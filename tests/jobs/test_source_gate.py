from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
from conftest import seed_training_ready_dataset
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ExecutionProfile, ResourceObservation
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"
BASE_TIME = datetime(2026, 10, 5, 15, tzinfo=UTC)


def _observation(when: datetime) -> dict:
    return ResourceObservation(
        observation_id=str(uuid4()),
        node_id="ubuntu",
        hostname="ubuntu",
        host_identity=HOST_IDENTITY,
        gpu_name="NVIDIA RTX A6000",
        gpu_uuid=GPU_UUID,
        free_mib=48_000,
        total_mib=49_140,
        gpu_processes=(),
        gpu_process_list_complete=True,
        process_table=(),
        process_table_complete=True,
        storage_path=STORAGE_PATH,
        filesystem_identity=FILESYSTEM_IDENTITY,
        filesystem_available_bytes=2 * 1024**4,
        observed_at=when,
    ).to_dict()


def test_dataset_tombstone_after_submit_blocks_gpu_admission(task7_database_url: str) -> None:
    async def exercise() -> None:
        dataset_version = await seed_training_ready_dataset(task7_database_url)
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        sources = DatasetSourceRegistry(database_url=task7_database_url)
        queue = JobQueue(repository=repository, sources=sources)
        profile = ExecutionProfile(
            model_kind="detr",
            config_version="detr-source-gate-v1",
            phase="training",
            memory_requirement_mib=8_192,
            artifact_reservation_bytes=16 * 1024**2,
            config={"input_size": 640, "micro_batch": 1},
            candidate=True,
        )
        await queue.register_profile(profile)
        measurement_id = uuid4()
        connection = await asyncpg.connect(task7_database_url)
        try:
            await connection.execute(
                """
                UPDATE gods_mlops_resource_profiles SET profile_state = 'measured', measurement_id = $4
                WHERE phase = 'training' AND model_kind = 'detr' AND config_version = $1
                  AND config_sha256 = $2 AND memory_requirement_mib = $3
                """,
                profile.config_version,
                profile.config_sha256,
                profile.memory_requirement_mib,
                measurement_id,
            )
        finally:
            await connection.close()
        job_id = await queue.submit(dataset_version, "detr", profile.config_version)
        now = [BASE_TIME]

        class Observer:
            def __init__(self) -> None:
                self.calls = 0

            async def observe(self):
                self.calls += 1
                return _observation(now[0] + timedelta(milliseconds=100))

        observer = Observer()
        admission = GpuAdmission(
            repository=repository,
            queue=queue,
            expected_host_identity=HOST_IDENTITY,
            expected_gpu_uuid=GPU_UUID,
            expected_filesystem_identity=FILESYSTEM_IDENTITY,
            expected_storage_path=STORAGE_PATH,
            observer=observer,
            clock=lambda: now[0],
        )
        for offset in range(0, 26, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            waiting = await admission.admit(job_id, _observation(now[0]))
        assert waiting["state"] == "waiting_gpu"
        assert waiting["reason_code"] == "idle_observation_window"

        connection = await asyncpg.connect(task7_database_url)
        try:
            sample_id = await connection.fetchval(
                "SELECT sample_id FROM dataset_items WHERE dataset_version = $1 LIMIT 1",
                dataset_version,
            )
            await connection.execute(
                """
                INSERT INTO dataset_source_invalidations (sample_id, reason)
                VALUES ($1, 'late_task7_source_tombstone')
                """,
                sample_id,
            )
        finally:
            await connection.close()

        now[0] = BASE_TIME + timedelta(seconds=30)
        blocked = await admission.admit(job_id, _observation(now[0]))
        assert blocked["state"] == "failed"
        assert blocked["reason_code"] == "source_sample_explicitly_invalidated"
        assert observer.calls == 0
        assert await repository.get_active_lease(GPU_UUID) is None
        assert await repository.artifact_reservation_for(job_id) is None
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())
