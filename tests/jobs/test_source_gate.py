from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
from conftest import seed_training_ready_dataset
from gods_mlops.datasets.publish import DatasetPublisher
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ExecutionProfile, ProcessIdentity, ResourceObservation
from gods_mlops.jobs.monitor import GpuJobMonitor
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"
BASE_TIME = datetime(2026, 10, 5, 15, tzinfo=UTC)
OWNER = ProcessIdentity(pid=43122, start_ticks=89123, uid=1009)


def _observation(when: datetime, *, gpu=(), processes=()) -> dict:
    return ResourceObservation(
        observation_id=str(uuid4()),
        node_id="ubuntu",
        hostname="ubuntu",
        host_identity=HOST_IDENTITY,
        gpu_name="NVIDIA RTX A6000",
        gpu_uuid=GPU_UUID,
        free_mib=48_000,
        total_mib=49_140,
        gpu_processes=tuple(gpu),
        gpu_process_list_complete=True,
        process_table=tuple(processes),
        process_table_complete=True,
        storage_path=STORAGE_PATH,
        filesystem_identity=FILESYSTEM_IDENTITY,
        filesystem_available_bytes=2 * 1024**4,
        observed_at=when,
    ).to_dict()


async def _training_job(database_url: str):
    dataset_version = await seed_training_ready_dataset(database_url)
    repository = PostgresJobQueueRepository(database_url=database_url)
    await repository.ensure_schema()
    sources = DatasetSourceRegistry(database_url=database_url)
    queue = JobQueue(repository=repository, sources=sources)
    profile = ExecutionProfile(
        model_kind="detr",
        config_version=f"detr-measured-{uuid4().hex}",
        phase="training",
        memory_requirement_mib=8_192,
        artifact_reservation_bytes=16 * 1024**2,
        config={"input_size": 640, "micro_batch": 1},
        candidate=True,
    )
    await queue.register_profile(profile)
    async with repository._pool.acquire() as connection:
        await connection.execute(
            """UPDATE gods_mlops_resource_profiles SET profile_state='measured', measurement_id=$2::uuid
               WHERE phase='training' AND model_kind='detr' AND config_version=$1""",
            profile.config_version,
            uuid4(),
        )
        sample_id = await connection.fetchval(
            "SELECT sample_id FROM dataset_items WHERE dataset_version=$1 LIMIT 1", dataset_version
        )
    job_id = await queue.submit(dataset_version, "detr", profile.config_version)
    return repository, queue, sources, dataset_version, sample_id, job_id


async def _invalidate_sample(database_url: str, sample_id: str) -> None:
    publisher = DatasetPublisher(database_url=database_url, objects=None)
    try:
        await publisher.invalidate_sample(str(sample_id))
    finally:
        await publisher.close()


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


def test_prelaunch_rechecks_a_task6_sample_invalidation_inside_lease_acquisition(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, sources, _version, sample_id, job_id = await _training_job(task7_database_url)
        now = [BASE_TIME]

        class PausedObserver:
            def __init__(self) -> None:
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def observe(self):
                self.entered.set()
                await self.release.wait()
                return _observation(now[0] + timedelta(milliseconds=100))

        observer = PausedObserver()
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
        try:
            for offset in range(0, 26, 5):
                now[0] = BASE_TIME + timedelta(seconds=offset)
                waiting = await admission.admit(job_id, _observation(now[0]))
            assert waiting["reason_code"] == "idle_observation_window"

            now[0] = BASE_TIME + timedelta(seconds=30)
            admission_task = asyncio.create_task(admission.admit(job_id, _observation(now[0])))
            await asyncio.wait_for(observer.entered.wait(), timeout=3)
            await _invalidate_sample(task7_database_url, sample_id)
        finally:
            observer.release.set()

        blocked = await admission_task
        assert blocked["state"] == "failed"
        assert blocked["reason_code"] == "source_sample_explicitly_invalidated"
        assert await repository.get_active_lease(GPU_UUID) is None
        assert await repository.artifact_reservation_for(job_id) is None
        assert await repository.is_next_eligible_job(job_id) is False
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_source_invalidated_running_job_releases_only_after_owner_exit_observation(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, sources, _version, sample_id, job_id = await _training_job(task7_database_url)
        now = [BASE_TIME]

        class Observer:
            async def observe(self):
                return _observation(now[0] + timedelta(milliseconds=100))

        admission = GpuAdmission(
            repository=repository,
            queue=queue,
            expected_host_identity=HOST_IDENTITY,
            expected_gpu_uuid=GPU_UUID,
            expected_filesystem_identity=FILESYSTEM_IDENTITY,
            expected_storage_path=STORAGE_PATH,
            observer=Observer(),
            clock=lambda: now[0],
        )
        for offset in range(0, 31, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            running = await admission.admit(job_id, _observation(now[0]))
        assert running["state"] == "running"
        token = running["lease_token"]
        assert await queue.bind_process(job_id, token, OWNER)
        monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)

        await _invalidate_sample(task7_database_url, sample_id)
        now[0] += timedelta(seconds=5)
        yielding = await monitor.observe(_observation(now[0], gpu=(OWNER,), processes=(OWNER,)))
        assert yielding["state"] == "yield_requested"
        assert yielding["reason_code"] == "source_sample_explicitly_invalidated"
        assert (await repository.get_active_lease(GPU_UUID))["lease_token"] == token

        now[0] += timedelta(seconds=5)
        released = await monitor.observe(_observation(now[0]))
        assert released["state"] == "failed"
        assert released["reason_code"] == "source_sample_explicitly_invalidated"
        assert released["retryable"] is False
        assert await repository.get_active_lease(GPU_UUID) is None
        assert await repository.is_next_eligible_job(job_id) is False
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_late_evaluation_leakage_overlay_does_not_block_training_ready_source(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, sources, dataset_version, _sample_id, job_id = await _training_job(task7_database_url)
        connection = await asyncpg.connect(task7_database_url)
        try:
            impact_id = uuid4()
            await connection.execute(
                "UPDATE dataset_versions SET evaluation_eligible=FALSE, evaluation_reasons=$2::jsonb WHERE dataset_version=$1",
                dataset_version,
                '["late_cross_boundary_link"]',
            )
            await connection.execute(
                """INSERT INTO dataset_split_leakage_impacts(impact_id,input_sha256,link_ids,sample_ids,splits)
                   VALUES($1,$2,$3::jsonb,$4::jsonb,$5::jsonb)""",
                impact_id,
                "b" * 64,
                '["late-link"]',
                '["00000000-0000-0000-0000-000000000001"]',
                '["train","test"]',
            )
            await connection.execute(
                "INSERT INTO dataset_version_leakage_impacts(dataset_version,impact_id) VALUES($1,$2)",
                dataset_version,
                impact_id,
            )
        finally:
            await connection.close()

        now = [BASE_TIME]

        class Observer:
            async def observe(self):
                return _observation(now[0] + timedelta(milliseconds=100))

        admission = GpuAdmission(repository=repository, queue=queue,
            expected_host_identity=HOST_IDENTITY, expected_gpu_uuid=GPU_UUID,
            expected_filesystem_identity=FILESYSTEM_IDENTITY, expected_storage_path=STORAGE_PATH,
            observer=Observer(), clock=lambda: now[0])
        for offset in range(0, 31, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            admitted = await admission.admit(job_id, _observation(now[0]))
        assert admitted["state"] == "running"
        assert (await queue.get(job_id))["dataset_version"] == dataset_version
        assert (await repository.get_active_lease(GPU_UUID))["job_id"] == job_id
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())
