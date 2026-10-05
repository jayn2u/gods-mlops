from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from conftest import seed_training_ready_dataset
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ExecutionProfile, ResourceObservation
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"
MIN_FREE_BYTES = 1024**4
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
        filesystem_available_bytes=2 * MIN_FREE_BYTES,
        observed_at=when,
    ).to_dict()


def test_fifo_queue_keeps_gpu_and_artifact_capacity_unallocated_for_later_job(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        await seed_training_ready_dataset(task7_database_url)
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        queue = JobQueue(
            repository=repository,
            sources=DatasetSourceRegistry(database_url=task7_database_url),
        )
        profile = ExecutionProfile(
            model_kind="detr",
            config_version="detr-probe-candidate-v1",
            phase="probe",
            memory_requirement_mib=8_192,
            artifact_reservation_bytes=32 * 1024**2,
            config={"model_revision": "5650961749fa93567c0d46fc7f43ea4f9e914107"},
            candidate=True,
        )
        await queue.register_profile(profile)
        first_id = await queue.submit_probe(
            probe_input_id="probe-fifo-first",
            input_sha256=hashlib.sha256(b"first probe input").hexdigest(),
            model_kind="detr",
            config_version=profile.config_version,
        )
        later_id = await queue.submit_probe(
            probe_input_id="probe-fifo-later",
            input_sha256=hashlib.sha256(b"later probe input").hexdigest(),
            model_kind="detr",
            config_version=profile.config_version,
        )
        now = [BASE_TIME]

        class FreshObserver:
            def __init__(self) -> None:
                self.calls = 0

            async def observe(self):
                self.calls += 1
                return _observation(now[0] + timedelta(milliseconds=100))

        observer = FreshObserver()
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
        for offset in range(0, 31, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            later = await admission.admit(later_id, _observation(now[0]))
        assert later["state"] == "waiting_gpu"
        assert later["reason_code"] == "fifo_queue_position"
        assert observer.calls == 0
        assert await repository.artifact_reservation_for(later_id) is None
        assert await repository.get_active_lease(GPU_UUID) is None

        now[0] = BASE_TIME + timedelta(seconds=31)
        admitted = await admission.admit(first_id, _observation(now[0]))
        assert admitted["state"] == "running"
        assert observer.calls == 1
        assert (await repository.get_active_lease(GPU_UUID))["job_id"] == first_id
        assert await repository.artifact_reservation_for(first_id) is not None
        assert await repository.artifact_reservation_for(later_id) is None
        await queue.close()
        await repository.close()

    asyncio.run(exercise())
