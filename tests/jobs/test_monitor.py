from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from conftest import seed_training_ready_dataset
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ExecutionProfile, ProcessIdentity, ResourceObservation
from gods_mlops.jobs.monitor import GpuJobMonitor
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"
MIN_FREE_BYTES = 1024**4
BASE_TIME = datetime(2026, 10, 5, 15, tzinfo=UTC)
OWNER = ProcessIdentity(pid=43122, start_ticks=89123, uid=1009)
EXTERNAL = ProcessIdentity(pid=52111, start_ticks=99113, uid=1009)


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
        filesystem_available_bytes=2 * MIN_FREE_BYTES,
        observed_at=when,
    ).to_dict()


async def _running_probe(database_url: str):
    await seed_training_ready_dataset(database_url)
    repository = PostgresJobQueueRepository(database_url=database_url)
    await repository.ensure_schema()
    queue = JobQueue(
        repository=repository,
        sources=DatasetSourceRegistry(database_url=database_url),
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
    job_id = await queue.submit_probe(
        probe_input_id=f"probe-monitor-{uuid4()}",
        input_sha256=hashlib.sha256(uuid4().bytes).hexdigest(),
        model_kind="detr",
        config_version=profile.config_version,
    )
    now = [BASE_TIME]

    class FreshObserver:
        async def observe(self) -> dict:
            return _observation(now[0] + timedelta(milliseconds=100))

    admission = GpuAdmission(
        repository=repository,
        queue=queue,
        expected_host_identity=HOST_IDENTITY,
        expected_gpu_uuid=GPU_UUID,
        expected_filesystem_identity=FILESYSTEM_IDENTITY,
        expected_storage_path=STORAGE_PATH,
        observer=FreshObserver(),
        clock=lambda: now[0],
    )
    for offset in range(0, 31, 5):
        now[0] = BASE_TIME + timedelta(seconds=offset)
        result = await admission.admit(job_id, _observation(now[0]))
    assert result["state"] == "running"
    lease_token = result["lease_token"]
    await queue.bind_process(job_id, lease_token, OWNER)
    monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)
    return repository, queue, admission, monitor, job_id, lease_token, now


def test_external_work_requests_own_job_yield_and_expired_live_owner_keeps_the_lease(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, _admission, monitor, job_id, token, now = await _running_probe(task7_database_url)
        now[0] += timedelta(seconds=5)
        external_start = _observation(
            now[0], gpu=(OWNER, EXTERNAL), processes=(OWNER, EXTERNAL)
        )
        await monitor.observe(external_start)
        job = await queue.get(job_id)
        assert job["state"] == "yield_requested"
        assert job["reason_code"] == "external_gpu_process_started"
        assert job["oom_retries"] == 0
        assert job["communication_retries"] == 0
        assert (await repository.get_active_lease(GPU_UUID))["lease_token"] == token

        # An expired lease and a SIGSTOP-like still-visible process are not release proof.
        now[0] += timedelta(seconds=60)
        await monitor.observe(
            _observation(now[0], gpu=(OWNER, EXTERNAL), processes=(OWNER, EXTERNAL))
        )
        assert (await repository.get_active_lease(GPU_UUID))["lease_token"] == token
        assert (await queue.get(job_id))["state"] == "yield_requested"
        await queue.close()
        await repository.close()

    asyncio.run(exercise())


def test_pid_reuse_does_not_release_vram_and_observed_exit_allows_fenced_resume(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, admission, monitor, job_id, old_token, now = await _running_probe(
            task7_database_url
        )
        now[0] += timedelta(seconds=5)
        await monitor.observe(
            _observation(now[0], gpu=(OWNER, EXTERNAL), processes=(OWNER, EXTERNAL))
        )
        now[0] += timedelta(seconds=60)
        pid_reuse = ProcessIdentity(pid=OWNER.pid, start_ticks=OWNER.start_ticks + 900, uid=1009)
        await monitor.observe(
            _observation(now[0], gpu=(pid_reuse,), processes=(pid_reuse,))
        )
        active = await repository.get_active_lease(GPU_UUID)
        assert active is not None
        assert active["lease_token"] == old_token

        # Once both the original process identity and that PID's CUDA allocation
        # disappear from complete observations, the monitor releases the old lease.
        now[0] += timedelta(seconds=5)
        await monitor.observe(_observation(now[0]))
        assert await repository.get_active_lease(GPU_UUID) is None
        assert (await queue.get(job_id))["state"] == "waiting_gpu"

        # A fresh uninterrupted idle window resumes the same job with a higher fence.
        for _ in range(6):
            now[0] += timedelta(seconds=5)
            result = await admission.admit(job_id, _observation(now[0]))
        assert result["state"] == "running"
        assert result["lease_token"] != old_token
        assert await queue.renew_lease(job_id, old_token) is False
        await queue.close()
        await repository.close()

    asyncio.run(exercise())
