from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Callable
from uuid import uuid4

from conftest import seed_training_ready_dataset
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ExecutionProfile, ProcessIdentity, ResourceObservation
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"
EXTERNAL_PROCESS = ProcessIdentity(pid=52111, start_ticks=99113, uid=1009)
OWNER = ProcessIdentity(pid=43122, start_ticks=89123, uid=1009)
MIN_FREE_BYTES = 1024**4
BASE_TIME = datetime(2026, 10, 5, 15, tzinfo=UTC)


def _observation(
    when: datetime,
    *,
    gpu_processes: tuple[ProcessIdentity, ...] = (),
    process_table: tuple[ProcessIdentity, ...] = (),
    **overrides,
) -> dict:
    return ResourceObservation(
        observation_id=overrides.pop("observation_id", None) or str(uuid4()),
        node_id="ubuntu",
        hostname="ubuntu",
        host_identity=HOST_IDENTITY,
        gpu_name="NVIDIA RTX A6000",
        gpu_uuid=GPU_UUID,
        free_mib=48_000,
        total_mib=49_140,
        gpu_processes=gpu_processes,
        gpu_process_list_complete=True,
        process_table=process_table,
        process_table_complete=True,
        storage_path=STORAGE_PATH,
        filesystem_identity=FILESYSTEM_IDENTITY,
        filesystem_available_bytes=2 * MIN_FREE_BYTES,
        observed_at=when,
    ).to_dict() | overrides


class SequenceObserver:
    def __init__(self, clock: Callable[[], datetime], snapshots: list[Callable[[datetime], dict]]):
        self._clock = clock
        self._snapshots = snapshots
        self.calls = 0

    async def observe(self) -> dict:
        snapshot = self._snapshots[self.calls](self._clock())
        self.calls += 1
        return snapshot


class TypedSequenceObserver:
    def __init__(self, clock: Callable[[], datetime], snapshot: Callable[[datetime], dict]):
        self._clock = clock
        self._snapshot = snapshot
        self.calls = 0

    async def observe(self) -> ResourceObservation:
        self.calls += 1
        return ResourceObservation.from_dict(self._snapshot(self._clock()))


async def _queue_with_probe(database_url: str) -> tuple[PostgresJobQueueRepository, JobQueue, str]:
    dataset_version = await seed_training_ready_dataset(database_url)
    repository = PostgresJobQueueRepository(database_url=database_url)
    await repository.ensure_schema()
    queue = JobQueue(
        repository=repository,
        sources=DatasetSourceRegistry(database_url=database_url),
    )
    await queue.register_profile(
        ExecutionProfile(
            model_kind="detr",
            config_version="detr-probe-candidate-v1",
            phase="probe",
            memory_requirement_mib=8_192,
            artifact_reservation_bytes=32 * 1024**2,
            config={"model_revision": "5650961749fa93567c0d46fc7f43ea4f9e914107"},
            candidate=True,
        )
    )
    job_id = await queue.submit_probe(
        probe_input_id="probe-task7-admission-v1",
        input_sha256=hashlib.sha256(b"immutable probe input").hexdigest(),
        model_kind="detr",
        config_version="detr-probe-candidate-v1",
    )
    return repository, queue, job_id


def _admission(repository, queue, now, observer=None):
    return GpuAdmission(
        repository=repository,
        queue=queue,
        expected_host_identity=HOST_IDENTITY,
        expected_gpu_uuid=GPU_UUID,
        expected_filesystem_identity=FILESYSTEM_IDENTITY,
        expected_storage_path=STORAGE_PATH,
        clock=lambda: now[0],
        observer=observer,
    )


def test_stable_idle_window_is_followed_by_a_fresh_prelaunch_external_process_check(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, job_id = await _queue_with_probe(task7_database_url)
        now = [BASE_TIME]
        external_after_idle = SequenceObserver(
            lambda: now[0],
            [
                lambda at: _observation(
                    at + timedelta(milliseconds=100),
                    gpu_processes=(EXTERNAL_PROCESS,),
                    process_table=(EXTERNAL_PROCESS,),
                ),
                lambda at: _observation(at + timedelta(milliseconds=100)),
            ],
        )
        admission = _admission(repository, queue, now, observer=external_after_idle)

        for offset in range(0, 31, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            result = await admission.admit(job_id, _observation(now[0]))
        assert result["state"] == "waiting_gpu"
        assert result["reason_code"] == "external_gpu_processes"
        assert external_after_idle.calls == 1
        assert await repository.get_active_lease(GPU_UUID) is None

        # The fresh read found another process after the 30-second idle series.
        # The queue has to observe a new uninterrupted idle period before claiming.
        for offset in range(35, 66, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            result = await admission.admit(job_id, _observation(now[0]))
        assert result["state"] == "running"
        assert external_after_idle.calls == 2
        assert (await repository.get_active_lease(GPU_UUID))["job_id"] == job_id
        await queue.close()
        await repository.close()

    asyncio.run(exercise())


def test_typed_resource_observation_can_complete_the_fresh_prelaunch_check(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, job_id = await _queue_with_probe(task7_database_url)
        now = [BASE_TIME]
        observer = TypedSequenceObserver(
            lambda: now[0],
            lambda at: _observation(at + timedelta(milliseconds=100)),
        )
        admission = _admission(repository, queue, now, observer=observer)

        for offset in range(0, 31, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            result = await admission.admit(job_id, _observation(now[0]))

        assert result["state"] == "running"
        assert observer.calls == 1
        assert (await repository.get_active_lease(GPU_UUID))["job_id"] == job_id
        await queue.close()
        await repository.close()

    asyncio.run(exercise())


def test_stale_prelaunch_wait_preserves_an_owner_yield_requested_after_the_read_started(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, job_id = await _queue_with_probe(task7_database_url)
        now = [BASE_TIME]

        class PausedObserver:
            def __init__(self) -> None:
                self.entered = asyncio.Event()
                self.release = asyncio.Event()

            async def observe(self) -> dict:
                self.entered.set()
                await self.release.wait()
                return _observation(
                    now[0] + timedelta(milliseconds=100),
                    gpu_processes=(OWNER, EXTERNAL_PROCESS),
                    process_table=(OWNER, EXTERNAL_PROCESS),
                )

        paused = PausedObserver()
        stale_admission = _admission(repository, queue, now, observer=paused)
        for offset in range(0, 26, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            wait = await stale_admission.admit(job_id, _observation(now[0]))
        assert wait["reason_code"] == "idle_observation_window"

        now[0] = BASE_TIME + timedelta(seconds=30)
        stale_task = asyncio.create_task(stale_admission.admit(job_id, _observation(now[0])))
        await asyncio.wait_for(paused.entered.wait(), timeout=3)

        now[0] = BASE_TIME + timedelta(seconds=35)

        class FreshObserver:
            async def observe(self) -> dict:
                return _observation(now[0] + timedelta(milliseconds=100))

        owner_admission = _admission(repository, queue, now, observer=FreshObserver())
        granted = await owner_admission.admit(job_id, _observation(now[0]))
        assert granted["state"] == "running"
        token = granted["lease_token"]
        assert await queue.bind_process(job_id, token, OWNER)
        await queue.request_yield(job_id, reason="external_gpu_process_started")

        paused.release.set()
        stale_result = await stale_task
        current = await queue.get(job_id)
        active = await repository.get_active_lease(GPU_UUID)
        assert stale_result["state"] == "yield_requested"
        assert current["state"] == "yield_requested"
        assert current["lease_token"] == token
        assert current["owner_pid"] == OWNER.pid
        assert current["owner_start_ticks"] == OWNER.start_ticks
        assert active["lease_token"] == token
        await queue.close()
        await repository.close()

    asyncio.run(exercise())


def test_transactional_wait_does_not_mutate_a_yield_requested_job_with_a_live_lease(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, job_id = await _queue_with_probe(task7_database_url)
        now = [BASE_TIME]

        class FreshObserver:
            async def observe(self) -> dict:
                return _observation(now[0] + timedelta(milliseconds=100))

        admission = _admission(repository, queue, now, observer=FreshObserver())
        for offset in range(0, 31, 5):
            now[0] = BASE_TIME + timedelta(seconds=offset)
            running = await admission.admit(job_id, _observation(now[0]))
        assert running["state"] == "running"
        token = running["lease_token"]
        assert await queue.bind_process(job_id, token, OWNER)
        await queue.request_yield(job_id, reason="external_gpu_process_started")
        observation = await repository.latest_observation("ubuntu")
        profile = await repository.get_profile(
            phase="probe", model_kind="detr", config_version="detr-probe-candidate-v1"
        )

        waited = await repository.acquire_gpu_lease(
            job_id=job_id,
            observation=observation,
            profile=profile,
            source_registry=queue.source_registry,
            now=now[0],
            lease_seconds=15,
            min_idle_seconds=30,
            min_idle_observations=7,
            min_filesystem_bytes=MIN_FREE_BYTES,
            safety_mib=4096,
        )
        assert waited["state"] == "yield_requested"
        assert waited["lease_token"] == token
        assert waited["owner_pid"] == OWNER.pid
        assert (await repository.get_active_lease(GPU_UUID))["lease_token"] == token
        await queue.close()
        await repository.close()

    asyncio.run(exercise())


def test_replayed_observation_resets_the_persisted_idle_window(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, job_id = await _queue_with_probe(task7_database_url)
        now = [BASE_TIME]
        admission = _admission(repository, queue, now)
        snapshot = _observation(now[0])
        first = await admission.admit(job_id, snapshot)
        assert first["reason_code"] == "idle_observation_window"
        initial_state = await repository.get_observation_state("ubuntu")
        assert initial_state["idle_observation_count"] == 1

        now[0] += timedelta(seconds=5)
        replay = await admission.admit(job_id, snapshot)
        assert replay["state"] == "waiting_gpu"
        assert replay["reason_code"] == "ubuntu_observation_replayed"
        after_replay = await repository.get_observation_state("ubuntu")
        assert after_replay["observation_id"] != initial_state["observation_id"]
        assert after_replay["failure_code"] is not None
        assert after_replay["idle_since"] is None
        assert after_replay["idle_observation_count"] == 0
        assert await repository.get_active_lease(GPU_UUID) is None
        await queue.close()
        await repository.close()

    asyncio.run(exercise())


def test_missing_stale_and_wrong_host_observations_fail_closed(task7_database_url: str) -> None:
    async def exercise() -> None:
        repository, queue, job_id = await _queue_with_probe(task7_database_url)
        now = [BASE_TIME]
        admission = _admission(repository, queue, now)

        missing = await admission.admit(job_id, {})
        assert missing["state"] == "waiting_gpu"
        assert missing["reason_code"] == "ubuntu_observation_unavailable"
        stale = await admission.admit(
            job_id, _observation(now[0] - timedelta(minutes=1))
        )
        assert stale["state"] == "waiting_gpu"
        assert stale["reason_code"] == "ubuntu_observation_stale"
        wrong_host = await admission.admit(
            job_id, _observation(now[0], host_identity="machine-sha256:wrong-host")
        )
        assert wrong_host["state"] == "waiting_gpu"
        assert wrong_host["reason_code"] == "ubuntu_observation_identity_mismatch"
        assert await repository.get_active_lease(GPU_UUID) is None
        assert await repository.artifact_reservation_for(job_id) is None
        await queue.close()
        await repository.close()

    asyncio.run(exercise())


def test_incomplete_process_observation_is_not_treated_as_gpu_idle(task7_database_url: str) -> None:
    async def exercise() -> None:
        repository, queue, job_id = await _queue_with_probe(task7_database_url)
        now = [BASE_TIME]
        admission = _admission(repository, queue, now)
        incomplete = await admission.admit(
            job_id,
            _observation(
                now[0],
                gpu_process_list_complete=False,
                process_table_complete=False,
            ),
        )
        assert incomplete["state"] == "waiting_gpu"
        assert incomplete["reason_code"] == "ubuntu_observation_incomplete"
        assert await repository.get_active_lease(GPU_UUID) is None
        await queue.close()
        await repository.close()

    asyncio.run(exercise())
