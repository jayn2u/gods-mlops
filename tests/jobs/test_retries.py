from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import asyncpg
from conftest import seed_training_ready_dataset
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ExecutionProfile, ProcessIdentity, ResourceObservation
from gods_mlops.jobs.monitor import GpuJobMonitor
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry
from gods_mlops.jobs.models import ExecutionProfile
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

BASE_TIME = datetime(2026, 10, 5, 15, tzinfo=UTC)
GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"
OWNER = ProcessIdentity(pid=43122, start_ticks=89123, uid=1009)


def _observation(when: datetime, *, processes: tuple[ProcessIdentity, ...] = ()) -> dict:
    return ResourceObservation(
        observation_id=str(uuid4()),
        node_id="ubuntu",
        hostname="ubuntu",
        host_identity=HOST_IDENTITY,
        gpu_name="NVIDIA RTX A6000",
        gpu_uuid=GPU_UUID,
        free_mib=48_000,
        total_mib=49_140,
        gpu_processes=processes,
        gpu_process_list_complete=True,
        process_table=processes,
        process_table_complete=True,
        storage_path=STORAGE_PATH,
        filesystem_identity=FILESYSTEM_IDENTITY,
        filesystem_available_bytes=2 * 1024**4,
        observed_at=when,
    ).to_dict()


async def _seed_measured_training_profile(
    *,
    database_url: str,
    repository: PostgresJobQueueRepository,
    queue: JobQueue,
    config_version: str,
    measured_memory_mib: int,
    config: dict,
    oom_alternatives: tuple[str, ...],
) -> None:
    config_cap = min(49_140, measured_memory_mib + 8_192)
    candidate = ExecutionProfile(
        model_kind="detr",
        config_version=config_version,
        phase="training",
        memory_requirement_mib=config_cap,
        artifact_reservation_bytes=32 * 1024**2,
        config=config,
        candidate=True,
        oom_alternatives=oom_alternatives,
    )
    await queue.register_profile(candidate)
    probe = ExecutionProfile(
        model_kind="detr",
        config_version=config_version,
        phase="probe",
        target_phase="training",
        memory_requirement_mib=config_cap,
        artifact_reservation_bytes=32 * 1024**2,
        config=config,
        candidate=True,
        oom_alternatives=oom_alternatives,
    )
    await queue.register_profile(probe)
    probe_id = await queue.submit_probe(
        probe_input_id=f"profile-proof-{config_version}",
        input_sha256=hashlib.sha256(config_version.encode()).hexdigest(),
        model_kind="detr",
        config_version=config_version,
    )
    job = await queue.get(probe_id)
    measurement_id = uuid4()
    checkpoint_sha = hashlib.sha256(f"checkpoint-{config_version}".encode()).hexdigest()
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO gods_mlops_profile_measurements (
                measurement_id, job_id, input_sha256, config_sha256, model_kind,
                target_phase, result_state, peak_allocated_mib, peak_reserved_mib,
                optimizer_steps, inference_steps, checkpoint_resumed,
                verification_details, checkpoint_sha256, exit_code
            ) VALUES ($1, $2::uuid, $3, $4, 'detr', 'training', 'succeeded',
                $5, $6, 3, 0, TRUE, '{"passed":true}'::jsonb, $7, 0)
            """,
            measurement_id,
            probe_id,
            job["input_sha256"],
            job["config_sha256"],
            measured_memory_mib - 512,
            measured_memory_mib,
            checkpoint_sha,
        )
        await connection.execute(
            """
            UPDATE gods_mlops_resource_profiles
            SET memory_requirement_mib = $4, profile_state = 'measured', measurement_id = $5
            WHERE phase = 'training' AND model_kind = 'detr' AND config_version = $1
              AND config_sha256 = $2 AND artifact_reservation_bytes = $3
            """,
            config_version,
            candidate.config_sha256,
            candidate.artifact_reservation_bytes,
            measured_memory_mib,
            measurement_id,
        )
        await connection.execute(
            "UPDATE gods_mlops_jobs SET state = 'completed', completed_at = now() WHERE job_id = $1::uuid",
            probe_id,
        )
    finally:
        await connection.close()


async def _active_training_job(
    *,
    database_url: str,
    queue: JobQueue,
    repository: PostgresJobQueueRepository,
    dataset_version: str,
    config_version: str,
):
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
    job_id = await queue.submit(dataset_version, "detr", config_version)
    for offset in range(0, 31, 5):
        now[0] = BASE_TIME + timedelta(seconds=offset)
        admitted = await admission.admit(job_id, _observation(now[0]))
    assert admitted["state"] == "running"
    await queue.bind_process(job_id, admitted["lease_token"], OWNER)
    monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)
    return job_id, admitted["lease_token"], admission, monitor, now


def test_communication_retries_wait_ten_thirty_ninety_then_fail(task7_database_url: str) -> None:
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
            config_version="comm-retry-probe-v1",
            phase="probe",
            memory_requirement_mib=8_192,
            artifact_reservation_bytes=32 * 1024**2,
            config={"purpose": "comm-retry"},
            candidate=True,
        )
        await queue.register_profile(profile)
        job_id = await queue.submit_probe(
            probe_input_id="comm-retry-fixture-v1",
            input_sha256=hashlib.sha256(b"comm retry fixture").hexdigest(),
            model_kind="detr",
            config_version=profile.config_version,
        )
        now = BASE_TIME
        for count, delay in enumerate((10, 30, 90), start=1):
            state = await queue.record_communication_failure(
                job_id=job_id,
                error_code="temporary_database_unavailable",
                now=now,
            )
            assert state["state"] == "waiting_gpu"
            assert state["retryable"] is True
            assert state["reason_code"] == "communication_retry_scheduled"
            assert state["communication_retries"] == count
            assert state["oom_retries"] == 0
            assert datetime.fromisoformat(state["next_retry_at"]) == now + timedelta(seconds=delay)
            now += timedelta(seconds=delay)
        terminal = await queue.record_communication_failure(
            job_id=job_id,
            error_code="temporary_database_unavailable",
            now=now,
        )
        assert terminal["state"] == "failed"
        assert terminal["retryable"] is False
        assert terminal["reason_code"] == "communication_retries_exhausted"
        assert terminal["communication_retries"] == 3
        assert terminal["oom_retries"] == 0
        assert await repository.get_active_lease("GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e") is None
        assert await repository.artifact_reservation_for(job_id) is None
        await queue.close()
        await repository.close()
        await queue._sources.close()

    asyncio.run(exercise())


def test_admission_honors_a_scheduled_communication_delay_without_allocating_resources(
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
            config_version="comm-delay-probe-v1",
            phase="probe",
            memory_requirement_mib=8_192,
            artifact_reservation_bytes=32 * 1024**2,
            config={"purpose": "comm-delay"},
            candidate=True,
        )
        await queue.register_profile(profile)
        job_id = await queue.submit_probe(
            probe_input_id="comm-delay-fixture-v1",
            input_sha256=hashlib.sha256(b"comm delay fixture").hexdigest(),
            model_kind="detr",
            config_version=profile.config_version,
        )
        await queue.record_communication_failure(
            job_id=job_id,
            error_code="temporary_database_unavailable",
            now=BASE_TIME,
        )
        now = BASE_TIME + timedelta(seconds=5)
        observation = ResourceObservation(
            observation_id=str(uuid4()),
            node_id="ubuntu",
            hostname="ubuntu",
            host_identity="machine-sha256:task7-test-ubuntu",
            gpu_name="NVIDIA RTX A6000",
            gpu_uuid=GPU_UUID,
            free_mib=48_000,
            total_mib=49_140,
            gpu_processes=(),
            gpu_process_list_complete=True,
            process_table=(),
            process_table_complete=True,
            storage_path="/data/jayn2u/gods-mlops",
            filesystem_identity="ext4:uuid=task7-test-data",
            filesystem_available_bytes=2 * 1024**4,
            observed_at=now,
        )

        class Observer:
            async def observe(self):
                raise AssertionError("retry delay must not run a pre-launch observation")

        admission = GpuAdmission(
            repository=repository,
            queue=queue,
            expected_host_identity="machine-sha256:task7-test-ubuntu",
            expected_gpu_uuid=GPU_UUID,
            expected_filesystem_identity="ext4:uuid=task7-test-data",
            expected_storage_path="/data/jayn2u/gods-mlops",
            observer=Observer(),
            clock=lambda: now,
        )
        delayed = await admission.admit(job_id, observation.to_dict())
        assert delayed["state"] == "waiting_gpu"
        assert delayed["reason_code"] == "communication_retry_delay"
        assert delayed["oom_retries"] == 0
        assert await repository.get_active_lease(GPU_UUID) is None
        assert await repository.artifact_reservation_for(job_id) is None
        await queue.close()
        await repository.close()
        await queue._sources.close()

    asyncio.run(exercise())


def test_oom_uses_at_most_two_measured_smaller_children_then_stops(task7_database_url: str) -> None:
    async def exercise() -> None:
        dataset_version = await seed_training_ready_dataset(task7_database_url)
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        queue = JobQueue(
            repository=repository,
            sources=DatasetSourceRegistry(database_url=task7_database_url),
        )
        await _seed_measured_training_profile(
            database_url=task7_database_url,
            repository=repository,
            queue=queue,
            config_version="detr-batch2-v1",
            measured_memory_mib=12_288,
            config={"micro_batch": 2, "input_size": 640},
            oom_alternatives=("detr-batch1-v1", "detr-batch1-checkpointed-v1"),
        )
        await _seed_measured_training_profile(
            database_url=task7_database_url,
            repository=repository,
            queue=queue,
            config_version="detr-batch1-v1",
            measured_memory_mib=8_192,
            config={"micro_batch": 1, "input_size": 640},
            oom_alternatives=("detr-batch1-checkpointed-v1",),
        )
        await _seed_measured_training_profile(
            database_url=task7_database_url,
            repository=repository,
            queue=queue,
            config_version="detr-batch1-checkpointed-v1",
            measured_memory_mib=4_096,
            config={"micro_batch": 1, "input_size": 640, "gradient_checkpointing": True},
            oom_alternatives=(),
        )
        parent_id, first_token, admission, monitor, now = await _active_training_job(
            database_url=task7_database_url,
            queue=queue,
            repository=repository,
            dataset_version=dataset_version,
            config_version="detr-batch2-v1",
        )
        first_retry = await queue.record_oom(parent_id, first_token)
        assert first_retry["state"] == "retrying"
        first_child_id = first_retry["retry_job_id"]
        first_child = await queue.get(first_child_id)
        assert (await queue.get(parent_id))["reason_detail"]["checkpoint_reused"] is False
        assert first_child["parent_job_id"] == parent_id
        assert first_child["retry_root_id"] == parent_id
        assert first_child["oom_retries"] == 1
        assert first_child["config_version"] == "detr-batch1-v1"
        assert first_child["input_sha256"] == (await queue.get(parent_id))["input_sha256"]

        async def release_and_resume(job_id: str, old_token: str):
            now[0] += timedelta(seconds=5)
            await monitor.observe(_observation(now[0], processes=(OWNER,)))
            now[0] += timedelta(seconds=5)
            await monitor.observe(_observation(now[0]))
            assert await repository.get_active_lease(GPU_UUID) is None
            for _ in range(6):
                now[0] += timedelta(seconds=5)
                result = await admission.admit(job_id, _observation(now[0]))
            assert result["state"] == "running"
            assert result["lease_token"] != old_token
            await queue.bind_process(job_id, result["lease_token"], OWNER)
            return result["lease_token"]

        first_child_token = await release_and_resume(first_child_id, first_token)
        second_retry = await queue.record_oom(first_child_id, first_child_token)
        assert second_retry["state"] == "retrying"
        second_child_id = second_retry["retry_job_id"]
        second_child = await queue.get(second_child_id)
        assert second_child["parent_job_id"] == first_child_id
        assert second_child["retry_root_id"] == parent_id
        assert second_child["oom_retries"] == 2
        assert second_child["config_version"] == "detr-batch1-checkpointed-v1"

        second_child_token = await release_and_resume(second_child_id, first_child_token)
        exhausted = await queue.record_oom(second_child_id, second_child_token)
        assert exhausted["state"] == "failed"
        assert exhausted["retry_job_id"] is None
        assert exhausted["reason_code"] == "oom_alternatives_exhausted"
        assert (await queue.get(parent_id))["state"] == "failed"
        await queue.close()
        await repository.close()
        await queue._sources.close()

    asyncio.run(exercise())


def test_micro_batch_one_without_a_measured_smaller_alternative_is_terminal(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        dataset_version = await seed_training_ready_dataset(task7_database_url)
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        queue = JobQueue(
            repository=repository,
            sources=DatasetSourceRegistry(database_url=task7_database_url),
        )
        await _seed_measured_training_profile(
            database_url=task7_database_url,
            repository=repository,
            queue=queue,
            config_version="detr-batch1-terminal-v1",
            measured_memory_mib=8_192,
            config={"micro_batch": 1, "input_size": 640},
            oom_alternatives=(),
        )
        job_id, token, _admission, _monitor, _now = await _active_training_job(
            database_url=task7_database_url,
            queue=queue,
            repository=repository,
            dataset_version=dataset_version,
            config_version="detr-batch1-terminal-v1",
        )
        result = await queue.record_oom(job_id, token)
        assert result["state"] == "failed"
        assert result["reason_code"] == "oom_no_validated_smaller_profile"
        assert result["retry_job_id"] is None
        await queue.close()
        await repository.close()
        await queue._sources.close()

    asyncio.run(exercise())


def test_oom_rejects_an_unmeasured_candidate_as_a_smaller_alternative(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        dataset_version = await seed_training_ready_dataset(task7_database_url)
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        queue = JobQueue(
            repository=repository,
            sources=DatasetSourceRegistry(database_url=task7_database_url),
        )
        await _seed_measured_training_profile(
            database_url=task7_database_url,
            repository=repository,
            queue=queue,
            config_version="detr-candidate-gate-base-v1",
            measured_memory_mib=12_288,
            config={"micro_batch": 2, "input_size": 640},
            oom_alternatives=("detr-unmeasured-small-v1",),
        )
        await queue.register_profile(
            ExecutionProfile(
                model_kind="detr",
                config_version="detr-unmeasured-small-v1",
                phase="training",
                memory_requirement_mib=4_096,
                artifact_reservation_bytes=32 * 1024**2,
                config={"micro_batch": 1, "input_size": 640},
                candidate=True,
            )
        )
        job_id, token, _admission, _monitor, _now = await _active_training_job(
            database_url=task7_database_url,
            queue=queue,
            repository=repository,
            dataset_version=dataset_version,
            config_version="detr-candidate-gate-base-v1",
        )
        result = await queue.record_oom(job_id, token)
        assert result["state"] == "failed"
        assert result["reason_code"] == "oom_alternative_not_measured_or_smaller"
        assert result["retry_job_id"] is None
        await queue.close()
        await repository.close()
        await queue._sources.close()

    asyncio.run(exercise())
