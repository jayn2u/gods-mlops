from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from conftest import seed_training_ready_dataset
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.checkpoints import CheckpointIdentity, FileCheckpointStore
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


async def _probe_queue(database_url: str, config_version: str):
    dataset_version = await seed_training_ready_dataset(database_url)
    repository = PostgresJobQueueRepository(database_url=database_url)
    await repository.ensure_schema()
    sources = DatasetSourceRegistry(database_url=database_url)
    queue = JobQueue(repository=repository, sources=sources)
    await queue.register_profile(
        ExecutionProfile(
            model_kind="detr",
            config_version=config_version,
            phase="probe",
            memory_requirement_mib=16_384,
            artifact_reservation_bytes=32 * 1024**2,
            config={"input_size": 640, "micro_batch": 1},
            candidate=True,
        )
    )
    probe_id = await queue.submit_probe(
        probe_input_id=f"probe-{config_version}",
        input_sha256=hashlib.sha256(config_version.encode()).hexdigest(),
        model_kind="detr",
        config_version=config_version,
    )
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
        admitted = await admission.admit(probe_id, _observation(now[0]))
    return repository, queue, admission, dataset_version, probe_id, admitted["lease_token"], now


def test_only_successful_optimizer_and_resume_probe_creates_measured_training_profile(
    task7_database_url: str,
    tmp_path,
) -> None:
    async def exercise() -> None:
        config_version = "detr-measured-config-v1"
        repository, queue, _admission, dataset_version, probe_id, token, _now = await _probe_queue(
            task7_database_url, config_version
        )
        job = await queue.get(probe_id)
        identity = CheckpointIdentity(
            job_id=probe_id,
            input_kind=job["input_kind"],
            input_id=job["input_id"],
            input_sha256=job["input_sha256"],
            phase=job["phase"],
            model_kind=job["model_kind"],
            config_version=job["config_version"],
            config_sha256=job["config_sha256"],
            dataset_version=job["dataset_version"],
        )
        store = FileCheckpointStore(root=tmp_path)
        checkpoint = await queue.save_checkpoint(
            store=store,
            job_id=probe_id,
            lease_token=token,
            identity=identity,
            payload=b"saved state before resume",
        )
        measured = await queue.record_probe_measurement(
            job_id=probe_id,
            lease_token=token,
            exit_code=0,
            peak_allocated_mib=7_500,
            peak_reserved_mib=8_192,
            optimizer_steps=3,
            checkpoint_resumed=True,
            checkpoint_sha256=checkpoint.sha256,
            verification_details={"passed": True, "probe_schema": "task7-v1"},
        )
        assert measured["profile_state"] == "measured"
        training_profile = await repository.get_profile(
            phase="training", model_kind="detr", config_version=config_version
        )
        assert training_profile is not None
        assert training_profile["profile_state"] == "measured"
        assert training_profile["measurement_id"] == measured["measurement_id"]
        assert training_profile["memory_requirement_mib"] == 8_192
        training_job = await queue.submit(dataset_version, "detr", config_version)
        assert (await queue.get(training_job))["profile_state_snapshot"] == "measured"
        await queue.close()
        await repository.close()
        await queue._sources.close()

    asyncio.run(exercise())


def test_failed_or_unresumed_probe_never_enters_the_measured_profile_allowlist(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        config_version = "detr-failed-config-v1"
        repository, queue, _admission, _dataset_version, probe_id, token, _now = await _probe_queue(
            task7_database_url, config_version
        )
        failed = await queue.record_probe_measurement(
            job_id=probe_id,
            lease_token=token,
            exit_code=0,
            peak_allocated_mib=7_500,
            peak_reserved_mib=8_192,
            optimizer_steps=2,
            checkpoint_resumed=False,
            checkpoint_sha256=None,
            verification_details={"passed": True, "probe_schema": "task7-v1"},
        )
        assert failed["profile_state"] == "candidate"
        assert failed["result_state"] == "failed"
        assert failed["reason_code"] == "probe_contract_not_met"
        assert await repository.get_profile(
            phase="training", model_kind="detr", config_version=config_version
        ) is None
        job = await queue.get(probe_id)
        assert job["state"] == "failed"
        await queue.close()
        await repository.close()
        await queue._sources.close()

    asyncio.run(exercise())


def test_detr_preparation_inference_promotes_preparation_profile_without_training_checkpoint(
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
        await queue.register_profile(
            ExecutionProfile(
                model_kind="detr",
                config_version="detr-preparation-candidate-v1",
                phase="probe",
                target_phase="preparation",
                memory_requirement_mib=16_384,
                artifact_reservation_bytes=32 * 1024**2,
                config={"draft_batch": 1},
                candidate=True,
            )
        )
        probe_id = await queue.submit_probe(
            probe_input_id="immutable-detector-preparation-probe-v1",
            input_sha256=hashlib.sha256(b"detr preparation fixture").hexdigest(),
            model_kind="detr",
            config_version="detr-preparation-candidate-v1",
        )
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
            admitted = await admission.admit(probe_id, _observation(now[0]))
        result = await queue.record_probe_measurement(
            job_id=probe_id,
            lease_token=admitted["lease_token"],
            exit_code=0,
            peak_allocated_mib=9_000,
            peak_reserved_mib=10_000,
            optimizer_steps=0,
            checkpoint_resumed=False,
            checkpoint_sha256=None,
            inference_steps=1,
            verification_details={"passed": True, "mode": "draft-inference"},
        )
        assert result["profile_state"] == "measured"
        assert result["phase"] == "preparation"
        measured_preparation = await repository.get_profile(
            phase="preparation", model_kind="detr", config_version="detr-preparation-candidate-v1"
        )
        assert measured_preparation["profile_state"] == "measured"
        assert measured_preparation["memory_requirement_mib"] == 10_000
        assert await repository.get_profile(
            phase="training", model_kind="detr", config_version="detr-preparation-candidate-v1"
        ) is None
        await queue.close()
        await repository.close()
        await queue._sources.close()

    asyncio.run(exercise())
