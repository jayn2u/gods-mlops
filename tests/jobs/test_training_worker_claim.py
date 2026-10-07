from __future__ import annotations

import asyncio
import importlib
import json
import os
from hashlib import sha256
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
from conftest import seed_training_ready_dataset
from gods_mlops.datasets.publish import DatasetObjectStore, DatasetPublisher
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ExecutionProfile, ProcessIdentity, ProbeInput, ResourceObservation
from gods_mlops.jobs.monitor import GpuJobMonitor
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.checkpoints import StaleCheckpointOwnerError
from gods_mlops.jobs.sources import DatasetSourceRegistry
from gods_mlops.training.claims import WorkerClaim, WorkerAuthorizationError
from gods_mlops.training.artifacts import FileResultArtifactStore, S3ResultArtifactStore
from gods_mlops.training.checkpoints import S3CheckpointStore

GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task8-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task8-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"
OWNER = ProcessIdentity(pid=43122, start_ticks=89123, uid=10001)


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


async def _training_run(database_url: str):
    dataset_version = await seed_training_ready_dataset(database_url)
    repository = PostgresJobQueueRepository(database_url=database_url)
    await repository.ensure_schema()
    sources = DatasetSourceRegistry(database_url=database_url)
    queue = JobQueue(repository=repository, sources=sources)
    profile = ExecutionProfile(
        model_kind="detr",
        config_version=f"task8-guard-{uuid4().hex}",
        phase="training",
        memory_requirement_mib=8_192,
        artifact_reservation_bytes=32 * 1024**2,
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
    now = [await repository.artifact_database_clock()]

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
    return repository, queue, sources, admission, dataset_version, sample_id, job_id, now


async def _probe_run(database_url: str):
    repository = PostgresJobQueueRepository(database_url=database_url)
    await repository.ensure_schema()
    sources = DatasetSourceRegistry(database_url=database_url)
    queue = JobQueue(repository=repository, sources=sources)
    model = importlib.import_module("gods_mlops.training.contracts").locked_model("detr")
    config_version = f"task8-storage-probe-{uuid4().hex}"
    profile = ExecutionProfile(
        model_kind="detr",
        config_version=config_version,
        phase="probe",
        target_phase="training",
        memory_requirement_mib=8_192,
        artifact_reservation_bytes=96 * 1024**2,
        config={"model_id": model.model_id, "model_revision": model.revision, "input_size": 640, "micro_batch": 1},
        candidate=True,
    )
    await queue.register_profile(profile)
    probe_input = ProbeInput(
        probe_input_id=f"task8-storage-probe-{uuid4().hex}",
        model_kind="detr",
        target_phase="training",
        config_version=config_version,
        manifest_object_key=f"probe-inputs/task8-tests/{uuid4().hex}.json",
        input_sha256="5" * 64,
        object_size_bytes=1024,
        fixture=True,
    )
    job_id = await queue.submit_probe(
        model_kind="detr", config_version=config_version, probe_input=probe_input
    )
    now = [datetime.now(UTC)]

    class Observer:
        async def observe(self):
            return _observation(now[0] + timedelta(milliseconds=100))

    admission = GpuAdmission(
        repository=repository,
        queue=queue,
        expected_node_id="ubuntu",
        expected_host_identity=HOST_IDENTITY,
        expected_gpu_uuid=GPU_UUID,
        expected_filesystem_identity=FILESYSTEM_IDENTITY,
        expected_storage_path=STORAGE_PATH,
        observer=Observer(),
        clock=lambda: now[0],
    )
    return repository, queue, sources, admission, job_id, now


async def _admit_probe(queue: JobQueue, admission: GpuAdmission, job_id: str, now) -> dict:
    admitted = None
    base_time = now[0]
    for offset in range(0, 31, 5):
        now[0] = base_time + timedelta(seconds=offset)
        admitted = await admission.admit(job_id, _observation(now[0]))
        if admitted.get("reason_code") == "ubuntu_observation_replayed":
            state = await queue.repository.get_observation_state("ubuntu")
            raise AssertionError(
                f"probe admission replayed at offset {offset}: now={now[0].isoformat()} "
                f"stored={state['observed_at']} id={state['observation_id']}"
            )
    assert admitted is not None and admitted["state"] == "running", admitted
    return admitted


def _guard_function():
    module = importlib.import_module("gods_mlops.training.claims")
    function = getattr(module, "validate_current_worker_claim", None)
    assert function is not None, "training worker must re-read its active Task 7 lease before CUDA"
    return function


def _require_method(owner, name: str):
    method = getattr(owner, name, None)
    assert method is not None, f"Task 8 queue must expose {name} for fenced result publication"
    return method


def _require_module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        pytest.fail(f"missing implementation module {name}: {error.name}", pytrace=False)


def _result_artifact_store(tmp_path: Path):
    endpoint = os.environ.get("GODS_MLOPS_TEST_S3_ENDPOINT")
    access_key = os.environ.get("GODS_MLOPS_TEST_S3_ACCESS_KEY")
    secret_key = os.environ.get("GODS_MLOPS_TEST_S3_SECRET_KEY")
    bucket = os.environ.get("GODS_MLOPS_TEST_S3_BUCKET")
    if all((endpoint, access_key, secret_key, bucket)):
        objects = DatasetObjectStore(
            endpoint_url=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region="us-east-1",
        )
        store = S3ResultArtifactStore(objects=objects, bucket=bucket, prefix=f"task8-tests/{uuid4()}")
        return store, objects
    return FileResultArtifactStore(root=tmp_path), None


def test_ordinary_worker_rejects_profile_revision_mismatch_before_cuda_or_runner(monkeypatch) -> None:
    worker = _require_module("gods_mlops.training.worker")
    error_type = WorkerAuthorizationError
    job = {
        "job_id": "a0320b59-663c-4cdc-b893-086bb970ea60",
        "state": "running",
        "lease_token": "8ad96890-3434-4f07-85bb-8cde17a2b009",
        "lease_generation": 1,
        "phase": "training",
        "target_phase": None,
        "input_kind": "dataset_version",
        "input_id": "dataset-task8-lock-test",
        "input_sha256": "1" * 64,
        "dataset_version": "dataset-task8-lock-test",
        "model_kind": "clip",
        "config_version": "clip-measured-lock-test-v1",
        "config_sha256": "2" * 64,
    }
    lease = {
        "job_id": job["job_id"],
        "lease_token": job["lease_token"],
        "fencing_token": 1,
        "gpu_uuid": GPU_UUID,
    }
    profile = {
        "profile_state": "measured",
        "config_json": {
            "model_kind": "clip",
            "model_id": "openai/clip-vit-base-patch16",
            "model_revision": "f" * 40,
            "resolution": 224,
            "micro_batch": 2,
            "contrastive_config_version": "task8-explicit-negatives-v1",
        },
    }

    class Repository:
        async def get_active_lease(self, gpu_uuid):
            return lease

        async def get_profile(self, *, phase, model_kind, config_version):
            return profile

        async def checkpoint_identity(self, job_id):
            raise AssertionError("a mismatched profile must fail before checkpoint or CUDA setup")

        async def close(self):
            return None

    class Sources:
        async def close(self):
            return None

    repository = Repository()
    sources = Sources()

    class Queue:
        source_registry = sources

        async def get(self, job_id):
            return job

    async def validate_claim(queue, claim, *, object_store):
        return job, lease

    monkeypatch.setattr(worker, "PostgresJobQueueRepository", lambda **kwargs: repository)
    monkeypatch.setattr(worker, "DatasetSourceRegistry", lambda **kwargs: sources)
    monkeypatch.setattr(worker, "JobQueue", lambda **kwargs: Queue())
    monkeypatch.setattr(worker, "dataset_object_store_from_environment", lambda: object())
    monkeypatch.setattr(worker, "validate_current_worker_claim", validate_claim)
    monkeypatch.setattr(worker, "_verify_claim_environment", lambda claim: None)
    monkeypatch.setattr(worker, "_runner_for", lambda *args: pytest.fail("runner must not start"))
    for name, value in {
        "GODS_MLOPS_DATABASE_URL": "postgresql://test:test@127.0.0.1:15439/gods",
        "GODS_MLOPS_JOB_ID": job["job_id"],
        "GODS_MLOPS_LEASE_TOKEN": job["lease_token"],
        "GODS_MLOPS_IMAGE_ID": "sha256:" + "a" * 64,
        "GODS_MLOPS_GPU_UUID": GPU_UUID,
    }.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(error_type, match="model ID or revision differs from the immutable lock"):
        asyncio.run(worker.run_worker())


def test_worker_start_rechecks_task6_current_source_invalidation_before_cuda(task7_database_url: str) -> None:
    async def exercise() -> None:
        guard = _guard_function()
        repository, queue, sources, admission, _version, sample_id, job_id, now = await _training_run(
            task7_database_url
        )
        base_time = now[0]
        for offset in range(0, 31, 5):
            now[0] = base_time + timedelta(seconds=offset)
            running = await admission.admit(job_id, _observation(now[0]))
        assert running["state"] == "running"
        lease = await repository.get_active_lease(GPU_UUID)
        claim = WorkerClaim.from_admitted_job(
            running, lease, image_id="sha256:" + "a" * 64
        )
        await guard(queue, claim)

        publisher = DatasetPublisher(database_url=task7_database_url, objects=None)
        try:
            await publisher.invalidate_sample(str(sample_id))
        finally:
            await publisher.close()

        with pytest.raises(WorkerAuthorizationError, match="source readiness changed.*source_sample_explicitly_invalidated"):
            await guard(queue, claim)

        worker = _require_module("gods_mlops.training.worker")
        cuda_calls = []
        model_loader_calls = []

        async def bind_after_invalidation(seconds):
            assert await queue.bind_process(job_id, lease["lease_token"], OWNER)

        def runner():
            cuda_calls.append("cuda")
            model_loader_calls.append("from_pretrained")

        with pytest.raises(WorkerAuthorizationError, match="source readiness changed.*source_sample_explicitly_invalidated"):
            await worker._run_after_owner_binding(
                repository,
                claim,
                runner,
                timeout_seconds=2,
                poll_interval_seconds=0.1,
                sleep=bind_after_invalidation,
                revalidate=lambda: worker._worker_claim_at_safe_boundary(queue, claim),
            )
        assert cuda_calls == []
        assert model_loader_calls == []
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_typed_probe_input_keeps_fixture_identity_out_of_published_dataset_queue(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, sources, _admission, _version, _sample_id, _job_id, _now = await _training_run(
            task7_database_url
        )
        config_version = f"task8-typed-probe-{uuid4().hex}"
        await queue.register_profile(
            ExecutionProfile(
                model_kind="detr",
                config_version=config_version,
                phase="probe",
                target_phase="training",
                memory_requirement_mib=16_384,
                artifact_reservation_bytes=64 * 1024**2,
                config={"input_size": 640, "micro_batch": 1, "model_revision": "5650961749fa93567c0d46fc7f43ea4f9e914107"},
                candidate=True,
            )
        )
        probe_input = ProbeInput(
            probe_input_id=f"task8-probe-{uuid4().hex}",
            model_kind="detr",
            target_phase="training",
            config_version=config_version,
            manifest_object_key="probes/task8/detr-probe.json",
            input_sha256="5" * 64,
            object_size_bytes=1024,
            fixture=True,
        )
        first = await queue.submit_probe(
            model_kind="detr", config_version=config_version, probe_input=probe_input
        )
        retry = await queue.submit_probe(
            model_kind="detr", config_version=config_version, probe_input=probe_input
        )
        assert retry == first

        changed = ProbeInput(
            probe_input_id=probe_input.probe_input_id,
            model_kind=probe_input.model_kind,
            target_phase=probe_input.target_phase,
            config_version=probe_input.config_version,
            manifest_object_key=probe_input.manifest_object_key,
            input_sha256="6" * 64,
            object_size_bytes=1024,
            fixture=True,
        )
        changed_job = await queue.submit_probe(
            model_kind="detr", config_version=config_version, probe_input=changed
        )
        explicit_rerun = await queue.submit_probe(
            model_kind="detr", config_version=config_version, probe_input=probe_input, rerun=True
        )
        assert changed_job != first
        assert explicit_rerun != first
        job = await queue.get(first)
        assert job["phase"] == "probe"
        assert job["input_kind"] == "probe_input"
        assert job["dataset_version"] is None
        assert job["source_refs"] == probe_input.as_dict()
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_delayed_worker_with_old_database_fence_cannot_start_after_recovery(
    task7_database_url: str, tmp_path: Path
) -> None:
    async def exercise() -> None:
        guard = _guard_function()
        repository, queue, sources, admission, _version, _sample_id, job_id, now = await _training_run(
            task7_database_url
        )
        base_time = now[0]
        for offset in range(0, 31, 5):
            now[0] = base_time + timedelta(seconds=offset)
            running = await admission.admit(job_id, _observation(now[0]))
        old_lease = await repository.get_active_lease(GPU_UUID)
        old_claim = WorkerClaim.from_admitted_job(
            running, old_lease, image_id="sha256:" + "a" * 64
        )
        assert await queue.bind_process(job_id, old_lease["lease_token"], OWNER)

        now[0] = base_time + timedelta(seconds=35)
        monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)
        released = await monitor.observe(_observation(now[0]))
        assert released["state"] == "waiting_gpu"
        assert await repository.get_active_lease(GPU_UUID) is None
        assert job_id not in await repository.list_released_terminal_artifact_reservations()

        for offset in range(40, 71, 5):
            now[0] = base_time + timedelta(seconds=offset)
            recovered = await admission.admit(job_id, _observation(now[0]))
        assert recovered["state"] == "running"
        new_lease = await repository.get_active_lease(GPU_UUID)
        assert new_lease["fencing_token"] == old_claim.fence + 1
        assert new_lease["lease_token"] != old_claim.lease_token

        with pytest.raises(WorkerAuthorizationError, match="stale"):
            await guard(queue, old_claim)
        store_module = _require_module("gods_mlops.training.artifacts")
        store = store_module.FileResultArtifactStore(root=tmp_path)
        save_artifact = _require_method(queue, "save_result_artifact")
        identity = await repository.checkpoint_identity(job_id)
        with pytest.raises(StaleCheckpointOwnerError, match="lease"):
            await save_artifact(
                store=store,
                job_id=job_id,
                lease_token=old_claim.lease_token,
                identity=identity,
                kind="model",
                payload=b"stale result bytes",
            )
        assert not list(tmp_path.rglob("*.artifact"))
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_result_artifact_commit_is_idempotent_and_completion_keeps_lease_until_observed_exit(
    task7_database_url: str, tmp_path: Path
) -> None:
    async def exercise() -> None:
        store, objects = _result_artifact_store(tmp_path)
        complete_job = _require_method(JobQueue, "complete_owned_job")
        repository, queue, sources, admission, _version, _sample_id, job_id, now = await _training_run(
            task7_database_url
        )
        base_time = now[0]
        save_artifact = _require_method(queue, "save_result_artifact")
        for offset in range(0, 31, 5):
            now[0] = base_time + timedelta(seconds=offset)
            running = await admission.admit(job_id, _observation(now[0]))
        lease = await repository.get_active_lease(GPU_UUID)
        token = lease["lease_token"]
        identity = await repository.checkpoint_identity(job_id)

        first = await save_artifact(
            store=store,
            job_id=job_id,
            lease_token=token,
            identity=identity,
            kind="model",
            payload=b"committed Task 8 model bundle",
        )
        retry = await save_artifact(
            store=store,
            job_id=job_id,
            lease_token=token,
            identity=identity,
            kind="model",
            payload=b"committed Task 8 model bundle",
        )
        assert retry.sha256 == first.sha256
        reservation = await repository.artifact_reservation_for(job_id)
        assert reservation["consumed_bytes"] == first.size_bytes
        assert reservation["state"] == "reserved"

        owner = ProcessIdentity(pid=43122, start_ticks=89123, uid=10001)
        assert await queue.bind_process(job_id, token, owner)
        completed = await queue.complete_owned_job(
            job_id=job_id,
            lease_token=token,
            identity=identity,
            details={"artifact_uri": first.uri, "artifact_sha256": first.sha256},
        )
        assert completed["state"] == "completed"
        assert (await repository.get_active_lease(GPU_UUID))["lease_token"] == token
        assert (await repository.artifact_reservation_for(job_id))["state"] == "settled"

        monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)
        now[0] = base_time + timedelta(seconds=35)
        still_owned = await monitor.observe(_observation(now[0], gpu=(owner,), processes=(owner,)))
        assert still_owned["state"] == "completed"
        assert (await repository.get_active_lease(GPU_UUID))["lease_token"] == token

        now[0] = base_time + timedelta(seconds=40)
        released = await monitor.observe(_observation(now[0]))
        assert released["state"] == "completed"
        assert await repository.get_active_lease(GPU_UUID) is None
        assert await repository.is_next_eligible_job(job_id) is False
        if objects is None:
            assert len(list(tmp_path.rglob("*.artifact"))) == 1
        else:
            assert objects.read_source(
                object_key=first.object_key,
                sha256_digest=first.sha256,
                size_bytes=first.size_bytes,
            ) == b"committed Task 8 model bundle"
            objects.delete_object(object_key=first.object_key)
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_s3_checkpoint_and_result_use_task7_fence_and_shared_global_reservation(
    task7_database_url: str, tmp_path: Path
) -> None:
    async def exercise() -> None:
        artifact_store, objects = _result_artifact_store(tmp_path)
        if objects is None:
            pytest.skip("loopback-only Task 8 S3 endpoint is not configured")
        result_prefix = f"task8-tests/{uuid4()}"
        artifact_store = S3ResultArtifactStore(
            objects=objects,
            bucket=os.environ["GODS_MLOPS_TEST_S3_BUCKET"],
            prefix=result_prefix,
        )
        checkpoint_store = S3CheckpointStore(
            objects=objects,
            bucket=os.environ["GODS_MLOPS_TEST_S3_BUCKET"],
            prefix=f"task8-tests/{uuid4()}",
        )
        repository, queue, sources, admission, _version, _sample_id, job_id, now = await _training_run(
            task7_database_url
        )
        base_time = now[0]
        baseline = await _storage_bytes(task7_database_url)
        for offset in range(0, 31, 5):
            now[0] = base_time + timedelta(seconds=offset)
            running = await admission.admit(job_id, _observation(now[0]))
        lease = await repository.get_active_lease(GPU_UUID)
        identity = await repository.checkpoint_identity(job_id)
        reservation = await repository.artifact_reservation_for(job_id)
        assert await _storage_bytes(task7_database_url) - baseline == reservation["reserved_bytes"]

        class LostCheckpointAck:
            lost = False

            def write_immutable(self, **kwargs):
                objects.write_immutable(**kwargs)
                if not self.lost:
                    self.lost = True
                    raise OSError("lost S3 checkpoint acknowledgement")

            def read_source(self, **kwargs):
                return objects.read_source(**kwargs)

            def delete_object(self, **kwargs):
                return objects.delete_object(**kwargs)

        lost_checkpoint_store = S3CheckpointStore(
            objects=LostCheckpointAck(),
            bucket=os.environ["GODS_MLOPS_TEST_S3_BUCKET"],
            prefix=checkpoint_store._prefix,
        )
        payload = b"optimizer and model checkpoint step three"
        with pytest.raises(OSError, match="lost S3 checkpoint acknowledgement"):
            await queue.save_checkpoint(
                store=lost_checkpoint_store,
                job_id=job_id,
                lease_token=lease["lease_token"],
                identity=identity,
                payload=payload,
            )
        assert await queue.load_checkpoint(store=checkpoint_store, job_id=job_id) is None
        checkpoint = await queue.save_checkpoint(
            store=checkpoint_store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=payload,
        )
        resumed = await queue.load_checkpoint(store=checkpoint_store, job_id=job_id)
        assert resumed.payload == payload
        assert resumed.sha256 == checkpoint.sha256

        class LostResultAck:
            lost = False

            def write_immutable(self, **kwargs):
                objects.write_immutable(**kwargs)
                if not self.lost:
                    self.lost = True
                    raise OSError("lost S3 result acknowledgement")

            def read_source(self, **kwargs):
                return objects.read_source(**kwargs)

        lost_result_store = S3ResultArtifactStore(
            objects=LostResultAck(),
            bucket=os.environ["GODS_MLOPS_TEST_S3_BUCKET"],
            prefix=result_prefix,
        )
        result_payload = b"measured model result bundle"
        with pytest.raises(OSError, match="lost S3 result acknowledgement"):
            await queue.save_result_artifact(
                store=lost_result_store,
                job_id=job_id,
                lease_token=lease["lease_token"],
                identity=identity,
                kind="model",
                payload=result_payload,
            )
        result_artifact = await queue.save_result_artifact(
            store=artifact_store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            kind="model",
            payload=result_payload,
        )
        assert objects.read_source(
            object_key=result_artifact.object_key,
            sha256_digest=result_artifact.sha256,
            size_bytes=result_artifact.size_bytes,
        ) == result_payload
        assert await queue.bind_process(
            job_id, lease["lease_token"], ProcessIdentity(43122, 89123, 10001)
        )
        complete = await queue.complete_owned_job(
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            details={"artifact_uri": result_artifact.uri, "artifact_sha256": result_artifact.sha256},
        )
        assert complete["state"] == "completed"
        settled = await repository.artifact_reservation_for(job_id)
        assert settled["state"] == "settled"
        assert settled["consumed_bytes"] == checkpoint.size_bytes + result_artifact.size_bytes
        assert await _storage_bytes(task7_database_url) == baseline + settled["consumed_bytes"]

        objects.delete_object(object_key=result_artifact.object_key)
        objects.delete_object(object_key=checkpoint.uri.removeprefix(f"s3://{os.environ['GODS_MLOPS_TEST_S3_BUCKET']}/"))
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_s3_result_put_lost_ack_remains_charged_until_verified_terminal_cleanup(
    task7_database_url: str,
) -> None:
    if not all(os.environ.get(name) for name in (
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )):
        pytest.skip("loopback-only Task 8 S3 endpoint is not configured")

    async def exercise() -> None:
        endpoint = os.environ["GODS_MLOPS_TEST_S3_ENDPOINT"]
        access = os.environ["GODS_MLOPS_TEST_S3_ACCESS_KEY"]
        secret = os.environ["GODS_MLOPS_TEST_S3_SECRET_KEY"]
        bucket = os.environ["GODS_MLOPS_TEST_S3_BUCKET"]
        objects = DatasetObjectStore(
            endpoint_url=endpoint, access_key=access, secret_key=secret, bucket=bucket, region="us-east-1"
        )
        repository, queue, sources, admission, job_id, now = await _probe_run(task7_database_url)
        baseline = await _storage_bytes(task7_database_url)
        await _admit_probe(queue, admission, job_id, now)
        lease = await repository.get_active_lease(GPU_UUID)
        identity = await repository.checkpoint_identity(job_id)
        payload = b"immutable result whose S3 verification acknowledgement was lost"
        delegate = S3ResultArtifactStore(objects=objects, bucket=bucket, prefix=f"task8-pending/{uuid4()}")

        class LostReadback:
            lost = False

            @property
            def _bucket(self):
                return delegate._bucket

            @property
            def _prefix(self):
                return delegate._prefix

            def prepare(self, **kwargs):
                return delegate.prepare(**kwargs)

            def commit(self, prepared):
                if not self.lost:
                    self.lost = True
                    objects.write_immutable(
                        object_key=prepared.object_key,
                        content=prepared.payload,
                        sha256_digest=prepared.sha256,
                        content_type="application/octet-stream",
                    )
                    raise OSError("lost result readback acknowledgement after S3 put")
                return delegate.commit(prepared)

        store = LostReadback()
        expected = store.prepare(
            identity=identity,
            kind="model",
            payload=payload,
            reservation_bytes=32 * 1024**2,
        )
        with pytest.raises(OSError, match="lost result readback acknowledgement"):
            await queue.save_result_artifact(
                store=store,
                job_id=job_id,
                lease_token=lease["lease_token"],
                identity=identity,
                kind="model",
                payload=payload,
            )
        reservation = await repository.artifact_reservation_for(job_id)
        assert reservation["consumed_bytes"] == len(payload)
        async with repository._pool.acquire() as connection:
            pending = await connection.fetchrow(
                """SELECT details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type='artifact_write_pending'""",
                job_id,
            )
        pending_details = json.loads(pending["details"])
        assert pending_details["uri"] == expected.uri
        assert pending_details["sha256"] == expected.sha256
        assert pending_details["size_bytes"] == len(payload)
        assert pending_details["identity"] == identity.as_dict()
        assert objects.read_source(
            object_key=expected.object_key, sha256_digest=expected.sha256, size_bytes=len(payload)
        ) == payload

        failed = await queue.record_probe_measurement(
            job_id=job_id,
            lease_token=lease["lease_token"],
            exit_code=1,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
            optimizer_steps=0,
            checkpoint_resumed=False,
            checkpoint_sha256=None,
            inference_steps=0,
            verification_details={"passed": False, "failure": "lost_result_readback_ack"},
        )
        assert failed["result_state"] == "failed"
        assert (await repository.artifact_reservation_for(job_id))["state"] == "settled"
        assert await _storage_bytes(task7_database_url) == baseline + len(payload)
        released = await repository.release_after_observed_exit(
            job_id=job_id,
            lease_token=lease["lease_token"],
            observation=ResourceObservation.from_dict(_observation(datetime.now(UTC))),
        )
        assert released is True

        foreign_payload = b"neighboring unowned S3 object"
        foreign_key = f"{expected.object_key}.foreign"
        objects.write_immutable(
            object_key=foreign_key,
            content=foreign_payload,
            sha256_digest=__import__("hashlib").sha256(foreign_payload).hexdigest(),
            content_type="application/octet-stream",
        )

        await queue.cleanup_pending_artifact_writes(object_store=objects, job_id=job_id)
        assert await _storage_bytes(task7_database_url) == baseline
        with pytest.raises(FileNotFoundError):
            objects.read_source(
                object_key=expected.object_key,
                sha256_digest=expected.sha256,
                size_bytes=len(payload),
            )
        assert objects.read_source(
            object_key=foreign_key,
            sha256_digest=__import__("hashlib").sha256(foreign_payload).hexdigest(),
            size_bytes=len(foreign_payload),
        ) == foreign_payload
        objects.delete_object(object_key=foreign_key)
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_new_fence_recovers_committed_probe_result_without_retraining_or_double_charge(
    task7_database_url: str, monkeypatch
) -> None:
    required = (
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )
    if any(not os.environ.get(name) for name in required):
        pytest.skip("loopback-only Task 8 S3 endpoint is not configured")

    async def exercise() -> None:
        from gods_mlops.datasets.manifest import canonical_json

        endpoint = os.environ["GODS_MLOPS_TEST_S3_ENDPOINT"]
        access = os.environ["GODS_MLOPS_TEST_S3_ACCESS_KEY"]
        secret = os.environ["GODS_MLOPS_TEST_S3_SECRET_KEY"]
        bucket = os.environ["GODS_MLOPS_TEST_S3_BUCKET"]
        objects = DatasetObjectStore(
            endpoint_url=endpoint, access_key=access, secret_key=secret, bucket=bucket, region="us-east-1"
        )
        writes: list[str] = []
        original_write = objects.write_immutable

        def count_write(*, object_key, content, sha256_digest, content_type):
            writes.append(object_key)
            original_write(
                object_key=object_key,
                content=content,
                sha256_digest=sha256_digest,
                content_type=content_type,
            )

        objects.write_immutable = count_write
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        sources = DatasetSourceRegistry(database_url=task7_database_url)
        queue = JobQueue(repository=repository, sources=sources)
        model = importlib.import_module("gods_mlops.training.contracts").locked_model("detr")
        config_version = f"task8-recovery-{uuid4().hex}"
        profile = ExecutionProfile(
            model_kind="detr",
            config_version=config_version,
            phase="probe",
            target_phase="training",
            memory_requirement_mib=8_192,
            artifact_reservation_bytes=96 * 1024**2,
            config={
                "model_id": model.model_id,
                "model_revision": model.revision,
                "input_size": 640,
                "micro_batch": 1,
            },
            candidate=True,
        )
        await queue.register_profile(profile)
        probe_input_id = f"task8-recovery-{uuid4().hex}"
        manifest = {
            "schema_version": 1,
            "fixture": True,
            "phase": "probe",
            "model_kind": "detr",
            "input_kind": "probe_input",
            "input_id": probe_input_id,
            "config_version": config_version,
        }
        manifest_payload = canonical_json(manifest)
        manifest_sha = sha256(manifest_payload).hexdigest()
        manifest_key = f"probe-inputs/task8-recovery/{uuid4().hex}.json"
        objects.write_immutable(
            object_key=manifest_key,
            content=manifest_payload,
            sha256_digest=manifest_sha,
            content_type="application/json",
        )
        probe_input = ProbeInput(
            probe_input_id=probe_input_id,
            model_kind="detr",
            target_phase="training",
            config_version=config_version,
            manifest_object_key=manifest_key,
            input_sha256=manifest_sha,
            object_size_bytes=len(manifest_payload),
            fixture=True,
        )
        job_id = await queue.submit_probe(
            model_kind="detr", config_version=config_version, probe_input=probe_input
        )
        baseline = await _storage_bytes(task7_database_url)
        now = [datetime.now(UTC)]

        class Observer:
            async def observe(self):
                return _observation(now[0] + timedelta(milliseconds=100))

        admission = GpuAdmission(
            repository=repository,
            queue=queue,
            expected_node_id="ubuntu",
            expected_host_identity=HOST_IDENTITY,
            expected_gpu_uuid=GPU_UUID,
            expected_filesystem_identity=FILESYSTEM_IDENTITY,
            expected_storage_path=STORAGE_PATH,
            observer=Observer(),
            clock=lambda: now[0],
        )
        first_job = await _admit_probe(queue, admission, job_id, now)
        first_lease = await repository.get_active_lease(GPU_UUID)
        identity = await repository.checkpoint_identity(job_id)
        checkpoint_store = S3CheckpointStore(objects=objects, bucket=bucket)
        result_store = S3ResultArtifactStore(objects=objects, bucket=bucket)
        checkpoint = await queue.save_checkpoint(
            store=checkpoint_store,
            job_id=job_id,
            lease_token=first_lease["lease_token"],
            identity=identity,
            payload=b"verified generation-one optimizer checkpoint",
        )
        result_payload = b"immutable generation-one probe result bundle"
        measurements = {
            "peak_vram_allocated_mib": 7_200,
            "peak_vram_reserved_mib": 8_000,
            "elapsed_seconds": 4.2,
            "optimizer_steps": 3,
            "inference_steps": 0,
            "checkpoint_resumed": True,
            "initial_weight_sha256": "3" * 64,
            "final_weight_sha256": "4" * 64,
            "model_id": model.model_id,
            "model_revision": model.revision,
            "losses": [1.1, 0.8, 0.6],
        }
        artifact = await queue.save_result_artifact(
            store=result_store,
            job_id=job_id,
            lease_token=first_lease["lease_token"],
            identity=identity,
            kind="model",
            payload=result_payload,
            runtime_measurements=measurements,
        )
        before_recovery = await repository.artifact_reservation_for(job_id)
        assert before_recovery["state"] == "reserved"
        assert before_recovery["consumed_bytes"] == checkpoint.size_bytes + artifact.size_bytes
        assert len(await repository.result_artifacts_for(job_id)) == 1
        result_write_count = len(writes)

        assert await queue.bind_process(job_id, first_lease["lease_token"], OWNER)
        monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)
        admission_base = now[0]
        now[0] = admission_base + timedelta(seconds=35)
        released = await monitor.observe(_observation(now[0]))
        assert released["state"] == "waiting_gpu"
        assert await repository.get_active_lease(GPU_UUID) is None
        for offset in range(40, 71, 5):
            now[0] = admission_base + timedelta(seconds=offset)
            recovered_job = await admission.admit(job_id, _observation(now[0]))
        assert recovered_job["state"] == "running"
        second_lease = await repository.get_active_lease(GPU_UUID)
        assert second_lease["fencing_token"] == first_lease["fencing_token"] + 1
        assert second_lease["lease_token"] != first_lease["lease_token"]
        assert (await repository.artifact_reservation_for(job_id))["consumed_bytes"] == before_recovery["consumed_bytes"]

        worker = _require_module("gods_mlops.training.worker")
        runner_calls: list[tuple] = []

        def fail_if_runner_starts(*args):
            runner_calls.append(args)
            raise AssertionError("recovery must not import or execute the model runner")

        monkeypatch.setattr(worker, "_runner_for", fail_if_runner_starts)
        monkeypatch.setattr(worker, "_verify_claim_environment", lambda claim: None)
        monkeypatch.setattr(worker, "dataset_object_store_from_environment", lambda: objects)
        for name, value in {
            "GODS_MLOPS_DATABASE_URL": task7_database_url,
            "GODS_MLOPS_JOB_ID": job_id,
            "GODS_MLOPS_LEASE_TOKEN": second_lease["lease_token"],
            "GODS_MLOPS_IMAGE_ID": "sha256:" + "a" * 64,
            "GODS_MLOPS_GPU_UUID": GPU_UUID,
            "GODS_MLOPS_S3_ENDPOINT_URL": endpoint,
            "GODS_MLOPS_S3_ACCESS_KEY": access,
            "GODS_MLOPS_S3_SECRET_KEY": secret,
            "GODS_MLOPS_S3_BUCKET": bucket,
            "GODS_MLOPS_S3_REGION": "us-east-1",
            "GODS_MLOPS_MODEL_LOCK": str(Path(__file__).resolve().parents[2] / "models" / "lock.json"),
        }.items():
            monkeypatch.setenv(name, value)
        assert await worker.run_worker() == 0
        assert runner_calls == []
        assert len(writes) == result_write_count
        assert len(await repository.result_artifacts_for(job_id)) == 1
        reservation = await repository.artifact_reservation_for(job_id)
        assert reservation["state"] == "settled"
        assert reservation["consumed_bytes"] == before_recovery["consumed_bytes"]
        assert await _storage_bytes(task7_database_url) == baseline + checkpoint.size_bytes + artifact.size_bytes
        assert (await queue.get(job_id))["state"] == "completed"
        measurement = await repository.profile_measurement_for_job(job_id)
        assert measurement is not None and measurement["result_state"] == "succeeded"

        assert await repository.release_after_observed_exit(
            job_id=job_id,
            lease_token=second_lease["lease_token"],
            observation=ResourceObservation.from_dict(_observation(now[0] + timedelta(seconds=1))),
        )
        objects.delete_object(object_key=manifest_key)
        objects.delete_object(object_key=checkpoint.uri.removeprefix(f"s3://{bucket}/"))
        objects.delete_object(object_key=artifact.object_key)
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_s3_checkpoint_aba_replacement_recharges_pruned_object_lifetime(
    task7_database_url: str,
) -> None:
    required = (
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )
    if any(not os.environ.get(name) for name in required):
        pytest.skip("loopback-only Task 8 S3 endpoint is not configured")

    async def exercise() -> None:
        endpoint = os.environ["GODS_MLOPS_TEST_S3_ENDPOINT"]
        access = os.environ["GODS_MLOPS_TEST_S3_ACCESS_KEY"]
        secret = os.environ["GODS_MLOPS_TEST_S3_SECRET_KEY"]
        bucket = os.environ["GODS_MLOPS_TEST_S3_BUCKET"]
        objects = DatasetObjectStore(
            endpoint_url=endpoint, access_key=access, secret_key=secret, bucket=bucket, region="us-east-1"
        )
        repository, queue, sources, admission, job_id, now = await _probe_run(task7_database_url)
        baseline = await _storage_bytes(task7_database_url)
        await _admit_probe(queue, admission, job_id, now)
        lease = await repository.get_active_lease(GPU_UUID)
        identity = await repository.checkpoint_identity(job_id)
        store = S3CheckpointStore(objects=objects, bucket=bucket, prefix=f"task8-aba/{uuid4()}")
        payload_a = b"checkpoint-A-retained-after-its-recreated-lifetime"
        payload_b = b"checkpoint-B-temporary-version-with-distinct-content"

        first_a = await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=payload_a,
        )
        await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=payload_b,
        )
        after_b = await repository.artifact_reservation_for(job_id)
        assert after_b["consumed_bytes"] == len(payload_b)
        with pytest.raises(FileNotFoundError):
            objects.read_source(
                object_key=first_a.uri.removeprefix(f"s3://{bucket}/"),
                sha256_digest=first_a.sha256,
                size_bytes=first_a.size_bytes,
            )

        recreated_a = await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=payload_a,
        )
        reservation = await repository.artifact_reservation_for(job_id)
        assert recreated_a.uri == first_a.uri
        assert reservation["consumed_bytes"] == recreated_a.size_bytes
        assert (await queue.load_checkpoint(store=store, job_id=job_id)).payload == payload_a

        failed = await queue.record_probe_measurement(
            job_id=job_id,
            lease_token=lease["lease_token"],
            exit_code=1,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
            optimizer_steps=0,
            checkpoint_resumed=False,
            checkpoint_sha256=None,
            inference_steps=0,
            verification_details={"passed": False, "failure": "test_checkpoint_aba_settlement"},
        )
        assert failed["result_state"] == "failed"
        assert await _storage_bytes(task7_database_url) == baseline + len(payload_a)
        assert objects.read_source(
            object_key=recreated_a.uri.removeprefix(f"s3://{bucket}/"),
            sha256_digest=recreated_a.sha256,
            size_bytes=recreated_a.size_bytes,
        ) == payload_a
        objects.delete_object(object_key=recreated_a.uri.removeprefix(f"s3://{bucket}/"))
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


async def _exercise_round1_checkpoint_prune_recreation(
    database_url: str, *, prune_was_already_completed: bool
) -> None:
    endpoint = os.environ["GODS_MLOPS_TEST_S3_ENDPOINT"]
    access = os.environ["GODS_MLOPS_TEST_S3_ACCESS_KEY"]
    secret = os.environ["GODS_MLOPS_TEST_S3_SECRET_KEY"]
    bucket = os.environ["GODS_MLOPS_TEST_S3_BUCKET"]
    objects = DatasetObjectStore(
        endpoint_url=endpoint, access_key=access, secret_key=secret, bucket=bucket, region="us-east-1"
    )
    repository, queue, sources, admission, job_id, now = await _probe_run(database_url)
    object_keys: set[str] = set()
    try:
        baseline = await _storage_bytes(database_url)
        await _admit_probe(queue, admission, job_id, now)
        lease = await repository.get_active_lease(GPU_UUID)
        identity = await repository.checkpoint_identity(job_id)
        store = S3CheckpointStore(objects=objects, bucket=bucket, prefix=f"task8-r1-compat/{uuid4()}")
        payload_a = b"round-one-checkpoint-A-retained-after-recreation"
        payload_b = b"round-one-checkpoint-B-temporary-replacement"

        first_a = await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=payload_a,
        )
        object_keys.add(first_a.uri.removeprefix(f"s3://{bucket}/"))
        if prune_was_already_completed:
            second = await queue.save_checkpoint(
                store=store,
                job_id=job_id,
                lease_token=lease["lease_token"],
                identity=identity,
                payload=payload_b,
            )
            object_keys.add(second.uri.removeprefix(f"s3://{bucket}/"))
            with pytest.raises(FileNotFoundError):
                objects.read_source(
                    object_key=first_a.uri.removeprefix(f"s3://{bucket}/"),
                    sha256_digest=first_a.sha256,
                    size_bytes=first_a.size_bytes,
                )
        else:
            original_prune_previous = store.prune_previous
            failed_once = False

            def lose_first_delete_ack(prepared):
                nonlocal failed_once
                if not failed_once:
                    failed_once = True
                    raise OSError("simulate round-one checkpoint prune acknowledgement loss")
                original_prune_previous(prepared)

            prepared_b = store.prepare(
                identity=identity,
                payload=payload_b,
                reservation_bytes=96 * 1024**2,
            )
            object_keys.add(prepared_b.object_key)
            store.prune_previous = lose_first_delete_ack
            with pytest.raises(OSError, match="acknowledgement loss"):
                await queue.save_checkpoint(
                    store=store,
                    job_id=job_id,
                    lease_token=lease["lease_token"],
                    identity=identity,
                    payload=payload_b,
                )
            store.prune_previous = original_prune_previous

        legacy_rows = await _rewrite_checkpoint_history_as_round1(
            database_url,
            job_id=job_id,
            uri=first_a.uri,
            sha256_digest=first_a.sha256,
        )
        assert "write_lifetime_id" not in legacy_rows["artifact_write_pending"]
        assert legacy_rows["artifact_write_pending"]["operation_id"]
        assert "write_lifetime_id" not in legacy_rows["checkpoint_committed"]
        assert legacy_rows["checkpoint_committed"]["operation_id"]
        assert "write_lifetime_id" not in legacy_rows["checkpoint_prune_pending"]
        assert "operation_id" not in legacy_rows["checkpoint_prune_pending"]
        if prune_was_already_completed:
            assert "checkpoint_pruned" in legacy_rows
            assert "write_lifetime_id" not in legacy_rows["checkpoint_pruned"]
            assert "operation_id" not in legacy_rows["checkpoint_pruned"]
        else:
            assert "checkpoint_pruned" not in legacy_rows

        recreated_a = await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=payload_a,
        )
        after_recreation = await repository.artifact_reservation_for(job_id)
        assert recreated_a.uri == first_a.uri
        assert after_recreation["consumed_bytes"] == recreated_a.size_bytes
        assert objects.read_source(
            object_key=recreated_a.uri.removeprefix(f"s3://{bucket}/"),
            sha256_digest=recreated_a.sha256,
            size_bytes=recreated_a.size_bytes,
        ) == payload_a

        # A retry of the same currently committed bytes remains idempotent and charged once.
        repeated = await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=payload_a,
        )
        assert repeated.uri == recreated_a.uri
        assert (await repository.artifact_reservation_for(job_id))["consumed_bytes"] == recreated_a.size_bytes

        failed = await queue.record_probe_measurement(
            job_id=job_id,
            lease_token=lease["lease_token"],
            exit_code=1,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
            optimizer_steps=0,
            checkpoint_resumed=False,
            checkpoint_sha256=None,
            inference_steps=0,
            verification_details={"passed": False, "failure": "test_round1_checkpoint_prune_compatibility"},
        )
        assert failed["result_state"] == "failed"
        reservation = await repository.artifact_reservation_for(job_id)
        assert reservation["state"] == "settled"
        assert reservation["consumed_bytes"] == len(payload_a)
        assert await _storage_bytes(database_url) == baseline + len(payload_a)
    finally:
        for object_key in object_keys:
            objects.delete_object(object_key=object_key)
        if "recreated_a" in locals():
            objects.delete_object(object_key=recreated_a.uri.removeprefix(f"s3://{bucket}/"))
        await queue.close()
        await repository.close()
        await sources.close()


def test_s3_round1_completed_prune_recreation_gets_a_fresh_charge(task7_database_url: str) -> None:
    required = (
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )
    if any(not os.environ.get(name) for name in required):
        pytest.skip("loopback-only Task 8 S3 endpoint is not configured")
    asyncio.run(_exercise_round1_checkpoint_prune_recreation(task7_database_url, prune_was_already_completed=True))


def test_s3_round1_pending_prune_recreation_gets_a_fresh_charge(task7_database_url: str) -> None:
    required = (
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )
    if any(not os.environ.get(name) for name in required):
        pytest.skip("loopback-only Task 8 S3 endpoint is not configured")
    asyncio.run(_exercise_round1_checkpoint_prune_recreation(task7_database_url, prune_was_already_completed=False))


def test_file_checkpoint_replacements_account_only_retained_bytes_and_keep_replacing(
    task7_database_url: str, tmp_path: Path, monkeypatch
) -> None:
    async def exercise() -> None:
        from gods_mlops.jobs.checkpoints import FileCheckpointStore

        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        sources = DatasetSourceRegistry(database_url=task7_database_url)
        queue = JobQueue(repository=repository, sources=sources)
        model = importlib.import_module("gods_mlops.training.contracts").locked_model("detr")
        config_version = f"task8-file-checkpoint-{uuid4().hex}"
        await queue.register_profile(
            ExecutionProfile(
                model_kind="detr",
                config_version=config_version,
                phase="probe",
                target_phase="training",
                memory_requirement_mib=8_192,
                artifact_reservation_bytes=9 * 1024**2,
                config={
                    "model_id": model.model_id,
                    "model_revision": model.revision,
                    "input_size": 640,
                    "micro_batch": 1,
                },
                candidate=True,
            )
        )
        job_id = await queue.submit_probe(
            model_kind="detr",
            config_version=config_version,
            probe_input_id=f"task8-file-checkpoint-{uuid4().hex}",
            input_sha256="5" * 64,
        )
        baseline = await _storage_bytes(task7_database_url)
        now = [datetime.now(UTC)]

        class Observer:
            async def observe(self):
                return _observation(now[0] + timedelta(milliseconds=100))

        admission = GpuAdmission(
            repository=repository,
            queue=queue,
            expected_node_id="ubuntu",
            expected_host_identity=HOST_IDENTITY,
            expected_gpu_uuid=GPU_UUID,
            expected_filesystem_identity=FILESYSTEM_IDENTITY,
            expected_storage_path=STORAGE_PATH,
            observer=Observer(),
            clock=lambda: now[0],
        )
        await _admit_probe(queue, admission, job_id, now)
        lease = await repository.get_active_lease(GPU_UUID)
        assert await queue.bind_process(job_id, lease["lease_token"], OWNER)
        assert await repository.list_released_terminal_artifact_reservations() == []
        identity = await repository.checkpoint_identity(job_id)
        store = FileCheckpointStore(root=tmp_path)
        checkpoint_directory = tmp_path / job_id
        payload_size = 2 * 1024**2
        latest = None
        for generation in range(5):
            payload = bytes([generation + 1]) * payload_size
            latest = await queue.save_checkpoint(
                store=store,
                job_id=job_id,
                lease_token=lease["lease_token"],
                identity=identity,
                payload=payload,
            )
            retained_bytes = sum(path.stat().st_size for path in checkpoint_directory.iterdir())
            reservation = await repository.artifact_reservation_for(job_id)
            assert reservation["consumed_bytes"] == retained_bytes
            assert len(list(checkpoint_directory.glob("*.checkpoint"))) == 1
            assert len(list(checkpoint_directory.glob("*.json"))) == 1
        assert latest is not None

        writer_operation_id = str(uuid4())
        writer_attempt_id = str(uuid4())
        async with repository._pool.acquire() as connection:
            await connection.execute(
                """INSERT INTO gods_mlops_job_events(
                       job_id,event_type,state,fencing_token,details
                   ) VALUES($1::uuid,'artifact_write_started','running',$2,$3::jsonb)""",
                job_id,
                lease["fencing_token"],
                json.dumps({
                    "operation_id": writer_operation_id,
                    "writer_attempt_id": writer_attempt_id,
                    "lease_token": lease["lease_token"],
                    "fencing_token": lease["fencing_token"],
                }),
            )

        failed = await queue.record_probe_measurement(
            job_id=job_id,
            lease_token=lease["lease_token"],
            exit_code=1,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
            optimizer_steps=0,
            checkpoint_resumed=False,
            checkpoint_sha256=None,
            inference_steps=0,
            verification_details={"passed": False, "failure": "test_file_checkpoint_quota"},
        )
        assert failed["result_state"] == "failed"
        retained_bytes = sum(path.stat().st_size for path in checkpoint_directory.iterdir())
        reservation = await repository.artifact_reservation_for(job_id)
        assert reservation["state"] == "reserved"
        assert reservation["consumed_bytes"] == retained_bytes
        assert await repository.list_released_terminal_artifact_reservations() == []
        async with repository._pool.acquire() as connection:
            failure_event_before = await connection.fetchrow(
                """SELECT event_id,reason_code,details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type='probe_measurement_failed'
                   ORDER BY event_id DESC LIMIT 1""",
                job_id,
            )
        original_settle = repository.settle_artifact_reservation
        fail_next_settlement = True

        async def fail_once(requested_job_id):
            nonlocal fail_next_settlement
            if fail_next_settlement:
                fail_next_settlement = False
                raise ConnectionError("temporary settlement connection failure")
            return await original_settle(requested_job_id)

        monkeypatch.setattr(repository, "settle_artifact_reservation", fail_once)

        monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)
        now[0] += timedelta(seconds=5)
        still_owned = await monitor.observe(
            _observation(now[0], gpu=(OWNER,), processes=(OWNER,))
        )
        assert still_owned["state"] == "failed"
        assert (await repository.get_active_lease(GPU_UUID))["lease_token"] == lease["lease_token"]
        assert (await repository.artifact_reservation_for(job_id))["state"] == "reserved"

        now[0] += timedelta(seconds=5)
        released = await monitor.observe(_observation(now[0]))
        assert released["state"] == "failed"
        assert await repository.get_active_lease(GPU_UUID) is None
        assert fail_next_settlement is False
        assert job_id in await repository.list_released_terminal_artifact_reservations()
        reservation = await repository.artifact_reservation_for(job_id)
        assert reservation["state"] == "reserved"

        # A restarted monitor sees the durable released terminal reservation even
        # though the live lease row is gone; the active writer still defers refund.
        monitor_after_restart = GpuJobMonitor(
            repository=repository,
            queue=queue,
            admission=admission,
        )
        now[0] += timedelta(seconds=5)
        assert await monitor_after_restart.observe(_observation(now[0])) is None
        assert (await repository.artifact_reservation_for(job_id))["state"] == "reserved"

        await repository.record_artifact_writer_quiescent(
            job_id=job_id,
            operation_id=writer_operation_id,
            writer_attempt_id=writer_attempt_id,
        )
        now[0] += timedelta(seconds=5)
        assert await monitor_after_restart.observe(_observation(now[0])) is None
        reservation = await repository.artifact_reservation_for(job_id)
        assert reservation["state"] == "settled"
        assert reservation["consumed_bytes"] == retained_bytes
        assert await _storage_bytes(task7_database_url) == baseline + retained_bytes
        assert await repository.list_released_terminal_artifact_reservations() == []
        job_after_settlement = await queue.get(job_id)
        assert job_after_settlement["state"] == "failed"
        assert job_after_settlement["reason_code"] == failure_event_before["reason_code"]
        async with repository._pool.acquire() as connection:
            failure_event_after = await connection.fetchrow(
                """SELECT event_id,reason_code,details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type='probe_measurement_failed'
                   ORDER BY event_id DESC LIMIT 1""",
                job_id,
            )
            release_count = await connection.fetchval(
                """SELECT count(*) FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type='lease_owner_released'""",
                job_id,
            )
            settled_count = await connection.fetchval(
                """SELECT count(*) FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type='artifact_reservation_settled'""",
                job_id,
            )
        assert failure_event_after == failure_event_before
        assert release_count == 1
        assert settled_count == 1

        settled_bytes = await _storage_bytes(task7_database_url)
        await queue.settle_artifact_reservation(job_id)
        assert (await repository.artifact_reservation_for(job_id))["state"] == "settled"
        assert await _storage_bytes(task7_database_url) == settled_bytes
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_terminal_reservation_query_requires_release_evidence_after_a_worker_lease(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository, queue, sources, admission, job_id, now = await _probe_run(task7_database_url)
        assert (await queue.get(job_id))["phase"] == "probe"
        base_time = now[0]
        for offset in range(0, 31, 5):
            now[0] = base_time + timedelta(seconds=offset)
            await admission.admit(job_id, _observation(now[0]))

        lease = await repository.get_active_lease(GPU_UUID)
        assert lease is not None
        assert lease["fencing_token"] > 0
        assert await repository.list_released_terminal_artifact_reservations() == []

        result = await queue.record_probe_measurement(
            job_id=job_id,
            lease_token=lease["lease_token"],
            exit_code=1,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
            optimizer_steps=0,
            checkpoint_resumed=False,
            checkpoint_sha256=None,
            inference_steps=0,
            verification_details={"passed": False, "failure": "test_no_release_proof"},
        )
        assert result["result_state"] == "failed"
        assert await repository.list_released_terminal_artifact_reservations() == []

        async with repository._pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM gods_mlops_gpu_leases WHERE job_id=$1::uuid",
                job_id,
            )
            await connection.execute(
                """UPDATE gods_mlops_jobs
                   SET lease_token=NULL,lease_expires_at=NULL,owner_pid=NULL,
                       owner_start_ticks=NULL,owner_uid=NULL
                   WHERE job_id=$1::uuid""",
                job_id,
            )

        failed_job = await queue.get(job_id)
        assert failed_job["state"] == "failed"
        assert failed_job["lease_generation"] == lease["fencing_token"]
        assert await repository.get_active_lease(GPU_UUID) is None
        assert await repository.list_released_terminal_artifact_reservations() == []
        assert (await repository.artifact_reservation_for(job_id))["state"] == "reserved"
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_s3_checkpoint_delete_failure_keeps_both_versions_charged_until_retry(
    task7_database_url: str,
) -> None:
    if not all(os.environ.get(name) for name in (
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )):
        pytest.skip("loopback-only Task 8 S3 endpoint is not configured")

    async def exercise() -> None:
        endpoint = os.environ["GODS_MLOPS_TEST_S3_ENDPOINT"]
        access = os.environ["GODS_MLOPS_TEST_S3_ACCESS_KEY"]
        secret = os.environ["GODS_MLOPS_TEST_S3_SECRET_KEY"]
        bucket = os.environ["GODS_MLOPS_TEST_S3_BUCKET"]
        objects = DatasetObjectStore(
            endpoint_url=endpoint, access_key=access, secret_key=secret, bucket=bucket, region="us-east-1"
        )
        repository, queue, sources, admission, job_id, now = await _probe_run(task7_database_url)
        baseline = await _storage_bytes(task7_database_url)
        await _admit_probe(queue, admission, job_id, now)
        lease = await repository.get_active_lease(GPU_UUID)
        identity = await repository.checkpoint_identity(job_id)
        delegate = S3CheckpointStore(objects=objects, bucket=bucket, prefix=f"task8-prune/{uuid4()}")

        class LoseFirstDelete:
            failed = False

            def prepare(self, **kwargs):
                return delegate.prepare(**kwargs)

            def commit(self, prepared):
                return delegate.commit(prepared)

            def load_uri(self, *args, **kwargs):
                return delegate.load_uri(*args, **kwargs)

            def prune_previous(self, prepared):
                if not self.failed and prepared.previous_uri:
                    self.failed = True
                    raise OSError("checkpoint prune acknowledgement failed")
                return delegate.prune_previous(prepared)

            def prune_uri(self, *args, **kwargs):
                return delegate.prune_uri(*args, **kwargs)

        store = LoseFirstDelete()
        first_payload = b"checkpoint generation one"
        first = await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=first_payload,
        )
        second_payload = b"checkpoint generation two is larger than the first"
        second_prepared = delegate.prepare(
            identity=identity,
            payload=second_payload,
            reservation_bytes=32 * 1024**2,
            replacement_reservation_bytes=64 * 1024**2,
            previous_uri=first.uri,
            previous_sha256=first.sha256,
            previous_size_bytes=first.size_bytes,
        )
        with pytest.raises(OSError, match="checkpoint prune acknowledgement"):
            await queue.save_checkpoint(
                store=store,
                job_id=job_id,
                lease_token=lease["lease_token"],
                identity=identity,
                payload=second_payload,
            )
        pending_reservation = await repository.artifact_reservation_for(job_id)
        assert pending_reservation["consumed_bytes"] == len(first_payload) + len(second_payload)
        assert objects.read_source(
            object_key=first.uri.removeprefix(f"s3://{bucket}/"),
            sha256_digest=first.sha256,
            size_bytes=len(first_payload),
        ) == first_payload

        resumed = await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=lease["lease_token"],
            identity=identity,
            payload=second_payload,
        )
        assert resumed.sha256 == sha256(second_payload).hexdigest()
        assert (await queue.load_checkpoint(store=delegate, job_id=job_id)).payload == second_payload
        assert (await repository.artifact_reservation_for(job_id))["consumed_bytes"] == len(second_payload)
        with pytest.raises(FileNotFoundError):
            objects.read_source(
                object_key=first.uri.removeprefix(f"s3://{bucket}/"),
                sha256_digest=first.sha256,
                size_bytes=len(first_payload),
            )

        failed = await queue.record_probe_measurement(
            job_id=job_id,
            lease_token=lease["lease_token"],
            exit_code=1,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
            optimizer_steps=0,
            checkpoint_resumed=False,
            checkpoint_sha256=None,
            inference_steps=0,
            verification_details={"passed": False, "failure": "test_terminal_settlement"},
        )
        assert failed["result_state"] == "failed"
        assert await _storage_bytes(task7_database_url) == baseline + len(second_payload)
        objects.delete_object(object_key=second_prepared.object_key)
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_model_draft_assignment_reuses_atomic_task5_request_after_worker_or_event_crash(
    task7_database_url: str,
) -> None:
    required = (
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )
    if any(not os.environ.get(name) for name in required):
        pytest.skip("loopback-only Task 8 S3 endpoint is not configured")

    async def exercise() -> None:
        from gods_mlops.annotations.label_studio import (
            LabelStudioTaskReference,
        )
        from gods_mlops.annotations.models import ReviewAssignmentConflictError
        from gods_mlops.annotations.storage import PostgresAnnotationRepository
        from gods_mlops.annotations.workflow import LabelStudioReviewWorkflow
        from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore

        class LocalLabelStudio:
            def __init__(self) -> None:
                self.next_task_id = 1001
                self.next_file_id = 2001
                self.references = {}
                self.tasks = {}
                self.predictions = []

            async def import_media_task(self, *, project_id, filename, image, prediction=None):
                prior = self.references.get((project_id, filename))
                if prior is not None:
                    if prediction is not None:
                        await self.attach_prediction(
                            project_id=project_id, task_id=prior.task_id, prediction=prediction
                        )
                    return prior
                reference = LabelStudioTaskReference(
                    project_id=project_id,
                    task_id=self.next_task_id,
                    file_upload_id=self.next_file_id,
                    media_path=f"/data/upload/{project_id}/{filename}",
                    filename=filename,
                )
                self.next_task_id += 1
                self.next_file_id += 1
                self.references[(project_id, filename)] = reference
                self.tasks[reference.task_id] = {
                    "id": reference.task_id,
                    "data": {"image": reference.media_path},
                    "annotations": [],
                    "predictions": [],
                }
                if prediction is not None:
                    await self.attach_prediction(
                        project_id=project_id, task_id=reference.task_id, prediction=prediction
                    )
                return reference

            async def get_task(self, task_id):
                return self.tasks[task_id]

            async def attach_prediction(self, *, project_id, task_id, prediction):
                del project_id
                current = self.tasks[task_id]["predictions"]
                if prediction not in current:
                    current.append(prediction)
                self.predictions.append((task_id, prediction["model_version"]))

        class NoopCleanup:
            async def delete_upload(self, **kwargs):
                return {"deleted": True, "already_absent": False}

        annotations = PostgresAnnotationRepository(database_url=task7_database_url)
        ingestion = PostgresIngestionRepository(database_url=task7_database_url)
        await ingestion.ensure_schema()
        await annotations.ensure_schema()
        endpoint = os.environ["GODS_MLOPS_TEST_S3_ENDPOINT"]
        access_key = os.environ["GODS_MLOPS_TEST_S3_ACCESS_KEY"]
        secret_key = os.environ["GODS_MLOPS_TEST_S3_SECRET_KEY"]
        bucket = os.environ["GODS_MLOPS_TEST_S3_BUCKET"]
        objects = S3SampleStore(
            endpoint_url=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region="us-east-1",
        )
        frame = b"task8 immutable frame fixture"
        frame_hash = __import__("hashlib").sha256(frame).hexdigest()
        sample_id = uuid4()
        object_key = f"samples/task8-handoff/{sample_id}/{frame_hash}.jpg"
        objects.ensure_object(object_key=object_key, image=frame, expected_sha256=frame_hash)
        connection = await asyncpg.connect(task7_database_url)
        try:
            now = datetime.now(UTC)
            await connection.execute(
                """INSERT INTO ingestion_samples(
                       sample_id,camera_id,capture_day,captured_at_utc,reason,sha256,
                       model_revision,processor_revision,object_key,object_size_bytes,
                       receipt_id,state,received_at,retention_until
                   ) VALUES($1,$2,$3,$4::timestamptz,'periodic',$5,'task8','task8',$6,$7,$8,
                       'received',$4::timestamptz,$4::timestamptz+interval '7 days')""",
                sample_id,
                uuid4(),
                now.date(),
                now,
                frame_hash,
                object_key,
                len(frame),
                uuid4(),
            )
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes=used_bytes+$1 WHERE singleton=TRUE",
                len(frame),
            )
        finally:
            await connection.close()

        label_studio = LocalLabelStudio()
        workflow = LabelStudioReviewWorkflow(
            repository=annotations,
            objects=objects,
            label_studio=label_studio,
            media_cleanup=NoopCleanup(),
        )
        request_key = __import__("hashlib").sha256(
            f"task8-job:{sample_id}:{frame_hash}:result-v1".encode()
        ).hexdigest()
        prediction = {
            "model_version": f"gods-mlops:job-{request_key[:12]}:result-{request_key[12:24]}",
            "result": [],
        }
        kwargs = {
            "sample_id": sample_id,
            "stage": "bbox",
            "project_id": 17,
            "bbox_revision": None,
            "media_object_key": object_key,
            "required_bytes": len(frame),
            "model_request_key": request_key,
            "source_sha256": frame_hash,
            "model_version": prediction["model_version"],
            "item_id": str(sample_id),
        }
        try:
            first, repeated = await asyncio.gather(
                workflow.prepare_assignment(**kwargs),
                workflow.prepare_assignment(**kwargs),
            )
            assert first.revision == repeated.revision

            # Simulate a crash after Task 5 prepared the assignment but before Task 8
            # could append its job-event handoff marker; the next CPU attempt must find it.
            recovered = await workflow.prepare_assignment(**kwargs)
            assert recovered.revision == first.revision

            async def human_assignment():
                return await workflow.prepare_assignment(
                    sample_id=sample_id,
                    stage="bbox",
                    project_id=17,
                    bbox_revision=None,
                    media_object_key=object_key,
                    required_bytes=len(frame),
                )

            model_retry, human_retry = await asyncio.wait_for(
                asyncio.gather(
                    workflow.prepare_assignment(**kwargs),
                    human_assignment(),
                    return_exceptions=True,
                ),
                timeout=10,
            )
            assert model_retry.revision == first.revision
            assert isinstance(
                human_retry, (ReviewAssignmentConflictError, asyncpg.UniqueViolationError)
            )

            with pytest.raises(ReviewAssignmentConflictError, match="source|request|assignment"):
                await workflow.prepare_assignment(
                    **{**kwargs, "project_id": 18, "model_request_key": "f" * 64}
                )
            with pytest.raises(ReviewAssignmentConflictError, match="source|request|assignment"):
                await workflow.prepare_assignment(
                    **{**kwargs, "source_sha256": "e" * 64}
                )

            provisioned = await workflow.provision_task(
                revision=first.revision,
                project_id=17,
                prediction=prediction,
            )
            retried = await workflow.provision_task(
                revision=first.revision,
                project_id=17,
                prediction=prediction,
            )
            assert provisioned["assignment_revision"] == retried["assignment_revision"]
            assert provisioned["label_studio_task_id"] == retried["label_studio_task_id"]
            assert label_studio.tasks[provisioned["label_studio_task_id"]]["annotations"] == []
            assert len(label_studio.tasks[provisioned["label_studio_task_id"]]["predictions"]) == 1

            async with annotations._pool.acquire() as connection:
                assignments = await connection.fetch(
                    "SELECT revision FROM review_assignments WHERE sample_id=$1 AND stage='bbox'",
                    sample_id,
                )
                media = await connection.fetchrow(
                    "SELECT project_id,upload_filename,sha256,object_size_bytes,state "
                    "FROM label_studio_media_uploads WHERE assignment_revision=$1::uuid",
                    first.revision,
                )
            assert len(assignments) == 1
            assert media["project_id"] == 17
            assert media["upload_filename"].startswith("gods-model-")
            assert media["sha256"].strip() == frame_hash
            assert media["object_size_bytes"] == len(frame)
            assert media["state"] == "uploaded"

            from gods_mlops.annotations.service import AnnotationService

            task_id = provisioned["label_studio_task_id"]
            label_studio.tasks[task_id]["annotations"] = [
                {
                    "id": 3001,
                    "was_cancelled": False,
                    "result": [
                        {
                            "from_name": "bbox",
                            "type": "rectanglelabels",
                            "original_width": 1,
                            "original_height": 1,
                            "value": {"x": 0, "y": 0, "width": 100, "height": 100,
                                      "rectanglelabels": ["person"]},
                        }
                    ],
                }
            ]
            frozen_bbox = await AnnotationService(
                repository=annotations, label_studio=label_studio
            ).finalize_annotation(str(sample_id), first.revision)
            bbox_revision = frozen_bbox["annotation_revision_id"]
            crop_id = uuid4()
            crop = b"task8 immutable reviewed crop fixture"
            crop_hash = __import__("hashlib").sha256(crop).hexdigest()
            crop_key = f"crops/task8-handoff/{sample_id}/{bbox_revision}/{crop_hash}.jpg"
            objects.ensure_object(object_key=crop_key, image=crop, expected_sha256=crop_hash)
            connection = await asyncpg.connect(task7_database_url)
            try:
                await connection.execute(
                    """INSERT INTO annotation_crops(
                           crop_id,sample_id,bbox_revision,region_index,region_id,object_key,
                           sha256,object_size_bytes,state,caption_state,provenance,crop_set_ready
                       ) VALUES($1,$2,$3::uuid,0,'person-0',$4,$5,$6,'ready','needs_review',
                                '{"source":"task8-test"}'::jsonb,TRUE)""",
                    crop_id,
                    sample_id,
                    bbox_revision,
                    crop_key,
                    crop_hash,
                    len(crop),
                )
                await connection.execute(
                    "UPDATE ingestion_storage_usage SET used_bytes=used_bytes+$1 WHERE singleton=TRUE",
                    len(crop),
                )
            finally:
                await connection.close()
            caption_request_key = __import__("hashlib").sha256(
                f"task8-job:{sample_id}:{crop_id}:{bbox_revision}:{crop_hash}:caption-result-v1".encode()
            ).hexdigest()
            caption_prediction = {
                "model_version": f"gods-mlops:caption-job-{caption_request_key[:20]}",
                "result": [
                    {
                        "from_name": "caption",
                        "to_name": "image",
                        "type": "textarea",
                        "value": {"text": ["dark jacket and blue trousers"]},
                    }
                ],
            }
            caption_kwargs = {
                "sample_id": sample_id,
                "stage": "caption",
                "project_id": 23,
                "bbox_revision": bbox_revision,
                "media_object_key": crop_key,
                "required_bytes": len(crop),
                "model_request_key": caption_request_key,
                "source_sha256": crop_hash,
                "model_version": caption_prediction["model_version"],
                "item_id": str(crop_id),
            }
            with pytest.raises(ReviewAssignmentConflictError, match="bbox revision changed"):
                await workflow.prepare_assignment(
                    **{**caption_kwargs, "bbox_revision": str(uuid4())}
                )
            caption_assignment = await workflow.prepare_assignment(**caption_kwargs)
            assert caption_assignment.bbox_revision == bbox_revision
            caption_task = await workflow.provision_task(
                revision=caption_assignment.revision,
                project_id=23,
                prediction=caption_prediction,
            )
            assert caption_task["stage"] == "caption"
            assert caption_task["bbox_revision"] == bbox_revision
            assert label_studio.tasks[caption_task["label_studio_task_id"]]["annotations"] == []

            from gods_mlops.jobs.models import AnnotationSourceSelection, ExecutionProfile
            from gods_mlops.jobs.sources import DatasetSourceRegistry

            queue_repository = PostgresJobQueueRepository(database_url=task7_database_url)
            sources = DatasetSourceRegistry(database_url=task7_database_url)
            queue = JobQueue(repository=queue_repository, sources=sources)
            config_version = f"task8-handoff-{uuid4().hex}"
            await queue.register_profile(
                ExecutionProfile(
                    model_kind="detr",
                    config_version=config_version,
                    phase="preparation",
                    memory_requirement_mib=1024,
                    artifact_reservation_bytes=1024,
                    config={"input_size": 640, "micro_batch": 1},
                    candidate=True,
                )
            )
            async with queue_repository._pool.acquire() as connection:
                await connection.execute(
                    """UPDATE gods_mlops_resource_profiles
                       SET profile_state='measured',measurement_id=$2::uuid
                       WHERE phase='preparation' AND model_kind='detr' AND config_version=$1""",
                    config_version,
                    uuid4(),
                )
            batch = await sources.prepare_annotation_batch(
                [
                    AnnotationSourceSelection(
                        item_kind="frame", item_id=str(sample_id), sha256=frame_hash
                    )
                ]
            )
            job_id = await queue.submit_preparation(
                batch=batch, model_kind="detr", config_version=config_version
            )
            job = await queue.get(job_id)
            identity = await queue_repository.checkpoint_identity(job_id)
            async with queue_repository._pool.acquire() as connection:
                await connection.execute(
                    "UPDATE gods_mlops_jobs SET state='completed',completed_at=now() WHERE job_id=$1::uuid",
                    job_id,
                )
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(job_id,event_type,state,details)
                       VALUES($1::uuid,'result_artifact_committed','running',$2::jsonb)""",
                    job_id,
                    json.dumps(
                        {
                            "kind": "drafts",
                            "sha256": "9" * 64,
                            "identity": identity.as_dict(),
                            "uri": "s3://task8-test/jobs/draft.json",
                            "size_bytes": 1,
                        }
                    ),
                )
            handoff_details = {
                "job_id": job_id,
                "item_id": str(sample_id),
                "sample_id": str(sample_id),
                "assignment_revision": first.revision,
                "project_id": 17,
                "model_request_key": request_key,
                "model_version": prediction["model_version"],
                "model_id": "PekingU/rtdetr_v2_r18vd",
                "model_revision": "5650961749fa93567c0d46fc7f43ea4f9e914107",
                "media_object_key": object_key,
                "source_sha256": frame_hash,
                "source_size_bytes": len(frame),
                "bbox_revision": None,
                "result_sha256": "9" * 64,
            }
            handoff = await queue_repository.record_preparation_review_assignment(**handoff_details)
            repeated_handoff = await queue_repository.record_preparation_review_assignment(**handoff_details)
            assert handoff["assignment_revision"] == first.revision
            assert repeated_handoff == handoff
            assert (await queue_repository.review_handoffs_for(job_id))[0] == handoff
            with pytest.raises(ValueError, match="project"):
                await queue_repository.record_preparation_review_assignment(
                    **{**handoff_details, "project_id": 18}
                )
            await queue.repository.close()
            await sources.close()
        finally:
            await annotations.close()
            await ingestion.close()
            objects.delete_object(object_key=object_key)
            if "crop_key" in locals():
                objects.delete_object(object_key=crop_key)

    asyncio.run(exercise())


async def _storage_bytes(database_url: str) -> int:
    connection = await asyncpg.connect(database_url)
    try:
        return int(await connection.fetchval("SELECT used_bytes FROM ingestion_storage_usage WHERE singleton=TRUE"))
    finally:
        await connection.close()


async def _rewrite_checkpoint_history_as_round1(
    database_url: str, *, job_id: str, uri: str, sha256_digest: str
) -> dict[str, dict]:
    """Remove only fields that did not exist in the round-1 durable event format."""
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """UPDATE gods_mlops_job_events SET details = CASE
                     WHEN event_type IN ('artifact_write_pending','checkpoint_committed')
                       THEN details - 'write_lifetime_id'
                     WHEN event_type IN ('checkpoint_prune_pending','checkpoint_pruned')
                       THEN details - 'write_lifetime_id' - 'operation_id'
                     ELSE details END
               WHERE job_id=$1::uuid
                 AND event_type IN ('artifact_write_pending','checkpoint_committed',
                                    'checkpoint_prune_pending','checkpoint_pruned')
                 AND COALESCE(details->>'uri',details->>'checkpoint_uri')=$2
                 AND details->>'sha256'=$3""",
            job_id,
            uri,
            sha256_digest,
        )
        rows = await connection.fetch(
            """SELECT event_type,details FROM gods_mlops_job_events
               WHERE job_id=$1::uuid AND event_type IN
                 ('artifact_write_pending','checkpoint_committed','checkpoint_prune_pending','checkpoint_pruned')
                 AND COALESCE(details->>'uri',details->>'checkpoint_uri')=$2
                 AND details->>'sha256'=$3
               ORDER BY event_id""",
            job_id,
            uri,
            sha256_digest,
        )
        return {
            row["event_type"]: json.loads(row["details"])
            if isinstance(row["details"], str)
            else row["details"]
            for row in rows
        }
    finally:
        await connection.close()
