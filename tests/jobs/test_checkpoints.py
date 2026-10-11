from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from datetime import timedelta

import pytest
from conftest import seed_training_ready_dataset

from gods_mlops.jobs.checkpoints import (
    CHECKPOINT_INTERVAL_SECONDS,
    CheckpointIdentity,
    CheckpointIntegrityError,
    FileCheckpointStore,
    StaleCheckpointOwnerError,
)
from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ExecutionProfile, ResourceObservation
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry
from test_monitor import BASE_TIME, GPU_UUID, HOST_IDENTITY, FILESYSTEM_IDENTITY, STORAGE_PATH


def _identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        job_id="job-task7-probe-v1",
        input_kind="probe_input",
        input_id="probe-fixture-v1",
        input_sha256="a" * 64,
        phase="probe",
        model_kind="detr",
        config_version="detr-probe-candidate-v1",
        config_sha256="b" * 64,
        dataset_version=None,
    )


def test_partial_files_are_not_resumeable_and_identity_mismatch_is_rejected(tmp_path) -> None:
    store = FileCheckpointStore(root=tmp_path)
    identity = _identity()
    job_directory = tmp_path / identity.job_id
    job_directory.mkdir()
    (job_directory / "checkpoint.partial").write_bytes(b"incomplete write")
    assert store.load(identity.job_id, expected_identity=identity) is None

    payload = b"complete optimizer and model state"
    saved = store.save(identity=identity, payload=payload, reservation_bytes=1024)
    assert saved.sha256 == sha256(payload).hexdigest()
    loaded = store.load(identity.job_id, expected_identity=identity)
    assert loaded is not None
    assert loaded.payload == payload
    assert loaded.identity == identity

    changed_input = replace(identity, input_sha256="c" * 64)
    assert store.load(identity.job_id, expected_identity=changed_input) is None


def test_file_checkpoint_store_save_replaces_only_the_prior_committed_lifetime(tmp_path) -> None:
    store = FileCheckpointStore(root=tmp_path)
    identity = _identity()

    first = store.save(
        identity=identity,
        payload=b"first committed local checkpoint",
        reservation_bytes=1024,
    )
    second = store.save(
        identity=identity,
        payload=b"second committed local checkpoint",
        reservation_bytes=1024,
    )

    assert second.payload == b"second committed local checkpoint"
    assert first.sha256 != second.sha256
    checkpoint_directory = tmp_path / identity.job_id
    assert len(list(checkpoint_directory.glob("*.checkpoint"))) == 1
    assert len(list(checkpoint_directory.glob("*.json"))) == 1


def test_checkpoint_hash_mismatch_is_never_returned_as_resumable(tmp_path) -> None:
    store = FileCheckpointStore(root=tmp_path)
    identity = _identity()
    saved = store.save(
        identity=identity,
        payload=b"valid checkpoint",
        reservation_bytes=1024,
    )
    saved.path.write_bytes(b"corrupted checkpoint")
    with pytest.raises(CheckpointIntegrityError, match="SHA-256"):
        store.load(identity.job_id, expected_identity=identity)


def test_checkpoint_payload_cannot_exceed_the_versioned_artifact_reservation(tmp_path) -> None:
    store = FileCheckpointStore(root=tmp_path)
    with pytest.raises(ValueError, match="reservation"):
        store.save(identity=_identity(), payload=b"too large", reservation_bytes=4)


def test_checkpoint_interval_is_five_minutes() -> None:
    assert CHECKPOINT_INTERVAL_SECONDS == 300


def test_checkpoint_commit_is_fenced_and_pins_the_current_job_input(task7_database_url, tmp_path) -> None:
    async def exercise() -> None:
        from test_monitor import _observation

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
        job_id = await queue.submit_probe(
            probe_input_id="checkpoint-task7-probe-v1",
            input_sha256="d" * 64,
            model_kind="detr",
            config_version=profile.config_version,
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
            admitted = await admission.admit(job_id, _observation(now[0]))
        old_token = admitted["lease_token"]
        job = await queue.get(job_id)
        identity = CheckpointIdentity(
            job_id=job_id,
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
        first = await queue.save_checkpoint(
            store=store,
            job_id=job_id,
            lease_token=old_token,
            identity=identity,
            payload=b"complete owned checkpoint",
        )
        assert first.sha256 == sha256(b"complete owned checkpoint").hexdigest()

        # Expire and release the original owner through complete process observations.
        owned = (ProcessIdentity(pid=43122, start_ticks=89123, uid=1009),)
        await queue.bind_process(job_id, old_token, owned[0])
        monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)
        await queue.request_yield(job_id, "task7_checkpoint_resume")
        now[0] += timedelta(seconds=5)
        await monitor.observe(_observation(now[0], gpu=(), processes=()))
        assert await repository.get_active_lease(GPU_UUID) is None
        for _ in range(6):
            now[0] += timedelta(seconds=5)
            admitted = await admission.admit(job_id, _observation(now[0]))
        new_token = admitted["lease_token"]
        assert new_token != old_token

        with pytest.raises(StaleCheckpointOwnerError):
            await queue.save_checkpoint(
                store=store,
                job_id=job_id,
                lease_token=old_token,
                identity=identity,
                payload=b"stale writer output",
            )
        assert store.load(job_id, expected_identity=identity).payload == b"complete owned checkpoint"

        changed_input = CheckpointIdentity(
            **{**identity.as_dict(), "input_sha256": "e" * 64}
        )
        with pytest.raises(ValueError, match="identity"):
            await queue.save_checkpoint(
                store=store,
                job_id=job_id,
                lease_token=new_token,
                identity=changed_input,
                payload=b"wrong input checkpoint",
            )
        await queue.close()
        await repository.close()

    import asyncio
    from gods_mlops.jobs.models import ProcessIdentity
    from gods_mlops.jobs.monitor import GpuJobMonitor

    asyncio.run(exercise())
