from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from gods_mlops.jobs.checkpoints import CheckpointIdentity
from gods_mlops.jobs.queue import JobQueue
from gods_mlops.training.artifacts import S3ResultArtifactStore
from gods_mlops.training.worker_adapter import KubernetesOwnedWorkerAdapter, build_gpu_worker_job


class _BlockingObjects:
    def __init__(self) -> None:
        self.writer_started = threading.Event()
        self.release_writer = threading.Event()
        self.writer_thread: int | None = None

    def write_immutable(self, *, object_key, content, sha256_digest, content_type) -> None:
        self.writer_thread = threading.get_ident()
        self.writer_started.set()
        if not self.release_writer.wait(timeout=5):
            raise TimeoutError("test writer was not released")
        self.object = content

    def read_source(self, *, object_key, sha256_digest, size_bytes) -> bytes:
        return self.object


class _Repository:
    def __init__(self, identity: CheckpointIdentity) -> None:
        self.identity = identity
        self.writer_events: list[tuple[str, str, str]] = []
        self.finalized = False
        self.job = {
            "job_id": identity.job_id,
            "phase": identity.phase,
            "model_kind": identity.model_kind,
            "config_version": identity.config_version,
        }

    async def get_job(self, job_id: str) -> dict:
        assert job_id == self.identity.job_id
        return self.job

    async def checkpoint_identity(self, job_id: str) -> CheckpointIdentity:
        assert job_id == self.identity.job_id
        return self.identity

    async def lease_is_current(self, job_id: str, lease_token: str) -> bool:
        return True

    async def get_profile(self, **_kwargs) -> dict:
        return {"result_reservation_bytes": 1024}

    async def begin_artifact_write(self, **_kwargs) -> str:
        return "operation-1"

    async def record_artifact_writer_started(self, *, job_id, operation_id, writer_attempt_id, **_kwargs) -> None:
        self.writer_events.append(("started", operation_id, writer_attempt_id))

    async def record_artifact_writer_quiescent(self, *, job_id, operation_id, writer_attempt_id, **_kwargs) -> None:
        self.writer_events.append(("quiescent", operation_id, writer_attempt_id))

    async def commit_result_artifact(self, *, prepared, store, verified_artifact=None, **_kwargs):
        self.finalized = True
        return verified_artifact if verified_artifact is not None else store.commit(prepared)


def test_s3_result_prepare_and_write_do_not_block_the_worker_event_loop() -> None:
    async def exercise() -> None:
        loop_thread = threading.get_ident()
        identity = CheckpointIdentity(
            job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
            input_kind="probe_input",
            input_id="probe-v1",
            input_sha256="1" * 64,
            phase="probe",
            model_kind="clip",
            config_version="probe-v1",
            config_sha256="2" * 64,
            dataset_version=None,
        )
        objects = _BlockingObjects()
        store = S3ResultArtifactStore(objects=objects, bucket="gods-test")
        original_prepare = store.prepare
        prepare_threads: list[int] = []

        def capture_prepare(**kwargs):
            prepare_threads.append(threading.get_ident())
            return original_prepare(**kwargs)

        store.prepare = capture_prepare
        repository = _Repository(identity)
        queue = JobQueue(repository=repository, sources=object())
        heartbeat_ticks = 0
        stop_heartbeat = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal heartbeat_ticks
            while not stop_heartbeat.is_set():
                heartbeat_ticks += 1
                await asyncio.sleep(0.001)

        heartbeat_task = asyncio.create_task(heartbeat())
        baseline_ticks = heartbeat_ticks
        release_observation: list[int] = []

        def release_after_a_short_block() -> None:
            time.sleep(0.05)
            release_observation.append(heartbeat_ticks)
            objects.release_writer.set()

        release_thread = threading.Thread(target=release_after_a_short_block)
        release_thread.start()
        artifact = await queue.save_result_artifact(
            store=store,
            job_id=identity.job_id,
            lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
            identity=identity,
            kind="probe",
            payload=b"result payload",
            runtime_measurements={"optimizer_steps": 3},
        )
        stop_heartbeat.set()
        await heartbeat_task
        release_thread.join(timeout=1)

        assert artifact.sha256
        assert [event[0] for event in repository.writer_events] == ["started", "quiescent"]
        assert repository.writer_events[0][1:] == repository.writer_events[1][1:]
        assert repository.finalized
        assert prepare_threads and prepare_threads[0] != loop_thread
        assert objects.writer_thread != loop_thread
        assert release_observation and release_observation[0] > baseline_ticks

    asyncio.run(exercise())


def test_kubernetes_worker_contract_carries_the_exact_deadline_authority() -> None:
    deadline = {
        "controller_invocation_id": str(uuid4()),
        "artifact_deadline_at": datetime(2026, 10, 7, 1, 2, 3, 456789, tzinfo=UTC),
    }
    job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    lease_token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
    job = {
        "job_id": job_id,
        "state": "running",
        "lease_token": lease_token,
        "lease_generation": 2,
        "phase": "training",
        "target_phase": "training",
        "input_kind": "dataset_version",
        "input_id": "dataset-test-v1",
        "input_sha256": "1" * 64,
        "dataset_version": "dataset-test-v1",
        "model_kind": "clip",
        "config_version": "clip-v1",
        "config_sha256": "2" * 64,
        "profile_state_snapshot": "measured",
    }
    lease = {
        "job_id": job_id,
        "lease_token": lease_token,
        "fencing_token": 2,
        "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
    }
    profile = {
        "phase": "training",
        "model_kind": "clip",
        "config_version": "clip-v1",
        "config_sha256": "2" * 64,
        "profile_state": "measured",
    }

    manifest = build_gpu_worker_job(
        job=job,
        lease=lease,
        profile=profile,
        namespace="gods-mlops",
        image="registry.example/gods-training@sha256:" + "a" * 64,
        artifact_deadline=deadline,
    )

    env = {item["name"]: item for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["GODS_MLOPS_WORKER_ARTIFACT_INVOCATION_ID"]["value"] == deadline["controller_invocation_id"]
    assert env["GODS_MLOPS_WORKER_ARTIFACT_DEADLINE_UTC"]["value"] == "2026-10-07T01:02:03.456789+00:00"
    assert "activeDeadlineSeconds" not in manifest["spec"]


def test_kubernetes_409_and_retained_job_recovery_reject_deadline_mismatch() -> None:
    deadline = {
        "controller_invocation_id": str(uuid4()),
        "artifact_deadline_at": datetime(2026, 10, 7, 1, 2, 3, tzinfo=UTC),
    }
    stale_deadline = {
        "controller_invocation_id": str(uuid4()),
        "artifact_deadline_at": datetime(2026, 10, 7, 1, 2, 4, tzinfo=UTC),
    }
    job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    lease_token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
    job = {
        "job_id": job_id,
        "state": "running",
        "lease_token": lease_token,
        "lease_generation": 2,
        "phase": "training",
        "target_phase": "training",
        "input_kind": "dataset_version",
        "input_id": "dataset-test-v1",
        "input_sha256": "1" * 64,
        "dataset_version": "dataset-test-v1",
        "model_kind": "clip",
        "config_version": "clip-v1",
        "config_sha256": "2" * 64,
        "profile_state_snapshot": "measured",
    }
    lease = {
        "job_id": job_id,
        "lease_token": lease_token,
        "fencing_token": 2,
        "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
    }
    profile = {
        "phase": "training",
        "model_kind": "clip",
        "config_version": "clip-v1",
        "config_sha256": "2" * 64,
        "profile_state": "measured",
    }
    image = "registry.example/gods-training@sha256:" + "a" * 64
    existing = build_gpu_worker_job(
        job=job,
        lease=lease,
        profile=profile,
        namespace="gods-mlops",
        image=image,
        artifact_deadline=stale_deadline,
    )

    class AlreadyExists(Exception):
        status = 409

    class BatchAPI:
        def __init__(self):
            self.created = 0
            self.read = 0

        def create_namespaced_job(self, *, namespace, body):
            self.created += 1
            raise AlreadyExists()

        def read_namespaced_job(self, *, name, namespace):
            self.read += 1
            return existing

    batch_api = BatchAPI()
    adapter = KubernetesOwnedWorkerAdapter(batch_api=batch_api, namespace="gods-mlops", image=image)
    with pytest.raises(ValueError, match="existing Kubernetes Job"):
        adapter.ensure_worker(job=job, lease=lease, profile=profile, artifact_deadline=deadline)
    with pytest.raises(ValueError, match="existing Kubernetes Job"):
        adapter.read_existing_worker(
            job={**job, "state": "yield_requested"},
            lease=lease,
            profile=profile,
            artifact_deadline=deadline,
        )
    assert batch_api.created == 1
    assert batch_api.read == 2


def test_docker_worker_environment_transports_the_bound_deadline(monkeypatch) -> None:
    from gods_mlops.training.docker_probe import _worker_environment
    from gods_mlops.training.claims import WorkerClaim

    job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
    job = {
        "job_id": job_id,
        "state": "running",
        "lease_token": token,
        "lease_generation": 2,
        "phase": "probe",
        "target_phase": "training",
        "input_kind": "probe_input",
        "input_id": "probe-v1",
        "input_sha256": "1" * 64,
        "dataset_version": None,
        "model_kind": "clip",
        "config_version": "probe-v1",
        "config_sha256": "2" * 64,
        "profile_state_snapshot": "candidate",
    }
    lease = {
        "job_id": job_id,
        "lease_token": token,
        "fencing_token": 2,
        "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
    }
    claim = WorkerClaim.from_admitted_job(job, lease, image_id="sha256:" + "a" * 64)
    invocation_id = str(uuid4())
    monkeypatch.setenv("GODS_MLOPS_S3_ENDPOINT_URL", "http://127.0.0.1:18333")
    monkeypatch.setenv("GODS_MLOPS_S3_BUCKET", "gods-test")
    monkeypatch.setenv("GODS_MLOPS_S3_ACCESS_KEY", "test-access")
    monkeypatch.setenv("GODS_MLOPS_S3_SECRET_KEY", "test-secret")

    environment = _worker_environment(
        claim=claim,
        database_url="postgresql://test:secret@127.0.0.1:15439/gods",
        database_remote_port=35439,
        storage_remote_port=38333,
        image_id="sha256:" + "a" * 64,
        gpu_uuid=lease["gpu_uuid"],
        artifact_deadline={
            "controller_invocation_id": invocation_id,
            "artifact_deadline_at": datetime(2026, 10, 7, 1, 2, 3, 456789, tzinfo=UTC),
        },
    )

    assert environment["GODS_MLOPS_WORKER_ARTIFACT_INVOCATION_ID"] == invocation_id
    assert environment["GODS_MLOPS_WORKER_ARTIFACT_DEADLINE_UTC"] == "2026-10-07T01:02:03.456789+00:00"


def test_docker_deadline_binding_uses_post_admission_remaining_budget() -> None:
    from gods_mlops.training.docker_probe import _bind_docker_artifact_deadline

    db_now = datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
    invocation_id = str(uuid4())
    monotonic_samples = iter((10.0, 12.0, 14.0))
    captured = {}

    class Repository:
        async def artifact_database_clock(self):
            return db_now

        async def bind_worker_artifact_deadline(self, **values):
            captured.update(values)
            return {
                "controller_invocation_id": values["controller_invocation_id"],
                "artifact_deadline_at": values["candidate_deadline_at"],
            }

    authority, remaining = asyncio.run(
        _bind_docker_artifact_deadline(
            Repository(),
            job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
            lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
            fencing_token=2,
            controller_invocation_id=invocation_id,
            run_deadline=50.0,
            clock=lambda: next(monotonic_samples),
        )
    )

    assert captured["candidate_deadline_at"] == db_now + timedelta(seconds=38)
    assert authority["controller_invocation_id"] == invocation_id
    assert remaining == 36.0


def test_explicit_worker_deadline_is_validated_before_repository_or_object_store_startup(monkeypatch) -> None:
    import gods_mlops.training.worker as worker
    from gods_mlops.training.artifact_deadlines import ArtifactDeadlineError, INVOCATION_ENV, DEADLINE_ENV

    monkeypatch.setenv("GODS_MLOPS_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("GODS_MLOPS_JOB_ID", "a0320b59-663c-4cdc-b893-086bb970ea60")
    monkeypatch.setenv("GODS_MLOPS_LEASE_TOKEN", "8ad96890-3434-4f07-85bb-8cde17a2b009")
    monkeypatch.setenv("GODS_MLOPS_IMAGE_ID", "sha256:" + "a" * 64)
    monkeypatch.setenv("GODS_MLOPS_GPU_UUID", "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e")
    monkeypatch.setenv(INVOCATION_ENV, str(uuid4()))
    monkeypatch.setenv(DEADLINE_ENV, "tomorrow")

    class MustNotStartRepository:
        def __init__(self, **_kwargs):
            pytest.fail("malformed deadline reached repository startup")

    monkeypatch.setattr(worker, "PostgresJobQueueRepository", MustNotStartRepository)

    with pytest.raises(ArtifactDeadlineError, match="malformed"):
        asyncio.run(worker.run_worker())


def test_worker_deadline_resolution_keeps_legacy_bound_and_anchors_explicit_authority() -> None:
    from gods_mlops.training.worker import resolve_worker_artifact_deadline

    invocation_id = str(uuid4())
    deadline = datetime(2026, 10, 7, 1, 2, 3, tzinfo=UTC)
    database_now = datetime(2026, 10, 7, 1, 2, 0, tzinfo=UTC)
    authority_record = {
        "controller_invocation_id": invocation_id,
        "artifact_deadline_at": deadline,
        "database_now": database_now,
    }
    calls = []

    class Repository:
        async def read_worker_artifact_deadline(self, **values):
            calls.append(values)
            return authority_record

    samples = iter((10.0, 10.4))
    legacy_deadline, legacy_authority = asyncio.run(
        resolve_worker_artifact_deadline(
            Repository(),
            job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
            lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
            fencing_token=2,
            environment_value=None,
            monotonic_clock=lambda: 10.0,
        )
    )
    explicit_deadline, explicit_authority = asyncio.run(
        resolve_worker_artifact_deadline(
            Repository(),
            job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
            lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
            fencing_token=2,
            environment_value=(invocation_id, deadline),
            monotonic_clock=lambda: next(samples),
        )
    )

    assert legacy_deadline == 40.0
    assert legacy_authority is None
    assert explicit_deadline == 13.0
    assert explicit_authority is authority_record
    assert len(calls) == 1


def test_worker_artifact_callback_rejects_expired_local_deadline_before_scheduling_io() -> None:
    from gods_mlops.training.worker import await_worker_artifact_operation

    started = []

    async def operation():
        started.append(True)
        return "unexpected"

    with pytest.raises(TimeoutError, match="artifact deadline"):
        asyncio.run(
            await_worker_artifact_operation(
                operation(),
                local_deadline=10.0,
                monotonic_clock=lambda: 10.0,
            )
        )
    assert started == []


def test_worker_artifact_callback_passes_only_full_remaining_attempt_time(monkeypatch) -> None:
    from gods_mlops.training.worker import await_worker_artifact_operation

    timeouts = []
    real_wait_for = asyncio.wait_for

    async def capture_timeout(operation, *, timeout):
        timeouts.append(timeout)
        return await real_wait_for(operation, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait_for", capture_timeout)
    result = asyncio.run(
        await_worker_artifact_operation(
            asyncio.sleep(0, result="done"),
            local_deadline=1000.0,
            monotonic_clock=lambda: 50.0,
        )
    )

    assert result == "done"
    assert timeouts == [950.0]


def test_runtime_measurement_snapshot_cannot_change_after_worker_dispatch() -> None:
    from gods_mlops.training.worker import _snapshot_runtime_measurements

    measurements = {"steps": [1, {"allocated": 64}], "phase": "evaluation"}
    snapshot = _snapshot_runtime_measurements(measurements)
    measurements["steps"][1]["allocated"] = 128
    measurements["phase"] = "training"

    assert snapshot == {"steps": [1, {"allocated": 64}], "phase": "evaluation"}


def test_deadline_binding_reuses_fence_or_invocation_budget_but_allows_a_new_attempt() -> None:
    from gods_mlops.training.artifact_deadlines import artifact_deadline_for_attempt

    first_invocation = str(uuid4())
    next_invocation = str(uuid4())
    original_deadline = datetime(2026, 10, 7, 1, 0, tzinfo=UTC)
    candidate = datetime(2026, 10, 7, 2, 0, tzinfo=UTC)
    current_fence = {
        "controller_invocation_id": first_invocation,
        "artifact_deadline_at": original_deadline,
        "lease_token": "lease-1",
    }

    same_fence = artifact_deadline_for_attempt(
        existing_fence=current_fence,
        existing_invocation=None,
        controller_invocation_id=next_invocation,
        lease_token="lease-1",
        candidate_deadline_at=candidate,
    )
    next_fence_same_invocation = artifact_deadline_for_attempt(
        existing_fence=None,
        existing_invocation=current_fence,
        controller_invocation_id=first_invocation,
        lease_token="lease-2",
        candidate_deadline_at=candidate,
    )
    new_attempt = artifact_deadline_for_attempt(
        existing_fence=None,
        existing_invocation=None,
        controller_invocation_id=next_invocation,
        lease_token="lease-3",
        candidate_deadline_at=candidate,
    )

    assert same_fence["controller_invocation_id"] == first_invocation
    assert same_fence["artifact_deadline_at"] == original_deadline
    assert next_fence_same_invocation["artifact_deadline_at"] == original_deadline
    assert new_attempt["controller_invocation_id"] == next_invocation
    assert new_attempt["artifact_deadline_at"] == candidate


def test_worker_deadline_anchor_subtracts_read_round_trip_and_rejects_mismatch() -> None:
    from gods_mlops.training.artifact_deadlines import (
        ArtifactDeadlineError,
        anchor_worker_deadline,
    )

    invocation_id = str(uuid4())
    deadline = datetime(2026, 10, 7, 1, 2, 3, tzinfo=UTC)
    database_now = datetime(2026, 10, 7, 1, 2, 0, tzinfo=UTC)
    record = {
        "controller_invocation_id": invocation_id,
        "artifact_deadline_at": deadline,
        "database_now": database_now,
    }

    monotonic_deadline = anchor_worker_deadline(
        environment_value=(invocation_id, deadline),
        authority_record=record,
        monotonic_before_read=10.0,
        monotonic_after_read=10.4,
    )

    assert monotonic_deadline == 13.0
    with pytest.raises(ArtifactDeadlineError, match="durable authority"):
        anchor_worker_deadline(
            environment_value=(str(uuid4()), deadline),
            authority_record=record,
            monotonic_before_read=10.0,
            monotonic_after_read=10.1,
        )
    with pytest.raises(ArtifactDeadlineError, match="expired"):
        anchor_worker_deadline(
            environment_value=(invocation_id, deadline),
            authority_record={**record, "database_now": deadline},
            monotonic_before_read=10.0,
            monotonic_after_read=10.0,
        )
def test_cancelled_result_wait_keeps_writer_tracked_and_cannot_publish_late() -> None:
    async def exercise() -> None:
        identity = CheckpointIdentity(
            job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
            input_kind="probe_input",
            input_id="probe-v1",
            input_sha256="1" * 64,
            phase="probe",
            model_kind="clip",
            config_version="probe-v1",
            config_sha256="2" * 64,
            dataset_version=None,
        )
        objects = _BlockingObjects()
        repository = _Repository(identity)
        store = S3ResultArtifactStore(objects=objects, bucket="gods-test")
        queue = JobQueue(repository=repository, sources=object())
        release_thread = threading.Thread(
            target=lambda: (time.sleep(0.05), objects.release_writer.set())
        )
        release_thread.start()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                queue.save_result_artifact(
                    store=store,
                    job_id=identity.job_id,
                    lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
                    identity=identity,
                    kind="probe",
                    payload=b"result payload",
                    runtime_measurements={"optimizer_steps": 3},
                ),
                timeout=0.005,
            )

        for _ in range(100):
            if any(event[0] == "quiescent" for event in repository.writer_events):
                break
            await asyncio.sleep(0.005)
        release_thread.join(timeout=1)

        assert [event[0] for event in repository.writer_events] == ["started", "quiescent"]
        assert repository.writer_events[0][1:] == repository.writer_events[1][1:]
        assert not repository.finalized

    asyncio.run(exercise())


def test_cleanup_detects_unmatched_artifact_writer_attempts() -> None:
    from gods_mlops.jobs.queue import _unmatched_artifact_writers

    started = {
        "operation_id": "operation-1",
        "writer_attempt_id": "attempt-1",
    }
    quiescent = {**started}

    assert _unmatched_artifact_writers([("artifact_write_started", started)]) == {
        ("operation-1", "attempt-1")
    }
    assert _unmatched_artifact_writers(
        [
            ("artifact_write_started", started),
            ("artifact_write_quiescent", quiescent),
        ]
    ) == set()
    assert _unmatched_artifact_writers([("artifact_write_quiescent", quiescent)]) == {
        ("operation-1", "attempt-1")
    }
    pending_without_start = {
        "operation_id": "operation-2",
        "writer_quiescence_required": True,
    }
    assert _unmatched_artifact_writers([("artifact_write_pending", pending_without_start)]) == {
        ("operation-2", "<missing-start>")
    }
