from __future__ import annotations

import asyncio
from hashlib import sha256
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
        self.writer_returned = threading.Event()
        self.release_writer = threading.Event()
        self.writer_thread: int | None = None

    def write_immutable(self, *, object_key, content, sha256_digest, content_type) -> None:
        self.writer_thread = threading.get_ident()
        self.writer_started.set()
        try:
            if not self.release_writer.wait(timeout=5):
                raise TimeoutError("test writer was not released")
            self.object = content
        finally:
            self.writer_returned.set()

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

    assert legacy_deadline is None
    assert legacy_authority is None
    assert explicit_deadline == 13.0
    assert explicit_authority is authority_record
    assert len(calls) == 1


def test_legacy_artifact_bound_starts_fresh_for_late_and_repeated_operations() -> None:
    from gods_mlops.training.worker import _artifact_operation_deadline

    first = _artifact_operation_deadline(None, monotonic_clock=lambda: 131.0)
    second = _artifact_operation_deadline(None, monotonic_clock=lambda: 165.0)

    assert first == 161.0
    assert second == 195.0


def test_legacy_artifact_operation_that_exceeds_30_seconds_times_out_per_call(monkeypatch) -> None:
    from gods_mlops.training.worker import (
        _artifact_operation_deadline,
        await_worker_artifact_operation,
    )

    seen_timeouts = []
    cancelled = asyncio.Event()

    async def slow_operation():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def fake_wait_for(operation, *, timeout):
        seen_timeouts.append(timeout)
        task = asyncio.create_task(operation)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    with pytest.raises(TimeoutError, match="exceeded its deadline"):
        asyncio.run(
            await_worker_artifact_operation(
                slow_operation(),
                local_deadline=_artifact_operation_deadline(None, monotonic_clock=lambda: 100.0),
                monotonic_clock=lambda: 100.0,
            )
        )

    assert seen_timeouts == [30.0]
    assert cancelled.is_set()


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


def test_cancelled_writer_watcher_never_records_quiescence_before_thread_return() -> None:
    from gods_mlops.jobs import queue as queue_module

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
        repository = _Repository(identity)
        queue = JobQueue(repository=repository, sources=object())
        objects = _BlockingObjects()
        store = SimpleNamespace(commit=lambda _prepared: _blocking_write(objects))
        watchers_before = set(queue_module._ACTIVE_ARTIFACT_WRITER_WATCHERS)
        writer_task = asyncio.create_task(
            queue._tracked_s3_artifact_write(
                job_id=identity.job_id,
                lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
                operation_id="operation-cancelled-watcher",
                store=store,
                prepared=SimpleNamespace(),
                artifact_deadline=None,
            )
        )
        assert await asyncio.to_thread(objects.writer_started.wait, 1)
        await asyncio.sleep(0)
        watchers = queue_module._ACTIVE_ARTIFACT_WRITER_WATCHERS - watchers_before
        assert len(watchers) == 1
        next(iter(watchers)).cancel()
        await asyncio.sleep(0.01)
        was_active = not objects.writer_returned.is_set()
        quiescent_before_release = any(event[0] == "quiescent" for event in repository.writer_events)
        objects.release_writer.set()
        with pytest.raises(asyncio.CancelledError):
            await writer_task
        assert objects.writer_returned.wait(timeout=1)

        assert was_active
        assert not quiescent_before_release
        assert not any(event[0] == "quiescent" for event in repository.writer_events)

    asyncio.run(exercise())


def test_event_loop_shutdown_cannot_turn_running_writer_into_quiescent() -> None:
    from gods_mlops.jobs import queue as queue_module

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
    repository = _Repository(identity)
    queue = JobQueue(repository=repository, sources=object())
    objects = _BlockingObjects()
    release_snapshot = {}

    def release_after_shutdown_cancellation() -> None:
        time.sleep(0.05)
        release_snapshot["thread_returned"] = objects.writer_returned.is_set()
        release_snapshot["quiescent"] = any(
            event[0] == "quiescent" for event in repository.writer_events
        )
        objects.release_writer.set()

    async def abandon_pending_writer() -> None:
        asyncio.create_task(
            queue._tracked_s3_artifact_write(
                job_id=identity.job_id,
                lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
                operation_id="operation-loop-shutdown",
                store=SimpleNamespace(commit=lambda _prepared: _blocking_write(objects)),
                prepared=SimpleNamespace(),
                artifact_deadline=None,
            )
        )
        assert await asyncio.to_thread(objects.writer_started.wait, 1)
        threading.Thread(target=release_after_shutdown_cancellation).start()

    asyncio.run(abandon_pending_writer())

    assert release_snapshot["thread_returned"] is False
    assert release_snapshot["quiescent"] is False


def test_checkpoint_prune_waits_for_writer_quiescence_before_object_delete() -> None:
    from gods_mlops.training.checkpoints import S3CheckpointStore

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

    class MemoryObjects:
        def __init__(self):
            self.objects = {}

        def write_immutable(self, *, object_key, content, sha256_digest, content_type):
            self.objects[object_key] = content

        def read_source(self, *, object_key, sha256_digest, size_bytes):
            if object_key not in self.objects:
                raise FileNotFoundError(object_key)
            return self.objects[object_key]

        def delete_object(self, *, object_key):
            self.objects.pop(object_key, None)

    objects = MemoryObjects()
    store = S3CheckpointStore(objects=objects, bucket="gods-test")
    prepared = store.prepare(identity=identity, payload=b"checkpoint A", reservation_bytes=1024)
    verified = store.commit(prepared)
    previous = {
        "uri": verified.uri,
        "sha256": verified.sha256,
        "size_bytes": verified.size_bytes,
        "metadata_size_bytes": 0,
        "identity": identity.as_dict(),
        "operation_id": "operation-A",
        "write_lifetime_id": "operation-A",
    }
    expected_previous = previous

    class Repository:
        def __init__(self):
            self.gate_calls = 0
            self.completed = []

        async def pending_checkpoint_prunes_for(self, _job_id):
            return [previous]

        async def checkpoint_prune_writer_gate(self, *, job_id, previous):
            self.gate_calls += 1
            assert job_id == identity.job_id
            assert previous == expected_previous
            return False

        async def complete_checkpoint_prune(self, *, job_id, previous):
            self.completed.append(previous)
            return True

    repository = Repository()
    queue = JobQueue(repository=repository, sources=object())

    deleted = asyncio.run(queue.retry_pending_checkpoint_prunes(store=store, job_id=identity.job_id))

    assert deleted == []
    assert repository.gate_calls == 1
    assert repository.completed == []
    assert prepared.object_key in objects.objects


def test_writer_start_is_rejected_while_its_checkpoint_lifetime_is_pruning() -> None:
    from gods_mlops.jobs.checkpoints import StaleCheckpointOwnerError
    from gods_mlops.jobs.queue import PostgresJobQueueRepository

    job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    lease_token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
    database_now = datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
    identity = {"job_id": job_id}
    prune = {
        "operation_id": "operation-A",
        "write_lifetime_id": "operation-A",
        "uri": "s3://gods-test/jobs/a0320b59/checkpoints/checkpoint-A",
        "sha256": "a" * 64,
    }
    pending = {
        "operation_id": "operation-B",
        "operation": "checkpoint",
        "uri": prune["uri"],
        "sha256": prune["sha256"],
        "identity": identity,
        "writer_quiescence_required": True,
    }
    job = {
        "job_id": job_id,
        "state": "running",
        "lease_token": lease_token,
        "lease_generation": 2,
    }
    lease = {
        "job_id": job_id,
        "lease_token": lease_token,
        "fencing_token": 2,
        "expires_at": database_now + timedelta(minutes=1),
    }
    rows = [
        {"event_type": "checkpoint_prune_pending", "details": prune},
        {"event_type": "artifact_write_pending", "details": pending},
    ]
    writes = []

    class Connection:
        def transaction(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def fetchrow(self, query, *_args):
            if "gods_mlops_worker_artifact_deadlines" in query:
                return None
            return None

        async def fetch(self, query, *_args):
            if "checkpoint_prune_pending" in query:
                return rows
            return [row for row in rows if row["event_type"] != "checkpoint_prune_pending"]

        async def fetchval(self, query, *_args):
            return database_now if "clock_timestamp" in query else False

        async def execute(self, query, *_args):
            writes.append(query)

    connection = Connection()

    class Pool:
        def acquire(self):
            return connection

    class Repository(PostgresJobQueueRepository):
        async def ensure_schema(self):
            return None

        async def _get_pool(self):
            return Pool()

        async def _lock_job_then_lease(self, *_args, **_kwargs):
            return job, lease

    with pytest.raises(StaleCheckpointOwnerError, match="prun"):
        asyncio.run(
            Repository(database_url="postgresql://unused").record_artifact_writer_started(
                job_id=job_id,
                lease_token=lease_token,
                operation_id="operation-B",
                writer_attempt_id=str(uuid4()),
            )
        )
    assert writes == []
    rows.append({"event_type": "checkpoint_pruned", "details": prune})
    asyncio.run(
        Repository(database_url="postgresql://unused").record_artifact_writer_started(
            job_id=job_id,
            lease_token=lease_token,
            operation_id="operation-B",
            writer_attempt_id=str(uuid4()),
        )
    )
    assert len(writes) == 1


def test_checkpoint_prune_gate_waits_for_durable_writer_quiescence() -> None:
    from gods_mlops.jobs.queue import PostgresJobQueueRepository

    job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    previous = {
        "uri": "s3://gods-test/jobs/a0320b59/checkpoints/checkpoint-A",
        "sha256": "a" * 64,
        "size_bytes": 12,
        "metadata_size_bytes": 0,
        "identity": {"job_id": job_id},
        "operation_id": "operation-A",
        "write_lifetime_id": "operation-A",
    }
    job = {"job_id": job_id, "state": "running", "checkpoint_uri": "s3://gods-test/checkpoint-B"}
    events = [
        {"event_type": "checkpoint_prune_pending", "details": previous},
        {
            "event_type": "artifact_write_started",
            "details": {"operation_id": "operation-A", "writer_attempt_id": "attempt-A"},
        },
    ]

    class Connection:
        def transaction(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def fetchrow(self, query, *_args):
            return job if "gods_mlops_jobs" in query else None

        async def fetch(self, query, *_args):
            return events

    connection = Connection()

    class Pool:
        def acquire(self):
            return connection

    class Repository(PostgresJobQueueRepository):
        async def ensure_schema(self):
            return None

        async def _get_pool(self):
            return Pool()

    ready = asyncio.run(
        Repository(database_url="postgresql://unused").checkpoint_prune_writer_gate(
            job_id=job_id,
            previous=previous,
        )
    )

    assert ready is False


def test_checkpoint_prune_refund_rechecks_writer_quiescence() -> None:
    from gods_mlops.jobs.queue import PostgresJobQueueRepository

    job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    previous = {
        "uri": "s3://gods-test/jobs/a0320b59/checkpoints/checkpoint-A",
        "sha256": "a" * 64,
        "size_bytes": 12,
        "metadata_size_bytes": 0,
        "identity": {"job_id": job_id},
        "operation_id": "operation-A",
        "write_lifetime_id": "operation-A",
    }
    job = {"job_id": job_id, "state": "running", "checkpoint_uri": "s3://gods-test/checkpoint-B"}
    rows = [
        {
            "event_id": 1,
            "event_type": "artifact_write_started",
            "details": {"operation_id": "operation-A", "writer_attempt_id": "attempt-A"},
        },
        {"event_id": 2, "event_type": "checkpoint_prune_pending", "details": previous},
    ]
    writes = []

    class Connection:
        def transaction(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def fetchrow(self, query, *_args):
            if "ingestion_storage_usage" in query:
                return {"used_bytes": 100}
            if "gods_mlops_jobs" in query:
                return job
            if "gods_mlops_artifact_reservations" in query:
                return {"state": "reserved", "reserved_bytes": 100, "consumed_bytes": 20}
            return None

        async def fetch(self, _query, *_args):
            return rows

        async def execute(self, query, *_args):
            writes.append(query)

    connection = Connection()

    class Pool:
        def acquire(self):
            return connection

    class Repository(PostgresJobQueueRepository):
        async def ensure_schema(self):
            return None

        async def _get_pool(self):
            return Pool()

    with pytest.raises(ValueError, match="active or unknown writer"):
        asyncio.run(
            Repository(database_url="postgresql://unused").complete_checkpoint_prune(
                job_id=job_id,
                previous=previous,
            )
        )
    assert writes == []


def test_checkpoint_replacement_uses_prune_gate_instead_of_direct_adapter_delete() -> None:
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
    previous = {
        "uri": "file:///tmp/checkpoint-A",
        "sha256": "a" * 64,
        "size_bytes": 12,
        "metadata_size_bytes": 0,
        "identity": identity.as_dict(),
        "operation_id": "operation-A",
        "write_lifetime_id": "operation-A",
    }
    verified = SimpleNamespace(
        identity=identity,
        sha256="b" * 64,
        size_bytes=12,
        uri="file:///tmp/checkpoint-B",
        path=None,
    )

    class Repository:
        def __init__(self):
            self.gate_calls = 0

        async def get_job(self, _job_id):
            return {"phase": "probe", "model_kind": "clip", "config_version": "probe-v1"}

        async def checkpoint_identity(self, _job_id):
            return identity

        async def lease_is_current(self, *_args):
            return True

        async def get_profile(self, **_kwargs):
            return {"checkpoint_reservation_bytes": 1024}

        async def checkpoint_metadata_for(self, _job_id):
            return previous

        async def pending_checkpoint_prunes_for(self, _job_id):
            return [previous]

        async def checkpoint_prune_writer_gate(self, **_kwargs):
            self.gate_calls += 1
            return False

        async def commit_checkpoint(self, **_kwargs):
            return verified

        async def complete_checkpoint_prune(self, **_kwargs):
            return True

    class Store:
        def __init__(self):
            self.direct_prunes = 0
            self.uri_prunes = 0

        def prepare(self, **_kwargs):
            return SimpleNamespace(
                identity=identity,
                sha256="b" * 64,
                size_bytes=12,
                metadata_size_bytes=0,
                object_key=None,
                uri="file:///tmp/checkpoint-B",
                previous_uri=previous["uri"],
                previous_sha256=previous["sha256"],
                previous_size_bytes=previous["size_bytes"],
                previous_metadata_size_bytes=0,
            )

        def prune_previous(self, _prepared):
            self.direct_prunes += 1

        def prune_uri(self, *_args, **_kwargs):
            self.uri_prunes += 1

    repository = Repository()
    store = Store()
    queue = JobQueue(repository=repository, sources=object())

    result = asyncio.run(
        queue.save_checkpoint(
            store=store,
            job_id=identity.job_id,
            lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
            identity=identity,
            payload=b"checkpoint B",
        )
    )

    assert result is verified
    assert repository.gate_calls == 2
    assert store.direct_prunes == 0
    assert store.uri_prunes == 0


def _blocking_write(objects: _BlockingObjects):
    objects.writer_thread = threading.get_ident()
    objects.writer_started.set()
    try:
        if not objects.release_writer.wait(timeout=5):
            raise TimeoutError("test writer was not released")
        return "writer-complete"
    finally:
        objects.writer_returned.set()


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


def test_checkpoint_and_result_finalizers_recheck_lease_after_reservation_wait() -> None:
    from gods_mlops.jobs.checkpoints import StaleCheckpointOwnerError
    from gods_mlops.jobs.queue import PostgresJobQueueRepository

    async def exercise(operation: str, *, expired_deadline: bool = False) -> None:
        start = datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
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
        job = {
            **identity.as_dict(),
            "state": "running",
            "lease_token": "8ad96890-3434-4f07-85bb-8cde17a2b009",
            "lease_generation": 2,
            "target_phase": "training",
            "checkpoint_uri": None,
            "checkpoint_sha256": None,
            "checkpoint_identity": None,
        }
        lease = {
            "job_id": identity.job_id,
            "lease_token": job["lease_token"],
            "fencing_token": 2,
            "expires_at": start + timedelta(seconds=60 if expired_deadline else 15),
        }
        invocation_id = str(uuid4()) if expired_deadline else None
        artifact_deadline_at = start + timedelta(seconds=15) if expired_deadline else None
        deadline_row = (
            {
                "job_id": identity.job_id,
                "fencing_token": 2,
                "lease_token": job["lease_token"],
                "controller_invocation_id": invocation_id,
                "artifact_deadline_at": artifact_deadline_at,
            }
            if expired_deadline
            else None
        )
        profile = {"checkpoint_reservation_bytes": 1024, "result_reservation_bytes": 1024}
        reservation = {"state": "reserved", "reserved_bytes": 4096, "consumed_bytes": 0}
        state = {"now": start, "clock_samples": [], "writes": []}

        class Connection:
            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def fetchrow(self, query, *_args):
                if "ingestion_storage_usage" in query:
                    return {"used_bytes": 0}
                if "gods_mlops_worker_artifact_deadlines" in query:
                    return deadline_row
                if "gods_mlops_resource_profiles" in query:
                    return profile
                if "gods_mlops_artifact_reservations" in query:
                    state["now"] = start + timedelta(seconds=20)
                    return reservation
                return None

            async def fetchval(self, query, *_args):
                if "clock_timestamp" in query:
                    state["clock_samples"].append(state["now"])
                    return state["now"]
                return False

            async def fetch(self, _query, *_args):
                return []

            async def execute(self, query, *_args):
                state["writes"].append(query)
                return "UPDATE 1"

        connection = Connection()

        class Pool:
            def acquire(self):
                return connection

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

            async def _lock_job_then_lease(self, *_args, **_kwargs):
                return job, lease

        repository = Repository(database_url="postgresql://unused")
        payload = b"artifact"
        digest = sha256(payload).hexdigest()
        prepared = SimpleNamespace(
            identity=identity,
            kind="model",
            uri=f"s3://gods-test/jobs/{identity.job_id}/{digest}.artifact",
            object_key=f"jobs/{identity.job_id}/{digest}.artifact",
            sha256=digest,
            size_bytes=len(payload),
            metadata_size_bytes=0,
            previous_uri=None,
            previous_sha256=None,
            previous_size_bytes=None,
        )
        store = SimpleNamespace(_bucket="gods-test", _prefix="jobs")
        verified = SimpleNamespace(
            identity=identity,
            kind="model",
            uri=prepared.uri,
            object_key=prepared.object_key,
            sha256=digest,
            size_bytes=len(payload),
            path=None,
        )

        with pytest.raises(
            TimeoutError if expired_deadline else StaleCheckpointOwnerError,
            match="expired" if expired_deadline else "unexpired fence",
        ):
            if operation == "checkpoint":
                await repository.commit_checkpoint(
                    job_id=identity.job_id,
                    lease_token=job["lease_token"],
                    identity=identity,
                    prepared=prepared,
                    store=store,
                    verified_artifact=verified,
                    artifact_invocation_id=invocation_id,
                    artifact_deadline_at=artifact_deadline_at,
                )
            else:
                await repository.commit_result_artifact(
                    job_id=identity.job_id,
                    lease_token=job["lease_token"],
                    identity=identity,
                    prepared=prepared,
                    store=store,
                    source_registry=object(),
                    verified_artifact=verified,
                    artifact_invocation_id=invocation_id,
                    artifact_deadline_at=artifact_deadline_at,
                )

        assert state["clock_samples"][-1] == start + timedelta(seconds=20)
        assert not any("checkpoint_uri =" in query for query in state["writes"])
        assert not any("checkpoint_committed" in query for query in state["writes"])
        assert not any("result_artifact_committed" in query for query in state["writes"])

    asyncio.run(exercise("checkpoint"))
    asyncio.run(exercise("result"))
    asyncio.run(exercise("checkpoint", expired_deadline=True))
    asyncio.run(exercise("result", expired_deadline=True))




def test_terminal_reservation_settlement_defers_while_worker_lease_is_retained() -> None:
    from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository

    async def exercise() -> None:
        job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
        writes = []

        class Connection:
            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def fetchrow(self, query, *_args):
                if "ingestion_storage_usage" in query:
                    return {"used_bytes": 100}
                if "gods_mlops_jobs" in query:
                    return {"job_id": job_id, "state": "failed"}
                if "gods_mlops_artifact_reservations" in query:
                    return {
                        "job_id": job_id,
                        "state": "reserved",
                        "reserved_bytes": 100,
                        "consumed_bytes": 10,
                    }
                return None

            async def fetchval(self, query, *_args):
                if "gods_mlops_gpu_leases" in query:
                    return True
                return False

            async def fetch(self, *_args):
                return []

            async def execute(self, query, *_args):
                writes.append(query)

        connection = Connection()

        class Pool:
            def acquire(self):
                return connection

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

            async def record_probe_measurement(self, **_kwargs):
                return {"result_state": "succeeded"}

        queue = JobQueue(
            repository=Repository(database_url="postgresql://unused"),
            sources=object(),
        )
        result = await queue.record_probe_measurement(
            job_id=job_id,
            lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
            exit_code=0,
            peak_allocated_mib=100,
            peak_reserved_mib=100,
            optimizer_steps=3,
            checkpoint_resumed=True,
            checkpoint_sha256="a" * 64,
        )

        assert result["result_state"] == "succeeded"
        assert not any("SET used_bytes" in query for query in writes)
        assert not any("state='settled'" in query for query in writes)

    asyncio.run(exercise())


def test_terminal_reservation_settlement_defers_without_refund_while_writer_is_unmatched() -> None:
    import json

    from gods_mlops.jobs.queue import PostgresJobQueueRepository

    async def exercise() -> None:
        job_id = "bbbbbbbb-2222-4333-8444-555555555555"
        writes: list[str] = []

        class Connection:
            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def fetchrow(self, query, *_args):
                if "ingestion_storage_usage" in query:
                    return {"used_bytes": 100}
                if "gods_mlops_jobs" in query:
                    return {"job_id": job_id, "state": "failed"}
                if "gods_mlops_artifact_reservations" in query:
                    return {
                        "job_id": job_id,
                        "state": "reserved",
                        "reserved_bytes": 100,
                        "consumed_bytes": 10,
                    }
                return None

            async def fetchval(self, query, *_args):
                if "gods_mlops_gpu_leases" in query:
                    return False
                return False

            async def fetch(self, query, *_args):
                if "artifact_write_started" in query:
                    return [{
                        "event_type": "artifact_write_started",
                        "details": json.dumps({
                            "operation_id": "checkpoint-operation",
                            "writer_attempt_id": "cccccccc-3333-4444-8555-666666666666",
                        }),
                    }]
                return []

            async def execute(self, query, *_args):
                writes.append(query)

        connection = Connection()

        class Pool:
            def acquire(self):
                return connection

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

        repository = Repository(database_url="postgresql://unused")

        result = await repository.settle_artifact_reservation(job_id)

        assert result["state"] == "failed"
        assert writes == []

    asyncio.run(exercise())


def test_released_terminal_reservation_sweep_retries_transient_and_writer_deferrals() -> None:
    from gods_mlops.jobs.queue import JobQueue

    async def exercise() -> None:
        job_id = "dddddddd-4444-4555-8666-777777777777"

        class Repository:
            def __init__(self) -> None:
                self.state = "reserved"
                self.writer_unmatched = True
                self.fail_next_settlement = True

            async def list_released_terminal_artifact_reservations(self, **_kwargs):
                return [job_id] if self.state == "reserved" else []

            async def settle_artifact_reservation(self, requested_job_id):
                assert requested_job_id == job_id
                if self.fail_next_settlement:
                    self.fail_next_settlement = False
                    raise ConnectionError("temporary database disconnect")
                if not self.writer_unmatched:
                    self.state = "settled"
                return {"job_id": job_id, "state": "failed"}

            async def artifact_reservation_for(self, requested_job_id):
                assert requested_job_id == job_id
                return {"job_id": job_id, "state": self.state}

        repository = Repository()
        queue = JobQueue(repository=repository, sources=object())
        reconcile = getattr(queue, "settle_released_terminal_artifact_reservations", None)
        assert callable(reconcile), "queue must expose a durable terminal-reservation reconciliation seam"

        failed_attempt = await reconcile()
        assert failed_attempt == {"examined": 1, "settled": 0, "deferred": 1}
        assert repository.state == "reserved"

        writer_deferred = await reconcile()
        assert writer_deferred == {"examined": 1, "settled": 0, "deferred": 1}
        assert repository.state == "reserved"

        repository.writer_unmatched = False
        quiescent = await reconcile()
        assert quiescent == {"examined": 1, "settled": 1, "deferred": 0}
        assert repository.state == "settled"

        repeated = await reconcile()
        assert repeated == {"examined": 0, "settled": 0, "deferred": 0}

    asyncio.run(exercise())


def test_released_terminal_reservation_sweep_surfaces_source_invariant_errors() -> None:
    from gods_mlops.jobs.queue import JobQueue

    async def exercise() -> None:
        class Repository:
            async def list_released_terminal_artifact_reservations(self, **_kwargs):
                return ["eeeeeeee-5555-4666-8777-888888888888"]

            async def settle_artifact_reservation(self, _job_id):
                raise ValueError("only terminal jobs can settle their artifact reservation")

        queue = JobQueue(repository=Repository(), sources=object())
        reconcile = getattr(queue, "settle_released_terminal_artifact_reservations", None)
        assert callable(reconcile), "queue must expose a durable terminal-reservation reconciliation seam"

        with pytest.raises(ValueError, match="only terminal jobs"):
            await reconcile()

    asyncio.run(exercise())
