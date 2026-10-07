from __future__ import annotations

import asyncio
import inspect
import json
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

os.environ.setdefault(
    "GODS_MLOPS_MODEL_LOCK",
    "/mnt/data/gods-mlops-clip-runtime-50d93e9/models/lock.json",
)

import gods_mlops.jobs.queue as queue_module
from gods_mlops.jobs.models import (
    EvaluationProbeCheckpointSource,
    ProbeInput,
    ProcessIdentity,
)
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.training.contracts import locked_model
from gods_mlops.training.probe_setup import candidate_profile

_NOW = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
_JOB_ID = "e4d82c26-8f0a-4f45-8b88-5fe84302d948"
_LEASE_TOKEN = "8ad96890-3434-4f07-85bb-8cde17a2b009"
_TRAINING_JOB_ID = "a0320b59-663c-4cdc-b893-086bb970ea60"
_SOURCE_CHECKPOINT_SHA = "b" * 64
_REASON = "task11_clip_evaluation_cursor"


def _require(value: Any, name: str):
    if not callable(value):
        pytest.fail(f"{name} is part of the evaluation progress queue contract")
    return value


def _typed_input_and_profile():
    model = locked_model("clip")
    training_identity = {
        "job_id": _TRAINING_JOB_ID,
        "input_kind": "probe_input",
        "input_id": "task8-clip-synthetic-probe-v1",
        "input_sha256": "1" * 64,
        "phase": "probe",
        "model_kind": "clip",
        "config_version": "task9-clip-224-microbatch2-symmetric-ce-fp64-probe-v1",
        "config_sha256": "2" * 64,
        "dataset_version": None,
    }
    source = EvaluationProbeCheckpointSource(
        training_probe_job_id=_TRAINING_JOB_ID,
        model_kind="clip",
        model_id=model.model_id,
        model_revision=model.revision,
        checkpoint_uri=(
            f"s3://gods-task8-test/jobs/{_TRAINING_JOB_ID}/checkpoints/"
            + "3" * 64
            + f"/{_SOURCE_CHECKPOINT_SHA}.checkpoint"
        ),
        checkpoint_sha256=_SOURCE_CHECKPOINT_SHA,
        checkpoint_size_bytes=123,
        checkpoint_identity=training_identity,
        worker_image_id="sha256:" + "4" * 64,
        source_commit="5" * 40,
        runtime_evidence_sha256="6" * 64,
    )
    profile = candidate_profile("clip", target_phase="evaluation")
    probe_input = ProbeInput(
        probe_input_id=f"task9-clip-evaluation-probe-{_SOURCE_CHECKPOINT_SHA}",
        model_kind="clip",
        target_phase="evaluation",
        config_version=profile.config_version,
        manifest_object_key="probe-inputs/task9-clip-evaluation/manifest.json",
        input_sha256="7" * 64,
        object_size_bytes=256,
        evaluation_checkpoint_source=source,
    )
    return probe_input, profile


def _target(probe_input: ProbeInput, profile, *, submitted_after: datetime | None = None) -> dict[str, Any]:
    source = probe_input.evaluation_checkpoint_source
    assert source is not None
    return {
        "phase": "probe",
        "target_phase": "evaluation",
        "model_kind": "clip",
        "stage": "queries",
        "next_index": 2,
        "total_items": 2,
        "completed_batch_count": 1,
        "input_sha256": probe_input.input_sha256,
        "config_sha256": profile.config_sha256,
        "training_probe_job_id": source.training_probe_job_id,
        "training_source_checkpoint_sha256": source.checkpoint_sha256,
        "submitted_after": (submitted_after or (_NOW - timedelta(seconds=1))).isoformat(),
    }


def _database_state(*, job_state: str = "queued", lease_generation: int = 0):
    probe_input, profile = _typed_input_and_profile()
    source = probe_input.evaluation_checkpoint_source
    assert source is not None
    job = {
        "job_id": _JOB_ID,
        "phase": "probe",
        "target_phase": "evaluation",
        "input_kind": "probe_input",
        "input_id": probe_input.probe_input_id,
        "input_sha256": probe_input.input_sha256,
        "source_refs": probe_input.as_dict(),
        "model_kind": "clip",
        "config_version": profile.config_version,
        "config_sha256": profile.config_sha256,
        "profile_state_snapshot": "candidate",
        "state": job_state,
        "lease_token": None,
        "lease_generation": lease_generation,
        "lease_expires_at": None,
        "owner_pid": None,
        "owner_start_ticks": None,
        "owner_uid": None,
        "checkpoint_sha256": None,
        "checkpoint_uri": None,
        "created_at": _NOW,
    }
    profile_row = {
        "phase": "probe",
        "target_phase": "evaluation",
        "model_kind": "clip",
        "config_version": profile.config_version,
        "config_sha256": profile.config_sha256,
        "profile_state": "candidate",
        "config_json": profile.config,
    }
    return {
        "job": job,
        "profile": profile_row,
        "lease": None,
        "deadline": None,
        "events": [],
        "next_event_id": 1,
        "now": _NOW,
        "probe_input": probe_input,
        "profile_object": profile,
        "source": source,
    }


class _FakeTransaction:
    def __init__(self, state):
        self.state = state
        self.snapshot = None

    async def __aenter__(self):
        self.snapshot = deepcopy(self.state)
        return self

    async def __aexit__(self, error_type, error, traceback):
        if error_type is not None:
            self.state.clear()
            self.state.update(self.snapshot)
        return False


class _FakeConnection:
    def __init__(self, state):
        self.state = state
        self.lock_order = []

    def transaction(self, *, readonly=False):
        return _FakeTransaction(self.state)

    async def fetchrow(self, query: str, *args):
        normalized = " ".join(query.split()).lower()
        if "from gods_mlops_jobs" in normalized:
            self.lock_order.append("job") if "for update" in normalized else None
            return deepcopy(self.state["job"])
        if "from gods_mlops_gpu_leases" in normalized:
            self.lock_order.append("lease") if "for update" in normalized else None
            lease = self.state["lease"]
            if lease is None:
                return None
            if len(args) > 1 and str(lease["lease_token"]) != str(args[1]):
                return None
            return deepcopy(lease)
        if "from gods_mlops_resource_profiles" in normalized:
            return deepcopy(self.state["profile"])
        if "from gods_mlops_worker_artifact_deadlines" in normalized:
            deadline = self.state["deadline"]
            if deadline is None:
                return None
            if len(args) > 1 and int(deadline["fencing_token"]) != int(args[1]):
                return None
            return deepcopy(deadline)
        if "from gods_mlops_job_events" in normalized:
            rows = self._matching_events(args)
            return deepcopy(rows[0]) if rows else None
        raise AssertionError(f"unexpected fake fetchrow query: {normalized}")

    async def fetch(self, query: str, *args):
        normalized = " ".join(query.split()).lower()
        if "from gods_mlops_job_events" in normalized:
            return deepcopy(self._matching_events(args))
        raise AssertionError(f"unexpected fake fetch query: {normalized}")

    async def fetchval(self, query: str, *args):
        normalized = " ".join(query.split()).lower()
        if "clock_timestamp()" in normalized:
            return self.state["now"]
        if "insert into gods_mlops_job_events" in normalized:
            if "evaluation_yield_armed" in normalized:
                job_id, reason, details = args
                event_type, event_state, fence = "evaluation_yield_armed", "queued", None
            elif "evaluation_batch_completed" in normalized:
                job_id, fence, details = args
                event_type, event_state, reason = "evaluation_batch_completed", "running", None
            elif "yield_requested" in normalized:
                job_id, reason, fence, details = args
                event_type, event_state = "yield_requested", "yield_requested"
            else:
                raise AssertionError(f"unexpected event insert: {normalized}")
            event_id = self.state["next_event_id"]
            self.state["next_event_id"] += 1
            self.state["events"].append(
                {
                    "event_id": event_id,
                    "job_id": str(job_id),
                    "event_type": event_type,
                    "state": event_state,
                    "reason_code": reason,
                    "fencing_token": fence,
                    "details": deepcopy(json.loads(details) if isinstance(details, str) else details),
                }
            )
            return event_id
        raise AssertionError(f"unexpected fake fetchval query: {normalized}")

    async def execute(self, query: str, *args):
        normalized = " ".join(query.split()).lower()
        if "update gods_mlops_jobs" in normalized:
            job_id, reason, details, lease_token, fence = args
            if (
                str(self.state["job"]["job_id"]) != str(job_id)
                or str(self.state["job"]["lease_token"]) != str(lease_token)
                or int(self.state["job"]["lease_generation"]) != int(fence)
                or self.state["job"]["state"] != "running"
            ):
                return "UPDATE 0"
            self.state["job"].update(
                state="yield_requested",
                reason_code=reason,
                reason_detail=deepcopy(json.loads(details) if isinstance(details, str) else details),
                retryable=True,
            )
            return "UPDATE 1"
        if "update gods_mlops_gpu_leases" in normalized:
            _, reason, lease_token, fence = args
            lease = self.state["lease"]
            if (
                lease is None
                or str(lease["lease_token"]) != str(lease_token)
                or int(lease["fencing_token"]) != int(fence)
            ):
                return "UPDATE 0"
            lease["yield_reason"] = reason
            return "UPDATE 1"
        raise AssertionError(f"unexpected fake execute query: {normalized}")

    def _matching_events(self, args):
        job_id = str(args[0]) if args else _JOB_ID
        return [event for event in self.state["events"] if event["job_id"] == job_id]


class _FakeAcquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, error_type, error, traceback):
        return False


class _FakePool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _FakeAcquire(self.connection)


def _repository(state, monkeypatch):
    repository = PostgresJobQueueRepository(database_url="postgresql://unused/fixture")
    connection = _FakeConnection(state)

    async def no_schema():
        return None

    async def fake_pool():
        return _FakePool(connection)

    async def source_authority_ok(connection, job):
        return None

    repository.ensure_schema = no_schema
    repository._get_pool = fake_pool
    monkeypatch.setattr(queue_module, "_evaluation_probe_checkpoint_source_error", source_authority_ok)
    return repository, connection


def _activate_generation_one(state):
    owner = ProcessIdentity(pid=12345, start_ticks=67890, uid=10001)
    lease_expiry = _NOW + timedelta(seconds=60)
    token = _LEASE_TOKEN
    state["job"].update(
        state="running",
        lease_token=token,
        lease_generation=1,
        lease_expires_at=lease_expiry,
        owner_pid=owner.pid,
        owner_start_ticks=owner.start_ticks,
        owner_uid=owner.uid,
    )
    state["lease"] = {
        "job_id": _JOB_ID,
        "lease_token": token,
        "fencing_token": 1,
        "expires_at": lease_expiry,
        "owner_pid": owner.pid,
        "owner_start_ticks": owner.start_ticks,
        "owner_uid": owner.uid,
        "yield_reason": None,
    }
    state["deadline"] = {
        "job_id": _JOB_ID,
        "fencing_token": 1,
        "lease_token": token,
        "controller_invocation_id": "8f8cfb84-98a4-46b8-aabc-778da1d58aa6",
        "artifact_deadline_at": _NOW + timedelta(minutes=10),
    }
    return owner, state["deadline"]


def _progress(state):
    source = state["source"]
    return {
        "phase": "probe",
        "target_phase": "evaluation",
        "model_kind": "clip",
        "stage": "queries",
        "next_index": 2,
        "total_items": 2,
        "completed_batch_count": 1,
        "input_sha256": state["job"]["input_sha256"],
        "config_sha256": state["job"]["config_sha256"],
        "training_probe_job_id": source.training_probe_job_id,
        "training_source_checkpoint_sha256": source.checkpoint_sha256,
    }


def test_queue_yield_request_accepts_only_the_optional_progress_target() -> None:
    parameters = inspect.signature(JobQueue.request_yield).parameters
    assert "after_progress" in parameters
    assert parameters["after_progress"].kind is inspect.Parameter.KEYWORD_ONLY


def test_immediate_queue_yield_keeps_its_existing_return_and_call_semantics() -> None:
    class Repository:
        def __init__(self):
            self.calls = []

        async def request_yield(self, job_id, reason, *, after_progress=None):
            self.calls.append((job_id, reason, after_progress))
            return {"state": "yield_requested"}

    queue = JobQueue.__new__(JobQueue)
    queue._repository = Repository()

    result = asyncio.run(queue.request_yield(_JOB_ID, "existing_immediate_reason"))

    assert result is None
    assert queue._repository.calls == [(_JOB_ID, "existing_immediate_reason", None)]


def test_arm_is_first_lease_only_and_same_fence_replay_does_not_duplicate_events(monkeypatch) -> None:
    request_yield = _require(PostgresJobQueueRepository.request_yield, "request_yield(after_progress=...)")
    pending = _require(
        getattr(PostgresJobQueueRepository, "pending_evaluation_progress_yield", None),
        "pending_evaluation_progress_yield",
    )
    state = _database_state()
    repository, connection = _repository(state, monkeypatch)
    target = _target(state["probe_input"], state["profile_object"])

    first = asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=target))
    replay = asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=target))

    assert first["arm_event_id"] == replay["arm_event_id"]
    assert first["request_id"] == replay["request_id"]
    assert first["job_state"] == "queued"
    assert first["lease_generation"] == 0
    assert first["pending_first_attempt"] is True
    assert state["job"]["state"] == "queued"
    assert [event["event_type"] for event in state["events"]] == ["evaluation_yield_armed"]
    assert connection.lock_order[:2] == ["job", "lease"]

    conflicting_target = _target(
        state["probe_input"],
        state["profile_object"],
        submitted_after=_NOW - timedelta(seconds=2),
    )
    with pytest.raises(RuntimeError, match="conflicts with a retained arm"):
        asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=conflicting_target))
    assert len(state["events"]) == 1

    _activate_generation_one(state)
    state["job"]["checkpoint_sha256"] = "d" * 64
    retained = asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=target))
    assert retained["arm_event_id"] == first["arm_event_id"]
    assert retained["pending_first_attempt"] is False
    pending_arm = asyncio.run(
        pending(
            repository,
            job_id=_JOB_ID,
            lease_token=_LEASE_TOKEN,
            fencing_token=1,
            input_sha256=state["job"]["input_sha256"],
            config_sha256=state["job"]["config_sha256"],
            training_source_checkpoint_sha256=state["source"].checkpoint_sha256,
        )
    )
    assert pending_arm["arm_event_id"] == first["arm_event_id"]
    assert pending_arm["training_source_checkpoint_sha256"] == _SOURCE_CHECKPOINT_SHA

    state["job"].update(state="running", lease_generation=2, lease_token="another-lease-token")
    state["lease"].update(fencing_token=2, lease_token="another-lease-token")
    stale = asyncio.run(
        pending(
            repository,
            job_id=_JOB_ID,
            lease_token="another-lease-token",
            fencing_token=2,
            input_sha256=state["job"]["input_sha256"],
            config_sha256=state["job"]["config_sha256"],
            training_source_checkpoint_sha256=state["source"].checkpoint_sha256,
        )
    )
    assert stale is None
    retained_after_retry = asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=target))
    assert retained_after_retry["arm_event_id"] == first["arm_event_id"]
    assert retained_after_retry["pending_first_attempt"] is False
    assert len(state["events"]) == 1


def test_progress_event_and_yield_transition_are_atomic_and_replayable(monkeypatch) -> None:
    request_yield = _require(PostgresJobQueueRepository.request_yield, "request_yield(after_progress=...)")
    pending = _require(
        getattr(PostgresJobQueueRepository, "pending_evaluation_progress_yield", None),
        "pending_evaluation_progress_yield",
    )
    record = _require(
        getattr(PostgresJobQueueRepository, "record_evaluation_progress", None),
        "record_evaluation_progress",
    )
    state = _database_state()
    repository, connection = _repository(state, monkeypatch)
    target = _target(state["probe_input"], state["profile_object"])
    armed = asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=target))
    owner, deadline = _activate_generation_one(state)
    arm = asyncio.run(
        pending(
            repository,
            job_id=_JOB_ID,
            lease_token=_LEASE_TOKEN,
            fencing_token=1,
            input_sha256=state["job"]["input_sha256"],
            config_sha256=state["job"]["config_sha256"],
            training_source_checkpoint_sha256=state["source"].checkpoint_sha256,
        )
    )

    triggered = asyncio.run(
        record(
            repository,
            job_id=_JOB_ID,
            lease_token=_LEASE_TOKEN,
            fencing_token=1,
            arm_event_id=arm["arm_event_id"],
            progress=_progress(state),
            owner=owner,
            artifact_deadline=deadline,
        )
    )

    assert triggered["status"] == "yield_requested"
    assert state["job"]["state"] == "yield_requested"
    assert state["lease"]["yield_reason"] == _REASON
    state["job"]["checkpoint_sha256"] = "d" * 64
    assert state["job"]["checkpoint_sha256"] == "d" * 64
    assert [event["event_type"] for event in state["events"]] == [
        "evaluation_yield_armed",
        "evaluation_batch_completed",
        "yield_requested",
    ]
    progress_event, consumed = state["events"][1:]
    assert progress_event["fencing_token"] == consumed["fencing_token"] == 1
    assert consumed["details"]["arm_event_id"] == armed["arm_event_id"]
    assert consumed["details"]["progress_event_id"] == progress_event["event_id"]
    assert consumed["details"]["training_source_checkpoint_sha256"] == _SOURCE_CHECKPOINT_SHA
    assert not any(event["event_type"] == "checkpoint_committed" for event in state["events"])

    state["job"]["checkpoint_sha256"] = "e" * 64
    replay = asyncio.run(
        record(
            repository,
            job_id=_JOB_ID,
            lease_token=_LEASE_TOKEN,
            fencing_token=1,
            arm_event_id=arm["arm_event_id"],
            progress=_progress(state),
            owner=owner,
            artifact_deadline=deadline,
        )
    )
    assert replay["idempotent_replay"] is True
    assert len(state["events"]) == 3
    assert state["events"][0]["details"]["training_source_checkpoint_sha256"] == _SOURCE_CHECKPOINT_SHA
    assert state["job"]["checkpoint_sha256"] == "e" * 64
    assert connection.lock_order[-2:] == ["job", "lease"]


@pytest.mark.parametrize(
    "invalidity",
    ["expired_lease", "stale_token", "stale_fence", "wrong_owner", "expired_deadline", "wrong_source"],
)
def test_invalid_fence_or_progress_never_publishes_events(invalidity: str, monkeypatch) -> None:
    request_yield = _require(PostgresJobQueueRepository.request_yield, "request_yield(after_progress=...)")
    record = _require(
        getattr(PostgresJobQueueRepository, "record_evaluation_progress", None),
        "record_evaluation_progress",
    )
    state = _database_state()
    repository, _ = _repository(state, monkeypatch)
    target = _target(state["probe_input"], state["profile_object"])
    armed = asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=target))
    owner, deadline = _activate_generation_one(state)
    progress = _progress(state)
    token = _LEASE_TOKEN
    fence = 1
    if invalidity == "expired_lease":
        state["lease"]["expires_at"] = _NOW - timedelta(seconds=1)
        state["job"]["lease_expires_at"] = state["lease"]["expires_at"]
    elif invalidity == "stale_token":
        token = "00000000-0000-4000-8000-000000000000"
    elif invalidity == "stale_fence":
        fence = 2
    elif invalidity == "wrong_owner":
        owner = ProcessIdentity(pid=12346, start_ticks=67891, uid=10001)
    elif invalidity == "expired_deadline":
        deadline = {**deadline, "artifact_deadline_at": _NOW - timedelta(seconds=1)}
    elif invalidity == "wrong_source":
        progress["training_source_checkpoint_sha256"] = "f" * 64

    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        asyncio.run(
            record(
                repository,
                job_id=_JOB_ID,
                lease_token=token,
                fencing_token=fence,
                arm_event_id=armed["arm_event_id"],
                progress=progress,
                owner=owner,
                artifact_deadline=deadline,
            )
        )
    assert [event["event_type"] for event in state["events"]] == ["evaluation_yield_armed"]


def test_arm_fails_closed_for_dedupe_or_admission_race(monkeypatch) -> None:
    if "after_progress" not in inspect.signature(PostgresJobQueueRepository.request_yield).parameters:
        pytest.fail("request_yield(after_progress=...) is part of the evaluation progress queue contract")
    request_yield = _require(PostgresJobQueueRepository.request_yield, "request_yield(after_progress=...)")
    state = _database_state(job_state="running", lease_generation=1)
    repository, _ = _repository(state, monkeypatch)
    target = _target(state["probe_input"], state["profile_object"])

    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=target))
    assert state["events"] == []

    state = _database_state()
    repository, _ = _repository(state, monkeypatch)
    old_target = _target(
        state["probe_input"],
        state["profile_object"],
        submitted_after=_NOW + timedelta(seconds=1),
    )
    with pytest.raises((RuntimeError, ValueError)):
        asyncio.run(request_yield(repository, _JOB_ID, _REASON, after_progress=old_target))
    assert state["events"] == []
