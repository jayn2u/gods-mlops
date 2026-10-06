from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
from uuid import UUID, uuid4

import pytest

from gods_mlops.jobs.queue import (
    DatasetNotReadyForTrainingError,
    JobQueue,
    PostgresJobQueueRepository,
    _operator_retry_dedupe_key,
)


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _Connection:
    def __init__(self):
        self.jobs_by_intent: dict[str, dict] = {}

    def transaction(self):
        return _Transaction()

    async def fetchval(self, query: str, *args):
        if "INSERT INTO gods_mlops_jobs" in query:
            job_id, phase, input_kind, input_id, input_sha256, dataset_version, source_refs, model_kind, target_phase, config_version, config_sha256, profile_state, rerun, dedupe_key, parent_job_id, retry_root_id, oom_retries, reservation_bytes = args
            if dedupe_key is not None and dedupe_key in self.jobs_by_intent:
                return None
            self.jobs_by_intent[dedupe_key] = {
                "job_id": job_id,
                "phase": phase,
                "input_kind": input_kind,
                "input_id": input_id,
                "input_sha256": input_sha256,
                "dataset_version": dataset_version,
                "source_refs": source_refs,
                "model_kind": model_kind,
                "target_phase": target_phase,
                "config_version": config_version,
                "config_sha256": config_sha256,
                "profile_state_snapshot": profile_state,
                "rerun": rerun,
                "dedupe_key": dedupe_key,
                "parent_job_id": parent_job_id,
                "retry_root_id": retry_root_id,
                "oom_retries": oom_retries,
                "artifact_reservation_bytes": reservation_bytes,
            }
            return job_id
        if "SELECT job_id FROM gods_mlops_jobs WHERE dedupe_key = $1" in query:
            row = self.jobs_by_intent.get(args[0])
            return row["job_id"] if row else None
        raise AssertionError(f"unexpected fetchval query: {query}")

    async def fetchrow(self, query: str, *args):
        if "FROM gods_mlops_jobs WHERE dedupe_key = $1" in query:
            return self.jobs_by_intent.get(args[0])
        raise AssertionError(f"unexpected fetchrow query: {query}")

    async def execute(self, query: str, *args):
        return "OK"


class _Pool:
    def __init__(self, connection: _Connection):
        self.connection = connection

    def acquire(self):
        return _Acquire(self.connection)


class _TrainingSource:
    dataset_version = "dataset-123"
    manifest_sha256 = "b" * 64
    target = "detr"

    def __init__(self, reasons=()):
        self.reasons = tuple(reasons)

    def training_block_reasons(self):
        return self.reasons


class _SourceRegistry:
    async def get_training_source(self, dataset_version):
        assert dataset_version == "dataset-123"
        return _TrainingSource()


class _RetryRepository:
    def __init__(self, parent, *, blocked_reasons=()):
        self.parent = parent
        self.blocked_reasons = blocked_reasons
        self.submissions = []

    async def get_job(self, job_id):
        assert job_id == self.parent["job_id"]
        return self.parent

    async def get_profile(self, *, phase, model_kind, config_version):
        assert (phase, model_kind, config_version) == ("training", "detr", "detr-config-v1")
        return {
            "target_phase": None,
            "config_version": config_version,
            "config_sha256": "a" * 64,
            "profile_state": "measured",
            "artifact_reservation_bytes": 4096,
        }

    async def enqueue(self, **request):
        self.submissions.append(request)
        return str(uuid4())


class _QueueOrderingConnection(_Connection):
    def __init__(self, jobs):
        super().__init__()
        self.jobs = jobs
        self.updates = {}

    async def fetch(self, query: str, *args):
        if "FROM gods_mlops_jobs" in query and "FOR UPDATE" in query:
            return [dict(job) for job in sorted(self.jobs, key=lambda item: item["job_id"])]
        raise AssertionError(f"unexpected fetch query: {query}")

    async def execute(self, query: str, *args):
        if "UPDATE gods_mlops_jobs SET queue_order" in query:
            queue_order, job_id = args
            self.updates[str(job_id)] = queue_order
            return "UPDATE 1"
        if "INSERT INTO gods_mlops_job_events" in query:
            return "INSERT 0 1"
        raise AssertionError(f"unexpected execute query: {query}")


class _QueueOrderingPool(_Pool):
    pass


class _CancelTransactionConnection(_Connection):
    def __init__(self, job):
        super().__init__()
        self.job = job
        self.cancel_events = 0

    async def fetchrow(self, query: str, *args):
        if "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid FOR UPDATE" in query:
            return dict(self.job)
        if "UPDATE gods_mlops_jobs SET state = 'cancelled'" in query:
            assert args[0] == str(self.job["job_id"])
            assert self.job["state"] in {"queued", "waiting_profile", "waiting_gpu", "waiting_storage", "waiting_capacity"}
            self.job.update(
                {
                    "state": "cancelled",
                    "reason_code": "operator_cancelled",
                    "reason_detail": json.loads(args[1]),
                    "retryable": False,
                }
            )
            return dict(self.job)
        raise AssertionError(f"unexpected fetchrow query: {query}")

    async def execute(self, query: str, *args):
        if "INSERT INTO gods_mlops_job_events" in query:
            self.cancel_events += 1
            return "INSERT 0 1"
        raise AssertionError(f"unexpected execute query: {query}")


class _CancelRepository:
    def __init__(self):
        self.state = "waiting_gpu"
        self.cancel_event_count = 0
        self.settlement_attempts = 0
        self.reservation_state = "reserved"

    async def get_job(self, job_id):
        return {"job_id": job_id, "state": self.state}

    async def cancel_queued_job(self, job_id, *, actor_sha256):
        if self.state == "cancelled":
            return {"job_id": job_id, "state": "cancelled", "already_cancelled": True}
        self.state = "cancelled"
        self.cancel_event_count += 1
        return {"job_id": job_id, "state": "cancelled", "already_cancelled": False}

    async def settle_artifact_reservation(self, job_id):
        self.settlement_attempts += 1
        if self.settlement_attempts == 1:
            raise OSError("private endpoint detail must not be surfaced")
        self.reservation_state = "settled"
        return {"job_id": job_id, "state": "cancelled"}

    async def artifact_reservation_for(self, job_id):
        return {"state": self.reservation_state}


def _queue_job(*, job_id: str, created_seconds: int, queue_order=None, state="queued"):
    return {
        "job_id": job_id,
        "state": state,
        "queue_order": queue_order,
        "created_at": datetime(2026, 10, 7, tzinfo=UTC) + timedelta(seconds=created_seconds),
        "lease_token": None,
        "owner_pid": None,
        "owner_start_ticks": None,
    }


def _cancel_job(*, job_id: str, state="queued", lease_token=None, owner_pid=None):
    return {
        "job_id": UUID(job_id),
        "phase": "training",
        "state": state,
        "lease_token": lease_token,
        "owner_pid": owner_pid,
        "owner_start_ticks": None,
        "reason_code": None,
        "reason_detail": {},
        "retryable": False,
        "completed_at": None,
        "updated_at": None,
        "queue_order": None,
    }


def _training_parent(*, state="failed"):
    return {
        "job_id": str(uuid4()),
        "retry_root_id": str(uuid4()),
        "phase": "training",
        "state": state,
        "lease_token": None,
        "owner_pid": None,
        "dataset_version": "dataset-123",
        "model_kind": "detr",
        "config_version": "detr-config-v1",
        "source_refs": {"dataset_version": "dataset-123", "manifest_sha256": "b" * 64, "target": "detr"},
    }


def test_operator_retry_intent_digest_is_scoped_and_never_contains_raw_token() -> None:
    parent_job_id = str(uuid4())
    intent_token = "random-intent-token-with-at-least-32-characters"

    first = _operator_retry_dedupe_key(
        parent_job_id=parent_job_id,
        operator_id="operator",
        intent_token=intent_token,
    )

    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
    assert intent_token not in first
    assert first == _operator_retry_dedupe_key(
        parent_job_id=parent_job_id,
        operator_id="operator",
        intent_token=intent_token,
    )
    assert first != _operator_retry_dedupe_key(
        parent_job_id=str(uuid4()),
        operator_id="operator",
        intent_token=intent_token,
    )
    assert first != _operator_retry_dedupe_key(
        parent_job_id=parent_job_id,
        operator_id="different-operator",
        intent_token=intent_token,
    )


def test_duplicate_retry_intent_returns_one_job_and_conflicting_request_is_rejected() -> None:
    async def exercise() -> None:
        connection = _Connection()
        repository = PostgresJobQueueRepository(database_url="postgresql://unused")
        repository._schema_ready = True
        repository._pool = _Pool(connection)
        parent_job_id = str(uuid4())
        intent_token = "random-intent-token-with-at-least-32-characters"
        key = _operator_retry_dedupe_key(
            parent_job_id=parent_job_id,
            operator_id="operator",
            intent_token=intent_token,
        )
        profile = {
            "target_phase": None,
            "config_version": "detr-config-v1",
            "config_sha256": "a" * 64,
            "profile_state": "measured",
            "artifact_reservation_bytes": 4096,
        }
        request = {
            "phase": "training",
            "input_kind": "dataset_version",
            "input_id": "dataset-123",
            "input_sha256": "b" * 64,
            "dataset_version": "dataset-123",
            "source_refs": {"dataset_version": "dataset-123", "manifest_sha256": "b" * 64},
            "model_kind": "detr",
            "profile": profile,
            "rerun": True,
            "parent_job_id": parent_job_id,
            "retry_root_id": parent_job_id,
            "operator_dedupe_key": key,
        }

        first = await repository.enqueue(**request)
        duplicate = await repository.enqueue(**request)

        assert UUID(first)
        assert duplicate == first
        assert len(connection.jobs_by_intent) == 1
        assert intent_token not in str(connection.jobs_by_intent)

        different_source = dict(request)
        different_source["input_sha256"] = "c" * 64
        different_source["source_refs"] = {"dataset_version": "dataset-123", "manifest_sha256": "c" * 64}
        with pytest.raises(ValueError, match="operator retry intent was reused for a different immutable request"):
            await repository.enqueue(**different_source)

    asyncio.run(exercise())


def test_operator_dedupe_key_requires_a_canonical_parent_and_high_entropy_intent() -> None:
    with pytest.raises(ValueError, match="parent_job_id must be a UUID"):
        _operator_retry_dedupe_key(
            parent_job_id="not-a-uuid",
            operator_id="operator",
            intent_token="random-intent-token-with-at-least-32-characters",
        )
    with pytest.raises(ValueError, match="intent token must contain at least 32 characters"):
        _operator_retry_dedupe_key(
            parent_job_id=str(uuid4()),
            operator_id="operator",
            intent_token="short",
        )


def test_retry_job_reuses_current_training_gates_and_binds_the_parent_intent() -> None:
    async def exercise() -> None:
        parent = _training_parent()
        repository = _RetryRepository(parent)
        queue = JobQueue(repository=repository, sources=_SourceRegistry())
        intent_token = "random-intent-token-with-at-least-32-characters"

        await queue.retry_job(
            parent["job_id"],
            operator_id="operator",
            intent_token=intent_token,
        )

        assert len(repository.submissions) == 1
        request = repository.submissions[0]
        assert request["rerun"] is True
        assert request["parent_job_id"] == parent["job_id"]
        assert request["retry_root_id"] == parent["retry_root_id"]
        assert request["operator_dedupe_key"] == _operator_retry_dedupe_key(
            parent_job_id=parent["job_id"],
            operator_id="operator",
            intent_token=intent_token,
        )
        assert intent_token not in str(request)

    asyncio.run(exercise())


def test_retry_job_rejects_probe_and_nonterminal_jobs_without_enqueueing() -> None:
    async def exercise() -> None:
        intent_token = "random-intent-token-with-at-least-32-characters"
        probe = _training_parent()
        probe["phase"] = "probe"
        probe_repository = _RetryRepository(probe)
        probe_queue = JobQueue(repository=probe_repository, sources=_SourceRegistry())
        with pytest.raises(ValueError, match="operator retry is not supported for probe jobs"):
            await probe_queue.retry_job(probe["job_id"], operator_id="operator", intent_token=intent_token)
        assert probe_repository.submissions == []

        running = _training_parent(state="running")
        running_repository = _RetryRepository(running)
        running_queue = JobQueue(repository=running_repository, sources=_SourceRegistry())
        with pytest.raises(ValueError, match="only completed, failed, or cancelled jobs can be rerun"):
            await running_queue.retry_job(running["job_id"], operator_id="operator", intent_token=intent_token)
        assert running_repository.submissions == []

    asyncio.run(exercise())


def test_retry_job_cannot_bypass_current_dataset_eligibility() -> None:
    async def exercise() -> None:
        parent = _training_parent()
        repository = _RetryRepository(parent)
        queue = JobQueue(repository=repository, sources=_BlockedSourceRegistry())

        with pytest.raises(DatasetNotReadyForTrainingError):
            await queue.retry_job(
                parent["job_id"],
                operator_id="operator",
                intent_token="random-intent-token-with-at-least-32-characters",
            )
        assert repository.submissions == []

    asyncio.run(exercise())


def test_reorder_materializes_fifo_then_moves_only_queue_order() -> None:
    async def exercise() -> None:
        older = str(uuid4())
        middle = str(uuid4())
        newer = str(uuid4())
        connection = _QueueOrderingConnection(
            [
                _queue_job(job_id=older, created_seconds=1),
                _queue_job(job_id=middle, created_seconds=2),
                _queue_job(job_id=newer, created_seconds=3),
            ]
        )
        repository = PostgresJobQueueRepository(database_url="postgresql://unused")
        repository._schema_ready = True
        repository._pool = _QueueOrderingPool(connection)

        positions = await repository.reorder_queued_jobs(
            middle,
            before_job_id=older,
            actor_sha256="d" * 64,
        )

        assert positions == [middle, older, newer]
        assert connection.updates == {middle: 1, older: 2, newer: 3}
        assert [job["created_at"] for job in connection.jobs] == [
            datetime(2026, 10, 7, tzinfo=UTC) + timedelta(seconds=1),
            datetime(2026, 10, 7, tzinfo=UTC) + timedelta(seconds=2),
            datetime(2026, 10, 7, tzinfo=UTC) + timedelta(seconds=3),
        ]

    asyncio.run(exercise())


def test_cancel_reports_cleanup_pending_then_retries_settlement_without_duplicate_event() -> None:
    async def exercise() -> None:
        repository = _CancelRepository()
        queue = JobQueue(repository=repository, sources=_SourceRegistry())
        job_id = str(uuid4())

        first = await queue.cancel_queued(job_id, operator_id="operator")
        second = await queue.cancel_queued(job_id, operator_id="operator")

        assert first["job"]["state"] == "cancelled"
        assert first["reservation_cleanup"] == "pending"
        assert first["settlement_error_type"] == "OSError"
        assert "private endpoint detail" not in str(first)
        assert second["job"]["state"] == "cancelled"
        assert second["reservation_cleanup"] == "settled"
        assert repository.cancel_event_count == 1
        assert repository.settlement_attempts == 2

    asyncio.run(exercise())


def test_repository_cancel_records_one_event_and_rejects_running_or_owned_jobs() -> None:
    async def exercise() -> None:
        job_id = str(uuid4())
        connection = _CancelTransactionConnection(_cancel_job(job_id=job_id))
        repository = PostgresJobQueueRepository(database_url="postgresql://unused")
        repository._schema_ready = True
        repository._pool = _Pool(connection)

        first = await repository.cancel_queued_job(job_id, actor_sha256="e" * 64)
        repeated = await repository.cancel_queued_job(job_id, actor_sha256="e" * 64)

        assert first["state"] == repeated["state"] == "cancelled"
        assert first["reason_code"] == "operator_cancelled"
        assert connection.cancel_events == 1

        running_connection = _CancelTransactionConnection(_cancel_job(job_id=str(uuid4()), state="running"))
        running_repository = PostgresJobQueueRepository(database_url="postgresql://unused")
        running_repository._schema_ready = True
        running_repository._pool = _Pool(running_connection)
        with pytest.raises(ValueError, match="only queued or waiting jobs can be cancelled"):
            await running_repository.cancel_queued_job(
                str(running_connection.job["job_id"]),
                actor_sha256="e" * 64,
            )
        assert running_connection.cancel_events == 0

    asyncio.run(exercise())


class _BlockedSourceRegistry:
    async def get_training_source(self, dataset_version):
        return _TrainingSource(("source_sample_explicitly_invalidated",))
