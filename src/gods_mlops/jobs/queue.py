"""Durable queue submissions and their immutable PostgreSQL identities."""

from __future__ import annotations

import asyncio
import json
import posixpath
import re
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from .models import (
    AnnotationPreparationBatch,
    DatasetTrainingSource,
    ExecutionProfile,
    ProbeInput,
    ProcessIdentity,
    ResourceObservation,
)
from .sources import DatasetSourceRegistry, DatasetSourceUnavailableError

COMMUNICATION_RETRY_DELAYS_SECONDS = (10, 30, 90)
_PROBE_RUNTIME_EVIDENCE_EVENT = "probe_runtime_evidence_committed"
_PROBE_RUNTIME_EVIDENCE_SCHEMA = "gods-mlops-probe-runtime-evidence-v1"
_PROBE_RUNTIME_EVIDENCE_FIELDS = (
    "event",
    "job_id",
    "model_kind",
    "target_phase",
    "config_version",
    "input_sha256",
    "docker_image_id",
    "image_source_commit",
    "source_commit",
)
_OPERATOR_QUEUE_STATES = (
    "queued",
    "waiting_profile",
    "waiting_gpu",
    "waiting_storage",
    "waiting_capacity",
)
_MAX_OPERATOR_RETRY_INTENT_GENERATIONS = 4096
_ACTIVE_ARTIFACT_WRITER_WATCHERS: set[asyncio.Task] = set()
_TRANSIENT_RESERVATION_ERRORS = (
    asyncpg.PostgresConnectionError,
    ConnectionError,
    OSError,
    TimeoutError,
)


class DatasetNotReadyForTrainingError(ValueError):
    """A queue request does not reference a current, training-ready target."""

    def __init__(self, reasons: list[str] | tuple[str, ...] | str) -> None:
        self.reasons = (reasons,) if isinstance(reasons, str) else tuple(sorted(set(reasons)))
        super().__init__(", ".join(self.reasons))


class DatasetNotReadyForEvaluationError(ValueError):
    """A queue request does not reference a current, evaluation-eligible target."""

    def __init__(self, reasons: list[str] | tuple[str, ...] | str) -> None:
        self.reasons = (reasons,) if isinstance(reasons, str) else tuple(sorted(set(reasons)))
        super().__init__(", ".join(self.reasons))


class ResourceProfileNotFoundError(ValueError):
    """The requested immutable model/config profile was never registered."""


class ResourceProfileConflictError(ValueError):
    """A profile name was reused for different configuration bytes."""


class ResultArtifactConflictError(ValueError):
    """A job attempted to publish different bytes for the same result artifact kind."""


class OperatorRetryIntentConflictError(ValueError):
    """An operator reused one retry intent for a different immutable request."""


class OperatorRetryIntentCapacityError(ValueError):
    """The bounded active server-side retry-intent generation table is full."""


class OperatorRetrySessionExpiredError(ValueError):
    """The signed operator session expired before its retry intent was read."""


class ObservationReplayError(ValueError):
    """An observation ID or timestamp was repeated and cannot extend the idle window."""

    def __init__(self, message: str, *, idle_window_reset: bool = False) -> None:
        super().__init__(message)
        self.idle_window_reset = idle_window_reset


class ResourceObservationRejectedError(ValueError):
    """A producer sample failed its persisted resource trust or completeness checks."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class PostgresJobQueueRepository:
    """Persist queue identities, profiles, and the shared lease state in PostgreSQL."""

    def __init__(
        self,
        *,
        database_url: str,
        expected_node_id: str | None = "ubuntu",
        expected_host_identity: str | None = None,
        expected_gpu_uuid: str | None = None,
        expected_filesystem_identity: str | None = None,
        expected_storage_path: str | None = None,
        observation_max_age_seconds: int = 10,
    ) -> None:
        self._database_url = database_url
        self._pool: asyncpg.Pool | None = None
        self._schema_ready = False
        self._expected_node_id = expected_node_id
        self._expected_observation_identity: tuple[str, ...] | None = None
        self._observation_max_age_seconds = observation_max_age_seconds
        supplied = (
            expected_host_identity,
            expected_gpu_uuid,
            expected_filesystem_identity,
            expected_storage_path,
        )
        if any(value is not None for value in supplied):
            if not expected_node_id or not all(value for value in supplied):
                raise ValueError("trusted Ubuntu observation identity must be configured completely")
            self.configure_observation_identity(
                expected_node_id=expected_node_id,
                expected_host_identity=expected_host_identity,
                expected_gpu_uuid=expected_gpu_uuid,
                expected_filesystem_identity=expected_filesystem_identity,
                expected_storage_path=expected_storage_path,
            )
        if observation_max_age_seconds <= 0:
            raise ValueError("maximum observation age must be positive")

    def configure_observation_identity(
        self,
        *,
        expected_node_id: str | None = "ubuntu",
        expected_host_identity: str | None,
        expected_gpu_uuid: str | None,
        expected_filesystem_identity: str | None,
        expected_storage_path: str | None,
    ) -> None:
        """Pin the trusted observer identity before either admission or producer writes."""
        values = (
            expected_node_id,
            expected_host_identity,
            expected_gpu_uuid,
            expected_filesystem_identity,
            expected_storage_path,
        )
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError("trusted Ubuntu observation identity must be configured completely")
        identity = (
            expected_node_id,
            "ubuntu",
            "NVIDIA RTX A6000",
            expected_host_identity,
            expected_gpu_uuid,
            expected_filesystem_identity,
            posixpath.normpath(expected_storage_path),
        )
        if self._expected_observation_identity is not None and self._expected_observation_identity != identity:
            raise ValueError("repository is already pinned to another Ubuntu observation identity")
        if self._expected_node_id is not None and self._expected_node_id != expected_node_id:
            raise ValueError("repository is already pinned to another Ubuntu observation node")
        self._expected_node_id = expected_node_id
        self._expected_observation_identity = identity

    @property
    def expected_node_id(self) -> str:
        return self._expected_node_id or "ubuntu"

    @staticmethod
    async def _lock_job_then_lease(
        connection: asyncpg.Connection,
        *,
        job_id: str,
        lease_token: str,
    ) -> tuple[asyncpg.Record | None, asyncpg.Record | None]:
        """Lock the job row before its lease row in every fenced lease transaction."""
        job = await connection.fetchrow(
            "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid FOR UPDATE",
            job_id,
        )
        if job is None:
            return None, None
        lease = await connection.fetchrow(
            """
            SELECT * FROM gods_mlops_gpu_leases
            WHERE job_id = $1::uuid AND lease_token = $2::uuid FOR UPDATE
            """,
            job_id,
            lease_token,
        )
        return job, lease

    async def artifact_database_clock(self) -> datetime:
        """Read the authority database clock used to anchor controller deadlines."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            return await connection.fetchval("SELECT clock_timestamp()")

    async def bind_worker_artifact_deadline(
        self,
        *,
        job_id: str,
        lease_token: str,
        fencing_token: int,
        controller_invocation_id: str,
        candidate_deadline_at: datetime,
    ) -> dict[str, Any]:
        """Bind one exact UTC deadline to the currently admitted worker fence."""
        from gods_mlops.training.artifact_deadlines import artifact_deadline_for_attempt

        try:
            invocation_id = str(UUID(controller_invocation_id))
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("controller invocation ID is invalid") from error
        if invocation_id != controller_invocation_id:
            raise ValueError("controller invocation ID is not canonical")
        if candidate_deadline_at.tzinfo is None or candidate_deadline_at.utcoffset() != UTC.utcoffset(None):
            raise ValueError("artifact deadline candidate must be a timezone-aware UTC timestamp")
        if fencing_token <= 0:
            raise ValueError("worker fence must be positive")

        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                database_now = await connection.fetchval("SELECT clock_timestamp()")
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease["lease_token"])
                    or int(job["lease_generation"]) != fencing_token
                    or int(lease["fencing_token"]) != fencing_token
                    or job["state"] != "running"
                    or lease["expires_at"] <= database_now
                ):
                    raise RuntimeError("worker artifact deadline requires the current unexpired running fence")

                same_fence = await connection.fetchrow(
                    """SELECT * FROM gods_mlops_worker_artifact_deadlines
                       WHERE job_id=$1::uuid AND fencing_token=$2 FOR UPDATE""",
                    job_id,
                    fencing_token,
                )
                same_invocation = None
                if same_fence is None:
                    same_invocation = await connection.fetchrow(
                        """SELECT * FROM gods_mlops_worker_artifact_deadlines
                           WHERE job_id=$1::uuid AND controller_invocation_id=$2::uuid
                           ORDER BY created_at, fencing_token LIMIT 1""",
                        job_id,
                        invocation_id,
                    )
                selected = artifact_deadline_for_attempt(
                    existing_fence=dict(same_fence) if same_fence is not None else None,
                    existing_invocation=dict(same_invocation) if same_invocation is not None else None,
                    controller_invocation_id=invocation_id,
                    lease_token=str(lease_token),
                    candidate_deadline_at=candidate_deadline_at,
                )
                deadline_at = selected["artifact_deadline_at"].astimezone(UTC)
                if deadline_at <= database_now:
                    raise TimeoutError("worker artifact deadline has expired")
                if same_fence is None:
                    await connection.execute(
                        """INSERT INTO gods_mlops_worker_artifact_deadlines(
                               job_id, fencing_token, lease_token, controller_invocation_id,
                               artifact_deadline_at
                           ) VALUES($1::uuid,$2,$3::uuid,$4::uuid,$5)""",
                        job_id,
                        fencing_token,
                        lease_token,
                        selected["controller_invocation_id"],
                        deadline_at,
                    )
                return {
                    "job_id": str(job_id),
                    "fencing_token": fencing_token,
                    "lease_token": str(lease_token),
                    "controller_invocation_id": selected["controller_invocation_id"],
                    "artifact_deadline_at": deadline_at,
                    "database_now": database_now,
                }

    async def read_worker_artifact_deadline(
        self,
        *,
        job_id: str,
        lease_token: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        """Read the exact current fence deadline and DB clock for worker startup."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                database_now = await connection.fetchval("SELECT clock_timestamp()")
                if (
                    job is None
                    or lease is None
                    or job["state"] != "running"
                    or str(job["lease_token"]) != str(lease["lease_token"])
                    or int(job["lease_generation"]) != fencing_token
                    or int(lease["fencing_token"]) != fencing_token
                    or lease["expires_at"] <= database_now
                ):
                    raise RuntimeError("worker artifact deadline has no current unexpired fence")
                row = await connection.fetchrow(
                    """SELECT * FROM gods_mlops_worker_artifact_deadlines
                       WHERE job_id=$1::uuid AND fencing_token=$2""",
                    job_id,
                    fencing_token,
                )
                if row is None or str(row["lease_token"]) != str(lease_token):
                    raise RuntimeError("worker artifact deadline authority record is missing or mismatched")
                if row["artifact_deadline_at"] <= database_now:
                    raise TimeoutError("worker artifact deadline has expired")
                return {
                    "job_id": str(job_id),
                    "fencing_token": fencing_token,
                    "lease_token": str(lease_token),
                    "controller_invocation_id": str(row["controller_invocation_id"]),
                    "artifact_deadline_at": row["artifact_deadline_at"],
                    "database_now": database_now,
                }

    async def artifact_deadline_for_fence(
        self, *, job_id: str, lease_token: str, fencing_token: int
    ) -> dict[str, Any] | None:
        """Read an existing deadline for immutable retained-Job comparison."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT * FROM gods_mlops_worker_artifact_deadlines
                   WHERE job_id=$1::uuid AND fencing_token=$2""",
                job_id,
                fencing_token,
            )
        if row is None:
            return None
        if str(row["lease_token"]) != str(lease_token):
            raise RuntimeError("retained worker deadline is bound to a different lease token")
        return {
            "job_id": str(job_id),
            "fencing_token": fencing_token,
            "lease_token": str(lease_token),
            "controller_invocation_id": str(row["controller_invocation_id"]),
            "artifact_deadline_at": row["artifact_deadline_at"],
        }

    async def _assert_artifact_write_deadline(
        self,
        connection: asyncpg.Connection,
        *,
        job: asyncpg.Record,
        lease: asyncpg.Record,
        database_now: datetime,
        expected_invocation_id: str | None = None,
        expected_deadline_at: datetime | None = None,
    ) -> None:
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError, StaleCheckpointOwnerError

        if (expected_invocation_id is None) != (expected_deadline_at is None):
            raise CheckpointIdentityError("artifact deadline authority fields are incomplete")
        if (
            str(job["lease_token"]) != str(lease["lease_token"])
            or int(job["lease_generation"]) != int(lease["fencing_token"])
            or lease["expires_at"] <= database_now
        ):
            raise StaleCheckpointOwnerError("artifact writer no longer owns the current unexpired fence")
        row = await connection.fetchrow(
            """SELECT * FROM gods_mlops_worker_artifact_deadlines
               WHERE job_id=$1::uuid AND fencing_token=$2 FOR SHARE""",
            job["job_id"],
            lease["fencing_token"],
        )
        if row is None:
            if expected_invocation_id is not None:
                raise CheckpointIdentityError("worker artifact deadline authority record is missing")
            return
        if str(row["lease_token"]) != str(lease["lease_token"]):
            raise StaleCheckpointOwnerError("artifact deadline belongs to a replaced lease token")
        if row["artifact_deadline_at"] <= database_now:
            raise TimeoutError("worker artifact deadline has expired")
        if expected_invocation_id is not None and (
            str(row["controller_invocation_id"]) != str(expected_invocation_id)
            or row["artifact_deadline_at"] != expected_deadline_at
        ):
            raise CheckpointIdentityError("worker artifact deadline differs from its durable authority record")

    async def _require_artifact_writers_quiescent(
        self,
        connection: asyncpg.Connection,
        *,
        job_id: str,
        operation_id: str,
        required: bool,
    ) -> None:
        from gods_mlops.jobs.checkpoints import StaleCheckpointOwnerError

        rows = await connection.fetch(
            """SELECT event_type,details FROM gods_mlops_job_events
               WHERE job_id=$1::uuid AND event_type IN
                 ('artifact_write_started','artifact_write_quiescent')
               ORDER BY event_id""",
            job_id,
        )
        attempts = [
            (row["event_type"], _json_value(row["details"]))
            for row in rows
            if str(_json_value(row["details"]).get("operation_id", "")) == operation_id
        ]
        starts = [event for event in attempts if event[0] == "artifact_write_started"]
        if (required and not starts) or _unmatched_artifact_writers(attempts):
            raise StaleCheckpointOwnerError("artifact publication is blocked until every writer is quiescent")

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        # The ingestion repository is the schema authority for the shared 1 TiB
        # ledger and applies the ordered annotation, dataset, and queue migrations.
        from gods_mlops.ingestion.storage import PostgresIngestionRepository

        ingestion = PostgresIngestionRepository(database_url=self._database_url)
        try:
            await ingestion.ensure_schema()
        finally:
            await ingestion.close()
        self._schema_ready = True

    async def register_profile(self, profile: ExecutionProfile) -> None:
        if not profile.candidate:
            raise ValueError("measured profiles can only be created from a successful probe")
        await self.ensure_schema()
        pool = await self._get_pool()
        payload = _canonical_json(profile.config)
        async with pool.acquire() as connection:
            async with connection.transaction():
                inserted = await connection.fetchval(
                    """
                    INSERT INTO gods_mlops_resource_profiles (
                        phase, target_phase, model_kind, config_version, config_sha256,
                        memory_requirement_mib, artifact_reservation_bytes,
                        checkpoint_reservation_bytes, result_reservation_bytes, config_json,
                        profile_state, measurement_id, oom_alternatives
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, 'candidate', NULL, $11::jsonb)
                    ON CONFLICT (phase, model_kind, config_version) DO NOTHING
                    RETURNING TRUE
                    """,
                    profile.phase,
                    profile.target_phase,
                    profile.model_kind,
                    profile.config_version,
                    profile.config_sha256,
                    profile.memory_requirement_mib,
                    profile.artifact_reservation_bytes,
                    profile.checkpoint_reservation_bytes,
                    profile.result_reservation_bytes,
                    payload,
                    json.dumps(list(profile.oom_alternatives)),
                )
                if inserted:
                    return
                current = await connection.fetchrow(
                    """
                    SELECT config_sha256, memory_requirement_mib, artifact_reservation_bytes,
                           checkpoint_reservation_bytes, result_reservation_bytes,
                           config_json, profile_state, oom_alternatives, target_phase
                    FROM gods_mlops_resource_profiles
                    WHERE phase = $1 AND model_kind = $2 AND config_version = $3
                    FOR UPDATE
                    """,
                    profile.phase,
                    profile.model_kind,
                    profile.config_version,
                )
                if (
                    current is None
                    or current["target_phase"] != profile.target_phase
                    or current["config_sha256"].strip() != profile.config_sha256
                    or current["memory_requirement_mib"] != profile.memory_requirement_mib
                    or current["artifact_reservation_bytes"] != profile.artifact_reservation_bytes
                    or current["checkpoint_reservation_bytes"] != profile.checkpoint_reservation_bytes
                    or current["result_reservation_bytes"] != profile.result_reservation_bytes
                    or _json_value(current["config_json"]) != profile.config
                    or _json_value(current["oom_alternatives"]) != list(profile.oom_alternatives)
                ):
                    raise ResourceProfileConflictError(
                        "a profile name is already bound to different immutable configuration"
                    )

    async def get_profile(self, *, phase: str, model_kind: str, config_version: str) -> dict[str, Any] | None:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT phase, target_phase, model_kind, config_version, config_sha256,
                       memory_requirement_mib, artifact_reservation_bytes,
                       checkpoint_reservation_bytes, result_reservation_bytes, config_json,
                       profile_state, measurement_id, oom_alternatives
                FROM gods_mlops_resource_profiles
                WHERE phase = $1 AND model_kind = $2 AND config_version = $3
                """,
                phase,
                model_kind,
                config_version,
            )
        return _profile_dict(row) if row is not None else None

    async def enqueue(
        self,
        *,
        phase: str,
        input_kind: str,
        input_id: str,
        input_sha256: str,
        dataset_version: str | None,
        source_refs: dict[str, Any],
        model_kind: str,
        profile: dict[str, Any],
        rerun: bool,
        parent_job_id: str | None = None,
        retry_root_id: str | None = None,
        oom_retries: int = 0,
        stable_source_identity: dict[str, Any] | None = None,
        operator_dedupe_key: str | None = None,
    ) -> str:
        if operator_dedupe_key is not None:
            _validate_digest(operator_dedupe_key, "operator_dedupe_key")
            if not rerun or parent_job_id is None:
                raise ValueError("operator retry deduplication requires an explicit parent rerun")
        await self.ensure_schema()
        stable_identity = {
            "phase": phase,
            "target_phase": profile["target_phase"] or phase,
            "input_kind": input_kind,
            "input_id": input_id,
            "input_sha256": input_sha256,
            "dataset_version": dataset_version,
            "model_kind": model_kind,
            "config_version": profile["config_version"],
            "config_sha256": profile["config_sha256"],
        }
        dedupe_key = _enqueue_dedupe_key(stable_identity, stable_source_identity)
        if rerun:
            # Preserve the historical un-deduplicated rerun path unless an
            # operator supplies a scoped idempotency intent.
            dedupe_key = operator_dedupe_key
        job_id = uuid4()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                inserted = await connection.fetchval(
                    """
                    INSERT INTO gods_mlops_jobs (
                        job_id, phase, input_kind, input_id, input_sha256, dataset_version,
                        source_refs, model_kind, target_phase, config_version, config_sha256,
                        profile_state_snapshot, state, rerun, dedupe_key,
                        parent_job_id, retry_root_id, oom_retries, artifact_reservation_bytes
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11,
                        $12, 'queued', $13, $14, $15::uuid, $16::uuid, $17, $18
                    )
                    ON CONFLICT (dedupe_key) WHERE dedupe_key IS NOT NULL DO NOTHING
                    RETURNING job_id
                    """,
                    job_id,
                    phase,
                    input_kind,
                    input_id,
                    input_sha256,
                    dataset_version,
                    _canonical_json(source_refs),
                    model_kind,
                    profile["target_phase"] or phase,
                    profile["config_version"],
                    profile["config_sha256"],
                    profile["profile_state"],
                    rerun,
                    dedupe_key,
                    parent_job_id,
                    retry_root_id,
                    oom_retries,
                    profile["artifact_reservation_bytes"],
                )
                if inserted is None:
                    if operator_dedupe_key is not None:
                        existing = await connection.fetchrow(
                            """
                            SELECT job_id, phase, input_kind, input_id, input_sha256, dataset_version,
                                   source_refs, model_kind, target_phase, config_version, config_sha256,
                                   profile_state_snapshot, artifact_reservation_bytes, parent_job_id,
                                   retry_root_id, oom_retries, rerun
                            FROM gods_mlops_jobs WHERE dedupe_key = $1
                            """,
                            dedupe_key,
                        )
                        if existing is None:
                            raise RuntimeError("operator retry intent dedupe record disappeared")
                        if not _operator_retry_request_matches(
                            existing,
                            phase=phase,
                            input_kind=input_kind,
                            input_id=input_id,
                            input_sha256=input_sha256,
                            dataset_version=dataset_version,
                            source_refs=source_refs,
                            model_kind=model_kind,
                            target_phase=profile["target_phase"] or phase,
                            config_version=profile["config_version"],
                            config_sha256=profile["config_sha256"],
                            profile_state=profile["profile_state"],
                            reservation_bytes=profile["artifact_reservation_bytes"],
                            parent_job_id=parent_job_id,
                            retry_root_id=retry_root_id,
                            oom_retries=oom_retries,
                        ):
                            raise OperatorRetryIntentConflictError(
                                "operator retry intent was reused for a different immutable request"
                            )
                        return str(existing["job_id"])
                    existing = await connection.fetchval(
                        "SELECT job_id FROM gods_mlops_jobs WHERE dedupe_key = $1",
                        dedupe_key,
                    )
                    if existing is None:
                        raise RuntimeError("automatic job submission dedupe record disappeared")
                    return str(existing)
                root_id = retry_root_id or str(inserted)
                await connection.execute(
                    "UPDATE gods_mlops_jobs SET retry_root_id = $2::uuid WHERE job_id = $1::uuid",
                    inserted,
                    root_id,
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (job_id, event_type, state, details)
                    VALUES ($1, 'submitted', 'queued', $2::jsonb)
                    """,
                    inserted,
                    _canonical_json(stable_identity),
                )
                return str(inserted)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid",
                job_id,
            )
        if row is None:
            raise KeyError(f"GPU job {job_id} does not exist")
        return _job_dict(row)

    async def get_operator_retry_intent_generation(
        self,
        scope_sha256: str,
        *,
        session_expires_at: datetime,
        now: datetime | None = None,
    ) -> int:
        """Read the current session-scoped retry generation without creating state."""
        _validate_digest(scope_sha256, "operator retry intent scope SHA-256")
        current = now or datetime.now(UTC)
        if current.tzinfo is None or session_expires_at.tzinfo is None:
            raise ValueError("retry intent read time and session expiry must be timezone-aware")
        if session_expires_at <= current:
            raise OperatorRetrySessionExpiredError("operator session expired before retry generation lookup")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT generation,expires_at FROM gods_mlops_operator_retry_intent_generations
                   WHERE scope_sha256=$1""",
                scope_sha256,
            )
        if row is None:
            return 0
        if row["expires_at"] <= current or row["expires_at"] != session_expires_at:
            raise OperatorRetrySessionExpiredError("operator retry generation has expired or changed session")
        return int(row["generation"])

    async def advance_operator_retry_intent_generation(
        self,
        scope_sha256: str,
        *,
        expires_at: datetime,
        now: datetime | None = None,
    ) -> int:
        """Atomically advance an explicit rerun generation, bounded by live sessions."""
        _validate_digest(scope_sha256, "operator retry intent scope SHA-256")
        current = now or datetime.now(UTC)
        if current.tzinfo is None or expires_at.tzinfo is None:
            raise ValueError("retry intent session expiry must be future and timezone-aware")
        if expires_at <= current:
            raise OperatorRetrySessionExpiredError("operator session expired before retry generation advance")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                # Serialize first-insert capacity checks as well as increments across workers.
                await connection.execute(
                    "LOCK TABLE gods_mlops_operator_retry_intent_generations IN SHARE ROW EXCLUSIVE MODE"
                )
                await connection.execute(
                    "DELETE FROM gods_mlops_operator_retry_intent_generations WHERE expires_at <= $1",
                    current,
                )
                row = await connection.fetchrow(
                    """SELECT generation FROM gods_mlops_operator_retry_intent_generations
                       WHERE scope_sha256=$1 FOR UPDATE""",
                    scope_sha256,
                )
                if row is not None:
                    generation = int(row["generation"]) + 1
                    await connection.execute(
                        """UPDATE gods_mlops_operator_retry_intent_generations
                           SET generation=$2,expires_at=$3,updated_at=$4 WHERE scope_sha256=$1""",
                        scope_sha256,
                        generation,
                        expires_at,
                        current,
                    )
                    return generation
                count = await connection.fetchval(
                    "SELECT COUNT(*) FROM gods_mlops_operator_retry_intent_generations"
                )
                if int(count) >= _MAX_OPERATOR_RETRY_INTENT_GENERATIONS:
                    raise OperatorRetryIntentCapacityError("operator retry intent capacity has been reached")
                await connection.execute(
                    """INSERT INTO gods_mlops_operator_retry_intent_generations
                       (scope_sha256,generation,expires_at,updated_at) VALUES ($1,1,$2,$3)""",
                    scope_sha256,
                    expires_at,
                    current,
                )
                return 1

    async def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """List recent job state with active queue positions from durable ordering."""
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("job list limit must be between 1 and 500")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                WITH queue_positions AS (
                    SELECT job_id,
                           row_number() OVER (
                               ORDER BY queue_order ASC NULLS LAST, created_at ASC, job_id ASC
                           ) AS queue_position
                    FROM gods_mlops_jobs
                    WHERE state = ANY($2::text[])
                )
                SELECT job.*, queue_positions.queue_position,
                       CASE
                           WHEN reservation.job_id IS NULL THEN 'not_reserved'
                           WHEN reservation.state IN ('settled', 'released') THEN reservation.state
                           WHEN reservation.state = 'reserved' THEN 'pending'
                           ELSE 'unknown'
                       END AS reservation_cleanup
                FROM gods_mlops_jobs AS job
                LEFT JOIN queue_positions USING (job_id)
                LEFT JOIN gods_mlops_artifact_reservations AS reservation USING (job_id)
                ORDER BY
                    CASE WHEN queue_positions.queue_position IS NULL THEN 1 ELSE 0 END,
                    queue_positions.queue_position ASC NULLS LAST,
                    job.created_at DESC,
                    job.job_id ASC
                LIMIT $1
                """,
                limit,
                list(_OPERATOR_QUEUE_STATES),
            )
        return [_job_dict(row) for row in rows]

    async def cancel_queued_job(self, job_id: str, *, actor_sha256: str) -> dict[str, Any]:
        """Terminalize only an unowned waiting job; settlement happens after this commit."""
        _validate_digest(actor_sha256, "operator actor SHA-256")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid FOR UPDATE",
                    job_id,
                )
                if row is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if row["state"] == "cancelled":
                    return _job_dict(row)
                if row["state"] not in _OPERATOR_QUEUE_STATES:
                    raise ValueError("only queued or waiting jobs can be cancelled")
                if row["lease_token"] is not None or row["owner_pid"] is not None or row["owner_start_ticks"] is not None:
                    raise ValueError("jobs with an active lease or process owner cannot be cancelled")
                cancelled = await connection.fetchrow(
                    """
                    UPDATE gods_mlops_jobs SET state = 'cancelled',
                        reason_code = 'operator_cancelled', reason_detail = $2::jsonb,
                        retryable = FALSE, completed_at = now(), updated_at = now()
                    WHERE job_id = $1::uuid AND state = ANY($3::text[])
                      AND lease_token IS NULL AND owner_pid IS NULL AND owner_start_ticks IS NULL
                    RETURNING *
                    """,
                    job_id,
                    _canonical_json({"actor_sha256": actor_sha256}),
                    list(_OPERATOR_QUEUE_STATES),
                )
                if cancelled is None:
                    raise ValueError("job changed before operator cancellation")
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events(job_id, event_type, state, reason_code, details)
                    VALUES($1::uuid, 'operator_cancelled', 'cancelled', 'operator_cancelled', $2::jsonb)
                    """,
                    job_id,
                    _canonical_json({"actor_sha256": actor_sha256}),
                )
                return _job_dict(cancelled)

    async def reorder_queued_jobs(
        self,
        job_id: str,
        *,
        before_job_id: str | None,
        actor_sha256: str,
    ) -> list[str]:
        """Persist a manual order without changing any creation or checkpoint history."""
        _validate_digest(actor_sha256, "operator actor SHA-256")
        try:
            target_id = str(UUID(job_id))
            before_id = str(UUID(before_job_id)) if before_job_id is not None else None
        except (TypeError, ValueError) as error:
            raise ValueError("queue reorder job IDs must be UUIDs") from error
        if target_id != job_id.lower() or (before_job_id is not None and before_id != before_job_id.lower()):
            raise ValueError("queue reorder job IDs must be canonical lowercase UUID text")
        if before_id == target_id:
            raise ValueError("a job cannot be moved before itself")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                rows = await connection.fetch(
                    """
                    SELECT job_id, state, queue_order, created_at,
                           lease_token, owner_pid, owner_start_ticks
                    FROM gods_mlops_jobs WHERE state = ANY($1::text[])
                    ORDER BY job_id FOR UPDATE
                    """,
                    list(_OPERATOR_QUEUE_STATES),
                )
                jobs = [_job_dict(row) for row in rows]
                if any(
                    job.get("lease_token") is not None
                    or job.get("owner_pid") is not None
                    or job.get("owner_start_ticks") is not None
                    for job in jobs
                ):
                    raise ValueError("queued jobs with an active lease or process owner cannot be reordered")
                ordered = sorted(jobs, key=_queue_order_sort_key)
                ordered_ids = [job["job_id"] for job in ordered]
                if target_id not in ordered_ids:
                    raise ValueError("only queued or waiting jobs can be reordered")
                ordered_ids.remove(target_id)
                if before_id is None:
                    ordered_ids.append(target_id)
                else:
                    if before_id not in ordered_ids:
                        raise ValueError("the destination job is not queued or waiting")
                    ordered_ids.insert(ordered_ids.index(before_id), target_id)
                for position, queued_id in enumerate(ordered_ids, start=1):
                    await connection.execute(
                        "UPDATE gods_mlops_jobs SET queue_order = $1 WHERE job_id = $2::uuid",
                        position,
                        queued_id,
                    )
                target_position = ordered_ids.index(target_id) + 1
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events(job_id, event_type, state, details)
                    VALUES($1::uuid, 'operator_queue_reordered',
                           (SELECT state FROM gods_mlops_jobs WHERE job_id = $1::uuid), $2::jsonb)
                    """,
                    target_id,
                    _canonical_json(
                        {
                            "actor_sha256": actor_sha256,
                            "before_job_id": before_id,
                            "position": target_position,
                        }
                    ),
                )
                return ordered_ids

    async def fail_job_for_source(self, job_id: str, reason_code: str, details: dict[str, Any]) -> dict[str, Any]:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await connection.fetchrow(
                    "SELECT state, lease_token FROM gods_mlops_jobs WHERE job_id = $1::uuid FOR UPDATE",
                    job_id,
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if job["lease_token"] is not None:
                    raise ValueError("cannot terminally block a job while its GPU lease is still active")
                if job["state"] not in {
                    "queued", "waiting_profile", "waiting_gpu", "waiting_storage", "waiting_capacity"
                }:
                    return _job_dict(
                        await connection.fetchrow(
                            "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid", job_id
                        )
                    )
                await connection.execute(
                    """
                    UPDATE gods_mlops_jobs SET state = 'failed', reason_code = $2,
                        reason_detail = $3::jsonb, retryable = FALSE,
                        completed_at = now(), updated_at = now()
                    WHERE job_id = $1::uuid
                    """,
                    job_id,
                    reason_code,
                    _canonical_json(details),
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, reason_code, details
                    ) VALUES ($1::uuid, 'source_readiness_failed', 'failed', $2, $3::jsonb)
                    """,
                    job_id,
                    reason_code,
                    _canonical_json(details),
                )
        return await self.get_job(job_id)

    async def is_next_eligible_job(self, job_id: str) -> bool:
        """Keep eligible work in durable FIFO order across every GPU phase."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                WITH eligible AS (
                    SELECT job_id FROM gods_mlops_jobs
                    WHERE state IN ('queued', 'waiting_gpu', 'waiting_storage', 'waiting_capacity')
                      AND (profile_state_snapshot = 'measured' OR phase = 'probe')
                    ORDER BY queue_order ASC NULLS LAST, created_at, job_id
                    LIMIT 1
                )
                SELECT EXISTS (SELECT 1 FROM eligible WHERE job_id = $1::uuid)
                """,
                job_id,
            )
        return bool(row[0])

    async def record_observation(
        self,
        observation: ResourceObservation,
        *,
        max_gap_seconds: int = 10,
        received_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Persist only complete, trusted producer samples and advance a contiguous idle window."""
        await self.ensure_schema()
        received_at = received_at or datetime.now(UTC)
        reason = self._observation_rejection_reason(observation, received_at)
        if reason is not None:
            await self.record_observation_failure(node_id=self.expected_node_id, failure_code=reason)
            raise ResourceObservationRejectedError(reason)
        pool = await self._get_pool()
        payload = _canonical_json(observation.to_dict())
        rejected: ObservationReplayError | ResourceObservationRejectedError | None = None
        state: dict[str, Any] | None = None
        async with pool.acquire() as connection:
            async with connection.transaction():
                current = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_gpu_observation_current WHERE node_id = $1 FOR UPDATE",
                    self.expected_node_id,
                )
                if current is not None:
                    if str(current["observation_id"]) == observation.observation_id:
                        rejected = ObservationReplayError(
                            "observation ID was already recorded", idle_window_reset=True
                        )
                    elif observation.observed_at <= current["observed_at"]:
                        rejected = ObservationReplayError(
                            "observation timestamp did not advance", idle_window_reset=True
                        )
                    elif current["failure_code"] is None and current["observation"] is not None:
                        previous = ResourceObservation.from_dict(_json_value(current["observation"]))
                        if self._observation_identity(previous) != self._expected_observation_identity:
                            rejected = ResourceObservationRejectedError(
                                "ubuntu_observation_history_identity_mismatch"
                            )
                if rejected is not None:
                    await self._record_observation_failure_in_transaction(
                        connection,
                        node_id=self.expected_node_id,
                        failure_code=str(rejected),
                    )
                else:
                    has_gpu_processes = bool(observation.gpu_processes)
                    if has_gpu_processes:
                        idle_since = None
                        idle_count = 0
                    elif current is not None and current["failure_code"] is None:
                        gap = (observation.observed_at - current["last_observed_at"]).total_seconds()
                        if current["idle_since"] is not None and 0 < gap <= max_gap_seconds:
                            idle_since = current["idle_since"]
                            idle_count = current["idle_observation_count"] + 1
                        else:
                            idle_since = observation.observed_at
                            idle_count = 1
                    else:
                        idle_since = observation.observed_at
                        idle_count = 1
                    row = await connection.fetchrow(
                        """
                        INSERT INTO gods_mlops_gpu_observation_current (
                            node_id, observation_id, observed_at, received_at, observation,
                            failure_code, failure_count, idle_since, idle_observation_count,
                            last_observed_at
                        ) VALUES ($1, $2::uuid, $3, now(), $4::jsonb, NULL, 0, $5, $6, $3)
                        ON CONFLICT (node_id) DO UPDATE SET
                            observation_id = EXCLUDED.observation_id,
                            observed_at = EXCLUDED.observed_at,
                            received_at = now(),
                            observation = EXCLUDED.observation,
                            failure_code = NULL,
                            failure_count = 0,
                            idle_since = EXCLUDED.idle_since,
                            idle_observation_count = EXCLUDED.idle_observation_count,
                            last_observed_at = EXCLUDED.last_observed_at
                        RETURNING *
                        """,
                        self.expected_node_id,
                        observation.observation_id,
                        observation.observed_at,
                        payload,
                        idle_since,
                        idle_count,
                    )
                    state = _observation_state_dict(row)
        if rejected is not None:
            raise rejected
        assert state is not None
        return state

    def _observation_rejection_reason(
        self,
        observation: ResourceObservation,
        received_at: datetime,
    ) -> str | None:
        if self._expected_observation_identity is None:
            return "ubuntu_observer_identity_not_configured"
        if received_at.tzinfo is None or observation.observed_at.tzinfo is None:
            return "ubuntu_observation_incomplete"
        age = (received_at - observation.observed_at).total_seconds()
        if age < -2 or age > self._observation_max_age_seconds:
            return "ubuntu_observation_stale"
        if self._observation_identity(observation) != self._expected_observation_identity:
            return "ubuntu_observation_identity_mismatch"
        if (
            not observation.gpu_process_list_complete
            or not observation.process_table_complete
            or observation.total_mib <= 0
            or observation.free_mib < 0
            or observation.free_mib > observation.total_mib
            or observation.filesystem_available_bytes < 0
        ):
            return "ubuntu_observation_incomplete"
        try:
            UUID(observation.observation_id)
        except ValueError:
            return "ubuntu_observation_incomplete"
        return None

    @staticmethod
    def _observation_identity(observation: ResourceObservation) -> tuple[str, ...]:
        return (
            observation.node_id,
            observation.hostname.lower(),
            observation.gpu_name,
            observation.host_identity,
            observation.gpu_uuid,
            observation.filesystem_identity,
            posixpath.normpath(observation.storage_path),
        )

    async def _record_observation_failure_in_transaction(
        self,
        connection: asyncpg.Connection,
        *,
        node_id: str,
        failure_code: str,
    ) -> None:
        await connection.execute(
            """
            INSERT INTO gods_mlops_gpu_observation_current (
                node_id, observation_id, observed_at, received_at, observation,
                failure_code, failure_count, idle_since, idle_observation_count, last_observed_at
            ) VALUES ($1, $2::uuid, now(), now(), NULL, $3, 1, NULL, 0, NULL)
            ON CONFLICT (node_id) DO UPDATE SET
                observation_id=EXCLUDED.observation_id, observed_at=EXCLUDED.observed_at,
                received_at=now(), observation=NULL, failure_code=EXCLUDED.failure_code,
                failure_count=gods_mlops_gpu_observation_current.failure_count+1,
                idle_since=NULL, idle_observation_count=0, last_observed_at=NULL
            """,
            node_id,
            uuid4(),
            failure_code[:128],
        )

    async def record_observation_failure(self, *, node_id: str, failure_code: str) -> None:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO gods_mlops_gpu_observation_current (
                    node_id, observation_id, observed_at, received_at, observation,
                    failure_code, failure_count, idle_since, idle_observation_count,
                    last_observed_at
                ) VALUES ($1, $2, now(), now(), NULL, $3, 1, NULL, 0, NULL)
                ON CONFLICT (node_id) DO UPDATE SET
                    observation_id = EXCLUDED.observation_id,
                    observed_at = EXCLUDED.observed_at,
                    received_at = now(),
                    observation = NULL,
                    failure_code = EXCLUDED.failure_code,
                    failure_count = gods_mlops_gpu_observation_current.failure_count + 1,
                    idle_since = NULL,
                    idle_observation_count = 0,
                    last_observed_at = NULL
                """,
                node_id,
                uuid4(),
                failure_code[:128],
            )

    async def get_observation_state(self, node_id: str) -> dict[str, Any] | None:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM gods_mlops_gpu_observation_current WHERE node_id = $1",
                node_id,
            )
        return _observation_state_dict(row) if row is not None else None

    async def latest_observation(self, node_id: str | None = None) -> ResourceObservation | None:
        node_id = node_id or self.expected_node_id
        state = await self.get_observation_state(node_id)
        if state is None or state["failure_code"] is not None or state["observation"] is None:
            return None
        return ResourceObservation.from_dict(state["observation"])

    async def update_wait_state(
        self,
        job_id: str,
        *,
        state: str,
        reason_code: str,
        details: dict[str, Any] | None = None,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        if state not in {"waiting_profile", "waiting_gpu", "waiting_storage", "waiting_capacity"}:
            raise ValueError("invalid retryable GPU job wait state")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                updated = await connection.fetchrow(
                    """
                    UPDATE gods_mlops_jobs SET state = $2, reason_code = $3,
                        reason_detail = $4::jsonb, retryable = TRUE, updated_at = now()
                    WHERE job_id = $1::uuid AND state IN (
                        'queued', 'waiting_profile', 'waiting_gpu', 'waiting_storage',
                        'waiting_capacity', 'yield_requested'
                    )
                      AND lease_token IS NULL
                    RETURNING job_id
                    """,
                    job_id,
                    state,
                    reason_code,
                    _canonical_json(details or {}),
                )
                if updated is None:
                    row = await connection.fetchrow("SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid", job_id)
                    if row is None:
                        raise KeyError(f"GPU job {job_id} does not exist")
                    return _job_dict(row)
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, reason_code, observation_id, details
                    ) VALUES ($1::uuid, 'admission_wait', $2, $3, $4::uuid, $5::jsonb)
                    """,
                    job_id,
                    state,
                    reason_code,
                    observation_id,
                    _canonical_json(details or {}),
                )
        return await self.get_job(job_id)

    async def get_active_lease(self, gpu_uuid: str) -> dict[str, Any] | None:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM gods_mlops_gpu_leases WHERE gpu_uuid = $1",
                gpu_uuid,
            )
        if row is None:
            return None
        result = dict(row)
        for key in ("job_id", "lease_token", "granted_observation_id"):
            result[key] = str(result[key])
        for key in ("expires_at", "granted_at"):
            if result[key] is not None:
                result[key] = result[key].isoformat()
        return result

    async def artifact_reservation_for(self, job_id: str) -> dict[str, Any] | None:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT * FROM gods_mlops_artifact_reservations WHERE job_id = $1::uuid",
                job_id,
            )
        if row is None:
            return None
        result = dict(row)
        result["job_id"] = str(result["job_id"])
        return result

    async def evaluation_probe_source_block_reasons(self, job_id: str) -> tuple[str, ...]:
        """Revalidate the immutable training-probe checkpoint bound to an eval probe."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction(readonly=True):
                job = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid",
                    job_id,
                )
                if job is None:
                    return ("evaluation_probe_checkpoint_source_invalid",)
                if job["phase"] != "probe" or job["target_phase"] != "evaluation":
                    return ()
                reason = await _evaluation_probe_checkpoint_source_error(connection, job)
        return (reason,) if reason else ()

    async def bind_lease_process(
        self,
        job_id: str,
        lease_token: str,
        owner: ProcessIdentity,
    ) -> bool:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease_token)
                    or job["state"] not in {"running", "yield_requested"}
                ):
                    return False
                if lease["owner_pid"] is not None and (
                    lease["owner_pid"] != owner.pid
                    or lease["owner_start_ticks"] != owner.start_ticks
                    or lease["owner_uid"] != owner.uid
                ):
                    return False
                await connection.execute(
                    """
                    UPDATE gods_mlops_gpu_leases
                    SET owner_pid = $3, owner_start_ticks = $4, owner_uid = $5
                    WHERE job_id = $1::uuid AND lease_token = $2::uuid
                    """,
                    job_id,
                    lease_token,
                    owner.pid,
                    owner.start_ticks,
                    owner.uid,
                )
                await connection.execute(
                    """
                    UPDATE gods_mlops_jobs SET owner_pid = $3, owner_start_ticks = $4,
                        owner_uid = $5, updated_at = now()
                    WHERE job_id = $1::uuid AND lease_token = $2::uuid
                    """,
                    job_id,
                    lease_token,
                    owner.pid,
                    owner.start_ticks,
                    owner.uid,
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, fencing_token, details
                    ) VALUES ($1::uuid, 'lease_owner_bound', 'running', $2, $3::jsonb)
                    """,
                    job_id,
                    lease["fencing_token"],
                    _canonical_json(owner.as_dict()),
                )
                return True

    async def request_yield(
        self,
        job_id: str,
        reason: str,
        *,
        after_progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not reason.strip() or len(reason) > 128:
            raise ValueError("yield reason must contain 1 to 128 characters")
        if after_progress is not None:
            return await self._arm_evaluation_progress_yield(
                job_id,
                reason,
                after_progress=after_progress,
            )
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await connection.fetchrow(
                    "SELECT state, lease_token, lease_generation FROM gods_mlops_jobs WHERE job_id = $1::uuid FOR UPDATE",
                    job_id,
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if job["state"] not in {"running", "yield_requested"} or job["lease_token"] is None:
                    return _job_dict(await connection.fetchrow(
                        "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid", job_id
                    ))
                await connection.execute(
                    """
                    UPDATE gods_mlops_jobs SET state = 'yield_requested', reason_code = $2,
                        reason_detail = $3::jsonb, retryable = TRUE, updated_at = now()
                    WHERE job_id = $1::uuid AND lease_token = $4
                    """,
                    job_id,
                    reason,
                    _canonical_json({"reason": reason}),
                    job["lease_token"],
                )
                await connection.execute(
                    """
                    UPDATE gods_mlops_gpu_leases SET yield_reason = $2
                    WHERE job_id = $1::uuid AND lease_token = $3
                    """,
                    job_id,
                    reason,
                    job["lease_token"],
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, reason_code, fencing_token, details
                    ) VALUES ($1::uuid, 'yield_requested', 'yield_requested', $2, $3, $4::jsonb)
                    """,
                    job_id,
                    reason,
                    job["lease_generation"],
                    _canonical_json({"action": "checkpoint_and_exit_own_process"}),
                )
        return await self.get_job(job_id)

    async def _arm_evaluation_progress_yield(
        self,
        job_id: str,
        reason: str,
        *,
        after_progress: dict[str, Any],
    ) -> dict[str, Any]:
        target = _normalize_evaluation_progress_target(after_progress)
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            job = await connection.fetchrow(
                "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid FOR UPDATE",
                job_id,
            )
            if job is None:
                raise KeyError(f"GPU job {job_id} does not exist")
            # Preserve the queue-wide job-before-lease order. For a queued
            # first attempt this proves that no lease was already created.
            lease = await connection.fetchrow(
                "SELECT * FROM gods_mlops_gpu_leases WHERE job_id=$1::uuid FOR UPDATE",
                job_id,
            )
            database_now = await connection.fetchval("SELECT clock_timestamp()")
            profile = await self._evaluation_progress_profile(connection, job)
            details = await _evaluation_progress_arm_details(
                connection,
                job,
                profile,
                reason=reason,
                target=target,
            )
            events = await connection.fetch(
                """SELECT event_id,event_type,state,fencing_token,details
                   FROM gods_mlops_job_events WHERE job_id=$1::uuid ORDER BY event_id""",
                job_id,
            )
            arms = [row for row in events if row["event_type"] == "evaluation_yield_armed"]
            request_id = details["request_id"]
            matching_arm = next(
                (
                    row
                    for row in arms
                    if _json_value(row["details"]).get("request_id") == request_id
                ),
                None,
            )
            if matching_arm is not None:
                if _canonical_json(_json_value(matching_arm["details"])) != _canonical_json(details):
                    raise RuntimeError("evaluation yield request identity conflicts with its retained arm")
                arm_event_id = int(matching_arm["event_id"])
                consumed = any(
                    row["event_type"] == "yield_requested"
                    and _json_value(row["details"]).get("arm_event_id") == arm_event_id
                    and int(row["fencing_token"] or 0) == int(details["expected_lease_generation"])
                    for row in events
                )
                return _evaluation_progress_arm_result(
                    arm_event_id=arm_event_id,
                    details=details,
                    job=job,
                    lease=lease,
                    consumed=consumed,
                )
            if arms:
                raise RuntimeError("evaluation progress yield request conflicts with a retained arm")
            submitted_after = datetime.fromisoformat(target["submitted_after"])
            if (
                job["state"] != "queued"
                or int(job["lease_generation"]) != 0
                or job["lease_token"] is not None
                or job["lease_expires_at"] is not None
                or lease is not None
                or job["created_at"] <= submitted_after
                or submitted_after >= database_now
            ):
                raise RuntimeError("evaluation yield arm requires this new unleased generation-zero job")
            arm_event_id = await connection.fetchval(
                """INSERT INTO gods_mlops_job_events (
                       job_id,event_type,state,reason_code,fencing_token,details
                   ) VALUES ($1::uuid,'evaluation_yield_armed','queued',$2,NULL,$3::jsonb)
                   RETURNING event_id""",
                job_id,
                reason,
                _canonical_json(details),
            )
            return _evaluation_progress_arm_result(
                arm_event_id=int(arm_event_id),
                details=details,
                job=job,
                lease=None,
                consumed=False,
            )

    async def pending_evaluation_progress_yield(
        self,
        *,
        job_id: str,
        lease_token: str,
        fencing_token: int,
        input_sha256: str,
        config_sha256: str,
        training_source_checkpoint_sha256: str,
    ) -> dict[str, Any] | None:
        """Read a matching first-generation arm without reading checkpoint payloads."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction(readonly=True):
            job = await connection.fetchrow(
                "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid",
                job_id,
            )
            if (
                job is None
                or job["state"] != "running"
                or str(job["lease_token"]) != str(lease_token)
                or int(job["lease_generation"]) != fencing_token
                or int(fencing_token) != 1
                or str(job["input_sha256"]).strip() != input_sha256
                or str(job["config_sha256"]).strip() != config_sha256
            ):
                return None
            lease = await connection.fetchrow(
                """SELECT * FROM gods_mlops_gpu_leases
                   WHERE job_id=$1::uuid AND lease_token=$2::uuid""",
                job_id,
                lease_token,
            )
            database_now = await connection.fetchval("SELECT clock_timestamp()")
            if (
                lease is None
                or int(lease["fencing_token"]) != fencing_token
                or lease["expires_at"] <= database_now
                or job["lease_expires_at"] != lease["expires_at"]
            ):
                return None
            profile = await self._evaluation_progress_profile(connection, job)
            source = _evaluation_progress_probe_input(job, profile)
            if source.evaluation_checkpoint_source is None or (
                source.evaluation_checkpoint_source.checkpoint_sha256
                != training_source_checkpoint_sha256
            ):
                raise RuntimeError("evaluation progress arm training checkpoint source changed")
            events = await connection.fetch(
                """SELECT event_id,event_type,state,fencing_token,details
                   FROM gods_mlops_job_events WHERE job_id=$1::uuid ORDER BY event_id""",
                job_id,
            )
            arms = [row for row in events if row["event_type"] == "evaluation_yield_armed"]
            matching = []
            for arm in arms:
                details = _json_value(arm["details"])
                if (
                    int(details.get("expected_lease_generation", 0)) == fencing_token
                    and details.get("input_sha256") == input_sha256
                    and details.get("config_sha256") == config_sha256
                    and details.get("training_source_checkpoint_sha256")
                    == training_source_checkpoint_sha256
                ):
                    matching.append((arm, details))
            if len(matching) > 1:
                raise RuntimeError("multiple evaluation progress arms target the same first lease")
            if not matching:
                return None
            arm, details = matching[0]
            arm_event_id = int(arm["event_id"])
            if _evaluation_progress_arm_consumed(events, arm_event_id, fencing_token):
                return None
            identity = await _evaluation_progress_arm_details(
                connection,
                job,
                profile,
                reason=str(details.get("reason", "")),
                target=details.get("target", {}),
            )
            if _canonical_json(identity) != _canonical_json(details):
                raise RuntimeError("evaluation progress arm no longer matches immutable job identity")
            return {
                "arm_event_id": arm_event_id,
                "request_id": str(details["request_id"]),
                "expected_lease_generation": fencing_token,
                "target": deepcopy(details["target"]),
                "training_probe_job_id": source.evaluation_checkpoint_source.training_probe_job_id,
                "training_source_checkpoint_sha256": source.evaluation_checkpoint_source.checkpoint_sha256,
            }

    async def record_evaluation_progress(
        self,
        *,
        job_id: str,
        lease_token: str,
        fencing_token: int,
        arm_event_id: int,
        progress: dict[str, Any],
        owner: ProcessIdentity,
        artifact_deadline: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Atomically publish a fenced batch boundary and consume its exact arm."""
        if not isinstance(owner, ProcessIdentity):
            raise TypeError("evaluation progress requires the bound worker process identity")
        if artifact_deadline is None:
            raise ValueError("evaluation progress requires the existing worker artifact deadline authority")
        progress_value = _normalize_evaluation_progress(progress)
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection, connection.transaction():
            job, lease = await self._lock_job_then_lease(
                connection,
                job_id=job_id,
                lease_token=lease_token,
            )
            database_now = await connection.fetchval("SELECT clock_timestamp()")
            if (
                job is None
                or lease is None
                or str(job["lease_token"]) != str(lease_token)
                or str(lease["lease_token"]) != str(lease_token)
                or int(job["lease_generation"]) != fencing_token
                or int(lease["fencing_token"]) != fencing_token
                or fencing_token != 1
                or job["state"] not in {"running", "yield_requested"}
                or lease["expires_at"] <= database_now
                or job["lease_expires_at"] != lease["expires_at"]
                or job["lease_expires_at"] <= database_now
            ):
                raise RuntimeError("evaluation progress does not own the current unexpired job-then-lease fence")
            bound_owner = (
                owner.pid,
                owner.start_ticks,
                owner.uid,
            )
            lease_owner = (
                lease["owner_pid"],
                lease["owner_start_ticks"],
                lease["owner_uid"],
            )
            job_owner = (
                job["owner_pid"],
                job["owner_start_ticks"],
                job["owner_uid"],
            )
            if (
                owner.pid <= 0
                or owner.start_ticks <= 0
                or owner.uid != 10001
                or lease_owner != bound_owner
                or job_owner != bound_owner
            ):
                raise RuntimeError("evaluation progress worker does not match the bound host owner")
            invocation_id = artifact_deadline.get("controller_invocation_id")
            deadline_at = artifact_deadline.get("artifact_deadline_at")
            if (
                str(artifact_deadline.get("job_id")) != job_id
                or str(artifact_deadline.get("lease_token")) != lease_token
                or int(artifact_deadline.get("fencing_token", -1)) != fencing_token
                or not isinstance(invocation_id, str)
                or not isinstance(deadline_at, datetime)
            ):
                raise ValueError("evaluation progress artifact deadline identity is incomplete")
            await self._assert_artifact_write_deadline(
                connection,
                job=job,
                lease=lease,
                database_now=database_now,
                expected_invocation_id=invocation_id,
                expected_deadline_at=deadline_at,
            )
            profile = await self._evaluation_progress_profile(connection, job)
            probe_input = _evaluation_progress_probe_input(job, profile)
            source = probe_input.evaluation_checkpoint_source
            if source is None:
                raise RuntimeError("evaluation progress has no typed training checkpoint source")
            events = await connection.fetch(
                """SELECT event_id,event_type,state,fencing_token,details
                   FROM gods_mlops_job_events WHERE job_id=$1::uuid ORDER BY event_id""",
                job_id,
            )
            arm = next(
                (
                    row
                    for row in events
                    if row["event_type"] == "evaluation_yield_armed"
                    and int(row["event_id"]) == arm_event_id
                ),
                None,
            )
            if arm is None:
                raise RuntimeError("evaluation progress arm event is missing")
            arm_details = _json_value(arm["details"])
            expected_arm_details = await _evaluation_progress_arm_details(
                connection,
                job,
                profile,
                reason=str(arm_details.get("reason", "")),
                target=arm_details.get("target", {}),
            )
            if (
                _canonical_json(expected_arm_details) != _canonical_json(arm_details)
                or int(arm_details.get("expected_lease_generation", 0)) != fencing_token
                or _canonical_json(arm_details.get("source_identity"))
                != _canonical_json(probe_input.as_dict())
                or arm_details.get("input_sha256") != str(job["input_sha256"]).strip()
                or arm_details.get("config_sha256") != str(job["config_sha256"]).strip()
                or arm_details.get("training_source_checkpoint_sha256") != source.checkpoint_sha256
                or arm_details.get("training_probe_job_id") != source.training_probe_job_id
                or _canonical_json(progress_value)
                != _canonical_json(
                    {
                        key: value
                        for key, value in arm_details.get("target", {}).items()
                        if key != "submitted_after"
                    }
                )
            ):
                raise RuntimeError("evaluation progress does not match its immutable armed boundary")
            progress_identity = {
                "request_id": arm_details["request_id"],
                "arm_event_id": arm_event_id,
                "job_id": job_id,
                "expected_lease_generation": fencing_token,
                "progress": progress_value,
                "owner": owner.as_dict(),
                "artifact_deadline": {
                    "controller_invocation_id": invocation_id,
                    "artifact_deadline_at": deadline_at.isoformat(),
                },
            }
            progress_id = sha256(_canonical_json(progress_identity).encode("utf-8")).hexdigest()
            progress_events = [
                row
                for row in events
                if row["event_type"] == "evaluation_batch_completed"
                and _json_value(row["details"]).get("progress_id") == progress_id
            ]
            consumed_events = [
                row
                for row in events
                if row["event_type"] == "yield_requested"
                and _json_value(row["details"]).get("arm_event_id") == arm_event_id
            ]
            expected_progress_details = progress_identity | {
                "progress_id": progress_id,
                "phase": "probe",
                "target_phase": "evaluation",
                "model_kind": "clip",
                "training_probe_job_id": source.training_probe_job_id,
                "training_source_checkpoint_sha256": source.checkpoint_sha256,
            }
            publication_now = await connection.fetchval("SELECT clock_timestamp()")
            current_lease_owner = (
                lease["owner_pid"],
                lease["owner_start_ticks"],
                lease["owner_uid"],
            )
            current_job_owner = (
                job["owner_pid"],
                job["owner_start_ticks"],
                job["owner_uid"],
            )
            if (
                lease["expires_at"] <= publication_now
                or job["lease_expires_at"] != lease["expires_at"]
                or job["lease_expires_at"] <= publication_now
                or current_lease_owner != bound_owner
                or current_job_owner != bound_owner
            ):
                raise RuntimeError("evaluation progress lease or process owner expired before publication")
            await self._assert_artifact_write_deadline(
                connection,
                job=job,
                lease=lease,
                database_now=publication_now,
                expected_invocation_id=invocation_id,
                expected_deadline_at=deadline_at,
            )
            if progress_events or consumed_events:
                if (
                    len(progress_events) != 1
                    or len(consumed_events) != 1
                    or _canonical_json(_json_value(progress_events[0]["details"]))
                    != _canonical_json(expected_progress_details)
                    or int(progress_events[0]["fencing_token"] or 0) != fencing_token
                    or _json_value(consumed_events[0]["details"]).get("progress_event_id")
                    != int(progress_events[0]["event_id"])
                    or _json_value(consumed_events[0]["details"]).get("progress_id") != progress_id
                    or _json_value(consumed_events[0]["details"]).get("request_id")
                    != arm_details["request_id"]
                    or int(consumed_events[0]["fencing_token"] or 0) != fencing_token
                    or job["state"] != "yield_requested"
                ):
                    raise RuntimeError("evaluation progress replay conflicts with its consumed yield marker")
                return {
                    "status": "yield_requested",
                    "arm_event_id": arm_event_id,
                    "progress_event_id": int(progress_events[0]["event_id"]),
                    "yield_event_id": int(consumed_events[0]["event_id"]),
                    "progress_id": progress_id,
                    "idempotent_replay": True,
                }
            if job["state"] != "running":
                raise RuntimeError("evaluation progress may only consume an armed running job")
            progress_event_id = await connection.fetchval(
                """INSERT INTO gods_mlops_job_events (
                       job_id,event_type,state,reason_code,fencing_token,details
                   ) VALUES ($1::uuid,'evaluation_batch_completed','running',NULL,$2,$3::jsonb)
                   RETURNING event_id""",
                job_id,
                fencing_token,
                _canonical_json(expected_progress_details),
            )
            reason = str(arm_details["reason"])
            reason_detail = {
                "reason": reason,
                "action": "checkpoint_after_evaluation_batch",
                "arm_event_id": arm_event_id,
                "progress_event_id": int(progress_event_id),
                "request_id": arm_details["request_id"],
                "progress_id": progress_id,
                "expected_lease_generation": fencing_token,
                "progress": progress_value,
            }
            updated = await connection.execute(
                """UPDATE gods_mlops_jobs SET state='yield_requested',reason_code=$2,
                       reason_detail=$3::jsonb,retryable=TRUE,updated_at=now()
                   WHERE job_id=$1::uuid AND lease_token=$4::uuid
                     AND lease_generation=$5 AND state='running'
                     AND lease_expires_at > clock_timestamp()
                     AND $6::timestamptz > clock_timestamp()
                     AND owner_pid=$8 AND owner_start_ticks=$9 AND owner_uid=$10
                     AND EXISTS (
                       SELECT 1 FROM gods_mlops_gpu_leases AS l
                       WHERE l.job_id=$1::uuid AND l.lease_token=$4::uuid
                         AND l.fencing_token=$5 AND l.expires_at > clock_timestamp()
                         AND l.owner_pid=$8 AND l.owner_start_ticks=$9 AND l.owner_uid=$10
                     )
                     AND EXISTS (
                       SELECT 1 FROM gods_mlops_worker_artifact_deadlines AS d
                       WHERE d.job_id=$1::uuid AND d.fencing_token=$5
                         AND d.lease_token=$4::uuid AND d.controller_invocation_id=$7::uuid
                         AND d.artifact_deadline_at=$6::timestamptz
                         AND d.artifact_deadline_at > clock_timestamp()
                     )""",
                job_id,
                reason,
                _canonical_json(reason_detail),
                lease_token,
                fencing_token,
                deadline_at,
                invocation_id,
                owner.pid,
                owner.start_ticks,
                owner.uid,
            )
            if updated != "UPDATE 1":
                raise RuntimeError("evaluation progress lost its running job fence")
            updated_lease = await connection.execute(
                """UPDATE gods_mlops_gpu_leases SET yield_reason=$2
                   WHERE job_id=$1::uuid AND lease_token=$3::uuid AND fencing_token=$4
                     AND expires_at > clock_timestamp()
                     AND $5::timestamptz > clock_timestamp()
                     AND owner_pid=$7 AND owner_start_ticks=$8 AND owner_uid=$9
                     AND EXISTS (
                       SELECT 1 FROM gods_mlops_worker_artifact_deadlines AS d
                       WHERE d.job_id=$1::uuid AND d.fencing_token=$4
                         AND d.lease_token=$3::uuid AND d.controller_invocation_id=$6::uuid
                         AND d.artifact_deadline_at=$5::timestamptz
                         AND d.artifact_deadline_at > clock_timestamp()
                     )""",
                job_id,
                reason,
                lease_token,
                fencing_token,
                deadline_at,
                invocation_id,
                owner.pid,
                owner.start_ticks,
                owner.uid,
            )
            if updated_lease != "UPDATE 1":
                raise RuntimeError("evaluation progress lost its active lease fence")
            yield_event_details = reason_detail | {
                "action": "checkpoint_and_exit_own_process",
                "phase": "probe",
                "target_phase": "evaluation",
                "model_kind": "clip",
                "input_sha256": str(job["input_sha256"]).strip(),
                "config_sha256": str(job["config_sha256"]).strip(),
                "training_probe_job_id": source.training_probe_job_id,
                "training_source_checkpoint_sha256": source.checkpoint_sha256,
            }
            yield_event_id = await connection.fetchval(
                """INSERT INTO gods_mlops_job_events (
                       job_id,event_type,state,reason_code,fencing_token,details
                   ) SELECT $1::uuid,'yield_requested','yield_requested',$2,$3,$4::jsonb
                   WHERE EXISTS (
                       SELECT 1
                       FROM gods_mlops_jobs AS j
                       JOIN gods_mlops_gpu_leases AS l ON l.job_id=j.job_id
                       JOIN gods_mlops_worker_artifact_deadlines AS d
                         ON d.job_id=j.job_id AND d.fencing_token=$3
                       WHERE j.job_id=$1::uuid AND j.state='yield_requested'
                         AND j.lease_token=$5::uuid AND l.lease_token=$5::uuid
                         AND j.lease_generation=$3 AND l.fencing_token=$3
                         AND j.owner_pid=$8 AND j.owner_start_ticks=$9 AND j.owner_uid=$10
                         AND l.owner_pid=$8 AND l.owner_start_ticks=$9 AND l.owner_uid=$10
                         AND j.lease_expires_at > clock_timestamp()
                         AND l.expires_at > clock_timestamp()
                         AND d.lease_token=$5::uuid AND d.controller_invocation_id=$6::uuid
                         AND d.artifact_deadline_at=$7::timestamptz
                         AND d.artifact_deadline_at > clock_timestamp()
                   )
                   RETURNING event_id""",
                job_id,
                reason,
                fencing_token,
                _canonical_json(yield_event_details),
                lease_token,
                invocation_id,
                deadline_at,
                owner.pid,
                owner.start_ticks,
                owner.uid,
            )
            if yield_event_id is None:
                raise RuntimeError("evaluation progress authority expired before yield publication")
            return {
                "status": "yield_requested",
                "arm_event_id": arm_event_id,
                "progress_event_id": int(progress_event_id),
                "yield_event_id": int(yield_event_id),
                "progress_id": progress_id,
                "idempotent_replay": False,
            }

    async def _evaluation_progress_profile(self, connection, job):
        profile = await connection.fetchrow(
            """SELECT phase,target_phase,model_kind,config_version,config_sha256,
                      profile_state,config_json
               FROM gods_mlops_resource_profiles
               WHERE phase='probe' AND model_kind=$1 AND config_version=$2 FOR SHARE""",
            job["model_kind"],
            job["config_version"],
        )
        if profile is None:
            raise RuntimeError("evaluation progress profile is unavailable")
        config = _json_value(profile["config_json"])
        if (
            profile["phase"] != "probe"
            or profile["target_phase"] != "evaluation"
            or profile["model_kind"] != "clip"
            or profile["config_version"] != job["config_version"]
            or str(profile["config_sha256"]).strip() != str(job["config_sha256"]).strip()
            or profile["profile_state"] != "candidate"
            or not isinstance(config, dict)
            or config.get("micro_batch") != 2
            or config.get("evaluation_batch_size") != 2
            or config.get("resolution") != 224
        ):
            raise RuntimeError("evaluation progress profile differs from the frozen CLIP fixture batch")
        return {**dict(profile), "config_json": config}

    async def renew_lease(self, job_id: str, lease_token: str, *, now=None, lease_seconds: int = 15) -> bool:
        await self.ensure_schema()
        renew_at = now or __import__("datetime").datetime.now(__import__("datetime").UTC)
        expires_at = renew_at + __import__("datetime").timedelta(seconds=lease_seconds)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease_token)
                    or job["state"] != "running"
                ):
                    return False
                await connection.execute(
                    "UPDATE gods_mlops_gpu_leases SET expires_at = $3 WHERE job_id = $1::uuid AND lease_token = $2::uuid",
                    job_id,
                    lease_token,
                    expires_at,
                )
                await connection.execute(
                    "UPDATE gods_mlops_jobs SET lease_expires_at = $3, updated_at = now() WHERE job_id = $1::uuid AND lease_token = $2::uuid",
                    job_id,
                    lease_token,
                    expires_at,
                )
                return True

    async def schedule_communication_retry(
        self,
        *,
        job_id: str,
        error_code: str,
        now,
        lease_token: str | None,
    ) -> dict[str, Any]:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid FOR UPDATE",
                    job_id,
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if job["lease_token"] is not None and str(job["lease_token"]) != lease_token:
                    raise ValueError("communication retry caller does not own the current lease")
                retries = int(job["communication_retries"])
                if retries >= len(COMMUNICATION_RETRY_DELAYS_SECONDS):
                    await connection.execute(
                        """
                        UPDATE gods_mlops_jobs SET state = 'failed', reason_code = 'communication_retries_exhausted',
                            reason_detail = $2::jsonb, retryable = FALSE,
                            completed_at = now(), updated_at = now()
                        WHERE job_id = $1::uuid
                        """,
                        job_id,
                        _canonical_json({"error_code": error_code, "retry_count": retries}),
                    )
                    await connection.execute(
                        """
                        INSERT INTO gods_mlops_job_events (
                            job_id, event_type, state, reason_code, fencing_token, details
                        ) VALUES ($1::uuid, 'communication_retries_exhausted', 'failed',
                            'communication_retries_exhausted', $2, $3::jsonb)
                        """,
                        job_id,
                        job["lease_generation"],
                        _canonical_json({"error_code": error_code, "retry_count": retries}),
                    )
                    return _job_dict(
                        await connection.fetchrow(
                            "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid", job_id
                        )
                    )

                delay = COMMUNICATION_RETRY_DELAYS_SECONDS[retries]
                next_retry_at = now + __import__("datetime").timedelta(seconds=delay)
                target_state = (
                    "yield_requested"
                    if job["lease_token"] is not None and job["state"] in {"running", "yield_requested"}
                    else "waiting_gpu"
                )
                details = {"error_code": error_code, "retry": retries + 1, "delay_seconds": delay}
                await connection.execute(
                    """
                    UPDATE gods_mlops_jobs SET state = $2, reason_code = 'communication_retry_scheduled',
                        reason_detail = $3::jsonb, retryable = TRUE,
                        communication_retries = communication_retries + 1,
                        next_retry_at = $4, updated_at = now()
                    WHERE job_id = $1::uuid
                    """,
                    job_id,
                    target_state,
                    _canonical_json(details),
                    next_retry_at,
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, reason_code, fencing_token, details
                    ) VALUES ($1::uuid, 'communication_retry_scheduled', $2,
                        'communication_retry_scheduled', $3, $4::jsonb)
                    """,
                    job_id,
                    target_state,
                    job["lease_generation"],
                    _canonical_json(details | {"next_retry_at": next_retry_at.isoformat()}),
                )
                return _job_dict(
                    await connection.fetchrow(
                        "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid", job_id
                    )
                )

    async def record_oom(self, *, job_id: str, lease_token: str) -> dict[str, Any]:
        """Create a new fenced config attempt only from a measured smaller alternative."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                if (
                    job is None
                    or lease is None
                    or job["lease_token"] != lease["lease_token"]
                    or job["state"] != "running"
                ):
                    raise ValueError("OOM report does not own a running fenced GPU lease")
                if job["oom_retries"] >= 2:
                    return await self._fail_oom_in_transaction(
                        connection, job, "oom_alternatives_exhausted", lease["fencing_token"]
                    )

                root_id = job["retry_root_id"] or job["job_id"]
                root = await connection.fetchrow(
                    "SELECT phase, model_kind, config_version FROM gods_mlops_jobs WHERE job_id = $1 FOR SHARE",
                    root_id,
                )
                if root is None:
                    raise RuntimeError("OOM retry root record is missing")
                root_profile = await connection.fetchrow(
                    """
                    SELECT oom_alternatives FROM gods_mlops_resource_profiles
                    WHERE phase = $1 AND model_kind = $2 AND config_version = $3
                    """,
                    root["phase"],
                    root["model_kind"],
                    root["config_version"],
                )
                alternatives = _json_value(root_profile["oom_alternatives"]) if root_profile else []
                attempt = int(job["oom_retries"])
                if attempt >= len(alternatives):
                    return await self._fail_oom_in_transaction(
                        connection, job, "oom_no_validated_smaller_profile", lease["fencing_token"]
                    )
                alternative_version = alternatives[attempt]
                current_profile = await connection.fetchrow(
                    """
                    SELECT memory_requirement_mib, artifact_reservation_bytes
                    FROM gods_mlops_resource_profiles
                    WHERE phase = $1 AND model_kind = $2 AND config_version = $3
                    """,
                    job["phase"],
                    job["model_kind"],
                    job["config_version"],
                )
                alternative = await connection.fetchrow(
                    """
                    SELECT * FROM gods_mlops_resource_profiles
                    WHERE phase = $1 AND model_kind = $2 AND config_version = $3
                    """,
                    job["phase"],
                    job["model_kind"],
                    alternative_version,
                )
                measured = None
                if alternative is not None and alternative["measurement_id"] is not None:
                    measured = await connection.fetchrow(
                        """
                        SELECT result_state, model_kind, target_phase, config_sha256
                        FROM gods_mlops_profile_measurements WHERE measurement_id = $1
                        """,
                        alternative["measurement_id"],
                    )
                if (
                    current_profile is None
                    or alternative is None
                    or alternative["profile_state"] != "measured"
                    or measured is None
                    or measured["result_state"] != "succeeded"
                    or measured["model_kind"] != job["model_kind"]
                    or measured["config_sha256"].strip() != alternative["config_sha256"].strip()
                    or measured["target_phase"] != job["target_phase"]
                    or alternative["memory_requirement_mib"] >= current_profile["memory_requirement_mib"]
                    or alternative["artifact_reservation_bytes"] > current_profile["artifact_reservation_bytes"]
                ):
                    return await self._fail_oom_in_transaction(
                        connection,
                        job,
                        "oom_alternative_not_measured_or_smaller",
                        lease["fencing_token"],
                    )

                child_id = uuid4()
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_jobs (
                        job_id, phase, input_kind, input_id, input_sha256, dataset_version,
                        source_refs, model_kind, target_phase, config_version, config_sha256,
                        profile_state_snapshot, state, rerun, dedupe_key, parent_job_id,
                        retry_root_id, oom_retries, artifact_reservation_bytes
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11,
                        'measured', 'queued', TRUE, NULL, $12::uuid, $13::uuid, $14, $15
                    )
                    """,
                    child_id,
                    job["phase"],
                    job["input_kind"],
                    job["input_id"],
                    job["input_sha256"].strip(),
                    job["dataset_version"],
                    _canonical_json(_json_value(job["source_refs"])),
                    job["model_kind"],
                    job["target_phase"],
                    alternative_version,
                    alternative["config_sha256"].strip(),
                    job_id,
                    root_id,
                    attempt + 1,
                    alternative["artifact_reservation_bytes"],
                )
                await connection.execute(
                    """
                    UPDATE gods_mlops_jobs SET state = 'retrying', retry_job_id = $2::uuid,
                        oom_retries = $3, reason_code = 'oom_retry_child_submitted',
                        reason_detail = $4::jsonb, retryable = TRUE, updated_at = now()
                    WHERE job_id = $1::uuid AND lease_token = $5::uuid
                    """,
                    job_id,
                    child_id,
                    attempt + 1,
                    _canonical_json({
                        "retry_job_id": str(child_id),
                        "root_job_id": str(root_id),
                        "config_version": alternative_version,
                        "checkpoint_reused": False,
                        "retry_attempt": attempt + 1,
                    }),
                    lease_token,
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, reason_code, fencing_token, details
                    ) VALUES ($1::uuid, 'oom_retry_child_submitted', 'retrying',
                        'oom_retry_child_submitted', $2, $3::jsonb)
                    """,
                    job_id,
                    lease["fencing_token"],
                    _canonical_json({"child_job_id": str(child_id), "config_version": alternative_version}),
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (job_id, event_type, state, details)
                    VALUES ($1, 'oom_retry_child_queued', 'queued', $2::jsonb)
                    """,
                    child_id,
                    _canonical_json({"parent_job_id": str(job_id), "retry_root_id": str(root_id)}),
                )
                parent = _job_dict(
                    await connection.fetchrow(
                        "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid", job_id
                    )
                )
                return {"state": "retrying", "retry_job_id": str(child_id), "job": parent}

    async def _fail_oom_in_transaction(
        self,
        connection: asyncpg.Connection,
        job: asyncpg.Record,
        reason_code: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        await connection.execute(
            """
            UPDATE gods_mlops_jobs SET state = 'failed', reason_code = $2,
                reason_detail = $3::jsonb, retryable = FALSE,
                completed_at = now(), updated_at = now()
            WHERE job_id = $1
            """,
            job["job_id"],
            reason_code,
            _canonical_json({"oom_retries": job["oom_retries"]}),
        )
        await connection.execute(
            """
            INSERT INTO gods_mlops_job_events (
                job_id, event_type, state, reason_code, fencing_token, details
            ) VALUES ($1, 'oom_terminal_failure', 'failed', $2, $3, $4::jsonb)
            """,
            job["job_id"],
            reason_code,
            fencing_token,
            _canonical_json({"oom_retries": job["oom_retries"]}),
        )
        root_id = job["retry_root_id"] or job["job_id"]
        if root_id != job["job_id"]:
            await connection.execute(
                """
                UPDATE gods_mlops_jobs SET state = 'failed', reason_code = $2,
                    reason_detail = $3::jsonb, retryable = FALSE,
                    completed_at = now(), updated_at = now()
                WHERE job_id = $1 AND state = 'retrying'
                """,
                root_id,
                reason_code,
                _canonical_json({"failed_job_id": str(job["job_id"])}),
            )
        return {"state": "failed", "retry_job_id": None, "reason_code": reason_code}

    async def release_after_observed_exit(
        self,
        *,
        job_id: str,
        lease_token: str,
        observation: ResourceObservation,
        terminal_reason: str | None = None,
    ) -> bool:
        """Release only after complete process and CUDA listings prove this owner exited."""
        if terminal_reason is not None and not 1 <= len(terminal_reason) <= 128:
            raise ValueError("terminal release reason must contain 1 to 128 characters")
        if not observation.gpu_process_list_complete or not observation.process_table_complete:
            return False
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                if job is None or lease is None or str(job["lease_token"]) != str(lease_token):
                    return False
                if lease["owner_pid"] is not None:
                    owner_pid = lease["owner_pid"]
                    owner_start = lease["owner_start_ticks"]
                    owner_is_live = any(
                        item.pid == owner_pid and item.start_ticks == owner_start
                        for item in observation.process_table
                    )
                    pid_uses_gpu = any(item.pid == owner_pid for item in observation.gpu_processes)
                    if owner_is_live or pid_uses_gpu:
                        return False
                elif observation.gpu_processes:
                    return False
                await connection.execute(
                    "DELETE FROM gods_mlops_gpu_leases WHERE gpu_uuid = $1 AND lease_token = $2::uuid",
                    observation.gpu_uuid,
                    lease_token,
                )
                released_job = await connection.fetchrow(
                    """
                    UPDATE gods_mlops_jobs SET
                        state = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN state
                            WHEN $4::text IS NOT NULL THEN 'failed'
                            ELSE 'waiting_gpu'
                        END,
                        reason_code = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN reason_code
                            WHEN $4::text IS NOT NULL THEN $4
                            ELSE 'yielded_checkpoint'
                        END,
                        reason_detail = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN reason_detail
                            WHEN $4::text IS NOT NULL THEN $3::jsonb
                            ELSE $3::jsonb
                        END,
                        retryable = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN retryable
                            WHEN $4::text IS NOT NULL THEN FALSE
                            ELSE TRUE
                        END,
                        completed_at = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN completed_at
                            WHEN $4::text IS NOT NULL THEN now()
                            ELSE completed_at
                        END,
                        lease_token = NULL, lease_expires_at = NULL,
                        owner_pid = NULL, owner_start_ticks = NULL, owner_uid = NULL,
                        updated_at = now()
                    WHERE job_id = $1::uuid AND lease_token = $2::uuid
                    RETURNING state, reason_code
                    """,
                    job_id,
                    lease_token,
                    _canonical_json({
                        "owner_process_exited": True,
                        "gpu_release_observed": True,
                        "source_blocked": terminal_reason,
                    }),
                    terminal_reason,
                )
                if released_job is None:
                    return False
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, reason_code, observation_id,
                        fencing_token, details
                    ) VALUES ($1::uuid, 'lease_owner_released', $2, $3,
                        $4::uuid, $5, $6::jsonb)
                    """,
                    job_id,
                    released_job["state"],
                    released_job["reason_code"],
                    observation.observation_id,
                    lease["fencing_token"],
                    _canonical_json({"owner_pid": lease["owner_pid"], "owner_start_ticks": lease["owner_start_ticks"]}),
                )
                return True

    async def release_unbound_expired_lease(
        self,
        *,
        job_id: str,
        lease_token: str,
        observation: ResourceObservation,
        terminal_reason: str | None = None,
        min_idle_seconds: int = 30,
        min_idle_observations: int = 7,
    ) -> bool:
        """Release only an expired unbound lease after a persisted complete idle proof."""
        if terminal_reason is not None and not 1 <= len(terminal_reason) <= 128:
            raise ValueError("terminal release reason must contain 1 to 128 characters")
        if min_idle_seconds < 30 or min_idle_observations < 7:
            raise ValueError("unbound lease recovery requires the full 30-second idle proof")
        if (
            not observation.gpu_process_list_complete
            or not observation.process_table_complete
            or observation.gpu_processes
        ):
            return False
        from datetime import datetime

        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                current = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_gpu_observation_current WHERE node_id = $1 FOR UPDATE",
                    observation.node_id,
                )
                if (
                    current is None
                    or current["failure_code"] is not None
                    or current["observation"] is None
                    or str(current["observation_id"]) != observation.observation_id
                ):
                    return False
                persisted = ResourceObservation.from_dict(_json_value(current["observation"]))
                idle_since = current["idle_since"]
                if isinstance(idle_since, str):
                    idle_since = datetime.fromisoformat(idle_since)
                if (
                    persisted.gpu_processes
                    or persisted.gpu_uuid != observation.gpu_uuid
                    or idle_since is None
                    or current["idle_observation_count"] < min_idle_observations
                    or (observation.observed_at - idle_since).total_seconds() < min_idle_seconds
                ):
                    return False
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease_token)
                    or lease["gpu_uuid"] != observation.gpu_uuid
                    or lease["owner_pid"] is not None
                    or lease["owner_start_ticks"] is not None
                    or lease["owner_uid"] is not None
                    or lease["expires_at"] > observation.observed_at
                ):
                    return False
                await connection.execute(
                    "DELETE FROM gods_mlops_gpu_leases WHERE gpu_uuid=$1 AND lease_token=$2::uuid",
                    observation.gpu_uuid,
                    lease_token,
                )
                released_job = await connection.fetchrow(
                    """
                    UPDATE gods_mlops_jobs SET
                        state = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN state
                            WHEN $3::text IS NOT NULL THEN 'failed'
                            ELSE 'waiting_gpu'
                        END,
                        reason_code = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN reason_code
                            WHEN $3::text IS NOT NULL THEN $3
                            ELSE 'expired_unbound_lease_released'
                        END,
                        reason_detail = $4::jsonb,
                        retryable = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN retryable
                            WHEN $3::text IS NOT NULL THEN FALSE
                            ELSE TRUE
                        END,
                        completed_at = CASE
                            WHEN state IN ('completed', 'failed', 'cancelled', 'retrying') THEN completed_at
                            WHEN $3::text IS NOT NULL THEN now()
                            ELSE completed_at
                        END,
                        lease_token = NULL, lease_expires_at = NULL,
                        owner_pid = NULL, owner_start_ticks = NULL, owner_uid = NULL,
                        updated_at = now()
                    WHERE job_id = $1::uuid AND lease_token = $2::uuid
                    RETURNING state, reason_code
                    """,
                    job_id,
                    lease_token,
                    terminal_reason,
                    _canonical_json({"unbound_lease_expired": True, "complete_gpu_idle_observed": True,
                                     "source_blocked": terminal_reason}),
                )
                if released_job is None:
                    return False
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id,event_type,state,reason_code,observation_id,fencing_token,details
                    ) VALUES ($1::uuid,'lease_owner_released',$2,$3,$4::uuid,$5,$6::jsonb)
                    """,
                    job_id,
                    released_job["state"],
                    released_job["reason_code"],
                    observation.observation_id,
                    lease["fencing_token"],
                    _canonical_json({"owner_pid": None, "owner_start_ticks": None,
                                     "idle_since": idle_since, "idle_observation_count": current["idle_observation_count"]}),
                )
                return True

    async def lease_is_current(self, job_id: str, lease_token: str) -> bool:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            return bool(
                await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM gods_mlops_gpu_leases AS lease
                        JOIN gods_mlops_jobs AS job USING (job_id)
                        WHERE lease.job_id = $1::uuid AND lease.lease_token = $2::uuid
                          AND job.lease_token = lease.lease_token
                          AND job.state IN ('running', 'yield_requested')
                    )
                    """,
                    job_id,
                    lease_token,
                )
            )

    async def checkpoint_identity(self, job_id: str):
        from gods_mlops.jobs.checkpoints import CheckpointIdentity

        job = await self.get_job(job_id)
        return CheckpointIdentity(
            job_id=job["job_id"],
            input_kind=job["input_kind"],
            input_id=job["input_id"],
            input_sha256=job["input_sha256"],
            phase=job["phase"],
            model_kind=job["model_kind"],
            config_version=job["config_version"],
            config_sha256=job["config_sha256"],
            dataset_version=job["dataset_version"],
        )

    async def begin_artifact_write(
        self,
        *,
        job_id: str,
        lease_token: str,
        identity,
        prepared,
        store,
        operation: str,
        source_registry: DatasetSourceRegistry | None = None,
        runtime_measurements: dict[str, Any] | None = None,
        artifact_invocation_id: str | None = None,
        artifact_deadline_at: datetime | None = None,
    ) -> str | None:
        """Persist an exact S3 write intent and charge its reserved bytes before object I/O."""
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError, StaleCheckpointOwnerError

        if operation not in {"checkpoint", "result"}:
            raise ValueError("artifact write operation must be checkpoint or result")
        uri = _prepared_artifact_uri(prepared, store)
        object_key = getattr(prepared, "object_key", None)
        digest = str(prepared.sha256).strip()
        size_bytes = int(prepared.size_bytes)
        metadata_size = int(getattr(prepared, "metadata_size_bytes", 0)) if operation == "checkpoint" else 0
        charge_bytes = size_bytes + metadata_size
        kind = str(prepared.kind) if operation == "result" else "checkpoint"
        expected = identity
        if prepared.identity != expected:
            raise CheckpointIdentityError("artifact write input, phase, model, or config identity changed")
        if runtime_measurements is not None and not isinstance(runtime_measurements, dict):
            raise ValueError("result runtime measurements must be a JSON object")
        measured_details = json.loads(_canonical_json(runtime_measurements)) if runtime_measurements is not None else None

        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                if operation == "result" or identity.phase == "evaluation":
                    usage = await connection.fetchrow(
                        "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton=TRUE FOR UPDATE"
                    )
                    if usage is None:
                        raise RuntimeError("the shared object-storage ledger is not initialized")
                    if identity.phase in {"training", "evaluation"}:
                        if source_registry is None:
                            raise ValueError("Task 6 source registry is required before dataset artifact I/O")
                        reasons = await _phase_source_block_reasons_in_transaction(
                            source_registry,
                            connection,
                            phase=identity.phase,
                            dataset_version=identity.dataset_version,
                            model_kind=identity.model_kind,
                        )
                        if reasons:
                            if identity.phase == "evaluation":
                                raise DatasetNotReadyForEvaluationError(reasons)
                            raise DatasetNotReadyForTrainingError(reasons)
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                allowed_states = {"running", "yield_requested"} if operation == "checkpoint" else {"running"}
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease["lease_token"])
                    or job["state"] not in allowed_states
                ):
                    raise StaleCheckpointOwnerError("artifact writer no longer owns the active GPU lease")
                database_now = await connection.fetchval("SELECT clock_timestamp()")
                await self._assert_artifact_write_deadline(
                    connection,
                    job=job,
                    lease=lease,
                    database_now=database_now,
                    expected_invocation_id=artifact_invocation_id,
                    expected_deadline_at=artifact_deadline_at,
                )
                if identity != _checkpoint_identity_from_row(job) or str(job["job_id"]) != job_id:
                    raise CheckpointIdentityError("artifact write identity differs from the immutable job")
                if identity.phase == "evaluation":
                    checkpoint_error = await _evaluation_checkpoint_source_error(connection, job)
                    if checkpoint_error:
                        raise CheckpointIdentityError("evaluation checkpoint source changed before artifact write")
                elif identity.phase == "probe" and job.get("target_phase") == "evaluation":
                    checkpoint_error = await _evaluation_probe_checkpoint_source_error(connection, job)
                    if checkpoint_error:
                        raise CheckpointIdentityError(
                            "evaluation probe checkpoint source changed before artifact intent"
                        )
                if (
                    operation == "checkpoint"
                    and str(job["checkpoint_uri"] or "") == uri
                    and str(job["checkpoint_sha256"] or "").strip() == digest
                    and _json_value(job["checkpoint_identity"] or {}) == identity.as_dict()
                ):
                    committed = await connection.fetchrow(
                        """SELECT details FROM gods_mlops_job_events
                           WHERE job_id=$1::uuid AND event_type='checkpoint_committed'
                             AND details->>'checkpoint_uri'=$2 AND details->>'sha256'=$3
                           ORDER BY event_id DESC LIMIT 1""",
                        job_id,
                        uri,
                        digest,
                    )
                    existing_operation_id = (
                        _json_value(committed["details"]).get("operation_id") if committed else None
                    )
                    return str(existing_operation_id) if existing_operation_id else None
                profile = await connection.fetchrow(
                    """SELECT checkpoint_reservation_bytes,result_reservation_bytes
                       FROM gods_mlops_resource_profiles
                       WHERE phase=$1 AND model_kind=$2 AND config_version=$3""",
                    job["phase"],
                    job["model_kind"],
                    job["config_version"],
                )
                if profile is None:
                    raise CheckpointIdentityError("artifact resource profile is unavailable")
                limit = profile[
                    "checkpoint_reservation_bytes" if operation == "checkpoint" else "result_reservation_bytes"
                ]
                replacement_limit = 2 * limit if operation == "checkpoint" else limit
                if charge_bytes <= 0 or size_bytes > limit or charge_bytes > replacement_limit:
                    raise ValueError("artifact write exceeds its exact versioned reservation")

                events = await connection.fetch(
                    """SELECT event_type,details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type IN
                         ('artifact_write_pending','result_artifact_committed','checkpoint_committed',
                          'artifact_write_deleted','checkpoint_pruned','checkpoint_prune_pending')
                       ORDER BY event_id""",
                    job_id,
                )
                same_kind_commits = []
                active_pending = []
                deleted_intents: set[str] = set()
                pruned_checkpoint_lifetimes: set[str] = set()
                legacy_pruned_checkpoint_objects: set[tuple[str, str]] = set()
                pruning_checkpoint_objects: dict[str, tuple[str, str]] = {}
                for row in events:
                    details = _json_value(row["details"])
                    event_type = row["event_type"]
                    if event_type == "checkpoint_prune_pending":
                        pruning_checkpoint_objects[_checkpoint_lifetime_id(details)] = (
                                str(details.get("uri") or details.get("checkpoint_uri") or ""),
                                str(details.get("sha256", "")),
                            )
                        continue
                    if event_type == "artifact_write_deleted":
                        deleted_intents.add(str(details.get("operation_id", "")))
                        continue
                    if event_type == "checkpoint_pruned":
                        lifetime_id = details.get("write_lifetime_id") or details.get("operation_id")
                        if lifetime_id:
                            pruned_checkpoint_lifetimes.add(str(lifetime_id))
                        # Round-1 prune events did not carry a lifetime/operation ID. A
                        # later retry emits a synthetic legacy-* lifetime, which needs
                        # the same URI/SHA fallback when paired with a round-1 write.
                        if (
                            not details.get("operation_id")
                            and (
                                not details.get("write_lifetime_id")
                                or str(details["write_lifetime_id"]).startswith("legacy-")
                            )
                        ):
                            legacy_pruned_checkpoint_objects.add(
                                (
                                    str(details.get("uri") or details.get("checkpoint_uri") or ""),
                                    str(details.get("sha256") or ""),
                                )
                            )
                        continue
                    event_uri = details.get("uri") or details.get("checkpoint_uri")
                    if operation == "result" and event_type == "result_artifact_committed":
                        if details.get("kind") == kind:
                            same_kind_commits.append((event_uri, details))
                        continue
                    if operation == "checkpoint" and event_type == "checkpoint_committed":
                        continue
                    if event_type == "artifact_write_pending" and details.get("operation") == operation:
                        if operation == "result" and details.get("kind") == kind:
                            active_pending.append(details)
                        elif operation == "checkpoint" and event_uri == uri:
                            active_pending.append(details)

                active_pruning_objects = {
                    object_identity
                    for lifetime, object_identity in pruning_checkpoint_objects.items()
                    if lifetime not in pruned_checkpoint_lifetimes
                }
                if operation == "checkpoint" and (uri, digest) in active_pruning_objects:
                    raise StaleCheckpointOwnerError("checkpoint object lifetime is being pruned")

                if operation == "result":
                    for existing_uri, details in same_kind_commits:
                        if (
                            existing_uri == uri
                            and details.get("sha256") == digest
                            and details.get("size_bytes") == size_bytes
                            and details.get("identity") == identity.as_dict()
                            and details.get("runtime_measurements") == measured_details
                        ):
                            existing_operation_id = details.get("operation_id")
                            return str(existing_operation_id) if existing_operation_id else None
                        raise ResultArtifactConflictError(
                            "result artifact kind already has different immutable bytes"
                        )
                for details in active_pending:
                    operation_id = str(details.get("operation_id", ""))
                    lifetime_id = str(details.get("write_lifetime_id") or operation_id)
                    legacy_pruned_object = (
                        operation == "checkpoint"
                        and not details.get("write_lifetime_id")
                        and (
                            str(details.get("uri") or details.get("checkpoint_uri") or ""),
                            str(details.get("sha256") or ""),
                        )
                        in legacy_pruned_checkpoint_objects
                    )
                    if operation_id in deleted_intents or (
                        operation == "checkpoint" and lifetime_id in pruned_checkpoint_lifetimes
                    ) or legacy_pruned_object:
                        continue
                    if (
                        details.get("uri") == uri
                        and details.get("sha256") == digest
                        and details.get("size_bytes") == size_bytes
                        and details.get("identity") == identity.as_dict()
                        and details.get("kind") == kind
                        and details.get("runtime_measurements") == measured_details
                    ):
                        return operation_id
                    if operation == "result":
                        raise ResultArtifactConflictError(
                            "result artifact kind already has a different pending immutable write"
                        )

                reservation = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_artifact_reservations WHERE job_id=$1::uuid FOR UPDATE",
                    job_id,
                )
                if reservation is None or reservation["state"] != "reserved":
                    raise ValueError("artifact write requires an active shared storage reservation")
                if reservation["consumed_bytes"] + charge_bytes > reservation["reserved_bytes"]:
                    raise ValueError("artifact write exceeds its remaining shared storage reservation")
                operation_id = str(uuid4())
                details = {
                    "operation_id": operation_id,
                    "operation": operation,
                    "kind": kind,
                    "bucket": getattr(store, "_bucket", None),
                    "prefix": getattr(store, "_prefix", None),
                    "uri": uri,
                    "object_key": object_key,
                    "sha256": digest,
                    "size_bytes": size_bytes,
                    "metadata_size_bytes": metadata_size,
                    "charge_bytes": charge_bytes,
                    "identity": identity.as_dict(),
                }
                if operation == "result":
                    details["runtime_measurements"] = measured_details
                if operation == "checkpoint":
                    details.update(
                        {
                            "write_lifetime_id": operation_id,
                            "previous_uri": getattr(prepared, "previous_uri", None),
                            "previous_sha256": getattr(prepared, "previous_sha256", None),
                            "previous_size_bytes": getattr(prepared, "previous_size_bytes", None),
                        }
                    )
                details["writer_quiescence_required"] = True
                await connection.execute(
                    """UPDATE gods_mlops_artifact_reservations
                       SET consumed_bytes=consumed_bytes+$2 WHERE job_id=$1::uuid""",
                    job_id,
                    charge_bytes,
                )
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(
                           job_id,event_type,state,fencing_token,details
                       ) VALUES($1::uuid,'artifact_write_pending',$2,$3,$4::jsonb)""",
                    job_id,
                    job["state"],
                    lease["fencing_token"],
                    _canonical_json(details),
                )
                return operation_id

    async def record_artifact_writer_started(
        self,
        *,
        job_id: str,
        lease_token: str,
        operation_id: str,
        writer_attempt_id: str,
        artifact_invocation_id: str | None = None,
        artifact_deadline_at: datetime | None = None,
    ) -> None:
        """Persist the writer attempt before its thread can touch the object store."""
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError, StaleCheckpointOwnerError

        attempt_id = str(UUID(writer_attempt_id))
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                database_now = await connection.fetchval("SELECT clock_timestamp()")
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease["lease_token"])
                    or job["state"] not in {"running", "yield_requested"}
                ):
                    raise StaleCheckpointOwnerError("artifact writer start no longer has the current GPU lease")
                await self._assert_artifact_write_deadline(
                    connection,
                    job=job,
                    lease=lease,
                    database_now=database_now,
                    expected_invocation_id=artifact_invocation_id,
                    expected_deadline_at=artifact_deadline_at,
                )
                events = await connection.fetch(
                    """SELECT event_type, details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type IN
                          ('artifact_write_pending','artifact_write_deleted','checkpoint_committed',
                          'result_artifact_committed','artifact_write_started','artifact_write_quiescent',
                          'checkpoint_prune_pending','checkpoint_pruned')
                       ORDER BY event_id""",
                    job_id,
                )
                pending = None
                committed = False
                committed_details = None
                deleted = False
                already_started = False
                prune_pending_by_lifetime: dict[str, dict[str, Any]] = {}
                pruned_lifetimes: set[str] = set()
                for row in events:
                    details = _json_value(row["details"])
                    if row["event_type"] == "checkpoint_prune_pending":
                        prune_pending_by_lifetime[_checkpoint_lifetime_id(details)] = details
                        continue
                    if row["event_type"] == "checkpoint_pruned":
                        pruned_lifetimes.add(_checkpoint_lifetime_id(details))
                        continue
                    if str(details.get("operation_id", "")) != operation_id:
                        continue
                    if row["event_type"] == "artifact_write_pending":
                        pending = details
                    elif row["event_type"] in {"checkpoint_committed", "result_artifact_committed"}:
                        committed = True
                        committed_details = details
                    elif row["event_type"] == "artifact_write_deleted":
                        deleted = True
                    elif (
                        row["event_type"] == "artifact_write_started"
                        and str(details.get("writer_attempt_id")) == attempt_id
                    ):
                        already_started = True
                if pending is not None and pending.get("operation") == "checkpoint":
                    writer_identity = pending
                else:
                    writer_identity = committed_details
                prune_pending = [
                    details
                    for lifetime, details in prune_pending_by_lifetime.items()
                    if lifetime not in pruned_lifetimes
                ]
                if writer_identity is not None and writer_identity.get("operation", "checkpoint") == "checkpoint":
                    writer_uri = str(writer_identity.get("uri") or writer_identity.get("checkpoint_uri") or "")
                    writer_sha = str(writer_identity.get("sha256", ""))
                    if any(
                        str(prune.get("operation_id", "")) == operation_id
                        or str(prune.get("write_lifetime_id", "")) == operation_id
                        or (
                            str(prune.get("uri", "")) == writer_uri
                            and str(prune.get("sha256", "")) == writer_sha
                        )
                        for prune in prune_pending
                    ):
                        raise StaleCheckpointOwnerError("checkpoint writer is blocked by a pending prune")
                if already_started:
                    return
                if deleted or (pending is None and not committed):
                    raise CheckpointIdentityError("artifact writer start has no pending or committed exact operation")
                if pending is not None:
                    if pending.get("operation") not in {"checkpoint", "result"}:
                        raise CheckpointIdentityError("artifact writer start has an invalid pending operation")
                    allowed_states = {"running", "yield_requested"} if pending["operation"] == "checkpoint" else {"running"}
                    if job["state"] not in allowed_states:
                        raise StaleCheckpointOwnerError("artifact writer start is forbidden for this job state")
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(
                           job_id,event_type,state,fencing_token,details
                       ) VALUES($1::uuid,'artifact_write_started',$2,$3,$4::jsonb)""",
                    job_id,
                    job["state"],
                    lease["fencing_token"],
                    _canonical_json(
                        {
                            "operation_id": operation_id,
                            "writer_attempt_id": attempt_id,
                            "lease_token": str(lease_token),
                            "fencing_token": int(lease["fencing_token"]),
                        }
                    ),
                )

    async def record_artifact_writer_quiescent(
        self,
        *,
        job_id: str,
        operation_id: str,
        writer_attempt_id: str,
    ) -> None:
        """Record that a submitted writer thread has actually returned."""
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError

        attempt_id = str(UUID(writer_attempt_id))
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await connection.fetchrow(
                    "SELECT state FROM gods_mlops_jobs WHERE job_id=$1::uuid", job_id
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                events = await connection.fetch(
                    """SELECT event_type, fencing_token, details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type IN
                         ('artifact_write_started','artifact_write_quiescent')
                       ORDER BY event_id""",
                    job_id,
                )
                started = None
                for row in events:
                    details = _json_value(row["details"])
                    if (
                        str(details.get("operation_id", "")) == operation_id
                        and str(details.get("writer_attempt_id", "")) == attempt_id
                    ):
                        if row["event_type"] == "artifact_write_started":
                            started = row
                        elif started is None:
                            raise CheckpointIdentityError("artifact writer quiescence has no durable start event")
                        elif row["event_type"] == "artifact_write_quiescent":
                            return
                if started is None:
                    raise CheckpointIdentityError("artifact writer quiescence has no durable start event")
                start_details = _json_value(started["details"])
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(
                           job_id,event_type,state,fencing_token,details
                       ) VALUES($1::uuid,'artifact_write_quiescent',$2,$3,$4::jsonb)""",
                    job_id,
                    job["state"],
                    started["fencing_token"],
                    _canonical_json(
                        {
                            "operation_id": operation_id,
                            "writer_attempt_id": attempt_id,
                            "lease_token": start_details.get("lease_token"),
                            "fencing_token": start_details.get("fencing_token"),
                        }
                    ),
                )

    async def commit_checkpoint(
        self,
        *,
        job_id: str,
        lease_token: str,
        identity,
        prepared,
        store,
        source_registry: DatasetSourceRegistry | None = None,
        operation_id: str | None = None,
        precharged: bool = False,
        verified_artifact=None,
        artifact_invocation_id: str | None = None,
        artifact_deadline_at: datetime | None = None,
    ):
        try:
            return await self._commit_checkpoint_transaction(
                job_id=job_id,
                lease_token=lease_token,
                identity=identity,
                prepared=prepared,
                store=store,
                source_registry=source_registry,
                operation_id=operation_id,
                precharged=precharged,
                verified_artifact=verified_artifact,
                artifact_invocation_id=artifact_invocation_id,
                artifact_deadline_at=artifact_deadline_at,
            )
        except BaseException:
            if operation_id and precharged and verified_artifact is not None:
                try:
                    committed = await asyncio.shield(
                        self._checkpoint_operation_matches(
                            job_id=job_id,
                            operation_id=operation_id,
                            identity=identity,
                            prepared=prepared,
                            store=store,
                            verified=verified_artifact,
                        )
                    )
                except BaseException:
                    committed = False
                if committed:
                    return verified_artifact
            raise

    async def _checkpoint_operation_matches(
        self, *, job_id: str, operation_id: str, identity, prepared, store, verified
    ) -> bool:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT j.checkpoint_uri,j.checkpoint_sha256,j.checkpoint_identity,e.details
                   FROM gods_mlops_jobs j JOIN gods_mlops_job_events e ON e.job_id=j.job_id
                   WHERE j.job_id=$1::uuid AND e.event_type='checkpoint_committed'
                     AND e.details->>'operation_id'=$2
                   ORDER BY e.event_id DESC LIMIT 1""",
                job_id,
                operation_id,
            )
        if row is None:
            return False
        details = _json_value(row["details"])
        uri = _prepared_artifact_uri(prepared, store)
        verified_uri = _verified_checkpoint_uri(verified)
        return (
            str(row["checkpoint_uri"] or "") == uri == verified_uri
            and str(row["checkpoint_sha256"] or "").strip() == str(prepared.sha256).strip()
            and _json_value(row["checkpoint_identity"] or {}) == identity.as_dict()
            and details.get("operation_id") == operation_id
            and details.get("checkpoint_uri") == uri
            and details.get("sha256") == prepared.sha256
            and details.get("size_bytes") == prepared.size_bytes
            and details.get("metadata_size_bytes", 0) == prepared.metadata_size_bytes
            and details.get("object_key") == getattr(prepared, "object_key", None)
            and details.get("identity") == identity.as_dict()
            and str(verified.sha256).strip() == str(prepared.sha256).strip()
            and int(verified.size_bytes) == int(prepared.size_bytes)
        )

    async def _commit_checkpoint_transaction(
        self,
        *,
        job_id: str,
        lease_token: str,
        identity,
        prepared,
        store,
        source_registry: DatasetSourceRegistry | None = None,
        operation_id: str | None = None,
        precharged: bool = False,
        verified_artifact=None,
        artifact_invocation_id: str | None = None,
        artifact_deadline_at: datetime | None = None,
    ):
        from gods_mlops.jobs.checkpoints import (
            CheckpointIdentityError,
            StaleCheckpointOwnerError,
        )

        write_lifetime_id = operation_id or str(uuid4())
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                if identity.phase == "evaluation":
                    usage = await connection.fetchrow(
                        "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton=TRUE FOR UPDATE"
                    )
                    if usage is None:
                        raise RuntimeError("the shared object-storage ledger is not initialized")
                    if source_registry is None:
                        raise ValueError("Task 6 source registry is required before evaluation checkpoint commit")
                    reasons = await _phase_source_block_reasons_in_transaction(
                        source_registry,
                        connection,
                        phase="evaluation",
                        dataset_version=identity.dataset_version,
                        model_kind=identity.model_kind,
                    )
                    if reasons:
                        raise DatasetNotReadyForEvaluationError(reasons)
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                database_now = await connection.fetchval("SELECT clock_timestamp()")
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease["lease_token"])
                    or int(job["lease_generation"]) != int(lease["fencing_token"])
                    or lease["expires_at"] <= database_now
                    or job["state"] not in {"running", "yield_requested"}
                ):
                    raise StaleCheckpointOwnerError("checkpoint writer no longer owns the active GPU lease")
                await self._assert_artifact_write_deadline(
                    connection,
                    job=job,
                    lease=lease,
                    database_now=database_now,
                    expected_invocation_id=artifact_invocation_id,
                    expected_deadline_at=artifact_deadline_at,
                )
                expected = _checkpoint_identity_from_row(job)
                if identity != expected or prepared.identity != expected:
                    raise CheckpointIdentityError("checkpoint input, phase, model, or config identity changed")
                if identity.phase == "evaluation":
                    checkpoint_error = await _evaluation_checkpoint_source_error(connection, job)
                    if checkpoint_error:
                        raise CheckpointIdentityError("evaluation checkpoint source changed before checkpoint commit")
                elif identity.phase == "probe" and job.get("target_phase") == "evaluation":
                    checkpoint_error = await _evaluation_probe_checkpoint_source_error(connection, job)
                    if checkpoint_error:
                        raise CheckpointIdentityError(
                            "evaluation probe checkpoint source changed before checkpoint commit"
                        )
                checkpoint_uri_expected = _prepared_artifact_uri(prepared, store)
                existing_marker = (
                    str(job["checkpoint_uri"]) == checkpoint_uri_expected
                    and str(job["checkpoint_sha256"] or "").strip() == str(prepared.sha256).strip()
                    and _json_value(job["checkpoint_identity"] or {}) == identity.as_dict()
                )
                profile = await connection.fetchrow(
                    """
                    SELECT checkpoint_reservation_bytes FROM gods_mlops_resource_profiles
                    WHERE phase = $1 AND model_kind = $2 AND config_version = $3
                    """,
                    job["phase"],
                    job["model_kind"],
                    job["config_version"],
                )
                reservation = await connection.fetchrow(
                    """
                    SELECT * FROM gods_mlops_artifact_reservations
                    WHERE job_id = $1::uuid FOR UPDATE
                    """,
                    job_id,
                )
                prior_checkpoint = None
                if job["checkpoint_uri"] is not None:
                    prior_row = await connection.fetchrow(
                        """SELECT details FROM gods_mlops_job_events
                           WHERE job_id=$1::uuid AND event_type='checkpoint_committed'
                             AND details->>'checkpoint_uri'=$2
                             AND details->>'sha256'=$3
                           ORDER BY event_id DESC LIMIT 1""",
                        job_id,
                        str(job["checkpoint_uri"]),
                        str(job["checkpoint_sha256"]).strip(),
                    )
                    if prior_row is None:
                        raise CheckpointIdentityError("current checkpoint marker has no matching committed event")
                    prior_details = _json_value(prior_row["details"])
                    prior_checkpoint = {
                        "uri": str(job["checkpoint_uri"]),
                        "sha256": str(job["checkpoint_sha256"]).strip(),
                        "size_bytes": int(prior_details["size_bytes"]),
                        "metadata_size_bytes": int(prior_details.get("metadata_size_bytes", 0)),
                        "operation_id": prior_details.get("operation_id"),
                        "write_lifetime_id": _checkpoint_lifetime_id(prior_details),
                        "identity": prior_details.get("identity"),
                    }
                result_exists = await connection.fetchval(
                    """SELECT EXISTS(SELECT 1 FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type='result_artifact_committed')""",
                    job_id,
                )
                if profile is None or reservation is None:
                    raise ValueError("checkpoint exceeds its active versioned artifact reservation")
                if result_exists and not existing_marker:
                    raise ValueError("checkpoint cannot be replaced after a result artifact is committed")
                if not existing_marker and (
                    reservation["state"] != "reserved"
                    or prepared.size_bytes + prepared.metadata_size_bytes > profile["checkpoint_reservation_bytes"]
                ):
                    raise ValueError("checkpoint exceeds its active versioned artifact reservation")
                if precharged and not existing_marker:
                    pending = await connection.fetchrow(
                        """SELECT details FROM gods_mlops_job_events
                           WHERE job_id=$1::uuid AND event_type='artifact_write_pending'
                             AND details->>'operation_id'=$2
                           ORDER BY event_id DESC LIMIT 1""",
                        job_id,
                        operation_id,
                    ) if operation_id else None
                    if pending is None:
                        raise CheckpointIdentityError("checkpoint write has no durable precharged intent")
                    pending_details = _json_value(pending["details"])
                    if (
                        pending_details.get("operation") != "checkpoint"
                        or pending_details.get("operation_id") != operation_id
                        or pending_details.get("uri") != checkpoint_uri_expected
                        or pending_details.get("object_key") != getattr(prepared, "object_key", None)
                        or pending_details.get("sha256") != prepared.sha256
                        or pending_details.get("size_bytes") != prepared.size_bytes
                        or pending_details.get("metadata_size_bytes", 0) != prepared.metadata_size_bytes
                        or pending_details.get("charge_bytes")
                        != prepared.size_bytes + prepared.metadata_size_bytes
                        or pending_details.get("previous_uri") != getattr(prepared, "previous_uri", None)
                        or pending_details.get("previous_sha256") != getattr(prepared, "previous_sha256", None)
                        or pending_details.get("previous_size_bytes")
                        != getattr(prepared, "previous_size_bytes", None)
                        or pending_details.get("identity") != identity.as_dict()
                        or reservation is None
                        or reservation["consumed_bytes"] < pending_details.get("charge_bytes", 0)
                    ):
                        raise CheckpointIdentityError("checkpoint write differs from its durable pending intent")
                    await self._require_artifact_writers_quiescent(
                        connection,
                        job_id=job_id,
                        operation_id=operation_id,
                        required=getattr(prepared, "object_key", None) is not None,
                    )
                if existing_marker:
                    if getattr(prepared, "object_key", None) is not None:
                        verified = verified_artifact
                        if verified is None:
                            raise CheckpointIdentityError("S3 checkpoint retry has no off-loop exact readback")
                        if operation_id:
                            await self._require_artifact_writers_quiescent(
                                connection,
                                job_id=job_id,
                                operation_id=operation_id,
                                required=True,
                            )
                    else:
                        verified = store.commit(prepared)
                    if (
                        verified.identity != identity
                        or str(verified.sha256).strip() != str(prepared.sha256).strip()
                        or int(verified.size_bytes) != int(prepared.size_bytes)
                        or _verified_checkpoint_uri(verified) != checkpoint_uri_expected
                    ):
                        raise CheckpointIdentityError("verified checkpoint retry differs from its commit marker")
                    return verified
                if getattr(prepared, "object_key", None) is not None:
                    verified = verified_artifact
                    if verified is None:
                        raise CheckpointIdentityError("S3 checkpoint finalization has no verified object")
                else:
                    verified = store.commit(prepared)
                if (
                    verified.identity != identity
                    or str(verified.sha256).strip() != str(prepared.sha256).strip()
                    or int(verified.size_bytes) != int(prepared.size_bytes)
                ):
                    raise CheckpointIdentityError("verified checkpoint differs from its immutable write intent")
                checkpoint_uri = _verified_checkpoint_uri(verified)
                if checkpoint_uri != checkpoint_uri_expected:
                    raise CheckpointIdentityError("verified checkpoint URI differs from its pending object identity")
                publication_now = await connection.fetchval("SELECT clock_timestamp()")
                await self._assert_artifact_write_deadline(
                    connection,
                    job=job,
                    lease=lease,
                    database_now=publication_now,
                    expected_invocation_id=artifact_invocation_id,
                    expected_deadline_at=artifact_deadline_at,
                )
                fence = int(lease["fencing_token"])
                updated = await connection.execute(
                    """UPDATE gods_mlops_jobs AS j
                       SET checkpoint_uri=$3, checkpoint_sha256=$4,
                           checkpoint_identity=$5::jsonb, updated_at=now()
                       WHERE j.job_id=$1::uuid AND j.lease_token=$2::uuid
                         AND j.lease_generation=$6 AND j.state IN ('running','yield_requested')
                         AND EXISTS (
                           SELECT 1 FROM gods_mlops_gpu_leases AS l
                           WHERE l.job_id=j.job_id AND l.lease_token=$2::uuid
                             AND l.fencing_token=$6 AND l.expires_at>clock_timestamp()
                         )
                         AND (
                           ($7::uuid IS NULL AND NOT EXISTS (
                             SELECT 1 FROM gods_mlops_worker_artifact_deadlines AS d
                             WHERE d.job_id=j.job_id AND d.fencing_token=$6
                           ))
                           OR EXISTS (
                             SELECT 1 FROM gods_mlops_worker_artifact_deadlines AS d
                             WHERE d.job_id=j.job_id AND d.fencing_token=$6
                               AND d.lease_token=$2::uuid
                               AND d.artifact_deadline_at>clock_timestamp()
                               AND ($7::uuid IS NULL OR (
                                 d.controller_invocation_id=$7::uuid
                                 AND d.artifact_deadline_at=$8
                               ))
                           )
                         )""",
                    job_id,
                    lease_token,
                    checkpoint_uri,
                    verified.sha256,
                    _canonical_json(identity.as_dict()),
                    fence,
                    artifact_invocation_id,
                    artifact_deadline_at,
                )
                if updated != "UPDATE 1":
                    raise StaleCheckpointOwnerError("checkpoint publication lost its current lease or deadline")
                if not precharged:
                    if reservation["consumed_bytes"] + verified.size_bytes + prepared.metadata_size_bytes > reservation["reserved_bytes"]:
                        raise ValueError("checkpoint exceeds its remaining shared storage reservation")
                    await connection.execute(
                        """UPDATE gods_mlops_artifact_reservations
                           SET consumed_bytes=consumed_bytes+$2 WHERE job_id=$1::uuid""",
                        job_id,
                        verified.size_bytes + prepared.metadata_size_bytes,
                    )
                publication_now = await connection.fetchval("SELECT clock_timestamp()")
                await self._assert_artifact_write_deadline(
                    connection,
                    job=job,
                    lease=lease,
                    database_now=publication_now,
                    expected_invocation_id=artifact_invocation_id,
                    expected_deadline_at=artifact_deadline_at,
                )
                event_details = _canonical_json(
                    {
                        "checkpoint_uri": checkpoint_uri,
                        "sha256": verified.sha256,
                        "size_bytes": verified.size_bytes,
                        "metadata_size_bytes": prepared.metadata_size_bytes,
                        "object_key": getattr(prepared, "object_key", None),
                        "operation_id": operation_id,
                        "write_lifetime_id": write_lifetime_id,
                        "identity": identity.as_dict(),
                    }
                )
                event_id = await connection.fetchval(
                    """INSERT INTO gods_mlops_job_events (
                           job_id,event_type,state,fencing_token,details
                       )
                       SELECT $1::uuid,'checkpoint_committed',$2,$3,$4::jsonb
                       WHERE EXISTS (
                           SELECT 1 FROM gods_mlops_jobs AS j
                           JOIN gods_mlops_gpu_leases AS l ON l.job_id=j.job_id
                           WHERE j.job_id=$1::uuid AND j.lease_token=$5::uuid
                             AND j.lease_generation=$3 AND j.state IN ('running','yield_requested')
                             AND l.lease_token=$5::uuid AND l.fencing_token=$3
                             AND l.expires_at>clock_timestamp()
                             AND (
                               ($6::uuid IS NULL AND NOT EXISTS (
                                 SELECT 1 FROM gods_mlops_worker_artifact_deadlines AS d
                                 WHERE d.job_id=j.job_id AND d.fencing_token=$3
                               ))
                               OR EXISTS (
                                 SELECT 1 FROM gods_mlops_worker_artifact_deadlines AS d
                                 WHERE d.job_id=j.job_id AND d.fencing_token=$3
                                   AND d.lease_token=$5::uuid
                                   AND d.artifact_deadline_at>clock_timestamp()
                                   AND ($6::uuid IS NULL OR (
                                     d.controller_invocation_id=$6::uuid
                                     AND d.artifact_deadline_at=$7
                                   ))
                               )
                             )
                       )
                       RETURNING event_id""",
                    job_id,
                    job["state"],
                    fence,
                    event_details,
                    lease_token,
                    artifact_invocation_id,
                    artifact_deadline_at,
                )
                if event_id is None:
                    raise StaleCheckpointOwnerError("checkpoint event publication lost its current lease or deadline")
                if prior_checkpoint is not None and prior_checkpoint["uri"] != checkpoint_uri:
                    prior_uri = getattr(prepared, "previous_uri", None)
                    if prior_uri != prior_checkpoint["uri"]:
                        raise CheckpointIdentityError("checkpoint replacement lost its previous object identity")
                    previous_sha = getattr(prepared, "previous_sha256", None)
                    previous_size = getattr(prepared, "previous_size_bytes", None)
                    if (
                        previous_sha != prior_checkpoint["sha256"]
                        or previous_size != prior_checkpoint["size_bytes"]
                    ):
                        raise CheckpointIdentityError("checkpoint replacement changed its previous object identity")
                    existing_prune = await connection.fetchval(
                        """SELECT EXISTS(SELECT 1 FROM gods_mlops_job_events
                           WHERE job_id=$1::uuid AND event_type='checkpoint_prune_pending'
                             AND details->>'write_lifetime_id'=$2)""",
                        job_id,
                        prior_checkpoint["write_lifetime_id"],
                    )
                    if not existing_prune:
                        await connection.execute(
                            """INSERT INTO gods_mlops_job_events(
                                   job_id,event_type,state,fencing_token,details
                               ) VALUES($1::uuid,'checkpoint_prune_pending',$2,$3,$4::jsonb)""",
                            job_id,
                            job["state"],
                            lease["fencing_token"],
                            _canonical_json(
                                {
                                    **prior_checkpoint,
                                    "identity": identity.as_dict(),
                                    "write_lifetime_id": prior_checkpoint["write_lifetime_id"],
                                    "replacement_uri": checkpoint_uri,
                                    "replacement_sha256": verified.sha256,
                                }
                            ),
                        )
        return verified

    async def pending_checkpoint_prunes_for(self, job_id: str) -> list[dict[str, Any]]:
        """List exact prior checkpoint objects that still carry a deletion intent."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT event_type,details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid
                     AND event_type IN ('checkpoint_prune_pending','checkpoint_pruned')
                   ORDER BY event_id""",
                job_id,
            )
        pending: dict[str, dict[str, Any]] = {}
        resolved: set[str] = set()
        for row in rows:
            details = _json_value(row["details"])
            key = _checkpoint_lifetime_id(details)
            if row["event_type"] == "checkpoint_prune_pending":
                pending[key] = details
            else:
                resolved.add(key)
        return [details for key, details in pending.items() if key not in resolved]

    async def checkpoint_prune_writer_gate(
        self, *, job_id: str, previous: dict[str, Any]
    ) -> bool:
        """Exclusively check the old checkpoint lifetime before unlocked deletion I/O."""
        await self.ensure_schema()
        pool = await self._get_pool()
        operation_id = str(previous.get("operation_id", ""))
        lifetime_id = _checkpoint_lifetime_id(previous)
        uri = str(previous.get("uri", ""))
        digest = str(previous.get("sha256", ""))
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid FOR UPDATE", job_id
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if (
                    str(job["checkpoint_uri"] or "") == uri
                    and str(job["checkpoint_sha256"] or "").strip() == digest
                ):
                    return False
                rows = await connection.fetch(
                    """SELECT event_id,event_type,details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type IN
                         ('checkpoint_prune_pending','checkpoint_pruned',
                          'artifact_write_pending','artifact_write_started','artifact_write_quiescent')
                       ORDER BY event_id""",
                    job_id,
                )
                prune_pending_id = None
                resolved = False
                for row in rows:
                    if row["event_type"] not in {"checkpoint_prune_pending", "checkpoint_pruned"}:
                        continue
                    details = _json_value(row["details"])
                    if _checkpoint_lifetime_id(details) != lifetime_id:
                        continue
                    if row["event_type"] == "checkpoint_pruned":
                        resolved = True
                    else:
                        prune_pending_id = int(row.get("event_id", 0))
                if resolved or prune_pending_id is None:
                    return False
                related_writer_events = []
                for row in rows:
                    event_type = row["event_type"]
                    if event_type not in {
                        "artifact_write_pending",
                        "artifact_write_started",
                        "artifact_write_quiescent",
                    }:
                        continue
                    details = _json_value(row["details"])
                    same_operation = (
                        str(details.get("operation_id", "")) in {operation_id, lifetime_id}
                    )
                    same_object = (
                        str(details.get("uri", "")) == uri
                        and str(details.get("sha256", "")) == digest
                    )
                    if not (same_operation or same_object):
                        continue
                    if (
                        event_type == "artifact_write_started"
                        and int(row.get("event_id", 0)) > prune_pending_id
                    ):
                        return False
                    related_writer_events.append((event_type, details))
                return not _unmatched_artifact_writers(related_writer_events)

    async def pending_artifact_writes_for_cleanup(self, job_id: str) -> list[dict[str, Any]]:
        """Return only uncommitted S3 writes after the terminal owner has released its lease."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await connection.fetchrow(
                    "SELECT state FROM gods_mlops_jobs WHERE job_id=$1::uuid FOR UPDATE", job_id
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if job["state"] not in {"completed", "failed", "cancelled"}:
                    raise ValueError("pending artifact writes can be cleaned only for terminal jobs")
                lease_exists = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM gods_mlops_gpu_leases WHERE job_id=$1::uuid)", job_id
                )
                if lease_exists:
                    raise ValueError("pending artifact writes cannot be cleaned before GPU lease release")
                rows = await connection.fetch(
                    """SELECT event_type,details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid
                         AND event_type IN ('artifact_write_pending','artifact_write_deleted',
                                            'result_artifact_committed','checkpoint_committed',
                                            'artifact_write_started','artifact_write_quiescent')
                       ORDER BY event_id""",
                    job_id,
                )
        writer_events = [
            (row["event_type"], _json_value(row["details"]))
            for row in rows
            if row["event_type"] in {
                "artifact_write_pending",
                "artifact_write_started",
                "artifact_write_quiescent",
            }
        ]
        if _unmatched_artifact_writers(writer_events):
            raise ValueError("pending artifact cleanup is blocked by an active or unknown object writer")
        pending: dict[str, dict[str, Any]] = {}
        resolved: set[str] = set()
        for row in rows:
            details = _json_value(row["details"])
            operation_id = str(details.get("operation_id", ""))
            if not operation_id:
                continue
            if row["event_type"] in {"artifact_write_deleted", "result_artifact_committed", "checkpoint_committed"}:
                resolved.add(operation_id)
            elif row["event_type"] == "artifact_write_pending":
                pending[operation_id] = details
        return [details for operation_id, details in pending.items() if operation_id not in resolved]

    async def complete_artifact_write_delete(self, *, job_id: str, details: dict[str, Any]) -> bool:
        """Release charged bytes only after exact S3 absence was verified by the caller."""
        await self.ensure_schema()
        operation_id = str(details.get("operation_id", ""))
        if not operation_id:
            raise ValueError("artifact deletion has no durable operation ID")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton=TRUE FOR UPDATE"
                )
                if usage is None:
                    raise RuntimeError("the shared object-storage ledger is not initialized")
                job = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid FOR UPDATE", job_id
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if job["state"] not in {"completed", "failed", "cancelled"}:
                    raise ValueError("artifact deletion requires a terminal job")
                lease_exists = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM gods_mlops_gpu_leases WHERE job_id=$1::uuid)", job_id
                )
                if lease_exists:
                    raise ValueError("artifact deletion requires an observed GPU lease release")
                reservation = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_artifact_reservations WHERE job_id=$1::uuid FOR UPDATE",
                    job_id,
                )
                if reservation is None:
                    raise RuntimeError("pending artifact write has no shared storage reservation")
                rows = await connection.fetch(
                    """SELECT event_type,details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type IN
                         ('artifact_write_pending','artifact_write_deleted',
                          'result_artifact_committed','checkpoint_committed',
                          'artifact_write_started','artifact_write_quiescent')
                       ORDER BY event_id""",
                    job_id,
                )
                writer_events = [
                    (row["event_type"], _json_value(row["details"]))
                    for row in rows
                    if str(_json_value(row["details"]).get("operation_id", "")) == operation_id
                    and row["event_type"] in {
                        "artifact_write_pending",
                        "artifact_write_started",
                        "artifact_write_quiescent",
                    }
                ]
                if _unmatched_artifact_writers(writer_events):
                    raise ValueError("artifact deletion is blocked by an active or unknown object writer")
                pending = None
                committed = False
                deleted = False
                for row in rows:
                    event_details = _json_value(row["details"])
                    if str(event_details.get("operation_id", "")) != operation_id:
                        continue
                    if row["event_type"] == "artifact_write_pending":
                        pending = event_details
                    elif row["event_type"] == "artifact_write_deleted":
                        deleted = True
                    elif row["event_type"] in {"result_artifact_committed", "checkpoint_committed"}:
                        committed = True
                if deleted:
                    return False
                if committed:
                    raise ValueError("committed artifact writes cannot be deleted as pending")
                if pending is None or pending != details:
                    raise ValueError("artifact deletion does not match its exact durable pending intent")
                if (
                    str(job["checkpoint_uri"] or "") == str(details.get("uri", ""))
                    and str(job["checkpoint_sha256"] or "").strip() == str(details.get("sha256", "")).strip()
                ):
                    raise ValueError("the currently committed checkpoint cannot be deleted as pending")
                charge_bytes = int(details.get("charge_bytes", 0))
                if charge_bytes <= 0 or reservation["consumed_bytes"] < charge_bytes:
                    raise RuntimeError("pending artifact deletion exceeds accounted retained bytes")
                if reservation["state"] == "settled":
                    if usage["used_bytes"] < charge_bytes:
                        raise RuntimeError("shared storage ledger is below the verified deleted artifact")
                    await connection.execute(
                        "UPDATE ingestion_storage_usage SET used_bytes=used_bytes-$1 WHERE singleton=TRUE",
                        charge_bytes,
                    )
                elif reservation["state"] != "reserved":
                    raise RuntimeError("pending artifact has an unsupported reservation state")
                await connection.execute(
                    "UPDATE gods_mlops_artifact_reservations SET consumed_bytes=consumed_bytes-$2 WHERE job_id=$1::uuid",
                    job_id,
                    charge_bytes,
                )
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(job_id,event_type,state,details)
                       VALUES($1::uuid,'artifact_write_deleted',$2,$3::jsonb)""",
                    job_id,
                    job["state"],
                    _canonical_json(
                        {
                            "operation_id": operation_id,
                            "uri": details["uri"],
                            "sha256": details["sha256"],
                            "size_bytes": int(details["size_bytes"]),
                            "charge_bytes": charge_bytes,
                            "identity": details["identity"],
                        }
                    ),
                )
                return True

    async def complete_checkpoint_prune(self, *, job_id: str, previous: dict[str, Any]) -> bool:
        """Release previous-checkpoint bytes only after the object store proved deletion."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton=TRUE FOR UPDATE"
                )
                if usage is None:
                    raise RuntimeError("the shared object-storage ledger is not initialized")
                job = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid FOR UPDATE", job_id
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if str(job["checkpoint_uri"] or "") == str(previous.get("uri")):
                    raise ValueError("cannot prune the currently committed checkpoint")
                reservation = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_artifact_reservations WHERE job_id=$1::uuid FOR UPDATE",
                    job_id,
                )
                if reservation is None:
                    raise RuntimeError("checkpoint prune has no artifact reservation")
                rows = await connection.fetch(
                    """SELECT event_id,event_type,details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type IN (
                         'checkpoint_prune_pending','checkpoint_pruned','artifact_write_pending',
                         'artifact_write_started','artifact_write_quiescent')
                       ORDER BY event_id""",
                    job_id,
                )
                uri = str(previous.get("uri", ""))
                digest = str(previous.get("sha256", ""))
                lifetime_id = _checkpoint_lifetime_id(previous)
                has_pending = False
                already_pruned = False
                prune_pending_id = None
                for row in rows:
                    details = _json_value(row["details"])
                    if _checkpoint_lifetime_id(details) == lifetime_id:
                        if row["event_type"] == "checkpoint_prune_pending":
                            has_pending = True
                            prune_pending_id = int(row["event_id"])
                        elif row["event_type"] == "checkpoint_pruned":
                            already_pruned = True
                if not has_pending or already_pruned:
                    return False
                writer_events = []
                for row in rows:
                    event_type = row["event_type"]
                    if event_type not in {
                        "artifact_write_pending",
                        "artifact_write_started",
                        "artifact_write_quiescent",
                    }:
                        continue
                    writer_details = _json_value(row["details"])
                    same_operation = str(writer_details.get("operation_id", "")) in {
                        str(previous.get("operation_id", "")),
                        lifetime_id,
                    }
                    same_object = (
                        str(writer_details.get("uri", "")) == uri
                        and str(writer_details.get("sha256", "")) == digest
                    )
                    if not (same_operation or same_object):
                        continue
                    if (
                        event_type == "artifact_write_started"
                        and int(row["event_id"]) > int(prune_pending_id)
                    ):
                        raise ValueError("checkpoint writer started after its prune exclusion")
                    writer_events.append((event_type, writer_details))
                if _unmatched_artifact_writers(writer_events):
                    raise ValueError("checkpoint prune refund is blocked by an active or unknown writer")
                charge_bytes = int(previous.get("size_bytes", 0)) + int(
                    previous.get("metadata_size_bytes", 0)
                )
                if charge_bytes <= 0 or reservation["consumed_bytes"] < charge_bytes:
                    raise RuntimeError("checkpoint prune exceeds accounted retained artifact bytes")
                consumed_after = reservation["consumed_bytes"] - charge_bytes
                await connection.execute(
                    "UPDATE gods_mlops_artifact_reservations SET consumed_bytes=$2 WHERE job_id=$1::uuid",
                    job_id,
                    consumed_after,
                )
                if reservation["state"] == "settled":
                    if usage["used_bytes"] < charge_bytes:
                        raise RuntimeError("shared storage ledger is below the verified deleted checkpoint")
                    await connection.execute(
                        "UPDATE ingestion_storage_usage SET used_bytes=used_bytes-$1 WHERE singleton=TRUE",
                        charge_bytes,
                    )
                elif reservation["state"] != "reserved":
                    raise RuntimeError("checkpoint prune has an unsupported reservation state")
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(
                           job_id,event_type,state,details
                       ) VALUES($1::uuid,'checkpoint_pruned',$2,$3::jsonb)""",
                    job_id,
                    job["state"],
                    _canonical_json(
                        {
                            "uri": uri,
                            "sha256": digest,
                            "size_bytes": int(previous["size_bytes"]),
                            "metadata_size_bytes": int(previous.get("metadata_size_bytes", 0)),
                            "identity": previous.get("identity"),
                            "operation_id": previous.get("operation_id"),
                            "write_lifetime_id": lifetime_id,
                        }
                    ),
                )
                return True

    async def checkpoint_metadata_for(self, job_id: str) -> dict[str, Any] | None:
        """Return the current DB commit marker and its verified payload size."""
        job = await self.get_job(job_id)
        uri = job.get("checkpoint_uri")
        digest = job.get("checkpoint_sha256")
        identity = job.get("checkpoint_identity")
        if uri is None:
            return None
        if digest is None or not isinstance(identity, dict):
            raise ValueError("checkpoint commit marker is incomplete")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT details FROM gods_mlops_job_events
                WHERE job_id = $1::uuid AND event_type = 'checkpoint_committed'
                  AND details->>'checkpoint_uri' = $2
                  AND details->>'sha256' = $3
                ORDER BY event_id DESC LIMIT 1
                """,
                job_id,
                uri,
                str(digest).strip(),
            )
        if row is None:
            raise ValueError("checkpoint database commit marker has no matching event")
        details = _json_value(row["details"])
        if details.get("identity") != identity:
            raise ValueError("checkpoint event identity does not match the active job row")
        return {
            "uri": uri,
            "sha256": str(digest).strip(),
            "size_bytes": int(details["size_bytes"]),
            "metadata_size_bytes": int(details.get("metadata_size_bytes", 0)),
            "operation_id": details.get("operation_id"),
            "write_lifetime_id": _checkpoint_lifetime_id(details),
            "identity": identity,
        }

    async def commit_result_artifact(
        self,
        *,
        job_id: str,
        lease_token: str,
        identity,
        prepared,
        store,
        source_registry: DatasetSourceRegistry,
        operation_id: str | None = None,
        precharged: bool = False,
        runtime_measurements: dict[str, Any] | None = None,
        verified_artifact=None,
        artifact_invocation_id: str | None = None,
        artifact_deadline_at: datetime | None = None,
    ):
        try:
            return await self._commit_result_artifact_transaction(
                job_id=job_id,
                lease_token=lease_token,
                identity=identity,
                prepared=prepared,
                store=store,
                source_registry=source_registry,
                operation_id=operation_id,
                precharged=precharged,
                runtime_measurements=runtime_measurements,
                verified_artifact=verified_artifact,
                artifact_invocation_id=artifact_invocation_id,
                artifact_deadline_at=artifact_deadline_at,
            )
        except BaseException:
            if operation_id and precharged and verified_artifact is not None:
                try:
                    committed = await asyncio.shield(
                        self._result_operation_matches(
                            job_id=job_id,
                            operation_id=operation_id,
                            identity=identity,
                            prepared=prepared,
                            verified=verified_artifact,
                            runtime_measurements=runtime_measurements,
                        )
                    )
                except BaseException:
                    committed = False
                if committed:
                    return verified_artifact
            raise

    async def _result_operation_matches(
        self, *, job_id: str, operation_id: str, identity, prepared, verified, runtime_measurements
    ) -> bool:
        measured_details = json.loads(_canonical_json(runtime_measurements)) if runtime_measurements is not None else None
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type='result_artifact_committed'
                     AND details->>'operation_id'=$2
                   ORDER BY event_id DESC LIMIT 1""",
                job_id,
                operation_id,
            )
        if row is None:
            return False
        details = _json_value(row["details"])
        return (
            details.get("operation_id") == operation_id
            and details.get("kind") == prepared.kind == verified.kind
            and details.get("uri") == _prepared_artifact_uri(prepared, store=None)
            and details.get("object_key") == getattr(prepared, "object_key", None)
            and details.get("sha256") == prepared.sha256 == verified.sha256
            and details.get("size_bytes") == prepared.size_bytes == verified.size_bytes
            and details.get("identity") == identity.as_dict()
            and details.get("runtime_measurements") == measured_details
            and verified.identity == identity
        )

    async def _commit_result_artifact_transaction(
        self,
        *,
        job_id: str,
        lease_token: str,
        identity,
        prepared,
        store,
        source_registry: DatasetSourceRegistry,
        operation_id: str | None = None,
        precharged: bool = False,
        runtime_measurements: dict[str, Any] | None = None,
        verified_artifact=None,
        artifact_invocation_id: str | None = None,
        artifact_deadline_at: datetime | None = None,
    ):
        """Publish one immutable S3/file result under the current fence and reserved quota."""
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError, StaleCheckpointOwnerError

        await self.ensure_schema()
        measured_details = json.loads(_canonical_json(runtime_measurements)) if runtime_measurements is not None else None
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                if usage is None:
                    raise RuntimeError("the shared object-storage ledger is not initialized")
                if identity.phase in {"training", "evaluation"}:
                    source_reasons = await _phase_source_block_reasons_in_transaction(
                        source_registry,
                        connection,
                        phase=identity.phase,
                        dataset_version=identity.dataset_version,
                        model_kind=identity.model_kind,
                    )
                    if source_reasons:
                        if identity.phase == "evaluation":
                            raise DatasetNotReadyForEvaluationError(source_reasons)
                        raise DatasetNotReadyForTrainingError(source_reasons)
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                database_now = await connection.fetchval("SELECT clock_timestamp()")
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease["lease_token"])
                    or int(job["lease_generation"]) != int(lease["fencing_token"])
                    or lease["expires_at"] <= database_now
                    or job["state"] != "running"
                ):
                    raise StaleCheckpointOwnerError("result artifact writer no longer owns the running GPU lease")
                await self._assert_artifact_write_deadline(
                    connection,
                    job=job,
                    lease=lease,
                    database_now=database_now,
                    expected_invocation_id=artifact_invocation_id,
                    expected_deadline_at=artifact_deadline_at,
                )
                expected = _checkpoint_identity_from_row(job)
                if identity != expected or prepared.identity != expected or str(job["job_id"]) != job_id:
                    raise CheckpointIdentityError("result artifact input, phase, model, or config identity changed")
                if identity.phase == "evaluation":
                    checkpoint_error = await _evaluation_checkpoint_source_error(connection, job)
                    if checkpoint_error:
                        raise CheckpointIdentityError("evaluation checkpoint source changed before result commit")
                elif identity.phase == "probe" and job.get("target_phase") == "evaluation":
                    checkpoint_error = await _evaluation_probe_checkpoint_source_error(connection, job)
                    if checkpoint_error:
                        raise CheckpointIdentityError(
                            "evaluation probe checkpoint source changed before result commit"
                        )
                profile = await connection.fetchrow(
                    """SELECT result_reservation_bytes FROM gods_mlops_resource_profiles
                       WHERE phase=$1 AND model_kind=$2 AND config_version=$3""",
                    job["phase"],
                    job["model_kind"],
                    job["config_version"],
                )
                reservation = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_artifact_reservations WHERE job_id=$1::uuid FOR UPDATE",
                    job_id,
                )
                if profile is None or reservation is None:
                    raise ValueError("result artifact exceeds its active versioned reservation")
                existing_rows = await connection.fetch(
                    """SELECT details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type='result_artifact_committed'""",
                    job_id,
                )
                existing = [_json_value(row["details"]) for row in existing_rows]
                same_kind = [item for item in existing if item.get("kind") == prepared.kind]
                if same_kind and any(
                    item.get("sha256") != prepared.sha256
                    or item.get("identity") != identity.as_dict()
                    or item.get("runtime_measurements") != measured_details
                    for item in same_kind
                ):
                    raise ResultArtifactConflictError("result artifact kind already has different immutable bytes")
                if same_kind:
                    prior = same_kind[-1]
                    if getattr(prepared, "object_key", None) is not None:
                        verified = verified_artifact
                        if verified is None:
                            raise CheckpointIdentityError("S3 result retry has no off-loop exact readback")
                        if operation_id:
                            await self._require_artifact_writers_quiescent(
                                connection,
                                job_id=job_id,
                                operation_id=operation_id,
                                required=True,
                            )
                    else:
                        verified = store.commit(prepared)
                    if prior.get("uri") != verified.uri or prior.get("size_bytes") != verified.size_bytes:
                        raise ResultArtifactConflictError("result artifact retry does not match its prior commit")
                    if (
                        verified.identity != identity
                        or verified.kind != prepared.kind
                        or verified.sha256 != prepared.sha256
                        or (operation_id is not None and prior.get("operation_id") != operation_id)
                    ):
                        raise ResultArtifactConflictError("result artifact retry verification differs from its marker")
                    return verified
                if (
                    reservation["state"] != "reserved"
                    or prepared.size_bytes > profile["result_reservation_bytes"]
                    or (
                        not precharged
                        and reservation["consumed_bytes"] + prepared.size_bytes > reservation["reserved_bytes"]
                    )
                ):
                    raise ValueError("result artifact exceeds its active versioned reservation")
                if precharged:
                    pending = await connection.fetchrow(
                        """SELECT details FROM gods_mlops_job_events
                           WHERE job_id=$1::uuid AND event_type='artifact_write_pending'
                             AND details->>'operation_id'=$2
                           ORDER BY event_id DESC LIMIT 1""",
                        job_id,
                        operation_id,
                    ) if operation_id else None
                    if pending is None:
                        raise CheckpointIdentityError("result write has no durable precharged intent")
                    pending_details = _json_value(pending["details"])
                    if (
                        pending_details.get("operation") != "result"
                        or pending_details.get("operation_id") != operation_id
                        or pending_details.get("kind") != prepared.kind
                        or pending_details.get("uri") != _prepared_artifact_uri(prepared, store)
                        or pending_details.get("object_key") != getattr(prepared, "object_key", None)
                        or pending_details.get("sha256") != prepared.sha256
                        or pending_details.get("size_bytes") != prepared.size_bytes
                        or pending_details.get("charge_bytes") != prepared.size_bytes
                        or pending_details.get("identity") != identity.as_dict()
                        or pending_details.get("runtime_measurements") != measured_details
                        or reservation["consumed_bytes"] < pending_details.get("charge_bytes", 0)
                    ):
                        raise CheckpointIdentityError("result write differs from its durable pending intent")
                    await self._require_artifact_writers_quiescent(
                        connection,
                        job_id=job_id,
                        operation_id=operation_id,
                        required=getattr(prepared, "object_key", None) is not None,
                    )
                if getattr(prepared, "object_key", None) is not None:
                    verified = verified_artifact
                    if verified is None:
                        raise CheckpointIdentityError("S3 result finalization has no verified object")
                else:
                    verified = store.commit(prepared)
                if (
                    verified.identity != identity
                    or verified.kind != prepared.kind
                    or str(verified.sha256).strip() != str(prepared.sha256).strip()
                    or int(verified.size_bytes) != int(prepared.size_bytes)
                    or verified.uri != _prepared_artifact_uri(prepared, store)
                ):
                    raise CheckpointIdentityError("verified result differs from its immutable write intent")
                details = {
                    "kind": prepared.kind,
                    "uri": verified.uri,
                    "sha256": verified.sha256,
                    "size_bytes": verified.size_bytes,
                    "identity": identity.as_dict(),
                    "object_key": getattr(prepared, "object_key", None),
                    "operation_id": operation_id,
                }
                details["runtime_measurements"] = measured_details
                if not precharged:
                    await connection.execute(
                        "UPDATE gods_mlops_artifact_reservations SET consumed_bytes=consumed_bytes+$2 WHERE job_id=$1::uuid",
                        job_id,
                        verified.size_bytes,
                    )
                publication_now = await connection.fetchval("SELECT clock_timestamp()")
                await self._assert_artifact_write_deadline(
                    connection,
                    job=job,
                    lease=lease,
                    database_now=publication_now,
                    expected_invocation_id=artifact_invocation_id,
                    expected_deadline_at=artifact_deadline_at,
                )
                event_id = await connection.fetchval(
                    """INSERT INTO gods_mlops_job_events(
                           job_id,event_type,state,fencing_token,details
                       )
                       SELECT $1::uuid,'result_artifact_committed','running',$2,$3::jsonb
                       WHERE EXISTS (
                           SELECT 1 FROM gods_mlops_jobs AS j
                           JOIN gods_mlops_gpu_leases AS l ON l.job_id=j.job_id
                           WHERE j.job_id=$1::uuid AND j.lease_token=$4::uuid
                             AND j.lease_generation=$2 AND j.state='running'
                             AND l.lease_token=$4::uuid AND l.fencing_token=$2
                             AND l.expires_at>clock_timestamp()
                             AND (
                               ($5::uuid IS NULL AND NOT EXISTS (
                                 SELECT 1 FROM gods_mlops_worker_artifact_deadlines AS d
                                 WHERE d.job_id=j.job_id AND d.fencing_token=$2
                               ))
                               OR EXISTS (
                                 SELECT 1 FROM gods_mlops_worker_artifact_deadlines AS d
                                 WHERE d.job_id=j.job_id AND d.fencing_token=$2
                                   AND d.lease_token=$4::uuid
                                   AND d.artifact_deadline_at>clock_timestamp()
                                   AND ($5::uuid IS NULL OR (
                                     d.controller_invocation_id=$5::uuid
                                     AND d.artifact_deadline_at=$6
                                   ))
                               )
                             )
                       )
                       RETURNING event_id""",
                    job_id,
                    lease["fencing_token"],
                    _canonical_json(details),
                    lease_token,
                    artifact_invocation_id,
                    artifact_deadline_at,
                )
                if event_id is None:
                    raise StaleCheckpointOwnerError("result event publication lost its current lease or deadline")
                return verified

    async def complete_owned_job(
        self,
        *,
        job_id: str,
        lease_token: str,
        identity,
        details: dict[str, Any],
        source_registry: DatasetSourceRegistry,
    ) -> dict[str, Any]:
        """Mark Task 8 work successful under its fence; the monitor still owns GPU release."""
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError, StaleCheckpointOwnerError

        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                if usage is None:
                    raise RuntimeError("the shared object-storage ledger is not initialized")
                if identity.phase in {"training", "evaluation"}:
                    source_reasons = await _phase_source_block_reasons_in_transaction(
                        source_registry,
                        connection,
                        phase=identity.phase,
                        dataset_version=identity.dataset_version,
                        model_kind=identity.model_kind,
                    )
                    if source_reasons:
                        if identity.phase == "evaluation":
                            raise DatasetNotReadyForEvaluationError(source_reasons)
                        raise DatasetNotReadyForTrainingError(source_reasons)
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                expected = _checkpoint_identity_from_row(job)
                if identity != expected:
                    raise CheckpointIdentityError("completion identity does not match the immutable job")
                if job["state"] == "completed":
                    return _job_dict(job)
                if (
                    lease is None
                    or str(job["lease_token"]) != str(lease_token)
                    or job["state"] != "running"
                    or job["phase"] not in {"preparation", "training", "evaluation"}
                ):
                    raise StaleCheckpointOwnerError("completion does not own a running GPU job lease")
                if identity.phase == "evaluation":
                    checkpoint_error = await _evaluation_checkpoint_source_error(connection, job)
                    if checkpoint_error:
                        raise CheckpointIdentityError("evaluation checkpoint source changed before completion")
                profile = await connection.fetchrow(
                    """SELECT result_reservation_bytes FROM gods_mlops_resource_profiles
                       WHERE phase=$1 AND model_kind=$2 AND config_version=$3""",
                    job["phase"],
                    job["model_kind"],
                    job["config_version"],
                )
                reservation = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_artifact_reservations WHERE job_id=$1::uuid FOR UPDATE",
                    job_id,
                )
                artifact_rows = await connection.fetch(
                    """SELECT details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type='result_artifact_committed'""",
                    job_id,
                )
                if profile is None or reservation is None or reservation["state"] != "reserved":
                    raise ValueError("successful job has no active artifact reservation")
                if not artifact_rows:
                    raise ValueError("successful job must commit a result artifact before completion")
                if reservation["consumed_bytes"] > reservation["reserved_bytes"]:
                    raise RuntimeError("job artifact reservation accounting is inconsistent")
                if usage["used_bytes"] < reservation["reserved_bytes"]:
                    raise RuntimeError("shared object-storage ledger is below the active reservation")
                canonical_details = _canonical_json(details)
                completed = await connection.fetchrow(
                    """UPDATE gods_mlops_jobs SET state='completed',reason_code=NULL,
                           reason_detail=$3::jsonb,retryable=FALSE,completed_at=now(),updated_at=now()
                       WHERE job_id=$1::uuid AND lease_token=$2::uuid RETURNING *""",
                    job_id,
                    lease_token,
                    canonical_details,
                )
                if completed is None:
                    raise StaleCheckpointOwnerError("completion fence changed before terminal commit")
                usage_after_settlement = (
                    usage["used_bytes"] - reservation["reserved_bytes"] + reservation["consumed_bytes"]
                )
                if usage_after_settlement < 0 or usage_after_settlement > 1024**4:
                    raise RuntimeError("settled artifact usage is outside the one TiB storage ledger")
                await connection.execute(
                    "UPDATE ingestion_storage_usage SET used_bytes=$1 WHERE singleton=TRUE",
                    usage_after_settlement,
                )
                await connection.execute(
                    """UPDATE gods_mlops_artifact_reservations SET state='settled',settled_at=now()
                       WHERE job_id=$1::uuid""",
                    job_id,
                )
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(
                           job_id,event_type,state,fencing_token,details
                       ) VALUES($1::uuid,'job_completed','completed',$2,$3::jsonb)""",
                    job_id,
                    lease["fencing_token"],
                    _canonical_json({"identity": identity.as_dict(), "result": details}),
                )
                return _job_dict(completed)

    async def result_artifacts_for(self, job_id: str) -> list[dict[str, Any]]:
        """Return only committed result artifact events for one durable Task 8 job."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type='result_artifact_committed'
                   ORDER BY created_at, event_id""",
                job_id,
            )
        return [_json_value(row["details"]) for row in rows]

    async def profile_measurement_for_job(self, job_id: str) -> dict[str, Any] | None:
        """Return one real model probe measurement and its evidence fields."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """SELECT measurement_id,input_sha256,config_sha256,model_kind,target_phase,
                          result_state,peak_allocated_mib,peak_reserved_mib,optimizer_steps,
                          inference_steps,checkpoint_resumed,verification_details,
                          checkpoint_sha256,exit_code,measured_at
                   FROM gods_mlops_profile_measurements WHERE job_id=$1::uuid""",
                job_id,
            )
        if row is None:
            return None
        result = dict(row)
        result["measurement_id"] = str(result["measurement_id"])
        result["input_sha256"] = result["input_sha256"].strip()
        result["config_sha256"] = result["config_sha256"].strip()
        result["checkpoint_sha256"] = (
            result["checkpoint_sha256"].strip() if result["checkpoint_sha256"] else None
        )
        result["verification_details"] = _json_value(result["verification_details"])
        result["measured_at"] = result["measured_at"].isoformat()
        return result

    async def probe_runtime_evidence_for_job(self, job_id: str) -> dict[str, Any] | None:
        """Read the unique immutable runtime provenance event for a completed training probe."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type=$2 ORDER BY event_id""",
                job_id,
                _PROBE_RUNTIME_EVIDENCE_EVENT,
            )
        if len(rows) != 1:
            return None
        details = _json_value(rows[0]["details"])
        if (
            not isinstance(details, dict)
            or details.get("schema") != _PROBE_RUNTIME_EVIDENCE_SCHEMA
            or details.get("job_id") != str(job_id)
        ):
            return None
        return details

    async def _record_probe_runtime_evidence(self, evidence_payload: bytes) -> dict[str, Any]:
        """Persist evidence produced by the trusted Docker probe controller, once per origin."""
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError
        from gods_mlops.jobs.models import EvaluationProbeCheckpointSource

        evidence, evidence_sha256, evidence_canonical_sha256 = _probe_runtime_evidence_projection(evidence_payload)
        job_id = evidence["job_id"]
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                prior_row = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid FOR UPDATE",
                    job_id,
                )
                if prior_row is None:
                    raise CheckpointIdentityError("runtime evidence has no completed probe origin")
                job = _job_dict(prior_row)
                job["checkpoint_identity"] = _json_value(job.get("checkpoint_identity"))
                probe_profile_row = await connection.fetchrow(
                    """SELECT phase,target_phase,model_kind,config_version,config_sha256,
                              profile_state,config_json
                       FROM gods_mlops_resource_profiles
                       WHERE phase='probe' AND model_kind=$1 AND config_version=$2""",
                    job.get("model_kind"),
                    job.get("config_version"),
                )
                training_profile_row = await connection.fetchrow(
                    """SELECT phase,target_phase,model_kind,config_version,config_sha256,
                              profile_state,config_json
                       FROM gods_mlops_resource_profiles
                       WHERE phase='training' AND model_kind=$1 AND config_version=$2""",
                    job.get("model_kind"),
                    job.get("config_version"),
                )
                measurement_row = await connection.fetchrow(
                    """SELECT measurement_id,input_sha256,config_sha256,model_kind,target_phase,
                              result_state,peak_allocated_mib,peak_reserved_mib,optimizer_steps,
                              inference_steps,checkpoint_resumed,verification_details,
                              checkpoint_sha256,exit_code,measured_at
                       FROM gods_mlops_profile_measurements WHERE job_id=$1::uuid""",
                    job_id,
                )
                commit_row = await connection.fetchrow(
                    """SELECT details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type='checkpoint_committed'
                       ORDER BY event_id DESC LIMIT 1""",
                    job_id,
                )
                result_rows = await connection.fetch(
                    """SELECT details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type='result_artifact_committed'
                       ORDER BY event_id""",
                    job_id,
                )
                if any(
                    item is None
                    for item in (probe_profile_row, training_profile_row, measurement_row, commit_row)
                ) or len(result_rows) != 1:
                    raise CheckpointIdentityError("runtime evidence origin rows are incomplete or ambiguous")

                probe_profile = dict(probe_profile_row)
                training_profile = dict(training_profile_row)
                probe_profile["config_json"] = _json_value(probe_profile["config_json"])
                training_profile["config_json"] = _json_value(training_profile["config_json"])
                measurement = dict(measurement_row)
                measurement["measurement_id"] = str(measurement["measurement_id"])
                measurement["input_sha256"] = measurement["input_sha256"].strip()
                measurement["config_sha256"] = measurement["config_sha256"].strip()
                measurement["checkpoint_sha256"] = (
                    measurement["checkpoint_sha256"].strip() if measurement["checkpoint_sha256"] else None
                )
                measurement["verification_details"] = _json_value(measurement["verification_details"])
                measured_at = measurement["measured_at"]
                measurement["measured_at"] = (
                    measured_at.isoformat() if hasattr(measured_at, "isoformat") else str(measured_at)
                )
                checkpoint_commit = _json_value(commit_row["details"])
                result_artifact = _json_value(result_rows[0]["details"])

                profile_config = probe_profile.get("config_json")
                if not isinstance(profile_config, dict):
                    raise CheckpointIdentityError("runtime evidence origin has no candidate profile config")
                try:
                    source = EvaluationProbeCheckpointSource(
                        training_probe_job_id=job_id,
                        model_kind=str(job["model_kind"]),
                        model_id=str(profile_config["model_id"]),
                        model_revision=str(profile_config["model_revision"]),
                        checkpoint_uri=str(checkpoint_commit.get("uri", checkpoint_commit.get("checkpoint_uri"))),
                        checkpoint_sha256=str(checkpoint_commit["sha256"]).strip(),
                        checkpoint_size_bytes=int(checkpoint_commit["size_bytes"]),
                        checkpoint_identity=job["checkpoint_identity"],
                        worker_image_id=evidence["docker_image_id"],
                        source_commit=evidence["source_commit"],
                        runtime_evidence_sha256=evidence_sha256,
                    )
                    authority = _probe_runtime_evidence_authority(
                        evidence,
                        evidence_sha256,
                        evidence_canonical_sha256,
                        job=job,
                        measurement=measurement,
                        checkpoint_commit=checkpoint_commit,
                        result_artifact=result_artifact,
                    )
                    _require_runtime_evidence_matches_committed_rows(
                        evidence_payload,
                        measurement=measurement,
                        result_artifact=result_artifact,
                    )
                    source.validate_training_probe_origin(
                        job,
                        probe_profile,
                        training_profile,
                        measurement,
                        checkpoint_commit,
                        authority,
                        result_artifact,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise CheckpointIdentityError("runtime evidence differs from its terminal probe origin") from error

                existing_rows = await connection.fetch(
                    """SELECT details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type=$2 ORDER BY event_id""",
                    job_id,
                    _PROBE_RUNTIME_EVIDENCE_EVENT,
                )
                if existing_rows:
                    existing = [_json_value(row["details"]) for row in existing_rows]
                    if len(existing) == 1:
                        prior = existing[0]
                        prior_without_exact_bytes = {key: value for key, value in prior.items() if key != "evidence_sha256"}
                        authority_without_exact_bytes = {
                            key: value for key, value in authority.items() if key != "evidence_sha256"
                        }
                        if prior_without_exact_bytes == authority_without_exact_bytes:
                            return prior
                    raise CheckpointIdentityError("probe job already has different runtime evidence")
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(job_id,event_type,state,details)
                       VALUES($1::uuid,$2,'completed',$3::jsonb)""",
                    job_id,
                    _PROBE_RUNTIME_EVIDENCE_EVENT,
                    _canonical_json(authority),
                )
                return authority

    async def review_handoffs_for(self, job_id: str) -> list[dict[str, Any]]:
        """Return CPU-published Task 5 assignment links attached after GPU exit."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type='preparation_review_assignment'
                   ORDER BY created_at, event_id""",
                job_id,
            )
        return [_json_value(row["details"]) for row in rows]

    async def record_preparation_review_assignment(
        self,
        *,
        job_id: str,
        item_id: str,
        sample_id: str,
        assignment_revision: str,
        project_id: int,
        model_request_key: str,
        model_version: str,
        model_id: str,
        model_revision: str,
        media_object_key: str,
        source_sha256: str,
        source_size_bytes: int,
        bbox_revision: str | None,
        result_sha256: str,
    ) -> dict[str, Any]:
        """Persist a Task 5 revision link after the GPU lease has been observed released."""
        from uuid import UUID

        UUID(job_id)
        UUID(sample_id)
        UUID(assignment_revision)
        _validate_digest(result_sha256, "result_sha256")
        _validate_digest(model_request_key, "model_request_key")
        _validate_digest(source_sha256, "source_sha256")
        if (
            not item_id
            or len(item_id) > 255
            or not model_version
            or len(model_version) > 255
            or not model_id
            or len(model_id) > 255
            or not model_revision
            or len(model_revision) > 255
            or project_id <= 0
            or source_size_bytes <= 0
            or not media_object_key
        ):
            raise ValueError("preparation review provenance is invalid")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid FOR UPDATE", job_id
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if (
                    job["phase"] != "preparation"
                    or job["state"] != "completed"
                    or job["lease_token"] is not None
                    or job["model_kind"] not in {"detr", "qwen"}
                ):
                    raise ValueError("Task 5 assignment handoff requires completed preparation after observed GPU exit")
                refs = _json_value(job["source_refs"])
                matches = [
                    item
                    for item in refs.get("items", [])
                    if str(item.get("item_id")) == item_id
                    and str(item.get("sample_id")) == sample_id
                    and item.get("object_key") == media_object_key
                    and str(item.get("sha256")) == source_sha256
                    and int(item.get("object_size_bytes", -1)) == source_size_bytes
                    and (
                        str(item.get("revision_id"))
                        if item.get("revision_id") is not None
                        else None
                    )
                    == bbox_revision
                ]
                if len(matches) != 1:
                    raise ValueError("Task 5 assignment handoff item is outside the completed immutable batch")
                if (job["model_kind"] == "detr") != (matches[0].get("item_kind") == "frame"):
                    raise ValueError("Task 5 assignment handoff model/source kind does not match the job")
                if job["model_kind"] == "qwen" and matches[0].get("item_kind") != "crop":
                    raise ValueError("Qwen assignment handoff requires the exact reviewed crop source")
                result_events = await connection.fetch(
                    """SELECT details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type='result_artifact_committed'""",
                    job_id,
                )
                if not any(
                    item.get("sha256") == result_sha256
                    and item.get("kind") == ("drafts" if job["model_kind"] == "detr" else "caption_drafts")
                    and item.get("identity") == _checkpoint_identity_from_row(job).as_dict()
                    for item in (_json_value(row["details"]) for row in result_events)
                ):
                    raise ValueError("Task 5 assignment handoff lacks a committed result artifact")
                assignment = await connection.fetchrow(
                    """SELECT assignment.sample_id,assignment.stage,assignment.bbox_revision,
                              assignment.media_object_key,assignment.required_bytes,assignment.state,
                              media.project_id,media.upload_filename,media.sha256,media.object_size_bytes,
                              media.state AS media_state
                       FROM review_assignments AS assignment
                       JOIN label_studio_media_uploads AS media
                         ON media.assignment_revision=assignment.revision
                       WHERE assignment.revision=$1::uuid FOR UPDATE OF assignment,media""",
                    assignment_revision,
                )
                request_marker = sha256(model_version.encode("utf-8")).hexdigest()[:16]
                expected_filename = f"gods-model-{model_request_key[:32]}-{request_marker}.jpg"
                expected_stage = "bbox" if job["model_kind"] == "detr" else "caption"
                if assignment is not None and assignment["project_id"] != project_id:
                    raise ValueError("Task 5 assignment project differs from the configured review project")
                if (
                    assignment is None
                    or str(assignment["sample_id"]) != sample_id
                    or assignment["stage"] != expected_stage
                    or assignment["media_object_key"] != media_object_key
                    or assignment["required_bytes"] != source_size_bytes
                    or assignment["upload_filename"] != expected_filename
                    or assignment["sha256"].strip() != source_sha256
                    or assignment["object_size_bytes"] != source_size_bytes
                    or assignment["bbox_revision"] != bbox_revision
                    or assignment["state"] not in {"provisioning", "active", "finalized"}
                    or assignment["media_state"] not in {"reserved", "uploaded", "deleted"}
                ):
                    raise ValueError("Task 5 assignment differs from the immutable source or model provenance")
                active_lease = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM gods_mlops_gpu_leases WHERE job_id=$1::uuid)",
                    job_id,
                )
                if active_lease:
                    raise ValueError("Task 5 assignment handoff cannot run while the GPU lease is active")
                details = {
                    "item_id": item_id,
                    "sample_id": sample_id,
                    "assignment_revision": assignment_revision,
                    "project_id": project_id,
                    "model_request_key": model_request_key,
                    "media_object_key": media_object_key,
                    "source_sha256": source_sha256,
                    "source_size_bytes": source_size_bytes,
                    "bbox_revision": bbox_revision,
                    "result_sha256": result_sha256,
                    "model_version": model_version,
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "input_kind": job["input_kind"],
                    "input_id": job["input_id"],
                    "input_sha256": job["input_sha256"].strip(),
                    "model_kind": job["model_kind"],
                    "config_version": job["config_version"],
                    "config_sha256": job["config_sha256"].strip(),
                }
                existing = await connection.fetch(
                    """SELECT details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid AND event_type='preparation_review_assignment'""",
                    job_id,
                )
                for row in existing:
                    current = _json_value(row["details"])
                    if current.get("item_id") == item_id:
                        if current != details:
                            raise ResultArtifactConflictError(
                                "Task 5 assignment handoff changed for an immutable preparation result"
                            )
                        return current
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(
                           job_id,event_type,state,details
                       ) VALUES($1::uuid,'preparation_review_assignment','completed',$2::jsonb)""",
                    job_id,
                    _canonical_json(details),
                )
                return details

    async def list_released_terminal_artifact_reservations(self) -> list[str]:
        """Find only terminal, unleased reservations with durable release proof."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """SELECT job.job_id
                   FROM gods_mlops_jobs AS job
                   JOIN gods_mlops_artifact_reservations AS reservation USING (job_id)
                   WHERE job.state = ANY($1::text[])
                     AND reservation.state = 'reserved'
                     AND job.lease_token IS NULL
                     AND job.lease_expires_at IS NULL
                     AND job.owner_pid IS NULL
                     AND job.owner_start_ticks IS NULL
                     AND job.owner_uid IS NULL
                     AND NOT EXISTS (
                       SELECT 1 FROM gods_mlops_gpu_leases AS lease
                       WHERE lease.job_id = job.job_id
                     )
                     AND (
                       COALESCE(job.lease_generation, 0) = 0
                       OR EXISTS (
                         SELECT 1 FROM gods_mlops_job_events AS event
                         WHERE event.job_id = job.job_id
                           AND event.event_type = 'lease_owner_released'
                           AND event.fencing_token = job.lease_generation
                       )
                     )
                   ORDER BY job.updated_at ASC, job.job_id ASC""",
                ["completed", "failed", "cancelled"],
            )
        return [str(row["job_id"]) for row in rows]

    async def settle_artifact_reservation(self, job_id: str) -> dict[str, Any] | None:
        """Settle a terminal probe/failure reservation once and release unused global quota."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton=TRUE FOR UPDATE"
                )
                if usage is None:
                    raise RuntimeError("the shared object-storage ledger is not initialized")
                job = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid FOR UPDATE", job_id
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                if job["state"] not in {"completed", "failed", "cancelled"}:
                    raise ValueError("only terminal jobs can settle their artifact reservation")
                lease_exists = await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM gods_mlops_gpu_leases WHERE job_id=$1::uuid)",
                    job_id,
                )
                if lease_exists:
                    # Terminal probe/failure results can be recorded before the monitor
                    # observes process exit. Keep the reservation intact until that release.
                    return _job_dict(job)
                reservation = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_artifact_reservations WHERE job_id=$1::uuid FOR UPDATE",
                    job_id,
                )
                if reservation is None or reservation["state"] == "settled":
                    return _job_dict(job)
                writer_rows = await connection.fetch(
                    """SELECT event_type,details FROM gods_mlops_job_events
                       WHERE job_id=$1::uuid
                         AND event_type IN ('artifact_write_pending','artifact_write_started',
                                            'artifact_write_quiescent')
                       ORDER BY event_id""",
                    job_id,
                )
                if _unmatched_artifact_writers(
                    [(row["event_type"], _json_value(row["details"])) for row in writer_rows]
                ):
                    # Keep the quota charged until a later monitor pass can prove
                    # that every durable writer attempt has become quiescent.
                    return _job_dict(job)
                if usage["used_bytes"] < reservation["reserved_bytes"]:
                    raise RuntimeError("shared object-storage ledger is below the active reservation")
                used_after = usage["used_bytes"] - reservation["reserved_bytes"] + reservation["consumed_bytes"]
                await connection.execute(
                    "UPDATE ingestion_storage_usage SET used_bytes=$1 WHERE singleton=TRUE",
                    used_after,
                )
                await connection.execute(
                    "UPDATE gods_mlops_artifact_reservations SET state='settled',settled_at=now() WHERE job_id=$1::uuid",
                    job_id,
                )
                await connection.execute(
                    """INSERT INTO gods_mlops_job_events(job_id,event_type,state,details)
                       VALUES($1::uuid,'artifact_reservation_settled',$2,$3::jsonb)""",
                    job_id,
                    job["state"],
                    _canonical_json({
                        "reserved_bytes": reservation["reserved_bytes"],
                        "consumed_bytes": reservation["consumed_bytes"],
                        "released_bytes": reservation["reserved_bytes"] - reservation["consumed_bytes"],
                    }),
                )
                return _job_dict(job)

    async def record_probe_measurement(
        self,
        *,
        job_id: str,
        lease_token: str,
        exit_code: int,
        peak_allocated_mib: int | None,
        peak_reserved_mib: int | None,
        optimizer_steps: int,
        checkpoint_resumed: bool,
        checkpoint_sha256: str | None,
        inference_steps: int = 0,
        verification_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Promote only successful, resumed DETR/CLIP probes into training profiles."""
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                job, lease = await self._lock_job_then_lease(
                    connection, job_id=job_id, lease_token=lease_token
                )
                if (
                    job is None
                    or lease is None
                    or str(job["lease_token"]) != str(lease["lease_token"])
                    or job["phase"] != "probe"
                ):
                    raise ValueError("probe measurement does not own a current probe lease")
                candidate = await connection.fetchrow(
                    """
                    SELECT * FROM gods_mlops_resource_profiles
                    WHERE phase = 'probe' AND model_kind = $1 AND config_version = $2
                    FOR UPDATE
                    """,
                    job["model_kind"],
                    job["config_version"],
                )
                if candidate is None or candidate["profile_state"] != "candidate":
                    raise ResourceProfileConflictError("probe is not backed by an immutable candidate profile")
                target_phase = candidate["target_phase"]
                if target_phase == "evaluation":
                    checkpoint_source_error = await _evaluation_probe_checkpoint_source_error(connection, job)
                    if checkpoint_source_error:
                        raise ValueError("evaluation probe checkpoint source changed before measurement commit")
                measurement_id = uuid4()
                details = verification_details or {}
                memory_measurement_ok = (
                    peak_allocated_mib is not None
                    and peak_reserved_mib is not None
                    and peak_allocated_mib > 0
                    and peak_reserved_mib >= peak_allocated_mib
                    and peak_reserved_mib <= candidate["memory_requirement_mib"]
                )
                training_probe_ok = (
                    target_phase == "training"
                    and job["model_kind"] in {"detr", "clip"}
                    and optimizer_steps >= 3
                    and checkpoint_resumed is True
                    and checkpoint_sha256 is not None
                    and job["checkpoint_sha256"] is not None
                    and job["checkpoint_sha256"].strip() == checkpoint_sha256
                )
                inference_probe_ok = target_phase in {"preparation", "evaluation"} and inference_steps >= 1
                contract_ok = (
                    exit_code == 0
                    and memory_measurement_ok
                    and details.get("passed") is True
                    and (training_probe_ok or inference_probe_ok)
                )
                reason = None
                if exit_code != 0:
                    reason = "probe_execution_failed"
                elif details.get("passed") is not True:
                    reason = "probe_verification_not_successful"
                elif target_phase in {"preparation", "evaluation"} and inference_steps < 1:
                    reason = "probe_contract_not_met"
                elif target_phase == "training" and (
                    job["model_kind"] not in {"detr", "clip"}
                    or optimizer_steps < 3
                    or not checkpoint_resumed
                    or checkpoint_sha256 is None
                    or job["checkpoint_sha256"] is None
                    or job["checkpoint_sha256"].strip() != checkpoint_sha256
                ):
                    reason = "probe_contract_not_met"
                elif not contract_ok:
                    reason = "probe_contract_not_met"
                result_state = "succeeded" if contract_ok else "failed"
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_profile_measurements (
                        measurement_id, job_id, input_sha256, config_sha256, model_kind,
                        target_phase, result_state, peak_allocated_mib, peak_reserved_mib,
                        optimizer_steps, inference_steps, checkpoint_resumed,
                        verification_details, checkpoint_sha256, exit_code
                    ) VALUES ($1, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                        $12, $13::jsonb, $14, $15)
                    """,
                    measurement_id,
                    job_id,
                    job["input_sha256"].strip(),
                    job["config_sha256"].strip(),
                    job["model_kind"],
                    target_phase,
                    result_state,
                    peak_allocated_mib,
                    peak_reserved_mib,
                    optimizer_steps,
                    inference_steps,
                    checkpoint_resumed,
                    _canonical_json(details),
                    checkpoint_sha256,
                    exit_code,
                )
                if not contract_ok:
                    await connection.execute(
                        """
                        UPDATE gods_mlops_jobs SET state = 'failed', reason_code = $2,
                            reason_detail = $3::jsonb, retryable = FALSE,
                            completed_at = now(), updated_at = now()
                        WHERE job_id = $1::uuid AND lease_token = $4::uuid
                        """,
                        job_id,
                        reason,
                        _canonical_json({
                            "exit_code": exit_code,
                            "optimizer_steps": optimizer_steps,
                            "inference_steps": inference_steps,
                            "checkpoint_resumed": checkpoint_resumed,
                            "verification_details": details,
                        }),
                        lease_token,
                    )
                    await connection.execute(
                        """
                        INSERT INTO gods_mlops_job_events (
                            job_id, event_type, state, reason_code, fencing_token, details
                        ) VALUES ($1::uuid, 'probe_measurement_failed', 'failed', $2, $3, $4::jsonb)
                        """,
                        job_id,
                        reason,
                        lease["fencing_token"],
                        _canonical_json({"measurement_id": str(measurement_id)}),
                    )
                    return {
                        "result_state": result_state,
                        "profile_state": "candidate",
                        "reason_code": reason,
                        "measurement_id": str(measurement_id),
                    }

                target_profile = await connection.fetchrow(
                    """
                    SELECT * FROM gods_mlops_resource_profiles
                    WHERE phase = $1 AND model_kind = $2 AND config_version = $3
                    FOR UPDATE
                    """,
                    target_phase,
                    job["model_kind"],
                    job["config_version"],
                )
                if target_profile is not None:
                    if (
                        target_profile["profile_state"] != "candidate"
                        or target_profile["config_sha256"].strip() != job["config_sha256"].strip()
                        or target_profile["artifact_reservation_bytes"] != candidate["artifact_reservation_bytes"]
                    ):
                        raise ResourceProfileConflictError(
                            "measured probe cannot replace a different immutable execution profile"
                        )
                    alternatives = target_profile["oom_alternatives"]
                else:
                    alternatives = candidate["oom_alternatives"]
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_resource_profiles (
                        phase, target_phase, model_kind, config_version, config_sha256,
                        memory_requirement_mib, artifact_reservation_bytes,
                        checkpoint_reservation_bytes, result_reservation_bytes,
                        config_json, profile_state, measurement_id, oom_alternatives
                    ) VALUES ($1, NULL, $2, $3, $4, $5, $6, $7, $8,
                        $9::jsonb, 'measured', $10, $11::jsonb)
                    ON CONFLICT (phase, model_kind, config_version) DO UPDATE SET
                        memory_requirement_mib = EXCLUDED.memory_requirement_mib,
                        profile_state = 'measured', measurement_id = EXCLUDED.measurement_id
                    """,
                    target_phase,
                    job["model_kind"],
                    job["config_version"],
                    job["config_sha256"].strip(),
                    peak_reserved_mib,
                    candidate["artifact_reservation_bytes"],
                    candidate["checkpoint_reservation_bytes"],
                    candidate["result_reservation_bytes"],
                    _canonical_json(_json_value(candidate["config_json"])),
                    measurement_id,
                    json.dumps(_json_value(alternatives)),
                )
                await connection.execute(
                    """
                    UPDATE gods_mlops_jobs SET state = 'completed', reason_code = NULL,
                        reason_detail = '{}'::jsonb, retryable = FALSE,
                        completed_at = now(), updated_at = now()
                    WHERE job_id = $1::uuid AND lease_token = $2::uuid
                    """,
                    job_id,
                    lease_token,
                )
                await connection.execute(
                    """
                    UPDATE gods_mlops_jobs SET profile_state_snapshot = 'measured',
                        state = CASE WHEN state = 'waiting_profile' THEN 'queued' ELSE state END,
                        reason_code = CASE WHEN state = 'waiting_profile' THEN NULL ELSE reason_code END,
                        retryable = CASE WHEN state = 'waiting_profile' THEN FALSE ELSE retryable END,
                        updated_at = now()
                    WHERE phase = $1 AND model_kind = $2 AND config_version = $3
                      AND state IN ('queued', 'waiting_profile')
                    """,
                    target_phase,
                    job["model_kind"],
                    job["config_version"],
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, fencing_token, details
                    ) VALUES ($1::uuid, 'probe_measurement_succeeded', 'completed', $2, $3::jsonb)
                    """,
                    job_id,
                    lease["fencing_token"],
                    _canonical_json({
                        "measurement_id": str(measurement_id),
                        "peak_allocated_mib": peak_allocated_mib,
                        "peak_reserved_mib": peak_reserved_mib,
                        "optimizer_steps": optimizer_steps,
                        "inference_steps": inference_steps,
                        "checkpoint_sha256": checkpoint_sha256,
                        "verification_details": details,
                    }),
                )
                return {
                    "result_state": result_state,
                    "profile_state": "measured",
                    "measurement_id": str(measurement_id),
                    "memory_requirement_mib": peak_reserved_mib,
                    "phase": target_phase,
                }

    async def acquire_gpu_lease(
        self,
        *,
        job_id: str,
        observation: ResourceObservation,
        profile: dict[str, Any],
        source_registry: DatasetSourceRegistry,
        now,
        lease_seconds: int,
        min_idle_seconds: int,
        min_idle_observations: int,
        min_filesystem_bytes: int,
        safety_mib: int,
    ) -> dict[str, Any]:
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtext($1))", observation.gpu_uuid
                )
                current_observation = await connection.fetchrow(
                    """
                    SELECT observation_id, observation, idle_since, idle_observation_count
                    FROM gods_mlops_gpu_observation_current
                    WHERE node_id = $1 FOR UPDATE
                    """,
                    observation.node_id,
                )
                if (
                    current_observation is None
                    or str(current_observation["observation_id"]) != observation.observation_id
                ):
                    return await self._wait_in_transaction(
                        connection,
                        job_id,
                        "waiting_gpu",
                        "ubuntu_observation_changed_before_lease",
                        observation.observation_id,
                    )
                persisted = ResourceObservation.from_dict(_json_value(current_observation["observation"]))
                if persisted.gpu_processes:
                    return await self._wait_in_transaction(
                        connection,
                        job_id,
                        "waiting_gpu",
                        "external_gpu_processes",
                        observation.observation_id,
                        {"external_pids": sorted(item.pid for item in persisted.gpu_processes)},
                    )
                if persisted.filesystem_available_bytes < min_filesystem_bytes:
                    return await self._wait_in_transaction(
                        connection,
                        job_id,
                        "waiting_storage",
                        "ubuntu_filesystem_headroom_below_minimum",
                        observation.observation_id,
                    )
                if persisted.free_mib < profile["memory_requirement_mib"] + safety_mib:
                    return await self._wait_in_transaction(
                        connection,
                        job_id,
                        "waiting_gpu",
                        "measured_profile_exceeds_current_free_memory",
                        observation.observation_id,
                        {
                            "required_mib": profile["memory_requirement_mib"],
                            "safety_mib": safety_mib,
                            "observed_free_mib": persisted.free_mib,
                        },
                    )
                idle_since = current_observation["idle_since"]
                idle_count = current_observation["idle_observation_count"]
                if (
                    idle_since is None
                    or (now - idle_since).total_seconds() < min_idle_seconds
                    or idle_count < min_idle_observations
                ):
                    return await self._wait_in_transaction(
                        connection,
                        job_id,
                        "waiting_gpu",
                        "idle_observation_window",
                        observation.observation_id,
                    )
                job = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid FOR UPDATE",
                    job_id,
                )
                if job is None:
                    raise KeyError(f"GPU job {job_id} does not exist")
                queue_head = await connection.fetchval(
                    """
                    SELECT job_id FROM gods_mlops_jobs
                    WHERE state IN ('queued', 'waiting_gpu', 'waiting_storage', 'waiting_capacity')
                      AND (profile_state_snapshot = 'measured' OR phase = 'probe')
                    ORDER BY queue_order ASC NULLS LAST, created_at, job_id
                    LIMIT 1
                    """
                )
                if queue_head is None or str(queue_head) != job_id:
                    return await self._wait_in_transaction(
                        connection,
                        job_id,
                        "waiting_gpu",
                        "fifo_queue_position",
                        observation.observation_id,
                    )
                existing = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_gpu_leases WHERE gpu_uuid = $1 FOR UPDATE",
                    observation.gpu_uuid,
                )
                if existing is not None:
                    return await self._wait_in_transaction(
                        connection,
                        job_id,
                        "waiting_gpu",
                        "gpu_lease_owned",
                        observation.observation_id,
                        {"owner_job_id": str(existing["job_id"]), "expires_at": existing["expires_at"].isoformat()},
                    )
                # Admission follows the shared-ledger-before-source lock order used by
                # the Task 4/5 quota writers. The dataset row stays SHARE-locked until
                # the lease and reservation commit, so sample invalidation cannot pass
                # this check and commit ahead of the lease.
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                if usage is None:
                    raise RuntimeError("the shared object-storage ledger is not initialized")
                if job["phase"] in {"training", "evaluation"}:
                    source_reasons = await _phase_source_block_reasons_in_transaction(
                        source_registry,
                        connection,
                        phase=job["phase"],
                        dataset_version=job["dataset_version"],
                        model_kind=job["model_kind"],
                    )
                    if job["phase"] == "evaluation":
                        checkpoint_source_error = await _evaluation_checkpoint_source_error(connection, job)
                        if checkpoint_source_error:
                            source_reasons = tuple(sorted(set([*source_reasons, checkpoint_source_error])))
                    if source_reasons:
                        evaluation = job["phase"] == "evaluation"
                        priority = (
                            (
                                "evaluation_checkpoint_source_invalid",
                                "source_sample_explicitly_invalidated",
                                "dataset_source_unavailable",
                                "dataset_not_evaluation_eligible",
                                "late_cross_boundary_link",
                            )
                            if evaluation
                            else (
                                "source_sample_explicitly_invalidated",
                                "dataset_source_unavailable",
                                "dataset_not_training_ready",
                            )
                        )
                        reason = next((item for item in priority if item in source_reasons), source_reasons[0])
                        source_reasons_key = "evaluation_block_reasons" if evaluation else "training_block_reasons"
                        updated = await connection.fetchrow(
                            """
                            UPDATE gods_mlops_jobs SET state='failed', reason_code=$2,
                                reason_detail=$3::jsonb, retryable=FALSE,
                                completed_at=now(), updated_at=now()
                            WHERE job_id=$1::uuid AND lease_token IS NULL
                            RETURNING *
                            """,
                            job_id,
                            reason,
                            _canonical_json({source_reasons_key: list(source_reasons)}),
                        )
                        if updated is None:
                            current_job = await connection.fetchrow(
                                "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid", job_id
                            )
                            if current_job is None:
                                raise KeyError(f"GPU job {job_id} does not exist")
                            return _job_dict(current_job)
                        await connection.execute(
                            """
                            INSERT INTO gods_mlops_job_events(job_id,event_type,state,reason_code,details)
                            VALUES($1::uuid,'source_readiness_failed','failed',$2,$3::jsonb)
                            """,
                            job_id,
                            reason,
                            _canonical_json({source_reasons_key: list(source_reasons), "lease_not_granted": True}),
                        )
                        return _job_dict(updated)
                if job["phase"] == "probe" and job["target_phase"] == "evaluation":
                    checkpoint_source_error = await _evaluation_probe_checkpoint_source_error(connection, job)
                    if checkpoint_source_error:
                        updated = await connection.fetchrow(
                            """UPDATE gods_mlops_jobs SET state='failed', reason_code=$2,
                                      reason_detail=$3::jsonb, retryable=FALSE,
                                      completed_at=now(), updated_at=now()
                               WHERE job_id=$1::uuid AND lease_token IS NULL
                               RETURNING *""",
                            job_id,
                            checkpoint_source_error,
                            _canonical_json({
                                "evaluation_probe_checkpoint_source_error": checkpoint_source_error
                            }),
                        )
                        if updated is not None:
                            await connection.execute(
                                """INSERT INTO gods_mlops_job_events(
                                       job_id,event_type,state,reason_code,details
                                   ) VALUES($1::uuid,'source_readiness_failed','failed',$2,$3::jsonb)""",
                                job_id,
                                checkpoint_source_error,
                                _canonical_json({
                                    "evaluation_probe_checkpoint_source_error": checkpoint_source_error,
                                    "lease_not_granted": True,
                                }),
                            )
                            return _job_dict(updated)
                reservation = await connection.fetchrow(
                    "SELECT * FROM gods_mlops_artifact_reservations WHERE job_id = $1::uuid FOR UPDATE",
                    job_id,
                )
                if reservation is None:
                    amount = profile["artifact_reservation_bytes"]
                    if usage["used_bytes"] + amount > 1024**4:
                        return await self._wait_in_transaction(
                            connection,
                            job_id,
                            "waiting_capacity",
                            "shared_artifact_capacity_unavailable",
                            observation.observation_id,
                            {"requested_bytes": amount, "used_bytes": usage["used_bytes"]},
                        )
                    await connection.execute(
                        "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                        amount,
                    )
                    await connection.execute(
                        """
                        INSERT INTO gods_mlops_artifact_reservations (
                            job_id, config_version, reserved_bytes, consumed_bytes, state
                        ) VALUES ($1::uuid, $2, $3, 0, 'reserved')
                        """,
                        job_id,
                        profile["config_version"],
                        amount,
                    )
                elif (
                    reservation["state"] != "reserved"
                    or reservation["config_version"] != profile["config_version"]
                    or reservation["reserved_bytes"] != profile["artifact_reservation_bytes"]
                ):
                    raise RuntimeError("job artifact reservation does not match its immutable profile")

                lease_token = uuid4()
                generation = int(job["lease_generation"]) + 1
                expires_at = now + __import__("datetime").timedelta(seconds=lease_seconds)
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_gpu_leases (
                        gpu_uuid, job_id, lease_token, fencing_token, memory_requirement_mib,
                        expires_at, granted_observation_id
                    ) VALUES ($1, $2::uuid, $3, $4, $5, $6, $7::uuid)
                    """,
                    observation.gpu_uuid,
                    job_id,
                    lease_token,
                    generation,
                    profile["memory_requirement_mib"],
                    expires_at,
                    observation.observation_id,
                )
                await connection.execute(
                    """
                    UPDATE gods_mlops_jobs SET state = 'running', reason_code = NULL,
                        reason_detail = '{}'::jsonb, retryable = FALSE,
                        lease_token = $2, lease_generation = $3, lease_expires_at = $4,
                        next_retry_at = NULL,
                        owner_pid = NULL, owner_start_ticks = NULL, owner_uid = NULL,
                        updated_at = now()
                    WHERE job_id = $1::uuid
                    """,
                    job_id,
                    lease_token,
                    generation,
                    expires_at,
                )
                await connection.execute(
                    """
                    INSERT INTO gods_mlops_job_events (
                        job_id, event_type, state, observation_id, fencing_token, details
                    ) VALUES ($1::uuid, 'lease_acquired', 'running', $2::uuid, $3, $4::jsonb)
                    """,
                    job_id,
                    observation.observation_id,
                    generation,
                    _canonical_json({
                        "gpu_uuid": observation.gpu_uuid,
                        "memory_requirement_mib": profile["memory_requirement_mib"],
                        "artifact_reservation_bytes": profile["artifact_reservation_bytes"],
                    }),
                )
        return await self.get_job(job_id)

    async def _wait_in_transaction(
        self,
        connection: asyncpg.Connection,
        job_id: str,
        state: str,
        reason_code: str,
        observation_id: str | None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        updated = await connection.fetchrow(
            """
            UPDATE gods_mlops_jobs SET state = $2, reason_code = $3,
                reason_detail = $4::jsonb, retryable = TRUE, updated_at = now()
            WHERE job_id = $1::uuid AND state IN (
                'queued', 'waiting_profile', 'waiting_gpu', 'waiting_storage',
                'waiting_capacity', 'yield_requested'
            )
              AND lease_token IS NULL
            RETURNING job_id
            """,
            job_id,
            state,
            reason_code,
            _canonical_json(details or {}),
        )
        if updated is None:
            row = await connection.fetchrow("SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid", job_id)
            if row is None:
                raise KeyError(f"GPU job {job_id} does not exist")
            return _job_dict(row)
        await connection.execute(
            """
            INSERT INTO gods_mlops_job_events (
                job_id, event_type, state, reason_code, observation_id, details
            ) VALUES ($1::uuid, 'admission_wait', $2, $3, $4::uuid, $5::jsonb)
            """,
            job_id,
            state,
            reason_code,
            observation_id,
            _canonical_json(details or {}),
        )
        row = await connection.fetchrow("SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid", job_id)
        return _job_dict(row)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._database_url, min_size=1, max_size=8)
        return self._pool


class JobQueue:
    """Validate phase input, then submit an immutable, idempotent queue record."""

    def __init__(
        self,
        *,
        repository: PostgresJobQueueRepository,
        sources: DatasetSourceRegistry,
    ) -> None:
        self._repository = repository
        self._sources = sources

    async def register_profile(self, profile: ExecutionProfile) -> None:
        await self._repository.register_profile(profile)

    async def submit(
        self,
        dataset_version: str,
        model_kind: str,
        config_version: str,
        rerun: bool = False,
        *,
        parent_job_id: str | None = None,
        retry_root_id: str | None = None,
        operator_id: str | None = None,
        intent_token: str | None = None,
    ) -> str:
        """Queue a published, current, model-compatible training input."""
        if model_kind not in {"detr", "clip"}:
            raise DatasetNotReadyForTrainingError("caption inference is not a training target")
        try:
            source: DatasetTrainingSource = await self._sources.get_training_source(dataset_version)
        except DatasetSourceUnavailableError:
            raise
        reasons = source.training_block_reasons()
        if source.target not in {model_kind, "both"}:
            reasons = tuple(sorted(set([*reasons, f"dataset_target_not_{model_kind}"])))
        if reasons:
            raise DatasetNotReadyForTrainingError(reasons)
        profile = await self._repository.get_profile(
            phase="training", model_kind=model_kind, config_version=config_version
        )
        if profile is None:
            raise ResourceProfileNotFoundError("training config has no versioned resource profile")
        operator_dedupe_key = _operator_retry_key_for_enqueue(
            rerun=rerun,
            parent_job_id=parent_job_id,
            operator_id=operator_id,
            intent_token=intent_token,
        )
        return await self._repository.enqueue(
            phase="training",
            input_kind="dataset_version",
            input_id=source.dataset_version,
            input_sha256=source.manifest_sha256,
            dataset_version=source.dataset_version,
            source_refs={
                "dataset_version": source.dataset_version,
                "manifest_sha256": source.manifest_sha256,
                "target": source.target,
            },
            model_kind=model_kind,
            profile=profile,
            rerun=rerun,
            parent_job_id=parent_job_id,
            retry_root_id=retry_root_id,
            operator_dedupe_key=operator_dedupe_key,
        )

    async def submit_evaluation(
        self,
        *,
        dataset_version: str,
        model_kind: str,
        config_version: str,
        checkpoint_source,
        evaluation_split: str = "test",
        baseline_metadata: dict[str, str | None] | None = None,
        rerun: bool = False,
        parent_job_id: str | None = None,
        retry_root_id: str | None = None,
        operator_id: str | None = None,
        intent_token: str | None = None,
    ) -> str:
        """Queue evaluation against a prior checkpoint with current source checks."""
        from gods_mlops.jobs.models import EvaluationCheckpointSource

        if model_kind not in {"detr", "clip"}:
            raise DatasetNotReadyForEvaluationError("evaluation supports DETR or CLIP checkpoints")
        if evaluation_split not in {"validation", "test"}:
            raise ValueError("evaluation_split must be validation or test")
        if not isinstance(checkpoint_source, EvaluationCheckpointSource):
            raise ValueError("evaluation requires a typed immutable prior-checkpoint source")
        if (
            checkpoint_source.dataset_version != dataset_version
            or checkpoint_source.model_kind != model_kind
        ):
            raise ValueError("evaluation checkpoint source differs from the requested dataset or model")
        baseline = None
        if baseline_metadata is not None:
            if not isinstance(baseline_metadata, dict) or set(baseline_metadata) - {"model_id", "revision"}:
                raise ValueError("baseline metadata must contain only model_id and revision")
            model_id = baseline_metadata.get("model_id")
            revision = baseline_metadata.get("revision")
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError("baseline model_id metadata must be non-empty")
            if revision is not None and (not isinstance(revision, str) or not revision.strip()):
                raise ValueError("baseline revision metadata must be non-empty")
            baseline = {"model_id": model_id, "revision": revision, "verified": False}
        try:
            source: DatasetTrainingSource = await self._sources.get_training_source(dataset_version)
        except DatasetSourceUnavailableError:
            raise
        reasons = source.evaluation_block_reasons()
        if source.target not in {model_kind, "both"}:
            reasons = tuple(sorted(set([*reasons, f"dataset_target_not_{model_kind}"])))
        if checkpoint_source.training_manifest_sha256 != source.manifest_sha256:
            reasons = tuple(sorted(set([*reasons, "checkpoint_training_manifest_mismatch"])))
        if reasons:
            raise DatasetNotReadyForEvaluationError(reasons)
        profile = await self._repository.get_profile(
            phase="evaluation", model_kind=model_kind, config_version=config_version
        )
        if profile is None:
            raise ResourceProfileNotFoundError("evaluation config has no phase-specific resource profile")
        if profile.get("profile_state") != "measured":
            raise ResourceProfileConflictError("evaluation jobs require a successfully measured phase-specific profile")
        operator_dedupe_key = _operator_retry_key_for_enqueue(
            rerun=rerun,
            parent_job_id=parent_job_id,
            operator_id=operator_id,
            intent_token=intent_token,
        )
        source_refs = {
            "schema": "gods-mlops-evaluation-source-v1",
            "dataset_version": source.dataset_version,
            "manifest_sha256": source.manifest_sha256,
            "target": source.target,
            "evaluation_split": evaluation_split,
            "checkpoint": checkpoint_source.as_dict(),
        }
        if baseline is not None:
            source_refs["baseline"] = baseline
        stable_source_identity = {
            "checkpoint": checkpoint_source.as_dict(),
            "evaluation_split": evaluation_split,
        }
        if baseline is not None:
            stable_source_identity["baseline"] = baseline
        return await self._repository.enqueue(
            phase="evaluation",
            input_kind="dataset_version",
            input_id=source.dataset_version,
            input_sha256=source.manifest_sha256,
            dataset_version=source.dataset_version,
            source_refs=source_refs,
            model_kind=model_kind,
            profile=profile,
            rerun=rerun,
            parent_job_id=parent_job_id,
            retry_root_id=retry_root_id,
            stable_source_identity=stable_source_identity,
            operator_dedupe_key=operator_dedupe_key,
        )

    async def submit_probe(
        self,
        *,
        model_kind: str,
        config_version: str,
        probe_input_id: str | None = None,
        input_sha256: str | None = None,
        probe_input: ProbeInput | None = None,
        rerun: bool = False,
    ) -> str:
        """Queue an immutable pre-publication resource probe on the shared queue."""
        if model_kind not in {"detr", "clip", "qwen"}:
            raise ValueError("probe model_kind must be detr, clip, or qwen")
        if probe_input is not None:
            if probe_input.model_kind != model_kind:
                raise ValueError("probe input model kind does not match the requested model")
            if (
                probe_input.target_phase == "evaluation"
                and probe_input.evaluation_checkpoint_source is None
            ):
                raise ValueError("evaluation profile probes require a verified training-probe checkpoint")
            if (
                probe_input.evaluation_checkpoint_source is not None
                and probe_input.evaluation_checkpoint_source.model_kind != model_kind
            ):
                raise ValueError("evaluation probe checkpoint model kind does not match the requested model")
            probe_input_id = probe_input.probe_input_id
            input_sha256 = probe_input.input_sha256
            source_refs = probe_input.as_dict()
            stable_source_identity = (
                {
                    "evaluation_checkpoint_source": probe_input.evaluation_checkpoint_source.as_dict()
                }
                if probe_input.evaluation_checkpoint_source is not None
                else None
            )
        else:
            if not isinstance(probe_input_id, str) or not probe_input_id.strip() or len(probe_input_id) > 255:
                raise ValueError("probe_input_id must contain 1 to 255 characters")
            source_refs = {"probe_input_id": probe_input_id, "input_sha256": input_sha256}
            stable_source_identity = None
        if not isinstance(input_sha256, str):
            raise ValueError("probe input SHA-256 is missing")
        _validate_digest(input_sha256, "input_sha256")
        profile = await self._repository.get_profile(
            phase="probe", model_kind=model_kind, config_version=config_version
        )
        if profile is None:
            raise ResourceProfileNotFoundError("probe config has no versioned resource profile")
        if profile["profile_state"] != "candidate":
            raise ValueError("resource probes require an unmeasured candidate profile")
        if probe_input is not None and profile["target_phase"] != probe_input.target_phase:
            raise ValueError("probe input target phase does not match its candidate profile")
        if probe_input is not None and probe_input.evaluation_checkpoint_source is not None:
            from gods_mlops.jobs.models import EvaluationProbeCheckpointSource

            checkpoint_source = EvaluationProbeCheckpointSource.from_dict(
                probe_input.evaluation_checkpoint_source.as_dict()
            )
            profile_config = profile.get("config_json")
            if isinstance(profile_config, str):
                profile_config = json.loads(profile_config)
            if (
                not isinstance(profile_config, dict)
                or checkpoint_source.model_kind != model_kind
                or checkpoint_source.model_id != profile_config.get("model_id")
                or checkpoint_source.model_revision != profile_config.get("model_revision")
            ):
                raise ValueError("evaluation probe checkpoint source differs from its immutable candidate profile")
        return await self._repository.enqueue(
            phase="probe",
            input_kind="probe_input",
            input_id=probe_input_id,
            input_sha256=input_sha256,
            dataset_version=None,
            source_refs=source_refs,
            model_kind=model_kind,
            profile=profile,
            rerun=rerun,
            stable_source_identity=stable_source_identity,
        )

    async def submit_preparation(
        self,
        *,
        batch: AnnotationPreparationBatch,
        model_kind: str,
        config_version: str,
        rerun: bool = False,
        parent_job_id: str | None = None,
        retry_root_id: str | None = None,
        operator_id: str | None = None,
        intent_token: str | None = None,
    ) -> str:
        """Queue GPU draft work from immutable frame/crop refs before publication."""
        if model_kind not in {"detr", "qwen"}:
            raise ValueError("preparation supports only DETR frame drafts and Qwen crop drafts")
        await self._sources.verify_annotation_batch(batch)
        profile = await self._repository.get_profile(
            phase="preparation", model_kind=model_kind, config_version=config_version
        )
        if profile is None:
            raise ResourceProfileNotFoundError("preparation config has no versioned resource profile")
        if profile.get("profile_state") != "measured":
            raise ValueError("preparation jobs require a measured phase-specific resource profile")
        expected_kind = "frame" if model_kind == "detr" else "crop"
        if not batch.items or any(item.item_kind != expected_kind for item in batch.items):
            raise ValueError(f"{model_kind} preparation requires a non-empty batch of {expected_kind} items")
        if model_kind in {"detr", "qwen"}:
            config = profile["config_json"]
            limit_key = "max_draft_frames" if model_kind == "detr" else "max_draft_images"
            limit = config.get(limit_key, 1) if isinstance(config, dict) else None
            if type(limit) is not int or limit < 1:
                raise ValueError(f"{model_kind} preparation profile needs a positive {limit_key} bound")
            if len(batch.items) > limit:
                raise ValueError(
                    f"{model_kind} preparation batch exceeds its versioned {limit_key} bound"
                )
        operator_dedupe_key = _operator_retry_key_for_enqueue(
            rerun=rerun,
            parent_job_id=parent_job_id,
            operator_id=operator_id,
            intent_token=intent_token,
        )
        return await self._repository.enqueue(
            phase="preparation",
            input_kind="annotation_batch",
            input_id=batch.batch_id,
            input_sha256=batch.input_sha256,
            dataset_version=None,
            source_refs=batch.as_dict(),
            model_kind=model_kind,
            profile=profile,
            rerun=rerun,
            parent_job_id=parent_job_id,
            retry_root_id=retry_root_id,
            operator_dedupe_key=operator_dedupe_key,
        )

    async def bind_process(
        self,
        job_id: str,
        lease_token: str,
        owner: ProcessIdentity,
    ) -> bool:
        return await self._repository.bind_lease_process(job_id, lease_token, owner)

    async def request_yield(
        self,
        job_id: str,
        reason: str,
        *,
        after_progress: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Ask the worker to yield now, or arm one exact CLIP evaluation boundary."""
        if after_progress is None:
            await self._repository.request_yield(job_id, reason)
            return None
        result = await self._repository.request_yield(
            job_id,
            reason,
            after_progress=after_progress,
        )
        return result

    async def pending_evaluation_progress_yield(
        self,
        *,
        job_id: str,
        lease_token: str,
        fencing_token: int,
        input_sha256: str,
        config_sha256: str,
        training_source_checkpoint_sha256: str,
    ) -> dict[str, Any] | None:
        return await self._repository.pending_evaluation_progress_yield(
            job_id=job_id,
            lease_token=lease_token,
            fencing_token=fencing_token,
            input_sha256=input_sha256,
            config_sha256=config_sha256,
            training_source_checkpoint_sha256=training_source_checkpoint_sha256,
        )

    async def record_evaluation_progress(
        self,
        *,
        job_id: str,
        lease_token: str,
        fencing_token: int,
        arm_event_id: int,
        progress: dict[str, Any],
        owner: ProcessIdentity,
        artifact_deadline: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return await self._repository.record_evaluation_progress(
            job_id=job_id,
            lease_token=lease_token,
            fencing_token=fencing_token,
            arm_event_id=arm_event_id,
            progress=progress,
            owner=owner,
            artifact_deadline=artifact_deadline,
        )

    async def renew_lease(self, job_id: str, lease_token: str, *, now=None) -> bool:
        from gods_mlops.jobs.admission import GPU_LEASE_SECONDS

        return await self._repository.renew_lease(
            job_id,
            lease_token,
            now=now,
            lease_seconds=GPU_LEASE_SECONDS,
        )

    async def _tracked_s3_artifact_write(
        self,
        *,
        job_id: str,
        lease_token: str,
        operation_id: str,
        store,
        prepared,
        artifact_deadline: dict[str, Any] | None,
    ):
        writer_attempt_id = str(uuid4())
        deadline_kwargs = _artifact_deadline_kwargs(artifact_deadline)
        await self._repository.record_artifact_writer_started(
            job_id=job_id,
            lease_token=lease_token,
            operation_id=operation_id,
            writer_attempt_id=writer_attempt_id,
            **deadline_kwargs,
        )
        loop = asyncio.get_running_loop()
        writer_future = loop.run_in_executor(None, store.commit, prepared)

        async def record_quiescence() -> None:
            try:
                await asyncio.shield(writer_future)
            except asyncio.CancelledError:
                # A cancelled observer cannot prove that the executor thread stopped.
                # Leave the durable start unmatched so cleanup remains blocked.
                raise
            except BaseException:
                pass
            await self._repository.record_artifact_writer_quiescent(
                job_id=job_id,
                operation_id=operation_id,
                writer_attempt_id=writer_attempt_id,
            )

        quiescence_task = asyncio.create_task(record_quiescence())
        _ACTIVE_ARTIFACT_WRITER_WATCHERS.add(quiescence_task)

        def release_watcher(task: asyncio.Task) -> None:
            _ACTIVE_ARTIFACT_WRITER_WATCHERS.discard(task)
            if not task.cancelled():
                task.exception()

        quiescence_task.add_done_callback(release_watcher)
        try:
            verified = await asyncio.shield(writer_future)
        except asyncio.CancelledError:
            raise
        except BaseException:
            await asyncio.shield(quiescence_task)
            raise
        await asyncio.shield(quiescence_task)
        return verified

    async def save_checkpoint(
        self,
        *,
        store,
        job_id: str,
        lease_token: str,
        identity,
        payload: bytes,
        artifact_deadline: dict[str, Any] | None = None,
    ):
        """Write payload bytes, then publish metadata only under the current fence."""
        from gods_mlops.jobs.checkpoints import (
            CheckpointIdentityError,
            StaleCheckpointOwnerError,
        )

        current_job = await self.get(job_id)
        expected = await self._repository.checkpoint_identity(job_id)
        if identity != expected:
            raise CheckpointIdentityError("checkpoint identity does not match the current immutable job input")
        if not await self._repository.lease_is_current(job_id, lease_token):
            raise StaleCheckpointOwnerError("checkpoint writer no longer owns the active GPU lease")
        await self.retry_pending_checkpoint_prunes(store=store, job_id=job_id)
        profile = await self._repository.get_profile(
            phase=current_job["phase"],
            model_kind=current_job["model_kind"],
            config_version=current_job["config_version"],
        )
        if profile is None:
            raise CheckpointIdentityError("checkpoint resource profile is no longer available")
        previous = await self._repository.checkpoint_metadata_for(job_id)
        prepared = await asyncio.to_thread(
            store.prepare,
            identity=identity,
            payload=payload,
            reservation_bytes=profile["checkpoint_reservation_bytes"],
            replacement_reservation_bytes=2 * profile["checkpoint_reservation_bytes"],
            previous_uri=previous["uri"] if previous else None,
            previous_sha256=previous["sha256"] if previous else None,
            previous_size_bytes=previous["size_bytes"] if previous else None,
            previous_metadata_size_bytes=previous["metadata_size_bytes"] if previous else None,
        )
        precharged = getattr(prepared, "object_key", None) is not None
        operation_id = None
        if precharged:
            operation_id = await self._repository.begin_artifact_write(
                job_id=job_id,
                lease_token=lease_token,
                identity=identity,
                prepared=prepared,
                store=store,
                operation="checkpoint",
                source_registry=self._sources,
                **_artifact_deadline_kwargs(artifact_deadline),
            )
        verified_artifact = None
        if precharged and operation_id:
            verified_artifact = await self._tracked_s3_artifact_write(
                job_id=job_id,
                lease_token=lease_token,
                operation_id=operation_id,
                store=store,
                prepared=prepared,
                artifact_deadline=artifact_deadline,
            )
        elif precharged:
            verified_artifact = await asyncio.to_thread(
                store.load_uri,
                _prepared_artifact_uri(prepared, store),
                expected_identity=identity,
                expected_sha256=prepared.sha256,
                expected_size_bytes=prepared.size_bytes,
            )
        verified = await self._repository.commit_checkpoint(
            job_id=job_id,
            lease_token=lease_token,
            identity=identity,
            prepared=prepared,
            store=store,
            source_registry=self._sources,
            operation_id=operation_id,
            precharged=precharged,
            verified_artifact=verified_artifact,
            **_artifact_deadline_kwargs(artifact_deadline),
        )
        if previous is not None and previous["uri"] != _verified_checkpoint_uri(verified):
            await self.retry_pending_checkpoint_prunes(store=store, job_id=job_id)
        return verified

    async def save_result_artifact(
        self,
        *,
        store,
        job_id: str,
        lease_token: str,
        identity,
        kind: str,
        payload: bytes,
        runtime_measurements: dict[str, Any] | None = None,
        artifact_deadline: dict[str, Any] | None = None,
    ):
        """Commit one result bundle under the Task 7 fence and shared reservation."""
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError, StaleCheckpointOwnerError

        measured_snapshot = (
            json.loads(_canonical_json(runtime_measurements))
            if runtime_measurements is not None
            else None
        )
        if measured_snapshot is not None and not isinstance(measured_snapshot, dict):
            raise ValueError("result runtime measurements must be a JSON object")
        current_job = await self.get(job_id)
        expected = await self._repository.checkpoint_identity(job_id)
        if identity != expected:
            raise CheckpointIdentityError("result identity does not match the current immutable job input")
        if not await self._repository.lease_is_current(job_id, lease_token):
            raise StaleCheckpointOwnerError("result writer no longer owns the active GPU lease")
        profile = await self._repository.get_profile(
            phase=current_job["phase"],
            model_kind=current_job["model_kind"],
            config_version=current_job["config_version"],
        )
        if profile is None:
            raise CheckpointIdentityError("result resource profile is no longer available")
        prepared = await asyncio.to_thread(
            store.prepare,
            identity=identity,
            kind=kind,
            payload=payload,
            reservation_bytes=profile["result_reservation_bytes"],
        )
        precharged = getattr(prepared, "object_key", None) is not None
        operation_id = None
        if precharged:
            operation_id = await self._repository.begin_artifact_write(
                job_id=job_id,
                lease_token=lease_token,
                identity=identity,
                prepared=prepared,
                store=store,
                operation="result",
                source_registry=self._sources,
                runtime_measurements=measured_snapshot,
                **_artifact_deadline_kwargs(artifact_deadline),
            )
        try:
            verified_artifact = None
            if precharged and operation_id:
                verified_artifact = await self._tracked_s3_artifact_write(
                    job_id=job_id,
                    lease_token=lease_token,
                    operation_id=operation_id,
                    store=store,
                    prepared=prepared,
                    artifact_deadline=artifact_deadline,
                )
            elif precharged:
                existing_details = {
                    "kind": prepared.kind,
                    "uri": _prepared_artifact_uri(prepared, store),
                    "sha256": prepared.sha256,
                    "size_bytes": prepared.size_bytes,
                    "identity": identity.as_dict(),
                    "object_key": getattr(prepared, "object_key", None),
                }
                verified_artifact, _payload = await asyncio.to_thread(
                    store.read_committed,
                    existing_details,
                    expected_identity=identity,
                )
            return await self._repository.commit_result_artifact(
                job_id=job_id,
                lease_token=lease_token,
                identity=identity,
                prepared=prepared,
                store=store,
                source_registry=self._sources,
                operation_id=operation_id,
                precharged=precharged,
                runtime_measurements=measured_snapshot,
                verified_artifact=verified_artifact,
                **_artifact_deadline_kwargs(artifact_deadline),
            )
        except BaseException:
            discard = getattr(store, "discard", None)
            if not precharged and callable(discard):
                discard(prepared)
            raise

    async def cleanup_pending_artifact_writes(self, *, object_store, job_id: str) -> list[str]:
        """Delete exact, uncommitted S3 writes after terminal lease release and verify absence."""
        pending = await self._repository.pending_artifact_writes_for_cleanup(job_id)
        deleted: list[str] = []
        for details in pending:
            object_key, digest, size_bytes = _validate_pending_artifact_intent(
                job_id=job_id, details=details, object_store=object_store
            )
            absent = False
            try:
                object_store.read_source(
                    object_key=object_key, sha256_digest=digest, size_bytes=size_bytes
                )
            except FileNotFoundError:
                absent = True
            if not absent:
                object_store.delete_object(object_key=object_key)
                try:
                    object_store.read_source(
                        object_key=object_key, sha256_digest=digest, size_bytes=size_bytes
                    )
                except FileNotFoundError:
                    absent = True
                if not absent:
                    raise OSError("pending artifact still exists after exact-object deletion")
            if await self._repository.complete_artifact_write_delete(job_id=job_id, details=details):
                deleted.append(str(details["uri"]))
        return deleted

    async def cleanup_pending_artifact_writes_from_environment(self, job_id: str) -> list[str]:
        """Lazily build the CPU controller's S3 client only when terminal writes need cleanup."""
        if not await self._repository.pending_artifact_writes_for_cleanup(job_id):
            await self._repository.settle_artifact_reservation(job_id)
            return []
        from gods_mlops.training.data import dataset_object_store_from_environment

        deleted = await self.cleanup_pending_artifact_writes(
            object_store=dataset_object_store_from_environment(), job_id=job_id
        )
        await self._repository.settle_artifact_reservation(job_id)
        return deleted

    async def retry_pending_checkpoint_prunes(self, *, store, job_id: str) -> list[str]:
        """Retry exact previous-checkpoint deletion intents after a lost delete acknowledgement."""
        prune_uri = getattr(store, "prune_uri", None)
        if not callable(prune_uri):
            return []
        writer_gate = getattr(self._repository, "checkpoint_prune_writer_gate", None)
        if not callable(writer_gate):
            raise RuntimeError("checkpoint pruning requires the durable writer lifecycle gate")
        completed = []
        for previous in await self._repository.pending_checkpoint_prunes_for(job_id):
            if not await writer_gate(job_id=job_id, previous=previous):
                continue
            await asyncio.to_thread(
                prune_uri,
                previous["uri"],
                sha256_digest=previous["sha256"],
                size_bytes=int(previous["size_bytes"]),
                job_id=job_id,
                identity=previous.get("identity"),
                metadata_size_bytes=int(previous.get("metadata_size_bytes", 0)),
            )
            if await self._repository.complete_checkpoint_prune(job_id=job_id, previous=previous):
                completed.append(previous["uri"])
        return completed

    async def complete_owned_job(
        self,
        *,
        job_id: str,
        lease_token: str,
        identity,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._repository.complete_owned_job(
            job_id=job_id,
            lease_token=lease_token,
            identity=identity,
            details=details,
            source_registry=self._sources,
        )

    async def settle_artifact_reservation(self, job_id: str) -> dict[str, Any] | None:
        return await self._repository.settle_artifact_reservation(job_id)

    async def settle_released_terminal_artifact_reservations(self) -> dict[str, int]:
        """Retry only terminal reservations whose GPU lease release is durable."""
        try:
            job_ids = await self._repository.list_released_terminal_artifact_reservations()
        except _TRANSIENT_RESERVATION_ERRORS:
            return {"examined": 0, "settled": 0, "deferred": 1}

        settled = 0
        deferred = 0
        for job_id in job_ids:
            try:
                await self._repository.settle_artifact_reservation(job_id)
                reservation = await self._repository.artifact_reservation_for(job_id)
            except _TRANSIENT_RESERVATION_ERRORS:
                deferred += 1
                continue
            if reservation is not None and reservation.get("state") == "settled":
                settled += 1
            else:
                # Missing or active-writer rows remain charged and discoverable.
                deferred += 1
        return {"examined": len(job_ids), "settled": settled, "deferred": deferred}

    async def load_checkpoint(self, *, store, job_id: str):
        """Load only the current DB-committed checkpoint for this immutable job identity."""
        from gods_mlops.jobs.checkpoints import CheckpointIntegrityError

        details = await self._repository.checkpoint_metadata_for(job_id)
        if details is None:
            return None
        identity = await self._repository.checkpoint_identity(job_id)
        if identity.as_dict() != details["identity"]:
            raise CheckpointIntegrityError("checkpoint identity changed in the job commit marker")
        if hasattr(store, "load_uri"):
            return await asyncio.to_thread(
                store.load_uri,
                details["uri"],
                expected_identity=identity,
                expected_sha256=details["sha256"],
                expected_size_bytes=details["size_bytes"],
            )
        checkpoint = await asyncio.to_thread(store.load, job_id, expected_identity=identity)
        if checkpoint is None or checkpoint.sha256 != details["sha256"]:
            raise CheckpointIntegrityError("local checkpoint does not match the database commit marker")
        return checkpoint

    async def record_probe_measurement(
        self,
        *,
        job_id: str,
        lease_token: str,
        exit_code: int,
        peak_allocated_mib: int | None,
        peak_reserved_mib: int | None,
        optimizer_steps: int,
        checkpoint_resumed: bool,
        checkpoint_sha256: str | None,
        inference_steps: int = 0,
        verification_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a probe result and promote only its measured runtime profile."""
        result = await self._repository.record_probe_measurement(
            job_id=job_id,
            lease_token=lease_token,
            exit_code=exit_code,
            peak_allocated_mib=peak_allocated_mib,
            peak_reserved_mib=peak_reserved_mib,
            optimizer_steps=optimizer_steps,
            checkpoint_resumed=checkpoint_resumed,
            checkpoint_sha256=checkpoint_sha256,
            inference_steps=inference_steps,
            verification_details=verification_details,
        )
        if result.get("result_state") in {"succeeded", "failed"}:
            await self._repository.settle_artifact_reservation(job_id)
        return result

    async def record_communication_failure(
        self,
        *,
        job_id: str,
        error_code: str,
        now=None,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        from datetime import UTC, datetime

        retry_at = now or datetime.now(UTC)
        if retry_at.tzinfo is None:
            raise ValueError("communication retry time must be timezone-aware")
        if not error_code.strip() or len(error_code) > 128:
            raise ValueError("communication error code must contain 1 to 128 characters")
        return await self._repository.schedule_communication_retry(
            job_id=job_id,
            error_code=error_code,
            now=retry_at,
            lease_token=lease_token,
        )

    async def record_oom(self, job_id: str, lease_token: str) -> dict[str, Any]:
        """Record OOM and start only the next measured smaller config attempt."""
        return await self._repository.record_oom(job_id=job_id, lease_token=lease_token)

    async def get(self, job_id: str) -> dict[str, Any]:
        return await self._repository.get_job(job_id)

    async def list_jobs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return await self._repository.list_jobs(limit=limit)

    async def retry_intent_generation(self, scope_sha256: str, *, session_expires_at: datetime) -> int:
        return await self._repository.get_operator_retry_intent_generation(
            scope_sha256,
            session_expires_at=session_expires_at,
        )

    async def advance_retry_intent_generation(
        self,
        scope_sha256: str,
        *,
        session_expires_at: datetime,
    ) -> int:
        return await self._repository.advance_operator_retry_intent_generation(
            scope_sha256,
            expires_at=session_expires_at,
        )

    async def get_job_detail(self, job_id: str) -> dict[str, Any]:
        job = await self.get(job_id)
        try:
            reservation = await self._repository.artifact_reservation_for(job_id)
        except Exception:  # noqa: BLE001 - unknown cleanup state remains visible after a failed readback
            reservation = {"state": "unknown"}
        return {
            **job,
            "reservation_cleanup": _reservation_cleanup_state(reservation),
            "checkpoint": await self._repository.checkpoint_metadata_for(job_id),
            "result_artifacts": await self._repository.result_artifacts_for(job_id),
            "profile_measurement": await self._repository.profile_measurement_for_job(job_id),
            "review_handoffs": await self._repository.review_handoffs_for(job_id),
        }

    async def cancel_queued(self, job_id: str, *, operator_id: str) -> dict[str, Any]:
        actor_sha256 = _operator_actor_sha256(operator_id)
        job = await self._repository.cancel_queued_job(job_id, actor_sha256=actor_sha256)
        settlement_error_type = None
        try:
            await self._repository.settle_artifact_reservation(job_id)
        except Exception as error:  # noqa: BLE001 - expose pending settlement without leaking backend details
            settlement_error_type = type(error).__name__
        try:
            reservation = await self._repository.artifact_reservation_for(job_id)
        except Exception as error:  # noqa: BLE001 - failed readback cannot prove settlement
            reservation = {"state": "unknown"}
            settlement_error_type = settlement_error_type or type(error).__name__
        cleanup_status = _reservation_cleanup_state(reservation)
        return {
            "job": await self.get(job_id),
            "reservation_cleanup": cleanup_status,
            "cleanup_pending": cleanup_status in {"pending", "unknown"},
            "settlement_error_type": settlement_error_type,
        }

    async def reorder_queued(
        self,
        job_id: str,
        *,
        before_job_id: str | None,
        operator_id: str,
    ) -> list[str]:
        return await self._repository.reorder_queued_jobs(
            job_id,
            before_job_id=before_job_id,
            actor_sha256=_operator_actor_sha256(operator_id),
        )

    async def retry_job(self, job_id: str, *, operator_id: str, intent_token: str) -> str:
        """Rerun a terminal public job through its original validation path."""
        parent = await self.get(job_id)
        if parent.get("phase") == "probe":
            raise ValueError("operator retry is not supported for probe jobs")
        if parent.get("state") not in {"completed", "failed", "cancelled"}:
            raise ValueError("only completed, failed, or cancelled jobs can be rerun")
        if parent.get("lease_token") is not None or parent.get("owner_pid") is not None:
            raise ValueError("jobs with a lease or process owner cannot be rerun")

        parent_job_id = str(UUID(str(parent["job_id"])))
        retry_root_id = str(UUID(str(parent.get("retry_root_id") or parent_job_id)))
        common = {
            "parent_job_id": parent_job_id,
            "retry_root_id": retry_root_id,
            "operator_id": operator_id,
            "intent_token": intent_token,
            "rerun": True,
        }
        phase = parent.get("phase")
        if phase == "training":
            return await self.submit(
                dataset_version=str(parent["dataset_version"]),
                model_kind=str(parent["model_kind"]),
                config_version=str(parent["config_version"]),
                **common,
            )
        if phase == "evaluation":
            from gods_mlops.jobs.models import EvaluationCheckpointSource

            refs = _json_value(parent.get("source_refs"))
            if not isinstance(refs, dict):
                raise ValueError("evaluation job source references are unavailable")
            checkpoint_source = EvaluationCheckpointSource.from_dict(refs.get("checkpoint"))
            baseline = refs.get("baseline")
            baseline_metadata = None
            if baseline is not None:
                if not isinstance(baseline, dict):
                    raise ValueError("evaluation baseline source is invalid")
                baseline_metadata = {
                    "model_id": baseline.get("model_id"),
                    "revision": baseline.get("revision"),
                }
            return await self.submit_evaluation(
                dataset_version=str(parent["dataset_version"]),
                model_kind=str(parent["model_kind"]),
                config_version=str(parent["config_version"]),
                checkpoint_source=checkpoint_source,
                evaluation_split=str(refs.get("evaluation_split", "test")),
                baseline_metadata=baseline_metadata,
                **common,
            )
        if phase == "preparation":
            from gods_mlops.jobs.models import AnnotationPreparationBatch, ImmutableAnnotationItem

            refs = _json_value(parent.get("source_refs"))
            if not isinstance(refs, dict) or not isinstance(refs.get("items"), list):
                raise ValueError("preparation job source references are unavailable")
            try:
                batch = AnnotationPreparationBatch(
                    batch_id=str(refs["batch_id"]),
                    input_sha256=str(refs["input_sha256"]),
                    items=tuple(
                        ImmutableAnnotationItem(
                            item_kind=str(item["item_kind"]),
                            item_id=str(item["item_id"]),
                            sample_id=str(item["sample_id"]),
                            sha256=str(item["sha256"]),
                            object_key=str(item["object_key"]),
                            object_size_bytes=int(item["object_size_bytes"]),
                            revision_id=(
                                str(item["revision_id"]) if item.get("revision_id") is not None else None
                            ),
                        )
                        for item in refs["items"]
                    ),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("preparation job source references are malformed") from error
            return await self.submit_preparation(
                batch=batch,
                model_kind=str(parent["model_kind"]),
                config_version=str(parent["config_version"]),
                **common,
            )
        raise ValueError("operator retry is not supported for this job phase")

    async def training_source_block_reasons(self, job_id: str) -> tuple[str, ...]:
        job = await self.get(job_id)
        if job["phase"] != "training":
            return ()
        try:
            source = await self._sources.get_training_source(job["dataset_version"])
        except DatasetSourceUnavailableError:
            return ("dataset_source_unavailable",)
        reasons = set(source.training_block_reasons())
        if source.target not in {job["model_kind"], "both"}:
            reasons.add(f"dataset_target_not_{job['model_kind']}")
        return tuple(sorted(reasons))

    async def evaluation_source_block_reasons(self, job_id: str) -> tuple[str, ...]:
        job = await self.get(job_id)
        if job["phase"] != "evaluation":
            return ()
        try:
            source = await self._sources.get_training_source(job["dataset_version"])
        except DatasetSourceUnavailableError:
            return ("dataset_source_unavailable",)
        reasons = set(source.evaluation_block_reasons())
        if source.target not in {job["model_kind"], "both"}:
            reasons.add(f"dataset_target_not_{job['model_kind']}")
        return tuple(sorted(reasons))

    async def evaluation_probe_source_block_reasons(self, job_id: str) -> tuple[str, ...]:
        return await self._repository.evaluation_probe_source_block_reasons(job_id)

    async def fail_for_source_readiness(
        self,
        job_id: str,
        reasons: tuple[str, ...],
    ) -> dict[str, Any]:
        job = await self.get(job_id)
        evaluation = job.get("phase") == "evaluation"
        priority = (
            (
                "source_sample_explicitly_invalidated",
                "dataset_source_unavailable",
                "dataset_not_evaluation_eligible",
                "late_cross_boundary_link",
            )
            if evaluation
            else (
                "source_sample_explicitly_invalidated",
                "dataset_source_unavailable",
                "dataset_not_training_ready",
            )
        )
        reason = next((item for item in priority if item in reasons), reasons[0])
        return await self._repository.fail_job_for_source(
            job_id,
            reason,
            {"evaluation_block_reasons" if evaluation else "training_block_reasons": list(reasons)},
        )

    async def close(self) -> None:
        """Close facade-owned integrations; the repository remains caller-owned."""

    @property
    def source_registry(self) -> DatasetSourceRegistry:
        """Expose the shared Task 6 source seam to the acquisition transaction."""
        return self._sources

    @property
    def repository(self) -> PostgresJobQueueRepository:
        return self._repository


async def _phase_source_block_reasons_in_transaction(
    source_registry: DatasetSourceRegistry,
    connection: asyncpg.Connection,
    *,
    phase: str,
    dataset_version: str | None,
    model_kind: str,
) -> tuple[str, ...]:
    if dataset_version is None:
        return ("dataset_source_unavailable",)
    if phase == "training":
        return await source_registry.training_block_reasons_in_transaction(
            connection,
            dataset_version=dataset_version,
            model_kind=model_kind,
        )
    if phase == "evaluation":
        return await source_registry.evaluation_block_reasons_in_transaction(
            connection,
            dataset_version=dataset_version,
            model_kind=model_kind,
        )
    return ()


async def _evaluation_probe_checkpoint_source_error(
    connection: asyncpg.Connection,
    evaluation_probe_job: Any,
) -> str | None:
    """Revalidate a probe-only prior checkpoint without loosening public evaluation origins."""
    from gods_mlops.jobs.models import EvaluationProbeCheckpointSource, ProbeInput

    job = dict(evaluation_probe_job)
    source_refs = _json_value(job.get("source_refs", {}))
    if (
        job.get("phase") != "probe"
        or job.get("target_phase") != "evaluation"
        or not isinstance(source_refs, dict)
        or source_refs.get("schema") != "gods-mlops-probe-input-v1"
    ):
        return "evaluation_probe_checkpoint_source_invalid"
    try:
        probe_input = ProbeInput.from_dict(source_refs)
        source = probe_input.evaluation_checkpoint_source
    except (TypeError, ValueError):
        return "evaluation_probe_checkpoint_source_invalid"
    if (
        source is None
        or probe_input.probe_input_id != job.get("input_id")
        or probe_input.model_kind != job.get("model_kind")
        or probe_input.target_phase != job.get("target_phase")
        or probe_input.config_version != job.get("config_version")
        or probe_input.input_sha256 != str(job.get("input_sha256", "")).strip()
    ):
        return "evaluation_probe_checkpoint_source_invalid"

    stable_identity = {
        "phase": job["phase"],
        "target_phase": job["target_phase"],
        "input_kind": job["input_kind"],
        "input_id": job["input_id"],
        "input_sha256": str(job["input_sha256"]).strip(),
        "dataset_version": job["dataset_version"],
        "model_kind": job["model_kind"],
        "config_version": job["config_version"],
        "config_sha256": str(job["config_sha256"]).strip(),
    }
    stable_source_identity = {"evaluation_checkpoint_source": source.as_dict()}
    dedupe_key = str(job.get("dedupe_key") or "").strip()
    if (
        (not job.get("rerun") and dedupe_key != _enqueue_dedupe_key(stable_identity, stable_source_identity))
        or (job.get("rerun") and dedupe_key)
    ):
        return "evaluation_probe_checkpoint_source_invalid"

    prior_row = await connection.fetchrow(
        "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid",
        source.training_probe_job_id,
    )
    if prior_row is None:
        return "evaluation_probe_checkpoint_source_invalid"
    prior_job = _job_dict(prior_row)
    prior_job["checkpoint_identity"] = _json_value(prior_job.get("checkpoint_identity"))
    probe_profile_row = await connection.fetchrow(
        """SELECT phase,target_phase,model_kind,config_version,config_sha256,
                  profile_state,config_json
           FROM gods_mlops_resource_profiles
           WHERE phase='probe' AND model_kind=$1 AND config_version=$2""",
        source.model_kind,
        source.checkpoint_identity["config_version"],
    )
    training_profile_row = await connection.fetchrow(
        """SELECT phase,target_phase,model_kind,config_version,config_sha256,
                  profile_state,config_json
           FROM gods_mlops_resource_profiles
           WHERE phase='training' AND model_kind=$1 AND config_version=$2""",
        source.model_kind,
        source.checkpoint_identity["config_version"],
    )
    measurement_row = await connection.fetchrow(
        """SELECT measurement_id,result_state,target_phase,model_kind,input_sha256,config_sha256,
                  optimizer_steps,checkpoint_resumed,checkpoint_sha256,exit_code,
                  verification_details
           FROM gods_mlops_profile_measurements WHERE job_id=$1::uuid""",
        source.training_probe_job_id,
    )
    commit_row = await connection.fetchrow(
        """SELECT details FROM gods_mlops_job_events
           WHERE job_id=$1::uuid AND event_type='checkpoint_committed'
             AND details->>'checkpoint_uri'=$2 AND details->>'sha256'=$3
           ORDER BY event_id DESC LIMIT 1""",
        source.training_probe_job_id,
        source.checkpoint_uri,
        source.checkpoint_sha256,
    )
    runtime_evidence_rows = await connection.fetch(
        """SELECT details FROM gods_mlops_job_events
           WHERE job_id=$1::uuid AND event_type='probe_runtime_evidence_committed'
           ORDER BY event_id""",
        source.training_probe_job_id,
    )
    result_artifact_rows = await connection.fetch(
        """SELECT details FROM gods_mlops_job_events
           WHERE job_id=$1::uuid AND event_type='result_artifact_committed'
           ORDER BY event_id""",
        source.training_probe_job_id,
    )
    if (
        any(item is None for item in (probe_profile_row, training_profile_row, measurement_row, commit_row))
        or len(runtime_evidence_rows) != 1
        or len(result_artifact_rows) != 1
    ):
        return "evaluation_probe_checkpoint_source_invalid"
    probe_profile = dict(probe_profile_row)
    training_profile = dict(training_profile_row)
    probe_profile["config_json"] = _json_value(probe_profile["config_json"])
    training_profile["config_json"] = _json_value(training_profile["config_json"])
    measurement = dict(measurement_row)
    measurement["measurement_id"] = str(measurement["measurement_id"])
    measurement["input_sha256"] = str(measurement["input_sha256"]).strip()
    measurement["config_sha256"] = str(measurement["config_sha256"]).strip()
    measurement["checkpoint_sha256"] = (
        str(measurement["checkpoint_sha256"]).strip() if measurement["checkpoint_sha256"] else None
    )
    measurement["verification_details"] = _json_value(measurement["verification_details"])
    checkpoint_commit = _json_value(commit_row["details"])
    runtime_evidence_record = _json_value(runtime_evidence_rows[0]["details"])
    result_artifact = _json_value(result_artifact_rows[0]["details"])
    try:
        source.validate_training_probe_origin(
            prior_job,
            probe_profile,
            training_profile,
            measurement,
            checkpoint_commit,
            runtime_evidence_record,
            result_artifact,
        )
    except (TypeError, ValueError):
        return "evaluation_probe_checkpoint_source_invalid"
    return None


async def _evaluation_checkpoint_source_error(connection: asyncpg.Connection, evaluation_job: Any) -> str | None:
    """Revalidate a typed prior checkpoint against its durable training/job rows."""
    from gods_mlops.jobs.models import EvaluationCheckpointSource

    evaluation_row = dict(evaluation_job)
    source_refs = _json_value(evaluation_row.get("source_refs", {}))
    if not isinstance(source_refs, dict) or source_refs.get("schema") != "gods-mlops-evaluation-source-v1":
        return "evaluation_checkpoint_source_invalid"
    try:
        source = EvaluationCheckpointSource.from_dict(source_refs.get("checkpoint"))
    except (TypeError, ValueError):
        return "evaluation_checkpoint_source_invalid"

    training_job = await connection.fetchrow(
        "SELECT * FROM gods_mlops_jobs WHERE job_id = $1::uuid",
        source.training_job_id,
    )
    if training_job is None:
        return "evaluation_checkpoint_source_invalid"
    training_row = dict(training_job)
    training_profile = await connection.fetchrow(
        """SELECT phase, model_kind, config_version, config_sha256,
                  profile_state, config_json
           FROM gods_mlops_resource_profiles
           WHERE phase = 'training' AND model_kind = $1 AND config_version = $2""",
        training_row.get("model_kind"),
        training_row.get("config_version"),
    )
    if training_profile is None:
        return "evaluation_checkpoint_source_invalid"
    checkpoint_event = await connection.fetchrow(
        """SELECT details FROM gods_mlops_job_events
           WHERE job_id = $1::uuid AND event_type = 'checkpoint_committed'
             AND details->>'checkpoint_uri' = $2 AND details->>'sha256' = $3
           ORDER BY event_id DESC LIMIT 1""",
        source.training_job_id,
        source.checkpoint_uri,
        source.checkpoint_sha256,
    )
    if checkpoint_event is None:
        return "evaluation_checkpoint_source_invalid"
    details = _json_value(checkpoint_event["details"])
    try:
        source.validate_training_origin(
            training_row,
            dict(training_profile),
            details,
        )
    except (TypeError, ValueError):
        return "evaluation_checkpoint_source_invalid"
    return None


def _enqueue_dedupe_key(
    stable_identity: dict[str, Any],
    stable_source_identity: dict[str, Any] | None = None,
) -> str:
    """Bind an optional typed immutable secondary input into automatic job identity."""
    value = stable_identity
    if stable_source_identity is not None:
        value = {**stable_identity, "stable_source_identity": stable_source_identity}
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _operator_actor_sha256(operator_id: str) -> str:
    if not isinstance(operator_id, str) or not operator_id.strip() or len(operator_id) > 255:
        raise ValueError("operator identity must contain 1 to 255 characters")
    return sha256(operator_id.encode("utf-8")).hexdigest()


def _reservation_cleanup_state(reservation: dict[str, Any] | None) -> str:
    if reservation is None:
        return "not_reserved"
    state = reservation.get("state")
    if state in {"settled", "released"}:
        return str(state)
    if state == "reserved":
        return "pending"
    return "unknown"


def _queue_order_sort_key(job: dict[str, Any]) -> tuple[bool, int, str, str]:
    queue_order = job.get("queue_order")
    return (
        queue_order is None,
        queue_order if queue_order is not None else 0,
        str(job.get("created_at", "")),
        str(job.get("job_id", "")),
    )


def _operator_retry_dedupe_key(*, parent_job_id: str, operator_id: str, intent_token: str) -> str:
    """Hash a browser retry intent into the existing durable queue dedupe key."""
    try:
        canonical_parent_id = str(UUID(parent_job_id))
    except (TypeError, ValueError) as error:
        raise ValueError("operator retry parent_job_id must be a UUID") from error
    if canonical_parent_id != parent_job_id.lower():
        raise ValueError("operator retry parent_job_id must be canonical lowercase UUID text")
    if not isinstance(operator_id, str) or not operator_id.strip() or len(operator_id) > 255:
        raise ValueError("operator retry operator_id must contain 1 to 255 characters")
    if not isinstance(intent_token, str) or len(intent_token) < 32 or len(intent_token) > 512:
        raise ValueError("operator retry intent token must contain at least 32 characters")
    material = {
        "kind": "operator-retry-v1",
        "parent_job_id": canonical_parent_id,
        "operator_sha256": sha256(operator_id.encode("utf-8")).hexdigest(),
        "intent_sha256": sha256(intent_token.encode("utf-8")).hexdigest(),
    }
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _operator_retry_key_for_enqueue(
    *,
    rerun: bool,
    parent_job_id: str | None,
    operator_id: str | None,
    intent_token: str | None,
) -> str | None:
    supplied = (parent_job_id is not None, operator_id is not None, intent_token is not None)
    if not any(supplied):
        return None
    if not rerun or not all(supplied):
        raise ValueError("operator retry intent requires rerun, parent_job_id, operator_id, and intent_token")
    assert parent_job_id is not None and operator_id is not None and intent_token is not None
    return _operator_retry_dedupe_key(
        parent_job_id=parent_job_id,
        operator_id=operator_id,
        intent_token=intent_token,
    )


def _operator_retry_request_matches(
    existing: Any,
    *,
    phase: str,
    input_kind: str,
    input_id: str,
    input_sha256: str,
    dataset_version: str | None,
    source_refs: dict[str, Any],
    model_kind: str,
    target_phase: str,
    config_version: str,
    config_sha256: str,
    profile_state: str,
    reservation_bytes: int,
    parent_job_id: str,
    retry_root_id: str | None,
    oom_retries: int,
) -> bool:
    """Require duplicate browser intent to describe the same immutable job."""
    expected_parent = str(UUID(parent_job_id))
    existing_parent = str(existing["parent_job_id"]) if existing["parent_job_id"] is not None else None
    expected_root = str(UUID(retry_root_id)) if retry_root_id is not None else None
    existing_root = str(existing["retry_root_id"]) if existing["retry_root_id"] is not None else None
    return (
        existing["phase"] == phase
        and existing["input_kind"] == input_kind
        and existing["input_id"] == input_id
        and str(existing["input_sha256"]).strip() == input_sha256
        and existing["dataset_version"] == dataset_version
        and _canonical_json(_json_value(existing["source_refs"])) == _canonical_json(source_refs)
        and existing["model_kind"] == model_kind
        and existing["target_phase"] == target_phase
        and existing["config_version"] == config_version
        and str(existing["config_sha256"]).strip() == config_sha256
        and existing["profile_state_snapshot"] == profile_state
        and existing["artifact_reservation_bytes"] == reservation_bytes
        and existing_parent == expected_parent
        and existing_root == expected_root
        and existing["oom_retries"] == oom_retries
        and existing["rerun"] is True
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _artifact_deadline_kwargs(authority: dict[str, Any] | None) -> dict[str, Any]:
    if authority is None:
        return {"artifact_invocation_id": None, "artifact_deadline_at": None}
    return {
        "artifact_invocation_id": authority.get("controller_invocation_id"),
        "artifact_deadline_at": authority.get("artifact_deadline_at"),
    }


def _unmatched_artifact_writers(events: list[tuple[str, dict[str, Any]]]) -> set[tuple[str, str]]:
    """Return writer attempts whose thread completion is not durably acknowledged."""
    started: set[tuple[str, str]] = set()
    quiescent: set[tuple[str, str]] = set()
    required_operations: set[str] = set()
    for event_type, details in events:
        operation_id = str(details.get("operation_id", ""))
        attempt_id = str(details.get("writer_attempt_id", ""))
        if event_type == "artifact_write_pending" and details.get("writer_quiescence_required") is True:
            if operation_id:
                required_operations.add(operation_id)
            else:
                started.add(("", ""))
        if not operation_id or not attempt_id:
            if event_type in {"artifact_write_started", "artifact_write_quiescent"}:
                started.add(("", ""))
            continue
        key = operation_id, attempt_id
        if event_type == "artifact_write_started":
            started.add(key)
        elif event_type == "artifact_write_quiescent":
            quiescent.add(key)
    started_operations = {operation_id for operation_id, _attempt_id in started}
    missing_start = {
        (operation_id, "<missing-start>")
        for operation_id in required_operations - started_operations
    }
    return (started - quiescent) | (quiescent - started) | missing_start


def _checkpoint_lifetime_id(details: dict[str, Any]) -> str:
    """Identify one committed checkpoint object lifetime, not only its reusable URI/hash."""
    lifetime_id = details.get("write_lifetime_id") or details.get("operation_id")
    if lifetime_id:
        return str(lifetime_id)
    legacy_identity = {
        "uri": str(details.get("uri", "")),
        "sha256": str(details.get("sha256", "")),
    }
    return "legacy-" + sha256(_canonical_json(legacy_identity).encode("utf-8")).hexdigest()


def _verified_checkpoint_uri(verified: Any) -> str:
    uri = getattr(verified, "uri", None)
    if isinstance(uri, str) and uri:
        return uri
    path = getattr(verified, "path", None)
    if path is None:
        raise ValueError("verified checkpoint has no durable URI")
    return path.resolve().as_uri()


def _prepared_artifact_uri(prepared: Any, store: Any) -> str:
    uri = getattr(prepared, "uri", None)
    if isinstance(uri, str) and uri:
        return uri
    object_key = getattr(prepared, "object_key", None)
    bucket = getattr(store, "_bucket", None)
    if isinstance(object_key, str) and object_key and isinstance(bucket, str) and bucket:
        return f"s3://{bucket}/{object_key}"
    for name in ("final_path", "data_path"):
        path = getattr(prepared, name, None)
        if path is not None:
            return path.resolve().as_uri()
    raise ValueError("prepared artifact has no durable object URI")


def _validate_pending_artifact_intent(*, job_id: str, details: dict[str, Any], object_store: Any) -> tuple[str, str, int]:
    """Reconstruct the only key the immutable Task 8 store could have written for this intent."""
    identity = details.get("identity")
    bucket = details.get("bucket")
    prefix = details.get("prefix")
    operation = details.get("operation")
    object_key = details.get("object_key")
    kind = details.get("kind")
    if not isinstance(identity, dict) or identity.get("job_id") != job_id:
        raise ValueError("pending artifact identity does not belong to the terminal job")
    try:
        UUID(job_id)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("pending artifact job ID is invalid") from error
    if not isinstance(bucket, str) or not bucket or getattr(object_store, "_bucket", None) != bucket:
        raise ValueError("pending artifact bucket does not match the configured object store")
    if (
        not isinstance(prefix, str)
        or not prefix
        or prefix.startswith("/")
        or prefix.endswith("/")
        or any(part in {"", ".", ".."} for part in prefix.split("/"))
        or posixpath.normpath(prefix) != prefix
    ):
        raise ValueError("pending artifact prefix is not a safe immutable prefix")
    if not isinstance(object_key, str) or not object_key or "\\" in object_key:
        raise ValueError("pending artifact object key is invalid")
    if any(part in {"", ".", ".."} for part in object_key.split("/")):
        raise ValueError("pending artifact object key is not a safe relative key")
    digest = str(details.get("sha256", "")).strip()
    _validate_digest(digest, "pending artifact SHA-256")
    identity_sha = sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    if operation == "result":
        if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", kind):
            raise ValueError("pending result kind is invalid")
        expected_key = f"{prefix}/{job_id}/results/{identity_sha}/{kind}-{digest}.artifact"
        expected_metadata_size = 0
    elif operation == "checkpoint" and kind == "checkpoint":
        expected_key = f"{prefix}/{job_id}/checkpoints/{identity_sha}/{digest}.checkpoint"
        expected_metadata_size = int(details.get("metadata_size_bytes", -1))
        if expected_metadata_size < 0:
            raise ValueError("pending checkpoint metadata size is invalid")
    else:
        raise ValueError("pending artifact operation or kind is invalid")
    if object_key != expected_key or details.get("uri") != f"s3://{bucket}/{expected_key}":
        raise ValueError("pending artifact URI does not match its exact immutable job key")
    size_bytes = int(details.get("size_bytes", 0))
    metadata_size = int(details.get("metadata_size_bytes", 0))
    if size_bytes <= 0 or metadata_size != expected_metadata_size:
        raise ValueError("pending artifact byte identity is invalid")
    if int(details.get("charge_bytes", -1)) != size_bytes + metadata_size:
        raise ValueError("pending artifact charge differs from its exact byte identity")
    return expected_key, digest, size_bytes


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _normalize_evaluation_progress_target(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "phase",
        "target_phase",
        "model_kind",
        "stage",
        "next_index",
        "total_items",
        "completed_batch_count",
        "input_sha256",
        "config_sha256",
        "training_probe_job_id",
        "training_source_checkpoint_sha256",
        "submitted_after",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("evaluation progress yield target fields are unsupported")
    if (
        value.get("phase") != "probe"
        or value.get("target_phase") != "evaluation"
        or value.get("model_kind") != "clip"
        or value.get("stage") != "queries"
        or type(value.get("next_index")) is not int
        or value["next_index"] != 2
        or type(value.get("total_items")) is not int
        or value["total_items"] != 2
        or type(value.get("completed_batch_count")) is not int
        or value["completed_batch_count"] != 1
    ):
        raise ValueError("evaluation progress yield target must be the first complete CLIP query batch")
    _validate_digest(str(value["input_sha256"]), "input_sha256")
    _validate_digest(str(value["config_sha256"]), "config_sha256")
    _validate_digest(str(value["training_source_checkpoint_sha256"]), "training source checkpoint SHA-256")
    try:
        training_job_id = str(UUID(value["training_probe_job_id"]))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("evaluation progress training-probe job ID is invalid") from error
    if training_job_id != value["training_probe_job_id"]:
        raise ValueError("evaluation progress training-probe job ID must be canonical")
    submitted_after = value["submitted_after"]
    if isinstance(submitted_after, str):
        try:
            submitted_after = datetime.fromisoformat(submitted_after)
        except ValueError as error:
            raise ValueError("evaluation progress submission time is invalid") from error
    if (
        not isinstance(submitted_after, datetime)
        or submitted_after.tzinfo is None
        or submitted_after.utcoffset() != UTC.utcoffset(None)
    ):
        raise ValueError("evaluation progress submission time must be timezone-aware UTC")
    normalized = dict(value)
    normalized["training_probe_job_id"] = training_job_id
    normalized["submitted_after"] = submitted_after.astimezone(UTC).isoformat()
    return normalized


def _normalize_evaluation_progress(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "phase",
        "target_phase",
        "model_kind",
        "stage",
        "next_index",
        "total_items",
        "completed_batch_count",
        "input_sha256",
        "config_sha256",
        "training_probe_job_id",
        "training_source_checkpoint_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("evaluation batch progress fields are unsupported")
    if (
        value.get("phase") != "probe"
        or value.get("target_phase") != "evaluation"
        or value.get("model_kind") != "clip"
        or value.get("stage") != "queries"
        or type(value.get("next_index")) is not int
        or type(value.get("total_items")) is not int
        or type(value.get("completed_batch_count")) is not int
    ):
        raise ValueError("evaluation batch progress identity is invalid")
    normalized = _normalize_evaluation_progress_target(
        {
            **value,
            "submitted_after": datetime(2000, 1, 1, tzinfo=UTC),
        }
    )
    normalized.pop("submitted_after")
    return normalized


def _evaluation_progress_probe_input(job: Any, profile: Any) -> ProbeInput:
    row = dict(job)
    source_refs = _json_value(row.get("source_refs", {}))
    if not isinstance(source_refs, dict):
        raise TypeError("evaluation probe source references are unavailable")
    try:
        probe_input = ProbeInput.from_dict(source_refs)
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation probe source references are invalid") from error
    source = probe_input.evaluation_checkpoint_source
    config = _json_value(profile.get("config_json"))
    if (
        row.get("phase") != "probe"
        or row.get("target_phase") != "evaluation"
        or row.get("model_kind") != "clip"
        or row.get("input_kind") != "probe_input"
        or row.get("dataset_version") is not None
        or row.get("profile_state_snapshot") != "candidate"
        or probe_input.target_phase != "evaluation"
        or probe_input.model_kind != "clip"
        or probe_input.probe_input_id != row.get("input_id")
        or probe_input.config_version != row.get("config_version")
        or probe_input.input_sha256 != str(row.get("input_sha256", "")).strip()
        or source is None
        or not isinstance(config, dict)
        or source.model_kind != "clip"
        or source.model_id != config.get("model_id")
        or source.model_revision != config.get("model_revision")
    ):
        raise ValueError("evaluation probe typed source differs from its immutable job/profile identity")
    from gods_mlops.training.contracts import locked_model

    locked = locked_model("clip")
    if source.model_id != locked.model_id or source.model_revision != locked.revision:
        raise ValueError("evaluation probe typed source differs from the locked CLIP model")
    return probe_input


async def _evaluation_progress_arm_details(
    connection: asyncpg.Connection,
    job: Any,
    profile: Any,
    *,
    reason: str,
    target: dict[str, Any],
) -> dict[str, Any]:
    row = dict(job)
    normalized_target = _normalize_evaluation_progress_target(target)
    if (
        normalized_target["input_sha256"] != str(row.get("input_sha256", "")).strip()
        or normalized_target["config_sha256"] != str(row.get("config_sha256", "")).strip()
    ):
        raise ValueError("evaluation progress target differs from its job input/config")
    probe_input = _evaluation_progress_probe_input(row, profile)
    source = probe_input.evaluation_checkpoint_source
    if source is None or (
        normalized_target["training_probe_job_id"] != source.training_probe_job_id
        or normalized_target["training_source_checkpoint_sha256"] != source.checkpoint_sha256
    ):
        raise ValueError("evaluation progress target differs from its typed training checkpoint source")
    source_error = await _evaluation_probe_checkpoint_source_error(connection, row)
    if source_error:
        raise ValueError(source_error)
    identity = {
        "schema_version": 1,
        "reason": reason,
        "expected_lease_generation": 1,
        "phase": "probe",
        "target_phase": "evaluation",
        "model_kind": "clip",
        "job_id": str(row["job_id"]),
        "input_id": str(row["input_id"]),
        "input_sha256": str(row["input_sha256"]).strip(),
        "config_version": str(row["config_version"]),
        "config_sha256": str(row["config_sha256"]).strip(),
        "source_identity": probe_input.as_dict(),
        "training_probe_job_id": source.training_probe_job_id,
        "training_source_checkpoint_sha256": source.checkpoint_sha256,
        "target": normalized_target,
    }
    identity["request_id"] = sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    return identity


def _evaluation_progress_arm_consumed(events: list[Any], arm_event_id: int, fencing_token: int) -> bool:
    return any(
        row["event_type"] == "yield_requested"
        and _json_value(row["details"]).get("arm_event_id") == arm_event_id
        and int(row["fencing_token"] or 0) == fencing_token
        for row in events
    )


def _evaluation_progress_arm_result(
    *,
    arm_event_id: int,
    details: dict[str, Any],
    job: Any,
    lease: Any,
    consumed: bool,
) -> dict[str, Any]:
    return {
        "arm_event_id": arm_event_id,
        "request_id": str(details["request_id"]),
        "expected_lease_generation": int(details["expected_lease_generation"]),
        "job_state": str(job["state"]),
        "lease_generation": int(job["lease_generation"]),
        "active_lease": lease is not None,
        "consumed": consumed,
        "pending_first_attempt": (
            job["state"] == "queued"
            and int(job["lease_generation"]) == 0
            and job["lease_token"] is None
            and lease is None
            and not consumed
        ),
        "target": deepcopy(details["target"]),
    }


def _profile_dict(row: asyncpg.Record) -> dict[str, Any]:
    return {
        "phase": row["phase"],
        "target_phase": row["target_phase"],
        "target_phase": row["target_phase"],
        "model_kind": row["model_kind"],
        "config_version": row["config_version"],
        "config_sha256": row["config_sha256"].strip(),
        "memory_requirement_mib": row["memory_requirement_mib"],
        "artifact_reservation_bytes": row["artifact_reservation_bytes"],
        "checkpoint_reservation_bytes": row["checkpoint_reservation_bytes"],
        "result_reservation_bytes": row["result_reservation_bytes"],
        "config_json": _json_value(row["config_json"]),
        "profile_state": row["profile_state"],
        "measurement_id": str(row["measurement_id"]) if row["measurement_id"] else None,
        "oom_alternatives": _json_value(row["oom_alternatives"]),
    }


def _job_dict(row: asyncpg.Record) -> dict[str, Any]:
    result = dict(row)
    for key, value in tuple(result.items()):
        if isinstance(value, UUID):
            result[key] = str(value)
        elif hasattr(value, "isoformat"):
            result[key] = value.isoformat()
        elif isinstance(value, str) and key in {"source_refs", "reason_detail", "checkpoint_identity"}:
            result[key] = json.loads(value)
    if result.get("input_sha256") is not None:
        result["input_sha256"] = result["input_sha256"].strip()
    if result.get("config_sha256") is not None:
        result["config_sha256"] = result["config_sha256"].strip()
    if result.get("dedupe_key") is not None:
        result["dedupe_key"] = result["dedupe_key"].strip()
    return result


def _probe_runtime_evidence_projection(payload: bytes) -> tuple[dict[str, str], str, str]:
    """Hash finalized evidence bytes and extract only the non-secret origin identity."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("probe runtime evidence payload must contain bytes")
    try:
        evidence = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("probe runtime evidence payload is invalid JSON") from error
    if not isinstance(evidence, dict):
        raise ValueError("probe runtime evidence payload must be an object")
    projection = {key: evidence.get(key) for key in _PROBE_RUNTIME_EVIDENCE_FIELDS}
    if (
        projection["event"] != "task8_real_model_probe_complete"
        or projection["target_phase"] != "training"
        or projection["model_kind"] not in {"detr", "clip"}
        or not isinstance(projection["job_id"], str)
        or not projection["job_id"]
        or not isinstance(projection["config_version"], str)
        or not projection["config_version"]
        or not isinstance(projection["input_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", projection["input_sha256"])
        or not isinstance(projection["docker_image_id"], str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", projection["docker_image_id"])
        or not isinstance(projection["source_commit"], str)
        or not re.fullmatch(r"[0-9a-f]{40,64}", projection["source_commit"])
        or projection["image_source_commit"] != projection["source_commit"]
    ):
        raise ValueError("probe runtime evidence identity is malformed")
    return projection, sha256(payload).hexdigest(), sha256(_canonical_json(evidence).encode("utf-8")).hexdigest()


def _probe_result_artifact_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("probe runtime evidence has no committed result artifact")
    fields = ("kind", "uri", "sha256", "size_bytes", "identity", "object_key")
    projected = {key: value.get(key) for key in fields}
    if (
        not isinstance(projected["kind"], str)
        or not isinstance(projected["uri"], str)
        or not isinstance(projected["sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", projected["sha256"])
        or type(projected["size_bytes"]) is not int
        or projected["size_bytes"] <= 0
        or not isinstance(projected["identity"], dict)
        or not isinstance(projected["object_key"], str)
    ):
        raise ValueError("probe runtime evidence committed result identity is invalid")
    return projected


def _probe_runtime_evidence_authority(
    evidence: dict[str, str],
    evidence_sha256: str,
    evidence_canonical_sha256: str,
    *,
    job: dict[str, Any],
    measurement: dict[str, Any],
    checkpoint_commit: dict[str, Any],
    result_artifact: dict[str, Any],
) -> dict[str, Any]:
    identity = _json_value(job.get("checkpoint_identity"))
    checkpoint_sha256 = str(checkpoint_commit.get("sha256", "")).strip()
    if not isinstance(identity, dict) or not checkpoint_sha256:
        raise ValueError("probe runtime evidence checkpoint origin is unavailable")
    return {
        "schema": _PROBE_RUNTIME_EVIDENCE_SCHEMA,
        "evidence_sha256": evidence_sha256,
        "evidence_canonical_sha256": evidence_canonical_sha256,
        "evidence_projection_sha256": sha256(_canonical_json(evidence).encode("utf-8")).hexdigest(),
        "evidence": evidence,
        "job_id": str(job.get("job_id", "")),
        "measurement_id": str(measurement.get("measurement_id", "")),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity": identity,
        "result_artifact": _probe_result_artifact_projection(result_artifact),
    }


def _require_runtime_evidence_matches_committed_rows(
    payload: bytes,
    *,
    measurement: dict[str, Any],
    result_artifact: dict[str, Any],
) -> None:
    value = json.loads(payload)
    if value.get("measurement") != measurement or value.get("result_artifact") != result_artifact:
        raise ValueError("finalized runtime evidence differs from durable measurement or result readback")


def _observation_state_dict(row: asyncpg.Record) -> dict[str, Any]:
    return {
        "node_id": row["node_id"],
        "observation_id": str(row["observation_id"]),
        "observed_at": row["observed_at"].isoformat(),
        "received_at": row["received_at"].isoformat(),
        "observation": _json_value(row["observation"]),
        "failure_code": row["failure_code"],
        "failure_count": row["failure_count"],
        "idle_since": row["idle_since"].isoformat() if row["idle_since"] is not None else None,
        "idle_observation_count": row["idle_observation_count"],
        "last_observed_at": (
            row["last_observed_at"].isoformat() if row["last_observed_at"] is not None else None
        ),
    }


def _checkpoint_identity_from_row(row: asyncpg.Record):
    from gods_mlops.jobs.checkpoints import CheckpointIdentity

    return CheckpointIdentity(
        job_id=str(row["job_id"]),
        input_kind=row["input_kind"],
        input_id=row["input_id"],
        input_sha256=row["input_sha256"].strip(),
        phase=row["phase"],
        model_kind=row["model_kind"],
        config_version=row["config_version"],
        config_sha256=row["config_sha256"].strip(),
        dataset_version=row["dataset_version"],
    )
