"""Wire-format and worker-side anchoring for authority-backed artifact deadlines."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID


DEADLINE_ENV = "GODS_MLOPS_WORKER_ARTIFACT_DEADLINE_UTC"
INVOCATION_ENV = "GODS_MLOPS_WORKER_ARTIFACT_INVOCATION_ID"
LEGACY_ARTIFACT_TIMEOUT_SECONDS = 30


class ArtifactDeadlineError(ValueError):
    """A worker deadline is malformed or differs from its durable authority row."""


def artifact_deadline_environment(record: dict[str, Any]) -> dict[str, str]:
    invocation_id = record.get("controller_invocation_id")
    deadline_at = record.get("artifact_deadline_at")
    try:
        parsed_invocation_id = UUID(str(invocation_id))
    except (TypeError, ValueError, AttributeError) as error:
        raise ArtifactDeadlineError("artifact deadline invocation ID is invalid") from error
    if str(parsed_invocation_id) != str(invocation_id).lower():
        raise ArtifactDeadlineError("artifact deadline invocation ID is not canonical")
    if not isinstance(deadline_at, datetime) or deadline_at.tzinfo is None:
        raise ArtifactDeadlineError("artifact deadline must be a timezone-aware UTC timestamp")
    if deadline_at.utcoffset() != timedelta(0):
        raise ArtifactDeadlineError("artifact deadline must be a timezone-aware UTC timestamp")
    return {
        INVOCATION_ENV: str(parsed_invocation_id),
        DEADLINE_ENV: deadline_at.astimezone(UTC).isoformat(timespec="microseconds"),
    }


def artifact_deadline_for_attempt(
    *,
    existing_fence: dict[str, Any] | None,
    existing_invocation: dict[str, Any] | None,
    controller_invocation_id: str,
    lease_token: str,
    candidate_deadline_at: datetime,
) -> dict[str, Any]:
    """Select the immutable deadline for a fence or its controller invocation."""
    if candidate_deadline_at.tzinfo is None or candidate_deadline_at.utcoffset() != timedelta(0):
        raise ArtifactDeadlineError("artifact deadline candidate must be a timezone-aware UTC timestamp")
    try:
        invocation_id = UUID(controller_invocation_id)
    except (TypeError, ValueError, AttributeError) as error:
        raise ArtifactDeadlineError("artifact deadline invocation ID is invalid") from error
    if str(invocation_id) != controller_invocation_id:
        raise ArtifactDeadlineError("artifact deadline invocation ID is not canonical")
    if existing_fence is not None:
        if str(existing_fence.get("lease_token")) != lease_token:
            raise ArtifactDeadlineError("artifact deadline fence is bound to a different lease token")
        return {
            "controller_invocation_id": str(existing_fence["controller_invocation_id"]),
            "artifact_deadline_at": existing_fence["artifact_deadline_at"],
            "lease_token": lease_token,
        }
    deadline_at = (
        existing_invocation["artifact_deadline_at"]
        if existing_invocation is not None
        else candidate_deadline_at.astimezone(UTC)
    )
    return {
        "controller_invocation_id": str(invocation_id),
        "artifact_deadline_at": deadline_at,
        "lease_token": lease_token,
    }


def parse_worker_deadline_environment(environment: dict[str, str]) -> tuple[str, datetime] | None:
    invocation_raw = environment.get(INVOCATION_ENV)
    deadline_raw = environment.get(DEADLINE_ENV)
    if invocation_raw is None and deadline_raw is None:
        return None
    if invocation_raw is None or deadline_raw is None:
        raise ArtifactDeadlineError("explicit artifact deadline requires both authority fields")
    try:
        invocation_id = UUID(invocation_raw)
    except (TypeError, ValueError, AttributeError) as error:
        raise ArtifactDeadlineError("artifact deadline invocation ID is invalid") from error
    if str(invocation_id) != invocation_raw:
        raise ArtifactDeadlineError("artifact deadline invocation ID is not canonical")
    try:
        deadline_at = datetime.fromisoformat(deadline_raw)
    except (TypeError, ValueError) as error:
        raise ArtifactDeadlineError("artifact deadline timestamp is malformed") from error
    if deadline_at.tzinfo is None or deadline_at.utcoffset() != timedelta(0):
        raise ArtifactDeadlineError("artifact deadline must be a UTC timestamp")
    if deadline_at.astimezone(UTC).isoformat(timespec="microseconds") != deadline_raw:
        raise ArtifactDeadlineError("artifact deadline timestamp is not canonical")
    return str(invocation_id), deadline_at.astimezone(UTC)


def anchor_worker_deadline(
    *,
    environment_value: tuple[str, datetime],
    authority_record: dict[str, Any],
    monotonic_before_read: float,
    monotonic_after_read: float,
) -> float:
    invocation_id, transported_deadline = environment_value
    authority_invocation_id = str(authority_record.get("controller_invocation_id", ""))
    authority_deadline = authority_record.get("artifact_deadline_at")
    database_now = authority_record.get("database_now")
    if (
        invocation_id != authority_invocation_id
        or not isinstance(authority_deadline, datetime)
        or transported_deadline != authority_deadline.astimezone(UTC)
    ):
        raise ArtifactDeadlineError("worker artifact deadline differs from its durable authority record")
    if not isinstance(database_now, datetime) or database_now.tzinfo is None:
        raise ArtifactDeadlineError("artifact deadline authority has no database clock sample")
    if monotonic_after_read < monotonic_before_read:
        raise ArtifactDeadlineError("worker monotonic clock moved backwards during deadline read")
    remaining = (transported_deadline - database_now.astimezone(UTC)).total_seconds()
    remaining -= monotonic_after_read - monotonic_before_read
    if remaining <= 0:
        raise ArtifactDeadlineError("worker artifact deadline has expired")
    return monotonic_after_read + remaining
