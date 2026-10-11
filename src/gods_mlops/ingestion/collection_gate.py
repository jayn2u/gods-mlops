"""Fail-closed live Ubuntu filesystem gate for new candidate receipts."""

from __future__ import annotations

import posixpath
from datetime import UTC, datetime
from typing import Any

from gods_mlops.jobs.models import ResourceObservation
from gods_mlops.jobs.queue import PostgresJobQueueRepository

MIN_UBUNTU_FILESYSTEM_FREE_BYTES = 1024**4
MAX_RESOURCE_OBSERVATION_AGE_SECONDS = 10


class CollectionPausedError(RuntimeError):
    """New collection is temporarily paused; the product spool may safely retry."""

    def __init__(self, reason_code: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.retryable = True
        self.details = details or {}

    def as_detail(self) -> dict[str, Any]:
        return {
            "code": "collection_paused",
            "reason": self.reason_code,
            "retryable": self.retryable,
            **({"details": self.details} if self.details else {}),
        }


class CollectionStorageGate:
    """Check the latest durable observation before accepting a new sample ID."""

    def __init__(
        self,
        *,
        repository: PostgresJobQueueRepository,
        expected_node_id: str | None = "ubuntu",
        expected_host_identity: str | None,
        expected_gpu_uuid: str | None,
        expected_filesystem_identity: str | None,
        expected_storage_path: str | None,
        min_free_bytes: int = MIN_UBUNTU_FILESYSTEM_FREE_BYTES,
        max_observation_age_seconds: int = MAX_RESOURCE_OBSERVATION_AGE_SECONDS,
        clock=None,
    ) -> None:
        self._repository = repository
        self._expected_node_id = expected_node_id
        self._expected_host_identity = expected_host_identity
        self._expected_gpu_uuid = expected_gpu_uuid
        self._expected_filesystem_identity = expected_filesystem_identity
        self._expected_storage_path = (
            posixpath.normpath(expected_storage_path) if expected_storage_path else None
        )
        self._min_free_bytes = min_free_bytes
        self._max_observation_age_seconds = max_observation_age_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        if all(
            (
                expected_node_id,
                expected_host_identity,
                expected_gpu_uuid,
                expected_filesystem_identity,
                expected_storage_path,
            )
        ):
            self._repository.configure_observation_identity(
                expected_node_id=expected_node_id,
                expected_host_identity=expected_host_identity,
                expected_gpu_uuid=expected_gpu_uuid,
                expected_filesystem_identity=expected_filesystem_identity,
                expected_storage_path=expected_storage_path,
            )

    async def ensure_available(self) -> ResourceObservation:
        if not all(
            (
                self._expected_node_id,
                self._expected_host_identity,
                self._expected_gpu_uuid,
                self._expected_filesystem_identity,
                self._expected_storage_path,
            )
        ):
            raise CollectionPausedError("ubuntu_observer_identity_not_configured")
        assert self._expected_node_id is not None
        state = await self._repository.get_observation_state(self._expected_node_id)
        if state is None or state["observation"] is None or state["failure_code"] is not None:
            failure = state["failure_code"] if state is not None else None
            raise CollectionPausedError(
                "ubuntu_observation_unavailable",
                details={"observer_failure": failure} if failure else None,
            )
        try:
            observation = ResourceObservation.from_dict(state["observation"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise CollectionPausedError("ubuntu_observation_incomplete") from error
        now = self._now()
        age_seconds = (now - observation.observed_at).total_seconds()
        if age_seconds < -2 or age_seconds > self._max_observation_age_seconds:
            raise CollectionPausedError(
                "ubuntu_observation_stale",
                details={"age_seconds": round(age_seconds, 3)},
            )
        if (
            observation.node_id != self._expected_node_id
            or observation.hostname.lower() != "ubuntu"
            or observation.host_identity != self._expected_host_identity
            or observation.gpu_uuid != self._expected_gpu_uuid
            or observation.filesystem_identity != self._expected_filesystem_identity
            or posixpath.normpath(observation.storage_path) != self._expected_storage_path
        ):
            raise CollectionPausedError("ubuntu_observation_identity_mismatch")
        if observation.filesystem_available_bytes < 0:
            raise CollectionPausedError("ubuntu_observation_incomplete")
        if observation.filesystem_available_bytes < self._min_free_bytes:
            raise CollectionPausedError(
                "ubuntu_filesystem_headroom_below_minimum",
                details={
                    "available_bytes": observation.filesystem_available_bytes,
                    "minimum_bytes": self._min_free_bytes,
                },
            )
        return observation

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("collection gate clock must return a timezone-aware datetime")
        return now.astimezone(UTC)
