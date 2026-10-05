"""Fail-closed observation policy and fenced lease acquisition for the Ubuntu A6000."""

from __future__ import annotations

import inspect
import posixpath
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import ResourceObservation
from .queue import (
    JobQueue,
    ObservationReplayError,
    PostgresJobQueueRepository,
    ResourceObservationRejectedError,
)

GPU_SAFETY_MIB = 4_096
GPU_IDLE_WINDOW_SECONDS = 30
GPU_MONITOR_INTERVAL_SECONDS = 5
GPU_LEASE_SECONDS = 15
OBSERVATION_MAX_AGE_SECONDS = 10
OBSERVATION_FUTURE_TOLERANCE_SECONDS = 2
UBUNTU_STORAGE_MIN_FREE_BYTES = 1024**4
_MAX_OBSERVATION_GAP_SECONDS = GPU_MONITOR_INTERVAL_SECONDS * 2
_MIN_IDLE_OBSERVATION_COUNT = GPU_IDLE_WINDOW_SECONDS // GPU_MONITOR_INTERVAL_SECONDS + 1


class GpuAdmission:
    """Admit a single job only after observed Ubuntu idleness and a last-moment reread."""

    def __init__(
        self,
        *,
        repository: PostgresJobQueueRepository,
        queue: JobQueue,
        expected_host_identity: str,
        expected_gpu_uuid: str,
        expected_filesystem_identity: str,
        expected_storage_path: str,
        expected_node_id: str = "ubuntu",
        observer: Any | None = None,
        clock=None,
        idle_window_seconds: int = GPU_IDLE_WINDOW_SECONDS,
        monitor_interval_seconds: int = GPU_MONITOR_INTERVAL_SECONDS,
        lease_seconds: int = GPU_LEASE_SECONDS,
        observation_max_age_seconds: int = OBSERVATION_MAX_AGE_SECONDS,
        min_filesystem_bytes: int = UBUNTU_STORAGE_MIN_FREE_BYTES,
        safety_mib: int = GPU_SAFETY_MIB,
    ) -> None:
        self._repository = repository
        self._queue = queue
        self._expected_host_identity = expected_host_identity
        self._expected_gpu_uuid = expected_gpu_uuid
        self._expected_filesystem_identity = expected_filesystem_identity
        self._expected_storage_path = posixpath.normpath(expected_storage_path)
        self._expected_node_id = expected_node_id
        self._observer = observer
        self._clock = clock or (lambda: datetime.now(UTC))
        self._idle_window_seconds = idle_window_seconds
        self._monitor_interval_seconds = monitor_interval_seconds
        self._lease_seconds = lease_seconds
        self._observation_max_age_seconds = observation_max_age_seconds
        self._min_filesystem_bytes = min_filesystem_bytes
        self._safety_mib = safety_mib
        self._repository.configure_observation_identity(
            expected_node_id=expected_node_id,
            expected_host_identity=expected_host_identity,
            expected_gpu_uuid=expected_gpu_uuid,
            expected_filesystem_identity=expected_filesystem_identity,
            expected_storage_path=expected_storage_path,
        )

    @property
    def expected_gpu_uuid(self) -> str:
        return self._expected_gpu_uuid

    async def observe_once(self) -> ResourceObservation:
        if self._observer is None:
            await self._record_observation_failure("observer_unreachable")
            raise ValueError("ubuntu_observation_unavailable")
        try:
            method = self._observer.observe if hasattr(self._observer, "observe") else self._observer
            value = method()
            if inspect.isawaitable(value):
                value = await value
        except Exception as error:  # noqa: BLE001 - a failed producer read invalidates prior idle proof
            await self._record_observation_failure("observer_unreachable")
            raise ValueError("ubuntu_observation_unavailable") from error
        try:
            observation = (
                value if isinstance(value, ResourceObservation) else ResourceObservation.from_dict(value)
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            await self._record_observation_failure("ubuntu_observation_unavailable")
            raise ValueError("ubuntu_observation_unavailable") from error
        try:
            now = self._now()
        except ValueError:
            await self._record_observation_failure("admission_clock_unavailable")
            raise
        reason = self._observation_reason(observation, now)
        if reason is not None:
            await self._record_observation_failure(reason)
            raise ValueError(reason)
        return observation

    async def admit(self, job_id: str, observation: dict[str, Any]) -> dict[str, Any]:
        """Persist the supplied observation and return a lease or a retryable wait."""
        try:
            parsed = ResourceObservation.from_dict(observation)
        except (KeyError, TypeError, ValueError, OverflowError):
            await self._record_observation_failure("ubuntu_observation_unavailable")
            return await self._wait(job_id, "waiting_gpu", "ubuntu_observation_unavailable")
        now = self._now()
        invalid_reason = self._observation_reason(parsed, now)
        if invalid_reason is not None:
            await self._record_observation_failure(invalid_reason)
            return await self._wait(job_id, "waiting_gpu", invalid_reason)
        try:
            state = await self._repository.record_observation(
                parsed,
                max_gap_seconds=_MAX_OBSERVATION_GAP_SECONDS,
                received_at=now,
            )
        except ObservationReplayError:
            return await self._wait(
                job_id,
                "waiting_gpu",
                "ubuntu_observation_replayed",
                observation_id=parsed.observation_id,
            )
        except ResourceObservationRejectedError as error:
            return await self._wait(
                job_id,
                "waiting_gpu",
                error.reason_code,
                observation_id=parsed.observation_id,
            )
        return await self._admit_from_current(job_id, parsed, state, now=now, prelaunch=False)

    async def record_observation(
        self,
        value: dict[str, Any] | ResourceObservation,
    ) -> tuple[ResourceObservation, dict[str, Any]]:
        """Validate and persist one trusted observer sample for monitor consumers."""
        try:
            observation = (
                value if isinstance(value, ResourceObservation) else ResourceObservation.from_dict(value)
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            await self._record_observation_failure("ubuntu_observation_unavailable")
            raise ValueError("ubuntu_observation_unavailable") from error
        now = self._now()
        reason = self._observation_reason(observation, now)
        if reason is not None:
            await self._record_observation_failure(reason)
            raise ValueError(reason)
        try:
            state = await self._repository.record_observation(
                observation,
                max_gap_seconds=_MAX_OBSERVATION_GAP_SECONDS,
                received_at=now,
            )
        except ResourceObservationRejectedError as error:
            raise ValueError(error.reason_code) from error
        return observation, state

    async def _admit_from_current(
        self,
        job_id: str,
        observation: ResourceObservation,
        state: dict[str, Any],
        *,
        now: datetime,
        prelaunch: bool,
    ) -> dict[str, Any]:
        job = await self._queue.get(job_id)
        if job["state"] in {"completed", "failed", "cancelled"}:
            return job
        source_reasons = await self._queue.training_source_block_reasons(job_id)
        if source_reasons:
            current_lease = await self._repository.get_active_lease(observation.gpu_uuid)
            if current_lease is not None and current_lease["job_id"] == job_id:
                reason = (
                    "source_sample_explicitly_invalidated"
                    if "source_sample_explicitly_invalidated" in source_reasons
                    else "dataset_source_unavailable"
                )
                await self._queue.request_yield(job_id, reason=reason)
                return await self._queue.get(job_id)
            return await self._queue.fail_for_source_readiness(job_id, source_reasons)
        next_retry_at = job.get("next_retry_at")
        if next_retry_at is not None:
            if isinstance(next_retry_at, str):
                next_retry_at = datetime.fromisoformat(next_retry_at)
            if now < next_retry_at:
                return await self._wait(
                    job_id,
                    "waiting_gpu",
                    "communication_retry_delay",
                    observation_id=observation.observation_id,
                    details={"retry_at": next_retry_at.isoformat()},
                )
        active_lease = await self._repository.get_active_lease(observation.gpu_uuid)
        if active_lease is not None:
            if active_lease["job_id"] == job_id:
                return job
            return await self._wait(
                job_id,
                "waiting_gpu",
                "gpu_lease_owned",
                observation_id=observation.observation_id,
                details={"owner_job_id": active_lease["job_id"]},
            )

        profile = await self._repository.get_profile(
            phase=job["phase"],
            model_kind=job["model_kind"],
            config_version=job["config_version"],
        )
        if profile is None:
            return await self._wait(
                job_id,
                "waiting_profile",
                "versioned_resource_profile_missing",
                observation_id=observation.observation_id,
            )
        if profile["profile_state"] == "rejected":
            return await self._wait(
                job_id,
                "waiting_profile",
                "resource_profile_rejected",
                observation_id=observation.observation_id,
            )
        if profile["profile_state"] != "measured" and job["phase"] != "probe":
            return await self._wait(
                job_id,
                "waiting_profile",
                "resource_profile_not_measured",
                observation_id=observation.observation_id,
            )
        if not await self._repository.is_next_eligible_job(job_id):
            return await self._wait(
                job_id,
                "waiting_gpu",
                "fifo_queue_position",
                observation_id=observation.observation_id,
            )

        reason = self._capacity_reason(observation, profile)
        if reason is not None:
            state_name = "waiting_storage" if reason == "ubuntu_filesystem_headroom_below_minimum" else "waiting_gpu"
            return await self._wait(
                job_id,
                state_name,
                reason,
                observation_id=observation.observation_id,
                details={
                    "observed_free_mib": observation.free_mib,
                    "required_memory_mib": profile["memory_requirement_mib"],
                    "safety_mib": self._safety_mib,
                    "filesystem_available_bytes": observation.filesystem_available_bytes,
                    "minimum_filesystem_bytes": self._min_filesystem_bytes,
                },
            )
        if observation.gpu_processes:
            return await self._wait(
                job_id,
                "waiting_gpu",
                "external_gpu_processes",
                observation_id=observation.observation_id,
                details={"external_pids": sorted(item.pid for item in observation.gpu_processes)},
            )
        if not self._idle_window_ready(state, now):
            return await self._wait(
                job_id,
                "waiting_gpu",
                "idle_observation_window",
                observation_id=observation.observation_id,
            )

        if not prelaunch:
            fresh = await self._read_prelaunch_observation(job_id, observation.observation_id)
            if isinstance(fresh, dict):
                current_job = await self._queue.get(job_id)
                if (
                    current_job.get("lease_token") is not None
                    or current_job["state"] in {"running", "yield_requested", "completed", "failed", "cancelled", "retrying"}
                ):
                    return current_job
                return fresh
            observation, state = fresh
            current_job = await self._queue.get(job_id)
            if (
                current_job.get("lease_token") is not None
                or current_job["state"] in {"running", "yield_requested", "completed", "failed", "cancelled", "retrying"}
            ):
                return current_job
            reason = self._capacity_reason(observation, profile)
            if reason is not None:
                state_name = (
                    "waiting_storage"
                    if reason == "ubuntu_filesystem_headroom_below_minimum"
                    else "waiting_gpu"
                )
                return await self._wait(
                    job_id,
                    state_name,
                    reason,
                    observation_id=observation.observation_id,
                )
            if observation.gpu_processes:
                return await self._wait(
                    job_id,
                    "waiting_gpu",
                    "external_gpu_processes",
                    observation_id=observation.observation_id,
                    details={"external_pids": sorted(item.pid for item in observation.gpu_processes)},
                )
            now = self._now()
            if not self._idle_window_ready(state, now):
                return await self._wait(
                    job_id,
                    "waiting_gpu",
                    "idle_observation_window",
                    observation_id=observation.observation_id,
                )

        return await self._repository.acquire_gpu_lease(
            job_id=job_id,
            observation=observation,
            profile=profile,
            source_registry=self._queue.source_registry,
            now=self._now(),
            lease_seconds=self._lease_seconds,
            min_idle_seconds=self._idle_window_seconds,
            min_idle_observations=_MIN_IDLE_OBSERVATION_COUNT,
            min_filesystem_bytes=self._min_filesystem_bytes,
            safety_mib=self._safety_mib,
        )

    async def _read_prelaunch_observation(
        self,
        job_id: str,
        previous_observation_id: str,
    ) -> tuple[ResourceObservation, dict[str, Any]] | dict[str, Any]:
        if self._observer is None:
            await self._record_observation_failure("observer_unreachable")
            return await self._wait(
                job_id,
                "waiting_gpu",
                "ubuntu_prelaunch_observation_unavailable",
                observation_id=previous_observation_id,
            )
        try:
            method = self._observer.observe if hasattr(self._observer, "observe") else self._observer
            value = method()
            if inspect.isawaitable(value):
                value = await value
            observation = value if isinstance(value, ResourceObservation) else ResourceObservation.from_dict(value)
        except Exception:  # noqa: BLE001 - observation failure must only defer GPU work
            await self._repository.record_observation_failure(
                node_id=self._expected_node_id,
                failure_code="observer_unreachable",
            )
            return await self._wait(
                job_id,
                "waiting_gpu",
                "ubuntu_prelaunch_observation_unavailable",
                observation_id=previous_observation_id,
            )
        now = self._now()
        reason = self._observation_reason(observation, now)
        if reason is not None:
            await self._record_observation_failure(reason)
            if reason == "ubuntu_observation_identity_mismatch":
                return await self._wait(
                    job_id,
                    "waiting_gpu",
                    reason,
                    observation_id=observation.observation_id,
                )
            return await self._wait(
                job_id,
                "waiting_gpu",
                "ubuntu_prelaunch_observation_unavailable",
                observation_id=observation.observation_id,
                details={"observation_reason": reason},
            )
        try:
            state = await self._repository.record_observation(
                observation,
                max_gap_seconds=_MAX_OBSERVATION_GAP_SECONDS,
                received_at=now,
            )
        except ObservationReplayError:
            return await self._wait(
                job_id,
                "waiting_gpu",
                "ubuntu_observation_replayed",
                observation_id=observation.observation_id,
            )
        except ResourceObservationRejectedError as error:
            return await self._wait(
                job_id,
                "waiting_gpu",
                error.reason_code,
                observation_id=observation.observation_id,
            )
        return observation, state

    async def _record_observation_failure(self, failure_code: str) -> None:
        await self._repository.record_observation_failure(
            node_id=self._expected_node_id,
            failure_code=failure_code,
        )

    def _observation_reason(self, observation: ResourceObservation, now: datetime) -> str | None:
        age = (now - observation.observed_at).total_seconds()
        if age < -OBSERVATION_FUTURE_TOLERANCE_SECONDS or age > self._observation_max_age_seconds:
            return "ubuntu_observation_stale"
        if (
            observation.node_id != self._expected_node_id
            or observation.hostname.lower() != "ubuntu"
            or observation.host_identity != self._expected_host_identity
            or observation.gpu_name != "NVIDIA RTX A6000"
            or observation.gpu_uuid != self._expected_gpu_uuid
            or observation.filesystem_identity != self._expected_filesystem_identity
            or posixpath.normpath(observation.storage_path) != self._expected_storage_path
        ):
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
        return None

    def _capacity_reason(self, observation: ResourceObservation, profile: dict[str, Any]) -> str | None:
        if observation.filesystem_available_bytes < self._min_filesystem_bytes:
            return "ubuntu_filesystem_headroom_below_minimum"
        required = int(profile["memory_requirement_mib"])
        if observation.free_mib < required + self._safety_mib:
            return "measured_profile_exceeds_current_free_memory"
        return None

    def _idle_window_ready(self, state: dict[str, Any], now: datetime) -> bool:
        idle_since = state.get("idle_since")
        if isinstance(idle_since, str):
            idle_since = datetime.fromisoformat(idle_since)
        return (
            idle_since is not None
            and (now - idle_since).total_seconds() >= self._idle_window_seconds
            and state.get("idle_observation_count", 0) >= _MIN_IDLE_OBSERVATION_COUNT
        )

    async def _wait(
        self,
        job_id: str,
        state: str,
        reason: str,
        *,
        observation_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._repository.update_wait_state(
            job_id,
            state=state,
            reason_code=reason,
            details=details,
            observation_id=observation_id,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("admission clock must return a timezone-aware datetime")
        return value.astimezone(UTC)
