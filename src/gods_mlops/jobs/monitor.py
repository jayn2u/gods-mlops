"""Five-second ownership monitor for the single Ubuntu A6000 lease."""

from __future__ import annotations

import asyncio

from .admission import GpuAdmission
from .models import ResourceObservation
from .queue import JobQueue, PostgresJobQueueRepository

MONITOR_INTERVAL_SECONDS = 5


class GpuJobMonitor:
    """Observe GPU process identities, request own-job yield, and prove lease release."""

    def __init__(
        self,
        *,
        repository: PostgresJobQueueRepository,
        queue: JobQueue,
        admission: GpuAdmission,
        interval_seconds: int = MONITOR_INTERVAL_SECONDS,
    ) -> None:
        if interval_seconds != MONITOR_INTERVAL_SECONDS:
            raise ValueError("GPU monitor interval is fixed at five seconds")
        self._repository = repository
        self._queue = queue
        self._admission = admission
        self._interval_seconds = interval_seconds

    async def observe(self, value: dict | ResourceObservation) -> dict | None:
        try:
            observation, _state = await self._admission.record_observation(value)
        except ValueError as error:
            lease = await self._repository.get_active_lease(self._admission.expected_gpu_uuid)
            if lease is None:
                return None
            job = await self._queue.get(lease["job_id"])
            if job["state"] == "running":
                await self._queue.request_yield(
                    lease["job_id"],
                    reason=str(error) if str(error) else "ubuntu_observation_unavailable",
                )
                return await self._queue.get(lease["job_id"])
            return job

        lease = await self._repository.get_active_lease(observation.gpu_uuid)
        if lease is None:
            return None
        job_id = lease["job_id"]
        source_reasons = await self._queue.training_source_block_reasons(job_id)
        source_reason = None
        if source_reasons:
            source_reason = (
                "source_sample_explicitly_invalidated"
                if "source_sample_explicitly_invalidated" in source_reasons
                else "dataset_source_unavailable"
            )
            if (await self._queue.get(job_id))["state"] in {"running", "yield_requested"}:
                await self._queue.request_yield(job_id, reason=source_reason)
        owner_pid = lease["owner_pid"]
        owner_start = lease["owner_start_ticks"]
        if owner_pid is None or owner_start is None:
            if observation.gpu_processes:
                if source_reason is None:
                    await self._queue.request_yield(
                        job_id,
                        reason="external_gpu_process_started_before_owner_bind",
                    )
                return await self._queue.get(job_id)
            if self._lease_expired(lease, observation) and await self._idle_window_ready(observation):
                await self._repository.release_unbound_expired_lease(
                    job_id=job_id,
                    lease_token=lease["lease_token"],
                    observation=observation,
                    terminal_reason=source_reason,
                )
            return await self._queue.get(job_id)

        owner_is_live = any(
            process.pid == owner_pid and process.start_ticks == owner_start
            for process in observation.process_table
        )
        owner_pid_still_uses_gpu = any(
            process.pid == owner_pid for process in observation.gpu_processes
        )
        external_processes = [
            process
            for process in observation.gpu_processes
            if process.pid != owner_pid or process.start_ticks != owner_start
        ]
        if owner_is_live:
            if external_processes:
                if source_reason is None:
                    await self._queue.request_yield(
                        job_id,
                        reason="external_gpu_process_started",
                    )
                return await self._queue.get(job_id)
            if (await self._queue.get(job_id))["state"] == "running":
                await self._queue.renew_lease(
                    job_id,
                    lease["lease_token"],
                    now=observation.observed_at,
                )
            return await self._queue.get(job_id)

        # A PID with a different kernel start time is a new process. Its GPU
        # allocation keeps the exclusive lease fenced until that PID also exits.
        if owner_pid_still_uses_gpu:
            job = await self._queue.get(job_id)
            if external_processes and job["state"] == "running" and source_reason is None:
                await self._queue.request_yield(
                    job_id,
                    reason="external_gpu_process_started",
                )
                return await self._queue.get(job_id)
            return job

        released = await self._repository.release_after_observed_exit(
            job_id=job_id,
            lease_token=lease["lease_token"],
            observation=observation,
            terminal_reason=source_reason,
        )
        job = await self._queue.get(job_id)
        if released and job["state"] in {"completed", "failed", "cancelled"}:
            await self._queue.settle_artifact_reservation(job_id)
        return job

    async def run_once(self) -> dict | None:
        """Read and reconcile one host observation from the configured Ubuntu producer."""
        try:
            observation = await self._admission.observe_once()
        except ValueError:
            # Admission already persisted the producer failure and reset its idle window.
            return await self.observe_failure()
        return await self.observe(observation)

    async def observe_failure(self) -> dict | None:
        lease = await self._repository.get_active_lease(self._admission.expected_gpu_uuid)
        if lease is None:
            return None
        job = await self._queue.get(lease["job_id"])
        if job["state"] == "running":
            await self._queue.request_yield(
                lease["job_id"],
                reason="ubuntu_observation_unavailable",
            )
            return await self._queue.get(lease["job_id"])
        return job

    async def run_forever(self) -> None:
        """Poll at the fixed policy cadence; an observer outage only requests own-job yield."""
        while True:
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001 - a failed monitor iteration cannot grant resources
                await self.observe_failure()
            await asyncio.sleep(self._interval_seconds)

    @staticmethod
    def _lease_expired(lease: dict, observation: ResourceObservation) -> bool:
        from datetime import datetime

        expiry = lease["expires_at"]
        if isinstance(expiry, str):
            expiry = datetime.fromisoformat(expiry)
        return expiry <= observation.observed_at

    async def _idle_window_ready(self, observation: ResourceObservation) -> bool:
        # An unbound lease has no owner identity to prove dead. It can only expire
        # after an independent full idle series is available in durable observation state.
        state = await self._repository.get_observation_state(observation.node_id)
        if state is None or state["idle_since"] is None:
            return False
        from datetime import datetime

        idle_since = state["idle_since"]
        if isinstance(idle_since, str):
            idle_since = datetime.fromisoformat(idle_since)
        return (
            (observation.observed_at - idle_since).total_seconds() >= 30
            and state["idle_observation_count"] >= 7
        )
