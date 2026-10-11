"""Run expiry only when an operator-owned caller explicitly requests it."""

from __future__ import annotations

from datetime import datetime
from functools import partial
from typing import Any

import anyio

from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore


class RetentionService:
    def __init__(
        self,
        *,
        repository: PostgresIngestionRepository,
        objects: S3SampleStore,
        annotations: PostgresAnnotationRepository,
    ) -> None:
        self._repository = repository
        self._objects = objects
        self._annotations = annotations

    async def expire_candidates(self, now: datetime, *, limit: int = 100) -> dict[str, Any]:
        """Delete expired, unprotected frames while preserving stable receipt tombstones."""
        if now.tzinfo is None:
            raise ValueError("retention time must be timezone-aware")
        claimed = await self._repository.claim_expired(now=now, limit=limit)
        deleted = 0
        deferred = 0
        crops_deleted = 0
        crops_deferred = 0
        for sample in claimed:
            crop_deleted, crop_deferred = await self._delete_unadopted_crops(
                now=now,
                sample_id=sample.sample_id,
                limit=limit,
            )
            crops_deleted += crop_deleted
            crops_deferred += crop_deferred
            if crop_deferred:
                deferred += 1
                continue
            try:
                await anyio.to_thread.run_sync(
                    partial(self._objects.delete_object, sample.object_key)
                )
                completed = await self._repository.finish_expiry(sample.sample_id)
            except Exception as error:  # noqa: BLE001 - preserve the tombstone reservation for retry
                await self._annotations.record_retention_event(
                    sample_id=sample.sample_id,
                    action="expiry_deferred",
                    reason="candidate_object_delete_failed",
                    details={"error_type": type(error).__name__},
                )
                deferred += 1
                continue
            if completed:
                deleted += 1
            else:
                deferred += 1
        other_deleted, other_deferred = await self._delete_unadopted_crops(
            now=now,
            sample_id=None,
            limit=limit,
        )
        crops_deleted += other_deleted
        crops_deferred += other_deferred
        return {
            "claimed": len(claimed),
            "deleted": deleted,
            "deferred": deferred,
            "crops_deleted": crops_deleted,
            "crops_deferred": crops_deferred,
        }

    async def _delete_unadopted_crops(self, *, now: datetime, sample_id, limit: int) -> tuple[int, int]:
        claimed = await self._annotations.claim_unadopted_crops(
            now=now,
            sample_id=sample_id,
            limit=limit,
        )
        deleted = 0
        deferred = 0
        for crop in claimed:
            try:
                await anyio.to_thread.run_sync(
                    partial(self._objects.delete_object, crop["object_key"])
                )
                finished = await self._annotations.finish_crop_expiry(crop["crop_id"])
            except Exception as error:  # noqa: BLE001 - keep purge_pending crop for retry
                await self._annotations.record_retention_event(
                    sample_id=crop["sample_id"],
                    action="crop_expiry_deferred",
                    reason="unadopted_crop_delete_failed",
                    details={"crop_id": str(crop["crop_id"]), "error_type": type(error).__name__},
                )
                deferred += 1
                continue
            if finished:
                deleted += 1
            else:
                deferred += 1
        return deleted, deferred
