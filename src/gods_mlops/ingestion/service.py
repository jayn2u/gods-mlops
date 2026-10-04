"""Coordinate crash-repairable candidate object and metadata persistence."""

from __future__ import annotations

from functools import partial
from hashlib import sha256
from typing import Protocol
from uuid import UUID

import anyio

from .schemas import (
    CandidateMetadata,
    SampleConflictError,
    SampleReceipt,
    SampleStorageError,
    GlobalObjectLimitError,
    SampleExpiredError,
)
from .storage import PostgresIngestionRepository, S3SampleStore, sample_object_key


class IngestionRepository(Protocol):
    """Describe durable sample reservation and receipt operations."""

    async def reserve(
        self,
        metadata: CandidateMetadata,
        object_key: str,
        object_size_bytes: int,
    ): ...

    async def mark_received(self, sample_id: UUID) -> SampleReceipt: ...

    async def record_failure(self, sample_id: UUID, code: str) -> None: ...

    async def ready(self) -> None: ...

    async def claim_expired(self, *, limit: int = 100): ...

    async def finish_expiry(self, sample_id: UUID) -> bool: ...


class SampleObjectStore(Protocol):
    """Describe the synchronous S3 operations executed away from the event loop."""

    def ensure_object(self, *, object_key: str, image: bytes, expected_sha256: str) -> None: ...

    def ready(self) -> None: ...

    def delete_object(self, object_key: str) -> None: ...


class IngestionService:
    """Accept identical retries and reject sample-ID reuse with different bytes."""

    def __init__(
        self,
        *,
        repository: IngestionRepository,
        objects: SampleObjectStore,
    ) -> None:
        self._repository = repository
        self._objects = objects

    async def receive(self, metadata: CandidateMetadata, image: bytes) -> SampleReceipt:
        """Persist quota+metadata, verify the immutable object, and then issue receipt."""
        if not image:
            raise ValueError("sample image must not be empty")
        if sha256(image).hexdigest() != metadata.sha256:
            raise ValueError("sample image does not match its declared SHA-256")
        object_key = sample_object_key(metadata)
        reservation = await self._repository.reserve(metadata, object_key, len(image))
        if reservation.sha256 != metadata.sha256:
            raise SampleConflictError("sample ID is already bound to different bytes")
        if reservation.state == "storage_limited":
            raise GlobalObjectLimitError("candidate object store has reached its global capacity")
        if reservation.state == "purge_pending":
            raise SampleStorageError("sample is being removed by the retention owner")
        if reservation.state == "expired":
            raise SampleExpiredError("sample object has expired and is no longer available")
        try:
            await anyio.to_thread.run_sync(
                partial(
                    self._objects.ensure_object,
                    object_key=reservation.object_key,
                    image=image,
                    expected_sha256=metadata.sha256,
                )
            )
        except Exception as error:  # noqa: BLE001 - preserve the pending row for a repair retry
            try:
                await self._repository.record_failure(metadata.sample_id, "object_store_failed")
            except Exception:  # noqa: BLE001 - failure recording cannot hide the original failure
                pass
            raise SampleStorageError("sample object could not be verified") from error
        try:
            return await self._repository.mark_received(metadata.sample_id)
        except Exception as error:  # noqa: BLE001 - verified object plus pending row is repairable
            raise SampleStorageError("sample metadata receipt could not be committed") from error

    async def ready(self) -> None:
        """Check both durable dependencies before reporting the receiver ready."""
        await self._repository.ready()
        await anyio.to_thread.run_sync(self._objects.ready)

    async def prune_expired(self, *, limit: int = 100) -> int:
        """Delete only expired, unselected objects and retain idempotency tombstones."""
        expired = await self._repository.claim_expired(limit=limit)
        removed = 0
        for sample in expired:
            await anyio.to_thread.run_sync(
                partial(self._objects.delete_object, sample.object_key)
            )
            removed += int(await self._repository.finish_expiry(sample.sample_id))
        return removed


__all__ = ["IngestionService"]
