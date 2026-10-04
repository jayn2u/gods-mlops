"""Validated candidate metadata and durable upload receipts."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
Revision = Annotated[str, Field(min_length=1, max_length=255)]


class CandidateReason(StrEnum):
    """Why the product worker retained this frame for review."""

    PERIODIC = "periodic"
    LOW_CONFIDENCE = "low_confidence"
    OPERATOR = "operator"


class CandidateMetadata(BaseModel):
    """Immutable source identity attached to one full-frame candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: UUID
    camera_id: UUID
    captured_at_utc: datetime
    reason: CandidateReason
    sha256: Sha256
    model_revision: Revision
    processor_revision: Revision

    @field_validator("captured_at_utc")
    @classmethod
    def normalize_capture_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("captured_at_utc must include a timezone")
        return value.astimezone(UTC)


class SampleReceipt(BaseModel):
    """Return only after the image object and metadata commit are verified."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_id: UUID
    sha256: Sha256
    receipt_id: UUID
    object_key: str


class SampleConflictError(ValueError):
    """A sample ID was reused for bytes other than its original image."""


class DailySampleLimitError(ValueError):
    """The camera has reached its collection limit for the captured day."""


class GlobalObjectLimitError(ValueError):
    """The candidate bucket has reached its shared 1 TiB object capacity."""


class SampleExpiredError(ValueError):
    """A retained sample tombstone proves its former object has expired."""


class SampleStorageError(RuntimeError):
    """Object or metadata persistence did not reach a receiptable state."""
