"""Shared annotation and review assignment value objects."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


class AnnotationError(RuntimeError):
    """Base class for an invalid label review transition."""


class AnnotationNotSubmittedError(AnnotationError):
    """Raised when Label Studio has only a prediction or a cancelled result."""


class ReviewAssignmentNotFoundError(AnnotationError):
    """Raised when a finalization refers to an unknown review revision."""


class ReviewAssignmentConflictError(AnnotationError):
    """Raised when review state changed before the requested transition."""


class ReviewQuotaExceededError(AnnotationError):
    """Raised when an assignment would exceed the protected-review byte quota."""


@dataclass(frozen=True, slots=True)
class ReviewAssignment:
    assignment_id: UUID
    revision: str
    sample_id: UUID
    stage: str
    bbox_revision: str | None
    label_studio_task_id: int | None
    media_object_key: str
    required_bytes: int
    state: str


def assignment_from_record(record: object) -> ReviewAssignment:
    return ReviewAssignment(
        assignment_id=record["assignment_id"],
        revision=str(record["revision"]),
        sample_id=record["sample_id"],
        stage=record["stage"],
        bbox_revision=record["bbox_revision"],
        label_studio_task_id=record["label_studio_task_id"],
        media_object_key=record["media_object_key"],
        required_bytes=record["required_bytes"],
        state=record["state"],
    )
