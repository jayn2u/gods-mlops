"""Coordinate Label Studio submissions and durable annotation snapshots."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from .label_studio import LabelStudioTaskReader, submitted_annotation
from .models import ReviewAssignmentNotFoundError
from .storage import PostgresAnnotationRepository


class AnnotationService:
    def __init__(self, *, repository: PostgresAnnotationRepository, label_studio: LabelStudioTaskReader) -> None:
        self._repository = repository
        self._label_studio = label_studio

    async def finalize_annotation(self, sample_id: str, revision: str) -> dict[str, Any]:
        """Persist a submitted Label Studio annotation as an immutable local revision."""
        sample_uuid = UUID(sample_id)
        revision_uuid = UUID(revision)
        assignment = await self._repository.assignment(sample_uuid, revision_uuid)
        if assignment.state == "finalized":
            return await self._repository.finalized_revision(assignment.revision)
        if assignment.state != "active" or assignment.label_studio_task_id is None:
            raise ReviewAssignmentNotFoundError("review task is not ready for submission finalization")
        task = await self._label_studio.get_task(assignment.label_studio_task_id)
        if not isinstance(task, dict) or task.get("id") != assignment.label_studio_task_id:
            raise ReviewAssignmentNotFoundError("Label Studio task does not match the assigned review")
        annotation = submitted_annotation(task)
        return await self._repository.finalize_submitted_annotation(
            assignment=assignment,
            annotation=annotation,
            task=task,
        )
