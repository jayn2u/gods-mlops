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

    @property
    def label_studio_configured(self) -> bool:
        return self._label_studio is not None

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

    async def list_reviews(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """List durable assignment and provenance state without labeling drafts as human-approved."""
        return await self._repository.list_review_assignments(limit=limit)

    async def start_bbox_review(self, sample_id: str, *, project_id: int, workflow) -> dict[str, Any]:
        """Start a frame review through the existing assignment and Label Studio workflow."""
        sample_uuid = UUID(sample_id)
        return await workflow.start_bbox_review(sample_id=sample_uuid, project_id=project_id)
