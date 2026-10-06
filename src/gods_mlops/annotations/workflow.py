"""Bridge durable review assignments to Label Studio CE task import and cleanup."""

from __future__ import annotations

from functools import partial
from typing import Any
from uuid import UUID

import anyio

from gods_mlops.ingestion.storage import S3SampleStore

from .label_studio import LabelStudioApiClient, LabelStudioMediaCleanupClient
from .models import ReviewAssignment
from .service import AnnotationService
from .storage import PostgresAnnotationRepository


class LabelStudioReviewWorkflow:
    def __init__(
        self,
        *,
        repository: PostgresAnnotationRepository,
        objects: S3SampleStore,
        label_studio: LabelStudioApiClient,
        media_cleanup: LabelStudioMediaCleanupClient,
    ) -> None:
        self._repository = repository
        self._objects = objects
        self._label_studio = label_studio
        self._media_cleanup = media_cleanup

    async def start_bbox_review(self, *, sample_id: UUID, project_id: int) -> dict[str, Any]:
        """Create a durable frame assignment, then provision its Label Studio task."""
        source = await self._repository.bbox_review_source(sample_id)
        assignment = await self.prepare_assignment(
            sample_id=sample_id,
            stage="bbox",
            project_id=project_id,
            bbox_revision=None,
            media_object_key=source["object_key"],
            required_bytes=source["object_size_bytes"],
        )
        return await self.provision_task(revision=assignment.revision, project_id=project_id)

    async def prepare_assignment(
        self,
        *,
        sample_id: UUID,
        stage: str,
        project_id: int,
        bbox_revision: str | None,
        media_object_key: str,
        required_bytes: int,
        expected_caption_revision_id: str | None = None,
        source_sha256: str | None = None,
        model_request_key: str | None = None,
        model_version: str | None = None,
        item_id: str | None = None,
    ) -> ReviewAssignment:
        """Reserve source media and review quota before any Label Studio upload."""
        if project_id <= 0:
            raise ValueError("Label Studio project ID must be positive")
        model_identity = (source_sha256, model_request_key, model_version, item_id)
        if any(value is not None for value in model_identity):
            if not all(isinstance(value, str) and value for value in model_identity):
                raise ValueError("model draft assignment requires complete immutable source/provenance")
            if expected_caption_revision_id is not None:
                raise ValueError("model draft assignments cannot replace a human caption revision")
            return await self._repository.create_model_draft_assignment(
                sample_id=sample_id,
                stage=stage,
                project_id=project_id,
                bbox_revision=bbox_revision,
                media_object_key=media_object_key,
                required_bytes=required_bytes,
                source_sha256=str(source_sha256),
                model_request_key=str(model_request_key),
                model_version=str(model_version),
                item_id=str(item_id),
            )
        return await self._repository.create_review_assignment(
            sample_id=sample_id,
            stage=stage,
            label_studio_task_id=None,
            bbox_revision=bbox_revision,
            media_object_key=media_object_key,
            required_bytes=required_bytes,
            expected_caption_revision_id=expected_caption_revision_id,
        )

    async def provision_task(
        self,
        *,
        revision: str,
        project_id: int,
        prediction: dict | None = None,
    ) -> dict[str, Any]:
        assignment = await self._repository.assignment_by_revision(revision)
        if assignment.state == "active" and assignment.label_studio_task_id is not None:
            media = await self._repository.label_studio_media_upload(revision)
            if media is None:
                raise RuntimeError("active Label Studio review has no media reservation")
            if prediction is not None and str(media["upload_filename"]).startswith("gods-model-"):
                if not _matches_model_request_filename(media["upload_filename"], prediction):
                    raise RuntimeError("active review assignment belongs to different model provenance")
                attach_prediction = getattr(self._label_studio, "attach_prediction", None)
                if not callable(attach_prediction):
                    raise RuntimeError("Label Studio client cannot idempotently attach the model prediction")
                await attach_prediction(
                    project_id=project_id,
                    task_id=assignment.label_studio_task_id,
                    prediction=prediction,
                )
            return _provisioned(assignment, media)
        if assignment.state != "provisioning":
            raise RuntimeError("review assignment is not available for Label Studio provisioning")

        source = await self._repository.review_media_source(assignment.sample_id, UUID(revision))
        prepared_media = await self._repository.label_studio_media_upload(revision)
        filename = (
            prepared_media["upload_filename"]
            if prepared_media is not None
            else f"gods-review-{revision}-{source['sha256'][:16]}.jpg"
        )
        if (
            prediction is not None
            and filename.startswith("gods-model-")
            and not _matches_model_request_filename(filename, prediction)
        ):
            raise RuntimeError("prepared review assignment belongs to different model provenance")
        image = await anyio.to_thread.run_sync(
            partial(
                self._objects.read_object,
                object_key=source["object_key"],
                expected_sha256=source["sha256"],
            )
        )
        if len(image) != source["object_size_bytes"]:
            raise OSError("Label Studio source media size did not match the persisted object metadata")
        reservation = await self._repository.reserve_label_studio_media(
            revision=revision,
            project_id=project_id,
            filename=filename,
            sha256_digest=source["sha256"],
            object_size_bytes=len(image),
        )
        reference = await self._label_studio.import_media_task(
            project_id=project_id,
            filename=filename,
            image=image,
            prediction=prediction,
        )
        media = await self._repository.bind_label_studio_media(
            revision=revision,
            task_id=reference.task_id,
            file_upload_id=reference.file_upload_id,
            upload_path=reference.media_path,
        )
        return {
            **_provisioned(await self._repository.assignment_by_revision(revision), media),
            "prediction_attached": prediction is not None,
            "media_reservation_reused": reservation["state"] != "reserved",
        }

    async def finalize_annotation(self, sample_id: str, revision: str) -> dict[str, Any]:
        snapshot = await AnnotationService(
            repository=self._repository,
            label_studio=self._label_studio,
        ).finalize_annotation(sample_id, revision)
        cleanup_pending = await self._cleanup_uploaded_media(revision)
        return {**snapshot, "label_studio_media_cleanup_pending": cleanup_pending}

    async def close_assignment(
        self,
        *,
        sample_id: UUID,
        revision: str,
        outcome: str,
        reason: str,
    ) -> dict[str, Any]:
        closed = await self._repository.close_review_assignment(
            sample_id=sample_id,
            revision=revision,
            outcome=outcome,
            reason=reason,
        )
        cleanup_pending = await self._cleanup_uploaded_media(revision)
        return {**closed, "label_studio_media_cleanup_pending": cleanup_pending}

    async def cleanup_pending_media(self, *, limit: int = 100) -> dict[str, int]:
        """Retry only terminal or explicitly pending Label Studio media deletions."""
        revisions = await self._repository.label_studio_media_cleanup_candidates(limit=limit)
        cleaned = 0
        deferred = 0
        for revision in revisions:
            if await self._cleanup_uploaded_media(revision):
                deferred += 1
            else:
                cleaned += 1
        return {"claimed": len(revisions), "cleaned": cleaned, "deferred": deferred}

    async def _cleanup_uploaded_media(self, revision: str) -> bool:
        media = await self._repository.label_studio_media_upload(revision)
        if media is None or media["state"] == "deleted":
            return False
        if media["state"] == "reserved":
            # An import request may have been accepted before its response was lost.
            # Keep its byte reservation until reconciliation recovers the remote file/task.
            return True
        await self._repository.mark_label_studio_media_delete_pending(revision)
        assignment = await self._repository.assignment_by_revision(revision)
        try:
            await self._label_studio.delete_review_task(
                project_id=media["project_id"],
                task_id=media["task_id"],
                file_upload_id=media["file_upload_id"],
            )
            await self._media_cleanup.delete_upload(
                project_id=media["project_id"],
                file_upload_id=media["file_upload_id"],
                upload_path=media["upload_path"],
                expected_sha256=media["sha256"],
                expected_size_bytes=media["object_size_bytes"],
            )
            await self._repository.finish_label_studio_media_cleanup(revision)
            return False
        except Exception as error:  # noqa: BLE001 - saved human labels remain final; retry cleanup later
            await self._repository.record_retention_event(
                sample_id=assignment.sample_id,
                action="label_media_cleanup_deferred",
                reason="label_studio_upload_cleanup_failed",
                details={"error_type": type(error).__name__},
            )
            return True


def _provisioned(assignment: ReviewAssignment, media: dict[str, Any]) -> dict[str, Any]:
    if assignment.label_studio_task_id is None or media["file_upload_id"] is None or not media["upload_path"]:
        raise RuntimeError("Label Studio media reservation has not been bound to its imported task")
    return {
        "sample_id": str(assignment.sample_id),
        "assignment_revision": assignment.revision,
        "stage": assignment.stage,
        "bbox_revision": assignment.bbox_revision,
        "label_studio_project_id": media["project_id"],
        "label_studio_task_id": assignment.label_studio_task_id,
        "label_studio_file_upload_id": media["file_upload_id"],
        "media_path": media["upload_path"],
        "media_sha256": media["sha256"],
        "media_size_bytes": media["object_size_bytes"],
        "state": assignment.state,
    }


def _matches_model_request_filename(filename: str, prediction: dict[str, Any]) -> bool:
    from hashlib import sha256

    model_version = prediction.get("model_version")
    if not isinstance(model_version, str) or not model_version:
        return False
    suffix = sha256(model_version.encode("utf-8")).hexdigest()[:16]
    return filename.endswith(f"-{suffix}.jpg")
