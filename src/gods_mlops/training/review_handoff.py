"""CPU-only idempotent Task 5 assignment handoff for verified model drafts."""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from hashlib import sha256
from typing import Any
from uuid import UUID

from gods_mlops.annotations.label_studio import LabelStudioApiClient, LabelStudioMediaCleanupClient
from gods_mlops.annotations.models import ReviewAssignmentConflictError
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.annotations.workflow import LabelStudioReviewWorkflow
from gods_mlops.datasets.manifest import canonical_json
from gods_mlops.datasets.publish import DatasetObjectStore
from gods_mlops.ingestion.storage import S3SampleStore
from gods_mlops.jobs.queue import JobQueue

from .claims import _annotation_batch_from_job
from .contracts import locked_model


class PreparationReviewHandoffError(RuntimeError):
    """A verified GPU draft could not be attached to its exact Task 5 review source."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(message)


async def publish_preparation_handoffs(queue: JobQueue, job_id: str) -> list[dict[str, Any]]:
    """Attach drafts only after the completed GPU worker lease is observed released."""
    job = await queue.get(job_id)
    if job.get("phase") != "preparation" or job.get("state") != "completed" or job.get("lease_token") is not None:
        raise PreparationReviewHandoffError(
            "preparation_assignment_gpu_not_released",
            "Task 5 review assignment requires completed preparation after observed GPU exit",
        )
    model_kind = str(job["model_kind"])
    if model_kind not in {"detr", "qwen"}:
        raise PreparationReviewHandoffError(
            "preparation_assignment_model_unsupported", "Task 5 draft handoff supports DETR and Qwen only"
        )
    project_id = _project_id(model_kind)
    try:
        batch = _annotation_batch_from_job(job)
        await queue.source_registry.verify_annotation_batch(batch)
    except Exception as error:
        raise PreparationReviewHandoffError(
            "preparation_assignment_source_changed",
            "the frame/crop or bbox revision changed before Task 5 assignment",
        ) from error

    model = locked_model(model_kind)
    artifacts = await queue.repository.result_artifacts_for(job_id)
    kind = "drafts" if model_kind == "detr" else "caption_drafts"
    matching_artifacts = [item for item in artifacts if item.get("kind") == kind]
    identity = await queue.repository.checkpoint_identity(job_id)
    matching_artifacts = [
        item for item in matching_artifacts if item.get("identity") == identity.as_dict()
    ]
    if len(matching_artifacts) != 1:
        raise PreparationReviewHandoffError(
            "preparation_assignment_artifact_missing",
            "completed GPU preparation has no unique committed result artifact for this input",
        )
    artifact = matching_artifacts[0]
    artifact_bytes = _read_result_artifact(artifact)
    try:
        document = json.loads(artifact_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationReviewHandoffError(
            "preparation_assignment_artifact_invalid", "committed preparation artifact is not valid JSON"
        ) from error
    _verify_document(job, document, model.model_id, model.revision, kind)

    refs = job.get("source_refs")
    source_items = refs.get("items") if isinstance(refs, dict) else None
    drafts = document.get("drafts")
    if not isinstance(source_items, list) or not isinstance(drafts, list):
        raise PreparationReviewHandoffError(
            "preparation_assignment_source_mismatch", "draft output does not contain its frozen source batch"
        )
    expected_source_kind = "frame" if model_kind == "detr" else "crop"
    by_id = {
        str(item.get("item_id")): item
        for item in source_items
        if isinstance(item, dict) and item.get("item_kind") == expected_source_kind
    }
    draft_by_id = {
        str(item.get("item_id")): item
        for item in drafts
        if isinstance(item, dict) and item.get("item_id")
    }
    if len(draft_by_id) != len(drafts) or not set(draft_by_id).issubset(by_id):
        raise PreparationReviewHandoffError(
            "preparation_assignment_source_mismatch", "draft item IDs do not match the exact frozen frame/crop refs"
        )
    if model_kind == "detr" and set(draft_by_id) != set(by_id):
        raise PreparationReviewHandoffError(
            "preparation_assignment_source_mismatch", "DETR draft output omitted one or more selected frames"
        )

    database_url = _required("GODS_MLOPS_DATABASE_URL")
    endpoint = _required("GODS_MLOPS_S3_ENDPOINT_URL")
    access_key = _required("GODS_MLOPS_S3_ACCESS_KEY")
    secret_key = _required("GODS_MLOPS_S3_SECRET_KEY")
    bucket = _required("GODS_MLOPS_S3_BUCKET")
    label_studio = LabelStudioApiClient(
        base_url=_required("GODS_MLOPS_LABEL_STUDIO_URL"),
        api_token=_required("GODS_MLOPS_LABEL_STUDIO_API_TOKEN"),
    )
    try:
        project = await label_studio.get_project(project_id)
        validate_project_configuration(model_kind, str(project.get("label_config", "")))
    except PreparationReviewHandoffError:
        raise
    except Exception as error:
        raise PreparationReviewHandoffError(
            "preparation_assignment_project_unavailable",
            "configured Task 5 review project could not be read or authenticated",
        ) from error
    cleanup = LabelStudioMediaCleanupClient(
        base_url=_required("GODS_MLOPS_LABEL_MEDIA_CLEANUP_URL"),
        token=_required("GODS_MLOPS_LABEL_MEDIA_CLEANUP_TOKEN"),
    )
    objects = S3SampleStore(
        endpoint_url=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region=os.environ.get("GODS_MLOPS_S3_REGION", "us-east-1"),
    )
    annotations = PostgresAnnotationRepository(database_url=database_url)
    await annotations.ensure_schema()
    workflow = LabelStudioReviewWorkflow(
        repository=annotations,
        objects=objects,
        label_studio=label_studio,
        media_cleanup=cleanup,
    )
    try:
        completed_events = await queue.repository.review_handoffs_for(job_id)
        event_by_item = {str(item.get("item_id")): item for item in completed_events}
        results = []
        for item_id, draft in sorted(draft_by_id.items()):
            source = by_id[item_id]
            sample_id = str(source["sample_id"])
            source_key = str(source["object_key"])
            source_sha = str(source["sha256"])
            source_size = int(source["object_size_bytes"])
            bbox_revision = (
                str(source["revision_id"]) if source.get("revision_id") is not None else None
            )
            prediction = _prediction(
                job=job,
                model_id=model.model_id,
                model_revision=model.revision,
                result_sha256=artifact["sha256"],
                draft=draft,
                model_kind=model_kind,
            )
            model_version = str(prediction["model_version"])
            request_key = sha256(
                canonical_json(
                    {
                        "job_id": job_id,
                        "item_id": item_id,
                        "sample_id": sample_id,
                        "item_kind": expected_source_kind,
                        "object_key": source_key,
                        "source_sha256": source_sha,
                        "source_size_bytes": source_size,
                        "bbox_revision": bbox_revision,
                        "project_id": project_id,
                        "input_sha256": job["input_sha256"],
                        "config_sha256": job["config_sha256"].strip(),
                        "model_id": model.model_id,
                        "model_revision": model.revision,
                        "result_sha256": artifact["sha256"],
                    }
                )
            ).hexdigest()
            existing_event = event_by_item.get(item_id)
            if existing_event is not None:
                if (
                    existing_event.get("project_id") != project_id
                    or existing_event.get("model_request_key") != request_key
                    or existing_event.get("result_sha256") != artifact["sha256"]
                    or existing_event.get("bbox_revision") != bbox_revision
                ):
                    raise PreparationReviewHandoffError(
                        "preparation_assignment_provenance_mismatch",
                        "existing Task 5 assignment event belongs to a different project or result",
                    )
                revision = str(existing_event["assignment_revision"])
                try:
                    assignment = await workflow._repository.assignment_by_revision(revision)
                    _verify_assignment(assignment, source, model_kind, project_id)
                    if assignment.state != "finalized":
                        await _provision_with_retry(
                            workflow,
                            revision=revision,
                            project_id=project_id,
                            prediction=prediction,
                        )
                except Exception as error:
                    raise PreparationReviewHandoffError(
                        "preparation_assignment_retry_failed",
                        "the existing Task 5 model-draft assignment could not be resumed",
                    ) from error
                results.append(existing_event)
                continue

            try:
                assignment = await workflow.prepare_assignment(
                    sample_id=UUID(sample_id),
                    stage="bbox" if model_kind == "detr" else "caption",
                    project_id=project_id,
                    bbox_revision=bbox_revision,
                    media_object_key=source_key,
                    required_bytes=source_size,
                    source_sha256=source_sha,
                    model_request_key=request_key,
                    model_version=model_version,
                    item_id=item_id,
                )
                event = await queue.repository.record_preparation_review_assignment(
                    job_id=job_id,
                    item_id=item_id,
                    sample_id=sample_id,
                    assignment_revision=assignment.revision,
                    project_id=project_id,
                    model_request_key=request_key,
                    model_version=model_version,
                    model_id=model.model_id,
                    model_revision=model.revision,
                    media_object_key=source_key,
                    source_sha256=source_sha,
                    source_size_bytes=source_size,
                    bbox_revision=bbox_revision,
                    result_sha256=artifact["sha256"],
                )
                if assignment.state != "finalized":
                    await _provision_with_retry(
                        workflow,
                        revision=assignment.revision,
                        project_id=project_id,
                        prediction=prediction,
                    )
                results.append(event)
            except PreparationReviewHandoffError:
                raise
            except ReviewAssignmentConflictError as error:
                raise PreparationReviewHandoffError(
                    "preparation_assignment_conflict",
                    "the exact Task 5 source already has a different human/model assignment",
                ) from error
            except Exception as error:
                raise PreparationReviewHandoffError(
                    "preparation_assignment_provision_failed",
                    "Task 5 assignment or Label Studio provisioning failed; retry will resume the same request",
                ) from error
        return results
    finally:
        await annotations.close()


def _project_id(model_kind: str) -> int:
    key = (
        "GODS_MLOPS_LABEL_STUDIO_BBOX_PROJECT_ID"
        if model_kind == "detr"
        else "GODS_MLOPS_LABEL_STUDIO_CAPTION_PROJECT_ID"
    )
    raw = os.environ.get(key)
    if not raw:
        raise PreparationReviewHandoffError(
            "preparation_assignment_project_missing",
            f"Task 5 project configuration {key} is required before draft handoff",
        )
    try:
        project_id = int(raw)
    except ValueError as error:
        raise PreparationReviewHandoffError(
            "preparation_assignment_project_invalid", f"Task 5 project configuration {key} is invalid"
        ) from error
    if project_id <= 0:
        raise PreparationReviewHandoffError(
            "preparation_assignment_project_invalid", f"Task 5 project configuration {key} must be positive"
        )
    return project_id


def validate_project_configuration(model_kind: str, label_config: str) -> None:
    """Reject a project whose operator-authored controls do not match this draft kind."""
    try:
        root = ET.fromstring(label_config)
    except ET.ParseError as error:
        raise PreparationReviewHandoffError(
            "preparation_assignment_project_mismatch", "Task 5 project has invalid XML label configuration"
        ) from error
    image = root.find(".//Image[@name='image'][@value='$image']")
    if image is None:
        raise PreparationReviewHandoffError(
            "preparation_assignment_project_mismatch", "Task 5 project must expose its assigned source image"
        )
    if model_kind == "detr":
        boxes = root.findall(".//RectangleLabels[@name='bbox'][@toName='image']")
        if not boxes or not any(label.get("value") == "person" for box in boxes for label in box.findall("Label")):
            raise PreparationReviewHandoffError(
                "preparation_assignment_project_mismatch",
                "configured bbox project must accept the person rectangle prediction",
            )
    elif model_kind == "qwen":
        captions = root.findall(".//TextArea[@name='caption'][@toName='image']")
        if not captions or not any(item.get("required", "").lower() == "true" for item in captions):
            raise PreparationReviewHandoffError(
                "preparation_assignment_project_mismatch",
                "configured caption project must expose the required crop-caption field",
            )
    else:
        raise PreparationReviewHandoffError(
            "preparation_assignment_model_unsupported", "Task 5 project validation supports DETR and Qwen only"
        )


def _read_result_artifact(event: dict[str, Any]) -> bytes:
    if not isinstance(event.get("uri"), str) or not event["uri"].startswith("s3://"):
        raise PreparationReviewHandoffError(
            "preparation_assignment_artifact_invalid", "Task 5 handoff requires a verified S3 result artifact"
        )
    bucket, _, key = event["uri"][5:].partition("/")
    if not bucket or not key or bucket != _required("GODS_MLOPS_S3_BUCKET"):
        raise PreparationReviewHandoffError(
            "preparation_assignment_artifact_invalid", "result artifact is outside the configured Task 6 bucket"
        )
    objects = DatasetObjectStore(
        endpoint_url=_required("GODS_MLOPS_S3_ENDPOINT_URL"),
        access_key=_required("GODS_MLOPS_S3_ACCESS_KEY"),
        secret_key=_required("GODS_MLOPS_S3_SECRET_KEY"),
        bucket=bucket,
        region=os.environ.get("GODS_MLOPS_S3_REGION", "us-east-1"),
    )
    try:
        return objects.read_source(
            object_key=key,
            sha256_digest=str(event["sha256"]),
            size_bytes=int(event["size_bytes"]),
        )
    except Exception as error:
        raise PreparationReviewHandoffError(
            "preparation_assignment_artifact_invalid", "committed result artifact failed Task 6 hash/size validation"
        ) from error


def _verify_document(job: dict[str, Any], document: Any, model_id: str, model_revision: str, kind: str) -> None:
    drafts = document.get("drafts") if isinstance(document, dict) else None
    drafts_are_typed = isinstance(drafts, list) and all(isinstance(item, dict) for item in drafts)
    drafts_have_required_content = drafts_are_typed and all(
        isinstance(item.get("caption"), str) and item["caption"].strip()
        if kind == "caption_drafts"
        else isinstance(item.get("detections"), list)
        for item in drafts
    )
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("model_kind") != job["model_kind"]
        or document.get("model_id") != model_id
        or document.get("model_revision") != model_revision
        or document.get("config_version") != job["config_version"]
        or document.get("input_id") != job["input_id"]
        or document.get("input_sha256") != job["input_sha256"].strip()
        or not drafts_are_typed
        or not drafts_have_required_content
    ):
        raise PreparationReviewHandoffError(
            "preparation_assignment_provenance_mismatch",
            "result artifact model/input/config identity differs from the completed Task 7 job",
        )


def _prediction(
    *,
    job: dict[str, Any],
    model_id: str,
    model_revision: str,
    result_sha256: str,
    draft: dict[str, Any],
    model_kind: str,
) -> dict[str, Any]:
    model_version = (
        f"gods-mlops:{job['job_id']}:{model_id}@{model_revision[:12]}:"
        f"config-{job['config_sha256'].strip()[:12]}:result-{result_sha256[:12]}"
    )
    if model_kind == "qwen":
        caption = str(draft.get("caption", "")).strip()
        if not caption:
            raise PreparationReviewHandoffError(
                "preparation_assignment_draft_invalid", "Qwen draft caption is empty"
            )
        return {
            "model_version": model_version,
            "result": [
                {
                    "from_name": "caption",
                    "to_name": "image",
                    "type": "textarea",
                    "value": {"text": [caption]},
                }
            ],
        }
    width = _positive_dimension(draft.get("image_width"))
    height = _positive_dimension(draft.get("image_height"))
    detections = draft.get("detections")
    if not isinstance(detections, list):
        raise PreparationReviewHandoffError(
            "preparation_assignment_draft_invalid", "DETR frame draft has no detection list"
        )
    results = []
    for index, detection in enumerate(detections):
        box = detection.get("bbox_xyxy") if isinstance(detection, dict) else None
        if not isinstance(box, list) or len(box) != 4:
            raise PreparationReviewHandoffError(
                "preparation_assignment_draft_invalid", "DETR draft contains a malformed xyxy box"
            )
        x1, y1, x2, y2 = (float(value) for value in box)
        left, top = _clamp(x1, 0.0, float(width)), _clamp(y1, 0.0, float(height))
        right, bottom = _clamp(x2, left, float(width)), _clamp(y2, top, float(height))
        if right <= left or bottom <= top:
            continue
        results.append(
            {
                "id": f"person-{index}",
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "score": float(detection.get("score", 0.0)),
                "value": {
                    "x": left / width * 100.0,
                    "y": top / height * 100.0,
                    "width": (right - left) / width * 100.0,
                    "height": (bottom - top) / height * 100.0,
                    "rotation": 0,
                    "rectanglelabels": ["person"],
                },
            }
        )
    return {"model_version": model_version, "result": results}


async def _provision_with_retry(
    workflow: LabelStudioReviewWorkflow,
    *,
    revision: str,
    project_id: int,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    try:
        return await workflow.provision_task(
            revision=revision, project_id=project_id, prediction=prediction
        )
    except ReviewAssignmentConflictError:
        current = await workflow._repository.assignment_by_revision(revision)
        if current.state != "active" or current.label_studio_task_id is None:
            raise
        # A concurrent provisioner may have committed the same task between import and bind.
        return await workflow.provision_task(
            revision=revision, project_id=project_id, prediction=prediction
        )


def _verify_assignment(assignment: Any, source: dict[str, Any], model_kind: str, project_id: int) -> None:
    expected_stage = "bbox" if model_kind == "detr" else "caption"
    expected_revision = (
        str(source["revision_id"]) if source.get("revision_id") is not None else None
    )
    if (
        str(assignment.sample_id) != str(source["sample_id"])
        or assignment.stage != expected_stage
        or assignment.bbox_revision != expected_revision
        or assignment.media_object_key != source["object_key"]
        or assignment.required_bytes != int(source["object_size_bytes"])
        or assignment.state not in {"provisioning", "active", "finalized"}
    ):
        raise PreparationReviewHandoffError(
            "preparation_assignment_provenance_mismatch",
            "Task 5 assignment no longer matches the exact frame/crop and bbox revision",
        )


def _positive_dimension(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PreparationReviewHandoffError(
            "preparation_assignment_draft_invalid", "DETR frame draft dimensions must be positive integers"
        )
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value):
        raise PreparationReviewHandoffError(
            "preparation_assignment_draft_invalid", "DETR draft contains a non-finite coordinate"
        )
    return min(max(value, lower), upper)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise PreparationReviewHandoffError(
            "preparation_assignment_service_config_missing",
            f"required Task 5 handoff setting {name} is not configured",
        )
    return value
