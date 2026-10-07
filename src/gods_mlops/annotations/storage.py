"""Durable PostgreSQL storage for immutable human annotation revisions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from importlib.resources import files
from pathlib import PurePosixPath
import re
from typing import Any
from uuid import UUID, uuid4

import asyncpg

from gods_mlops.ingestion.schemas import GlobalObjectLimitError
from gods_mlops.ingestion.storage import GLOBAL_OBJECT_LIMIT

from .models import (
    ReviewAssignment,
    ReviewAssignmentConflictError,
    ReviewAssignmentNotFoundError,
    ReviewQuotaExceededError,
    assignment_from_record,
)

_REVIEW_EXCEPTION_BYTES = 100 * 1024**3
_MAX_REVIEW_MEDIA_BYTES = 20 * 1024**2
_MIGRATIONS = (
    (1, "0001_label_review.sql"),
    (2, "0002_bbox_revision_heads.sql"),
    (3, "0003_retention_events.sql"),
    (4, "0004_annotation_crops.sql"),
    (5, "0005_crop_batch_completion.sql"),
    (6, "0006_dataset_adoptions.sql"),
    (7, "0007_revisioned_relevance.sql"),
    (8, "0008_label_studio_media_state.sql"),
    (9, "0009_crop_expiry_reconciliation.sql"),
    (10, "0010_relevance_review_dependencies.sql"),
    (11, "0011_immutable_datasets.sql"),
    (12, "0012_dataset_authority_and_invalidations.sql"),
    (13, "0013_gpu_job_queue.sql"),
    (14, "0014_operator_queue_order.sql"),
    (15, "0015_operator_retry_intent_generations.sql"),
    (16, "0016_worker_artifact_deadlines.sql"),
)
_MIGRATION_RESOURCES = files("gods_mlops.migrations")


class PostgresAnnotationRepository:
    def __init__(
        self,
        *,
        database_url: str,
        review_exception_bytes: int = _REVIEW_EXCEPTION_BYTES,
    ) -> None:
        if not 1 <= review_exception_bytes <= _REVIEW_EXCEPTION_BYTES:
            raise ValueError("review_exception_bytes must be between 1 and 100 GiB")
        self._database_url = database_url
        self._review_exception_bytes = review_exception_bytes
        self._pool: asyncpg.Pool | None = None

    async def ensure_schema(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock(731940115)")
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gods_mlops_schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                for version, filename in _MIGRATIONS:
                    applied = await connection.fetchval(
                        "SELECT EXISTS (SELECT 1 FROM gods_mlops_schema_migrations WHERE version = $1)",
                        version,
                    )
                    if applied:
                        continue
                    migration_sql = _MIGRATION_RESOURCES.joinpath(filename).read_text(encoding="utf-8")
                    await connection.execute(migration_sql)
                    await connection.execute(
                        "INSERT INTO gods_mlops_schema_migrations(version) VALUES ($1)", version
                    )

    async def create_review_assignment(
        self,
        *,
        sample_id: UUID,
        stage: str,
        label_studio_task_id: int | None,
        bbox_revision: str | None,
        media_object_key: str,
        required_bytes: int,
        expected_caption_revision_id: str | None = None,
    ) -> ReviewAssignment:
        if stage not in {"bbox", "caption", "relevance"}:
            raise ValueError("review stage must be bbox, caption, or relevance")
        if expected_caption_revision_id is not None and stage != "caption":
            raise ValueError("only caption assignments can replace a caption revision")
        if (label_studio_task_id is not None and label_studio_task_id <= 0) or required_bytes <= 0:
            raise ValueError("task ID and required media bytes must be positive")
        assignment_state = "provisioning" if label_studio_task_id is None else "active"
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                sample = await connection.fetchrow(
                    """
                    SELECT state, object_key, object_size_bytes
                    FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE
                    """,
                    sample_id,
                )
                if sample is None or sample["state"] != "received":
                    raise ReviewAssignmentConflictError("only a received sample can be assigned for review")
                if stage == "bbox" and (
                    media_object_key != sample["object_key"]
                    or required_bytes != sample["object_size_bytes"]
                ):
                    raise ReviewAssignmentConflictError("bbox review must use the received frame object")
                if stage == "caption":
                    if not bbox_revision:
                        raise ReviewAssignmentConflictError("caption review requires a bbox revision")
                    crop = await connection.fetchrow(
                        """
                        SELECT bbox_revision, object_size_bytes, state, caption_state,
                               caption_revision_id
                        FROM annotation_crops
                        WHERE sample_id = $1 AND object_key = $2 FOR UPDATE
                        """,
                        sample_id,
                        media_object_key,
                    )
                    if (
                        crop is None
                        or crop["state"] != "ready"
                        or crop["bbox_revision"] != UUID(bbox_revision)
                        or required_bytes != crop["object_size_bytes"]
                    ):
                        raise ReviewAssignmentConflictError(
                            "caption review must reference a stored crop from this bbox revision"
                        )
                    if expected_caption_revision_id is not None:
                        if (
                            crop["caption_state"] != "reviewed"
                            or crop["caption_revision_id"] != UUID(expected_caption_revision_id)
                        ):
                            raise ReviewAssignmentConflictError("caption revision changed before edit")
                        await connection.execute(
                            "UPDATE annotation_crops SET caption_state = 'needs_review', updated_at = now() WHERE sample_id = $1 AND object_key = $2",
                            sample_id,
                            media_object_key,
                        )
                        await self._invalidate_pending_caption_source_reviews(
                            connection,
                            UUID(expected_caption_revision_id),
                            reason="query_caption_revision_changed",
                        )
                    elif crop["caption_state"] != "needs_review":
                        raise ReviewAssignmentConflictError(
                            "caption review must reference an unreviewed stored crop"
                        )

                await connection.execute(
                    """
                    INSERT INTO review_storage_usage (singleton, active_bytes)
                    VALUES (TRUE, 0) ON CONFLICT (singleton) DO NOTHING
                    """
                )
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                review_usage = await connection.fetchrow(
                    "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                if usage is None or review_usage is None or usage["used_bytes"] < required_bytes:
                    raise ReviewAssignmentConflictError("review media is not accounted in the object quota")
                next_review_bytes = review_usage["active_bytes"] + required_bytes
                if next_review_bytes > self._review_exception_bytes:
                    await self._record_capacity_stop(
                        sample_id=sample_id,
                        reason="active_review_quota_exceeded",
                    )
                    raise ReviewQuotaExceededError("active review media exceeds the 100 GiB exception quota")

                assignment_id = uuid4()
                revision = uuid4()
                record = await connection.fetchrow(
                    """
                    INSERT INTO review_assignments (
                        assignment_id, revision, sample_id, stage, bbox_revision,
                        label_studio_task_id, media_object_key, required_bytes, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING assignment_id, revision, sample_id, stage, bbox_revision,
                              label_studio_task_id, media_object_key, required_bytes, state
                    """,
                    assignment_id,
                    revision,
                    sample_id,
                    stage,
                    bbox_revision,
                    label_studio_task_id,
                    media_object_key,
                    required_bytes,
                    assignment_state,
                )
                if stage == "bbox":
                    head = await connection.fetchrow(
                        """
                        INSERT INTO sample_annotation_heads (
                            sample_id, current_bbox_assignment_revision
                        ) VALUES ($1, $2)
                        ON CONFLICT (sample_id) DO UPDATE
                          SET current_bbox_assignment_revision = EXCLUDED.current_bbox_assignment_revision,
                              updated_at = now()
                          WHERE sample_annotation_heads.current_bbox_assignment_revision IS NULL
                        RETURNING sample_id
                        """,
                        sample_id,
                        revision,
                    )
                    if head is None:
                        raise ReviewAssignmentConflictError("another bbox revision is already active")
                    await connection.execute(
                        "UPDATE ingestion_samples SET selected = TRUE WHERE sample_id = $1",
                        sample_id,
                    )
                await connection.execute(
                    "UPDATE review_storage_usage SET active_bytes = $1 WHERE singleton = TRUE",
                    next_review_bytes,
                )
                return assignment_from_record(record)

    async def create_model_draft_assignment(
        self,
        *,
        sample_id: UUID,
        stage: str,
        project_id: int,
        bbox_revision: str | None,
        media_object_key: str,
        required_bytes: int,
        source_sha256: str,
        model_request_key: str,
        model_version: str,
        item_id: str,
    ) -> ReviewAssignment:
        """Atomically find or prepare one source-bound model draft review assignment."""
        if stage not in {"bbox", "caption"}:
            raise ValueError("model drafts can prepare only bbox or caption review assignments")
        if project_id <= 0 or required_bytes <= 0 or not media_object_key:
            raise ValueError("model draft review requires project, source key, and positive byte count")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", source_sha256)
            or not re.fullmatch(r"[0-9a-f]{64}", model_request_key)
            or not model_version
            or len(model_version) > 255
        ):
            raise ValueError("model draft review identity or provenance is malformed")
        if stage == "bbox" and bbox_revision is not None:
            raise ReviewAssignmentConflictError("detector draft must bind the exact received frame")
        if stage == "bbox" and item_id != str(sample_id):
            raise ReviewAssignmentConflictError("detector draft item ID must be its exact frame sample ID")
        if stage == "caption" and bbox_revision is None:
            raise ReviewAssignmentConflictError("caption draft must bind its exact bbox revision")
        request_marker = sha256(model_version.encode("utf-8")).hexdigest()[:16]
        filename = f"gods-model-{model_request_key[:32]}-{request_marker}.jpg"
        revision_id = UUID(bbox_revision) if bbox_revision is not None else None
        bbox_revision_value = str(revision_id) if revision_id is not None else None
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                if stage == "bbox":
                    source = await connection.fetchrow(
                        """SELECT state,object_key,sha256,object_size_bytes
                           FROM ingestion_samples WHERE sample_id=$1 FOR UPDATE""",
                        sample_id,
                    )
                    if (
                        source is None
                        or source["state"] != "received"
                        or source["object_key"] != media_object_key
                        or source["sha256"].strip() != source_sha256
                        or source["object_size_bytes"] != required_bytes
                    ):
                        raise ReviewAssignmentConflictError("detector draft frame source changed before assignment")
                    source_sample_id = sample_id
                else:
                    try:
                        UUID(item_id)
                    except ValueError as error:
                        raise ReviewAssignmentConflictError("caption draft crop ID is malformed") from error
                    sample = await connection.fetchrow(
                        "SELECT state FROM ingestion_samples WHERE sample_id=$1 FOR UPDATE",
                        sample_id,
                    )
                    if sample is None or sample["state"] not in {"received", "expired"}:
                        raise ReviewAssignmentConflictError("caption draft parent sample is unavailable")
                    source = await connection.fetchrow(
                        """SELECT crop_id,sample_id,bbox_revision,object_key,sha256,
                                  object_size_bytes,state,caption_state,crop_set_ready
                           FROM annotation_crops WHERE crop_id=$1::uuid FOR UPDATE""",
                        item_id,
                    )
                    if (
                        source is None
                        or source["sample_id"] != sample_id
                        or source["bbox_revision"] != revision_id
                        or source["object_key"] != media_object_key
                        or source["sha256"].strip() != source_sha256
                        or source["object_size_bytes"] != required_bytes
                        or source["state"] != "ready"
                        or source["caption_state"] != "needs_review"
                        or not source["crop_set_ready"]
                    ):
                        raise ReviewAssignmentConflictError("caption draft crop or bbox revision changed before assignment")
                    source_sample_id = source["sample_id"]

                existing = await connection.fetch(
                    """SELECT assignment.assignment_id,assignment.revision,assignment.sample_id,
                              assignment.stage,assignment.bbox_revision,assignment.label_studio_task_id,
                              assignment.media_object_key,assignment.required_bytes,assignment.state,
                              media.project_id,media.upload_filename,media.sha256,media.object_size_bytes
                       FROM review_assignments AS assignment
                       JOIN label_studio_media_uploads AS media
                         ON media.assignment_revision=assignment.revision
                       WHERE assignment.sample_id=$1 AND assignment.stage=$2
                         AND assignment.media_object_key=$3
                         AND assignment.bbox_revision IS NOT DISTINCT FROM $4
                         AND media.upload_filename=$5""",
                    source_sample_id,
                    stage,
                    media_object_key,
                    bbox_revision_value,
                    filename,
                )
                if existing:
                    if len(existing) != 1:
                        raise ReviewAssignmentConflictError("model draft request key matches multiple assignments")
                    prior = existing[0]
                    if (
                        prior["project_id"] != project_id
                        or prior["sha256"].strip() != source_sha256
                        or prior["object_size_bytes"] != required_bytes
                    ):
                        raise ReviewAssignmentConflictError("model draft request provenance or project changed")
                    if prior["state"] not in {"provisioning", "active", "finalized"}:
                        raise ReviewAssignmentConflictError("model draft assignment is already terminal")
                    return assignment_from_record(prior)

                conflict = await connection.fetchrow(
                    """SELECT revision FROM review_assignments
                       WHERE sample_id=$1 AND stage=$2 AND state IN ('provisioning','active')""",
                    source_sample_id,
                    stage,
                )
                if conflict is not None:
                    raise ReviewAssignmentConflictError(
                        "another human or model review assignment is already active for this source"
                    )
                # Match Task 5's existing lock order: source row, shared quota, then
                # active-review quota. This serializes model retries with human assignment.
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton=TRUE FOR UPDATE"
                )
                if usage is None:
                    raise RuntimeError("shared object quota ledger is missing")
                if usage["used_bytes"] < required_bytes:
                    raise ReviewAssignmentConflictError("model draft source bytes are absent from the shared quota")
                next_global_bytes = usage["used_bytes"] + required_bytes
                if next_global_bytes > GLOBAL_OBJECT_LIMIT:
                    await self._record_capacity_stop(
                        sample_id=source_sample_id,
                        reason="global_object_quota_exceeded",
                        action="model_draft_review_blocked",
                    )
                    raise GlobalObjectLimitError("model draft review copy exceeds the shared one TiB quota")
                review_usage = await connection.fetchrow(
                    "SELECT active_bytes FROM review_storage_usage WHERE singleton=TRUE FOR UPDATE"
                )
                if review_usage is None:
                    raise RuntimeError("active review quota ledger is missing")
                next_review_bytes = review_usage["active_bytes"] + 2 * required_bytes
                if next_review_bytes > self._review_exception_bytes:
                    await self._record_capacity_stop(
                        sample_id=source_sample_id,
                        reason="active_review_quota_exceeded",
                        action="model_draft_review_blocked",
                    )
                    raise ReviewQuotaExceededError("model draft review exceeds the 100 GiB active-review quota")

                assignment_id = uuid4()
                new_revision = uuid4()
                record = await connection.fetchrow(
                    """INSERT INTO review_assignments(
                           assignment_id,revision,sample_id,stage,bbox_revision,
                           label_studio_task_id,media_object_key,required_bytes,state
                       ) VALUES($1,$2,$3,$4,$5,NULL,$6,$7,'provisioning')
                       RETURNING assignment_id,revision,sample_id,stage,bbox_revision,
                                 label_studio_task_id,media_object_key,required_bytes,state""",
                    assignment_id,
                    new_revision,
                    source_sample_id,
                    stage,
                    bbox_revision_value,
                    media_object_key,
                    required_bytes,
                )
                if stage == "bbox":
                    head = await connection.fetchrow(
                        """INSERT INTO sample_annotation_heads(sample_id,current_bbox_assignment_revision)
                           VALUES($1,$2)
                           ON CONFLICT(sample_id) DO UPDATE SET
                             current_bbox_assignment_revision=EXCLUDED.current_bbox_assignment_revision,
                             updated_at=now()
                           WHERE sample_annotation_heads.current_bbox_assignment_revision IS NULL
                           RETURNING sample_id""",
                        source_sample_id,
                        new_revision,
                    )
                    if head is None:
                        raise ReviewAssignmentConflictError("bbox assignment head changed before model draft creation")
                    await connection.execute(
                        "UPDATE ingestion_samples SET selected=TRUE WHERE sample_id=$1", source_sample_id
                    )
                await connection.execute(
                    "UPDATE ingestion_storage_usage SET used_bytes=$1 WHERE singleton=TRUE",
                    next_global_bytes,
                )
                await connection.execute(
                    """UPDATE review_storage_usage SET active_bytes=$1 WHERE singleton=TRUE""",
                    next_review_bytes,
                )
                await connection.execute(
                    """INSERT INTO label_studio_media_uploads(
                           assignment_revision,project_id,upload_filename,sha256,
                           object_size_bytes,state
                       ) VALUES($1,$2,$3,$4,$5,'reserved')""",
                    new_revision,
                    project_id,
                    filename,
                    source_sha256,
                    required_bytes,
                )
                return assignment_from_record(record)

    async def mark_bbox_edit(
        self,
        *,
        sample_id: UUID,
        expected_revision: str,
    ) -> ReviewAssignment:
        """Supersede the active bbox revision and atomically open its replacement."""
        old_revision = UUID(expected_revision)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                sample = await connection.fetchrow(
                    """
                    SELECT state, object_key, object_size_bytes
                    FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE
                    """,
                    sample_id,
                )
                if sample is None or sample["state"] != "received":
                    raise ReviewAssignmentConflictError("bbox edits require a live received frame")
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                review_usage = await connection.fetchrow(
                    "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                old = await connection.fetchrow(
                    """
                    SELECT assignment_id, revision, sample_id, stage, bbox_revision,
                           label_studio_task_id, media_object_key, required_bytes, state
                    FROM review_assignments WHERE sample_id = $1 AND revision = $2 FOR UPDATE
                    """,
                    sample_id,
                    old_revision,
                )
                head = await connection.fetchrow(
                    "SELECT current_bbox_assignment_revision FROM sample_annotation_heads WHERE sample_id = $1 FOR UPDATE",
                    sample_id,
                )
                if (
                    old is None
                    or old["stage"] != "bbox"
                    or old["state"] != "active"
                    or head is None
                    or head["current_bbox_assignment_revision"] != old_revision
                ):
                    raise ReviewAssignmentConflictError("bbox revision changed before edit replacement")
                if usage is None or usage["used_bytes"] < sample["object_size_bytes"] or review_usage is None:
                    raise ReviewAssignmentConflictError("frame bytes are not accounted for review")
                required_bytes = sample["object_size_bytes"]
                next_review_bytes = review_usage["active_bytes"] - old["required_bytes"] + required_bytes
                if next_review_bytes < 0:
                    raise RuntimeError("review quota accounting became negative")
                if next_review_bytes > self._review_exception_bytes:
                    raise ReviewQuotaExceededError("active review media exceeds the 100 GiB exception quota")

                new_assignment_id = uuid4()
                new_revision = uuid4()
                await connection.execute(
                    "UPDATE review_assignments SET state = 'superseded', finished_at = now() WHERE revision = $1",
                    old_revision,
                )
                record = await connection.fetchrow(
                    """
                    INSERT INTO review_assignments (
                        assignment_id, revision, sample_id, stage, bbox_revision,
                        label_studio_task_id, media_object_key, required_bytes, state
                    ) VALUES ($1, $2, $3, 'bbox', $4, NULL, $5, $6, 'provisioning')
                    RETURNING assignment_id, revision, sample_id, stage, bbox_revision,
                              label_studio_task_id, media_object_key, required_bytes, state
                    """,
                    new_assignment_id,
                    new_revision,
                    sample_id,
                    old["bbox_revision"],
                    sample["object_key"],
                    required_bytes,
                )
                await connection.execute(
                    "UPDATE sample_annotation_heads SET current_bbox_assignment_revision = $2, updated_at = now() WHERE sample_id = $1",
                    sample_id,
                    new_revision,
                )
                await connection.execute(
                    "UPDATE review_storage_usage SET active_bytes = $1 WHERE singleton = TRUE",
                    next_review_bytes,
                )
                return assignment_from_record(record)

    async def close_review_assignment(
        self,
        *,
        sample_id: UUID,
        revision: str,
        outcome: str,
        reason: str,
    ) -> dict[str, Any]:
        if outcome not in {"cancelled", "rejected"}:
            raise ValueError("review outcome must be cancelled or rejected")
        if not reason.strip() or len(reason) > 128:
            raise ValueError("review close reason must contain 1 to 128 characters")
        revision_id = UUID(revision)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                sample = await connection.fetchrow(
                    "SELECT state FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE",
                    sample_id,
                )
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                review_usage = await connection.fetchrow(
                    "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                assignment = await connection.fetchrow(
                    """
                    SELECT stage, state, media_object_key, required_bytes
                    FROM review_assignments WHERE sample_id = $1 AND revision = $2 FOR UPDATE
                    """,
                    sample_id,
                    revision_id,
                )
                if sample is None or assignment is None:
                    raise ReviewAssignmentNotFoundError("review assignment does not exist")
                if assignment["state"] != "active" or review_usage is None:
                    raise ReviewAssignmentConflictError("only an active review can be closed")
                if review_usage["active_bytes"] < assignment["required_bytes"]:
                    raise RuntimeError("review quota accounting became negative")
                if assignment["stage"] == "bbox":
                    await connection.execute(
                        """
                        UPDATE sample_annotation_heads SET current_bbox_assignment_revision = NULL,
                            updated_at = now()
                        WHERE sample_id = $1 AND current_bbox_assignment_revision = $2
                        """,
                        sample_id,
                        revision_id,
                    )
                    protected = await connection.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM dataset_adoptions
                            WHERE sample_id = $1 AND target = 'detr'
                        )
                        """,
                        sample_id,
                    )
                    await connection.execute(
                        "UPDATE ingestion_samples SET selected = $2 WHERE sample_id = $1",
                        sample_id,
                        protected,
                    )
                elif assignment["stage"] == "caption":
                    updated = await connection.execute(
                        """
                        UPDATE annotation_crops SET caption_state = $3, updated_at = now()
                        WHERE sample_id = $1 AND object_key = $2
                          AND state = 'ready' AND caption_state = 'needs_review'
                        """,
                        sample_id,
                        assignment["media_object_key"],
                        outcome,
                    )
                    if updated != "UPDATE 1":
                        raise ReviewAssignmentConflictError("caption crop changed before review close")
                await connection.execute(
                    "UPDATE review_assignments SET state = $2, finished_at = now() WHERE revision = $1",
                    revision_id,
                    outcome,
                )
                await connection.execute(
                    """
                    UPDATE review_storage_usage SET active_bytes = active_bytes - $1
                    WHERE singleton = TRUE AND active_bytes >= $1
                    """,
                    assignment["required_bytes"],
                )
                await connection.execute(
                    """
                    INSERT INTO retention_policy_events (sample_id, action, reason, details)
                    VALUES ($1, $2, $3, $4::jsonb)
                    """,
                    sample_id,
                    outcome,
                    reason,
                    json.dumps({"stage": assignment["stage"], "assignment_revision": revision}),
                )
                return {
                    "sample_id": str(sample_id),
                    "revision": revision,
                    "stage": assignment["stage"],
                    "state": outcome,
                    "reason": reason,
                }

    async def bbox_crop_source(self, *, sample_id: UUID, bbox_revision: UUID) -> dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            record = await connection.fetchrow(
                """
                SELECT sample.state, sample.object_key, sample.sha256,
                       revision.result, revision.provenance,
                       revision.label_studio_task_id, revision.label_studio_annotation_id,
                       head.latest_bbox_revision, head.current_bbox_assignment_revision
                FROM annotation_revisions AS revision
                JOIN ingestion_samples AS sample ON sample.sample_id = revision.sample_id
                JOIN sample_annotation_heads AS head ON head.sample_id = sample.sample_id
                WHERE revision.sample_id = $1
                  AND revision.annotation_revision_id = $2
                  AND revision.stage = 'bbox'
                """,
                sample_id,
                bbox_revision,
            )
        if record is None:
            raise ReviewAssignmentNotFoundError("bbox annotation revision does not exist")
        if record["state"] != "received":
            raise ReviewAssignmentConflictError("the source frame has expired and cannot produce new crops")
        if (
            record["latest_bbox_revision"] != bbox_revision
            or record["current_bbox_assignment_revision"] is not None
        ):
            raise ReviewAssignmentConflictError("bbox revision changed before crop creation")
        return {
            "sample_id": str(sample_id),
            "object_key": record["object_key"],
            "sha256": record["sha256"].strip(),
            "result": _json_value(record["result"]),
            "annotation_provenance": _json_value(record["provenance"]),
            "label_studio_task_id": record["label_studio_task_id"],
            "label_studio_annotation_id": record["label_studio_annotation_id"],
        }

    async def crops_for_revision(self, *, sample_id: UUID, bbox_revision: UUID) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT crop_id, sample_id, bbox_revision, region_index, region_id,
                       object_key, sha256, object_size_bytes, state, caption_state,
                       caption_revision_id, provenance, parent_available, regeneration_available,
                       crop_set_ready
                FROM annotation_crops
                WHERE sample_id = $1 AND bbox_revision = $2
                ORDER BY region_index
                """,
                sample_id,
                bbox_revision,
            )
        return [_crop_dict(row) for row in rows]

    async def reserve_crops(
        self,
        *,
        sample_id: UUID,
        bbox_revision: UUID,
        specs: list[dict[str, Any]],
        source: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not specs:
            return []
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                sample = await connection.fetchrow(
                    "SELECT state, object_key, sha256 FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE",
                    sample_id,
                )
                head = await connection.fetchrow(
                    """
                    SELECT latest_bbox_revision, current_bbox_assignment_revision
                    FROM sample_annotation_heads WHERE sample_id = $1 FOR UPDATE
                    """,
                    sample_id,
                )
                if (
                    sample is None
                    or sample["state"] != "received"
                    or head is None
                    or head["latest_bbox_revision"] != bbox_revision
                    or head["current_bbox_assignment_revision"] is not None
                ):
                    raise ReviewAssignmentConflictError("bbox revision changed before crop reservation")
                await self._invalidate_pending_bbox_crop_reviews(
                    connection,
                    sample_id=sample_id,
                    current_bbox_revision=bbox_revision,
                )
                existing = await connection.fetch(
                    """
                    SELECT crop_id, sample_id, bbox_revision, region_index, region_id,
                           object_key, sha256, object_size_bytes, state, caption_state,
                           caption_revision_id, provenance, parent_available, regeneration_available,
                           crop_set_ready
                    FROM annotation_crops WHERE sample_id = $1 AND bbox_revision = $2
                    ORDER BY region_index FOR UPDATE
                    """,
                    sample_id,
                    bbox_revision,
                )
                if existing:
                    if len(existing) != len(specs):
                        raise ReviewAssignmentConflictError("crop set already exists with different geometry")
                    for row, spec in zip(existing, specs, strict=True):
                        if (
                            row["region_index"] != spec["region_index"]
                            or row["region_id"] != spec["region_id"]
                            or row["sha256"].strip() != spec["sha256"]
                            or row["object_size_bytes"] != len(spec["image"])
                            or row["state"] == "deleted"
                        ):
                            raise ReviewAssignmentConflictError("crop revision is immutable")
                    return [_crop_dict(row) for row in existing]

                total_bytes = sum(len(spec["image"]) for spec in specs)
                usage = await connection.fetchrow(
                    """
                    UPDATE ingestion_storage_usage
                    SET used_bytes = used_bytes + $1
                    WHERE singleton = TRUE AND used_bytes + $1 <= $2
                    RETURNING used_bytes
                    """,
                    total_bytes,
                    GLOBAL_OBJECT_LIMIT,
                )
                if usage is None:
                    await self._record_capacity_stop(
                        sample_id=sample_id,
                        reason="global_object_quota_exceeded",
                        action="crop_blocked",
                    )
                    raise GlobalObjectLimitError("crop bytes exceed the shared one TiB object quota")

                reserved: list[dict[str, Any]] = []
                for spec in specs:
                    crop_id = uuid4()
                    object_key = f"crops/{sample_id}/{bbox_revision}/{crop_id}.jpg"
                    provenance = {
                        "frame_sample_id": str(sample_id),
                        "frame_object_key": source["object_key"],
                        "frame_sha256": sample["sha256"].strip(),
                        "bbox_revision": str(bbox_revision),
                        "bbox_result_sha256": source["annotation_provenance"]["result_sha256"],
                        "label_studio_task_id": source["label_studio_task_id"],
                        "label_studio_annotation_id": source["label_studio_annotation_id"],
                        "region_id": spec["region_id"],
                        "region_index": spec["region_index"],
                    }
                    record = await connection.fetchrow(
                        """
                        INSERT INTO annotation_crops (
                            crop_id, sample_id, bbox_revision, region_index, region_id,
                            object_key, sha256, object_size_bytes, state, caption_state, provenance
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending', 'needs_review', $9::jsonb)
                        RETURNING crop_id, sample_id, bbox_revision, region_index, region_id,
                                  object_key, sha256, object_size_bytes, state, caption_state,
                                  caption_revision_id, provenance, parent_available, regeneration_available,
                                  crop_set_ready
                        """,
                        crop_id,
                        sample_id,
                        bbox_revision,
                        spec["region_index"],
                        spec["region_id"],
                        object_key,
                        spec["sha256"],
                        len(spec["image"]),
                        json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                    )
                    reserved.append(_crop_dict(record))
                return reserved

    async def mark_crop_ready(self, crop_id: UUID) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                updated = await connection.fetchrow(
                    """
                    UPDATE annotation_crops SET state = 'ready', updated_at = now()
                    WHERE crop_id = $1 AND state IN ('pending', 'ready')
                    RETURNING sample_id, bbox_revision
                    """,
                    crop_id,
                )
                if updated is None:
                    raise ReviewAssignmentConflictError("crop reservation is no longer writable")
                state = await connection.fetchrow(
                    """
                    SELECT count(*) AS total,
                           count(*) FILTER (WHERE state = 'ready') AS ready
                    FROM annotation_crops WHERE sample_id = $1 AND bbox_revision = $2
                    """,
                    updated["sample_id"],
                    updated["bbox_revision"],
                )
                if state["total"] > 0 and state["ready"] == state["total"]:
                    await connection.execute(
                        """
                        UPDATE annotation_crops SET crop_set_ready = TRUE, updated_at = now()
                        WHERE sample_id = $1 AND bbox_revision = $2
                        """,
                        updated["sample_id"],
                        updated["bbox_revision"],
                    )

    async def claim_unadopted_crops(
        self,
        *,
        now: datetime,
        sample_id: UUID | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if now.tzinfo is None:
            raise ValueError("crop retention time must be timezone-aware")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                rows = await connection.fetch(
                    """
                    SELECT crop.crop_id, crop.sample_id, crop.object_key,
                           crop.sha256, crop.object_size_bytes, crop.state,
                           crop.bbox_revision
                    FROM annotation_crops AS crop
                    JOIN ingestion_samples AS sample ON sample.sample_id = crop.sample_id
                    WHERE (crop.state = 'purge_pending'
                           OR (crop.state = 'pending' AND
                               sample.state IN ('purge_pending', 'expired'))
                           OR (crop.state = 'ready' AND (
                               sample.state IN ('purge_pending', 'expired')
                               OR crop.caption_state IN ('cancelled', 'rejected')
                           )))
                      AND ($1::uuid IS NULL OR crop.sample_id = $1)
                      AND NOT EXISTS (
                          SELECT 1 FROM dataset_adoptions AS adoption
                          WHERE adoption.target = 'clip' AND adoption.crop_id = crop.crop_id
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM review_assignments AS assignment
                          WHERE assignment.sample_id = crop.sample_id
                            AND assignment.stage = 'caption'
                            AND assignment.state IN ('provisioning', 'active')
                            AND assignment.media_object_key = crop.object_key
                      )
                    ORDER BY crop.created_at, crop.crop_id
                    LIMIT $2 FOR UPDATE OF crop SKIP LOCKED
                    """,
                    sample_id,
                    limit,
                )
                claimed: list[dict[str, Any]] = []
                for row in rows:
                    if row["state"] in {"pending", "ready"}:
                        await connection.execute(
                            "UPDATE annotation_crops SET state = 'purge_pending', updated_at = now() WHERE crop_id = $1 AND state = $2",
                            row["crop_id"],
                            row["state"],
                        )
                    claimed.append(
                        {
                            "crop_id": row["crop_id"],
                            "sample_id": row["sample_id"],
                            "object_key": row["object_key"],
                            "sha256": row["sha256"].strip(),
                            "object_size_bytes": row["object_size_bytes"],
                            "bbox_revision": row["bbox_revision"],
                        }
                    )
                return claimed

    async def finish_crop_expiry(self, crop_id: UUID) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT sample_id, bbox_revision, object_size_bytes, quota_released
                    FROM annotation_crops
                    WHERE crop_id = $1 AND state = 'purge_pending'
                    FOR UPDATE
                    """,
                    crop_id,
                )
                if row is None:
                    return False
                if not row["quota_released"]:
                    usage = await connection.fetchrow(
                        """
                        UPDATE ingestion_storage_usage SET used_bytes = used_bytes - $1
                        WHERE singleton = TRUE AND used_bytes >= $1 RETURNING used_bytes
                        """,
                        row["object_size_bytes"],
                    )
                    if usage is None:
                        raise RuntimeError("object quota accounting is inconsistent during crop retention")
                await connection.execute(
                    """
                    UPDATE annotation_crops
                    SET state = 'deleted', quota_released = TRUE,
                        crop_set_ready = FALSE, updated_at = now()
                    WHERE crop_id = $1
                    """,
                    crop_id,
                )
                await connection.execute(
                    """
                    UPDATE annotation_crops SET crop_set_ready = FALSE, updated_at = now()
                    WHERE sample_id = $1 AND bbox_revision = $2
                    """,
                    row["sample_id"],
                    row["bbox_revision"],
                )
                return True

    async def mark_crop_cleanup_pending(self, crop_id: UUID) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    "SELECT state, quota_released FROM annotation_crops WHERE crop_id = $1 FOR UPDATE",
                    crop_id,
                )
                if row is None:
                    raise ReviewAssignmentNotFoundError("crop cleanup reservation does not exist")
                if row["state"] == "purge_pending":
                    return
                if row["state"] != "deleted":
                    raise ReviewAssignmentConflictError("late crop cleanup is not eligible for reconciliation")
                # Older tombstones reached deleted only after releasing their byte reservation.
                await connection.execute(
                    """
                    UPDATE annotation_crops
                    SET state = 'purge_pending', quota_released = TRUE, updated_at = now()
                    WHERE crop_id = $1
                    """,
                    crop_id,
                )

    async def adopt_for_dataset(
        self,
        *,
        dataset_version: str,
        target: str,
        sample_id: UUID,
        bbox_revision: UUID,
        crop_id: UUID | None = None,
        caption_revision: UUID | None = None,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                return await self.adopt_for_dataset_in_transaction(
                    connection,
                    dataset_version=dataset_version,
                    target=target,
                    sample_id=sample_id,
                    bbox_revision=bbox_revision,
                    crop_id=crop_id,
                    caption_revision=caption_revision,
                )

    async def adopt_for_dataset_in_transaction(
        self,
        connection: asyncpg.Connection,
        *,
        dataset_version: str,
        target: str,
        sample_id: UUID,
        bbox_revision: UUID,
        crop_id: UUID | None = None,
        caption_revision: UUID | None = None,
    ) -> dict[str, Any]:
        """Adopt one current annotation source within its caller's transaction.

        A dataset publisher uses this seam after it has acquired all sample, annotation-head,
        and crop locks. The adoption, publication intent, and any relevance freeze can then
        commit or roll back together without creating a retention race.
        """
        if not dataset_version.strip() or len(dataset_version) > 255:
            raise ValueError("dataset_version must contain 1 to 255 characters")
        if target not in {"detr", "clip"}:
            raise ValueError("dataset target must be detr or clip")
        if (target == "clip") != (crop_id is not None and caption_revision is not None):
            raise ValueError("CLIP adoption requires one crop and its reviewed caption")
        sample = await connection.fetchrow(
            "SELECT state FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE",
            sample_id,
        )
        if sample is None:
            raise ReviewAssignmentNotFoundError("dataset source frame does not exist")
        if target == "detr":
            if sample["state"] != "received":
                raise ReviewAssignmentConflictError("DETR adoption requires a retained source frame")
            head = await connection.fetchrow(
                """
                SELECT latest_bbox_revision, current_bbox_assignment_revision
                FROM sample_annotation_heads WHERE sample_id = $1 FOR UPDATE
                """,
                sample_id,
            )
            if (
                head is None
                or head["latest_bbox_revision"] != bbox_revision
                or head["current_bbox_assignment_revision"] is not None
            ):
                raise ReviewAssignmentConflictError("DETR adoption must use the latest finalized bbox")
            prior = await connection.fetchrow(
                """
                SELECT adoption_id, dataset_version, target, sample_id,
                       annotation_revision_id, crop_id, caption_revision_id
                FROM dataset_adoptions
                WHERE dataset_version = $1 AND target = 'detr' AND sample_id = $2
                """,
                dataset_version,
                sample_id,
            )
            if prior is not None:
                return _adoption_dict(prior)
            exists = await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM annotation_revisions
                    WHERE sample_id = $1 AND annotation_revision_id = $2 AND stage = 'bbox'
                )
                """,
                sample_id,
                bbox_revision,
            )
            if not exists:
                raise ReviewAssignmentNotFoundError("finalized DETR annotation revision is missing")
            record = await connection.fetchrow(
                """
                INSERT INTO dataset_adoptions (
                    adoption_id, dataset_version, target, sample_id, annotation_revision_id
                ) VALUES ($1, $2, 'detr', $3, $4)
                RETURNING adoption_id, dataset_version, target, sample_id,
                          annotation_revision_id, crop_id, caption_revision_id
                """,
                uuid4(),
                dataset_version,
                sample_id,
                bbox_revision,
            )
            await connection.execute(
                "UPDATE ingestion_samples SET selected = TRUE WHERE sample_id = $1",
                sample_id,
            )
            return _adoption_dict(record)

        crop = await connection.fetchrow(
            """
            SELECT bbox_revision, state, caption_state, caption_revision_id, crop_set_ready
            FROM annotation_crops WHERE sample_id = $1 AND crop_id = $2 FOR UPDATE
            """,
            sample_id,
            crop_id,
        )
        if (
            crop is None
            or crop["bbox_revision"] != bbox_revision
            or crop["state"] != "ready"
            or not crop["crop_set_ready"]
            or crop["caption_state"] != "reviewed"
            or crop["caption_revision_id"] != caption_revision
        ):
            raise ReviewAssignmentConflictError(
                "CLIP adoption requires the exact reviewed crop revision"
            )
        exists = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM annotation_revisions
                WHERE sample_id = $1 AND annotation_revision_id = $2 AND stage = 'caption'
            )
            """,
            sample_id,
            caption_revision,
        )
        if not exists:
            raise ReviewAssignmentNotFoundError("reviewed caption revision is missing")
        prior = await connection.fetchrow(
            """
            SELECT adoption_id, dataset_version, target, sample_id,
                   annotation_revision_id, crop_id, caption_revision_id
            FROM dataset_adoptions
            WHERE dataset_version = $1 AND target = 'clip' AND crop_id = $2
            """,
            dataset_version,
            crop_id,
        )
        if prior is not None:
            return _adoption_dict(prior)
        record = await connection.fetchrow(
            """
            INSERT INTO dataset_adoptions (
                adoption_id, dataset_version, target, sample_id,
                annotation_revision_id, crop_id, caption_revision_id
            ) VALUES ($1, $2, 'clip', $3, $4, $5, $6)
            RETURNING adoption_id, dataset_version, target, sample_id,
                      annotation_revision_id, crop_id, caption_revision_id
            """,
            uuid4(),
            dataset_version,
            sample_id,
            bbox_revision,
            crop_id,
            caption_revision,
        )
        return _adoption_dict(record)

    async def record_relevance_judgments(
        self,
        *,
        query_id: str,
        query_text: str,
        query_revision: str,
        gallery: list[dict[str, str]],
        judgments: list[dict[str, str]],
        provenance: dict[str, Any],
        query_source_caption_revision_id: str | None = None,
    ) -> dict[str, Any]:
        """Prepare and immediately freeze a submitted human matrix through the review seam."""
        if not query_id or len(query_id) > 255 or not query_text.strip() or len(query_text) > 4096:
            raise ValueError("query ID and exact query text are required within the stored bounds")
        if not query_revision or len(query_revision) > 255:
            raise ValueError("query_revision must contain 1 to 255 characters")
        if not gallery or not isinstance(provenance, dict):
            raise ValueError("a complete gallery and human provenance are required")
        if provenance.get("source") != "label_studio" or not provenance.get("label_studio_annotation_id"):
            raise ValueError("relevance truth must identify its submitted Label Studio annotation")
        if not (provenance.get("reviewer_id") or provenance.get("completed_by")):
            raise ValueError("relevance truth must record the human reviewer")

        gallery_ids: set[UUID] = set()
        for item in gallery:
            if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
                raise ValueError("each gallery crop requires its ID and SHA-256")
            crop_id = UUID(item["crop_id"])
            if crop_id in gallery_ids:
                raise ValueError("gallery contains a duplicate crop ID")
            gallery_ids.add(crop_id)
        judgment_by_id: dict[UUID, str] = {}
        for item in judgments:
            if not isinstance(item, dict):
                raise ValueError("each relevance judgment must be an object")
            crop_id = UUID(item["crop_id"])
            label = item.get("judgment")
            if label not in {"relevant", "not_relevant", "uncertain"}:
                raise ValueError("judgment must be relevant, not_relevant, or uncertain")
            if crop_id in judgment_by_id:
                raise ValueError("relevance matrix contains a duplicate crop judgment")
            judgment_by_id[crop_id] = label
        if set(judgment_by_id) != gallery_ids:
            raise ValueError("every selected gallery crop must have exactly one human judgment")
        labels = list(judgment_by_id.values())
        if "relevant" not in labels or "not_relevant" not in labels:
            raise ValueError("evaluation truth requires at least one positive and one negative")

        draft = await self.prepare_relevance_review(
            query_id=query_id,
            query_text=query_text,
            query_revision=query_revision,
            gallery=gallery,
            query_source_caption_revision_id=query_source_caption_revision_id,
        )
        return await self.freeze_relevance_review(
            review_id=draft["review_id"],
            judgments=judgments,
            provenance=provenance,
        )

    async def prepare_relevance_review(
        self,
        *,
        query_id: str,
        query_text: str,
        query_revision: str,
        gallery: list[dict[str, str]],
        query_source_caption_revision_id: str | None = None,
    ) -> dict[str, Any]:
        """Snapshot exact gallery and optional caption-source dependencies for later freeze."""
        if not query_id or len(query_id) > 255 or not query_text.strip() or len(query_text) > 4096:
            raise ValueError("query ID and exact query text are required within the stored bounds")
        if not query_revision or len(query_revision) > 255:
            raise ValueError("query_revision must contain 1 to 255 characters")
        if not gallery:
            raise ValueError("a complete gallery is required before relevance review")
        source_caption_id = UUID(query_source_caption_revision_id) if query_source_caption_revision_id else None

        gallery_by_id: dict[UUID, str] = {}
        for item in gallery:
            if not isinstance(item, dict) or not isinstance(item.get("sha256"), str):
                raise ValueError("each gallery crop requires its ID and SHA-256")
            crop_id = UUID(item["crop_id"])
            crop_sha = item["sha256"].lower()
            if len(crop_sha) != 64 or any(char not in "0123456789abcdef" for char in crop_sha):
                raise ValueError("gallery crop SHA-256 must be lowercase hexadecimal")
            if crop_id in gallery_by_id:
                raise ValueError("gallery contains a duplicate crop ID")
            gallery_by_id[crop_id] = crop_sha
        gallery_records = [
            {"crop_id": str(crop_id), "sha256": gallery_by_id[crop_id]}
            for crop_id in sorted(gallery_by_id, key=str)
        ]
        query_sha256 = sha256(
            json.dumps(
                {"query_id": query_id, "query_revision": query_revision, "query_text": query_text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        gallery_sha256 = sha256(
            json.dumps(gallery_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                stale_reason = await self._lock_current_relevance_dependencies(
                    connection,
                    gallery=gallery_records,
                    query_source_caption_revision_id=source_caption_id,
                )
                if stale_reason is not None:
                    raise ReviewAssignmentConflictError(stale_reason)
                review_id = uuid4()
                record = await connection.fetchrow(
                    """
                    INSERT INTO relevance_matrix_review_drafts (
                        review_id, query_id, query_text, query_revision,
                        query_source_caption_revision_id, query_sha256, gallery_sha256, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'pending')
                    RETURNING review_id, query_id, query_text, query_revision,
                              query_source_caption_revision_id, query_sha256, gallery_sha256,
                              state, invalidation_reason, frozen_relevance_revision_id, created_at
                    """,
                    review_id,
                    query_id,
                    query_text,
                    query_revision,
                    source_caption_id,
                    query_sha256,
                    gallery_sha256,
                )
                await connection.executemany(
                    """
                    INSERT INTO relevance_matrix_review_crops (review_id, crop_id, crop_sha256)
                    VALUES ($1, $2, $3)
                    """,
                    [
                        (review_id, crop_id, gallery_by_id[crop_id])
                        for crop_id in sorted(gallery_by_id, key=str)
                    ],
                )
        return {
            "review_id": str(record["review_id"]),
            "query_id": record["query_id"],
            "query_text": record["query_text"],
            "query_revision": record["query_revision"],
            "query_source_caption_revision_id": (
                str(record["query_source_caption_revision_id"])
                if record["query_source_caption_revision_id"] is not None
                else None
            ),
            "query_sha256": record["query_sha256"].strip(),
            "gallery_sha256": record["gallery_sha256"].strip(),
            "state": record["state"],
            "invalidation_reason": record["invalidation_reason"],
            "frozen_relevance_revision_id": None,
            "created_at": record["created_at"].isoformat(),
            "gallery": gallery_records,
        }

    async def relevance_review(self, review_id: str) -> dict[str, Any]:
        """Return current-selection state, invalidating stale pending dependencies on read."""
        review_uuid = UUID(review_id)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                review = await connection.fetchrow(
                    "SELECT * FROM relevance_matrix_review_drafts WHERE review_id = $1",
                    review_uuid,
                )
                if review is None:
                    raise ReviewAssignmentNotFoundError("relevance review draft does not exist")
                gallery_rows = await connection.fetch(
                    """
                    SELECT crop_id, crop_sha256 FROM relevance_matrix_review_crops
                    WHERE review_id = $1 ORDER BY crop_id
                    """,
                    review_uuid,
                )
                gallery = [
                    {"crop_id": str(row["crop_id"]), "sha256": row["crop_sha256"].strip()}
                    for row in gallery_rows
                ]
                if review["state"] == "pending":
                    stale_reason = await self._lock_current_relevance_dependencies(
                        connection,
                        gallery=gallery,
                        query_source_caption_revision_id=review["query_source_caption_revision_id"],
                    )
                    current = await connection.fetchrow(
                        "SELECT * FROM relevance_matrix_review_drafts WHERE review_id = $1 FOR UPDATE",
                        review_uuid,
                    )
                    if current["state"] == "pending" and stale_reason is not None:
                        await connection.execute(
                            """
                            UPDATE relevance_matrix_review_drafts
                            SET state = 'needs_review', invalidation_reason = $2, updated_at = now()
                            WHERE review_id = $1 AND state = 'pending'
                            """,
                            review_uuid,
                            stale_reason,
                        )
                        review = await connection.fetchrow(
                            "SELECT * FROM relevance_matrix_review_drafts WHERE review_id = $1",
                            review_uuid,
                        )
                    else:
                        review = current
        return _relevance_review_dict(review, gallery)

    async def freeze_relevance_review(
        self,
        *,
        review_id: str,
        judgments: list[dict[str, str]],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        """Freeze a pending matrix in its own transaction for non-publication callers."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                result = await self.freeze_relevance_review_in_transaction(
                    connection,
                    review_id=review_id,
                    judgments=judgments,
                    provenance=provenance,
                )
        if result["state"] == "needs_review":
            raise ReviewAssignmentConflictError(result["invalidation_reason"])
        if result.get("revision") is not None:
            return result["revision"]
        return await self.relevance_revision(result["frozen_relevance_revision_id"])

    async def freeze_relevance_review_in_transaction(
        self,
        connection: asyncpg.Connection,
        *,
        review_id: str,
        judgments: list[dict[str, str]],
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        """Freeze truth on a caller-owned publication transaction and return its outcome.

        Lock order is sample rows by UUID, annotation heads by sample UUID, crop rows by
        crop UUID, then this draft row. Caption and bbox edits use the same order before
        invalidating drafts. A publisher must call this helper inside its transaction,
        publish only when the returned state is ``frozen``, and commit its publication rows
        together with the returned frozen revision. A ``needs_review`` result is a durable
        stop signal; the caller must not publish that draft.
        """
        if not isinstance(provenance, dict):
            raise ValueError("human provenance is required")
        if provenance.get("source") != "label_studio" or not provenance.get("label_studio_annotation_id"):
            raise ValueError("relevance truth must identify its submitted Label Studio annotation")
        if not (provenance.get("reviewer_id") or provenance.get("completed_by")):
            raise ValueError("relevance truth must record the human reviewer")

        review_uuid = UUID(review_id)
        review = await connection.fetchrow(
            "SELECT * FROM relevance_matrix_review_drafts WHERE review_id = $1",
            review_uuid,
        )
        if review is None:
            raise ReviewAssignmentNotFoundError("relevance review draft does not exist")
        if review["state"] == "frozen":
            return {
                "state": "frozen",
                "review_id": str(review_uuid),
                "frozen_relevance_revision_id": str(review["frozen_relevance_revision_id"]),
                "revision": None,
            }
        gallery_rows = await connection.fetch(
            """
            SELECT crop_id, crop_sha256 FROM relevance_matrix_review_crops
            WHERE review_id = $1 ORDER BY crop_id
            """,
            review_uuid,
        )
        gallery = [
            {"crop_id": str(row["crop_id"]), "sha256": row["crop_sha256"].strip()}
            for row in gallery_rows
        ]
        stale_reason = await self._lock_current_relevance_dependencies(
            connection,
            gallery=gallery,
            query_source_caption_revision_id=review["query_source_caption_revision_id"],
        )
        current = await connection.fetchrow(
            "SELECT * FROM relevance_matrix_review_drafts WHERE review_id = $1 FOR UPDATE",
            review_uuid,
        )
        if current["state"] == "frozen":
            return {
                "state": "frozen",
                "review_id": str(review_uuid),
                "frozen_relevance_revision_id": str(current["frozen_relevance_revision_id"]),
                "revision": None,
            }
        if current["state"] != "pending":
            return {
                "state": current["state"],
                "review_id": str(review_uuid),
                "invalidation_reason": current["invalidation_reason"],
                "frozen_relevance_revision_id": None,
                "revision": None,
            }
        if stale_reason is not None:
            await connection.execute(
                """
                UPDATE relevance_matrix_review_drafts
                SET state = 'needs_review', invalidation_reason = $2, updated_at = now()
                WHERE review_id = $1 AND state = 'pending'
                """,
                review_uuid,
                stale_reason,
            )
            return {
                "state": "needs_review",
                "review_id": str(review_uuid),
                "invalidation_reason": stale_reason,
                "frozen_relevance_revision_id": None,
                "revision": None,
            }

        gallery_by_id = {UUID(item["crop_id"]): item["sha256"] for item in gallery}
        judgment_by_id: dict[UUID, str] = {}
        for item in judgments:
            if not isinstance(item, dict):
                raise ValueError("each relevance judgment must be an object")
            crop_id = UUID(item["crop_id"])
            label = item.get("judgment")
            if label not in {"relevant", "not_relevant", "uncertain"}:
                raise ValueError("judgment must be relevant, not_relevant, or uncertain")
            if crop_id in judgment_by_id:
                raise ValueError("relevance matrix contains a duplicate crop judgment")
            judgment_by_id[crop_id] = label
        if set(judgment_by_id) != set(gallery_by_id):
            raise ValueError("every selected gallery crop must have exactly one human judgment")
        labels = list(judgment_by_id.values())
        if "relevant" not in labels or "not_relevant" not in labels:
            raise ValueError("evaluation truth requires at least one positive and one negative")

        gallery_records = [
            {"crop_id": str(crop_id), "sha256": gallery_by_id[crop_id]}
            for crop_id in sorted(gallery_by_id, key=str)
        ]
        judgments_records = [
            {
                "crop_id": str(crop_id),
                "sha256": gallery_by_id[crop_id],
                "judgment": judgment_by_id[crop_id],
            }
            for crop_id in sorted(judgment_by_id, key=str)
        ]
        gallery_json = json.dumps(
            gallery_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        judgments_json = json.dumps(
            judgments_records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        query_json = json.dumps(
            {
                "query_id": current["query_id"],
                "query_revision": current["query_revision"],
                "query_text": current["query_text"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        query_sha256 = sha256(query_json.encode("utf-8")).hexdigest()
        gallery_sha256 = sha256(gallery_json.encode("utf-8")).hexdigest()
        judgments_sha256 = sha256(judgments_json.encode("utf-8")).hexdigest()
        unresolved_count = labels.count("uncertain")
        status = "unresolved" if unresolved_count else "complete"
        revision_id = uuid4()
        saved_provenance = {
            **provenance,
            "query_sha256": query_sha256,
            "gallery_sha256": gallery_sha256,
            "judgments_sha256": judgments_sha256,
            "gallery_count": len(gallery_records),
            "query_source_caption_revision_id": (
                str(current["query_source_caption_revision_id"])
                if current["query_source_caption_revision_id"] is not None
                else None
            ),
        }
        revision_record = await connection.fetchrow(
            """
            INSERT INTO relevance_matrix_revisions (
                relevance_revision_id, query_id, query_text, query_revision,
                query_sha256, gallery_sha256, judgments_sha256, status,
                evaluation_eligible, provenance
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            RETURNING relevance_revision_id, query_id, query_text, query_revision,
                      query_sha256, gallery_sha256, judgments_sha256, status,
                      evaluation_eligible, provenance, created_at
            """,
            revision_id,
            current["query_id"],
            current["query_text"],
            current["query_revision"],
            query_sha256,
            gallery_sha256,
            judgments_sha256,
            status,
            status == "complete",
            json.dumps(saved_provenance, ensure_ascii=False, sort_keys=True),
        )
        await connection.executemany(
            """
            INSERT INTO relevance_judgments (
                relevance_revision_id, crop_id, crop_sha256, judgment
            ) VALUES ($1, $2, $3, $4)
            """,
            [
                (revision_id, crop_id, gallery_by_id[crop_id], judgment_by_id[crop_id])
                for crop_id in sorted(judgment_by_id, key=str)
            ],
        )
        await connection.execute(
            """
            UPDATE relevance_matrix_review_drafts
            SET state = 'frozen', frozen_relevance_revision_id = $2,
                invalidation_reason = NULL, updated_at = now()
            WHERE review_id = $1 AND state = 'pending'
            """,
            review_uuid,
            revision_id,
        )
        revision = _relevance_revision_dict(
            revision_record,
            positive_count=labels.count("relevant"),
            negative_count=labels.count("not_relevant"),
            uncertain_count=unresolved_count,
        )
        return {
            "state": "frozen",
            "review_id": str(review_uuid),
            "frozen_relevance_revision_id": str(revision_id),
            "revision": revision,
        }

    async def _lock_current_relevance_dependencies(
        self,
        connection: asyncpg.Connection,
        *,
        gallery: list[dict[str, str]],
        query_source_caption_revision_id: UUID | None,
    ) -> str | None:
        gallery_ids = [UUID(item["crop_id"]) for item in gallery]
        gallery_id_set = set(gallery_ids)
        gallery_sources = await connection.fetch(
            "SELECT crop_id, sample_id FROM annotation_crops WHERE crop_id = ANY($1::uuid[])",
            gallery_ids,
        )
        if len(gallery_sources) != len(gallery_id_set):
            return "gallery_crop_revision_changed"
        sample_ids = {row["sample_id"] for row in gallery_sources}
        source_revision = None
        source_crop_ids: set[UUID] = set()
        if query_source_caption_revision_id is not None:
            source_revision = await connection.fetchrow(
                """
                SELECT sample_id, stage FROM annotation_revisions
                WHERE annotation_revision_id = $1
                """,
                query_source_caption_revision_id,
            )
            if source_revision is None or source_revision["stage"] != "caption":
                return "query_caption_revision_changed"
            sample_ids.add(source_revision["sample_id"])
            source_crops = await connection.fetch(
                "SELECT crop_id FROM annotation_crops WHERE caption_revision_id = $1",
                query_source_caption_revision_id,
            )
            source_crop_ids = {row["crop_id"] for row in source_crops}

        sorted_samples = sorted(sample_ids, key=str)
        if sorted_samples:
            await connection.fetch(
                """
                SELECT sample_id FROM ingestion_samples
                WHERE sample_id = ANY($1::uuid[])
                ORDER BY sample_id FOR SHARE
                """,
                sorted_samples,
            )
        head_rows = await connection.fetch(
            """
            SELECT sample_id, latest_bbox_revision, current_bbox_assignment_revision
            FROM sample_annotation_heads
            WHERE sample_id = ANY($1::uuid[])
            ORDER BY sample_id FOR SHARE
            """,
            sorted_samples,
        ) if sorted_samples else []
        heads = {row["sample_id"]: row for row in head_rows}
        dependency_ids = sorted(gallery_id_set | source_crop_ids, key=str)
        crop_rows = await connection.fetch(
            """
            SELECT crop_id, sample_id, bbox_revision, sha256, state, caption_state,
                   caption_revision_id, crop_set_ready
            FROM annotation_crops
            WHERE crop_id = ANY($1::uuid[])
            ORDER BY crop_id FOR SHARE
            """,
            dependency_ids,
        ) if dependency_ids else []
        crops = {row["crop_id"]: row for row in crop_rows}

        for item in gallery:
            crop_id = UUID(item["crop_id"])
            crop = crops.get(crop_id)
            if (
                crop is None
                or crop["sha256"].strip() != item["sha256"]
                or crop["state"] != "ready"
                or not crop["crop_set_ready"]
            ):
                return "gallery_crop_revision_changed"
            head = heads.get(crop["sample_id"])
            if (
                head is None
                or head["latest_bbox_revision"] != crop["bbox_revision"]
                or head["current_bbox_assignment_revision"] is not None
            ):
                return "gallery_crop_revision_changed"

        if query_source_caption_revision_id is not None:
            matching = [
                crop for crop in crops.values()
                if crop["caption_revision_id"] == query_source_caption_revision_id
            ]
            if len(matching) != 1:
                return "query_caption_revision_changed"
            crop = matching[0]
            head = heads.get(crop["sample_id"])
            if (
                crop["state"] != "ready"
                or crop["caption_state"] != "reviewed"
                or head is None
                or head["latest_bbox_revision"] != crop["bbox_revision"]
                or head["current_bbox_assignment_revision"] is not None
            ):
                return "query_caption_revision_changed"
        return None

    async def relevance_revision(self, revision_id: str) -> dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            revision = await connection.fetchrow(
                """
                SELECT relevance_revision_id, query_id, query_text, query_revision,
                       query_sha256, gallery_sha256, judgments_sha256, status,
                       evaluation_eligible, provenance, created_at
                FROM relevance_matrix_revisions WHERE relevance_revision_id = $1
                """,
                UUID(revision_id),
            )
            if revision is None:
                raise ReviewAssignmentNotFoundError("relevance matrix revision does not exist")
            judgments = await connection.fetch(
                """
                SELECT crop_id, crop_sha256, judgment FROM relevance_judgments
                WHERE relevance_revision_id = $1 ORDER BY crop_id
                """,
                UUID(revision_id),
            )
        return {
            **_relevance_revision_dict(
                revision,
                positive_count=sum(row["judgment"] == "relevant" for row in judgments),
                negative_count=sum(row["judgment"] == "not_relevant" for row in judgments),
                uncertain_count=sum(row["judgment"] == "uncertain" for row in judgments),
            ),
            "judgments": [
                {
                    "crop_id": str(row["crop_id"]),
                    "sha256": row["crop_sha256"].strip(),
                    "judgment": row["judgment"],
                }
                for row in judgments
            ],
        }

    async def assignment(self, sample_id: UUID, revision: UUID) -> ReviewAssignment:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            record = await connection.fetchrow(
                """
                SELECT assignment_id, revision, sample_id, stage, bbox_revision,
                       label_studio_task_id, media_object_key, required_bytes, state
                FROM review_assignments WHERE sample_id = $1 AND revision = $2
                """,
                sample_id,
                revision,
            )
        if record is None:
            raise ReviewAssignmentNotFoundError("review revision does not exist for this sample")
        return assignment_from_record(record)

    async def assignment_by_revision(self, revision: str) -> ReviewAssignment:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            record = await connection.fetchrow(
                """
                SELECT assignment_id, revision, sample_id, stage, bbox_revision,
                       label_studio_task_id, media_object_key, required_bytes, state
                FROM review_assignments WHERE revision = $1
                """,
                UUID(revision),
            )
        if record is None:
            raise ReviewAssignmentNotFoundError("review revision does not exist")
        return assignment_from_record(record)

    async def bbox_review_source(self, sample_id: UUID) -> dict[str, Any]:
        """Resolve only a currently received frame for a new bbox assignment."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT sample_id, state, object_key, object_size_bytes
                FROM ingestion_samples WHERE sample_id = $1
                """,
                sample_id,
            )
        if row is None or row["state"] != "received":
            raise ReviewAssignmentConflictError("only a received sample can be assigned for bbox review")
        return {
            "sample_id": str(row["sample_id"]),
            "object_key": row["object_key"],
            "object_size_bytes": row["object_size_bytes"],
        }

    async def list_review_assignments(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return saved assignment state and exact Label Studio provenance for the operator UI."""
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("review list limit must be between 1 and 500")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT assignment.sample_id, assignment.revision, assignment.stage,
                       assignment.state AS assignment_state, assignment.label_studio_task_id,
                       assignment.created_at, assignment.finished_at,
                       media.project_id, revision.annotation_revision_id,
                       revision.provenance
                FROM review_assignments AS assignment
                LEFT JOIN annotation_revisions AS revision
                  ON revision.assignment_revision = assignment.revision
                LEFT JOIN label_studio_media_uploads AS media
                  ON media.assignment_revision = assignment.revision
                ORDER BY assignment.created_at DESC, assignment.revision ASC
                LIMIT $1
                """,
                limit,
            )
        result = []
        for row in rows:
            provenance = _json_value(row["provenance"]) if row["provenance"] is not None else None
            human_review_recorded = bool(
                isinstance(provenance, dict)
                and provenance.get("source") == "label_studio"
                and provenance.get("label_studio_annotation_id")
                and (provenance.get("reviewer_id") or provenance.get("completed_by"))
            )
            result.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "revision": str(row["revision"]),
                    "stage": row["stage"],
                    "state": row["assignment_state"],
                    "label_studio_task_id": row["label_studio_task_id"],
                    "project_id": row["project_id"],
                    "created_at": row["created_at"].isoformat(),
                    "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
                    "annotation_revision_id": (
                        str(row["annotation_revision_id"])
                        if row["annotation_revision_id"] is not None
                        else None
                    ),
                    "provenance": provenance,
                    "human_review_recorded": human_review_recorded,
                }
            )
        return result

    async def review_media_source(self, sample_id: UUID, revision: UUID) -> dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            assignment = await connection.fetchrow(
                """
                SELECT stage, state, media_object_key, required_bytes
                FROM review_assignments WHERE sample_id = $1 AND revision = $2
                """,
                sample_id,
                revision,
            )
            if assignment is None or assignment["state"] != "provisioning":
                raise ReviewAssignmentConflictError("only a provisioning assignment can upload Label Studio media")
            if assignment["stage"] == "bbox":
                source = await connection.fetchrow(
                    """
                    SELECT state, object_key, sha256, object_size_bytes
                    FROM ingestion_samples WHERE sample_id = $1
                    """,
                    sample_id,
                )
                if (
                    source is None
                    or source["state"] != "received"
                    or source["object_key"] != assignment["media_object_key"]
                    or source["object_size_bytes"] != assignment["required_bytes"]
                ):
                    raise ReviewAssignmentConflictError("bbox review source frame changed")
            elif assignment["stage"] == "caption":
                source = await connection.fetchrow(
                    """
                    SELECT state, object_key, sha256, object_size_bytes
                    FROM annotation_crops
                    WHERE sample_id = $1 AND object_key = $2
                    """,
                    sample_id,
                    assignment["media_object_key"],
                )
                if (
                    source is None
                    or source["state"] != "ready"
                    or source["object_size_bytes"] != assignment["required_bytes"]
                ):
                    raise ReviewAssignmentConflictError("caption review crop changed")
            else:
                raise ReviewAssignmentConflictError("relevance tasks do not upload media through this flow")
        return {
            "sample_id": str(sample_id),
            "revision": str(revision),
            "stage": assignment["stage"],
            "object_key": source["object_key"],
            "sha256": source["sha256"].strip(),
            "object_size_bytes": source["object_size_bytes"],
            "required_bytes": assignment["required_bytes"],
        }

    async def reserve_label_studio_media(
        self,
        *,
        revision: str,
        project_id: int,
        filename: str,
        sha256_digest: str,
        object_size_bytes: int,
    ) -> dict[str, Any]:
        if project_id <= 0 or object_size_bytes <= 0 or len(sha256_digest) != 64:
            raise ValueError("Label Studio media project, bytes, and SHA-256 are required")
        revision_id = UUID(revision)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                identity = await connection.fetchrow(
                    "SELECT sample_id FROM review_assignments WHERE revision = $1",
                    revision_id,
                )
                if identity is None:
                    raise ReviewAssignmentNotFoundError("review assignment does not exist")
                sample = await connection.fetchrow(
                    "SELECT sample_id FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE",
                    identity["sample_id"],
                )
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                review_usage = await connection.fetchrow(
                    "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                assignment = await connection.fetchrow(
                    "SELECT state FROM review_assignments WHERE revision = $1 FOR UPDATE",
                    revision_id,
                )
                if sample is None or assignment is None or assignment["state"] != "provisioning":
                    raise ReviewAssignmentConflictError("review assignment is no longer provisioning")
                current = await connection.fetchrow(
                    "SELECT * FROM label_studio_media_uploads WHERE assignment_revision = $1 FOR UPDATE",
                    revision_id,
                )
                if current is not None:
                    if (
                        current["project_id"] != project_id
                        or current["upload_filename"] != filename
                        or current["sha256"].strip() != sha256_digest
                        or current["object_size_bytes"] != object_size_bytes
                        or current["state"] not in {"reserved", "uploaded", "delete_pending"}
                    ):
                        raise ReviewAssignmentConflictError("Label Studio media reservation changed")
                    return _label_studio_upload_dict(current)
                if usage is None or review_usage is None:
                    raise RuntimeError("shared object or active-review quota ledger is missing")
                if usage["used_bytes"] + object_size_bytes > GLOBAL_OBJECT_LIMIT:
                    await self._record_capacity_stop(
                        sample_id=identity["sample_id"],
                        reason="global_object_quota_exceeded",
                        action="label_studio_upload_blocked",
                    )
                    raise GlobalObjectLimitError("Label Studio media exceeds the shared one TiB object quota")
                next_review_bytes = review_usage["active_bytes"] + object_size_bytes
                if next_review_bytes > self._review_exception_bytes:
                    await self._record_capacity_stop(
                        sample_id=identity["sample_id"],
                        reason="active_review_quota_exceeded",
                        action="label_studio_upload_blocked",
                    )
                    raise ReviewQuotaExceededError("active review media exceeds the 100 GiB exception quota")
                await connection.execute(
                    "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                    object_size_bytes,
                )
                await connection.execute(
                    "UPDATE review_storage_usage SET active_bytes = $1 WHERE singleton = TRUE",
                    next_review_bytes,
                )
                record = await connection.fetchrow(
                    """
                    INSERT INTO label_studio_media_uploads (
                        assignment_revision, project_id, upload_filename,
                        sha256, object_size_bytes, state
                    ) VALUES ($1, $2, $3, $4, $5, 'reserved')
                    RETURNING assignment_revision, project_id, upload_filename, upload_path,
                              file_upload_id, task_id, sha256, object_size_bytes, state
                    """,
                    revision_id,
                    project_id,
                    filename,
                    sha256_digest,
                    object_size_bytes,
                )
                return _label_studio_upload_dict(record)

    async def bind_label_studio_media(
        self,
        *,
        revision: str,
        task_id: int,
        file_upload_id: int,
        upload_path: str,
    ) -> dict[str, Any]:
        if min(task_id, file_upload_id) <= 0:
            raise ValueError("Label Studio task and file-upload IDs must be positive")
        revision_id = UUID(revision)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                media = await connection.fetchrow(
                    "SELECT * FROM label_studio_media_uploads WHERE assignment_revision = $1 FOR UPDATE",
                    revision_id,
                )
                assignment = await connection.fetchrow(
                    """
                    SELECT sample_id, stage, bbox_revision, state,
                           media_object_key, required_bytes
                    FROM review_assignments WHERE revision = $1 FOR UPDATE
                    """,
                    revision_id,
                )
                if media is None or assignment is None:
                    raise ReviewAssignmentNotFoundError("reserved Label Studio media assignment does not exist")
                if media["state"] == "uploaded":
                    if (
                        media["task_id"] != task_id
                        or media["file_upload_id"] != file_upload_id
                        or media["upload_path"] != upload_path
                    ):
                        raise ReviewAssignmentConflictError("Label Studio upload retry resolved to another task")
                    return _label_studio_upload_dict(media)
                if assignment["state"] != "provisioning" or media["state"] != "reserved":
                    raise ReviewAssignmentConflictError("review assignment cannot bind Label Studio media")
                if assignment["stage"] == "caption":
                    crop = await connection.fetchrow(
                        """
                        SELECT bbox_revision, state, caption_state, object_size_bytes
                        FROM annotation_crops
                        WHERE sample_id = $1 AND object_key = $2 FOR UPDATE
                        """,
                        assignment["sample_id"],
                        assignment["media_object_key"],
                    )
                    if (
                        crop is None
                        or crop["state"] != "ready"
                        or crop["caption_state"] != "needs_review"
                        or crop["bbox_revision"] != UUID(assignment["bbox_revision"])
                        or crop["object_size_bytes"] != assignment["required_bytes"]
                    ):
                        raise ReviewAssignmentConflictError("caption crop expired or changed before media bind")
                media = await connection.fetchrow(
                    """
                    UPDATE label_studio_media_uploads
                    SET task_id = $2, file_upload_id = $3, upload_path = $4,
                        state = 'uploaded', updated_at = now()
                    WHERE assignment_revision = $1
                    RETURNING assignment_revision, project_id, upload_filename, upload_path,
                              file_upload_id, task_id, sha256, object_size_bytes, state
                    """,
                    revision_id,
                    task_id,
                    file_upload_id,
                    upload_path,
                )
                await connection.execute(
                    "UPDATE review_assignments SET label_studio_task_id = $2, state = 'active' WHERE revision = $1",
                    revision_id,
                    task_id,
                )
                return _label_studio_upload_dict(media)

    async def register_existing_label_studio_media(
        self,
        *,
        revision: str,
        project_id: int,
        task_id: int,
        file_upload_id: int,
        filename: str,
        upload_path: str,
        sha256_digest: str,
        object_size_bytes: int,
    ) -> dict[str, Any]:
        """Account a pre-existing CE upload after its task and bytes are verified externally."""
        if min(project_id, task_id, file_upload_id, object_size_bytes) <= 0:
            raise ValueError("Label Studio media IDs and size must be positive")
        if object_size_bytes > _MAX_REVIEW_MEDIA_BYTES:
            raise ValueError("Label Studio review media exceeds the 20 MiB object limit")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256_digest):
            raise ValueError("Label Studio review media SHA-256 must be 64 lowercase hex characters")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}\.(jpg|jpeg|png)", filename):
            raise ValueError("Label Studio review filename must be a simple JPEG or PNG name")
        media_path = PurePosixPath(upload_path)
        if (
            not media_path.is_absolute()
            or media_path.parts != ("/", "data", "upload", str(project_id), filename)
        ):
            raise ValueError("Label Studio media path must identify the recorded file under its project")

        revision_id = UUID(revision)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                assignment = await connection.fetchrow(
                    "SELECT sample_id, stage, state, label_studio_task_id FROM review_assignments WHERE revision = $1 FOR UPDATE",
                    revision_id,
                )
                if assignment is None:
                    raise ReviewAssignmentNotFoundError("review assignment does not exist")
                if (
                    assignment["stage"] not in {"bbox", "caption"}
                    or assignment["state"] != "active"
                    or assignment["label_studio_task_id"] != task_id
                ):
                    raise ReviewAssignmentConflictError("existing Label Studio task does not match an active review")
                sample = await connection.fetchrow(
                    "SELECT state FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE",
                    assignment["sample_id"],
                )
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                review_usage = await connection.fetchrow(
                    "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                if sample is None or sample["state"] != "received" or usage is None or review_usage is None:
                    raise ReviewAssignmentConflictError("existing review media has no live sample or quota ledger")
                current = await connection.fetchrow(
                    "SELECT * FROM label_studio_media_uploads WHERE assignment_revision = $1 FOR UPDATE",
                    revision_id,
                )
                if current is not None:
                    if (
                        current["project_id"] != project_id
                        or current["task_id"] != task_id
                        or current["file_upload_id"] != file_upload_id
                        or current["upload_filename"] != filename
                        or current["upload_path"] != upload_path
                        or current["sha256"].strip() != sha256_digest
                        or current["object_size_bytes"] != object_size_bytes
                        or current["state"] not in {"uploaded", "delete_pending", "deleted"}
                    ):
                        raise ReviewAssignmentConflictError("existing Label Studio media registration changed")
                    return _label_studio_upload_dict(current)

                if usage["used_bytes"] + object_size_bytes > GLOBAL_OBJECT_LIMIT:
                    await self._record_capacity_stop(
                        sample_id=assignment["sample_id"],
                        reason="global_object_quota_exceeded",
                        action="label_studio_upload_blocked",
                    )
                    raise GlobalObjectLimitError("Label Studio media exceeds the shared one TiB object quota")
                next_review_bytes = review_usage["active_bytes"] + object_size_bytes
                if next_review_bytes > self._review_exception_bytes:
                    await self._record_capacity_stop(
                        sample_id=assignment["sample_id"],
                        reason="active_review_quota_exceeded",
                        action="label_studio_upload_blocked",
                    )
                    raise ReviewQuotaExceededError("active review media exceeds the 100 GiB exception quota")

                await connection.execute(
                    "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                    object_size_bytes,
                )
                await connection.execute(
                    "UPDATE review_storage_usage SET active_bytes = $1 WHERE singleton = TRUE",
                    next_review_bytes,
                )
                record = await connection.fetchrow(
                    """
                    INSERT INTO label_studio_media_uploads (
                        assignment_revision, project_id, upload_filename, upload_path,
                        file_upload_id, task_id, sha256, object_size_bytes, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'uploaded')
                    RETURNING assignment_revision, project_id, upload_filename, upload_path,
                              file_upload_id, task_id, sha256, object_size_bytes, state
                    """,
                    revision_id,
                    project_id,
                    filename,
                    upload_path,
                    file_upload_id,
                    task_id,
                    sha256_digest,
                    object_size_bytes,
                )
                return _label_studio_upload_dict(record)

    async def label_studio_media_upload(self, revision: str) -> dict[str, Any] | None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            record = await connection.fetchrow(
                "SELECT * FROM label_studio_media_uploads WHERE assignment_revision = $1",
                UUID(revision),
            )
        return _label_studio_upload_dict(record) if record is not None else None

    async def label_studio_media_cleanup_candidates(self, *, limit: int = 100) -> list[str]:
        if not 1 <= limit <= 1000:
            raise ValueError("Label Studio cleanup limit must be between 1 and 1000")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            revisions = await connection.fetch(
                """
                SELECT media.assignment_revision
                FROM label_studio_media_uploads AS media
                JOIN review_assignments AS assignment
                  ON assignment.revision = media.assignment_revision
                WHERE media.state = 'delete_pending'
                   OR (media.state = 'uploaded' AND assignment.state IN (
                       'finalized', 'cancelled', 'rejected', 'expired', 'superseded'
                   ))
                ORDER BY media.updated_at, media.assignment_revision
                LIMIT $1
                """,
                limit,
            )
        return [str(row["assignment_revision"]) for row in revisions]

    async def mark_label_studio_media_delete_pending(self, revision: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            result = await connection.execute(
                """
                UPDATE label_studio_media_uploads SET state = 'delete_pending', updated_at = now()
                WHERE assignment_revision = $1 AND state IN ('uploaded', 'delete_pending')
                """,
                UUID(revision),
            )
        if result != "UPDATE 1":
            raise ReviewAssignmentConflictError("uploaded Label Studio media is not ready for deletion")

    async def finish_label_studio_media_cleanup(self, revision: str) -> bool:
        revision_id = UUID(revision)
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                media = await connection.fetchrow(
                    "SELECT * FROM label_studio_media_uploads WHERE assignment_revision = $1 FOR UPDATE",
                    revision_id,
                )
                if media is None:
                    return False
                if media["state"] == "deleted":
                    return False
                if media["state"] != "delete_pending":
                    raise ReviewAssignmentConflictError("Label Studio media bytes were not marked for deletion")
                usage = await connection.fetchrow(
                    """
                    UPDATE ingestion_storage_usage SET used_bytes = used_bytes - $1
                    WHERE singleton = TRUE AND used_bytes >= $1 RETURNING used_bytes
                    """,
                    media["object_size_bytes"],
                )
                review = await connection.fetchrow(
                    """
                    UPDATE review_storage_usage SET active_bytes = active_bytes - $1
                    WHERE singleton = TRUE AND active_bytes >= $1 RETURNING active_bytes
                    """,
                    media["object_size_bytes"],
                )
                if usage is None or review is None:
                    raise RuntimeError("shared review-media quota accounting is inconsistent")
                await connection.execute(
                    "UPDATE label_studio_media_uploads SET state = 'deleted', updated_at = now() WHERE assignment_revision = $1",
                    revision_id,
                )
                return True

    async def finalized_revision(self, revision: str) -> dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            record = await connection.fetchrow(
                """
                SELECT annotation_revision_id, assignment_revision, sample_id, stage,
                       label_studio_task_id, label_studio_annotation_id, result,
                       result_sha256, provenance, submitted_at
                FROM annotation_revisions WHERE assignment_revision = $1
                """,
                UUID(revision),
            )
        if record is None:
            raise ReviewAssignmentConflictError("finalized review has no immutable annotation snapshot")
        return _revision_dict(record)

    async def finalize_submitted_annotation(
        self,
        *,
        assignment: ReviewAssignment,
        annotation: dict[str, Any],
        task: dict[str, Any],
    ) -> dict[str, Any]:
        canonical_result = json.dumps(annotation["result"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result_sha256 = sha256(canonical_result.encode("utf-8")).hexdigest()
        annotation_revision_id = uuid4()
        provenance = {
            "source": "label_studio",
            "label_studio_task_id": assignment.label_studio_task_id,
            "label_studio_annotation_id": annotation["id"],
            "created_at": annotation.get("created_at"),
            "updated_at": annotation.get("updated_at"),
            "completed_by": annotation.get("completed_by"),
            "lead_time": annotation.get("lead_time"),
            "was_cancelled": False,
            "result_sha256": result_sha256,
        }
        canonical_provenance = json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                sample = await connection.fetchrow(
                    "SELECT state FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE",
                    assignment.sample_id,
                )
                current = await connection.fetchrow(
                    "SELECT * FROM review_assignments WHERE sample_id = $1 AND revision = $2 FOR UPDATE",
                    assignment.sample_id,
                    UUID(assignment.revision),
                )
                if sample is None or current is None:
                    raise ReviewAssignmentNotFoundError("review revision no longer exists")
                existing = await connection.fetchrow(
                    """
                    SELECT annotation_revision_id, assignment_revision, sample_id, stage,
                           label_studio_task_id, label_studio_annotation_id, result,
                           result_sha256, provenance, submitted_at
                    FROM annotation_revisions WHERE assignment_revision = $1
                    """,
                    UUID(assignment.revision),
                )
                if current["state"] == "finalized" and existing is not None:
                    return _revision_dict(existing)
                sample_is_live = sample["state"] == "received"
                crop_is_live = False
                if current["stage"] == "caption" and sample["state"] == "expired":
                    crop_is_live = await connection.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM annotation_crops
                            WHERE sample_id = $1 AND object_key = $2
                              AND bbox_revision = $3 AND state = 'ready'
                        )
                        """,
                        assignment.sample_id,
                        current["media_object_key"],
                        UUID(assignment.bbox_revision),
                    )
                if current["state"] != "active" or not (sample_is_live or crop_is_live):
                    raise ReviewAssignmentConflictError("review changed before annotation finalization")
                if task.get("id") != current["label_studio_task_id"]:
                    raise ReviewAssignmentConflictError("Label Studio task ID does not match the active revision")
                if current["stage"] == "bbox":
                    head = await connection.fetchrow(
                        "SELECT current_bbox_assignment_revision FROM sample_annotation_heads WHERE sample_id = $1 FOR UPDATE",
                        assignment.sample_id,
                    )
                    if head is None or head["current_bbox_assignment_revision"] != UUID(assignment.revision):
                        raise ReviewAssignmentConflictError("bbox revision changed before annotation finalization")
                elif current["stage"] == "caption":
                    crop = await connection.fetchrow(
                        """
                        SELECT bbox_revision, state, caption_state, caption_revision_id
                        FROM annotation_crops
                        WHERE sample_id = $1 AND object_key = $2 FOR UPDATE
                        """,
                        assignment.sample_id,
                        current["media_object_key"],
                    )
                    if (
                        crop is None
                        or crop["bbox_revision"] != UUID(assignment.bbox_revision)
                        or crop["state"] != "ready"
                        or crop["caption_state"] != "needs_review"
                    ):
                        raise ReviewAssignmentConflictError("caption crop changed before finalization")

                saved = await connection.fetchrow(
                    """
                    INSERT INTO annotation_revisions (
                        annotation_revision_id, sample_id, assignment_revision, stage,
                        label_studio_task_id, label_studio_annotation_id, result,
                        result_sha256, provenance, submitted_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9::jsonb,
                              COALESCE($10::timestamptz, now()))
                    RETURNING annotation_revision_id, assignment_revision, sample_id, stage,
                              label_studio_task_id, label_studio_annotation_id, result,
                              result_sha256, provenance, submitted_at
                    """,
                    annotation_revision_id,
                    assignment.sample_id,
                    UUID(assignment.revision),
                    assignment.stage,
                    current["label_studio_task_id"],
                    annotation["id"],
                    canonical_result,
                    result_sha256,
                    canonical_provenance,
                    _parse_label_studio_time(
                        annotation.get("updated_at") or annotation.get("created_at")
                    ),
                )
                await connection.execute(
                    "UPDATE review_assignments SET state = 'finalized', finished_at = now() WHERE revision = $1",
                    UUID(assignment.revision),
                )
                if assignment.stage == "bbox":
                    await self._invalidate_pending_bbox_crop_reviews(
                        connection,
                        sample_id=assignment.sample_id,
                        current_bbox_revision=annotation_revision_id,
                    )
                    await connection.execute(
                        """
                        UPDATE sample_annotation_heads
                        SET current_bbox_assignment_revision = NULL,
                            latest_bbox_revision = $2, updated_at = now()
                        WHERE sample_id = $1
                        """,
                        assignment.sample_id,
                        annotation_revision_id,
                    )
                    await connection.execute(
                        "UPDATE ingestion_samples SET selected = FALSE WHERE sample_id = $1",
                        assignment.sample_id,
                    )
                elif assignment.stage == "caption":
                    prior_caption_revision_id = crop["caption_revision_id"]
                    if prior_caption_revision_id is not None and prior_caption_revision_id != annotation_revision_id:
                        await self._invalidate_pending_caption_source_reviews(
                            connection,
                            prior_caption_revision_id,
                            reason="query_caption_revision_changed",
                        )
                    updated = await connection.execute(
                        """
                        UPDATE annotation_crops
                        SET caption_state = 'reviewed', caption_revision_id = $3, updated_at = now()
                        WHERE sample_id = $1 AND object_key = $2
                          AND caption_state = 'needs_review' AND state = 'ready'
                        """,
                        assignment.sample_id,
                        current["media_object_key"],
                        annotation_revision_id,
                    )
                    if updated != "UPDATE 1":
                        raise ReviewAssignmentConflictError("caption review changed before commit")
                await connection.execute(
                    """
                    UPDATE review_storage_usage SET active_bytes = active_bytes - $1
                    WHERE singleton = TRUE AND active_bytes >= $1
                    """,
                    current["required_bytes"],
                )
                return _revision_dict(saved)

    async def assignment_state(self, revision: str) -> str | None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            state = await connection.fetchval(
                "SELECT state FROM review_assignments WHERE revision = $1",
                UUID(revision),
            )
        return state

    async def annotation_revision_count(self, sample_id: UUID) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            count = await connection.fetchval(
                "SELECT count(*) FROM annotation_revisions WHERE sample_id = $1", sample_id
            )
        return int(count)

    async def _invalidate_pending_caption_source_reviews(
        self,
        connection: asyncpg.Connection,
        caption_revision_id: UUID,
        *,
        reason: str,
    ) -> None:
        await connection.execute(
            """
            UPDATE relevance_matrix_review_drafts
            SET state = 'needs_review', invalidation_reason = $2, updated_at = now()
            WHERE state = 'pending'
              AND query_source_caption_revision_id = $1
            """,
            caption_revision_id,
            reason,
        )

    async def _invalidate_pending_bbox_crop_reviews(
        self,
        connection: asyncpg.Connection,
        *,
        sample_id: UUID,
        current_bbox_revision: UUID,
    ) -> None:
        await connection.execute(
            """
            UPDATE relevance_matrix_review_drafts AS review
            SET state = 'needs_review',
                invalidation_reason = 'gallery_crop_revision_changed',
                updated_at = now()
            WHERE review.state = 'pending'
              AND EXISTS (
                  SELECT 1
                  FROM relevance_matrix_review_crops AS dependency
                  JOIN annotation_crops AS crop ON crop.crop_id = dependency.crop_id
                  WHERE dependency.review_id = review.review_id
                    AND crop.sample_id = $1
                    AND crop.bbox_revision <> $2
              )
            """,
            sample_id,
            current_bbox_revision,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _record_capacity_stop(
        self,
        *,
        sample_id: UUID,
        reason: str,
        action: str = "assignment_blocked",
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO retention_policy_events (sample_id, action, reason)
                VALUES ($1, $2, $3)
                """,
                sample_id,
                action,
                reason,
            )

    async def record_retention_event(
        self,
        *,
        sample_id: UUID | None,
        action: str,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO retention_policy_events (sample_id, action, reason, details)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                sample_id,
                action,
                reason,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
            )

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=1,
                max_size=8,
                command_timeout=10,
            )
        return self._pool


def _revision_dict(record: Any) -> dict[str, Any]:
    return {
        "annotation_revision_id": str(record["annotation_revision_id"]),
        "assignment_revision": str(record["assignment_revision"]),
        "sample_id": str(record["sample_id"]),
        "stage": record["stage"],
        "label_studio_task_id": record["label_studio_task_id"],
        "label_studio_annotation_id": record["label_studio_annotation_id"],
        "result": _json_value(record["result"]),
        "result_sha256": record["result_sha256"].strip(),
        "provenance": _json_value(record["provenance"]),
        "submitted_at": record["submitted_at"].isoformat(),
    }


def _crop_dict(record: Any) -> dict[str, Any]:
    return {
        "crop_id": str(record["crop_id"]),
        "sample_id": str(record["sample_id"]),
        "bbox_revision": str(record["bbox_revision"]),
        "region_index": record["region_index"],
        "region_id": record["region_id"],
        "object_key": record["object_key"],
        "sha256": record["sha256"].strip(),
        "object_size_bytes": record["object_size_bytes"],
        "state": record["state"],
        "caption_state": record["caption_state"],
        "caption_revision_id": str(record["caption_revision_id"]) if record["caption_revision_id"] else None,
        "provenance": _json_value(record["provenance"]),
        "parent_available": record["parent_available"],
        "parent_regenerable": record["regeneration_available"],
        "crop_set_ready": record["crop_set_ready"],
    }


def _adoption_dict(record: Any) -> dict[str, Any]:
    return {
        "adoption_id": str(record["adoption_id"]),
        "dataset_version": record["dataset_version"],
        "target": record["target"],
        "sample_id": str(record["sample_id"]),
        "annotation_revision_id": str(record["annotation_revision_id"]),
        "crop_id": str(record["crop_id"]) if record["crop_id"] else None,
        "caption_revision_id": str(record["caption_revision_id"]) if record["caption_revision_id"] else None,
    }


def _relevance_revision_dict(
    record: Any,
    *,
    positive_count: int,
    negative_count: int,
    uncertain_count: int,
) -> dict[str, Any]:
    return {
        "relevance_revision_id": str(record["relevance_revision_id"]),
        "query_id": record["query_id"],
        "query_text": record["query_text"],
        "query_revision": record["query_revision"],
        "query_sha256": record["query_sha256"].strip(),
        "gallery_sha256": record["gallery_sha256"].strip(),
        "judgments_sha256": record["judgments_sha256"].strip(),
        "status": record["status"],
        "evaluation_eligible": record["evaluation_eligible"],
        "provenance": _json_value(record["provenance"]),
        "created_at": record["created_at"].isoformat(),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "uncertain_count": uncertain_count,
    }


def _relevance_review_dict(record: Any, gallery: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "review_id": str(record["review_id"]),
        "query_id": record["query_id"],
        "query_text": record["query_text"],
        "query_revision": record["query_revision"],
        "query_source_caption_revision_id": (
            str(record["query_source_caption_revision_id"])
            if record["query_source_caption_revision_id"] is not None
            else None
        ),
        "query_sha256": record["query_sha256"].strip(),
        "gallery_sha256": record["gallery_sha256"].strip(),
        "state": record["state"],
        "invalidation_reason": record["invalidation_reason"],
        "frozen_relevance_revision_id": (
            str(record["frozen_relevance_revision_id"])
            if record["frozen_relevance_revision_id"] is not None
            else None
        ),
        "created_at": record["created_at"].isoformat(),
        "is_current": record["state"] == "pending",
        "gallery": gallery,
    }


def _label_studio_upload_dict(record: Any) -> dict[str, Any]:
    return {
        "assignment_revision": str(record["assignment_revision"]),
        "project_id": record["project_id"],
        "upload_filename": record["upload_filename"],
        "upload_path": record["upload_path"],
        "file_upload_id": record["file_upload_id"],
        "task_id": record["task_id"],
        "sha256": record["sha256"].strip(),
        "object_size_bytes": record["object_size_bytes"],
        "state": record["state"],
    }


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _parse_label_studio_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str):
        raise ValueError("Label Studio submission time must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Label Studio submission time must be valid ISO-8601") from error
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
