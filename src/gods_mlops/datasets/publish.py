"""Publish immutable training inputs and evaluation truth snapshots."""

from __future__ import annotations

import json
import os
import asyncio
from datetime import timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.ingestion.storage import GLOBAL_OBJECT_LIMIT, PostgresIngestionRepository

from .manifest import canonical_json, content_sha256, dataset_code_sha256
from .split import plan_splits


class DatasetQuotaExceededError(RuntimeError):
    """A dataset copy could not reserve bytes in the shared one TiB ledger."""


class DatasetPublicationError(RuntimeError):
    """A dataset selection cannot be published as a coherent immutable input."""


class DatasetObjectStore:
    """Verify source and dataset objects through an S3-compatible endpoint."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str,
    ) -> None:
        if not endpoint_url.startswith(("http://", "https://")) or "@" in endpoint_url:
            raise ValueError("S3 endpoint must be an HTTP(S) URL without embedded credentials")
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(
                signature_version="s3v4",
                connect_timeout=2,
                read_timeout=30,
                retries={"mode": "standard", "total_max_attempts": 3},
                s3={"addressing_style": "path"},
            ),
        )

    def read_source(self, *, object_key: str, sha256_digest: str, size_bytes: int) -> bytes:
        try:
            response: dict[str, Any] = self._client.get_object(Bucket=self._bucket, Key=object_key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "NotFound", "404"}:
                raise FileNotFoundError(f"required dataset source is missing: {object_key}") from error
            raise
        payload = _read_body(response["Body"], expected_size=size_bytes, expected_sha256=sha256_digest)
        return payload

    def write_immutable(
        self,
        *,
        object_key: str,
        content: bytes,
        sha256_digest: str,
        content_type: str,
    ) -> None:
        if sha256(content).hexdigest() != sha256_digest:
            raise OSError("dataset object content does not match its reserved SHA-256")
        if not self._matches(object_key, len(content), sha256_digest):
            self._client.put_object(
                Bucket=self._bucket,
                Key=object_key,
                Body=content,
                ContentType=content_type,
                Metadata={"sha256": sha256_digest},
            )
        if not self._matches(object_key, len(content), sha256_digest):
            raise OSError("immutable dataset object failed read-after-write verification")

    def delete_object(self, *, object_key: str) -> None:
        """Delete one explicitly owned regenerable artifact after a replacement commits."""
        if not object_key or object_key.startswith("/") or ".." in object_key.split("/"):
            raise ValueError("S3 object key is not a safe relative key")
        self._client.delete_object(Bucket=self._bucket, Key=object_key)

    def _matches(self, object_key: str, expected_size: int, expected_sha256: str) -> bool:
        try:
            response: dict[str, Any] = self._client.get_object(Bucket=self._bucket, Key=object_key)
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "NotFound", "404"}:
                return False
            raise
        try:
            _read_body(response["Body"], expected_size=expected_size, expected_sha256=expected_sha256)
        except OSError:
            return False
        return True


class DatasetPublisher:
    """Atomically freeze dataset lineage, reserve shared storage, then publish S3 bytes."""

    def __init__(self, *, database_url: str, objects: DatasetObjectStore) -> None:
        self._database_url = database_url
        self._objects = objects
        self._annotations = PostgresAnnotationRepository(database_url=database_url)
        self._pool: asyncpg.Pool | None = None
        self._schema_ready = False

    @classmethod
    def from_environment(cls) -> "DatasetPublisher":
        return cls(
            database_url=_required("GODS_MLOPS_DATABASE_URL"),
            objects=DatasetObjectStore(
                endpoint_url=_required("GODS_MLOPS_S3_ENDPOINT_URL"),
                access_key=_required("GODS_MLOPS_S3_ACCESS_KEY"),
                secret_key=_required("GODS_MLOPS_S3_SECRET_KEY"),
                bucket=_required("GODS_MLOPS_S3_BUCKET"),
                region=os.environ.get("GODS_MLOPS_S3_REGION", "us-east-1"),
            ),
        )

    async def list_publications(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """List immutable dataset status and the original eligibility reasons."""
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("dataset list limit must be between 1 and 500")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT dataset_version, manifest_sha256, manifest_object_key, split_counts,
                       state, training_ready, training_reasons, evaluation_eligible,
                       evaluation_reasons, relevance_revision_ids, published_at
                FROM dataset_versions
                ORDER BY created_at DESC, dataset_version ASC
                LIMIT $1
                """,
                limit,
            )
        return [_response(row) for row in rows]

    async def list_publication_candidates(self, *, limit: int = 100) -> dict[str, list[dict[str, Any]]]:
        """List received frames and stored crops with their current review state."""
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("publication candidate limit must be between 1 and 500")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            samples = await connection.fetch(
                """
                SELECT sample.sample_id, sample.camera_id, sample.captured_at_utc,
                       sample.reason, sample.state, sample.selected, sample.object_size_bytes,
                       head.latest_bbox_revision, head.current_bbox_assignment_revision
                FROM ingestion_samples AS sample
                LEFT JOIN sample_annotation_heads AS head USING (sample_id)
                WHERE sample.state = 'received'
                ORDER BY sample.captured_at_utc DESC, sample.sample_id ASC
                LIMIT $1
                """,
                limit,
            )
            crops = await connection.fetch(
                """
                SELECT crop.crop_id, crop.sample_id, crop.bbox_revision,
                       crop.caption_revision_id, crop.state, crop.caption_state, crop.sha256
                FROM annotation_crops AS crop
                JOIN ingestion_samples AS sample USING (sample_id)
                WHERE crop.state = 'ready' AND sample.state IN ('received', 'purge_pending', 'expired')
                ORDER BY crop.updated_at DESC, crop.crop_id ASC
                LIMIT $1
                """,
                limit,
            )
        return {
            "samples": [
                {
                    "sample_id": str(row["sample_id"]),
                    "camera_id": str(row["camera_id"]),
                    "captured_at_utc": row["captured_at_utc"].isoformat(),
                    "reason": row["reason"],
                    "state": row["state"],
                    "selected": bool(row["selected"]),
                    "object_size_bytes": row["object_size_bytes"],
                    "latest_bbox_revision": (
                        str(row["latest_bbox_revision"]) if row["latest_bbox_revision"] is not None else None
                    ),
                    "bbox_reviewed": (
                        row["latest_bbox_revision"] is not None
                        and row["current_bbox_assignment_revision"] is None
                    ),
                }
                for row in samples
            ],
            "crops": [
                {
                    "crop_id": str(row["crop_id"]),
                    "sample_id": str(row["sample_id"]),
                    "bbox_revision": str(row["bbox_revision"]),
                    "caption_revision_id": (
                        str(row["caption_revision_id"])
                        if row["caption_revision_id"] is not None
                        else None
                    ),
                    "state": row["state"],
                    "caption_state": row["caption_state"],
                    "sha256": row["sha256"].strip(),
                }
                for row in crops
            ],
        }

    async def publication_selection(
        self,
        *,
        target: str,
        sample_ids: list[str] | tuple[str, ...] = (),
        crop_ids: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Pin only current finalized review revisions for the existing publisher."""
        if target not in {"detr", "clip", "both"}:
            raise ValueError("selection target must be detr, clip, or both")
        normalized_samples = _uuid_list(list(sample_ids), "sample_ids")
        normalized_crops = _uuid_list(list(crop_ids), "crop_ids")
        if target in {"detr", "both"} and not normalized_samples:
            raise ValueError("DETR publication selection requires reviewed samples")
        if target in {"clip", "both"} and not normalized_crops:
            raise ValueError("CLIP publication selection requires reviewed crops")

        await self.ensure_schema()
        pool = await self._get_pool()
        bbox_revisions: dict[str, str] = {}
        caption_revisions: dict[str, str] = {}
        async with pool.acquire() as connection:
            if normalized_samples:
                rows = await connection.fetch(
                    """
                    SELECT head.sample_id, head.latest_bbox_revision AS bbox_revision,
                           head.current_bbox_assignment_revision
                    FROM sample_annotation_heads AS head
                    JOIN annotation_revisions AS revision
                      ON revision.annotation_revision_id = head.latest_bbox_revision
                    WHERE head.sample_id = ANY($1::uuid[]) AND revision.stage = 'bbox'
                    """,
                    [UUID(value) for value in normalized_samples],
                )
                for row in rows:
                    if row["current_bbox_assignment_revision"] is not None:
                        continue
                    bbox_revisions[str(row["sample_id"])] = str(row["bbox_revision"])
            if normalized_crops:
                rows = await connection.fetch(
                    """
                    SELECT crop.crop_id, crop.caption_revision_id
                    FROM annotation_crops AS crop
                    JOIN annotation_revisions AS revision
                      ON revision.annotation_revision_id = crop.caption_revision_id
                    WHERE crop.crop_id = ANY($1::uuid[])
                      AND crop.state = 'ready' AND crop.caption_state = 'reviewed'
                      AND revision.stage = 'caption'
                    """,
                    [UUID(value) for value in normalized_crops],
                )
                caption_revisions = {
                    str(row["crop_id"]): str(row["caption_revision_id"])
                    for row in rows
                }
        if target in {"detr", "both"} and set(bbox_revisions) != set(normalized_samples):
            raise DatasetPublicationError("every selected sample needs a finalized bbox review")
        if target in {"clip", "both"} and set(caption_revisions) != set(normalized_crops):
            raise DatasetPublicationError("every selected crop needs a finalized caption review")
        return _normalize_selection(
            {
                "target": target,
                "sample_ids": normalized_samples,
                "crop_ids": normalized_crops,
                "bbox_revisions": bbox_revisions,
                "caption_revisions": caption_revisions,
                "evaluation_gallery_crop_ids": [],
                "event_links": [],
                "clip_links": [],
                "relevance_reviews": [],
            }
        )

    async def publish_dataset(self, selection: dict[str, Any], config_version: str) -> dict[str, Any]:
        normalized = _normalize_selection(selection)
        if not isinstance(config_version, str) or not config_version.strip() or len(config_version) > 255:
            raise ValueError("config_version must contain 1 to 255 characters")
        code_sha = dataset_code_sha256()
        input_sha = sha256(
            canonical_json(normalized)
            + b"\n"
            + config_version.encode("utf-8")
            + b"\n"
            + code_sha.encode("ascii")
        ).hexdigest()
        dataset_version = f"dataset-{input_sha[:24]}"
        config_payload = canonical_json({"config_version": config_version})
        config_sha = content_sha256(config_payload)
        config_key = f"datasets/{dataset_version}/config.json"
        await self.ensure_schema()

        prepared = await self._prepare_publication(
            selection=normalized,
            config_version=config_version,
            config_sha=config_sha,
            config_key=config_key,
            code_sha=code_sha,
            input_sha=input_sha,
            dataset_version=dataset_version,
        )
        if prepared["mode"] == "response":
            return prepared["response"]

        try:
            await self._publish_objects(prepared, config_payload=config_payload)
        except (FileNotFoundError, _ImmutableSourceError) as error:
            await self._mark_blocked(
                dataset_version=dataset_version,
                reason="required_source_unavailable",
                detail={"error_type": type(error).__name__},
            )
            return await self._result(dataset_version)
        except Exception as error:  # noqa: BLE001 - retain the input identity and byte reservation for retry
            await self._record_publish_failure(dataset_version, type(error).__name__)
            raise

        return await self._finish_publication(dataset_version)

    async def register_model_lineage(self, *, model_id: str, dataset_version: str) -> dict[str, Any]:
        """Record model lineage and backfill any durable impacts without creating artifacts."""
        if not model_id.strip() or len(model_id) > 255:
            raise ValueError("model_id must contain 1 to 255 characters")
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                version_row = await connection.fetchrow(
                    "SELECT state, training_ready, evaluation_eligible, published_at FROM dataset_versions WHERE dataset_version = $1 FOR UPDATE",
                    dataset_version,
                )
                if version_row is None or version_row["published_at"] is None:
                    raise DatasetPublicationError("model lineage requires a previously published dataset")
                await connection.execute(
                    """
                    INSERT INTO dataset_model_lineage (model_id, dataset_version)
                    VALUES ($1, $2) ON CONFLICT DO NOTHING
                    """,
                    model_id,
                    dataset_version,
                )
                await connection.execute(
                    """
                    INSERT INTO dataset_model_impacts (model_id, dataset_version, sample_id, reason)
                    SELECT $1, invalidation.dataset_version, invalidation.sample_id,
                           'source_sample_explicitly_invalidated'
                    FROM dataset_invalidations AS invalidation
                    WHERE invalidation.dataset_version = $2
                    ON CONFLICT DO NOTHING
                    """,
                    model_id,
                    dataset_version,
                )
                await connection.execute(
                    """
                    INSERT INTO dataset_model_impacts (model_id, dataset_version, sample_id, reason)
                    SELECT DISTINCT $1, $2, item.sample_id, 'evaluation_split_leakage'
                    FROM dataset_version_leakage_impacts AS version_impact
                    JOIN dataset_split_leakage_impacts AS impact USING (impact_id)
                    JOIN dataset_items AS item ON item.dataset_version = version_impact.dataset_version
                      AND item.sample_id::text IN (
                          SELECT jsonb_array_elements_text(impact.sample_ids)
                      )
                    WHERE version_impact.dataset_version = $2
                    ON CONFLICT DO NOTHING
                    """,
                    model_id,
                    dataset_version,
                )
                impact_rows = await connection.fetch(
                    """
                    SELECT sample_id, reason FROM dataset_model_impacts
                    WHERE model_id = $1 AND dataset_version = $2
                    ORDER BY reason, sample_id
                    """,
                    model_id,
                    dataset_version,
                )
                return {
                    "model_id": model_id,
                    "dataset_version": dataset_version,
                    "training_eligible": bool(version_row["training_ready"]),
                    "evaluation_eligible": bool(version_row["evaluation_eligible"]),
                    "impacts": [
                        {"sample_id": str(row["sample_id"]), "reason": row["reason"]}
                        for row in impact_rows
                    ],
                }

    async def invalidate_sample(self, sample_id: str) -> dict[str, Any]:
        from .deletion import invalidate_sample_in_repository

        await self.ensure_schema()
        return await invalidate_sample_in_repository(self, sample_id)

    async def preview_sample_invalidation(self, sample_id: str) -> dict[str, Any]:
        """Read dataset, model, review, and active-job impact before explicit invalidation."""
        try:
            sample_uuid = UUID(sample_id)
        except (TypeError, ValueError) as error:
            raise ValueError("sample_id must be a UUID") from error
        await self.ensure_schema()
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            sample = await connection.fetchrow(
                "SELECT sample_id, state, selected FROM ingestion_samples WHERE sample_id = $1",
                sample_uuid,
            )
            if sample is None:
                return {
                    "sample_id": str(sample_uuid),
                    "sample_exists": False,
                    "physical_deletion": False,
                    "datasets": [],
                    "dataset_states": {},
                    "models": [],
                    "active_review_count": 0,
                    "active_job_count": 0,
                    "retained_for_review": False,
                    "block_training": False,
                    "block_evaluation": False,
                }
            versions = await connection.fetch(
                """
                SELECT version.dataset_version, version.state
                FROM dataset_versions AS version
                WHERE version.dataset_version IN (
                    SELECT item.dataset_version FROM dataset_items AS item WHERE item.sample_id = $1
                ) AND version.state IN ('publishing', 'published', 'invalidated')
                ORDER BY version.dataset_version
                """,
                sample_uuid,
            )
            version_ids = [row["dataset_version"] for row in versions]
            if version_ids:
                model_rows = await connection.fetch(
                    """
                    SELECT DISTINCT model_id FROM dataset_model_lineage
                    WHERE dataset_version = ANY($1::text[]) ORDER BY model_id
                    """,
                    version_ids,
                )
                active_jobs = await connection.fetchval(
                    """
                    SELECT count(DISTINCT job.job_id)
                    FROM gods_mlops_jobs AS job
                    JOIN dataset_items AS item ON item.dataset_version = job.dataset_version
                    WHERE item.sample_id = $1
                      AND job.state IN ('queued', 'waiting_profile', 'waiting_gpu', 'waiting_storage',
                                        'waiting_capacity', 'running', 'yield_requested', 'retrying')
                    """,
                    sample_uuid,
                )
            else:
                model_rows = []
                active_jobs = 0
            active_reviews = await connection.fetchval(
                """
                SELECT count(*) FROM review_assignments
                WHERE sample_id = $1 AND state IN ('active', 'provisioning')
                """,
                sample_uuid,
            )
        return {
            "sample_id": str(sample_uuid),
            "sample_exists": True,
            "sample_state": sample["state"],
            "physical_deletion": False,
            "datasets": version_ids,
            "dataset_states": {row["dataset_version"]: row["state"] for row in versions},
            "models": [row["model_id"] for row in model_rows],
            "active_review_count": int(active_reviews or 0),
            "active_job_count": int(active_jobs or 0),
            "retained_for_review": bool(sample["selected"] or active_reviews),
            "block_training": True,
            "block_evaluation": True,
        }

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        ingestion = PostgresIngestionRepository(database_url=self._database_url)
        try:
            await ingestion.ensure_schema()
        finally:
            await ingestion.close()
        await self._annotations.ensure_schema()
        await self._get_pool()
        self._schema_ready = True

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        await self._annotations.close()
        self._schema_ready = False

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=1,
                max_size=8,
                command_timeout=30,
            )
        return self._pool

    async def _prepare_publication(
        self,
        *,
        selection: dict[str, Any],
        config_version: str,
        config_sha: str,
        config_key: str,
        code_sha: str,
        input_sha: str,
        dataset_version: str,
    ) -> dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock(731940116)")
                existing = await connection.fetchrow(
                    "SELECT * FROM dataset_versions WHERE input_sha256 = $1 FOR UPDATE",
                    input_sha,
                )
                if existing is not None:
                    if existing["state"] == "publishing":
                        return await self._resume_publication(connection, existing)
                    return {"mode": "response", "response": _response(existing)}
                prior_block = await connection.fetchrow(
                    "SELECT reason_codes, details FROM dataset_publication_blocks WHERE input_sha256 = $1",
                    input_sha,
                )
                if prior_block is not None:
                    return {
                        "mode": "response",
                        "response": _blocked_result(
                            dataset_version,
                            input_sha,
                            _json_value(prior_block["reason_codes"]),
                            _json_value(prior_block["details"]),
                        ),
                    }

                candidate = await self._load_candidate(connection, selection)
                if candidate.get("blocked"):
                    return await self._persist_block(
                        connection,
                        dataset_version=dataset_version,
                        input_sha=input_sha,
                        reasons=candidate["reasons"],
                        details=candidate["details"],
                    )

                source_sample_ids = candidate["source_sample_ids"]
                review_context = await self._load_review_context(connection, selection)
                authority = await self._load_split_authority_context(
                    connection,
                    candidate=candidate,
                    selection=selection,
                )
                candidate["samples_by_id"].update(authority["samples_by_id"])
                candidate["context_sample_ids"].update(
                    set(authority["samples_by_id"]) - set(candidate["source_sample_ids"])
                )
                lock_sample_ids = {UUID(value) for value in source_sample_ids}
                lock_sample_ids.update(review_context["sample_ids"])
                lock_sample_ids.update(UUID(value) for value in candidate["context_sample_ids"])
                lock_crop_ids = {UUID(value) for value in candidate["crop_ids"]}
                lock_crop_ids.update(review_context["crop_ids"])
                sample_lock_rows = await connection.fetch(
                    """
                    SELECT sample_id FROM ingestion_samples
                    WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id FOR UPDATE
                    """,
                    sorted(lock_sample_ids, key=str),
                )
                if {row["sample_id"] for row in sample_lock_rows} != lock_sample_ids:
                    return await self._persist_block(
                        connection,
                        dataset_version=dataset_version,
                        input_sha=input_sha,
                        reasons=["required_source_unavailable"],
                        details={"missing_sample_ids": sorted(map(str, lock_sample_ids - {row["sample_id"] for row in sample_lock_rows}))},
                    )
                await connection.fetch(
                    """
                    SELECT sample_id FROM sample_annotation_heads
                    WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id FOR UPDATE
                    """,
                    sorted(lock_sample_ids, key=str),
                )
                if lock_crop_ids:
                    await connection.fetch(
                        """
                        SELECT crop_id FROM annotation_crops
                        WHERE crop_id = ANY($1::uuid[]) ORDER BY crop_id FOR UPDATE
                        """,
                        sorted(lock_crop_ids, key=str),
                    )

                candidate = await self._load_candidate(connection, selection)
                if candidate.get("blocked"):
                    return await self._persist_block(
                        connection,
                        dataset_version=dataset_version,
                        input_sha=input_sha,
                        reasons=candidate["reasons"],
                        details=candidate["details"],
                    )
                authority = await self._load_split_authority_context(
                    connection,
                    candidate=candidate,
                    selection=selection,
                )
                authority_sample_ids = set(authority["samples_by_id"])
                authority_lock_ids = {UUID(value) for value in authority_sample_ids}
                already_locked_ids = lock_sample_ids
                missing_lock_ids = authority_lock_ids - already_locked_ids
                if missing_lock_ids:
                    lock_rows = await connection.fetch(
                        """
                        SELECT sample_id FROM ingestion_samples
                        WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id FOR UPDATE
                        """,
                        sorted(missing_lock_ids, key=str),
                    )
                    if {row["sample_id"] for row in lock_rows} != missing_lock_ids:
                        return await self._persist_block(
                            connection,
                            dataset_version=dataset_version,
                            input_sha=input_sha,
                            reasons=["required_source_unavailable"],
                            details={"missing_sample_ids": sorted(map(str, missing_lock_ids - {row["sample_id"] for row in lock_rows}))},
                        )
                candidate["samples_by_id"].update(authority["samples_by_id"])
                candidate["context_sample_ids"].update(
                    authority_sample_ids - set(candidate["source_sample_ids"])
                )
                invalidated_sources = await connection.fetch(
                    """
                    SELECT sample_id FROM dataset_source_invalidations
                    WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id
                    """,
                    [UUID(sample_id) for sample_id in sorted(candidate["source_sample_ids"])],
                )
                if invalidated_sources:
                    return await self._persist_block(
                        connection,
                        dataset_version=dataset_version,
                        input_sha=input_sha,
                        reasons=["source_explicitly_invalidated"],
                        details={"sample_ids": [str(row["sample_id"]) for row in invalidated_sources]},
                    )
                await self._persist_group_link_members(connection, selection)
                split_plan = await self._assign_splits(connection, candidate, selection, authority)
                if split_plan["blocked"]:
                    return await self._persist_block(
                        connection,
                        dataset_version=dataset_version,
                        input_sha=input_sha,
                        reasons=(
                            ["late_cross_boundary_link"]
                            if split_plan["leakage_impacts"]
                            else split_plan.get("block_reasons") or ["invalid_independent_split_structure"]
                        ),
                        details={
                            "leakage_impacts": split_plan["leakage_impacts"],
                            "block_reasons": split_plan.get("block_reasons", []),
                            "split_error": split_plan.get("error"),
                        },
                    )

                items = self._build_items(candidate, split_plan, selection, dataset_version)
                split_counts = _split_counts(items, split_plan, candidate)
                training_reasons, evaluation_reasons = _readiness_reasons(
                    selection["target"], split_counts
                )
                structural_error = _structural_split_error(
                    items,
                    split_counts,
                    candidate,
                    has_authority=split_plan["has_authority"],
                )
                if structural_error:
                    return await self._persist_block(
                        connection,
                        dataset_version=dataset_version,
                        input_sha=input_sha,
                        reasons=[structural_error],
                        details={"split_counts": split_counts},
                    )
                if training_reasons:
                    return await self._persist_block(
                        connection,
                        dataset_version=dataset_version,
                        input_sha=input_sha,
                        reasons=training_reasons,
                        details={"split_counts": split_counts, "scope": "training_inputs"},
                    )

                if selection["target"] in {"clip", "both"}:
                    relevance = await self._freeze_available_relevance(
                        connection,
                        selection=selection,
                        review_context=review_context,
                        candidate=candidate,
                        items=items,
                        split_plan=split_plan,
                    )
                else:
                    relevance = {"reasons": [], "matrices": [], "revision_ids": []}
                evaluation_reasons.extend(relevance["reasons"])
                if selection["target"] in {"clip", "both"} and not relevance["matrices"]:
                    if "human_relevance_truth_missing" not in evaluation_reasons:
                        evaluation_reasons.append("human_relevance_truth_missing")

                adoptions = await self._adopt_sources(
                    connection,
                    dataset_version=dataset_version,
                    target=selection["target"],
                    candidate=candidate,
                    items=items,
                )
                manifest, object_plans = self._make_manifest(
                    dataset_version=dataset_version,
                    input_sha=input_sha,
                    config_version=config_version,
                    config_sha=config_sha,
                    config_key=config_key,
                    code_sha=code_sha,
                    selection=selection,
                    candidate=candidate,
                    split_plan=split_plan,
                    split_counts=split_counts,
                    items=items,
                    relevance=relevance,
                    training_reasons=training_reasons,
                    evaluation_reasons=sorted(set(evaluation_reasons)),
                    adoptions=adoptions,
                )
                manifest_bytes = canonical_json(manifest)
                manifest_sha = content_sha256(manifest_bytes)
                manifest_key = f"datasets/{dataset_version}/manifest.json"
                config_payload = canonical_json({"config_version": config_version})
                object_plans[config_key] = {
                    "object_key": config_key,
                    "purpose": "config",
                    "sha256": config_sha,
                    "size_bytes": len(config_payload),
                    "content_type": "application/json",
                    "source_object_key": None,
                }
                object_plans[manifest_key] = {
                    "object_key": manifest_key,
                    "purpose": "manifest",
                    "sha256": manifest_sha,
                    "size_bytes": len(manifest_bytes),
                    "content_type": "application/json",
                    "source_object_key": None,
                }
                total_reservation = sum(plan["size_bytes"] for plan in object_plans.values())
                usage = await connection.fetchrow(
                    "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE FOR UPDATE"
                )
                if usage is None or usage["used_bytes"] + total_reservation > GLOBAL_OBJECT_LIMIT:
                    raise DatasetQuotaExceededError("dataset copies exceed the shared one TiB object quota")
                updated_usage = await connection.fetchrow(
                    """
                    UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1
                    WHERE singleton = TRUE AND used_bytes + $1 <= $2
                    RETURNING used_bytes
                    """,
                    total_reservation,
                    GLOBAL_OBJECT_LIMIT,
                )
                if updated_usage is None:
                    raise DatasetQuotaExceededError("dataset copies exceed the shared one TiB object quota")

                response_metadata = {
                    "schema_version": 1,
                    "dataset_version": dataset_version,
                    "input_sha256": input_sha,
                    "target": selection["target"],
                    "config_version": config_version,
                    "config_sha256": config_sha,
                    "code_sha256": code_sha,
                    "state": "publishing",
                    "manifest_object_key": manifest_key,
                    "manifest_sha256": manifest_sha,
                    "manifest_size_bytes": len(manifest_bytes),
                    "manifest_json": manifest,
                    "split_counts": split_counts,
                    "training_ready": not training_reasons,
                    "training_reasons": training_reasons,
                    "evaluation_eligible": not evaluation_reasons,
                    "evaluation_reasons": sorted(set(evaluation_reasons)),
                    "relevance_revision_ids": relevance["revision_ids"],
                }
                await connection.execute(
                    """
                    INSERT INTO dataset_versions (
                        dataset_version, input_sha256, target, config_version, config_sha256,
                        code_sha256, state, manifest_object_key, manifest_sha256,
                        manifest_size_bytes, manifest_json, split_counts, training_ready,
                        training_reasons, evaluation_eligible, evaluation_reasons,
                        relevance_revision_ids
                    ) VALUES ($1, $2, $3, $4, $5, $6, 'publishing', $7, $8, $9,
                              $10::jsonb, $11::jsonb, $12, $13::jsonb, $14, $15::jsonb, $16::jsonb)
                    """,
                    dataset_version,
                    input_sha,
                    selection["target"],
                    config_version,
                    config_sha,
                    code_sha,
                    manifest_key,
                    manifest_sha,
                    len(manifest_bytes),
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                    json.dumps(split_counts, ensure_ascii=False, sort_keys=True),
                    not training_reasons,
                    json.dumps(training_reasons),
                    not evaluation_reasons,
                    json.dumps(sorted(set(evaluation_reasons))),
                    json.dumps(relevance["revision_ids"]),
                )
                for sample_id, split in split_plan["assignments"].items():
                    sample = candidate["samples_by_id"][sample_id]
                    await connection.execute(
                        """
                        INSERT INTO dataset_sample_splits (
                            sample_id, split, group_id, camera_id, capture_day, first_dataset_version
                        ) VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT (sample_id) DO NOTHING
                        """,
                        UUID(sample_id),
                        split["split"],
                        split["group_id"],
                        sample["camera_id"],
                        sample["capture_day"],
                        dataset_version,
                    )
                for item in items:
                    await connection.execute(
                        """
                        INSERT INTO dataset_items (
                            dataset_version, item_kind, item_id, sample_id, target, split,
                            group_id, component_id, source_object_key, object_key,
                            source_sha256, object_size_bytes, snapshot
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb)
                        """,
                        dataset_version,
                        item["item_kind"],
                        UUID(item["item_id"]),
                        UUID(item["sample_id"]),
                        item["target"],
                        item["split"],
                        item["group_id"],
                        item["component_id"],
                        item["source_object_key"],
                        item["object_key"],
                        item["source_sha256"],
                        item["object_size_bytes"],
                        json.dumps(item["snapshot"], ensure_ascii=False, sort_keys=True),
                    )
                for plan in object_plans.values():
                    await connection.execute(
                        """
                        INSERT INTO dataset_objects (
                            dataset_version, object_key, purpose, sha256, size_bytes, state
                        ) VALUES ($1, $2, $3, $4, $5, 'reserved')
                        """,
                        dataset_version,
                        plan["object_key"],
                        plan["purpose"],
                        plan["sha256"],
                        plan["size_bytes"],
                    )
                prepared = {
                    "mode": "publish",
                    "dataset_version": dataset_version,
                    "manifest": manifest,
                    "manifest_bytes": manifest_bytes,
                    "manifest_key": manifest_key,
                    "manifest_sha": manifest_sha,
                    "config_payload": config_payload,
                    "config_key": config_key,
                    "object_plans": object_plans,
                    "items": items,
                }
                return prepared

    async def _load_candidate(self, connection: asyncpg.Connection, selection: dict[str, Any]) -> dict[str, Any]:
        frame_ids = [UUID(value) for value in selection["sample_ids"]]
        crop_ids = [UUID(value) for value in selection["crop_ids"]]
        crop_rows = []
        if crop_ids:
            crop_rows = await connection.fetch(
                """
                SELECT crop.crop_id, crop.sample_id, crop.bbox_revision, crop.object_key,
                       crop.sha256, crop.object_size_bytes, crop.state, crop.caption_state,
                       crop.caption_revision_id, crop.provenance, crop.parent_available,
                       crop.regeneration_available, crop.crop_set_ready
                FROM annotation_crops AS crop WHERE crop.crop_id = ANY($1::uuid[])
                ORDER BY crop.crop_id
                """,
                crop_ids,
            )
        crop_map = {row["crop_id"]: row for row in crop_rows}
        missing_crop_ids = sorted(str(crop_id) for crop_id in set(crop_ids) - set(crop_map))
        if missing_crop_ids:
            return _candidate_block("required_crop_unavailable", {"crop_ids": missing_crop_ids})
        source_sample_ids = set(frame_ids) | {row["sample_id"] for row in crop_rows}
        context_sample_ids = {
            UUID(sample_id)
            for link in selection["event_links"] + selection["clip_links"]
            for sample_id in link["sample_ids"]
        }
        all_sample_ids = source_sample_ids | context_sample_ids
        samples = await connection.fetch(
            """
            SELECT sample_id, camera_id, capture_day, captured_at_utc, sha256,
                   model_revision, processor_revision, object_key, object_size_bytes, state
            FROM ingestion_samples WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id
            """,
            sorted(all_sample_ids, key=str),
        )
        samples_by_id = {str(row["sample_id"]): row for row in samples}
        missing_samples = sorted(str(sample_id) for sample_id in all_sample_ids if str(sample_id) not in samples_by_id)
        if missing_samples:
            return _candidate_block("required_sample_unavailable", {"sample_ids": missing_samples})

        heads: dict[UUID, asyncpg.Record] = {}
        bbox_revision_ids: set[UUID] = set()
        head_ids = set(frame_ids) | {row["sample_id"] for row in crop_rows}
        if head_ids:
            head_rows = await connection.fetch(
                """
                SELECT sample_id, latest_bbox_revision, current_bbox_assignment_revision
                FROM sample_annotation_heads WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id
                """,
                sorted(head_ids, key=str),
            )
            heads = {row["sample_id"]: row for row in head_rows}
            for sample_id in frame_ids:
                head = heads.get(sample_id)
                if head is None or head["latest_bbox_revision"] is None or head["current_bbox_assignment_revision"] is not None:
                    return _candidate_block(
                        "finalized_bbox_required",
                        {"sample_id": str(sample_id), "reason": "missing_or_pending_bbox_revision"},
                    )
                bbox_revision_ids.add(head["latest_bbox_revision"])
        for row in crop_rows:
            bbox_revision_ids.add(row["bbox_revision"])

        bbox_rows = await connection.fetch(
            """
            SELECT annotation_revision_id, sample_id, stage, result, result_sha256,
                   provenance, label_studio_task_id, label_studio_annotation_id, submitted_at
            FROM annotation_revisions WHERE annotation_revision_id = ANY($1::uuid[])
            """,
            sorted(bbox_revision_ids, key=str),
        ) if bbox_revision_ids else []
        annotations_by_id = {row["annotation_revision_id"]: row for row in bbox_rows}
        caption_ids = {row["caption_revision_id"] for row in crop_rows if row["caption_revision_id"]}
        caption_rows = await connection.fetch(
            """
            SELECT annotation_revision_id, sample_id, stage, result, result_sha256,
                   provenance, label_studio_task_id, label_studio_annotation_id, submitted_at
            FROM annotation_revisions WHERE annotation_revision_id = ANY($1::uuid[])
            """,
            sorted(caption_ids, key=str),
        ) if caption_ids else []
        annotations_by_id.update({row["annotation_revision_id"]: row for row in caption_rows})

        for sample_id in frame_ids:
            sample = samples_by_id[str(sample_id)]
            head = heads[sample_id]
            revision = annotations_by_id.get(head["latest_bbox_revision"])
            if sample["state"] != "received" or revision is None or revision["stage"] != "bbox":
                return _candidate_block(
                    "finalized_bbox_required",
                    {"sample_id": str(sample_id), "reason": "retained_frame_or_bbox_missing"},
                )
            if str(head["latest_bbox_revision"]) != selection["bbox_revisions"][str(sample_id)]:
                return _candidate_block(
                    "bbox_revision_changed_before_publication",
                    {"sample_id": str(sample_id), "current_revision": str(head["latest_bbox_revision"])},
                )
        for row in crop_rows:
            sample = samples_by_id[str(row["sample_id"])]
            head = heads.get(row["sample_id"])
            bbox = annotations_by_id.get(row["bbox_revision"])
            caption = annotations_by_id.get(row["caption_revision_id"])
            if sample["state"] not in {"received", "purge_pending", "expired"}:
                return _candidate_block("required_crop_parent_unavailable", {"sample_id": str(row["sample_id"])})
            if (
                row["state"] != "ready"
                or not row["crop_set_ready"]
                or row["caption_state"] != "reviewed"
                or caption is None
                or caption["stage"] != "caption"
                or caption["sample_id"] != row["sample_id"]
            ):
                return _candidate_block("reviewed_crop_caption_required", {"crop_id": str(row["crop_id"])})
            if bbox is None or bbox["stage"] != "bbox" or bbox["sample_id"] != row["sample_id"]:
                return _candidate_block("crop_bbox_revision_missing", {"crop_id": str(row["crop_id"])})
            if (
                head is None
                or head["latest_bbox_revision"] != row["bbox_revision"]
                or head["current_bbox_assignment_revision"] is not None
            ):
                return _candidate_block("crop_bbox_revision_not_current", {"crop_id": str(row["crop_id"])})
            if str(row["caption_revision_id"]) != selection["caption_revisions"][str(row["crop_id"])]:
                return _candidate_block(
                    "caption_revision_changed_before_publication",
                    {"crop_id": str(row["crop_id"]), "current_revision": str(row["caption_revision_id"])},
                )

        return {
            "source_sample_ids": {str(item) for item in source_sample_ids},
            "context_sample_ids": {str(item) for item in context_sample_ids - source_sample_ids},
            "samples_by_id": samples_by_id,
            "frame_ids": {str(item) for item in frame_ids},
            "crop_ids": {str(item) for item in crop_ids},
            "crop_rows": crop_map,
            "heads": heads,
            "annotations_by_id": annotations_by_id,
        }

    async def _load_split_authority_context(
        self,
        connection: asyncpg.Connection,
        *,
        candidate: dict[str, Any],
        selection: dict[str, Any],
    ) -> dict[str, Any]:
        """Expand current samples through persisted camera/day and event/clip edges."""
        samples_by_id = dict(candidate["samples_by_id"])
        links_by_key: dict[tuple[str, str], set[str]] = {}
        for kind, selection_key in (("event", "event_links"), ("clip", "clip_links")):
            for link in selection[selection_key]:
                links_by_key[(kind, link["link_id"])] = set(link["sample_ids"])

        while True:
            known_ids = {UUID(sample_id) for sample_id in samples_by_id}
            if known_ids:
                day_rows = await connection.fetch(
                    """
                    SELECT related.sample_id, related.camera_id, related.capture_day,
                           related.captured_at_utc, related.sha256, related.model_revision,
                           related.processor_revision, related.object_key, related.object_size_bytes,
                           related.state
                    FROM ingestion_samples AS related
                    WHERE EXISTS (
                        SELECT 1 FROM ingestion_samples AS seed
                        WHERE seed.sample_id = ANY($1::uuid[])
                          AND seed.camera_id = related.camera_id
                          AND seed.capture_day = related.capture_day
                    )
                    ORDER BY related.sample_id
                    """,
                    sorted(known_ids, key=str),
                )
                for row in day_rows:
                    samples_by_id[str(row["sample_id"])] = row

            known_ids = {UUID(sample_id) for sample_id in samples_by_id}
            link_keys = sorted(links_by_key)
            kind_ids = [key[0] for key in link_keys]
            link_ids = [key[1] for key in link_keys]
            edge_rows = await connection.fetch(
                """
                SELECT link_kind, link_id, sample_id FROM dataset_group_link_members
                WHERE sample_id = ANY($1::uuid[])
                   OR EXISTS (
                       SELECT 1 FROM unnest($2::text[], $3::text[]) AS wanted(kind, id)
                       WHERE wanted.kind = dataset_group_link_members.link_kind
                         AND wanted.id = dataset_group_link_members.link_id
                   )
                ORDER BY link_kind, link_id, sample_id
                """,
                sorted(known_ids, key=str),
                kind_ids,
                link_ids,
            )
            before = (len(samples_by_id), sum(len(members) for members in links_by_key.values()))
            for row in edge_rows:
                links_by_key.setdefault((row["link_kind"], row["link_id"]), set()).add(str(row["sample_id"]))
            linked_ids = {
                sample_id
                for members in links_by_key.values()
                for sample_id in members
            }
            missing_link_ids = sorted(linked_ids - set(samples_by_id))
            if missing_link_ids:
                member_rows = await connection.fetch(
                    """
                    SELECT sample_id, camera_id, capture_day, captured_at_utc, sha256,
                           model_revision, processor_revision, object_key, object_size_bytes, state
                    FROM ingestion_samples WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id
                    """,
                    [UUID(sample_id) for sample_id in missing_link_ids],
                )
                samples_by_id.update({str(row["sample_id"]): row for row in member_rows})
            after = (len(samples_by_id), sum(len(members) for members in links_by_key.values()))
            if after == before:
                break

        sample_ids = sorted(samples_by_id)
        prior_rows = await connection.fetch(
            """
            SELECT sample_id, split, group_id FROM dataset_sample_splits
            WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id
            """,
            [UUID(sample_id) for sample_id in sample_ids],
        )
        prior = {
            str(row["sample_id"]): {"split": row["split"], "group_id": row["group_id"]}
            for row in prior_rows
        }
        has_authority = bool(
            await connection.fetchval("SELECT EXISTS (SELECT 1 FROM dataset_sample_splits)")
        )
        return {
            "samples_by_id": samples_by_id,
            "prior": prior,
            "has_authority": has_authority,
            "event_links": [
                {"link_id": link_id, "sample_ids": sorted(members)}
                for (kind, link_id), members in sorted(links_by_key.items())
                if kind == "event"
            ],
            "clip_links": [
                {"link_id": link_id, "sample_ids": sorted(members)}
                for (kind, link_id), members in sorted(links_by_key.items())
                if kind == "clip"
            ],
        }

    async def _persist_group_link_members(
        self,
        connection: asyncpg.Connection,
        selection: dict[str, Any],
    ) -> None:
        for kind, selection_key in (("event", "event_links"), ("clip", "clip_links")):
            for link in selection[selection_key]:
                for sample_id in link["sample_ids"]:
                    await connection.execute(
                        """
                        INSERT INTO dataset_group_link_members (link_kind, link_id, sample_id)
                        VALUES ($1, $2, $3) ON CONFLICT DO NOTHING
                        """,
                        kind,
                        link["link_id"],
                        UUID(sample_id),
                    )

    async def _assign_splits(
        self,
        connection: asyncpg.Connection,
        candidate: dict[str, Any],
        selection: dict[str, Any],
        authority: dict[str, Any],
    ) -> dict[str, Any]:
        prior = authority["prior"]
        samples = [
            {
                "sample_id": sample_id,
                "camera_id": str(row["camera_id"]),
                "capture_day": row["capture_day"],
                "captured_at_utc": row["captured_at_utc"],
            }
            for sample_id, row in candidate["samples_by_id"].items()
        ]
        try:
            split_plan = plan_splits(
                samples,
                event_links=authority["event_links"],
                clip_links=authority["clip_links"],
                prior_assignments=prior,
                has_authority=authority["has_authority"],
            )
            split_plan["event_links"] = authority["event_links"]
            split_plan["clip_links"] = authority["clip_links"]
            return split_plan
        except ValueError as error:
            return {
                "blocked": True,
                "block_reasons": ["invalid_independent_split_structure"],
                "leakage_impacts": [],
                "error": str(error),
                "assignments": {},
                "excluded": [],
                "group_counts": {"train": 0, "validation": 0, "test": 0},
                "prior": prior,
                "has_authority": authority["has_authority"],
            }

    def _build_items(
        self,
        candidate: dict[str, Any],
        split_plan: dict[str, Any],
        selection: dict[str, Any],
        dataset_version: str,
    ) -> list[dict[str, Any]]:
        assignments = split_plan["assignments"]
        items: list[dict[str, Any]] = []
        for sample_id in sorted(candidate["frame_ids"]):
            if sample_id not in assignments:
                continue
            sample = candidate["samples_by_id"][sample_id]
            head = candidate["heads"][UUID(sample_id)]
            annotation = candidate["annotations_by_id"][head["latest_bbox_revision"]]
            result = _json_value(annotation["result"])
            result_sha = annotation["result_sha256"].strip()
            if content_sha256(canonical_json(result)) != result_sha:
                raise DatasetPublicationError("finalized bbox result hash does not match its snapshot")
            is_positive = _has_person(result)
            source_sha = sample["sha256"].strip()
            destination = f"datasets/{dataset_version}/media/{source_sha}.jpg"
            snapshot = {
                "sample": {
                    "sample_id": sample_id,
                    "camera_id": str(sample["camera_id"]),
                    "capture_day": sample["capture_day"].isoformat(),
                    "captured_at_utc": sample["captured_at_utc"].astimezone(timezone.utc).isoformat(),
                    "sha256": source_sha,
                    "object_size_bytes": sample["object_size_bytes"],
                    "model_revision": sample["model_revision"],
                    "processor_revision": sample["processor_revision"],
                },
                "annotation": _annotation_snapshot(annotation, result),
                "has_person": is_positive,
            }
            items.append(
                {
                    "item_kind": "frame",
                    "item_id": sample_id,
                    "sample_id": sample_id,
                    "target": "detr",
                    "split": assignments[sample_id]["split"],
                    "group_id": assignments[sample_id]["group_id"],
                    "component_id": split_plan["component_by_sample"][sample_id],
                    "source_object_key": sample["object_key"],
                    "object_key": destination,
                    "source_sha256": source_sha,
                    "object_size_bytes": sample["object_size_bytes"],
                    "snapshot": snapshot,
                }
            )
        for crop_id in sorted(candidate["crop_ids"]):
            row = candidate["crop_rows"][UUID(crop_id)]
            sample_id = str(row["sample_id"])
            if sample_id not in assignments:
                continue
            sample = candidate["samples_by_id"][sample_id]
            bbox = candidate["annotations_by_id"][row["bbox_revision"]]
            caption = candidate["annotations_by_id"][row["caption_revision_id"]]
            bbox_result = _json_value(bbox["result"])
            caption_result = _json_value(caption["result"])
            bbox_sha, caption_sha = bbox["result_sha256"].strip(), caption["result_sha256"].strip()
            if content_sha256(canonical_json(bbox_result)) != bbox_sha:
                raise DatasetPublicationError("crop bbox result hash does not match its snapshot")
            if content_sha256(canonical_json(caption_result)) != caption_sha:
                raise DatasetPublicationError("reviewed caption result hash does not match its snapshot")
            caption_text = _caption_text(caption_result)
            crop_sha = row["sha256"].strip()
            destination = f"datasets/{dataset_version}/media/{crop_sha}.jpg"
            provenance = _json_value(row["provenance"])
            parent_available = bool(row["parent_available"] and sample["state"] == "received")
            regeneration_available = bool(row["regeneration_available"] and sample["state"] == "received")
            snapshot = {
                "sample": {
                    "sample_id": sample_id,
                    "camera_id": str(sample["camera_id"]),
                    "capture_day": sample["capture_day"].isoformat(),
                    "captured_at_utc": sample["captured_at_utc"].astimezone(timezone.utc).isoformat(),
                },
                "crop": {
                    "crop_id": crop_id,
                    "bbox_revision": str(row["bbox_revision"]),
                    "sha256": crop_sha,
                    "object_size_bytes": row["object_size_bytes"],
                    "source_object_key": row["object_key"],
                    "parent_available": parent_available,
                    "regeneration_available": regeneration_available,
                    "provenance": provenance,
                },
                "bbox_annotation": _annotation_snapshot(bbox, bbox_result),
                "caption": {
                    "revision_id": str(caption["annotation_revision_id"]),
                    "sha256": caption_sha,
                    "text": caption_text,
                    "provenance": _json_value(caption["provenance"]),
                    "label_studio_task_id": caption["label_studio_task_id"],
                    "label_studio_annotation_id": caption["label_studio_annotation_id"],
                    "submitted_at": caption["submitted_at"].astimezone(timezone.utc).isoformat(),
                },
            }
            items.append(
                {
                    "item_kind": "crop",
                    "item_id": crop_id,
                    "sample_id": sample_id,
                    "target": "clip",
                    "split": assignments[sample_id]["split"],
                    "group_id": assignments[sample_id]["group_id"],
                    "component_id": split_plan["component_by_sample"][sample_id],
                    "source_object_key": row["object_key"],
                    "object_key": destination,
                    "source_sha256": crop_sha,
                    "object_size_bytes": row["object_size_bytes"],
                    "snapshot": snapshot,
                }
            )
        return items

    async def _load_review_context(
        self,
        connection: asyncpg.Connection,
        selection: dict[str, Any],
    ) -> dict[str, Any]:
        review_ids = [UUID(item["review_id"]) for item in selection["relevance_reviews"]]
        if not review_ids:
            return {"sample_ids": set(), "crop_ids": set(), "drafts": {}}
        rows = await connection.fetch(
            "SELECT * FROM relevance_matrix_review_drafts WHERE review_id = ANY($1::uuid[])",
            review_ids,
        )
        drafts = {row["review_id"]: row for row in rows}
        gallery_rows = await connection.fetch(
            """
            SELECT review_id, crop_id, crop_sha256 FROM relevance_matrix_review_crops
            WHERE review_id = ANY($1::uuid[]) ORDER BY review_id, crop_id
            """,
            review_ids,
        )
        gallery_by_review: dict[UUID, list[dict[str, str]]] = {}
        crop_ids = {UUID(value) for value in selection["evaluation_gallery_crop_ids"]}
        for row in gallery_rows:
            gallery_by_review.setdefault(row["review_id"], []).append(
                {"crop_id": str(row["crop_id"]), "sha256": row["crop_sha256"].strip()}
            )
            crop_ids.add(row["crop_id"])
        sample_ids: set[UUID] = set()
        if crop_ids:
            parent_rows = await connection.fetch(
                "SELECT DISTINCT sample_id FROM annotation_crops WHERE crop_id = ANY($1::uuid[])",
                sorted(crop_ids, key=str),
            )
            sample_ids.update(row["sample_id"] for row in parent_rows)
        caption_ids = {
            row["query_source_caption_revision_id"]
            for row in rows
            if row["query_source_caption_revision_id"] is not None
        }
        if caption_ids:
            caption_rows = await connection.fetch(
                "SELECT sample_id FROM annotation_revisions WHERE annotation_revision_id = ANY($1::uuid[])",
                sorted(caption_ids, key=str),
            )
            sample_ids.update(row["sample_id"] for row in caption_rows)
        return {"sample_ids": sample_ids, "crop_ids": crop_ids, "drafts": drafts, "gallery": gallery_by_review}

    async def _freeze_available_relevance(
        self,
        connection: asyncpg.Connection,
        *,
        selection: dict[str, Any],
        review_context: dict[str, Any],
        candidate: dict[str, Any],
        items: list[dict[str, Any]],
        split_plan: dict[str, Any],
    ) -> dict[str, Any]:
        reasons: list[str] = []
        matrices: list[dict[str, Any]] = []
        revisions: list[str] = []
        clip_items = {item["item_id"]: item for item in items if item["item_kind"] == "crop"}
        evaluation_gallery = set(selection["evaluation_gallery_crop_ids"])
        if not selection["relevance_reviews"] or not evaluation_gallery:
            return {"reasons": ["human_relevance_truth_missing"], "matrices": [], "revision_ids": []}
        if not evaluation_gallery.issubset(clip_items):
            reasons.append("evaluation_gallery_not_in_published_test_set")
        elif any(clip_items[crop_id]["split"] != "test" for crop_id in evaluation_gallery):
            reasons.append("evaluation_gallery_not_in_published_test_set")
        if reasons:
            return {"reasons": reasons, "matrices": [], "revision_ids": []}

        request_by_id = {UUID(item["review_id"]): item for item in selection["relevance_reviews"]}
        for review_id in sorted(request_by_id, key=str):
            request = request_by_id[review_id]
            draft = await connection.fetchrow(
                "SELECT * FROM relevance_matrix_review_drafts WHERE review_id = $1",
                review_id,
            )
            if draft is None:
                reasons.append("human_relevance_truth_missing")
                matrices.append({"review_id": str(review_id), "state": "missing", "evaluation_eligible": False})
                continue
            gallery = review_context["gallery"].get(review_id, [])
            gallery_ids = {item["crop_id"] for item in gallery}
            if gallery_ids != evaluation_gallery:
                reasons.append("human_relevance_gallery_mismatch")
                matrices.append(_draft_snapshot(draft, gallery, state="gallery_mismatch"))
                continue

            revision_id = draft["frozen_relevance_revision_id"] if draft["state"] == "frozen" else None
            if draft["state"] == "pending":
                judgments = request.get("judgments")
                provenance = request.get("provenance")
                if not _complete_human_matrix(gallery, judgments, provenance):
                    reasons.append("human_relevance_truth_missing_or_incomplete")
                    matrices.append(_draft_snapshot(draft, gallery, state="pending"))
                    continue
                frozen = await self._annotations.freeze_relevance_review_in_transaction(
                    connection,
                    review_id=str(review_id),
                    judgments=judgments,
                    provenance=provenance,
                )
                if frozen["state"] != "frozen":
                    reasons.append("human_relevance_truth_invalidated")
                    current_draft = await connection.fetchrow(
                        "SELECT * FROM relevance_matrix_review_drafts WHERE review_id = $1",
                        review_id,
                    )
                    matrices.append(_draft_snapshot(current_draft, gallery, state=frozen["state"]))
                    continue
                revision_id = UUID(frozen["frozen_relevance_revision_id"])
            elif draft["state"] != "frozen":
                reasons.append("human_relevance_truth_invalidated")
                matrices.append(_draft_snapshot(draft, gallery, state=draft["state"]))
                continue

            revision = await self._read_relevance_revision(connection, revision_id)
            if revision is None:
                reasons.append("human_relevance_truth_missing")
                continue
            revisions.append(str(revision_id))
            selected_gallery = [
                {"crop_id": crop_id, "sha256": clip_items[crop_id]["source_sha256"]}
                for crop_id in sorted(evaluation_gallery)
            ]
            revision_gallery = [
                {"crop_id": item["crop_id"], "sha256": item["sha256"]}
                for item in revision["judgments"]
            ]
            valid_gallery = selected_gallery == [
                {"crop_id": item["crop_id"], "sha256": item["sha256"]}
                for item in sorted(revision_gallery, key=lambda row: row["crop_id"])
            ]
            revision["gallery_matches_selection"] = valid_gallery
            matrices.append(revision)
            if not valid_gallery:
                reasons.append("human_relevance_gallery_mismatch")
            if not revision["evaluation_eligible"] or revision["status"] != "complete":
                reasons.append("human_relevance_truth_unresolved")
        return {
            "reasons": sorted(set(reasons)),
            "matrices": matrices,
            "revision_ids": revisions,
        }

    async def _read_relevance_revision(
        self,
        connection: asyncpg.Connection,
        revision_id: UUID,
    ) -> dict[str, Any] | None:
        row = await connection.fetchrow(
            "SELECT * FROM relevance_matrix_revisions WHERE relevance_revision_id = $1",
            revision_id,
        )
        if row is None:
            return None
        judgments = await connection.fetch(
            """
            SELECT crop_id, crop_sha256, judgment FROM relevance_judgments
            WHERE relevance_revision_id = $1 ORDER BY crop_id
            """,
            revision_id,
        )
        return {
            "relevance_revision_id": str(row["relevance_revision_id"]),
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "query_revision": row["query_revision"],
            "query_sha256": row["query_sha256"].strip(),
            "gallery_sha256": row["gallery_sha256"].strip(),
            "judgments_sha256": row["judgments_sha256"].strip(),
            "status": row["status"],
            "evaluation_eligible": row["evaluation_eligible"],
            "provenance": _json_value(row["provenance"]),
            "judgments": [
                {
                    "crop_id": str(item["crop_id"]),
                    "sha256": item["crop_sha256"].strip(),
                    "judgment": item["judgment"],
                }
                for item in judgments
            ],
        }

    async def _adopt_sources(
        self,
        connection: asyncpg.Connection,
        *,
        dataset_version: str,
        target: str,
        candidate: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        adoptions: list[dict[str, Any]] = []
        for item in items:
            sample_id = UUID(item["sample_id"])
            if item["item_kind"] == "frame":
                head = candidate["heads"][sample_id]
                adoptions.append(
                    await self._annotations.adopt_for_dataset_in_transaction(
                        connection,
                        dataset_version=dataset_version,
                        target="detr",
                        sample_id=sample_id,
                        bbox_revision=head["latest_bbox_revision"],
                    )
                )
            else:
                crop = candidate["crop_rows"][UUID(item["item_id"])]
                adoptions.append(
                    await self._annotations.adopt_for_dataset_in_transaction(
                        connection,
                        dataset_version=dataset_version,
                        target="clip",
                        sample_id=sample_id,
                        bbox_revision=crop["bbox_revision"],
                        crop_id=crop["crop_id"],
                        caption_revision=crop["caption_revision_id"],
                    )
                )
        return adoptions

    def _make_manifest(
        self,
        *,
        dataset_version: str,
        input_sha: str,
        config_version: str,
        config_sha: str,
        config_key: str,
        code_sha: str,
        selection: dict[str, Any],
        candidate: dict[str, Any],
        split_plan: dict[str, Any],
        split_counts: dict[str, Any],
        items: list[dict[str, Any]],
        relevance: dict[str, Any],
        training_reasons: list[str],
        evaluation_reasons: list[str],
        adoptions: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        object_plans: dict[str, dict[str, Any]] = {}
        manifest_items: list[dict[str, Any]] = []
        for item in items:
            kind = item["item_kind"]
            object_plans.setdefault(
                item["object_key"],
                {
                    "object_key": item["object_key"],
                    "purpose": kind,
                    "sha256": item["source_sha256"],
                    "size_bytes": item["object_size_bytes"],
                    "content_type": "image/jpeg",
                    "source_object_key": item["source_object_key"],
                },
            )
            manifest_items.append(
                {
                    "kind": kind,
                    "item_id": item["item_id"],
                    "sample_id": item["sample_id"],
                    "target": item["target"],
                    "split": item["split"],
                    "group_id": item["group_id"],
                    "component_id": item["component_id"],
                    "object": {
                        "key": item["object_key"],
                        "sha256": item["source_sha256"],
                        "size_bytes": item["object_size_bytes"],
                    },
                    "snapshot": item["snapshot"],
                }
            )
        manifest = {
            "schema_version": 1,
            "dataset_version": dataset_version,
            "input_sha256": input_sha,
            "target": selection["target"],
            "time_semantics": {"timezone": "Asia/Seoul", "capture_day": "captured_at_utc converted to Asia/Seoul"},
            "split_policy": {
                "target_ratio": {"train": 0.6, "validation": 0.2, "test": 0.2},
                "grouping": ["camera_capture_day", "event_links", "clip_links"],
                "event_links": split_plan.get("event_links", selection["event_links"]),
                "clip_links": split_plan.get("clip_links", selection["clip_links"]),
                "midnight_boundary_exclusion_seconds": 300,
                "new_groups": ["train", "validation"],
                "assignments": [
                    {
                        "sample_id": sample_id,
                        **assignment,
                        "component_id": split_plan["component_by_sample"].get(sample_id),
                    }
                    for sample_id, assignment in sorted(split_plan["assignments"].items())
                    if sample_id in candidate["source_sample_ids"]
                ],
                "excluded": split_plan["excluded"],
            },
            "split_counts": split_counts,
            "config": {
                "version": config_version,
                "sha256": config_sha,
                "object_key": config_key,
                "snapshot": {"config_version": config_version},
            },
            "code": {
                "package": "gods_mlops.datasets",
                "version": _package_version(),
                "sha256": code_sha,
            },
            "items": sorted(manifest_items, key=lambda item: (item["split"], item["kind"], item["item_id"])),
            "training": {"ready": not training_reasons, "reason_codes": training_reasons},
            "evaluation": {
                "eligible": not evaluation_reasons,
                "reason_codes": evaluation_reasons,
                "gallery_crop_ids": sorted(selection["evaluation_gallery_crop_ids"]),
                "relevance_matrices": relevance["matrices"],
            },
            "annotation_adoptions": sorted(
                [
                    {
                        "dataset_version": item["dataset_version"],
                        "target": item["target"],
                        "sample_id": item["sample_id"],
                        "annotation_revision_id": item["annotation_revision_id"],
                        "crop_id": item["crop_id"],
                        "caption_revision_id": item["caption_revision_id"],
                    }
                    for item in adoptions
                ],
                key=lambda item: (item["target"], item["sample_id"], str(item["crop_id"])),
            ),
        }
        return manifest, object_plans

    async def _resume_publication(
        self,
        connection: asyncpg.Connection,
        version_row: asyncpg.Record,
    ) -> dict[str, Any]:
        dataset_version = version_row["dataset_version"]
        rows = await connection.fetch(
            """
            SELECT item_kind, source_object_key, source_sha256, object_size_bytes, object_key
            FROM dataset_items WHERE dataset_version = $1 ORDER BY item_kind, item_id
            """,
            dataset_version,
        )
        plans: dict[str, dict[str, Any]] = {}
        for row in rows:
            plans.setdefault(
                row["object_key"],
                {
                    "object_key": row["object_key"],
                    "purpose": row["item_kind"],
                    "sha256": row["source_sha256"].strip(),
                    "size_bytes": row["object_size_bytes"],
                    "content_type": "image/jpeg",
                    "source_object_key": row["source_object_key"],
                },
            )
        object_rows = await connection.fetch(
            "SELECT object_key, purpose, sha256, size_bytes FROM dataset_objects WHERE dataset_version = $1",
            dataset_version,
        )
        manifest = _json_value(version_row["manifest_json"])
        for row in object_rows:
            if row["purpose"] == "config":
                plans[row["object_key"]] = {
                    "object_key": row["object_key"],
                    "purpose": "config",
                    "sha256": row["sha256"].strip(),
                    "size_bytes": row["size_bytes"],
                    "content_type": "application/json",
                    "source_object_key": None,
                }
            elif row["purpose"] == "manifest":
                plans[row["object_key"]] = {
                    "object_key": row["object_key"],
                    "purpose": "manifest",
                    "sha256": row["sha256"].strip(),
                    "size_bytes": row["size_bytes"],
                    "content_type": "application/json",
                    "source_object_key": None,
                }
        return {
            "mode": "publish",
            "dataset_version": dataset_version,
            "manifest": manifest,
            "manifest_bytes": canonical_json(manifest),
            "manifest_key": version_row["manifest_object_key"],
            "manifest_sha": version_row["manifest_sha256"].strip(),
            "config_payload": canonical_json(manifest["config"]["snapshot"]),
            "config_key": manifest["config"]["object_key"],
            "object_plans": plans,
            "items": rows,
        }

    async def _publish_objects(self, prepared: dict[str, Any], *, config_payload: bytes) -> None:
        for item in prepared["items"]:
            source = await asyncio.to_thread(
                self._objects.read_source,
                object_key=item["source_object_key"],
                sha256_digest=item["source_sha256"].strip(),
                size_bytes=item["object_size_bytes"],
            )
            await asyncio.to_thread(
                self._objects.write_immutable,
                object_key=item["object_key"],
                content=source,
                sha256_digest=item["source_sha256"].strip(),
                content_type="image/jpeg",
            )
        await asyncio.to_thread(
            self._objects.write_immutable,
            object_key=prepared["config_key"],
            content=config_payload,
            sha256_digest=content_sha256(config_payload),
            content_type="application/json",
        )
        await asyncio.to_thread(
            self._objects.write_immutable,
            object_key=prepared["manifest_key"],
            content=prepared["manifest_bytes"],
            sha256_digest=prepared["manifest_sha"],
            content_type="application/json",
        )

    async def _finish_publication(self, dataset_version: str) -> dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    "SELECT state, training_reasons, evaluation_reasons FROM dataset_versions WHERE dataset_version = $1 FOR UPDATE",
                    dataset_version,
                )
                if row is None:
                    raise RuntimeError("dataset publication reservation disappeared")
                if row["state"] == "invalidated":
                    return await self._result_with_connection(connection, dataset_version)
                if row["state"] != "publishing":
                    return await self._result_with_connection(connection, dataset_version)
                invalidated_samples = await connection.fetch(
                    """
                    SELECT DISTINCT item.sample_id
                    FROM dataset_items AS item
                    JOIN dataset_source_invalidations AS source_invalidation USING (sample_id)
                    WHERE item.dataset_version = $1 ORDER BY item.sample_id
                    """,
                    dataset_version,
                )
                if invalidated_samples:
                    sample_ids = [row["sample_id"] for row in invalidated_samples]
                    training_reasons = sorted(
                        set([*_json_value(row["training_reasons"]), "sample_explicitly_invalidated"])
                    )
                    evaluation_reasons = sorted(
                        set([*_json_value(row["evaluation_reasons"]), "sample_explicitly_invalidated"])
                    )
                    await connection.execute(
                        """
                        INSERT INTO dataset_invalidations (dataset_version, sample_id, reason)
                        SELECT $1, sample_id, 'sample_explicitly_invalidated'
                        FROM unnest($2::uuid[]) AS source(sample_id)
                        ON CONFLICT DO NOTHING
                        """,
                        dataset_version,
                        sample_ids,
                    )
                    await connection.execute(
                        """
                        UPDATE dataset_versions SET state = 'invalidated', training_ready = FALSE,
                            evaluation_eligible = FALSE, training_reasons = $2::jsonb,
                            evaluation_reasons = $3::jsonb, invalidated_at = now()
                        WHERE dataset_version = $1 AND state = 'publishing'
                        """,
                        dataset_version,
                        json.dumps(training_reasons),
                        json.dumps(evaluation_reasons),
                    )
                    return await self._result_with_connection(connection, dataset_version)
                leakage_overlay = await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM dataset_version_leakage_impacts
                        WHERE dataset_version = $1
                    )
                    """,
                    dataset_version,
                )
                if leakage_overlay:
                    evaluation_reasons = sorted(
                        set([*_json_value(row["evaluation_reasons"]), "late_cross_boundary_link"])
                    )
                    await connection.execute(
                        """
                        UPDATE dataset_versions SET evaluation_eligible = FALSE,
                            evaluation_reasons = $2::jsonb
                        WHERE dataset_version = $1 AND state = 'publishing'
                        """,
                        dataset_version,
                        json.dumps(evaluation_reasons),
                    )
                await connection.execute(
                    """
                    UPDATE dataset_objects SET state = 'verified', verified_at = now()
                    WHERE dataset_version = $1 AND state = 'reserved'
                    """,
                    dataset_version,
                )
                await connection.execute(
                    """
                    UPDATE dataset_versions SET state = 'published', published_at = now(), last_error = NULL
                    WHERE dataset_version = $1 AND state = 'publishing'
                    """,
                    dataset_version,
                )
                return await self._result_with_connection(connection, dataset_version)

    async def _mark_blocked(self, *, dataset_version: str, reason: str, detail: dict[str, Any]) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    UPDATE dataset_versions SET state = 'blocked', training_ready = FALSE,
                        evaluation_eligible = FALSE, training_reasons = $2::jsonb,
                        evaluation_reasons = $2::jsonb, last_error = $3
                    WHERE dataset_version = $1 AND state = 'publishing'
                    """,
                    dataset_version,
                    json.dumps([reason]),
                    json.dumps(detail, ensure_ascii=False, sort_keys=True)[:1000],
                )

    async def _record_publish_failure(self, dataset_version: str, error_type: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE dataset_versions SET last_error = $2 WHERE dataset_version = $1 AND state = 'publishing'",
                dataset_version,
                error_type[:128],
            )

    async def _result(self, dataset_version: str) -> dict[str, Any]:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            return await self._result_with_connection(connection, dataset_version)

    async def _result_with_connection(
        self,
        connection: asyncpg.Connection,
        dataset_version: str,
    ) -> dict[str, Any]:
        row = await connection.fetchrow(
            "SELECT * FROM dataset_versions WHERE dataset_version = $1",
            dataset_version,
        )
        if row is None:
            raise RuntimeError("dataset publication record does not exist")
        return _response(row)

    async def _persist_block(
        self,
        connection: asyncpg.Connection,
        *,
        dataset_version: str,
        input_sha: str,
        reasons: list[str],
        details: dict[str, Any],
    ) -> dict[str, Any]:
        await connection.execute(
            """
            INSERT INTO dataset_publication_blocks (input_sha256, dataset_version, reason_codes, details)
            VALUES ($1, $2, $3::jsonb, $4::jsonb) ON CONFLICT (input_sha256) DO NOTHING
            """,
            input_sha,
            dataset_version,
            json.dumps(sorted(set(reasons))),
            json.dumps(details, ensure_ascii=False, sort_keys=True, default=str),
        )
        for impact in details.get("leakage_impacts", []):
            link_ids_json = json.dumps(impact["link_ids"])
            impact_row = await connection.fetchrow(
                """
                INSERT INTO dataset_split_leakage_impacts (
                    impact_id, input_sha256, link_ids, sample_ids, splits
                ) VALUES ($1, $2, $3::jsonb, $4::jsonb, $5::jsonb)
                ON CONFLICT (input_sha256, link_ids) DO NOTHING
                RETURNING impact_id
                """,
                uuid4(),
                input_sha,
                link_ids_json,
                json.dumps(impact["sample_ids"]),
                json.dumps(impact["splits"]),
            )
            if impact_row is None:
                impact_row = await connection.fetchrow(
                    """
                    SELECT impact_id FROM dataset_split_leakage_impacts
                    WHERE input_sha256 = $1 AND link_ids = $2::jsonb
                    """,
                    input_sha,
                    link_ids_json,
                )
            await self._mark_existing_evaluation_leakage(
                connection,
                impact_id=impact_row["impact_id"],
                sample_ids=[UUID(sample_id) for sample_id in impact["sample_ids"]],
            )
        return {
            "mode": "response",
            "response": _blocked_result(dataset_version, input_sha, sorted(set(reasons)), details),
        }

    async def _mark_existing_evaluation_leakage(
        self,
        connection: asyncpg.Connection,
        *,
        impact_id: UUID,
        sample_ids: list[UUID],
    ) -> None:
        """Attach a late link to prior manifests that already span its fixed splits."""
        versions = await connection.fetch(
            """
            SELECT version.dataset_version, version.evaluation_reasons,
                   array_agg(DISTINCT item.sample_id) AS sample_ids
            FROM dataset_versions AS version
            JOIN dataset_items AS item USING (dataset_version)
            JOIN dataset_sample_splits AS split USING (sample_id)
            WHERE item.sample_id = ANY($1::uuid[])
              AND version.state IN ('publishing', 'published', 'invalidated')
            GROUP BY version.dataset_version, version.evaluation_reasons
            HAVING count(DISTINCT item.sample_id) > 1
               AND count(DISTINCT split.split) > 1
            ORDER BY version.dataset_version
            """,
            sample_ids,
        )
        for row in versions:
            version_id = row["dataset_version"]
            reasons = sorted(set([*_json_value(row["evaluation_reasons"]), "late_cross_boundary_link"]))
            await connection.execute(
                """
                UPDATE dataset_versions SET evaluation_eligible = FALSE,
                    evaluation_reasons = $2::jsonb
                WHERE dataset_version = $1 AND state IN ('publishing', 'published', 'invalidated')
                """,
                version_id,
                json.dumps(reasons),
            )
            await connection.execute(
                """
                INSERT INTO dataset_version_leakage_impacts (dataset_version, impact_id)
                VALUES ($1, $2) ON CONFLICT DO NOTHING
                """,
                version_id,
                impact_id,
            )
            await connection.execute(
                """
                INSERT INTO dataset_model_impacts (model_id, dataset_version, sample_id, reason)
                SELECT lineage.model_id, lineage.dataset_version, item.sample_id,
                       'evaluation_split_leakage'
                FROM dataset_model_lineage AS lineage
                JOIN dataset_items AS item USING (dataset_version)
                WHERE lineage.dataset_version = $1
                  AND item.sample_id = ANY($2::uuid[])
                ON CONFLICT DO NOTHING
                """,
                version_id,
                sample_ids,
            )


async def publish_dataset(
    selection: dict[str, Any],
    config_version: str,
    *,
    publisher: DatasetPublisher | None = None,
) -> dict[str, Any]:
    """Publish an immutable dataset version using the configured Gods stores."""
    owned = publisher is None
    service = publisher or DatasetPublisher.from_environment()
    try:
        return await service.publish_dataset(selection, config_version)
    finally:
        if owned:
            await service.close()


class _ImmutableSourceError(OSError):
    pass


def _read_body(body: Any, *, expected_size: int, expected_sha256: str) -> bytes:
    digest = sha256()
    chunks: list[bytes] = []
    size = 0
    try:
        while chunk := body.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            chunks.append(chunk)
    finally:
        body.close()
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise _ImmutableSourceError("object bytes do not match the immutable source reservation")
    return b"".join(chunks)


def _normalize_selection(selection: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise ValueError("selection must be an object")
    target = selection.get("target")
    if target not in {"detr", "clip", "both"}:
        raise ValueError("selection target must be detr, clip, or both")
    sample_ids = _uuid_list(selection.get("sample_ids", []), "sample_ids")
    crop_ids = _uuid_list(selection.get("crop_ids", []), "crop_ids")
    gallery_ids = _uuid_list(selection.get("evaluation_gallery_crop_ids", []), "evaluation_gallery_crop_ids")
    bbox_revisions = _normalize_revision_map(selection.get("bbox_revisions", {}), "bbox_revisions")
    caption_revisions = _normalize_revision_map(selection.get("caption_revisions", {}), "caption_revisions")
    if target in {"detr", "both"} and not sample_ids:
        raise ValueError("DETR selection requires sample_ids")
    if target in {"clip", "both"} and not crop_ids:
        raise ValueError("CLIP selection requires crop_ids")
    if target in {"detr", "both"} and set(bbox_revisions) != set(sample_ids):
        raise ValueError("bbox_revisions must pin one finalized bbox revision for every selected sample")
    if target in {"clip", "both"} and set(caption_revisions) != set(crop_ids):
        raise ValueError("caption_revisions must pin one reviewed caption revision for every selected crop")
    if not set(gallery_ids).issubset(crop_ids):
        raise ValueError("evaluation gallery crop IDs must be selected CLIP crop IDs")
    event_links = _normalize_selection_links(selection.get("event_links", []), "event")
    clip_links = _normalize_selection_links(selection.get("clip_links", []), "clip")
    relevance_reviews = []
    for item in selection.get("relevance_reviews", []):
        if not isinstance(item, dict):
            raise ValueError("each relevance review selection must be an object")
        review_id = str(UUID(str(item.get("review_id", ""))))
        judgments = item.get("judgments")
        if judgments is not None and not isinstance(judgments, list):
            raise ValueError("relevance judgments must be a list")
        provenance = item.get("provenance")
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError("relevance provenance must be an object")
        normalized_judgments = []
        for judgment in judgments or []:
            if not isinstance(judgment, dict):
                raise ValueError("each relevance judgment must be an object")
            normalized_judgments.append(
                {"crop_id": str(UUID(str(judgment.get("crop_id", "")))), "judgment": judgment.get("judgment")}
            )
        relevance_reviews.append(
            {
                "review_id": review_id,
                "judgments": normalized_judgments if judgments is not None else None,
                "provenance": provenance,
            }
        )
    relevance_reviews.sort(key=lambda item: item["review_id"])
    normalized = {
        "target": target,
        "sample_ids": sample_ids,
        "crop_ids": crop_ids,
        "bbox_revisions": bbox_revisions,
        "caption_revisions": caption_revisions,
        "evaluation_gallery_crop_ids": gallery_ids,
        "event_links": event_links,
        "clip_links": clip_links,
        "relevance_reviews": relevance_reviews,
    }
    canonical_json(normalized)
    return normalized


def _uuid_list(values: Any, label: str) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    normalized = [str(UUID(str(value))) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must not contain duplicate IDs")
    return sorted(normalized)


def _normalize_revision_map(values: Any, label: str) -> dict[str, str]:
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be an object keyed by source ID")
    try:
        normalized = {str(UUID(str(key))): str(UUID(str(value))) for key, value in values.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain UUID source and revision IDs") from error
    if len(normalized) != len(values):
        raise ValueError(f"{label} contains duplicate normalized source IDs")
    return dict(sorted(normalized.items()))


def _normalize_selection_links(links: Any, kind: str) -> list[dict[str, Any]]:
    if not isinstance(links, list):
        raise ValueError(f"{kind}_links must be a list")
    normalized = []
    seen = set()
    for link in links:
        if not isinstance(link, dict):
            raise ValueError(f"each {kind} link must be an object")
        link_id = str(link.get("link_id", "")).strip()
        sample_ids = _uuid_list(link.get("sample_ids", []), f"{kind}_link.sample_ids")
        if not link_id or len(link_id) > 255 or len(sample_ids) < 2:
            raise ValueError(f"{kind} links require an ID and at least two sample IDs")
        if link_id in seen:
            raise ValueError(f"duplicate {kind} link ID {link_id}")
        seen.add(link_id)
        normalized.append({"link_id": link_id, "sample_ids": sample_ids})
    normalized.sort(key=lambda item: item["link_id"])
    return normalized


def _candidate_block(reason: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"blocked": True, "reasons": [reason], "details": details}


def _annotation_snapshot(row: asyncpg.Record, result: Any) -> dict[str, Any]:
    return {
        "revision_id": str(row["annotation_revision_id"]),
        "stage": row["stage"],
        "sha256": row["result_sha256"].strip(),
        "result": result,
        "provenance": _json_value(row["provenance"]),
        "label_studio_task_id": row["label_studio_task_id"],
        "label_studio_annotation_id": row["label_studio_annotation_id"],
        "submitted_at": row["submitted_at"].astimezone(timezone.utc).isoformat(),
    }


def _has_person(result: Any) -> bool:
    return any(
        isinstance(item, dict)
        and item.get("from_name") == "bbox"
        and item.get("type") == "rectanglelabels"
        and isinstance(item.get("value"), dict)
        and "person" in item["value"].get("rectanglelabels", [])
        for item in result
    )


def _caption_text(result: Any) -> str:
    captions = []
    for item in result:
        if not isinstance(item, dict) or item.get("from_name") != "caption" or item.get("type") != "textarea":
            continue
        text_values = item.get("value", {}).get("text")
        if isinstance(text_values, list):
            captions.extend(value for value in text_values if isinstance(value, str))
    if len(captions) != 1 or not captions[0].strip():
        raise DatasetPublicationError("reviewed crop must have exactly one non-empty caption")
    return captions[0]


def _complete_human_matrix(gallery: list[dict[str, str]], judgments: Any, provenance: Any) -> bool:
    if not isinstance(judgments, list) or not isinstance(provenance, dict):
        return False
    if provenance.get("source") != "label_studio" or not provenance.get("label_studio_annotation_id"):
        return False
    if not (provenance.get("reviewer_id") or provenance.get("completed_by")):
        return False
    try:
        normalized = [(str(UUID(item["crop_id"])), item.get("judgment")) for item in judgments]
    except (KeyError, TypeError, ValueError):
        return False
    if len(normalized) != len(gallery) or {crop_id for crop_id, _ in normalized} != {item["crop_id"] for item in gallery}:
        return False
    labels = [label for _, label in normalized]
    return all(label in {"relevant", "not_relevant", "uncertain"} for label in labels) and {
        "relevant",
        "not_relevant",
    }.issubset(labels)


def _draft_snapshot(draft: asyncpg.Record, gallery: list[dict[str, str]], *, state: str | None = None) -> dict[str, Any]:
    return {
        "review_id": str(draft["review_id"]),
        "state": state or draft["state"],
        "query_id": draft["query_id"],
        "query_text": draft["query_text"],
        "query_revision": draft["query_revision"],
        "query_source_caption_revision_id": (
            str(draft["query_source_caption_revision_id"])
            if draft["query_source_caption_revision_id"] is not None
            else None
        ),
        "query_sha256": draft["query_sha256"].strip(),
        "gallery_sha256": draft["gallery_sha256"].strip(),
        "gallery": gallery,
        "invalidation_reason": draft["invalidation_reason"],
        "evaluation_eligible": False,
    }


def _split_counts(items: list[dict[str, Any]], split_plan: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    counts = {
        split: {
            "groups": 0,
            "frames": 0,
            "positive_frames": 0,
            "negative_frames": 0,
            "crop_caption_pairs": 0,
        }
        for split in ("train", "validation", "test")
    }
    groups: dict[str, set[str]] = {split: set() for split in counts}
    for item in items:
        split = item["split"]
        groups[split].add(item["component_id"])
        if item["item_kind"] == "frame":
            counts[split]["frames"] += 1
            counts[split]["positive_frames" if item["snapshot"]["has_person"] else "negative_frames"] += 1
        else:
            counts[split]["crop_caption_pairs"] += 1
    for split in counts:
        counts[split]["groups"] = len(groups[split])
    return counts


def _structural_split_error(
    items: list[dict[str, Any]],
    split_counts: dict[str, Any],
    candidate: dict[str, Any],
    *,
    has_authority: bool = False,
) -> str | None:
    if not items:
        return "no_samples_after_midnight_boundary_exclusion"
    if has_authority:
        return None
    if any(split_counts[split]["groups"] == 0 for split in ("train", "validation", "test")):
        return "invalid_independent_split_structure"
    groups_by_camera: dict[str, set[str]] = {}
    for item in items:
        sample = candidate["samples_by_id"][item["sample_id"]]
        groups_by_camera.setdefault(str(sample["camera_id"]), set()).add(item["component_id"])
    if any(len(groups) < 3 for groups in groups_by_camera.values()):
        return "fewer_than_three_independent_day_groups_per_camera"
    return None


def _readiness_reasons(target: str, counts: dict[str, Any]) -> tuple[list[str], list[str]]:
    training: list[str] = []
    evaluation: list[str] = []
    if target in {"detr", "both"}:
        train = counts["train"]
        if train["frames"] < 20:
            training.append("insufficient_train_detr_frames")
        if train["positive_frames"] < 10:
            training.append("insufficient_train_detr_positive_frames")
        if train["negative_frames"] < 5:
            training.append("insufficient_train_detr_negative_frames")
        for split in ("validation", "test"):
            values = counts[split]
            if values["frames"] < 20:
                evaluation.append(f"insufficient_{split}_detr_frames")
            if values["positive_frames"] < 10:
                evaluation.append(f"insufficient_{split}_detr_positive_frames")
            if values["negative_frames"] < 5:
                evaluation.append(f"insufficient_{split}_detr_negative_frames")
    if target in {"clip", "both"}:
        if counts["train"]["crop_caption_pairs"] < 20:
            training.append("insufficient_train_clip_pairs")
        for split in ("validation", "test"):
            if counts[split]["crop_caption_pairs"] < 20:
                evaluation.append(f"insufficient_{split}_clip_pairs")
    return training, evaluation


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _response(row: asyncpg.Record) -> dict[str, Any]:
    manifest_available = row["published_at"] is not None
    return {
        "dataset_version": row["dataset_version"],
        "manifest_hash": row["manifest_sha256"].strip() if manifest_available else None,
        "manifest_object_key": row["manifest_object_key"] if manifest_available else None,
        "split_counts": _json_value(row["split_counts"]),
        "state": row["state"],
        "training_ready": row["training_ready"],
        "training_reasons": _json_value(row["training_reasons"]),
        "evaluation_eligible": row["evaluation_eligible"],
        "evaluation_reasons": _json_value(row["evaluation_reasons"]),
        "relevance_revision_ids": _json_value(row["relevance_revision_ids"]),
    }


def _blocked_result(
    dataset_version: str,
    input_sha: str,
    reasons: list[str],
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_version": dataset_version,
        "manifest_hash": None,
        "manifest_object_key": None,
        "split_counts": details.get("split_counts", {}),
        "state": "blocked",
        "training_ready": False,
        "training_reasons": sorted(set(reasons)),
        "evaluation_eligible": False,
        "evaluation_reasons": sorted(set(reasons)),
        "reason_details": details,
        "input_sha256": input_sha,
        "relevance_revision_ids": [],
    }


def _package_version() -> str:
    try:
        return version("gods-mlops")
    except PackageNotFoundError:
        return "uninstalled"


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"required setting is missing: {name}")
    return value
