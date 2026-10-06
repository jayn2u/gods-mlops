"""Public read seam for immutable dataset versions and their current overlays."""

from __future__ import annotations

import json

import asyncpg

from .models import (
    AnnotationPreparationBatch,
    AnnotationSourceSelection,
    DatasetTrainingSource,
    ImmutableAnnotationItem,
)


class DatasetSourceUnavailableError(ValueError):
    """The requested published dataset cannot be used for the selected GPU phase."""


class DatasetSourceRegistry:
    """Read Task 6's durable readiness and impact overlay without private publisher APIs."""

    def __init__(self, *, database_url: str) -> None:
        self._database_url = database_url
        self._pool: asyncpg.Pool | None = None

    async def get_training_source(self, dataset_version: str) -> DatasetTrainingSource:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT version.dataset_version, version.target, version.manifest_sha256,
                       version.state, version.training_ready, version.training_reasons,
                       version.evaluation_eligible, version.evaluation_reasons,
                       (SELECT count(DISTINCT item.sample_id)
                        FROM dataset_items AS item
                        JOIN dataset_source_invalidations AS invalidation USING (sample_id)
                        WHERE item.dataset_version = version.dataset_version) AS invalidated_source_count,
                       (SELECT count(*) FROM dataset_version_leakage_impacts AS impact
                        WHERE impact.dataset_version = version.dataset_version) AS leakage_impact_count
                FROM dataset_versions AS version
                WHERE version.dataset_version = $1 AND version.published_at IS NOT NULL
                """,
                dataset_version,
            )
        if row is None:
            raise DatasetSourceUnavailableError("dataset version is not a published immutable source")
        return DatasetTrainingSource(
            dataset_version=row["dataset_version"],
            target=row["target"],
            manifest_sha256=row["manifest_sha256"].strip(),
            state=row["state"],
            training_ready=bool(row["training_ready"]),
            training_reasons=tuple(sorted(_json_list(row["training_reasons"]))),
            evaluation_eligible=bool(row["evaluation_eligible"]),
            evaluation_reasons=tuple(sorted(_json_list(row["evaluation_reasons"]))),
            invalidated_source_count=int(row["invalidated_source_count"]),
            leakage_impact_count=int(row["leakage_impact_count"]),
        )

    async def training_manifest_reference(self, dataset_version: str) -> dict[str, str | int]:
        """Return the published immutable manifest S3 reference for a current dataset."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                """
                SELECT manifest_object_key, manifest_sha256, manifest_size_bytes
                FROM dataset_versions
                WHERE dataset_version = $1 AND published_at IS NOT NULL
                  AND state = 'published'
                """,
                dataset_version,
            )
        if row is None or not row["manifest_object_key"]:
            raise DatasetSourceUnavailableError("published dataset manifest is unavailable")
        return {
            "object_key": str(row["manifest_object_key"]),
            "sha256": row["manifest_sha256"].strip(),
            "size_bytes": int(row["manifest_size_bytes"]),
        }

    async def training_block_reasons_in_transaction(
        self,
        connection: asyncpg.Connection,
        *,
        dataset_version: str,
        model_kind: str,
    ) -> tuple[str, ...]:
        """Recheck mutable training eligibility while the admission transaction locks its version."""
        row = await connection.fetchrow(
            """
            SELECT version.dataset_version, version.target, version.manifest_sha256,
                   version.state, version.training_ready, version.training_reasons,
                   version.evaluation_eligible, version.evaluation_reasons,
                   (SELECT count(DISTINCT item.sample_id)
                    FROM dataset_items AS item
                    JOIN dataset_source_invalidations AS invalidation USING (sample_id)
                    WHERE item.dataset_version = version.dataset_version) AS invalidated_source_count,
                   (SELECT count(*) FROM dataset_version_leakage_impacts AS impact
                    WHERE impact.dataset_version = version.dataset_version) AS leakage_impact_count
            FROM dataset_versions AS version
            WHERE version.dataset_version = $1 AND version.published_at IS NOT NULL
            FOR SHARE OF version
            """,
            dataset_version,
        )
        if row is None:
            return ("dataset_source_unavailable",)
        source = DatasetTrainingSource(
            dataset_version=row["dataset_version"],
            target=row["target"],
            manifest_sha256=row["manifest_sha256"].strip(),
            state=row["state"],
            training_ready=bool(row["training_ready"]),
            training_reasons=tuple(sorted(_json_list(row["training_reasons"]))),
            evaluation_eligible=bool(row["evaluation_eligible"]),
            evaluation_reasons=tuple(sorted(_json_list(row["evaluation_reasons"]))),
            invalidated_source_count=int(row["invalidated_source_count"]),
            leakage_impact_count=int(row["leakage_impact_count"]),
        )
        reasons = set(source.training_block_reasons())
        if source.target not in {model_kind, "both"}:
            reasons.add(f"dataset_target_not_{model_kind}")
        return tuple(sorted(reasons))

    async def evaluation_block_reasons_in_transaction(
        self,
        connection: asyncpg.Connection,
        *,
        dataset_version: str,
        model_kind: str,
    ) -> tuple[str, ...]:
        """Recheck current evaluation eligibility under the version SHARE lock."""
        row = await connection.fetchrow(
            """
            SELECT version.dataset_version, version.target, version.manifest_sha256,
                   version.state, version.training_ready, version.training_reasons,
                   version.evaluation_eligible, version.evaluation_reasons,
                   (SELECT count(DISTINCT item.sample_id)
                    FROM dataset_items AS item
                    JOIN dataset_source_invalidations AS invalidation USING (sample_id)
                    WHERE item.dataset_version = version.dataset_version) AS invalidated_source_count,
                   (SELECT count(*) FROM dataset_version_leakage_impacts AS impact
                    WHERE impact.dataset_version = version.dataset_version) AS leakage_impact_count
            FROM dataset_versions AS version
            WHERE version.dataset_version = $1 AND version.published_at IS NOT NULL
            FOR SHARE OF version
            """,
            dataset_version,
        )
        if row is None:
            return ("dataset_source_unavailable",)
        source = DatasetTrainingSource(
            dataset_version=row["dataset_version"],
            target=row["target"],
            manifest_sha256=row["manifest_sha256"].strip(),
            state=row["state"],
            training_ready=bool(row["training_ready"]),
            training_reasons=tuple(sorted(_json_list(row["training_reasons"]))),
            evaluation_eligible=bool(row["evaluation_eligible"]),
            evaluation_reasons=tuple(sorted(_json_list(row["evaluation_reasons"]))),
            invalidated_source_count=int(row["invalidated_source_count"]),
            leakage_impact_count=int(row["leakage_impact_count"]),
        )
        reasons = set(source.evaluation_block_reasons())
        if source.target not in {model_kind, "both"}:
            reasons.add(f"dataset_target_not_{model_kind}")
        return tuple(sorted(reasons))

    async def prepare_annotation_batch(
        self,
        selections: list[AnnotationSourceSelection] | tuple[AnnotationSourceSelection, ...],
    ) -> AnnotationPreparationBatch:
        """Resolve frame/crop IDs to current immutable object references before publication."""
        normalized = tuple(sorted(selections, key=lambda item: (item.item_kind, item.item_id)))
        if not normalized:
            raise DatasetSourceUnavailableError("annotation preparation batch cannot be empty")
        if len({(item.item_kind, item.item_id) for item in normalized}) != len(normalized):
            raise DatasetSourceUnavailableError("annotation preparation batch has duplicate source IDs")
        pool = await self._get_pool()
        resolved: list[ImmutableAnnotationItem] = []
        async with pool.acquire() as connection:
            async with connection.transaction():
                for selection in normalized:
                    if selection.item_kind == "frame":
                        row = await connection.fetchrow(
                            """
                            SELECT sample_id, sha256, object_key, object_size_bytes, state
                            FROM ingestion_samples WHERE sample_id = $1::uuid FOR SHARE
                            """,
                            selection.item_id,
                        )
                        if (
                            row is None
                            or row["state"] != "received"
                            or row["sha256"].strip() != selection.sha256
                        ):
                            raise DatasetSourceUnavailableError(
                                "immutable frame source is unavailable or changed"
                            )
                        resolved.append(
                            ImmutableAnnotationItem(
                                item_kind="frame",
                                item_id=str(row["sample_id"]),
                                sample_id=str(row["sample_id"]),
                                sha256=row["sha256"].strip(),
                                object_key=row["object_key"],
                                object_size_bytes=row["object_size_bytes"],
                            )
                        )
                    else:
                        row = await connection.fetchrow(
                            """
                            SELECT crop_id, sample_id, bbox_revision, sha256, object_key,
                                   object_size_bytes, state, crop_set_ready
                            FROM annotation_crops WHERE crop_id = $1::uuid FOR SHARE
                            """,
                            selection.item_id,
                        )
                        if (
                            row is None
                            or row["state"] != "ready"
                            or not row["crop_set_ready"]
                            or row["sha256"].strip() != selection.sha256
                            or str(row["bbox_revision"]) != selection.revision_id
                        ):
                            raise DatasetSourceUnavailableError(
                                "immutable crop source is unavailable or changed"
                            )
                        resolved.append(
                            ImmutableAnnotationItem(
                                item_kind="crop",
                                item_id=str(row["crop_id"]),
                                sample_id=str(row["sample_id"]),
                                sha256=row["sha256"].strip(),
                                object_key=row["object_key"],
                                object_size_bytes=row["object_size_bytes"],
                                revision_id=str(row["bbox_revision"]),
                            )
                        )
        payload = {
            "schema": "annotation-preparation-input-v1",
            "items": [item.as_dict() for item in resolved],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = __import__("hashlib").sha256(encoded.encode("utf-8")).hexdigest()
        return AnnotationPreparationBatch(
            batch_id=f"annotation-batch-{digest[:24]}",
            input_sha256=digest,
            items=tuple(resolved),
        )

    async def verify_annotation_batch(self, batch: AnnotationPreparationBatch) -> None:
        selections = [
            AnnotationSourceSelection(
                item_kind=item.item_kind,
                item_id=item.item_id,
                sha256=item.sha256,
                revision_id=item.revision_id,
            )
            for item in batch.items
        ]
        current = await self.prepare_annotation_batch(selections)
        if current != batch:
            raise DatasetSourceUnavailableError("immutable annotation preparation batch changed")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._database_url, min_size=1, max_size=4)
        return self._pool


def _json_list(value: object) -> list[str]:
    if isinstance(value, str):
        decoded = json.loads(value)
    else:
        decoded = value
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded]
