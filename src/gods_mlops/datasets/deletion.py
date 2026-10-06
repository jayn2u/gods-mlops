"""Persist explicit sample invalidation impacts without erasing dataset history."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import asyncpg


async def invalidate_sample(
    sample_id: str,
    *,
    publisher: Any | None = None,
) -> dict[str, Any]:
    """Block affected dataset eligibility and record registered model impacts."""
    owned = publisher is None
    if publisher is None:
        from .publish import DatasetPublisher

        publisher = DatasetPublisher.from_environment()
    try:
        return await publisher.invalidate_sample(sample_id)
    finally:
        if owned:
            await publisher.close()


async def preview_invalidate_sample(
    sample_id: str,
    *,
    publisher: Any | None = None,
) -> dict[str, Any]:
    """Read the existing eligibility and in-use impact without changing source state."""
    owned = publisher is None
    if publisher is None:
        from .publish import DatasetPublisher

        publisher = DatasetPublisher.from_environment()
    try:
        return await publisher.preview_sample_invalidation(sample_id)
    finally:
        if owned:
            await publisher.close()


async def invalidate_sample_in_repository(publisher: Any, sample_id: str) -> dict[str, Any]:
    sample_uuid = UUID(sample_id)
    pool: asyncpg.Pool = await publisher._get_pool()
    async with pool.acquire() as connection:
        async with connection.transaction():
            sample = await connection.fetchrow(
                "SELECT sample_id FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE",
                sample_uuid,
            )
            if sample is None:
                return {
                    "sample_id": str(sample_uuid),
                    "datasets": [],
                    "models": [],
                    "block_training": False,
                    "block_evaluation": False,
                }
            await connection.execute(
                """
                INSERT INTO dataset_source_invalidations (sample_id, reason)
                VALUES ($1, 'sample_explicitly_invalidated')
                ON CONFLICT (sample_id) DO NOTHING
                """,
                sample_uuid,
            )
            versions = await connection.fetch(
                """
                SELECT version.dataset_version, version.state,
                       version.training_reasons, version.evaluation_reasons
                FROM dataset_versions AS version
                WHERE version.dataset_version IN (
                    SELECT item.dataset_version FROM dataset_items AS item WHERE item.sample_id = $1
                ) AND version.state IN ('publishing', 'published', 'invalidated')
                ORDER BY version.dataset_version FOR UPDATE OF version
                """,
                sample_uuid,
            )
            version_ids = [row["dataset_version"] for row in versions]
            if not version_ids:
                return {
                    "sample_id": str(sample_uuid),
                    "datasets": [],
                    "models": [],
                    "block_training": True,
                    "block_evaluation": True,
                }
            model_rows = await connection.fetch(
                """
                SELECT DISTINCT model_id FROM dataset_model_lineage
                WHERE dataset_version = ANY($1::text[]) ORDER BY model_id
                """,
                version_ids,
            )
            models = [row["model_id"] for row in model_rows]
            await connection.execute(
                """
                INSERT INTO dataset_invalidations (dataset_version, sample_id, reason)
                SELECT versions.dataset_version, $2, 'sample_explicitly_invalidated'
                FROM unnest($1::text[]) AS versions(dataset_version)
                ON CONFLICT (dataset_version, sample_id) DO NOTHING
                """,
                version_ids,
                sample_uuid,
            )
            await connection.execute(
                """
                INSERT INTO dataset_model_impacts (model_id, dataset_version, sample_id, reason)
                SELECT lineage.model_id, lineage.dataset_version, $2, 'source_sample_explicitly_invalidated'
                FROM dataset_model_lineage AS lineage
                WHERE lineage.dataset_version = ANY($1::text[])
                ON CONFLICT DO NOTHING
                """,
                version_ids,
                sample_uuid,
            )
            for row in versions:
                training_reasons = _json_value(row["training_reasons"])
                evaluation_reasons = _json_value(row["evaluation_reasons"])
                training_reasons = sorted(set([*training_reasons, "sample_explicitly_invalidated"]))
                evaluation_reasons = sorted(set([*evaluation_reasons, "sample_explicitly_invalidated"]))
                await connection.execute(
                    """
                    UPDATE dataset_versions SET state = 'invalidated',
                        training_ready = FALSE, evaluation_eligible = FALSE,
                        training_reasons = $2::jsonb, evaluation_reasons = $3::jsonb,
                        invalidated_at = COALESCE(invalidated_at, now())
                    WHERE dataset_version = $1 AND state IN ('publishing', 'published', 'invalidated')
                    """,
                    row["dataset_version"],
                    json.dumps(training_reasons),
                    json.dumps(evaluation_reasons),
                )
            return {
                "sample_id": str(sample_uuid),
                "datasets": version_ids,
                "models": models,
                "block_training": True,
                "block_evaluation": True,
            }


def _json_value(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value
