"""Run bounded object and annotation retention work for the scheduled Kubernetes job."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
import os
import sys
from typing import Any

from gods_mlops.annotations.label_studio import LabelStudioApiClient, LabelStudioMediaCleanupClient
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.annotations.workflow import LabelStudioReviewWorkflow
from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore

from .service import RetentionService

_BATCH_LIMIT = 100


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"required setting is missing: {name}")
    return value


async def run_once(*, now: datetime | None = None, limit: int = _BATCH_LIMIT) -> dict[str, Any]:
    """Retry terminal review-media cleanup, then expire at most 100 frames/crops."""
    if not 1 <= limit <= _BATCH_LIMIT:
        raise ValueError(f"retention batch limit must be between 1 and {_BATCH_LIMIT}")
    database_url = _required("GODS_MLOPS_DATABASE_URL")
    ingestion = PostgresIngestionRepository(database_url=database_url)
    annotations = PostgresAnnotationRepository(database_url=database_url)
    objects = S3SampleStore(
        endpoint_url=_required("GODS_MLOPS_S3_ENDPOINT_URL"),
        access_key=_required("GODS_MLOPS_S3_ACCESS_KEY"),
        secret_key=_required("GODS_MLOPS_S3_SECRET_KEY"),
        bucket=_required("GODS_MLOPS_S3_BUCKET"),
        region=os.environ.get("GODS_MLOPS_S3_REGION", "us-east-1"),
    )
    label_studio = LabelStudioApiClient(
        base_url=_required("GODS_MLOPS_LABEL_STUDIO_URL"),
        api_token=_required("GODS_MLOPS_LABEL_STUDIO_API_TOKEN"),
    )
    media_cleanup = LabelStudioMediaCleanupClient(
        base_url=_required("GODS_MLOPS_LABEL_MEDIA_CLEANUP_URL"),
        token=_required("GODS_MLOPS_LABEL_MEDIA_CLEANUP_TOKEN"),
    )
    try:
        await ingestion.ensure_schema()
        await annotations.ensure_schema()
        run_at = now or datetime.now(UTC)
        cleanup_result = await LabelStudioReviewWorkflow(
            repository=annotations,
            objects=objects,
            label_studio=label_studio,
            media_cleanup=media_cleanup,
        ).cleanup_pending_media(limit=limit)
        expiry_result = await RetentionService(
            repository=ingestion,
            objects=objects,
            annotations=annotations,
        ).expire_candidates(run_at, limit=limit)
        return {
            "status": "completed",
            "run_at": run_at.isoformat(),
            "label_studio_media": cleanup_result,
            "retention": expiry_result,
        }
    finally:
        await annotations.close()
        await ingestion.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one bounded Gods review and object retention pass")
    parser.add_argument("--limit", type=int, default=_BATCH_LIMIT, help="maximum records handled in this pass (1-100)")
    args = parser.parse_args(argv)
    try:
        summary = asyncio.run(run_once(limit=args.limit))
    except Exception as error:  # noqa: BLE001 - do not emit credentials or endpoint details
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
