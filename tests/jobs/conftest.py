from __future__ import annotations

import asyncio
import os
import hashlib
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest


async def seed_training_ready_dataset(database_url: str) -> str:
    """Create one non-empty published training source with eval readiness separate."""
    from gods_mlops.jobs.queue import PostgresJobQueueRepository

    repository = PostgresJobQueueRepository(database_url=database_url)
    await repository.ensure_schema()
    sample_id = uuid4()
    now = datetime(2026, 10, 5, tzinfo=UTC)
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO ingestion_samples (
                sample_id, camera_id, capture_day, captured_at_utc, reason, sha256,
                model_revision, processor_revision, object_key, object_size_bytes,
                receipt_id, state, received_at, retention_until
            ) VALUES ($1, $2, DATE '2026-10-05', $3, 'periodic', $4,
                'detector-test', 'processor-test', $5, 1, $6, 'received', $3,
                $3::timestamptz + interval '7 days')
            """,
            sample_id,
            uuid4(),
            now,
            hashlib.sha256(b"frame").hexdigest(),
            f"samples/{sample_id}.jpg",
            uuid4(),
        )
        dataset_version = f"dataset-test-{uuid4().hex}"
        await connection.execute(
            """
            INSERT INTO dataset_versions (
                dataset_version, input_sha256, target, config_version, config_sha256,
                code_sha256, state, manifest_object_key, manifest_sha256,
                manifest_size_bytes, manifest_json, split_counts, training_ready,
                training_reasons, evaluation_eligible, evaluation_reasons, published_at
            ) VALUES ($1, $2, 'detr', 'dataset-config-v1', $3, $4, 'published',
                $5, $6, 1, '{}'::jsonb, '{}'::jsonb, TRUE, '[]'::jsonb,
                FALSE, '["evaluation_minimum_not_met"]'::jsonb, $7)
            """,
            dataset_version,
            hashlib.sha256(dataset_version.encode()).hexdigest(),
            hashlib.sha256(b"dataset-config").hexdigest(),
            hashlib.sha256(b"dataset-code").hexdigest(),
            f"datasets/{dataset_version}/manifest.json",
            hashlib.sha256(b"manifest").hexdigest(),
            now,
        )
        await connection.execute(
            """
            INSERT INTO dataset_items (
                dataset_version, item_kind, item_id, sample_id, target, split,
                group_id, component_id, source_object_key, object_key, source_sha256,
                object_size_bytes, snapshot
            ) VALUES ($1, 'frame', $2, $2, 'detr', 'train', 'group-test', 'component-test',
                'source.jpg', 'copy.jpg', $3, 1, '{}'::jsonb)
            """,
            dataset_version,
            sample_id,
            hashlib.sha256(b"frame").hexdigest(),
        )
    finally:
        await connection.close()
        await repository.close()
    return dataset_version


@pytest.fixture
def task7_database_url() -> str:
    """Create a fresh, loopback-only DB so every global queue state starts empty."""
    admin_url = os.environ.get("GODS_MLOPS_TASK7_ADMIN_DATABASE_URL")
    if not admin_url:
        pytest.skip("loopback-only disposable Task 7 PostgreSQL admin URL is not configured")
    parsed = urlsplit(admin_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        pytest.fail("Task 7 integration tests refuse a non-loopback PostgreSQL endpoint")
    database_name = f"gods_task7_{uuid4().hex[:16]}"

    async def create_database() -> None:
        connection = await asyncpg.connect(admin_url)
        try:
            await connection.execute(f'CREATE DATABASE "{database_name}"')
        finally:
            await connection.close()

    asyncio.run(create_database())
    target_url = urlunsplit(
        (parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, parsed.fragment)
    )
    try:
        yield target_url
    finally:
        async def drop_database() -> None:
            connection = await asyncpg.connect(admin_url)
            try:
                await connection.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = $1",
                    database_name,
                )
                await connection.execute(f'DROP DATABASE IF EXISTS "{database_name}"')
            finally:
                await connection.close()

        asyncio.run(drop_database())
