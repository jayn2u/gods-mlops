from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import asyncpg
import pytest

from gods_mlops.jobs.models import AnnotationSourceSelection, ExecutionProfile
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry, DatasetSourceUnavailableError


def test_gpu_preparation_uses_verified_frame_and_crop_refs_without_a_published_dataset(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        sample_id = uuid4()
        bbox_revision = uuid4()
        crop_id = uuid4()
        bbox_assignment_revision = uuid4()
        frame_sha = hashlib.sha256(b"immutable frame bytes").hexdigest()
        crop_sha = hashlib.sha256(b"immutable crop bytes").hexdigest()
        bbox_result = [{"boxes": [[1, 2, 3, 4]], "labels": ["person"]}]
        bbox_sha = hashlib.sha256(json.dumps(bbox_result, separators=(",", ":")).encode()).hexdigest()
        connection = await asyncpg.connect(task7_database_url)
        try:
            await connection.execute(
                """
                INSERT INTO ingestion_samples (
                    sample_id, camera_id, capture_day, captured_at_utc, reason, sha256,
                    model_revision, processor_revision, object_key, object_size_bytes,
                    receipt_id, state, received_at, retention_until
                ) VALUES ($1, $2, DATE '2026-10-05', $3, 'operator', $4,
                    'detector-source-revision', 'processor-source-revision', $5, 20,
                    $6, 'received', $3, $3::timestamptz + interval '7 days')
                """,
                sample_id,
                uuid4(),
                datetime(2026, 10, 5, tzinfo=UTC),
                frame_sha,
                f"samples/{sample_id}.jpg",
                uuid4(),
            )
            await connection.execute(
                """
                INSERT INTO review_assignments (
                    assignment_id, revision, sample_id, stage, label_studio_task_id,
                    media_object_key, required_bytes, state, finished_at
                ) VALUES ($1, $2, $3, 'bbox', 1, $4, 20, 'finalized', now())
                """,
                uuid4(),
                bbox_assignment_revision,
                sample_id,
                f"samples/{sample_id}.jpg",
            )
            await connection.execute(
                """
                INSERT INTO annotation_revisions (
                    annotation_revision_id, sample_id, assignment_revision, stage,
                    label_studio_task_id, label_studio_annotation_id, result,
                    result_sha256, provenance, submitted_at
                ) VALUES ($1, $2, $3, 'bbox', 1, 1, $4::jsonb, $5, '{}'::jsonb, now())
                """,
                bbox_revision,
                sample_id,
                bbox_assignment_revision,
                json.dumps(bbox_result),
                bbox_sha,
            )
            await connection.execute(
                """
                INSERT INTO annotation_crops (
                    crop_id, sample_id, bbox_revision, region_index, region_id,
                    object_key, sha256, object_size_bytes, state, caption_state,
                    provenance, crop_set_ready
                ) VALUES ($1, $2, $3, 0, 'person-0', $4, $5, 12, 'ready',
                    'needs_review', $6::jsonb, TRUE)
                """,
                crop_id,
                sample_id,
                bbox_revision,
                f"crops/{sample_id}/{bbox_revision}/{crop_id}.jpg",
                crop_sha,
                json.dumps({"frame_sha256": frame_sha, "bbox_sha256": bbox_sha}),
            )
        finally:
            await connection.close()

        sources = DatasetSourceRegistry(database_url=task7_database_url)
        batch = await sources.prepare_annotation_batch(
            [
                AnnotationSourceSelection("frame", str(sample_id), frame_sha),
                AnnotationSourceSelection("crop", str(crop_id), crop_sha, str(bbox_revision)),
            ]
        )
        queue = JobQueue(repository=repository, sources=sources)
        await queue.register_profile(
            ExecutionProfile(
                model_kind="qwen",
                config_version="caption-preparation-v1",
                phase="preparation",
                memory_requirement_mib=16_384,
                artifact_reservation_bytes=32 * 1024**2,
                config={"input_tokens": 4096, "output_tokens": 128},
                candidate=True,
            )
        )
        first = await queue.submit_preparation(
            batch=batch,
            model_kind="qwen",
            config_version="caption-preparation-v1",
        )
        retry = await queue.submit_preparation(
            batch=batch,
            model_kind="qwen",
            config_version="caption-preparation-v1",
        )
        rerun = await queue.submit_preparation(
            batch=batch,
            model_kind="qwen",
            config_version="caption-preparation-v1",
            rerun=True,
        )
        assert first == retry
        assert rerun != first
        job = await queue.get(first)
        assert job["phase"] == "preparation"
        assert job["input_kind"] == "annotation_batch"
        assert job["dataset_version"] is None
        assert job["input_id"] == batch.batch_id
        assert job["input_sha256"] == batch.input_sha256
        assert {item["item_kind"] for item in job["source_refs"]["items"]} == {"frame", "crop"}
        assert await connection_closed_dataset_count(task7_database_url) == 0
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


async def connection_closed_dataset_count(database_url: str) -> int:
    connection = await asyncpg.connect(database_url)
    try:
        return await connection.fetchval("SELECT count(*) FROM dataset_versions")
    finally:
        await connection.close()


def test_changed_crop_revision_cannot_reuse_an_older_preparation_batch(task7_database_url: str) -> None:
    async def exercise() -> None:
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        sample_id = uuid4()
        crop_id = uuid4()
        bbox_revision = uuid4()
        frame_sha = hashlib.sha256(b"frame").hexdigest()
        crop_sha = hashlib.sha256(b"crop").hexdigest()
        connection = await asyncpg.connect(task7_database_url)
        try:
            await connection.execute(
                """
                INSERT INTO ingestion_samples (
                    sample_id, camera_id, capture_day, captured_at_utc, reason, sha256,
                    model_revision, processor_revision, object_key, object_size_bytes,
                    receipt_id, state
                ) VALUES ($1, $2, DATE '2026-10-05', now(), 'operator', $3,
                    'detector', 'processor', $4, 8, $5, 'received')
                """,
                sample_id,
                uuid4(),
                frame_sha,
                f"samples/{sample_id}.jpg",
                uuid4(),
            )
            assignment_revision = uuid4()
            await connection.execute(
                """
                INSERT INTO review_assignments (
                    assignment_id, revision, sample_id, stage, label_studio_task_id,
                    media_object_key, required_bytes, state
                ) VALUES ($1, $2, $3, 'bbox', 1, 'frame.jpg', 8, 'finalized')
                """,
                uuid4(), assignment_revision, sample_id,
            )
            await connection.execute(
                """
                INSERT INTO annotation_revisions (
                    annotation_revision_id, sample_id, assignment_revision, stage,
                    label_studio_task_id, label_studio_annotation_id, result,
                    result_sha256, provenance, submitted_at
                ) VALUES ($1, $2, $3, 'bbox', 1, 1, '{}'::jsonb, $4, '{}'::jsonb, now())
                """,
                bbox_revision, sample_id, assignment_revision, hashlib.sha256(b"{}").hexdigest(),
            )
            await connection.execute(
                """
                INSERT INTO annotation_crops (
                    crop_id, sample_id, bbox_revision, region_index, region_id,
                    object_key, sha256, object_size_bytes, state, provenance, crop_set_ready
                ) VALUES ($1, $2, $3, 0, 'p0', 'crop.jpg', $4, 4, 'ready', '{}'::jsonb, TRUE)
                """,
                crop_id, sample_id, bbox_revision, crop_sha,
            )
        finally:
            await connection.close()
        sources = DatasetSourceRegistry(database_url=task7_database_url)
        batch = await sources.prepare_annotation_batch(
            [AnnotationSourceSelection("crop", str(crop_id), crop_sha, str(bbox_revision))]
        )
        connection = await asyncpg.connect(task7_database_url)
        try:
            await connection.execute(
                "UPDATE annotation_crops SET state = 'deleted' WHERE crop_id = $1", crop_id
            )
        finally:
            await connection.close()
        with pytest.raises(DatasetSourceUnavailableError, match="immutable crop source"):
            await sources.verify_annotation_batch(batch)
        await repository.close()
        await sources.close()

    asyncio.run(exercise())
