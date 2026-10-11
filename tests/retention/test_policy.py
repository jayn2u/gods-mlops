from __future__ import annotations

import asyncio
import hashlib
import io
import os
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import asyncpg
import boto3
import pytest
from PIL import Image

from gods_mlops.annotations.models import ReviewAssignmentConflictError, ReviewQuotaExceededError
from gods_mlops.annotations.crops import CropService
from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.ingestion.schemas import CandidateMetadata, SampleExpiredError
from gods_mlops.ingestion.service import IngestionService
from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore
from gods_mlops.retention.service import RetentionService


class FakeSubmittedTask:
    def __init__(self, task: dict) -> None:
        self.task = task

    async def get_task(self, task_id: int) -> dict:
        assert task_id == self.task["id"]
        return self.task


async def _seed_candidate(database_url: str, *, size_bytes: int) -> tuple[object, str]:
    ingestion = PostgresIngestionRepository(database_url=database_url)
    await ingestion.ensure_schema()
    sample_id = uuid4()
    camera_id = uuid4()
    digest = uuid4().hex + uuid4().hex
    object_key = f"samples/{camera_id}/{sample_id}/{digest}.jpg"
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO ingestion_samples (
                sample_id, camera_id, capture_day, captured_at_utc, reason, sha256,
                model_revision, processor_revision, object_key, object_size_bytes,
                receipt_id, state, received_at, retention_until
            ) VALUES ($1, $2, DATE '2026-10-05', TIMESTAMPTZ '2026-10-05 00:00:00+00',
                'periodic', $3, 'detector-test', 'processor-test', $4, $5,
                $6, 'received', now(), now() + interval '7 days')
            """,
            sample_id,
            camera_id,
            digest,
            object_key,
            size_bytes,
            uuid4(),
        )
        await connection.execute(
            "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
            size_bytes,
        )
    finally:
        await connection.close()
        await ingestion.close()
    return sample_id, object_key


def test_review_quota_is_a_subset_and_records_capacity_stop() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")

    async def exercise() -> None:
        initial = PostgresIngestionRepository(database_url=database_url)
        await initial.ensure_schema()
        connection = await asyncpg.connect(database_url)
        try:
            initial_review_bytes = await connection.fetchval(
                "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE"
            )
        finally:
            await connection.close()
            await initial.close()
        repository = PostgresAnnotationRepository(
            database_url=database_url,
            review_exception_bytes=initial_review_bytes + 128,
        )
        await repository.ensure_schema()
        first_sample, first_key = await _seed_candidate(database_url, size_bytes=128)
        second_sample, second_key = await _seed_candidate(database_url, size_bytes=128)
        await repository.create_review_assignment(
            sample_id=first_sample,
            stage="bbox",
            label_studio_task_id=uuid4().int % 2_000_000_000 + 1,
            bbox_revision="detector-draft-0",
            media_object_key=first_key,
            required_bytes=128,
        )
        with pytest.raises(ReviewQuotaExceededError):
            await repository.create_review_assignment(
                sample_id=second_sample,
                stage="bbox",
                label_studio_task_id=uuid4().int % 2_000_000_000 + 1,
                bbox_revision="detector-draft-0",
                media_object_key=second_key,
                required_bytes=128,
            )

        connection = await asyncpg.connect(database_url)
        try:
            review_bytes = await connection.fetchval(
                "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE"
            )
            object_bytes = await connection.fetchval(
                "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE"
            )
            stop_count = await connection.fetchval(
                """
                SELECT count(*) FROM retention_policy_events
                WHERE sample_id = $1 AND reason = 'active_review_quota_exceeded'
                """,
                second_sample,
            )
        finally:
            await connection.close()
            await repository.close()
        assert review_bytes == initial_review_bytes + 128
        assert object_bytes >= 256
        assert stop_count == 1

    asyncio.run(exercise())


def test_expired_candidate_with_active_review_is_not_claimed() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")
    now = datetime.now(timezone.utc)

    async def exercise() -> None:
        sample_id, object_key = await _seed_candidate(database_url, size_bytes=64)
        ingestion = PostgresIngestionRepository(database_url=database_url)
        annotations = PostgresAnnotationRepository(database_url=database_url)
        await ingestion.ensure_schema()
        await annotations.ensure_schema()
        await annotations.create_review_assignment(
            sample_id=sample_id,
            stage="bbox",
            label_studio_task_id=uuid4().int % 2_000_000_000 + 1,
            bbox_revision="detector-draft-0",
            media_object_key=object_key,
            required_bytes=64,
        )
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(
                "UPDATE ingestion_samples SET retention_until = $2 WHERE sample_id = $1",
                sample_id,
                now - timedelta(minutes=1),
            )
        finally:
            await connection.close()

        claimed = await ingestion.claim_expired(now=now)
        assert all(sample.sample_id != sample_id for sample in claimed)
        assert await ingestion.sample_state(sample_id) == "received"
        await annotations.close()
        await ingestion.close()

    asyncio.run(exercise())


def test_assignment_and_expiry_compete_on_the_sample_row_lock() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")
    now = datetime.now(timezone.utc)

    async def exercise() -> None:
        sample_id, object_key = await _seed_candidate(database_url, size_bytes=64)
        ingestion = PostgresIngestionRepository(database_url=database_url)
        annotations = PostgresAnnotationRepository(database_url=database_url)
        await ingestion.ensure_schema()
        await annotations.ensure_schema()
        setup = await asyncpg.connect(database_url)
        try:
            await setup.execute(
                "UPDATE ingestion_samples SET retention_until = $2 WHERE sample_id = $1",
                sample_id,
                now - timedelta(minutes=1),
            )
        finally:
            await setup.close()

        assignment_lock = await asyncpg.connect(database_url)
        await assignment_lock.execute("BEGIN")
        await assignment_lock.fetchrow(
            "SELECT sample_id FROM ingestion_samples WHERE sample_id = $1 FOR UPDATE",
            sample_id,
        )
        assignment_task = asyncio.create_task(
            annotations.create_review_assignment(
                sample_id=sample_id,
                stage="bbox",
                label_studio_task_id=uuid4().int % 2_000_000_000 + 1,
                bbox_revision="detector-draft-0",
                media_object_key=object_key,
                required_bytes=64,
            )
        )
        await asyncio.sleep(0.05)
        try:
            claimed_while_assignment_holds_row = await ingestion.claim_expired(now=now)
            assert all(item.sample_id != sample_id for item in claimed_while_assignment_holds_row)
        finally:
            await assignment_lock.execute("COMMIT")
            await assignment_lock.close()
        assignment = await assignment_task

        connection = await asyncpg.connect(database_url)
        try:
            selected_guard = await connection.fetchval(
                "SELECT selected FROM ingestion_samples WHERE sample_id = $1", sample_id
            )
        finally:
            await connection.close()
        claimed_after_assignment = await ingestion.claim_expired(now=now)
        assert selected_guard is True
        assert all(item.sample_id != sample_id for item in claimed_after_assignment)
        assert await ingestion.sample_state(sample_id) == "received"
        assert await annotations.assignment_state(assignment.revision) == "active"
        await annotations.close()
        await ingestion.close()

    asyncio.run(exercise())


def test_expiry_claim_wins_race_and_later_assignment_cannot_reuse_frame() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")
    now = datetime.now(timezone.utc)

    async def exercise() -> None:
        sample_id, object_key = await _seed_candidate(database_url, size_bytes=64)
        ingestion = PostgresIngestionRepository(database_url=database_url)
        annotations = PostgresAnnotationRepository(database_url=database_url)
        await ingestion.ensure_schema()
        await annotations.ensure_schema()
        setup = await asyncpg.connect(database_url)
        function_name = f"gods_t5_delay_{sample_id.hex}"
        trigger_name = f"gods_t5_delay_{sample_id.hex}"
        try:
            await setup.execute(
                "UPDATE ingestion_samples SET retention_until = $2 WHERE sample_id = $1",
                sample_id,
                now - timedelta(minutes=1),
            )
            await setup.execute(
                f"""
                CREATE FUNCTION public.{function_name}() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN
                    IF NEW.sample_id = '{sample_id}'::uuid AND NEW.state = 'purge_pending' THEN
                        PERFORM pg_sleep(5);
                    END IF;
                    RETURN NEW;
                END;
                $$
                """
            )
            await setup.execute(
                f"""
                CREATE TRIGGER {trigger_name} BEFORE UPDATE OF state ON ingestion_samples
                FOR EACH ROW EXECUTE FUNCTION public.{function_name}()
                """
            )
        finally:
            await setup.close()

        expiry_task = asyncio.create_task(ingestion.claim_expired(now=now))
        observer = None
        revision_task = None
        try:
            observer = await asyncpg.connect(database_url)
            deadline = asyncio.get_running_loop().time() + 3
            while asyncio.get_running_loop().time() < deadline:
                sleeping = await observer.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM pg_stat_activity
                        WHERE datname = current_database() AND pid <> pg_backend_pid()
                          AND wait_event = 'PgSleep'
                    )
                    """
                )
                if sleeping:
                    break
                await asyncio.sleep(0.01)
            assert sleeping is True, "expiry transition did not enter its guarded update"
            revision_task = asyncio.create_task(
                annotations.create_review_assignment(
                sample_id=sample_id,
                stage="bbox",
                label_studio_task_id=uuid4().int % 2_000_000_000 + 1,
                bbox_revision="detector-draft-0",
                media_object_key=object_key,
                required_bytes=64,
            )
            )
            claimed = await expiry_task
            outcome = await asyncio.gather(revision_task, return_exceptions=True)
            assert any(item.sample_id == sample_id for item in claimed)
            assert isinstance(outcome[0], ReviewAssignmentConflictError)
            assert await ingestion.sample_state(sample_id) == "purge_pending"
        finally:
            if observer is not None:
                await observer.close()
            if not expiry_task.done():
                await asyncio.wait_for(expiry_task, timeout=8)
            if revision_task is not None and not revision_task.done():
                await asyncio.wait_for(revision_task, timeout=8)
            cleanup = await asyncpg.connect(database_url)
            try:
                await cleanup.execute(
                    f"DROP TRIGGER IF EXISTS {trigger_name} ON ingestion_samples"
                )
                await cleanup.execute(f"DROP FUNCTION IF EXISTS public.{function_name}()")
            finally:
                await cleanup.close()
                await annotations.close()
                await ingestion.close()

    asyncio.run(exercise())


def test_seven_day_expiry_deletes_s3_object_and_keeps_replay_tombstone() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    s3_endpoint = os.environ.get("GODS_MLOPS_TEST_S3_ENDPOINT")
    access_key = os.environ.get("GODS_MLOPS_TEST_S3_ACCESS_KEY")
    secret_key = os.environ.get("GODS_MLOPS_TEST_S3_SECRET_KEY")
    bucket = os.environ.get("GODS_MLOPS_TEST_S3_BUCKET")
    if not all((database_url, s3_endpoint, access_key, secret_key, bucket)):
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    now = datetime.now(timezone.utc)
    image = b"durable frame bytes eligible for seven day expiry"
    digest = hashlib.sha256(image).hexdigest()
    sample_id, camera_id = uuid4(), uuid4()
    object_key = f"samples/{camera_id}/{sample_id}/{digest}.jpg"
    metadata = CandidateMetadata(
        sample_id=sample_id,
        camera_id=camera_id,
        captured_at_utc=now - timedelta(days=8),
        reason="periodic",
        sha256=digest,
        model_revision="detector-test",
        processor_revision="processor-test",
    )
    objects = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    objects.ensure_object(object_key=object_key, image=image, expected_sha256=digest)

    async def exercise() -> None:
        ingestion = PostgresIngestionRepository(database_url=database_url)
        await ingestion.ensure_schema()
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(
                """
                INSERT INTO ingestion_samples (
                    sample_id, camera_id, capture_day, captured_at_utc, reason, sha256,
                    model_revision, processor_revision, object_key, object_size_bytes,
                    receipt_id, state, received_at, retention_until
                ) VALUES ($1, $2, DATE '2026-10-05', $3, 'periodic', $4,
                    'detector-test', 'processor-test', $5, $6, $7, 'received', $8, $9)
                """,
                sample_id,
                camera_id,
                metadata.captured_at_utc,
                digest,
                object_key,
                len(image),
                uuid4(),
                metadata.captured_at_utc,
                now - timedelta(seconds=1),
            )
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                len(image),
            )
        finally:
            await connection.close()
        charged_usage = await ingestion.storage_bytes()
        annotation_repository = PostgresAnnotationRepository(database_url=database_url)
        await annotation_repository.ensure_schema()
        service = RetentionService(
            repository=ingestion,
            objects=objects,
            annotations=annotation_repository,
        )
        try:
            result = await service.expire_candidates(now)
            assert result["deleted"] >= 1
            assert result["deferred"] == 0
            assert await ingestion.sample_state(sample_id) == "expired"
            assert await ingestion.count_sample(sample_id) == 1
            assert await ingestion.storage_bytes() <= charged_usage - len(image)
            with pytest.raises(FileNotFoundError):
                objects.read_object(object_key=object_key, expected_sha256=digest)
            with pytest.raises(SampleExpiredError):
                await IngestionService(repository=ingestion, objects=objects).receive(metadata, image)
        finally:
            await annotation_repository.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_clip_adoption_keeps_crop_caption_and_records_lost_parent() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    s3_endpoint = os.environ.get("GODS_MLOPS_TEST_S3_ENDPOINT")
    access_key = os.environ.get("GODS_MLOPS_TEST_S3_ACCESS_KEY")
    secret_key = os.environ.get("GODS_MLOPS_TEST_S3_SECRET_KEY")
    bucket = os.environ.get("GODS_MLOPS_TEST_S3_BUCKET")
    if not all((database_url, s3_endpoint, access_key, secret_key, bucket)):
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    now = datetime.now(timezone.utc)
    image = Image.new("RGB", (80, 40), "white")
    frame_buffer = io.BytesIO()
    image.save(frame_buffer, format="JPEG", quality=92)
    frame_bytes = frame_buffer.getvalue()
    frame_sha = hashlib.sha256(frame_bytes).hexdigest()
    sample_id, camera_id = uuid4(), uuid4()
    frame_key = f"samples/{camera_id}/{sample_id}/{frame_sha}.jpg"
    objects = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    objects.ensure_object(object_key=frame_key, image=frame_bytes, expected_sha256=frame_sha)
    bbox_task_id = uuid4().int % 2_000_000_000 + 1
    bbox_task = {
        "id": bbox_task_id,
        "annotations": [
            {
                "id": uuid4().int % 2_000_000_000 + 1,
                "was_cancelled": False,
                "completed_by": {"id": 7},
                "result": [
                    {
                        "id": "person",
                        "from_name": "bbox",
                        "type": "rectanglelabels",
                        "original_width": 80,
                        "original_height": 40,
                        "value": {"x": 10, "y": 10, "width": 80, "height": 80, "rectanglelabels": ["person"]},
                    }
                ],
            }
        ],
    }

    async def exercise() -> None:
        ingestion = PostgresIngestionRepository(database_url=database_url)
        await ingestion.ensure_schema()
        before_usage = await ingestion.storage_bytes()
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(
                """
                INSERT INTO ingestion_samples (
                    sample_id, camera_id, capture_day, captured_at_utc, reason, sha256,
                    model_revision, processor_revision, object_key, object_size_bytes,
                    receipt_id, state, received_at, retention_until
                ) VALUES ($1, $2, DATE '2026-10-05', $3, 'periodic', $4,
                    'detector-test', 'processor-test', $5, $6, $7, 'received', $3, $8)
                """,
                sample_id,
                camera_id,
                now - timedelta(days=8),
                frame_sha,
                frame_key,
                len(frame_bytes),
                uuid4(),
                now - timedelta(seconds=1),
            )
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                len(frame_bytes),
            )
        finally:
            await connection.close()

        annotations = PostgresAnnotationRepository(database_url=database_url)
        await annotations.ensure_schema()
        assignment = await annotations.create_review_assignment(
            sample_id=sample_id,
            stage="bbox",
            label_studio_task_id=bbox_task_id,
            bbox_revision="detector-draft-0",
            media_object_key=frame_key,
            required_bytes=len(frame_bytes),
        )
        try:
            bbox_revision = await AnnotationService(
                repository=annotations,
                label_studio=FakeSubmittedTask(bbox_task),
            ).finalize_annotation(str(sample_id), assignment.revision)
            crop_batch = await CropService(repository=annotations, objects=objects).create_crop(
                str(sample_id), bbox_revision["annotation_revision_id"]
            )
            crop = crop_batch["crops"][0]
            caption_task_id = uuid4().int % 2_000_000_000 + 1
            caption_task = {
                "id": caption_task_id,
                "annotations": [
                    {
                        "id": uuid4().int % 2_000_000_000 + 1,
                        "was_cancelled": False,
                        "completed_by": {"id": 7},
                        "result": [
                            {
                                "from_name": "caption",
                                "type": "textarea",
                                "value": {"text": ["white shirt"]},
                            }
                        ],
                    }
                ],
            }
            caption_assignment = await annotations.create_review_assignment(
                sample_id=sample_id,
                stage="caption",
                label_studio_task_id=caption_task_id,
                bbox_revision=bbox_revision["annotation_revision_id"],
                media_object_key=crop["object_key"],
                required_bytes=crop["object_size_bytes"],
            )
            caption_revision = await AnnotationService(
                repository=annotations,
                label_studio=FakeSubmittedTask(caption_task),
            ).finalize_annotation(str(sample_id), caption_assignment.revision)
            adoption = await annotations.adopt_for_dataset(
                dataset_version="clip-test-v1",
                target="clip",
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision["annotation_revision_id"]),
                crop_id=UUID(crop["crop_id"]),
                caption_revision=UUID(caption_revision["annotation_revision_id"]),
            )

            result = await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)
            assert result["deleted"] >= 1
            assert await ingestion.sample_state(sample_id) == "expired"
            with pytest.raises(FileNotFoundError):
                objects.read_object(object_key=frame_key, expected_sha256=frame_sha)
            retained = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision["annotation_revision_id"]),
            )
            assert retained[0]["crop_id"] == crop["crop_id"]
            assert retained[0]["caption_revision_id"] == caption_revision["annotation_revision_id"]
            assert retained[0]["parent_available"] is False
            assert retained[0]["parent_regenerable"] is False
            assert objects.read_object(
                object_key=crop["object_key"],
                expected_sha256=crop["sha256"],
            )
            crop_again = await CropService(repository=annotations, objects=objects).create_crop(
                str(sample_id), bbox_revision["annotation_revision_id"]
            )
            assert crop_again["crop_id"] == crop["crop_id"]
            assert adoption["target"] == "clip"
            assert await ingestion.storage_bytes() >= crop["object_size_bytes"]
        finally:
            await annotations.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_rejected_caption_makes_unadopted_crop_eligible_for_removal() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    s3_endpoint = os.environ.get("GODS_MLOPS_TEST_S3_ENDPOINT")
    access_key = os.environ.get("GODS_MLOPS_TEST_S3_ACCESS_KEY")
    secret_key = os.environ.get("GODS_MLOPS_TEST_S3_SECRET_KEY")
    bucket = os.environ.get("GODS_MLOPS_TEST_S3_BUCKET")
    if not all((database_url, s3_endpoint, access_key, secret_key, bucket)):
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    now = datetime.now(timezone.utc)
    sample_id, camera_id = uuid4(), uuid4()
    frame_bytes, crop_bytes = b"frame bytes retained through caption rejection", b"unadopted crop bytes"
    frame_sha, crop_sha = hashlib.sha256(frame_bytes).hexdigest(), hashlib.sha256(crop_bytes).hexdigest()
    frame_key = f"samples/{camera_id}/{sample_id}/{frame_sha}.jpg"
    crop_id = uuid4()
    crop_key = f"crops/{sample_id}/{crop_id}.jpg"
    objects = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    objects.ensure_object(object_key=frame_key, image=frame_bytes, expected_sha256=frame_sha)
    objects.ensure_object(object_key=crop_key, image=crop_bytes, expected_sha256=crop_sha)
    bbox_task_id = uuid4().int % 2_000_000_000 + 1
    caption_task_id = uuid4().int % 2_000_000_000 + 1

    async def exercise() -> None:
        ingestion = PostgresIngestionRepository(database_url=database_url)
        await ingestion.ensure_schema()
        prior_usage = await ingestion.storage_bytes()
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(
                """
                INSERT INTO ingestion_samples (
                    sample_id, camera_id, capture_day, captured_at_utc, reason, sha256,
                    model_revision, processor_revision, object_key, object_size_bytes,
                    receipt_id, state, received_at, retention_until
                ) VALUES ($1, $2, DATE '2026-10-05', now(), 'periodic', $3,
                    'detector-test', 'processor-test', $4, $5, $6, 'received', now(), now() + interval '7 days')
                """,
                sample_id,
                camera_id,
                frame_sha,
                frame_key,
                len(frame_bytes),
                uuid4(),
            )
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                len(frame_bytes),
            )
        finally:
            await connection.close()

        annotations = PostgresAnnotationRepository(database_url=database_url)
        await annotations.ensure_schema()
        bbox_assignment = await annotations.create_review_assignment(
            sample_id=sample_id,
            stage="bbox",
            label_studio_task_id=bbox_task_id,
            bbox_revision="detector-draft-0",
            media_object_key=frame_key,
            required_bytes=len(frame_bytes),
        )
        bbox_revision = await AnnotationService(
            repository=annotations,
            label_studio=FakeSubmittedTask(
                {
                    "id": bbox_task_id,
                    "annotations": [
                        {
                            "id": uuid4().int % 2_000_000_000 + 1,
                            "was_cancelled": False,
                            "completed_by": {"id": 7},
                            "result": [{"from_name": "bbox", "type": "rectanglelabels", "value": {"x": 1}}],
                        }
                    ],
                }
            ),
        ).finalize_annotation(str(sample_id), bbox_assignment.revision)
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                len(crop_bytes),
            )
            await connection.execute(
                """
                INSERT INTO annotation_crops (
                    crop_id, sample_id, bbox_revision, region_index, region_id,
                    object_key, sha256, object_size_bytes, state, caption_state,
                    provenance, crop_set_ready
                ) VALUES ($1, $2, $3, 0, 'person-0', $4, $5, $6, 'ready', 'needs_review', '{}'::jsonb, TRUE)
                """,
                crop_id,
                sample_id,
                UUID(bbox_revision["annotation_revision_id"]),
                crop_key,
                crop_sha,
                len(crop_bytes),
            )
        finally:
            await connection.close()
        caption_assignment = await annotations.create_review_assignment(
            sample_id=sample_id,
            stage="caption",
            label_studio_task_id=caption_task_id,
            bbox_revision=bbox_revision["annotation_revision_id"],
            media_object_key=crop_key,
            required_bytes=len(crop_bytes),
        )
        try:
            await annotations.close_review_assignment(
                sample_id=sample_id,
                revision=caption_assignment.revision,
                outcome="rejected",
                reason="human_rejected_caption",
            )
            result = await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)
            crop_row = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision["annotation_revision_id"]),
            )
            assert result["crops_deleted"] >= 1
            assert await ingestion.sample_state(sample_id) == "received"
            assert crop_row[0]["state"] == "deleted"
            assert crop_row[0]["caption_state"] == "rejected"
            assert await ingestion.storage_bytes() >= len(frame_bytes)
            with pytest.raises(FileNotFoundError):
                objects.read_object(object_key=crop_key, expected_sha256=crop_sha)
            assert objects.read_object(object_key=frame_key, expected_sha256=frame_sha) == frame_bytes
        finally:
            await annotations.close()
            await ingestion.close()

    asyncio.run(exercise())
