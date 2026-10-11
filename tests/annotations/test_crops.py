from __future__ import annotations

import asyncio
import hashlib
import io
import os
import threading
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import asyncpg
import boto3
import pytest
from PIL import Image, ImageDraw

from gods_mlops.annotations.crops import CropService
from gods_mlops.annotations.label_studio import LabelStudioTaskReference
from gods_mlops.annotations.models import ReviewAssignmentConflictError
from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.annotations.workflow import LabelStudioReviewWorkflow
from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore
from gods_mlops.retention.service import RetentionService


class SubmittedBBoxTask:
    def __init__(self, task: dict) -> None:
        self.task = task

    async def get_task(self, task_id: int) -> dict:
        assert task_id == self.task["id"]
        return self.task


def _configured() -> tuple[str, str, str, str, str] | None:
    keys = (
        "GODS_MLOPS_TEST_DATABASE_URL",
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )
    values = tuple(os.environ.get(key, "") for key in keys)
    return values if all(values) else None


async def _seed_single_bbox_source(
    *,
    database_url: str,
    objects: S3SampleStore,
) -> tuple[PostgresIngestionRepository, PostgresAnnotationRepository, UUID, str, bytes, str, int]:
    ingestion = PostgresIngestionRepository(database_url=database_url)
    await ingestion.ensure_schema()
    annotations = PostgresAnnotationRepository(database_url=database_url)
    await annotations.ensure_schema()
    initial_usage = await ingestion.storage_bytes()
    sample_id, camera_id = uuid4(), uuid4()
    image = Image.new("RGB", (100, 50), "white")
    frame_buffer = io.BytesIO()
    image.save(frame_buffer, format="JPEG", quality=92)
    frame_bytes = frame_buffer.getvalue()
    frame_sha = hashlib.sha256(frame_bytes).hexdigest()
    frame_key = f"samples/{camera_id}/{sample_id}/{frame_sha}.jpg"
    objects.ensure_object(object_key=frame_key, image=frame_bytes, expected_sha256=frame_sha)
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
            datetime.now(timezone.utc),
            frame_sha,
            frame_key,
            len(frame_bytes),
            uuid4(),
            datetime.now(timezone.utc) + timedelta(days=7),
        )
        await connection.execute(
            "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
            len(frame_bytes),
        )
    finally:
        await connection.close()
    task_id = uuid4().int % 2_000_000_000 + 1
    assignment = await annotations.create_review_assignment(
        sample_id=sample_id,
        stage="bbox",
        label_studio_task_id=task_id,
        bbox_revision="detector-draft-0",
        media_object_key=frame_key,
        required_bytes=len(frame_bytes),
    )
    bbox_task = {
        "id": task_id,
        "annotations": [
            {
                "id": uuid4().int % 2_000_000_000 + 1,
                "was_cancelled": False,
                "completed_by": {"id": 11, "email": "reviewer@example.invalid"},
                "result": [
                    {
                        "id": "person",
                        "from_name": "bbox",
                        "type": "rectanglelabels",
                        "original_width": 100,
                        "original_height": 50,
                        "value": {
                            "x": 10,
                            "y": 10,
                            "width": 70,
                            "height": 80,
                            "rectanglelabels": ["person"],
                        },
                    }
                ],
            }
        ],
    }
    bbox_revision = await AnnotationService(
        repository=annotations,
        label_studio=SubmittedBBoxTask(bbox_task),
    ).finalize_annotation(str(sample_id), assignment.revision)
    return (
        ingestion,
        annotations,
        sample_id,
        frame_key,
        frame_bytes,
        bbox_revision["annotation_revision_id"],
        initial_usage,
    )


def test_create_crop_writes_verified_s3_media_with_bbox_revision_provenance() -> None:
    configured = _configured()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    database_url, s3_endpoint, access_key, secret_key, bucket = configured
    task_id = uuid4().int % 2_000_000_000 + 1
    frame_id = uuid4()
    camera_id = uuid4()
    image = Image.new("RGB", (100, 50), "white")
    ImageDraw.Draw(image).rectangle((20, 10, 79, 39), fill=(180, 20, 40))
    frame_bytes = io.BytesIO()
    image.save(frame_bytes, format="JPEG", quality=95)
    original = frame_bytes.getvalue()
    original_sha = hashlib.sha256(original).hexdigest()
    frame_key = f"samples/{camera_id}/{frame_id}/{original_sha}.jpg"
    objects = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    objects.ensure_object(object_key=frame_key, image=original, expected_sha256=original_sha)
    task = {
        "id": task_id,
        "data": {"sample_id": str(frame_id)},
        "predictions": [],
        "annotations": [
            {
                "id": uuid4().int % 2_000_000_000 + 1,
                "was_cancelled": False,
                "completed_by": {"id": 7},
                "result": [
                    {
                        "id": "person-1",
                        "from_name": "bbox",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "original_width": 100,
                        "original_height": 50,
                        "value": {
                            "x": 20,
                            "y": 20,
                            "width": 60,
                            "height": 60,
                            "rectanglelabels": ["person"],
                        },
                    }
                ],
            }
        ],
    }

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
                ) VALUES ($1, $2, DATE '2026-10-05', TIMESTAMPTZ '2026-10-05 00:00:00+00',
                    'periodic', $3, 'detector-test', 'processor-test', $4, $5,
                    $6, 'received', now(), now() + interval '7 days')
                """,
                frame_id,
                camera_id,
                original_sha,
                frame_key,
                len(original),
                uuid4(),
            )
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                len(original),
            )
        finally:
            await connection.close()

        repository = PostgresAnnotationRepository(database_url=database_url)
        await repository.ensure_schema()
        assignment = await repository.create_review_assignment(
            sample_id=frame_id,
            stage="bbox",
            label_studio_task_id=task_id,
            bbox_revision="detector-draft-0",
            media_object_key=frame_key,
            required_bytes=len(original),
        )
        try:
            finalized = await AnnotationService(
                repository=repository,
                label_studio=SubmittedBBoxTask(task),
            ).finalize_annotation(str(frame_id), assignment.revision)
            crop_batch = await CropService(repository=repository, objects=objects).create_crop(
                str(frame_id), finalized["annotation_revision_id"]
            )
            assert len(crop_batch["crops"]) == 1
            crop = crop_batch["crops"][0]
            assert crop["sample_id"] == str(frame_id)
            assert crop["bbox_revision"] == finalized["annotation_revision_id"]
            assert crop["caption_state"] == "needs_review"
            assert crop["parent_regenerable"] is True

            client = boto3.client(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="us-east-1",
            )
            response = client.get_object(Bucket=bucket, Key=crop["object_key"])
            crop_bytes = response["Body"].read()
            assert hashlib.sha256(crop_bytes).hexdigest() == crop["sha256"]
            with Image.open(io.BytesIO(crop_bytes)) as saved_crop:
                assert saved_crop.size == (60, 30)

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
                                "to_name": "image",
                                "type": "textarea",
                                "value": {"text": ["red jacket and dark trousers"]},
                            }
                        ],
                    }
                ],
            }
            caption_assignment = await repository.create_review_assignment(
                sample_id=frame_id,
                stage="caption",
                label_studio_task_id=caption_task_id,
                bbox_revision=finalized["annotation_revision_id"],
                media_object_key=crop["object_key"],
                required_bytes=crop["object_size_bytes"],
            )
            caption_revision = await AnnotationService(
                repository=repository,
                label_studio=SubmittedBBoxTask(caption_task),
            ).finalize_annotation(str(frame_id), caption_assignment.revision)
            refreshed = await repository.crops_for_revision(
                sample_id=frame_id,
                bbox_revision=UUID(finalized["annotation_revision_id"]),
            )
            assert refreshed[0]["caption_state"] == "reviewed"
            assert refreshed[0]["caption_revision_id"] == caption_revision["annotation_revision_id"]

            next_task_id = uuid4().int % 2_000_000_000 + 1
            next_task = {
                "id": next_task_id,
                "annotations": [
                    {
                        "id": uuid4().int % 2_000_000_000 + 1,
                        "was_cancelled": False,
                        "completed_by": {"id": 7},
                        "result": [
                            {
                                "id": "person-1-edited",
                                "from_name": "bbox",
                                "type": "rectanglelabels",
                                "original_width": 100,
                                "original_height": 50,
                                "value": {
                                    "x": 15,
                                    "y": 15,
                                    "width": 65,
                                    "height": 65,
                                    "rectanglelabels": ["person"],
                                },
                            }
                        ],
                    }
                ],
            }
            next_assignment = await repository.create_review_assignment(
                sample_id=frame_id,
                stage="bbox",
                label_studio_task_id=next_task_id,
                bbox_revision=finalized["annotation_revision_id"],
                media_object_key=frame_key,
                required_bytes=len(original),
            )
            next_finalized = await AnnotationService(
                repository=repository,
                label_studio=SubmittedBBoxTask(next_task),
            ).finalize_annotation(str(frame_id), next_assignment.revision)
            next_crop_batch = await CropService(repository=repository, objects=objects).create_crop(
                str(frame_id), next_finalized["annotation_revision_id"]
            )
            next_crop = next_crop_batch["crops"][0]
            old_crop = (await repository.crops_for_revision(
                sample_id=frame_id,
                bbox_revision=UUID(finalized["annotation_revision_id"]),
            ))[0]
            assert next_crop["crop_id"] != crop["crop_id"]
            assert next_crop["caption_state"] == "needs_review"
            assert old_crop["caption_state"] == "reviewed"
            assert old_crop["caption_revision_id"] == caption_revision["annotation_revision_id"]
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_crop_retry_reuses_reserved_bytes_and_hides_incomplete_crop_set() -> None:
    configured = _configured()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    database_url, s3_endpoint, access_key, secret_key, bucket = configured
    frame_id = uuid4()
    camera_id = uuid4()
    task_id = uuid4().int % 2_000_000_000 + 1
    image = Image.new("RGB", (100, 50), "white")
    frame_buffer = io.BytesIO()
    image.save(frame_buffer, format="JPEG", quality=92)
    frame_bytes = frame_buffer.getvalue()
    frame_sha = hashlib.sha256(frame_bytes).hexdigest()
    frame_key = f"samples/{camera_id}/{frame_id}/{frame_sha}.jpg"
    base_store = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    base_store.ensure_object(object_key=frame_key, image=frame_bytes, expected_sha256=frame_sha)
    task = {
        "id": task_id,
        "annotations": [
            {
                "id": uuid4().int % 2_000_000_000 + 1,
                "was_cancelled": False,
                "result": [
                    {
                        "id": "left-person",
                        "from_name": "bbox",
                        "type": "rectanglelabels",
                        "original_width": 100,
                        "original_height": 50,
                        "value": {"x": 0, "y": 0, "width": 25, "height": 100, "rectanglelabels": ["person"]},
                    },
                    {
                        "id": "right-person",
                        "from_name": "bbox",
                        "type": "rectanglelabels",
                        "original_width": 100,
                        "original_height": 50,
                        "value": {"x": 75, "y": 0, "width": 25, "height": 100, "rectanglelabels": ["person"]},
                    },
                    {
                        "id": "middle-person",
                        "from_name": "bbox",
                        "type": "rectanglelabels",
                        "original_width": 100,
                        "original_height": 50,
                        "value": {"x": 40, "y": 20, "width": 20, "height": 60, "rectanglelabels": ["person"]},
                    },
                ],
            }
        ],
    }

    class LoseSecondWriteResponse(S3SampleStore):
        writes = 0

        def ensure_object(self, *, object_key: str, image: bytes, expected_sha256: str) -> None:
            super().ensure_object(object_key=object_key, image=image, expected_sha256=expected_sha256)
            self.writes += 1
            if self.writes == 2:
                raise OSError("injected crop write acknowledgment loss")

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
                ) VALUES ($1, $2, DATE '2026-10-05', TIMESTAMPTZ '2026-10-05 00:00:00+00',
                    'periodic', $3, 'detector-test', 'processor-test', $4, $5,
                    $6, 'received', now(), now() + interval '7 days')
                """,
                frame_id,
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

        repository = PostgresAnnotationRepository(database_url=database_url)
        await repository.ensure_schema()
        assignment = await repository.create_review_assignment(
            sample_id=frame_id,
            stage="bbox",
            label_studio_task_id=task_id,
            bbox_revision="detector-draft-0",
            media_object_key=frame_key,
            required_bytes=len(frame_bytes),
        )
        flaky_store = LoseSecondWriteResponse(
            endpoint_url=s3_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region="us-east-1",
        )
        try:
            finalized = await AnnotationService(
                repository=repository,
                label_studio=SubmittedBBoxTask(task),
            ).finalize_annotation(str(frame_id), assignment.revision)
            failed_service = CropService(repository=repository, objects=flaky_store)
            with pytest.raises(OSError, match="acknowledgment loss"):
                await failed_service.create_crop(str(frame_id), finalized["annotation_revision_id"])

            usage_after_failed_response = await ingestion.storage_bytes()
            reserved = await repository.crops_for_revision(
                sample_id=frame_id,
                bbox_revision=UUID(finalized["annotation_revision_id"]),
            )
            assert len(reserved) == 3
            assert all(crop["crop_set_ready"] is False for crop in reserved)

            retried = await CropService(repository=repository, objects=base_store).create_crop(
                str(frame_id), finalized["annotation_revision_id"]
            )
            completed = await repository.crops_for_revision(
                sample_id=frame_id,
                bbox_revision=UUID(finalized["annotation_revision_id"]),
            )
            assert retried["crop_ids"] == [crop["crop_id"] for crop in reserved]
            assert len(completed) == 3
            assert all(crop["state"] == "ready" for crop in completed)
            assert all(crop["crop_set_ready"] is True for crop in completed)
            assert await ingestion.storage_bytes() == usage_after_failed_response
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_unadopted_pending_crop_expiry_removes_lost_ack_object_and_releases_bytes() -> None:
    configured = _configured()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    database_url, s3_endpoint, access_key, secret_key, bucket = configured
    objects = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    now = datetime.now(timezone.utc)

    class LoseCropPutResponse(S3SampleStore):
        def ensure_object(self, *, object_key: str, image: bytes, expected_sha256: str) -> None:
            super().ensure_object(object_key=object_key, image=image, expected_sha256=expected_sha256)
            if object_key.startswith("crops/"):
                raise OSError("injected crop PUT acknowledgment loss")

    async def exercise() -> None:
        ingestion, annotations, sample_id, frame_key, frame_bytes, bbox_revision, initial_usage = (
            await _seed_single_bbox_source(database_url=database_url, objects=objects)
        )
        flaky_store = LoseCropPutResponse(
            endpoint_url=s3_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region="us-east-1",
        )
        try:
            with pytest.raises(OSError, match="acknowledgment loss"):
                await CropService(repository=annotations, objects=flaky_store).create_crop(
                    str(sample_id), bbox_revision
                )
            pending = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision),
            )
            assert len(pending) == 1
            assert pending[0]["state"] == "pending"
            assert pending[0]["crop_set_ready"] is False
            assert objects.read_object(
                object_key=pending[0]["object_key"],
                expected_sha256=pending[0]["sha256"],
            )

            connection = await asyncpg.connect(database_url)
            try:
                await connection.execute(
                    "UPDATE ingestion_samples SET retention_until = $2 WHERE sample_id = $1",
                    sample_id,
                    now - timedelta(seconds=1),
                )
            finally:
                await connection.close()

            await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)

            expired = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision),
            )
            assert expired[0]["state"] == "deleted"
            assert await ingestion.sample_state(sample_id) == "expired"
            with pytest.raises(FileNotFoundError):
                objects.read_object(object_key=frame_key, expected_sha256=hashlib.sha256(frame_bytes).hexdigest())
            with pytest.raises(FileNotFoundError):
                objects.read_object(
                    object_key=expired[0]["object_key"],
                    expected_sha256=expired[0]["sha256"],
                )
            assert await ingestion.storage_bytes() == initial_usage
        finally:
            await annotations.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_expiry_fences_a_late_crop_writer_and_retries_failed_cleanup() -> None:
    configured = _configured()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    database_url, s3_endpoint, access_key, secret_key, bucket = configured
    objects = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    now = datetime.now(timezone.utc)
    write_started = threading.Event()
    release_write = threading.Event()

    class BlockLateCropPut(S3SampleStore):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.fail_delete_once = True

        def ensure_object(self, *, object_key: str, image: bytes, expected_sha256: str) -> None:
            if object_key.startswith("crops/"):
                write_started.set()
                if not release_write.wait(timeout=10):
                    raise TimeoutError("crop upload test writer was not released")
            super().ensure_object(object_key=object_key, image=image, expected_sha256=expected_sha256)

        def delete_object(self, object_key: str) -> None:
            if object_key.startswith("crops/") and self.fail_delete_once:
                self.fail_delete_once = False
                raise OSError("injected late crop cleanup failure")
            super().delete_object(object_key)

    async def exercise() -> None:
        ingestion, annotations, sample_id, frame_key, frame_bytes, bbox_revision, initial_usage = (
            await _seed_single_bbox_source(database_url=database_url, objects=objects)
        )
        late_store = BlockLateCropPut(
            endpoint_url=s3_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region="us-east-1",
        )
        writer = None
        try:
            writer = asyncio.create_task(
                CropService(repository=annotations, objects=late_store).create_crop(
                    str(sample_id), bbox_revision
                )
            )
            assert await asyncio.wait_for(asyncio.to_thread(write_started.wait, 5), timeout=6)
            connection = await asyncpg.connect(database_url)
            try:
                await connection.execute(
                    "UPDATE ingestion_samples SET retention_until = $2 WHERE sample_id = $1",
                    sample_id,
                    now - timedelta(seconds=1),
                )
            finally:
                await connection.close()

            await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)
            release_write.set()
            outcome = await asyncio.gather(writer, return_exceptions=True)
            assert isinstance(outcome[0], OSError)

            queued = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision),
            )
            assert queued[0]["state"] == "purge_pending"
            assert await ingestion.storage_bytes() == initial_usage
            with pytest.raises(FileNotFoundError):
                objects.read_object(object_key=frame_key, expected_sha256=hashlib.sha256(frame_bytes).hexdigest())
            assert objects.read_object(
                object_key=queued[0]["object_key"],
                expected_sha256=queued[0]["sha256"],
            )

            retried = await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)
            assert retried["crops_deleted"] >= 1
            final_rows = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision),
            )
            assert final_rows[0]["state"] == "deleted"
            with pytest.raises(FileNotFoundError):
                objects.read_object(
                    object_key=final_rows[0]["object_key"],
                    expected_sha256=final_rows[0]["sha256"],
                )
            assert await ingestion.storage_bytes() == initial_usage
        finally:
            release_write.set()
            if writer is not None and not writer.done():
                await asyncio.gather(writer, return_exceptions=True)
            await annotations.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_lost_crop_ack_after_expiry_requeues_late_object_for_cleanup() -> None:
    configured = _configured()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    database_url, s3_endpoint, access_key, secret_key, bucket = configured
    objects = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    now = datetime.now(timezone.utc)
    write_started = threading.Event()
    release_write = threading.Event()

    class LateLostAckStore(S3SampleStore):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.fail_cleanup_once = True

        def ensure_object(self, *, object_key: str, image: bytes, expected_sha256: str) -> None:
            if object_key.startswith("crops/"):
                write_started.set()
                if not release_write.wait(timeout=10):
                    raise TimeoutError("late crop upload test writer was not released")
                super().ensure_object(object_key=object_key, image=image, expected_sha256=expected_sha256)
                raise OSError("injected late crop PUT acknowledgment loss")
            super().ensure_object(object_key=object_key, image=image, expected_sha256=expected_sha256)

        def delete_object(self, object_key: str) -> None:
            if object_key.startswith("crops/") and self.fail_cleanup_once:
                self.fail_cleanup_once = False
                raise OSError("injected late crop cleanup failure")
            super().delete_object(object_key)

    async def exercise() -> None:
        ingestion, annotations, sample_id, frame_key, frame_bytes, bbox_revision, initial_usage = (
            await _seed_single_bbox_source(database_url=database_url, objects=objects)
        )
        late_store = LateLostAckStore(
            endpoint_url=s3_endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region="us-east-1",
        )
        writer = None
        try:
            writer = asyncio.create_task(
                CropService(repository=annotations, objects=late_store).create_crop(
                    str(sample_id), bbox_revision
                )
            )
            assert await asyncio.wait_for(asyncio.to_thread(write_started.wait, 5), timeout=6)
            connection = await asyncpg.connect(database_url)
            try:
                await connection.execute(
                    "UPDATE ingestion_samples SET retention_until = $2 WHERE sample_id = $1",
                    sample_id,
                    now - timedelta(seconds=1),
                )
            finally:
                await connection.close()

            await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)
            expired = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision),
            )
            assert expired[0]["state"] == "deleted"
            assert await ingestion.sample_state(sample_id) == "expired"
            assert await ingestion.storage_bytes() == initial_usage
            # Simulate a deleted row created by the pre-flag implementation, which
            # had already released its bytes but stored no quota_released marker.
            connection = await asyncpg.connect(database_url)
            try:
                await connection.execute(
                    "UPDATE annotation_crops SET quota_released = FALSE WHERE crop_id = $1",
                    UUID(expired[0]["crop_id"]),
                )
            finally:
                await connection.close()

            release_write.set()
            outcome = await asyncio.gather(writer, return_exceptions=True)
            assert isinstance(outcome[0], OSError)
            requeued = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision),
            )
            assert requeued[0]["state"] == "purge_pending"
            assert await ingestion.storage_bytes() == initial_usage
            connection = await asyncpg.connect(database_url)
            try:
                quota_released = await connection.fetchval(
                    "SELECT quota_released FROM annotation_crops WHERE crop_id = $1",
                    UUID(requeued[0]["crop_id"]),
                )
            finally:
                await connection.close()
            assert quota_released is True
            assert objects.read_object(
                object_key=requeued[0]["object_key"],
                expected_sha256=requeued[0]["sha256"],
            )

            retried = await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)
            assert retried["crops_deleted"] >= 1
            final_rows = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision),
            )
            assert final_rows[0]["state"] == "deleted"
            with pytest.raises(FileNotFoundError):
                objects.read_object(
                    object_key=final_rows[0]["object_key"],
                    expected_sha256=final_rows[0]["sha256"],
                )
            assert await ingestion.storage_bytes() == initial_usage
        finally:
            release_write.set()
            if writer is not None and not writer.done():
                await asyncio.gather(writer, return_exceptions=True)
            await annotations.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_caption_provisioning_protects_crop_until_label_studio_media_binds() -> None:
    configured = _configured()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    database_url, s3_endpoint, access_key, secret_key, bucket = configured
    objects = S3SampleStore(
        endpoint_url=s3_endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    import_started = asyncio.Event()
    allow_bind = asyncio.Event()
    now = datetime.now(timezone.utc)

    class PausedLabelStudio:
        async def import_media_task(
            self,
            *,
            project_id: int,
            filename: str,
            image: bytes,
            prediction: dict | None = None,
        ) -> LabelStudioTaskReference:
            import_started.set()
            await allow_bind.wait()
            return LabelStudioTaskReference(
                project_id=project_id,
                task_id=uuid4().int % 2_000_000_000 + 1,
                file_upload_id=uuid4().int % 2_000_000_000 + 1,
                media_path=f"/data/upload/{project_id}/{filename}",
                filename=filename,
            )

        async def delete_review_task(self, **_kwargs) -> None:
            return None

    class LocalMediaCleanup:
        async def delete_upload(self, **_kwargs) -> dict[str, bool]:
            return {"deleted": True, "already_absent": False}

    async def exercise() -> None:
        ingestion, annotations, sample_id, frame_key, frame_bytes, bbox_revision, initial_usage = (
            await _seed_single_bbox_source(database_url=database_url, objects=objects)
        )
        crop_batch = await CropService(repository=annotations, objects=objects).create_crop(
            str(sample_id), bbox_revision
        )
        crop = crop_batch["crops"][0]
        workflow = LabelStudioReviewWorkflow(
            repository=annotations,
            objects=objects,
            label_studio=PausedLabelStudio(),
            media_cleanup=LocalMediaCleanup(),
        )
        assignment = await workflow.prepare_assignment(
            sample_id=sample_id,
            stage="caption",
            project_id=31,
            bbox_revision=bbox_revision,
            media_object_key=crop["object_key"],
            required_bytes=crop["object_size_bytes"],
        )
        assert assignment.state == "provisioning"
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(
                "UPDATE ingestion_samples SET retention_until = $2 WHERE sample_id = $1",
                sample_id,
                now - timedelta(seconds=1),
            )
        finally:
            await connection.close()

        provision = asyncio.create_task(
            workflow.provision_task(revision=assignment.revision, project_id=31)
        )
        try:
            await asyncio.wait_for(import_started.wait(), timeout=5)
            await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)
            assert await ingestion.sample_state(sample_id) == "expired"

            allow_bind.set()
            bound = await provision
            assert bound["state"] == "active"
            assert bound["label_studio_task_id"] > 0
            current_crop = await annotations.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=UUID(bbox_revision),
            )
            assert current_crop[0]["state"] == "ready"
            assert current_crop[0]["caption_state"] == "needs_review"
            assert current_crop[0]["parent_available"] is False
            assert current_crop[0]["parent_regenerable"] is False
            assert objects.read_object(
                object_key=crop["object_key"],
                expected_sha256=crop["sha256"],
            )
            with pytest.raises(FileNotFoundError):
                objects.read_object(object_key=frame_key, expected_sha256=hashlib.sha256(frame_bytes).hexdigest())
            assert await ingestion.storage_bytes() == initial_usage + 2 * crop["object_size_bytes"]

            await workflow.close_assignment(
                sample_id=sample_id,
                revision=assignment.revision,
                outcome="rejected",
                reason="test_cleanup",
            )
            await RetentionService(
                repository=ingestion,
                objects=objects,
                annotations=annotations,
            ).expire_candidates(now)
            assert await ingestion.storage_bytes() == initial_usage
        finally:
            allow_bind.set()
            if not provision.done():
                await asyncio.gather(provision, return_exceptions=True)
            await annotations.close()
            await ingestion.close()

    asyncio.run(exercise())
