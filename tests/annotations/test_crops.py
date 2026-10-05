from __future__ import annotations

import asyncio
import hashlib
import io
import os
from uuid import UUID, uuid4

import asyncpg
import boto3
import pytest
from PIL import Image, ImageDraw

from gods_mlops.annotations.crops import CropService
from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore


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
