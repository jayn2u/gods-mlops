from __future__ import annotations

import asyncio
import json
import os
from uuid import UUID, uuid4

import asyncpg
import pytest

from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.ingestion.storage import PostgresIngestionRepository


class SubmittedBBox:
    def __init__(self, task: dict) -> None:
        self.task = task

    async def get_task(self, task_id: int) -> dict:
        assert task_id == self.task["id"]
        return self.task


def test_uncertain_relevance_is_versioned_and_ineligible_until_corrected() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")
    sample_id, camera_id = uuid4(), uuid4()
    task_id = uuid4().int % 2_000_000_000 + 1
    frame_sha = (uuid4().hex + uuid4().hex)[:64]
    frame_key = f"samples/{camera_id}/{sample_id}/{frame_sha}.jpg"
    task = {
        "id": task_id,
        "annotations": [
            {
                "id": uuid4().int % 2_000_000_000 + 1,
                "was_cancelled": False,
                "completed_by": {"id": 11, "email": "human@example.invalid"},
                "result": [
                    {
                        "id": "person",
                        "from_name": "bbox",
                        "type": "rectanglelabels",
                        "original_width": 100,
                        "original_height": 50,
                        "value": {
                            "x": 20,
                            "y": 20,
                            "width": 30,
                            "height": 40,
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
                ) VALUES ($1, $2, DATE '2026-10-05', now(), 'periodic', $3,
                    'detector-test', 'processor-test', $4, 128, $5, 'received', now(), now() + interval '7 days')
                """,
                sample_id,
                camera_id,
                frame_sha,
                frame_key,
                uuid4(),
            )
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + 128 WHERE singleton = TRUE"
            )
        finally:
            await connection.close()

        repository = PostgresAnnotationRepository(database_url=database_url)
        await repository.ensure_schema()
        assignment = await repository.create_review_assignment(
            sample_id=sample_id,
            stage="bbox",
            label_studio_task_id=task_id,
            bbox_revision="detector-draft-0",
            media_object_key=frame_key,
            required_bytes=128,
        )
        try:
            bbox_revision = await AnnotationService(
                repository=repository,
                label_studio=SubmittedBBox(task),
            ).finalize_annotation(str(sample_id), assignment.revision)
            gallery = []
            connection = await asyncpg.connect(database_url)
            try:
                for index, digest in enumerate(("1" * 64, "2" * 64, "3" * 64)):
                    crop_id = uuid4()
                    await connection.execute(
                        """
                        INSERT INTO annotation_crops (
                            crop_id, sample_id, bbox_revision, region_index, region_id,
                            object_key, sha256, object_size_bytes, state, caption_state,
                            provenance, crop_set_ready
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, 32, 'ready', 'needs_review', $8::jsonb, TRUE)
                        """,
                        crop_id,
                        sample_id,
                        UUID(bbox_revision["annotation_revision_id"]),
                        index,
                        f"region-{index}",
                        f"crops/{sample_id}/{crop_id}.jpg",
                        digest,
                        json.dumps({"source": "test"}),
                    )
                    gallery.append({"crop_id": str(crop_id), "sha256": digest})
                await connection.execute(
                    "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + 96 WHERE singleton = TRUE"
                )
            finally:
                await connection.close()

            provenance = {
                "source": "label_studio",
                "label_studio_task_id": 52,
                "label_studio_annotation_id": 63,
                "reviewer_id": "human-11",
            }
            first = await repository.record_relevance_judgments(
                query_id="night-shift-query",
                query_text="a person wearing a red jacket",
                query_revision="query-sha-v1",
                gallery=gallery,
                judgments=[
                    {"crop_id": gallery[0]["crop_id"], "judgment": "relevant"},
                    {"crop_id": gallery[1]["crop_id"], "judgment": "not_relevant"},
                    {"crop_id": gallery[2]["crop_id"], "judgment": "uncertain"},
                ],
                provenance=provenance,
            )
            assert first["status"] == "unresolved"
            assert first["evaluation_eligible"] is False
            assert first["positive_count"] == 1
            assert first["negative_count"] == 1
            assert first["uncertain_count"] == 1

            with pytest.raises(ValueError, match="every selected gallery crop"):
                await repository.record_relevance_judgments(
                    query_id="night-shift-query",
                    query_text="a person wearing a red jacket",
                    query_revision="query-sha-v1",
                    gallery=gallery,
                    judgments=[{"crop_id": gallery[0]["crop_id"], "judgment": "relevant"}],
                    provenance=provenance,
                )

            corrected = await repository.record_relevance_judgments(
                query_id="night-shift-query",
                query_text="a person wearing a red jacket",
                query_revision="query-sha-v1",
                gallery=gallery,
                judgments=[
                    {"crop_id": item["crop_id"], "judgment": "relevant" if index == 0 else "not_relevant"}
                    for index, item in enumerate(gallery)
                ],
                provenance={**provenance, "label_studio_annotation_id": 64},
            )
            assert corrected["relevance_revision_id"] != first["relevance_revision_id"]
            assert corrected["status"] == "complete"
            assert corrected["evaluation_eligible"] is True
            previous = await repository.relevance_revision(first["relevance_revision_id"])
            assert {item["judgment"] for item in previous["judgments"]} == {
                "relevant",
                "not_relevant",
                "uncertain",
            }
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())
