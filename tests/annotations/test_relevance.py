from __future__ import annotations

import asyncio
import json
import os
from uuid import UUID, uuid4

import asyncpg
import pytest

from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.models import ReviewAssignmentConflictError
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.annotations.workflow import LabelStudioReviewWorkflow
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


def test_pending_relevance_dependencies_invalidate_but_frozen_truth_remains_immutable() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")
    sample_id, camera_id = uuid4(), uuid4()
    frame_sha = "d" * 64
    frame_key = f"samples/{camera_id}/{sample_id}/{frame_sha}.jpg"
    bbox_task_id = uuid4().int % 2_000_000_000 + 1
    bbox_result = [
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
    ]

    class MemoryLabelStudio:
        def __init__(self) -> None:
            self.tasks: dict[int, dict] = {}

        async def get_task(self, task_id: int) -> dict:
            return self.tasks[task_id]

        async def delete_review_task(self, **_kwargs) -> None:
            return None

    class MemoryMediaCleanup:
        async def delete_upload(self, **_kwargs) -> dict[str, bool]:
            return {"deleted": True, "already_absent": False}

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
        bbox_assignment = await repository.create_review_assignment(
            sample_id=sample_id,
            stage="bbox",
            label_studio_task_id=bbox_task_id,
            bbox_revision="bbox-draft-0",
            media_object_key=frame_key,
            required_bytes=128,
        )
        old_bbox = await AnnotationService(
            repository=repository,
            label_studio=SubmittedBBox(
                {
                    "id": bbox_task_id,
                    "annotations": [
                        {
                            "id": uuid4().int % 2_000_000_000 + 1,
                            "was_cancelled": False,
                            "completed_by": {"id": 11, "email": "human@example.invalid"},
                            "result": bbox_result,
                        }
                    ],
                }
            ),
        ).finalize_annotation(str(sample_id), bbox_assignment.revision)
        bbox_revision_id = UUID(old_bbox["annotation_revision_id"])
        crop_ids = (uuid4(), uuid4())
        gallery = [
            {"crop_id": str(crop_ids[0]), "sha256": "1" * 64},
            {"crop_id": str(crop_ids[1]), "sha256": "2" * 64},
        ]
        crop_keys = (f"crops/{sample_id}/{crop_ids[0]}.jpg", f"crops/{sample_id}/{crop_ids[1]}.jpg")
        connection = await asyncpg.connect(database_url)
        try:
            for index, (crop_id, digest, object_key) in enumerate(zip(crop_ids, ("1" * 64, "2" * 64), crop_keys, strict=True)):
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
                    bbox_revision_id,
                    index,
                    f"region-{index}",
                    object_key,
                    digest,
                    json.dumps({"source": "test"}),
                )
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + 64 WHERE singleton = TRUE"
            )
        finally:
            await connection.close()

        caption_task_id = uuid4().int % 2_000_000_000 + 1
        caption_text = "a person in a red jacket"
        caption_assignment = await repository.create_review_assignment(
            sample_id=sample_id,
            stage="caption",
            label_studio_task_id=caption_task_id,
            bbox_revision=str(bbox_revision_id),
            media_object_key=crop_keys[0],
            required_bytes=32,
        )
        old_caption = await AnnotationService(
            repository=repository,
            label_studio=SubmittedBBox(
                {
                    "id": caption_task_id,
                    "annotations": [
                        {
                            "id": uuid4().int % 2_000_000_000 + 1,
                            "was_cancelled": False,
                            "completed_by": {"id": 11, "email": "human@example.invalid"},
                            "result": [
                                {
                                    "from_name": "caption",
                                    "type": "textarea",
                                    "value": {"text": [caption_text]},
                                }
                            ],
                        }
                    ],
                }
            ),
        ).finalize_annotation(str(sample_id), caption_assignment.revision)
        old_caption_id = old_caption["annotation_revision_id"]
        provenance = {
            "source": "label_studio",
            "label_studio_task_id": 52,
            "label_studio_annotation_id": 63,
            "reviewer_id": "human-11",
        }
        judgments = [
            {"crop_id": str(crop_ids[0]), "judgment": "relevant"},
            {"crop_id": str(crop_ids[1]), "judgment": "not_relevant"},
        ]
        common = {
            "query_id": "caption-query",
            "query_text": caption_text,
            "query_revision": "query-v1",
            "gallery": gallery,
        }
        try:
            frozen_draft = await repository.prepare_relevance_review(
                **common,
                query_source_caption_revision_id=old_caption_id,
            )
            frozen = await repository.freeze_relevance_review(
                review_id=frozen_draft["review_id"],
                judgments=judgments,
                provenance=provenance,
            )
            caption_dependent = await repository.prepare_relevance_review(
                **common,
                query_source_caption_revision_id=old_caption_id,
            )
            freehand_same_text = await repository.prepare_relevance_review(
                **common,
                query_source_caption_revision_id=None,
            )
            transaction_review = await repository.prepare_relevance_review(
                **common,
                query_source_caption_revision_id=None,
            )
            missed_callback = await repository.prepare_relevance_review(
                **common,
                query_source_caption_revision_id=old_caption_id,
            )

            class RollbackPublication(Exception):
                pass

            publication_connection = await asyncpg.connect(database_url)
            try:
                with pytest.raises(RollbackPublication):
                    async with publication_connection.transaction():
                        frozen_in_publication = await repository.freeze_relevance_review_in_transaction(
                            publication_connection,
                            review_id=transaction_review["review_id"],
                            judgments=judgments,
                            provenance={**provenance, "label_studio_annotation_id": 66},
                        )
                        assert frozen_in_publication["state"] == "frozen"
                        raise RollbackPublication()
            finally:
                await publication_connection.close()
            assert (await repository.relevance_review(transaction_review["review_id"]))["state"] == "pending"

            edited_caption_task_id = uuid4().int % 2_000_000_000 + 1
            edited_filename = f"caption-edit-{uuid4().hex}.jpg"
            label_studio = MemoryLabelStudio()
            workflow = LabelStudioReviewWorkflow(
                repository=repository,
                objects=None,
                label_studio=label_studio,
                media_cleanup=MemoryMediaCleanup(),
            )
            edited_caption_assignment = await workflow.prepare_assignment(
                sample_id=sample_id,
                stage="caption",
                project_id=17,
                bbox_revision=str(bbox_revision_id),
                media_object_key=crop_keys[0],
                required_bytes=32,
                expected_caption_revision_id=old_caption_id,
            )
            await repository.reserve_label_studio_media(
                revision=edited_caption_assignment.revision,
                project_id=17,
                filename=edited_filename,
                sha256_digest="1" * 64,
                object_size_bytes=32,
            )
            await repository.bind_label_studio_media(
                revision=edited_caption_assignment.revision,
                task_id=edited_caption_task_id,
                file_upload_id=uuid4().int % 2_000_000_000 + 1,
                upload_path=f"/data/upload/17/{edited_filename}",
            )
            label_studio.tasks[edited_caption_task_id] = {
                "id": edited_caption_task_id,
                "annotations": [
                    {
                        "id": uuid4().int % 2_000_000_000 + 1,
                        "was_cancelled": False,
                        "completed_by": {"id": 12, "email": "human2@example.invalid"},
                        "result": [
                            {
                                "from_name": "caption",
                                "type": "textarea",
                                "value": {"text": ["a person in a blue jacket"]},
                            }
                        ],
                    }
                ],
            }
            await workflow.finalize_annotation(str(sample_id), edited_caption_assignment.revision)

            caption_review = await repository.relevance_review(caption_dependent["review_id"])
            freehand_review = await repository.relevance_review(freehand_same_text["review_id"])
            assert caption_review["state"] == "needs_review"
            assert caption_review["invalidation_reason"] == "query_caption_revision_changed"
            assert freehand_review["state"] == "pending"

            connection = await asyncpg.connect(database_url)
            try:
                await connection.execute(
                    "UPDATE relevance_matrix_review_drafts SET state = 'pending', invalidation_reason = NULL WHERE review_id = $1",
                    UUID(missed_callback["review_id"]),
                )
            finally:
                await connection.close()
            with pytest.raises(ReviewAssignmentConflictError):
                await repository.freeze_relevance_review(
                    review_id=missed_callback["review_id"],
                    judgments=judgments,
                    provenance={**provenance, "label_studio_annotation_id": 64},
                )
            assert (await repository.relevance_review(missed_callback["review_id"]))["state"] == "needs_review"

            edited_bbox_task_id = uuid4().int % 2_000_000_000 + 1
            edited_bbox_assignment = await repository.create_review_assignment(
                sample_id=sample_id,
                stage="bbox",
                label_studio_task_id=edited_bbox_task_id,
                bbox_revision=str(bbox_revision_id),
                media_object_key=frame_key,
                required_bytes=128,
            )
            new_bbox = await AnnotationService(
                repository=repository,
                label_studio=SubmittedBBox(
                    {
                        "id": edited_bbox_task_id,
                        "annotations": [
                            {
                                "id": uuid4().int % 2_000_000_000 + 1,
                                "was_cancelled": False,
                                "completed_by": {"id": 13, "email": "human3@example.invalid"},
                                "result": [{**bbox_result[0], "value": {**bbox_result[0]["value"], "x": 25}}],
                            }
                        ],
                    }
                ),
            ).finalize_annotation(str(sample_id), edited_bbox_assignment.revision)
            assert new_bbox["annotation_revision_id"] != old_bbox["annotation_revision_id"]

            connection = await asyncpg.connect(database_url)
            try:
                await connection.execute(
                    "UPDATE relevance_matrix_review_drafts SET state = 'pending', invalidation_reason = NULL WHERE review_id = $1",
                    UUID(freehand_same_text["review_id"]),
                )
            finally:
                await connection.close()
            with pytest.raises(ReviewAssignmentConflictError):
                await repository.freeze_relevance_review(
                    review_id=freehand_same_text["review_id"],
                    judgments=judgments,
                    provenance={**provenance, "label_studio_annotation_id": 65},
                )
            crop_review = await repository.relevance_review(freehand_same_text["review_id"])
            assert crop_review["state"] == "needs_review"
            assert crop_review["invalidation_reason"] == "gallery_crop_revision_changed"

            frozen_before = await repository.relevance_revision(frozen["relevance_revision_id"])
            frozen_after = await repository.relevance_revision(frozen["relevance_revision_id"])
            assert frozen_after["query_sha256"] == frozen_before["query_sha256"]
            assert frozen_after["gallery_sha256"] == frozen_before["gallery_sha256"]
            assert frozen_after["judgments_sha256"] == frozen_before["judgments_sha256"]
            assert frozen_after["judgments"] == frozen_before["judgments"]
            assert (await repository.relevance_review(frozen_draft["review_id"]))["state"] == "frozen"
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())
