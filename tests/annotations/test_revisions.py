from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import asyncpg
import pytest

from gods_mlops.annotations.models import AnnotationNotSubmittedError, ReviewAssignmentConflictError
from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.ingestion.storage import PostgresIngestionRepository


class FakeLabelStudio:
    def __init__(self, task: dict) -> None:
        self.task = task

    async def get_task(self, task_id: int) -> dict:
        assert task_id == self.task["id"]
        return self.task


class BlockingLabelStudio(FakeLabelStudio):
    def __init__(self, task: dict, started: asyncio.Event, resume: asyncio.Event) -> None:
        super().__init__(task)
        self.started = started
        self.resume = resume

    async def get_task(self, task_id: int) -> dict:
        self.started.set()
        await self.resume.wait()
        return await super().get_task(task_id)


async def _seed_received_frame(database_url: str, *, size_bytes: int = 128) -> tuple[PostgresIngestionRepository, object, str]:
    ingestion = PostgresIngestionRepository(database_url=database_url)
    await ingestion.ensure_schema()
    sample_id = uuid4()
    camera_id = uuid4()
    digest = "c" * 64
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
    return ingestion, sample_id, object_key


def test_prediction_only_label_studio_task_cannot_finalize_in_postgres() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")

    sample_id = uuid4()
    camera_id = uuid4()
    receipt_id = uuid4()
    frame_sha256 = "a" * 64
    frame_object_key = f"samples/{camera_id}/{sample_id}/{frame_sha256}.jpg"
    ls_task = {
        "id": 271,
        "data": {"sample_id": str(sample_id), "bbox_revision": "bbox-0"},
        "predictions": [
            {
                "model_version": "rtdetr_v2_r18vd@56509617",
                "result": [
                    {
                        "from_name": "bbox",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "value": {"x": 10, "y": 20, "width": 30, "height": 40, "rectanglelabels": ["person"]},
                    }
                ],
            }
        ],
        "annotations": [],
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
                    'periodic', $3, 'detector-test', 'processor-test', $4, 128,
                    $5, 'received', now(), now() + interval '7 days')
                ON CONFLICT (sample_id) DO NOTHING
                """,
                sample_id,
                camera_id,
                frame_sha256,
                frame_object_key,
                receipt_id,
            )
            await connection.execute(
                """
                UPDATE ingestion_storage_usage SET used_bytes = used_bytes + 128
                WHERE singleton = TRUE
                """
            )
        finally:
            await connection.close()

        repository = PostgresAnnotationRepository(database_url=database_url)
        await repository.ensure_schema()
        assignment = await repository.create_review_assignment(
            sample_id=sample_id,
            stage="bbox",
            label_studio_task_id=ls_task["id"],
            bbox_revision="bbox-0",
            media_object_key=frame_object_key,
            required_bytes=128,
        )
        service = AnnotationService(repository=repository, label_studio=FakeLabelStudio(ls_task))

        try:
            with pytest.raises(AnnotationNotSubmittedError):
                await service.finalize_annotation(str(sample_id), assignment.revision)

            assert await repository.assignment_state(assignment.revision) == "active"
            assert await repository.annotation_revision_count(sample_id) == 0
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_submitted_annotation_is_snapshotted_and_repeat_finalize_is_immutable() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")
    task_id = uuid4().int % 2_000_000_000 + 1
    annotation_id = uuid4().int % 2_000_000_000 + 1
    task = {
        "id": task_id,
        "predictions": [{"result": [{"fake": "prediction"}]}],
        "annotations": [
            {
                "id": annotation_id,
                "was_cancelled": False,
                "created_at": "2026-10-05T01:00:00Z",
                "updated_at": "2026-10-05T01:05:00Z",
                "completed_by": {"id": 7, "email": "reviewer@example.invalid"},
                "result": [{"from_name": "bbox", "value": {"rectanglelabels": ["person"]}}],
            }
        ],
    }

    async def exercise() -> None:
        ingestion, sample_id, frame_key = await _seed_received_frame(database_url)
        repository = PostgresAnnotationRepository(database_url=database_url)
        await repository.ensure_schema()
        assignment = await repository.create_review_assignment(
            sample_id=sample_id,
            stage="bbox",
            label_studio_task_id=task_id,
            bbox_revision="bbox-0",
            media_object_key=frame_key,
            required_bytes=128,
        )
        label_studio = FakeLabelStudio(task)
        service = AnnotationService(repository=repository, label_studio=label_studio)
        try:
            first = await service.finalize_annotation(str(sample_id), assignment.revision)
            assert first["label_studio_annotation_id"] == annotation_id
            assert first["result"] == task["annotations"][0]["result"]
            assert first["provenance"]["completed_by"]["id"] == 7
            task["annotations"] = []
            second = await service.finalize_annotation(str(sample_id), assignment.revision)
            assert second == first
            assert await repository.annotation_revision_count(sample_id) == 1
            assert await repository.assignment_state(assignment.revision) == "finalized"
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_bbox_edit_wins_race_with_old_finalize_without_mixing_revisions() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")
    task_id = uuid4().int % 2_000_000_000 + 1
    task = {
        "id": task_id,
        "annotations": [
            {"id": uuid4().int % 2_000_000_000 + 1, "was_cancelled": False, "result": [{"value": {"x": 1}}]}
        ],
    }

    async def exercise() -> None:
        ingestion, sample_id, frame_key = await _seed_received_frame(database_url)
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
        started = asyncio.Event()
        resume = asyncio.Event()
        service = AnnotationService(
            repository=repository,
            label_studio=BlockingLabelStudio(task, started, resume),
        )
        finalization = asyncio.create_task(
            service.finalize_annotation(str(sample_id), assignment.revision)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            edited = await repository.mark_bbox_edit(
                sample_id=sample_id,
                expected_revision=assignment.revision,
            )
        finally:
            resume.set()
        outcome = await asyncio.gather(finalization, return_exceptions=True)
        try:
            assert isinstance(outcome[0], ReviewAssignmentConflictError)
            assert edited.revision != assignment.revision
            assert edited.bbox_revision == assignment.bbox_revision
            assert await repository.assignment_state(assignment.revision) == "superseded"
            assert await repository.assignment_state(edited.revision) == "provisioning"
            assert await repository.annotation_revision_count(sample_id) == 0
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())


def test_existing_label_studio_upload_is_charged_to_both_shared_ledgers() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("isolated PostgreSQL test endpoint is not configured")

    async def exercise() -> None:
        ingestion, sample_id, frame_key = await _seed_received_frame(database_url)
        global_bytes_after_seed = await ingestion.storage_bytes()
        connection = await asyncpg.connect(database_url)
        try:
            review_bytes_before_assignment = await connection.fetchval(
                "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE"
            )
        finally:
            await connection.close()
        repository = PostgresAnnotationRepository(database_url=database_url)
        await repository.ensure_schema()
        assignment = await repository.create_review_assignment(
            sample_id=sample_id,
            stage="bbox",
            label_studio_task_id=271,
            bbox_revision="existing-label-studio-task",
            media_object_key=frame_key,
            required_bytes=128,
        )
        try:
            media = await repository.register_existing_label_studio_media(
                revision=assignment.revision,
                project_id=17,
                task_id=271,
                file_upload_id=381,
                filename="fixture.jpg",
                upload_path="/data/upload/17/fixture.jpg",
                sha256_digest="d" * 64,
                object_size_bytes=64,
            )
            assert media["state"] == "uploaded"
            assert media["object_size_bytes"] == 64
            assert await repository.register_existing_label_studio_media(
                revision=assignment.revision,
                project_id=17,
                task_id=271,
                file_upload_id=381,
                filename="fixture.jpg",
                upload_path="/data/upload/17/fixture.jpg",
                sha256_digest="d" * 64,
                object_size_bytes=64,
            ) == media
            assert await ingestion.storage_bytes() == global_bytes_after_seed + 64
            connection = await asyncpg.connect(database_url)
            try:
                assert await connection.fetchval(
                    "SELECT active_bytes FROM review_storage_usage WHERE singleton = TRUE"
                ) == review_bytes_before_assignment + 192
            finally:
                await connection.close()
            await repository.close_review_assignment(
                sample_id=sample_id,
                revision=assignment.revision,
                outcome="cancelled",
                reason="test_cleanup",
            )
            await repository.mark_label_studio_media_delete_pending(assignment.revision)
            assert assignment.revision in await repository.label_studio_media_cleanup_candidates()
            assert await repository.finish_label_studio_media_cleanup(assignment.revision) is True
            assert (await repository.label_studio_media_upload(assignment.revision))["state"] == "deleted"
            assert await ingestion.storage_bytes() == global_bytes_after_seed
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())
