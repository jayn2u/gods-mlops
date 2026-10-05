from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from uuid import uuid4

import asyncpg
import pytest
from PIL import Image

from gods_mlops.annotations.label_studio import (
    LabelStudioApiClient,
    LabelStudioMediaCleanupClient,
    LabelStudioTaskReference,
)
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.annotations.workflow import LabelStudioReviewWorkflow
from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore


def test_label_studio_api_client_uses_bearer_access_tokens_with_explicit_legacy_option(monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self, _limit: int) -> bytes:
            return self._body

    def fake_urlopen(request, *, timeout):
        path = request.full_url.removeprefix("http://label-studio.invalid")
        authorization = request.get_header("Authorization")
        seen.append((path, authorization))
        body = b'{"access":"short-lived-access"}' if path == "/api/token/refresh/" else b"{}"
        return Response(body)

    monkeypatch.setattr("gods_mlops.annotations.label_studio.request.urlopen", fake_urlopen)
    default_client = LabelStudioApiClient(base_url="http://label-studio.invalid", api_token="refresh-token")
    assert default_client._request_json("/api/projects/", method="GET") == {}
    legacy_client = LabelStudioApiClient(
        base_url="http://label-studio.invalid",
        api_token="legacy-token",
        auth_scheme="Token",
    )
    assert legacy_client._request_json("/api/projects/", method="GET") == {}
    assert seen == [
        ("/api/token/refresh/", "Bearer refresh-token"),
        ("/api/projects/", "Bearer short-lived-access"),
        ("/api/projects/", "Token legacy-token"),
    ]


def test_pinned_label_studio_import_retries_lost_response_and_keeps_prediction_read_only() -> None:
    base_url = os.environ.get("GODS_MLOPS_TEST_LABEL_STUDIO_URL")
    api_token = os.environ.get("GODS_MLOPS_TEST_LABEL_STUDIO_API_TOKEN")
    cleanup_url = os.environ.get("GODS_MLOPS_TEST_LABEL_STUDIO_CLEANUP_URL")
    cleanup_token = os.environ.get("GODS_MLOPS_TEST_LABEL_STUDIO_CLEANUP_TOKEN")
    data_dir = os.environ.get("GODS_MLOPS_TEST_LABEL_STUDIO_DATA_DIR")
    if not all((base_url, api_token, cleanup_url, cleanup_token, data_dir)):
        pytest.skip("isolated Label Studio endpoint, credentials, and colocated cleanup sidecar are not configured")

    class LoseFirstImportResponse(LabelStudioApiClient):
        lost = False

        def _request_json(self, path: str, *, method: str, **kwargs):
            response = super()._request_json(path, method=method, **kwargs)
            if method == "POST" and "/import?" in path and not self.lost:
                self.lost = True
                raise RuntimeError("injected lost import response")
            return response

    async def exercise() -> None:
        client = LoseFirstImportResponse(
            base_url=base_url,
            api_token=api_token,
            auth_scheme=os.environ.get("GODS_MLOPS_TEST_LABEL_STUDIO_AUTH_SCHEME", "Bearer"),
        )
        cleanup_client = LabelStudioMediaCleanupClient(base_url=cleanup_url, token=cleanup_token)
        title = f"Gods Task 5 {uuid4().hex[:12]}"
        project_id = await client.create_project(
            title=title,
            label_config=(
                "<View><Image name='image' value='$image'/>"
                "<RectangleLabels name='bbox' toName='image'><Label value='person'/></RectangleLabels></View>"
            ),
        )
        image = Image.new("RGB", (48, 24), "white")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=90)
        media = output.getvalue()
        digest = hashlib.sha256(media).hexdigest()
        data_path = Path(data_dir) / "media" / "upload" / str(project_id)
        filename = f"review-{uuid4().hex}-{digest[:16]}.jpg"
        prediction = {
            "model_version": f"rtdetr-test-{uuid4().hex[:8]}",
            "result": [
                {
                    "id": "person-1",
                    "from_name": "bbox",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": 48,
                    "original_height": 24,
                    "value": {"x": 10, "y": 10, "width": 60, "height": 80, "rectanglelabels": ["person"]},
                }
            ],
        }
        with pytest.raises(RuntimeError, match="injected lost import response"):
            await client.import_media_task(
                project_id=project_id,
                filename=filename,
                image=media,
                prediction=prediction,
            )

        reference = await client.import_media_task(
            project_id=project_id,
            filename=filename,
            image=media,
            prediction=prediction,
        )
        retry = await client.import_media_task(
            project_id=project_id,
            filename=filename,
            image=media,
            prediction=prediction,
        )
        task = await client.get_task(reference.task_id)
        assert retry.task_id == reference.task_id
        assert retry.file_upload_id == reference.file_upload_id
        assert reference.media_path.startswith(f"/data/upload/{project_id}/")
        assert "?" not in reference.media_path
        assert task["annotations"] == []
        assert len(task["predictions"]) == 1
        assert task["predictions"][0]["model_version"] == prediction["model_version"]
        assert task["predictions"][0]["result"] == prediction["result"]

        uploaded_file = data_path / PurePosixPath(reference.media_path).name
        assert uploaded_file.is_file()
        assert uploaded_file.stat().st_size == len(media)
        assert hashlib.sha256(uploaded_file.read_bytes()).hexdigest() == digest

        await client.delete_review_task(
            project_id=project_id,
            task_id=reference.task_id,
            file_upload_id=reference.file_upload_id,
        )
        assert uploaded_file.is_file(), "native task/file metadata deletion must not be trusted to remove uploaded bytes"
        removed = await cleanup_client.delete_upload(
            project_id=project_id,
            file_upload_id=reference.file_upload_id,
            upload_path=reference.media_path,
            expected_sha256=digest,
            expected_size_bytes=len(media),
        )
        assert removed in (
            {"deleted": True, "already_absent": False},
            {"deleted": False, "already_absent": True},
        )
        assert not uploaded_file.exists(), "the colocated sidecar must remove the Label Studio media bytes"

    asyncio.run(exercise())


def test_bbox_edit_cleanup_cannot_delete_replacement_task_and_retry_is_idempotent() -> None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    endpoint_url = os.environ.get("GODS_MLOPS_TEST_S3_ENDPOINT")
    access_key = os.environ.get("GODS_MLOPS_TEST_S3_ACCESS_KEY")
    secret_key = os.environ.get("GODS_MLOPS_TEST_S3_SECRET_KEY")
    bucket = os.environ.get("GODS_MLOPS_TEST_S3_BUCKET")
    if not all((database_url, endpoint_url, access_key, secret_key, bucket)):
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")

    class LocalLabelStudio:
        def __init__(self) -> None:
            self.next_task_id = uuid4().int % 1_000_000_000 + 1
            self.next_file_id = uuid4().int % 1_000_000_000 + 1
            self.by_filename: dict[str, LabelStudioTaskReference] = {}
            self.tasks: dict[int, dict] = {}
            self.deleted: list[tuple[int, int]] = []
            self.lose_next_import_response = False

        async def import_media_task(
            self,
            *,
            project_id: int,
            filename: str,
            image: bytes,
            prediction: dict | None = None,
        ) -> LabelStudioTaskReference:
            prior = self.by_filename.get(filename)
            if prior is not None:
                return prior
            reference = LabelStudioTaskReference(
                project_id=project_id,
                task_id=self.next_task_id,
                file_upload_id=self.next_file_id,
                media_path=f"/data/upload/{project_id}/{filename}",
                filename=filename,
            )
            self.next_task_id += 1
            self.next_file_id += 1
            self.by_filename[filename] = reference
            self.tasks[reference.task_id] = {
                "id": reference.task_id,
                "data": {"image": reference.media_path},
                "annotations": [],
                "predictions": [prediction] if prediction else [],
                "image_bytes": image,
            }
            if self.lose_next_import_response:
                self.lose_next_import_response = False
                raise RuntimeError("injected lost replacement import response")
            return reference

        async def get_task(self, task_id: int) -> dict:
            return self.tasks[task_id]

        async def delete_review_task(self, *, project_id: int, task_id: int, file_upload_id: int) -> None:
            reference = next(
                reference
                for reference in self.by_filename.values()
                if reference.task_id == task_id
            )
            assert reference.project_id == project_id
            assert reference.file_upload_id == file_upload_id
            self.deleted.append((task_id, file_upload_id))
            self.tasks.pop(task_id)

    class LocalMediaCleanup:
        def __init__(self) -> None:
            self.deleted: list[tuple[int, int]] = []

        async def delete_upload(
            self,
            *,
            project_id: int,
            file_upload_id: int,
            upload_path: str,
            expected_sha256: str,
            expected_size_bytes: int,
        ) -> dict[str, bool]:
            self.deleted.append((project_id, file_upload_id))
            return {"deleted": True, "already_absent": False}

    async def exercise() -> None:
        ingestion = PostgresIngestionRepository(database_url=database_url)
        await ingestion.ensure_schema()
        repository = PostgresAnnotationRepository(database_url=database_url)
        await repository.ensure_schema()
        initial_object_bytes = await ingestion.storage_bytes()
        sample_id, camera_id = uuid4(), uuid4()
        frame = Image.new("RGB", (64, 32), "navy")
        encoded = io.BytesIO()
        frame.save(encoded, format="JPEG", quality=90)
        frame_bytes = encoded.getvalue()
        frame_sha = hashlib.sha256(frame_bytes).hexdigest()
        frame_key = f"samples/{camera_id}/{sample_id}/{frame_sha}.jpg"
        objects = S3SampleStore(
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            region="us-east-1",
        )
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
                    'detector-test', 'processor-test', $5, $6, $7, 'received', now(), now() + interval '7 days')
                """,
                sample_id,
                camera_id,
                datetime.now(timezone.utc),
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

        label_studio = LocalLabelStudio()
        media_cleanup = LocalMediaCleanup()
        workflow = LabelStudioReviewWorkflow(
            repository=repository,
            objects=objects,
            label_studio=label_studio,
            media_cleanup=media_cleanup,
        )
        try:
            first = await workflow.prepare_assignment(
                sample_id=sample_id,
                stage="bbox",
                project_id=17,
                bbox_revision="draft-0",
                media_object_key=frame_key,
                required_bytes=len(frame_bytes),
            )
            first_task = await workflow.provision_task(revision=first.revision, project_id=17)
            label_studio.tasks[first_task["label_studio_task_id"]]["annotations"] = [
                {
                    "id": 1,
                    "was_cancelled": False,
                    "completed_by": {"id": 11, "email": "reviewer@example.invalid"},
                    "created_at": "2026-10-05T01:00:00Z",
                    "result": [{"from_name": "bbox", "value": {"x": 1}}],
                }
            ]
            first_snapshot = await workflow.finalize_annotation(str(sample_id), first.revision)

            editing = await workflow.prepare_assignment(
                sample_id=sample_id,
                stage="bbox",
                project_id=17,
                bbox_revision=first_snapshot["annotation_revision_id"],
                media_object_key=frame_key,
                required_bytes=len(frame_bytes),
            )
            editing_task = await workflow.provision_task(revision=editing.revision, project_id=17)
            label_studio.lose_next_import_response = True
            replacement = await repository.mark_bbox_edit(
                sample_id=sample_id,
                expected_revision=editing.revision,
            )

            assert replacement.state == "provisioning"
            assert replacement.label_studio_task_id is None

            with pytest.raises(RuntimeError, match="injected lost replacement import response"):
                await workflow.provision_task(revision=replacement.revision, project_id=17)
            charged_after_lost_ack = await ingestion.storage_bytes()
            replacement_task = await workflow.provision_task(revision=replacement.revision, project_id=17)
            assert await ingestion.storage_bytes() == charged_after_lost_ack
            assert replacement_task["label_studio_task_id"] not in {
                first_task["label_studio_task_id"],
                editing_task["label_studio_task_id"],
            }
            assert replacement_task["label_studio_file_upload_id"] != editing_task["label_studio_file_upload_id"]

            cleanup = await workflow.cleanup_pending_media()
            assert cleanup["cleaned"] >= 1
            old_media = await repository.label_studio_media_upload(editing.revision)
            assert old_media is not None and old_media["state"] == "deleted"
            assert (editing_task["label_studio_task_id"], editing_task["label_studio_file_upload_id"]) in label_studio.deleted
            assert (
                replacement_task["label_studio_task_id"],
                replacement_task["label_studio_file_upload_id"],
            ) not in label_studio.deleted
            assert replacement_task["label_studio_task_id"] in label_studio.tasks

            label_studio.tasks[replacement_task["label_studio_task_id"]]["annotations"] = [
                {
                    "id": 2,
                    "was_cancelled": False,
                    "completed_by": {"id": 12, "email": "reviewer2@example.invalid"},
                    "created_at": "2026-10-05T01:05:00Z",
                    "result": [{"from_name": "bbox", "value": {"x": 2}}],
                }
            ]
            replacement_snapshot = await workflow.finalize_annotation(str(sample_id), replacement.revision)
            previous_snapshot = await repository.finalized_revision(first.revision)
            assert previous_snapshot["annotation_revision_id"] == first_snapshot["annotation_revision_id"]
            assert previous_snapshot["result_sha256"] == first_snapshot["result_sha256"]
            assert previous_snapshot["result"] == first_snapshot["result"]
            assert replacement_snapshot["annotation_revision_id"] != first_snapshot["annotation_revision_id"]
            assert await repository.assignment_state(replacement.revision) == "finalized"
            assert await repository.label_studio_media_upload(replacement.revision) is not None
            assert await ingestion.storage_bytes() == initial_object_bytes + len(frame_bytes)
        finally:
            await repository.close()
            await ingestion.close()

    asyncio.run(exercise())
