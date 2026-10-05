from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest
from PIL import Image

from gods_mlops.annotations.label_studio import LabelStudioApiClient, LabelStudioMediaCleanupClient


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
