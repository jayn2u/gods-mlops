"""Label Studio task submission semantics and HTTP client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import re
from typing import Any, Protocol
from urllib import error, request
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

from .models import AnnotationNotSubmittedError


class LabelStudioTaskReader(Protocol):
    async def get_task(self, task_id: int) -> dict: ...


@dataclass(frozen=True, slots=True)
class LabelStudioTaskReference:
    project_id: int
    task_id: int
    file_upload_id: int
    media_path: str
    filename: str


def submitted_annotation(task: dict) -> dict:
    """Return the latest completed human annotation, never a read-only prediction."""
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        raise AnnotationNotSubmittedError("Label Studio task contains no submitted annotation")

    submitted = [
        annotation
        for annotation in annotations
        if isinstance(annotation, dict)
        and isinstance(annotation.get("id"), int)
        and annotation.get("was_cancelled") is not True
        and isinstance(annotation.get("result"), list)
    ]
    if not submitted:
        raise AnnotationNotSubmittedError("Label Studio task contains no submitted annotation")
    return max(
        submitted,
        key=lambda annotation: str(annotation.get("updated_at") or annotation.get("created_at") or ""),
    )


def _safe_image_filename(filename: str) -> str:
    if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}\.(jpg|jpeg|png)", filename):
        raise ValueError("Label Studio media filename must be a simple JPEG or PNG filename")
    return filename


def _multipart_file_body(*, boundary: str, filename: str, image: bytes) -> bytes:
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode("ascii")
    return prefix + image + f"\r\n--{boundary}--\r\n".encode("ascii")


def _positive_id(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"Label Studio {name} ID was missing or invalid")
    return value


def _uploaded_media_path(task: dict, project_id: int) -> str:
    data = task.get("data")
    value = data.get("image") if isinstance(data, dict) else None
    if not isinstance(value, str):
        raise RuntimeError("Label Studio uploaded task has no image field")
    parsed = urlsplit(value)
    path = PurePosixPath(parsed.path)
    if parsed.query or parsed.fragment or path.parts[:4] != ("/", "data", "upload", str(project_id)):
        raise RuntimeError("Label Studio task media path is outside its project upload directory")
    if len(path.parts) != 5 or any(part in {".", ".."} for part in path.parts):
        raise RuntimeError("Label Studio task media path is not a single uploaded filename")
    return str(path)


def _file_upload_id(task: dict) -> int | None:
    value = task.get("file_upload_id", task.get("file_upload"))
    if isinstance(value, dict):
        value = value.get("id")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _result_items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("results", "tasks", "files"):
            result = value.get(key)
            if isinstance(result, list):
                return result
    return []


def _has_more_results(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("next"))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class LabelStudioApiClient:
    """CE REST adapter; Bearer mode stores a refresh token and never persists access tokens."""

    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        timeout_seconds: float = 8.0,
        auth_scheme: str = "Bearer",
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Label Studio base URL must be an HTTP(S) origin without embedded credentials")
        if not api_token.strip():
            raise ValueError("Label Studio API token must not be empty")
        if auth_scheme not in {"Bearer", "Token"}:
            raise ValueError("Label Studio API authentication scheme must be Bearer or Token")
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._auth_scheme = auth_scheme
        self._timeout_seconds = timeout_seconds

    async def get_task(self, task_id: int) -> dict:
        if task_id <= 0:
            raise ValueError("Label Studio task ID must be positive")
        return await asyncio.to_thread(self._get_task, task_id)

    async def create_project(self, *, title: str, label_config: str) -> int:
        if not title.strip() or not label_config.strip():
            raise ValueError("Label Studio project requires a title and label configuration")
        result = await asyncio.to_thread(
            self._request_json,
            "/api/projects/",
            method="POST",
            payload={"title": title, "label_config": label_config},
        )
        if not isinstance(result, dict) or not isinstance(result.get("id"), int):
            raise RuntimeError("Label Studio project response did not contain its ID")
        return result["id"]

    async def import_media_task(
        self,
        *,
        project_id: int,
        filename: str,
        image: bytes,
        prediction: dict | None = None,
    ) -> LabelStudioTaskReference:
        """Upload one assigned image and import its prediction as a read-only result."""
        if project_id <= 0 or not image:
            raise ValueError("project ID and media bytes are required")
        safe_filename = _safe_image_filename(filename)
        existing = await asyncio.to_thread(
            self._find_existing_task,
            project_id,
            safe_filename,
        )
        if existing is not None:
            if prediction is not None:
                await asyncio.to_thread(self._ensure_prediction, project_id, existing.task_id, prediction)
            return existing

        boundary = f"gods-label-studio-{uuid4().hex}"
        body = _multipart_file_body(boundary=boundary, filename=safe_filename, image=image)
        query = urlencode({"return_task_ids": "true", "commit_to_project": "true"})
        response = await asyncio.to_thread(
            self._request_json,
            f"/api/projects/{project_id}/import?{query}",
            method="POST",
            body=body,
            content_type=f"multipart/form-data; boundary={boundary}",
        )
        if not isinstance(response, dict):
            raise RuntimeError("Label Studio import response was not an object")
        task_ids = response.get("task_ids")
        upload_ids = response.get("file_upload_ids")
        if response.get("task_count") != 1 or not isinstance(task_ids, list) or len(task_ids) != 1:
            raise RuntimeError("Label Studio must import exactly one task for one review image")
        if not isinstance(upload_ids, list) or len(upload_ids) != 1:
            raise RuntimeError("Label Studio did not return exactly one uploaded media ID")
        task_id = _positive_id(task_ids[0], "task")
        file_upload_id = _positive_id(upload_ids[0], "uploaded file")
        task = await self.get_task(task_id)
        media_path = _uploaded_media_path(task, project_id)
        reference = LabelStudioTaskReference(
            project_id=project_id,
            task_id=task_id,
            file_upload_id=file_upload_id,
            media_path=media_path,
            filename=safe_filename,
        )
        if prediction is not None:
            await asyncio.to_thread(self._ensure_prediction, project_id, task_id, prediction)
        return reference

    async def delete_review_task(self, *, project_id: int, task_id: int, file_upload_id: int) -> None:
        if project_id <= 0 or task_id <= 0 or file_upload_id <= 0:
            raise ValueError("Label Studio project, task, and file IDs must be positive")
        await asyncio.to_thread(self._delete_task, task_id)
        await asyncio.to_thread(
            self._request_json,
            f"/api/projects/{project_id}/file-uploads",
            method="DELETE",
            payload={"file_upload_ids": [file_upload_id]},
        )

    def _get_task(self, task_id: int) -> dict:
        result = self._request_json(f"/api/tasks/{task_id}/", method="GET")
        if not isinstance(result, dict):
            raise RuntimeError("Label Studio task response was not a JSON object")
        return result

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        payload: dict | list | None = None,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> Any:
        token = self._api_token
        if self._auth_scheme == "Bearer":
            token = self._refresh_access_token()
        return self._send_json(
            path,
            method=method,
            payload=payload,
            body=body,
            content_type=content_type,
            auth_scheme=self._auth_scheme,
            token=token,
        )

    def _refresh_access_token(self) -> str:
        response = self._send_json(
            "/api/token/refresh/",
            method="POST",
            payload={"refresh": self._api_token},
            auth_scheme="Bearer",
            token=self._api_token,
        )
        access_token = response.get("access") if isinstance(response, dict) else None
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError("Label Studio refresh endpoint did not return an access token")
        return access_token

    def _send_json(
        self,
        path: str,
        *,
        method: str,
        payload: dict | list | None = None,
        body: bytes | None = None,
        content_type: str | None = None,
        auth_scheme: str,
        token: str,
    ) -> Any:
        headers = {"Authorization": f"{auth_scheme} {token}", "Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        api_request = request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(api_request, timeout=self._timeout_seconds) as response:
                response_body = response.read(5 * 1024 * 1024 + 1)
        except error.HTTPError as exception:
            raise RuntimeError(f"Label Studio request failed with HTTP {exception.code}") from exception
        except error.URLError as exception:
            raise RuntimeError("Label Studio request failed before a response was received") from exception
        if len(response_body) > 5 * 1024 * 1024:
            raise RuntimeError("Label Studio response exceeded the 5 MiB limit")
        if not response_body:
            return {}
        try:
            return json.loads(response_body)
        except json.JSONDecodeError as exception:
            raise RuntimeError("Label Studio response was not valid JSON") from exception

    def _ensure_prediction(self, project_id: int, task_id: int, prediction: dict) -> None:
        if not isinstance(prediction, dict) or not isinstance(prediction.get("result"), list):
            raise ValueError("pre-annotation must contain a result array")
        task = self._get_task(task_id)
        result_json = _canonical_json(prediction["result"])
        model_version = prediction.get("model_version", "gods-mlops-unknown")
        for existing in task.get("predictions", []):
            if (
                isinstance(existing, dict)
                and existing.get("model_version") == model_version
                and _canonical_json(existing.get("result", [])) == result_json
            ):
                return
        prediction_item = {
            "task": task_id,
            "model_version": model_version,
            "result": prediction["result"],
        }
        self._request_json(
            f"/api/projects/{project_id}/import/predictions",
            method="POST",
            payload=[prediction_item],
        )

    def _delete_task(self, task_id: int) -> None:
        try:
            self._request_json(f"/api/tasks/{task_id}/", method="DELETE")
        except RuntimeError as error:
            if "HTTP 404" not in str(error):
                raise

    def _find_existing_task(self, project_id: int, filename: str) -> LabelStudioTaskReference | None:
        files = self._request_json(
            f"/api/projects/{project_id}/file-uploads?all=true",
            method="GET",
        )
        file_items = _result_items(files)
        upload_ids = {
            _positive_id(item["id"], "uploaded file")
            for item in file_items
            if filename in str(item.get("file") or item.get("url") or "")
        }
        if not upload_ids:
            return None
        if len(upload_ids) > 1:
            raise RuntimeError("multiple Label Studio uploads match one review assignment marker")

        for page in range(1, 51):
            query = urlencode({"project": project_id, "page": page, "page_size": 100, "ordering": "-id"})
            tasks = self._request_json(f"/api/tasks/?{query}", method="GET")
            for task in _result_items(tasks):
                if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
                    continue
                image_value = str(task["data"].get("image", ""))
                file_id = _file_upload_id(task)
                if filename not in image_value and file_id not in upload_ids:
                    continue
                task_id = _positive_id(task.get("id"), "task")
                detail = self._get_task(task_id)
                image_path = _uploaded_media_path(detail, project_id)
                matched_file_id = _file_upload_id(detail) or file_id
                if matched_file_id not in upload_ids:
                    matched_file_id = next(iter(upload_ids))
                return LabelStudioTaskReference(
                    project_id=project_id,
                    task_id=task_id,
                    file_upload_id=matched_file_id,
                    media_path=image_path,
                    filename=filename,
                )
            if not _has_more_results(tasks):
                break
        raise RuntimeError("Label Studio retained the review image but the matching task could not be reconciled")


class LabelStudioMediaCleanupClient:
    """Call the Label Studio-pod cleanup sidecar with a header-only credential."""

    def __init__(self, *, base_url: str, token: str, timeout_seconds: float = 8.0) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("cleanup base URL must be an HTTP(S) origin without embedded credentials")
        if len(token) < 32:
            raise ValueError("cleanup service token must contain at least 32 characters")
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout_seconds = timeout_seconds

    async def delete_upload(
        self,
        *,
        project_id: int,
        file_upload_id: int,
        upload_path: str,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> dict[str, bool]:
        payload = json.dumps(
            {
                "project_id": project_id,
                "file_upload_id": file_upload_id,
                "upload_path": upload_path,
                "sha256": expected_sha256,
                "size_bytes": expected_size_bytes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        api_request = request.Request(
            f"{self._base_url}/internal/review-media/delete",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        return await asyncio.to_thread(self._delete_upload, api_request)

    def _delete_upload(self, api_request: request.Request) -> dict[str, bool]:
        try:
            with request.urlopen(api_request, timeout=self._timeout_seconds) as response:
                payload = response.read(64 * 1024 + 1)
        except error.HTTPError as exception:
            raise RuntimeError(f"Label Studio media cleanup failed with HTTP {exception.code}") from exception
        except error.URLError as exception:
            raise RuntimeError("Label Studio media cleanup failed before a response was received") from exception
        if len(payload) > 64 * 1024:
            raise RuntimeError("Label Studio cleanup response exceeded the 64 KiB limit")
        try:
            response = json.loads(payload)
        except json.JSONDecodeError as exception:
            raise RuntimeError("Label Studio cleanup response was not valid JSON") from exception
        if not isinstance(response, dict) or not response.get("deleted") and not response.get("already_absent"):
            raise RuntimeError("Label Studio cleanup did not confirm the upload is absent")
        return {"deleted": bool(response.get("deleted")), "already_absent": bool(response.get("already_absent"))}
