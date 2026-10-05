"""Label Studio Pod sidecar for verified cleanup of temporary review uploads."""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

_MAX_REVIEW_MEDIA_BYTES = 20 * 1024 * 1024


class CleanupRequest(BaseModel):
    project_id: int = Field(gt=0)
    file_upload_id: int = Field(gt=0)
    upload_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=_MAX_REVIEW_MEDIA_BYTES)


def delete_review_upload(
    *,
    upload_root: Path,
    project_id: int,
    upload_path: str,
    expected_sha256: str,
    expected_size_bytes: int,
) -> bool:
    """Delete only a hash-matched regular upload under the requested project directory."""
    parsed = urlsplit(upload_path)
    path = Path(parsed.path)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or not path.is_absolute()
        or len(path.parts) != 5
        or path.parts[:4] != ("/", "data", "upload", str(project_id))
    ):
        raise ValueError("upload path must be one Label Studio /data/upload project file")
    filename = path.parts[4]
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("upload path must contain one regular filename")

    root = upload_root.resolve(strict=True)
    project_dir = root / str(project_id)
    if project_dir.is_symlink():
        raise ValueError("Label Studio project upload directory must not be a symlink")
    candidate = project_dir / filename
    if candidate.is_symlink():
        raise ValueError("Label Studio upload must not be a symlink")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root) or resolved.parent != project_dir.resolve(strict=False):
        raise ValueError("Label Studio upload path escaped its project directory")
    if not candidate.exists():
        return False
    if not candidate.is_file():
        raise ValueError("Label Studio upload is not a regular file")

    digest = hashlib.sha256()
    actual_size = 0
    with candidate.open("rb") as media:
        while chunk := media.read(1024 * 1024):
            actual_size += len(chunk)
            if actual_size > expected_size_bytes or actual_size > _MAX_REVIEW_MEDIA_BYTES:
                raise ValueError("Label Studio upload size did not match its reserved size")
            digest.update(chunk)
    if actual_size != expected_size_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError("Label Studio upload does not match its reserved size and SHA-256")

    candidate.unlink()
    directory_fd = os.open(project_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return True


def create_cleanup_app(*, upload_root: Path, token: str) -> FastAPI:
    if len(token) < 32:
        raise ValueError("cleanup service token must contain at least 32 characters")
    if upload_root.is_symlink():
        raise ValueError("Label Studio upload root must not be a symlink")
    upload_root.mkdir(parents=True, exist_ok=True)
    root = upload_root.resolve(strict=True)
    app = FastAPI(title="Gods Label Studio media cleanup")

    @app.get("/readyz")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/internal/review-media/delete")
    async def delete(request: CleanupRequest, authorization: str | None = Header(default=None)) -> dict[str, bool]:
        expected = f"Bearer {token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail={"code": "invalid_cleanup_token"})
        try:
            removed = delete_review_upload(
                upload_root=root,
                project_id=request.project_id,
                upload_path=request.upload_path,
                expected_sha256=request.sha256,
                expected_size_bytes=request.size_bytes,
            )
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=409, detail={"code": "media_cleanup_conflict"}) from error
        return {"deleted": removed, "already_absent": not removed}

    return app


def create_application() -> FastAPI:
    root = os.environ.get("GODS_MLOPS_LABEL_STUDIO_UPLOAD_ROOT", "").strip()
    token = os.environ.get("GODS_MLOPS_LABEL_MEDIA_CLEANUP_TOKEN", "").strip()
    if not root:
        raise ValueError("required setting is missing: GODS_MLOPS_LABEL_STUDIO_UPLOAD_ROOT")
    if not token:
        raise ValueError("required setting is missing: GODS_MLOPS_LABEL_MEDIA_CLEANUP_TOKEN")
    return create_cleanup_app(upload_root=Path(root), token=token)


def main() -> None:
    import uvicorn

    uvicorn.run(create_application(), host="0.0.0.0", port=8090, access_log=False)


if __name__ == "__main__":
    main()
