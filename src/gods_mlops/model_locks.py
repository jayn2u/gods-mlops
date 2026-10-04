"""Model lock loading and local artifact validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from pathlib import PurePosixPath
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass


class LockValidationError(ValueError):
    """A lock file or prepared model cache does not satisfy its contract."""


@dataclass(frozen=True)
class ModelFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class ModelLock:
    model_id: str
    revision: str
    files: tuple[ModelFile, ...]


@dataclass(frozen=True)
class ModelLockSet:
    models: tuple[ModelLock, ...]


def load_model_lock(path: Path) -> ModelLockSet:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LockValidationError(f"cannot read model lock {path}: {exc}") from exc

    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise LockValidationError("model lock must declare schema_version 1")
    raw_models = document.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise LockValidationError("model lock must contain at least one model")

    models: list[ModelLock] = []
    seen_ids: set[str] = set()
    for item in raw_models:
        if not isinstance(item, dict):
            raise LockValidationError("each model entry must be an object")
        model_id = item.get("model_id")
        if not isinstance(model_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", model_id
        ) is None:
            raise LockValidationError(f"model_id must be a safe Hugging Face owner/name: {model_id!r}")
        if model_id in seen_ids:
            raise LockValidationError(f"duplicate model_id: {model_id}")
        seen_ids.add(model_id)

        revision = item.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise LockValidationError(
                f"{model_id} must use an immutable 40-character revision commit, got {revision!r}"
            )

        raw_files = item.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise LockValidationError(f"{model_id} must lock at least one file")
        files: list[ModelFile] = []
        seen_paths: set[str] = set()
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise LockValidationError(f"{model_id} file entries must be objects")
            relative_path = raw_file.get("path")
            if not isinstance(relative_path, str) or not _is_safe_relative_path(relative_path):
                raise LockValidationError(f"{model_id} has unsafe file path: {relative_path!r}")
            if relative_path in seen_paths:
                raise LockValidationError(f"{model_id} has duplicate file path: {relative_path}")
            seen_paths.add(relative_path)
            digest = raw_file.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise LockValidationError(f"{model_id}:{relative_path} must have a SHA-256 digest")
            files.append(ModelFile(path=relative_path, sha256=digest))
        models.append(ModelLock(model_id=model_id, revision=revision, files=tuple(files)))

    return ModelLockSet(models=tuple(models))


def validate_model_cache(model: ModelLock, model_dir: Path) -> None:
    """Raise when any locked artifact is missing or has the wrong digest."""
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise LockValidationError(f"model cache directory is missing or unsafe: {model_dir}")

    root = model_dir.resolve(strict=True)
    for locked_file in model.files:
        candidate = root.joinpath(*PurePosixPath(locked_file.path).parts)
        if not candidate.exists():
            raise LockValidationError(f"missing required file: {locked_file.path}")
        if candidate.is_symlink() or not candidate.is_file():
            raise LockValidationError(f"required file is not a regular local file: {locked_file.path}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise LockValidationError(f"required file escapes model cache: {locked_file.path}")
        actual_digest = _sha256_file(candidate)
        if actual_digest != locked_file.sha256:
            raise LockValidationError(
                f"SHA-256 mismatch for {locked_file.path}: expected {locked_file.sha256}, got {actual_digest}"
            )


def model_cache_path(cache_root: Path, model: ModelLock) -> Path:
    """Return the immutable cache location for a model revision."""
    return cache_root / "snapshots" / model.model_id / model.revision


def prepare_model(
    model: ModelLock,
    cache_root: Path,
    *,
    downloader: Callable[[ModelLock, Path], None] | None = None,
) -> Path:
    """Download to a private staging directory and publish only after digest checks."""
    target = model_cache_path(cache_root, model)
    if target.exists() or target.is_symlink():
        validate_model_cache(model, target)
        return target

    staging_root = cache_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f"{model.revision[:8]}-", dir=staging_root))
    try:
        if downloader is None:
            _download_from_hugging_face(model, staging_dir)
        else:
            downloader(model, staging_dir)
        validate_model_cache(model, staging_dir)

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(staging_dir, target)
        except OSError:
            if target.exists() or target.is_symlink():
                validate_model_cache(model, target)
                return target
            raise
        validate_model_cache(model, target)
        return target
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


def validate_all_model_caches(lock: ModelLockSet, cache_root: Path) -> None:
    """Require every locked model revision to be complete before declaring readiness."""
    for model in lock.models:
        validate_model_cache(model, model_cache_path(cache_root, model))


def _is_safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return bool(path.parts) and not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _download_from_hugging_face(model: ModelLock, staging_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model.model_id,
        revision=model.revision,
        local_dir=staging_dir,
        allow_patterns=[file.path for file in model.files],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
