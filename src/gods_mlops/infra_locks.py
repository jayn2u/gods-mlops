"""Validation for immutable OCI image lock entries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from gods_mlops.model_locks import LockValidationError


@dataclass(frozen=True)
class ImageLock:
    name: str
    source: str
    tag: str
    digest: str
    license: str | None = None
    source_repository: str | None = None


@dataclass(frozen=True)
class ImageLockSet:
    images: tuple[ImageLock, ...]


def load_image_lock(path: Path) -> ImageLockSet:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise LockValidationError(f"cannot read image lock {path}: {exc}") from exc

    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise LockValidationError("image lock must declare schema_version 1")
    raw_images = document.get("images")
    if not isinstance(raw_images, list) or not raw_images:
        raise LockValidationError("image lock must contain at least one image")

    images: list[ImageLock] = []
    seen_names: set[str] = set()
    for item in raw_images:
        if not isinstance(item, dict):
            raise LockValidationError("each image entry must be an object")
        name = item.get("name")
        source = item.get("source")
        tag = item.get("tag")
        digest = item.get("digest")
        if not isinstance(name, str) or not name.strip() or name in seen_names:
            raise LockValidationError(f"image name must be non-empty and unique: {name!r}")
        if not isinstance(source, str) or not source.strip() or "@" in source or any(c.isspace() for c in source):
            raise LockValidationError(f"{name} must have an image source without an embedded digest")
        if not isinstance(tag, str) or not tag.strip() or any(c.isspace() for c in tag):
            raise LockValidationError(f"{name} must have an image tag")
        if not isinstance(digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise LockValidationError(f"{name} must have an immutable sha256 digest")
        license_name = item.get("license")
        source_repository = item.get("source_repository")
        if license_name is not None and not isinstance(license_name, str):
            raise LockValidationError(f"{name} license must be a string")
        if source_repository is not None and not isinstance(source_repository, str):
            raise LockValidationError(f"{name} source_repository must be a string")
        seen_names.add(name)
        images.append(
            ImageLock(
                name=name,
                source=source,
                tag=tag,
                digest=digest,
                license=license_name,
                source_repository=source_repository,
            )
        )
    return ImageLockSet(images=tuple(images))
