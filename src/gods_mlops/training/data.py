"""Read frozen Task 5/6 media references with their committed content hashes."""

from __future__ import annotations

import os
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any


class TrainingSourceIntegrityError(ValueError):
    """Frozen media bytes do not match their Task 5/6 or probe identity."""


def read_verified_media(item: dict[str, Any], *, object_store: Any = None) -> bytes:
    """Read one local fixture or immutable object and verify both size and SHA-256."""
    if not isinstance(item, dict):
        raise ValueError("training media item must be an object")
    image_path = item.get("image_path")
    if isinstance(image_path, str):
        payload = Path(image_path).read_bytes()
        expected_sha = item.get("image_sha256", item.get("sha256"))
        expected_size = item.get("image_size_bytes", item.get("size_bytes"))
    else:
        ref = item.get("object") if isinstance(item.get("object"), dict) else item
        object_key = ref.get("key", ref.get("object_key"))
        expected_sha = ref.get("sha256", item.get("source_sha256"))
        expected_size = ref.get("size_bytes", item.get("object_size_bytes"))
        if not isinstance(object_key, str) or not object_key:
            raise ValueError("training media item has no frozen object key")
        if object_store is None:
            raise ValueError("training media object store is not configured")
        if not isinstance(expected_sha, str) or not isinstance(expected_size, int):
            raise ValueError("training media object reference lacks frozen hash or size")
        payload = object_store.read_source(
            object_key=object_key,
            sha256_digest=expected_sha,
            size_bytes=expected_size,
        )
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("training media SHA-256 is missing or malformed")
    digest = sha256(payload).hexdigest()
    if digest != expected_sha:
        raise TrainingSourceIntegrityError("training media SHA-256 differs from its immutable manifest")
    if expected_size is not None and len(payload) != int(expected_size):
        raise TrainingSourceIntegrityError("training media byte count differs from its immutable manifest")
    return payload


def load_rgb_image(item: dict[str, Any], *, object_store: Any = None):
    from PIL import Image, UnidentifiedImageError

    payload = read_verified_media(item, object_store=object_store)
    try:
        image = Image.open(BytesIO(payload))
        image.verify()
        image = Image.open(BytesIO(payload)).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise TrainingSourceIntegrityError("training media is not a readable image") from error
    return image


def dataset_object_store_from_environment():
    from gods_mlops.datasets.publish import DatasetObjectStore

    settings = {
        "endpoint_url": os.environ.get("GODS_MLOPS_S3_ENDPOINT_URL"),
        "access_key": os.environ.get("GODS_MLOPS_S3_ACCESS_KEY"),
        "secret_key": os.environ.get("GODS_MLOPS_S3_SECRET_KEY"),
        "bucket": os.environ.get("GODS_MLOPS_S3_BUCKET"),
        "region": os.environ.get("GODS_MLOPS_S3_REGION", "us-east-1"),
    }
    if any(not settings[key] for key in ("endpoint_url", "access_key", "secret_key", "bucket")):
        raise ValueError("training S3 object store settings are incomplete")
    return DatasetObjectStore(**settings)
