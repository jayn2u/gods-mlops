"""Canonical immutable dataset manifest serialization and source identity."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Serialize a JSON value with the byte ordering used for every pinned hash."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def dataset_code_sha256() -> str:
    """Hash the code that snapshots annotations, splits, and publishes data."""
    package = Path(__file__).resolve().parents[1]
    relevant = [
        *sorted((package / "datasets").glob("*.py")),
        package / "annotations" / "storage.py",
        package / "ingestion" / "storage.py",
    ]
    digest = sha256()
    for path in relevant:
        if not path.is_file():
            continue
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
