"""Source and model provenance derived inside the bound evaluation worker."""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path
from typing import Any

_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVALUATOR_FILES = (
    "evaluation/detection.py",
    "evaluation/eligibility.py",
    "evaluation/provenance.py",
    "evaluation/report.py",
    "evaluation/retrieval.py",
    "training/clip.py",
    "training/detector.py",
    "training/worker.py",
)


def evaluator_source_revision() -> str:
    """Return a deterministic digest of the source files that produce evaluation artifacts."""
    package_root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for relative in _EVALUATOR_FILES:
        source_path = package_root / relative
        payload = source_path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def worker_image_digest(image_reference: str) -> str:
    """Extract only the immutable digest from a pinned worker image reference."""
    if not isinstance(image_reference, str) or "@" not in image_reference:
        raise ValueError("evaluation worker image must be pinned by an immutable digest")
    _name, _separator, digest = image_reference.rpartition("@")
    if not _name or not _IMAGE_DIGEST.fullmatch(digest):
        raise ValueError("evaluation worker image must be pinned by an immutable digest")
    return digest


def build_owned_evaluation_provenance(
    *,
    worker_image_id: str,
    model_revision: str,
    model_kind: str,
    prediction_payload: dict[str, Any],
) -> dict[str, Any]:
    """Bind report provenance to the verified worker claim, locked model and runner output."""
    if not isinstance(worker_image_id, str) or not _IMAGE_DIGEST.fullmatch(worker_image_id):
        raise ValueError("owned evaluation worker image identity is invalid")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("owned evaluation model revision is unavailable")
    if prediction_payload.get("model_revision") != model_revision:
        raise ValueError("evaluation prediction model revision differs from the verified model")
    provenance: dict[str, Any] = {
        "evaluator_revision": evaluator_source_revision(),
        "worker_image_id": worker_image_id,
        "model_revision": model_revision,
    }
    if model_kind == "detr":
        provenance["person_class_mapping"] = _person_class_mapping(
            prediction_payload.get("person_class_mapping"), expected_model_revision=model_revision
        )
    elif model_kind != "clip":
        raise ValueError("owned evaluation provenance supports DETR or CLIP")
    return provenance


def _person_class_mapping(value: Any, *, expected_model_revision: str) -> dict[str, Any]:
    fields = {
        "model_revision",
        "model_class_index",
        "model_class_name",
        "coco_category_id",
        "coco_category_name",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("DETR person class mapping provenance is missing or unsupported")
    if (
        value.get("model_revision") != expected_model_revision
        or type(value.get("model_class_index")) is not int
        or value["model_class_index"] < 0
        or value.get("model_class_name") != "person"
        or type(value.get("coco_category_id")) is not int
        or value["coco_category_id"] != 1
        or value.get("coco_category_name") != "person"
    ):
        raise ValueError("DETR person class mapping must bind the verified model class to COCO person")
    return dict(value)
