"""Immutable input and model-version checks for every runner invocation."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from gods_mlops.model_locks import LockValidationError, load_model_lock
from gods_mlops.datasets.manifest import canonical_json

_MODEL_IDS = {
    "detr": "PekingU/rtdetr_v2_r18vd",
    "clip": "openai/clip-vit-base-patch16",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}


def model_lock_path() -> Path:
    configured = __import__("os").environ.get("GODS_MLOPS_MODEL_LOCK")
    return Path(configured) if configured else Path(__file__).resolve().parents[3] / "models" / "lock.json"


def locked_model(model_kind: str, *, lock_path: Path | None = None):
    try:
        model_id = _MODEL_IDS[model_kind]
    except KeyError as error:
        raise ValueError("model_kind must be detr, clip, or qwen") from error
    lock = load_model_lock(lock_path or model_lock_path())
    for model in lock.models:
        if model.model_id == model_id:
            return model
    raise LockValidationError(f"model is not present in the immutable lock: {model_id}")


def validate_model_revision(config: dict[str, Any], *, lock_path: Path | None = None) -> dict[str, str]:
    model = locked_model(str(config.get("model_kind")), lock_path=lock_path)
    if config.get("model_id") != model.model_id or config.get("model_revision") != model.revision:
        raise ValueError("runner model ID or revision differs from the immutable model lock")
    return {"model_id": model.model_id, "model_revision": model.revision}


def validate_manifest_identity(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Reject a manifest whose immutable input, model, phase, or config differs from its job."""
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise ValueError("runner config and manifest must be JSON objects")
    validate_model_revision(config)
    phase = str(config.get("phase", ""))
    model_kind = str(config.get("model_kind", ""))
    if model_kind not in _MODEL_IDS:
        raise ValueError("runner model_kind must be detr, clip, or qwen")
    if not config.get("config_version"):
        raise ValueError("runner config version is missing")
    expected_sha = str(config.get("input_sha256", ""))
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("runner input hash must be a lowercase SHA-256 digest")
    dataset_version = config.get("dataset_version")
    input_id = str(config.get("input_id", ""))
    input_kind = str(config.get("input_kind", ""))
    if phase == "training":
        if input_kind != "dataset_version" or not dataset_version or str(dataset_version) != input_id:
            raise ValueError("training requires a published dataset version")
        if manifest.get("dataset_version") != input_id or manifest.get("schema_version") != 1:
            raise ValueError("manifest dataset version does not match the immutable job")
        if sha256(canonical_json(manifest)).hexdigest() != expected_sha:
            raise ValueError("manifest content hash does not match the immutable job")
        if manifest.get("schema_version") != 1 or manifest.get("training", {}).get("ready") is not True:
            raise ValueError("training manifest is not training-ready")
        if manifest.get("target") not in {model_kind, "both"}:
            raise ValueError("published dataset target does not include this model")
    elif phase == "evaluation":
        if input_kind != "dataset_version" or not dataset_version or str(dataset_version) != input_id:
            raise ValueError("evaluation requires a published dataset version")
        if manifest.get("dataset_version") != input_id or manifest.get("schema_version") != 1:
            raise ValueError("manifest dataset version does not match the immutable evaluation job")
        if sha256(canonical_json(manifest)).hexdigest() != expected_sha:
            raise ValueError("evaluation manifest content hash does not match the immutable job")
        if manifest.get("training", {}).get("ready") is not True:
            raise ValueError("evaluation manifest was not published training-ready")
        if manifest.get("target") not in {model_kind, "both"}:
            raise ValueError("published dataset target does not include this evaluation model")
        if manifest.get("evaluation", {}).get("eligible") is not True:
            raise ValueError("evaluation manifest was published with insufficient evaluation inputs")
    elif phase == "preparation":
        if input_kind != "annotation_batch" or dataset_version is not None:
            raise ValueError("preparation requires a pre-publication annotation batch")
        if manifest.get("batch_id") != input_id or manifest.get("input_id") != input_id:
            raise ValueError("preparation batch identity does not match the immutable job")
        if str(manifest.get("input_sha256", "")).strip() != expected_sha:
            raise ValueError("preparation input hash does not match the immutable job")
        items = manifest.get("items")
        expected_kind = "frame" if model_kind == "detr" else "crop" if model_kind == "qwen" else None
        if expected_kind is None or not isinstance(items, list) or not items:
            raise ValueError("preparation must use detector frames or caption crops")
        if any(not isinstance(item, dict) or item.get("item_kind") != expected_kind for item in items):
            raise ValueError(f"{model_kind} preparation requires immutable {expected_kind} inputs")
        batch_payload = {"schema": "annotation-preparation-input-v1", "items": items}
        if sha256(canonical_json(batch_payload)).hexdigest() != expected_sha:
            raise ValueError("preparation batch content hash does not match the immutable job")
    elif phase == "probe":
        if input_kind != "probe_input" or dataset_version is not None:
            raise ValueError("model readiness probe requires a separate immutable probe input")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("fixture") is not True
            or manifest.get("phase") != phase
            or manifest.get("model_kind") != model_kind
            or manifest.get("input_kind") != input_kind
            or manifest.get("input_id") != input_id
            or manifest.get("config_version") != config.get("config_version")
        ):
            raise ValueError("probe manifest must explicitly identify its synthetic fixture")
        if sha256(canonical_json(manifest)).hexdigest() != expected_sha:
            raise ValueError("probe manifest content hash does not match the immutable job")
    else:
        raise ValueError("runner phase must be probe, preparation, or training")
    manifest_model = manifest.get("model_kind")
    if manifest_model is not None and manifest_model != model_kind:
        raise ValueError("manifest model kind does not match the immutable job")
    manifest_config_version = manifest.get("config_version")
    if (
        phase != "evaluation"
        and manifest_config_version is not None
        and manifest_config_version != config.get("config_version")
    ):
        raise ValueError("manifest config version does not match the immutable job")
    return {
        "phase": phase,
        "model_kind": model_kind,
        "input_kind": input_kind,
        "input_id": input_id,
        "input_sha256": expected_sha,
        "dataset_version": str(dataset_version) if dataset_version is not None else None,
        "config_version": str(config["config_version"]),
        "model_id": str(config["model_id"]),
        "model_revision": str(config["model_revision"]),
    }


def load_manifest(manifest_uri: str, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    if manifest_uri.startswith("s3://"):
        bucket, _, key = manifest_uri[5:].partition("/")
        if not bucket or not key:
            raise ValueError("S3 manifest URI must contain a bucket and object key")
        if config is None:
            raise ValueError("S3 manifest reads require the immutable job input identity")
        expected_sha = config.get("input_sha256")
        expected_size = config.get("manifest_size_bytes")
        if not isinstance(expected_sha, str) or not isinstance(expected_size, int) or expected_size <= 0:
            raise ValueError("S3 manifest job identity must include its frozen hash and size")
        from .data import dataset_object_store_from_environment

        if bucket != __import__("os").environ.get("GODS_MLOPS_S3_BUCKET"):
            raise ValueError("S3 manifest bucket differs from the configured immutable object store")
        payload = dataset_object_store_from_environment().read_source(
            object_key=key,
            sha256_digest=expected_sha,
            size_bytes=expected_size,
        )
    else:
        path = Path(manifest_uri.removeprefix("file://"))
        payload = path.read_bytes()
    try:
        result = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest is not valid UTF-8 JSON") from error
    if not isinstance(result, dict):
        raise ValueError("manifest must contain a JSON object")
    return result
