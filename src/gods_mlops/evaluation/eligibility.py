"""Fail-closed dataset, checkpoint, and product-release eligibility checks."""

from __future__ import annotations

import io
import math
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_REQUEST_FIELDS = {
    "dataset_version",
    "manifest_sha256",
    "training_job_id",
    "checkpoint_sha256",
    "model_kind",
    "evaluation_config_version",
    "evaluation_split",
    "baseline",
}
_PRODUCT_GATE_REQUIREMENTS = [
    "trusted Gods Watching quality policy matches the exact evaluator, baseline, CUHK dataset, and protocol",
    "the imported package pins a CUHK held-out test report to the exact candidate weights and metric definition",
    "verified package evidence binds baseline identity, package hash, product-case hash, and evaluator revision",
    "real product crop-search text and image Recall@5 are each at least 0.8 and strictly exceed baseline",
]


def validate_evaluation_request(
    manifest_uri: str,
    checkpoint_uri: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate the explicit immutable inputs accepted by public ``evaluate``."""
    manifest_bucket, manifest_key = _s3_object_uri(manifest_uri, "manifest")
    checkpoint_bucket, checkpoint_key = _s3_object_uri(checkpoint_uri, "checkpoint")
    if manifest_bucket != checkpoint_bucket:
        raise ValueError("manifest and checkpoint must use the same trusted S3 bucket")
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be an object")
    unexpected = sorted(set(config) - _REQUEST_FIELDS)
    if unexpected:
        raise ValueError("unsupported evaluation config field: " + unexpected[0])
    missing = sorted(_REQUEST_FIELDS - {"baseline"} - set(config))
    if missing:
        raise ValueError("evaluation config is missing required field: " + missing[0])

    dataset_version = _safe_identifier(config["dataset_version"], "dataset_version")
    training_job_id = _uuid(config["training_job_id"], "training_job_id")
    model_kind = config["model_kind"]
    if model_kind not in {"detr", "clip"}:
        raise ValueError("evaluation model_kind must be detr or clip")
    config_version = config["evaluation_config_version"]
    if not isinstance(config_version, str) or not config_version.strip() or len(config_version) > 255:
        raise ValueError("evaluation_config_version must contain 1 to 255 characters")
    split = config["evaluation_split"]
    if split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test")
    manifest_sha256 = _digest(config["manifest_sha256"], "manifest_sha256")
    checkpoint_sha256 = _digest(config["checkpoint_sha256"], "checkpoint_sha256")
    if manifest_key != f"datasets/{dataset_version}/manifest.json":
        raise ValueError("manifest URI does not match the requested immutable dataset version")
    expected_checkpoint_prefix = f"jobs/{training_job_id}/checkpoints/"
    if not checkpoint_key.startswith(expected_checkpoint_prefix) or not checkpoint_key.endswith(
        f"/{checkpoint_sha256}.checkpoint"
    ):
        raise ValueError("checkpoint URI does not match the requested training job and content hash")

    baseline = config.get("baseline")
    if baseline is not None:
        if not isinstance(baseline, Mapping) or set(baseline) - {"model_id", "revision"}:
            raise ValueError("baseline must contain only optional model_id and revision metadata")
        baseline_id = baseline.get("model_id")
        baseline_revision = baseline.get("revision")
        if not isinstance(baseline_id, str) or not baseline_id.strip():
            raise ValueError("baseline model_id metadata must be a non-empty string")
        if baseline_revision is not None and (
            not isinstance(baseline_revision, str) or not baseline_revision.strip()
        ):
            raise ValueError("baseline revision metadata must be a non-empty string")
        baseline_result = {
            "model_id": baseline_id,
            "revision": baseline_revision,
            "verified": False,
        }
    else:
        baseline_result = None

    return {
        "dataset_version": dataset_version,
        "manifest_sha256": manifest_sha256,
        "training_job_id": training_job_id,
        "checkpoint_sha256": checkpoint_sha256,
        "model_kind": model_kind,
        "evaluation_config_version": config_version,
        "evaluation_split": split,
        "baseline": baseline_result,
    }


def evaluation_readiness(
    manifest: Mapping[str, Any],
    current_eligibility: Mapping[str, Any],
    *,
    model_kind: str,
    evaluation_split: str,
) -> dict[str, Any]:
    """Combine immutable readiness with the current durable impact overlay."""
    training_snapshot = manifest.get("training")
    evaluation_snapshot = manifest.get("evaluation")
    if not isinstance(training_snapshot, Mapping) or not isinstance(evaluation_snapshot, Mapping):
        return _readiness_result(
            False,
            ["manifest_readiness_missing"],
            False,
            ["manifest_readiness_missing"],
        )
    if not isinstance(current_eligibility, Mapping):
        return _readiness_result(False, ["current_eligibility_unavailable"], False, ["current_eligibility_unavailable"])

    training_reasons = _string_reasons(training_snapshot.get("reason_codes"))
    evaluation_reasons = _string_reasons(evaluation_snapshot.get("reason_codes"))
    if training_snapshot.get("ready") is not True:
        training_reasons.add("dataset_not_training_ready")
    if current_eligibility.get("training_eligible") is not True:
        training_reasons.add("current_dataset_not_training_eligible")

    target = manifest.get("target")
    if target not in {model_kind, "both"}:
        evaluation_reasons.add(f"dataset_target_not_{model_kind}")
    if current_eligibility.get("evaluation_eligible") is not True:
        evaluation_reasons.add("dataset_not_evaluation_eligible")
    impacts = current_eligibility.get("impacts")
    if isinstance(impacts, list):
        for impact in impacts:
            if not isinstance(impact, Mapping):
                evaluation_reasons.add("current_impact_invalid")
                continue
            reason = impact.get("reason")
            if isinstance(reason, str) and reason:
                evaluation_reasons.add(reason)
                if reason == "evaluation_split_leakage":
                    evaluation_reasons.add("late_cross_boundary_link")
                if reason == "source_sample_explicitly_invalidated":
                    training_reasons.add(reason)
                    evaluation_reasons.add(reason)

    counts = manifest.get("split_counts")
    split_counts = counts.get(evaluation_split) if isinstance(counts, Mapping) else None
    if not isinstance(split_counts, Mapping):
        evaluation_reasons.add("evaluation_split_counts_missing")
    elif _valid_nonnegative_int(split_counts.get("groups")) is False or split_counts.get("groups", 0) < 1:
        evaluation_reasons.add("evaluation_split_groups_missing")
    elif model_kind == "detr":
        if _valid_nonnegative_int(split_counts.get("frames")) is False or split_counts.get("frames", 0) < 20:
            evaluation_reasons.add(f"insufficient_{evaluation_split}_detr_frames")
        if (
            _valid_nonnegative_int(split_counts.get("positive_frames")) is False
            or split_counts.get("positive_frames", 0) < 10
        ):
            evaluation_reasons.add(f"insufficient_{evaluation_split}_detr_positive_frames")
        if (
            _valid_nonnegative_int(split_counts.get("negative_frames")) is False
            or split_counts.get("negative_frames", 0) < 5
        ):
            evaluation_reasons.add(f"insufficient_{evaluation_split}_detr_negative_frames")
    elif model_kind == "clip":
        if (
            _valid_nonnegative_int(split_counts.get("crop_caption_pairs")) is False
            or split_counts.get("crop_caption_pairs", 0) < 20
        ):
            evaluation_reasons.add(f"insufficient_{evaluation_split}_clip_pairs")
        if not evaluation_snapshot.get("gallery_crop_ids"):
            evaluation_reasons.add("human_relevance_truth_missing")
        matrices = evaluation_snapshot.get("relevance_matrices")
        if not isinstance(matrices, list) or not matrices:
            evaluation_reasons.add("human_relevance_truth_missing")
        elif any(
            not isinstance(matrix, Mapping)
            or matrix.get("status") != "complete"
            or matrix.get("evaluation_eligible") is not True
            for matrix in matrices
        ):
            evaluation_reasons.add("human_relevance_truth_unresolved")

    training_ready = not training_reasons
    return _readiness_result(
        training_ready,
        sorted(training_reasons),
        not evaluation_reasons,
        sorted(evaluation_reasons),
        impacts=impacts if isinstance(impacts, list) else [],
    )


def deployment_eligibility(report: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the current Gods Watching release gate and fail closed."""
    reasons: set[str] = set()
    if report.get("status") != "complete":
        reasons.add("internal_evaluation_incomplete")
    if report.get("model_kind") != "clip":
        reasons.add("clip_product_quality_evidence_required")
    baseline = report.get("baseline")
    if not isinstance(baseline, Mapping) or baseline.get("verified") is not True:
        reasons.add("baseline_evidence_missing")
    product_evidence = report.get("product_gate_evidence")
    if not isinstance(product_evidence, Mapping):
        reasons.update(
            {
                "cuhk_report_missing",
                "trusted_quality_policy_missing",
                "product_retrieval_evidence_missing",
            }
        )
    else:
        if product_evidence.get("gods_watching_verified") is not True:
            reasons.add("gods_watching_release_gate_unverified")
        cuhk_report = product_evidence.get("cuhk_report")
        if not isinstance(cuhk_report, Mapping):
            reasons.add("cuhk_report_missing")
        else:
            if cuhk_report.get("dataset_split") != "test":
                reasons.add("cuhk_held_out_test_required")
            required_cuhk = (
                "dataset_sha256",
                "protocol",
                "source_checkpoint",
                "source_checkpoint_revision",
                "candidate_weights_sha256",
                "evaluation_code_revision",
                "metric_definition",
            )
            if any(not isinstance(cuhk_report.get(key), str) or not cuhk_report[key].strip() for key in required_cuhk):
                reasons.add("cuhk_report_incomplete")
            for key in ("baseline_score", "candidate_score"):
                value = cuhk_report.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    reasons.add("cuhk_report_score_invalid")
            if (
                isinstance(cuhk_report.get("baseline_score"), (int, float))
                and isinstance(cuhk_report.get("candidate_score"), (int, float))
                and cuhk_report["candidate_score"] <= cuhk_report["baseline_score"]
            ):
                reasons.add("cuhk_held_out_score_not_improved")
        policy = product_evidence.get("trusted_quality_policy")
        if not isinstance(policy, Mapping) or product_evidence.get("quality_evidence_verified") is not True:
            reasons.add("trusted_quality_policy_missing")
        retrieval = product_evidence.get("product_retrieval")
        if not isinstance(retrieval, Mapping):
            reasons.add("product_retrieval_evidence_missing")
        else:
            for modality in ("text", "image"):
                baseline_score = retrieval.get(f"{modality}_baseline_recall_at_5")
                candidate_score = retrieval.get(f"{modality}_candidate_recall_at_5")
                if (
                    isinstance(baseline_score, bool)
                    or isinstance(candidate_score, bool)
                    or not isinstance(baseline_score, (int, float))
                    or not isinstance(candidate_score, (int, float))
                    or not math.isfinite(baseline_score)
                    or not math.isfinite(candidate_score)
                ):
                    reasons.add(f"product_{modality}_retrieval_evidence_missing")
                elif candidate_score < 0.8 or candidate_score <= baseline_score:
                    reasons.add(f"product_{modality}_recall_at_5_below_gate")
    return {
        "eligible": not reasons,
        "status": "eligible" if not reasons else "blocked",
        "reasons": sorted(reasons),
        "requirements": list(_PRODUCT_GATE_REQUIREMENTS),
        "gate_owner": "Gods Watching model_selection.quality.assess_quality",
    }


def load_verified_model_state(
    payload: bytes,
    *,
    expected_identity: Mapping[str, Any],
    expected_model_revision: str,
    model_kind: str,
) -> Mapping[str, Any]:
    """Extract weights from a verified training checkpoint without restoring optimizer state."""
    import torch

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("verified evaluation checkpoint payload is missing")
    if not isinstance(expected_identity, Mapping) or expected_identity.get("phase") != "training":
        raise ValueError("evaluation checkpoint identity must be an originating training job")
    if expected_identity.get("model_kind") != model_kind:
        raise ValueError("evaluation checkpoint belongs to another model kind")
    if expected_identity.get("input_kind") != "dataset_version" or not expected_identity.get("dataset_version"):
        raise ValueError("evaluation checkpoint is not bound to a published training dataset")
    if not isinstance(expected_model_revision, str) or not expected_model_revision:
        raise ValueError("locked model revision is required to load evaluation weights")
    try:
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - convert checkpoint decode failures to a bounded contract error
        raise ValueError("evaluation checkpoint payload is invalid") from error
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != "gods-mlops-training-checkpoint-v1":
        raise ValueError("evaluation checkpoint format is unsupported")
    if checkpoint.get("identity") != dict(expected_identity):
        raise ValueError("evaluation checkpoint identity differs from its database commit marker")
    if checkpoint.get("model_revision") != expected_model_revision:
        raise ValueError("evaluation checkpoint model revision differs from the immutable model lock")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("evaluation checkpoint has no model weights")
    return state


def apply_verified_model_weights(
    model: Any,
    config: Mapping[str, Any],
    *,
    model_kind: str,
    model_revision: str,
) -> None:
    """Load prior training weights for eval inference without restoring optimizer state."""
    payload = config.get("_evaluation_checkpoint_payload")
    identity = config.get("_evaluation_checkpoint_identity")
    state = load_verified_model_state(
        payload,
        expected_identity=identity,
        expected_model_revision=model_revision,
        model_kind=model_kind,
    )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:  # noqa: BLE001 - map model-shape failures to a bounded checkpoint contract error
        raise ValueError("evaluation checkpoint weights do not match the locked model architecture") from error


def _readiness_result(
    training_ready: bool,
    training_reasons: list[str],
    evaluation_eligible: bool,
    evaluation_reasons: list[str],
    *,
    impacts: list[Any] | None = None,
) -> dict[str, Any]:
    return {
        "training_ready": training_ready,
        "training_reasons": sorted(set(training_reasons)),
        "evaluation_eligible": evaluation_eligible,
        "evaluation_reasons": sorted(set(evaluation_reasons)),
        "impacts": list(impacts or []),
    }


def _s3_object_uri(value: Any, name: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ValueError(f"{name} URI must be an S3 object URI")
    parsed = urlsplit(value)
    key = parsed.path.lstrip("/")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or ".." in key.split("/")
        or "\\" in key
    ):
        raise ValueError(f"{name} URI must be a safe S3 object URI")
    return parsed.netloc, key


def _safe_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} is invalid")
    return value


def _uuid(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a UUID")
    try:
        return str(UUID(value))
    except ValueError as error:
        raise ValueError(f"{name} must be a UUID") from error


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _string_reasons(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _valid_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0
