"""Evaluation report identity, sample counts, and owned prediction artifact."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from gods_mlops.datasets.manifest import canonical_json

from .detection import evaluate_detections, manifest_person_boxes_xywh
from .eligibility import (
    deployment_eligibility,
    evaluation_readiness,
    validate_evaluation_request,
)
from .retrieval import evaluate_retrieval


def evaluate(manifest_uri: str, checkpoint_uri: str, config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one immutable checkpoint through the Task 7 owned GPU worker."""
    request = validate_evaluation_request(manifest_uri, checkpoint_uri, config)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_evaluate_with_task7_owned_worker(manifest_uri, checkpoint_uri, request))
    raise RuntimeError("evaluate() must be called outside an active asyncio event loop")


def build_evaluation_report(
    *,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    current_eligibility: Mapping[str, Any],
    metrics: Mapping[str, Any],
    predictions: Any,
) -> dict[str, Any]:
    """Bind computed metrics and predictions to verified, immutable identities."""
    if not isinstance(candidate, Mapping) or not isinstance(source, Mapping):
        raise ValueError("evaluation report needs candidate and immutable source identities")
    if not isinstance(evaluation_config, Mapping) or not isinstance(current_eligibility, Mapping):
        raise ValueError("evaluation report needs profile identity and current dataset status")
    if not isinstance(metrics, Mapping):
        raise ValueError("evaluation report metrics must be an object")

    metric_status = metrics.get("status")
    metric_values = metrics.get("metrics")
    if not isinstance(metric_values, Mapping):
        metric_values = {}
    counts = metrics.get("counts")
    if not isinstance(counts, Mapping):
        counts = {}
    reasons = _reason_set(metrics.get("reasons"))
    reasons.update(_reason_set(current_eligibility.get("evaluation_reasons")))
    reasons.update(_impact_reasons(current_eligibility.get("impacts")))
    training_ready = current_eligibility.get("training_ready") is True or current_eligibility.get(
        "training_eligible"
    ) is True
    evaluation_eligible = current_eligibility.get("evaluation_eligible") is True
    if not training_ready:
        reasons.add("dataset_not_training_ready")
    if not evaluation_eligible:
        reasons.add("dataset_not_evaluation_eligible")
    if metric_status != "complete":
        if not reasons:
            reasons.add("evaluation_metrics_unavailable")

    report = {
        "schema_version": 1,
        "execution_status": "succeeded",
        "status": "complete" if not reasons and metric_status == "complete" else "insufficient",
        "model_kind": source.get("target") if source.get("target") in {"detr", "clip"} else candidate.get("model_kind"),
        "candidate": dict(candidate),
        "baseline": dict(baseline) if baseline is not None else None,
        "source": dict(source),
        "evaluation_config": dict(evaluation_config),
        "training_ready": training_ready,
        "training_reasons": sorted(_reason_set(current_eligibility.get("training_reasons"))),
        "evaluation_eligible": evaluation_eligible,
        "current_impacts": list(current_eligibility.get("impacts", []))
        if isinstance(current_eligibility.get("impacts"), list)
        else [],
        "metric_status": metric_status if metric_status in {"complete", "insufficient"} else "insufficient",
        "metrics": dict(metric_values),
        "sample_counts": dict(counts),
        "metric_settings": dict(metrics.get("settings", {}))
        if isinstance(metrics.get("settings"), Mapping)
        else {},
        "insufficient_reasons": sorted(reasons),
        "predictions": predictions,
    }
    report["deployment_eligibility"] = deployment_eligibility(report)
    return report


async def _evaluate_with_task7_owned_worker(
    manifest_uri: str,
    checkpoint_uri: str,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Preflight immutable sources, submit one measured eval job, and verify its artifact."""
    from gods_mlops.datasets.publish import DatasetPublisher
    from gods_mlops.jobs.queue import (
        DatasetNotReadyForEvaluationError,
        JobQueue,
        PostgresJobQueueRepository,
        ResourceProfileNotFoundError,
    )
    from gods_mlops.jobs.sources import DatasetSourceRegistry, DatasetSourceUnavailableError
    from gods_mlops.training.artifacts import S3ResultArtifactStore
    from gods_mlops.training.checkpoints import S3CheckpointStore
    from gods_mlops.training.controller import TrainingController, build_controller_from_environment
    from gods_mlops.training.data import dataset_object_store_from_environment
    from gods_mlops.training.worker import _load_verified_evaluation_checkpoint

    candidate: dict[str, Any] = {
        "training_job_id": request["training_job_id"],
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": request["checkpoint_sha256"],
        "model_id": "sha256:" + request["checkpoint_sha256"],
        "model_kind": request["model_kind"],
        "verified": False,
    }
    source: dict[str, Any] = {
        "dataset_version": request["dataset_version"],
        "manifest_uri": manifest_uri,
        "manifest_sha256": request["manifest_sha256"],
        "target": request["model_kind"],
    }
    evaluation_config: dict[str, Any] = {
        "version": request["evaluation_config_version"],
        "sha256": None,
        "split": request["evaluation_split"],
        "state": "unresolved",
    }
    current: dict[str, Any] = {
        "training_ready": False,
        "training_eligible": False,
        "evaluation_eligible": False,
        "training_reasons": [],
        "evaluation_reasons": ["current_eligibility_unavailable"],
        "impacts": [],
    }

    def not_started(reason: str, *, execution_status: str = "not_started") -> dict[str, Any]:
        reasons = [reason]
        report = build_evaluation_report(
            candidate=candidate,
            baseline=request.get("baseline"),
            source=source,
            evaluation_config=evaluation_config,
            current_eligibility=current,
            metrics={
                "status": "insufficient",
                "metrics": {},
                "counts": {},
                "settings": {},
                "reasons": reasons,
            },
            predictions={},
        )
        report["execution_status"] = execution_status
        report["artifact"] = None
        return report

    database_url = os.environ.get("GODS_MLOPS_DATABASE_URL")
    if not database_url:
        return not_started("current_eligibility_unavailable")

    repository = PostgresJobQueueRepository(database_url=database_url)
    sources = DatasetSourceRegistry(database_url=database_url)
    queue = JobQueue(repository=repository, sources=sources)
    objects = None
    publisher = None
    controller: TrainingController | None = None
    try:
        try:
            objects = dataset_object_store_from_environment()
            publisher = DatasetPublisher.from_environment()
            training_job = await queue.get(request["training_job_id"])
        except Exception:  # noqa: BLE001 - report missing current authority without exposing environment values
            return not_started("current_eligibility_unavailable")

        if (
            training_job.get("state") != "completed"
            or training_job.get("phase") != "training"
            or training_job.get("target_phase") != "training"
            or training_job.get("model_kind") != request["model_kind"]
            or training_job.get("dataset_version") != request["dataset_version"]
            or training_job.get("input_kind") != "dataset_version"
            or training_job.get("input_sha256") != request["manifest_sha256"]
        ):
            return not_started("candidate_training_job_not_successful")

        try:
            dataset_reference = await sources.training_manifest_reference(request["dataset_version"])
            training_profile = await repository.get_profile(
                phase="training",
                model_kind=request["model_kind"],
                config_version=training_job["config_version"],
            )
            checkpoint_identity = await repository.checkpoint_identity(request["training_job_id"])
            checkpoint_metadata = await repository.checkpoint_metadata_for(request["training_job_id"])
        except Exception:  # noqa: BLE001 - immutable source metadata could not be verified
            return not_started("candidate_checkpoint_unverified")

        bucket = os.environ.get("GODS_MLOPS_S3_BUCKET", "")
        expected_manifest_uri = f"s3://{bucket}/{dataset_reference['object_key']}"
        if (
            not bucket
            or expected_manifest_uri != manifest_uri
            or dataset_reference["sha256"] != request["manifest_sha256"]
            or training_profile is None
            or training_profile.get("profile_state") != "measured"
            or training_profile.get("config_sha256") != training_job.get("config_sha256")
            or checkpoint_metadata is None
            or checkpoint_identity.as_dict() != checkpoint_metadata.get("identity")
        ):
            return not_started("candidate_checkpoint_or_manifest_identity_mismatch")

        train_settings = training_profile.get("config_json")
        if not isinstance(train_settings, dict):
            return not_started("candidate_training_config_unavailable")
        model_revision = train_settings.get("model_revision")
        if not isinstance(model_revision, str) or not model_revision:
            return not_started("candidate_training_model_revision_unavailable")
        from gods_mlops.jobs.models import EvaluationCheckpointSource

        try:
            checkpoint_source = EvaluationCheckpointSource(
                training_job_id=request["training_job_id"],
                dataset_version=request["dataset_version"],
                model_kind=request["model_kind"],
                training_manifest_sha256=request["manifest_sha256"],
                checkpoint_uri=checkpoint_uri,
                checkpoint_sha256=request["checkpoint_sha256"],
                checkpoint_size_bytes=int(checkpoint_metadata["size_bytes"]),
                checkpoint_identity=checkpoint_identity.as_dict(),
                model_revision=model_revision,
            )
            checkpoint_store = S3CheckpointStore(objects=objects, bucket=bucket)
            await _load_verified_evaluation_checkpoint(queue, checkpoint_store, checkpoint_source)
        except Exception:  # noqa: BLE001 - failed DB marker or object bytes cannot establish candidate identity
            return not_started("candidate_checkpoint_unverified")

        candidate.update(
            {
                "checkpoint_size_bytes": checkpoint_source.checkpoint_size_bytes,
                "checkpoint_identity": checkpoint_source.checkpoint_identity,
                "model_revision": checkpoint_source.model_revision,
                "verified": True,
            }
        )
        try:
            manifest = __import__("gods_mlops.training.contracts", fromlist=["load_manifest"]).load_manifest(
                manifest_uri,
                config={
                    "input_sha256": request["manifest_sha256"],
                    "manifest_size_bytes": dataset_reference["size_bytes"],
                },
            )
        except Exception:  # noqa: BLE001 - a bad immutable manifest is insufficient evidence
            return not_started("evaluation_manifest_unverified")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("dataset_version") != request["dataset_version"]
            or manifest.get("target") not in {request["model_kind"], "both"}
        ):
            return not_started("evaluation_manifest_identity_mismatch")
        source.update(
            {
                "target": manifest.get("target"),
                "manifest_code": manifest.get("code"),
                "split_policy": manifest.get("split_policy"),
                "split_counts": manifest.get("split_counts"),
            }
        )

        try:
            current = await publisher.register_model_lineage(
                model_id=candidate["model_id"],
                dataset_version=request["dataset_version"],
            )
        except Exception:  # noqa: BLE001 - fail closed when current lineage cannot be read
            return not_started("current_eligibility_unavailable")
        readiness = evaluation_readiness(
            manifest,
            current,
            model_kind=request["model_kind"],
            evaluation_split=request["evaluation_split"],
        )
        current = readiness
        evaluation_config = {
            "version": request["evaluation_config_version"],
            "sha256": None,
            "split": request["evaluation_split"],
            "state": "unresolved",
        }
        if not readiness["training_ready"]:
            return not_started("dataset_not_training_ready")
        if not readiness["evaluation_eligible"]:
            return not_started("dataset_not_evaluation_eligible")

        profile = await repository.get_profile(
            phase="evaluation",
            model_kind=request["model_kind"],
            config_version=request["evaluation_config_version"],
        )
        if profile is None:
            return not_started("evaluation_profile_missing")
        evaluation_config = {
            "version": profile["config_version"],
            "sha256": profile["config_sha256"],
            "measurement_id": profile.get("measurement_id"),
            "state": profile["profile_state"],
            "split": request["evaluation_split"],
            "settings": profile.get("config_json", {}),
        }
        if profile.get("profile_state") != "measured":
            return not_started("evaluation_profile_not_measured")

        worker_image = os.environ.get("GODS_MLOPS_WORKER_IMAGE")
        if not worker_image:
            return not_started("evaluation_worker_image_unavailable")
        job_id = await queue.submit_evaluation(
            dataset_version=request["dataset_version"],
            model_kind=request["model_kind"],
            config_version=request["evaluation_config_version"],
            checkpoint_source=checkpoint_source,
            evaluation_split=request["evaluation_split"],
            baseline_metadata=(
                {"model_id": request["baseline"]["model_id"], "revision": request["baseline"]["revision"]}
                if request.get("baseline") is not None
                else None
            ),
        )
        controller = build_controller_from_environment(
            worker_image=worker_image,
            namespace=os.environ.get("GODS_MLOPS_WORKER_NAMESPACE", "gods-mlops"),
        )
        job_result = await controller.run(job_id)
        if job_result.get("state") != "completed":
            try:
                current = await publisher.register_model_lineage(
                    model_id=candidate["model_id"],
                    dataset_version=request["dataset_version"],
                )
                current = evaluation_readiness(
                    manifest,
                    current,
                    model_kind=request["model_kind"],
                    evaluation_split=request["evaluation_split"],
                )
            except Exception:  # noqa: BLE001 - preserve the worker failure if overlay readback is unavailable
                pass
            return not_started(str(job_result.get("reason_code") or "evaluation_worker_failed"), execution_status="failed")

        artifacts = await repository.result_artifacts_for(job_id)
        if len(artifacts) != 1 or artifacts[0].get("kind") != "evaluation":
            return not_started("evaluation_result_artifact_missing", execution_status="failed")
        result_identity = await repository.checkpoint_identity(job_id)
        result_store = S3ResultArtifactStore(objects=objects, bucket=bucket)
        verified_artifact = result_store.verify_committed(
            artifacts[0],
            expected_identity=result_identity,
        )
        result_bytes = objects.read_source(
            object_key=verified_artifact.object_key,
            sha256_digest=verified_artifact.sha256,
            size_bytes=verified_artifact.size_bytes,
        )
        report = json.loads(result_bytes)
        if (
            not isinstance(report, dict)
            or report.get("candidate", {}).get("checkpoint_sha256") != request["checkpoint_sha256"]
            or report.get("source", {}).get("manifest_sha256") != request["manifest_sha256"]
            or report.get("evaluation_config", {}).get("sha256") != profile["config_sha256"]
            or report.get("evaluation_job", {}).get("job_id") != job_id
        ):
            raise ValueError("evaluation artifact provenance differs from the current immutable request")
        current = await publisher.register_model_lineage(
            model_id=candidate["model_id"],
            dataset_version=request["dataset_version"],
        )
        final_readiness = evaluation_readiness(
            manifest,
            current,
            model_kind=request["model_kind"],
            evaluation_split=request["evaluation_split"],
        )
        report["current_eligibility_rechecked"] = final_readiness
        if not final_readiness["evaluation_eligible"]:
            report["status"] = "insufficient"
            report["evaluation_eligible"] = False
            report["insufficient_reasons"] = sorted(
                set(report.get("insufficient_reasons", []))
                | set(final_readiness["evaluation_reasons"])
                | {"dataset_not_evaluation_eligible"}
            )
            report["deployment_eligibility"] = deployment_eligibility(report)
        report["artifact"] = {
            "uri": verified_artifact.uri,
            "sha256": verified_artifact.sha256,
            "size_bytes": verified_artifact.size_bytes,
            "kind": verified_artifact.kind,
        }
        return report
    except DatasetNotReadyForEvaluationError as error:
        return not_started(str(error))
    except ResourceProfileNotFoundError:
        return not_started("evaluation_profile_missing")
    except DatasetSourceUnavailableError:
        return not_started("evaluation_source_unavailable")
    finally:
        if controller is not None:
            await controller.close()
        if publisher is not None:
            await publisher.close()
        await sources.close()
        await repository.close()


def evaluate_prediction_payload(
    *,
    manifest: Mapping[str, Any],
    prediction_payload: Mapping[str, Any],
    model_kind: str,
    evaluation_split: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    current_eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    """Score worker-produced predictions against the frozen manifest and current overlay."""
    expected_input_sha = source.get("manifest_sha256")
    if not isinstance(expected_input_sha, str) or prediction_payload.get("input_sha256") != expected_input_sha:
        raise ValueError("evaluation predictions do not bind to the immutable manifest hash")
    if prediction_payload.get("model_kind") != model_kind:
        raise ValueError("evaluation prediction model kind differs from the queued model")
    if manifest.get("dataset_version") != source.get("dataset_version"):
        raise ValueError("evaluation manifest dataset version differs from the immutable source")

    readiness = evaluation_readiness(
        manifest,
        current_eligibility,
        model_kind=model_kind,
        evaluation_split=evaluation_split,
    )
    if model_kind == "detr":
        metrics = evaluate_detections(
            frames=_detection_frames(manifest, prediction_payload, evaluation_split),
            predictions=_detection_predictions(prediction_payload),
            score_threshold=evaluation_config.get("score_threshold", 0.3),
            max_detections=evaluation_config.get("max_detections", 100),
        )
    elif model_kind == "clip":
        evaluation_truth = manifest.get("evaluation")
        if not isinstance(evaluation_truth, Mapping):
            raise ValueError("CLIP manifest evaluation truth is unavailable")
        metrics = evaluate_retrieval(
            query_embeddings=prediction_payload.get("query_embeddings", {}),
            crop_embeddings=prediction_payload.get("crop_embeddings", {}),
            gallery_crop_ids=evaluation_truth.get("gallery_crop_ids", []),
            relevance_matrices=evaluation_truth.get("relevance_matrices", []),
        )
    else:
        raise ValueError("evaluation prediction scoring supports DETR or CLIP")

    enriched_source = {
        **dict(source),
        "schema_version": manifest.get("schema_version"),
        "target": manifest.get("target"),
        "manifest_code": manifest.get("code"),
        "split_policy": manifest.get("split_policy"),
    }
    enriched_config = {
        **dict(evaluation_config),
        "model_kind": model_kind,
        "split": evaluation_split,
    }
    return build_evaluation_report(
        candidate=candidate,
        baseline=baseline,
        source=enriched_source,
        evaluation_config=enriched_config,
        current_eligibility=readiness,
        metrics=metrics,
        predictions=dict(prediction_payload),
    )


def encode_evaluation_cursor(
    *,
    identity: Mapping[str, Any],
    candidate_checkpoint_sha256: str,
    manifest_sha256: str,
    config_sha256: str,
    next_index: int,
    predictions: list[dict[str, Any]] | dict[str, Any],
) -> bytes:
    """Serialize only evaluation progress; no optimizer/training cursor is reused."""
    _validate_evaluation_identity(identity)
    _validate_digest(candidate_checkpoint_sha256, "candidate checkpoint")
    _validate_digest(manifest_sha256, "evaluation manifest")
    _validate_digest(config_sha256, "evaluation config")
    if identity.get("input_sha256") != manifest_sha256 or identity.get("config_sha256") != config_sha256:
        raise ValueError("evaluation cursor identity differs from its manifest or config")
    if type(next_index) is not int or next_index < 0:
        raise ValueError("evaluation cursor index must be a non-negative integer")
    if not isinstance(predictions, (list, dict)):
        raise ValueError("evaluation cursor predictions must be a JSON array or object")
    return canonical_json(
        {
            "format": "gods-mlops-evaluation-cursor-v1",
            "identity": dict(identity),
            "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "next_index": next_index,
            "predictions": predictions,
        }
    )


def decode_evaluation_cursor(
    payload: bytes,
    *,
    expected_identity: Mapping[str, Any],
    expected_candidate_checkpoint_sha256: str,
    expected_manifest_sha256: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Validate a fenced eval-job cursor against both sources before resuming inference."""
    import json

    _validate_evaluation_identity(expected_identity)
    _validate_digest(expected_candidate_checkpoint_sha256, "candidate checkpoint")
    _validate_digest(expected_manifest_sha256, "evaluation manifest")
    _validate_digest(expected_config_sha256, "evaluation config")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("evaluation cursor payload is missing")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation cursor payload is not valid JSON") from error
    expected_keys = {
        "format",
        "identity",
        "candidate_checkpoint_sha256",
        "manifest_sha256",
        "config_sha256",
        "next_index",
        "predictions",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("evaluation cursor payload fields are unsupported")
    if value.get("format") != "gods-mlops-evaluation-cursor-v1":
        raise ValueError("payload is not an evaluation cursor")
    if value.get("identity") != dict(expected_identity):
        raise ValueError("evaluation cursor job identity changed during resume")
    if value.get("candidate_checkpoint_sha256") != expected_candidate_checkpoint_sha256:
        raise ValueError("evaluation cursor candidate checkpoint changed during resume")
    if value.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("evaluation cursor manifest changed during resume")
    if value.get("config_sha256") != expected_config_sha256:
        raise ValueError("evaluation cursor config changed during resume")
    if type(value.get("next_index")) is not int or value["next_index"] < 0:
        raise ValueError("evaluation cursor index is invalid")
    if not isinstance(value.get("predictions"), (list, dict)):
        raise ValueError("evaluation cursor predictions are invalid")
    return value


def _reason_set(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple)):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _impact_reasons(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    reasons = set()
    for impact in value:
        if not isinstance(impact, Mapping):
            continue
        reason = impact.get("reason")
        if isinstance(reason, str) and reason:
            reasons.add(reason)
            if reason == "evaluation_split_leakage":
                reasons.add("late_cross_boundary_link")
    return reasons


def _detection_frames(
    manifest: Mapping[str, Any],
    prediction_payload: Mapping[str, Any],
    evaluation_split: str,
) -> list[dict[str, Any]]:
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("DETR manifest has no immutable frame list")
    selected = [
        item
        for item in items
        if isinstance(item, Mapping)
        and (item.get("item_kind") == "frame" or item.get("kind") == "frame")
        and item.get("split") == evaluation_split
    ]
    predictions = _detection_predictions(prediction_payload)
    by_id = {item["item_id"]: item for item in predictions}
    selected_ids = {str(item.get("item_id", item.get("sample_id", ""))) for item in selected}
    if set(by_id) != selected_ids:
        raise ValueError("DETR worker predictions do not cover the exact immutable evaluation frame set")
    frames = []
    for item in selected:
        item_id = str(item.get("item_id", item.get("sample_id", "")))
        prediction = by_id[item_id]
        width, height = prediction.get("image_width"), prediction.get("image_height")
        frames.append(
            {
                "item_id": item_id,
                "width": width,
                "height": height,
                "person_boxes_xywh": manifest_person_boxes_xywh(
                    item,
                    image_width=width,
                    image_height=height,
                ),
            }
        )
    return frames


def _detection_predictions(prediction_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    drafts = prediction_payload.get("drafts")
    if not isinstance(drafts, list):
        raise ValueError("DETR worker prediction artifact has no draft rows")
    normalized = []
    seen = set()
    for draft in drafts:
        if not isinstance(draft, Mapping):
            raise ValueError("DETR worker prediction row is invalid")
        item_id = draft.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise ValueError("DETR worker prediction IDs must be unique non-empty strings")
        seen.add(item_id)
        normalized.append(
            {
                "item_id": item_id,
                "image_width": draft.get("image_width"),
                "image_height": draft.get("image_height"),
                "detections": draft.get("detections", []),
            }
        )
    return normalized


def _validate_evaluation_identity(identity: Mapping[str, Any]) -> None:
    required = {
        "job_id",
        "input_kind",
        "input_id",
        "input_sha256",
        "phase",
        "model_kind",
        "config_version",
        "config_sha256",
        "dataset_version",
    }
    if not isinstance(identity, Mapping) or set(identity) != required:
        raise ValueError("evaluation cursor needs a complete Task 7 job identity")
    if (
        identity.get("phase") != "evaluation"
        or identity.get("input_kind") != "dataset_version"
        or identity.get("input_id") != identity.get("dataset_version")
    ):
        raise ValueError("evaluation cursor must be scoped to its own evaluation job")
    _validate_digest(identity.get("input_sha256"), "evaluation identity input")
    _validate_digest(identity.get("config_sha256"), "evaluation identity config")


def _validate_digest(value: Any, name: str) -> None:
    import re

    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
