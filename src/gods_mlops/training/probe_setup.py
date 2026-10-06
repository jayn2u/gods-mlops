"""Versioned candidate profiles and deterministic synthetic Task 8 probe inputs."""

from __future__ import annotations

import io
import asyncio
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from gods_mlops.datasets.manifest import canonical_json, content_sha256
from gods_mlops.jobs.models import EvaluationProbeCheckpointSource, ExecutionProfile, ProbeInput

from .contracts import locked_model

_PROFILE_VERSIONS = {
    "detr": "task8-detr-640-microbatch1-probe-v1",
    "clip": "task8-clip-224-microbatch2-explicit-negative-probe-v1",
    "qwen": "task8-qwen-bounded-crop-caption-probe-v1",
}
_DETR_PREPARATION_PROFILE_VERSION = "task8-detr-640-frame-drafts-preparation-probe-v1"
_DETR_EVALUATION_PROFILE_VERSION = "task9-detr-person-coco-evaluation-probe-v1"
_CLIP_EVALUATION_PROFILE_VERSION = "task9-clip-retrieval-evaluation-probe-v1"
_A6000_CANDIDATE_MEMORY_MIB = 36_000
_PROBE_ARTIFACT_RESERVATION_BYTES = 8 * 1024**3


def candidate_profile(model_kind: str, *, target_phase: str | None = None) -> ExecutionProfile:
    """Return the exact unmeasured probe config; it never allowlists itself."""
    model = locked_model(model_kind)
    requested_target = target_phase or ("preparation" if model_kind == "qwen" else "training")
    if model_kind == "detr":
        if requested_target == "training":
            config = {
                "model_id": model.model_id,
                "model_revision": model.revision,
                "input_size": 640,
                "micro_batch": 1,
                "optimizer_steps": 3,
                "learning_rate": 1e-5,
                "weight_decay": 1e-4,
            }
            version = _PROFILE_VERSIONS[model_kind]
        elif requested_target == "preparation":
            config = {
                "model_id": model.model_id,
                "model_revision": model.revision,
                "input_size": 640,
                "micro_batch": 1,
                "score_threshold": 0.3,
                "max_draft_frames": 1,
            }
            version = _DETR_PREPARATION_PROFILE_VERSION
        elif requested_target == "evaluation":
            config = {
                "model_id": model.model_id,
                "model_revision": model.revision,
                "input_size": 640,
                "micro_batch": 1,
                "score_threshold": 0.3,
                "max_detections": 100,
                "max_evaluation_frames": 1,
            }
            version = _DETR_EVALUATION_PROFILE_VERSION
        else:
            raise ValueError("DETR readiness target must be training, preparation, or evaluation")
    elif model_kind == "clip":
        if requested_target == "training":
            config = {
                "model_id": model.model_id,
                "model_revision": model.revision,
                "resolution": 224,
                "micro_batch": 2,
                "gradient_accumulation_steps": 1,
                "contrastive_config_version": "task8-explicit-negatives-v1",
                "optimizer_steps": 3,
                "learning_rate": 1e-5,
                "weight_decay": 1e-4,
            }
            version = _PROFILE_VERSIONS[model_kind]
        elif requested_target == "evaluation":
            config = {
                "model_id": model.model_id,
                "model_revision": model.revision,
                "resolution": 224,
                "micro_batch": 2,
                "evaluation_batch_size": 2,
            }
            version = _CLIP_EVALUATION_PROFILE_VERSION
        else:
            raise ValueError("CLIP readiness target must be training or evaluation")
    elif model_kind == "qwen":
        if requested_target != "preparation":
            raise ValueError("Qwen readiness probe target must be preparation")
        config = {
            "model_id": model.model_id,
            "model_revision": model.revision,
            "max_input_tokens": 4096,
            "max_new_tokens": 128,
            "max_image_pixels": 1_048_576,
            "max_draft_images": 1,
        }
        version = _PROFILE_VERSIONS[model_kind]
    else:
        raise ValueError("candidate probe model must be detr, clip, or qwen")
    return ExecutionProfile(
        model_kind=model_kind,
        config_version=version,
        phase="probe",
        target_phase=requested_target,
        memory_requirement_mib=_A6000_CANDIDATE_MEMORY_MIB,
        artifact_reservation_bytes=_PROBE_ARTIFACT_RESERVATION_BYTES,
        config=config,
        candidate=True,
    )


def create_probe_input(
    *,
    objects: Any,
    model_kind: str,
    target_phase: str | None = None,
    evaluation_checkpoint_source: EvaluationProbeCheckpointSource | None = None,
) -> ProbeInput:
    """Store tiny deterministic synthetic media and a typed immutable S3 manifest."""
    profile = candidate_profile(model_kind, target_phase=target_phase)
    if profile.target_phase == "evaluation":
        if evaluation_checkpoint_source is None:
            raise ValueError("evaluation profile probes require a verified training-probe checkpoint")
        if (
            evaluation_checkpoint_source.model_kind != model_kind
            or evaluation_checkpoint_source.model_id != profile.config["model_id"]
            or evaluation_checkpoint_source.model_revision != profile.config["model_revision"]
        ):
            raise ValueError("evaluation probe checkpoint does not match the locked model profile")
        probe_input_id = (
            f"task9-{model_kind}-evaluation-probe-"
            f"{evaluation_checkpoint_source.checkpoint_sha256}"
        )
    else:
        if evaluation_checkpoint_source is not None:
            raise ValueError("training-probe checkpoints are only valid for evaluation profile probes")
        probe_input_id = (
            "task8-detr-preparation-synthetic-probe-v1"
            if model_kind == "detr" and profile.target_phase == "preparation"
            else f"task8-{model_kind}-synthetic-probe-v1"
        )
    media = _probe_media(model_kind)
    item_objects = []
    for label, image_bytes in media.items():
        digest = sha256(image_bytes).hexdigest()
        key = f"probe-inputs/{probe_input_id}/media/{label}-{digest}.jpg"
        objects.write_immutable(
            object_key=key,
            content=image_bytes,
            sha256_digest=digest,
            content_type="image/jpeg",
        )
        item_objects.append(
            {"key": key, "sha256": digest, "size_bytes": len(image_bytes)}
        )

    common = {
        "schema_version": 1,
        "fixture": True,
        "phase": "probe",
        "model_kind": model_kind,
        "input_kind": "probe_input",
        "input_id": probe_input_id,
        "config_version": profile.config_version,
        "model_id": profile.config["model_id"],
        "model_revision": profile.config["model_revision"],
    }
    if model_kind == "detr":
        manifest = {
            **common,
            "items": [
                {
                    "kind": "frame",
                    "item_kind": "frame",
                    "item_id": "task8-probe-frame-1",
                    "sample_id": "task8-probe-frame-1",
                    "object": item_objects[0],
                    "snapshot": {
                        "bbox_annotation": {
                            "result": [
                                {
                                    "from_name": "bbox",
                                    "to_name": "image",
                                    "type": "rectanglelabels",
                                    "original_width": 640,
                                    "original_height": 640,
                                    "value": {
                                        "x": 32,
                                        "y": 8,
                                        "width": 36,
                                        "height": 84,
                                        "rectanglelabels": ["person"],
                                    },
                                }
                            ]
                        }
                    },
                }
            ],
        }
    elif model_kind == "clip":
        pairs = []
        for image_id, text, positive_index in (
            ("task8-crop-red", "person wearing a red top and dark trousers", 0),
            ("task8-crop-blue", "person wearing a blue coat and light trousers", 1),
        ):
            media_item = {
                "kind": "crop",
                "item_id": image_id,
                "object": item_objects[positive_index],
            }
            other = "person wearing a blue coat and light trousers" if positive_index == 0 else "person wearing a red top and dark trousers"
            pairs.append(
                {
                    "image_id": image_id,
                    "text": text,
                    "negative_texts": [other],
                    "media": media_item,
                }
            )
        manifest = {
            **common,
            **(
                {"contrastive_config_version": profile.config["contrastive_config_version"]}
                if "contrastive_config_version" in profile.config
                else {}
            ),
            "pairs": pairs,
        }
    else:
        manifest = {
            **common,
            "items": [
                {
                    "item_kind": "crop",
                    "item_id": "task8-qwen-synthetic-crop",
                    "sample_id": "task8-qwen-synthetic-frame",
                    "sha256": item_objects[0]["sha256"],
                    "object_key": item_objects[0]["key"],
                    "object_size_bytes": item_objects[0]["size_bytes"],
                    "revision_id": "task8-synthetic-bbox-revision",
                }
            ],
        }

    manifest_bytes = canonical_json(manifest)
    manifest_sha256 = content_sha256(manifest_bytes)
    manifest_key = f"probe-inputs/{probe_input_id}/manifest.json"
    objects.write_immutable(
        object_key=manifest_key,
        content=manifest_bytes,
        sha256_digest=manifest_sha256,
        content_type="application/json",
    )
    return ProbeInput(
        probe_input_id=probe_input_id,
        model_kind=model_kind,
        target_phase=profile.target_phase or "training",
        config_version=profile.config_version,
        manifest_object_key=manifest_key,
        input_sha256=manifest_sha256,
        object_size_bytes=len(manifest_bytes),
        fixture=True,
        evaluation_checkpoint_source=evaluation_checkpoint_source,
    )


async def load_evaluation_probe_checkpoint_source(
    *,
    repository: Any,
    objects: Any,
    bucket: str,
    training_probe_job_id: str | None,
    model_kind: str,
) -> EvaluationProbeCheckpointSource:
    """Build the probe-only source from a completed Task 8 probe and its recorded evidence."""
    from .checkpoints import S3CheckpointStore

    if not training_probe_job_id:
        raise ValueError("evaluation profile probes require a training-target probe checkpoint source")
    try:
        job = await repository.get_job(training_probe_job_id)
        identity = await repository.checkpoint_identity(training_probe_job_id)
        checkpoint_commit = await repository.checkpoint_metadata_for(training_probe_job_id)
        probe_profile = await repository.get_profile(
            phase="probe",
            model_kind=model_kind,
            config_version=str(job.get("config_version", "")),
        )
        training_profile = await repository.get_profile(
            phase="training",
            model_kind=model_kind,
            config_version=str(job.get("config_version", "")),
        )
        measurement = await repository.profile_measurement_for_job(training_probe_job_id)
        runtime_evidence, evidence_sha256 = _recorded_training_probe_evidence(
            training_probe_job_id, model_kind
        )
        model = locked_model(model_kind)
        if checkpoint_commit is None:
            raise ValueError("training-target probe has no current DB checkpoint commit marker")
        source = EvaluationProbeCheckpointSource(
            training_probe_job_id=training_probe_job_id,
            model_kind=model_kind,
            model_id=model.model_id,
            model_revision=model.revision,
            checkpoint_uri=str(checkpoint_commit["uri"]),
            checkpoint_sha256=str(checkpoint_commit["sha256"]).strip(),
            checkpoint_size_bytes=int(checkpoint_commit["size_bytes"]),
            checkpoint_identity=identity.as_dict(),
            worker_image_id=str(runtime_evidence["docker_image_id"]),
            source_commit=str(runtime_evidence["image_source_commit"]),
            runtime_evidence_sha256=evidence_sha256,
        )
        source.validate_training_probe_origin(
            job,
            probe_profile,
            training_profile,
            measurement,
            checkpoint_commit,
            runtime_evidence,
            evidence_sha256,
        )
        checkpoint_store = S3CheckpointStore(objects=objects, bucket=bucket)
        verified = await asyncio.to_thread(
            checkpoint_store.load_uri,
            source.checkpoint_uri,
            expected_identity=identity,
            expected_sha256=source.checkpoint_sha256,
            expected_size_bytes=source.checkpoint_size_bytes,
        )
        if verified.identity.as_dict() != source.checkpoint_identity:
            raise ValueError("S3 checkpoint payload identity differs from the training-probe source")
        return source
    except Exception as error:  # noqa: BLE001 - fail closed without leaking local DB/S3 details
        raise ValueError("training-target probe checkpoint provenance could not be verified") from error


def _recorded_training_probe_evidence(job_id: str, model_kind: str) -> tuple[dict[str, Any], str]:
    root = Path(__file__).resolve().parents[3]
    output_root = root / "output"
    matches = sorted(output_root.rglob(f"{model_kind}-{job_id}.json")) if output_root.is_dir() else []
    if len(matches) != 1:
        raise ValueError("training-probe runtime evidence is missing or ambiguous")
    payload = matches[0].read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("training-probe runtime evidence is not an object")
    return value, sha256(payload).hexdigest()


def _probe_media(model_kind: str) -> dict[str, bytes]:
    if model_kind == "detr":
        return {"frame": _synthetic_person("#a0a0a0", "#232323", size=(640, 640))}
    if model_kind == "clip":
        return {
            "crop-red": _synthetic_person("#b52a2a", "#202020", size=(224, 224)),
            "crop-blue": _synthetic_person("#244daa", "#dedede", size=(224, 224)),
        }
    if model_kind == "qwen":
        return {"crop": _synthetic_person("#d1a125", "#262626", size=(224, 224))}
    raise ValueError("probe media model must be detr, clip, or qwen")


def _synthetic_person(top_color: str, trouser_color: str, *, size: tuple[int, int]) -> bytes:
    image = Image.new("RGB", size, "#c9d1d8")
    draw = ImageDraw.Draw(image)
    width, height = size
    scale = min(width, height) / 224
    center = width / 2
    head = (center - 18 * scale, 12 * scale, center + 18 * scale, 48 * scale)
    draw.ellipse(head, fill="#c98f69")
    draw.rectangle(
        (center - 30 * scale, 50 * scale, center + 30 * scale, 140 * scale),
        fill=top_color,
    )
    draw.rectangle(
        (center - 29 * scale, 140 * scale, center - 3 * scale, 215 * scale),
        fill=trouser_color,
    )
    draw.rectangle(
        (center + 3 * scale, 140 * scale, center + 29 * scale, 215 * scale),
        fill=trouser_color,
    )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()
