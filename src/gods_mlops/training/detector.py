"""Locked RT-DETRv2 R18 person detector training and draft inference."""

from __future__ import annotations

import gc
from hashlib import sha256
from time import perf_counter
from typing import Any

from PIL import Image
from .claims import WorkerYieldRequested

from .contracts import load_manifest, locked_model, validate_manifest_identity
from .data import load_rgb_image


def to_coco_annotations(
    item: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    person_class_id: int,
) -> dict[str, Any]:
    """Convert the frozen Task 5 percentage bbox results to RT-DETR COCO boxes."""
    snapshot = item.get("snapshot", {})
    bbox = snapshot.get("bbox_annotation", item.get("bbox_annotation", {}))
    results = bbox.get("result", []) if isinstance(bbox, dict) else []
    annotations: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("bbox annotation contains a non-object result")
        if result.get("from_name") != "bbox" or result.get("type") != "rectanglelabels":
            continue
        value = result.get("value")
        if not isinstance(value, dict) or "person" not in value.get("rectanglelabels", []):
            continue
        source_width = float(result.get("original_width", image_width))
        source_height = float(result.get("original_height", image_height))
        if source_width <= 0 or source_height <= 0:
            raise ValueError("bbox annotation image dimensions are invalid")
        scale_x, scale_y = image_width / source_width, image_height / source_height
        x = float(value["x"]) * source_width / 100.0 * scale_x
        y = float(value["y"]) * source_height / 100.0 * scale_y
        width = float(value["width"]) * source_width / 100.0 * scale_x
        height = float(value["height"]) * source_height / 100.0 * scale_y
        x = min(max(0.0, x), float(image_width))
        y = min(max(0.0, y), float(image_height))
        width = min(max(0.0, width), float(image_width) - x)
        height = min(max(0.0, height), float(image_height) - y)
        if width <= 0 or height <= 0:
            raise ValueError("person bbox has no visible area in the frozen frame")
        annotations.append(
            {
                "bbox": [x, y, width, height],
                "category_id": int(person_class_id),
                "area": width * height,
                "iscrowd": 0,
            }
        )
    return {"image_id": str(item.get("item_id", item.get("sample_id", ""))), "annotations": annotations}


def run(config: dict[str, Any], manifest_uri: str, output_uri: str) -> dict[str, Any]:
    """Run one Task 8 detector job after the CPU controller supplied an active claim."""
    import torch
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    from .data import dataset_object_store_from_environment
    from .runner_support import (
        assert_weights_changed,
        cache_directory,
        checkpoint_identity,
        model_bundle,
        optimizer,
        require_cuda,
        resource_measurements,
        restore_checkpoint,
        selected_trainable_weights,
        serialize_checkpoint,
        step_optimizer,
        write_json,
    )

    if int(config.get("input_size", 640)) != 640:
        raise ValueError("RT-DETRv2 training configuration must pin 640px input")
    phase = str(config.get("phase", ""))
    target_phase = str(config.get("target_phase") or phase)
    if int(config.get("micro_batch", 1)) != 1:
        raise ValueError("RT-DETRv2 versioned profile requires micro-batch 1")
    manifest = load_manifest(manifest_uri, config=config)
    identity = validate_manifest_identity(config, manifest)
    model_lock = locked_model("detr")
    model_path = cache_directory("detr", config)
    object_store = _object_store_if_needed(manifest)
    items = _detector_items(manifest, config)
    images = [load_rgb_image(item, object_store=object_store) for item in items]
    started = perf_counter()
    device = require_cuda()
    torch.cuda.reset_peak_memory_stats(0)
    processor = RTDetrImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = RTDetrV2ForObjectDetection.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    )
    model.to(device)
    id2label = {int(key): str(value).lower() for key, value in model.config.id2label.items()}
    person_class_id = next((key for key, value in id2label.items() if value == "person"), None)
    if person_class_id is None:
        raise ValueError("locked RT-DETRv2 revision has no person class mapping")
    output_path = _output_path(output_uri)

    if target_phase in {"preparation", "evaluation"}:
        return _draft_detections(
            config=config,
            identity=identity,
            model=model,
            processor=processor,
            items=items,
            images=images,
            person_class_id=person_class_id,
            device=device,
            output_uri=output_uri,
            output_path=output_path,
            started=started,
        )

    optimizer_instance = optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    weights_before = selected_trainable_weights(model)
    resume_payload = config.get("_resume_payload")
    start_step = 0
    if isinstance(resume_payload, bytes):
        start_step = restore_checkpoint(
            resume_payload,
            model=model,
            optimizer_instance=optimizer_instance,
            expected_identity=checkpoint_identity(config),
            expected_revision=model_lock.revision,
            device=device,
            scaler=scaler,
        )
    count = int(config.get("optimizer_steps", 3))
    if count < 3:
        raise ValueError("detector measurement requires at least three optimizer steps")
    input_batches = [
        _processor_inputs(processor, item, image, index, person_class_id)
        for index, (item, image) in enumerate(zip(items, images, strict=True))
    ]
    steps_done = start_step
    yielded = False
    for step in range(count):
        if not _worker_should_continue(config):
            yielded = True
            break
        inputs = _move_inputs(input_batches[(start_step + step) % len(input_batches)], device)
        step_optimizer(
            model,
            optimizer_instance,
            inputs,
            scaler=scaler,
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        )
        steps_done += 1
        if not _worker_should_continue(config):
            yielded = True
            break
    if yielded:
        partial_checkpoint = serialize_checkpoint(
            model=model,
            optimizer_instance=optimizer_instance,
            optimizer_steps=steps_done,
            identity=checkpoint_identity(config),
            model_revision=model_lock.revision,
            precision="float16-autocast",
            scaler=scaler,
        )
        commit_checkpoint = config.get("_commit_checkpoint")
        if callable(commit_checkpoint):
            commit_checkpoint(partial_checkpoint)
            partial_checkpoint = b""
        return {"status": "yielded", "checkpoint_payload": partial_checkpoint}
    first_payload = serialize_checkpoint(
        model=model,
        optimizer_instance=optimizer_instance,
        optimizer_steps=steps_done,
        identity=checkpoint_identity(config),
        model_revision=model_lock.revision,
        precision="float16-autocast",
        scaler=scaler,
    )
    resume_verified = False
    resumed_steps = 0
    resume_verified = isinstance(resume_payload, bytes)
    if phase == "probe":
        if not _worker_should_continue(config):
            commit_checkpoint = config.get("_commit_checkpoint")
            if callable(commit_checkpoint):
                commit_checkpoint(first_payload)
                first_payload = b""
            return {"status": "yielded", "checkpoint_payload": first_payload}
        resumed_steps = 1
        del optimizer_instance, scaler, model
        gc.collect()
        torch.cuda.empty_cache()
        model = RTDetrV2ForObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.float32,
        ).to(device)
        optimizer_instance = optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        restored_step = restore_checkpoint(
            first_payload,
            model=model,
            optimizer_instance=optimizer_instance,
            expected_identity=checkpoint_identity(config),
            expected_revision=model_lock.revision,
            device=device,
            scaler=scaler,
        )
        if restored_step != steps_done:
            raise ValueError("detector checkpoint resumed at the wrong optimizer step")
        inputs = _move_inputs(input_batches[restored_step % len(input_batches)], device)
        step_optimizer(model, optimizer_instance, inputs, scaler=scaler)
        resume_verified = True
        if not _worker_should_continue(config):
            yielded_checkpoint = serialize_checkpoint(
                model=model,
                optimizer_instance=optimizer_instance,
                optimizer_steps=steps_done + resumed_steps,
                identity=checkpoint_identity(config),
                model_revision=model_lock.revision,
                precision="float16-autocast",
                scaler=scaler,
            )
            commit_checkpoint = config.get("_commit_checkpoint")
            if callable(commit_checkpoint):
                commit_checkpoint(yielded_checkpoint)
                yielded_checkpoint = b""
            return {"status": "yielded", "checkpoint_payload": yielded_checkpoint}
    initial_weight_sha, final_weight_sha = assert_weights_changed(weights_before, model)
    optimizer_steps = steps_done + resumed_steps
    checkpoint_payload = serialize_checkpoint(
        model=model,
        optimizer_instance=optimizer_instance,
        optimizer_steps=optimizer_steps,
        identity=checkpoint_identity(config),
        model_revision=model_lock.revision,
        precision="float16-autocast",
        scaler=scaler,
    )
    checkpoint_sha = sha256(checkpoint_payload).hexdigest()
    measurements = resource_measurements(started, optimizer_steps=optimizer_steps)
    measurements.update(
        {
            "input_size": 640,
            "micro_batch": 1,
            "initial_optimizer_steps": steps_done,
            "resume_steps": resumed_steps,
            "checkpoint_resumed": resume_verified,
            "initial_weight_sha256": initial_weight_sha,
            "final_weight_sha256": final_weight_sha,
            "model_id": model_lock.model_id,
            "model_revision": model_lock.revision,
        }
    )
    metrics = {"status": "succeeded", "resource_measurements": measurements}
    result_payload, result_sha = model_bundle(model, output_uri=output_uri, metrics=metrics)
    checkpoint_uri = None
    result_uri = None
    commit_checkpoint = config.get("_commit_checkpoint")
    if callable(commit_checkpoint):
        checkpoint_uri = commit_checkpoint(checkpoint_payload).uri
        checkpoint_payload = b""
    commit_result = config.get("_commit_result_artifact")
    if callable(commit_result):
        result_uri = commit_result("model", result_payload, measurements).uri
        result_payload = b""
    return {
        "status": "succeeded",
        "checkpoint_uri": checkpoint_uri,
        "checkpoint_sha256": checkpoint_sha,
        "hash": result_sha,
        "result_uri": result_uri,
        "result_artifact_kind": "model",
        "result_artifact_payload": result_payload,
        "checkpoint_payload": checkpoint_payload,
        "resource_measurements": measurements,
    }


def _draft_detections(
    *,
    config: dict[str, Any],
    identity: dict[str, Any],
    model: Any,
    processor: Any,
    items: list[dict[str, Any]],
    images: list[Image.Image],
    person_class_id: int,
    device: Any,
    output_uri: str,
    output_path: Path,
    started: float,
) -> dict[str, Any]:
    import torch

    threshold = float(config.get("score_threshold", 0.3))
    if not 0 <= threshold <= 1:
        raise ValueError("detector score threshold must be between zero and one")
    model.eval()
    drafts = []
    with torch.inference_mode():
        for item, image in zip(items, images, strict=True):
            if not _worker_should_continue(config):
                raise WorkerYieldRequested("Task 7 requested a cooperative detector draft stop")
            inputs = processor(images=image, return_tensors="pt")
            inputs = _move_inputs(inputs, device)
            outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([[image.height, image.width]], device=device),
                threshold=threshold,
            )[0]
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=True):
                if int(label) != person_class_id:
                    continue
                detections.append(
                    {
                        "score": float(score.cpu()),
                        "bbox_xyxy": [float(value) for value in box.cpu().tolist()],
                    }
                )
            if not _worker_should_continue(config):
                raise WorkerYieldRequested("Task 7 requested a cooperative detector draft stop")
            drafts.append(
                {
                    "item_id": str(item.get("item_id", item.get("sample_id"))),
                    "image_width": image.width,
                    "image_height": image.height,
                    "detections": detections,
                }
            )
    output = {
        "schema_version": 1,
        "model_kind": "detr",
        "model_id": identity["model_id"],
        "model_revision": identity["model_revision"],
        "config_version": identity["config_version"],
        "input_id": identity["input_id"],
        "input_sha256": identity["input_sha256"],
        "score_threshold": threshold,
        "drafts": drafts,
    }
    encoded = write_json(output_path / "draft-detections.json", output)
    digest = sha256(encoded).hexdigest()
    measurements = _draft_resource_measurements(
        resource_measurements(started, optimizer_steps=0, inference_steps=len(images)),
        identity=identity,
    )
    result_uri = None
    commit_result = config.get("_commit_result_artifact")
    if callable(commit_result):
        result_uri = commit_result("drafts", encoded, measurements).uri
        encoded = b""
    return {
        "status": "succeeded",
        "checkpoint_uri": None,
        "hash": digest,
        "result_uri": result_uri,
        "result_artifact_kind": "drafts",
        "result_artifact_payload": encoded,
        "resource_measurements": measurements,
    }


def _detector_items(manifest: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("detector manifest has no immutable frame items")
    if config.get("phase") == "training":
        selected = [item for item in items if item.get("kind") == "frame" and item.get("split") == "train"]
    elif config.get("phase") == "preparation":
        selected = [item for item in items if item.get("item_kind") == "frame"]
    else:
        selected = [item for item in items if item.get("kind") == "frame" or item.get("item_kind") == "frame"]
    if not selected:
        raise ValueError("detector manifest has no frame items for this phase")
    if config.get("target_phase") == "preparation" or config.get("phase") == "preparation":
        limit = config.get("max_draft_frames", 1)
        if type(limit) is not int or limit < 1:
            raise ValueError("DETR preparation profile needs a positive max_draft_frames bound")
        if len(selected) > limit:
            raise ValueError("DETR preparation input exceeds its versioned max_draft_frames bound")
    return selected


def _draft_resource_measurements(
    measurements: dict[str, Any], *, identity: dict[str, Any]
) -> dict[str, Any]:
    """Persist the same locked model identity used by the DETR draft document."""
    model_id = identity.get("model_id")
    model_revision = identity.get("model_revision")
    if not isinstance(model_id, str) or not model_id or not isinstance(model_revision, str) or not model_revision:
        raise ValueError("DETR draft measurements require the locked model identity")
    return {
        **measurements,
        "input_size": 640,
        "micro_batch": 1,
        "model_id": model_id,
        "model_revision": model_revision,
    }


def _processor_inputs(
    processor: Any,
    item: dict[str, Any],
    image: Image.Image,
    image_id: int,
    person_class_id: int,
) -> dict[str, Any]:
    target = to_coco_annotations(item, image_width=image.width, image_height=image.height, person_class_id=person_class_id)
    target["image_id"] = image_id
    return processor(images=image, annotations=target, return_tensors="pt")


def _move_inputs(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    moved = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = [
                {name: item.to(device) if isinstance(item, torch.Tensor) else item for name, item in label.items()}
                if isinstance(label, dict)
                else label
                for label in value
            ]
        else:
            moved[key] = value
    return moved


def _object_store_if_needed(manifest: dict[str, Any]):
    items = manifest.get("items", [])
    needs_store = any(
        isinstance(item, dict)
        and not isinstance(item.get("image_path"), str)
        and (isinstance(item.get("object"), dict) or isinstance(item.get("object_key"), str))
        for item in items
    )
    if not needs_store:
        return None
    from .data import dataset_object_store_from_environment

    return dataset_object_store_from_environment()


def _output_path(output_uri: str) -> Path:
    from .runner_support import output_directory

    directory = output_directory(output_uri)
    return directory


def _worker_should_continue(config: dict[str, Any]) -> bool:
    check = config.get("_assert_current")
    return bool(check()) if callable(check) else True
