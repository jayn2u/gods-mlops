"""CLIP training contracts and real contrastive runner."""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter
from typing import Any

from .contracts import load_manifest, locked_model, validate_manifest_identity
from .data import load_rgb_image


def validate_contrastive_pairs(manifest: dict[str, Any]) -> int:
    """Require versioned, distinct positives and explicit in-batch negatives."""
    version = manifest.get("contrastive_config_version")
    pairs = manifest.get("pairs")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("CLIP contrastive configuration must be versioned")
    if not isinstance(pairs, list) or len(pairs) < 2:
        raise ValueError("CLIP contrastive training requires at least two positive pairs")
    image_ids: set[str] = set()
    positive_texts: set[str] = set()
    negatives_by_image: dict[str, set[str]] = {}
    positive_by_image: dict[str, str] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("CLIP contrastive pairs must be objects")
        image_id = str(pair.get("image_id", "")).strip()
        text = str(pair.get("text", "")).strip()
        negative_texts = pair.get("negative_texts")
        if not image_id or not text:
            raise ValueError("CLIP contrastive pairs need distinct image IDs and positive text")
        if image_id in image_ids or text in positive_texts:
            raise ValueError("CLIP positive pairs must use distinct images and texts")
        if not isinstance(negative_texts, list) or not negative_texts:
            raise ValueError("CLIP training requires explicit negative examples for each pair")
        negatives = {str(item).strip() for item in negative_texts if str(item).strip()}
        if text in negatives:
            raise ValueError("a CLIP negative example cannot equal its positive text")
        image_ids.add(image_id)
        positive_texts.add(text)
        negatives_by_image[image_id] = negatives
        positive_by_image[image_id] = text
    for image_id, negatives in negatives_by_image.items():
        positive = positive_by_image[image_id]
        if not any(text in positive_texts and text != positive for text in negatives):
            raise ValueError(f"CLIP pair {image_id} has no explicit cross-example negative")
    return len(pairs)


def run(config: dict[str, Any], manifest_uri: str, output_uri: str) -> dict[str, Any]:
    """Train the locked CLIP model with explicit in-batch negatives and resumable state."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

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
    )

    if config.get("phase") not in {"probe", "training"}:
        raise ValueError("CLIP runner supports only probe and published training phases")
    if config.get("target_phase", config.get("phase")) == "preparation":
        raise ValueError("CLIP has no pre-publication caption-draft preparation stage")
    if int(config.get("resolution", 224)) != 224:
        raise ValueError("CLIP ViT-B/16 training requires the versioned 224px resolution")
    micro_batch = int(config.get("micro_batch", 1))
    if micro_batch < 2:
        raise ValueError("CLIP contrastive training requires a micro-batch of at least two")
    if int(config.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("gradient accumulation cannot substitute for explicit CLIP negatives")
    steps = int(config.get("optimizer_steps", 3))
    if steps < 3:
        raise ValueError("CLIP measurement requires at least three optimizer steps")

    manifest = load_manifest(manifest_uri, config=config)
    identity = validate_manifest_identity(config, manifest)
    pairs, object_store = _pairs_from_manifest(manifest, config, micro_batch)
    pair_count = validate_contrastive_pairs(
        {
            "contrastive_config_version": config.get("contrastive_config_version"),
            "pairs": pairs,
        }
    )
    started = perf_counter()
    model_lock = locked_model("clip")
    model_path = cache_directory("clip", config)
    device = require_cuda()
    torch.cuda.reset_peak_memory_stats(0)
    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
    model = CLIPModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32)
    model.to(device)
    model.train()
    optimizer_instance = optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    initial_weights = selected_trainable_weights(model)
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
    images = [pair["image"] for pair in pairs]
    texts = [pair["text"] for pair in pairs]
    encoded = processor(text=texts, images=images, return_tensors="pt", padding=True)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    losses = []
    steps_done = start_step
    for _ in range(steps):
        if not _worker_should_continue(config):
            return _yield_training_checkpoint(
                config=config,
                model=model,
                optimizer_instance=optimizer_instance,
                scaler=scaler,
                optimizer_steps=steps_done,
                model_revision=model_lock.revision,
            )
        from .runner_support import step_optimizer

        losses.append(
            step_optimizer(
                model,
                optimizer_instance,
                encoded,
                scaler=scaler,
                model_kwargs={"return_loss": True},
                max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            )
        )
        steps_done += 1
        if not _worker_should_continue(config):
            return _yield_training_checkpoint(
                config=config,
                model=model,
                optimizer_instance=optimizer_instance,
                scaler=scaler,
                optimizer_steps=steps_done,
                model_revision=model_lock.revision,
            )
    total_steps = steps_done
    initial_probe_steps = steps
    resumed = isinstance(resume_payload, bytes)
    resume_steps = 0
    if config.get("phase") == "probe":
        first_checkpoint = serialize_checkpoint(
            model=model,
            optimizer_instance=optimizer_instance,
            optimizer_steps=total_steps,
            identity=checkpoint_identity(config),
            model_revision=model_lock.revision,
            precision="float16-autocast",
            scaler=scaler,
        )
        del model, optimizer_instance, scaler
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        model = CLIPModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32).to(device)
        model.train()
        optimizer_instance = optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        resumed_step = restore_checkpoint(
            first_checkpoint,
            model=model,
            optimizer_instance=optimizer_instance,
            expected_identity=checkpoint_identity(config),
            expected_revision=model_lock.revision,
            device=device,
            scaler=scaler,
        )
        if resumed_step != total_steps:
            raise ValueError("CLIP checkpoint resumed at the wrong optimizer step")
        if not _worker_should_continue(config):
            return _yield_training_checkpoint(
                config=config,
                model=model,
                optimizer_instance=optimizer_instance,
                scaler=scaler,
                optimizer_steps=total_steps,
                model_revision=model_lock.revision,
            )
        resumed = True
        resume_steps = 1
        losses.append(
            step_optimizer(
                model,
                optimizer_instance,
                encoded,
                scaler=scaler,
                model_kwargs={"return_loss": True},
                max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            )
        )
        total_steps += 1
        if not _worker_should_continue(config):
            return _yield_training_checkpoint(
                config=config,
                model=model,
                optimizer_instance=optimizer_instance,
                scaler=scaler,
                optimizer_steps=total_steps,
                model_revision=model_lock.revision,
            )
    initial_weight_sha, final_weight_sha = assert_weights_changed(initial_weights, model)
    checkpoint_payload = serialize_checkpoint(
        model=model,
        optimizer_instance=optimizer_instance,
        optimizer_steps=total_steps,
        identity=checkpoint_identity(config),
        model_revision=model_lock.revision,
        precision="float16-autocast",
        scaler=scaler,
    )
    checkpoint_sha = sha256(checkpoint_payload).hexdigest()
    measurements = resource_measurements(started, optimizer_steps=total_steps)
    measurements.update(
        {
            "resolution": 224,
            "micro_batch": micro_batch,
            "gradient_accumulation_steps": 1,
            "contrastive_config_version": config["contrastive_config_version"],
            "pair_count": pair_count,
            "initial_optimizer_steps": initial_probe_steps,
            "resume_steps": resume_steps,
            "checkpoint_resumed": resumed,
            "losses": losses,
            "initial_weight_sha256": initial_weight_sha,
            "final_weight_sha256": final_weight_sha,
            "model_id": model_lock.model_id,
            "model_revision": model_lock.revision,
            "source_manifest_kind": identity["input_kind"],
            "s3_object_store_verified": object_store is not None,
        }
    )
    result_payload, result_sha = model_bundle(
        model,
        output_uri=output_uri,
        metrics={"status": "succeeded", "resource_measurements": measurements},
    )
    checkpoint_uri = None
    result_uri = None
    commit_checkpoint = config.get("_commit_checkpoint")
    if callable(commit_checkpoint):
        checkpoint_uri = commit_checkpoint(checkpoint_payload).uri
        checkpoint_payload = b""
    commit_result = config.get("_commit_result_artifact")
    if callable(commit_result):
        result_uri = commit_result("model", result_payload).uri
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


def _pairs_from_manifest(
    manifest: dict[str, Any],
    config: dict[str, Any],
    micro_batch: int,
) -> tuple[list[dict[str, Any]], Any]:
    from .data import dataset_object_store_from_environment

    phase = config["phase"]
    raw_pairs = manifest.get("pairs") if phase == "probe" else None
    if raw_pairs is None:
        raw_items = [
            item for item in manifest.get("items", [])
            if item.get("kind") == "crop" and item.get("split") == "train"
        ]
        if not raw_items:
            raise ValueError("CLIP manifest has no training crop-caption pairs")
        texts = [str(item.get("snapshot", {}).get("caption", {}).get("text", "")).strip() for item in raw_items]
        unique: list[tuple[dict[str, Any], str]] = []
        seen_ids: set[str] = set()
        seen_texts: set[str] = set()
        for item, text in zip(raw_items, texts, strict=True):
            item_id = str(item.get("item_id", ""))
            if item_id and text and item_id not in seen_ids and text not in seen_texts:
                unique.append((item, text))
                seen_ids.add(item_id)
                seen_texts.add(text)
        selected = unique[:micro_batch]
        positive_texts = [text for _item, text in selected]
        raw_pairs = [
            {
                "image_id": str(item["item_id"]),
                "text": text,
                "negative_texts": [other for other in positive_texts if other != text],
                "media": item,
            }
            for item, text in selected
        ]
    if not isinstance(raw_pairs, list):
        raise ValueError("CLIP manifest has no explicit pair list")
    selected_pairs = raw_pairs[:micro_batch]
    if len(selected_pairs) < micro_batch:
        raise ValueError("CLIP manifest does not contain the complete versioned contrastive micro-batch")
    media_items = [pair.get("media", pair) for pair in selected_pairs]
    object_store = None
    if any(not isinstance(item.get("image_path"), str) for item in media_items):
        object_store = dataset_object_store_from_environment()
    pairs = []
    for pair, media in zip(selected_pairs, media_items, strict=True):
        image = load_rgb_image(media, object_store=object_store)
        pairs.append({**pair, "image": image})
    return pairs, object_store


def _worker_should_continue(config: dict[str, Any]) -> bool:
    check = config.get("_assert_current")
    return bool(check()) if callable(check) else True


def _yield_training_checkpoint(
    *,
    config: dict[str, Any],
    model: Any,
    optimizer_instance: Any,
    scaler: Any,
    optimizer_steps: int,
    model_revision: str,
) -> dict[str, Any]:
    from .runner_support import checkpoint_identity, serialize_checkpoint

    payload = serialize_checkpoint(
        model=model,
        optimizer_instance=optimizer_instance,
        optimizer_steps=optimizer_steps,
        identity=checkpoint_identity(config),
        model_revision=model_revision,
        precision="float16-autocast",
        scaler=scaler,
    )
    commit = config.get("_commit_checkpoint")
    if callable(commit):
        commit(payload)
        payload = b""
    return {"status": "yielded", "checkpoint_payload": payload}
