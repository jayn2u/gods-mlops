"""CLIP training contracts and real contrastive runner."""

from __future__ import annotations

from hashlib import sha256
from time import perf_counter
from typing import Any

from .claims import WorkerYieldRequested
from .contracts import load_manifest, locked_model, validate_manifest_identity
from .data import load_rgb_image


def validate_contrastive_pairs(
    manifest: dict[str, Any], *, require_negatives: bool = True
) -> int:
    """Validate distinct positives and optionally require their in-batch negatives."""
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
        if require_negatives and (not isinstance(negative_texts, list) or not negative_texts):
            raise ValueError("CLIP training requires explicit negative examples for each pair")
        negatives = (
            {str(item).strip() for item in negative_texts if str(item).strip()}
            if isinstance(negative_texts, list)
            else set()
        )
        if text in negatives:
            raise ValueError("a CLIP negative example cannot equal its positive text")
        image_ids.add(image_id)
        positive_texts.add(text)
        negatives_by_image[image_id] = negatives
        positive_by_image[image_id] = text
    if require_negatives:
        for image_id, negatives in negatives_by_image.items():
            positive = positive_by_image[image_id]
            if not any(text in positive_texts and text != positive for text in negatives):
                raise ValueError(f"CLIP pair {image_id} has no explicit cross-example negative")
    return len(pairs)


def clip_evaluation_sources(
    manifest: dict[str, Any], *, evaluation_split: str
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Resolve the frozen human query set and exact reviewed test gallery."""
    if evaluation_split != "test":
        raise ValueError("CLIP retrieval evaluation is pinned to the frozen test gallery")
    evaluation = manifest.get("evaluation")
    items = manifest.get("items")
    if not isinstance(evaluation, dict) or not isinstance(items, list):
        raise ValueError("CLIP manifest has no frozen evaluation truth and item list")
    gallery_ids = evaluation.get("gallery_crop_ids")
    matrices = evaluation.get("relevance_matrices")
    if not isinstance(gallery_ids, list) or not gallery_ids or any(
        not isinstance(item, str) or not item for item in gallery_ids
    ):
        raise ValueError("CLIP evaluation gallery IDs are missing")
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("CLIP evaluation gallery has duplicate crop IDs")
    if not isinstance(matrices, list) or not matrices:
        raise ValueError("CLIP evaluation human relevance matrices are missing")
    matrices_by_query: dict[str, dict[str, Any]] = {}
    for matrix in matrices:
        if not isinstance(matrix, dict):
            raise ValueError("CLIP human relevance matrix is invalid")
        query_id = matrix.get("query_id")
        query_text = matrix.get("query_text")
        if not isinstance(query_id, (str, int)) or not str(query_id).strip():
            raise ValueError("CLIP human relevance query ID is missing")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("CLIP frozen human query text is missing")
        query_id = str(query_id)
        if query_id in matrices_by_query:
            raise ValueError("CLIP human relevance matrices contain a duplicate query")
        if matrix.get("status") != "complete" or matrix.get("evaluation_eligible") is not True:
            raise ValueError("CLIP human relevance truth is unresolved")
        if matrix.get("gallery_matches_selection") is not True:
            raise ValueError("CLIP human relevance gallery differs from the frozen selection")
        judgments = matrix.get("judgments")
        if not isinstance(judgments, list):
            raise ValueError("CLIP human relevance judgments are missing")
        judgments_by_crop = {
            str(row.get("crop_id")): row.get("judgment")
            for row in judgments
            if isinstance(row, dict) and row.get("crop_id") is not None
        }
        if len(judgments_by_crop) != len(judgments) or set(judgments_by_crop) != set(gallery_ids):
            raise ValueError("CLIP human relevance matrix does not cover the frozen gallery")
        values = set(judgments_by_crop.values())
        if "uncertain" in values or not {"relevant", "not_relevant"} <= values:
            raise ValueError("CLIP human relevance truth needs resolved positive and negative judgments")
        matrices_by_query[query_id] = {"query_id": query_id, "query_text": query_text}

    items_by_id = {
        str(item.get("item_id")): item
        for item in items
        if isinstance(item, dict)
        and (item.get("item_kind") == "crop" or item.get("kind") == "crop")
        and item.get("split") == "test"
    }
    if not set(gallery_ids).issubset(items_by_id):
        raise ValueError("CLIP frozen gallery references a crop outside the test split")
    return (
        [matrices_by_query[query_id] for query_id in sorted(matrices_by_query)],
        [items_by_id[crop_id] for crop_id in sorted(gallery_ids)],
    )


def encode_evaluation_batch(
    model: Any,
    processor: Any,
    *,
    texts: list[str],
    images: list[Any],
    device: Any,
) -> tuple[list[list[float]], list[list[float]]]:
    """Encode frozen text queries and gallery crops in inference-only mode."""
    import torch
    import torch.nn.functional as functional

    if not texts and not images:
        raise ValueError("CLIP evaluation batch must contain queries or gallery images")
    model.eval()
    with torch.inference_mode():
        text_features: list[list[float]] = []
        if texts:
            encoded_text = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            encoded_text = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded_text.items()
            }
            features = model.get_text_features(**encoded_text)
            text_features = functional.normalize(features.float(), p=2, dim=-1).cpu().tolist()
        image_features: list[list[float]] = []
        if images:
            encoded_images = processor(images=images, return_tensors="pt")
            encoded_images = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in encoded_images.items()
            }
            features = model.get_image_features(**encoded_images)
            image_features = functional.normalize(features.float(), p=2, dim=-1).cpu().tolist()
    return text_features, image_features


def run(config: dict[str, Any], manifest_uri: str, output_uri: str) -> dict[str, Any]:
    """Train CLIP or run fenced inference against an immutable evaluation gallery."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    from .data import dataset_object_store_from_environment
    from .runner_support import (
        assert_weights_changed,
        cache_directory,
        checkpoint_identity,
        model_bundle,
        optimizer,
        OptimizerStepStats,
        require_cuda,
        resource_measurements,
        restore_checkpoint,
        selected_trainable_weights,
        serialize_checkpoint,
    )

    phase = str(config.get("phase", ""))
    target_phase = str(config.get("target_phase") or phase)
    if phase not in {"probe", "training", "evaluation"}:
        raise ValueError("CLIP runner supports probe, training, and evaluation phases")
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
    if phase != "evaluation" and target_phase != "evaluation" and steps < 3:
        raise ValueError("CLIP measurement requires at least three optimizer steps")

    manifest = load_manifest(manifest_uri, config=config)
    identity = validate_manifest_identity(config, manifest)
    if phase == "evaluation" or (phase == "probe" and target_phase == "evaluation"):
        started = perf_counter()
        model_lock = locked_model("clip")
        model_path = cache_directory("clip", config)
        device = require_cuda()
        torch.cuda.reset_peak_memory_stats(0)
        processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        model = CLIPModel.from_pretrained(model_path, local_files_only=True, dtype=torch.float32).to(device)
        config["_device"] = device
        if phase == "evaluation":
            from gods_mlops.evaluation.eligibility import apply_verified_model_weights

            apply_verified_model_weights(
                model,
                config,
                model_kind="clip",
                model_revision=model_lock.revision,
            )
            queries, gallery_items = clip_evaluation_sources(
                manifest,
                evaluation_split=str(config.get("evaluation_split", "test")),
            )
            object_store = dataset_object_store_from_environment()
            gallery_images = [
                load_rgb_image(item, object_store=object_store) for item in gallery_items
            ]
        else:
            pairs, object_store = _pairs_from_manifest(manifest, config, micro_batch)
            queries = [
                {"query_id": str(pair["image_id"]), "query_text": str(pair["text"])}
                for pair in pairs
            ]
            gallery_items = [pair.get("media", pair) for pair in pairs]
            gallery_images = [pair["image"] for pair in pairs]
        return _run_clip_evaluation(
            config,
            model=model,
            processor=processor,
            queries=queries,
            gallery_items=gallery_items,
            gallery_images=gallery_images,
            identity=identity,
            model_revision=model_lock.revision,
            started=started,
        )

    pairs, object_store = _pairs_from_manifest(manifest, config, micro_batch)
    pair_count = validate_contrastive_pairs(
        {
            "contrastive_config_version": config.get("contrastive_config_version"),
            "pairs": pairs,
        },
        require_negatives=config.get("phase") == "probe",
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
    step_stats = OptimizerStepStats()
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
            step_stats=step_stats,
        )
    probe_encoded = (
        _encode_contrastive_batch(pairs, processor=processor, device=device)
        if config.get("phase") == "probe"
        else None
    )
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
                step_stats=step_stats,
            )
        if config.get("phase") == "probe":
            encoded = probe_encoded
        else:
            batch = _contrastive_batch(pairs, micro_batch=micro_batch, optimizer_step=steps_done)
            validate_contrastive_pairs(
                {
                    "contrastive_config_version": config.get("contrastive_config_version"),
                    "pairs": batch,
                }
            )
            encoded = _encode_contrastive_batch(
                batch, processor=processor, device=device, object_store=object_store
            )
        from .runner_support import step_optimizer

        try:
            loss = step_optimizer(
                model,
                optimizer_instance,
                encoded,
                scaler=scaler,
                model_kwargs={"return_loss": True},
                max_grad_norm=float(config.get("max_grad_norm", 1.0)),
                step_stats=step_stats,
                before_attempt=lambda: _require_optimizer_attempt(config),
            )
        except WorkerYieldRequested:
            return _yield_training_checkpoint(
                config=config,
                model=model,
                optimizer_instance=optimizer_instance,
                scaler=scaler,
                optimizer_steps=steps_done,
                model_revision=model_lock.revision,
                step_stats=step_stats,
            )
        losses.append(loss)
        steps_done += 1
        if not _worker_should_continue(config):
            return _yield_training_checkpoint(
                config=config,
                model=model,
                optimizer_instance=optimizer_instance,
                scaler=scaler,
                optimizer_steps=steps_done,
                model_revision=model_lock.revision,
                step_stats=step_stats,
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
            step_stats=step_stats,
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
            step_stats=step_stats,
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
                step_stats=step_stats,
            )
        try:
            loss = step_optimizer(
                model,
                optimizer_instance,
                probe_encoded,
                scaler=scaler,
                model_kwargs={"return_loss": True},
                max_grad_norm=float(config.get("max_grad_norm", 1.0)),
                step_stats=step_stats,
                before_attempt=lambda: _require_optimizer_attempt(config),
            )
        except WorkerYieldRequested:
            return _yield_training_checkpoint(
                config=config,
                model=model,
                optimizer_instance=optimizer_instance,
                scaler=scaler,
                optimizer_steps=total_steps,
                model_revision=model_lock.revision,
                step_stats=step_stats,
            )
        resumed = True
        resume_steps = 1
        losses.append(loss)
        total_steps += 1
        if not _worker_should_continue(config):
            return _yield_training_checkpoint(
                config=config,
                model=model,
                optimizer_instance=optimizer_instance,
                scaler=scaler,
                optimizer_steps=total_steps,
                model_revision=model_lock.revision,
                step_stats=step_stats,
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
        step_stats=step_stats,
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
            "training_examples_consumed": max(0, total_steps - start_step) * micro_batch,
            "next_batch_offset": (total_steps * micro_batch) % pair_count,
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
            "optimizer_step_attempts": step_stats.attempts,
            "amp_overflow_skips": step_stats.amp_overflow_skips,
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


def _run_clip_evaluation(
    config: dict[str, Any],
    *,
    model: Any,
    processor: Any,
    queries: list[dict[str, str]],
    gallery_items: list[dict[str, Any]],
    gallery_images: list[Any],
    identity: dict[str, Any],
    model_revision: str,
    started: float,
) -> dict[str, Any]:
    import torch

    from gods_mlops.datasets.manifest import canonical_json
    from gods_mlops.evaluation.report import decode_evaluation_cursor, encode_evaluation_cursor
    from .runner_support import resource_measurements

    phase = str(config.get("phase"))
    if len(gallery_items) != len(gallery_images):
        raise ValueError("CLIP gallery item and image counts do not match")
    if not queries or not gallery_items:
        raise ValueError("CLIP evaluation requires frozen queries and gallery crops")
    batch_size = config.get("evaluation_batch_size", config.get("micro_batch", 2))
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("CLIP evaluation batch size must be a positive integer")
    query_ids = [str(item["query_id"]) for item in queries]
    gallery_ids = [str(item["item_id"]) for item in gallery_items]
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("CLIP evaluation query and crop IDs must be unique")

    state: dict[str, Any] = {
        "stage": "queries",
        "query_embeddings": {},
        "crop_embeddings": {},
    }
    start_index = 0
    resume_payload = config.get("_evaluation_resume_payload")
    if phase == "evaluation" and isinstance(resume_payload, bytes):
        cursor = decode_evaluation_cursor(
            resume_payload,
            expected_identity=config["_evaluation_job_identity"],
            expected_candidate_checkpoint_sha256=str(config["_evaluation_checkpoint_sha256"]),
            expected_manifest_sha256=str(config["input_sha256"]),
            expected_config_sha256=str(config["config_sha256"]),
        )
        state = cursor["predictions"]
        start_index = cursor["next_index"]
        if state.get("stage") not in {"queries", "gallery"}:
            raise ValueError("CLIP evaluation cursor stage is invalid")

    inference_steps = 0
    stage_order = {"queries": 0, "gallery": 1, "complete": 2}
    for stage in ("queries", "gallery"):
        if stage_order[stage] < stage_order.get(state["stage"], -1):
            continue
        position = start_index if stage == state["stage"] else 0
        size = len(queries) if stage == "queries" else len(gallery_items)
        while position < size:
            if not _worker_should_continue(config):
                if phase != "evaluation":
                    return {"status": "yielded", "resource_measurements": {"optimizer_steps": 0}}
                cursor_payload = encode_evaluation_cursor(
                    identity=config["_evaluation_job_identity"],
                    candidate_checkpoint_sha256=str(config["_evaluation_checkpoint_sha256"]),
                    manifest_sha256=str(config["input_sha256"]),
                    config_sha256=str(config["config_sha256"]),
                    next_index=position,
                    predictions={**state, "stage": stage},
                )
                return {"status": "yielded", "checkpoint_payload": cursor_payload}
            stop = min(position + batch_size, size)
            if stage == "queries":
                text_batch = [item["query_text"] for item in queries[position:stop]]
                image_batch = []
            else:
                text_batch = []
                image_batch = gallery_images[position:stop]
            text_embeddings, image_embeddings = encode_evaluation_batch(
                model,
                processor,
                texts=text_batch,
                images=image_batch,
                device=config.get("_device", "cuda:0"),
            )
            if stage == "queries":
                state["query_embeddings"].update(
                    {
                        query_ids[index]: embedding
                        for index, embedding in zip(range(position, stop), text_embeddings, strict=True)
                    }
                )
            else:
                state["crop_embeddings"].update(
                    {
                        gallery_ids[index]: embedding
                        for index, embedding in zip(range(position, stop), image_embeddings, strict=True)
                    }
                )
            position = stop
            inference_steps += 1
        state["stage"] = "gallery" if stage == "queries" else "complete"
        start_index = 0

    if not _worker_should_continue(config):
        if phase != "evaluation":
            return {"status": "yielded", "resource_measurements": {"optimizer_steps": 0}}
        cursor_payload = encode_evaluation_cursor(
            identity=config["_evaluation_job_identity"],
            candidate_checkpoint_sha256=str(config["_evaluation_checkpoint_sha256"]),
            manifest_sha256=str(config["input_sha256"]),
            config_sha256=str(config["config_sha256"]),
            next_index=len(gallery_items),
            predictions={**state, "stage": "gallery"},
        )
        return {"status": "yielded", "checkpoint_payload": cursor_payload}

    measurements = resource_measurements(started, optimizer_steps=0, inference_steps=inference_steps)
    measurements.update(
        {
            "precision": "float32-inference",
            "model_id": identity["model_id"],
            "model_revision": model_revision,
            "evaluation_batch_size": batch_size,
        }
    )
    if phase == "probe":
        document = {
            "schema_version": 1,
            "model_kind": "clip",
            "input_id": identity["input_id"],
            "input_sha256": identity["input_sha256"],
            "query_count": len(queries),
            "gallery_count": len(gallery_items),
            "source_fixture": True,
            "human_relevance_truth_used": False,
        }
        payload = canonical_json(document)
        result_kind = "evaluation_probe"
    else:
        document = {
            "schema_version": 1,
            "model_kind": "clip",
            "input_id": identity["input_id"],
            "input_sha256": identity["input_sha256"],
            "query_embeddings": state["query_embeddings"],
            "crop_embeddings": state["crop_embeddings"],
        }
        payload = canonical_json(document)
        result_kind = "evaluation_predictions"
    return {
        "status": "succeeded",
        "checkpoint_uri": None,
        "hash": sha256(payload).hexdigest(),
        "result_uri": None,
        "result_artifact_kind": result_kind,
        "result_artifact_payload": payload,
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
        if len(unique) < micro_batch:
            raise ValueError("published CLIP input is smaller than its complete contrastive micro-batch")
        raw_pairs = [
            {
                "image_id": str(item["item_id"]),
                "text": text,
                "negative_texts": [],
                "media": item,
            }
            for item, text in unique
        ]
    if not isinstance(raw_pairs, list):
        raise ValueError("CLIP manifest has no explicit pair list")
    selected_pairs = raw_pairs[:micro_batch] if phase == "probe" else raw_pairs
    if len(selected_pairs) < micro_batch:
        raise ValueError("CLIP manifest does not contain the complete versioned contrastive micro-batch")
    media_items = [pair.get("media", pair) for pair in selected_pairs]
    object_store = None
    if any(not isinstance(item.get("image_path"), str) for item in media_items):
        object_store = dataset_object_store_from_environment()
    pairs = []
    for pair, media in zip(selected_pairs, media_items, strict=True):
        if phase == "probe":
            image = load_rgb_image(media, object_store=object_store)
            pairs.append({**pair, "image": image})
        else:
            pairs.append({**pair, "media": media})
    return pairs, object_store


def _contrastive_batch(
    pairs: list[dict[str, Any]], *, micro_batch: int, optimizer_step: int
) -> list[dict[str, Any]]:
    """Select a deterministic cyclic micro-batch; optimizer step is the resume cursor."""
    if micro_batch < 2 or optimizer_step < 0 or len(pairs) < micro_batch:
        raise ValueError("CLIP batch cursor requires a complete positive micro-batch")
    start = (optimizer_step * micro_batch) % len(pairs)
    selected = [pairs[(start + offset) % len(pairs)] for offset in range(micro_batch)]
    batch_texts = [pair["text"] for pair in selected]
    return [
        {**pair, "negative_texts": [text for text in batch_texts if text != pair["text"]]}
        for pair in selected
    ]


def _encode_contrastive_batch(
    pairs: list[dict[str, Any]], *, processor, device, object_store=None
) -> dict[str, Any]:
    images = [
        pair["image"]
        if "image" in pair
        else load_rgb_image(pair["media"], object_store=object_store)
        for pair in pairs
    ]
    encoded = processor(
        text=[pair["text"] for pair in pairs], images=images, return_tensors="pt", padding=True
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _worker_should_continue(config: dict[str, Any]) -> bool:
    check = config.get("_assert_current")
    return bool(check()) if callable(check) else True


def _require_optimizer_attempt(config: dict[str, Any]) -> None:
    if not _worker_should_continue(config):
        raise WorkerYieldRequested("Task 7 requested a cooperative CLIP training stop")


def _yield_training_checkpoint(
    *,
    config: dict[str, Any],
    model: Any,
    optimizer_instance: Any,
    scaler: Any,
    optimizer_steps: int,
    model_revision: str,
    step_stats: Any,
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
        step_stats=step_stats,
    )
    commit = config.get("_commit_checkpoint")
    if callable(commit):
        commit(payload)
        payload = b""
    return {"status": "yielded", "checkpoint_payload": payload}
