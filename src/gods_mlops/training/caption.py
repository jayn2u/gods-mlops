"""Bounded, inference-only Qwen2.5-VL caption draft generation."""

from __future__ import annotations

from hashlib import sha256
from math import sqrt
from time import perf_counter
from typing import Any

from PIL import Image
from .claims import WorkerYieldRequested

from .contracts import load_manifest, locked_model, validate_manifest_identity
from .data import load_rgb_image

MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 128
MAX_IMAGE_PIXELS = 1_048_576
_PROMPT = (
    "Describe only visible clothing, colors, carried items, and pose. "
    "Do not guess identity, name, age, occupation, or any detail hidden by the image."
)


def validate_generation_config(config: dict[str, Any]) -> dict[str, int]:
    """Keep the caption model's image and token budgets within the approved bound."""
    limits = {
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_image_pixels": MAX_IMAGE_PIXELS,
    }
    values = {}
    for name, limit in limits.items():
        value = config.get(name, limit)
        if not isinstance(value, int) or value < 1 or value > limit:
            raise ValueError(f"{name} must be between 1 and {limit}")
        values[name] = value
    if len(_PROMPT) > int(config.get("max_prompt_chars", 8192)):
        raise ValueError("caption prompt exceeds its versioned character bound")
    return values


def run(config: dict[str, Any], manifest_uri: str, output_uri: str) -> dict[str, Any]:
    """Generate English drafts from immutable reviewed crops; this runner never trains Qwen."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

    from .data import dataset_object_store_from_environment
    from .runner_support import cache_directory, output_directory, require_cuda, resource_measurements, write_json

    bounds = validate_generation_config(config)
    if config.get("model_kind") != "qwen" or config.get("target_phase", "preparation") != "preparation":
        raise ValueError("Qwen caption generation is preparation inference only")
    manifest = load_manifest(manifest_uri, config=config)
    identity = validate_manifest_identity(config, manifest)
    model_lock = locked_model("qwen")
    model_path = cache_directory("qwen", config)
    items = _caption_items(manifest, config)
    object_store = _object_store_if_needed(items)
    images = [
        _bound_image(load_rgb_image(item, object_store=object_store), bounds["max_image_pixels"])
        for item in items
    ]
    device = require_cuda()
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats(0)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    drafts = []
    max_input_seen = 0
    with torch.inference_mode():
        for item, image in zip(items, images, strict=True):
            if not _worker_should_continue(config):
                raise WorkerYieldRequested("Task 7 requested a cooperative Qwen caption stop")
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": _PROMPT}],
                }
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
                truncation=True,
                max_length=bounds["max_input_tokens"],
            )
            if inputs.input_ids.shape[-1] > bounds["max_input_tokens"]:
                raise ValueError("Qwen prompt exceeded max_input_tokens after processor truncation")
            max_input_seen = max(max_input_seen, int(inputs.input_ids.shape[-1]))
            inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
            generated = model.generate(**inputs, max_new_tokens=bounds["max_new_tokens"], do_sample=False)
            trimmed = [output[len(source) :] for source, output in zip(inputs["input_ids"], generated, strict=True)]
            caption = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            if not caption:
                raise ValueError("Qwen returned an empty caption draft")
            if not _worker_should_continue(config):
                raise WorkerYieldRequested("Task 7 requested a cooperative Qwen caption stop")
            drafts.append(
                {
                    "item_id": str(item.get("item_id", item.get("crop_id", ""))),
                    "caption": caption,
                    "review_required": True,
                }
            )
    result_document = {
        "schema_version": 1,
        "model_kind": "qwen",
        "model_id": model_lock.model_id,
        "model_revision": model_lock.revision,
        "config_version": config["config_version"],
        "input_id": identity["input_id"],
        "input_sha256": identity["input_sha256"],
        "drafts": drafts,
    }
    output_path = output_directory(output_uri)
    encoded = write_json(output_path / "caption-drafts.json", result_document)
    digest = sha256(encoded).hexdigest()
    measurements = resource_measurements(started, optimizer_steps=0, inference_steps=len(drafts))
    measurements.update(
        {
            "max_input_tokens": bounds["max_input_tokens"],
            "max_new_tokens": bounds["max_new_tokens"],
            "max_image_pixels": bounds["max_image_pixels"],
            "max_input_tokens_observed": max_input_seen,
            "model_id": model_lock.model_id,
            "model_revision": model_lock.revision,
        }
    )
    result_uri = None
    commit_artifact = config.get("_commit_result_artifact")
    if callable(commit_artifact):
        result_uri = commit_artifact("caption_drafts", encoded, measurements).uri
        encoded = b""
    return {
        "status": "succeeded",
        "checkpoint_uri": None,
        "hash": digest,
        "result_uri": result_uri,
        "result_artifact_kind": "caption_drafts",
        "result_artifact_payload": encoded,
        "resource_measurements": measurements,
    }


def _caption_items(manifest: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Qwen caption input must be a typed non-empty crop batch")
    if any(
        not isinstance(item, dict)
        or not (item.get("item_kind") == "crop" or (item.get("kind") == "crop" and item.get("split") == "train"))
        for item in items
    ):
        raise ValueError("Qwen caption input must contain only eligible crop items")
    selected = list(items)
    limit = int(config.get("max_draft_images", 1))
    if limit < 1:
        raise ValueError("max_draft_images must be positive")
    if len(selected) > limit:
        raise ValueError("Qwen caption batch exceeds its versioned max_draft_images bound")
    return selected


def _object_store_if_needed(items: list[dict[str, Any]]):
    if any(not isinstance(item.get("image_path"), str) for item in items):
        return dataset_object_store_from_environment()
    return None


def _bound_image(image: Image.Image, maximum_pixels: int) -> Image.Image:
    if image.width * image.height <= maximum_pixels:
        return image
    scale = sqrt(maximum_pixels / (image.width * image.height))
    width = max(1, int(image.width * scale))
    height = max(1, int(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _worker_should_continue(config: dict[str, Any]) -> bool:
    check = config.get("_assert_current")
    return bool(check()) if callable(check) else True
