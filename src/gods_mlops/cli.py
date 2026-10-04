"""Command-line tools for validating locks and preparing model caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from gods_mlops.infra_locks import load_image_lock
from gods_mlops.model_locks import (
    LockValidationError,
    load_model_lock,
    model_cache_path,
    prepare_model,
    validate_all_model_caches,
    validate_model_cache,
)


APP_ROOT = Path(os.environ.get("GODS_MLOPS_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_MODEL_LOCK = APP_ROOT / "models" / "lock.json"
DEFAULT_IMAGE_LOCK = APP_ROOT / "infra" / "versions.lock.yaml"
DEFAULT_CACHE_ROOT = Path("/mnt/model-cache")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gods-mlops")
    commands = parser.add_subparsers(dest="command", required=True)

    check_locks = commands.add_parser("check-locks", help="validate model and OCI image lock files")
    check_locks.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    check_locks.add_argument("--image-lock", type=Path, default=DEFAULT_IMAGE_LOCK)

    prepare = commands.add_parser("prepare-models", help="download and verify locked model files")
    prepare.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    prepare.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    prepare.add_argument("--model-id", help="prepare only one locked model")

    check_models = commands.add_parser("check-models", help="verify every required local model file")
    check_models.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    check_models.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    check_models.add_argument("--model-id", help="verify only one locked model")

    check_training = commands.add_parser(
        "check-training-image", help="import the pinned training and evaluation dependencies"
    )
    check_training.add_argument("--require-cuda", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "check-locks":
            models = load_model_lock(args.model_lock)
            images = load_image_lock(args.image_lock)
            print(f"lock structure valid: {len(models.models)} model revisions, {len(images.images)} images")
            print("model cache readiness: not checked")
            return 0

        if args.command == "prepare-models":
            model_set = load_model_lock(args.model_lock)
            selected = _select_models(model_set.models, args.model_id)
            for model in selected:
                path = prepare_model(model, args.cache_root)
                print(f"prepared and verified: {model.model_id}@{model.revision} -> {path}")
            return 0

        if args.command == "check-models":
            model_set = load_model_lock(args.model_lock)
            selected = _select_models(model_set.models, args.model_id)
            if args.model_id is None:
                validate_all_model_caches(model_set, args.cache_root)
            else:
                for model in selected:
                    validate_model_cache(model, model_cache_path(args.cache_root, model))
            print(f"model files ready: {len(selected)} locked model revisions")
            return 0

        if args.command == "check-training-image":
            return _check_training_image(require_cuda=args.require_cuda)
    except LockValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


def training_preflight_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gods-mlops-training-preflight")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    return _check_training_image(require_cuda=args.require_cuda)


def _select_models(models: tuple, requested_id: str | None) -> tuple:
    if requested_id is None:
        return models
    selected = tuple(model for model in models if model.model_id == requested_id)
    if not selected:
        raise LockValidationError(f"model is not present in the immutable lock: {requested_id}")
    return selected


def _check_training_image(*, require_cuda: bool) -> int:
    import numpy
    import torch
    import torchvision
    from PIL import __version__ as pillow_version
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from transformers import (
        CLIPModel,
        CLIPProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
        RTDetrImageProcessor,
        RTDetrV2ForObjectDetection,
    )

    cuda_available = torch.cuda.is_available()
    report = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": cuda_available,
        "numpy": numpy.__version__,
        "pillow": pillow_version,
        "evaluation": [COCO.__name__, COCOeval.__name__],
        "model_classes": [
            RTDetrV2ForObjectDetection.__name__,
            RTDetrImageProcessor.__name__,
            CLIPModel.__name__,
            CLIPProcessor.__name__,
            Qwen2_5_VLForConditionalGeneration.__name__,
            Qwen2_5_VLProcessor.__name__,
        ],
    }
    print(json.dumps(report, sort_keys=True))
    if require_cuda and not cuda_available:
        print("ERROR: CUDA is required for this training preflight", file=sys.stderr)
        return 1
    return 0
