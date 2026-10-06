"""Real-model readiness probe dispatcher using explicit non-dataset inputs."""

from __future__ import annotations

from typing import Any

from .contracts import load_manifest, validate_manifest_identity


def run(config: dict[str, Any], manifest_uri: str, output_uri: str) -> dict[str, Any]:
    """Dispatch one locked model probe; probe fixtures never become published datasets."""
    if config.get("phase") != "probe" or config.get("input_kind") != "probe_input":
        raise ValueError("readiness probes require an explicit probe_input job identity")
    manifest = load_manifest(manifest_uri, config=config)
    validate_manifest_identity(config, manifest)
    model_kind = config.get("model_kind")
    target_phase = config.get("target_phase")
    if model_kind == "detr":
        from .detector import run as run_detector

        return run_detector(config, manifest_uri, output_uri)
    if model_kind == "clip" and target_phase in {"training", "evaluation"}:
        from .clip import run as run_clip

        return run_clip(config, manifest_uri, output_uri)
    if model_kind == "qwen" and target_phase == "preparation":
        from .caption import run as run_caption

        return run_caption(config, manifest_uri, output_uri)
    raise ValueError("probe model/phase combination is unsupported")
