"""Shared execution helpers for the three locked model runners."""

from __future__ import annotations

import gc
import io
import json
import os
import subprocess
import tarfile
import tempfile
import time
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch

from gods_mlops.jobs.checkpoints import CheckpointIdentity
from gods_mlops.model_locks import model_cache_path, validate_model_cache
from .contracts import locked_model, model_lock_path
from .placement import validate_worker_placement


def require_cuda(expected_gpu_uuid: str | None = None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("training and model readiness probes require the admitted CUDA device")
    expected_gpu_uuid = expected_gpu_uuid or os.environ.get("GODS_MLOPS_GPU_UUID", "")
    node_name = os.environ.get("GODS_MLOPS_NODE_NAME", "")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("training worker cannot verify the admitted visible CUDA device") from error
    visible_uuids = []
    visible_names = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",", 1)]
        if len(fields) == 2:
            visible_uuids.append(fields[0])
            visible_names.append(fields[1])
    validate_worker_placement(
        node_name=node_name,
        admitted_gpu_uuid=expected_gpu_uuid,
        visible_gpu_uuids=visible_uuids,
        visible_gpu_names=visible_names,
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("training worker CUDA visibility does not match its one-GPU lease")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA RTX A6000":
        raise RuntimeError("PyTorch visible CUDA device is not the admitted NVIDIA RTX A6000")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def cache_directory(model_kind: str, config: dict[str, Any]) -> Path:
    cache_root = Path(
        config.get("model_cache_root")
        or os.environ.get("GODS_MLOPS_MODEL_CACHE_ROOT", "/mnt/model-cache")
    )
    model = locked_model(model_kind, lock_path=model_lock_path())
    directory = model_cache_path(cache_root, model)
    validate_model_cache(model, directory)
    return directory


def checkpoint_identity(config: dict[str, Any]) -> CheckpointIdentity:
    return CheckpointIdentity(
        job_id=str(config["job_id"]),
        input_kind=str(config["input_kind"]),
        input_id=str(config["input_id"]),
        input_sha256=str(config["input_sha256"]),
        phase=str(config["phase"]),
        model_kind=str(config["model_kind"]),
        config_version=str(config["config_version"]),
        config_sha256=str(config["config_sha256"]),
        dataset_version=str(config["dataset_version"]) if config.get("dataset_version") is not None else None,
    )


def optimizer(model: Any, config: dict[str, Any]):
    learning_rate = float(config.get("learning_rate", 1e-5))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if not 0 < learning_rate <= 1e-2 or not 0 <= weight_decay <= 1:
        raise ValueError("optimizer learning rate or weight decay is outside the supported range")
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def step_optimizer(
    model: Any,
    optimizer_instance: Any,
    inputs: dict[str, Any],
    *,
    scaler: Any,
    model_kwargs: dict[str, Any] | None = None,
    max_grad_norm: float = 1.0,
) -> float:
    optimizer_instance.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(**inputs, **(model_kwargs or {}))
        loss = output.loss
    if loss is None or not torch.isfinite(loss):
        raise ValueError("model training loss is not finite")
    scalar = float(loss.detach().float().cpu())
    if scalar <= 0:
        raise ValueError("model training produced no finite non-zero learning signal")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer_instance)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer_instance)
    scaler.update()
    torch.cuda.synchronize()
    return scalar


def cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def serialize_checkpoint(
    *,
    model: Any,
    optimizer_instance: Any,
    optimizer_steps: int,
    identity: CheckpointIdentity,
    model_revision: str,
    precision: str,
    scaler: Any,
) -> bytes:
    payload = io.BytesIO()
    torch.save(
        {
            "format": "gods-mlops-training-checkpoint-v1",
            "identity": identity.as_dict(),
            "model_revision": model_revision,
            "precision": precision,
            "optimizer_steps": optimizer_steps,
            "model_state_dict": cpu_state(model.state_dict()),
            "optimizer_state_dict": cpu_state(optimizer_instance.state_dict()),
            "scaler_state_dict": cpu_state(scaler.state_dict()),
        },
        payload,
    )
    return payload.getvalue()


def restore_checkpoint(
    payload: bytes,
    *,
    model: Any,
    optimizer_instance: Any,
    expected_identity: CheckpointIdentity,
    expected_revision: str,
    device: torch.device,
    scaler: Any,
) -> int:
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "gods-mlops-training-checkpoint-v1":
        raise ValueError("checkpoint format is unsupported")
    if checkpoint.get("identity") != expected_identity.as_dict():
        raise ValueError("checkpoint job, fence input, phase, model, or config identity changed")
    if checkpoint.get("model_revision") != expected_revision:
        raise ValueError("checkpoint model revision differs from the immutable lock")
    step = checkpoint.get("optimizer_steps")
    if not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint optimizer step is invalid")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer_instance.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    model.to(device)
    for state in optimizer_instance.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    return step


def assert_weights_changed(before: list[tuple[str, torch.Tensor]], model: Any) -> tuple[str, str]:
    current_by_name = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    current = [(name, current_by_name[name]) for name, _value in before if name in current_by_name]
    if len(current) != len(before) or not any(
        not torch.equal(old, new) for (_name, old), (_current_name, new) in zip(before, current)
    ):
        raise ValueError("optimizer steps did not update any trainable model weight")
    initial_digest = _tensor_digest(before)
    final_digest = _tensor_digest(current)
    if initial_digest == final_digest:
        raise ValueError("training weight fingerprint did not change")
    return initial_digest, final_digest


def selected_trainable_weights(model: Any, *, maximum: int = 4) -> list[tuple[str, torch.Tensor]]:
    selected = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and 1 < parameter.numel() <= 65_536:
            selected.append((name, parameter.detach().cpu().clone()))
        if len(selected) >= maximum:
            break
    if not selected:
        raise ValueError("model exposes no bounded trainable weights for optimizer verification")
    return selected


def _tensor_digest(tensors: list[tuple[str, torch.Tensor]]) -> str:
    digest = sha256()
    for name, tensor in tensors:
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def output_directory(output_uri: str) -> Path:
    parsed = urlsplit(output_uri)
    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path if parsed.scheme == "file" else output_uri)
    elif parsed.scheme == "s3":
        path = Path(tempfile.mkdtemp(prefix="gods-mlops-result-"))
    else:
        raise ValueError("runner output URI must be a local stage or S3 artifact prefix")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def write_json(path: Path, value: Any) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    path.write_bytes(encoded)
    return encoded


def model_bundle(model: Any, *, output_uri: str, metrics: dict[str, Any]) -> tuple[bytes, str]:
    import tarfile

    directory = output_directory(output_uri)
    model_directory = directory / "model"
    model_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    model.save_pretrained(model_directory, safe_serialization=True)
    write_json(directory / "metrics.json", metrics)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(directory).as_posix(), recursive=False)
    result = buffer.getvalue()
    return result, sha256(result).hexdigest()


def resource_measurements(started_at: float, *, optimizer_steps: int, inference_steps: int = 0) -> dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "peak_vram_allocated_mib": int(torch.cuda.max_memory_allocated(0) / 1024**2),
        "peak_vram_reserved_mib": int(torch.cuda.max_memory_reserved(0) / 1024**2),
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
        "optimizer_steps": optimizer_steps,
        "inference_steps": inference_steps,
        "precision": "float16-autocast",
    }
