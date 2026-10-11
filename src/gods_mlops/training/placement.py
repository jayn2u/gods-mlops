"""Runtime placement checks for the admitted Ubuntu A6000 worker."""

from __future__ import annotations

from typing import Sequence


def validate_worker_placement(
    *,
    node_name: str,
    admitted_gpu_uuid: str,
    visible_gpu_uuids: Sequence[str],
    visible_gpu_names: Sequence[str],
) -> None:
    """Fail closed unless this process sees the exact single admitted RTX A6000."""
    if node_name != "ubuntu":
        raise ValueError("training worker must run on the admitted Ubuntu node")
    if not admitted_gpu_uuid.startswith("GPU-"):
        raise ValueError("admitted GPU UUID is unavailable")
    if len(visible_gpu_uuids) != 1 or len(visible_gpu_names) != 1:
        raise ValueError("training worker must see exactly one admitted GPU")
    if visible_gpu_uuids[0].strip() != admitted_gpu_uuid:
        raise ValueError("visible CUDA GPU UUID differs from the admitted lease")
    if visible_gpu_names[0].strip() != "NVIDIA RTX A6000":
        raise ValueError("visible CUDA device is not the admitted NVIDIA RTX A6000")
