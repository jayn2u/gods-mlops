from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module, util

import pytest


def _live_probe_api():
    name = "gods_mlops.jobs.live_probe"
    assert util.find_spec(name) is not None, "the committed live probe harness is missing"
    return import_module(name)


def test_proc_start_ticks_handles_spaces_and_parentheses_in_the_process_name() -> None:
    stat = "123 (python worker (cuda)) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 987654"
    assert _live_probe_api().proc_start_ticks(stat) == 987654


def test_proc_start_ticks_rejects_incomplete_process_records() -> None:
    with pytest.raises(ValueError, match="start time"):
        _live_probe_api().proc_start_ticks("123 (python) S 1 2")


def test_cached_training_image_runs_the_requested_python_command() -> None:
    module = _live_probe_api()
    assert hasattr(module, "python_entrypoint_command"), "the image entrypoint override is missing"
    assert module.python_entrypoint_command(
        "gods-mlops-training:task1", ["-m", "gods_mlops.jobs.live_probe", "worker"]
    ) == ["--entrypoint", "python", "gods-mlops-training:task1", "-m", "gods_mlops.jobs.live_probe", "worker"]


def test_worker_and_contender_program_need_no_control_plane_packages() -> None:
    module = _live_probe_api()
    assert hasattr(module, "WORKER_PROGRAM"), "the cached-image worker program is missing"
    assert "import torch" in module.WORKER_PROGRAM
    assert "asyncpg" not in module.WORKER_PROGRAM
    assert "gods_mlops" not in module.WORKER_PROGRAM
    if util.find_spec("torch") is None:
        pytest.skip("local venv omits torch; the committed harness runs this exact preflight in the cached image")
    blocker = """
import importlib.abc, sys
class BlockControlPlane(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'asyncpg' or fullname.startswith('gods_mlops'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockControlPlane())
"""
    result = subprocess.run(
        [sys.executable, "-c", blocker + module.WORKER_PROGRAM, "preflight"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["event"] == "preflight"


def test_non_gpu_preflight_runs_the_committed_worker_program_without_gpu_flags() -> None:
    module = _live_probe_api()
    assert hasattr(module, "training_image_preflight_command"), "the image runtime preflight helper is missing"
    assert module.training_image_preflight_command("gods-mlops-training:task1") == [
        "--entrypoint", "python", "gods-mlops-training:task1", "-c", module.WORKER_PROGRAM, "preflight"
    ]
