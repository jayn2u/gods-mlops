from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

TEST_TRAINING_IMAGE = "registry.example/gods-mlops-training@sha256:" + "a" * 64


def _require(module_name: str, symbol: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(f"missing implementation module {module_name}: {error.name}", pytrace=False)
    value = getattr(module, symbol, None)
    assert value is not None, f"{module_name}.{symbol} is part of the KFP contract"
    return value


def _compiled_pipeline(destination: Path) -> tuple[dict, str]:
    serialized = destination.read_text(encoding="utf-8")
    documents = list(yaml.safe_load_all(serialized))
    pipeline = next(item for item in documents if isinstance(item, dict) and "pipelineInfo" in item)
    return pipeline, serialized


def test_published_training_pipeline_compiles_cpu_admission_and_sequential_models(tmp_path: Path) -> None:
    compile_train = _require("pipelines.train", "compile_pipeline")

    destination = tmp_path / "train.yaml"
    compile_train(destination, training_image=TEST_TRAINING_IMAGE)
    pipeline, _serialized = _compiled_pipeline(destination)
    tasks = pipeline["root"]["dag"]["tasks"]

    assert set(tasks) >= {"run-training-job", "run-training-job-2"}
    for task_name in ("run-training-job", "run-training-job-2"):
        assert "nvidia.com/gpu" not in json.dumps(tasks[task_name])
        assert "accelerator" not in json.dumps(tasks[task_name])
    assert tasks["run-training-job-2"]["dependentTasks"] == ["run-training-job"]


def test_preparation_pipeline_compiles_one_typed_cpu_batch_without_published_dataset_step(tmp_path: Path) -> None:
    compile_prepare = _require("pipelines.prepare", "compile_pipeline")

    destination = tmp_path / "prepare.yaml"
    compile_prepare(destination, training_image=TEST_TRAINING_IMAGE)
    pipeline, _serialized = _compiled_pipeline(destination)
    tasks = pipeline["root"]["dag"]["tasks"]
    serialized = json.dumps(tasks)

    assert "run-preparation-job" in tasks
    assert "dataset_version" not in serialized
    assert "nvidia.com/gpu" not in json.dumps(tasks["run-preparation-job"])


def test_pipeline_components_keep_credentials_and_lease_tokens_out_of_parameters_and_logs(
    tmp_path: Path,
) -> None:
    compile_train = _require("pipelines.train", "compile_pipeline")
    destination = tmp_path / "train.yaml"

    compile_train(destination, training_image=TEST_TRAINING_IMAGE)

    pipeline, serialized = _compiled_pipeline(destination)
    spec = pipeline["root"]["dag"]["tasks"]
    assert "GODS_MLOPS_DATABASE_URL" not in json.dumps(pipeline["root"]["inputDefinitions"])
    assert "GODS_MLOPS_LEASE_TOKEN" not in serialized
    assert "password" not in serialized.lower()
    assert "gods-mlops-ingestion-credentials" in serialized
    assert "DATABASE_URL" in serialized
    assert "dataset_version" in json.dumps(pipeline["root"]["inputDefinitions"])
    assert "config_version" in json.dumps(pipeline["root"]["inputDefinitions"])
    assert "nvidia.com/gpu" not in json.dumps(spec)


def test_training_cli_import_is_safe_for_cpu_controller_process() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gods_mlops.training.cli; "
            "import gods_mlops.training.controller; import gods_mlops.training.worker; "
            "import gods_mlops.training.docker_probe; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_prepare_pipeline_uses_reviewed_crop_refs_not_dataset_or_clip_drafts(tmp_path: Path) -> None:
    compile_prepare = _require("pipelines.prepare", "compile_pipeline")
    destination = tmp_path / "prepare.yaml"

    compile_prepare(destination, training_image=TEST_TRAINING_IMAGE)

    pipeline, serialized = _compiled_pipeline(destination)
    inputs = json.dumps(pipeline["root"]["inputDefinitions"])
    assert "source_selections_json" in inputs
    assert "dataset_version" not in inputs
    assert "clip" not in serialized.lower()
    assert "GODS_MLOPS_LABEL_STUDIO_BBOX_PROJECT_ID" in serialized
    assert "GODS_MLOPS_LABEL_STUDIO_CAPTION_PROJECT_ID" in serialized
    assert "gods-label-studio-credentials" in serialized
    assert "LABEL_STUDIO_API_TOKEN" in serialized
    assert "MEDIA_CLEANUP_TOKEN" in serialized
