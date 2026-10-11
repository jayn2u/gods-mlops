from __future__ import annotations

import hashlib
import json

import pytest

from gods_mlops.jobs.models import ExecutionProfile, ProbeInput
from gods_mlops.training.probe_setup import candidate_profile, create_probe_input


CLIP_EXECUTION_CONFIG_VERSION = "task9-clip-224-microbatch2-symmetric-ce-fp64-probe-v1"
CLIP_MANIFEST_CONFIG_VERSION = "task8-clip-224-microbatch2-explicit-negative-probe-v1"
CLIP_INPUT_ID = "task8-clip-synthetic-probe-v1"
CLIP_INPUT_SHA256 = "b9f04dcf067c0dc4d53fa4b39bd88a991b9f956ca91590ba6d95a2293416731e"
CLIP_MANIFEST_KEY = f"probe-inputs/{CLIP_INPUT_ID}/manifest.json"
CLIP_MODEL_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"


class _ImmutableObjects:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    def write_immutable(self, *, object_key: str, content: bytes, sha256_digest: str, content_type: str) -> None:
        assert hashlib.sha256(content).hexdigest() == sha256_digest
        previous = self.values.get(object_key)
        if previous is not None and previous != content:
            raise AssertionError("an immutable probe object was rewritten")
        self.values[object_key] = content

    def read_source(self, *, object_key: str, sha256_digest: str, size_bytes: int) -> bytes:
        payload = self.values[object_key]
        assert len(payload) == size_bytes
        assert hashlib.sha256(payload).hexdigest() == sha256_digest
        return payload


def _registered_clip_profile(*, config: dict | None = None, config_sha256: str | None = None) -> dict:
    base = candidate_profile("clip")
    selected_config = dict(config if config is not None else base.config)
    selected = ExecutionProfile(
        model_kind="clip",
        config_version=base.config_version,
        phase="probe",
        target_phase="training",
        memory_requirement_mib=base.memory_requirement_mib,
        artifact_reservation_bytes=base.artifact_reservation_bytes,
        config=selected_config,
        candidate=True,
    )
    return {
        "phase": "probe",
        "model_kind": "clip",
        "config_version": selected.config_version,
        "config_sha256": config_sha256 or selected.config_sha256,
        "target_phase": "training",
        "profile_state": "candidate",
        "config_json": selected.config,
        "memory_requirement_mib": selected.memory_requirement_mib,
        "artifact_reservation_bytes": selected.artifact_reservation_bytes,
    }


def _worker_fixture(probe_input: ProbeInput, profile: dict) -> tuple[dict, dict, object]:
    from gods_mlops.training.claims import WorkerClaim

    lease_token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
    job = {
        "job_id": "e4d82c26-8f0a-4f45-8b88-5fe84302d948",
        "state": "running",
        "lease_token": lease_token,
        "lease_generation": 1,
        "phase": "probe",
        "target_phase": "training",
        "input_kind": "probe_input",
        "input_id": probe_input.probe_input_id,
        "input_sha256": probe_input.input_sha256,
        "dataset_version": None,
        "model_kind": "clip",
        "config_version": profile["config_version"],
        "config_sha256": profile["config_sha256"],
        "source_refs": probe_input.as_dict(),
    }
    lease = {
        "job_id": job["job_id"],
        "lease_token": lease_token,
        "fencing_token": 1,
        "gpu_uuid": GPU_UUID,
    }
    claim = WorkerClaim.from_admitted_job(job, lease, image_id="sha256:" + "a" * 64)
    return job, lease, claim


def _clip_runner_config(profile: dict, probe_input: ProbeInput) -> dict:
    return {
        **profile["config_json"],
        "phase": "probe",
        "target_phase": "training",
        "input_kind": "probe_input",
        "input_id": probe_input.probe_input_id,
        "input_sha256": probe_input.input_sha256,
        "dataset_version": None,
        "model_kind": "clip",
        "config_version": profile["config_version"],
        "config_sha256": profile["config_sha256"],
    }


def test_clip_fix5_profile_uses_new_policy_but_reuses_exact_frozen_manifest() -> None:
    profile = candidate_profile("clip")

    assert profile.phase == "probe"
    assert profile.target_phase == "training"
    assert profile.candidate is True
    assert profile.config_version == CLIP_EXECUTION_CONFIG_VERSION
    assert profile.config_sha256 != "f9dab1521d78a8297c2c0ae29bfa1de7b5a60d3f114435d96ca58a97bb2a0a6f"
    assert profile.config["model_id"] == "openai/clip-vit-base-patch16"
    assert profile.config["model_revision"] == CLIP_MODEL_REVISION
    assert profile.config["resolution"] == 224
    assert profile.config["micro_batch"] == 2
    assert profile.config["gradient_accumulation_steps"] == 1
    assert profile.config["contrastive_config_version"] == "task8-explicit-negatives-v1"
    assert profile.config["optimizer_steps"] == 3
    assert profile.config["learning_rate"] == 1e-5
    assert profile.config["weight_decay"] == 1e-4
    assert profile.config["probe_manifest_config_version"] == CLIP_MANIFEST_CONFIG_VERSION
    assert profile.config["forward_precision"] == "float16-autocast"
    assert profile.config["training_loss_policy_version"] == "task9-clip-symmetric-ce-fp64-v1"
    assert profile.config["training_loss_objective"] == "symmetric_identity_cross_entropy"
    assert profile.config["training_loss_reduction_precision"] == "float64"

    objects = _ImmutableObjects()
    probe_input = create_probe_input(objects=objects, model_kind="clip")
    manifest_bytes = objects.read_source(
        object_key=CLIP_MANIFEST_KEY,
        sha256_digest=CLIP_INPUT_SHA256,
        size_bytes=1268,
    )
    manifest = json.loads(manifest_bytes)

    assert probe_input == ProbeInput.from_dict(probe_input.as_dict())
    assert probe_input.config_version == CLIP_EXECUTION_CONFIG_VERSION
    assert probe_input.probe_input_id == CLIP_INPUT_ID
    assert probe_input.input_sha256 == CLIP_INPUT_SHA256
    assert probe_input.manifest_object_key == CLIP_MANIFEST_KEY
    assert probe_input.object_size_bytes == 1268
    assert manifest["config_version"] == CLIP_MANIFEST_CONFIG_VERSION
    assert manifest["model_kind"] == "clip"
    assert manifest["model_id"] == "openai/clip-vit-base-patch16"
    assert manifest["model_revision"] == CLIP_MODEL_REVISION
    assert probe_input.as_dict().keys() == {
        "schema",
        "probe_input_id",
        "input_kind",
        "model_kind",
        "target_phase",
        "config_version",
        "manifest_object_key",
        "input_sha256",
        "object_size_bytes",
        "fixture",
        "dataset_version",
    }

    with pytest.raises(ValueError, match="typed immutable reference"):
        probe_input.verify(objects)
    assert probe_input.verify(
        objects,
        expected_manifest_config_version=CLIP_MANIFEST_CONFIG_VERSION,
    ) == manifest

    second = create_probe_input(objects=objects, model_kind="clip")
    assert second.input_sha256 == CLIP_INPUT_SHA256
    assert objects.values[CLIP_MANIFEST_KEY] == manifest_bytes

    evaluation_profile = candidate_profile("clip", target_phase="evaluation")
    assert "probe_manifest_config_version" not in evaluation_profile.config
    assert "training_loss_policy_version" not in evaluation_profile.config


def test_admitted_clip_probe_claim_returns_only_its_profile_bound_manifest_pin() -> None:
    from gods_mlops.training.claims import validate_worker_claim

    objects = _ImmutableObjects()
    probe_input = create_probe_input(objects=objects, model_kind="clip")
    profile = _registered_clip_profile()
    job, lease, claim = _worker_fixture(probe_input, profile)

    pin = validate_worker_claim(claim, job=job, lease=lease, profile=profile)

    assert pin.manifest_config_version == CLIP_MANIFEST_CONFIG_VERSION
    assert pin.execution_config_version == CLIP_EXECUTION_CONFIG_VERSION
    assert pin.execution_config_sha256 == profile["config_sha256"]
    assert pin.input_id == CLIP_INPUT_ID
    assert pin.input_sha256 == CLIP_INPUT_SHA256


@pytest.mark.parametrize(
    "mutation",
    ["wrong_profile_pin", "wrong_profile_hash", "wrong_input_id", "wrong_input_sha", "wrong_model_revision"],
)
def test_admitted_clip_probe_claim_rejects_a_pin_detached_from_profile_or_input(mutation: str) -> None:
    from gods_mlops.training.claims import WorkerAuthorizationError, validate_worker_claim

    objects = _ImmutableObjects()
    original = create_probe_input(objects=objects, model_kind="clip")
    config = dict(candidate_profile("clip").config)
    if mutation == "wrong_profile_pin":
        config["probe_manifest_config_version"] = "task8-clip-unknown-probe-v1"
    elif mutation == "wrong_model_revision":
        config["model_revision"] = "0" * 40
    profile = _registered_clip_profile(config=config)
    probe_input = original
    if mutation in {"wrong_input_id", "wrong_input_sha"}:
        probe_input = ProbeInput(
            probe_input_id="task8-clip-other-synthetic-probe-v1" if mutation == "wrong_input_id" else original.probe_input_id,
            model_kind=original.model_kind,
            target_phase=original.target_phase,
            config_version=original.config_version,
            manifest_object_key=original.manifest_object_key,
            input_sha256="c" * 64 if mutation == "wrong_input_sha" else original.input_sha256,
            object_size_bytes=original.object_size_bytes,
        )
    job, lease, claim = _worker_fixture(probe_input, profile)
    if mutation == "wrong_profile_hash":
        profile["config_sha256"] = "f" * 64

    with pytest.raises(WorkerAuthorizationError):
        validate_worker_claim(claim, job=job, lease=lease, profile=profile)


def test_current_clip_probe_claim_verifies_the_old_manifest_with_the_admitted_pin() -> None:
    import asyncio

    from gods_mlops.training.claims import validate_current_worker_claim

    objects = _ImmutableObjects()
    probe_input = create_probe_input(objects=objects, model_kind="clip")
    profile = _registered_clip_profile()
    job, lease, claim = _worker_fixture(probe_input, profile)

    class Repository:
        async def get_active_lease(self, _gpu_uuid):
            return lease

        async def get_profile(self, **_kwargs):
            return profile

        async def lease_is_current(self, _job_id, _lease_token):
            return True

    class Queue:
        repository = Repository()

        async def get(self, _job_id):
            return job

    checked_job, checked_lease = asyncio.run(
        validate_current_worker_claim(Queue(), claim, object_store=objects)
    )
    assert checked_job == job
    assert checked_lease == lease


def test_worker_manifest_passes_only_the_registered_profile_pin_to_runner_validation(tmp_path) -> None:
    import asyncio
    from pathlib import Path

    from gods_mlops.training.claims import validate_worker_claim
    from gods_mlops.training.contracts import validate_manifest_identity
    from gods_mlops.training.worker import _worker_manifest

    objects = _ImmutableObjects()
    probe_input = create_probe_input(objects=objects, model_kind="clip")
    profile = _registered_clip_profile()
    job, lease, claim = _worker_fixture(probe_input, profile)
    pin = validate_worker_claim(claim, job=job, lease=lease, profile=profile)

    manifest_uri, extra = asyncio.run(
        _worker_manifest(
            job,
            None,
            objects,
            root=tmp_path,
            claim=claim,
            profile=profile,
        )
    )
    manifest_path = Path(manifest_uri)
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest_path.read_bytes() == objects.values[CLIP_MANIFEST_KEY]
    assert extra["_probe_manifest_pin"] == pin
    config = {**_clip_runner_config(profile, probe_input), **extra}
    assert validate_manifest_identity(
        config,
        manifest,
        probe_manifest_pin=extra["_probe_manifest_pin"],
    )["config_version"] == CLIP_EXECUTION_CONFIG_VERSION


def test_committed_result_recovery_reuses_the_verified_profile_pin() -> None:
    import asyncio
    import importlib
    from types import SimpleNamespace

    from gods_mlops.jobs.checkpoints import CheckpointIdentity
    from gods_mlops.training.worker import _recover_committed_result

    objects = _ImmutableObjects()
    probe_input = create_probe_input(objects=objects, model_kind="clip")
    profile = _registered_clip_profile()
    job, lease, claim = _worker_fixture(probe_input, profile)
    model = SimpleNamespace(
        model_id=profile["config_json"]["model_id"],
        revision=profile["config_json"]["model_revision"],
    )
    checkpoint_sha256 = "c" * 64
    result_sha256 = "e" * 64
    measurements = {
        "model_id": model.model_id,
        "model_revision": model.revision,
        "peak_vram_allocated_mib": 1200,
        "peak_vram_reserved_mib": 2000,
        "optimizer_steps": 4,
        "checkpoint_resumed": True,
        "initial_weight_sha256": "a" * 64,
        "final_weight_sha256": "b" * 64,
        "losses": [2.0e-8, 2.1e-8, 2.2e-8, 2.3e-8],
    }
    identity = CheckpointIdentity(
        job_id=claim.job_id,
        input_kind=claim.input_kind,
        input_id=claim.input_id,
        input_sha256=claim.input_sha256,
        phase=claim.phase,
        model_kind=claim.model_kind,
        config_version=claim.config_version,
        config_sha256=claim.config_sha256,
        dataset_version=None,
    )
    record = {}

    class Repository:
        async def result_artifacts_for(self, _job_id):
            return [{"kind": "model", "runtime_measurements": measurements}]

        async def get_active_lease(self, _gpu_uuid):
            return lease

        async def get_profile(self, **_kwargs):
            return profile

        async def lease_is_current(self, _job_id, _lease_token):
            return True

    class Queue:
        repository = Repository()
        source_registry = object()

        async def get(self, _job_id):
            return job

        async def load_checkpoint(self, *, store, job_id):
            assert store is checkpoint_store
            assert job_id == claim.job_id
            return SimpleNamespace(sha256=checkpoint_sha256)

        async def record_probe_measurement(self, **kwargs):
            record.update(kwargs)
            return {"result_state": "succeeded"}

    class ResultStore:
        def verify_committed(self, details, *, expected_identity):
            assert details["kind"] == "model"
            assert expected_identity == identity
            return SimpleNamespace(
                sha256=result_sha256,
                object_key=None,
                size_bytes=100,
            )

    queue = Queue()
    checkpoint_store = object()
    contracts = importlib.import_module("gods_mlops.training.contracts")
    config = _clip_runner_config(profile, probe_input)
    exit_code = asyncio.run(
        _recover_committed_result(
            job=job,
            queue=queue,
            claim=claim,
            identity=identity,
            objects=objects,
            checkpoint_store=checkpoint_store,
            result_store=ResultStore(),
            model=model,
            contracts=contracts,
            config=config,
            profile=profile,
        )
    )

    assert exit_code == 0
    assert record["verification_details"]["passed"] is True
    assert record["checkpoint_sha256"] == checkpoint_sha256
    assert config["_probe_manifest_pin"].manifest_config_version == CLIP_MANIFEST_CONFIG_VERSION


def test_promoted_public_training_profile_keeps_manifest_pin_inert() -> None:
    from gods_mlops.training.claims import validate_worker_claim

    profile = _registered_clip_profile()
    public_profile = {
        **profile,
        "phase": "training",
        "target_phase": None,
        "profile_state": "measured",
        "measurement_id": "21c4d375-c5e6-4122-bdcb-62329392bf97",
    }
    job = {
        "job_id": "e4d82c26-8f0a-4f45-8b88-5fe84302d948",
        "state": "running",
        "lease_token": "8ad96890-3434-4f07-85bb-8cde17a2b009",
        "lease_generation": 1,
        "phase": "training",
        "target_phase": None,
        "input_kind": "dataset_version",
        "input_id": "dataset-clip-v1",
        "input_sha256": "d" * 64,
        "dataset_version": "dataset-clip-v1",
        "model_kind": "clip",
        "config_version": profile["config_version"],
        "config_sha256": profile["config_sha256"],
    }
    lease = {
        "job_id": job["job_id"],
        "lease_token": job["lease_token"],
        "fencing_token": 1,
        "gpu_uuid": GPU_UUID,
    }
    from gods_mlops.training.claims import WorkerClaim

    claim = WorkerClaim.from_admitted_job(job, lease, image_id="sha256:" + "a" * 64)

    assert validate_worker_claim(claim, job=job, lease=lease, profile=public_profile) is None


def test_clip_probe_uses_the_selected_loss_policy_for_both_initial_and_restored_updates(
    monkeypatch,
    tmp_path,
) -> None:
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    import torch

    clip = importlib.import_module("gods_mlops.training.clip")
    runner_support = importlib.import_module("gods_mlops.training.runner_support")
    transformers = importlib.import_module("transformers")
    profile = candidate_profile("clip")
    config = {
        **profile.config,
        "job_id": "e4d82c26-8f0a-4f45-8b88-5fe84302d948",
        "phase": "probe",
        "target_phase": "training",
        "input_kind": "probe_input",
        "input_id": CLIP_INPUT_ID,
        "input_sha256": CLIP_INPUT_SHA256,
        "dataset_version": None,
        "model_kind": "clip",
        "config_version": profile.config_version,
        "config_sha256": profile.config_sha256,
    }

    class FakeClipModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(
                torch.tensor(
                    [[29.984375, 5.48828125], [13.34375, 30.03125]],
                    dtype=torch.float32,
                )
            )

        def forward(self, **_kwargs):
            return SimpleNamespace(
                loss=None,
                logits_per_image=self.logits.to(dtype=torch.float16),
            )

    class FakeScaler:
        def __init__(self, *_args, **_kwargs) -> None:
            self.scale_value = 65536.0

        def scale(self, loss):
            return loss * self.scale_value

        def unscale_(self, optimizer):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(self.scale_value)

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return None

        def get_scale(self):
            return self.scale_value

        def state_dict(self):
            return {"scale": self.scale_value}

        def load_state_dict(self, state):
            self.scale_value = float(state["scale"])

    models = [FakeClipModel(), FakeClipModel()]
    monkeypatch.setattr(
        transformers.CLIPModel,
        "from_pretrained",
        lambda *_args, **_kwargs: models.pop(0),
    )
    monkeypatch.setattr(
        transformers.CLIPProcessor,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(clip, "load_manifest", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        clip,
        "validate_manifest_identity",
        lambda *_args, **_kwargs: {
            "input_kind": "probe_input",
            "input_id": CLIP_INPUT_ID,
            "input_sha256": CLIP_INPUT_SHA256,
        },
    )
    monkeypatch.setattr(runner_support, "cache_directory", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(runner_support, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(torch.amp, "GradScaler", FakeScaler)
    monkeypatch.setattr(
        clip,
        "_pairs_from_manifest",
        lambda *_args, **_kwargs: (
            [
                {
                    "image_id": "crop-red",
                    "text": "red coat",
                    "negative_texts": ["blue coat"],
                },
                {
                    "image_id": "crop-blue",
                    "text": "blue coat",
                    "negative_texts": ["red coat"],
                },
            ],
            None,
        ),
    )
    monkeypatch.setattr(clip, "_encode_contrastive_batch", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner_support, "resource_measurements", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner_support,
        "model_bundle",
        lambda *_args, **_kwargs: (b"model-bundle", "e" * 64),
    )

    original_step_optimizer = runner_support.step_optimizer
    observed_options = []

    def record_step_options(*args, **kwargs):
        observed_options.append(
            (
                kwargs["model_kwargs"],
                kwargs["loss_from_output"],
                kwargs["require_finite_nonzero_gradients"],
            )
        )
        return original_step_optimizer(*args, **kwargs)

    monkeypatch.setattr(runner_support, "step_optimizer", record_step_options)

    result = clip.run(config, "unused-manifest", str(tmp_path / "model"))

    assert len(observed_options) == 4
    assert all(options[0] == {"return_loss": False} for options in observed_options)
    assert all(options[1] is clip._symmetric_clip_loss_from_output for options in observed_options)
    assert all(options[2] is True for options in observed_options)
    assert result["resource_measurements"]["resume_steps"] == 1
    assert result["resource_measurements"]["finite_nonzero_gradient_updates"] == 4
    assert result["resource_measurements"]["training_loss_reduction_precision"] == "float64"


def test_runner_accepts_only_the_verified_clip_probe_pin_and_keeps_public_paths_strict() -> None:
    from gods_mlops.training.claims import validate_worker_claim
    from gods_mlops.training.contracts import validate_manifest_identity

    objects = _ImmutableObjects()
    probe_input = create_probe_input(objects=objects, model_kind="clip")
    profile = _registered_clip_profile()
    job, lease, claim = _worker_fixture(probe_input, profile)
    pin = validate_worker_claim(claim, job=job, lease=lease, profile=profile)
    manifest = probe_input.verify(
        objects,
        expected_manifest_config_version=CLIP_MANIFEST_CONFIG_VERSION,
    )
    config = _clip_runner_config(profile, probe_input)

    identity = validate_manifest_identity(config, manifest, probe_manifest_pin=pin)
    assert identity["config_version"] == CLIP_EXECUTION_CONFIG_VERSION
    assert identity["input_id"] == CLIP_INPUT_ID
    assert identity["input_sha256"] == CLIP_INPUT_SHA256

    with pytest.raises(ValueError):
        validate_manifest_identity(config, manifest)
    for unsupported_context in (
        {"phase": "training"},
        {"target_phase": "evaluation"},
        {"model_kind": "detr"},
    ):
        with pytest.raises(ValueError):
            validate_manifest_identity(
                {**config, **unsupported_context},
                manifest,
                probe_manifest_pin=pin,
            )

    public_profile_config = dict(profile["config_json"])
    public_config = {
        **public_profile_config,
        "phase": "training",
        "target_phase": "training",
        "input_kind": "dataset_version",
        "input_id": "dataset-clip-v1",
        "input_sha256": "d" * 64,
        "dataset_version": "dataset-clip-v1",
        "model_kind": "clip",
        "config_version": CLIP_EXECUTION_CONFIG_VERSION,
        "config_sha256": profile["config_sha256"],
    }
    public_manifest = {
        "schema_version": 1,
        "dataset_version": "dataset-clip-v1",
        "training": {"ready": True},
        "target": "clip",
        "config_version": CLIP_EXECUTION_CONFIG_VERSION,
    }
    from gods_mlops.datasets.manifest import canonical_json, content_sha256

    public_config["input_sha256"] = content_sha256(canonical_json(public_manifest))
    assert validate_manifest_identity(public_config, public_manifest)["phase"] == "training"
    with pytest.raises(ValueError):
        validate_manifest_identity(public_config, public_manifest, probe_manifest_pin=pin)

    evaluation_probe_config = {
        **_clip_runner_config(profile, probe_input),
        "target_phase": "evaluation",
    }
    with pytest.raises(ValueError):
        validate_manifest_identity(evaluation_probe_config, manifest, probe_manifest_pin=pin)


def test_clip_probe_worker_dispatch_forwards_profile_pin_through_both_real_runners(
    monkeypatch,
    tmp_path,
) -> None:
    import asyncio
    import importlib
    from contextlib import nullcontext
    from dataclasses import replace
    from types import SimpleNamespace

    import torch

    from gods_mlops.training.claims import validate_worker_claim
    from gods_mlops.training.worker import _runner_for, _worker_manifest

    probe = importlib.import_module("gods_mlops.training.probe")
    clip = importlib.import_module("gods_mlops.training.clip")
    training_data = importlib.import_module("gods_mlops.training.data")
    runner_support = importlib.import_module("gods_mlops.training.runner_support")
    transformers = importlib.import_module("transformers")

    objects = _ImmutableObjects()
    probe_input = create_probe_input(objects=objects, model_kind="clip")
    profile = _registered_clip_profile()
    job, lease, claim = _worker_fixture(probe_input, profile)
    pin = validate_worker_claim(claim, job=job, lease=lease, profile=profile)
    manifest_root = tmp_path / "manifest"
    manifest_root.mkdir()
    manifest_uri, extra = asyncio.run(
        _worker_manifest(
            job,
            None,
            objects,
            root=manifest_root,
            claim=claim,
            profile=profile,
        )
    )
    manifest_bytes = objects.read_source(
        object_key=CLIP_MANIFEST_KEY,
        sha256_digest=CLIP_INPUT_SHA256,
        size_bytes=1268,
    )
    assert (manifest_root / "probe-manifest.json").read_bytes() == manifest_bytes
    assert extra["_probe_manifest_pin"] == pin
    config = {
        **_clip_runner_config(profile, probe_input),
        "job_id": claim.job_id,
        **extra,
    }

    # Keep the actual manifest/media path; only the external object-store transport is in memory.
    monkeypatch.setattr(training_data, "dataset_object_store_from_environment", lambda: objects)

    class FakeClipModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(
                torch.tensor(
                    [[29.984375, 5.48828125], [13.34375, 30.03125]],
                    dtype=torch.float32,
                )
            )

        def forward(self, **_kwargs):
            return SimpleNamespace(
                loss=None,
                logits_per_image=self.logits.to(dtype=torch.float16),
            )

    class FakeScaler:
        def __init__(self, *_args, **_kwargs) -> None:
            self.scale_value = 65536.0

        def scale(self, loss):
            return loss * self.scale_value

        def unscale_(self, optimizer):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(self.scale_value)

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            return None

        def get_scale(self):
            return self.scale_value

        def state_dict(self):
            return {"scale": self.scale_value}

        def load_state_dict(self, state):
            self.scale_value = float(state["scale"])

    models = [FakeClipModel(), FakeClipModel()]
    model_loads = []

    def load_fake_model(*args, **kwargs):
        model_loads.append((args, kwargs))
        return models.pop(0)

    monkeypatch.setattr(transformers.CLIPModel, "from_pretrained", load_fake_model)
    monkeypatch.setattr(transformers.CLIPProcessor, "from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner_support, "cache_directory", lambda *_args, **_kwargs: tmp_path)
    monkeypatch.setattr(runner_support, "require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(torch.amp, "GradScaler", FakeScaler)
    monkeypatch.setattr(clip, "_encode_contrastive_batch", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner_support, "model_bundle", lambda *_args, **_kwargs: (b"model-bundle", "e" * 64))

    runner = _runner_for("probe", "clip", "training")
    assert runner is probe.run
    result = runner(config, manifest_uri, str(tmp_path / "output"))

    assert result["status"] == "succeeded"
    assert len(model_loads) == 2
    assert result["resource_measurements"]["initial_optimizer_steps"] == 3
    assert result["resource_measurements"]["resume_steps"] == 1
    assert result["resource_measurements"]["optimizer_steps"] == 4
    assert result["resource_measurements"]["finite_nonzero_gradient_updates"] == 4
    assert result["resource_measurements"]["training_loss_reduction_precision"] == "float64"

    model_load_count = len(model_loads)
    missing_pin_config = dict(config)
    missing_pin_config.pop("_probe_manifest_pin")
    with pytest.raises(ValueError):
        runner(missing_pin_config, manifest_uri, str(tmp_path / "missing-pin-output"))
    assert len(model_loads) == model_load_count

    invalid_pin_config = {
        **config,
        "_probe_manifest_pin": replace(pin, input_sha256="c" * 64),
    }
    with pytest.raises(ValueError):
        clip.run(invalid_pin_config, manifest_uri, str(tmp_path / "invalid-pin-output"))
    assert len(model_loads) == model_load_count
