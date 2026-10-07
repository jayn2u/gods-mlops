from __future__ import annotations

import asyncio
import importlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from gods_mlops.jobs.models import ResourceObservation


def _require(module_name: str, symbol: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(f"missing implementation module {module_name}: {error.name}", pytrace=False)
    value = getattr(module, symbol, None)
    assert value is not None, f"{module_name}.{symbol} is part of the training contract"
    return value


def _probe_job(*, generation: int = 2, state: str = "running") -> tuple[dict, dict, dict]:
    job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
    job = {
        "job_id": job_id,
        "state": state,
        "lease_token": token,
        "lease_generation": generation,
        "phase": "probe",
        "target_phase": "training",
        "input_kind": "probe_input",
        "input_id": "task8-clip-probe-v1",
        "input_sha256": "1" * 64,
        "dataset_version": None,
        "model_kind": "clip",
        "config_version": "clip-real-probe-v1",
        "config_sha256": "2" * 64,
        "profile_state_snapshot": "candidate",
    }
    lease = {
        "job_id": job_id,
        "lease_token": token,
        "fencing_token": generation,
        "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
    }
    profile = {
        "phase": "probe",
        "target_phase": "training",
        "model_kind": "clip",
        "config_version": "clip-real-probe-v1",
        "config_sha256": "2" * 64,
        "profile_state": "candidate",
    }
    return job, lease, profile


def test_delayed_worker_with_expired_fence_fails_closed_before_runner_starts() -> None:
    claim_type = _require("gods_mlops.training.claims", "WorkerClaim")
    validate_claim = _require("gods_mlops.training.claims", "validate_worker_claim")
    job, current_lease, profile = _probe_job(generation=2)

    stale_claim = replace(
        claim_type.from_admitted_job(job, current_lease, image_id="sha256:" + "a" * 64),
        fence=1,
    )

    with pytest.raises(ValueError, match="fence"):
        validate_claim(stale_claim, job=job, lease=current_lease, profile=profile)


def test_training_worker_requires_a_measured_profile_and_exact_config_identity() -> None:
    claim_type = _require("gods_mlops.training.claims", "WorkerClaim")
    validate_claim = _require("gods_mlops.training.claims", "validate_worker_claim")
    job, lease, profile = _probe_job()
    training_job = {
        **job,
        "phase": "training",
        "target_phase": "training",
        "dataset_version": job["input_id"],
        "input_kind": "dataset_version",
        "profile_state_snapshot": "measured",
    }
    training_lease = {**lease, "fencing_token": training_job["lease_generation"]}
    measured_profile = {**profile, "phase": "training", "target_phase": None, "profile_state": "measured"}
    measured_profile["config_version"] = training_job["config_version"]
    training_job["input_id"] = training_job["dataset_version"]
    training_job["input_sha256"] = "3" * 64
    claim = claim_type.from_admitted_job(training_job, training_lease, image_id="sha256:" + "a" * 64)

    validate_claim(claim, job=training_job, lease=training_lease, profile=measured_profile)

    with pytest.raises(ValueError, match="config"):
        validate_claim(claim, job=training_job, lease=training_lease, profile={**measured_profile, "config_sha256": "4" * 64})

    with pytest.raises(ValueError, match="measured"):
        validate_claim(claim, job=training_job, lease=training_lease, profile={**measured_profile, "profile_state": "candidate"})


def test_published_clip_batches_cover_examples_and_resume_from_optimizer_cursor() -> None:
    pairs_from_manifest = _require("gods_mlops.training.clip", "_pairs_from_manifest")
    batch_for_step = _require("gods_mlops.training.clip", "_contrastive_batch")
    validate_pairs = _require("gods_mlops.training.clip", "validate_contrastive_pairs")
    items = [
        {
            "kind": "crop",
            "split": "train",
            "item_id": f"crop-{index}",
            "image_path": f"/unused/{index}.jpg",
            "snapshot": {"caption": {"text": f"distinct person description {index}"}},
        }
        for index in range(5)
    ]
    pairs, object_store = pairs_from_manifest(
        {"items": items}, {"phase": "training", "micro_batch": 2}, 2
    )

    assert object_store is None
    assert len(pairs) == 5
    assert sum(len(pair["negative_texts"]) for pair in pairs) == 0
    batches = [batch_for_step(pairs, micro_batch=2, optimizer_step=step) for step in range(3)]
    consumed = {pair["image_id"] for batch in batches for pair in batch}
    assert consumed == {item["item_id"] for item in items}
    assert all(validate_pairs({"contrastive_config_version": "test-v1", "pairs": batch}) == 2 for batch in batches)
    assert all(len(pair["negative_texts"]) == 1 for batch in batches for pair in batch)
    assert [pair["image_id"] for pair in batch_for_step(pairs, micro_batch=2, optimizer_step=3)] == [
        "crop-1",
        "crop-2",
    ]
    assert batches[0] != batch_for_step(pairs, micro_batch=2, optimizer_step=3)


def test_qwen_caption_batch_is_complete_or_rejected_at_its_versioned_bound() -> None:
    caption_items = _require("gods_mlops.training.caption", "_caption_items")
    items = [
        {"item_kind": "crop", "item_id": "crop-a"},
        {"item_kind": "crop", "item_id": "crop-b"},
    ]

    with pytest.raises(ValueError, match="exceeds its versioned max_draft_images bound"):
        caption_items({"items": items}, {"max_draft_images": 1})
    assert caption_items({"items": items}, {"max_draft_images": 2}) == items


def test_detr_preparation_probe_enforces_its_measured_frame_bound() -> None:
    detector_items = _require("gods_mlops.training.detector", "_detector_items")
    frames = [
        {"kind": "frame", "item_kind": "frame", "item_id": f"frame-{index}"}
        for index in range(2)
    ]
    config = {"phase": "probe", "target_phase": "preparation", "max_draft_frames": 1}

    with pytest.raises(ValueError, match="exceeds its versioned max_draft_frames bound"):
        detector_items({"items": frames}, config)
    assert detector_items({"items": frames[:1]}, config) == frames[:1]


@pytest.mark.parametrize("phase", ["preparation", "probe"])
def test_detr_draft_result_recovery_uses_the_runner_measurement_shape_without_cuda(
    phase: str, monkeypatch, tmp_path
) -> None:
    from types import SimpleNamespace

    worker = _require("gods_mlops.training.worker", "_recover_committed_result")
    claim_type = _require("gods_mlops.training.claims", "WorkerClaim")
    identity_type = _require("gods_mlops.jobs.checkpoints", "CheckpointIdentity")
    contracts = importlib.import_module("gods_mlops.training.contracts")
    model = contracts.locked_model("detr")
    job_id = "detr-draft-recovery-v1"
    config_version = "detr-preparation-recovery-v1"
    if phase == "preparation":
        input_id = "annotation-batch-detr-recovery-v1"
        input_kind = "annotation_batch"
        items = [
            {
                "item_kind": "frame",
                "item_id": "frame-recovery-v1",
                "sample_id": "sample-recovery-v1",
                "sha256": "3" * 64,
                "object_key": "samples/frame-recovery-v1.jpg",
                "object_size_bytes": 512,
                "revision_id": None,
            }
        ]
        input_sha256 = sha256(
            __import__("gods_mlops.datasets.manifest", fromlist=["canonical_json"]).canonical_json(
                {"schema": "annotation-preparation-input-v1", "items": items}
            )
        ).hexdigest()
        manifest = {
            "schema_version": 1,
            "batch_id": input_id,
            "input_id": input_id,
            "input_sha256": input_sha256,
            "model_kind": "detr",
            "config_version": config_version,
            "items": items,
        }
        dataset_version = None
    else:
        input_id = "task8-detr-preparation-synthetic-probe-v1"
        input_kind = "probe_input"
        manifest = {
            "schema_version": 1,
            "fixture": True,
            "phase": "probe",
            "model_kind": "detr",
            "input_kind": "probe_input",
            "input_id": input_id,
            "config_version": config_version,
            "model_id": model.model_id,
            "model_revision": model.revision,
            "items": [
                {
                    "kind": "frame",
                    "item_kind": "frame",
                    "item_id": "task8-probe-frame-1",
                    "sample_id": "task8-probe-frame-1",
                    "object": {"key": "probe-inputs/frame.jpg", "sha256": "3" * 64, "size_bytes": 512},
                    "snapshot": {"bbox_annotation": {"result": []}},
                }
            ],
        }
        input_sha256 = sha256(
            __import__("gods_mlops.datasets.manifest", fromlist=["canonical_json"]).canonical_json(manifest)
        ).hexdigest()
        dataset_version = None

    identity = identity_type(
        job_id=job_id,
        input_kind=input_kind,
        input_id=input_id,
        input_sha256=input_sha256,
        phase=phase,
        model_kind="detr",
        config_version=config_version,
        config_sha256="4" * 64,
        dataset_version=dataset_version,
    )
    claim = claim_type(
        job_id=job_id,
        lease_token="d" * 36,
        fence=2,
        gpu_uuid="GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
        phase=phase,
        target_phase="preparation",
        input_kind=input_kind,
        input_id=input_id,
        input_sha256=input_sha256,
        dataset_version=None,
        model_kind="detr",
        config_version=config_version,
        config_sha256="4" * 64,
        image_id="sha256:" + "a" * 64,
    )
    config = {
        **identity.as_dict(),
        "phase": phase,
        "target_phase": "preparation",
        "model_kind": "detr",
        "model_id": model.model_id,
        "model_revision": model.revision,
    }
    result_document = {
        "schema_version": 1,
        "model_kind": "detr",
        "model_id": model.model_id,
        "model_revision": model.revision,
        "config_version": config_version,
        "input_id": input_id,
        "input_sha256": input_sha256,
        "score_threshold": 0.3,
        "drafts": [
            {"item_id": "frame-recovery-v1", "image_width": 640, "image_height": 640, "detections": []}
        ],
    }
    from gods_mlops.datasets.manifest import canonical_json

    payload = canonical_json(result_document)
    digest = sha256(payload).hexdigest()
    draft_measurements = _require("gods_mlops.training.detector", "_draft_resource_measurements")
    measurements = draft_measurements(
        {
            "peak_vram_allocated_mib": 4_000,
            "peak_vram_reserved_mib": 5_000,
            "elapsed_seconds": 1.2,
            "optimizer_steps": 0,
            "inference_steps": 1,
            "precision": "float16-autocast",
        },
        identity={"model_id": model.model_id, "model_revision": model.revision},
    )
    artifact_details = {
        "kind": "drafts",
        "uri": "s3://task8-test/jobs/detr-draft-recovery-v1/result.artifact",
        "sha256": digest,
        "size_bytes": len(payload),
        "identity": identity.as_dict(),
        "object_key": "jobs/detr-draft-recovery-v1/result.artifact",
        "operation_id": "e" * 36,
        "runtime_measurements": measurements,
    }

    class Repository:
        async def result_artifacts_for(self, requested_job_id):
            assert requested_job_id == job_id
            return [artifact_details]

    class Queue:
        repository = Repository()

        async def complete_owned_job(self, **kwargs):
            self.completed = kwargs

        async def record_probe_measurement(self, **kwargs):
            self.probe_measurement = kwargs
            return {"result_state": "succeeded"}

    queue = Queue()

    class ResultStore:
        def verify_committed(self, details, *, expected_identity):
            assert details is artifact_details
            assert expected_identity == identity
            assert sha256(payload).hexdigest() == details["sha256"]
            return SimpleNamespace(
                identity=identity,
                kind="drafts",
                sha256=digest,
                size_bytes=len(payload),
                uri=details["uri"],
            )

    async def manifest_for_job(_job, _queue, _objects, *, root):
        path = root / "frozen-detr-preparation-manifest.json"
        path.write_bytes(canonical_json(manifest))
        return path.resolve().as_uri(), {}

    async def current_claim(_queue, _claim, *, object_store=None):
        return {}, {}

    monkeypatch.setattr(importlib.import_module("gods_mlops.training.worker"), "_worker_manifest", manifest_for_job)
    monkeypatch.setattr(importlib.import_module("gods_mlops.training.worker"), "validate_current_worker_claim", current_claim)

    async def recover():
        return await worker(
            job={"job_id": job_id},
            queue=queue,
            claim=claim,
            identity=identity,
            objects=object(),
            checkpoint_store=object(),
            result_store=ResultStore(),
            model=model,
            contracts=contracts,
            config=config,
        )

    assert asyncio.run(recover()) == 0
    if phase == "probe":
        assert queue.probe_measurement["inference_steps"] == 1
    else:
        assert queue.completed["details"]["result_uri"] == artifact_details["uri"]


def test_kubernetes_worker_process_requires_owned_pod_and_exact_cgroup_identity() -> None:
    resolve = _require("gods_mlops.training.worker_adapter", "resolve_owned_gpu_process")
    job_uid = "25f66a14-6178-4290-911f-d28f294adf84"
    container_id = "b" * 64
    pod = {
        "metadata": {
            "uid": "acb5a886-e1f6-48cf-a2c8-cf85ca3e9362",
            "ownerReferences": [{"kind": "Job", "uid": job_uid, "controller": True}],
        },
        "status": {
            "containerStatuses": [{"name": "trainer", "containerID": f"containerd://{container_id}"}]
        },
    }
    identity = {"pid": 7312, "start_ticks": 238191, "uid": 10001}
    matching_cgroup = (
        "/kubepods.slice/kubepods-burstable-podacb5a886_e1f6_48cf_a2c8_cf85ca3e9362.slice/"
        f"cri-containerd-{container_id}.scope"
    )
    observation = {
        "observation_id": "10ad4be7-6f88-4773-a860-2e89867cad66",
        "node_id": "ubuntu",
        "hostname": "ubuntu",
        "host_identity": "machine-sha256:test",
        "gpu_name": "NVIDIA RTX A6000",
        "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
        "free_mib": 48_000,
        "total_mib": 49_140,
        "gpu_processes": [],
        "process_table": [{**identity, "cgroup_paths": [matching_cgroup]}],
        "gpu_process_list_complete": True,
        "process_table_complete": True,
        "storage_path": "/data",
        "filesystem_identity": "ext4:uuid=test",
        "filesystem_available_bytes": 2**40,
        "observed_at": datetime.now(UTC).isoformat(),
    }

    owner = resolve(job_uid=job_uid, pod=pod, observation=observation)

    assert owner.as_dict() == {**identity, "cgroup_paths": [matching_cgroup]}

    wrong_container = {
        **observation,
        "process_table": [
            {**identity, "cgroup_paths": [matching_cgroup.replace(container_id, "c" * 64)]}
        ],
    }
    with pytest.raises(ValueError, match="owned GPU process"):
        resolve(job_uid=job_uid, pod=pod, observation=wrong_container)

    foreign_pod = {**pod, "metadata": {**pod["metadata"], "ownerReferences": [{"kind": "Job", "uid": "d" * 36}]}}
    with pytest.raises(ValueError, match="owner"):
        resolve(job_uid=job_uid, pod=foreign_pod, observation=observation)


def test_docker_probe_owner_uses_exact_container_cgroup_and_fresh_host_identity() -> None:
    resolve = _require("gods_mlops.training.worker_adapter", "resolve_owned_docker_process")
    container_id = "d" * 64
    cgroup = f"/system.slice/docker-{container_id}.scope"
    process = {"pid": 7312, "start_ticks": 238191, "uid": 10001, "cgroup_paths": [cgroup]}
    observation = {
        "observation_id": "10ad4be7-6f88-4773-a860-2e89867cad66",
        "node_id": "ubuntu",
        "hostname": "ubuntu",
        "host_identity": "machine-sha256:test",
        "gpu_name": "NVIDIA RTX A6000",
        "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
        "free_mib": 48_000,
        "total_mib": 49_140,
        "gpu_processes": [],
        "gpu_process_list_complete": True,
        "process_table": [process],
        "process_table_complete": True,
        "storage_path": "/data",
        "filesystem_identity": "ext4:uuid=test",
        "filesystem_available_bytes": 2**40,
        "observed_at": datetime.now(UTC).isoformat(),
    }

    owner = resolve(container_id=container_id, observation=observation)

    assert owner.as_dict() == process
    with pytest.raises(ValueError, match="unavailable or ambiguous"):
        resolve(
            container_id="e" * 64,
            observation=observation,
        )
    with pytest.raises(ValueError, match="UID"):
        resolve(
            container_id=container_id,
            observation={
                **observation,
                "process_table": [{**process, "uid": 1009}],
            },
        )


def test_strict_ssh_probe_adapter_rejects_non_loopback_postgres_and_s3_targets() -> None:
    module = _require("gods_mlops.training.docker_probe", "_loopback_database")
    s3_endpoint = _require("gods_mlops.training.docker_probe", "_loopback_endpoint")
    rewrite_dsn = _require("gods_mlops.training.docker_probe", "_remote_database_url")

    assert module("postgresql://probe:secret@127.0.0.1:15439/probe") == ("127.0.0.1", 15439)
    assert s3_endpoint("http://localhost:18333") == ("localhost", 18333)
    assert rewrite_dsn("postgresql://probe:secret@127.0.0.1:15439/probe", 35439) == (
        "postgresql://probe:secret@127.0.0.1:35439/probe"
    )
    with pytest.raises(ValueError, match="loopback PostgreSQL"):
        module("postgresql://probe:secret@10.10.0.8:15439/probe")
    with pytest.raises(ValueError, match="loopback S3"):
        s3_endpoint("http://storage.example:18333")
    with pytest.raises(ValueError, match="embed credentials"):
        s3_endpoint("http://probe:secret@127.0.0.1:18333")


def test_delayed_worker_never_calls_cuda_or_model_loader_before_current_host_bind() -> None:
    module = _require("gods_mlops.training.worker", "_run_after_owner_binding")
    claim_type = _require("gods_mlops.training.claims", "WorkerClaim")
    auth_error = _require("gods_mlops.training.claims", "WorkerAuthorizationError")
    job, lease, _profile = _probe_job(generation=2)
    claim = claim_type.from_admitted_job(job, lease, image_id="sha256:" + "a" * 64)
    identity = {"job": 0, "token": lease["lease_token"], "fence": 2, "owner": None}

    class Repository:
        async def get_job(self, job_id):
            assert job_id == claim.job_id
            return {"state": "running", "lease_token": identity["token"]}

        async def get_active_lease(self, gpu_uuid):
            assert gpu_uuid == claim.gpu_uuid
            return {
                "job_id": claim.job_id,
                "lease_token": identity["token"],
                "fencing_token": identity["fence"],
                "owner_pid": identity["owner"][0] if identity["owner"] else None,
                "owner_start_ticks": identity["owner"][1] if identity["owner"] else None,
                "owner_uid": identity["owner"][2] if identity["owner"] else None,
            }

        async def lease_is_current(self, job_id, lease_token):
            return job_id == claim.job_id and lease_token == identity["token"]

    cuda_calls = []
    model_loader_calls = []

    def runner():
        cuda_calls.append("cuda")
        model_loader_calls.append("from_pretrained")

    current = [0.0]

    async def no_bind_sleep(seconds):
        current[0] += seconds

    with pytest.raises(auth_error, match="timed out waiting"):
        asyncio.run(
            module(
                Repository(), claim, runner, timeout_seconds=2,
                poll_interval_seconds=0.1, sleep=no_bind_sleep, clock=lambda: current[0],
            )
        )
    assert cuda_calls == []
    assert model_loader_calls == []

    identity["token"] = "b" * 36
    identity["fence"] = 3
    with pytest.raises(auth_error, match="revoked"):
        asyncio.run(module(Repository(), claim, runner, timeout_seconds=1))
    assert cuda_calls == []
    assert model_loader_calls == []

    identity["token"] = lease["lease_token"]
    identity["fence"] = 2
    identity["owner"] = (7312, 238191, 10001)
    asyncio.run(module(Repository(), claim, runner, timeout_seconds=1))
    assert cuda_calls == ["cuda"]
    assert model_loader_calls == ["from_pretrained"]


def test_worker_revalidates_mutable_source_after_host_binding_before_cuda() -> None:
    module = _require("gods_mlops.training.worker", "_run_after_owner_binding")
    claim_type = _require("gods_mlops.training.claims", "WorkerClaim")
    auth_error = _require("gods_mlops.training.claims", "WorkerAuthorizationError")
    job, lease, _profile = _probe_job(generation=2)
    claim = claim_type.from_admitted_job(job, lease, image_id="sha256:" + "a" * 64)
    identity = {"owner": None, "source_current": True}

    class Repository:
        async def get_job(self, job_id):
            return {"state": "running", "lease_token": claim.lease_token}

        async def get_active_lease(self, gpu_uuid):
            owner = identity["owner"]
            return {
                "job_id": claim.job_id,
                "lease_token": claim.lease_token,
                "fencing_token": claim.fence,
                "owner_pid": owner[0] if owner else None,
                "owner_start_ticks": owner[1] if owner else None,
                "owner_uid": owner[2] if owner else None,
            }

        async def lease_is_current(self, job_id, lease_token):
            return job_id == claim.job_id and lease_token == claim.lease_token

    async def invalidate_source_during_wait(seconds):
        identity["source_current"] = False
        identity["owner"] = (7312, 238191, 10001)

    async def revalidate_source():
        if not identity["source_current"]:
            raise auth_error("training source readiness changed during owner binding")
        return True

    cuda_calls = []
    model_loader_calls = []

    def runner():
        cuda_calls.append("cuda")
        model_loader_calls.append("from_pretrained")

    with pytest.raises(auth_error, match="source readiness changed during owner binding"):
        asyncio.run(
            module(
                Repository(), claim, runner, timeout_seconds=2,
                poll_interval_seconds=0.1, sleep=invalidate_source_during_wait,
                revalidate=revalidate_source,
            )
        )
    assert cuda_calls == []
    assert model_loader_calls == []


def test_gpu_process_observation_preserves_host_cgroup_identity() -> None:
    cgroups = ["/kubepods.slice/cri-containerd-" + "b" * 64 + ".scope"]
    observation = ResourceObservation.from_dict(
        {
            "observation_id": "10ad4be7-6f88-4773-a860-2e89867cad66",
            "node_id": "ubuntu",
            "hostname": "ubuntu",
            "host_identity": "machine-sha256:test",
            "gpu_name": "NVIDIA RTX A6000",
            "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
            "free_mib": 48_000,
            "total_mib": 49_140,
            "gpu_processes": [{"pid": 7312, "start_ticks": 238191, "uid": 10001, "cgroup_paths": cgroups}],
            "gpu_process_list_complete": True,
            "process_table": [{"pid": 7312, "start_ticks": 238191, "uid": 10001}],
            "process_table_complete": True,
            "storage_path": "/data",
            "filesystem_identity": "ext4:uuid=test",
            "filesystem_available_bytes": 2**40,
            "observed_at": datetime.now(UTC).isoformat(),
        }
    )

    assert observation.to_dict()["gpu_processes"][0]["cgroup_paths"] == cgroups


def test_owned_gpu_job_is_idempotently_pinned_to_ubuntu_and_one_gpu() -> None:
    build_job = _require("gods_mlops.training.worker_adapter", "build_gpu_worker_job")
    job, lease, profile = _probe_job()

    manifest = build_job(
        job=job,
        lease=lease,
        profile=profile,
        namespace="gods-mlops",
        image="registry.example/gods-mlops-training@sha256:" + "a" * 64,
    )
    pod_spec = manifest["spec"]["template"]["spec"]

    assert manifest["metadata"]["name"].endswith("-fence-2")
    assert pod_spec["nodeSelector"] == {"kubernetes.io/hostname": "ubuntu"}
    assert pod_spec["containers"][0]["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert pod_spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert pod_spec["containers"][0]["image"] == "registry.example/gods-mlops-training@sha256:" + "a" * 64
    env = {item["name"]: item for item in pod_spec["containers"][0]["env"]}
    assert env["GODS_MLOPS_NODE_NAME"]["valueFrom"]["fieldRef"]["fieldPath"] == "spec.nodeName"
    assert env["GODS_MLOPS_MODEL_LOCK"]["value"] == "/app/models/lock.json"


def test_yielding_job_reconciliation_reads_exact_owned_job_without_creating_a_replacement() -> None:
    adapter_type = _require("gods_mlops.training.worker_adapter", "KubernetesOwnedWorkerAdapter")
    build_job = _require("gods_mlops.training.worker_adapter", "build_gpu_worker_job")
    job, lease, profile = _probe_job()
    namespace = "gods-mlops"
    image = "registry.example/gods-mlops-training@sha256:" + "a" * 64
    existing = build_job(
        job=job,
        lease=lease,
        profile=profile,
        namespace=namespace,
        image=image,
    )
    existing["metadata"]["uid"] = "25f66a14-6178-4290-911f-d28f294adf84"

    class BatchAPI:
        def __init__(self) -> None:
            self.created = []
            self.read = []

        def create_namespaced_job(self, *, namespace, body):
            self.created.append(body)
            raise AssertionError("a retained yield must never create a replacement Job")

        def read_namespaced_job(self, *, name, namespace):
            self.read.append((name, namespace))
            return existing

    batch_api = BatchAPI()
    adapter = adapter_type(batch_api=batch_api, namespace=namespace, image=image)

    actual = adapter.read_existing_worker(
        job={**job, "state": "yield_requested"}, lease=lease, profile=profile
    )

    assert actual is existing
    assert actual["metadata"]["uid"] == existing["metadata"]["uid"]
    assert batch_api.read == [(existing["metadata"]["name"], namespace)]
    assert batch_api.created == []


def test_owned_job_conflict_rejects_matching_labels_with_a_different_pod_contract() -> None:
    adapter_type = _require("gods_mlops.training.worker_adapter", "KubernetesOwnedWorkerAdapter")
    job, lease, profile = _probe_job()
    expected_image = "registry.example/gods-mlops-training@sha256:" + "a" * 64
    conflicting_image = "registry.example/gods-mlops-training@sha256:" + "b" * 64

    class AlreadyExists(Exception):
        status = 409

    class BatchAPI:
        def create_namespaced_job(self, *, namespace, body):
            raise AlreadyExists()

        def read_namespaced_job(self, *, name, namespace):
            existing = build(
                job=job,
                lease=lease,
                profile=profile,
                namespace=namespace,
                image=expected_image,
            )
            existing["spec"]["template"]["spec"]["containers"][0]["image"] = conflicting_image
            return existing

    build = _require("gods_mlops.training.worker_adapter", "build_gpu_worker_job")
    adapter = adapter_type(batch_api=BatchAPI(), namespace="gods-mlops", image=expected_image)

    with pytest.raises(ValueError, match="existing Kubernetes Job"):
        adapter.ensure_worker(job=job, lease=lease, profile=profile)


def test_worker_requires_the_admitted_ubuntu_node_and_actual_a6000_identity() -> None:
    validate_placement = _require("gods_mlops.training.placement", "validate_worker_placement")
    gpu_uuid = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"

    validate_placement(
        node_name="ubuntu",
        admitted_gpu_uuid=gpu_uuid,
        visible_gpu_uuids=[gpu_uuid],
        visible_gpu_names=["NVIDIA RTX A6000"],
    )

    with pytest.raises(ValueError, match="Ubuntu"):
        validate_placement(
            node_name="vis-lab",
            admitted_gpu_uuid=gpu_uuid,
            visible_gpu_uuids=[gpu_uuid],
            visible_gpu_names=["NVIDIA RTX A6000"],
        )
    with pytest.raises(ValueError, match="UUID"):
        validate_placement(
            node_name="ubuntu",
            admitted_gpu_uuid=gpu_uuid,
            visible_gpu_uuids=["GPU-" + "b" * 36],
            visible_gpu_names=["NVIDIA RTX A6000"],
        )
    with pytest.raises(ValueError, match="A6000"):
        validate_placement(
            node_name="ubuntu",
            admitted_gpu_uuid=gpu_uuid,
            visible_gpu_uuids=[gpu_uuid],
            visible_gpu_names=["NVIDIA L40S"],
        )
    with pytest.raises(ValueError, match="exactly one"):
        validate_placement(
            node_name="ubuntu",
            admitted_gpu_uuid=gpu_uuid,
            visible_gpu_uuids=[gpu_uuid, "GPU-" + "b" * 36],
            visible_gpu_names=["NVIDIA RTX A6000", "NVIDIA RTX A6000"],
        )


def test_runner_manifest_rejects_changed_version_or_input_hash() -> None:
    validate_manifest = _require("gods_mlops.training.contracts", "validate_manifest_identity")
    fixture_payload = {
        "schema_version": 1,
        "fixture": True,
        "phase": "probe",
        "model_kind": "clip",
        "input_kind": "probe_input",
        "input_id": "task8-clip-probe-v1",
        "config_version": "clip-real-probe-v1",
        "model_id": "openai/clip-vit-base-patch16",
        "model_revision": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
        "pairs": [
            {"image_id": "crop-a", "image_sha256": "3" * 64, "text": "blue coat"},
            {"image_id": "crop-b", "image_sha256": "4" * 64, "text": "red shirt"},
        ],
    }
    input_sha256 = sha256(
        json.dumps(fixture_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    config = {
        **{
            key: fixture_payload[key]
            for key in ("phase", "model_kind", "input_kind", "input_id", "config_version", "model_id", "model_revision")
        },
        "input_sha256": input_sha256,
    }
    manifest = fixture_payload

    validate_manifest(config, manifest)

    with pytest.raises(ValueError):
        validate_manifest({**config, "input_sha256": "f" * 64}, manifest)

    with pytest.raises(ValueError):
        validate_manifest(config, {**manifest, "input_id": "other-probe-v1"})

    with pytest.raises(ValueError, match="content hash"):
        validate_manifest(
            config,
            {
                **manifest,
                "pairs": [{**manifest["pairs"][0], "image_sha256": "5" * 64}, manifest["pairs"][1]],
            },
        )


def test_typed_probe_input_round_trips_model_phase_config_and_blob_identity() -> None:
    probe_input = _require("gods_mlops.jobs.models", "ProbeInput")
    manifest = {
        "schema_version": 1,
        "fixture": True,
        "phase": "probe",
        "model_kind": "detr",
        "input_kind": "probe_input",
        "input_id": "task8-detr-real-probe-v1",
        "config_version": "detr-probe-config-v1",
        "model_id": "PekingU/rtdetr_v2_r18vd",
        "model_revision": "5650961749fa93567c0d46fc7f43ea4f9e914107",
        "items": [],
    }
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    value = probe_input(
        probe_input_id="task8-detr-real-probe-v1",
        model_kind="detr",
        target_phase="training",
        config_version="detr-probe-config-v1",
        manifest_object_key="probes/task8/detr.json",
        input_sha256=sha256(payload).hexdigest(),
        object_size_bytes=len(payload),
        fixture=True,
    )

    assert probe_input.from_dict(value.as_dict()) == value
    assert value.input_kind == "probe_input"

    objects = _ImmutableObjects()
    objects.values[value.manifest_object_key] = payload
    assert value.verify(objects) == manifest

    class TamperedObjects:
        def read_source(self, **kwargs):
            return payload + b"tampered"

    with pytest.raises(ValueError, match="frozen size or SHA-256"):
        value.verify(TamperedObjects())

    with pytest.raises(ValueError, match="SHA-256"):
        probe_input.from_dict({**value.as_dict(), "input_sha256": "x" * 64})

    with pytest.raises(ValueError, match="Qwen|training"):
        probe_input(
            probe_input_id="task8-qwen-probe-v1",
            model_kind="qwen",
            target_phase="training",
            config_version="qwen-probe-config-v1",
            manifest_object_key="probes/task8/qwen.json",
            input_sha256=sha256(payload).hexdigest(),
            object_size_bytes=len(payload),
            fixture=True,
        )


def test_versioned_probe_profiles_and_probe_only_synthetic_media_are_separate_from_dataset_inputs() -> None:
    module = importlib.import_module("gods_mlops.training.probe_setup")

    class Objects:
        def __init__(self):
            self.values = {}

        def write_immutable(self, *, object_key, content, sha256_digest, content_type):
            assert sha256(content).hexdigest() == sha256_digest
            assert content_type in {"image/jpeg", "application/json"}
            current = self.values.get(object_key)
            assert current is None or current == content
            self.values[object_key] = content

        def read_source(self, *, object_key, sha256_digest, size_bytes):
            value = self.values[object_key]
            assert len(value) == size_bytes
            assert sha256(value).hexdigest() == sha256_digest
            return value

    expected_versions = {
        "detr": "task8-detr-640-microbatch1-probe-v1",
        "clip": "task9-clip-224-microbatch2-symmetric-ce-fp64-probe-v1",
        "qwen": "task8-qwen-bounded-crop-caption-probe-v1",
    }
    for model_kind, config_version in expected_versions.items():
        profile = module.candidate_profile(model_kind)
        assert profile.phase == "probe"
        assert profile.config_version == config_version
        assert profile.candidate is True
        if model_kind == "clip":
            assert profile.config["micro_batch"] == 2
            assert profile.config["gradient_accumulation_steps"] == 1
            assert profile.config["training_loss_reduction_precision"] == "float64"
        if model_kind == "qwen":
            assert profile.target_phase == "preparation"

        objects = Objects()
        probe_input = module.create_probe_input(objects=objects, model_kind=model_kind)
        manifest = (
            probe_input.verify(
                objects,
                expected_manifest_config_version=profile.config["probe_manifest_config_version"],
            )
            if model_kind == "clip"
            else probe_input.verify(objects)
        )
        assert probe_input.fixture is True
        assert probe_input.input_kind == "probe_input"
        assert "dataset_version" not in manifest
        if model_kind == "clip":
            assert len(manifest["pairs"]) == 2
            assert all(pair["negative_texts"] for pair in manifest["pairs"])
        else:
            expected_item_kind = "frame" if model_kind == "detr" else "crop"
            assert all(item["item_kind"] == expected_item_kind for item in manifest["items"])

    prep_profile = module.candidate_profile("detr", target_phase="preparation")
    assert prep_profile.target_phase == "preparation"
    assert prep_profile.config_version == "task8-detr-640-frame-drafts-preparation-probe-v1"
    assert prep_profile.config["max_draft_frames"] == 1
    prep_objects = Objects()
    prep_input = module.create_probe_input(
        objects=prep_objects, model_kind="detr", target_phase="preparation"
    )
    prep_manifest = prep_input.verify(prep_objects)
    assert prep_input.target_phase == "preparation"
    assert len(prep_manifest["items"]) == 1


def test_training_cli_dispatches_detr_preparation_measurement_target(monkeypatch) -> None:
    cli = _require("gods_mlops.training.cli", "main")
    docker_probe = importlib.import_module("gods_mlops.training.docker_probe")
    captured = {}

    async def fake_probe(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(docker_probe, "run_docker_model_probe", fake_probe)
    result = cli(
        [
            "run-probe",
            "--model-kind",
            "detr",
            "--target-phase",
            "preparation",
            "--worker-image",
            "registry.example/gods-training@sha256:" + "a" * 64,
            "--worker-image-id",
            "sha256:" + "b" * 64,
        ]
    )

    assert result == 0
    assert captured["model_kind"] == "detr"
    assert captured["target_phase"] == "preparation"


def test_training_runtime_locks_scipy_for_transformers_rtdetr_hungarian_loss() -> None:
    import tomllib
    from pathlib import Path

    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scipy = next(
        dependency for dependency in project["project"]["dependencies"] if dependency.startswith("scipy")
    )

    assert scipy == "scipy==1.18.1"


def test_strict_ssh_model_probe_accepts_only_loopback_service_tunnels() -> None:
    module = _require("gods_mlops.training.docker_probe", "_loopback_database")
    loopback_endpoint = _require("gods_mlops.training.docker_probe", "_loopback_endpoint")
    rewrite = _require("gods_mlops.training.docker_probe", "_remote_database_url")

    assert module("postgresql://probe:secret@127.0.0.1:15439/probe") == ("127.0.0.1", 15439)
    assert loopback_endpoint("http://localhost:18333") == ("localhost", 18333)
    tunneled = rewrite("postgresql://probe:secret@127.0.0.1:15439/probe", 35439)
    assert tunneled == "postgresql://probe:secret@127.0.0.1:35439/probe"
    with pytest.raises(ValueError, match="loopback PostgreSQL"):
        module("postgresql://probe:secret@10.0.0.5:15439/probe")
    with pytest.raises(ValueError, match="loopback S3"):
        loopback_endpoint("http://storage.internal:18333")
    with pytest.raises(ValueError, match="embed credentials"):
        loopback_endpoint("http://probe:secret@127.0.0.1:18333")


def test_docker_probe_starts_only_a_pinned_uid_worker_with_readonly_cache_and_loopback_services(
    monkeypatch,
) -> None:
    module = importlib.import_module("gods_mlops.training.docker_probe")
    claim_type = _require("gods_mlops.training.claims", "WorkerClaim")
    job, lease, _profile = _probe_job()
    claim = claim_type.from_admitted_job(job, lease, image_id="sha256:" + "a" * 64)
    for name, value in {
        "GODS_MLOPS_S3_ENDPOINT_URL": "http://127.0.0.1:18333",
        "GODS_MLOPS_S3_BUCKET": "gods-test",
        "GODS_MLOPS_S3_ACCESS_KEY": "test-access",
        "GODS_MLOPS_S3_SECRET_KEY": "test-secret",
    }.items():
        monkeypatch.setenv(name, value)
    environment = module._worker_environment(
        claim=claim,
        database_url="postgresql://test:secret@127.0.0.1:15439/gods",
        database_remote_port=35439,
        storage_remote_port=38333,
        image_id="sha256:" + "a" * 64,
        gpu_uuid="GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
    )

    assert environment["GODS_MLOPS_DATABASE_URL"] == "postgresql://test:secret@127.0.0.1:35439/gods"
    assert environment["GODS_MLOPS_S3_ENDPOINT_URL"] == "http://127.0.0.1:38333"
    assert environment["GODS_MLOPS_LEASE_TOKEN"] == lease["lease_token"]
    assert environment["GODS_MLOPS_FENCE"] == "2"
    assert environment["GODS_MLOPS_MODEL_CACHE_ROOT"] == "/mnt/model-cache"
    assert environment["GODS_MLOPS_MODEL_LOCK"] == "/app/models/lock.json"
    assert "GODS_MLOPS_WORKER_ARTIFACT_DEADLINE_UTC" not in environment
    assert "GODS_MLOPS_WORKER_ARTIFACT_INVOCATION_ID" not in environment

    captured = {}

    async def fake_ssh(target, port, identity_file, known_hosts, command, *, input_bytes=None, timeout=30):
        captured.update(
            target=target,
            command=command,
            payload=json.loads(input_bytes),
            timeout=timeout,
        )
        return "d" * 64

    monkeypatch.setattr(module, "_remote_ssh", fake_ssh)
    worker_image = "registry.example/gods-training@sha256:" + "b" * 64
    container_id = asyncio.run(
        module._start_worker_container(
            ssh_target="ubuntu-test",
            ssh_port=22,
            identity_file="/tmp/probe-key",
            known_hosts="/tmp/probe-known-hosts",
            image=worker_image,
            container_name="gods-task8-test",
            gpu_uuid="GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
            model_cache_root="/data/model-cache",
            environment=environment,
        )
    )

    args = captured["payload"]["docker_args"]
    assert container_id == "d" * 64
    assert captured["command"] == ["python3", "-c", module._REMOTE_DOCKER_RUNNER]
    assert "--pid=host" in args
    assert "--network=host" in args
    assert "--user" in args and args[args.index("--user") + 1] == "10001:10001"
    assert "--gpus" in args and args[args.index("--gpus") + 1] == "device=GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
    assert "type=bind,source=/data/model-cache,target=/mnt/model-cache,readonly" in args
    assert args[-2:] == [worker_image, "worker"]
    assert captured["payload"]["environment"] == environment
    assert "test-secret" not in " ".join(captured["command"] + args)


def test_probe_remote_docker_runner_writes_private_ephemeral_environment_file(monkeypatch, capsys) -> None:
    import io
    import subprocess
    import sys
    from pathlib import Path

    module = importlib.import_module("gods_mlops.training.docker_probe")
    expected_environment = {"GODS_MLOPS_SECRET": "private-test-secret", "GODS_MLOPS_FENCE": "8"}
    stdin = io.StringIO(
        json.dumps(
            {
                "environment": expected_environment,
                "docker_args": ["--detach", "registry.example/gods-training@sha256:" + "b" * 64, "worker"],
            }
        )
    )
    seen = {}

    def fake_run(args, **kwargs):
        env_path = Path(args[args.index("--env-file") + 1])
        seen["env_path"] = env_path
        seen["mode"] = env_path.stat().st_mode & 0o777
        seen["content"] = env_path.read_text(encoding="utf-8")
        seen["args"] = args
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(args, 0, "d" * 64, "")

    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(subprocess, "run", fake_run)
    exec(module._REMOTE_DOCKER_RUNNER, {})

    assert seen["mode"] == 0o600
    assert seen["content"] == "GODS_MLOPS_FENCE=8\nGODS_MLOPS_SECRET=private-test-secret\n"
    assert seen["args"][-3:] == ["--detach", "registry.example/gods-training@sha256:" + "b" * 64, "worker"]
    assert not seen["env_path"].exists()
    assert "private-test-secret" not in capsys.readouterr().out


def test_training_probe_finalizer_registers_the_exact_readback_evidence_bytes(monkeypatch) -> None:
    from types import SimpleNamespace

    from gods_mlops.jobs.checkpoints import CheckpointIdentity

    module = importlib.import_module("gods_mlops.training.docker_probe")
    identity = CheckpointIdentity(
        job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
        input_kind="probe_input",
        input_id="task8-clip-probe-v1",
        input_sha256="1" * 64,
        phase="probe",
        model_kind="clip",
        config_version="clip-real-probe-v1",
        config_sha256="2" * 64,
        dataset_version=None,
    )
    checkpoint_digest = "3" * 64
    checkpoint_commit = {
        "uri": f"s3://gods-test/jobs/{identity.job_id}/checkpoints/{'4' * 64}/{checkpoint_digest}.checkpoint",
        "sha256": checkpoint_digest,
        "size_bytes": 1234,
        "identity": identity.as_dict(),
    }
    evidence_value = {
        "event": "task8_real_model_probe_complete",
        "job_id": identity.job_id,
        "model_kind": "clip",
        "target_phase": "training",
        "config_version": identity.config_version,
        "input_sha256": identity.input_sha256,
        "docker_image_id": "sha256:" + "a" * 64,
        "image_source_commit": "b" * 40,
        "source_commit": "b" * 40,
    }
    evidence = SimpleNamespace(target_phase="training", job_id=identity.job_id, as_dict=lambda: evidence_value)
    order = []

    class Repository:
        async def checkpoint_identity(self, _job_id):
            return identity

        async def checkpoint_metadata_for(self, _job_id):
            return checkpoint_commit

        async def _record_probe_runtime_evidence(self, payload):
            order.append(("register", payload))
            return {"evidence_sha256": sha256(payload).hexdigest()}

    class CheckpointStore:
        def __init__(self, *, objects, bucket):
            self.objects = objects
            self.bucket = bucket

        def load_uri(self, uri, *, expected_identity, expected_sha256, expected_size_bytes):
            order.append(("checkpoint_read", uri, expected_sha256, expected_size_bytes))
            return SimpleNamespace(
                identity=expected_identity,
                sha256=expected_sha256,
                size_bytes=expected_size_bytes,
            )

    monkeypatch.setattr("gods_mlops.training.checkpoints.S3CheckpointStore", CheckpointStore)
    payload = asyncio.run(
        module._finalize_probe_runtime_evidence(
            repository=Repository(),
            objects=object(),
            bucket="gods-test",
            evidence=evidence,
        )
    )

    expected = (json.dumps(evidence_value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert payload == expected
    assert order[0] == ("checkpoint_read", checkpoint_commit["uri"], checkpoint_digest, 1234)
    assert order[1] == ("register", expected)


def test_clip_runner_requires_versioned_distinct_negative_pairs() -> None:
    validate_clip_pairs = _require("gods_mlops.training.clip", "validate_contrastive_pairs")
    good = {
        "contrastive_config_version": "clip-explicit-negatives-v1",
        "pairs": [
            {"image_id": "crop-a", "text": "blue jacket", "negative_texts": ["red shirt"]},
            {"image_id": "crop-b", "text": "red shirt", "negative_texts": ["blue jacket"]},
        ],
    }

    assert validate_clip_pairs(good) == 2

    with pytest.raises(ValueError, match="at least two"):
        validate_clip_pairs({**good, "pairs": good["pairs"][:1]})

    with pytest.raises(ValueError, match="negative"):
        validate_clip_pairs({**good, "pairs": [{"image_id": "crop-a", "text": "blue jacket"}, good["pairs"][1]]})


def test_training_media_reader_verifies_object_bytes_against_immutable_manifest_hash() -> None:
    read_verified_media = _require("gods_mlops.training.data", "read_verified_media")
    expected = b"immutable dataset crop bytes"
    item = {
        "kind": "crop",
        "item_id": "crop-17",
        "object": {
            "key": "datasets/dataset-v1/media/crop-17.jpg",
            "sha256": sha256(expected).hexdigest(),
            "size_bytes": len(expected),
        },
    }

    class ObjectStore:
        def __init__(self, value: bytes) -> None:
            self.value = value
            self.read = None

        def read_source(self, **kwargs):
            self.read = kwargs
            return self.value

    valid_store = ObjectStore(expected)
    assert read_verified_media(item, object_store=valid_store) == expected
    assert valid_store.read == {
        "object_key": item["object"]["key"],
        "sha256_digest": item["object"]["sha256"],
        "size_bytes": item["object"]["size_bytes"],
    }

    corrupt_store = ObjectStore(b"changed dataset crop")
    with pytest.raises(ValueError, match="SHA-256"):
        read_verified_media(item, object_store=corrupt_store)


def test_detector_target_conversion_uses_the_frozen_labelstudio_box_and_person_class() -> None:
    to_coco = _require("gods_mlops.training.detector", "to_coco_annotations")
    item = {
        "item_id": "frame-1",
        "snapshot": {
            "bbox_annotation": {
                "result": [
                    {
                        "from_name": "bbox",
                        "type": "rectanglelabels",
                        "original_width": 100,
                        "original_height": 50,
                        "value": {
                            "x": 20,
                            "y": 20,
                            "width": 30,
                            "height": 40,
                            "rectanglelabels": ["person"],
                        },
                    }
                ]
            }
        },
    }

    coco = to_coco(item, image_width=200, image_height=100, person_class_id=1)

    assert coco == {
        "image_id": "frame-1",
        "annotations": [
            {"bbox": [40.0, 20.0, 60.0, 40.0], "category_id": 1, "area": 2400.0, "iscrowd": 0}
        ],
    }


def test_detector_moves_mapping_based_processor_labels_to_device(monkeypatch) -> None:
    import sys
    from collections import UserDict
    from types import SimpleNamespace

    class FakeTensor:
        def __init__(self, device):
            self.device = device

        def to(self, device):
            return FakeTensor(device)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=FakeTensor))
    move_inputs = _require("gods_mlops.training.detector", "_move_inputs")
    target_device = object()
    processor_labels = UserDict(
        {
            "class_labels": FakeTensor("cpu"),
            "boxes": FakeTensor("cpu"),
            "area": FakeTensor("cpu"),
            "iscrowd": FakeTensor("cpu"),
        }
    )

    moved = move_inputs(
        {"pixel_values": FakeTensor("cpu"), "labels": [processor_labels]},
        target_device,
    )

    assert moved["pixel_values"].device is target_device
    assert moved["labels"][0]["class_labels"].device is target_device
    assert moved["labels"][0]["boxes"].device is target_device


def test_qwen_generation_config_bounds_prompt_image_and_new_tokens() -> None:
    validate = _require("gods_mlops.training.caption", "validate_generation_config")
    allowed = {
        "max_input_tokens": 4096,
        "max_new_tokens": 128,
        "max_image_pixels": 1_048_576,
    }

    validate(allowed)

    with pytest.raises(ValueError, match="max_new_tokens"):
        validate({**allowed, "max_new_tokens": 129})

    with pytest.raises(ValueError, match="max_input_tokens"):
        validate({**allowed, "max_input_tokens": 4097})

    with pytest.raises(ValueError, match="max_image_pixels"):
        validate({**allowed, "max_image_pixels": 1_048_577})


def test_task5_review_handoff_requires_distinct_configured_projects_and_exact_prediction_shapes(
    monkeypatch,
) -> None:
    handoff = _require("gods_mlops.training.review_handoff", "_project_id")
    error_type = _require("gods_mlops.training.review_handoff", "PreparationReviewHandoffError")
    build_prediction = _require("gods_mlops.training.review_handoff", "_prediction")
    validate_project = _require("gods_mlops.training.review_handoff", "validate_project_configuration")
    monkeypatch.delenv("GODS_MLOPS_LABEL_STUDIO_BBOX_PROJECT_ID", raising=False)
    monkeypatch.delenv("GODS_MLOPS_LABEL_STUDIO_CAPTION_PROJECT_ID", raising=False)

    with pytest.raises(error_type) as missing_bbox:
        handoff("detr")
    assert missing_bbox.value.reason_code == "preparation_assignment_project_missing"
    with pytest.raises(error_type) as missing_caption:
        handoff("qwen")
    assert missing_caption.value.reason_code == "preparation_assignment_project_missing"

    monkeypatch.setenv("GODS_MLOPS_LABEL_STUDIO_BBOX_PROJECT_ID", "17")
    monkeypatch.setenv("GODS_MLOPS_LABEL_STUDIO_CAPTION_PROJECT_ID", "23")
    assert handoff("detr") == 17
    assert handoff("qwen") == 23

    validate_project(
        "detr",
        "<View><Image name='image' value='$image'/><RectangleLabels name='bbox' toName='image'>"
        "<Label value='person'/></RectangleLabels></View>",
    )
    validate_project(
        "qwen",
        "<View><Image name='image' value='$image'/><TextArea name='caption' toName='image' required='true'/></View>",
    )
    with pytest.raises(error_type) as mismatched_project:
        validate_project(
            "qwen",
            "<View><Image name='image' value='$image'/><RectangleLabels name='bbox' toName='image'>"
            "<Label value='person'/></RectangleLabels></View>",
        )
    assert mismatched_project.value.reason_code == "preparation_assignment_project_mismatch"

    job = {
        "job_id": "a0320b59-663c-4cdc-b893-086bb970ea60",
        "config_sha256": "2" * 64,
    }
    detr = build_prediction(
        job=job,
        model_id="PekingU/rtdetr_v2_r18vd",
        model_revision="5650961749fa93567c0d46fc7f43ea4f9e914107",
        result_sha256="3" * 64,
        draft={
            "image_width": 200,
            "image_height": 100,
            "detections": [{"score": 0.9, "bbox_xyxy": [40, 20, 100, 60]}],
        },
        model_kind="detr",
    )
    assert detr["result"][0]["from_name"] == "bbox"
    assert detr["result"][0]["value"] == {
        "x": 20.0,
        "y": 20.0,
        "width": 30.0,
        "height": 40.0,
        "rotation": 0,
        "rectanglelabels": ["person"],
    }

    qwen = build_prediction(
        job=job,
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        model_revision="cc594898137f460bfe9f0759e9844b3ce807cfb5",
        result_sha256="4" * 64,
        draft={"caption": "dark jacket and blue trousers"},
        model_kind="qwen",
    )
    assert qwen["result"] == [
        {
            "from_name": "caption",
            "to_name": "image",
            "type": "textarea",
            "value": {"text": ["dark jacket and blue trousers"]},
        }
    ]


class _ImmutableObjects:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.lost_ack_once = False
        self.puts = 0

    def write_immutable(self, *, object_key: str, content: bytes, sha256_digest: str, content_type: str) -> None:
        assert sha256(content).hexdigest() == sha256_digest
        current = self.values.get(object_key)
        if current is not None and current != content:
            raise OSError("immutable object key contains different bytes")
        if current is None:
            self.values[object_key] = content
            self.puts += 1
        if self.lost_ack_once:
            self.lost_ack_once = False
            raise OSError("lost object acknowledgement")

    def read_source(self, *, object_key: str, sha256_digest: str, size_bytes: int) -> bytes:
        content = self.values[object_key]
        assert len(content) == size_bytes
        assert sha256(content).hexdigest() == sha256_digest
        return content

    def delete_object(self, object_key: str) -> None:
        self.values.pop(object_key, None)


def test_s3_result_artifact_lost_ack_retry_reuses_verified_immutable_object() -> None:
    module = _require("gods_mlops.training.artifacts", "S3ResultArtifactStore")
    objects = _ImmutableObjects()
    objects.lost_ack_once = True
    identity_type = _require("gods_mlops.jobs.checkpoints", "CheckpointIdentity")
    identity = identity_type(
        job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
        input_kind="probe_input",
        input_id="task8-artifact-probe-v1",
        input_sha256="1" * 64,
        phase="probe",
        model_kind="detr",
        config_version="detr-real-probe-v1",
        config_sha256="2" * 64,
        dataset_version=None,
    )
    store = module(objects=objects, bucket="gods-test", prefix="task8")
    prepared = store.prepare(
        identity=identity,
        kind="model",
        payload=b"immutable model bundle",
        reservation_bytes=1024,
    )

    with pytest.raises(OSError, match="lost object acknowledgement"):
        store.commit(prepared)
    committed = store.commit(prepared)
    retry = store.commit(prepared)

    assert committed.uri == retry.uri
    assert committed.sha256 == sha256(b"immutable model bundle").hexdigest()
    assert objects.puts == 1
    assert objects.read_source(
        object_key=prepared.object_key,
        sha256_digest=committed.sha256,
        size_bytes=committed.size_bytes,
    ) == b"immutable model bundle"


def test_s3_checkpoint_retry_and_resume_preserve_full_queue_identity() -> None:
    module = _require("gods_mlops.training.checkpoints", "S3CheckpointStore")
    identity_type = _require("gods_mlops.jobs.checkpoints", "CheckpointIdentity")
    identity = identity_type(
        job_id="a0320b59-663c-4cdc-b893-086bb970ea60",
        input_kind="probe_input",
        input_id="task8-checkpoint-probe-v1",
        input_sha256="1" * 64,
        phase="probe",
        model_kind="clip",
        config_version="clip-real-probe-v1",
        config_sha256="2" * 64,
        dataset_version=None,
    )
    objects = _ImmutableObjects()
    objects.lost_ack_once = True
    store = module(objects=objects, bucket="gods-test", prefix="task8")
    prepared = store.prepare(
        identity=identity,
        payload=b"model optimizer global step 3",
        reservation_bytes=1024,
    )

    with pytest.raises(OSError, match="lost object acknowledgement"):
        store.commit(prepared)
    committed = store.commit(prepared)
    resumed = store.load_uri(
        committed.uri,
        expected_identity=identity,
        expected_sha256=committed.sha256,
        expected_size_bytes=committed.size_bytes,
    )

    assert resumed.payload == b"model optimizer global step 3"
    assert resumed.identity == identity
    assert resumed.sha256 == committed.sha256
    assert objects.puts == 1
