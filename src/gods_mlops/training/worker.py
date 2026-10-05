"""Fenced entry point for one admitted Kubernetes GPU Job."""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from concurrent.futures import TimeoutError as FutureTimeout
from hashlib import sha256
from pathlib import Path
from typing import Any

from gods_mlops.datasets.manifest import canonical_json
from gods_mlops.datasets.publish import DatasetPublisher
from gods_mlops.jobs.models import ProbeInput
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

from .artifacts import S3ResultArtifactStore
from .checkpoints import S3CheckpointStore
from .claims import (
    WorkerAuthorizationError,
    WorkerClaim,
    WorkerYieldRequested,
    validate_current_worker_claim,
)
from .data import dataset_object_store_from_environment


async def run_worker() -> int:
    """Re-read the exact lease and source before importing any model/GPU runtime."""
    database_url = _required("GODS_MLOPS_DATABASE_URL")
    job_id = _required("GODS_MLOPS_JOB_ID")
    lease_token = _required("GODS_MLOPS_LEASE_TOKEN")
    image_id = _required("GODS_MLOPS_IMAGE_ID")
    gpu_uuid = _required("GODS_MLOPS_GPU_UUID")
    repository = PostgresJobQueueRepository(database_url=database_url)
    sources = DatasetSourceRegistry(database_url=database_url)
    queue = JobQueue(repository=repository, sources=sources)
    objects = dataset_object_store_from_environment()
    try:
        job = await queue.get(job_id)
        lease = await repository.get_active_lease(gpu_uuid)
        if lease is None or lease.get("job_id") != job_id:
            raise WorkerAuthorizationError("worker has no matching current GPU lease")
        profile = await repository.get_profile(
            phase=job["phase"], model_kind=job["model_kind"], config_version=job["config_version"]
        )
        claim = WorkerClaim.from_admitted_job(job, lease, image_id=image_id)
        if claim.lease_token != lease_token:
            raise WorkerAuthorizationError("worker lease token differs from its current database fence")
        _verify_claim_environment(claim)
        job, lease = await validate_current_worker_claim(queue, claim, object_store=objects)
        if profile is None:
            raise WorkerAuthorizationError("worker profile disappeared before model execution")

        model = __import__("gods_mlops.training.contracts", fromlist=["locked_model"]).locked_model(
            claim.model_kind
        )
        config = dict(profile["config_json"])
        config.update(
            {
                "job_id": claim.job_id,
                "phase": claim.phase,
                "target_phase": claim.target_phase,
                "input_kind": claim.input_kind,
                "input_id": claim.input_id,
                "input_sha256": claim.input_sha256,
                "dataset_version": claim.dataset_version,
                "model_kind": claim.model_kind,
                "model_id": model.model_id,
                "model_revision": model.revision,
                "config_version": claim.config_version,
                "config_sha256": claim.config_sha256,
                "image_id": claim.image_id,
                "gpu_uuid": claim.gpu_uuid,
                "model_cache_root": os.environ.get("GODS_MLOPS_MODEL_CACHE_ROOT", "/mnt/model-cache"),
            }
        )
        identity = await repository.checkpoint_identity(job_id)
        checkpoint_store = S3CheckpointStore(objects=objects, bucket=_required("GODS_MLOPS_S3_BUCKET"))
        result_store = S3ResultArtifactStore(objects=objects, bucket=_required("GODS_MLOPS_S3_BUCKET"))
        if claim.phase == "training" or (claim.phase == "probe" and claim.target_phase == "training"):
            previous = await queue.load_checkpoint(store=checkpoint_store, job_id=job_id)
            if previous is not None:
                config["_resume_payload"] = previous.payload

        event_loop = asyncio.get_running_loop()

        async def still_current() -> bool:
            current = await queue.get(job_id)
            return current.get("state") == "running" and await repository.lease_is_current(
                job_id, lease_token
            )

        def from_runner(coroutine):
            future = asyncio.run_coroutine_threadsafe(coroutine, event_loop)
            try:
                return future.result(timeout=30)
            except FutureTimeout as error:
                future.cancel()
                raise RuntimeError("fenced Task 7 database operation timed out") from error

        def assert_current() -> bool:
            return bool(from_runner(still_current()))

        def commit_checkpoint(payload: bytes):
            return from_runner(
                queue.save_checkpoint(
                    store=checkpoint_store,
                    job_id=job_id,
                    lease_token=lease_token,
                    identity=identity,
                    payload=payload,
                )
            )

        def commit_result(kind: str, payload: bytes):
            return from_runner(
                queue.save_result_artifact(
                    store=result_store,
                    job_id=job_id,
                    lease_token=lease_token,
                    identity=identity,
                    kind=kind,
                    payload=payload,
                )
            )

        config["_assert_current"] = assert_current
        config["_commit_checkpoint"] = commit_checkpoint
        config["_commit_result_artifact"] = commit_result
        if not await still_current():
            raise WorkerAuthorizationError("worker lease was revoked before CUDA startup")

        with tempfile.TemporaryDirectory(prefix="gods-mlops-worker-") as stage_directory:
            stage_root = Path(stage_directory)
            manifest_root = stage_root / "manifest"
            manifest_root.mkdir(mode=0o700)
            output_root = stage_root / "output"
            result = None
            try:
                manifest_uri, extra = await _worker_manifest(
                    job, queue, objects, root=manifest_root
                )
                config.update(extra)
                result = await _run_after_owner_binding(
                    repository,
                    claim,
                    lambda: _runner_for(claim.phase, claim.model_kind, claim.target_phase)(
                        config, manifest_uri, str(output_root)
                    ),
                )
            except WorkerYieldRequested:
                return 0
        if result.get("status") == "yielded":
            payload = result.get("checkpoint_payload")
            if isinstance(payload, bytes) and payload:
                await queue.save_checkpoint(
                    store=checkpoint_store,
                    job_id=job_id,
                    lease_token=lease_token,
                    identity=identity,
                    payload=payload,
                )
            return 0
        if result.get("status") != "succeeded":
            raise RuntimeError("model runner did not complete successfully")

        checkpoint_digest = result.get("checkpoint_sha256")
        checkpoint_payload = result.get("checkpoint_payload")
        if isinstance(checkpoint_payload, bytes) and checkpoint_payload:
            verified = await queue.save_checkpoint(
                store=checkpoint_store,
                job_id=job_id,
                lease_token=lease_token,
                identity=identity,
                payload=checkpoint_payload,
            )
            checkpoint_digest = verified.sha256
        result_payload = result.get("result_artifact_payload")
        if isinstance(result_payload, bytes) and result_payload:
            artifact = await queue.save_result_artifact(
                store=result_store,
                job_id=job_id,
                lease_token=lease_token,
                identity=identity,
                kind=str(result["result_artifact_kind"]),
                payload=result_payload,
            )
            result["result_uri"] = artifact.uri

        measurements = result["resource_measurements"]
        if claim.phase == "probe":
            verification = _probe_verification(claim, result, measurements, checkpoint_digest)
            measured = await queue.record_probe_measurement(
                job_id=job_id,
                lease_token=lease_token,
                exit_code=0,
                peak_allocated_mib=_optional_int(measurements.get("peak_vram_allocated_mib")),
                peak_reserved_mib=_optional_int(measurements.get("peak_vram_reserved_mib")),
                optimizer_steps=int(measurements.get("optimizer_steps", 0)),
                checkpoint_resumed=measurements.get("checkpoint_resumed") is True,
                checkpoint_sha256=checkpoint_digest,
                inference_steps=int(measurements.get("inference_steps", 0)),
                verification_details=verification,
            )
            return 0 if measured.get("result_state") == "succeeded" else 1

        if claim.phase == "training":
            publisher = DatasetPublisher.from_environment()
            try:
                await publisher.register_model_lineage(
                    model_id=str(result.get("result_uri") or f"gods-job-{job_id}"),
                    dataset_version=str(claim.dataset_version),
                )
            finally:
                await publisher.close()
        await queue.complete_owned_job(
            job_id=job_id,
            lease_token=lease_token,
            identity=identity,
            details={
                "result_uri": result.get("result_uri"),
                "result_sha256": result.get("hash"),
                "model_id": model.model_id,
                "model_revision": model.revision,
                "resource_measurements": measurements,
            },
        )
        return 0
    finally:
        await repository.close()
        await sources.close()


async def _worker_manifest(
    job: dict[str, Any], queue: JobQueue, objects: Any, *, root: Path
) -> tuple[str, dict[str, Any]]:
    phase = str(job["phase"])
    refs = job.get("source_refs")
    if not isinstance(refs, dict):
        raise WorkerAuthorizationError("worker source references are unavailable")
    if phase == "training":
        reference = await queue.source_registry.training_manifest_reference(str(job["dataset_version"]))
        if reference["sha256"] != job["input_sha256"]:
            raise WorkerAuthorizationError("published manifest hash differs from the queued training input")
        bucket = _required("GODS_MLOPS_S3_BUCKET")
        return (
            f"s3://{bucket}/{reference['object_key']}",
            {"manifest_size_bytes": reference["size_bytes"]},
        )
    if phase == "probe":
        if refs.get("schema") != "gods-mlops-probe-input-v1":
            raise WorkerAuthorizationError("real model probes require typed immutable probe input refs")
        probe_input = ProbeInput.from_dict(refs)
        manifest = probe_input.verify(objects)
        path = root / "probe-manifest.json"
        path.write_bytes(canonical_json(manifest))
        return str(path), {}
    if phase == "preparation":
        items = refs.get("items")
        if not isinstance(items, list) or not items:
            raise WorkerAuthorizationError("preparation batch refs are empty")
        manifest = {
            "schema_version": 1,
            "batch_id": refs.get("batch_id"),
            "input_id": refs.get("batch_id"),
            "input_sha256": refs.get("input_sha256"),
            "model_kind": job["model_kind"],
            "config_version": job["config_version"],
            "items": items,
        }
        payload = canonical_json(manifest)
        path = root / "preparation-manifest.json"
        path.write_bytes(payload)
        return str(path), {}
    raise WorkerAuthorizationError("worker phase is unsupported")


async def _run_after_owner_binding(
    repository: PostgresJobQueueRepository,
    claim: WorkerClaim,
    runner,
    *,
    timeout_seconds: int = 60,
    poll_interval_seconds: float = 1.0,
    sleep=asyncio.sleep,
    clock=time.monotonic,
):
    """Gate every CUDA/model import behind a fresh host-PID owner binding."""
    if timeout_seconds <= 0 or poll_interval_seconds <= 0:
        raise ValueError("worker owner-binding wait bounds must be positive")
    deadline = clock() + timeout_seconds
    while clock() < deadline:
        job = await repository.get_job(claim.job_id)
        lease = await repository.get_active_lease(claim.gpu_uuid)
        if (
            job.get("state") != "running"
            or job.get("lease_token") != claim.lease_token
            or lease is None
            or lease.get("job_id") != claim.job_id
            or str(lease.get("lease_token")) != claim.lease_token
            or int(lease.get("fencing_token", -1)) != claim.fence
            or not await repository.lease_is_current(claim.job_id, claim.lease_token)
        ):
            raise WorkerAuthorizationError("worker lease was revoked before its host PID binding")
        owner = (
            lease.get("owner_pid"),
            lease.get("owner_start_ticks"),
            lease.get("owner_uid"),
        )
        if all(value is not None for value in owner):
            if int(owner[0]) <= 0 or int(owner[1]) <= 0 or int(owner[2]) != 10001:
                raise WorkerAuthorizationError("bound host PID/start/UID differs from the training worker")
            return await asyncio.to_thread(runner)
        await sleep(poll_interval_seconds)
    raise WorkerAuthorizationError("worker timed out waiting for its exact host PID/start/UID binding")


def _runner_for(phase: str, model_kind: str, target_phase: str):
    if phase == "probe":
        from .probe import run

        return run
    if phase == "training" and model_kind == "detr":
        from .detector import run

        return run
    if phase == "training" and model_kind == "clip":
        from .clip import run

        return run
    if phase == "preparation" and model_kind == "detr" and target_phase == "preparation":
        from .detector import run

        return run
    if phase == "preparation" and model_kind == "qwen" and target_phase == "preparation":
        from .caption import run

        return run
    raise WorkerAuthorizationError("model runner does not support this phase and target")


def _probe_verification(
    claim: WorkerClaim,
    result: dict[str, Any],
    measurements: dict[str, Any],
    checkpoint_digest: str | None,
) -> dict[str, Any]:
    result_hash = result.get("hash")
    result_hash_valid = isinstance(result_hash, str) and len(result_hash) == 64 and all(
        char in "0123456789abcdef" for char in result_hash
    )
    steps = int(measurements.get("optimizer_steps", 0))
    resumed = measurements.get("checkpoint_resumed") is True
    weights_updated = (
        isinstance(measurements.get("initial_weight_sha256"), str)
        and isinstance(measurements.get("final_weight_sha256"), str)
        and measurements["initial_weight_sha256"] != measurements["final_weight_sha256"]
    )
    training_verified = (
        claim.target_phase == "training"
        and steps >= 3
        and resumed
        and checkpoint_digest is not None
        and weights_updated
    )
    inference_verified = claim.target_phase == "preparation" and int(
        measurements.get("inference_steps", 0)
    ) >= 1
    if claim.model_kind == "clip" and claim.target_phase == "training":
        losses = measurements.get("losses")
        training_verified = training_verified and isinstance(losses, list) and bool(losses) and all(
            isinstance(loss, (int, float)) and loss > 0 for loss in losses
        )
    return {
        "passed": result_hash_valid and (training_verified or inference_verified),
        "model_kind": claim.model_kind,
        "target_phase": claim.target_phase,
        "result_sha256": result_hash,
        "learning_signal_verified": weights_updated if claim.target_phase == "training" else None,
        "checkpoint_resume_verified": resumed,
    }


def _verify_claim_environment(claim: WorkerClaim) -> None:
    fields = {
        "GODS_MLOPS_JOB_ID": claim.job_id,
        "GODS_MLOPS_LEASE_TOKEN": claim.lease_token,
        "GODS_MLOPS_FENCE": str(claim.fence),
        "GODS_MLOPS_GPU_UUID": claim.gpu_uuid,
        "GODS_MLOPS_PHASE": claim.phase,
        "GODS_MLOPS_TARGET_PHASE": claim.target_phase,
        "GODS_MLOPS_INPUT_KIND": claim.input_kind,
        "GODS_MLOPS_INPUT_ID": claim.input_id,
        "GODS_MLOPS_INPUT_SHA256": claim.input_sha256,
        "GODS_MLOPS_DATASET_VERSION": claim.dataset_version or "",
        "GODS_MLOPS_MODEL_KIND": claim.model_kind,
        "GODS_MLOPS_CONFIG_VERSION": claim.config_version,
        "GODS_MLOPS_CONFIG_SHA256": claim.config_sha256,
        "GODS_MLOPS_IMAGE_ID": claim.image_id,
    }
    for name, expected in fields.items():
        if os.environ.get(name) != expected:
            raise WorkerAuthorizationError(f"worker environment does not match the admitted {name} claim")


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise WorkerAuthorizationError(f"required worker environment variable {name} is unavailable")
    return value
