"""Fenced entry point for one admitted Kubernetes GPU Job."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from concurrent.futures import TimeoutError as FutureTimeout
from hashlib import sha256
from pathlib import Path
from typing import Any

from gods_mlops.datasets.manifest import canonical_json
from gods_mlops.datasets.publish import DatasetPublisher
from gods_mlops.jobs.models import EvaluationCheckpointSource, EvaluationProbeCheckpointSource, ProbeInput
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
    evaluation_publisher = None
    evaluation_model_id = None
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

        contracts = __import__(
            "gods_mlops.training.contracts", fromlist=["locked_model", "validate_model_revision"]
        )
        profile_config = dict(profile["config_json"])
        try:
            contracts.validate_model_revision({**profile_config, "model_kind": claim.model_kind})
        except ValueError as error:
            raise WorkerAuthorizationError(
                "worker profile model ID or revision differs from the immutable lock"
            ) from error
        model = contracts.locked_model(claim.model_kind)
        config = {**profile_config, "model_kind": claim.model_kind}
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
        elif claim.phase == "evaluation":
            source_refs = job.get("source_refs")
            if not isinstance(source_refs, dict):
                raise WorkerAuthorizationError("evaluation source references are unavailable")
            checkpoint_source = EvaluationCheckpointSource.from_dict(source_refs.get("checkpoint"))
            if checkpoint_source.model_revision != model.revision:
                raise WorkerAuthorizationError("evaluation checkpoint model revision differs from the immutable lock")
            prior_checkpoint = await _load_verified_evaluation_checkpoint(
                queue,
                checkpoint_store,
                checkpoint_source,
            )
            config["_evaluation_checkpoint_payload"] = prior_checkpoint.payload
            config["_evaluation_checkpoint_identity"] = prior_checkpoint.identity.as_dict()
            config["_evaluation_checkpoint_sha256"] = prior_checkpoint.sha256
            config["_evaluation_checkpoint_model_revision"] = checkpoint_source.model_revision
            evaluation_model_id = "sha256:" + prior_checkpoint.sha256
            evaluation_publisher = DatasetPublisher.from_environment()
            current_lineage = await evaluation_publisher.register_model_lineage(
                model_id=evaluation_model_id,
                dataset_version=checkpoint_source.dataset_version,
            )
            if current_lineage.get("training_eligible") is not True or current_lineage.get(
                "evaluation_eligible"
            ) is not True:
                raise WorkerAuthorizationError("candidate checkpoint is not currently evaluation-eligible")
            config["_evaluation_candidate"] = {
                "training_job_id": checkpoint_source.training_job_id,
                "checkpoint_uri": checkpoint_source.checkpoint_uri,
                "checkpoint_sha256": checkpoint_source.checkpoint_sha256,
                "checkpoint_size_bytes": checkpoint_source.checkpoint_size_bytes,
                "checkpoint_identity": checkpoint_source.checkpoint_identity,
                "model_revision": checkpoint_source.model_revision,
                "model_id": evaluation_model_id,
                "model_kind": checkpoint_source.model_kind,
                "verified": True,
            }
            config["_evaluation_baseline"] = source_refs.get("baseline")
            evaluation_resume = await queue.load_checkpoint(store=checkpoint_store, job_id=job_id)
            if evaluation_resume is not None:
                config["_evaluation_resume_payload"] = evaluation_resume.payload
            config["_evaluation_job_identity"] = identity.as_dict()
        elif claim.phase == "probe" and claim.target_phase == "evaluation":
            source_refs = job.get("source_refs")
            try:
                probe_input = ProbeInput.from_dict(source_refs)
                checkpoint_source = probe_input.evaluation_checkpoint_source
            except (TypeError, ValueError) as error:
                raise WorkerAuthorizationError("evaluation probe checkpoint source is invalid") from error
            if checkpoint_source is None or checkpoint_source.model_revision != model.revision:
                raise WorkerAuthorizationError("evaluation probe has no matching locked prior checkpoint")
            prior_checkpoint = await _load_verified_evaluation_probe_checkpoint(
                queue,
                checkpoint_store,
                checkpoint_source,
            )
            config["_evaluation_checkpoint_payload"] = prior_checkpoint.payload
            config["_evaluation_checkpoint_identity"] = prior_checkpoint.identity.as_dict()
            config["_evaluation_checkpoint_sha256"] = prior_checkpoint.sha256
            config["_evaluation_checkpoint_model_revision"] = checkpoint_source.model_revision
            config["_evaluation_probe_checkpoint_source"] = checkpoint_source.as_dict()
            config["_evaluation_job_identity"] = identity.as_dict()
            evaluation_resume = await queue.load_checkpoint(store=checkpoint_store, job_id=job_id)
            if evaluation_resume is not None:
                config["_evaluation_resume_payload"] = evaluation_resume.payload

        event_loop = asyncio.get_running_loop()

        async def still_current() -> bool:
            return await _worker_claim_at_safe_boundary(queue, claim, object_store=objects)

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

        def commit_result(kind: str, payload: bytes, runtime_measurements=None):
            return from_runner(
                queue.save_result_artifact(
                    store=result_store,
                    job_id=job_id,
                    lease_token=lease_token,
                    identity=identity,
                    kind=kind,
                    payload=payload,
                    runtime_measurements=runtime_measurements,
                )
            )

        config["_assert_current"] = assert_current
        config["_commit_checkpoint"] = commit_checkpoint
        if claim.phase != "evaluation":
            config["_commit_result_artifact"] = commit_result
        if not await still_current():
            raise WorkerAuthorizationError("worker lease was revoked before CUDA startup")

        recovered_exit = await _recover_committed_result(
            job=job,
            queue=queue,
            claim=claim,
            identity=identity,
            objects=objects,
            checkpoint_store=checkpoint_store,
            result_store=result_store,
            model=model,
            contracts=contracts,
            config=config,
        )
        if recovered_exit is not None:
            return recovered_exit

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
                    revalidate=still_current,
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

        if claim.phase == "evaluation":
            if evaluation_publisher is None or evaluation_model_id is None:
                raise WorkerAuthorizationError("evaluation lineage publisher is unavailable")

            prediction_payload = result.get("result_artifact_payload")
            if not isinstance(prediction_payload, bytes) or not prediction_payload:
                raise WorkerAuthorizationError("evaluation runner returned no immutable predictions")
            try:
                predictions = json.loads(prediction_payload)
                manifest = contracts.load_manifest(manifest_uri, config=config)
                contracts.validate_manifest_identity(config, manifest)
            except Exception as error:  # noqa: BLE001 - reject changed inputs before report creation
                raise WorkerAuthorizationError("evaluation prediction or manifest identity is invalid") from error
            current_lineage = await evaluation_publisher.register_model_lineage(
                model_id=evaluation_model_id,
                dataset_version=str(claim.dataset_version),
            )
            checkpoint_source = EvaluationCheckpointSource.from_dict(job["source_refs"]["checkpoint"])
            evaluation_report = _build_owned_evaluation_report(
                manifest=manifest,
                prediction_payload=predictions,
                claim=claim,
                candidate=config["_evaluation_candidate"],
                baseline=config.get("_evaluation_baseline"),
                source={
                    "dataset_version": str(claim.dataset_version),
                    "manifest_uri": manifest_uri,
                    "manifest_sha256": claim.input_sha256,
                    "training_manifest_sha256": checkpoint_source.training_manifest_sha256,
                    "target": manifest.get("target"),
                },
                evaluation_config={
                    "version": claim.config_version,
                    "sha256": claim.config_sha256,
                    "measurement_id": profile.get("measurement_id"),
                    "split": config["evaluation_split"],
                    "settings": profile.get("config_json", {}),
                },
                current_lineage=current_lineage,
                model_revision=model.revision,
            )
            evaluation_report["evaluation_job"] = {
                "job_id": claim.job_id,
                "phase": claim.phase,
                "config_version": claim.config_version,
                "config_sha256": claim.config_sha256,
                "profile_measurement_id": profile.get("measurement_id"),
            }
            evaluation_report["runtime_measurements"] = result.get("resource_measurements", {})
            result_payload = canonical_json(evaluation_report)
            result["hash"] = sha256(result_payload).hexdigest()
            result["result_artifact_kind"] = "evaluation"
            result["result_artifact_payload"] = result_payload

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
        if claim.phase == "probe" and claim.target_phase == "evaluation":
            if not _evaluation_probe_result_source_verified(
                result_payload,
                result.get("hash"),
                config.get("_evaluation_probe_checkpoint_source"),
            ):
                raise WorkerAuthorizationError(
                    "evaluation probe result provenance differs from its typed checkpoint source"
                )
        if isinstance(result_payload, bytes) and result_payload:
            artifact = await queue.save_result_artifact(
                store=result_store,
                job_id=job_id,
                lease_token=lease_token,
                identity=identity,
                kind=str(result["result_artifact_kind"]),
                payload=result_payload,
                runtime_measurements=result.get("resource_measurements"),
            )
            result["result_uri"] = artifact.uri

        measurements = result["resource_measurements"]
        if claim.phase == "probe":
            verification = _probe_verification(
                claim,
                result,
                measurements,
                checkpoint_digest,
                result_artifact_payload=result.get("result_artifact_payload"),
                evaluation_probe_checkpoint_source=config.get("_evaluation_probe_checkpoint_source"),
            )
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
        elif claim.phase == "evaluation":
            if evaluation_publisher is None or evaluation_model_id is None:
                raise WorkerAuthorizationError("evaluation lineage publisher is unavailable")
            final_lineage = await evaluation_publisher.register_model_lineage(
                model_id=evaluation_model_id,
                dataset_version=str(claim.dataset_version),
            )
            if final_lineage.get("evaluation_eligible") is not True:
                raise WorkerAuthorizationError("evaluation eligibility changed before fenced completion")
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
        if evaluation_publisher is not None:
            await evaluation_publisher.close()
        await repository.close()
        await sources.close()


def _build_owned_evaluation_report(
    *,
    manifest: dict[str, Any],
    prediction_payload: dict[str, Any],
    claim: Any,
    candidate: dict[str, Any],
    baseline: dict[str, Any] | None,
    source: dict[str, Any],
    evaluation_config: dict[str, Any],
    current_lineage: dict[str, Any],
    model_revision: str,
) -> dict[str, Any]:
    """Compose raw current authority with the owned prediction and bound runtime identity."""
    from gods_mlops.evaluation.provenance import build_owned_evaluation_provenance
    from gods_mlops.evaluation.report import evaluate_prediction_payload

    if claim.phase != "evaluation" or claim.target_phase != "evaluation":
        raise WorkerAuthorizationError("owned evaluation report requires an evaluation-phase worker claim")
    if candidate.get("model_revision") != model_revision:
        raise WorkerAuthorizationError("candidate checkpoint model revision differs from the loaded model")
    provenance = build_owned_evaluation_provenance(
        worker_image_id=claim.image_id,
        model_revision=model_revision,
        model_kind=claim.model_kind,
        prediction_payload=prediction_payload,
    )
    report = evaluate_prediction_payload(
        manifest=manifest,
        prediction_payload=prediction_payload,
        model_kind=claim.model_kind,
        evaluation_split=str(evaluation_config["split"]),
        candidate=candidate,
        baseline=baseline,
        source=source,
        evaluation_config=evaluation_config,
        current_eligibility=current_lineage,
        evaluator_provenance=provenance,
    )
    if report.get("training_ready") is not True or report.get("evaluation_eligible") is not True:
        raise WorkerAuthorizationError("evaluation eligibility changed before report publication")
    return report


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
    if phase == "evaluation":
        reference = await queue.source_registry.training_manifest_reference(str(job["dataset_version"]))
        if reference["sha256"] != job["input_sha256"]:
            raise WorkerAuthorizationError("published evaluation manifest hash differs from the queued source")
        checkpoint_source = EvaluationCheckpointSource.from_dict(refs.get("checkpoint"))
        if (
            refs.get("dataset_version") != job["dataset_version"]
            or refs.get("manifest_sha256") != reference["sha256"]
            or checkpoint_source.dataset_version != job["dataset_version"]
            or checkpoint_source.training_manifest_sha256 != reference["sha256"]
        ):
            raise WorkerAuthorizationError("evaluation source references do not match the published dataset")
        bucket = _required("GODS_MLOPS_S3_BUCKET")
        return (
            f"s3://{bucket}/{reference['object_key']}",
            {
                "manifest_size_bytes": reference["size_bytes"],
                "evaluation_split": refs["evaluation_split"],
            },
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


async def _load_verified_evaluation_checkpoint(queue, store, source: EvaluationCheckpointSource):
    """Load model bytes only from a successful training job's current DB commit marker."""
    try:
        training_job = await queue.get(source.training_job_id)
        actual_identity = await queue.repository.checkpoint_identity(source.training_job_id)
        metadata = await queue.repository.checkpoint_metadata_for(source.training_job_id)
    except Exception as error:  # noqa: BLE001 - do not expose DB/object identities in worker logs
        raise WorkerAuthorizationError("evaluation training checkpoint metadata is unavailable") from error
    if (
        training_job.get("state") != "completed"
        or training_job.get("phase") != "training"
        or training_job.get("target_phase") != "training"
        or training_job.get("model_kind") != source.model_kind
        or training_job.get("dataset_version") != source.dataset_version
        or training_job.get("input_kind") != "dataset_version"
        or training_job.get("input_id") != source.dataset_version
        or training_job.get("input_sha256") != source.training_manifest_sha256
        or metadata is None
    ):
        raise WorkerAuthorizationError("evaluation requires a successful training checkpoint for this dataset")
    if (
        actual_identity.as_dict() != source.checkpoint_identity
        or metadata.get("identity") != source.checkpoint_identity
        or metadata.get("uri") != source.checkpoint_uri
        or metadata.get("sha256") != source.checkpoint_sha256
        or metadata.get("size_bytes") != source.checkpoint_size_bytes
    ):
        raise WorkerAuthorizationError("evaluation checkpoint metadata differs from the immutable source reference")
    try:
        verified = store.load_uri(
            source.checkpoint_uri,
            expected_identity=actual_identity,
            expected_sha256=source.checkpoint_sha256,
            expected_size_bytes=source.checkpoint_size_bytes,
        )
    except Exception as error:  # noqa: BLE001 - map storage details to a bounded authorization failure
        raise WorkerAuthorizationError("evaluation checkpoint bytes failed immutable verification") from error
    if verified.identity != actual_identity or verified.sha256 != source.checkpoint_sha256:
        raise WorkerAuthorizationError("verified evaluation checkpoint identity changed during load")
    return verified


async def _load_verified_evaluation_probe_checkpoint(
    queue,
    store,
    source: EvaluationProbeCheckpointSource,
):
    """Read weights from a successful training-target probe without its optimizer state."""
    try:
        training_probe = await queue.get(source.training_probe_job_id)
        actual_identity = await queue.repository.checkpoint_identity(source.training_probe_job_id)
        metadata = await queue.repository.checkpoint_metadata_for(source.training_probe_job_id)
        probe_profile = await queue.repository.get_profile(
            phase="probe",
            model_kind=source.model_kind,
            config_version=source.checkpoint_identity["config_version"],
        )
        training_profile = await queue.repository.get_profile(
            phase="training",
            model_kind=source.model_kind,
            config_version=source.checkpoint_identity["config_version"],
        )
        measurement = await queue.repository.profile_measurement_for_job(source.training_probe_job_id)
        runtime_evidence_record = await queue.repository.probe_runtime_evidence_for_job(
            source.training_probe_job_id
        )
        result_artifacts = await queue.repository.result_artifacts_for(source.training_probe_job_id)
    except Exception as error:  # noqa: BLE001 - keep source metadata out of worker logs
        raise WorkerAuthorizationError("evaluation probe training checkpoint metadata is unavailable") from error
    if (
        metadata is None
        or measurement is None
        or probe_profile is None
        or training_profile is None
        or runtime_evidence_record is None
        or len(result_artifacts) != 1
    ):
        raise WorkerAuthorizationError("evaluation probe training checkpoint origin is incomplete")
    source.validate_training_probe_origin(
        training_probe,
        probe_profile,
        training_profile,
        measurement,
        metadata,
        runtime_evidence_record,
        result_artifacts[0],
    )
    if actual_identity.as_dict() != source.checkpoint_identity:
        raise WorkerAuthorizationError("evaluation probe checkpoint identity differs from its DB origin")
    try:
        verified = store.load_uri(
            source.checkpoint_uri,
            expected_identity=actual_identity,
            expected_sha256=source.checkpoint_sha256,
            expected_size_bytes=source.checkpoint_size_bytes,
        )
    except Exception as error:  # noqa: BLE001 - map storage details to a bounded authorization failure
        raise WorkerAuthorizationError("evaluation probe checkpoint bytes failed immutable verification") from error
    if verified.identity != actual_identity or verified.sha256 != source.checkpoint_sha256:
        raise WorkerAuthorizationError("verified evaluation probe checkpoint identity changed during load")
    return verified


def apply_verified_evaluation_probe_checkpoint_weights(
    model: Any,
    config: dict[str, Any],
    *,
    model_kind: str,
    model_revision: str,
) -> None:
    """Load only the verified model state from a training probe checkpoint."""
    import io
    import torch
    from collections.abc import Mapping

    try:
        source = EvaluationProbeCheckpointSource.from_dict(
            config.get("_evaluation_probe_checkpoint_source")
        )
        payload = config.get("_evaluation_checkpoint_payload")
        expected_identity = config.get("_evaluation_checkpoint_identity")
        if (
            not isinstance(payload, bytes)
            or sha256(payload).hexdigest() != source.checkpoint_sha256
            or source.model_kind != model_kind
            or source.model_revision != model_revision
            or source.checkpoint_identity != expected_identity
            or config.get("_evaluation_checkpoint_sha256") != source.checkpoint_sha256
        ):
            raise ValueError("evaluation probe checkpoint binding changed before model load")
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - prevent fallback to pretrained weights
        raise WorkerAuthorizationError("evaluation probe checkpoint bytes failed model-state validation") from error
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("format") != "gods-mlops-training-checkpoint-v1"
        or checkpoint.get("identity") != source.checkpoint_identity
        or checkpoint.get("model_revision") != source.model_revision
    ):
        raise WorkerAuthorizationError("evaluation probe checkpoint payload provenance is invalid")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise WorkerAuthorizationError("evaluation probe checkpoint contains no model weights")
    try:
        model.load_state_dict(state, strict=True)
    except Exception as error:  # noqa: BLE001 - architecture mismatch fails before inference
        raise WorkerAuthorizationError("evaluation probe checkpoint weights do not match the locked model") from error




async def _recover_committed_result(
    *,
    job: dict[str, Any],
    queue: JobQueue,
    claim: WorkerClaim,
    identity,
    objects: Any,
    checkpoint_store: S3CheckpointStore,
    result_store: S3ResultArtifactStore,
    model,
    contracts,
    config: dict[str, Any],
) -> int | None:
    """Finish a fenced retry from a verified result without importing or invoking a model runner."""
    artifacts = await queue.repository.result_artifacts_for(claim.job_id)
    if not artifacts:
        return None
    if len(artifacts) != 1:
        raise WorkerAuthorizationError("committed result recovery requires exactly one immutable artifact")
    expected_kind = _expected_result_kind(claim)
    details = artifacts[0]
    if details.get("kind") != expected_kind:
        raise WorkerAuthorizationError("committed result kind differs from the immutable job phase")
    try:
        verified = result_store.verify_committed(details, expected_identity=identity)
    except Exception as error:
        raise WorkerAuthorizationError("committed result bytes or identity failed recovery verification") from error

    measurements = details.get("runtime_measurements")
    if not isinstance(measurements, dict):
        raise WorkerAuthorizationError("committed result has no durable runner measurements for recovery")
    if (
        measurements.get("model_id") != model.model_id
        or measurements.get("model_revision") != model.revision
    ):
        raise WorkerAuthorizationError("committed result measurements differ from the immutable model lock")

    checkpoint_sha256 = None
    if claim.phase == "training" or (claim.phase == "probe" and claim.target_phase == "training"):
        checkpoint = await queue.load_checkpoint(store=checkpoint_store, job_id=claim.job_id)
        if checkpoint is None:
            raise WorkerAuthorizationError("committed result has no verified matching checkpoint for recovery")
        checkpoint_sha256 = checkpoint.sha256

    # Re-verify the exact frozen input bytes before using a prior result to complete this fence.
    with tempfile.TemporaryDirectory(prefix="gods-mlops-recovery-") as stage_directory:
        manifest_root = Path(stage_directory) / "manifest"
        manifest_root.mkdir(mode=0o700)
        manifest_uri, extra = await _worker_manifest(job, queue, objects, root=manifest_root)
        config.update(extra)
        manifest = contracts.load_manifest(manifest_uri, config=config)
        contracts.validate_manifest_identity(config, manifest)
    await validate_current_worker_claim(queue, claim, object_store=objects)

    if claim.phase == "probe":
        result_payload = None
        evaluation_probe_source = None
        if claim.target_phase == "evaluation":
            if verified.object_key is None:
                raise WorkerAuthorizationError("evaluation probe result has no immutable S3 object key")
            result_payload = objects.read_source(
                object_key=verified.object_key,
                sha256_digest=verified.sha256,
                size_bytes=verified.size_bytes,
            )
            try:
                evaluation_probe_source = ProbeInput.from_dict(job.get("source_refs")).evaluation_checkpoint_source
            except (TypeError, ValueError) as error:
                raise WorkerAuthorizationError("evaluation probe checkpoint source is invalid during result recovery") from error
        verification = _probe_verification(
            claim,
            {"hash": verified.sha256},
            measurements,
            checkpoint_sha256,
            result_artifact_payload=result_payload,
            evaluation_probe_checkpoint_source=(
                evaluation_probe_source.as_dict() if evaluation_probe_source is not None else None
            ),
        )
        measured = await queue.record_probe_measurement(
            job_id=claim.job_id,
            lease_token=claim.lease_token,
            exit_code=0,
            peak_allocated_mib=_optional_int(measurements.get("peak_vram_allocated_mib")),
            peak_reserved_mib=_optional_int(measurements.get("peak_vram_reserved_mib")),
            optimizer_steps=int(measurements.get("optimizer_steps", 0)),
            checkpoint_resumed=measurements.get("checkpoint_resumed") is True,
            checkpoint_sha256=checkpoint_sha256,
            inference_steps=int(measurements.get("inference_steps", 0)),
            verification_details=verification,
        )
        return 0 if measured.get("result_state") == "succeeded" else 1

    if claim.phase == "training":
        publisher = DatasetPublisher.from_environment()
        try:
            await publisher.register_model_lineage(
                model_id=verified.uri,
                dataset_version=str(claim.dataset_version),
            )
        finally:
            await publisher.close()
    await queue.complete_owned_job(
        job_id=claim.job_id,
        lease_token=claim.lease_token,
        identity=identity,
        details={
            "result_uri": verified.uri,
            "result_sha256": verified.sha256,
            "model_id": model.model_id,
            "model_revision": model.revision,
            "resource_measurements": measurements,
        },
    )
    return 0


def _expected_result_kind(claim: WorkerClaim) -> str:
    if claim.phase == "training" or (claim.phase == "probe" and claim.target_phase == "training"):
        return "model"
    if claim.phase == "evaluation":
        return "evaluation"
    if claim.phase == "probe" and claim.target_phase == "evaluation":
        return "drafts" if claim.model_kind == "detr" else "evaluation_probe"
    if claim.model_kind == "detr" and claim.target_phase == "preparation":
        return "drafts"
    if claim.model_kind == "qwen" and claim.target_phase == "preparation":
        return "caption_drafts"
    raise WorkerAuthorizationError("model phase has no committed result recovery contract")


async def _run_after_owner_binding(
    repository: PostgresJobQueueRepository,
    claim: WorkerClaim,
    runner,
    *,
    revalidate=None,
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
            if revalidate is not None and not await revalidate():
                raise WorkerYieldRequested("worker claim changed while awaiting host PID binding")
            return await asyncio.to_thread(runner)
        await sleep(poll_interval_seconds)
    raise WorkerAuthorizationError("worker timed out waiting for its exact host PID/start/UID binding")


async def _worker_claim_at_safe_boundary(queue, claim: WorkerClaim, *, object_store: Any = None) -> bool:
    """Recheck the mutable Task 6 source before continuing past a safe runner boundary."""
    job = await queue.get(claim.job_id)
    if job.get("state") == "yield_requested":
        return False
    if job.get("state") != "running":
        return False
    if not await queue.repository.lease_is_current(claim.job_id, claim.lease_token):
        return False
    await validate_current_worker_claim(queue, claim, object_store=object_store)
    return True


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
    if phase == "evaluation" and model_kind == "detr":
        from .detector import run

        return run
    if phase == "evaluation" and model_kind == "clip":
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
    *,
    result_artifact_payload: bytes | None = None,
    evaluation_probe_checkpoint_source: dict[str, Any] | None = None,
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
    inference_verified = claim.target_phase in {"preparation", "evaluation"} and int(
        measurements.get("inference_steps", 0)
    ) >= 1
    evaluation_probe_checkpoint_verified: bool | None = None
    if claim.phase == "probe" and claim.target_phase == "evaluation":
        evaluation_probe_checkpoint_verified = _evaluation_probe_result_source_verified(
            result_artifact_payload,
            result_hash,
            evaluation_probe_checkpoint_source,
        )
    if claim.model_kind == "clip" and claim.target_phase == "training":
        losses = measurements.get("losses")
        training_verified = training_verified and isinstance(losses, list) and bool(losses) and all(
            isinstance(loss, (int, float)) and loss > 0 for loss in losses
        )
    return {
        "passed": result_hash_valid
        and (training_verified or inference_verified)
        and evaluation_probe_checkpoint_verified is not False,
        "model_kind": claim.model_kind,
        "target_phase": claim.target_phase,
        "result_sha256": result_hash,
        "learning_signal_verified": weights_updated if claim.target_phase == "training" else None,
        "checkpoint_resume_verified": resumed,
        "evaluation_probe_checkpoint_verified": evaluation_probe_checkpoint_verified,
    }


def _evaluation_probe_result_source_verified(
    payload: bytes | None,
    result_sha256: Any,
    source_value: dict[str, Any] | None,
) -> bool:
    """Require the immutable probe result to name the exact bound training checkpoint."""
    if not isinstance(payload, bytes) or not payload:
        return False
    if not isinstance(result_sha256, str) or sha256(payload).hexdigest() != result_sha256:
        return False
    try:
        source = EvaluationProbeCheckpointSource.from_dict(source_value)
        document = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    recorded = document.get("evaluation_probe_checkpoint") if isinstance(document, dict) else None
    return isinstance(recorded, dict) and recorded == {
        "training_probe_job_id": source.training_probe_job_id,
        "checkpoint_sha256": source.checkpoint_sha256,
        "model_revision": source.model_revision,
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
