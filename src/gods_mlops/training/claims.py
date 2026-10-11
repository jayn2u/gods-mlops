"""Fenced execution claims shared by CPU controllers and GPU workers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .contracts import (
    CLIP_FIX5_EXECUTION_CONFIG_VERSION,
    CLIP_FIX5_INPUT_ID,
    CLIP_FIX5_INPUT_SHA256,
    CLIP_FIX5_MANIFEST_CONFIG_VERSION,
    CLIP_FIX5_MANIFEST_OBJECT_KEY,
    CLIP_FIX5_MANIFEST_SIZE_BYTES,
    ProbeManifestVersionPin,
    locked_model,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WorkerAuthorizationError(ValueError):
    """The durable job, profile, or active lease does not authorize this worker."""


@dataclass(frozen=True, slots=True)
class WorkerClaim:
    job_id: str
    lease_token: str
    fence: int
    gpu_uuid: str
    phase: str
    target_phase: str
    input_kind: str
    input_id: str
    input_sha256: str
    dataset_version: str | None
    model_kind: str
    config_version: str
    config_sha256: str
    image_id: str

    @classmethod
    def from_admitted_job(
        cls,
        job: dict[str, Any],
        lease: dict[str, Any],
        *,
        image_id: str,
    ) -> "WorkerClaim":
        """Copy immutable job identity and the currently granted fencing generation."""
        return cls(
            job_id=str(job["job_id"]),
            lease_token=str(lease["lease_token"]),
            fence=int(lease["fencing_token"]),
            gpu_uuid=str(lease["gpu_uuid"]),
            phase=str(job["phase"]),
            target_phase=str(job.get("target_phase") or job["phase"]),
            input_kind=str(job["input_kind"]),
            input_id=str(job["input_id"]),
            input_sha256=str(job["input_sha256"]).strip(),
            dataset_version=(str(job["dataset_version"]) if job.get("dataset_version") is not None else None),
            model_kind=str(job["model_kind"]),
            config_version=str(job["config_version"]),
            config_sha256=str(job["config_sha256"]).strip(),
            image_id=image_id,
        )

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "job_id": self.job_id,
            "lease_token": self.lease_token,
            "fence": self.fence,
            "gpu_uuid": self.gpu_uuid,
            "phase": self.phase,
            "target_phase": self.target_phase,
            "input_kind": self.input_kind,
            "input_id": self.input_id,
            "input_sha256": self.input_sha256,
            "dataset_version": self.dataset_version,
            "model_kind": self.model_kind,
            "config_version": self.config_version,
            "config_sha256": self.config_sha256,
            "image_id": self.image_id,
        }


def validate_worker_claim(
    claim: WorkerClaim,
    *,
    job: dict[str, Any],
    lease: dict[str, Any],
    profile: dict[str, Any] | None,
    allowed_job_states: frozenset[str] = frozenset({"running"}),
) -> ProbeManifestVersionPin | None:
    """Fail closed unless the latest durable job, profile, and fence exactly match."""
    if profile is None:
        raise WorkerAuthorizationError("worker resource profile is unavailable")
    if job.get("state") not in allowed_job_states:
        if allowed_job_states == frozenset({"running"}):
            raise WorkerAuthorizationError("worker job is not currently running")
        raise WorkerAuthorizationError("worker job is not in an authorized ownership state")
    if lease.get("job_id") != claim.job_id or job.get("job_id") != claim.job_id:
        raise WorkerAuthorizationError("worker job identity does not match its active lease")
    if str(job.get("lease_token")) != claim.lease_token or str(lease.get("lease_token")) != claim.lease_token:
        raise WorkerAuthorizationError("worker lease token is stale")
    if int(job.get("lease_generation", -1)) != claim.fence or int(lease.get("fencing_token", -1)) != claim.fence:
        raise WorkerAuthorizationError("worker fence is stale")
    if str(lease.get("gpu_uuid")) != claim.gpu_uuid:
        raise WorkerAuthorizationError("worker lease is bound to another GPU")
    for name in (
        "phase",
        "input_kind",
        "input_id",
        "model_kind",
        "config_version",
    ):
        if str(job.get(name)) != getattr(claim, name):
            raise WorkerAuthorizationError(f"worker {name} does not match the immutable job")
    target_phase = str(job.get("target_phase") or job.get("phase"))
    if target_phase != claim.target_phase:
        raise WorkerAuthorizationError("worker target phase does not match the immutable job")
    input_hash = str(job.get("input_sha256", "")).strip()
    config_hash = str(job.get("config_sha256", "")).strip()
    if input_hash != claim.input_sha256 or not _SHA256.fullmatch(input_hash):
        raise WorkerAuthorizationError("worker input hash does not match the immutable job")
    if config_hash != claim.config_sha256 or not _SHA256.fullmatch(config_hash):
        raise WorkerAuthorizationError("worker config hash does not match the immutable job")
    dataset_version = job.get("dataset_version")
    if (str(dataset_version) if dataset_version is not None else None) != claim.dataset_version:
        raise WorkerAuthorizationError("worker dataset version does not match the immutable job")
    if profile.get("phase") != claim.phase or profile.get("model_kind") != claim.model_kind:
        raise WorkerAuthorizationError("worker profile belongs to another phase or model")
    if profile.get("config_version") != claim.config_version:
        raise WorkerAuthorizationError("worker config version does not match its profile")
    if str(profile.get("config_sha256", "")).strip() != claim.config_sha256:
        raise WorkerAuthorizationError("worker config hash does not match its profile")
    profile_target = str(profile.get("target_phase") or profile.get("phase"))
    if profile_target != claim.target_phase:
        raise WorkerAuthorizationError("worker profile target phase does not match the job")
    profile_state = profile.get("profile_state")
    probe_manifest_pin = None
    if claim.phase == "probe":
        if profile_state != "candidate":
            raise WorkerAuthorizationError("probe worker requires a candidate measurement profile")
        if claim.input_kind != "probe_input" or claim.dataset_version is not None:
            raise WorkerAuthorizationError("probe worker requires a separate immutable probe input")
        source_refs = job.get("source_refs")
        if isinstance(source_refs, dict) and source_refs.get("schema") == "gods-mlops-probe-input-v1":
            from gods_mlops.jobs.models import ProbeInput

            try:
                probe_input = ProbeInput.from_dict(source_refs)
            except (TypeError, ValueError) as error:
                raise WorkerAuthorizationError("typed probe checkpoint/source reference is invalid") from error
            if (
                probe_input.probe_input_id != claim.input_id
                or probe_input.input_sha256 != claim.input_sha256
                or probe_input.model_kind != claim.model_kind
                or probe_input.target_phase != claim.target_phase
                or probe_input.config_version != claim.config_version
            ):
                raise WorkerAuthorizationError("typed probe source identity differs from the admitted job")
            probe_manifest_pin = clip_probe_manifest_version_pin(
                claim,
                job=job,
                profile=profile,
                probe_input=probe_input,
            )
            if claim.target_phase == "evaluation":
                checkpoint_source = probe_input.evaluation_checkpoint_source
                profile_config = profile.get("config_json", {})
                if isinstance(profile_config, str):
                    try:
                        profile_config = json.loads(profile_config)
                    except json.JSONDecodeError as error:
                        raise WorkerAuthorizationError("evaluation probe profile config is invalid") from error
                if (
                    checkpoint_source is None
                    or checkpoint_source.model_kind != claim.model_kind
                    or not isinstance(profile_config, dict)
                    or checkpoint_source.model_id != profile_config.get("model_id")
                    or checkpoint_source.model_revision != profile_config.get("model_revision")
                ):
                    raise WorkerAuthorizationError(
                        "evaluation probe has no matching typed training-probe checkpoint source"
                    )
    else:
        if profile_state != "measured":
            raise WorkerAuthorizationError("training worker profile is not measured and allowlisted")
    if claim.phase == "training" and claim.input_kind != "dataset_version":
        raise WorkerAuthorizationError("training worker requires a published dataset version")
    if claim.phase == "evaluation":
        from gods_mlops.jobs.models import EvaluationCheckpointSource

        if claim.input_kind != "dataset_version" or claim.dataset_version != claim.input_id:
            raise WorkerAuthorizationError("evaluation worker requires its published dataset version")
        source_refs = job.get("source_refs")
        if not isinstance(source_refs, dict) or source_refs.get("schema") != "gods-mlops-evaluation-source-v1":
            raise WorkerAuthorizationError("evaluation worker has no typed immutable source reference")
        if (
            source_refs.get("dataset_version") != claim.dataset_version
            or source_refs.get("manifest_sha256") != claim.input_sha256
            or source_refs.get("target") not in {claim.model_kind, "both"}
            or source_refs.get("evaluation_split") not in {"validation", "test"}
        ):
            raise WorkerAuthorizationError("evaluation source dataset identity differs from the immutable job")
        baseline = source_refs.get("baseline")
        if baseline is not None and (
            not isinstance(baseline, dict)
            or set(baseline) != {"model_id", "revision", "verified"}
            or baseline.get("verified") is not False
            or not isinstance(baseline.get("model_id"), str)
            or not baseline["model_id"].strip()
        ):
            raise WorkerAuthorizationError("evaluation baseline is metadata only and cannot claim verification")
        try:
            checkpoint_source = EvaluationCheckpointSource.from_dict(source_refs.get("checkpoint"))
        except (TypeError, ValueError) as error:
            raise WorkerAuthorizationError("evaluation checkpoint source is invalid") from error
        if (
            checkpoint_source.dataset_version != claim.dataset_version
            or checkpoint_source.training_manifest_sha256 != claim.input_sha256
            or checkpoint_source.model_kind != claim.model_kind
        ):
            raise WorkerAuthorizationError("evaluation checkpoint source differs from the immutable dataset job")
    if claim.phase == "preparation" and claim.input_kind != "annotation_batch":
        raise WorkerAuthorizationError("preparation worker requires an immutable annotation batch")
    if claim.image_id != "sha256:" + claim.image_id.removeprefix("sha256:") or not _SHA256.fullmatch(
        claim.image_id.removeprefix("sha256:")
    ):
        raise WorkerAuthorizationError("worker image identity must be an immutable SHA-256 ID")
    return probe_manifest_pin


def clip_probe_manifest_version_pin(
    claim: WorkerClaim,
    *,
    job: dict[str, Any],
    profile: dict[str, Any],
    probe_input: Any,
) -> ProbeManifestVersionPin | None:
    """Resolve the one admitted v2 CLIP fixture pin; every other path stays strict."""
    if (
        claim.phase != "probe"
        or claim.target_phase != "training"
        or claim.model_kind != "clip"
        or claim.config_version != CLIP_FIX5_EXECUTION_CONFIG_VERSION
    ):
        return None
    if (
        profile.get("phase") != "probe"
        or profile.get("model_kind") != "clip"
        or profile.get("config_version") != CLIP_FIX5_EXECUTION_CONFIG_VERSION
        or profile.get("target_phase") != "training"
        or profile.get("profile_state") != "candidate"
        or job.get("phase") != "probe"
        or job.get("target_phase") != "training"
        or job.get("input_kind") != "probe_input"
        or job.get("model_kind") != "clip"
        or job.get("config_version") != CLIP_FIX5_EXECUTION_CONFIG_VERSION
        or claim.input_kind != "probe_input"
    ):
        raise WorkerAuthorizationError("CLIP manifest pin requires its registered training-probe candidate")
    config = profile.get("config_json")
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError as error:
            raise WorkerAuthorizationError("CLIP manifest pin profile config is invalid") from error
    if not isinstance(config, dict):
        raise WorkerAuthorizationError("CLIP manifest pin profile config is unavailable")
    try:
        serialized_config = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise WorkerAuthorizationError("CLIP manifest pin profile config is not canonical JSON") from error
    config_hash = sha256(serialized_config).hexdigest()
    registered_hash = str(profile.get("config_sha256", "")).strip()
    if (
        config_hash != registered_hash
        or config_hash != claim.config_sha256
        or config_hash != str(job.get("config_sha256", "")).strip()
    ):
        raise WorkerAuthorizationError("CLIP manifest pin profile config hash differs from the admitted job")
    if (
        config.get("probe_manifest_config_version") != CLIP_FIX5_MANIFEST_CONFIG_VERSION
        or config.get("forward_precision") != "float16-autocast"
        or config.get("training_loss_policy_version") != "task9-clip-symmetric-ce-fp64-v1"
        or config.get("training_loss_objective") != "symmetric_identity_cross_entropy"
        or config.get("training_loss_reduction_precision") != "float64"
    ):
        raise WorkerAuthorizationError("CLIP manifest pin profile does not select the approved loss policy")
    model = locked_model("clip")
    if config.get("model_id") != model.model_id or config.get("model_revision") != model.revision:
        raise WorkerAuthorizationError("CLIP manifest pin model differs from its immutable model lock")
    if (
        claim.input_id != CLIP_FIX5_INPUT_ID
        or claim.input_sha256 != CLIP_FIX5_INPUT_SHA256
        or str(job.get("input_id", "")) != CLIP_FIX5_INPUT_ID
        or str(job.get("input_sha256", "")).strip() != CLIP_FIX5_INPUT_SHA256
        or probe_input.probe_input_id != CLIP_FIX5_INPUT_ID
        or probe_input.input_sha256 != CLIP_FIX5_INPUT_SHA256
        or probe_input.config_version != CLIP_FIX5_EXECUTION_CONFIG_VERSION
        or probe_input.model_kind != "clip"
        or probe_input.target_phase != "training"
        or probe_input.manifest_object_key != CLIP_FIX5_MANIFEST_OBJECT_KEY
        or probe_input.object_size_bytes != CLIP_FIX5_MANIFEST_SIZE_BYTES
        or probe_input.fixture is not True
    ):
        raise WorkerAuthorizationError("CLIP manifest pin is restricted to the exact frozen probe input")
    return ProbeManifestVersionPin(
        manifest_config_version=CLIP_FIX5_MANIFEST_CONFIG_VERSION,
        execution_config_version=CLIP_FIX5_EXECUTION_CONFIG_VERSION,
        execution_config_sha256=config_hash,
        input_id=CLIP_FIX5_INPUT_ID,
        input_sha256=CLIP_FIX5_INPUT_SHA256,
        model_id=model.model_id,
        model_revision=model.revision,
    )


class WorkerYieldRequested(RuntimeError):
    """The current Task 7 lease asked this model runner to stop at a safe boundary."""


async def validate_current_worker_claim(
    queue: Any,
    claim: WorkerClaim,
    *,
    object_store: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-read the durable source, job, profile, and lease immediately before CUDA."""
    job = await queue.get(claim.job_id)
    lease = await queue.repository.get_active_lease(claim.gpu_uuid)
    if lease is None:
        raise WorkerAuthorizationError("worker no longer has an active GPU lease")
    profile = await queue.repository.get_profile(
        phase=claim.phase,
        model_kind=claim.model_kind,
        config_version=claim.config_version,
    )
    probe_manifest_pin = validate_worker_claim(claim, job=job, lease=lease, profile=profile)
    if not await queue.repository.lease_is_current(claim.job_id, claim.lease_token):
        raise WorkerAuthorizationError("worker fence is no longer current")
    if claim.phase == "training":
        reasons = await queue.training_source_block_reasons(claim.job_id)
        if reasons:
            raise WorkerAuthorizationError(
                "training source readiness changed: " + ", ".join(reasons)
            )
    elif claim.phase == "evaluation":
        reasons = await queue.evaluation_source_block_reasons(claim.job_id)
        if reasons:
            raise WorkerAuthorizationError(
                "evaluation source readiness changed: " + ", ".join(reasons)
            )
    elif claim.phase == "preparation":
        batch = _annotation_batch_from_job(job)
        try:
            await queue.source_registry.verify_annotation_batch(batch)
        except Exception as error:
            raise WorkerAuthorizationError("immutable preparation source changed") from error
    elif claim.phase == "probe":
        source_refs = job.get("source_refs")
        if isinstance(source_refs, dict) and source_refs.get("schema") == "gods-mlops-probe-input-v1":
            from gods_mlops.jobs.models import ProbeInput

            if object_store is None:
                raise WorkerAuthorizationError("typed probe manifest object store is unavailable")
            try:
                probe_input = ProbeInput.from_dict(source_refs)
                probe_input.verify(
                    object_store,
                    expected_manifest_config_version=(
                        probe_manifest_pin.manifest_config_version
                        if probe_manifest_pin is not None
                        else None
                    ),
                )
            except Exception as error:
                raise WorkerAuthorizationError("immutable probe manifest bytes changed") from error
            if claim.target_phase == "evaluation":
                reasons = await queue.evaluation_probe_source_block_reasons(claim.job_id)
                if reasons:
                    raise WorkerAuthorizationError(
                        "evaluation probe checkpoint source changed: " + ", ".join(reasons)
                    )
    return job, lease


def _annotation_batch_from_job(job: dict[str, Any]):
    from gods_mlops.jobs.models import AnnotationPreparationBatch, ImmutableAnnotationItem

    refs = job.get("source_refs")
    if not isinstance(refs, dict) or not isinstance(refs.get("items"), list):
        raise WorkerAuthorizationError("preparation source references are unavailable")
    try:
        items = tuple(
            ImmutableAnnotationItem(
                item_kind=str(item["item_kind"]),
                item_id=str(item["item_id"]),
                sample_id=str(item["sample_id"]),
                sha256=str(item["sha256"]),
                object_key=str(item["object_key"]),
                object_size_bytes=int(item["object_size_bytes"]),
                revision_id=str(item["revision_id"]) if item.get("revision_id") is not None else None,
            )
            for item in refs["items"]
        )
        return AnnotationPreparationBatch(
            batch_id=str(refs["batch_id"]),
            input_sha256=str(refs["input_sha256"]),
            items=items,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise WorkerAuthorizationError("preparation source references are malformed") from error
