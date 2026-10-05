"""Fenced execution claims shared by CPU controllers and GPU workers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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
) -> None:
    """Fail closed unless the latest durable job, profile, and fence exactly match."""
    if profile is None:
        raise WorkerAuthorizationError("worker resource profile is unavailable")
    if job.get("state") != "running":
        raise WorkerAuthorizationError("worker job is not currently running")
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
    if claim.phase == "probe":
        if profile_state != "candidate":
            raise WorkerAuthorizationError("probe worker requires a candidate measurement profile")
        if claim.input_kind != "probe_input" or claim.dataset_version is not None:
            raise WorkerAuthorizationError("probe worker requires a separate immutable probe input")
        source_refs = job.get("source_refs")
        if isinstance(source_refs, dict) and source_refs.get("schema") == "gods-mlops-probe-input-v1":
            from gods_mlops.jobs.models import ProbeInput

            probe_input = ProbeInput.from_dict(source_refs)
            if (
                probe_input.probe_input_id != claim.input_id
                or probe_input.input_sha256 != claim.input_sha256
                or probe_input.model_kind != claim.model_kind
                or probe_input.target_phase != claim.target_phase
                or probe_input.config_version != claim.config_version
            ):
                raise WorkerAuthorizationError("typed probe source identity differs from the admitted job")
    else:
        if profile_state != "measured":
            raise WorkerAuthorizationError("training worker profile is not measured and allowlisted")
    if claim.phase == "training" and claim.input_kind != "dataset_version":
        raise WorkerAuthorizationError("training worker requires a published dataset version")
    if claim.phase == "preparation" and claim.input_kind != "annotation_batch":
        raise WorkerAuthorizationError("preparation worker requires an immutable annotation batch")
    if claim.image_id != "sha256:" + claim.image_id.removeprefix("sha256:") or not _SHA256.fullmatch(
        claim.image_id.removeprefix("sha256:")
    ):
        raise WorkerAuthorizationError("worker image identity must be an immutable SHA-256 ID")


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
    validate_worker_claim(claim, job=job, lease=lease, profile=profile)
    if not await queue.repository.lease_is_current(claim.job_id, claim.lease_token):
        raise WorkerAuthorizationError("worker fence is no longer current")
    if claim.phase == "training":
        reasons = await queue.training_source_block_reasons(claim.job_id)
        if reasons:
            raise WorkerAuthorizationError(
                "training source readiness changed: " + ", ".join(reasons)
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
                ProbeInput.from_dict(source_refs).verify(object_store)
            except Exception as error:
                raise WorkerAuthorizationError("immutable probe manifest bytes changed") from error
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
