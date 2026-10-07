"""Narrow Kubernetes worker construction and trusted host-process resolution."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from gods_mlops.jobs.models import ProcessIdentity, ResourceObservation
from .artifact_deadlines import artifact_deadline_environment
from .claims import WorkerClaim, WorkerAuthorizationError, validate_worker_claim

_IMAGE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]*@sha256:[0-9a-f]{64}$")
_RUNTIME_ID = re.compile(r"^[0-9a-f]{64}$")


def build_gpu_worker_job(
    *,
    job: dict[str, Any],
    lease: dict[str, Any],
    profile: dict[str, Any],
    namespace: str,
    image: str,
    model_cache_claim: str = "gods-mlops-model-cache",
    checkpoint_claim: str = "gods-mlops-artifacts",
    artifact_deadline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic Ubuntu/A6000 worker Job for an already admitted lease."""
    return _build_gpu_worker_job(
        job=job,
        lease=lease,
        profile=profile,
        namespace=namespace,
        image=image,
        model_cache_claim=model_cache_claim,
        checkpoint_claim=checkpoint_claim,
        allowed_job_states=frozenset({"running"}),
        artifact_deadline=artifact_deadline,
    )


def _build_gpu_worker_job(
    *,
    job: dict[str, Any],
    lease: dict[str, Any],
    profile: dict[str, Any],
    namespace: str,
    image: str,
    model_cache_claim: str,
    checkpoint_claim: str,
    allowed_job_states: frozenset[str],
    artifact_deadline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the immutable worker contract for creation or read-only recovery."""
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]{0,61}[a-z0-9])?", namespace):
        raise ValueError("worker namespace is invalid")
    if not _IMAGE_REF.fullmatch(image):
        raise ValueError("GPU worker image must be pinned by an immutable OCI digest")
    digest = image.rsplit("@", 1)[1]
    image_id = digest
    claim = WorkerClaim.from_admitted_job(job, lease, image_id=image_id)
    validate_worker_claim(
        claim,
        job=job,
        lease=lease,
        profile=profile,
        allowed_job_states=allowed_job_states,
    )
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]{0,61}[a-z0-9])?", model_cache_claim):
        raise ValueError("model cache claim name is invalid")
    if not re.fullmatch(r"[a-z0-9](?:[-a-z0-9.]{0,61}[a-z0-9])?", checkpoint_claim):
        raise ValueError("checkpoint claim name is invalid")

    job_id = claim.job_id
    fence = claim.fence
    name = f"gods-mlops-{job_id.replace('-', '')[:12]}-fence-{fence}"
    labels = {
        "app.kubernetes.io/name": "gods-mlops-training-worker",
        "gods.io/job-id": job_id,
        "gods.io/fence": str(fence),
    }
    annotations = {
        "gods.io/input-kind": claim.input_kind,
        "gods.io/input-id": claim.input_id,
        "gods.io/input-sha256": claim.input_sha256,
        "gods.io/model-kind": claim.model_kind,
        "gods.io/phase": claim.phase,
        "gods.io/target-phase": claim.target_phase,
        "gods.io/config-version": claim.config_version,
        "gods.io/config-sha256": claim.config_sha256,
        "gods.io/image-id": claim.image_id,
    }
    env = [
        {"name": "GODS_MLOPS_DATABASE_URL", "valueFrom": {"secretKeyRef": {"name": "gods-mlops-ingestion-credentials", "key": "DATABASE_URL"}}},
        {"name": "GODS_MLOPS_S3_ENDPOINT_URL", "valueFrom": {"configMapKeyRef": {"name": "gods-mlops-ingestion-config", "key": "GODS_MLOPS_S3_ENDPOINT_URL"}}},
        {"name": "GODS_MLOPS_S3_BUCKET", "valueFrom": {"configMapKeyRef": {"name": "gods-mlops-ingestion-config", "key": "GODS_MLOPS_S3_BUCKET"}}},
        {"name": "GODS_MLOPS_S3_REGION", "valueFrom": {"configMapKeyRef": {"name": "gods-mlops-ingestion-config", "key": "GODS_MLOPS_S3_REGION"}}},
        {"name": "GODS_MLOPS_S3_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": "gods-mlops-storage-s3-secret", "key": "admin_access_key_id"}}},
        {"name": "GODS_MLOPS_S3_SECRET_KEY", "valueFrom": {"secretKeyRef": {"name": "gods-mlops-storage-s3-secret", "key": "admin_secret_access_key"}}},
        {"name": "GODS_MLOPS_JOB_ID", "value": claim.job_id},
        {"name": "GODS_MLOPS_LEASE_TOKEN", "value": claim.lease_token},
        {"name": "GODS_MLOPS_FENCE", "value": str(claim.fence)},
        {"name": "GODS_MLOPS_GPU_UUID", "value": claim.gpu_uuid},
        {"name": "GODS_MLOPS_PHASE", "value": claim.phase},
        {"name": "GODS_MLOPS_TARGET_PHASE", "value": claim.target_phase},
        {"name": "GODS_MLOPS_INPUT_KIND", "value": claim.input_kind},
        {"name": "GODS_MLOPS_INPUT_ID", "value": claim.input_id},
        {"name": "GODS_MLOPS_INPUT_SHA256", "value": claim.input_sha256},
        {"name": "GODS_MLOPS_DATASET_VERSION", "value": claim.dataset_version or ""},
        {"name": "GODS_MLOPS_MODEL_KIND", "value": claim.model_kind},
        {"name": "GODS_MLOPS_CONFIG_VERSION", "value": claim.config_version},
        {"name": "GODS_MLOPS_CONFIG_SHA256", "value": claim.config_sha256},
        {"name": "GODS_MLOPS_IMAGE_ID", "value": claim.image_id},
        {"name": "GODS_MLOPS_NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "GODS_MLOPS_MODEL_LOCK", "value": "/app/models/lock.json"},
        {"name": "GODS_MLOPS_MODEL_CACHE_ROOT", "value": "/mnt/model-cache"},
        {"name": "GODS_MLOPS_CHECKPOINT_ROOT", "value": "/mnt/gods-objects/checkpoints"},
    ]
    if artifact_deadline is not None:
        env.extend(
            {"name": name, "value": value}
            for name, value in artifact_deadline_environment(artifact_deadline).items()
        )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "backoffLimit": 0,
            "parallelism": 1,
            "completions": 1,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "nodeSelector": {"kubernetes.io/hostname": "ubuntu"},
                    "securityContext": {"runAsNonRoot": True, "runAsUser": 10001, "runAsGroup": 10001},
                    "containers": [
                        {
                            "name": "trainer",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["gods-mlops-training"],
                            "args": ["worker"],
                            "env": env,
                            "resources": {
                                "requests": {"cpu": "2", "memory": "8Gi", "nvidia.com/gpu": "1"},
                                "limits": {"cpu": "8", "memory": "24Gi", "nvidia.com/gpu": "1"},
                            },
                            "volumeMounts": [
                                {"name": "model-cache", "mountPath": "/mnt/model-cache", "readOnly": True},
                                {"name": "gods-objects", "mountPath": "/mnt/gods-objects"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "model-cache", "persistentVolumeClaim": {"claimName": model_cache_claim, "readOnly": True}},
                        {"name": "gods-objects", "persistentVolumeClaim": {"claimName": checkpoint_claim}},
                    ],
                },
            },
        },
    }


class KubernetesOwnedWorkerAdapter:
    """Create a deterministic Job only for the exact current Task 7 lease."""

    def __init__(self, *, batch_api: Any, namespace: str, image: str) -> None:
        self._batch_api = batch_api
        self._namespace = namespace
        self._image = image

    def ensure_worker(
        self,
        *,
        job: dict[str, Any],
        lease: dict[str, Any],
        profile: dict[str, Any],
        artifact_deadline: dict[str, Any] | None = None,
    ) -> Any:
        body = build_gpu_worker_job(
            job=job,
            lease=lease,
            profile=profile,
            namespace=self._namespace,
            image=self._image,
            artifact_deadline=artifact_deadline,
        )
        name = body["metadata"]["name"]
        try:
            return self._batch_api.create_namespaced_job(namespace=self._namespace, body=body)
        except Exception as error:  # the only idempotency conflict accepted is the same owned fence
            if getattr(error, "status", None) != 409:
                raise
            existing = self._batch_api.read_namespaced_job(name=name, namespace=self._namespace)
            self._validate_existing_worker(existing, body, cause=error)
            return existing

    def read_existing_worker(
        self,
        *,
        job: dict[str, Any],
        lease: dict[str, Any],
        profile: dict[str, Any],
        artifact_deadline: dict[str, Any] | None = None,
    ) -> Any | None:
        """Read the exact owned worker for a retained non-running lease, never creating one."""
        allowed_states = frozenset({"yield_requested", "completed", "failed", "cancelled"})
        if job.get("state") not in allowed_states:
            raise WorkerAuthorizationError("only a retained yielding or terminal job can reuse a worker")
        body = _build_gpu_worker_job(
            job=job,
            lease=lease,
            profile=profile,
            namespace=self._namespace,
            image=self._image,
            model_cache_claim="gods-mlops-model-cache",
            checkpoint_claim="gods-mlops-artifacts",
            allowed_job_states=allowed_states,
            artifact_deadline=artifact_deadline,
        )
        name = body["metadata"]["name"]
        try:
            existing = self._batch_api.read_namespaced_job(name=name, namespace=self._namespace)
        except Exception as error:  # a missing old worker is not permission to launch a replacement
            if getattr(error, "status", None) == 404:
                return None
            raise
        self._validate_existing_worker(existing, body)
        return existing

    def _validate_existing_worker(self, existing: Any, body: dict[str, Any], *, cause=None) -> None:
        existing_dict = _to_plain_dict(existing)
        expected_meta = body["metadata"]
        actual_meta = existing_dict.get("metadata", {})
        if (
            actual_meta.get("name") != expected_meta["name"]
            or actual_meta.get("namespace") != self._namespace
            or actual_meta.get("labels") != expected_meta["labels"]
            or actual_meta.get("annotations") != expected_meta["annotations"]
            or not _same_worker_pod_contract(existing_dict, body)
        ):
            error = WorkerAuthorizationError("existing Kubernetes Job is not this job/fence/input/config")
            if cause is not None:
                raise error from cause
            raise error


def resolve_owned_gpu_process(
    *,
    job_uid: str,
    pod: Any,
    observation: ResourceObservation | dict[str, Any],
    now: datetime | None = None,
    max_age_seconds: int = 10,
) -> ProcessIdentity:
    """Match a current Pod's owner UID and runtime ID to one fresh host /proc cgroup."""
    try:
        expected_job_uid = str(UUID(job_uid))
    except ValueError as error:
        raise ValueError("owned Job UID is invalid") from error
    observation = observation if isinstance(observation, ResourceObservation) else ResourceObservation.from_dict(observation)
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    age = (current_time - observation.observed_at).total_seconds()
    if age < 0 or age > max_age_seconds:
        raise ValueError("host process observation is stale")
    if observation.node_id != "ubuntu" or observation.hostname.lower() != "ubuntu":
        raise ValueError("owned worker process was not observed on Ubuntu")
    if not observation.process_table_complete or not observation.gpu_process_list_complete:
        raise ValueError("host process observation is incomplete")

    metadata = _get(pod, "metadata")
    pod_uid = _get(metadata, "uid")
    try:
        pod_uid = str(UUID(str(pod_uid)))
    except (ValueError, TypeError) as error:
        raise ValueError("owned worker Pod UID is unavailable") from error
    owner_refs = _get(metadata, "ownerReferences", "owner_references") or []
    matching_owners = [
        owner
        for owner in owner_refs
        if _get(owner, "kind") == "Job"
        and str(_get(owner, "uid")) == expected_job_uid
        and _get(owner, "controller") is True
    ]
    if len(matching_owners) != 1:
        raise ValueError("Pod owner reference does not identify the admitted Kubernetes Job")

    status = _get(pod, "status")
    statuses = _get(status, "containerStatuses", "container_statuses") or []
    worker_statuses = [
        item for item in statuses if _get(item, "name") == "trainer" and _get(item, "containerID", "container_id")
    ]
    if len(worker_statuses) != 1:
        raise ValueError("owned worker container runtime identity is unavailable")
    container_id = str(_get(worker_statuses[0], "containerID", "container_id"))
    if "://" not in container_id:
        raise ValueError("owned worker container runtime identity is malformed")
    runtime, container_id = container_id.split("://", 1)
    if runtime not in {"containerd", "docker", "cri-o"} or not _RUNTIME_ID.fullmatch(container_id):
        raise ValueError("owned worker container runtime identity is unsupported")

    expected_pod_marker = "pod" + pod_uid.replace("-", "")
    matching: list[ProcessIdentity] = []
    for process in observation.process_table:
        path_segments = [re.sub(r"[^a-z0-9]", "", segment.lower()) for path in process.cgroup_paths for segment in path.split("/")]
        has_pod = any(expected_pod_marker in segment for segment in path_segments)
        runtime_segments = [segment.removesuffix("scope") for segment in path_segments]
        runtime_markers = {
            container_id,
            "cricontainerd" + container_id,
            "containerd" + container_id,
            "docker" + container_id,
            "crio" + container_id,
        }
        has_container = any(segment in runtime_markers for segment in runtime_segments)
        if has_pod and has_container:
            matching.append(process)
    if len(matching) != 1:
        raise ValueError("owned GPU process identity is unavailable or ambiguous")
    owner = matching[0]
    if owner.uid != 10001:
        raise ValueError("owned worker host process UID does not match the training image")
    return owner


def resolve_owned_docker_process(
    *,
    container_id: str,
    observation: ResourceObservation | dict[str, Any],
    now: datetime | None = None,
    max_age_seconds: int = 10,
) -> ProcessIdentity:
    """Bind a probe-only Docker container ID to its fresh host /proc cgroup identity."""
    if not _RUNTIME_ID.fullmatch(container_id):
        raise ValueError("owned Docker probe container ID is invalid")
    observation = observation if isinstance(observation, ResourceObservation) else ResourceObservation.from_dict(observation)
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    age = (current_time - observation.observed_at).total_seconds()
    if age < 0 or age > max_age_seconds:
        raise ValueError("host process observation is stale")
    if observation.node_id != "ubuntu" or observation.hostname.lower() != "ubuntu":
        raise ValueError("owned Docker probe process was not observed on Ubuntu")
    if not observation.process_table_complete or not observation.gpu_process_list_complete:
        raise ValueError("host process observation is incomplete")
    expected = container_id.lower()
    matches = []
    for process in observation.process_table:
        cgroup_segments = [
            re.sub(r"[^a-z0-9]", "", segment.lower()).removesuffix("scope")
            for path in process.cgroup_paths
            for segment in path.split("/")
        ]
        if any(segment.endswith(expected) for segment in cgroup_segments):
            matches.append(process)
    if len(matches) != 1:
        raise ValueError("owned Docker probe host PID is unavailable or ambiguous")
    owner = matches[0]
    if owner.uid != 10001:
        raise ValueError("owned Docker probe host process UID does not match the training image")
    return owner


def _get(value: Any, camel: str, snake: str | None = None) -> Any:
    if isinstance(value, dict):
        return value.get(camel, value.get(snake) if snake else None)
    attr = snake or _camel_to_snake(camel)
    return getattr(value, attr, None)


def _camel_to_snake(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError("Kubernetes API returned an unsupported Job object")


def _same_worker_pod_contract(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    actual_pod = _normal_keys(actual).get("spec", {}).get("template", {}).get("spec", {})
    expected_pod = _normal_keys(expected).get("spec", {}).get("template", {}).get("spec", {})
    actual_containers = actual_pod.get("containers") or []
    expected_containers = expected_pod.get("containers") or []
    if len(actual_containers) != 1 or len(expected_containers) != 1:
        return False
    fields = ("restart_policy", "automount_service_account_token", "node_selector")
    if any(actual_pod.get(field) != expected_pod.get(field) for field in fields):
        return False
    expected_security = expected_pod.get("security_context") or {}
    actual_security = actual_pod.get("security_context") or {}
    if any(
        actual_security.get(field) != expected_security.get(field)
        for field in ("run_as_non_root", "run_as_user", "run_as_group")
    ):
        return False
    if not _same_pod_volumes(actual_pod.get("volumes") or [], expected_pod.get("volumes") or []):
        return False
    container_fields = (
        "name",
        "image",
        "image_pull_policy",
        "command",
        "args",
        "resources",
    )
    actual_container, expected_container = actual_containers[0], expected_containers[0]
    return (
        all(actual_container.get(field) == expected_container.get(field) for field in container_fields)
        and _without_none(actual_container.get("env") or [])
        == _without_none(expected_container.get("env") or [])
        and _same_volume_mounts(
            actual_container.get("volume_mounts") or [],
            expected_container.get("volume_mounts") or [],
        )
    )


def _normal_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {_camel_to_snake(str(key)): _normal_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normal_keys(item) for item in value]
    return value


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def _same_pod_volumes(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    if {item.get("name") for item in actual} != {item.get("name") for item in expected}:
        return False
    actual_by_name = {item.get("name"): item for item in actual}
    for item in expected:
        current = actual_by_name[item.get("name")]
        expected_pvc = item.get("persistent_volume_claim")
        actual_pvc = current.get("persistent_volume_claim")
        if (expected_pvc is None) != (actual_pvc is None):
            return False
        if expected_pvc is None:
            continue
        if (
            not isinstance(actual_pvc, dict)
            or actual_pvc.get("claim_name") != expected_pvc.get("claim_name")
            or bool(actual_pvc.get("read_only", False)) != bool(expected_pvc.get("read_only", False))
        ):
            return False
    return True


def _same_volume_mounts(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    def key(item: dict[str, Any]):
        return item.get("name"), item.get("mount_path")

    if {key(item) for item in actual} != {key(item) for item in expected}:
        return False
    actual_by_key = {key(item): item for item in actual}
    return all(
        bool(actual_by_key[key(item)].get("read_only", False)) == bool(item.get("read_only", False))
        for item in expected
    )
