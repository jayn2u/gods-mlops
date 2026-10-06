"""CPU-only controller for Task 7 admission and owned Kubernetes GPU workers."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import AnnotationSourceSelection, ResourceObservation
from gods_mlops.jobs.monitor import GpuJobMonitor
from gods_mlops.jobs.observer import UbuntuResourceObserver
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

from .worker_adapter import KubernetesOwnedWorkerAdapter, resolve_owned_gpu_process


class TrainingController:
    """Wait on the durable queue and create only the currently admitted Job."""

    def __init__(
        self,
        *,
        queue: JobQueue,
        repository: PostgresJobQueueRepository,
        admission: GpuAdmission,
        monitor: GpuJobMonitor,
        observer: UbuntuResourceObserver,
        worker_adapter: KubernetesOwnedWorkerAdapter,
        core_api: Any,
        namespace: str,
        interval_seconds: int = 5,
        sleep=asyncio.sleep,
        clock=time.monotonic,
    ) -> None:
        self._queue = queue
        self._repository = repository
        self._admission = admission
        self._monitor = monitor
        self._observer = observer
        self._worker_adapter = worker_adapter
        self._core_api = core_api
        self._namespace = namespace
        self._interval_seconds = interval_seconds
        self._sleep = sleep
        self._clock = clock
        if interval_seconds != 5:
            raise ValueError("Task 7 admission and monitor cadence is fixed at five seconds")

    async def run(self, job_id: str, *, timeout_seconds: int = 86_400) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("controller timeout must be positive")
        deadline = self._clock() + timeout_seconds
        while self._clock() < deadline:
            job = await self._queue.get(job_id)
            if job.get("state") in {"completed", "failed", "cancelled"}:
                lease = await self._repository.get_active_lease(self._admission.expected_gpu_uuid)
                if lease is None or lease.get("job_id") != job_id:
                    await self._cleanup_terminal_artifact_writes(job_id)
                    return job

            if not job.get("lease_token"):
                observation = await self._observer_call()
                job = await self._admission.admit(job_id, observation.to_dict())

            lease = await self._repository.get_active_lease(self._admission.expected_gpu_uuid)
            if lease is None or lease.get("job_id") != job_id:
                await self._sleep(self._interval_seconds)
                continue

            profile = await self._repository.get_profile(
                phase=job["phase"],
                model_kind=job["model_kind"],
                config_version=job["config_version"],
            )
            if profile is None:
                await self._sleep(self._interval_seconds)
                continue
            worker_job = None
            if job.get("state") == "running":
                worker_job = self._worker_adapter.ensure_worker(job=job, lease=lease, profile=profile)
            elif job.get("state") in {"yield_requested", "completed", "failed", "cancelled"}:
                # A retained lease can outlive the job's runnable state. Read its
                # exact owned Job for observation, but never turn a yield/terminal
                # state into a new GPU launch.
                worker_job = self._worker_adapter.read_existing_worker(
                    job=job, lease=lease, profile=profile
                )
            worker_uid = _get(_get(worker_job, "metadata"), "uid") if worker_job is not None else None
            if worker_job is not None and not worker_uid:
                raise RuntimeError("created Kubernetes worker Job has no authoritative UID")

            observation = await self._observer_call()
            if lease.get("owner_pid") is None and worker_uid:
                pods = self._core_api.list_namespaced_pod(
                    namespace=self._namespace,
                    label_selector=(
                        f"gods.io/job-id={job_id},gods.io/fence={lease['fencing_token']}"
                    ),
                )
                for pod in _get(pods, "items") or []:
                    try:
                        owner = resolve_owned_gpu_process(
                            job_uid=str(worker_uid),
                            pod=pod,
                            observation=observation,
                        )
                    except ValueError:
                        continue
                    if not await self._queue.bind_process(job_id, lease["lease_token"], owner):
                        raise RuntimeError("current Task 7 lease refused the owned worker PID binding")
                    break

            await self._monitor.observe(observation)
            current = await self._queue.get(job_id)
            active = await self._repository.get_active_lease(self._admission.expected_gpu_uuid)
            if current.get("state") in {"completed", "failed", "cancelled"} and (
                active is None or active.get("job_id") != job_id
            ):
                await self._cleanup_terminal_artifact_writes(job_id)
                return current
            await self._sleep(self._interval_seconds)
        raise TimeoutError(f"training controller timed out for job {job_id}")

    async def _observer_call(self) -> ResourceObservation:
        method = self._observer.observe
        if asyncio.iscoroutinefunction(method):
            value = await method()
        else:
            value = await asyncio.to_thread(method)
        return value if isinstance(value, ResourceObservation) else ResourceObservation.from_dict(value)

    async def _cleanup_terminal_artifact_writes(self, job_id: str) -> None:
        cleanup = getattr(self._queue, "cleanup_pending_artifact_writes_from_environment", None)
        if callable(cleanup):
            await cleanup(job_id)

    async def close(self) -> None:
        await self._repository.close()
        await self._queue.source_registry.close()


def build_controller_from_environment(*, worker_image: str, namespace: str) -> TrainingController:
    """Build the CPU controller lazily so a waiting component never imports torch."""
    from kubernetes import client, config

    config.load_incluster_config()
    database_url = _required("GODS_MLOPS_DATABASE_URL")
    node_id = _required("GODS_MLOPS_UBUNTU_NODE_ID")
    host_identity = _required("GODS_MLOPS_UBUNTU_HOST_IDENTITY")
    gpu_uuid = _required("GODS_MLOPS_UBUNTU_GPU_UUID")
    filesystem_identity = _required("GODS_MLOPS_UBUNTU_FILESYSTEM_IDENTITY")
    storage_path = _required("GODS_MLOPS_UBUNTU_STORAGE_PATH")
    observer = UbuntuResourceObserver(
        node_id=node_id,
        host_identity=host_identity,
        ssh_target=_required("GODS_MLOPS_UBUNTU_SSH_TARGET"),
        gpu_uuid=gpu_uuid,
        filesystem_identity=filesystem_identity,
        storage_path=storage_path,
        ssh_port=int(os.environ["GODS_MLOPS_UBUNTU_SSH_PORT"])
        if os.environ.get("GODS_MLOPS_UBUNTU_SSH_PORT")
        else None,
        identity_file=os.environ.get("GODS_MLOPS_UBUNTU_SSH_IDENTITY_FILE"),
        known_hosts_file=os.environ.get("GODS_MLOPS_UBUNTU_SSH_KNOWN_HOSTS"),
        timeout_seconds=int(os.environ.get("GODS_MLOPS_UBUNTU_SSH_TIMEOUT_SECONDS", "10")),
    )
    repository = PostgresJobQueueRepository(
        database_url=database_url,
        expected_node_id=node_id,
        expected_host_identity=host_identity,
        expected_gpu_uuid=gpu_uuid,
        expected_filesystem_identity=filesystem_identity,
        expected_storage_path=storage_path,
    )
    sources = DatasetSourceRegistry(database_url=database_url)
    queue = JobQueue(repository=repository, sources=sources)
    admission = GpuAdmission(
        repository=repository,
        queue=queue,
        expected_node_id=node_id,
        expected_host_identity=host_identity,
        expected_gpu_uuid=gpu_uuid,
        expected_filesystem_identity=filesystem_identity,
        expected_storage_path=storage_path,
        observer=observer,
    )
    monitor = GpuJobMonitor(repository=repository, queue=queue, admission=admission)
    worker_adapter = KubernetesOwnedWorkerAdapter(
        batch_api=client.BatchV1Api(), namespace=namespace, image=worker_image
    )
    return TrainingController(
        queue=queue,
        repository=repository,
        admission=admission,
        monitor=monitor,
        observer=observer,
        worker_adapter=worker_adapter,
        core_api=client.CoreV1Api(),
        namespace=namespace,
    )


async def submit_training(
    *, dataset_version: str, model_kind: str, config_version: str
) -> tuple[JobQueue, str]:
    repository, queue = _queue_from_environment()
    job_id = await queue.submit(dataset_version, model_kind, config_version)
    return queue, job_id


async def submit_evaluation(
    *,
    dataset_version: str,
    model_kind: str,
    config_version: str,
    checkpoint_source,
    evaluation_split: str = "test",
    baseline_metadata: dict[str, str | None] | None = None,
    rerun: bool = False,
) -> tuple[JobQueue, str]:
    """Submit one prior-checkpoint evaluation through the existing durable queue."""
    repository, queue = _queue_from_environment()
    job_id = await queue.submit_evaluation(
        dataset_version=dataset_version,
        model_kind=model_kind,
        config_version=config_version,
        checkpoint_source=checkpoint_source,
        evaluation_split=evaluation_split,
        baseline_metadata=baseline_metadata,
        rerun=rerun,
    )
    return queue, job_id


async def submit_preparation(
    *, source_selections: list[dict[str, Any]], model_kind: str, config_version: str
) -> tuple[JobQueue, str]:
    if model_kind not in {"detr", "qwen"}:
        raise ValueError("preparation supports RT-DETR frame drafts and Qwen crop captions only")
    repository, queue = _queue_from_environment()
    selections = [
        AnnotationSourceSelection(
            item_kind=str(item["item_kind"]),
            item_id=str(item["item_id"]),
            sha256=str(item["sha256"]),
            revision_id=str(item["revision_id"]) if item.get("revision_id") is not None else None,
        )
        for item in source_selections
    ]
    expected_kind = "frame" if model_kind == "detr" else "crop"
    if any(item.item_kind != expected_kind for item in selections):
        raise ValueError(f"{model_kind} preparation requires only {expected_kind} source refs")
    batch = await queue.source_registry.prepare_annotation_batch(selections)
    job_id = await queue.submit_preparation(
        batch=batch, model_kind=model_kind, config_version=config_version
    )
    return queue, job_id


def _queue_from_environment() -> tuple[PostgresJobQueueRepository, JobQueue]:
    database_url = _required("GODS_MLOPS_DATABASE_URL")
    repository = PostgresJobQueueRepository(database_url=database_url)
    sources = DatasetSourceRegistry(database_url=database_url)
    return repository, JobQueue(repository=repository, sources=sources)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable {name} is not configured")
    return value


def _get(value: Any, camel: str, snake: str | None = None) -> Any:
    if isinstance(value, dict):
        return value.get(camel, value.get(snake) if snake else None)
    attr = snake or "".join(("_" + char.lower()) if char.isupper() else char for char in camel)
    return getattr(value, attr, None)
