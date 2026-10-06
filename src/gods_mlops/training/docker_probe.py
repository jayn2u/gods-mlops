"""Strict-SSH Docker adapter for bounded real-model probes on the Ubuntu A6000."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from gods_mlops.jobs.admission import GpuAdmission
from gods_mlops.jobs.models import ProbeInput, ResourceObservation
from gods_mlops.jobs.monitor import GpuJobMonitor
from gods_mlops.jobs.observer import UbuntuResourceObserver
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

from .claims import WorkerClaim
from .contracts import locked_model
from .data import dataset_object_store_from_environment
from .placement import validate_worker_placement
from .probe_setup import candidate_profile, create_probe_input, load_evaluation_probe_checkpoint_source
from .worker_adapter import resolve_owned_docker_process

_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}$")
_CONTAINER_ID = re.compile(r"^[0-9a-f]{64}$")

_REMOTE_DOCKER_RUNNER = r"""
import json, os, subprocess, sys, tempfile

payload=json.load(sys.stdin)
environment=payload["environment"]
for key,value in environment.items():
    if not isinstance(key,str) or not isinstance(value,str) or "\n" in value or "\r" in value:
        raise ValueError("probe container environment is malformed")
fd,path=tempfile.mkstemp(prefix="gods-task8-probe-env-",dir="/tmp")
os.fchmod(fd,0o600)
try:
    with os.fdopen(fd,"w",encoding="utf-8") as output:
        for key,value in sorted(environment.items()):
            output.write(key+"="+value+"\n")
    result=subprocess.run(["docker","run","--env-file",path,*payload["docker_args"]],
        capture_output=True,text=True,check=False,timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:] or "remote Docker run failed")
    print(result.stdout.strip())
finally:
    try: os.unlink(path)
    except FileNotFoundError: pass
"""


@dataclass(frozen=True, slots=True)
class DockerProbeEvidence:
    job_id: str
    model_kind: str
    target_phase: str
    config_version: str
    input_sha256: str
    image_reference: str
    docker_image_id: str
    image_source_commit: str
    source_commit: str
    container_id: str
    fencing_tokens: list[int]
    owner_processes: list[dict[str, int]]
    owner_pid: int
    owner_start_ticks: int
    owner_uid: int
    measurement: dict
    result_artifact: dict
    output: dict
    evaluation_checkpoint_source: dict | None = None

    def as_dict(self) -> dict:
        result = {
            "event": "task8_real_model_probe_complete",
            "job_id": self.job_id,
            "model_kind": self.model_kind,
            "target_phase": self.target_phase,
            "config_version": self.config_version,
            "input_sha256": self.input_sha256,
            "image_reference": self.image_reference,
            "docker_image_id": self.docker_image_id,
            "image_source_commit": self.image_source_commit,
            "source_commit": self.source_commit,
            "container_id": self.container_id,
            "fencing_tokens": self.fencing_tokens,
            "owner_processes": self.owner_processes,
            "worker_uid": self.owner_uid,
            "owner_pid_start": [self.owner_pid, self.owner_start_ticks],
            "measurement": self.measurement,
            "result_artifact": self.result_artifact,
            "output": self.output,
        }
        if self.evaluation_checkpoint_source is not None:
            result["evaluation_checkpoint_source"] = self.evaluation_checkpoint_source
        return result


async def _finalize_probe_runtime_evidence(
    *,
    repository: PostgresJobQueueRepository,
    objects,
    bucket: str,
    evidence: DockerProbeEvidence,
) -> bytes:
    """Finalize exact controller evidence bytes, then persist authority for training probes."""
    payload = (json.dumps(evidence.as_dict(), sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if evidence.target_phase == "training":
        from .checkpoints import S3CheckpointStore

        identity = await repository.checkpoint_identity(evidence.job_id)
        commit = await repository.checkpoint_metadata_for(evidence.job_id)
        if commit is None or identity.as_dict() != commit["identity"]:
            raise DockerProbeError("successful training probe has no matching committed checkpoint identity")
        checkpoint_store = S3CheckpointStore(objects=objects, bucket=bucket)
        verified_checkpoint = await asyncio.to_thread(
            checkpoint_store.load_uri,
            commit["uri"],
            expected_identity=identity,
            expected_sha256=commit["sha256"],
            expected_size_bytes=commit["size_bytes"],
        )
        if (
            verified_checkpoint.identity != identity
            or verified_checkpoint.sha256 != commit["sha256"]
            or verified_checkpoint.size_bytes != commit["size_bytes"]
        ):
            raise DockerProbeError("successful training probe checkpoint failed exact S3 readback")
        await repository._record_probe_runtime_evidence(payload)
    return payload


class DockerProbeError(RuntimeError):
    """A bounded real-model Docker probe failed before or during execution."""


class DockerProbeYielded(DockerProbeError):
    """The fenced probe checkpointed and can resume after the next admission window."""


async def run_docker_model_probe(
    *,
    model_kind: str,
    worker_image: str,
    expected_image_id: str,
    target_phase: str | None = None,
    training_probe_job_id: str | None = None,
    model_cache_root: str = "/data/jayn2u/gods-mlops-model-preparation",
    timeout_seconds: int = 3600,
    evidence_directory: str | Path = "output/task8-real-model-probes",
) -> DockerProbeEvidence:
    """Submit one typed candidate through Task 7, bind Docker host PID, and run its worker."""
    if model_kind not in {"detr", "clip", "qwen"}:
        raise ValueError("real-model probe model_kind must be detr, clip, or qwen")
    profile = candidate_profile(model_kind, target_phase=target_phase)
    if profile.target_phase == "evaluation" and training_probe_job_id is None:
        raise ValueError("evaluation profile probes require a successful training-probe job ID")
    if profile.target_phase != "evaluation" and training_probe_job_id is not None:
        raise ValueError("training-probe checkpoint source is only valid for evaluation profile calibration")
    if not _IMAGE_REF.fullmatch(worker_image) or not _IMAGE_ID.fullmatch(expected_image_id):
        raise ValueError("probe requires the source-matched image reference and exact Docker image ID")
    if not model_cache_root.startswith("/") or any(char in model_cache_root for char in ":,\n\r"):
        raise ValueError("probe model-cache bind path is invalid")
    if not 120 <= timeout_seconds <= 14_400:
        raise ValueError("real-model probe timeout must be between two minutes and four hours")
    source_commit = _require_committed_source()
    _verify_runtime_environment()

    database_url = _required("GODS_MLOPS_DATABASE_URL")
    endpoint_url = _required("GODS_MLOPS_S3_ENDPOINT_URL")
    db_local_host, db_local_port = _loopback_database(database_url)
    s3_local_host, s3_local_port = _loopback_endpoint(endpoint_url)
    tunnel_database_port = int(os.environ.get("GODS_MLOPS_PROBE_REMOTE_DATABASE_PORT", "35439"))
    tunnel_s3_port = int(os.environ.get("GODS_MLOPS_PROBE_REMOTE_S3_PORT", "38333"))
    if not (1 <= tunnel_database_port <= 65_535 and 1 <= tunnel_s3_port <= 65_535):
        raise ValueError("remote probe tunnel ports are invalid")

    node_id = _required("GODS_MLOPS_UBUNTU_NODE_ID")
    host_identity = _required("GODS_MLOPS_UBUNTU_HOST_IDENTITY")
    gpu_uuid = _required("GODS_MLOPS_UBUNTU_GPU_UUID")
    filesystem_identity = _required("GODS_MLOPS_UBUNTU_FILESYSTEM_IDENTITY")
    storage_path = _required("GODS_MLOPS_UBUNTU_STORAGE_PATH")
    ssh_target = _required("GODS_MLOPS_UBUNTU_SSH_TARGET")
    ssh_identity_file = _required("GODS_MLOPS_UBUNTU_SSH_IDENTITY_FILE")
    ssh_known_hosts = _required("GODS_MLOPS_UBUNTU_SSH_KNOWN_HOSTS")
    ssh_port = int(os.environ.get("GODS_MLOPS_UBUNTU_SSH_PORT", "22"))
    if not 1 <= ssh_port <= 65_535:
        raise ValueError("Ubuntu SSH port is invalid")

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
    observer = UbuntuResourceObserver(
        node_id=node_id,
        host_identity=host_identity,
        ssh_target=ssh_target,
        ssh_port=ssh_port,
        identity_file=ssh_identity_file,
        known_hosts_file=ssh_known_hosts,
        gpu_uuid=gpu_uuid,
        filesystem_identity=filesystem_identity,
        storage_path=storage_path,
        timeout_seconds=int(os.environ.get("GODS_MLOPS_UBUNTU_SSH_TIMEOUT_SECONDS", "10")),
    )
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
    tunnel: subprocess.Popen | None = None
    container_id: str | None = None
    owned_containers: list[str] = []
    job_id: str | None = None
    try:
        await repository.ensure_schema()
        await queue.register_profile(profile)
        if profile.config_version is None:
            raise RuntimeError("candidate profile lost its immutable config version")
        objects = dataset_object_store_from_environment()
        evaluation_checkpoint_source = None
        if profile.target_phase == "evaluation":
            evaluation_checkpoint_source = await load_evaluation_probe_checkpoint_source(
                repository=repository,
                objects=objects,
                bucket=_required("GODS_MLOPS_S3_BUCKET"),
                training_probe_job_id=training_probe_job_id,
                model_kind=model_kind,
            )
        probe_input = create_probe_input(
            objects=objects,
            model_kind=model_kind,
            target_phase=profile.target_phase,
            evaluation_checkpoint_source=evaluation_checkpoint_source,
        )
        docker_image_id = await _remote_ssh(
            ssh_target,
            ssh_port,
            ssh_identity_file,
            ssh_known_hosts,
            ["docker", "image", "inspect", "--format", "{{.Id}}", worker_image],
            timeout=30,
        )
        if docker_image_id != expected_image_id:
            raise DockerProbeError("cached Ubuntu Docker image ID differs from the committed source-matched image")
        image_source_commit = await _remote_ssh(
            ssh_target,
            ssh_port,
            ssh_identity_file,
            ssh_known_hosts,
            [
                "docker", "image", "inspect", "--format",
                "{{ index .Config.Labels \"org.opencontainers.image.revision\" }}",
                worker_image,
            ],
            timeout=30,
        )
        if image_source_commit != source_commit:
            raise DockerProbeError("cached Docker image label differs from the committed Task 8 source")
        model = locked_model(model_kind)
        await _verify_model_cache(
            ssh_target=ssh_target,
            ssh_port=ssh_port,
            identity_file=ssh_identity_file,
            known_hosts=ssh_known_hosts,
            image=worker_image,
            cache_root=model_cache_root,
            model_id=model.model_id,
        )

        tunnel = _start_reverse_tunnel(
            ssh_target=ssh_target,
            ssh_port=ssh_port,
            identity_file=ssh_identity_file,
            known_hosts=ssh_known_hosts,
            database_local=(db_local_host, db_local_port),
            database_remote_port=tunnel_database_port,
            storage_local=(s3_local_host, s3_local_port),
            storage_remote_port=tunnel_s3_port,
        )
        await _wait_for_reverse_tunnel(
            ssh_target=ssh_target,
            ssh_port=ssh_port,
            identity_file=ssh_identity_file,
            known_hosts=ssh_known_hosts,
            ports=(tunnel_database_port, tunnel_s3_port),
            process=tunnel,
        )
        job_id = await queue.submit_probe(
            model_kind=model_kind,
            config_version=profile.config_version,
            probe_input=probe_input,
            rerun=False,
        )
        current_profile = await repository.get_profile(
            phase="probe", model_kind=model_kind, config_version=profile.config_version
        )
        if current_profile is None or current_profile["profile_state"] != "candidate":
            raise DockerProbeError("candidate profile is not available for a new real-model measurement")
        run_deadline = time.monotonic() + timeout_seconds
        owner = None
        lease_generations: list[int] = []
        owner_processes: list[dict[str, int]] = []
        while True:
            remaining = int(run_deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("real-model probe exceeded its bounded end-to-end timeout")
            lease = await _admit_probe(
                job_id=job_id,
                repository=repository,
                queue=queue,
                admission=admission,
                max_wait_seconds=min(
                    remaining,
                    int(os.environ.get("GODS_MLOPS_PROBE_MAX_WAIT_SECONDS", "1800")),
                ),
            )
            claim = WorkerClaim.from_admitted_job(
                await queue.get(job_id), lease, image_id=expected_image_id
            )
            lease_generations.append(claim.fence)
            container_name = f"gods-task8-{uuid4().hex[:12]}"
            environment = _worker_environment(
                claim=claim,
                database_url=database_url,
                database_remote_port=tunnel_database_port,
                storage_remote_port=tunnel_s3_port,
                image_id=expected_image_id,
                gpu_uuid=gpu_uuid,
            )
            container_id = await _start_worker_container(
                ssh_target=ssh_target,
                ssh_port=ssh_port,
                identity_file=ssh_identity_file,
                known_hosts=ssh_known_hosts,
                image=worker_image,
                container_name=container_name,
                gpu_uuid=gpu_uuid,
                model_cache_root=model_cache_root,
                environment=environment,
            )
            owned_containers.append(container_id)
            try:
                owner = await _bind_docker_worker(
                    container_id=container_id,
                    job_id=job_id,
                    lease_token=claim.lease_token,
                    repository=repository,
                    queue=queue,
                    admission=admission,
                    monitor=monitor,
                    ssh_target=ssh_target,
                    ssh_port=ssh_port,
                    identity_file=ssh_identity_file,
                    known_hosts=ssh_known_hosts,
                    timeout_seconds=min(60, remaining),
                )
                owner_processes.append(
                    {"pid": owner.pid, "start_ticks": owner.start_ticks, "uid": int(owner.uid or 0)}
                )
                await _monitor_docker_worker(
                    container_id=container_id,
                    job_id=job_id,
                    lease_token=claim.lease_token,
                    owner=owner,
                    repository=repository,
                    queue=queue,
                    monitor=monitor,
                    admission=admission,
                    ssh_target=ssh_target,
                    ssh_port=ssh_port,
                    identity_file=ssh_identity_file,
                    known_hosts=ssh_known_hosts,
                    timeout_seconds=max(1, int(run_deadline - time.monotonic())),
                )
                break
            except DockerProbeYielded:
                await _remote_ssh(
                    ssh_target, ssh_port, ssh_identity_file, ssh_known_hosts,
                    ["docker", "rm", "--force", container_id], timeout=30,
                )
                container_id = None
                continue
        measurement = await repository.profile_measurement_for_job(job_id)
        if measurement is None or measurement["result_state"] != "succeeded":
            raise DockerProbeError("real-model probe completed without a successful profile measurement")
        artifact = await _probe_result_artifact(
            queue,
            objects,
            job_id,
            identity=await repository.checkpoint_identity(job_id),
            target_phase=profile.target_phase or "training",
        )
        evidence = DockerProbeEvidence(
            job_id=job_id,
            model_kind=model_kind,
            target_phase=profile.target_phase or "training",
            config_version=profile.config_version,
            input_sha256=claim.input_sha256,
            image_reference=worker_image,
            docker_image_id=docker_image_id,
            image_source_commit=image_source_commit,
            source_commit=source_commit,
            container_id=container_id,
            fencing_tokens=lease_generations,
            owner_processes=owner_processes,
            owner_pid=owner.pid,
            owner_start_ticks=owner.start_ticks,
            owner_uid=int(owner.uid or 0),
            measurement=measurement,
            result_artifact=artifact[0],
            output=artifact[1],
            evaluation_checkpoint_source=(
                evaluation_checkpoint_source.as_dict()
                if evaluation_checkpoint_source is not None
                else None
            ),
        )
        evidence_payload = await _finalize_probe_runtime_evidence(
            repository=repository,
            objects=objects,
            bucket=_required("GODS_MLOPS_S3_BUCKET"),
            evidence=evidence,
        )
        evidence_path = Path(evidence_directory) / f"{model_kind}-{job_id}.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_bytes(evidence_payload)
        print(json.dumps({"evidence": str(evidence_path), **evidence.as_dict()}, sort_keys=True))
        return evidence
    finally:
        for owned_container in reversed(owned_containers):
            try:
                await _remote_ssh(
                    ssh_target,
                    ssh_port,
                    ssh_identity_file,
                    ssh_known_hosts,
                    ["docker", "rm", "--force", owned_container],
                    timeout=30,
                )
            except Exception:
                pass
            if job_id is not None:
                try:
                    await _reconcile_probe_lease(
                        job_id=job_id,
                        repository=repository,
                        queue=queue,
                        admission=admission,
                        monitor=monitor,
                    )
                except Exception:
                    pass
        if tunnel is not None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.kill()
                tunnel.wait(timeout=5)
        await queue.close()
        await sources.close()
        await repository.close()


def _verify_runtime_environment() -> None:
    status = subprocess.run(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            "src/gods_mlops", "pyproject.toml", "uv.lock", "models/lock.json",
            "infra/versions.lock.yaml", "images/training/Dockerfile",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise DockerProbeError("commit the source-matched Task 8 worker before model probe execution")


def _require_committed_source() -> str:
    root = Path(__file__).resolve().parents[3]
    _verify_runtime_environment()
    commit = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise DockerProbeError("Task 8 source commit identity is invalid")
    return commit


def _loopback_database(database_url: str) -> tuple[str, int]:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port is None:
        raise ValueError("strict-SSH probe adapter requires a loopback PostgreSQL endpoint")
    return parsed.hostname, parsed.port


def _loopback_endpoint(endpoint_url: str) -> tuple[str, int]:
    parsed = urlsplit(endpoint_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.port is None:
        raise ValueError("strict-SSH probe adapter requires a loopback S3 endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("probe S3 endpoint must not embed credentials, query, or fragment data")
    return parsed.hostname, parsed.port


def _remote_database_url(database_url: str, port: int) -> str:
    parsed = urlsplit(database_url)
    userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
    return urlunsplit((parsed.scheme, f"{userinfo}127.0.0.1:{port}", parsed.path, parsed.query, parsed.fragment))


def _remote_endpoint_url(local_endpoint: str, port: int) -> str:
    parsed = urlsplit(local_endpoint)
    return urlunsplit((parsed.scheme, f"127.0.0.1:{port}", parsed.path, "", ""))


def _start_reverse_tunnel(
    *,
    ssh_target: str,
    ssh_port: int,
    identity_file: str,
    known_hosts: str,
    database_local: tuple[str, int],
    database_remote_port: int,
    storage_local: tuple[str, int],
    storage_remote_port: int,
) -> subprocess.Popen:
    command = [
        "ssh", "-N", "-T", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}",
        "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2", "-p", str(ssh_port), "-i", identity_file,
        "-R", f"127.0.0.1:{database_remote_port}:{database_local[0]}:{database_local[1]}",
        "-R", f"127.0.0.1:{storage_remote_port}:{storage_local[0]}:{storage_local[1]}",
        ssh_target,
    ]
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def _wait_for_reverse_tunnel(
    *,
    ssh_target: str,
    ssh_port: int,
    identity_file: str,
    known_hosts: str,
    ports: tuple[int, int],
    process: subprocess.Popen,
) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise DockerProbeError("strict-SSH reverse tunnel exited before probe startup")
        try:
            for port in ports:
                await _remote_ssh(
                    ssh_target, ssh_port, identity_file, known_hosts,
                    ["python3", "-c", "import socket,sys; s=socket.create_connection(('127.0.0.1',int(sys.argv[1])),2); s.close()", str(port)],
                    timeout=5,
                )
            return
        except Exception:
            await asyncio.sleep(0.5)
    raise TimeoutError("strict-SSH local service tunnels did not become reachable on Ubuntu loopback")


async def _remote_ssh(
    target: str,
    port: int,
    identity_file: str,
    known_hosts: str,
    command: list[str],
    *,
    input_bytes: bytes | None = None,
    timeout: int = 30,
) -> str:
    args = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={known_hosts}", "-o", "IdentitiesOnly=yes",
        "-o", f"ConnectTimeout={min(timeout, 10)}", "-p", str(port), "-i", identity_file,
        target, shlex.join(command),
    ]
    result = await asyncio.to_thread(
        subprocess.run,
        args,
        input=input_bytes,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode:
        raise DockerProbeError("strict-SSH Docker probe command failed")
    return result.stdout.decode("utf-8", "strict").strip()


async def _verify_model_cache(
    *,
    ssh_target: str,
    ssh_port: int,
    identity_file: str,
    known_hosts: str,
    image: str,
    cache_root: str,
    model_id: str,
) -> None:
    result = await _remote_ssh(
        ssh_target,
        ssh_port,
        identity_file,
        known_hosts,
        [
            "docker", "run", "--rm", "--pull=never", "--network=none", "--user", "10001:10001",
            "--mount", f"type=bind,source={cache_root},target=/mnt/model-cache,readonly",
            "--entrypoint", "gods-mlops", image, "check-models", "--model-id", model_id,
            "--cache-root", "/mnt/model-cache",
        ],
        timeout=900,
    )
    if "model files ready: 1 locked model revision" not in result:
        raise DockerProbeError("UID 10001 could not verify every locked model file in the host cache")


def _worker_environment(
    *,
    claim: WorkerClaim,
    database_url: str,
    database_remote_port: int,
    storage_remote_port: int,
    image_id: str,
    gpu_uuid: str,
) -> dict[str, str]:
    values = {
        "GODS_MLOPS_DATABASE_URL": _remote_database_url(database_url, database_remote_port),
        "GODS_MLOPS_S3_ENDPOINT_URL": _remote_endpoint_url(_required("GODS_MLOPS_S3_ENDPOINT_URL"), storage_remote_port),
        "GODS_MLOPS_S3_BUCKET": _required("GODS_MLOPS_S3_BUCKET"),
        "GODS_MLOPS_S3_REGION": os.environ.get("GODS_MLOPS_S3_REGION", "us-east-1"),
        "GODS_MLOPS_S3_ACCESS_KEY": _required("GODS_MLOPS_S3_ACCESS_KEY"),
        "GODS_MLOPS_S3_SECRET_KEY": _required("GODS_MLOPS_S3_SECRET_KEY"),
        "GODS_MLOPS_JOB_ID": claim.job_id,
        "GODS_MLOPS_LEASE_TOKEN": claim.lease_token,
        "GODS_MLOPS_FENCE": str(claim.fence),
        "GODS_MLOPS_GPU_UUID": gpu_uuid,
        "GODS_MLOPS_PHASE": claim.phase,
        "GODS_MLOPS_TARGET_PHASE": claim.target_phase,
        "GODS_MLOPS_INPUT_KIND": claim.input_kind,
        "GODS_MLOPS_INPUT_ID": claim.input_id,
        "GODS_MLOPS_INPUT_SHA256": claim.input_sha256,
        "GODS_MLOPS_DATASET_VERSION": "",
        "GODS_MLOPS_MODEL_KIND": claim.model_kind,
        "GODS_MLOPS_CONFIG_VERSION": claim.config_version,
        "GODS_MLOPS_CONFIG_SHA256": claim.config_sha256,
        "GODS_MLOPS_IMAGE_ID": image_id,
        "GODS_MLOPS_NODE_NAME": "ubuntu",
        "GODS_MLOPS_MODEL_LOCK": "/app/models/lock.json",
        "GODS_MLOPS_MODEL_CACHE_ROOT": "/mnt/model-cache",
        "GODS_MLOPS_CHECKPOINT_ROOT": "/tmp/gods-mlops-checkpoints",
        "HOME": "/tmp",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    return values


async def _start_worker_container(
    *,
    ssh_target: str,
    ssh_port: int,
    identity_file: str,
    known_hosts: str,
    image: str,
    container_name: str,
    gpu_uuid: str,
    model_cache_root: str,
    environment: dict[str, str],
) -> str:
    docker_args = [
        "--detach", "--pull=never", "--name", container_name,
        "--label", "gods-mlops.task=task8-real-model-probe",
        "--label", f"gods-mlops.probe={container_name}",
        "--pid=host", "--network=host", "--gpus", f"device={gpu_uuid}",
        "--user", "10001:10001", "--workdir", "/tmp",
        "--tmpfs", "/tmp:rw,nosuid,size=8g",
        "--mount", f"type=bind,source={model_cache_root},target=/mnt/model-cache,readonly",
        "--entrypoint", "gods-mlops-training", image, "worker",
    ]
    payload = json.dumps({"docker_args": docker_args, "environment": environment}, sort_keys=True).encode()
    container_id = await _remote_ssh(
        ssh_target,
        ssh_port,
        identity_file,
        known_hosts,
        ["python3", "-c", _REMOTE_DOCKER_RUNNER],
        input_bytes=payload,
        timeout=120,
    )
    if not _CONTAINER_ID.fullmatch(container_id):
        raise DockerProbeError("remote Docker daemon did not return one full container ID")
    return container_id


async def _admit_probe(
    *,
    job_id: str,
    repository: PostgresJobQueueRepository,
    queue: JobQueue,
    admission: GpuAdmission,
    max_wait_seconds: int,
) -> dict:
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        observation = await admission.observe_once()
        job = await admission.admit(job_id, observation.to_dict())
        if job.get("state") in {"failed", "cancelled"}:
            raise DockerProbeError(
                f"candidate probe job is already terminal: {job.get('reason_code') or job['state']}"
            )
        if job.get("lease_token"):
            lease = await repository.get_active_lease(observation.gpu_uuid)
            if (
                lease is None
                or lease.get("job_id") != job_id
                or lease.get("lease_token") != job.get("lease_token")
            ):
                raise DockerProbeError("Task 7 admission returned a probe without its current lease")
            return lease
        await asyncio.sleep(5)
    job = await queue.get(job_id)
    raise TimeoutError(
        f"real-model probe remained {job.get('state')} for {max_wait_seconds}s: {job.get('reason_code')}"
    )


async def _bind_docker_worker(
    *,
    container_id: str,
    job_id: str,
    lease_token: str,
    repository: PostgresJobQueueRepository,
    queue: JobQueue,
    admission: GpuAdmission,
    monitor: GpuJobMonitor,
    ssh_target: str,
    ssh_port: int,
    identity_file: str,
    known_hosts: str,
    timeout_seconds: int,
):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        observation = await admission.observe_once()
        state = await _remote_ssh(
            ssh_target, ssh_port, identity_file, known_hosts,
            ["docker", "inspect", "--format", "{{.State.Status}}", container_id],
        )
        if state in {"exited", "dead"}:
            await monitor.observe(observation)
            raise DockerProbeError("probe worker exited before a current host owner bind")
        lease = await repository.get_active_lease(observation.gpu_uuid)
        if lease is None or lease.get("job_id") != job_id or lease.get("lease_token") != lease_token:
            raise DockerProbeError("probe lease changed before Docker host identity could be bound")
        if lease.get("owner_pid") is not None:
            owner = next(
                (
                    item for item in observation.process_table
                    if item.pid == lease["owner_pid"]
                    and item.start_ticks == lease["owner_start_ticks"]
                    and item.uid == 10001
                ),
                None,
            )
            if owner is not None:
                await monitor.observe(observation)
                return owner
            await monitor.observe(observation)
            await asyncio.sleep(5)
            continue
        process_pid_text = await _remote_ssh(
            ssh_target, ssh_port, identity_file, known_hosts,
            ["docker", "inspect", "--format", "{{.State.Pid}}", container_id],
        )
        try:
            process_pid = int(process_pid_text)
        except ValueError:
            process_pid = 0
        if process_pid > 0:
            try:
                owner = resolve_owned_docker_process(
                    container_id=container_id,
                    observation=observation,
                )
            except ValueError:
                owner = None
            if owner is not None and owner.pid == process_pid:
                if not await queue.bind_process(job_id, lease_token, owner):
                    raise DockerProbeError("Task 7 lease rejected the Docker ID→host PID binding")
                await monitor.observe(observation)
                return owner
        await monitor.observe(observation)
        await asyncio.sleep(5)
    raise TimeoutError("strict-SSH observer did not resolve the exact Docker container host PID/cgroup")


async def _monitor_docker_worker(
    *,
    container_id: str,
    job_id: str,
    lease_token: str,
    owner,
    repository: PostgresJobQueueRepository,
    queue: JobQueue,
    monitor: GpuJobMonitor,
    admission: GpuAdmission,
    ssh_target: str,
    ssh_port: int,
    identity_file: str,
    known_hosts: str,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    failure_recorded = False
    while time.monotonic() < deadline:
        observation = await admission.observe_once()
        job = await queue.get(job_id)
        container_state = await _remote_ssh(
            ssh_target, ssh_port, identity_file, known_hosts,
            ["docker", "inspect", "--format", "{{.State.Status}}", container_id],
        )
        if container_state == "exited" and not failure_recorded and job.get("state") in {"running", "yield_requested"}:
            exit_text = await _remote_ssh(
                ssh_target, ssh_port, identity_file, known_hosts,
                ["docker", "inspect", "--format", "{{.State.ExitCode}}", container_id],
            )
            try:
                exit_code = int(exit_text)
            except ValueError:
                exit_code = 1
            if exit_code != 0 and job.get("state") == "running":
                await queue.record_probe_measurement(
                    job_id=job_id,
                    lease_token=lease_token,
                    exit_code=exit_code,
                    peak_allocated_mib=None,
                    peak_reserved_mib=None,
                    optimizer_steps=0,
                    checkpoint_resumed=False,
                    checkpoint_sha256=None,
                    inference_steps=0,
                    verification_details={"passed": False, "failure": "worker_exit_nonzero"},
                )
                failure_recorded = True
        status = await monitor.observe(observation)
        active = await repository.get_active_lease(admission.expected_gpu_uuid)
        current = status or await queue.get(job_id)
        if current.get("state") in {"completed", "failed", "cancelled"} and (
            active is None or active.get("job_id") != job_id
        ):
            if current.get("state") != "completed":
                raise DockerProbeError(f"real-model probe ended in state {current.get('state')}")
            if any((owner.pid, owner.start_ticks) == (proc.pid, proc.start_ticks)
                   for proc in observation.process_table + observation.gpu_processes):
                raise DockerProbeError("Docker probe completed before the host observer proved process exit")
            return
        if active is None and current.get("state") in {"waiting_gpu", "retrying"}:
            raise DockerProbeYielded("probe checkpointed at a safe boundary and awaits a new lease")
        await asyncio.sleep(5)
    raise TimeoutError("Task 7 monitor did not observe the Docker probe exit and release its GPU lease")


async def _probe_result_artifact(
    queue: JobQueue,
    objects,
    job_id: str,
    *,
    identity,
    target_phase: str,
) -> tuple[dict, dict]:
    artifacts = await queue.repository.result_artifacts_for(job_id)
    if target_phase == "preparation":
        expected_kind = "drafts" if identity.model_kind == "detr" else "caption_drafts"
    elif target_phase == "evaluation":
        expected_kind = "drafts" if identity.model_kind == "detr" else "evaluation_probe"
    else:
        expected_kind = "model"
    matches = [
        item for item in artifacts
        if item.get("kind") == expected_kind and item.get("identity") == identity.as_dict()
    ]
    if len(matches) != 1:
        raise DockerProbeError("successful model probe has no unique fenced S3 result artifact")
    artifact = matches[0]
    bucket, _, key = artifact["uri"][5:].partition("/")
    if bucket != _required("GODS_MLOPS_S3_BUCKET") or not key:
        raise DockerProbeError("probe result artifact is outside the shared Task 6 bucket")
    payload = objects.read_source(
        object_key=key,
        sha256_digest=artifact["sha256"],
        size_bytes=artifact["size_bytes"],
    )
    if target_phase == "preparation":
        output = {
            "result_document": json.loads(payload),
            "resource_measurements": artifact.get("runtime_measurements"),
        }
    elif target_phase == "evaluation":
        output = {
            "fixture_probe_document": json.loads(payload),
            "resource_measurements": artifact.get("runtime_measurements"),
            "human_relevance_truth_used": False,
        }
    elif identity.model_kind == "qwen":
        output = json.loads(payload)
    else:
        import tarfile
        from io import BytesIO

        with tarfile.open(fileobj=BytesIO(payload), mode="r:") as archive:
            metrics = json.load(archive.extractfile("metrics.json"))
        output = metrics
    return artifact, output


async def _admit_and_reconcile_on_failure(
    *,
    job_id: str,
    repository: PostgresJobQueueRepository,
    queue: JobQueue,
    admission: GpuAdmission,
    monitor: GpuJobMonitor,
) -> None:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if await repository.get_active_lease(admission.expected_gpu_uuid) is None:
            return
        try:
            observation = await admission.observe_once()
            await monitor.observe(observation)
        except Exception:
            pass
        await asyncio.sleep(5)


async def _reconcile_probe_lease(
    *,
    job_id: str,
    repository: PostgresJobQueueRepository,
    queue: JobQueue,
    admission: GpuAdmission,
    monitor: GpuJobMonitor,
) -> None:
    await _admit_and_reconcile_on_failure(
        job_id=job_id,
        repository=repository,
        queue=queue,
        admission=admission,
        monitor=monitor,
    )


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required strict-SSH probe setting {name} is not configured")
    return value
