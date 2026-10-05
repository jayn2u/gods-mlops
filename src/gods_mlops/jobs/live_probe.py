"""One owned, real-CUDA queue/checkpoint lifecycle acceptance harness."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import posixpath
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from .admission import GPU_MONITOR_INTERVAL_SECONDS, GpuAdmission
from .checkpoints import FileCheckpointStore
from .models import ExecutionProfile, ProcessIdentity, ResourceObservation
from .monitor import GpuJobMonitor
from .observer import UbuntuResourceObserver
from .queue import JobQueue, PostgresJobQueueRepository
from .sources import DatasetSourceRegistry

CUDA_BYTES = 16 * 1024**2
ARTIFACT_BYTES = 32 * 1024**2
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")

WORKER_PROGRAM = r'''
import base64, hashlib, json, os, pathlib, sys, time
import torch

def emit(event, **values):
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)

def process_identity():
    stat = pathlib.Path("/proc/self/stat").read_text(encoding="ascii")
    fields = stat[stat.rfind(")") + 2:].split()
    if len(fields) <= 19:
        raise RuntimeError("procfs record lacks process start ticks")
    return {"pid": os.getpid(), "start_ticks": int(fields[19]), "uid": os.getuid()}

mode = sys.argv[1]
if mode == "preflight":
    emit("preflight", torch_version=torch.__version__)
    raise SystemExit(0)

if mode == "contender":
    allocation_bytes, duration = int(sys.argv[2]), int(sys.argv[3])
    owner = process_identity()
    tensor = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda:0")
    tensor.fill_(2)
    torch.cuda.synchronize()
    emit("contender", **owner, allocation_bytes=allocation_bytes)
    time.sleep(duration)
    del tensor
    torch.cuda.synchronize()
    raise SystemExit(0)

if mode != "worker":
    raise ValueError("unsupported live probe mode")

work = pathlib.Path(sys.argv[2])
job_id, source_sha = sys.argv[3], sys.argv[4]
resume_step, previous_sha = int(sys.argv[5]), sys.argv[6] or None
previous_payload = base64.b64decode(sys.argv[7], validate=True) if sys.argv[7] else None
allocation_bytes, timeout = int(sys.argv[8]), int(sys.argv[9])
if previous_payload is None:
    if resume_step != 0 or previous_sha is not None:
        raise ValueError("fresh worker received an invalid resume identity")
else:
    if hashlib.sha256(previous_payload).hexdigest() != previous_sha:
        raise ValueError("resume payload does not match the committed checkpoint SHA")
    previous = json.loads(previous_payload)
    if int(previous["step"]) != resume_step or previous["job_id"] != job_id:
        raise ValueError("resume payload belongs to another step or job")

owner = process_identity()
tensor = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda:0")
tensor.fill_(1)
torch.cuda.synchronize()
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
emit("allocated", **owner, job_id=job_id, allocation_bytes=allocation_bytes,
     resumed_from_step=resume_step, previous_checkpoint_sha256=previous_sha,
     cuda_free_mib=int(free_bytes / 1024**2), cuda_total_mib=int(total_bytes / 1024**2))
control = work / "control.json"
pending = work / "pending.json"
ack = work / "ack.json"
deadline = time.monotonic() + timeout
while time.monotonic() < deadline and not control.exists():
    time.sleep(0.25)
if not control.exists():
    raise TimeoutError("controller did not request a cooperative checkpoint")
request = json.loads(control.read_text(encoding="utf-8"))
payload = json.dumps({
    "format": "task7-live-cuda-probe-v1", "job_id": job_id, "step": resume_step + 1,
    "resumed_from_step": resume_step, "source_code_sha256": source_sha,
    "allocation_bytes": allocation_bytes, "owner_pid": owner["pid"],
    "owner_start_ticks": owner["start_ticks"], "model_training_performed": False,
    "yield_reason": request["reason"],
}, sort_keys=True, separators=(",", ":")).encode("utf-8")
digest = hashlib.sha256(payload).hexdigest()
temporary = pending.with_suffix(".partial")
temporary.write_bytes(payload)
temporary.replace(pending)
emit("checkpoint_pending", job_id=job_id, pid=owner["pid"], start_ticks=owner["start_ticks"],
     sha256=digest, step=resume_step + 1, size_bytes=len(payload))
while time.monotonic() < deadline and not ack.exists():
    time.sleep(0.25)
if not ack.exists() or json.loads(ack.read_text(encoding="utf-8")).get("sha256") != digest:
    raise TimeoutError("controller did not acknowledge the committed checkpoint SHA")
del tensor
torch.cuda.synchronize()
torch.cuda.empty_cache()
emit("checkpoint_acknowledged", job_id=job_id, pid=owner["pid"], start_ticks=owner["start_ticks"],
     sha256=digest, step=resume_step + 1,
     allocated_after_release=int(torch.cuda.memory_allocated(0)),
     reserved_after_release=int(torch.cuda.memory_reserved(0)))
'''


def proc_start_ticks(stat_record: str) -> int:
    """Read procfs start ticks despite spaces and parentheses in the process name."""
    end_name = stat_record.rfind(")")
    fields = stat_record[end_name + 2 :].split() if end_name >= 0 else []
    if len(fields) <= 19:
        raise ValueError("process stat lacks a start time")
    try:
        ticks = int(fields[19])
    except ValueError as error:
        raise ValueError("process stat lacks a valid start time") from error
    if ticks <= 0:
        raise ValueError("process stat lacks a valid start time")
    return ticks


def python_entrypoint_command(image: str, command: list[str]) -> list[str]:
    """Override the cached image's gods-mlops entrypoint for a Python command."""
    return ["--entrypoint", "python", image, *command]


def training_image_preflight_command(image: str) -> list[str]:
    """Run the exact committed worker code without exposing a GPU to Docker."""
    return python_entrypoint_command(image, ["-c", WORKER_PROGRAM, "preflight"])


def _source_revision() -> str:
    root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", "src/gods_mlops/jobs", "src/gods_mlops/migrations/0013_gpu_job_queue.sql", "pyproject.toml"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("commit the Task 7 runtime package before recording live probe evidence")
    revision = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise RuntimeError("source commit does not have a full Git object ID")
    return revision


def _ssh(args: argparse.Namespace, command: list[str], *, input_bytes: bytes | None = None, timeout: int = 30) -> str:
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes",
           "-o", f"UserKnownHostsFile={args.ssh_known_hosts}", "-i", args.ssh_identity_file]
    if args.ssh_port:
        ssh += ["-p", str(args.ssh_port)]
    result = subprocess.run(
        [*ssh, args.ssh_target, shlex.join(command)], check=False, capture_output=True, input=input_bytes, timeout=timeout
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-1500:].strip() or "remote command failed")
    return result.stdout.decode("utf-8", "strict").strip()


def _events(logs: str) -> list[dict[str, Any]]:
    events = []
    for line in logs.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("event"), str):
            events.append(item)
    return events


async def _wait_event(args: argparse.Namespace, container: str, event: str, timeout: int = 90) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        logs = _ssh(args, ["docker", "logs", container], timeout=20)
        events = _events(logs)
        if event in {item["event"] for item in events}:
            return events
        state = _ssh(args, ["docker", "inspect", "--format", "{{.State.Status}}", container])
        if state in {"exited", "dead"}:
            raise RuntimeError(f"owned container exited before {event}: {logs[-1000:]}")
        await asyncio.sleep(0.5)
    raise TimeoutError(f"container did not emit {event}")


def _container(args: argparse.Namespace, *, run_id: str, root: str, uid: str, gid: str,
               name: str, gpu_uuid: str, command: list[str], with_gpu: bool = True) -> str:
    docker = ["docker", "run", "--detach", "--pull=never", "--name", name,
              "--label", "gods-mlops.task=task7-live-probe", "--label", f"gods-mlops.probe={run_id}",
              "--pid=host", "--network=none", "--user", f"{uid}:{gid}",
              "--mount", f"type=bind,source={root}/work,target={root}/work"]
    if with_gpu:
        docker += ["--gpus", f"device={gpu_uuid}"]
    return_code = _ssh(args, [*docker, *python_entrypoint_command(
        args.training_image, ["-c", WORKER_PROGRAM, *command]
    )], timeout=60).splitlines()[-1]
    if not re.fullmatch(r"[0-9a-f]{12,64}", return_code):
        raise RuntimeError("Docker did not return an owned container ID")
    return return_code


async def _observe(admission: GpuAdmission, args: argparse.Namespace) -> ResourceObservation:
    observation = await admission.observe_once()
    identity = (observation.host_identity, observation.gpu_uuid, observation.filesystem_identity,
                posixpath.normpath(observation.storage_path))
    expected = (args.host_identity, args.gpu_uuid, args.filesystem_identity, posixpath.normpath(args.storage_path))
    if identity != expected:
        raise RuntimeError("strict observer did not match the pinned host, GPU, filesystem, and /data path")
    return observation


async def _admit(repo: PostgresJobQueueRepository, admission: GpuAdmission, args: argparse.Namespace,
                 job_id: str) -> tuple[dict[str, Any], ResourceObservation, ResourceObservation]:
    started = time.monotonic()
    first_idle = None
    while True:
        sample = await _observe(admission, args)
        if not sample.gpu_processes and first_idle is None:
            first_idle = sample
        result = await admission.admit(job_id, sample.to_dict())
        if result["state"] == "running":
            lease = await repo.get_active_lease(args.gpu_uuid)
            if lease is None or lease["job_id"] != job_id:
                raise RuntimeError("queue returned running without this job's durable GPU lease")
            return lease, first_idle or sample, sample
        if time.monotonic() - started > args.max_wait_seconds:
            raise TimeoutError(f"job remained {result['state']}: {result.get('reason_code')}")
        await asyncio.sleep(GPU_MONITOR_INTERVAL_SECONDS)


async def _confirm_and_bind_owner(queue: JobQueue, monitor: GpuJobMonitor, admission: GpuAdmission,
                                 args: argparse.Namespace, job_id: str, lease_token: str,
                                 container: str) -> tuple[ProcessIdentity, ResourceObservation, list[dict[str, Any]]]:
    events = await _wait_event(args, container, "allocated")
    allocated = next(item for item in events if item["event"] == "allocated")
    owner = ProcessIdentity(int(allocated["pid"]), int(allocated["start_ticks"]), int(allocated["uid"]))
    for _ in range(12):
        sample = await _observe(admission, args)
        key = (owner.pid, owner.start_ticks)
        if key in {(item.pid, item.start_ticks) for item in sample.gpu_processes} and key in {
            (item.pid, item.start_ticks) for item in sample.process_table
        }:
            if not await queue.bind_process(job_id, lease_token, owner):
                raise RuntimeError("observed worker PID/start identity did not bind to the GPU lease")
            return owner, sample, events
        await asyncio.sleep(GPU_MONITOR_INTERVAL_SECONDS)
    raise TimeoutError("Ubuntu observer did not confirm the exact owned PID on the GPU and process table")


async def _monitor_until_release(repo: PostgresJobQueueRepository, monitor: GpuJobMonitor,
                                 admission: GpuAdmission, args: argparse.Namespace,
                                 owner: ProcessIdentity, contender: ProcessIdentity | None = None) -> ResourceObservation:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        sample = await _observe(admission, args)
        await monitor.observe(sample)
        if await repo.get_active_lease(args.gpu_uuid) is None:
            key = (owner.pid, owner.start_ticks)
            if key in {(item.pid, item.start_ticks) for item in sample.process_table + sample.gpu_processes}:
                raise RuntimeError("monitor released the lease before the exact owner PID/CUDA process disappeared")
            if contender is not None and (contender.pid, contender.start_ticks) not in {
                (item.pid, item.start_ticks) for item in sample.gpu_processes
            }:
                raise RuntimeError("controlled contender exited before owner release was observed")
            return sample
        await asyncio.sleep(GPU_MONITOR_INTERVAL_SECONDS)
    raise TimeoutError("monitor did not release the observed owner process")


async def _checkpoint_on_yield(args: argparse.Namespace, queue: JobQueue, repo: PostgresJobQueueRepository,
                               store: FileCheckpointStore, job_id: str, lease_token: str,
                               work_dir: str, reason: str, container: str) -> Any:
    current = await queue.get(job_id)
    if current["state"] == "running":
        await queue.request_yield(job_id, reason=reason)
    elif current["state"] != "yield_requested":
        raise RuntimeError(f"owned worker cannot checkpoint from queue state {current['state']}")
    control = json.dumps({"reason": reason}, sort_keys=True, separators=(",", ":")).encode()
    _ssh(args, ["tee", f"{work_dir}/control.json"], input_bytes=control)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            payload = _ssh(args, ["cat", f"{work_dir}/pending.json"]).encode()
            break
        except RuntimeError as error:
            if "No such file" not in str(error):
                raise
            await asyncio.sleep(0.25)
    else:
        raise TimeoutError("owned worker did not prepare a checkpoint payload")
    identity = await repo.checkpoint_identity(job_id)
    checkpoint = await queue.save_checkpoint(
        store=store, job_id=job_id, lease_token=lease_token, identity=identity, payload=payload
    )
    pending_events = _events(_ssh(args, ["docker", "logs", container]))
    pending = next(item for item in pending_events if item["event"] == "checkpoint_pending")
    if checkpoint.sha256 != pending["sha256"]:
        raise RuntimeError("controller checkpoint hash differs from the worker's pending bytes")
    ack = json.dumps({"sha256": checkpoint.sha256}, separators=(",", ":")).encode()
    _ssh(args, ["tee", f"{work_dir}/ack.json"], input_bytes=ack)
    await _wait_event(args, container, "checkpoint_acknowledged", timeout=30)
    if (await queue.get(job_id))["checkpoint_sha256"] != checkpoint.sha256:
        raise RuntimeError("checkpoint hash is not durable in the queue row")
    return checkpoint


async def _run(args: argparse.Namespace) -> None:
    source_sha = _source_revision()
    if args.storage_path != "/data" or not _IMAGE_ID.fullmatch(args.training_image_id):
        raise ValueError("live probe requires the pinned /data path and immutable cached image ID")
    database = urlsplit(args.database_url)
    if database.scheme not in {"postgres", "postgresql"} or database.hostname not in {"127.0.0.1", "localhost", "::1"} or database.port is None:
        raise ValueError("the disposable PostgreSQL database must use local loopback and an explicit port")
    run_id = uuid4().hex
    root = f"/data/jayn2u/gods-mlops-task7-probe-{run_id}"
    repo = PostgresJobQueueRepository(database_url=args.database_url)
    queue = JobQueue(repository=repo, sources=DatasetSourceRegistry(database_url=args.database_url))
    observer = UbuntuResourceObserver(
        ssh_target=args.ssh_target, gpu_uuid=args.gpu_uuid, storage_path=args.storage_path,
        ssh_port=args.ssh_port, identity_file=args.ssh_identity_file, known_hosts_file=args.ssh_known_hosts,
    )
    admission = GpuAdmission(repository=repo, queue=queue, expected_host_identity=args.host_identity,
        expected_gpu_uuid=args.gpu_uuid, expected_filesystem_identity=args.filesystem_identity,
        expected_storage_path=args.storage_path, observer=observer)
    monitor = GpuJobMonitor(repository=repo, queue=queue, admission=admission)
    containers: list[str] = []
    job_id: str | None = None
    try:
        await repo.ensure_schema()
        profile = ExecutionProfile(
            model_kind="detr", config_version=f"task7-live-{run_id}", phase="probe", target_phase="training",
            memory_requirement_mib=1024, artifact_reservation_bytes=ARTIFACT_BYTES,
            config={"purpose": "task7-cuda-lifecycle-only", "cuda_bytes": CUDA_BYTES,
                    "model_training_performed": False, "source_sha256": source_sha, "image_id": args.training_image_id},
        )
        await queue.register_profile(profile)
        input_sha = hashlib.sha256(run_id.encode()).hexdigest()
        job_id = await queue.submit_probe(probe_input_id=f"task7-live-{run_id}", input_sha256=input_sha,
            model_kind="detr", config_version=profile.config_version, rerun=True)
        _ssh(args, ["mkdir", "-m", "700", "-p", f"{root}/work/first", f"{root}/work/contender", f"{root}/work/resume"])
        uid, gid = _ssh(args, ["id", "-u"]), _ssh(args, ["id", "-g"])
        image_id = _ssh(args, ["docker", "image", "inspect", "--format", "{{.Id}}", args.training_image])
        if image_id != args.training_image_id:
            raise RuntimeError("cached training image does not match its pinned image ID")
        image_user = _ssh(args, ["docker", "image", "inspect", "--format", "{{.Config.User}}", args.training_image])
        preflight = _ssh(args, ["docker", "run", "--rm", "--pull=never", "--network=none", "--user", f"{uid}:{gid}",
                                *training_image_preflight_command(args.training_image)], timeout=60)
        preflight_event = next(item for item in _events(preflight) if item["event"] == "preflight")

        started = time.monotonic()
        lease, initial_idle, _ = await _admit(repo, admission, args, job_id)
        initial_admission_seconds = time.monotonic() - started
        first_generation = int(lease["fencing_token"])
        first_token = str(lease["lease_token"])
        first = _container(args, run_id=run_id, root=root, uid=uid, gid=gid, name=f"gods-task7-{run_id[:10]}-first",
            gpu_uuid=args.gpu_uuid, command=["worker", f"{root}/work/first", job_id, source_sha, "0", "", "",
            str(CUDA_BYTES), str(args.worker_timeout_seconds)])
        containers.append(first)
        owner, owner_observation, _ = await _confirm_and_bind_owner(queue, monitor, admission, args, job_id, first_token, first)
        status = await monitor.observe(owner_observation)
        if status and status["state"] == "yield_requested":
            with tempfile.TemporaryDirectory(prefix=f"gods-task7-early-{run_id[:8]}-") as failure_root:
                failure_store = FileCheckpointStore(root=failure_root)
                await _checkpoint_on_yield(args, queue, repo, failure_store, job_id, first_token,
                    f"{root}/work/first", "external_gpu_process_started_before_test_contender", first)
                await _monitor_until_release(repo, monitor, admission, args, owner)
            raise RuntimeError("external GPU work arrived before the controlled contender was started")

        contender_id = _container(args, run_id=run_id, root=root, uid=uid, gid=gid, name=f"gods-task7-{run_id[:10]}-contender",
            gpu_uuid=args.gpu_uuid, command=["contender", str(CUDA_BYTES), str(args.contender_duration_seconds)])
        containers.append(contender_id)
        contender_logs = await _wait_event(args, contender_id, "contender", timeout=30)
        contender_event = next(item for item in contender_logs if item["event"] == "contender")
        contender = ProcessIdentity(int(contender_event["pid"]), int(contender_event["start_ticks"]))

        external_deadline = time.monotonic() + 60
        external_sample = None
        yield_started = time.monotonic()
        while time.monotonic() < external_deadline:
            sample = await _observe(admission, args)
            status = await monitor.observe(sample)
            if any((item.pid, item.start_ticks) == (contender.pid, contender.start_ticks) for item in sample.gpu_processes):
                external_sample = sample
            if status and status["state"] == "yield_requested":
                break
            await asyncio.sleep(GPU_MONITOR_INTERVAL_SECONDS)
        else:
            raise TimeoutError("monitor did not request yield after observing the controlled GPU contender")

        with tempfile.TemporaryDirectory(prefix=f"gods-task7-{run_id[:8]}-") as checkpoint_root:
            store = FileCheckpointStore(root=checkpoint_root)
            first_checkpoint = await _checkpoint_on_yield(args, queue, repo, store, job_id, first_token,
                f"{root}/work/first", "external_gpu_process_started", first)
            first_release = await _monitor_until_release(repo, monitor, admission, args, owner, contender)
            if external_sample is None or not any((item.pid, item.start_ticks) == (contender.pid, contender.start_ticks)
                                                  for item in first_release.gpu_processes):
                raise RuntimeError("owner release was not observed while the controlled contender held CUDA")
            first_job = await queue.get(job_id)
            if first_job["checkpoint_sha256"] != first_checkpoint.sha256:
                raise RuntimeError("first checkpoint SHA does not match the durable job row")

            resume_started = time.monotonic()
            resumed_lease, all_free, _ = await _admit(repo, admission, args, job_id)
            resume_admission_seconds = time.monotonic() - resume_started
            if all_free.gpu_processes or all_free.free_mib < initial_idle.free_mib - 64:
                raise RuntimeError("GPU free memory did not return after the owned contenders exited")
            identity = await repo.checkpoint_identity(job_id)
            verified = store.load(job_id, expected_identity=identity)
            if verified is None or verified.sha256 != first_checkpoint.sha256:
                raise RuntimeError("resume did not read back the committed checkpoint bytes")

            previous_payload = base64.b64encode(verified.payload).decode("ascii")
            resume_token = str(resumed_lease["lease_token"])
            resumed = _container(args, run_id=run_id, root=root, uid=uid, gid=gid, name=f"gods-task7-{run_id[:10]}-resume",
                gpu_uuid=args.gpu_uuid, command=["worker", f"{root}/work/resume", job_id, source_sha,
                str(json.loads(verified.payload)["step"]), verified.sha256, previous_payload,
                str(CUDA_BYTES), str(args.worker_timeout_seconds)])
            containers.append(resumed)
            resumed_owner, resumed_observation, resumed_events = await _confirm_and_bind_owner(
                queue, monitor, admission, args, job_id, resume_token, resumed)
            if not any(item["event"] == "allocated" and item["previous_checkpoint_sha256"] == verified.sha256
                       for item in resumed_events):
                raise RuntimeError("resumed worker did not consume the verified prior checkpoint")
            await monitor.observe(resumed_observation)
            second_checkpoint = await _checkpoint_on_yield(args, queue, repo, store, job_id, resume_token,
                f"{root}/work/resume", "task7_resume_verified", resumed)
            second_release = await _monitor_until_release(repo, monitor, admission, args, resumed_owner)
            final_job = await queue.get(job_id)
            if final_job["checkpoint_sha256"] != second_checkpoint.sha256:
                raise RuntimeError("resumed checkpoint SHA does not match the durable job row")
            if await repo.get_active_lease(args.gpu_uuid) is not None:
                raise RuntimeError("a GPU lease remains after both owned workers exited")
            profile_state = await repo.get_profile(phase="probe", model_kind="detr", config_version=profile.config_version)
            if profile_state is None or profile_state["profile_state"] != "candidate":
                raise RuntimeError("a tiny CUDA lifecycle probe incorrectly promoted a training profile")

            result = {
                "event": "task7_live_probe_complete", "source_code_sha256": source_sha, "job_id": job_id,
                "phase": "probe", "model_training_performed": False, "target_profile_promoted": False,
                "host_identity": args.host_identity, "gpu_uuid": args.gpu_uuid,
                "filesystem_identity": args.filesystem_identity, "storage_path": args.storage_path,
                "training_image_id": image_id, "training_image_default_user": image_user,
                "probe_container_user": f"{uid}:{gid}", "worker_uid": owner.uid,
                "worker_torch_version": preflight_event["torch_version"],
                "config_sha256": profile.config_sha256, "input_sha256": input_sha,
                "lease_generations": [first_generation, resumed_lease["fencing_token"]],
                "owner_pid_start": [owner.pid, owner.start_ticks],
                "contender_pid_start": [contender.pid, contender.start_ticks],
                "owner_absent_after_release": all((owner.pid, owner.start_ticks) != (p.pid, p.start_ticks)
                                                   for p in first_release.process_table),
                "owner_absent_from_cuda_after_release": all((owner.pid, owner.start_ticks) != (p.pid, p.start_ticks)
                                                            for p in first_release.gpu_processes),
                "contender_present_during_release": any((contender.pid, contender.start_ticks) == (p.pid, p.start_ticks)
                                                        for p in first_release.gpu_processes),
                "free_mib": {"initial_idle": initial_idle.free_mib, "owner_allocated": owner_observation.free_mib,
                             "owner_released_contender_active": first_release.free_mib, "all_gpu_free": all_free.free_mib,
                             "returned_delta": all_free.free_mib - initial_idle.free_mib},
                "first_checkpoint_sha256": first_checkpoint.sha256, "first_checkpoint_step": json.loads(first_checkpoint.payload)["step"],
                "resumed_from_step": json.loads(verified.payload)["step"], "resumed_checkpoint_sha256": verified.sha256,
                "final_checkpoint_sha256": second_checkpoint.sha256, "final_checkpoint_step": json.loads(second_checkpoint.payload)["step"],
                "final_state": final_job["state"], "final_lease_present": False,
                "timings_seconds": {"initial_idle_admission": round(initial_admission_seconds, 2),
                                    "external_yield_and_release": round(time.monotonic() - yield_started, 2),
                                    "resume_idle_admission": round(resume_admission_seconds, 2)},
            }
            print(json.dumps(result, sort_keys=True))
    finally:
        for container in reversed(containers):
            try:
                _ssh(args, ["docker", "rm", "--force", container])
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
        if job_id is not None:
            try:
                for _ in range(3):
                    if await repo.get_active_lease(args.gpu_uuid) is None:
                        break
                    await monitor.observe(await _observe(admission, args))
                    await asyncio.sleep(GPU_MONITOR_INTERVAL_SECONDS)
            except Exception:  # noqa: BLE001 - reconcile only this task-owned lease.
                pass
        try:
            _ssh(args, ["rm", "-rf", "--", root])
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        await queue.close()
        await repo.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gods_mlops.jobs.live_probe")
    run = parser.add_subparsers(dest="command", required=True).add_parser("run", help="run one owned CUDA lifecycle")
    for key in ("database_url", "ssh_target", "ssh_identity_file", "ssh_known_hosts", "host_identity", "gpu_uuid",
                "filesystem_identity", "training_image", "training_image_id"):
        run.add_argument("--" + key.replace("_", "-"), required=True)
    run.add_argument("--ssh-port", type=int)
    run.add_argument("--storage-path", default="/data")
    run.add_argument("--max-wait-seconds", type=int, default=1800)
    run.add_argument("--worker-timeout-seconds", type=int, default=300)
    run.add_argument("--contender-duration-seconds", type=int, default=25)
    args = parser.parse_args()
    try:
        if args.storage_path != "/data" or not _IMAGE_ID.fullmatch(args.training_image_id):
            raise ValueError("live probe requires /data and an immutable cached image ID")
        asyncio.run(_run(args))
    except Exception as error:  # noqa: BLE001 - concise, non-secret CLI failure.
        print(f"task7 live probe failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
