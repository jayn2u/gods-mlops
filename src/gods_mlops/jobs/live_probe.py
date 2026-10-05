"""Narrow, owned CUDA lifecycle harness for Task 7 acceptance evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import posixpath
import random
import re
import shlex
import subprocess
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit
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
_SOURCE_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def remote_database_url(database_url: str, forwarded_port: int) -> str:
    """Change only the endpoint so a worker uses a loopback SSH reverse tunnel."""
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or parsed.hostname is None or parsed.port is None:
        raise ValueError("a PostgreSQL URL with an explicit port is required")
    if not 1 <= forwarded_port <= 65_535:
        raise ValueError("SSH tunnel PostgreSQL port is invalid")
    auth = ""
    if parsed.username is not None:
        auth = quote(unquote(parsed.username), safe="")
        if parsed.password is not None:
            auth += ":" + quote(unquote(parsed.password), safe="")
        auth += "@"
    return urlunsplit((parsed.scheme, f"{auth}127.0.0.1:{forwarded_port}", parsed.path, parsed.query, ""))


def proc_start_ticks(stat_record: str) -> int:
    """Read start time from procfs when a process name contains spaces/parens."""
    end_name = stat_record.rfind(")")
    fields = stat_record[end_name + 2 :].split() if end_name >= 0 else []
    if len(fields) <= 19:
        raise ValueError("process stat lacks a start time")
    try:
        result = int(fields[19])
    except ValueError as error:
        raise ValueError("process stat lacks a valid start time") from error
    if result <= 0:
        raise ValueError("process stat lacks a valid start time")
    return result


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
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes", "-o", f"UserKnownHostsFile={args.ssh_known_hosts}", "-i", args.ssh_identity_file]
    if args.ssh_port:
        ssh += ["-p", str(args.ssh_port)]
    result = subprocess.run(
        [*ssh, args.ssh_target, shlex.join(command)],
        check=False,
        capture_output=True,
        input=input_bytes,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace")[-1500:].strip() or "remote command failed")
    return result.stdout.decode("utf-8", "strict").strip()


def _package_archive() -> bytes:
    source_root = Path(__file__).resolve().parents[2]
    bundle = BytesIO()
    with tarfile.open(fileobj=bundle, mode="w:gz") as archive:
        for item in sorted(source_root.rglob("*")):
            if "__pycache__" not in item.parts and item.suffix != ".pyc":
                archive.add(item, arcname=item.relative_to(source_root).as_posix())
    return bundle.getvalue()


def _events(logs: str) -> list[dict[str, Any]]:
    results = []
    for line in logs.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("event"), str):
            results.append(item)
    return results


async def _wait_event(args: argparse.Namespace, container: str, event: str, timeout: int = 90) -> list[dict[str, Any]]:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        logs = _ssh(args, ["docker", "logs", container], timeout=20)
        items = _events(logs)
        if event in {item["event"] for item in items}:
            return items
        state = _ssh(args, ["docker", "inspect", "--format", "{{.State.Status}}", container])
        if state in {"exited", "dead"}:
            raise RuntimeError(f"owned container exited before {event}: {logs[-1000:]}")
        await asyncio.sleep(0.5)
    raise TimeoutError(f"container did not emit {event}")


async def _observe(admission: GpuAdmission, args: argparse.Namespace) -> ResourceObservation:
    value = await admission.observe_once()
    expected = (
        args.host_identity,
        args.gpu_uuid,
        args.filesystem_identity,
        posixpath.normpath(args.storage_path),
    )
    actual = (
        value.host_identity,
        value.gpu_uuid,
        value.filesystem_identity,
        posixpath.normpath(value.storage_path),
    )
    if actual != expected:
        raise RuntimeError("strict observer did not match the pinned host, GPU, filesystem, and /data path")
    return value


async def _admit(
    repository: PostgresJobQueueRepository, admission: GpuAdmission, args: argparse.Namespace, job_id: str
) -> tuple[dict[str, Any], ResourceObservation, ResourceObservation]:
    started = time.monotonic()
    first_idle: ResourceObservation | None = None
    while True:
        sample = await _observe(admission, args)
        if not sample.gpu_processes and first_idle is None:
            first_idle = sample
        result = await admission.admit(job_id, sample.to_dict())
        if result["state"] == "running":
            lease = await repository.get_active_lease(args.gpu_uuid)
            if lease is None or lease["job_id"] != job_id:
                raise RuntimeError("queue returned running without this job's durable GPU lease")
            return lease, first_idle or sample, sample
        if time.monotonic() - started > args.max_wait_seconds:
            raise TimeoutError(f"job remained {result['state']}: {result.get('reason_code')}")
        await asyncio.sleep(GPU_MONITOR_INTERVAL_SECONDS)


async def _monitor_until_release(
    repository: PostgresJobQueueRepository,
    monitor: GpuJobMonitor,
    admission: GpuAdmission,
    args: argparse.Namespace,
    owner: ProcessIdentity,
    contender: ProcessIdentity | None = None,
    timeout: int = 60,
) -> tuple[ResourceObservation, dict[str, Any]]:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        sample = await _observe(admission, args)
        status = await monitor.observe(sample)
        if await repository.get_active_lease(args.gpu_uuid) is None:
            process_table = {(item.pid, item.start_ticks) for item in sample.process_table}
            gpu_processes = {(item.pid, item.start_ticks) for item in sample.gpu_processes}
            key = (owner.pid, owner.start_ticks)
            if key in process_table or key in gpu_processes:
                raise RuntimeError("monitor released the lease before its owner PID/CUDA process disappeared")
            if contender is not None and (contender.pid, contender.start_ticks) not in gpu_processes:
                raise RuntimeError("external contender was not still present when owner release was observed")
            return sample, status or {}
        await asyncio.sleep(GPU_MONITOR_INTERVAL_SECONDS)
    raise TimeoutError("monitor did not release the observed owner process")


async def _worker(args: argparse.Namespace) -> int:
    import torch

    dsn = os.environ["GODS_MLOPS_DATABASE_URL"]
    repo = PostgresJobQueueRepository(database_url=dsn)
    queue = JobQueue(repository=repo, sources=DatasetSourceRegistry(database_url=dsn))
    tensor = None
    try:
        job = await queue.get(args.job_id)
        if job["state"] != "running" or job["lease_token"] != args.lease_token:
            raise RuntimeError("worker lacks the current fenced lease")
        identity = await repo.checkpoint_identity(args.job_id)
        store = FileCheckpointStore(root=args.checkpoint_root)
        previous = store.load(args.job_id, expected_identity=identity)
        if bool(previous) != args.resume or (previous and job["checkpoint_sha256"] != previous.sha256):
            raise RuntimeError("resumed worker did not find the exact durable checkpoint")
        previous_data = json.loads(previous.payload) if previous else {"step": 0}
        owner = ProcessIdentity(os.getpid(), proc_start_ticks(Path("/proc/self/stat").read_text()), os.getuid())
        if not await queue.bind_process(args.job_id, args.lease_token, owner):
            raise RuntimeError("worker PID/start-time identity did not bind to its lease")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("container did not expose exactly the pinned CUDA device")
        torch.cuda.set_device(0)
        tensor = torch.empty(CUDA_BYTES, dtype=torch.uint8, device="cuda:0")
        tensor.fill_(1)
        torch.cuda.synchronize()
        emit = lambda name, **fields: print(json.dumps({"event": name, **fields}, sort_keys=True), flush=True)
        emit("allocated", pid=owner.pid, start_ticks=owner.start_ticks, uid=owner.uid,
             resumed_from_step=int(previous_data["step"]), previous_checkpoint_sha256=previous.sha256 if previous else None)
        await queue.renew_lease(args.job_id, args.lease_token)
        deadline = time.monotonic() + args.max_seconds
        while time.monotonic() < deadline:
            job = await queue.get(args.job_id)
            if job["state"] == "yield_requested":
                payload = json.dumps({"step": int(previous_data["step"]) + 1, "allocation_bytes": CUDA_BYTES,
                                      "model_training_performed": False}, sort_keys=True).encode()
                checkpoint = await queue.save_checkpoint(
                    store=store, job_id=args.job_id, lease_token=args.lease_token, identity=identity, payload=payload
                )
                del tensor
                tensor = None
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                await asyncio.sleep(0.1)
                emit("checkpoint", sha256=checkpoint.sha256, step=int(previous_data["step"]) + 1,
                     allocated_after_release=int(torch.cuda.memory_allocated(0)),
                     reserved_after_release=int(torch.cuda.memory_reserved(0)))
                return 0
            if job["state"] != "running":
                raise RuntimeError(f"worker observed unexpected state {job['state']}")
            await queue.renew_lease(args.job_id, args.lease_token)
            await asyncio.sleep(1)
        raise TimeoutError("worker did not receive a bounded yield request")
    finally:
        if tensor is not None:
            del tensor
        await repo.close()


async def _contender(args: argparse.Namespace) -> int:
    import torch

    owner = ProcessIdentity(os.getpid(), proc_start_ticks(Path("/proc/self/stat").read_text()), os.getuid())
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("container did not expose exactly the pinned CUDA device")
    torch.cuda.set_device(0)
    tensor = torch.empty(CUDA_BYTES, dtype=torch.uint8, device="cuda:0")
    tensor.fill_(2)
    torch.cuda.synchronize()
    print(json.dumps({"event": "contender", "pid": owner.pid, "start_ticks": owner.start_ticks}, sort_keys=True), flush=True)
    await asyncio.sleep(args.duration_seconds)
    del tensor
    torch.cuda.synchronize()
    return 0


async def _run(args: argparse.Namespace) -> None:
    source_sha = _source_revision()
    run_id = uuid4().hex
    root = f"/data/jayn2u/gods-mlops-task7-probe-{run_id}"
    ssh_base = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes",
                "-o", f"UserKnownHostsFile={args.ssh_known_hosts}", "-i", args.ssh_identity_file]
    if args.ssh_port:
        ssh_base += ["-p", str(args.ssh_port)]
    if args.storage_path != "/data" or not _IMAGE_ID.fullmatch(args.training_image_id):
        raise ValueError("live probe requires the pinned /data filesystem and immutable cached image ID")
    dsn = urlsplit(args.database_url)
    if dsn.hostname not in {"127.0.0.1", "localhost", "::1"} or dsn.port is None:
        raise ValueError("the disposable PostgreSQL endpoint must be on local loopback with an explicit port")

    remote = lambda cmd, **kwargs: _ssh(args, cmd, **kwargs)
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
    tunnel: asyncio.subprocess.Process | None = None
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

        remote(["mkdir", "-m", "700", "-p", f"{root}/source", f"{root}/work"])
        archive = _package_archive()
        remote(["tar", "-xzf", "-", "-C", f"{root}/source"], input_bytes=archive, timeout=60)
        uid, gid = remote(["id", "-u"]), remote(["id", "-g"])
        image_id = remote(["docker", "image", "inspect", "--format", "{{.Id}}", args.training_image])
        if image_id != args.training_image_id:
            raise RuntimeError("cached training image does not match its pinned ID")
        remote(["docker", "run", "--rm", "--pull=never", "--network=host", "--user", f"{uid}:{gid}",
                "--mount", f"type=bind,source={root}/source,target=/probe/source,readonly", "--env", "PYTHONPATH=/probe/source",
                args.training_image, "python", "-c", "import asyncpg, torch"], timeout=60)

        forwarded_port = random.randint(51_000, 61_000)
        tunnel = await asyncio.create_subprocess_exec(*ssh_base, "-N", "-o", "ExitOnForwardFailure=yes", "-R",
            f"127.0.0.1:{forwarded_port}:127.0.0.1:{dsn.port}", args.ssh_target,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await asyncio.sleep(0.5)
        if tunnel.returncode is not None:
            raise RuntimeError("loopback-only reverse SSH tunnel to the disposable database failed")
        worker_dsn = remote_database_url(args.database_url, forwarded_port)

        started = time.monotonic()
        lease, initial_idle, _ = await _admit(repo, admission, args, job_id)
        first_lease_seconds = time.monotonic() - started
        owner_id = str(lease["lease_token"])
        first_generation = int(lease["fencing_token"])
        base_command = ["python", "-m", "gods_mlops.jobs.live_probe"]
        first_worker = _ssh_container(args, run_id, root, uid, gid, args.training_image, args.gpu_uuid,
            f"gods-task7-{run_id[:10]}-first", [*base_command, "worker", "--job-id", job_id, "--lease-token", owner_id,
            "--checkpoint-root", f"{root}/work/checkpoints", "--max-seconds", str(args.worker_timeout_seconds)], worker_dsn)
        containers.append(first_worker)
        first_events = await _wait_event(args, first_worker, "allocated")
        started_event = next(item for item in first_events if item["event"] == "allocated")
        owner = ProcessIdentity(int(started_event["pid"]), int(started_event["start_ticks"]), int(started_event["uid"]))

        owner_observation = None
        for _ in range(12):
            sample = await _observe(admission, args)
            status = await monitor.observe(sample)
            gpu = {(p.pid, p.start_ticks) for p in sample.gpu_processes}
            table = {(p.pid, p.start_ticks) for p in sample.process_table}
            if (owner.pid, owner.start_ticks) in gpu & table:
                owner_observation = sample
                break
            if status and status["state"] == "yield_requested":
                raise RuntimeError("external CUDA work began before the owned worker was confirmed")
            await asyncio.sleep(5)
        if owner_observation is None:
            raise TimeoutError("Ubuntu observer did not confirm the owned process on the GPU")

        contender_id = _ssh_container(args, run_id, root, uid, gid, args.training_image, args.gpu_uuid,
            f"gods-task7-{run_id[:10]}-contender", [*base_command, "contender", "--duration-seconds",
            str(args.contender_duration_seconds)], None)
        containers.append(contender_id)
        contender_events = await _wait_event(args, contender_id, "contender", timeout=30)
        contender_event = next(item for item in contender_events if item["event"] == "contender")
        contender = ProcessIdentity(int(contender_event["pid"]), int(contender_event["start_ticks"]))

        external_seen = None
        yield_started = time.monotonic()
        for _ in range(12):
            sample = await _observe(admission, args)
            status = await monitor.observe(sample)
            if any((p.pid, p.start_ticks) == (contender.pid, contender.start_ticks) for p in sample.gpu_processes):
                external_seen = sample
            if status and status["state"] == "yield_requested":
                break
            await asyncio.sleep(5)
        else:
            raise TimeoutError("monitor did not request yield after observing the owned contender")

        first_release, _ = await _monitor_until_release(repo, monitor, admission, args, owner, contender)
        owner_release_seconds = time.monotonic() - yield_started
        first_events = _events(remote(["docker", "logs", first_worker]))
        checkpoint = next((item for item in first_events if item["event"] == "checkpoint"), None)
        job = await queue.get(job_id)
        if checkpoint is None or job["checkpoint_sha256"] != checkpoint["sha256"]:
            raise RuntimeError("first atomic checkpoint hash was not durable in the queue")
        if not any((p.pid, p.start_ticks) == (contender.pid, contender.start_ticks) for p in first_release.gpu_processes):
            raise RuntimeError("owner release observation did not retain the controlled contender PID")

        resume_started = time.monotonic()
        resumed_lease, all_free, _ = await _admit(repo, admission, args, job_id)
        resume_wait_seconds = time.monotonic() - resume_started
        if all_free.gpu_processes or all_free.free_mib < initial_idle.free_mib - 64:
            raise RuntimeError("GPU did not return idle memory after both controlled containers exited")
        second_worker = _ssh_container(args, run_id, root, uid, gid, args.training_image, args.gpu_uuid,
            f"gods-task7-{run_id[:10]}-resume", [*base_command, "worker", "--job-id", job_id,
            "--lease-token", str(resumed_lease["lease_token"]), "--checkpoint-root", f"{root}/work/checkpoints",
            "--max-seconds", str(args.worker_timeout_seconds), "--resume"], worker_dsn)
        containers.append(second_worker)
        resumed_events = await _wait_event(args, second_worker, "allocated")
        resumed_event = next(item for item in resumed_events if item["event"] == "allocated")
        if resumed_event["resumed_from_step"] != checkpoint["step"] or resumed_event["previous_checkpoint_sha256"] != checkpoint["sha256"]:
            raise RuntimeError("same job did not resume the exact committed checkpoint")
        await queue.request_yield(job_id, reason="task7_probe_resume_verified")
        resumed_owner = ProcessIdentity(int(resumed_event["pid"]), int(resumed_event["start_ticks"]), int(resumed_event["uid"]))
        final_release, _ = await _monitor_until_release(repo, monitor, admission, args, resumed_owner)
        resumed_events = _events(remote(["docker", "logs", second_worker]))
        final_checkpoint = next((item for item in resumed_events if item["event"] == "checkpoint"), None)
        final_job = await queue.get(job_id)
        if final_checkpoint is None or final_job["checkpoint_sha256"] != final_checkpoint["sha256"]:
            raise RuntimeError("resumed worker failed to commit its next fenced checkpoint")
        if await repo.get_active_lease(args.gpu_uuid) is not None:
            raise RuntimeError("a GPU lease remains after the controlled worker exited")
        if (await repo.get_profile(phase="probe", model_kind="detr", config_version=profile.config_version))["profile_state"] != "candidate":
            raise RuntimeError("tiny CUDA allocation incorrectly promoted a real training profile")

        print(json.dumps({
            "event": "task7_live_probe_complete", "source_code_sha256": source_sha, "job_id": job_id,
            "phase": "probe", "model_training_performed": False, "target_profile_promoted": False,
            "host_identity": args.host_identity, "gpu_uuid": args.gpu_uuid,
            "filesystem_identity": args.filesystem_identity, "storage_path": args.storage_path,
            "training_image_id": image_id, "config_sha256": profile.config_sha256, "input_sha256": input_sha,
            "lease_generations": [first_generation, resumed_lease["fencing_token"]],
            "owner_pid_start": [owner.pid, owner.start_ticks], "contender_pid_start": [contender.pid, contender.start_ticks],
            "owner_absent_after_release": all((owner.pid, owner.start_ticks) != (p.pid, p.start_ticks) for p in first_release.process_table),
            "owner_absent_from_cuda_after_release": all((owner.pid, owner.start_ticks) != (p.pid, p.start_ticks) for p in first_release.gpu_processes),
            "contender_present_during_release": any((contender.pid, contender.start_ticks) == (p.pid, p.start_ticks) for p in first_release.gpu_processes),
            "free_mib": {"initial_idle": initial_idle.free_mib, "owner_allocated": owner_observation.free_mib,
                         "owner_released_contender_active": first_release.free_mib, "all_gpu_free": all_free.free_mib},
            "first_checkpoint_sha256": checkpoint["sha256"], "first_checkpoint_step": checkpoint["step"],
            "resumed_from_step": resumed_event["resumed_from_step"], "resumed_checkpoint_sha256": resumed_event["previous_checkpoint_sha256"],
            "final_checkpoint_sha256": final_checkpoint["sha256"], "final_checkpoint_step": final_checkpoint["step"],
            "final_state": final_job["state"], "final_lease_present": False,
            "timings_seconds": {"initial_idle_admission": round(first_lease_seconds, 2),
                                "external_yield_release": round(owner_release_seconds, 2),
                                "resume_idle_admission": round(resume_wait_seconds, 2)},
        }, sort_keys=True))
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
                    await asyncio.sleep(5)
            except Exception:  # noqa: BLE001 - cleanup only reconciles the probe's own fenced lease.
                pass
        if tunnel and tunnel.returncode is None:
            tunnel.terminate()
            try:
                await asyncio.wait_for(tunnel.wait(), 5)
            except asyncio.TimeoutError:
                tunnel.kill()
                await tunnel.wait()
        try:
            _ssh(args, ["rm", "-rf", "--", root])
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        await queue.close()
        await repo.close()


def _ssh_container(args: argparse.Namespace, run_id: str, root: str, uid: str, gid: str,
                   image: str, gpu_uuid: str, name: str, command: list[str], dsn: str | None) -> str:
    parts = ["docker", "run", "--detach", "--pull=never", "--name", name, "--label",
             "gods-mlops.task=task7-live-probe", "--label", f"gods-mlops.probe={run_id}",
             "--gpus", f"device={gpu_uuid}", "--pid=host", "--network=host", "--user", f"{uid}:{gid}",
             "--mount", f"type=bind,source={root}/source,target=/probe/source,readonly", "--mount",
             f"type=bind,source={root}/work,target={root}/work", "--env", "PYTHONPATH=/probe/source"]
    if dsn:
        parts += ["--env", f"GODS_MLOPS_DATABASE_URL={dsn}"]
    output = _ssh(args, [*parts, image, *command], timeout=60).splitlines()[-1]
    if not re.fullmatch(r"[0-9a-f]{12,64}", output):
        raise RuntimeError("Docker did not return a container ID")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m gods_mlops.jobs.live_probe")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one owned queue/checkpoint/resume GPU lifecycle")
    for key in ("database_url", "ssh_target", "ssh_identity_file", "ssh_known_hosts", "host_identity", "gpu_uuid", "filesystem_identity", "training_image", "training_image_id"):
        run.add_argument("--" + key.replace("_", "-"), required=True)
    run.add_argument("--ssh-port", type=int)
    run.add_argument("--storage-path", default="/data")
    run.add_argument("--max-wait-seconds", type=int, default=1800)
    run.add_argument("--worker-timeout-seconds", type=int, default=300)
    run.add_argument("--contender-duration-seconds", type=int, default=25)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--lease-token", required=True)
    worker.add_argument("--checkpoint-root", required=True)
    worker.add_argument("--max-seconds", type=int, default=300)
    worker.add_argument("--resume", action="store_true")
    contender = commands.add_parser("contender", help=argparse.SUPPRESS)
    contender.add_argument("--duration-seconds", type=int, default=25)
    args = parser.parse_args()
    try:
        if args.command == "run":
            asyncio.run(_run(args))
        elif args.command == "worker":
            asyncio.run(_worker(args))
        else:
            asyncio.run(_contender(args))
    except Exception as error:  # noqa: BLE001 - CLI boundary reports a concise failure.
        print(f"task7 live probe failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
