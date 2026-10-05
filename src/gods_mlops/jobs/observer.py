"""Read-only Ubuntu GPU/filesystem producer and durable five-second observer loop."""

from __future__ import annotations

import asyncio
import json
import os
import posixpath
import re
import shlex
import subprocess
from typing import Any

from .admission import GPU_MONITOR_INTERVAL_SECONDS
from .models import ResourceObservation
from .queue import (
    ObservationReplayError,
    PostgresJobQueueRepository,
    ResourceObservationRejectedError,
)

_REMOTE_PROGRAM = r"""
import datetime, hashlib, json, os, pathlib, subprocess, sys, uuid

settings = json.load(sys.stdin)
storage_path = settings["storage_path"]
gpu_uuid = settings["gpu_uuid"]
node_id = settings["node_id"]

def run(command):
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    return result.stdout.strip()

def identity_for(pid):
    proc = pathlib.Path("/proc") / str(pid)
    raw = (proc / "stat").read_text(encoding="ascii")
    fields = raw[raw.rfind(")") + 2:].split()
    if len(fields) <= 19:
        raise ValueError("process stat lacks kernel start time")
    status = (proc / "status").read_text(encoding="ascii")
    uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
    uid = int(uid_line.split()[1])
    return {"pid": int(pid), "start_ticks": int(fields[19]), "uid": uid}

machine_id = pathlib.Path("/etc/machine-id").read_text(encoding="ascii").strip()
host_identity = "machine-sha256:" + hashlib.sha256(machine_id.encode("ascii")).hexdigest()
gpu_line = run([
    "nvidia-smi", "--id=" + gpu_uuid,
    "--query-gpu=name,uuid,memory.free,memory.total",
    "--format=csv,noheader,nounits",
])
gpu_fields = [field.strip() for field in gpu_line.split(",")]
if len(gpu_fields) != 4 or gpu_fields[1] != gpu_uuid:
    raise ValueError("expected Ubuntu GPU is unavailable")
compute_output = run([
    "nvidia-smi", "--id=" + gpu_uuid,
    "--query-compute-apps=gpu_uuid,pid,used_memory",
    "--format=csv,noheader,nounits",
])
gpu_processes = []
for line in compute_output.splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) == 3 and fields[0] == gpu_uuid:
        gpu_processes.append(identity_for(int(fields[1])))

process_table = []
process_table_complete = True
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    try:
        process_table.append(identity_for(int(entry.name)))
    except FileNotFoundError:
        # A process exited while the complete procfs listing was being read.
        continue
    except (PermissionError, ProcessLookupError, ValueError):
        process_table_complete = False

stat = os.stat(storage_path)
space = os.statvfs(storage_path)
mount_info = json.loads(run([
    "findmnt", "--json", "--output", "SOURCE,FSTYPE,UUID,TARGET", "--target", storage_path
]))["filesystems"]
if len(mount_info) != 1 or not mount_info[0].get("uuid"):
    raise ValueError("configured storage path has no stable filesystem UUID")
filesystem = mount_info[0]
filesystem_identity = str(filesystem["fstype"]) + ":uuid=" + str(filesystem["uuid"])
observation = {
    "observation_id": str(uuid.uuid4()),
    "node_id": node_id,
    "hostname": os.uname().nodename.split(".")[0],
    "host_identity": host_identity,
    "gpu_name": gpu_fields[0],
    "gpu_uuid": gpu_fields[1],
    "free_mib": int(gpu_fields[2]),
    "total_mib": int(gpu_fields[3]),
    "gpu_processes": gpu_processes,
    "gpu_process_list_complete": True,
    "process_table": process_table,
    "process_table_complete": process_table_complete,
    "storage_path": storage_path,
    "filesystem_identity": filesystem_identity,
    "filesystem_available_bytes": int(space.f_bavail * space.f_frsize),
    "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
print(json.dumps(observation, separators=(",", ":"), sort_keys=True))
"""


class UbuntuObservationUnavailableError(RuntimeError):
    """SSH or a required Ubuntu read-only sensor could not produce a fresh sample."""


class UbuntuResourceObserver:
    """Poll exactly the configured Ubuntu GPU and storage path through trusted SSH."""

    def __init__(
        self,
        *,
        node_id: str,
        host_identity: str,
        ssh_target: str,
        gpu_uuid: str,
        filesystem_identity: str,
        storage_path: str,
        ssh_port: int | None = None,
        identity_file: str | None = None,
        known_hosts_file: str | None = None,
        timeout_seconds: int = 10,
    ) -> None:
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", node_id)
            or not host_identity.startswith("machine-sha256:")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@:-]{0,254}", ssh_target)
            or not gpu_uuid.startswith("GPU-")
            or ":uuid=" not in filesystem_identity
            or not storage_path.startswith("/")
        ):
            raise ValueError("observer requires pinned node, host, GPU, filesystem, and storage identities")
        if not 1 <= timeout_seconds <= 30:
            raise ValueError("observer timeout must be between 1 and 30 seconds")
        if ssh_port is not None and not 1 <= ssh_port <= 65_535:
            raise ValueError("observer SSH port is invalid")
        self._node_id = node_id
        self._host_identity = host_identity
        self._ssh_target = ssh_target
        self._gpu_uuid = gpu_uuid
        self._filesystem_identity = filesystem_identity
        self._storage_path = posixpath.normpath(storage_path)
        self._ssh_port = ssh_port
        self._identity_file = identity_file
        self._known_hosts_file = known_hosts_file
        self._timeout_seconds = timeout_seconds

    def observe(self) -> ResourceObservation:
        command = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={min(self._timeout_seconds, 10)}",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if self._identity_file:
            command.extend(["-i", self._identity_file])
        if self._known_hosts_file:
            command.extend(["-o", f"UserKnownHostsFile={self._known_hosts_file}"])
        if self._ssh_port is not None:
            command.extend(["-p", str(self._ssh_port)])
        command.extend([self._ssh_target, "python3 -c " + shlex.quote(_REMOTE_PROGRAM)])
        request = json.dumps(
            {
                "node_id": self._node_id,
                "gpu_uuid": self._gpu_uuid,
                "storage_path": self._storage_path,
            },
            separators=(",", ":"),
        )
        try:
            result = subprocess.run(
                command,
                input=request,
                capture_output=True,
                text=True,
                check=False,
                timeout=self._timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise UbuntuObservationUnavailableError("Ubuntu observer SSH command failed") from error
        if result.returncode != 0:
            raise UbuntuObservationUnavailableError("Ubuntu observer did not return a complete observation")
        try:
            observation = ResourceObservation.from_dict(json.loads(result.stdout))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OverflowError) as error:
            raise UbuntuObservationUnavailableError("Ubuntu observer returned invalid observation JSON") from error
        identity = (
            observation.node_id,
            observation.host_identity,
            observation.gpu_uuid,
            observation.filesystem_identity,
            posixpath.normpath(observation.storage_path),
        )
        expected = (
            self._node_id,
            self._host_identity,
            self._gpu_uuid,
            self._filesystem_identity,
            self._storage_path,
        )
        if identity != expected:
            raise UbuntuObservationUnavailableError("Ubuntu observer returned the wrong resource identity")
        return observation


async def run_observer_forever(
    *,
    repository: PostgresJobQueueRepository,
    observer: UbuntuResourceObserver,
    interval_seconds: int = GPU_MONITOR_INTERVAL_SECONDS,
) -> None:
    if interval_seconds != GPU_MONITOR_INTERVAL_SECONDS:
        raise ValueError("Ubuntu resource observation cadence is fixed at five seconds")
    await repository.ensure_schema()
    while True:
        try:
            observation = await asyncio.to_thread(observer.observe)
            await repository.record_observation(observation)
        except (ObservationReplayError, ResourceObservationRejectedError):
            # Persistence already reset the durable idle window in the rejecting transaction.
            pass
        except Exception:  # noqa: BLE001 - leave latest good data stale and fail closed
            await repository.record_observation_failure(
                node_id=repository.expected_node_id,
                failure_code="observer_unreachable",
            )
        await asyncio.sleep(interval_seconds)


async def observe_once_and_persist(
    *,
    repository: PostgresJobQueueRepository,
    observer: UbuntuResourceObserver,
) -> ResourceObservation:
    """One inspectable observation for tests and controlled runtime probes."""
    await repository.ensure_schema()
    try:
        observation = await asyncio.to_thread(observer.observe)
        await repository.record_observation(observation)
        return observation
    except (ObservationReplayError, ResourceObservationRejectedError):
        raise
    except Exception:
        await repository.record_observation_failure(
            node_id=repository.expected_node_id,
            failure_code="observer_unreachable",
        )
        raise


def main() -> None:
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
    asyncio.run(_run_and_close(repository, observer))


async def _run_and_close(
    repository: PostgresJobQueueRepository,
    observer: UbuntuResourceObserver,
) -> None:
    try:
        await run_observer_forever(repository=repository, observer=observer)
    finally:
        await repository.close()


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise ValueError(f"required setting is missing: {name}")
    return value


if __name__ == "__main__":
    main()
