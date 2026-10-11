#!/usr/bin/env python3
"""Read-only check for leftover K3s-owned container processes after stop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any


def assess_runtime_quiescence(evidence: dict[str, Any]) -> dict[str, Any]:
    failures = []
    if evidence.get("service_active") is not False:
        failures.append({"check": "service_state", "reason": "managed systemd service is active or its state is unknown"})
    if evidence.get("runtime_socket_responsive") is not False:
        failures.append({"check": "runtime_socket_state", "reason": "K3s CRI socket is responsive or its state is unknown"})
    if evidence.get("process_scan_complete") is not True:
        failures.append({"check": "process_scan", "reason": "the host process scan did not complete"})
    if not isinstance(evidence.get("owned_process_count"), int) or evidence["owned_process_count"] != 0:
        failures.append({"check": "owned_runtime_processes_remain", "reason": "K3s-owned processes or Kubernetes pod processes remain"})
    if not isinstance(evidence.get("ambiguous_process_count"), int) or evidence["ambiguous_process_count"] != 0:
        failures.append({"check": "ambiguous_runtime_processes", "reason": "a containerd shim could not be assigned safely to a runtime"})
    return {
        "status": "verified_stopped" if not failures else "pending",
        "failures": failures,
        "evidence": {
            "service_active": evidence.get("service_active"),
            "runtime_socket_responsive": evidence.get("runtime_socket_responsive"),
            "process_scan_complete": evidence.get("process_scan_complete"),
            "owned_process_count": evidence.get("owned_process_count"),
            "ambiguous_process_count": evidence.get("ambiguous_process_count"),
            "process_kinds": sorted(set(evidence.get("process_kinds", []))),
        },
    }


def collect_runtime_evidence(
    *,
    service: str,
    data_dir: str,
    socket_path: str,
    proc_root: str | Path = "/proc",
    current_pid: int | None = None,
) -> dict[str, Any]:
    service_state = subprocess.run(
        ["systemctl", "is-active", service],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if service_state.stdout.strip() == "active":
        service_active: bool | None = True
    elif service_state.stdout.strip() == "inactive":
        service_active = False
    else:
        service_active = None

    socket_state = _socket_responsive(socket_path)
    owned_count = 0
    ambiguous_count = 0
    kinds: list[str] = []
    scan_complete = True
    service_group = f"/system.slice/{service}.service"
    proc_root_path = Path(proc_root)
    current_pid = current_pid or os.getpid()
    try:
        process_dirs = list(proc_root_path.iterdir())
        verifier_chain = _verifier_process_chain(
            proc_root_path,
            current_pid=current_pid,
        )
    except OSError:
        return {
            "service_active": service_active,
            "runtime_socket_responsive": socket_state,
            "process_scan_complete": False,
            "owned_process_count": 0,
            "ambiguous_process_count": 0,
            "process_kinds": [],
        }

    for process_dir in process_dirs:
        if not process_dir.name.isdigit():
            continue
        if int(process_dir.name) in verifier_chain:
            continue
        try:
            comm = (process_dir / "comm").read_text(encoding="utf-8").strip()
            cgroup = (process_dir / "cgroup").read_text(encoding="utf-8")
            arguments = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
            executable = str((process_dir / "exe").readlink())
        except FileNotFoundError:
            continue
        except PermissionError:
            scan_complete = False
            continue
        except OSError:
            scan_complete = False
            continue

        args_lower = arguments.lower()
        cgroup_lower = cgroup.lower()
        process_lower = comm.lower()
        executable_lower = executable.lower().removesuffix(" (deleted)")
        executable_name = executable_lower.rsplit("/", 1)[-1]
        is_docker = (
            "moby" in args_lower
            or "/system.slice/docker.service" in cgroup_lower
            or "/docker/" in cgroup_lower
        )
        if is_docker:
            continue
        service_owned = service_group in cgroup_lower
        data_owned = data_dir in arguments or data_dir in cgroup
        socket_owned = socket_path in arguments or socket_path in cgroup
        kubernetes_pod = "kubepods" in cgroup_lower
        is_k3s = process_lower == "k3s" or executable_name == "k3s"
        is_containerd = "containerd" in process_lower or "containerd" in args_lower or "containerd" in executable_name
        is_shim = "containerd-shim" in process_lower or "containerd-shim" in args_lower or "containerd-shim" in executable_name
        if service_owned or data_owned or socket_owned or kubernetes_pod:
            owned_count += 1
            if kubernetes_pod:
                kinds.append("kubernetes-pod-process")
            elif is_k3s:
                kinds.append("k3s-runtime")
            elif is_shim:
                kinds.append("k3s-containerd-shim")
            elif is_containerd:
                kinds.append("k3s-containerd")
            else:
                kinds.append("unknown-owned-runtime-process")
        elif is_k3s or is_containerd or is_shim:
            ambiguous_count += 1
            kinds.append("unclassified-runtime-process")

    return {
        "service_active": service_active,
        "runtime_socket_responsive": socket_state,
        "process_scan_complete": scan_complete,
        "owned_process_count": owned_count,
        "ambiguous_process_count": ambiguous_count,
        "process_kinds": kinds,
    }


def _verifier_process_chain(proc_root: Path, *, current_pid: int) -> set[int]:
    """Exclude only this probe and its shell wrappers, never unrelated runtimes."""
    excluded = {current_pid}
    visited = {current_pid}
    pid = current_pid
    while True:
        try:
            stat_line = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
            cmdline = (proc_root / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except FileNotFoundError:
            break
        # The executable comm can contain spaces, so parse fields after its final ')'.
        fields = stat_line[stat_line.rfind(")") + 2 :].split()
        if len(fields) < 2:
            raise OSError("cannot determine runtime verifier parent process")
        parent_pid = int(fields[1])
        if parent_pid <= 1 or parent_pid in visited:
            break
        try:
            parent_cmdline = (proc_root / str(parent_pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except FileNotFoundError:
            break
        if "runtime_probe.py" not in parent_cmdline:
            break
        excluded.add(parent_pid)
        visited.add(parent_pid)
        pid = parent_pid
    return excluded


def _socket_responsive(path: str) -> bool | None:
    endpoint = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    endpoint.settimeout(0.25)
    try:
        endpoint.connect(path)
        return True
    except (FileNotFoundError, ConnectionRefusedError):
        return False
    except OSError:
        return None
    finally:
        endpoint.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices={"k3s", "k3s-agent"}, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--socket", default="/run/k3s/containerd/containerd.sock")
    args = parser.parse_args(argv)
    evidence = collect_runtime_evidence(service=args.service, data_dir=args.data_dir, socket_path=args.socket)
    report = assess_runtime_quiescence(evidence)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "verified_stopped" else 1


if __name__ == "__main__":
    raise SystemExit(main())
