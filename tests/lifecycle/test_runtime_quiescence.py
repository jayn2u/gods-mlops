from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from gods_mlops.lifecycle.recovery import (
    active_workload_pods,
    build_daemonset_snapshot,
    inspect_k3s_runtime_state,
    owned_daemonsets,
    validate_daemonset_snapshot,
    validate_emptydir_policy,
    verify_restored_daemonsets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PROBE_PATH = REPO_ROOT / "infra" / "ansible" / "runtime_probe.py"
SPEC = importlib.util.spec_from_file_location("gods_runtime_probe", RUNTIME_PROBE_PATH)
assert SPEC and SPEC.loader
RUNTIME_PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME_PROBE)


def test_daemonset_pods_are_counted_as_active_owned_workloads() -> None:
    pods = {
        "items": [
            {
                "metadata": {
                    "namespace": "nvidia-device-plugin",
                    "name": "nvidia-device-plugin-ds-abc",
                    "ownerReferences": [{"kind": "DaemonSet", "name": "nvidia-device-plugin-ds"}],
                },
                "spec": {"nodeName": "ubuntu"},
                "status": {"phase": "Running"},
            },
            {
                "metadata": {
                    "namespace": "kube-system",
                    "name": "kube-apiserver-vis-lab",
                    "annotations": {"kubernetes.io/config.mirror": "mirror"},
                },
                "spec": {"nodeName": "vis-lab"},
                "status": {"phase": "Running"},
            },
            {
                "metadata": {"namespace": "gods-mlops", "name": "completed-job"},
                "spec": {"nodeName": "ubuntu"},
                "status": {"phase": "Succeeded"},
            },
        ]
    }

    remaining = active_workload_pods(pods)

    assert remaining == [
        {
            "namespace": "nvidia-device-plugin",
            "name": "nvidia-device-plugin-ds-abc",
            "phase": "Running",
            "node": "ubuntu",
            "owner_kind": "DaemonSet",
        }
    ]


def test_only_known_owned_daemonsets_can_be_removed_for_reclaim() -> None:
    known = {
        "items": [
            {
                "metadata": {
                    "namespace": "kube-system",
                    "name": "istio-cni-node",
                    "labels": {"app.kubernetes.io/name": "istio-cni"},
                }
            },
            {
                "metadata": {
                    "namespace": "nvidia-device-plugin",
                    "name": "nvidia-device-plugin",
                    "labels": {"app.kubernetes.io/instance": "nvidia-device-plugin"},
                }
            },
        ]
    }

    result = owned_daemonsets(known)

    assert result["status"] == "verified"
    assert result["delete"] == [
        {"namespace": "kube-system", "name": "istio-cni-node"},
        {"namespace": "nvidia-device-plugin", "name": "nvidia-device-plugin"},
    ]
    with pytest.raises(ValueError, match="unrecognized DaemonSet"):
        owned_daemonsets({"items": [{"metadata": {"namespace": "other", "name": "unknown"}}]})


def test_daemonset_snapshot_strips_server_metadata_and_requires_plan_and_digest_confirmation() -> None:
    document = {
        "items": [
            {
                "apiVersion": "apps/v1",
                "kind": "DaemonSet",
                "metadata": {
                    "namespace": "kube-system",
                    "name": "istio-cni-node",
                    "uid": "server-uid",
                    "resourceVersion": "10",
                    "managedFields": [{"manager": "apiserver"}],
                    "labels": {"app.kubernetes.io/name": "istio-cni"},
                },
                "spec": {"selector": {"matchLabels": {"k8s-app": "istio-cni-node"}}, "template": {"spec": {"containers": [{"name": "install-cni"}]}}},
                "status": {"numberReady": 2},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "DaemonSet",
                "metadata": {
                    "namespace": "nvidia-device-plugin",
                    "name": "nvidia-device-plugin",
                    "labels": {"app.kubernetes.io/instance": "nvidia-device-plugin"},
                },
                "spec": {"selector": {"matchLabels": {"app": "nvidia-device-plugin"}}, "template": {"spec": {"containers": [{"name": "plugin"}]}}},
                "status": {"numberReady": 2},
            },
        ]
    }

    snapshot = build_daemonset_snapshot(document, plan_id="a" * 64)
    summary = validate_daemonset_snapshot(
        snapshot,
        confirmation=snapshot["snapshot_sha256"],
        expected_plan_id="a" * 64,
    )

    assert summary["status"] == "snapshot_valid"
    assert len(snapshot["objects"]) == 2
    assert all("status" not in item for item in snapshot["objects"])
    assert "uid" not in snapshot["objects"][0]["metadata"]
    with pytest.raises(ValueError, match="different reclaim plan"):
        validate_daemonset_snapshot(snapshot, confirmation=snapshot["snapshot_sha256"], expected_plan_id="b" * 64)
    with pytest.raises(ValueError, match="confirmation digest"):
        validate_daemonset_snapshot(snapshot, confirmation="0" * 64, expected_plan_id="a" * 64)


def test_restored_daemonsets_must_match_the_captured_objects() -> None:
    live = {
        "items": [
            {
                "apiVersion": "apps/v1",
                "kind": "DaemonSet",
                "metadata": {"namespace": "kube-system", "name": "istio-cni-node", "labels": {"app.kubernetes.io/name": "istio-cni"}},
                "spec": {"selector": {"matchLabels": {"k8s-app": "istio-cni-node"}}, "template": {"spec": {"containers": [{"name": "install-cni"}]}}},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "DaemonSet",
                "metadata": {"namespace": "nvidia-device-plugin", "name": "nvidia-device-plugin", "labels": {"app.kubernetes.io/instance": "nvidia-device-plugin"}},
                "spec": {"selector": {"matchLabels": {"app": "nvidia-device-plugin"}}, "template": {"spec": {"containers": [{"name": "plugin"}]}}},
            },
        ]
    }
    snapshot = build_daemonset_snapshot(live, plan_id="c" * 64)

    assert verify_restored_daemonsets(snapshot["objects"], live)["status"] == "verified"
    modified = {"items": [dict(item) for item in live["items"]]}
    modified["items"][0] = {**modified["items"][0], "spec": {"selector": {}, "template": {}}}
    with pytest.raises(ValueError, match="do not match"):
        verify_restored_daemonsets(snapshot["objects"], modified)


def test_runtime_probe_marks_remaining_daemonset_container_processes_pending() -> None:
    report = RUNTIME_PROBE.assess_runtime_quiescence({
        "service_active": False,
        "runtime_socket_responsive": False,
        "process_scan_complete": True,
        "owned_process_count": 1,
        "ambiguous_process_count": 0,
        "process_kinds": ["kubernetes-pod-process"],
    })

    assert report["status"] == "pending"
    assert "owned_runtime_processes_remain" in {item["check"] for item in report["failures"]}


def test_runtime_probe_accepts_stopped_service_with_absent_socket_and_no_owned_processes() -> None:
    report = RUNTIME_PROBE.assess_runtime_quiescence({
        "service_active": False,
        "runtime_socket_responsive": False,
        "process_scan_complete": True,
        "owned_process_count": 0,
        "ambiguous_process_count": 0,
        "process_kinds": [],
    })

    assert report["status"] == "verified_stopped"
    assert report["failures"] == []


def test_runtime_probe_fails_closed_when_socket_is_unresponsive_but_state_is_unknown() -> None:
    report = RUNTIME_PROBE.assess_runtime_quiescence({
        "service_active": False,
        "runtime_socket_responsive": None,
        "process_scan_complete": False,
        "owned_process_count": 0,
        "ambiguous_process_count": 1,
        "process_kinds": ["unclassified-containerd-shim"],
    })

    assert report["status"] == "pending"
    assert {item["check"] for item in report["failures"]} >= {
        "runtime_socket_state",
        "process_scan",
        "ambiguous_runtime_processes",
    }


def test_runtime_collector_excludes_its_own_verifier_chain_but_counts_real_k3s_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    _patch_proc_root(monkeypatch, proc_root)
    verifier_pid = os.getpid()
    _write_proc_process(
        proc_root,
        verifier_pid,
        comm="python3",
        cmdline="python3 /tmp/ansible/runtime_probe.py --service k3s-agent --data-dir /srv/gods/k3s --socket /run/k3s.sock",
        cgroup="0::/user.slice/session.scope",
        ppid=99,
    )
    _write_proc_process(
        proc_root,
        99,
        comm="sh",
        cmdline="sh -c /tmp/ansible/runtime_probe.py --service k3s-agent --data-dir /srv/gods/k3s --socket /run/k3s.sock",
        cgroup="0::/user.slice/session.scope",
        ppid=1,
    )
    _write_proc_process(
        proc_root,
        200,
        comm="k3s",
        cmdline="/usr/local/bin/k3s agent --data-dir /srv/gods/k3s",
        cgroup="0::/system.slice/k3s-agent.service",
        ppid=1,
    )
    monkeypatch.setattr(RUNTIME_PROBE.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"stdout": "inactive\n"})())
    monkeypatch.setattr(RUNTIME_PROBE, "_socket_responsive", lambda _path: False)

    evidence = RUNTIME_PROBE.collect_runtime_evidence(
        service="k3s-agent",
        data_dir="/srv/gods/k3s",
        socket_path="/run/k3s.sock",
    )

    assert evidence["process_scan_complete"] is True
    assert evidence["owned_process_count"] == 1
    assert evidence["process_kinds"] == ["k3s-runtime"]
    assert RUNTIME_PROBE.assess_runtime_quiescence(evidence)["status"] == "pending"


def test_runtime_collector_fails_closed_on_unknown_process_in_managed_cgroup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    _patch_proc_root(monkeypatch, proc_root)
    _write_proc_process(
        proc_root,
        300,
        comm="future-runtime-helper",
        cmdline="future-runtime-helper --serve",
        cgroup="0::/system.slice/k3s.service",
        ppid=1,
    )
    monkeypatch.setattr(RUNTIME_PROBE.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"stdout": "inactive\n"})())
    monkeypatch.setattr(RUNTIME_PROBE, "_socket_responsive", lambda _path: False)

    evidence = RUNTIME_PROBE.collect_runtime_evidence(
        service="k3s",
        data_dir="/srv/gods/k3s",
        socket_path="/run/k3s.sock",
    )

    assert evidence["owned_process_count"] == 1
    assert evidence["process_kinds"] == ["unknown-owned-runtime-process"]
    assert RUNTIME_PROBE.assess_runtime_quiescence(evidence)["status"] == "pending"


def test_runtime_collector_treats_failed_service_state_as_unknown_not_quiescent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    _patch_proc_root(monkeypatch, proc_root)
    monkeypatch.setattr(RUNTIME_PROBE.subprocess, "run", lambda *args, **kwargs: type("Result", (), {"stdout": "failed\n"})())
    monkeypatch.setattr(RUNTIME_PROBE, "_socket_responsive", lambda _path: False)

    evidence = RUNTIME_PROBE.collect_runtime_evidence(
        service="k3s",
        data_dir="/srv/gods/k3s",
        socket_path="/run/k3s.sock",
    )

    assert evidence["service_active"] is None
    assert RUNTIME_PROBE.assess_runtime_quiescence(evidence)["status"] == "pending"


def _write_proc_process(
    proc_root: Path,
    pid: int,
    *,
    comm: str,
    cmdline: str,
    cgroup: str,
    ppid: int,
) -> None:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "comm").write_text(f"{comm}\n", encoding="utf-8")
    (process / "cmdline").write_bytes(cmdline.encode("utf-8").replace(b" ", b"\0"))
    (process / "cgroup").write_text(f"{cgroup}\n", encoding="utf-8")
    (process / "stat").write_text(f"{pid} ({comm}) S {ppid} 0 0\n", encoding="utf-8")
    (process / "exe").symlink_to(f"/usr/bin/{comm}")


def _patch_proc_root(monkeypatch: pytest.MonkeyPatch, proc_root: Path) -> None:
    class ProcPath:
        def __init__(self, value: str | Path) -> None:
            path = Path(value)
            self.path = proc_root / path.relative_to("/proc") if path.is_absolute() and path.parts[:2] == ("/", "proc") else path

        @property
        def name(self) -> str:
            return self.path.name

        def iterdir(self) -> list[ProcPath]:
            return [ProcPath(path) for path in self.path.iterdir()]

        def __truediv__(self, child: str) -> ProcPath:
            return ProcPath(self.path / child)

        def read_text(self, **kwargs: object) -> str:
            return self.path.read_text(**kwargs)

        def read_bytes(self) -> bytes:
            return self.path.read_bytes()

        def readlink(self) -> str:
            return self.path.readlink().as_posix()

    monkeypatch.setattr(RUNTIME_PROBE, "Path", ProcPath)


def test_cold_server_state_hashes_sqlite_token_tls_credentials_and_manifests(tmp_path: Path) -> None:
    root = tmp_path / "k3s"
    for relative in (
        "server/db",
        "server/tls",
        "server/cred",
        "server/manifests",
    ):
        (root / relative).mkdir(parents=True)
    (root / "server/db/state.db").write_bytes(b"sqlite datastore")
    (root / "server/token").write_bytes(b"server token")
    (root / "server/tls/server-ca.crt").write_bytes(b"ca")
    (root / "server/cred/admin.kubeconfig").write_bytes(b"private config")
    (root / "server/manifests/coredns.yaml").write_bytes(b"manifest")
    (root / "agent/containerd").mkdir(parents=True)
    (root / "agent/containerd/cache").write_bytes(b"excluded server runtime cache")
    (root / "agent/etc").mkdir()
    (root / "agent/etc/config.toml").write_bytes(b"agent config")

    state = inspect_k3s_runtime_state(root, role="server")

    assert state["datastore"] == "sqlite"
    assert len(state["server_db_sha256"]) == 64
    assert len(state["server_token_sha256"]) == 64
    assert set(state) == {
        "datastore",
        "server_db_sha256",
        "server_token_sha256",
        "server_tls_sha256",
        "server_cred_sha256",
        "server_manifests_sha256",
        "server_agent_state_sha256",
        "server_agent_excluded_runtime",
        "service_exec_start_sha256",
    }


def test_cold_agent_state_excludes_only_containerd_cache_and_rejects_unsupported_server_store(tmp_path: Path) -> None:
    root = tmp_path / "k3s"
    (root / "agent/etc").mkdir(parents=True)
    (root / "agent/etc/config.toml").write_text("runtime config", encoding="utf-8")
    (root / "agent/containerd").mkdir()
    (root / "agent/containerd/cache").write_text("cache one", encoding="utf-8")
    first = inspect_k3s_runtime_state(root, role="worker")
    (root / "agent/containerd/cache").write_text("cache two", encoding="utf-8")
    second = inspect_k3s_runtime_state(root, role="worker")
    assert first == second

    (root / "server/db/etcd").mkdir(parents=True)
    (root / "server/db/state.db").write_text("stale sqlite", encoding="utf-8")
    with pytest.raises(ValueError, match="embedded etcd"):
        inspect_k3s_runtime_state(root, role="server")


def test_cold_server_state_rejects_special_database_files(tmp_path: Path) -> None:
    root = tmp_path / "k3s"
    for relative in ("server/db", "server/tls", "server/cred", "server/manifests"):
        (root / relative).mkdir(parents=True)
    (root / "agent/etc").mkdir(parents=True)
    (root / "agent/etc/config.toml").write_bytes(b"agent config")
    (root / "server/db/state.db").write_bytes(b"sqlite datastore")
    (root / "server/token").write_bytes(b"server token")
    for relative in ("server/tls/ca.crt", "server/cred/admin", "server/manifests/coredns.yaml"):
        (root / relative).write_bytes(b"data")
    os.mkfifo(root / "server/db/unknown")
    with pytest.raises(ValueError, match="special filesystem objects"):
        inspect_k3s_runtime_state(root, role="server")


def test_emptydir_policy_allows_only_rendered_disposable_and_known_istio_runtime_volumes() -> None:
    pods = {
        "items": [
            {
                "metadata": {
                    "namespace": "istio-system",
                    "name": "gateway-abc",
                    "labels": {"app": "istio-ingressgateway"},
                },
                "spec": {
                    "nodeName": "vis-lab",
                    "containers": [{"name": "istio-proxy"}],
                    "volumes": [{"name": name, "emptyDir": {}} for name in [
                        "workload-socket", "credential-socket", "workload-certs", "istio-envoy", "istio-data"
                    ]],
                },
                "status": {"phase": "Running"},
            },
            {
                "metadata": {
                    "namespace": "kubeflow",
                    "name": "model-catalog-server-123",
                    "labels": {"app.kubernetes.io/name": "model-catalog", "app.kubernetes.io/component": "server"},
                    "ownerReferences": [{"kind": "ReplicaSet", "name": "model-catalog-server"}],
                },
                "spec": {
                    "nodeName": "ubuntu",
                    "containers": [{"name": "server"}, {"name": "istio-proxy"}],
                    "volumes": [
                        {"name": "perf-data", "emptyDir": {}},
                        {"name": "istio-envoy", "emptyDir": {}},
                        {"name": "istio-data", "emptyDir": {}},
                    ],
                },
                "status": {"phase": "Running"},
            },
        ]
    }

    result = validate_emptydir_policy(pods)

    assert result["status"] == "verified"
    assert [item["classification"] for item in result["approved_pods"]] == [
        "rendered-disposable-plus-injected-istio-runtime",
        "rendered-disposable",
    ]
    assert result["blockers"] == []


def test_emptydir_policy_blocks_unrendered_workload_data_before_any_pod_delete() -> None:
    pods = {
        "items": [
            {
                "metadata": {
                    "namespace": "gods-mlops",
                    "name": "operator-cache",
                    "labels": {"app": "operator"},
                    "ownerReferences": [{"kind": "StatefulSet", "name": "operator"}],
                },
                "spec": {"nodeName": "ubuntu", "containers": [{"name": "operator"}], "volumes": [{"name": "important-cache", "emptyDir": {}}]},
                "status": {"phase": "Running"},
            }
        ]
    }

    result = validate_emptydir_policy(pods)

    assert result["status"] == "blocked"
    assert result["approved_pods"] == []
    assert result["blockers"][0]["volumes"] == ["important-cache"]


def test_emptydir_policy_accepts_known_injected_istio_runtime_mounts_only_with_proxy_owner() -> None:
    pod = {
        "metadata": {
            "namespace": "gods-mlops",
            "name": "review-api-123",
            "labels": {"app": "review-api"},
            "ownerReferences": [{"kind": "ReplicaSet", "name": "review-api"}],
        },
        "spec": {
            "nodeName": "ubuntu",
            "containers": [{"name": "api"}, {"name": "istio-proxy"}],
            "volumes": [{"name": "istio-envoy", "emptyDir": {}}],
        },
        "status": {"phase": "Running"},
    }

    result = validate_emptydir_policy({"items": [pod]})

    assert result["status"] == "verified"
    assert result["approved_pods"][0]["classification"] == "injected-istio-runtime"
