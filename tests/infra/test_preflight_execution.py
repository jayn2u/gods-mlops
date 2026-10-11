from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from gods_mlops.lifecycle.inventory import load_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYBOOK = REPO_ROOT / "infra" / "ansible" / "preflight.yml"
SITE_PLAYBOOK = REPO_ROOT / "infra" / "ansible" / "site.yml"
TEARDOWN_PLAYBOOK = REPO_ROOT / "infra" / "ansible" / "teardown.yml"


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _run_preflight(
    tmp_path: Path,
    *,
    existing_owned_root: bool,
    authenticated_root: bool = False,
) -> dict:
    ansible_playbook = shutil.which("ansible-playbook")
    if not ansible_playbook:
        pytest.skip("ansible-playbook is required for the preflight execution contract")

    root = tmp_path / "gods-mlops"
    k3s_root = root / "k3s"
    preserved_root = tmp_path / "labclip-data"
    report_path = tmp_path / "preflight.json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    if authenticated_root:
        _write_fake_command(
            fake_bin / "sudo",
            'if [ "$1" = "-n" ] && [ "$2" = "true" ]; then exit 1; fi\nwhile [ "$#" -gt 0 ]; do case "$1" in -H|-S|-n) shift ;; -p|-u|-i) shift 2 ;; --) shift; break ;; *) break ;; esac; done\nGODS_TEST_ROOT=1 exec "$@"',
        )
        _write_fake_command(fake_bin / "id", 'if [ "$GODS_TEST_ROOT" = "1" ]; then printf "0\\n"; else exec /usr/bin/id -u; fi')
    else:
        _write_fake_command(fake_bin / "sudo", "exit 1")
    _write_fake_command(fake_bin / "findmnt", "printf '/\\n'")
    _write_fake_command(
        fake_bin / "ip",
        'if [ "$1" = "route" ]; then printf "local 127.0.0.1 dev lo src 127.0.0.1\\n"; else printf "1: lo inet 127.0.0.1/8 scope host lo\\n"; fi',
    )
    _write_fake_command(fake_bin / "systemctl", 'printf "inactive\\n"; exit 3')
    _write_fake_command(
        fake_bin / "k3s",
        'printf "k3s version v1.36.2+k3s1 (test)\\n"',
    )

    if existing_owned_root:
        root.mkdir()
        marker = {
            "schema_version": 1,
            "owner": "gods-mlops",
            "data_root": str(root),
            "k3s_data_dir": str(k3s_root),
            "k3s_version": "v1.36.2+k3s1",
        }
        (root / ".gods-mlops-owner.json").write_text(json.dumps(marker), encoding="utf-8")

    inventory = {
        "all": {
            "children": {
                "gods_cluster": {
                    "hosts": {
                        "preflight-test": {
                            "ansible_connection": "local",
                            "ansible_python_interpreter": sys.executable,
                            "gods_expected_root_uid": os.getuid(),
                            "gods_container_uid": 10001,
                            "gods_container_gid": 10001,
                            "gods_data_root": str(root),
                            "gods_k3s_data_dir": str(k3s_root),
                            "gods_owner_marker_name": ".gods-mlops-owner.json",
                            "gods_storage_mount": "/",
                            "gods_min_free_gib": 0,
                            "gods_data_directories": [],
                            "gods_preserve_paths": [str(preserved_root)],
                            "gods_preserve_patterns": [],
                            "gods_preserve_services": [],
                            "gods_network_flows": [],
                            "gods_peer_ip": "127.0.0.1",
                            "k3s_node_ip": "127.0.0.1",
                            **({
                                "ansible_become_exe": str(fake_bin / "sudo"),
                                "ansible_become_method": "sudo",
                                "ansible_become_pass": "test-only",
                            } if authenticated_root else {}),
                        }
                    }
                }
            }
        }
    }
    inventory_path = tmp_path / "inventory.yml"
    inventory_path.write_text(yaml.safe_dump(inventory), encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    extra_vars = f"gods_preflight_report_path={report_path}"
    if authenticated_root:
        extra_vars += " gods_preflight_use_become=true"
    result = subprocess.run(
        [ansible_playbook, str(PLAYBOOK), "-i", str(inventory_path), "-e", extra_vars],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    if authenticated_root:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, "missing authenticated or non-interactive sudo must leave the report blocked"
    assert report_path.is_file(), result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == ("ready" if authenticated_root else "blocked")
    assert len(report["nodes"]) == 1
    assert report["nodes"][0]["permissions"]["sudo_noninteractive"] is False
    assert report["nodes"][0]["permissions"]["effective_uid_is_root"] is authenticated_root
    assert isinstance(report["nodes"][0]["storage"]["owner_marker_valid"], bool)
    return report["nodes"][0]


def test_preflight_json_uses_native_boolean_for_a_new_root(tmp_path: Path) -> None:
    node = _run_preflight(tmp_path, existing_owned_root=False)
    assert node["storage"]["owner_marker_valid"] is True
    assert node["storage"]["data_root_exists"] is False


def test_preflight_json_uses_native_boolean_for_a_valid_existing_root(
    tmp_path: Path,
) -> None:
    node = _run_preflight(tmp_path, existing_owned_root=True)
    assert node["storage"]["owner_marker_valid"] is True
    assert node["storage"]["data_root_exists"] is True


def test_preflight_accepts_authenticated_become_root_when_sudo_n_is_unavailable(tmp_path: Path) -> None:
    node = _run_preflight(tmp_path, existing_owned_root=False, authenticated_root=True)
    assert node["status"] == "ready"
    assert node["permissions"]["effective_uid_is_root"] is True
    assert node["permissions"]["sudo_noninteractive"] is False


def test_site_always_imports_fresh_preflight_before_installation_plays() -> None:
    plays = yaml.safe_load(SITE_PLAYBOOK.read_text(encoding="utf-8"))
    assert plays[0].get("ansible.builtin.import_playbook") == "preflight.yml"
    assert plays[0].get("vars", {}).get("gods_preflight_use_become") is True
    assert [play.get("hosts") for play in plays[1:]] == ["gods_server", "gods_gpu_worker"]


def test_teardown_checks_both_host_ownership_before_controller_or_cluster_work(tmp_path: Path) -> None:
    ansible_playbook = shutil.which("ansible-playbook")
    if not ansible_playbook:
        pytest.skip("ansible-playbook is required for the teardown execution contract")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    call_log = tmp_path / "controller-calls.log"
    become_log = tmp_path / "become-calls.log"
    _write_fake_command(
        fake_bin / "sudo",
        'printf "%s\\n" "$*" >> "$GODS_TEST_BECOME_LOG"\nwhile [ "$#" -gt 0 ]; do case "$1" in -H|-S|-n) shift ;; -p|-u|-i) shift 2 ;; --) shift; break ;; *) break ;; esac; done\nGODS_TEST_ROOT=1 exec "$@"',
    )
    _write_fake_command(fake_bin / "gods-mlops", 'printf "%s\\n" "$*" >> "$GODS_TEST_CALL_LOG"; exit 98')
    inventory = {
        "all": {
            "children": {
                "gods_server": {"hosts": {"vis-lab": _teardown_test_host(tmp_path / "server", fake_bin / "sudo")}},
                "gods_gpu_worker": {"hosts": {"ubuntu": _teardown_test_host(tmp_path / "worker", fake_bin / "sudo")}},
                "gods_cluster": {"children": {"gods_server": None, "gods_gpu_worker": None}},
            }
        }
    }
    inventory_path = tmp_path / "inventory.yml"
    inventory_path.write_text(yaml.safe_dump(inventory), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["GODS_TEST_CALL_LOG"] = str(call_log)
    env["GODS_TEST_BECOME_LOG"] = str(become_log)

    result = subprocess.run(
        [ansible_playbook, "-v", str(TEARDOWN_PLAYBOOK), "-i", str(inventory_path), "-e", f"gods_reclaim_plan_id={'a' * 64}", "--ask-become-pass"],
        cwd=REPO_ROOT,
        env=env,
        input="test-only\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0
    assert "ownership" in (result.stdout + result.stderr).lower()
    assert not call_log.exists(), "controller plan/API operations must not start before both nodes prove ownership"
    assert become_log.is_file(), "the protected-root metadata precheck must run through authenticated become"


def test_teardown_ownership_precheck_succeeds_with_a_protected_cluster_marker_under_become(tmp_path: Path) -> None:
    ansible_playbook = shutil.which("ansible-playbook")
    if not ansible_playbook:
        pytest.skip("ansible-playbook is required for the teardown execution contract")
    first_play = yaml.safe_load(TEARDOWN_PLAYBOOK.read_text(encoding="utf-8"))[0]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    become_log = tmp_path / "become-calls.log"
    _write_fake_command(
        fake_bin / "sudo",
        'printf "%s\\n" "$*" >> "$GODS_TEST_BECOME_LOG"\nwhile [ "$#" -gt 0 ]; do case "$1" in -H|-S|-n) shift ;; -p|-u|-i) shift 2 ;; --) shift; break ;; *) break ;; esac; done\nGODS_TEST_ROOT=1 exec "$@"',
    )
    roots = {"vis-lab": tmp_path / "server", "ubuntu": tmp_path / "worker"}
    hosts = {}
    for node, root in roots.items():
        k3s_root = root / "k3s"
        k3s_root.mkdir(parents=True)
        root_marker = root / ".gods-mlops-owner.json"
        cluster_marker = k3s_root / ".gods-mlops-cluster-owner.json"
        marker = {"owner": "gods-mlops", "data_root": str(root), "k3s_data_dir": str(k3s_root)}
        root_marker.write_text(json.dumps(marker), encoding="utf-8")
        cluster_marker.write_text(json.dumps(marker), encoding="utf-8")
        k3s_root.chmod(0o700)
        cluster_marker.chmod(0o600)
        hosts[node] = _teardown_test_host(root, fake_bin / "sudo")
    inventory = {
        "all": {"children": {
            "gods_server": {"hosts": {"vis-lab": hosts["vis-lab"]}},
            "gods_gpu_worker": {"hosts": {"ubuntu": hosts["ubuntu"]}},
            "gods_cluster": {"children": {"gods_server": None, "gods_gpu_worker": None}},
        }}
    }
    inventory_path = tmp_path / "inventory.yml"
    inventory_path.write_text(yaml.safe_dump(inventory), encoding="utf-8")
    playbook_path = tmp_path / "ownership-gate.yml"
    playbook_path.write_text(yaml.safe_dump([first_play]), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["GODS_TEST_BECOME_LOG"] = str(become_log)

    result = subprocess.run(
        [ansible_playbook, "-v", str(playbook_path), "-i", str(inventory_path), "--ask-become-pass"],
        cwd=REPO_ROOT,
        env=env,
        input="test-only\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert become_log.is_file()
    assert "-u root" in become_log.read_text(encoding="utf-8")




@pytest.mark.parametrize("server_step", ["runtime_verification_pending", "retained_hash_pending", "service_stopped"])
def test_teardown_partial_and_completed_offline_retries_skip_kubernetes_api(
    tmp_path: Path, server_step: str
) -> None:
    ansible_playbook = shutil.which("ansible-playbook")
    if not ansible_playbook:
        pytest.skip("ansible-playbook is required for the teardown execution contract")
    source_plays = yaml.safe_load(TEARDOWN_PLAYBOOK.read_text(encoding="utf-8"))
    controller = next(play for play in source_plays if play.get("name") == "Verify backups, the exact plan, and a safe workload stop on the controller")
    resume_start = next(index for index, task in enumerate(controller["tasks"]) if task.get("name", "").startswith("Select online or offline reclaim"))
    resume_tasks = controller["tasks"][resume_start:]
    plan_id = "a" * 64
    snapshot_path = tmp_path / "daemonsets.json"
    snapshot_sha256 = "b" * 64
    test_plays = [
        {
            "name": "Provide durable state from isolated nodes",
            "hosts": "gods_server",
            "connection": "local",
            "gather_facts": False,
            "tasks": [{"ansible.builtin.set_fact": {
                "gods_lifecycle_service_active": False,
                "gods_reclaim_state_exists": True,
                "gods_reclaim_state_doc": {
                    "owner": "gods-mlops", "node": "vis-lab", "plan_id": plan_id,
                    "step": server_step, "daemonset_snapshot_path": str(snapshot_path),
                    "daemonset_snapshot_sha256": snapshot_sha256,
                },
            }}],
        },
        {
            "name": "Provide a completed worker state",
            "hosts": "gods_gpu_worker",
            "connection": "local",
            "gather_facts": False,
            "tasks": [{"ansible.builtin.set_fact": {
                "gods_lifecycle_service_active": False,
                "gods_reclaim_state_exists": True,
                "gods_reclaim_state_doc": {
                    "owner": "gods-mlops", "node": "ubuntu", "plan_id": plan_id,
                    "step": "service_stopped", "daemonset_snapshot_path": str(snapshot_path),
                    "daemonset_snapshot_sha256": snapshot_sha256,
                },
            }}],
        },
        {
            "name": "Run production controller resume tasks",
            "hosts": "localhost",
            "connection": "local",
            "gather_facts": False,
            "vars": {
                "gods_lifecycle_inventory": str(tmp_path / "inventory.yml"),
                "gods_kubeconfig": str(tmp_path / "missing-kubeconfig"),
            },
            "tasks": resume_tasks,
        },
    ]
    playbook_path = tmp_path / "offline-resume.yml"
    playbook_path.write_text(yaml.safe_dump(test_plays), encoding="utf-8")
    inventory = {
        "all": {"children": {
            "gods_server": {"hosts": {"vis-lab": {"ansible_connection": "local", "ansible_python_interpreter": sys.executable}}},
            "gods_gpu_worker": {"hosts": {"ubuntu": {"ansible_connection": "local", "ansible_python_interpreter": sys.executable}}},
            "gods_cluster": {"children": {"gods_server": None, "gods_gpu_worker": None}},
        }}
    }
    inventory_path = tmp_path / "inventory.yml"
    inventory_path.write_text(yaml.safe_dump(inventory), encoding="utf-8")
    call_log = tmp_path / "cli.log"
    kubectl_log = tmp_path / "kubectl.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_command(fake_bin / "gods-mlops", 'printf "%s\\n" "$*" >> "$GODS_TEST_CALL_LOG"; exit 0')
    _write_fake_command(fake_bin / "kubectl", 'printf "%s\\n" "$*" >> "$GODS_TEST_KUBECTL_LOG"; exit 98')
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["GODS_TEST_CALL_LOG"] = str(call_log)
    env["GODS_TEST_KUBECTL_LOG"] = str(kubectl_log)

    result = subprocess.run(
        [ansible_playbook, str(playbook_path), "-i", str(inventory_path), "-e", f"gods_reclaim_plan_id={plan_id}"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = call_log.read_text(encoding="utf-8")
    assert "verify-daemonset-snapshot" in calls
    assert "verify-daemonset-scope" not in calls
    assert "capture-daemonsets" not in calls
    assert "verify-emptydir-policy" not in calls
    assert not kubectl_log.exists()


@pytest.mark.parametrize(
    ("worker_status", "server_play_reached"),
    [("service_stopped", True), ("retained_hash_pending", False)],
)
def test_teardown_runs_worker_gate_before_server_and_summary_after_both(
    tmp_path: Path, worker_status: str, server_play_reached: bool
) -> None:
    ansible_playbook = shutil.which("ansible-playbook")
    if not ansible_playbook:
        pytest.skip("ansible-playbook is required for the teardown execution contract")
    source_plays = yaml.safe_load(TEARDOWN_PLAYBOOK.read_text(encoding="utf-8"))
    worker_gate = next(play for play in source_plays if play.get("name") == "Require worker cold evidence before stopping the control plane")
    final_summary = next(play for play in source_plays if play.get("name") == "Require both nodes to have cold retained-state evidence before reporting reclaim complete")
    server_marker = tmp_path / "server-play-ran"
    playbook = [
        {
            "hosts": "gods_gpu_worker",
            "connection": "local",
            "gather_facts": False,
            "tasks": [{"ansible.builtin.set_fact": {"gods_reclaim_node_status": worker_status}}],
        },
        worker_gate,
        {
            "hosts": "gods_server",
            "connection": "local",
            "gather_facts": False,
            "tasks": [
                {"ansible.builtin.copy": {"dest": str(server_marker), "content": "reached"}},
                {"ansible.builtin.set_fact": {"gods_reclaim_node_status": "service_stopped"}},
            ],
        },
        final_summary,
    ]
    playbook_path = tmp_path / "completion-order.yml"
    playbook_path.write_text(yaml.safe_dump(playbook), encoding="utf-8")
    inventory = {
        "all": {"children": {
            "gods_server": {"hosts": {"vis-lab": {"ansible_connection": "local", "ansible_python_interpreter": sys.executable}}},
            "gods_gpu_worker": {"hosts": {"ubuntu": {"ansible_connection": "local", "ansible_python_interpreter": sys.executable}}},
            "gods_cluster": {"children": {"gods_server": None, "gods_gpu_worker": None}},
        }}
    }
    inventory_path = tmp_path / "inventory.yml"
    inventory_path.write_text(yaml.safe_dump(inventory), encoding="utf-8")

    result = subprocess.run(
        [ansible_playbook, str(playbook_path), "-i", str(inventory_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert (result.returncode == 0) is server_play_reached, result.stdout + result.stderr
    assert server_marker.exists() is server_play_reached


def test_teardown_emptydir_drain_uses_eviction_and_stops_on_pdb_block(tmp_path: Path) -> None:
    ansible_playbook = shutil.which("ansible-playbook")
    if not ansible_playbook:
        pytest.skip("ansible-playbook is required for the teardown execution contract")
    source_plays = yaml.safe_load(TEARDOWN_PLAYBOOK.read_text(encoding="utf-8"))
    drain_play = next(play for play in source_plays if play.get("name") == "Drain approved Gods workloads before stopping K3s")
    playbook = [
        {
            "hosts": "localhost",
            "connection": "local",
            "gather_facts": False,
            "tasks": [{"ansible.builtin.set_fact": {
                "gods_reclaim_offline_resume": False,
                "gods_daemonsets_to_stop": [],
                "gods_emptydir_approved_pods": [{"namespace": "kubeflow", "name": "db-0", "volumes": ["scratch"]}],
            }}],
        },
        drain_play,
    ]
    playbook_path = tmp_path / "pdb-drain.yml"
    playbook_path.write_text(yaml.safe_dump(playbook), encoding="utf-8")
    inventory = {
        "all": {"children": {
            "gods_server": {"hosts": {"vis-lab": {"ansible_connection": "local", "ansible_python_interpreter": sys.executable}}},
            "gods_gpu_worker": {"hosts": {"ubuntu": {"ansible_connection": "local", "ansible_python_interpreter": sys.executable}}},
            "gods_cluster": {"children": {"gods_server": None, "gods_gpu_worker": None}},
        }}
    }
    inventory_path = tmp_path / "inventory.yml"
    inventory_path.write_text(yaml.safe_dump(inventory), encoding="utf-8")
    kubectl_log = tmp_path / "kubectl.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_command(
        fake_bin / "kubectl",
        'printf "%s\\n" "$*" >> "$GODS_TEST_KUBECTL_LOG"\ncase " $* " in *" cordon "*) printf "node cordoned\\n" ;; *" drain "*) printf "cannot evict as it would violate the pod disruption budget\\n" >&2; exit 1 ;; *" uncordon "*) printf "node uncordoned\\n" ;; *) exit 98 ;; esac',
    )
    _write_fake_command(
        fake_bin / "gods-mlops",
        'printf "{\\"status\\":\\"verified\\",\\"approved_pods\\":[{\\"namespace\\":\\"kubeflow\\",\\"name\\":\\"db-0\\",\\"volumes\\":[\\"scratch\\"]}]}\\n"',
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["GODS_TEST_KUBECTL_LOG"] = str(kubectl_log)

    result = subprocess.run(
        [ansible_playbook, str(playbook_path), "-i", str(inventory_path), "-e", f"gods_kubeconfig_file={tmp_path / 'kubeconfig'}"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0
    commands = kubectl_log.read_text(encoding="utf-8").splitlines()
    drain = next(command for command in commands if command.startswith("--kubeconfig") and " drain " in f" {command} ")
    assert "--delete-emptydir-data" in drain
    assert "delete pod" not in "\n".join(commands)
    assert any("uncordon" in command for command in commands)


@pytest.mark.parametrize("state_present", [False, True])
def test_purge_requires_plan_bound_cold_state_and_current_quiescence(
    tmp_path: Path, state_present: bool
) -> None:
    ansible_playbook = shutil.which("ansible-playbook")
    if not ansible_playbook:
        pytest.skip("ansible-playbook is required for the purge execution contract")
    root = tmp_path / "gods-mlops"
    k3s_root = root / "k3s"
    k3s_root.mkdir(parents=True)
    target = root / "objects"
    target.mkdir()
    (target / "data.bin").write_text("retained fixture", encoding="utf-8")
    root_marker = root / ".gods-mlops-owner.json"
    cluster_marker = k3s_root / ".gods-mlops-cluster-owner.json"
    plan_id = "c" * 64
    root_marker.write_text(json.dumps({"owner": "gods-mlops", "data_root": str(root), "k3s_data_dir": str(k3s_root)}), encoding="utf-8")
    cluster_marker.write_text(json.dumps({"owner": "gods-mlops", "data_root": str(root), "k3s_data_dir": str(k3s_root)}), encoding="utf-8")
    if state_present:
        required_by_node = load_inventory(REPO_ROOT / "infra" / "ansible" / "inventory.example.yml")["requirements_by_node"]
        assert len(required_by_node["vis-lab"]["retained_paths"]) == 1
        assert len(required_by_node["ubuntu"]["retained_paths"]) == 9
        state = {
            "schema_version": 1,
            "owner": "gods-mlops",
            "node": "ubuntu",
            "plan_id": plan_id,
            "step": "service_stopped",
            "service": "k3s-agent",
            "k3s_data_dir": str(k3s_root),
            "runtime_status": "verified_stopped",
            "cold_retained_paths": {identifier: "a" * 64 for identifier in required_by_node["ubuntu"]["retained_paths"]},
            "cold_k3s_state": {"agent_state_sha256": "b" * 64},
        }
        state_path = root / ".gods-mlops-reclaim-state.json"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        state_path.chmod(0o600)
    targets = {
        "schema_version": 1,
        "owner": "gods-mlops",
        "owned_roots": [str(root)],
        "protected_paths": [],
        "targets": [{"node": "ubuntu", "path": str(target), "owner_marker": str(root_marker)}],
    }
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps(targets), encoding="utf-8")
    inventory = {
        "all": {"children": {
            "gods_server": {"hosts": {}},
            "gods_gpu_worker": {"hosts": {"ubuntu": {
                "ansible_connection": "local",
                "ansible_python_interpreter": sys.executable,
                "ansible_become_exe": str(tmp_path / "bin" / "sudo"),
                "ansible_become_method": "sudo",
                "ansible_become_pass": "test-only",
                "gods_data_root": str(root),
                "gods_k3s_data_dir": str(k3s_root),
                "gods_owner_marker_name": root_marker.name,
                "gods_cluster_marker_name": cluster_marker.name,
            }}},
            "gods_cluster": {"children": {"gods_gpu_worker": None}},
        }}
    }
    inventory_path = tmp_path / "inventory.yml"
    inventory_path.write_text(yaml.safe_dump(inventory), encoding="utf-8")
    target_digest = hashlib.sha256(json.dumps({
        "schema_version": 1,
        "owner": "gods-mlops",
        "owned_roots": [str(root)],
        "protected_paths": [],
        "targets": [{"node": "ubuntu", "path": str(target), "owner_marker": str(root_marker)}],
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_command(fake_bin / "sudo", 'if [ "$1" = "-n" ] && [ "$2" = "true" ]; then exit 1; fi\nwhile [ "$#" -gt 0 ]; do case "$1" in -H|-S|-n) shift ;; -p|-u|-i) shift 2 ;; --) shift; break ;; *) break ;; esac; done\nGODS_TEST_ROOT=1 exec "$@"')
    _write_fake_command(fake_bin / "id", 'if [ "$GODS_TEST_ROOT" = "1" ]; then printf "0\\n"; else exec /usr/bin/id -u; fi')
    expected_by_node = load_inventory(REPO_ROOT / "infra" / "ansible" / "inventory.example.yml")["requirements_by_node"]
    purge_precheck = {
        "status": "verified",
        "target_sha256": target_digest,
        "plan_id": plan_id,
        "target_count": 1,
        "required_retained_paths_by_node": {
            node: requirements["retained_paths"] for node, requirements in expected_by_node.items()
        },
    }
    _write_fake_command(
        fake_bin / "gods-mlops",
        f'printf "%s\\n" \'{json.dumps(purge_precheck, separators=(",", ":"))}\'',
    )
    _write_fake_command(
        fake_bin / "python3",
        'case "$1" in *runtime_probe.py*) printf "%s\\n" "$*" >> "$GODS_TEST_RUNTIME_LOG"; printf "{\\"status\\":\\"pending\\",\\"failures\\":[{\\"check\\":\\"owned_runtime_processes_remain\\"}]}\\n"; exit 1 ;; *) exec /usr/bin/python3 "$@" ;; esac',
    )
    runtime_log = tmp_path / "runtime-probe.log"
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    env["GODS_TEST_RUNTIME_LOG"] = str(runtime_log)
    extra_vars = [
        f"@{targets_path}",
        f"gods_purge_targets_file={targets_path}",
        f"gods_purge_target_digest={target_digest}",
        f"gods_reclaim_plan_id={plan_id}",
        f"gods_lifecycle_inventory_file={REPO_ROOT / 'infra' / 'ansible' / 'inventory.example.yml'}",
    ]
    command = [ansible_playbook, str(REPO_ROOT / "infra" / "ansible" / "purge.yml"), "-i", str(inventory_path)]
    for value in extra_vars:
        command.extend(["-e", value])

    result = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False, timeout=60)

    assert result.returncode != 0
    assert target.is_dir(), "purge must leave the temporary target untouched when evidence or current quiescence fails"
    assert runtime_log.exists() is state_present



def _teardown_test_host(root: Path, sudo_path: Path) -> dict[str, object]:
    return {
        "ansible_connection": "local",
        "ansible_python_interpreter": sys.executable,
        "ansible_become_exe": str(sudo_path),
        "ansible_become_method": "sudo",
        "ansible_become_pass": "test-only",
        "gods_expected_root_uid": os.getuid(),
        "gods_data_root": str(root),
        "gods_k3s_data_dir": str(root / "k3s"),
        "gods_owner_marker_name": ".gods-mlops-owner.json",
        "gods_cluster_marker_name": ".gods-mlops-cluster-owner.json",
    }
