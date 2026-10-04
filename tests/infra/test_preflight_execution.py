from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAYBOOK = REPO_ROOT / "infra" / "ansible" / "preflight.yml"
SITE_PLAYBOOK = REPO_ROOT / "infra" / "ansible" / "site.yml"


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _run_preflight(tmp_path: Path, *, existing_owned_root: bool) -> dict:
    ansible_playbook = shutil.which("ansible-playbook")
    if not ansible_playbook:
        pytest.skip("ansible-playbook is required for the preflight execution contract")

    root = tmp_path / "gods-mlops"
    k3s_root = root / "k3s"
    preserved_root = tmp_path / "labclip-data"
    report_path = tmp_path / "preflight.json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
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
    result = subprocess.run(
        [
            ansible_playbook,
            str(PLAYBOOK),
            "-i",
            str(inventory_path),
            "-e",
            f"gods_preflight_report_path={report_path}",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )

    assert result.returncode != 0, "sudo failure must leave the report blocked"
    assert report_path.is_file(), result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert len(report["nodes"]) == 1
    assert report["nodes"][0]["permissions"]["sudo_noninteractive"] is False
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


def test_site_always_imports_fresh_preflight_before_installation_plays() -> None:
    plays = yaml.safe_load(SITE_PLAYBOOK.read_text(encoding="utf-8"))
    assert plays[0].get("ansible.builtin.import_playbook") == "preflight.yml"
    assert [play.get("hosts") for play in plays[1:]] == ["gods_server", "gods_gpu_worker"]
