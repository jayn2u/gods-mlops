from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import gods_mlops.cli as cli
from gods_mlops.lifecycle.inventory import load_inventory, plan_reclaim
from gods_mlops.lifecycle.recovery import purge_targets_digest
from gods_mlops.lifecycle.recovery import hash_tree


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repository_inventory_includes_external_model_staging_and_data_roots() -> None:
    inventory = load_inventory(REPO_ROOT / "infra" / "ansible" / "inventory.example.yml")
    plan = plan_reclaim(inventory)

    assert {node["name"] for node in inventory["nodes"]} == {"vis-lab", "ubuntu"}
    assert "/data/jayn2u/gods-mlops-model-preparation" in plan["preserved_paths"]
    assert "/mnt/data/labclip-k3s" in plan["preserved_paths"]
    assert "/data/jayn2u/gods-mlops/k3s" in {item["path"] for item in plan["retained_paths"]}
    assert set(inventory["required_database_restores"]) == {
        "gods-mlops-kfp-metadata-postgres",
        "gods-mlops-katib-mysql",
        "gods-mlops-model-catalog-postgres",
        "gods-mlops-kfp-mysql",
    }
    assert len(inventory["required_retained_paths"]) == 9


def test_purge_without_exact_confirmation_is_only_a_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    targets = {
        "schema_version": 1,
        "owner": "gods-mlops",
        "owned_roots": [str(tmp_path / "managed")],
        "protected_paths": [],
        "targets": [
            {
                "node": "ubuntu",
                "path": str(tmp_path / "managed" / "objects"),
                "owner_marker": str(tmp_path / "managed" / ".gods-mlops-owner.json"),
            }
        ],
    }
    targets_path = tmp_path / "purge-targets.json"
    targets_path.write_text(json.dumps(targets), encoding="utf-8")
    monkeypatch.setattr(cli, "_run_ansible", lambda _: pytest.fail("dry-run must not launch Ansible"))

    result = cli.main(["lifecycle", "purge", "--targets", str(targets_path)])

    output = json.loads(capsys.readouterr().out)
    assert result == 2
    assert output["dry_run"] is True
    assert output["target_sha256"] == purge_targets_digest(targets)
    assert output["confirmation_required"] == f"--confirm-targets {output['target_sha256']}"


def test_purge_with_wrong_confirmation_never_launches_ansible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    targets = {
        "schema_version": 1,
        "owner": "gods-mlops",
        "owned_roots": [str(tmp_path / "managed")],
        "protected_paths": [],
        "targets": [
            {
                "node": "ubuntu",
                "path": str(tmp_path / "managed" / "objects"),
                "owner_marker": str(tmp_path / "managed" / ".gods-mlops-owner.json"),
            }
        ],
    }
    targets_path = tmp_path / "purge-targets.json"
    targets_path.write_text(json.dumps(targets), encoding="utf-8")
    monkeypatch.setattr(cli, "_run_ansible", lambda _: pytest.fail("wrong digest must not launch Ansible"))

    result = cli.main([
        "lifecycle",
        "purge",
        "--targets",
        str(targets_path),
        "--confirm-targets",
        "0" * 64,
    ])

    assert result == 1
    assert "confirmation digest" in capsys.readouterr().err


def test_hash_tree_command_prints_only_path_and_digest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "backup"
    target.mkdir()
    (target / "data.bin").write_bytes(b"private recovery bytes")

    result = cli.main(["lifecycle", "hash-tree", "--path", str(target)])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output == {"path": str(target), "sha256": hash_tree(target)}
    assert "private recovery bytes" not in json.dumps(output)


def test_authenticated_privilege_probe_passes_ask_become_through_to_ansible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "_run_ansible", lambda arguments: calls.append(arguments) or 0)

    result = cli.main(["lifecycle", "verify-privilege", "--ask-become-pass"])

    assert result == 0
    assert len(calls) == 1
    assert calls[0][-1] == "--ask-become-pass"
    assert str(cli.DEFAULT_PRIVILEGE_PROBE) in calls[0]


def test_remote_tree_hash_uses_the_owning_node_and_optional_become_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(cli, "_run_ansible", lambda arguments: calls.append(arguments) or 0)

    result = cli.main([
        "lifecycle",
        "hash-tree",
        "--path",
        "/data/jayn2u/gods-mlops/objects",
        "--node",
        "ubuntu",
        "--ask-become-pass",
    ])

    assert result == 0
    assert calls[0][calls[0].index("--limit") + 1] == "ubuntu"
    assert calls[0][-1] == "--ask-become-pass"
    assert str(cli.DEFAULT_HASH_PATH_PLAYBOOK) in calls[0]


def test_reclaim_requires_structural_manifest_coverage_before_ansible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = REPO_ROOT / "infra" / "ansible" / "inventory.example.yml"
    inventory = load_inventory(inventory_path)
    plan = plan_reclaim(inventory)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "owner": "gods-mlops", "plan_id": plan["plan_id"]}), encoding="utf-8")
    monkeypatch.setattr(cli, "_run_ansible", lambda _: pytest.fail("unverified manifest must not launch Ansible"))

    result = cli.main([
        "lifecycle",
        "reclaim",
        "--inventory",
        str(inventory_path),
        "--manifest",
        str(manifest_path),
        "--confirm-plan",
        plan["plan_id"],
    ])

    assert result == 1
    output = capsys.readouterr().err
    assert "retained_paths" in output


def test_reconnect_does_not_invoke_site_until_retained_manifest_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = REPO_ROOT / "infra" / "ansible" / "inventory.example.yml"
    inventory = load_inventory(inventory_path)
    plan = plan_reclaim(inventory)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "owner": "gods-mlops", "plan_id": plan["plan_id"]}), encoding="utf-8")
    monkeypatch.setattr(cli, "_run_ansible", lambda _: pytest.fail("unverified recovery must not launch site.yml"))

    result = cli.main([
        "lifecycle",
        "reconnect",
        "--manifest",
        str(manifest_path),
        "--inventory",
        str(inventory_path),
        "--daemonset-snapshot",
        str(tmp_path / "snapshot.json"),
        "--confirm-plan",
        "0" * 64,
        "--confirm-snapshot-sha256",
        "0" * 64,
    ])

    assert result == 1
    assert "retained_paths" in capsys.readouterr().err


def test_playbooks_stop_only_marked_services_and_authenticate_become() -> None:
    teardown = yaml.safe_load((REPO_ROOT / "infra" / "ansible" / "teardown.yml").read_text(encoding="utf-8"))
    purge = yaml.safe_load((REPO_ROOT / "infra" / "ansible" / "purge.yml").read_text(encoding="utf-8"))
    probe = yaml.safe_load((REPO_ROOT / "infra" / "ansible" / "privilege-probe.yml").read_text(encoding="utf-8"))
    reconnect = yaml.safe_load((REPO_ROOT / "infra" / "ansible" / "reconnect.yml").read_text(encoding="utf-8"))
    teardown_text = (REPO_ROOT / "infra" / "ansible" / "teardown.yml").read_text(encoding="utf-8")

    remote_reclaim = next(item for item in teardown if item.get("hosts") == "gods_gpu_worker")
    controller_reclaim = next(item for item in teardown if item.get("name") == "Drain approved Gods workloads before stopping K3s")
    assert remote_reclaim["become"] is True
    assert any(
        task.get("ansible.builtin.systemd", {}).get("state") == "stopped"
        for task in remote_reclaim["tasks"]
    )
    assert "/usr/local/bin/k3s-uninstall.sh" not in teardown_text
    controller_blocks = [task["block"] for task in controller_reclaim["tasks"] if "block" in task]
    drain_tasks = [
        task
        for task in controller_blocks[0]
        if "drain" in task.get("ansible.builtin.command", {}).get("argv", [])
    ]
    assert len(drain_tasks) == 1
    assert "drain" in drain_tasks[0]["ansible.builtin.command"]["argv"]
    assert "--ignore-daemonsets" in drain_tasks[0]["ansible.builtin.command"]["argv"]
    assert "--delete-emptydir-data" not in teardown_text
    assert "verify-daemonset-scope" in teardown_text
    assert "runtime_probe.py" in teardown_text
    assert "runtime_verification_pending" in teardown_text
    assert controller_blocks
    assert purge[1]["become"] is True
    assert any("ansible.builtin.file" in task and task["ansible.builtin.file"].get("state") == "absent" for task in purge[1]["tasks"])
    assert probe[0]["become"] is True
    assert any(task.get("ansible.builtin.command", {}).get("argv") == ["id", "-u"] for task in probe[0]["tasks"])
    assert reconnect[0]["hosts"] == "localhost"
    assert any(
        task.get("ansible.builtin.command", {}).get("argv", [])[1:3] == ["lifecycle", "check-retained-manifest"]
        for task in reconnect[0]["tasks"]
    )
    assert reconnect[1]["ansible.builtin.import_playbook"] == "verify-retained.yml"
    assert any(item.get("ansible.builtin.import_playbook") == "site.yml" for item in reconnect)
