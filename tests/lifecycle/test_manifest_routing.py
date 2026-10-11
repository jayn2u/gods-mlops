from __future__ import annotations

import json
from pathlib import Path

import pytest

import gods_mlops.cli as cli
from gods_mlops.lifecycle.inventory import (
    InventoryError,
    load_inventory,
    plan_reclaim,
    slice_retained_manifest_for_node,
    validate_retained_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest(inventory: dict) -> dict:
    retained = []
    for identifier, expected in inventory["retained_requirements_by_id"].items():
        entry = {
            "id": identifier,
            "node": expected["node"],
            "source_path": expected["source_path"],
            "source_owner": expected["source_owner"],
            "role": expected["role"],
        }
        if expected["role"] != "database":
            entry.update({
                "backup_path": f"{expected['recovery_root']}/snapshots/{identifier}",
                "expected_tree_sha256": "a" * 64,
                "backup_owner": {"uid": 0, "gid": 0},
                "backup_mode": "0700",
            })
        retained.append(entry)

    databases = []
    for node, requirements in inventory["requirements_by_node"].items():
        for identifier in requirements["database_restores"]:
            recovery_root = next(item["recovery_root"] for item in inventory["retained_requirements_by_id"].values() if item["node"] == node)
            databases.append({
                "id": identifier,
                "node": node,
                "backup_dump_path": f"{recovery_root}/db/{identifier}-source.sql",
                "restored_dump_path": f"{recovery_root}/db/{identifier}-restored.sql",
                "expected_sha256": "b" * 64,
                "expected_uid": 0,
                "expected_mode": "0600",
            })

    credentials = []
    for node, requirements in inventory["requirements_by_node"].items():
        for identifier in requirements["credentials"]:
            recovery_root = next(item["recovery_root"] for item in inventory["retained_requirements_by_id"].values() if item["node"] == node)
            credentials.append({
                "id": identifier,
                "node": node,
                "path": f"{recovery_root}/secrets/{identifier}",
                "expected_sha256": "c" * 64,
                "expected_uid": 0,
                "expected_mode": "0600",
            })
            expected_source = inventory.get("credential_source_by_id", {}).get(identifier)
            if expected_source:
                credentials[-1].update({
                    "source_path": expected_source,
                    "source_expected_uid": 0,
                    "source_expected_mode": "0600",
                })
    return {
        "schema_version": 1,
        "owner": "gods-mlops",
        "plan_id": plan_reclaim(inventory)["plan_id"],
        "retained_paths": retained,
        "database_restores": databases,
        "credentials": credentials,
    }


def test_manifest_routes_each_source_backup_and_restore_check_to_its_owning_node() -> None:
    inventory = load_inventory(REPO_ROOT / "infra" / "ansible" / "inventory.example.yml")
    manifest = _manifest(inventory)

    result = validate_retained_manifest(manifest, inventory)
    vislab_slice = slice_retained_manifest_for_node(manifest, inventory, "vis-lab")
    ubuntu_slice = slice_retained_manifest_for_node(manifest, inventory, "ubuntu")

    assert result["status"] == "manifest_valid"
    assert {item["id"] for item in vislab_slice["retained_paths"]} == {
        "gods-mlops-spool",
    }
    assert not vislab_slice["database_restores"]
    assert {item["id"] for item in vislab_slice["credentials"]} == {
        "gods-k3s-token",
        "gods-ingestion-port-forward-kubeconfig",
    }
    assert len(ubuntu_slice["database_restores"]) == 5
    assert len(ubuntu_slice["credentials"]) == 5
    assert all(item["node"] == "ubuntu" for item in ubuntu_slice["retained_paths"])


def test_manifest_rejects_cross_node_paths_shared_mount_assumptions_and_missing_ids() -> None:
    inventory = load_inventory(REPO_ROOT / "infra" / "ansible" / "inventory.example.yml")
    manifest = _manifest(inventory)
    manifest["retained_paths"][0]["node"] = "ubuntu" if manifest["retained_paths"][0]["node"] == "vis-lab" else "vis-lab"
    with pytest.raises(InventoryError, match="wrong node"):
        validate_retained_manifest(manifest, inventory)

    manifest = _manifest(inventory)
    manifest["retained_paths"].pop()
    with pytest.raises(InventoryError, match="coverage mismatch"):
        validate_retained_manifest(manifest, inventory)

    manifest = _manifest(inventory)
    manifest["credentials"][0]["path"] = "/mnt/data/shared/secret"
    with pytest.raises(InventoryError, match="under its recovery root"):
        validate_retained_manifest(manifest, inventory)


def test_manifest_cli_returns_structural_coverage_and_node_scoped_slices(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory_path = REPO_ROOT / "infra" / "ansible" / "inventory.example.yml"
    inventory = load_inventory(inventory_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(inventory)), encoding="utf-8")

    result = cli.main([
        "lifecycle",
        "check-retained-manifest",
        "--manifest",
        str(manifest_path),
        "--inventory",
        str(inventory_path),
    ])
    coverage = json.loads(capsys.readouterr().out)
    assert result == 0
    assert coverage["status"] == "manifest_valid"

    result = cli.main([
        "lifecycle",
        "export-retained-node",
        "--manifest",
        str(manifest_path),
        "--inventory",
        str(inventory_path),
        "--node",
        "vis-lab",
    ])
    node_manifest = json.loads(capsys.readouterr().out)
    assert result == 0
    assert node_manifest["node"] == "vis-lab"
    assert node_manifest["retained_paths"]
    assert not node_manifest["database_restores"]
