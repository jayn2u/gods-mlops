from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pytest

from gods_mlops.lifecycle.inventory import InventoryError, plan_reclaim
from gods_mlops.lifecycle.recovery import (
    hash_tree,
    purge_targets_digest,
    validate_purge_targets,
    verify_recovery_artifacts,
    verify_retained,
)


def _inventory() -> dict:
    return {
        "schema_version": 1,
        "owner": "gods-mlops",
        "protected_paths": [
            "/data/jayn2u/minio",
            "/data/jayn2u/labclip-k3s",
            "/data/docker/volumes/rtsp-video-loop_app",
            "/data/jayn2u/gods-mlops-model-preparation",
        ],
        "protected_patterns": ["/data/docker/volumes/rtsp-video-loop_*"],
        "protected_services": ["rtsp-video-loop-app-1", "rtsp-video-loop-mediamtx-1"],
        "required_retained_paths": ["gods-mlops-objects", "gods-mlops-metadata-postgres"],
        "required_database_restores": ["gods-mlops-metadata-postgres"],
        "required_credentials": ["gods-k3s-token", "kubeflow-dex-operator"],
        "nodes": [
            {
                "name": "vis-lab",
                "role": "server",
                "reachable": True,
                "data_root": "/mnt/data/gods-mlops-runtime",
                "k3s_data_dir": "/mnt/data/gods-mlops-runtime/k3s",
                "service": "k3s",
                "ownership": {"owner": "gods-mlops"},
                "managed_paths": [
                    {"id": "gods-mlops-spool", "path": "/mnt/data/gods-mlops-runtime/spool"}
                ],
                "completion": {"plan_id": None, "steps": []},
            },
            {
                "name": "ubuntu",
                "role": "worker",
                "reachable": False,
                "data_root": "/data/jayn2u/gods-mlops",
                "k3s_data_dir": "/data/jayn2u/gods-mlops/k3s",
                "service": "k3s-agent",
                "ownership": {"owner": "gods-mlops"},
                "managed_paths": [
                    {"id": "gods-mlops-objects", "path": "/data/jayn2u/gods-mlops/objects"}
                ],
                "completion": {"plan_id": None, "steps": []},
            },
        ],
    }


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _retained_manifest(tmp_path: Path) -> dict:
    source = tmp_path / "source" / "objects"
    backup = tmp_path / "backup" / "objects"
    _write(source / "sample.bin", b"object bytes")
    _write(backup / "sample.bin", b"object bytes")
    source_hash = hash_tree(source)
    backup.chmod(0o700)

    source_dump = tmp_path / "backup" / "metadata.sql"
    restored_dump = tmp_path / "restore-check" / "metadata.sql"
    dump_hash = _write(source_dump, b"canonical pg_dump bytes")
    _write(restored_dump, b"canonical pg_dump bytes")
    source_dump.chmod(0o600)
    restored_dump.chmod(0o600)

    credential = tmp_path / "secrets" / "dex-operator"
    credential_hash = _write(credential, b"never include this in output")
    credential.chmod(0o600)

    uid = os.getuid()
    gid = os.getgid()
    return {
        "schema_version": 1,
        "owner": "gods-mlops",
        "retained_paths": [
            {
                "id": "gods-mlops-objects",
                "source_path": str(source),
                "backup_path": str(backup),
                "expected_tree_sha256": source_hash,
                "source_owner": {"uid": uid, "gid": gid},
                "backup_owner": {"uid": uid, "gid": gid},
                "backup_mode": "0700",
            }
        ],
        "database_restores": [
            {
                "id": "gods-mlops-metadata-postgres",
                "backup_dump_path": str(source_dump),
                "restored_dump_path": str(restored_dump),
                "expected_sha256": dump_hash,
                "expected_uid": uid,
                "expected_mode": "0600",
            }
        ],
        "credentials": [
            {
                "id": "gods-k3s-token",
                "path": str(credential),
                "expected_sha256": credential_hash,
                "expected_uid": uid,
                "expected_mode": "0600",
            },
            {
                "id": "kubeflow-dex-operator",
                "path": str(credential),
                "expected_sha256": credential_hash,
                "expected_uid": uid,
                "expected_mode": "0600",
            },
        ],
    }


def test_plan_reclaim_keeps_offline_nodes_pending_and_protects_existing_services() -> None:
    plan = plan_reclaim(_inventory())

    assert plan["dry_run"] is True
    assert plan["plan_valid"] is True
    assert {item["node"] for item in plan["actions"] if "node" in item} == {"vis-lab", "ubuntu"}
    ubuntu_action = next(item for item in plan["actions"] if item["node"] == "ubuntu")
    assert ubuntu_action["state"] == "pending_offline"
    assert any(item["kind"] == "node_unreachable" for item in plan["impacts"])
    assert {item["path"] for item in plan["retained_paths"]} >= {
        "/mnt/data/gods-mlops-runtime/spool",
        "/data/jayn2u/gods-mlops/objects",
    }
    assert "rtsp-video-loop-app-1" in plan["preserved_services"]
    assert {item["kind"] for item in plan["actions"]} >= {
        "drain_node",
        "stop_daemonset",
        "stop_service",
    }
    assert {item["name"] for item in plan["owned_daemonsets"] if "name" in item} == {
        "istio-cni-node",
    }
    assert "/data/docker/volumes/rtsp-video-loop_app" in plan["preserved_paths"]
    assert "/data/jayn2u/labclip-k3s" in plan["preserved_paths"]
    assert "/data/jayn2u/gods-mlops-model-preparation" in plan["preserved_paths"]


def test_plan_reclaim_blocks_an_ownership_mismatch_without_targeting_its_service() -> None:
    inventory = _inventory()
    inventory["nodes"][1]["ownership"] = {"owner": "labclip"}

    plan = plan_reclaim(inventory)

    assert plan["plan_valid"] is False
    assert not any(item.get("node") == "ubuntu" for item in plan["actions"])
    assert any(
        item["kind"] == "ownership_mismatch" and item["node"] == "ubuntu"
        for item in plan["impacts"]
    )
    assert any(
        item["path"] == "/data/jayn2u/gods-mlops/objects"
        for item in plan["retained_paths"]
    )


def test_plan_reclaim_is_stable_after_a_node_records_completion() -> None:
    inventory = _inventory()
    first = plan_reclaim(inventory)
    inventory["nodes"][0]["completion"] = {
        "plan_id": first["plan_id"],
        "steps": ["service_stopped"],
    }

    second = plan_reclaim(inventory)

    assert second["plan_id"] == first["plan_id"]
    assert next(item for item in second["actions"] if item["node"] == "vis-lab")[
        "state"
    ] == "completed"
    assert next(item for item in second["actions"] if item["node"] == "ubuntu")[
        "state"
    ] == "pending_offline"


def test_plan_reclaim_rejects_rtsp_paths_as_managed_targets() -> None:
    inventory = _inventory()
    inventory["nodes"][0]["managed_paths"].append(
        {"id": "bad-rtsp-target", "path": "/data/docker/volumes/rtsp-video-loop_app"}
    )

    with pytest.raises(InventoryError, match="outside"):
        plan_reclaim(inventory)


def test_verify_retained_checks_content_ownership_restore_and_secret_permissions(
    tmp_path: Path,
) -> None:
    manifest = _retained_manifest(tmp_path)
    required = {
        "retained_paths": ["gods-mlops-objects"],
        "database_restores": ["gods-mlops-metadata-postgres"],
        "credentials": ["gods-k3s-token", "kubeflow-dex-operator"],
    }

    result = verify_retained(manifest, requirements=required)

    assert result["status"] == "verified"
    assert result["failures"] == []
    assert "never include this in output" not in json.dumps(result)


def test_pre_stop_recovery_gate_hashes_the_backup_but_checks_live_source_ownership_only(
    tmp_path: Path,
) -> None:
    manifest = _retained_manifest(tmp_path)
    source = Path(manifest["retained_paths"][0]["source_path"])
    (source / "sample.bin").write_bytes(b"live database has changed before quiescence")
    required = {
        "retained_paths": ["gods-mlops-objects"],
        "database_restores": ["gods-mlops-metadata-postgres"],
        "credentials": ["gods-k3s-token", "kubeflow-dex-operator"],
    }

    pre_stop = verify_recovery_artifacts(manifest, requirements=required)
    post_stop = verify_retained(manifest, requirements=required)

    assert pre_stop["status"] == "verified"
    assert post_stop["status"] == "failed"
    assert "source_hash" in {failure["check"] for failure in post_stop["failures"]}


def test_live_database_path_uses_logical_restore_evidence_then_cold_hash(tmp_path: Path) -> None:
    manifest = _retained_manifest(tmp_path)
    database_path = manifest["retained_paths"][0]
    database_path.update({"id": "gods-mlops-katib-mysql", "role": "database"})
    for key in ("backup_path", "backup_owner", "backup_mode", "expected_tree_sha256"):
        database_path.pop(key)
    (Path(database_path["source_path"]) / "sample.bin").write_bytes(b"changed while database was live")
    required = {
        "retained_paths": ["gods-mlops-katib-mysql"],
        "database_restores": ["gods-mlops-metadata-postgres"],
        "credentials": ["gods-k3s-token", "kubeflow-dex-operator"],
    }

    pre_stop = verify_recovery_artifacts(manifest, requirements=required)
    cold = verify_retained(manifest, requirements=required)

    assert pre_stop["status"] == "verified"
    assert cold["status"] == "verified"
    assert cold["cold_retained_paths"]["gods-mlops-katib-mysql"]


def test_verify_retained_fails_closed_on_changed_backup_missing_database_restore_and_secret_mode(
    tmp_path: Path,
) -> None:
    manifest = _retained_manifest(tmp_path)
    Path(manifest["retained_paths"][0]["backup_path"], "sample.bin").write_bytes(b"changed")
    Path(manifest["database_restores"][0]["restored_dump_path"]).unlink()
    Path(manifest["credentials"][0]["path"]).chmod(0o644)

    result = verify_retained(manifest)

    assert result["status"] == "failed"
    assert {failure["check"] for failure in result["failures"]} >= {
        "backup_hash",
        "database_restore",
        "credential_permissions",
    }


def test_verify_retained_requires_coverage_for_every_inventory_item(tmp_path: Path) -> None:
    manifest = _retained_manifest(tmp_path)

    result = verify_retained(
        manifest,
        requirements={
            "retained_paths": ["gods-mlops-objects", "gods-mlops-katib-mysql"],
            "database_restores": ["gods-mlops-metadata-postgres", "gods-mlops-katib-mysql"],
            "credentials": ["gods-k3s-token", "kubeflow-dex-operator", "s3-credentials"],
        },
    )

    assert result["status"] == "failed"
    assert {failure["check"] for failure in result["failures"]} >= {
        "required_retained_path",
        "required_database_restore",
        "required_credential",
    }


def test_purge_requires_exact_targets_inside_owned_roots_and_excludes_preserved_data(
    tmp_path: Path,
) -> None:
    targets = {
        "schema_version": 1,
        "owner": "gods-mlops",
        "owned_roots": ["/data/jayn2u/gods-mlops"],
        "protected_paths": ["/data/jayn2u/labclip-k3s", "/data/docker/volumes/rtsp-video-loop_app"],
        "targets": [
            {
                "node": "ubuntu",
                "path": "/data/jayn2u/gods-mlops/objects",
                "owner_marker": "/data/jayn2u/gods-mlops/.gods-mlops-owner.json",
            }
        ],
    }

    fingerprint = purge_targets_digest(targets)
    validated = validate_purge_targets(targets, confirmation=fingerprint)

    assert validated["targets"][0]["path"].endswith("/objects")
    assert len(fingerprint) == 64
    assert validate_purge_targets(targets, confirmation=fingerprint) == validated


def test_purge_rejects_missing_confirmation_empty_list_and_protected_targets() -> None:
    targets = {
        "schema_version": 1,
        "owner": "gods-mlops",
        "owned_roots": ["/data/jayn2u/gods-mlops"],
        "protected_paths": ["/data/jayn2u/gods-mlops/metadata"],
        "targets": [
            {
                "node": "ubuntu",
                "path": "/data/jayn2u/gods-mlops/objects",
                "owner_marker": "/data/jayn2u/gods-mlops/.gods-mlops-owner.json",
            }
        ],
    }
    with_extraneous_root = dict(targets, targets=[])

    try:
        validate_purge_targets(targets)
        assert False, "purge validation must require the exact target digest"
    except ValueError as exc:
        assert "confirm-targets" in str(exc).lower()

    try:
        validate_purge_targets(with_extraneous_root, confirmation="0" * 64)
        assert False, "purge validation must reject an empty target list"
    except ValueError as exc:
        assert "target" in str(exc).lower()

    protected = dict(targets)
    protected["targets"] = [
        {
            "node": "ubuntu",
                "path": "/data/jayn2u/gods-mlops/metadata",
            "owner_marker": "/data/jayn2u/gods-mlops/.gods-mlops-owner.json",
        }
    ]
    try:
        validate_purge_targets(protected, confirmation="0" * 64)
        assert False, "purge validation must never admit preserved paths"
    except ValueError as exc:
        assert "protected" in str(exc).lower()


def test_tree_hash_records_symlink_text_without_following_and_rejects_special_files(tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("private bytes", encoding="utf-8")
    root = tmp_path / "tree"
    root.mkdir()
    link = root / "linked.txt"
    link.symlink_to(target)

    first_hash = hash_tree(root)
    target.write_text("different private bytes", encoding="utf-8")
    assert hash_tree(root) == first_hash

    special = root / "socket-placeholder"
    os.mkfifo(special)
    with pytest.raises(ValueError, match="special filesystem objects"):
        hash_tree(root)


def test_retained_verification_records_and_later_compares_cold_sqlite_state(tmp_path: Path) -> None:
    manifest = _retained_manifest(tmp_path)
    k3s_root = tmp_path / "k3s"
    for relative in ("server/db", "server/tls", "server/cred", "server/manifests"):
        (k3s_root / relative).mkdir(parents=True)
    (k3s_root / "agent/etc").mkdir(parents=True)
    (k3s_root / "agent/etc/config.toml").write_bytes(b"agent config")
    (k3s_root / "server/db/state.db").write_bytes(b"cold sqlite bytes")
    token_path = Path(manifest["credentials"][0]["path"])
    server_token = k3s_root / "server/token"
    server_token.write_bytes(token_path.read_bytes())
    server_token.chmod(0o600)
    manifest["credentials"][0].update({
        "source_path": str(server_token),
        "source_expected_uid": os.getuid(),
        "source_expected_mode": "0600",
    })
    (k3s_root / "server/tls/ca.crt").write_bytes(b"ca")
    (k3s_root / "server/cred/admin.kubeconfig").write_bytes(b"credential reference")
    (k3s_root / "server/manifests/coredns.yaml").write_bytes(b"manifest")
    manifest.update({
        "node": "vis-lab",
        "node_role": "server",
        "k3s_data_dir": str(k3s_root),
        "k3s_expected_uid": None,
    })

    captured = verify_retained(manifest)
    assert captured["status"] == "verified"
    assert captured["cold_k3s_state"]["datastore"] == "sqlite"

    manifest["cold_k3s_state_expected"] = True
    manifest["expected_cold_k3s_state"] = captured["cold_k3s_state"]
    assert verify_retained(manifest)["status"] == "verified"
    (k3s_root / "server/db/state.db").write_bytes(b"changed sqlite bytes")
    changed = verify_retained(manifest)
    assert changed["status"] == "failed"
    assert "k3s_authoritative_state" in {failure["check"] for failure in changed["failures"]}
