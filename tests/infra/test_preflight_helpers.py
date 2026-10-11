from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / "infra" / "ansible" / "preflight_helpers.py"


def _run_paths(
    data_root: Path,
    k3s_data_dir: Path,
    data_directories: list[dict[str, str]],
    preserved_paths: list[str],
    preserved_patterns: list[str] | None = None,
    expected_root_uid: int = 0,
    expected_data_uid: int = 10001,
    expected_data_gid: int = 10001,
    storage_mount: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    assert HELPER.is_file(), "the executable preflight helper must be implemented"
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "paths",
            "--data-root",
            str(data_root),
            "--storage-mount",
            str(storage_mount or data_root.parent),
            "--k3s-data-dir",
            str(k3s_data_dir),
            "--data-directories-json",
            json.dumps(data_directories),
            "--preserved-paths-json",
            json.dumps(preserved_paths),
            "--preserved-patterns-json",
            json.dumps(preserved_patterns or []),
            "--expected-root-uid",
            str(expected_root_uid),
            "--expected-data-uid",
            str(expected_data_uid),
            "--expected-data-gid",
            str(expected_data_gid),
            "--expected-k3s-version",
            "v1.36.2+k3s1",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result, json.loads(result.stdout) if result.stdout.strip() else {}


def _directory(path: Path, role: str = "objects") -> dict[str, str]:
    return {"path": str(path), "role": role, "capacity": "1Gi"}


def test_path_guard_allows_new_gods_paths_separate_from_preserved_data(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "gods-mlops"
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [str(tmp_path / "minio-data"), str(tmp_path / "labclip-cache")],
    )

    assert result.returncode == 0, result.stderr
    assert document["status"] == "ready"
    assert document["blockers"] == []
    assert document["root_owned"] is False
    assert document["owner_marker_valid"] is True


def test_path_guard_accepts_a_root_owned_marker_as_native_boolean(tmp_path: Path) -> None:
    data_root = tmp_path / "gods-mlops"
    data_root.mkdir()
    marker = {
        "schema_version": 1,
        "owner": "gods-mlops",
        "data_root": str(data_root),
        "k3s_data_dir": str(data_root / "k3s"),
        "k3s_version": "v1.36.2+k3s1",
    }
    (data_root / ".gods-mlops-owner.json").write_text(json.dumps(marker), encoding="utf-8")
    objects = data_root / "objects"
    objects.mkdir()

    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [],
        expected_root_uid=os.getuid(),
        expected_data_uid=os.getuid(),
        expected_data_gid=os.getgid(),
    )

    assert result.returncode == 0, result.stderr
    assert document["root_owned"] is True
    assert document["owner_marker_valid"] is True
    assert type(document["owner_marker_valid"]) is bool


def test_path_guard_blocks_an_existing_root_with_a_wrong_marker(tmp_path: Path) -> None:
    data_root = tmp_path / "gods-mlops"
    data_root.mkdir()
    (data_root / ".gods-mlops-owner.json").write_text(
        json.dumps({"schema_version": 1, "owner": "labclip"}), encoding="utf-8"
    )

    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [],
        expected_root_uid=os.getuid(),
    )

    assert result.returncode != 0
    assert document["root_owned"] is True
    assert document["owner_marker_valid"] is False
    assert any("marker" in blocker for blocker in document["blockers"])


def test_path_guard_rejects_a_gods_root_inside_a_preserved_tree(
    tmp_path: Path,
) -> None:
    preserved_root = tmp_path / "minio-data"
    data_root = preserved_root / "gods-mlops"
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [str(preserved_root)],
    )

    assert result.returncode != 0
    assert any("preserved" in blocker for blocker in document["blockers"])


def test_path_guard_rejects_parent_traversal_even_when_a_path_normalizes_under_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "gods-mlops"
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects" / ".." / "cache")],
        [],
    )

    assert result.returncode != 0
    assert any("traversal" in blocker for blocker in document["blockers"])


def test_path_guard_rejects_symlink_ancestors_without_following_them(
    tmp_path: Path,
) -> None:
    preserved_root = tmp_path / "preserved"
    preserved_root.mkdir()
    link = tmp_path / "linked-root"
    link.symlink_to(preserved_root, target_is_directory=True)
    data_root = link / "gods-mlops"
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [str(preserved_root)],
    )

    assert result.returncode != 0
    assert any("symlink" in blocker for blocker in document["blockers"])
    assert not (preserved_root / "gods-mlops").exists()


def test_path_guard_rejects_storage_directories_outside_the_gods_root(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "gods-mlops"
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(tmp_path / "outside" / "objects")],
        [],
    )

    assert result.returncode != 0
    assert any("inside the Gods root" in blocker for blocker in document["blockers"])


def test_path_guard_rejects_a_gods_root_outside_the_approved_mount(tmp_path: Path) -> None:
    approved_mount = tmp_path / "data"
    approved_mount.mkdir()
    data_root = tmp_path / "gods-mlops"
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [],
        storage_mount=approved_mount,
    )

    assert result.returncode != 0
    assert any("beneath the approved storage mount" in blocker for blocker in document["blockers"])


def _write_existing_root_marker(data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    marker = {
        "schema_version": 1,
        "owner": "gods-mlops",
        "data_root": str(data_root),
        "k3s_data_dir": str(data_root / "k3s"),
        "k3s_version": "v1.36.2+k3s1",
    }
    (data_root / ".gods-mlops-owner.json").write_text(json.dumps(marker), encoding="utf-8")


def test_path_guard_rejects_a_regular_file_instead_of_an_existing_storage_directory(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "gods-mlops"
    _write_existing_root_marker(data_root)
    (data_root / "objects").write_text("preserve", encoding="utf-8")
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [],
        expected_root_uid=os.getuid(),
        expected_data_uid=os.getuid(),
        expected_data_gid=os.getgid(),
    )

    assert result.returncode != 0
    assert any("must be a directory" in blocker for blocker in document["blockers"])


@pytest.mark.parametrize("owner_field", ["uid", "gid"])
def test_path_guard_rejects_existing_storage_directories_with_wrong_owner(
    tmp_path: Path,
    owner_field: str,
) -> None:
    data_root = tmp_path / "gods-mlops"
    _write_existing_root_marker(data_root)
    objects = data_root / "objects"
    objects.mkdir()
    expected_uid = os.getuid() + (1 if owner_field == "uid" else 0)
    expected_gid = os.getgid() + (1 if owner_field == "gid" else 0)

    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(objects)],
        [],
        expected_root_uid=os.getuid(),
        expected_data_uid=expected_uid,
        expected_data_gid=expected_gid,
    )

    assert result.returncode != 0
    assert any("unexpected owner" in blocker for blocker in document["blockers"])


def test_path_guard_rejects_a_storage_path_matching_a_preserved_pattern(
    tmp_path: Path,
) -> None:
    preserved_pattern = str(tmp_path / "rtsp-video-loop_*")
    data_root = tmp_path / "rtsp-video-loop_app" / "gods-mlops"
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [],
        [preserved_pattern],
    )

    assert result.returncode != 0
    assert any("preserved" in blocker for blocker in document["blockers"])


def test_path_guard_canonicalizes_preserved_symlink_ancestors(tmp_path: Path) -> None:
    preserved_target = tmp_path / "preserved-data"
    preserved_target.mkdir()
    preserved_alias = tmp_path / "labclip-data"
    preserved_alias.symlink_to(preserved_target, target_is_directory=True)
    data_root = preserved_target / "gods-mlops"
    result, document = _run_paths(
        data_root,
        data_root / "k3s",
        [_directory(data_root / "objects")],
        [str(preserved_alias)],
    )

    assert result.returncode != 0
    assert any("preserved" in blocker for blocker in document["blockers"])


def test_disk_probe_executes_df_and_parses_a_single_byte_count(tmp_path: Path) -> None:
    assert HELPER.is_file(), "the executable preflight helper must be implemented"
    result = subprocess.run(
        [sys.executable, str(HELPER), "disk", "--path", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    document = json.loads(result.stdout)
    assert document["status"] == "ready"
    assert isinstance(document["available_bytes"], int)
    assert document["available_bytes"] > 0
