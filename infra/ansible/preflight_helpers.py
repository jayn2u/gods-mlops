"""Read-only helpers for host disk and data-path preflight checks."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any


def _normalized_absolute(raw_path: str, label: str, blockers: list[str]) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        blockers.append(f"{label} must be a non-empty absolute path")
        return None
    if "\x00" in raw_path:
        blockers.append(f"{label} contains a NUL byte")
        return None
    raw = Path(raw_path)
    if not raw.is_absolute():
        blockers.append(f"{label} must be absolute: {raw_path}")
        return None
    if ".." in raw.parts:
        blockers.append(f"{label} contains path traversal: {raw_path}")
        return None
    return Path(os.path.normpath(raw_path))


def _has_symlink_ancestor(path: Path) -> str | None:
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            return f"cannot inspect path ancestor {cursor}: {exc}"
        if stat.S_ISLNK(mode):
            return f"path has a symlink ancestor: {cursor}"
    return None


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _pattern_fixed_prefix(pattern: str) -> Path:
    parts = Path(pattern).parts
    fixed_parts: list[str] = []
    for part in parts:
        if any(marker in part for marker in "*?["):
            break
        fixed_parts.append(part)
    if not fixed_parts:
        return Path(Path(pattern).anchor or "/")
    return Path(*fixed_parts)


def _overlaps_pattern(candidate: Path, pattern: str) -> bool:
    prefix = _pattern_fixed_prefix(pattern)
    if candidate == prefix or candidate in prefix.parents:
        return True
    ancestor = candidate
    while True:
        if fnmatch.fnmatchcase(str(ancestor), pattern):
            return True
        if ancestor.parent == ancestor:
            break
        ancestor = ancestor.parent
    return False


def validate_paths(
    data_root_raw: str,
    storage_mount_raw: str,
    k3s_data_dir_raw: str,
    data_directories: list[dict[str, Any]],
    preserved_paths_raw: list[str],
    preserved_patterns_raw: list[str],
    owner_marker_name: str = ".gods-mlops-owner.json",
    expected_owner: str = "gods-mlops",
    expected_k3s_version: str = "v1.36.2+k3s1",
    expected_root_uid: int = 0,
    expected_data_uid: int = 10001,
    expected_data_gid: int = 10001,
) -> dict[str, Any]:
    """Check path ownership boundaries without creating or modifying files."""
    blockers: list[str] = []
    data_root = _normalized_absolute(data_root_raw, "Gods data root", blockers)
    storage_mount = _normalized_absolute(storage_mount_raw, "storage mount", blockers)
    k3s_data_dir = _normalized_absolute(k3s_data_dir_raw, "K3s data directory", blockers)
    preserved_paths: list[Path] = []
    preserved_patterns: list[str] = []

    for raw_path in preserved_paths_raw:
        path = _normalized_absolute(raw_path, "preserved path", blockers)
        if path is not None:
            preserved_paths.append(path)
            try:
                resolved = path.resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                blockers.append(f"cannot resolve preserved path {path}: {exc}")
            else:
                if resolved != path:
                    preserved_paths.append(resolved)
    for raw_pattern in preserved_patterns_raw:
        pattern = _normalized_absolute(raw_pattern, "preserved path pattern", blockers)
        if pattern is not None:
            preserved_patterns.append(str(pattern))
            fixed_prefix = _pattern_fixed_prefix(str(pattern))
            pattern_parts = Path(pattern).parts
            suffix_parts = pattern_parts[len(fixed_prefix.parts) :]
            try:
                resolved_prefix = fixed_prefix.resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                blockers.append(f"cannot resolve preserved path pattern prefix {fixed_prefix}: {exc}")
            else:
                if resolved_prefix != fixed_prefix and suffix_parts:
                    preserved_patterns.append(str(resolved_prefix.joinpath(*suffix_parts)))

    planned_paths: list[tuple[Path, str]] = []
    root_exists = False
    root_owned = False
    owner_marker_valid = True
    if data_root is not None:
        planned_paths.append((data_root, "Gods data root"))
        if data_root == Path(data_root.anchor):
            blockers.append("Gods data root cannot be a filesystem root")
        if storage_mount is not None and (data_root == storage_mount or storage_mount not in data_root.parents):
            blockers.append("Gods data root must be strictly beneath the approved storage mount")
        root_symlink_error = _has_symlink_ancestor(data_root)
        if root_symlink_error:
            blockers.append(f"Gods data root: {root_symlink_error}")
            owner_marker_valid = False
        if not owner_marker_name or Path(owner_marker_name).name != owner_marker_name:
            blockers.append("ownership marker name must be a file name inside the Gods root")
            owner_marker_valid = False
        try:
            root_stat = data_root.lstat()
        except FileNotFoundError:
            root_stat = None
        except OSError as exc:
            root_stat = None
            blockers.append(f"cannot inspect Gods data root {data_root}: {exc}")

        if root_stat is not None:
            root_exists = True
            owner_marker_valid = False
            root_owned = stat.S_ISDIR(root_stat.st_mode) and root_stat.st_uid == expected_root_uid
            if not stat.S_ISDIR(root_stat.st_mode):
                blockers.append("existing Gods root is not a directory")
            if root_stat.st_uid != expected_root_uid:
                blockers.append("existing Gods root has an unexpected owner")
            if (
                owner_marker_name
                and Path(owner_marker_name).name == owner_marker_name
                and stat.S_ISDIR(root_stat.st_mode)
                and not root_symlink_error
            ):
                marker_path = data_root / owner_marker_name
                try:
                    marker_fd = os.open(marker_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                except OSError as exc:
                    owner_marker_valid = False
                    blockers.append(f"existing Gods root has no readable regular ownership marker: {exc}")
                else:
                    try:
                        marker_stat = os.fstat(marker_fd)
                        if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_uid != expected_root_uid:
                            owner_marker_valid = False
                            blockers.append("existing Gods ownership marker has an unexpected type or owner")
                        else:
                            raw_marker = os.read(marker_fd, 65537)
                            if len(raw_marker) > 65536:
                                owner_marker_valid = False
                                blockers.append("existing Gods ownership marker is too large")
                            else:
                                try:
                                    marker = json.loads(raw_marker.decode("utf-8"))
                                except (UnicodeDecodeError, json.JSONDecodeError):
                                    owner_marker_valid = False
                                    blockers.append("existing Gods ownership marker is not valid JSON")
                                else:
                                    if not isinstance(marker, dict):
                                        marker = {}
                                        marker_matches = False
                                    else:
                                        marker_matches = (
                                            root_owned
                                            and marker.get("schema_version") == 1
                                            and marker.get("owner") == expected_owner
                                            and marker.get("data_root") == str(data_root)
                                            and marker.get("k3s_data_dir") == str(k3s_data_dir)
                                            and marker.get("k3s_version") == expected_k3s_version
                                        )
                                    owner_marker_valid = marker_matches
                                    if not marker_matches:
                                        blockers.append("existing Gods ownership marker does not match the locked deployment")
                    finally:
                        os.close(marker_fd)
    if k3s_data_dir is not None:
        planned_paths.append((k3s_data_dir, "K3s data directory"))
        if data_root is not None and (k3s_data_dir == data_root or data_root not in k3s_data_dir.parents):
            blockers.append("K3s data directory must be strictly inside the Gods root")

    normalized_directories: list[dict[str, str]] = []
    for index, entry in enumerate(data_directories):
        if not isinstance(entry, dict):
            blockers.append(f"storage directory {index} must be an object")
            continue
        path = _normalized_absolute(entry.get("path", ""), f"storage directory {index}", blockers)
        if path is None:
            continue
        role = str(entry.get("role", f"storage-{index}"))
        if data_root is not None and (path == data_root or data_root not in path.parents):
            blockers.append(f"storage directory {role} must be strictly inside the Gods root")
        planned_paths.append((path, f"storage directory {role}"))
        normalized_directories.append(
            {
                "path": str(path),
                "role": role,
                "capacity": str(entry.get("capacity", "")),
            }
        )
        symlink_error = _has_symlink_ancestor(path)
        if symlink_error:
            blockers.append(f"storage directory {role}: {symlink_error}")
            continue
        try:
            directory_stat = path.lstat()
        except FileNotFoundError:
            directory_stat = None
        except OSError as exc:
            directory_stat = None
            blockers.append(f"cannot inspect storage directory {role} {path}: {exc}")
        if directory_stat is not None:
            if not stat.S_ISDIR(directory_stat.st_mode):
                blockers.append(f"existing storage directory {role} must be a directory, not a file or symlink")
            if directory_stat.st_uid != expected_data_uid or directory_stat.st_gid != expected_data_gid:
                blockers.append(
                    f"existing storage directory {role} has unexpected owner/group; "
                    f"expected {expected_data_uid}:{expected_data_gid}, got "
                    f"{directory_stat.st_uid}:{directory_stat.st_gid}"
                )

    for path, label in planned_paths:
        symlink_error = _has_symlink_ancestor(path)
        if symlink_error:
            blockers.append(f"{label}: {symlink_error}")
        for preserved in preserved_paths:
            if _overlaps(path, preserved):
                blockers.append(f"{label} overlaps preserved path {preserved}")
        for pattern in preserved_patterns:
            if _overlaps_pattern(path, pattern):
                blockers.append(f"{label} overlaps preserved path pattern {pattern}")

    directory_paths = [Path(item["path"]) for item in normalized_directories]
    for index, path in enumerate(directory_paths):
        for other in directory_paths[index + 1 :]:
            if _overlaps(path, other):
                blockers.append(f"storage directories overlap: {path} and {other}")
        if k3s_data_dir is not None and _overlaps(path, k3s_data_dir):
            blockers.append(f"storage directory overlaps K3s data directory: {path}")

    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "data_root": str(data_root) if data_root is not None else None,
        "storage_mount": str(storage_mount) if storage_mount is not None else None,
        "root_exists": root_exists,
        "root_owned": root_owned,
        "owner_marker_valid": owner_marker_valid,
        "k3s_data_dir": str(k3s_data_dir) if k3s_data_dir is not None else None,
        "data_directories": normalized_directories,
    }


def probe_disk(path: str) -> dict[str, Any]:
    """Read available bytes using GNU df's compatible machine output mode."""
    try:
        result = subprocess.run(
            ["df", "-B1", "--no-sync", "--output=avail", path],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "blocked", "path": path, "available_bytes": 0, "error": str(exc)}

    if result.returncode != 0:
        return {
            "status": "blocked",
            "path": path,
            "available_bytes": 0,
            "error": result.stderr.strip() or f"df exited {result.returncode}",
        }
    try:
        byte_counts = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        if len(byte_counts) != 1:
            raise ValueError(f"expected one byte-count row, got {len(byte_counts)}")
        available_bytes = byte_counts[0]
    except ValueError:
        return {
            "status": "blocked",
            "path": path,
            "available_bytes": 0,
            "error": f"df returned a non-integer byte count: {result.stdout!r}",
        }
    if available_bytes < 0:
        return {
            "status": "blocked",
            "path": path,
            "available_bytes": available_bytes,
            "error": "df returned a negative available-byte count",
        }
    return {"status": "ready", "path": path, "available_bytes": available_bytes}


def _json_argument(raw: str, label: str) -> list[Any]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    paths_parser = subparsers.add_parser("paths")
    paths_parser.add_argument("--data-root", required=True)
    paths_parser.add_argument("--storage-mount", required=True)
    paths_parser.add_argument("--k3s-data-dir", required=True)
    paths_parser.add_argument("--data-directories-json", required=True)
    paths_parser.add_argument("--preserved-paths-json", required=True)
    paths_parser.add_argument("--preserved-patterns-json", required=True)
    paths_parser.add_argument("--owner-marker-name", default=".gods-mlops-owner.json")
    paths_parser.add_argument("--expected-owner", default="gods-mlops")
    paths_parser.add_argument("--expected-k3s-version", default="v1.36.2+k3s1")
    paths_parser.add_argument("--expected-root-uid", type=int, default=0)
    paths_parser.add_argument("--expected-data-uid", type=int, default=10001)
    paths_parser.add_argument("--expected-data-gid", type=int, default=10001)

    disk_parser = subparsers.add_parser("disk")
    disk_parser.add_argument("--path", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "disk":
            result = probe_disk(args.path)
        else:
            result = validate_paths(
                data_root_raw=args.data_root,
                storage_mount_raw=args.storage_mount,
                k3s_data_dir_raw=args.k3s_data_dir,
                data_directories=_json_argument(args.data_directories_json, "data_directories"),
                preserved_paths_raw=_json_argument(args.preserved_paths_json, "preserved_paths"),
                preserved_patterns_raw=_json_argument(args.preserved_patterns_json, "preserved_patterns"),
                owner_marker_name=args.owner_marker_name,
                expected_owner=args.expected_owner,
                expected_k3s_version=args.expected_k3s_version,
                expected_root_uid=args.expected_root_uid,
                expected_data_uid=args.expected_data_uid,
                expected_data_gid=args.expected_data_gid,
            )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "blocked", "blockers": [str(exc)]}

    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
