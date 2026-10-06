from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import struct
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit
import zlib

import pytest
import yaml

from gods_mlops.lifecycle.inventory import (
    load_inventory,
    plan_reclaim,
    slice_retained_manifest_for_node,
    validate_retained_manifest,
)
from gods_mlops.lifecycle.recovery import (
    validate_daemonset_snapshot,
    verify_restored_daemonsets,
)


_CONFIG_ENV = "GODS_MLOPS_LIFECYCLE_E2E_CONFIG"
_ACK_ENV = "GODS_MLOPS_LIFECYCLE_E2E_ROOT_ACK"
_STAGES = (
    "deploy",
    "reapply",
    "baseline",
    "reclaim_interrupted",
    "reclaim_resume",
    "reclaim_repeat",
    "reconnect",
    "continuity",
    "rtsp_preserved",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parents[2]


def _open_absolute_directory(path: str) -> tuple[int, os.stat_result]:
    if not isinstance(path, str) or not path.startswith("/") or "\x00" in path:
        raise ValueError("evidence root must be an absolute directory")
    parts = path.split("/")[1:]
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("evidence root path is not canonical")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for part in parts:
            next_fd = os.open(part, flags | nofollow, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("evidence root must be a directory")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ValueError("evidence root must be current-user-owned and private")
        return fd, metadata
    except OSError as exc:
        os.close(fd)
        raise ValueError("evidence root cannot be opened without following symlinks") from exc
    except Exception:
        os.close(fd)
        raise


def _read_private_file(path: str, *, exact_mode: int | None = None) -> bytes:
    if not isinstance(path, str) or not path.startswith("/") or "\x00" in path:
        raise ValueError("configuration path must be absolute")
    parts = path.split("/")[1:]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("configuration path is not canonical")
    directory = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"
    directory_fd, _ = _open_absolute_directory_for_read(directory)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(parts[-1], flags, dir_fd=directory_fd)
    except OSError as exc:
        os.close(directory_fd)
        raise ValueError("configuration must be a regular file without symlinks") from exc
    os.close(directory_fd)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("configuration must be a regular file")
        if metadata.st_uid != os.geteuid():
            raise ValueError("configuration must be owned by the current user")
        mode = stat.S_IMODE(metadata.st_mode)
        if exact_mode is not None and mode != exact_mode:
            raise ValueError("configuration file must have mode 0600")
        if exact_mode is None and mode & 0o077:
            raise ValueError("evidence files must be private to the current user")
        return _read_fd(fd, limit=_MAX_ARTIFACT_BYTES)
    except OSError as exc:
        raise ValueError("configuration file cannot be read safely") from exc
    finally:
        os.close(fd)


def _open_absolute_directory_for_read(path: str) -> tuple[int, os.stat_result]:
    if not isinstance(path, str) or not path.startswith("/") or "\x00" in path:
        raise ValueError("file parent must be an absolute directory")
    parts = path.split("/")[1:]
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("file parent path is not canonical")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for part in parts:
            next_fd = os.open(part, flags | nofollow, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("file parent must be a directory")
        return fd, metadata
    except OSError as exc:
        os.close(fd)
        raise ValueError("file parent cannot be opened without following symlinks") from exc
    except Exception:
        os.close(fd)
        raise


def _read_fd(fd: int, *, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, limit + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ValueError("evidence file exceeds the configured size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _relative_parts(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value or "\x00" in value:
        raise ValueError("artifact path must be a safe relative path")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("artifact path must be a safe relative path")
    return parts


def _safe_reference(value: Any) -> bool:
    try:
        _relative_parts(value)
        return True
    except ValueError:
        return False


def _read_beneath(root_fd: int, relative_path: str) -> bytes:
    parts = _relative_parts(relative_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = flags | getattr(os, "O_DIRECTORY", 0) | nofollow
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            metadata = os.fstat(fd)
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise ValueError("artifact parent directory is not private and user-owned")
        file_fd = os.open(parts[-1], flags | nofollow, dir_fd=fd)
    except OSError as exc:
        raise ValueError("artifact must be a regular file without symlinks") from exc
    finally:
        os.close(fd)
    try:
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("artifact must be a regular file")
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("artifact must be current-user-owned with mode 0600")
        return _read_fd(file_fd, limit=_MAX_ARTIFACT_BYTES)
    finally:
        os.close(file_fd)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _parse_json_object(payload: bytes, *, what: str) -> dict[str, Any]:
    try:
        result = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{what} is not valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{what} must be a JSON object")
    return result


def _validate_lifecycle_config(config: dict[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "fixture_id",
        "expected_source_commit",
        "evidence_root",
        "bundle_path",
        "bundle_sha256",
        "plan_id",
        "snapshot_sha256",
    }
    if set(config) != expected_keys or config.get("schema_version") != 1:
        raise ValueError("lifecycle config has an unsupported schema")
    try:
        parsed_uuid = uuid.UUID(config["fixture_id"])
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("lifecycle config fixture_id must be a UUID") from exc
    if str(parsed_uuid) != config["fixture_id"]:
        raise ValueError("lifecycle config fixture_id must use canonical UUID spelling")
    if not isinstance(config["expected_source_commit"], str) or not _COMMIT.fullmatch(config["expected_source_commit"]):
        raise ValueError("lifecycle config source commit must be a full Git SHA")
    for name in ("bundle_sha256", "plan_id", "snapshot_sha256"):
        if not _is_sha256(config[name]):
            raise ValueError(f"lifecycle config {name} must be a SHA-256 digest")
    _relative_parts(config["bundle_path"])
    if not isinstance(config["evidence_root"], str) or not config["evidence_root"].startswith("/"):
        raise ValueError("evidence root must be an absolute directory")


def _load_lifecycle_config(environ: Mapping[str, str]) -> dict[str, Any] | None:
    """Read the opt-in pointer only; this function never starts lifecycle work."""
    config_path = environ.get(_CONFIG_ENV)
    if config_path is None:
        return None
    payload = _read_private_file(config_path, exact_mode=0o600)
    config = _parse_json_object(payload, what="lifecycle config")
    _validate_lifecycle_config(config)
    root_fd, _ = _open_absolute_directory(config["evidence_root"])
    os.close(root_fd)
    return config


def _load_bound_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    """Load a root-acknowledgeable bundle and verify every file before parsing it."""
    if not isinstance(config, dict):
        raise ValueError("lifecycle config must be an object")
    _validate_lifecycle_config(config)
    root_fd, _ = _open_absolute_directory(config.get("evidence_root"))
    try:
        bundle_bytes = _read_beneath(root_fd, config.get("bundle_path"))
        if not _is_sha256(config.get("bundle_sha256")) or _sha256(bundle_bytes) != config["bundle_sha256"]:
            raise ValueError("bundle hash mismatch")
        bundle = _parse_json_object(bundle_bytes, what="lifecycle bundle")
        if bundle.get("schema_version") != 1:
            raise ValueError("lifecycle bundle has an unsupported schema")
        for key in ("fixture_id", "source_commit", "plan_id", "snapshot_sha256"):
            expected_key = "expected_source_commit" if key == "source_commit" else key
            if bundle.get(key) != config.get(expected_key):
                raise ValueError(f"bundle {key} does not match protected config")

        entries = bundle.get("artifacts")
        if not isinstance(entries, list) or not entries:
            raise ValueError("lifecycle bundle must reference native artifacts")
        values: dict[str, Any] = {}
        paths: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"id", "path", "sha256", "media_type"}:
                raise ValueError("lifecycle artifact reference is malformed")
            artifact_id = entry["id"]
            if not isinstance(artifact_id, str) or not _ARTIFACT_ID.fullmatch(artifact_id) or artifact_id in values:
                raise ValueError("lifecycle artifact identifiers must be unique safe labels")
            _relative_parts(entry["path"])
            if not _is_sha256(entry["sha256"]):
                raise ValueError("lifecycle artifact hash must be a SHA-256 digest")
            payload = _read_beneath(root_fd, entry["path"])
            if _sha256(payload) != entry["sha256"]:
                raise ValueError(f"artifact hash mismatch: {artifact_id}")
            media_type = entry["media_type"]
            try:
                if media_type == "json":
                    value: Any = json.loads(payload)
                elif media_type == "yaml":
                    documents = list(yaml.safe_load_all(payload))
                    value = documents[0] if len(documents) == 1 else documents
                elif media_type == "bytes":
                    value = payload
                else:
                    raise ValueError("unsupported artifact media type")
            except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
                raise ValueError(f"artifact is malformed: {artifact_id}") from exc
            values[artifact_id] = value
            paths[artifact_id] = entry["path"]
            hashes[artifact_id] = entry["sha256"]
        return {
            "bundle": bundle,
            "artifact_values": values,
            "artifact_paths": paths,
            "artifact_hashes": hashes,
            "bundle_sha256": config["bundle_sha256"],
            "evidence_root": config["evidence_root"],
        }
    finally:
        os.close(root_fd)


def _iso_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256(encoded)


def _artifact_value(artifacts: dict[str, Any], artifact_id: Any) -> Any:
    values = artifacts.get("artifact_values")
    if not isinstance(values, dict) or not isinstance(artifact_id, str):
        return None
    return values.get(artifact_id)


def _artifact_is_still_bound(artifacts: dict[str, Any], artifact_id: Any) -> bool:
    if not isinstance(artifact_id, str):
        return False
    paths = artifacts.get("artifact_paths")
    hashes = artifacts.get("artifact_hashes")
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        return False
    relative_path = paths.get(artifact_id)
    expected_hash = hashes.get(artifact_id)
    if not isinstance(relative_path, str) or not _is_sha256(expected_hash):
        return False
    try:
        root_fd, _ = _open_absolute_directory(artifacts.get("evidence_root"))
        try:
            return _sha256(_read_beneath(root_fd, relative_path)) == expected_hash
        finally:
            os.close(root_fd)
    except (OSError, ValueError):
        return False


def _add_error(errors: dict[str, list[str]], case: str, code: str) -> None:
    if code not in errors[case]:
        errors[case].append(code)


def _native_receipt_binding_errors(
    receipt: dict[str, Any],
    bundle: dict[str, Any],
    artifacts: dict[str, Any],
    observation_id: Any,
) -> list[str]:
    errors: list[str] = []
    output_ids = receipt.get("native_output_artifact_ids")
    typed_id = receipt.get("native_receipt_artifact_id")
    typed = _artifact_value(artifacts, typed_id)
    if (
        not isinstance(output_ids, list)
        or not all(isinstance(item, str) for item in output_ids)
        or len(output_ids) != len(set(output_ids))
        or observation_id not in output_ids
        or not isinstance(typed_id, str)
        or typed_id not in output_ids
        or len(set(output_ids) - {observation_id, typed_id}) < 1
    ):
        return ["native_outputs_not_independently_bound"]
    if not isinstance(typed, dict):
        return ["typed_native_receipt_missing"]
    artifact_hashes = artifacts.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        return ["native_artifact_hash_index_invalid"]
    raw_ids = set(output_ids) - {observation_id, typed_id}
    expected = {
        "schema_version": 1,
        "kind": "native_stage_receipt",
        "stage": receipt.get("stage"),
        "attempt_id": receipt.get("attempt_id"),
        "source_commit": bundle.get("source_commit"),
        "plan_id": bundle.get("plan_id"),
        "exit_status": receipt.get("exit_status"),
        "started_at": receipt.get("started_at"),
        "ended_at": receipt.get("ended_at"),
        "output_artifact_ids": sorted(raw_ids),
        "output_sha256": {identifier: artifact_hashes.get(identifier) for identifier in raw_ids},
    }
    if any(typed.get(key) != value for key, value in expected.items()):
        errors.append("typed_native_receipt_binding_mismatch")
    return errors


def _native_provenance_errors(
    provenance: Any,
    receipts: list[dict[str, Any]],
    artifacts: dict[str, Any],
) -> list[str]:
    if not isinstance(provenance, dict) or provenance.get("kind") != "native":
        return []
    errors: list[str] = []
    origin = provenance.get("origin")
    normalized_origin = origin.casefold() if isinstance(origin, str) else ""
    if (
        provenance.get("execution_mode") != "recorded_native"
        or provenance.get("offline_fixture") is True
        or provenance.get("fixture_only") is True
        or "source test fixture" in normalized_origin
        or "offline fixture" in normalized_origin
    ):
        errors.append("native_provenance_contradicts_offline_fixture")
    for receipt in receipts:
        recipe = _artifact_value(artifacts, receipt.get("recipe_artifact_id"))
        native_receipt = _artifact_value(artifacts, receipt.get("native_receipt_artifact_id"))
        if isinstance(recipe, dict) and recipe.get("fixture_only") is True:
            errors.append("native_recipe_marked_fixture_only")
            break
        if isinstance(native_receipt, dict) and native_receipt.get("fixture_only") is True:
            errors.append("native_receipt_marked_fixture_only")
            break
        output_ids = receipt.get("native_output_artifact_ids")
        if isinstance(output_ids, list) and any(
            isinstance(_artifact_value(artifacts, identifier), bytes)
            and b"synthetic native output for " in _artifact_value(artifacts, identifier)
            for identifier in output_ids
        ):
            errors.append("native_output_marked_fixture_only")
            break
    return errors


def _hash_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(key, str) and key and _is_sha256(digest) for key, digest in value.items()
    )


def _identity_digest_mapping(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(
        isinstance(name, str)
        and name
        and isinstance(digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
        for name, digest in value.items()
    )


def _valid_input_versions(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(
        isinstance(name, str)
        and bool(name)
        and isinstance(record, dict)
        and isinstance(record.get("version"), str)
        and bool(record["version"])
        and _is_sha256(record.get("sha256"))
        for name, record in value.items()
    )


def _deployment_fingerprint(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    cluster = value.get("cluster")
    nodes = value.get("nodes")
    pv_bindings = value.get("pv_bindings")
    if not isinstance(cluster, dict) or not isinstance(nodes, dict) or not isinstance(pv_bindings, list):
        return None
    normalized_nodes: dict[str, Any] = {}
    for name, node in nodes.items():
        if not isinstance(node, dict):
            return None
        normalized_nodes[str(name)] = {
            key: node.get(key)
            for key in (
                "uid",
                "data_root",
                "k3s_data_dir",
                "root_uid",
                "data_root_uid",
                "k3s_data_dir_uid",
                "data_root_mode",
                "k3s_data_dir_mode",
                "owner_marker_uid",
                "cluster_marker_uid",
                "owner_marker_mode",
                "cluster_marker_mode",
                "owner_marker",
                "cluster_marker",
                "service_exec_start_sha256",
            )
        }
    normalized_pvs: list[dict[str, Any]] = []
    for binding in pv_bindings:
        if not isinstance(binding, dict):
            return None
        normalized_pvs.append(
            {key: binding.get(key) for key in ("id", "volume_name", "claim", "node", "local_path", "phase", "reclaim_policy")}
        )
    return {
        "cluster_uid": cluster.get("uid"),
        "nodes": normalized_nodes,
        "pv_bindings": sorted(normalized_pvs, key=lambda item: str(item.get("id"))),
        "credential_identity_sha256": value.get("credential_identity_sha256"),
        "image_digests": value.get("image_digests"),
    }


def _check_deploy_observation(
    observation: Any,
    inventory: dict[str, Any],
    bundle: dict[str, Any],
    artifacts: dict[str, Any],
    deploy_receipt: dict[str, Any] | None,
    errors: list[str],
) -> None:
    if not isinstance(observation, dict):
        errors.append("deploy_observation_missing")
        return
    expected_nodes = {node["name"]: node for node in inventory["nodes"]}
    preflight = observation.get("preflight")
    cluster = observation.get("cluster")
    nodes = observation.get("nodes")
    identities = bundle.get("identities")
    if not isinstance(identities, dict):
        identities = {}
    kubeconfig_identity = identities.get("kubeconfig")
    input_versions = identities.get("input_versions")
    k3s_identity = input_versions.get("k3s") if isinstance(input_versions, dict) else None
    expected_k3s_version = k3s_identity.get("version") if isinstance(k3s_identity, dict) else None
    if (
        not isinstance(preflight, dict)
        or preflight.get("status") != "verified"
        or preflight.get("authenticated") is not True
        or preflight.get("become_uid") != 0
        or preflight.get("source_commit") != bundle.get("source_commit")
        or preflight.get("image_digests") != identities.get("images")
        or not _is_sha256(preflight.get("render_sha256"))
    ):
        errors.append("authenticated_preflight_missing")
    if not isinstance(cluster, dict) or not cluster.get("uid") or not cluster.get("context"):
        errors.append("generated_kubeconfig_identity_missing")
    kubeconfig = observation.get("generated_kubeconfig")
    kubeconfig_artifact_id = kubeconfig.get("artifact_id") if isinstance(kubeconfig, dict) else None
    kubeconfig_facts = _artifact_value(artifacts, kubeconfig_artifact_id)
    deploy_outputs = deploy_receipt.get("native_output_artifact_ids") if isinstance(deploy_receipt, dict) else None
    if not isinstance(deploy_outputs, list):
        deploy_outputs = []
    facts_match = (
        isinstance(kubeconfig_facts, dict)
        and isinstance(kubeconfig_identity, dict)
        and _safe_reference(kubeconfig_identity.get("reference"))
        and kubeconfig_facts.get("schema_version") == 1
        and all(
            kubeconfig_facts.get(key) == kubeconfig_identity.get(key)
            for key in ("reference", "sha256", "mode", "context", "server", "certificate_authority_sha256", "cluster_uid")
        )
        and kubeconfig_artifact_id in deploy_outputs
    )
    cluster_match = (
        isinstance(cluster, dict)
        and isinstance(kubeconfig_facts, dict)
        and cluster.get("uid") == kubeconfig_facts.get("cluster_uid")
        and cluster.get("context") == kubeconfig_facts.get("context")
        and cluster.get("server") == kubeconfig_facts.get("server")
        and cluster.get("certificate_authority_sha256") == kubeconfig_facts.get("certificate_authority_sha256")
    )
    if (
        not isinstance(kubeconfig, dict)
        or not isinstance(kubeconfig_identity, dict)
        or kubeconfig.get("mode") != "0600"
        or kubeconfig.get("reference") != kubeconfig_identity.get("reference")
    ):
        errors.append("generated_kubeconfig_not_bound")
    elif (
        not _is_sha256(kubeconfig.get("sha256"))
        or not _is_sha256(kubeconfig.get("certificate_authority_sha256"))
        or not facts_match
        or not cluster_match
        or any(kubeconfig.get(key) != kubeconfig_facts.get(key) for key in ("sha256", "context", "server", "certificate_authority_sha256", "cluster_uid"))
    ):
        errors.append("generated_kubeconfig_not_bound")
    if not isinstance(nodes, dict) or set(nodes) != set(expected_nodes):
        errors.append("node_identity_set_mismatch")
    else:
        for name, expected in expected_nodes.items():
            actual = nodes[name]
            if not isinstance(actual, dict):
                errors.append("node_root_or_readiness_mismatch")
                break
            owner_marker = actual.get("owner_marker")
            cluster_marker = actual.get("cluster_marker")
            if (
                actual.get("uid") is None
                or actual.get("ready") is not True
                or actual.get("data_root") != expected["data_root"]
                or actual.get("k3s_data_dir") != expected["k3s_data_dir"]
                or actual.get("service") != expected["service"]
                or actual.get("root_uid") != 0
                or actual.get("data_root_uid") != 0
                or actual.get("k3s_data_dir_uid") != 0
                or actual.get("data_root_mode") != "0755"
                or actual.get("k3s_data_dir_mode") != "0700"
                or actual.get("owner_marker_uid") != 0
                or actual.get("cluster_marker_uid") != 0
                or actual.get("owner_marker_mode") != "0644"
                or actual.get("cluster_marker_mode") != "0600"
                or not isinstance(owner_marker, dict)
                or owner_marker.get("schema_version") != 1
                or owner_marker.get("owner") != "gods-mlops"
                or owner_marker.get("data_root") != expected["data_root"]
                or owner_marker.get("k3s_data_dir") != expected["k3s_data_dir"]
                or not owner_marker.get("k3s_version")
                or owner_marker.get("k3s_version") != expected_k3s_version
                or not isinstance(cluster_marker, dict)
                or cluster_marker.get("schema_version") != 1
                or cluster_marker.get("owner") != "gods-mlops"
                or cluster_marker.get("data_root") != expected["data_root"]
                or cluster_marker.get("k3s_data_dir") != expected["k3s_data_dir"]
                or cluster_marker.get("k3s_version") != owner_marker.get("k3s_version")
                or not _is_sha256(actual.get("service_exec_start_sha256"))
            ):
                errors.append("node_root_or_readiness_mismatch")
                break
    expected_pvs = {
        item["id"]: item for item in inventory.get("retained_paths", []) if isinstance(item, dict) and item.get("claim")
    }
    bindings = observation.get("pv_bindings")
    if not isinstance(bindings, list):
        errors.append("retained_pv_bindings_missing")
    else:
        actual_pvs = {item.get("id"): item for item in bindings if isinstance(item, dict)}
        if len(actual_pvs) != len(bindings) or set(actual_pvs) != set(expected_pvs):
            errors.append("retained_pv_binding_set_mismatch")
        else:
            for identifier, expected in expected_pvs.items():
                actual = actual_pvs[identifier]
                if any(
                    actual.get(key) != expected_value
                    for key, expected_value in {
                        "volume_name": identifier,
                        "claim": expected["claim"],
                        "node": expected["node"],
                        "local_path": expected["path"],
                        "phase": "Bound",
                        "reclaim_policy": "Retain",
                    }.items()
                ):
                    errors.append("retained_pv_binding_changed")
                    break
    if not _identity_digest_mapping(identities.get("images")) or observation.get("image_digests") != identities.get("images"):
        errors.append("image_identity_mismatch")
    if not _valid_input_versions(identities.get("input_versions")) or observation.get("input_versions") != identities.get("input_versions"):
        errors.append("input_identity_mismatch")
    readiness = observation.get("workload_readiness")
    if not isinstance(readiness, dict) or readiness.get("status") != "ready" or readiness.get("unready") != []:
        errors.append("workload_readiness_missing")


def _check_reapply_observation(observation: Any, deploy: Any, errors: list[str]) -> None:
    if not isinstance(observation, dict):
        errors.append("reapply_observation_missing")
        return
    before = _deployment_fingerprint(observation.get("before"))
    after = _deployment_fingerprint(observation.get("after"))
    baseline = _deployment_fingerprint(deploy)
    after_readiness = observation.get("after", {}).get("workload_readiness") if isinstance(observation.get("after"), dict) else None
    if before is None or after is None or baseline is None or before != after or after != baseline:
        errors.append("reapply_identity_or_retained_binding_changed")
    if not isinstance(after_readiness, dict) or after_readiness.get("status") != "ready" or after_readiness.get("unready") != []:
        errors.append("reapply_readiness_missing")


def _expected_report_checks(manifest: dict[str, Any], inventory: dict[str, Any], node: str, operation: str) -> dict[str, int]:
    sliced = slice_retained_manifest_for_node(manifest, inventory, node)
    if operation == "verify-recovery-artifacts":
        return {
            "source_ownership": len(sliced["retained_paths"]),
            "backup_paths": sum(item.get("role") != "database" for item in sliced["retained_paths"]),
            "database_restores": len(sliced["database_restores"]),
            "credentials": len(sliced["credentials"]),
        }
    return {
        "retained_paths": len(sliced["retained_paths"]),
        "database_restores": len(sliced["database_restores"]),
        "credentials": len(sliced["credentials"]),
    }


def _node_report_manifest(
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    node: str,
    operation: str,
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    sliced = slice_retained_manifest_for_node(manifest, inventory, node)
    if operation == "verify-retained":
        if not isinstance(state, dict):
            raise ValueError("cold verification report requires the native stopped-state record")
        cold_state = state.get("cold_k3s_state")
        cold_paths = state.get("cold_retained_paths")
        if not isinstance(cold_state, dict) or not isinstance(cold_paths, dict):
            raise ValueError("cold verification report is missing measured hashes")
        sliced.update(
            {
                "cold_k3s_state_expected": True,
                "expected_cold_k3s_state": cold_state,
                "cold_retained_paths_expected": True,
                "expected_cold_retained_paths": cold_paths,
                "service_exec_start_sha256": cold_state.get("service_exec_start_sha256", ""),
            }
        )
    return sliced


def _check_native_verification_report(
    report: Any,
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    node: str,
    operation: str,
    state: dict[str, Any] | None = None,
) -> bool:
    if not isinstance(report, dict) or report.get("status") != "verified" or report.get("failures") != []:
        return False
    try:
        sliced = _node_report_manifest(manifest, inventory, node, operation, state)
        expected_checks = _expected_report_checks(manifest, inventory, node, operation)
    except (ValueError, KeyError, TypeError):
        return False
    return report.get("manifest_sha256") == _canonical_sha(sliced) and report.get("checks") == expected_checks


def _state_artifact(
    artifact_id: Any,
    artifacts: dict[str, Any],
    node: str,
    inventory: dict[str, Any],
    plan_id: str,
    snapshot_sha256: str,
    *,
    complete: bool,
) -> dict[str, Any] | None:
    state = _artifact_value(artifacts, artifact_id)
    expected = next((item for item in inventory["nodes"] if item["name"] == node), None)
    if not isinstance(state, dict) or expected is None:
        return None
    if (
        state.get("schema_version") != 1
        or state.get("owner") != "gods-mlops"
        or state.get("node") != node
        or state.get("plan_id") != plan_id
        or state.get("daemonset_snapshot_sha256") != snapshot_sha256
        or state.get("service") != expected["service"]
        or state.get("k3s_data_dir") != expected["k3s_data_dir"]
    ):
        return None
    if not complete:
        if state.get("step") not in {"runtime_verification_pending", "retained_hash_pending", "service_stopped"}:
            return None
        return state
    retained_ids = inventory["requirements_by_node"][node]["retained_paths"]
    cold_paths = state.get("cold_retained_paths")
    cold_k3s = state.get("cold_k3s_state")
    if (
        state.get("step") != "service_stopped"
        or state.get("runtime_status") != "verified_stopped"
        or ("retained_failures" in state and state.get("retained_failures") != [])
        or not isinstance(cold_paths, dict)
        or set(cold_paths) != set(retained_ids)
        or not all(_is_sha256(value) for value in cold_paths.values())
        or not isinstance(cold_k3s, dict)
    ):
        return None
    expected_cold_keys = (
        {
            "server_db_sha256",
            "server_token_sha256",
            "server_tls_sha256",
            "server_cred_sha256",
            "server_manifests_sha256",
            "server_agent_state_sha256",
            "service_exec_start_sha256",
        }
        if expected["role"] == "server"
        else {"agent_state_sha256", "service_exec_start_sha256"}
    )
    if not expected_cold_keys.issubset(cold_k3s) or not all(_is_sha256(cold_k3s[key]) for key in expected_cold_keys):
        return None
    if expected["role"] == "server" and cold_k3s.get("datastore") != "sqlite":
        return None
    return state


def _node_state_map(
    observation: Any,
    artifacts: dict[str, Any],
    inventory: dict[str, Any],
    plan_id: str,
    snapshot_sha256: str,
    *,
    complete: bool,
) -> dict[str, dict[str, Any]] | None:
    if not isinstance(observation, dict) or not isinstance(observation.get("node_state_artifacts"), dict):
        return None
    expected = {node["name"] for node in inventory["nodes"]}
    refs = observation["node_state_artifacts"]
    if set(refs) != expected:
        return None
    result: dict[str, dict[str, Any]] = {}
    for node in expected:
        state = _state_artifact(refs[node], artifacts, node, inventory, plan_id, snapshot_sha256, complete=complete)
        if state is None:
            return None
        result[node] = state
    return result


def _check_node_reports(
    report_refs: Any,
    artifacts: dict[str, Any],
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    operation: str,
    states: dict[str, dict[str, Any]] | None = None,
) -> bool:
    expected_nodes = {node["name"] for node in inventory["nodes"]}
    if not isinstance(report_refs, dict) or set(report_refs) != expected_nodes:
        return False
    if operation == "verify-retained" and (not isinstance(states, dict) or set(states) != expected_nodes):
        return False
    return all(
        _check_native_verification_report(
            _artifact_value(artifacts, report_refs[node]),
            manifest,
            inventory,
            node,
            operation,
            states.get(node) if states is not None else None,
        )
        for node in expected_nodes
    )


def _check_retained_ids(records: Any) -> bool:
    if not isinstance(records, list) or not records:
        return False
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            return False
        if record["id"] in ids or not _is_sha256(record.get("sha256")):
            return False
        ids.add(record["id"])
    return True


def _check_rtsp_snapshots(
    observation: Any,
    protected_services: list[str],
    artifacts: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
) -> bool:
    if not isinstance(observation, dict):
        return False
    phases = {"before": "baseline", "during": "reclaim_resume", "after": "reconnect"}
    snapshots = [(phase, observation.get(phase)) for phase in phases]
    normalized: list[dict[str, Any]] = []
    capture_ids: set[str] = set()
    capture_paths: set[str] = set()
    capture_hashes: set[str] = set()
    capture_times: list[datetime] = []
    expected_containers = set(protected_services)
    if not expected_containers or any(not isinstance(snapshot, dict) for _, snapshot in snapshots):
        return False
    for phase, snapshot in snapshots:
        stage = phases[phase]
        receipt = receipts.get(stage)
        capture_id = snapshot.get("capture_artifact_id")
        observed_at = _iso_time(snapshot.get("observed_at"))
        capture = _artifact_value(artifacts, capture_id)
        artifact_paths = artifacts.get("artifact_paths")
        artifact_hashes = artifacts.get("artifact_hashes")
        capture_path = artifact_paths.get(capture_id) if isinstance(artifact_paths, dict) else None
        capture_hash = artifact_hashes.get(capture_id) if isinstance(artifact_hashes, dict) else None
        stage_outputs = receipt.get("native_output_artifact_ids") if isinstance(receipt, dict) else None
        if (
            not isinstance(capture_id, str)
            or capture_id in capture_ids
            or not isinstance(receipt, dict)
            or not isinstance(stage_outputs, list)
            or capture_id not in stage_outputs
            or observed_at is None
            or not isinstance(capture, dict)
            or not isinstance(capture_path, str)
            or not _is_sha256(capture_hash)
            or capture_path in capture_paths
            or capture_hash in capture_hashes
        ):
            return False
        capture_ids.add(capture_id)
        capture_paths.add(capture_path)
        capture_hashes.add(capture_hash)
        start = _iso_time(receipt.get("started_at"))
        end = _iso_time(receipt.get("ended_at"))
        if start is None or end is None or not start <= observed_at <= end:
            return False
        capture_times.append(observed_at)
        containers = snapshot.get("containers")
        content = snapshot.get("selected_content_sha256")
        probe = snapshot.get("read_only_stream_probe")
        if any(
            capture.get(key) != snapshot.get(key)
            for key in ("observed_at", "containers", "selected_content_sha256", "read_only_stream_probe")
        ) or capture.get("phase") != phase or capture.get("stage") != stage:
            return False
        if not isinstance(containers, dict) or set(containers) != expected_containers or not _hash_mapping(content):
            return False
        if not isinstance(probe, dict) or probe.get("status") != "readable" or not isinstance(probe.get("bytes_read"), int) or probe["bytes_read"] <= 0:
            return False
        current: dict[str, Any] = {}
        for name, container in containers.items():
            if (
                not isinstance(container, dict)
                or not container.get("container_id")
                or not isinstance(container.get("image_id"), str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", container.get("image_id", "")) is None
                or container.get("running") is not True
                or _iso_time(container.get("started_at")) is None
                or _iso_time(container.get("started_at")) > observed_at
                or not isinstance(container.get("mounts"), list)
                or not container["mounts"]
                or any(
                    not isinstance(mount, dict)
                    or not isinstance(mount.get("name"), str)
                    or not mount["name"]
                    or not isinstance(mount.get("source"), str)
                    or not mount["source"]
                    or not isinstance(mount.get("destination"), str)
                    or not mount["destination"]
                    for mount in container.get("mounts", [])
                )
            ):
                return False
            current[name] = {
                "container_id": container["container_id"],
                "image_id": container["image_id"],
                "running": container["running"],
                "started_at": container["started_at"],
                "mounts": sorted(container["mounts"], key=lambda item: json.dumps(item, sort_keys=True)),
            }
        normalized.append({"containers": current, "content": content})
    return capture_times[0] < capture_times[1] < capture_times[2] and normalized[0] == normalized[1] == normalized[2]


def _valid_png(payload: Any) -> bool:
    if not isinstance(payload, bytes) or len(payload) < 33 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    if payload[12:16] != b"IHDR" or int.from_bytes(payload[8:12], "big") != 13:
        return False
    width = int.from_bytes(payload[16:20], "big")
    height = int.from_bytes(payload[20:24], "big")
    return width > 0 and height > 0 and payload.endswith(b"IEND\xaeB`\x82")


def _valid_row_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_loopback_origin(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return (
            parsed.scheme in {"http", "https"}
            and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        return False


def _within_receipt_interval(value: datetime | None, receipt: dict[str, Any] | None) -> bool:
    if value is None or not isinstance(receipt, dict):
        return False
    start = _iso_time(receipt.get("started_at"))
    end = _iso_time(receipt.get("ended_at"))
    return start is not None and end is not None and start <= value <= end


def _check_logical_readback(
    continuity: Any,
    baseline: Any,
    manifest: Any,
    artifacts: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
) -> bool:
    if not isinstance(continuity, dict) or not isinstance(baseline, dict) or not isinstance(manifest, dict):
        return False
    metadata = continuity.get("logical_restore")
    if not isinstance(metadata, dict):
        return False
    database_entries = manifest.get("database_restores")
    if not isinstance(database_entries, list):
        return False
    restore_id = metadata.get("database_restore_manifest_id")
    manifest_entry = next(
        (item for item in database_entries if isinstance(item, dict) and item.get("id") == restore_id),
        None,
    )
    report = _artifact_value(artifacts, metadata.get("access_report_artifact_id"))
    if not isinstance(manifest_entry, dict) or not isinstance(report, dict):
        return False
    database_ids = [
        metadata.get("source_database_id"),
        metadata.get("restore_database_id"),
        metadata.get("backup_validation_database_id"),
        metadata.get("restore_validation_database_id"),
    ]
    if any(not isinstance(value, str) or not value for value in database_ids) or len(set(database_ids)) != len(database_ids):
        return False
    row_counts = (metadata.get("source_row_count"), metadata.get("restored_row_count"), metadata.get("readback_row_count"))
    if not all(_valid_row_count(value) for value in row_counts):
        return False
    hashes = (
        metadata.get("source_query_sha256"),
        metadata.get("restored_query_sha256"),
        metadata.get("readback_query_sha256"),
    )
    if not all(_is_sha256(value) for value in hashes) or len(set(hashes)) != 1:
        return False
    if (
        metadata.get("restore_status") != "verified"
        or metadata.get("backup_dump_path") != manifest_entry.get("backup_dump_path")
        or metadata.get("restored_dump_path") != manifest_entry.get("restored_dump_path")
        or metadata.get("backup_dump_path") == metadata.get("restored_dump_path")
    ):
        return False
    bound_fields = (
        "database_restore_manifest_id",
        "source_database_id",
        "restore_database_id",
        "backup_validation_database_id",
        "restore_validation_database_id",
        "backup_dump_path",
        "restored_dump_path",
        "source_query_sha256",
        "restored_query_sha256",
        "readback_query_sha256",
        "source_row_count",
        "restored_row_count",
        "readback_row_count",
    )
    if any(report.get(key) != metadata.get(key) for key in bound_fields):
        return False
    if report.get("status") != "verified" or report.get("readback_status") != "verified":
        return False
    report_time = _iso_time(report.get("observed_at"))
    reconnect_receipt = receipts.get("reconnect")
    reconnect_ended = _iso_time(reconnect_receipt.get("ended_at")) if isinstance(reconnect_receipt, dict) else None
    if (
        not _within_receipt_interval(report_time, receipts.get("continuity"))
        or reconnect_ended is None
        or report_time < reconnect_ended
    ):
        return False
    return True


def _check_operator_authentication(
    continuity: Any,
    baseline: Any,
    bundle: dict[str, Any],
    artifacts: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
) -> bool:
    if not isinstance(continuity, dict) or not isinstance(baseline, dict):
        return False
    identities = bundle.get("identities")
    operator = identities.get("operator") if isinstance(identities, dict) else None
    auth = continuity.get("authentication")
    if not isinstance(operator, dict) or not isinstance(auth, dict):
        return False
    browser_receipt = _artifact_value(artifacts, auth.get("browser_receipt_artifact_id"))
    screenshot_id = auth.get("screenshot_artifact_id")
    if not isinstance(screenshot_id, str):
        return False
    screenshot = _artifact_value(artifacts, screenshot_id)
    artifact_hashes = artifacts.get("artifact_hashes")
    continuity_receipt = receipts.get("continuity")
    continuity_outputs = continuity_receipt.get("native_output_artifact_ids") if isinstance(continuity_receipt, dict) else None
    if not isinstance(continuity_outputs, list):
        return False
    identity_hash = baseline.get("account_identity_sha256")
    credential_hash = baseline.get("credential_identity_sha256")
    if not isinstance(browser_receipt, dict) or not isinstance(artifact_hashes, dict):
        return False
    captured_at = _iso_time(auth.get("observed_at"))
    reconnect_receipt = receipts.get("reconnect")
    reconnect_ended = _iso_time(reconnect_receipt.get("ended_at")) if isinstance(reconnect_receipt, dict) else None
    if (
        not _is_sha256(identity_hash)
        or not _is_sha256(credential_hash)
        or identity_hash != operator.get("account_identity_sha256")
        or credential_hash != operator.get("credential_identity_sha256")
        or auth.get("account_identity_sha256") != identity_hash
        or auth.get("credential_identity_sha256") != credential_hash
        or auth.get("login_status") != 303
        or auth.get("redirect_followed") is not True
        or auth.get("protected_access_status") != 200
        or auth.get("protected_path") != operator.get("protected_path")
        or auth.get("operator_service") != operator.get("service_id")
        or not _is_loopback_origin(operator.get("origin"))
        or auth.get("origin") != operator.get("origin")
        or not _is_sha256(auth.get("session_fingerprint_sha256"))
        or any(key in auth for key in ("session_id", "session_token", "cookie", "password"))
        or captured_at is None
        or not _within_receipt_interval(captured_at, receipts.get("continuity"))
        or reconnect_ended is None
        or captured_at < reconnect_ended
        or auth.get("browser_receipt_artifact_id") not in continuity_outputs
        or screenshot_id not in continuity_outputs
        or not isinstance(browser_receipt.get("screenshot_sha256"), str)
        or browser_receipt.get("screenshot_sha256") != artifact_hashes.get(screenshot_id)
        or not _valid_png(screenshot)
    ):
        return False
    return all(
        browser_receipt.get(key) == auth.get(key)
        for key in (
            "login_status",
            "redirect_followed",
            "protected_access_status",
            "protected_path",
            "account_identity_sha256",
            "credential_identity_sha256",
            "session_fingerprint_sha256",
            "operator_service",
            "origin",
            "observed_at",
        )
    ) and not any(key in browser_receipt for key in ("session_id", "session_token", "cookie", "password"))


def _verify_lifecycle_evidence(config: dict[str, Any], artifacts: dict[str, Any]) -> dict[str, Any]:
    """Check hash-bound receipts only; never invoke deployment or recovery commands."""
    errors = {case: [] for case in _STAGES}
    bundle = artifacts.get("bundle") if isinstance(artifacts, dict) else None
    values = artifacts.get("artifact_values") if isinstance(artifacts, dict) else None
    if not isinstance(bundle, dict) or not isinstance(values, dict):
        return {
            "cases": {case: {"status": "failed", "reasons": ["bundle_not_loaded"]} for case in _STAGES},
            "performed_by_this_test": False,
            "overall": "incomplete_or_failed",
            "lifecycle_pass_claimed": False,
        }

    identity_errors: list[str] = []
    for bundle_key, config_key in (
        ("fixture_id", "fixture_id"),
        ("source_commit", "expected_source_commit"),
        ("plan_id", "plan_id"),
        ("snapshot_sha256", "snapshot_sha256"),
    ):
        if bundle.get(bundle_key) != config.get(config_key):
            identity_errors.append("protected_config_binding_mismatch")
    if not isinstance(bundle.get("source_commit"), str) or not _COMMIT.fullmatch(bundle["source_commit"]):
        identity_errors.append("source_commit_invalid")
    if not _is_sha256(bundle.get("plan_id")) or not _is_sha256(bundle.get("snapshot_sha256")):
        identity_errors.append("plan_or_snapshot_digest_invalid")
    for case in ("deploy", "reapply"):
        errors[case].extend(identity_errors)
    for case in ("baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
        errors[case].extend(identity_errors)

    raw_artifact_hashes = artifacts.get("artifact_hashes")
    raw_artifact_paths = artifacts.get("artifact_paths")
    if not isinstance(raw_artifact_hashes, dict) or not isinstance(raw_artifact_paths, dict):
        for case in _STAGES:
            _add_error(errors, case, "artifact_index_invalid")
    else:
        artifact_hashes = raw_artifact_hashes
        artifact_paths = raw_artifact_paths
        try:
            root_fd, _ = _open_absolute_directory(artifacts.get("evidence_root"))
            try:
                for artifact_id, relative_path in artifact_paths.items():
                    payload = _read_beneath(root_fd, relative_path)
                    if _sha256(payload) != artifact_hashes.get(artifact_id):
                        for case in _STAGES:
                            _add_error(errors, case, "artifact_changed_after_load")
                        break
            finally:
                os.close(root_fd)
        except (OSError, ValueError):
            for case in _STAGES:
                _add_error(errors, case, "artifact_path_changed_after_load")
    artifact_hashes = raw_artifact_hashes if isinstance(raw_artifact_hashes, dict) else {}
    artifact_paths = raw_artifact_paths if isinstance(raw_artifact_paths, dict) else {}

    receipts = bundle.get("stages")
    receipt_by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(receipts, list):
        for case in _STAGES:
            _add_error(errors, case, "stage_receipts_missing")
    else:
        previous_end: datetime | None = None
        names: list[str] = []
        attempt_names: dict[str, str] = {}
        observation_id = bundle.get("observations_artifact_id")
        for receipt in receipts:
            if not isinstance(receipt, dict):
                continue
            name = receipt.get("stage")
            if not isinstance(name, str) or name not in _STAGES:
                continue
            names.append(name)
            receipt_by_name[name] = receipt
            start = _iso_time(receipt.get("started_at"))
            end = _iso_time(receipt.get("ended_at"))
            attempt = receipt.get("attempt_id")
            output_ids = receipt.get("native_output_artifact_ids")
            if not _valid_uuid(attempt):
                _add_error(errors, name, "attempt_id_invalid")
            else:
                previous_stage = attempt_names.get(attempt)
                if previous_stage is not None:
                    _add_error(errors, previous_stage, "attempt_id_duplicate")
                    _add_error(errors, name, "attempt_id_duplicate")
                attempt_names[attempt] = name
            if start is None or end is None or end <= start or (previous_end is not None and start < previous_end):
                _add_error(errors, name, "stage_timestamps_invalid")
            else:
                previous_end = end
            if receipt.get("source_commit") != bundle.get("source_commit"):
                _add_error(errors, name, "stage_source_commit_mismatch")
            if receipt.get("plan_id") != bundle.get("plan_id"):
                _add_error(errors, name, "stage_plan_binding_mismatch")
            if name in {"baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity", "rtsp_preserved"} and receipt.get("snapshot_sha256") != bundle.get("snapshot_sha256"):
                _add_error(errors, name, "stage_snapshot_binding_mismatch")
            recipe_id = receipt.get("recipe_artifact_id")
            if (
                not isinstance(recipe_id, str)
                or recipe_id not in artifact_hashes
                or receipt.get("recipe_sha256") != artifact_hashes.get(recipe_id)
            ):
                _add_error(errors, name, "recipe_artifact_unbound")
            if (
                not isinstance(output_ids, list)
                or len(output_ids) < 2
                or any(not isinstance(item, str) or item not in artifact_hashes for item in output_ids)
                or observation_id not in output_ids
            ):
                _add_error(errors, name, "native_outputs_missing_or_unbound")
            for code in _native_receipt_binding_errors(receipt, bundle, artifacts, observation_id):
                _add_error(errors, name, code)
            status = receipt.get("exit_status")
            if type(status) is not int or (name == "reclaim_interrupted" and status == 0) or (name != "reclaim_interrupted" and status != 0):
                _add_error(errors, name, "stage_exit_status_unexpected")
        if names != list(_STAGES) or len(set(names)) != len(_STAGES):
            for case in _STAGES:
                if case not in receipt_by_name:
                    _add_error(errors, case, "stage_receipt_missing")
            for name in names:
                if name in receipt_by_name:
                    _add_error(errors, name, "stage_order_or_duplicate_invalid")

    provenance = bundle.get("provenance")
    provenance_kind = provenance.get("kind") if isinstance(provenance, dict) else None
    if provenance_kind not in {"native", "synthetic_offline"}:
        for case in _STAGES:
            _add_error(errors, case, "provenance_kind_invalid")
    for code in _native_provenance_errors(
        provenance,
        [receipt for receipt in receipt_by_name.values()],
        artifacts,
    ):
        for case in _STAGES:
            _add_error(errors, case, code)

    bindings = bundle.get("bindings")
    inventory = None
    manifest = None
    snapshot = None
    plan = None
    if not isinstance(bindings, dict):
        for case in ("baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
            _add_error(errors, case, "native_lifecycle_bindings_missing")
    else:
        inventory_id = bindings.get("inventory_artifact_id")
        storage_id = bindings.get("storage_artifact_id")
        manifest_id = bindings.get("manifest_artifact_id")
        snapshot_id = bindings.get("snapshot_artifact_id")
        inventory_path = artifact_paths.get(inventory_id) if isinstance(inventory_id, str) else None
        storage_path = artifact_paths.get(storage_id) if isinstance(storage_id, str) else None
        if not isinstance(inventory_path, str) or not isinstance(storage_path, str):
            for case in ("deploy", "reapply", "baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
                _add_error(errors, case, "inventory_or_storage_artifact_missing")
        else:
            evidence_root = artifacts.get("evidence_root")
            inventory_full_path = str(Path(evidence_root) / inventory_path)
            storage_full_path = str(Path(evidence_root) / storage_path)
            try:
                if not _artifact_is_still_bound(artifacts, inventory_id) or not _artifact_is_still_bound(artifacts, storage_id):
                    raise ValueError("inventory evidence changed")
                inventory = load_inventory(inventory_full_path, storage_path=storage_full_path)
                plan = plan_reclaim(inventory)
                if not _artifact_is_still_bound(artifacts, inventory_id) or not _artifact_is_still_bound(artifacts, storage_id):
                    raise ValueError("inventory evidence changed")
            except (OSError, ValueError, yaml.YAMLError):
                for case in ("deploy", "reapply", "baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
                    _add_error(errors, case, "inventory_or_storage_invalid")
            else:
                if plan.get("plan_id") != bundle.get("plan_id") or plan.get("plan_valid") is not True:
                    for case in ("deploy", "reapply", "baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
                        _add_error(errors, case, "reclaim_plan_mismatch_or_blocked")
        manifest = _artifact_value(artifacts, manifest_id)
        snapshot = _artifact_value(artifacts, snapshot_id)
        if not isinstance(manifest, dict):
            for case in ("baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
                _add_error(errors, case, "recovery_manifest_missing")
        elif inventory is not None:
            try:
                validate_retained_manifest(manifest, inventory)
            except (OSError, ValueError, KeyError, TypeError):
                for case in ("baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
                    _add_error(errors, case, "recovery_manifest_invalid")
        if not isinstance(snapshot, dict):
            for case in ("baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
                _add_error(errors, case, "daemonset_snapshot_missing")
        else:
            try:
                snapshot_summary = validate_daemonset_snapshot(
                    snapshot,
                    confirmation=str(bundle.get("snapshot_sha256", "")),
                    expected_plan_id=str(bundle.get("plan_id", "")),
                )
                if snapshot_summary.get("snapshot_sha256") != bundle.get("snapshot_sha256"):
                    raise ValueError("digest mismatch")
            except (OSError, ValueError, KeyError, TypeError):
                for case in ("baseline", "reclaim_interrupted", "reclaim_resume", "reclaim_repeat", "reconnect", "continuity"):
                    _add_error(errors, case, "daemonset_snapshot_invalid")

    observations = _artifact_value(artifacts, bundle.get("observations_artifact_id"))
    stage_observations = observations.get("stages") if isinstance(observations, dict) else None
    if not isinstance(stage_observations, dict):
        for case in _STAGES:
            _add_error(errors, case, "stage_observations_missing")
        stage_observations = {}

    if provenance_kind in {"native", "synthetic_offline"} and inventory is not None and isinstance(stage_observations, dict):
        deploy = stage_observations.get("deploy")
        _check_deploy_observation(
            deploy,
            inventory,
            bundle,
            artifacts,
            receipt_by_name.get("deploy"),
            errors["deploy"],
        )
        _check_reapply_observation(stage_observations.get("reapply"), deploy, errors["reapply"])

        baseline = stage_observations.get("baseline")
        if not isinstance(baseline, dict):
            errors["baseline"].append("baseline_observation_missing")
        else:
            if manifest is None or not _check_node_reports(baseline.get("recovery_report_artifacts"), artifacts, manifest, inventory, "verify-recovery-artifacts"):
                errors["baseline"].append("per_node_backup_restore_verification_missing")
            selected = baseline.get("selected_records")
            if not isinstance(selected, dict) or any(not _check_retained_ids(selected.get(key)) for key in ("datasets", "annotation_revisions", "immutable_objects")):
                errors["baseline"].append("baseline_data_lineage_missing")
            if not isinstance(selected, dict) or not isinstance(selected.get("job_history"), list) or not selected["job_history"] or any(not isinstance(item, dict) or not item.get("id") for item in selected["job_history"]):
                errors["baseline"].append("baseline_job_history_missing")
            if not _is_sha256(baseline.get("credential_identity_sha256")):
                errors["baseline"].append("baseline_credential_identity_missing")

        interrupted = stage_observations.get("reclaim_interrupted")
        interrupted_states = _node_state_map(interrupted, artifacts, inventory, bundle["plan_id"], bundle["snapshot_sha256"], complete=False)
        interrupt_receipt = receipt_by_name.get("reclaim_interrupted", {})
        controller = interrupted.get("controller") if isinstance(interrupted, dict) else None
        if (
            interrupted_states is None
            or not isinstance(controller, dict)
            or controller.get("attempt_id") != interrupt_receipt.get("attempt_id")
            or controller.get("interrupt_kind") != "externally_supervised_ansible_abort"
            or not controller.get("durable_boundary")
            or controller.get("observed") is not True
        ):
            errors["reclaim_interrupted"].append("native_interruption_record_missing")
        elif all(state.get("step") == "service_stopped" for state in interrupted_states.values()):
            errors["reclaim_interrupted"].append("interruption_has_no_partial_node_state")
        service_status = interrupted.get("service_status") if isinstance(interrupted, dict) else None
        expected_node_names = {node["name"] for node in inventory["nodes"]}
        if (
            not isinstance(service_status, dict)
            or set(service_status) != expected_node_names
            or any(
                not isinstance(value, dict)
                or type(value.get("active")) is not bool
                or _iso_time(value.get("observed_at")) is None
                or not isinstance(value.get("runtime_status"), str)
                or not value["runtime_status"]
                for value in service_status.values()
            )
        ):
            errors["reclaim_interrupted"].append("measured_service_and_runtime_status_missing")
        elif interrupted_states is None or any(
            (
                interrupted_states[node].get("runtime_status") is not None
                and service_status[node].get("runtime_status") != interrupted_states[node].get("runtime_status")
            )
            or (
                interrupted_states[node].get("step") == "runtime_verification_pending"
                and service_status[node].get("runtime_status") not in {"pending", "verified_stopped"}
            )
            or (
                interrupted_states[node].get("step") == "service_stopped"
                and service_status[node].get("active") is not False
            )
            for node in expected_node_names
        ):
            errors["reclaim_interrupted"].append("measured_service_status_conflicts_with_node_record")

        resume = stage_observations.get("reclaim_resume")
        resume_states = _node_state_map(resume, artifacts, inventory, bundle["plan_id"], bundle["snapshot_sha256"], complete=True)
        if resume_states is None:
            errors["reclaim_resume"].append("both_nodes_lack_verified_cold_state")
        elif manifest is None or not _check_node_reports(
            resume.get("retained_report_artifacts"), artifacts, manifest, inventory, "verify-retained", resume_states
        ):
            errors["reclaim_resume"].append("native_cold_verification_reports_missing")
        task_order = resume.get("task_order") if isinstance(resume, dict) else None
        worker_name = next((node["name"] for node in inventory["nodes"] if node["role"] == "worker"), None)
        server_name = next((node["name"] for node in inventory["nodes"] if node["role"] == "server"), None)
        if (
            not isinstance(task_order, dict)
            or _iso_time(task_order.get("worker_verified_at")) is None
            or _iso_time(task_order.get("server_stop_started_at")) is None
            or _iso_time(task_order.get("worker_verified_at")) >= _iso_time(task_order.get("server_stop_started_at"))
            or resume_states is None
            or worker_name not in resume_states
            or server_name not in resume_states
        ):
            errors["reclaim_resume"].append("worker_before_server_order_missing")

        repeat = stage_observations.get("reclaim_repeat")
        repeat_states = _node_state_map(repeat, artifacts, inventory, bundle["plan_id"], bundle["snapshot_sha256"], complete=True)
        if resume_states is None or repeat_states is None:
            errors["reclaim_repeat"].append("repeat_cold_state_missing")
        else:
            for node in resume_states:
                if (
                    repeat_states[node].get("cold_retained_paths") != resume_states[node].get("cold_retained_paths")
                    or repeat_states[node].get("cold_k3s_state") != resume_states[node].get("cold_k3s_state")
                ):
                    errors["reclaim_repeat"].append("repeat_cold_state_changed")
                    break
        if manifest is None or not isinstance(repeat, dict) or not _check_node_reports(
            repeat.get("retained_report_artifacts"), artifacts, manifest, inventory, "verify-retained", repeat_states
        ):
            errors["reclaim_repeat"].append("repeat_native_cold_reports_missing")
        offline = repeat.get("offline_retry") if isinstance(repeat, dict) else None
        trace = _artifact_value(artifacts, offline.get("trace_artifact_id")) if isinstance(offline, dict) else None
        if (
            not isinstance(offline, dict)
            or offline.get("selected") is not True
            or not isinstance(offline.get("executed_task_names"), list)
            or "Require matching saved node state and snapshot before an API-free resume" not in offline.get("executed_task_names", [])
            or not isinstance(trace, dict)
            or trace.get("api_request_events") != []
            or trace.get("selected_branch") != "offline_resume"
        ):
            errors["reclaim_repeat"].append("api_free_repeat_branch_not_proven")

        reconnect = stage_observations.get("reconnect")
        reconnect_states = _node_state_map(reconnect, artifacts, inventory, bundle["plan_id"], bundle["snapshot_sha256"], complete=True)
        if resume_states is None or reconnect_states is None:
            errors["reconnect"].append("reconnect_cold_state_missing")
        elif any(
            reconnect_states[node].get("cold_retained_paths") != resume_states[node].get("cold_retained_paths")
            or reconnect_states[node].get("cold_k3s_state") != resume_states[node].get("cold_k3s_state")
            for node in resume_states
        ):
            errors["reconnect"].append("reconnect_cold_state_changed")
        live_daemonsets_id = reconnect.get("live_daemonsets_artifact_id") if isinstance(reconnect, dict) else None
        live_daemonsets = _artifact_value(artifacts, live_daemonsets_id)
        if snapshot is None or not isinstance(live_daemonsets, dict):
            errors["reconnect"].append("restored_daemonset_readback_missing")
        else:
            try:
                restored = verify_restored_daemonsets(snapshot["objects"], live_daemonsets)
                if restored.get("status") != "verified":
                    errors["reconnect"].append("restored_daemonset_mismatch")
            except (ValueError, KeyError, TypeError):
                errors["reconnect"].append("restored_daemonset_mismatch")
        ready_nodes = reconnect.get("ready_nodes") if isinstance(reconnect, dict) else None
        expected_names = {node["name"] for node in inventory["nodes"]}
        if not isinstance(ready_nodes, dict) or set(ready_nodes) != expected_names or any(
            not isinstance(item, dict) or item.get("ready") is not True for item in ready_nodes.values()
        ):
            errors["reconnect"].append("reconnected_cluster_readiness_missing")
        if manifest is None or not isinstance(reconnect, dict) or not _check_node_reports(
            reconnect.get("pre_start_retained_report_artifacts"),
            artifacts,
            manifest,
            inventory,
            "verify-retained",
            reconnect_states,
        ):
            errors["reconnect"].append("pre_start_retained_verification_missing")
        reconnect_fingerprint = _deployment_fingerprint(reconnect.get("deployment_readback")) if isinstance(reconnect, dict) else None
        deploy_fingerprint = _deployment_fingerprint(deploy)
        if deploy_fingerprint is None or reconnect_fingerprint is None or reconnect_fingerprint != deploy_fingerprint:
            errors["reconnect"].append("reconnected_pv_or_identity_binding_changed")

        continuity = stage_observations.get("continuity")
        if not isinstance(baseline, dict) or not isinstance(continuity, dict):
            errors["continuity"].append("continuity_observation_missing")
        else:
            before = baseline.get("selected_records")
            after = continuity.get("selected_records")
            if not isinstance(before, dict) or after != before:
                errors["continuity"].append("selected_data_or_job_history_changed")
            if not _check_logical_readback(continuity, baseline, manifest, artifacts, receipt_by_name):
                errors["continuity"].append("distinct_database_restore_and_readback_missing")
            if not _check_operator_authentication(continuity, baseline, bundle, artifacts, receipt_by_name):
                errors["continuity"].append("fresh_preserved_credential_access_missing")
        rtsp = stage_observations.get("rtsp_preserved")
        services = inventory.get("protected_services", [])
        if not _check_rtsp_snapshots(rtsp, services, artifacts, receipt_by_name):
            errors["rtsp_preserved"].append("original_rtsp_identity_mount_or_stream_changed")

    cases: dict[str, dict[str, Any]] = {}
    for case in _STAGES:
        reasons = errors[case]
        if reasons:
            status = "failed"
        elif provenance_kind == "synthetic_offline":
            status = "not_run"
            reasons = ["synthetic_offline_fixture"]
        elif provenance_kind == "native":
            status = "verified_evidence"
        else:
            status = "failed"
        cases[case] = {"status": status, "reasons": reasons}
    overall = "evidence_consistent" if all(item["status"] == "verified_evidence" for item in cases.values()) else "incomplete_or_failed"
    return {
        "cases": cases,
        "performed_by_this_test": False,
        "overall": overall,
        "lifecycle_pass_claimed": False,
        "separate_gates": {
            "product_browser_detection_search": "not_run",
            "kubeflow_label_studio_operator_ui": "not_run",
            "real_operating_data_model_runs": "not_run",
            "model_quality": "not_run",
            "submodule_adoption": "not_run",
        },
    }


def _valid_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def _root_ack_matches(config: dict[str, Any], environ: Mapping[str, str]) -> bool:
    return environ.get(_ACK_ENV) == f"{config.get('fixture_id')}:{config.get('bundle_sha256')}"


def _daemonset_document() -> dict[str, Any]:
    return {
        "items": [
            {
                "apiVersion": "apps/v1",
                "kind": "DaemonSet",
                "metadata": {
                    "namespace": "kube-system",
                    "name": "istio-cni-node",
                    "labels": {"app.kubernetes.io/name": "istio-cni"},
                },
                "spec": {"selector": {"matchLabels": {"k8s-app": "istio-cni-node"}}, "template": {"spec": {"containers": [{"name": "install-cni"}]}}},
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
            },
        ]
    }


def _fixture_manifest(inventory: dict[str, Any]) -> dict[str, Any]:
    retained: list[dict[str, Any]] = []
    for identifier, expected in inventory["retained_requirements_by_id"].items():
        entry = {
            "id": identifier,
            "node": expected["node"],
            "source_path": expected["source_path"],
            "source_owner": expected["source_owner"],
            "role": expected["role"],
        }
        if expected["role"] != "database":
            entry.update(
                {
                    "backup_path": f"{expected['recovery_root']}/snapshots/{identifier}",
                    "expected_tree_sha256": "a" * 64,
                    "backup_owner": {"uid": 0, "gid": 0},
                    "backup_mode": "0700",
                }
            )
        retained.append(entry)
    databases: list[dict[str, Any]] = []
    for node, requirements in inventory["requirements_by_node"].items():
        for identifier in requirements["database_restores"]:
            recovery_root = next(
                entry["recovery_root"]
                for entry in inventory["retained_requirements_by_id"].values()
                if entry["node"] == node
            )
            databases.append(
                {
                    "id": identifier,
                    "node": node,
                    "backup_dump_path": f"{recovery_root}/db/{identifier}-source.sql",
                    "restored_dump_path": f"{recovery_root}/db/{identifier}-restored.sql",
                    "expected_sha256": "b" * 64,
                    "expected_uid": 0,
                    "expected_mode": "0600",
                    "dump_format": "postgresql-sql-v1",
                    "provenance": {"source": "synthetic fixture only"},
                }
            )
    credentials: list[dict[str, Any]] = []
    for node, requirements in inventory["requirements_by_node"].items():
        recovery_root = next(
            (entry["recovery_root"] for entry in inventory["retained_requirements_by_id"].values() if entry["node"] == node),
            next(item["recovery_root"] for item in inventory["nodes"] if item["name"] == node),
        )
        for identifier in requirements["credentials"]:
            entry = {
                "id": identifier,
                "node": node,
                "path": f"{recovery_root}/secrets/{identifier}",
                "expected_sha256": "c" * 64,
                "expected_uid": 0,
                "expected_mode": "0600",
            }
            expected_source = inventory.get("credential_source_by_id", {}).get(identifier)
            if expected_source:
                entry.update(
                    {
                        "source_path": expected_source,
                        "source_expected_uid": 0,
                        "source_expected_mode": "0600",
                    }
                )
            credentials.append(entry)
    return {
        "schema_version": 1,
        "owner": "gods-mlops",
        "plan_id": plan_reclaim(inventory)["plan_id"],
        "retained_paths": retained,
        "database_restores": databases,
        "credentials": credentials,
    }


def _complete_state(
    inventory: dict[str, Any],
    node: dict[str, Any],
    plan_id: str,
    snapshot_sha256: str,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": 1,
        "owner": "gods-mlops",
        "node": node["name"],
        "plan_id": plan_id,
        "step": "service_stopped",
        "runtime_status": "verified_stopped",
        "service": node["service"],
        "k3s_data_dir": node["k3s_data_dir"],
        "daemonset_snapshot_path": "/secure/recovery/daemonsets.json",
        "daemonset_snapshot_sha256": snapshot_sha256,
        "cold_retained_paths": {
            identifier: hashlib.sha256(f"{node['name']}:{identifier}".encode()).hexdigest()
            for identifier in inventory["requirements_by_node"][node["name"]]["retained_paths"]
        },
    }
    if node["role"] == "server":
        state["cold_k3s_state"] = {
            "datastore": "sqlite",
            "server_db_sha256": "1" * 64,
            "server_token_sha256": "2" * 64,
            "server_tls_sha256": "3" * 64,
            "server_cred_sha256": "4" * 64,
            "server_manifests_sha256": "5" * 64,
            "server_agent_state_sha256": "6" * 64,
            "service_exec_start_sha256": "7" * 64,
        }
    else:
        state["cold_k3s_state"] = {
            "agent_state_sha256": "8" * 64,
            "service_exec_start_sha256": "9" * 64,
        }
    return state


def _fixture_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


def _make_evidence_fixture(root: Path, *, provenance_kind: str = "synthetic_offline") -> tuple[dict[str, Any], dict[str, Any]]:
    from gods_mlops.lifecycle.recovery import build_daemonset_snapshot

    root.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    inventory_path = root / "inventory.yml"
    storage_path = root / "retained-storage.yml"
    _write_private(inventory_path, (REPO_ROOT / "infra/ansible/inventory.example.yml").read_bytes())
    _write_private(storage_path, (REPO_ROOT / "infra/kubeflow/retained-storage.yaml").read_bytes())
    inventory = load_inventory(inventory_path, storage_path=storage_path)
    plan_id = plan_reclaim(inventory)["plan_id"]
    manifest = _fixture_manifest(inventory)
    validate_retained_manifest(manifest, inventory)
    snapshot = build_daemonset_snapshot(_daemonset_document(), plan_id=plan_id)
    snapshot_sha256 = snapshot["snapshot_sha256"]
    source_commit = "a" * 40
    fixture_id = "a51a8330-fc99-4792-9471-279e4893de00"
    identities = {
        "images": {"training": "sha256:" + "1" * 64, "operator": "sha256:" + "2" * 64},
        "input_versions": {
            "dataset": {"version": "synthetic-v1", "sha256": "3" * 64},
            "k3s": {"version": "v1.36.2+k3s1", "sha256": "4" * 64},
        },
        "kubeconfig": {
            "reference": "infra/ansible/generated/kubeconfig",
            "sha256": "5" * 64,
            "mode": "0600",
            "context": "default",
            "server": "https://k3s.example.invalid:6443",
            "certificate_authority_sha256": "6" * 64,
            "cluster_uid": "synthetic-cluster-uid",
        },
        "operator": {
            "service_id": "gods-mlops-operator",
            "origin": "http://127.0.0.1:8080",
            "protected_path": "/samples",
            "account_identity_sha256": "7" * 64,
            "credential_identity_sha256": "c" * 64,
        },
    }
    deploy_nodes: dict[str, Any] = {}
    for node in inventory["nodes"]:
        deploy_nodes[node["name"]] = {
            "uid": f"{node['name']}-uid",
            "ready": True,
            "data_root": node["data_root"],
            "k3s_data_dir": node["k3s_data_dir"],
            "service": node["service"],
            "root_uid": 0,
            "data_root_uid": 0,
            "k3s_data_dir_uid": 0,
            "data_root_mode": "0755",
            "k3s_data_dir_mode": "0700",
            "owner_marker_uid": 0,
            "cluster_marker_uid": 0,
            "owner_marker_mode": "0644",
            "cluster_marker_mode": "0600",
            "owner_marker": {
                "schema_version": 1,
                "owner": "gods-mlops",
                "data_root": node["data_root"],
                "k3s_data_dir": node["k3s_data_dir"],
                "k3s_version": "v1.36.2+k3s1",
            },
            "cluster_marker": {
                "schema_version": 1,
                "owner": "gods-mlops",
                "data_root": node["data_root"],
                "k3s_data_dir": node["k3s_data_dir"],
                "k3s_version": "v1.36.2+k3s1",
            },
            "service_exec_start_sha256": "f" * 64,
        }
    deploy_pvs = [
        {
            "id": item["id"],
            "volume_name": item["id"],
            "claim": item["claim"],
            "node": item["node"],
            "local_path": item["path"],
            "phase": "Bound",
            "reclaim_policy": "Retain",
        }
        for item in inventory["retained_paths"]
        if item.get("claim")
    ]
    deploy = {
        "preflight": {
            "status": "verified",
            "authenticated": True,
            "become_uid": 0,
            "source_commit": source_commit,
            "image_digests": identities["images"],
            "render_sha256": "d" * 64,
        },
        "cluster": {
            "uid": "synthetic-cluster-uid",
            "context": "default",
            "server": "https://k3s.example.invalid:6443",
            "certificate_authority_sha256": "6" * 64,
        },
        "generated_kubeconfig": {
            **identities["kubeconfig"],
            "artifact_id": "kubeconfig-facts",
        },
        "nodes": deploy_nodes,
        "pv_bindings": deploy_pvs,
        "image_digests": identities["images"],
        "input_versions": identities["input_versions"],
        "workload_readiness": {"status": "ready", "unready": []},
        "credential_identity_sha256": "c" * 64,
    }
    selected_records = {
        "datasets": [{"id": "dataset-1", "sha256": "4" * 64}],
        "annotation_revisions": [{"id": "annotation-rev-1", "sha256": "5" * 64}],
        "immutable_objects": [{"id": "object-version-1", "sha256": "6" * 64}],
        "job_history": [{"id": "job-1", "status": "complete"}],
    }

    artifacts: list[dict[str, str]] = []

    def add_artifact(artifact_id: str, relative_path: str, value: Any, media_type: str) -> None:
        if media_type == "json":
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        elif media_type == "yaml":
            payload = value
        elif media_type == "bytes":
            payload = value
        else:
            raise AssertionError(media_type)
        _write_private(root / relative_path, payload)
        artifacts.append({"id": artifact_id, "path": relative_path, "sha256": _sha256(payload), "media_type": media_type})

    add_artifact("inventory", "inventory.yml", inventory_path.read_bytes(), "yaml")
    add_artifact("storage", "retained-storage.yml", storage_path.read_bytes(), "yaml")
    add_artifact("manifest", "manifest.json", manifest, "json")
    add_artifact("snapshot", "snapshot.json", snapshot, "json")
    add_artifact(
        "kubeconfig-facts",
        "kubeconfig-facts.json",
        {"schema_version": 1, **identities["kubeconfig"]},
        "json",
    )
    stage_times = {
        stage: (
            f"2026-10-07T10:{2 + 2 * index:02d}:00+09:00",
            f"2026-10-07T10:{3 + 2 * index:02d}:00+09:00",
        )
        for index, stage in enumerate(_STAGES)
    }
    for node in inventory["nodes"]:
        sliced = slice_retained_manifest_for_node(manifest, inventory, node["name"])
        completed_node_state = _complete_state(inventory, node, plan_id, snapshot_sha256)
        cold_sliced = _node_report_manifest(manifest, inventory, node["name"], "verify-retained", completed_node_state)
        recovery_checks = {
            "source_ownership": len(sliced["retained_paths"]),
            "backup_paths": sum(item.get("role") != "database" for item in sliced["retained_paths"]),
            "database_restores": len(sliced["database_restores"]),
            "credentials": len(sliced["credentials"]),
        }
        cold_checks = {
            "retained_paths": len(sliced["retained_paths"]),
            "database_restores": len(sliced["database_restores"]),
            "credentials": len(sliced["credentials"]),
        }
        add_artifact(
            f"baseline-report-{node['name']}",
            f"baseline-report-{node['name']}.json",
            {"status": "verified", "failures": [], "manifest_sha256": _canonical_sha(sliced), "checks": recovery_checks},
            "json",
        )
        add_artifact(
            f"resume-report-{node['name']}",
            f"resume-report-{node['name']}.json",
            {"status": "verified", "failures": [], "manifest_sha256": _canonical_sha(cold_sliced), "checks": cold_checks},
            "json",
        )
        add_artifact(
            f"repeat-report-{node['name']}",
            f"repeat-report-{node['name']}.json",
            {"status": "verified", "failures": [], "manifest_sha256": _canonical_sha(cold_sliced), "checks": cold_checks},
            "json",
        )

    node_state_artifacts: dict[str, str] = {}
    interrupted_state_artifacts: dict[str, str] = {}
    repeat_state_artifacts: dict[str, str] = {}
    reconnect_state_artifacts: dict[str, str] = {}
    for node in inventory["nodes"]:
        name = node["name"]
        state = _complete_state(inventory, node, plan_id, snapshot_sha256)
        add_artifact(f"resume-state-{name}", f"resume-state-{name}.json", state, "json")
        add_artifact(f"repeat-state-{name}", f"repeat-state-{name}.json", state, "json")
        add_artifact(f"reconnect-state-{name}", f"reconnect-state-{name}.json", state, "json")
        node_state_artifacts[name] = f"resume-state-{name}"
        repeat_state_artifacts[name] = f"repeat-state-{name}"
        reconnect_state_artifacts[name] = f"reconnect-state-{name}"
        if node["role"] == "server":
            interrupted_state = {
                "schema_version": 1,
                "owner": "gods-mlops",
                "node": name,
                "plan_id": plan_id,
                "step": "runtime_verification_pending",
                "service": node["service"],
                "k3s_data_dir": node["k3s_data_dir"],
                "daemonset_snapshot_path": "/secure/recovery/daemonsets.json",
                "daemonset_snapshot_sha256": snapshot_sha256,
            }
        else:
            interrupted_state = state
        add_artifact(f"interrupted-state-{name}", f"interrupted-state-{name}.json", interrupted_state, "json")
        interrupted_state_artifacts[name] = f"interrupted-state-{name}"

    add_artifact("offline-trace", "offline-trace.json", {"selected_branch": "offline_resume", "api_request_events": []}, "json")
    database_restore = manifest["database_restores"][0]
    query_sha256 = "9" * 64
    readback_at = "2026-10-07T10:16:30+09:00"
    logical_restore = {
        "database_restore_manifest_id": database_restore["id"],
        "source_database_id": "source-db",
        "restore_database_id": "restored-db",
        "backup_validation_database_id": "backup-validation-db",
        "restore_validation_database_id": "restore-validation-db",
        "backup_dump_path": database_restore["backup_dump_path"],
        "restored_dump_path": database_restore["restored_dump_path"],
        "source_query_sha256": query_sha256,
        "restored_query_sha256": query_sha256,
        "readback_query_sha256": query_sha256,
        "source_row_count": 2,
        "restored_row_count": 2,
        "readback_row_count": 2,
        "restore_status": "verified",
        "access_report_artifact_id": "database-access",
    }
    database_access = {
        "status": "verified",
        "readback_status": "verified",
        "observed_at": readback_at,
        **logical_restore,
    }
    add_artifact("database-access", "database-access.json", database_access, "json")
    screenshot_bytes = _fixture_png()
    add_artifact("browser-screenshot", "browser-screenshot.png", screenshot_bytes, "bytes")
    browser_observation = {
        "login_status": 303,
        "redirect_followed": True,
        "protected_access_status": 200,
        "protected_path": identities["operator"]["protected_path"],
        "account_identity_sha256": identities["operator"]["account_identity_sha256"],
        "credential_identity_sha256": identities["operator"]["credential_identity_sha256"],
        "session_fingerprint_sha256": "a" * 64,
        "operator_service": identities["operator"]["service_id"],
        "origin": identities["operator"]["origin"],
        "observed_at": readback_at,
    }
    add_artifact(
        "browser-receipt",
        "browser-receipt.json",
        {**browser_observation, "screenshot_sha256": _sha256(screenshot_bytes)},
        "json",
    )
    live_daemonsets = _daemonset_document()
    add_artifact("restored-daemonsets", "restored-daemonsets.json", live_daemonsets, "json")

    rtsp_containers = {
        name: {
            "container_id": f"synthetic-{name}-container-id",
            "image_id": "sha256:" + "7" * 64,
            "running": True,
            "started_at": "2026-10-07T09:00:00+09:00",
            "mounts": [{"name": f"{name}-volume", "destination": "/recordings", "source": "/srv/recordings"}],
        }
        for name in inventory["protected_services"]
    }
    rtsp_snapshots: dict[str, dict[str, Any]] = {}
    for phase, stage in (("before", "baseline"), ("during", "reclaim_resume"), ("after", "reconnect")):
        start_at = stage_times[stage][0]
        observed_at = start_at.replace(":00+09:00", ":30+09:00")
        capture_id = f"rtsp-capture-{phase}"
        snapshot = {
            "capture_artifact_id": capture_id,
            "observed_at": observed_at,
            "containers": rtsp_containers,
            "selected_content_sha256": {"recording-1": "8" * 64},
            "read_only_stream_probe": {"status": "readable", "bytes_read": 1024},
        }
        capture = {
            "schema_version": 1,
            "phase": phase,
            "stage": stage,
            "observed_at": observed_at,
            "containers": snapshot["containers"],
            "selected_content_sha256": snapshot["selected_content_sha256"],
            "read_only_stream_probe": snapshot["read_only_stream_probe"],
        }
        add_artifact(capture_id, f"{capture_id}.json", capture, "json")
        rtsp_snapshots[phase] = snapshot
    observations = {
        "schema_version": 1,
        "stages": {
            "deploy": deploy,
            "reapply": {"before": deploy, "after": deploy},
            "baseline": {
                "recovery_report_artifacts": {node["name"]: f"baseline-report-{node['name']}" for node in inventory["nodes"]},
                "selected_records": selected_records,
                "credential_identity_sha256": "c" * 64,
                "account_identity_sha256": identities["operator"]["account_identity_sha256"],
            },
            "reclaim_interrupted": {
                "controller": {
                    "interrupt_kind": "externally_supervised_ansible_abort",
                    "durable_boundary": "worker verified; server runtime pending",
                    "observed": True,
                },
                "node_state_artifacts": interrupted_state_artifacts,
                "service_status": {
                    node["name"]: {
                        "active": node["role"] == "server",
                        "runtime_status": "pending" if node["role"] == "server" else "verified_stopped",
                        "observed_at": "2026-10-07T10:09:00+09:00",
                    }
                    for node in inventory["nodes"]
                },
            },
            "reclaim_resume": {
                "node_state_artifacts": node_state_artifacts,
                "retained_report_artifacts": {node["name"]: f"resume-report-{node['name']}" for node in inventory["nodes"]},
                "task_order": {"worker_verified_at": "2026-10-07T10:10:00+09:00", "server_stop_started_at": "2026-10-07T10:11:00+09:00"},
            },
            "reclaim_repeat": {
                "node_state_artifacts": repeat_state_artifacts,
                "retained_report_artifacts": {node["name"]: f"repeat-report-{node['name']}" for node in inventory["nodes"]},
                "offline_retry": {
                    "selected": True,
                    "executed_task_names": ["Require matching saved node state and snapshot before an API-free resume"],
                    "trace_artifact_id": "offline-trace",
                },
            },
            "reconnect": {
                "node_state_artifacts": reconnect_state_artifacts,
                "pre_start_retained_report_artifacts": {node["name"]: f"resume-report-{node['name']}" for node in inventory["nodes"]},
                "live_daemonsets_artifact_id": "restored-daemonsets",
                "ready_nodes": {node["name"]: {"ready": True, "uid": deploy_nodes[node["name"]]["uid"]} for node in inventory["nodes"]},
                "deployment_readback": deploy,
            },
            "continuity": {
                "selected_records": selected_records,
                "logical_restore": logical_restore,
                "authentication": {
                    **browser_observation,
                    "browser_receipt_artifact_id": "browser-receipt",
                    "screenshot_artifact_id": "browser-screenshot",
                },
            },
            "rtsp_preserved": rtsp_snapshots,
        },
    }
    add_artifact("observations", "observations.json", observations, "json")

    stage_receipts: list[dict[str, Any]] = []
    extra_outputs = {
        "deploy": ["kubeconfig-facts"],
        "baseline": ["rtsp-capture-before"],
        "reclaim_resume": ["rtsp-capture-during"],
        "reconnect": ["rtsp-capture-after"],
        "continuity": ["database-access", "browser-receipt", "browser-screenshot"],
    }
    for stage in _STAGES:
        attempt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"synthetic:{stage}"))
        start, end = stage_times[stage]
        recipe_id = f"recipe-{stage}"
        add_artifact(
            recipe_id,
            f"{recipe_id}.json",
            {"fixture_only": True, "execution_mode": "fixture_only", "stage": stage},
            "json",
        )
        raw_id = f"native-output-{stage}"
        add_artifact(raw_id, f"{raw_id}.bin", f"synthetic native output for {stage}".encode(), "bytes")
        output_ids = [raw_id, *extra_outputs.get(stage, [])]
        native_receipt_id = f"native-receipt-{stage}"
        native_receipt = {
            "schema_version": 1,
            "kind": "native_stage_receipt",
            "fixture_only": True,
            "stage": stage,
            "attempt_id": attempt_id,
            "source_commit": source_commit,
            "plan_id": plan_id,
            "exit_status": 130 if stage == "reclaim_interrupted" else 0,
            "started_at": start,
            "ended_at": end,
            "output_artifact_ids": sorted(output_ids),
            "output_sha256": {
                identifier: next(item["sha256"] for item in artifacts if item["id"] == identifier)
                for identifier in output_ids
            },
        }
        add_artifact(native_receipt_id, f"{native_receipt_id}.json", native_receipt, "json")
        stage_receipts.append(
            {
                "stage": stage,
                "attempt_id": attempt_id,
                "started_at": start,
                "ended_at": end,
                "exit_status": 130 if stage == "reclaim_interrupted" else 0,
                "source_commit": source_commit,
                "plan_id": plan_id,
                **({"snapshot_sha256": snapshot_sha256} if stage not in {"deploy", "reapply"} else {}),
                "recipe_artifact_id": recipe_id,
                "recipe_sha256": next(item["sha256"] for item in artifacts if item["id"] == recipe_id),
                "native_receipt_artifact_id": native_receipt_id,
                "native_output_artifact_ids": ["observations", native_receipt_id, *output_ids],
            }
        )
    observations["stages"]["reclaim_interrupted"]["controller"]["attempt_id"] = next(
        item["attempt_id"] for item in stage_receipts if item["stage"] == "reclaim_interrupted"
    )
    observations_payload = json.dumps(observations, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    _write_private(root / "observations.json", observations_payload)
    next(item for item in artifacts if item["id"] == "observations")["sha256"] = _sha256(observations_payload)
    bundle = {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "source_commit": source_commit,
        "plan_id": plan_id,
        "snapshot_sha256": snapshot_sha256,
        "provenance": {
            "kind": provenance_kind,
            "execution_mode": "fixture_only",
            "origin": "source test fixture; no live operations",
            "data_origin": "synthetic",
        },
        "identities": identities,
        "bindings": {
            "inventory_artifact_id": "inventory",
            "storage_artifact_id": "storage",
            "manifest_artifact_id": "manifest",
            "snapshot_artifact_id": "snapshot",
        },
        "observations_artifact_id": "observations",
        "artifacts": artifacts,
        "stages": stage_receipts,
    }
    bundle_payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    _write_private(root / "bundle.json", bundle_payload)
    config = {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "expected_source_commit": source_commit,
        "evidence_root": str(root),
        "bundle_path": "bundle.json",
        "bundle_sha256": _sha256(bundle_payload),
        "plan_id": plan_id,
        "snapshot_sha256": snapshot_sha256,
    }
    config_path = root / "config.json"
    _write_private(config_path, json.dumps(config, sort_keys=True).encode())
    return config, bundle


def _write_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(payload)
    path.chmod(0o600)


def test_lifecycle_config_is_opt_out_when_environment_has_no_config() -> None:
    assert _load_lifecycle_config({}) is None


def test_lifecycle_config_requires_current_user_mode_0600_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_private(
        config_path,
        json.dumps(
            {
                "schema_version": 1,
                "fixture_id": "a51a8330-fc99-4792-9471-279e4893de00",
                "expected_source_commit": "a" * 40,
                "evidence_root": str(tmp_path),
                "bundle_path": "bundle.json",
                "bundle_sha256": "b" * 64,
                "plan_id": "c" * 64,
                "snapshot_sha256": "d" * 64,
            }
        ).encode(),
    )
    config_path.chmod(0o644)

    with pytest.raises(ValueError, match="mode 0600"):
        _load_lifecycle_config({"GODS_MLOPS_LIFECYCLE_E2E_CONFIG": str(config_path)})


def test_bound_artifact_loader_rejects_path_escape_and_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    bundle = {
        "schema_version": 1,
        "fixture_id": "a51a8330-fc99-4792-9471-279e4893de00",
        "source_commit": "a" * 40,
        "plan_id": "c" * 64,
        "snapshot_sha256": "d" * 64,
        "artifacts": [{"id": "inventory", "path": "../inventory.yml", "sha256": "e" * 64, "media_type": "yaml"}],
    }
    bundle_bytes = json.dumps(bundle, sort_keys=True).encode()
    _write_private(root / "bundle.json", bundle_bytes)
    config = {
        "schema_version": 1,
        "evidence_root": str(root),
        "bundle_path": "bundle.json",
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "fixture_id": bundle["fixture_id"],
        "expected_source_commit": bundle["source_commit"],
        "plan_id": bundle["plan_id"],
        "snapshot_sha256": bundle["snapshot_sha256"],
    }

    with pytest.raises(ValueError, match="relative path"):
        _load_bound_artifacts(config)

    bundle["artifacts"][0]["path"] = "inventory.yml"
    bundle_bytes = json.dumps(bundle, sort_keys=True).encode()
    _write_private(root / "bundle.json", bundle_bytes)
    config["bundle_sha256"] = hashlib.sha256(bundle_bytes).hexdigest()
    _write_private(root / "inventory.yml", b"all: {}\n")

    with pytest.raises(ValueError, match="hash mismatch"):
        _load_bound_artifacts(config)


def test_bound_artifact_loader_rejects_symlink_substitution(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    outside = tmp_path / "outside.json"
    _write_private(outside, b"{}")
    (root / "artifact.json").symlink_to(outside)
    bundle = {
        "schema_version": 1,
        "fixture_id": "a51a8330-fc99-4792-9471-279e4893de00",
        "source_commit": "a" * 40,
        "plan_id": "c" * 64,
        "snapshot_sha256": "d" * 64,
        "artifacts": [{"id": "receipt", "path": "artifact.json", "sha256": hashlib.sha256(b"{}").hexdigest(), "media_type": "json"}],
    }
    bundle_bytes = json.dumps(bundle, sort_keys=True).encode()
    _write_private(root / "bundle.json", bundle_bytes)
    config = {
        "schema_version": 1,
        "evidence_root": str(root),
        "bundle_path": "bundle.json",
        "bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "fixture_id": bundle["fixture_id"],
        "expected_source_commit": bundle["source_commit"],
        "plan_id": bundle["plan_id"],
        "snapshot_sha256": bundle["snapshot_sha256"],
    }

    with pytest.raises(ValueError, match="regular file"):
        _load_bound_artifacts(config)


def _resign_artifact(root: Path, config: dict[str, Any], bundle: dict[str, Any], artifact_id: str, value: Any) -> None:
    entry = next(item for item in bundle["artifacts"] if item["id"] == artifact_id)
    if entry["media_type"] == "json":
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    elif entry["media_type"] == "yaml":
        if isinstance(value, bytes):
            payload = value
        elif isinstance(value, list):
            payload = "".join(yaml.safe_dump(item, sort_keys=True) + "---\n" for item in value).encode()
        else:
            payload = yaml.safe_dump(value, sort_keys=True).encode()
    elif entry["media_type"] == "bytes":
        payload = value
    else:
        raise AssertionError(entry["media_type"])
    _write_private(root / entry["path"], payload)
    entry["sha256"] = _sha256(payload)
    bundle_payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    _write_private(root / config["bundle_path"], bundle_payload)
    config["bundle_sha256"] = _sha256(bundle_payload)


def _loaded_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = tmp_path / "evidence"
    config, _ = _make_evidence_fixture(root)
    artifacts = _load_bound_artifacts(config)
    return root, config, artifacts["bundle"], artifacts


def test_synthetic_fixture_checks_consistency_without_claiming_live_evidence(tmp_path: Path) -> None:
    _, config, _, artifacts = _loaded_fixture(tmp_path)

    result = _verify_lifecycle_evidence(config, artifacts)

    assert result["performed_by_this_test"] is False
    assert result["lifecycle_pass_claimed"] is False
    assert result["overall"] == "incomplete_or_failed"
    assert {item["status"] for item in result["cases"].values()} == {"not_run"}
    assert result["separate_gates"]["product_browser_detection_search"] == "not_run"


def test_wrong_plan_snapshot_or_missing_manifest_ids_fail_after_bundle_load(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    manifest = _artifact_value(_load_bound_artifacts(config), "manifest")
    manifest["plan_id"] = "0" * 64
    _resign_artifact(root, config, bundle, "manifest", manifest)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["baseline"]["status"] == "failed"
    assert "recovery_manifest_invalid" in result["cases"]["baseline"]["reasons"]

    root, config, bundle, _ = _loaded_fixture(tmp_path / "snapshot")
    snapshot = _artifact_value(_load_bound_artifacts(config), "snapshot")
    snapshot["plan_id"] = "0" * 64
    _resign_artifact(root, config, bundle, "snapshot", snapshot)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["reconnect"]["status"] == "failed"
    assert "daemonset_snapshot_invalid" in result["cases"]["reconnect"]["reasons"]

    root, config, bundle, _ = _loaded_fixture(tmp_path / "coverage")
    manifest = _artifact_value(_load_bound_artifacts(config), "manifest")
    manifest["retained_paths"].pop()
    _resign_artifact(root, config, bundle, "manifest", manifest)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["reclaim_resume"]["status"] == "failed"


def test_changed_inventory_and_source_binding_are_rejected(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    inventory = _artifact_value(_load_bound_artifacts(config), "inventory")
    inventory["all"]["children"]["gods_server"]["hosts"]["vis-lab"]["gods_preserve_paths"].append(
        "/srv/fixture-preserved-path"
    )
    _resign_artifact(root, config, bundle, "inventory", inventory)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["deploy"]["status"] == "failed"
    assert "reclaim_plan_mismatch_or_blocked" in result["cases"]["deploy"]["reasons"]

    _, config, bundle, _ = _loaded_fixture(tmp_path / "source")
    bundle["source_commit"] = "b" * 40
    payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    _write_private(Path(config["evidence_root"]) / config["bundle_path"], payload)
    config["bundle_sha256"] = _sha256(payload)
    with pytest.raises(ValueError, match="source_commit"):
        _load_bound_artifacts(config)


def test_interrupted_and_resumed_states_must_match_plan_and_have_cold_proof(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    state = _artifact_value(_load_bound_artifacts(config), "interrupted-state-vis-lab")
    state["plan_id"] = "0" * 64
    _resign_artifact(root, config, bundle, "interrupted-state-vis-lab", state)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["reclaim_interrupted"]["status"] == "failed"

    root, config, bundle, _ = _loaded_fixture(tmp_path / "resume")
    state = _artifact_value(_load_bound_artifacts(config), "resume-state-ubuntu")
    state["cold_retained_paths"] = {}
    _resign_artifact(root, config, bundle, "resume-state-ubuntu", state)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["reclaim_resume"]["status"] == "failed"
    assert "both_nodes_lack_verified_cold_state" in result["cases"]["reclaim_resume"]["reasons"]


def test_repeat_rejects_changed_cold_state_and_missing_offline_trace(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    state = _artifact_value(_load_bound_artifacts(config), "repeat-state-ubuntu")
    state["cold_k3s_state"]["agent_state_sha256"] = "0" * 64
    _resign_artifact(root, config, bundle, "repeat-state-ubuntu", state)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["reclaim_repeat"]["status"] == "failed"
    assert "repeat_cold_state_changed" in result["cases"]["reclaim_repeat"]["reasons"]


def test_continuity_requires_distinct_restore_and_fresh_preserved_authentication(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["continuity"]["logical_restore"]["restored_dump_path"] = "/recovery/source.sql"
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["continuity"]["status"] == "failed"
    assert "distinct_database_restore_and_readback_missing" in result["cases"]["continuity"]["reasons"]

    root, config, bundle, _ = _loaded_fixture(tmp_path / "auth")
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["continuity"]["authentication"]["login_status"] = 401
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["continuity"]["status"] == "failed"
    assert "fresh_preserved_credential_access_missing" in result["cases"]["continuity"]["reasons"]

    root, config, bundle, _ = _loaded_fixture(tmp_path / "credential")
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["continuity"]["authentication"]["credential_identity_sha256"] = "0" * 64
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["continuity"]["status"] == "failed"
    assert "fresh_preserved_credential_access_missing" in result["cases"]["continuity"]["reasons"]

    root, config, bundle, _ = _loaded_fixture(tmp_path / "missing-receipt")
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["continuity"]["authentication"]["browser_receipt_artifact_id"] = "missing-login-receipt"
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["continuity"]["status"] == "failed"
    assert "fresh_preserved_credential_access_missing" in result["cases"]["continuity"]["reasons"]

    root, config, bundle, _ = _loaded_fixture(tmp_path / "restore")
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["continuity"]["logical_restore"]["restore_status"] = "failed"
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["continuity"]["status"] == "failed"
    assert "distinct_database_restore_and_readback_missing" in result["cases"]["continuity"]["reasons"]


def test_rtsp_case_rejects_container_replacement_or_changed_media_digest(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["rtsp_preserved"]["after"]["containers"]["rtsp-video-loop-app-1"]["container_id"] = "replacement"
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["rtsp_preserved"]["status"] == "failed"
    assert "original_rtsp_identity_mount_or_stream_changed" in result["cases"]["rtsp_preserved"]["reasons"]

    root, config, bundle, _ = _loaded_fixture(tmp_path / "media")
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["rtsp_preserved"]["after"]["selected_content_sha256"]["recording-1"] = "0" * 64
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["rtsp_preserved"]["status"] == "failed"


def test_root_ack_must_bind_the_exact_fixture_and_bundle_hash() -> None:
    config = {"fixture_id": "fixture", "bundle_sha256": "digest"}

    assert not _root_ack_matches(config, {_ACK_ENV: "fixture:different"})
    assert _root_ack_matches(config, {_ACK_ENV: "fixture:digest"})


def test_lifecycle_cycle_evidence_from_root_acknowledged_bundle() -> None:
    config = _load_lifecycle_config(os.environ)
    if config is None:
        pytest.skip(f"set {_CONFIG_ENV} to a protected completed evidence bundle")
    if not _root_ack_matches(config, os.environ):
        pytest.skip(f"set {_ACK_ENV} to <fixture_id>:<bundle_sha256> after root review")

    artifacts = _load_bound_artifacts(config)
    report = _verify_lifecycle_evidence(config, artifacts)

    assert report["performed_by_this_test"] is False
    assert report["lifecycle_pass_claimed"] is False
    assert report["overall"] == "evidence_consistent"
    assert all(case["status"] == "verified_evidence" for case in report["cases"].values())


def test_native_completed_state_accepts_the_literal_producer_record(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    state = _artifact_value(_load_bound_artifacts(config), "resume-state-ubuntu")
    state.pop("retained_failures", None)
    _resign_artifact(root, config, bundle, "resume-state-ubuntu", state)

    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))

    assert result["cases"]["reclaim_resume"]["status"] == "not_run"

    root, config, bundle, _ = _loaded_fixture(tmp_path / "contradiction")
    state = _artifact_value(_load_bound_artifacts(config), "resume-state-ubuntu")
    state["retained_failures"] = ["tree_hash"]
    _resign_artifact(root, config, bundle, "resume-state-ubuntu", state)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["reclaim_resume"]["status"] == "failed"


def test_first_runtime_pending_state_may_lack_a_runtime_result(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    state = _artifact_value(_load_bound_artifacts(config), "interrupted-state-vis-lab")
    state.pop("runtime_status", None)
    state.pop("retained_failures", None)
    state.pop("cold_retained_paths", None)
    state.pop("cold_k3s_state", None)
    _resign_artifact(root, config, bundle, "interrupted-state-vis-lab", state)

    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))

    assert result["cases"]["reclaim_interrupted"]["status"] == "not_run"


def test_native_receipt_bindings_reject_duplicate_outputs_and_attempt_ids(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    bundle["stages"][0]["native_output_artifact_ids"] = ["observations", "observations"]
    _resign_artifact(root, config, bundle, "observations", _artifact_value(_load_bound_artifacts(config), "observations"))
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["deploy"]["status"] == "failed"

    root, config, bundle, _ = _loaded_fixture(tmp_path / "attempt")
    bundle["stages"][1]["attempt_id"] = bundle["stages"][0]["attempt_id"]
    _resign_artifact(root, config, bundle, "observations", _artifact_value(_load_bound_artifacts(config), "observations"))
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["reapply"]["status"] == "failed"

    for field, wrong_value in (
        ("stage", "wrong-stage"),
        ("source_commit", "b" * 40),
        ("plan_id", "0" * 64),
        ("attempt_id", str(uuid.uuid4())),
    ):
        root, config, bundle, _ = _loaded_fixture(tmp_path / field)
        native_receipt = _artifact_value(_load_bound_artifacts(config), "native-receipt-deploy")
        native_receipt[field] = wrong_value
        _resign_artifact(root, config, bundle, "native-receipt-deploy", native_receipt)
        result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
        assert result["cases"]["deploy"]["status"] == "failed"


def test_native_provenance_rejects_explicit_fixture_only_recipe_and_outputs(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    bundle["provenance"]["kind"] = "native"
    _resign_artifact(root, config, bundle, "observations", _artifact_value(_load_bound_artifacts(config), "observations"))

    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))

    assert result["overall"] == "incomplete_or_failed"
    assert all(case["status"] == "failed" for case in result["cases"].values())


def test_recorded_native_execution_may_use_synthetic_input_data() -> None:
    provenance = {
        "kind": "native",
        "execution_mode": "recorded_native",
        "origin": "operator-recorded Ansible execution",
        "data_origin": "synthetic",
    }
    receipt = {
        "recipe_artifact_id": "recipe",
        "native_receipt_artifact_id": "native-receipt",
        "native_output_artifact_ids": ["observations", "native-receipt", "native-output"],
    }
    artifacts = {
        "artifact_values": {
            "recipe": {"fixture_only": False},
            "native-receipt": {"fixture_only": False},
            "native-output": b"recorded native output for a synthetic dataset run",
        }
    }

    assert _native_provenance_errors(provenance, [receipt], artifacts) == []


def test_continuity_requires_bound_row_counts_and_authenticated_receipt(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    logical_restore = observations["stages"]["continuity"]["logical_restore"]
    logical_restore.pop("source_row_count", None)
    logical_restore.pop("restored_row_count", None)
    access_report = _artifact_value(_load_bound_artifacts(config), "database-access")
    access_report.pop("source_rows", None)
    access_report.pop("restored_rows", None)
    _resign_artifact(root, config, bundle, "database-access", access_report)
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["continuity"]["status"] == "failed"

    root, config, bundle, _ = _loaded_fixture(tmp_path / "stale-auth")
    browser_receipt = _artifact_value(_load_bound_artifacts(config), "browser-receipt")
    browser_receipt.update(
        {
            "account_identity_sha256": "0" * 64,
            "credential_identity_sha256": "0" * 64,
            "session_fingerprint_sha256": "0" * 64,
            "operator_service": "unrelated-service",
            "origin": "https://unrelated.invalid",
            "observed_at": "2000-01-01T00:00:00Z",
        }
    )
    _resign_artifact(root, config, bundle, "browser-receipt", browser_receipt)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["continuity"]["status"] == "failed"

    root, config, bundle, _ = _loaded_fixture(tmp_path / "image")
    _resign_artifact(root, config, bundle, "browser-screenshot", b"not an image")
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["continuity"]["status"] == "failed"


def test_native_login_303_followed_by_protected_200_is_accepted(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    browser_receipt = _artifact_value(_load_bound_artifacts(config), "browser-receipt")
    browser_receipt["login_status"] = 303
    _resign_artifact(root, config, bundle, "browser-receipt", browser_receipt)
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["continuity"]["authentication"]["login_status"] = 303
    _resign_artifact(root, config, bundle, "observations", observations)

    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))

    assert result["cases"]["continuity"]["status"] == "not_run"


def test_rtsp_snapshots_require_distinct_time_bound_native_captures(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    snapshots = observations["stages"]["rtsp_preserved"]
    for phase in ("before", "during", "after"):
        snapshots[phase]["observed_at"] = "2000-01-01T00:00:00Z"
        for container in snapshots[phase]["containers"].values():
            container["started_at"] = "2099-01-01T00:00:00Z"
    _resign_artifact(root, config, bundle, "observations", observations)

    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))

    assert result["cases"]["rtsp_preserved"]["status"] == "failed"

    root, config, bundle, _ = _loaded_fixture(tmp_path / "duplicate")
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    snapshots = observations["stages"]["rtsp_preserved"]
    snapshots["during"] = dict(snapshots["before"])
    snapshots["after"] = dict(snapshots["before"])
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["rtsp_preserved"]["status"] == "failed"


def test_explicitly_bound_default_kubeconfig_alias_is_allowed_but_wrong_cluster_fails(tmp_path: Path) -> None:
    root, config, bundle, _ = _loaded_fixture(tmp_path)
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["deploy"]["cluster"]["context"] = "default"
    observations["stages"]["deploy"]["generated_kubeconfig"]["context"] = "default"
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["deploy"]["status"] == "not_run"

    root, config, bundle, _ = _loaded_fixture(tmp_path / "wrong-cluster")
    observations = _artifact_value(_load_bound_artifacts(config), "observations")
    observations["stages"]["deploy"]["cluster"]["uid"] = "other-cluster-uid"
    _resign_artifact(root, config, bundle, "observations", observations)
    result = _verify_lifecycle_evidence(config, _load_bound_artifacts(config))
    assert result["cases"]["deploy"]["status"] == "failed"
