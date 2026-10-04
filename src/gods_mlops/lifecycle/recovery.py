#!/usr/bin/env python3
"""Recovery artifact verification and explicitly confirmed purge targets."""

from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any


OWNER = "gods-mlops"
OWNER_MARKER = ".gods-mlops-owner.json"
SHA256_LENGTH = 64
DATABASE_DUMP_FORMATS = {
    "postgresql-sql-v1": {
        "backup_tool": "pg_dump",
        "restore_tool": "psql",
        "redump_tool": "pg_dump",
        "required_options": {"--format=plain", "--column-inserts", "--rows-per-insert=1", "--no-owner", "--no-acl"},
    },
    "mysql-sql-v1": {
        "backup_tool": "mysqldump",
        "restore_tool": "mysql",
        "redump_tool": "mysqldump",
        "required_options": {
            "--skip-comments",
            "--skip-dump-date",
            "--skip-extended-insert",
            "--order-by-primary",
            "--routines",
            "--events",
            "--triggers",
            "--no-tablespaces",
            "--single-transaction",
            "--set-gtid-purged=OFF",
        },
    },
}


def verify_retained(
    manifest: dict[str, Any],
    *,
    requirements: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Verify real paths and hashes; manifest booleans are never accepted as proof."""
    failures: list[dict[str, str]] = []
    checks = {"retained_paths": 0, "database_restores": 0, "credentials": 0}
    require_every_category = requirements is None
    requirements = requirements or {}

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("owner") != OWNER:
        failures.append({"check": "manifest_identity", "id": "manifest", "reason": "expected schema_version 1 and owner gods-mlops"})
        return _verification_result(manifest if isinstance(manifest, dict) else {}, checks, failures)

    retained_paths = manifest.get("retained_paths")
    database_restores = manifest.get("database_restores")
    credentials = manifest.get("credentials")
    for key, value in (("retained_paths", retained_paths), ("database_restores", database_restores), ("credentials", credentials)):
        if not isinstance(value, list):
            failures.append({"check": f"{key}_manifest", "id": key, "reason": "entry list is missing or invalid"})
            if key == "retained_paths":
                retained_paths = []
            elif key == "database_restores":
                database_restores = []
            else:
                credentials = []
        elif not value and (require_every_category or requirements.get(key, [])):
            failures.append({"check": f"{key}_manifest", "id": key, "reason": "at least one verified entry is required"})

    retained_ids: set[str] = set()
    cold_retained_paths: dict[str, str] = {}
    for entry in retained_paths:
        identifier = _entry_id(entry, failures, "retained_path")
        if identifier is None:
            continue
        retained_ids.add(identifier)
        checks["retained_paths"] += 1
        if isinstance(entry, dict) and entry.get("role") == "database":
            source_hash = _verify_owned_tree(entry.get("source_path"), entry.get("source_owner"), identifier, "source", failures)
        else:
            source_hash = _verify_tree_pair(entry, identifier, failures)
        if source_hash is not None:
            cold_retained_paths[identifier] = source_hash

    database_ids: set[str] = set()
    for entry in database_restores:
        identifier = _entry_id(entry, failures, "database_restore")
        if identifier is None:
            continue
        database_ids.add(identifier)
        checks["database_restores"] += 1
        _verify_database_restore(entry, identifier, failures)

    credential_ids: set[str] = set()
    for entry in credentials:
        identifier = _entry_id(entry, failures, "credential")
        if identifier is None:
            continue
        credential_ids.add(identifier)
        checks["credentials"] += 1
        _verify_credential(entry, identifier, failures)

    _check_required_ids("retained_paths", retained_ids, requirements, failures)
    _check_required_ids("database_restores", database_ids, requirements, failures)
    _check_required_ids("credentials", credential_ids, requirements, failures)
    if manifest.get("cold_retained_paths_expected") is True:
        expected_paths = manifest.get("expected_cold_retained_paths")
        if not isinstance(expected_paths, dict):
            failures.append({"check": "cold_retained_paths", "id": "manifest", "reason": "recorded cold path hashes are missing"})
        elif expected_paths != cold_retained_paths:
            failures.append({"check": "cold_retained_paths", "id": "manifest", "reason": "retained path hashes differ from stopped-state evidence"})
    cold_k3s_state = None
    if manifest.get("k3s_data_dir") and manifest.get("node_role"):
        try:
            cold_k3s_state = inspect_k3s_runtime_state(
                manifest["k3s_data_dir"],
                role=manifest["node_role"],
                expected_uid=manifest.get("k3s_expected_uid"),
                service_exec_start_sha256=manifest.get("service_exec_start_sha256"),
            )
        except (OSError, ValueError) as exc:
            failures.append({"check": "k3s_authoritative_state", "id": f"{manifest.get('node', 'node')}-k3s-state", "reason": str(exc)})
        else:
            expected_cold_state = manifest.get("expected_cold_k3s_state")
            if manifest.get("cold_k3s_state_expected") is True:
                if not isinstance(expected_cold_state, dict):
                    failures.append({"check": "k3s_authoritative_state", "id": f"{manifest.get('node', 'node')}-k3s-state", "reason": "cold K3s state evidence is missing"})
                elif expected_cold_state != cold_k3s_state:
                    failures.append({"check": "k3s_authoritative_state", "id": f"{manifest.get('node', 'node')}-k3s-state", "reason": "cold K3s state hashes differ from the recorded stopped-state evidence"})
    report = _verification_result(manifest, checks, failures)
    report["cold_retained_paths"] = cold_retained_paths
    if cold_k3s_state is not None:
        report["cold_k3s_state"] = cold_k3s_state
    return report


def verify_recovery_artifacts(
    manifest: dict[str, Any],
    *,
    requirements: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Pre-stop gate: check source ownership and real backups without hashing live data."""
    failures: list[dict[str, str]] = []
    checks = {"source_ownership": 0, "backup_paths": 0, "database_restores": 0, "credentials": 0}
    require_every_category = requirements is None
    requirements = requirements or {}
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("owner") != OWNER:
        failures.append({"check": "manifest_identity", "id": "manifest", "reason": "expected schema_version 1 and owner gods-mlops"})
        return _verification_result(manifest if isinstance(manifest, dict) else {}, checks, failures)

    retained_paths = manifest.get("retained_paths")
    database_restores = manifest.get("database_restores")
    credentials = manifest.get("credentials")
    lists = (("retained_paths", retained_paths), ("database_restores", database_restores), ("credentials", credentials))
    for key, value in lists:
        if not isinstance(value, list):
            failures.append({"check": f"{key}_manifest", "id": key, "reason": "entry list is missing or invalid"})
            if key == "retained_paths":
                retained_paths = []
            elif key == "database_restores":
                database_restores = []
            else:
                credentials = []
        elif not value and (require_every_category or requirements.get(key, [])):
            failures.append({"check": f"{key}_manifest", "id": key, "reason": "at least one verified entry is required"})

    retained_ids: set[str] = set()
    for entry in retained_paths:
        identifier = _entry_id(entry, failures, "retained_path")
        if identifier is None:
            continue
        retained_ids.add(identifier)
        checks["source_ownership"] += 1
        if not isinstance(entry, dict):
            continue
        _verify_source_identity(entry.get("source_path"), entry.get("source_owner"), identifier, failures)
        if entry.get("role") != "database":
            checks["backup_paths"] += 1
            _verify_backup_tree(entry, identifier, failures)

    database_ids: set[str] = set()
    for entry in database_restores:
        identifier = _entry_id(entry, failures, "database_restore")
        if identifier is None:
            continue
        database_ids.add(identifier)
        checks["database_restores"] += 1
        _verify_database_restore(entry, identifier, failures)

    credential_ids: set[str] = set()
    for entry in credentials:
        identifier = _entry_id(entry, failures, "credential")
        if identifier is None:
            continue
        credential_ids.add(identifier)
        checks["credentials"] += 1
        _verify_credential(entry, identifier, failures)

    _check_required_ids("retained_paths", retained_ids, requirements, failures)
    _check_required_ids("database_restores", database_ids, requirements, failures)
    _check_required_ids("credentials", credential_ids, requirements, failures)
    return _verification_result(manifest, checks, failures)


def owned_daemonsets(document: dict[str, Any], *, require_complete: bool = False) -> dict[str, Any]:
    """Return only the known Gods plugin and Kubeflow CNI DaemonSets to stop."""
    items = document.get("items") if isinstance(document, dict) else None
    if not isinstance(items, list):
        raise ValueError("Kubernetes DaemonSet response must contain an items list")
    allowed: dict[tuple[str, str], bool] = {}
    for item in items:
        metadata = item.get("metadata", {})
        namespace = str(metadata.get("namespace", ""))
        name = str(metadata.get("name", ""))
        labels = metadata.get("labels") or {}
        if namespace == "kube-system" and name == "istio-cni-node" and labels.get("app.kubernetes.io/name") == "istio-cni":
            allowed[(namespace, name)] = True
            continue
        if namespace == "nvidia-device-plugin" and labels.get("app.kubernetes.io/instance") == "nvidia-device-plugin":
            allowed[(namespace, name)] = True
            continue
        raise ValueError(f"unrecognized DaemonSet blocks reclaim: {namespace}/{name}")
    has_istio = any(namespace == "kube-system" and name == "istio-cni-node" for namespace, name in allowed)
    has_nvidia = any(namespace == "nvidia-device-plugin" for namespace, _ in allowed)
    if require_complete and not (has_istio and has_nvidia):
        missing = []
        if not has_istio:
            missing.append("kube-system/istio-cni-node")
        if not has_nvidia:
            missing.append("nvidia-device-plugin/*")
        raise ValueError(f"cannot capture or restore reclaim state; required owned DaemonSets are missing: {missing}")
    return {
        "status": "verified",
        "delete": [{"namespace": namespace, "name": name} for namespace, name in sorted(allowed)],
    }


def build_daemonset_snapshot(document: dict[str, Any], *, plan_id: str) -> dict[str, Any]:
    scoped = owned_daemonsets(document, require_complete=True)
    by_identity = {
        (str(item.get("metadata", {}).get("namespace", "")), str(item.get("metadata", {}).get("name", ""))): item
        for item in document.get("items", [])
    }
    objects = [
        _sanitize_daemonset(by_identity[(item["namespace"], item["name"])])
        for item in scoped["delete"]
    ]
    digest = _json_sha256(objects)
    return {
        "schema_version": 1,
        "owner": OWNER,
        "plan_id": plan_id,
        "objects": objects,
        "snapshot_sha256": digest,
    }


def validate_daemonset_snapshot(
    snapshot: dict[str, Any],
    *,
    confirmation: str,
    expected_plan_id: str,
) -> dict[str, Any]:
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1 or snapshot.get("owner") != OWNER:
        raise ValueError("DaemonSet snapshot must declare schema_version 1 and owner gods-mlops")
    if snapshot.get("plan_id") != expected_plan_id:
        raise ValueError("DaemonSet snapshot belongs to a different reclaim plan")
    objects = snapshot.get("objects")
    if not isinstance(objects, list):
        raise ValueError("DaemonSet snapshot objects must be a list")
    scoped = owned_daemonsets({"items": objects}, require_complete=True)
    actual_digest = _json_sha256(objects)
    if snapshot.get("snapshot_sha256") != actual_digest:
        raise ValueError("DaemonSet snapshot digest does not match its objects")
    if confirmation != actual_digest:
        raise ValueError(f"snapshot confirmation digest mismatch; expected {actual_digest}")
    return {
        "status": "snapshot_valid",
        "plan_id": expected_plan_id,
        "snapshot_sha256": actual_digest,
        "objects": scoped["delete"],
        "apply_objects": objects,
    }


def verify_restored_daemonsets(snapshot_objects: list[dict[str, Any]], live_document: dict[str, Any]) -> dict[str, Any]:
    scoped = owned_daemonsets(live_document, require_complete=True)
    by_identity = {
        (str(item.get("metadata", {}).get("namespace", "")), str(item.get("metadata", {}).get("name", ""))): item
        for item in live_document.get("items", [])
    }
    actual = [
        _sanitize_daemonset(by_identity[(item["namespace"], item["name"])])
        for item in scoped["delete"]
    ]
    expected = sorted(snapshot_objects, key=lambda item: (item["metadata"].get("namespace", ""), item["metadata"]["name"]))
    if actual != expected:
        raise ValueError("live owned DaemonSet objects do not match the captured snapshot")
    return {"status": "verified", "objects": scoped["delete"]}


def active_workload_pods(
    document: dict[str, Any],
    *,
    ignore_daemonsets: bool = False,
) -> list[dict[str, str]]:
    """List scheduled active pods, including DaemonSets; static mirror pods are excluded."""
    items = document.get("items") if isinstance(document, dict) else None
    if not isinstance(items, list):
        raise ValueError("Kubernetes Pod response must contain an items list")
    active: list[dict[str, str]] = []
    for item in items:
        metadata = item.get("metadata", {})
        if (metadata.get("annotations") or {}).get("kubernetes.io/config.mirror"):
            continue
        phase = str(item.get("status", {}).get("phase", "Unknown"))
        node = str(item.get("spec", {}).get("nodeName", ""))
        if phase not in {"Running", "Pending", "Unknown"} or not node:
            continue
        owners = metadata.get("ownerReferences") or []
        owner_kind = ",".join(sorted({str(owner.get("kind", "unknown")) for owner in owners})) or "unowned"
        if ignore_daemonsets and owner_kind == "DaemonSet":
            continue
        active.append({
            "namespace": str(metadata.get("namespace", "default")),
            "name": str(metadata.get("name", "unknown")),
            "phase": phase,
            "node": node,
            "owner_kind": owner_kind,
        })
    return sorted(active, key=lambda item: (item["namespace"], item["name"]))


def validate_emptydir_policy(document: dict[str, Any]) -> dict[str, Any]:
    """Allow only rendered disposable volumes and known injected Istio runtime mounts."""
    items = document.get("items") if isinstance(document, dict) else None
    if not isinstance(items, list):
        raise ValueError("Kubernetes Pod response must contain an items list")
    rendered_policies = [
        ("istio-system", {"app": "cluster-local-gateway"}, {"workload-socket", "credential-socket", "workload-certs", "istio-envoy", "istio-data"}),
        ("istio-system", {"app": "istio-ingressgateway"}, {"workload-socket", "credential-socket", "workload-certs", "istio-envoy", "istio-data"}),
        ("istio-system", {"app": "istiod"}, {"local-certs"}),
        ("kubeflow", {"app.kubernetes.io/name": "model-catalog", "app.kubernetes.io/component": "server"}, {"perf-data"}),
        ("kubeflow", {"app.kubernetes.io/instance": "spark-operator", "app.kubernetes.io/component": "controller"}, {"tmp"}),
        ("kubeflow", {"app.kubernetes.io/instance": "spark-operator", "app.kubernetes.io/component": "webhook"}, {"serving-certs"}),
    ]
    istio_injected_namespaces = {"gods-mlops", "kubeflow"}
    istio_runtime_volumes = {"workload-socket", "credential-socket", "workload-certs", "istio-envoy", "istio-data"}
    approved: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for pod in items:
        metadata = pod.get("metadata", {})
        if (metadata.get("annotations") or {}).get("kubernetes.io/config.mirror"):
            continue
        spec = pod.get("spec", {})
        node = str(spec.get("nodeName", ""))
        phase = str(pod.get("status", {}).get("phase", "Unknown"))
        if not node or phase not in {"Running", "Pending", "Unknown"}:
            continue
        empty_names = {
            str(volume.get("name"))
            for volume in spec.get("volumes", [])
            if isinstance(volume, dict) and "emptyDir" in volume
        }
        if not empty_names:
            continue
        namespace = str(metadata.get("namespace", "default"))
        labels = metadata.get("labels") or {}
        owners = metadata.get("ownerReferences") or []
        owner_kinds = {str(owner.get("kind", "")) for owner in owners}
        matching_workload_volumes = [
            allowed_volumes
            for allowed_namespace, required_labels, allowed_volumes in rendered_policies
            if namespace == allowed_namespace and all(labels.get(key) == value for key, value in required_labels.items())
        ]
        workload_volume_names = set().union(*matching_workload_volumes) if matching_workload_volumes else set()
        injected_names = empty_names.intersection(istio_runtime_volumes)
        other_names = empty_names.difference(workload_volume_names).difference(istio_runtime_volumes)
        has_proxy = "istio-proxy" in {str(container.get("name")) for container in spec.get("containers", [])}
        rendered_disposable = bool(matching_workload_volumes) and not other_names and (not injected_names or has_proxy)
        injected_runtime_only = (
            namespace in istio_injected_namespaces
            and has_proxy
            and empty_names.issubset(istio_runtime_volumes)
            and bool(owner_kinds.intersection({"ReplicaSet", "StatefulSet", "DaemonSet", "Job"}))
        )
        if rendered_disposable or injected_runtime_only:
            if rendered_disposable and injected_names and any(name not in workload_volume_names for name in injected_names):
                classification = "rendered-disposable-plus-injected-istio-runtime"
            else:
                classification = "rendered-disposable" if rendered_disposable else "injected-istio-runtime"
            approved.append({
                "namespace": namespace,
                "name": str(metadata.get("name", "unknown")),
                "node": node,
                "volumes": sorted(empty_names),
                "classification": classification,
            })
        else:
            blockers.append({
                "namespace": namespace,
                "name": str(metadata.get("name", "unknown")),
                "node": node,
                "volumes": sorted(empty_names),
                "reason": "emptyDir volume is outside the explicit disposable-volume policy",
            })
    approved.sort(key=lambda item: (item["node"], item["namespace"], item["name"]))
    blockers.sort(key=lambda item: (item["node"], item["namespace"], item["name"]))
    return {
        "status": "verified" if not blockers else "blocked",
        "approved_pods": approved,
        "blockers": blockers,
    }


def hash_tree(path: str | Path, *, excluded_paths: tuple[str, ...] = ()) -> str:
    """Hash a file or directory tree by relative names, types, and file bytes."""
    root = Path(path)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ValueError(f"path is unavailable: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"symlink roots are not accepted: {root}")
    digest = hashlib.sha256()
    if stat.S_ISREG(root_stat.st_mode):
        _add_file(digest, ".", root)
        return digest.hexdigest()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"path is not a regular file or directory: {root}")

    excluded = {PurePosixPath(item) for item in excluded_paths}
    stack = [(root, PurePosixPath("."))]
    while stack:
        current, relative = stack.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda item: item.name, reverse=True)
        except OSError as exc:
            raise ValueError(f"directory cannot be read: {current}") from exc
        for entry in entries:
            child = Path(entry.path)
            relative_child = relative / entry.name
            if any(relative_child == item or item in relative_child.parents for item in excluded):
                continue
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as exc:
                raise ValueError(f"entry cannot be inspected: {child}") from exc
            if stat.S_ISLNK(mode):
                target = os.readlink(child)
                _add_record(digest, b"L", os.fsencode(relative_child.as_posix()), os.fsencode(target))
                continue
            if stat.S_ISDIR(mode):
                _add_record(digest, b"D", relative_child.as_posix().encode("utf-8"))
                stack.append((child, relative_child))
            elif stat.S_ISREG(mode):
                _add_file(digest, relative_child.as_posix(), child)
            else:
                raise ValueError(f"special filesystem objects are not accepted: {child}")
    return digest.hexdigest()


def inspect_k3s_runtime_state(
    data_dir: str | Path,
    *,
    role: str,
    expected_uid: int | None = None,
    service_exec_start_sha256: str | None = None,
) -> dict[str, str]:
    """Hash authoritative cold K3s state while excluding the reproducible agent containerd cache."""
    root = Path(data_dir)
    if role == "server":
        database = root / "server" / "db"
        sqlite_file = database / "state.db"
        etcd_path = database / "etcd"
        if etcd_path.exists() or etcd_path.is_symlink():
            raise ValueError("K3s state contains embedded etcd files, but the configured Gods server uses the default SQLite datastore")
        try:
            sqlite_metadata = sqlite_file.lstat()
        except OSError as exc:
            raise ValueError("K3s SQLite state.db is missing; authoritative server state cannot be verified") from exc
        if not stat.S_ISREG(sqlite_metadata.st_mode):
            raise ValueError("K3s SQLite state.db is missing; authoritative server state cannot be verified")
        required = {
            "server/db": database,
            "server/token": root / "server" / "token",
            "server/tls": root / "server" / "tls",
            "server/cred": root / "server" / "cred",
            "server/manifests": root / "server" / "manifests",
            "agent": root / "agent",
        }
        missing = [label for label, path in required.items() if not path.exists()]
        if missing:
            raise ValueError(f"required SQLite server state paths are missing: {missing}")
        for label, path in required.items():
            metadata = path.lstat()
            expected_type = stat.S_ISDIR(metadata.st_mode) if label != "server/token" else stat.S_ISREG(metadata.st_mode)
            if not expected_type:
                raise ValueError(f"required K3s server state path has an unexpected filesystem type: {label}")
            if expected_uid is not None and (metadata.st_uid != expected_uid or metadata.st_gid != 0):
                raise ValueError(f"required K3s server state path is not root-owned: {label}")
        return {
            "datastore": "sqlite",
            "server_db_sha256": hash_tree(database),
            "server_token_sha256": _file_sha256(required["server/token"]),
            "server_tls_sha256": hash_tree(required["server/tls"]),
            "server_cred_sha256": hash_tree(required["server/cred"]),
            "server_manifests_sha256": hash_tree(required["server/manifests"]),
            "server_agent_state_sha256": hash_tree(required["agent"], excluded_paths=("containerd",)),
            "server_agent_excluded_runtime": "containerd",
            "service_exec_start_sha256": service_exec_start_sha256 or "",
        }
    if role == "worker":
        agent_root = root / "agent"
        if not agent_root.is_dir():
            raise ValueError("K3s agent state directory is missing")
        metadata = agent_root.lstat()
        if expected_uid is not None and (metadata.st_uid != expected_uid or metadata.st_gid != 0):
            raise ValueError("K3s agent state directory is not root-owned")
        return {
            "agent_state_sha256": hash_tree(agent_root, excluded_paths=("containerd",)),
            "agent_excluded_runtime": "containerd",
            "service_exec_start_sha256": service_exec_start_sha256 or "",
        }
    raise ValueError(f"unsupported Gods K3s node role for cold state: {role}")


def purge_targets_digest(targets: dict[str, Any]) -> str:
    """Return the exact confirmation digest for a structurally valid target set."""
    normalized = _normalize_purge_targets(targets)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_purge_targets(
    targets: dict[str, Any],
    *,
    confirmation: str | None = None,
) -> dict[str, Any]:
    """Validate exact, in-root targets and require their digest to authorize purge."""
    normalized = _normalize_purge_targets(targets)
    expected = purge_targets_digest(normalized)
    if not confirmation:
        raise ValueError(f"purge requires --confirm-targets {expected}")
    if confirmation != expected:
        raise ValueError("purge target confirmation digest does not match the supplied target list")
    return normalized


def _verify_tree_pair(entry: Any, identifier: str, failures: list[dict[str, str]]) -> str | None:
    if not isinstance(entry, dict):
        return None
    expected = entry.get("expected_tree_sha256")
    source = entry.get("source_path")
    backup = entry.get("backup_path")
    if not _valid_sha256(expected):
        failures.append({"check": "tree_hash_manifest", "id": identifier, "reason": "expected_tree_sha256 must be a SHA-256 digest"})
        return None
    source_hash = _verify_owned_tree(source, entry.get("source_owner"), identifier, "source", failures)
    backup_hash = _verify_owned_tree(
        backup,
        entry.get("backup_owner"),
        identifier,
        "backup",
        failures,
        expected_mode=entry.get("backup_mode"),
    )
    if source_hash is not None and source_hash != expected:
        failures.append({"check": "source_hash", "id": identifier, "reason": "source tree does not match its declared digest"})
    if backup_hash is not None and backup_hash != expected:
        failures.append({"check": "backup_hash", "id": identifier, "reason": "backup tree does not match its declared digest"})
    if source_hash is not None and backup_hash is not None and source_hash != backup_hash:
        failures.append({"check": "backup_hash", "id": identifier, "reason": "source and backup tree hashes differ"})
    return source_hash


def _verify_backup_tree(entry: dict[str, Any], identifier: str, failures: list[dict[str, str]]) -> None:
    expected = entry.get("expected_tree_sha256")
    if not _valid_sha256(expected):
        failures.append({"check": "tree_hash_manifest", "id": identifier, "reason": "expected_tree_sha256 must be a SHA-256 digest"})
        return
    backup_hash = _verify_owned_tree(
        entry.get("backup_path"),
        entry.get("backup_owner"),
        identifier,
        "backup",
        failures,
        expected_mode=entry.get("backup_mode"),
    )
    if backup_hash is not None and backup_hash != expected:
        failures.append({"check": "backup_hash", "id": identifier, "reason": "backup tree does not match its declared digest"})


def _verify_source_identity(
    value: Any,
    owner: Any,
    identifier: str,
    failures: list[dict[str, str]],
) -> None:
    if not isinstance(value, str) or not value.startswith("/"):
        failures.append({"check": "source_path", "id": identifier, "reason": "an absolute source path is required"})
        return
    if not isinstance(owner, dict) or not isinstance(owner.get("uid"), int) or not isinstance(owner.get("gid"), int):
        failures.append({"check": "source_ownership", "id": identifier, "reason": "expected uid and gid are required"})
        return
    path = Path(value)
    try:
        metadata = path.lstat()
    except OSError as exc:
        failures.append({"check": "source_path", "id": identifier, "reason": f"source path is unavailable: {exc}"})
        return
    if stat.S_ISLNK(metadata.st_mode) or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
        failures.append({"check": "source_ownership", "id": identifier, "reason": "source root must be a real directory or regular file"})
    if metadata.st_uid != owner["uid"] or metadata.st_gid != owner["gid"]:
        failures.append({"check": "source_ownership", "id": identifier, "reason": "source path uid/gid does not match the manifest"})


def _verify_owned_tree(
    value: Any,
    owner: Any,
    identifier: str,
    label: str,
    failures: list[dict[str, str]],
    *,
    expected_mode: Any = None,
) -> str | None:
    if not isinstance(value, str) or not value.startswith("/"):
        failures.append({"check": f"{label}_path", "id": identifier, "reason": "an absolute path is required"})
        return None
    if not isinstance(owner, dict) or not isinstance(owner.get("uid"), int) or not isinstance(owner.get("gid"), int):
        failures.append({"check": f"{label}_ownership", "id": identifier, "reason": "expected uid and gid are required"})
        return None
    path = Path(value)
    try:
        metadata = path.lstat()
        actual_uid = metadata.st_uid
        actual_gid = metadata.st_gid
        actual_hash = hash_tree(path)
    except (OSError, ValueError) as exc:
        failures.append({"check": f"{label}_path", "id": identifier, "reason": str(exc)})
        return None
    if stat.S_ISLNK(metadata.st_mode):
        failures.append({"check": f"{label}_ownership", "id": identifier, "reason": "symlink roots are not accepted"})
    if actual_uid != owner["uid"] or actual_gid != owner["gid"]:
        failures.append({"check": f"{label}_ownership", "id": identifier, "reason": "path uid/gid does not match the manifest"})
    if expected_mode is not None:
        try:
            required_mode = int(str(expected_mode), 8)
        except ValueError:
            failures.append({"check": f"{label}_permissions", "id": identifier, "reason": "expected_mode must be an octal mode"})
        else:
            if stat.S_IMODE(metadata.st_mode) != required_mode:
                failures.append({"check": f"{label}_permissions", "id": identifier, "reason": "path mode does not match the manifest"})
    return actual_hash


def _verify_database_restore(entry: Any, identifier: str, failures: list[dict[str, str]]) -> None:
    if not isinstance(entry, dict):
        return
    expected = entry.get("expected_sha256")
    if not _valid_sha256(expected):
        failures.append({"check": "database_restore", "id": identifier, "reason": "expected_sha256 must be a SHA-256 digest"})
        return
    source_path = entry.get("backup_dump_path")
    restored_path = entry.get("restored_dump_path")
    if _same_file_identity(source_path, restored_path):
        failures.append({"check": "database_restore_separation", "id": identifier, "reason": "the restore proof must be a separate file and inode from the backup dump"})
    dump_format = _verify_database_dump_provenance(entry, identifier, failures)
    source_file_hash = _verify_file_hash(source_path, identifier, "database_backup", failures)
    restored_file_hash = _verify_file_hash(restored_path, identifier, "database_restore", failures)
    source_hash = restored_hash = None
    if dump_format is not None and source_file_hash is not None:
        source_hash = _verify_canonical_database_dump(source_path, identifier, "database_backup", dump_format, failures)
    if dump_format is not None and restored_file_hash is not None:
        restored_hash = _verify_canonical_database_dump(restored_path, identifier, "database_restore", dump_format, failures)
    expected_uid = entry.get("expected_uid")
    expected_mode = entry.get("expected_mode")
    if not isinstance(expected_uid, int) or expected_mode != "0600":
        failures.append({"check": "database_restore_permissions", "id": identifier, "reason": "root-owned mode-0600 database dumps are required"})
    else:
        for value, label in ((source_path, "database_backup"), (restored_path, "database_restore")):
            _verify_private_file_metadata(value, identifier, label, expected_uid, 0o600, failures)
    if source_hash is not None and source_hash != expected:
        failures.append({"check": "database_restore", "id": identifier, "reason": "database backup digest does not match"})
    if restored_hash is not None and restored_hash != expected:
        failures.append({"check": "database_restore", "id": identifier, "reason": "restored database dump digest does not match the backup"})


def _verify_database_dump_provenance(
    entry: dict[str, Any], identifier: str, failures: list[dict[str, str]]
) -> str | None:
    dump_format = entry.get("dump_format")
    policy = DATABASE_DUMP_FORMATS.get(dump_format) if isinstance(dump_format, str) else None
    provenance = entry.get("provenance")
    if policy is None or not isinstance(provenance, dict):
        failures.append({"check": "database_restore_provenance", "id": identifier, "reason": "a supported dump format and executable provenance are required"})
        return None
    required_fields = (
        "backup_tool",
        "backup_tool_version",
        "restore_tool",
        "restore_tool_version",
        "redump_tool",
        "redump_tool_version",
    )
    if any(not isinstance(provenance.get(field), str) or not provenance[field].strip() for field in required_fields):
        failures.append({"check": "database_restore_provenance", "id": identifier, "reason": "tool names and exact client versions must be recorded"})
        return None
    tools_match = all(provenance[field] == policy[field] for field in ("backup_tool", "restore_tool", "redump_tool"))
    versions_match = provenance["backup_tool_version"] == provenance["redump_tool_version"]
    options = provenance.get("options")
    options_valid = isinstance(options, list) and all(isinstance(option, str) for option in options)
    if not tools_match or not versions_match or not options_valid:
        failures.append({"check": "database_restore_provenance", "id": identifier, "reason": "backup and redump must use the same supported dumper version and explicit options"})
        return None
    missing_options = policy["required_options"].difference(options)
    if missing_options:
        failures.append({"check": "database_restore_provenance", "id": identifier, "reason": f"deterministic dump options are missing: {sorted(missing_options)}"})
        return None
    return str(dump_format)


def _verify_canonical_database_dump(
    value: Any,
    identifier: str,
    label: str,
    dump_format: str,
    failures: list[dict[str, str]],
) -> str | None:
    try:
        return canonical_database_dump_sha256(value, dump_format=dump_format)
    except (OSError, UnicodeError, ValueError) as exc:
        failures.append({"check": label, "id": identifier, "reason": f"database dump cannot be canonicalized: {exc}"})
        return None


def canonical_database_dump_sha256(value: str | Path, *, dump_format: str) -> str:
    """Hash a deterministic logical dump while preserving statement/table order."""
    if dump_format not in DATABASE_DUMP_FORMATS:
        raise ValueError(f"unsupported canonical SQL dump format: {dump_format}")
    path = Path(value)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("database dump must be a regular non-symlink file")
    canonical_lines: list[str] = []
    insert_block: list[str] = []
    insert_target: str | None = None

    def flush_inserts() -> None:
        if insert_block:
            canonical_lines.extend(sorted(insert_block))
            insert_block.clear()

    identifier = r'(?:`[^`]+`|"(?:""|[^"])+"|[A-Za-z_][A-Za-z0-9_$]*)'
    insert_target_pattern = re.compile(
        rf"^INSERT\s+INTO\s+(?P<table>{identifier}(?:\s*\.\s*{identifier})?)(?:\s*\(|\s+VALUES\b)",
        flags=re.IGNORECASE,
    )
    with path.open("r", encoding="utf-8", newline=None) as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            if dump_format == "postgresql-sql-v1":
                if re.match(r"^\\(?:un)?restrict(?:\s|$)", line):
                    continue
                if line.startswith(("-- Dumped from database version", "-- Dumped by pg_dump version", "-- Dump completed on")):
                    continue
            elif line.startswith("--"):
                continue
            elif re.match(r"^USE\s+.+;\s*$", line, flags=re.IGNORECASE):
                # The validation database has a different name from the source.
                line = "USE <logical-database>;"
            if line.upper().startswith("COPY "):
                raise ValueError("COPY data ordering is not canonical; use the recorded one-row INSERT options")
            if line.upper().startswith("INSERT INTO "):
                if not line.endswith(";"):
                    raise ValueError("multi-line INSERT statements are not supported by canonical SQL dump v1")
                match = insert_target_pattern.match(line)
                if match is None:
                    raise ValueError("INSERT target cannot be identified for scoped row comparison")
                target = match.group("table")
                if insert_target is not None and target != insert_target:
                    flush_inserts()
                insert_target = target
                insert_block.append(line)
                continue
            flush_inserts()
            insert_target = None
            canonical_lines.append(line)
    flush_inserts()
    canonical_bytes = ("\n".join(canonical_lines) + "\n").encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def _same_file_identity(source_path: Any, restored_path: Any) -> bool:
    if not isinstance(source_path, str) or not isinstance(restored_path, str):
        return False
    if os.path.abspath(os.path.normpath(source_path)) == os.path.abspath(os.path.normpath(restored_path)):
        return True
    try:
        source_metadata = Path(source_path).lstat()
        restored_metadata = Path(restored_path).lstat()
    except OSError:
        return False
    return (source_metadata.st_dev, source_metadata.st_ino) == (
        restored_metadata.st_dev,
        restored_metadata.st_ino,
    )


def _verify_credential(entry: Any, identifier: str, failures: list[dict[str, str]]) -> None:
    if not isinstance(entry, dict):
        return
    path_value = entry.get("path")
    expected = entry.get("expected_sha256")
    expected_uid = entry.get("expected_uid")
    expected_mode = entry.get("expected_mode")
    if not _valid_sha256(expected):
        failures.append({"check": "credential_hash", "id": identifier, "reason": "expected_sha256 must be a SHA-256 digest"})
        return
    if not isinstance(expected_uid, int) or expected_mode != "0600":
        failures.append({"check": "credential_permissions", "id": identifier, "reason": "expected_uid and restrictive mode 0600 are required"})
        return
    file_hash = _verify_file_hash(path_value, identifier, "credential", failures)
    if not isinstance(path_value, str) or not path_value.startswith("/"):
        return
    try:
        metadata = Path(path_value).lstat()
    except OSError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        failures.append({"check": "credential_permissions", "id": identifier, "reason": "credential path must be a regular file, not a symlink"})
    if metadata.st_uid != expected_uid or stat.S_IMODE(metadata.st_mode) != 0o600:
        failures.append({"check": "credential_permissions", "id": identifier, "reason": "credential owner or mode is not restrictive"})
    if file_hash is not None and file_hash != expected:
        failures.append({"check": "credential_hash", "id": identifier, "reason": "credential file digest does not match"})
    source_path = entry.get("source_path")
    if source_path is not None:
        source_hash = _verify_file_hash(source_path, identifier, "credential_source", failures)
        if source_hash is not None and source_hash != expected:
            failures.append({"check": "credential_source", "id": identifier, "reason": "live source credential digest does not match the preserved copy"})
        source_uid = entry.get("source_expected_uid")
        source_mode = entry.get("source_expected_mode")
        if not isinstance(source_uid, int) or source_mode != "0600":
            failures.append({"check": "credential_source_permissions", "id": identifier, "reason": "source credential owner and mode must be explicit"})
        else:
            _verify_private_file_metadata(source_path, identifier, "credential_source", source_uid, 0o600, failures)


def _verify_file_hash(value: Any, identifier: str, label: str, failures: list[dict[str, str]]) -> str | None:
    if not isinstance(value, str) or not value.startswith("/"):
        failures.append({"check": label, "id": identifier, "reason": "an absolute file path is required"})
        return None
    path = Path(value)
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("expected a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError) as exc:
        failures.append({"check": label, "id": identifier, "reason": f"file is unavailable or invalid: {exc}"})
        return None


def _verify_private_file_metadata(
    value: Any,
    identifier: str,
    label: str,
    expected_uid: int,
    expected_mode: int,
    failures: list[dict[str, str]],
) -> None:
    if not isinstance(value, str) or not value.startswith("/"):
        return
    try:
        metadata = Path(value).lstat()
    except OSError:
        return
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        failures.append({"check": f"{label}_permissions", "id": identifier, "reason": "database dump must be a regular file"})
    if metadata.st_uid != expected_uid or stat.S_IMODE(metadata.st_mode) != expected_mode:
        failures.append({"check": f"{label}_permissions", "id": identifier, "reason": "database dump owner or mode is not restrictive"})


def _file_sha256(path: Path) -> str:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"expected a regular file for authoritative K3s state: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _check_required_ids(
    key: str,
    actual: set[str],
    requirements: dict[str, list[str]],
    failures: list[dict[str, str]],
) -> None:
    expected = requirements.get(key, [])
    if not isinstance(expected, list):
        failures.append({"check": f"required_{key}", "id": key, "reason": "requirements must be a list"})
        return
    check_name = {
        "retained_paths": "required_retained_path",
        "database_restores": "required_database_restore",
        "credentials": "required_credential",
    }[key]
    for identifier in expected:
        if str(identifier) not in actual:
            failures.append({"check": check_name, "id": str(identifier), "reason": "required recovery item is absent from the manifest"})


def _entry_id(entry: Any, failures: list[dict[str, str]], category: str) -> str | None:
    if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
        failures.append({"check": f"{category}_manifest", "id": category, "reason": "each entry must have a non-empty id"})
        return None
    return entry["id"]


def _verification_result(
    manifest: dict[str, Any],
    checks: dict[str, int],
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "status": "verified" if not failures else "failed",
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "checks": checks,
        "failures": failures,
    }


def _normalize_purge_targets(targets: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(targets, dict) or targets.get("schema_version") != 1 or targets.get("owner") != OWNER:
        raise ValueError("purge target file must declare schema_version 1 and owner gods-mlops")
    roots = _absolute_paths(targets.get("owned_roots"), "owned_roots")
    protected = _absolute_paths(targets.get("protected_paths", []), "protected_paths")
    raw_targets = targets.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("purge requires a non-empty explicit target list")
    normalized_targets = []
    seen: list[str] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            raise ValueError("each purge target must be an object")
        node = item.get("node")
        path = _absolute_path(item.get("path"), "purge target path")
        owner_marker = _absolute_path(item.get("owner_marker"), "purge owner marker")
        root = next((root for root in roots if _within(path, root)), None)
        if root is None:
            raise ValueError(f"purge target is outside every owned data root: {path}")
        if path == root:
            raise ValueError(f"purge targets must be exact children; preserve the root ownership marker: {path}")
        if owner_marker != str(PurePosixPath(root) / OWNER_MARKER):
            raise ValueError(f"purge target owner marker does not identify the root: {path}")
        if any(_overlaps(path, protected_path) for protected_path in protected):
            raise ValueError(f"purge target overlaps protected data: {path}")
        if any(_overlaps(path, existing) for existing in seen):
            raise ValueError(f"purge target list contains duplicate or nested targets: {path}")
        if not isinstance(node, str) or not node:
            raise ValueError("each purge target requires a node name")
        seen.append(path)
        normalized_targets.append({"node": node, "path": path, "owner_marker": owner_marker})
    normalized_targets.sort(key=lambda item: (item["node"], item["path"]))
    return {
        "schema_version": 1,
        "owner": OWNER,
        "owned_roots": roots,
        "protected_paths": protected,
        "targets": normalized_targets,
    }


def _absolute_paths(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or (label == "owned_roots" and not value):
        raise ValueError(f"purge {label} must be a non-empty list" if label == "owned_roots" else "purge protected_paths must be a list")
    paths = [_absolute_path(item, f"{label} entry") for item in value]
    return sorted(set(paths))


def _absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute path")
    if any(character in value for character in "*?[]"):
        raise ValueError(f"{label} may not contain wildcard characters")
    path = PurePosixPath(value)
    if ".." in path.parts:
        raise ValueError(f"{label} may not contain traversal components")
    return str(path)


def _within(path: str, root: str) -> bool:
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return candidate == parent or parent in candidate.parents


def _overlaps(first: str, second: str) -> bool:
    return _within(first, second) or _within(second, first)


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _add_file(digest: Any, relative: str, path: Path) -> None:
    file_digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                file_digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"file cannot be read: {path}") from exc
    _add_record(digest, b"F", relative.encode("utf-8"), str(size).encode("ascii"), file_digest.hexdigest().encode("ascii"))


def _add_record(digest: Any, *parts: bytes) -> None:
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)


def _sanitize_daemonset(value: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(value.get("metadata", {}))
    for key in (
        "uid",
        "resourceVersion",
        "managedFields",
        "creationTimestamp",
        "generation",
        "selfLink",
        "deletionTimestamp",
        "deletionGracePeriodSeconds",
        "ownerReferences",
    ):
        metadata.pop(key, None)
    annotations = dict(metadata.get("annotations") or {})
    annotations.pop("kubectl.kubernetes.io/last-applied-configuration", None)
    if annotations:
        metadata["annotations"] = annotations
    else:
        metadata.pop("annotations", None)
    result = {
        "apiVersion": value.get("apiVersion", "apps/v1"),
        "kind": "DaemonSet",
        "metadata": metadata,
        "spec": value.get("spec", {}),
    }
    return result


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices={"verify-recovery-artifacts", "verify-retained", "hash-tree", "hash-database-dump"})
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--format", choices=set(DATABASE_DUMP_FORMATS))
    args = parser.parse_args(argv)
    if args.operation == "hash-database-dump":
        if args.path is None or args.format is None:
            parser.error("hash-database-dump requires --path and --format")
        print(json.dumps({
            "path": str(args.path),
            "format": args.format,
            "sha256": canonical_database_dump_sha256(args.path, dump_format=args.format),
        }, sort_keys=True))
        return 0
    if args.operation == "hash-tree":
        if args.path is None:
            parser.error("hash-tree requires --path")
        path_info = args.path.lstat()
        print(json.dumps({
            "path": str(args.path),
            "sha256": hash_tree(args.path),
            "uid": path_info.st_uid,
            "gid": path_info.st_gid,
            "mode": f"{stat.S_IMODE(path_info.st_mode):04o}",
        }, sort_keys=True))
        return 0
    if args.manifest is None:
        parser.error("manifest verification requires --manifest")
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "failures": [{"check": "manifest_read", "id": "manifest", "reason": str(exc)}]}))
        return 1
    if not isinstance(manifest, dict):
        print(json.dumps({"status": "failed", "failures": [{"check": "manifest_identity", "id": "manifest", "reason": "expected a JSON object"}]}))
        return 1
    requirements = manifest.get("requirements")
    verifier = verify_recovery_artifacts if args.operation == "verify-recovery-artifacts" else verify_retained
    report = verifier(manifest, requirements=requirements)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
