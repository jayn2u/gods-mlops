"""Inventory normalization and deterministic data-preserving reclaim plans."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


OWNER = "gods-mlops"
OWNER_MARKER = ".gods-mlops-owner.json"
CLUSTER_MARKER = ".gods-mlops-cluster-owner.json"
ALLOWED_SERVICES = {"k3s", "k3s-agent"}
OWNED_DAEMONSETS = [
    {"namespace": "kube-system", "name": "istio-cni-node"},
    {
        "namespace": "nvidia-device-plugin",
        "selector": "app.kubernetes.io/instance=nvidia-device-plugin",
    },
]


class InventoryError(ValueError):
    """Raised when an inventory cannot be used as a safe lifecycle source."""


def load_inventory(
    inventory_path: str | Path,
    *,
    preflight_path: str | Path | None = None,
    storage_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load an Ansible inventory and normalize only its explicit Gods hosts."""
    source = Path(inventory_path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InventoryError(f"cannot read lifecycle inventory: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("all"), dict):
        raise InventoryError("inventory must be an Ansible YAML inventory with an all group")

    all_vars = raw["all"].get("vars", {})
    children = raw["all"].get("children", {})
    if not isinstance(children, dict):
        raise InventoryError("inventory all.children must be a mapping")

    role_groups = (("gods_server", "server"), ("gods_gpu_worker", "worker"))
    nodes: list[dict[str, Any]] = []
    for group_name, role in role_groups:
        group = children.get(group_name, {})
        hosts = group.get("hosts", {}) if isinstance(group, dict) else {}
        if not isinstance(hosts, dict):
            raise InventoryError(f"inventory group {group_name} must define a hosts mapping")
        for name, host_vars in hosts.items():
            if not isinstance(host_vars, dict):
                raise InventoryError(f"host {name} variables must be a mapping")
            data_root = _absolute(host_vars.get("gods_data_root"), f"{name}.gods_data_root")
            recovery_root = _absolute(host_vars.get("gods_recovery_root"), f"{name}.gods_recovery_root")
            data_dir_name = str(host_vars.get("gods_k3s_data_dir_name", all_vars.get("gods_k3s_data_dir_name", "k3s")))
            k3s_data_dir = str(PurePosixPath(data_root) / data_dir_name)
            managed_paths = []
            for item in host_vars.get("gods_data_directories", []):
                if not isinstance(item, dict):
                    raise InventoryError(f"{name}.gods_data_directories entries must be mappings")
                item_path = _absolute(item.get("path"), f"{name}.gods_data_directories.path")
                if not _is_within(item_path, data_root, allow_equal=False):
                    raise InventoryError(f"managed path is outside its Gods data root: {item_path}")
                managed_paths.append({
                    "id": str(item.get("role") or Path(item_path).name),
                    "path": item_path,
                    "role": str(item.get("role") or "managed-data"),
                })
            marker_name = str(host_vars.get("gods_owner_marker_name", all_vars.get("gods_owner_marker_name", OWNER_MARKER)))
            node = {
                "name": str(name),
                "role": role,
                "reachable": None,
                "preflight_status": None,
                "data_root": data_root,
                "recovery_root": recovery_root,
                "data_owner": {
                    "uid": int(host_vars.get("gods_container_uid", all_vars.get("gods_container_uid", 10001))),
                    "gid": int(host_vars.get("gods_container_gid", all_vars.get("gods_container_gid", 10001))),
                },
                "k3s_data_dir": k3s_data_dir,
                "owner_marker": str(PurePosixPath(data_root) / marker_name),
                "cluster_marker": str(PurePosixPath(k3s_data_dir) / CLUSTER_MARKER),
                "service": "k3s" if role == "server" else "k3s-agent",
                "ownership": None,
                "managed_paths": managed_paths,
                "preserve_paths": _string_paths(host_vars.get("gods_preserve_paths", []), f"{name}.gods_preserve_paths"),
                "preserve_patterns": [str(value) for value in host_vars.get("gods_preserve_patterns", [])],
                "preserve_services": [str(value) for value in host_vars.get("gods_preserve_services", [])],
                "completion": {"plan_id": None, "steps": []},
                "recovery_credential_ids": [
                    str(value)
                    for value in host_vars.get(
                        "gods_recovery_credential_ids",
                        all_vars.get("gods_recovery_credential_ids", []),
                    )
                ],
            }
            nodes.append(node)
    if not nodes:
        raise InventoryError("inventory contains no gods_server or gods_gpu_worker hosts")
    names = [node["name"] for node in nodes]
    if len(names) != len(set(names)):
        raise InventoryError("a host may appear in only one Gods lifecycle role")

    observations = _load_json(preflight_path) if preflight_path else None
    _apply_preflight(nodes, observations)
    storage_file = Path(storage_path) if storage_path else source.parent.parent / "kubeflow" / "retained-storage.yaml"
    storage = _load_storage(storage_file, nodes) if storage_file.is_file() else {
        "retained_paths": [],
        "database_restores": [],
        "required_database_restores": [],
    }

    protected_paths = sorted({path for node in nodes for path in node["preserve_paths"]})
    protected_patterns = sorted({pattern for node in nodes for pattern in node["preserve_patterns"]})
    protected_services = sorted({service for node in nodes for service in node["preserve_services"]})
    retained_paths = [
        {"id": f"{node['name']}-{item['id']}", "path": item["path"], "node": node["name"], "role": item["role"]}
        for node in nodes
        for item in node["managed_paths"]
    ]
    retained_paths.extend(storage["retained_paths"])
    storage_path_set = {item["path"] for item in storage["retained_paths"]}
    required_retained_ids = {item["id"] for item in storage["retained_paths"]}
    for node in nodes:
        retained_paths.append({
            "id": f"{node['name']}-k3s-state",
            "path": node["k3s_data_dir"],
            "node": node["name"],
            "role": "k3s-state",
        })
        required_retained_ids.update(
            f"{node['name']}-{item['id']}"
            for item in node["managed_paths"]
            if item["path"] not in storage_path_set
        )
    credential_ids = sorted({item for node in nodes for item in node["recovery_credential_ids"]})
    credential_source_by_id = {}
    for node in nodes:
        for credential_id in node["recovery_credential_ids"]:
            if credential_id == "gods-k3s-token":
                if node["role"] != "server":
                    raise InventoryError("gods-k3s-token backup must be checked on the server node")
                credential_source_by_id[credential_id] = str(PurePosixPath(node["k3s_data_dir"]) / "server" / "token")

    requirements_by_node: dict[str, dict[str, Any]] = {
        node["name"]: {
            "retained_paths": [],
            "database_restores": [],
            "credentials": list(node["recovery_credential_ids"]),
        }
        for node in nodes
    }
    retained_requirements_by_id: dict[str, dict[str, Any]] = {}
    for item in storage["retained_paths"]:
        node = next(value for value in nodes if value["name"] == item["node"])
        expected_owner = {"uid": 0, "gid": 0} if item["role"] == "k3s-state" else node["data_owner"]
        retained_requirements_by_id[item["id"]] = {
            "node": item["node"],
            "source_path": item["path"],
            "source_owner": expected_owner,
            "recovery_root": node["recovery_root"],
            "role": item["role"],
        }
        requirements_by_node[item["node"]]["retained_paths"].append(item["id"])
    for node in nodes:
        for item in node["managed_paths"]:
            if item["path"] in storage_path_set:
                continue
            identifier = f"{node['name']}-{item['id']}"
            if identifier not in retained_requirements_by_id:
                retained_requirements_by_id[identifier] = {
                    "node": node["name"],
                    "source_path": item["path"],
                    "source_owner": node["data_owner"],
                    "recovery_root": node["recovery_root"],
                    "role": item["role"],
                }
                requirements_by_node[node["name"]]["retained_paths"].append(identifier)
    for item in storage["database_restores"]:
        requirements_by_node[item["node"]]["database_restores"].append(item["id"])
    for node_requirements in requirements_by_node.values():
        for key, values in node_requirements.items():
            node_requirements[key] = sorted(set(values))

    return {
        "schema_version": 1,
        "owner": OWNER,
        "nodes": nodes,
        "protected_paths": protected_paths,
        "protected_patterns": protected_patterns,
        "protected_services": protected_services,
        "retained_paths": _unique_records(retained_paths, key="id"),
        "required_retained_paths": sorted(required_retained_ids),
        "required_database_restores": storage["required_database_restores"],
        "required_credentials": credential_ids,
        "credential_source_by_id": credential_source_by_id,
        "requirements_by_node": requirements_by_node,
        "retained_requirements_by_id": retained_requirements_by_id,
        "preflight_status": observations.get("status") if observations else None,
    }


def plan_reclaim(inventory: dict[str, Any]) -> dict[str, Any]:
    """Create a dry-run service-stop plan while treating data as retained."""
    if inventory.get("schema_version") != 1 or inventory.get("owner") != OWNER:
        raise InventoryError("lifecycle inventory must declare schema_version 1 and owner gods-mlops")
    nodes = inventory.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise InventoryError("lifecycle inventory must contain at least one node")

    protected_paths = sorted({_absolute(path, "protected path") for path in inventory.get("protected_paths", [])})
    protected_patterns = [str(pattern) for pattern in inventory.get("protected_patterns", [])]
    protected_services = sorted({str(service) for service in inventory.get("protected_services", [])})
    retained: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    impacts: list[dict[str, Any]] = []
    executable = True

    seen_nodes: set[str] = set()
    for node in sorted(nodes, key=lambda item: str(item.get("name", ""))):
        name = str(node.get("name", ""))
        if not name or name in seen_nodes:
            raise InventoryError("lifecycle node names must be present and unique")
        seen_nodes.add(name)
        data_root = _absolute(node.get("data_root"), f"{name}.data_root")
        k3s_data_dir = _absolute(node.get("k3s_data_dir"), f"{name}.k3s_data_dir")
        service = str(node.get("service", ""))
        if service not in ALLOWED_SERVICES:
            raise InventoryError(f"refusing unowned service target for {name}: {service!r}")
        role = str(node.get("role", ""))
        if (role == "server" and service != "k3s") or (role == "worker" and service != "k3s-agent"):
            raise InventoryError(f"service {service} does not match the declared role for {name}")

        ownership = node.get("ownership")
        if isinstance(ownership, dict) and ownership.get("owner") not in (None, OWNER):
            executable = False
            for item in node.get("managed_paths", []):
                if isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"].startswith("/"):
                    retained.append({
                        "id": str(item.get("id") or f"{name}-{Path(item['path']).name}"),
                        "path": item["path"],
                        "node": name,
                        "role": str(item.get("role") or "managed-data"),
                        "state": "ownership_unverified",
                    })
            retained.append({
                "id": f"{name}-k3s-state",
                "path": k3s_data_dir,
                "node": name,
                "role": "k3s-state",
                "state": "ownership_unverified",
            })
            impacts.append({
                "kind": "ownership_mismatch",
                "node": name,
                "detail": "managed markers do not identify this node as Gods-owned; no service action is planned",
            })
            continue

        paths = node.get("managed_paths", [])
        if not isinstance(paths, list):
            raise InventoryError(f"{name}.managed_paths must be a list")
        node_paths: list[dict[str, Any]] = []
        protected_overlap = False
        for item in paths:
            if not isinstance(item, dict):
                raise InventoryError(f"{name}.managed_paths entries must be mappings")
            path = _absolute(item.get("path"), f"{name}.managed_paths.path")
            if not _is_within(path, data_root, allow_equal=False):
                raise InventoryError(f"managed path is outside {name}'s Gods data root: {path}")
            if _is_protected(path, protected_paths, protected_patterns):
                protected_overlap = True
                impacts.append({
                    "kind": "protected_path_overlap",
                    "node": name,
                    "path": path,
                    "detail": "a proposed managed path overlaps an explicitly preserved path or pattern",
                })
                continue
            record = {
                "id": str(item.get("id") or f"{name}-{Path(path).name}"),
                "path": path,
                "node": name,
                "role": str(item.get("role") or "managed-data"),
            }
            node_paths.append(record)
            retained.append(record)
        if protected_overlap:
            executable = False
            continue

        reachable = node.get("reachable")
        completion = node.get("completion") or {}
        if _is_protected(k3s_data_dir, protected_paths, protected_patterns):
            executable = False
            impacts.append({
                "kind": "protected_path_overlap",
                "node": name,
                "path": k3s_data_dir,
                "detail": "the K3s state directory overlaps an explicitly preserved path or pattern",
            })
            continue

        action = {
            "kind": "drain_node",
            "node": name,
            "data_dir": k3s_data_dir,
            "state": "pending",
            "constraints": [
                "DaemonSet and static mirror pods are accounted for explicitly",
                "PodDisruptionBudgets are honored",
                "emptyDir data is not deleted automatically",
            ],
        }
        if reachable is False:
            action["state"] = "pending_offline"
            impacts.append({
                "kind": "node_unreachable",
                "node": name,
                "detail": "the node remains pending; rerun the same reclaim after connectivity returns",
            })
        actions.append(action)
        service_action = {
            "kind": "stop_service",
            "node": name,
            "service": service,
            "data_dir": k3s_data_dir,
            "state": "pending_offline" if reachable is False else "pending",
        }
        if completion.get("plan_id") == inventory.get("plan_id") and "service_stopped" in completion.get("steps", []):
            service_action["state"] = "completed"
        actions.append(service_action)
        retained.append({
            "id": f"{name}-k3s-state",
            "path": k3s_data_dir,
            "node": name,
            "role": "k3s-state",
        })

    for record in inventory.get("retained_paths", []):
        if not isinstance(record, dict):
            raise InventoryError("retained_paths entries must be mappings")
        path = _absolute(record.get("path"), "retained path")
        retained.append({
            **record,
            "path": path,
            "id": str(record.get("id") or path),
        })
    retained = _unique_records(retained, key="id")
    actions.append({
        "kind": "delete_approved_emptydir_pods",
        "policy": "rendered-disposable-and-injected-istio-runtime-only",
        "state": "pending",
    })
    for target in OWNED_DAEMONSETS:
        actions.append({"kind": "stop_daemonset", **target, "state": "pending"})
    actions.sort(key=lambda item: (
        {"drain_node": 0, "delete_approved_emptydir_pods": 1, "stop_daemonset": 2, "stop_service": 3}.get(item["kind"], 9),
        0 if item.get("service") == "k3s-agent" else 1,
        item.get("node", ""),
        item.get("namespace", ""),
        item.get("name", item.get("selector", "")),
    ))

    # Dynamic probe results and completion records are excluded so a rerun keeps
    # the same confirmation digest after one host has completed or reconnected.
    intent = {
        "schema_version": 1,
        "owner": OWNER,
        "actions": [
            {key: action[key] for key in ("kind", "node", "service", "data_dir", "namespace", "name", "selector", "policy") if key in action}
            for action in actions
        ],
        "retained_paths": [{key: item[key] for key in ("id", "path", "node", "role") if key in item} for item in retained],
        "protected_paths": protected_paths,
        "protected_patterns": sorted(protected_patterns),
        "protected_services": protected_services,
        "required_retained_paths": sorted({str(value) for value in inventory.get("required_retained_paths", [])}),
        "required_database_restores": sorted({str(value) for value in inventory.get("required_database_restores", [])}),
        "required_credentials": sorted({str(value) for value in inventory.get("required_credentials", [])}),
    }
    plan_id = hashlib.sha256(_canonical_json(intent)).hexdigest()
    for action in actions:
        if "node" not in action:
            continue
        node = next(item for item in nodes if item["name"] == action["node"])
        completion = node.get("completion") or {}
        if completion.get("plan_id") == plan_id and "service_stopped" in completion.get("steps", []):
            action["state"] = "completed"
    impacts.extend([
        {"kind": "platform_unavailable", "detail": "stopping the Gods K3s services interrupts Kubeflow, training, and Kubernetes workloads until reapply"},
        {"kind": "workload_quiescence", "detail": "drain schedulable workloads, stop only known platform DaemonSets, and require an empty K3s-owned runtime before recording completion"},
        {"kind": "drain_constraint", "detail": "PodDisruptionBudgets are honored; known disposable emptyDir volumes are handled by an explicit policy and unknown ones block reclaim"},
        {"kind": "data_retained", "detail": "managed local data, K3s state, credentials, and recovery files remain in place"},
        {"kind": "purge_separate", "detail": "permanent data removal is a separate command with an exact target digest"},
    ])
    return {
        "schema_version": 1,
        "owner": OWNER,
        "operation": "reclaim",
        "dry_run": True,
        "plan_valid": executable,
        "plan_id": plan_id,
        "actions": actions,
        "owned_daemonsets": OWNED_DAEMONSETS,
        "retained_paths": retained,
        "preserved_paths": protected_paths,
        "preserved_patterns": sorted(protected_patterns),
        "preserved_services": protected_services,
        "requirements": {
            "retained_paths": intent["required_retained_paths"],
            "database_restores": intent["required_database_restores"],
            "credentials": intent["required_credentials"],
        },
        "execution_preconditions": [
            "verify real backup hashes, database restore dumps, and credentials on their owning nodes",
            "verify source ownership before stop and cold retained-data hashes after quiescence",
            "authenticate Ansible become and prove effective uid 0 on each target node",
            "verify owner markers and the service ExecStart data directory on every target node",
        ],
        "impacts": sorted(impacts, key=lambda item: (item.get("node", ""), item["kind"], item.get("path", ""))),
    }


def validate_retained_manifest(manifest: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    """Validate exact per-node recovery coverage without reading any data paths."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("owner") != OWNER:
        raise InventoryError("recovery manifest must declare schema_version 1 and owner gods-mlops")
    if inventory.get("schema_version") != 1 or inventory.get("owner") != OWNER:
        raise InventoryError("lifecycle inventory must declare schema_version 1 and owner gods-mlops")
    expected_plan_id = plan_reclaim(inventory)["plan_id"]
    if manifest.get("plan_id") != expected_plan_id:
        raise InventoryError(f"recovery manifest plan_id must match the current reclaim plan {expected_plan_id}")
    requirements_by_node = inventory.get("requirements_by_node")
    retained_expected = inventory.get("retained_requirements_by_id")
    if not isinstance(requirements_by_node, dict) or not isinstance(retained_expected, dict):
        raise InventoryError("inventory is missing per-node recovery requirements")

    node_roots = {node["name"]: node["recovery_root"] for node in inventory["nodes"]}
    node_sources = {identifier: record["node"] for identifier, record in retained_expected.items()}
    expected_database_node = {
        identifier: node_name
        for node_name, node_requirements in requirements_by_node.items()
        for identifier in node_requirements["database_restores"]
    }
    expected_credential_node = {
        identifier: node_name
        for node_name, node_requirements in requirements_by_node.items()
        for identifier in node_requirements["credentials"]
    }

    retained_entries = _entries_by_id(manifest.get("retained_paths"), "retained_paths")
    database_entries = _entries_by_id(manifest.get("database_restores"), "database_restores")
    credential_entries = _entries_by_id(manifest.get("credentials"), "credentials")
    expected_retained = set(retained_expected)
    expected_databases = set(expected_database_node)
    expected_credentials = set(expected_credential_node)
    _require_exact_ids("retained_paths", set(retained_entries), expected_retained)
    _require_exact_ids("database_restores", set(database_entries), expected_databases)
    _require_exact_ids("credentials", set(credential_entries), expected_credentials)

    for identifier, entry in retained_entries.items():
        node_name = node_sources[identifier]
        expected = retained_expected[identifier]
        if entry.get("node") != node_name:
            raise InventoryError(f"retained path {identifier} is assigned to the wrong node")
        if _absolute(entry.get("source_path"), f"{identifier}.source_path") != expected["source_path"]:
            raise InventoryError(f"retained path {identifier} does not name its configured source path")
        if entry.get("source_owner") != expected["source_owner"]:
            raise InventoryError(f"retained path {identifier} has unexpected source ownership")
        if entry.get("role") != expected["role"]:
            raise InventoryError(f"retained path {identifier} has the wrong recovery role")
        if expected["role"] == "database":
            if any(key in entry for key in ("backup_path", "backup_owner", "backup_mode", "expected_tree_sha256")):
                raise InventoryError(f"database path {identifier} uses logical restore evidence rather than a raw active-tree backup")
        else:
            if entry.get("backup_owner") != {"uid": 0, "gid": 0} or entry.get("backup_mode") != "0700":
                raise InventoryError(f"retained path {identifier} backup must be root-owned mode 0700")
            backup_path = _absolute(entry.get("backup_path"), f"{identifier}.backup_path")
            root = node_roots[node_name]
            if not _is_within(backup_path, root, allow_equal=False):
                raise InventoryError(f"retained path {identifier} backup is outside {node_name}'s recovery root")
            if _paths_overlap(backup_path, expected["source_path"]):
                raise InventoryError(f"retained path {identifier} backup overlaps its live source")
            if not _valid_sha256(entry.get("expected_tree_sha256")):
                raise InventoryError(f"retained path {identifier} requires a real SHA-256 tree digest")

    for identifier, entry in database_entries.items():
        node_name = expected_database_node[identifier]
        if entry.get("node") != node_name:
            raise InventoryError(f"database restore {identifier} is assigned to the wrong node")
        if not _valid_sha256(entry.get("expected_sha256")):
            raise InventoryError(f"database restore {identifier} requires a real SHA-256 digest")
        if entry.get("expected_uid") != 0 or entry.get("expected_mode") != "0600":
            raise InventoryError(f"database restore {identifier} dumps must be root-owned mode 0600")
        for key in ("backup_dump_path", "restored_dump_path"):
            path = _absolute(entry.get(key), f"{identifier}.{key}")
            if not _is_within(path, node_roots[node_name], allow_equal=False):
                raise InventoryError(f"database restore {identifier} files must remain on {node_name}")

    for identifier, entry in credential_entries.items():
        node_name = expected_credential_node[identifier]
        if entry.get("node") != node_name:
            raise InventoryError(f"credential {identifier} is assigned to the wrong node")
        if not _valid_sha256(entry.get("expected_sha256")):
            raise InventoryError(f"credential {identifier} requires a real SHA-256 digest")
        if entry.get("expected_uid") != 0 or entry.get("expected_mode") != "0600":
            raise InventoryError(f"credential {identifier} must be root-owned mode 0600")
        path = _absolute(entry.get("path"), f"{identifier}.path")
        if not _is_within(path, node_roots[node_name], allow_equal=False):
            raise InventoryError(f"credential {identifier} must be stored on {node_name} under its recovery root")
        expected_source = inventory.get("credential_source_by_id", {}).get(identifier)
        if expected_source:
            if _absolute(entry.get("source_path"), f"{identifier}.source_path") != expected_source:
                raise InventoryError(f"credential {identifier} source does not match the K3s server token path")
            if entry.get("source_expected_uid") != 0 or entry.get("source_expected_mode") != "0600":
                raise InventoryError(f"credential {identifier} source must be root-owned mode 0600")
        elif "source_path" in entry:
            raise InventoryError(f"credential {identifier} has an unrecognized live source path")

    return {
        "status": "manifest_valid",
        "plan_id": expected_plan_id,
        "requirements_by_node": requirements_by_node,
        "counts": {
            "retained_paths": len(retained_entries),
            "database_restores": len(database_entries),
            "credentials": len(credential_entries),
        },
    }


def slice_retained_manifest_for_node(
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    node_name: str,
) -> dict[str, Any]:
    coverage = validate_retained_manifest(manifest, inventory)
    if node_name not in coverage["requirements_by_node"]:
        raise InventoryError(f"node is not in the Gods lifecycle inventory: {node_name}")
    return {
        "schema_version": 1,
        "owner": OWNER,
        "plan_id": manifest["plan_id"],
        "node": node_name,
        "node_role": next(node["role"] for node in inventory["nodes"] if node["name"] == node_name),
        "k3s_data_dir": next(node["k3s_data_dir"] for node in inventory["nodes"] if node["name"] == node_name),
        "k3s_expected_uid": 0,
        "requirements": coverage["requirements_by_node"][node_name],
        "retained_paths": [entry for entry in manifest["retained_paths"] if entry["node"] == node_name],
        "database_restores": [entry for entry in manifest["database_restores"] if entry["node"] == node_name],
        "credentials": [entry for entry in manifest["credentials"] if entry["node"] == node_name],
    }

def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"cannot read read-only preflight report: {exc}") from exc
    if not isinstance(value, dict):
        raise InventoryError("preflight report must contain a JSON object")
    return value


def _apply_preflight(nodes: list[dict[str, Any]], report: dict[str, Any] | None) -> None:
    if report is None:
        return
    observed = report.get("nodes", [])
    if not isinstance(observed, list):
        raise InventoryError("preflight report nodes must be a list")
    by_name = {str(item.get("node")): item for item in observed if isinstance(item, dict)}
    for node in nodes:
        item = by_name.get(node["name"])
        if item is None:
            node["reachable"] = False
            node["preflight_status"] = "missing"
            continue
        status = str(item.get("status", "unknown"))
        node["preflight_status"] = status
        node["reachable"] = status not in {"unreachable", "failed", "missing"}


def _load_storage(path: Path, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        docs = [item for item in yaml.safe_load_all(path.read_text(encoding="utf-8")) if item]
    except (OSError, yaml.YAMLError) as exc:
        raise InventoryError(f"cannot read retained storage manifest: {exc}") from exc
    volumes = [item for item in docs if item.get("kind") == "PersistentVolume"]
    claims = [item for item in docs if item.get("kind") == "PersistentVolumeClaim"]
    claims_by_volume = {
        item.get("spec", {}).get("volumeName"): item
        for item in claims
        if item.get("spec", {}).get("volumeName")
    }
    required_paths: list[dict[str, Any]] = []
    database_restores: list[dict[str, str]] = []
    roots = [node["data_root"] for node in nodes]
    for volume in volumes:
        volume_name = volume.get("metadata", {}).get("name")
        claim = claims_by_volume.get(volume_name)
        claim_ref = volume.get("spec", {}).get("claimRef", {})
        name = claim.get("metadata", {}).get("name") if claim else claim_ref.get("name")
        namespace = claim.get("metadata", {}).get("namespace") if claim else claim_ref.get("namespace")
        if not name or not namespace:
            raise InventoryError(f"retained volume {volume_name} has no explicit namespace/name claim mapping")
        if volume.get("spec", {}).get("persistentVolumeReclaimPolicy") != "Retain":
            raise InventoryError(f"volume {volume_name} does not have Retain policy")
        if claim and (claim_ref.get("name") != name or claim_ref.get("namespace") != namespace):
            raise InventoryError(f"claim {namespace}/{name} and PV {volume_name} claimRef disagree")
        local_path = _absolute(volume.get("spec", {}).get("local", {}).get("path"), f"PV/{volume_name}.local.path")
        if not any(_is_within(local_path, root, allow_equal=False) for root in roots):
            raise InventoryError(f"retained volume path is outside the new Gods data roots: {local_path}")
        pv_id = str(volume_name)
        required_paths.append({
            "id": pv_id,
            "path": local_path,
            "node": next(node["name"] for node in nodes if _is_within(local_path, node["data_root"], allow_equal=False)),
            "role": str(volume.get("metadata", {}).get("labels", {}).get("gods.io/recovery-kind", "persistent-volume")),
            "claim": f"{namespace}/{name}",
        })
        recovery_kind = volume.get("metadata", {}).get("labels", {}).get("gods.io/recovery-kind")
        if recovery_kind == "database":
            database_restores.append({"id": pv_id, "node": required_paths[-1]["node"]})
    return {
        "retained_paths": required_paths,
        "database_restores": sorted(database_restores, key=lambda item: item["id"]),
        "required_database_restores": sorted(item["id"] for item in database_restores),
    }


def _unique_records(records: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        identifier = str(record[key])
        prior = result.get(identifier)
        if prior is not None and prior != record:
            raise InventoryError(f"duplicate lifecycle id has conflicting records: {identifier}")
        result[identifier] = record
    return [result[item] for item in sorted(result)]


def _absolute(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise InventoryError(f"{label} must be an absolute path")
    path = PurePosixPath(value)
    if ".." in path.parts or any(part in {"", "."} for part in path.parts[1:]):
        raise InventoryError(f"{label} must be normalized and may not contain traversal components")
    return str(path)


def _string_paths(values: Any, label: str) -> list[str]:
    if not isinstance(values, list):
        raise InventoryError(f"{label} must be a list")
    return [_absolute(value, label) for value in values]


def _entries_by_id(values: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise InventoryError(f"recovery manifest {label} must be a list")
    result: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise InventoryError(f"every recovery manifest {label} entry requires an id")
        identifier = item["id"]
        if identifier in result:
            raise InventoryError(f"duplicate recovery manifest id in {label}: {identifier}")
        result[identifier] = item
    return result


def _require_exact_ids(label: str, actual: set[str], expected: set[str]) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise InventoryError(f"{label} ID coverage mismatch; missing={missing}, unexpected={unexpected}")


def _is_within(path: str, root: str, *, allow_equal: bool) -> bool:
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    return (allow_equal and candidate == parent) or parent in candidate.parents


def _paths_overlap(first: str, second: str) -> bool:
    first_path = PurePosixPath(first)
    second_path = PurePosixPath(second)
    return first_path == second_path or first_path in second_path.parents or second_path in first_path.parents


def _is_protected(path: str, protected_paths: list[str], patterns: list[str]) -> bool:
    candidate = PurePosixPath(path)
    if any(candidate == PurePosixPath(item) or candidate in PurePosixPath(item).parents or PurePosixPath(item) in candidate.parents for item in protected_paths):
        return True
    candidates = [str(candidate), *(str(parent) for parent in candidate.parents)]
    return any(fnmatch.fnmatchcase(value, pattern) for value in candidates for pattern in patterns)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
