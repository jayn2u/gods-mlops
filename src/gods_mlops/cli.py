"""Command-line tools for validating locks and preparing model caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import stat

from gods_mlops.infra_locks import load_image_lock
from gods_mlops.model_locks import (
    LockValidationError,
    load_model_lock,
    model_cache_path,
    prepare_model,
    validate_all_model_caches,
    validate_model_cache,
)
from gods_mlops.lifecycle.inventory import (
    InventoryError,
    load_inventory,
    plan_reclaim,
    slice_retained_manifest_for_node,
    validate_retained_manifest,
)
from gods_mlops.lifecycle.recovery import (
    active_workload_pods,
    build_daemonset_snapshot,
    hash_tree,
    owned_daemonsets,
    purge_targets_digest,
    validate_purge_targets,
    validate_daemonset_snapshot,
    validate_emptydir_policy,
    verify_restored_daemonsets,
)


APP_ROOT = Path(os.environ.get("GODS_MLOPS_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_MODEL_LOCK = APP_ROOT / "models" / "lock.json"
DEFAULT_IMAGE_LOCK = APP_ROOT / "infra" / "versions.lock.yaml"
DEFAULT_CACHE_ROOT = Path("/mnt/model-cache")
DEFAULT_ANSIBLE_INVENTORY = APP_ROOT / "infra" / "ansible" / "inventory.example.yml"
DEFAULT_TEARDOWN_PLAYBOOK = APP_ROOT / "infra" / "ansible" / "teardown.yml"
DEFAULT_PURGE_PLAYBOOK = APP_ROOT / "infra" / "ansible" / "purge.yml"
DEFAULT_PRIVILEGE_PROBE = APP_ROOT / "infra" / "ansible" / "privilege-probe.yml"
DEFAULT_RECONNECT_PLAYBOOK = APP_ROOT / "infra" / "ansible" / "reconnect.yml"
DEFAULT_VERIFY_RETAINED_PLAYBOOK = APP_ROOT / "infra" / "ansible" / "verify-retained.yml"
DEFAULT_HASH_PATH_PLAYBOOK = APP_ROOT / "infra" / "ansible" / "hash-path.yml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gods-mlops")
    commands = parser.add_subparsers(dest="command", required=True)

    check_locks = commands.add_parser("check-locks", help="validate model and OCI image lock files")
    check_locks.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    check_locks.add_argument("--image-lock", type=Path, default=DEFAULT_IMAGE_LOCK)

    prepare = commands.add_parser("prepare-models", help="download and verify locked model files")
    prepare.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    prepare.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    prepare.add_argument("--model-id", help="prepare only one locked model")

    check_models = commands.add_parser("check-models", help="verify every required local model file")
    check_models.add_argument("--model-lock", type=Path, default=DEFAULT_MODEL_LOCK)
    check_models.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    check_models.add_argument("--model-id", help="verify only one locked model")

    check_training = commands.add_parser(
        "check-training-image", help="import the pinned training and evaluation dependencies"
    )
    check_training.add_argument("--require-cuda", action="store_true")

    commands.add_parser("operator-ui", help="start the loopback-only single-operator web interface")

    lifecycle = commands.add_parser("lifecycle", help="plan and verify the Gods data-preserving lifecycle")
    lifecycle_commands = lifecycle.add_subparsers(dest="lifecycle_command", required=True)
    for command_name, help_text in (
        ("plan-reclaim", "show an exact dry-run service-stop and preservation plan"),
        ("verify-retained", "verify recovery copies, database restore artifacts, and credentials"),
    ):
        lifecycle_command = lifecycle_commands.add_parser(command_name, help=help_text)
        lifecycle_command.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
        lifecycle_command.add_argument("--preflight", type=Path)
        lifecycle_command.add_argument("--storage", type=Path)
    lifecycle_commands.choices["plan-reclaim"].add_argument("--output", type=Path)
    lifecycle_commands.choices["verify-retained"].add_argument("--manifest", type=Path, required=True)
    lifecycle_commands.choices["verify-retained"].add_argument("--confirm-plan", required=True)
    lifecycle_commands.choices["verify-retained"].add_argument("--ask-become-pass", action="store_true", help="let Ansible prompt for normal sudo authentication")

    reclaim = lifecycle_commands.add_parser("reclaim", help="verify retention then stop only marked Gods K3s services")
    reclaim.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    reclaim.add_argument("--preflight", type=Path)
    reclaim.add_argument("--storage", type=Path)
    reclaim.add_argument("--manifest", type=Path, required=True)
    reclaim.add_argument("--confirm-plan", required=True, help="exact plan_id printed by plan-reclaim")
    reclaim.add_argument("--ask-become-pass", action="store_true", help="let Ansible prompt for normal sudo authentication")

    reconnect = lifecycle_commands.add_parser("reconnect", help="verify retained state before reapplying the existing Gods K3s roots")
    reconnect.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    reconnect.add_argument("--preflight", type=Path)
    reconnect.add_argument("--storage", type=Path)
    reconnect.add_argument("--manifest", type=Path, required=True)
    reconnect.add_argument("--daemonset-snapshot", type=Path, required=True)
    reconnect.add_argument("--confirm-plan", required=True, help="exact plan_id bound to the saved DaemonSet snapshot")
    reconnect.add_argument("--confirm-snapshot-sha256", required=True, help="exact digest printed by the captured snapshot")
    reconnect.add_argument("--ask-become-pass", action="store_true", help="let Ansible prompt for normal sudo authentication")

    purge = lifecycle_commands.add_parser("purge", help="permanently delete only a separately confirmed target list")
    purge.add_argument("--targets", type=Path, required=True)
    purge.add_argument("--confirm-targets", help="exact digest printed when targets are reviewed")
    purge.add_argument("--confirm-plan", help="exact reclaim plan_id printed by lifecycle plan-reclaim")
    purge.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    purge.add_argument("--ask-become-pass", action="store_true", help="let Ansible prompt for normal sudo authentication")

    verify_purge = lifecycle_commands.add_parser(
        "verify-purge-targets", help=argparse.SUPPRESS
    )
    verify_purge.add_argument("--targets", type=Path, required=True)
    verify_purge.add_argument("--confirm-targets", required=True)
    verify_purge.add_argument("--confirm-plan", required=True)
    verify_purge.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)

    check_manifest = lifecycle_commands.add_parser("check-retained-manifest", help=argparse.SUPPRESS)
    check_manifest.add_argument("--manifest", type=Path, required=True)
    check_manifest.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    check_manifest.add_argument("--preflight", type=Path)
    check_manifest.add_argument("--storage", type=Path)

    export_node_manifest = lifecycle_commands.add_parser("export-retained-node", help=argparse.SUPPRESS)
    export_node_manifest.add_argument("--manifest", type=Path, required=True)
    export_node_manifest.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    export_node_manifest.add_argument("--preflight", type=Path)
    export_node_manifest.add_argument("--storage", type=Path)
    export_node_manifest.add_argument("--node", required=True)

    capture_daemonsets = lifecycle_commands.add_parser("capture-daemonsets", help=argparse.SUPPRESS)
    capture_daemonsets.add_argument("--kubeconfig", type=Path, required=True)
    capture_daemonsets.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    capture_daemonsets.add_argument("--plan-id", required=True)
    capture_daemonsets.add_argument("--output", type=Path)

    verify_ds_snapshot = lifecycle_commands.add_parser("verify-daemonset-snapshot", help=argparse.SUPPRESS)
    verify_ds_snapshot.add_argument("--snapshot", type=Path, required=True)
    verify_ds_snapshot.add_argument("--confirm-plan", required=True)
    verify_ds_snapshot.add_argument("--confirm-snapshot-sha256", required=True)
    verify_ds_snapshot.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)

    restore_ds_snapshot = lifecycle_commands.add_parser("restore-daemonset-snapshot", help=argparse.SUPPRESS)
    restore_ds_snapshot.add_argument("--snapshot", type=Path, required=True)
    restore_ds_snapshot.add_argument("--confirm-plan", required=True)
    restore_ds_snapshot.add_argument("--confirm-snapshot-sha256", required=True)
    restore_ds_snapshot.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    restore_ds_snapshot.add_argument("--kubeconfig", type=Path, required=True)

    hash_tree_command = lifecycle_commands.add_parser("hash-tree", help="print a deterministic SHA-256 for a regular path tree")
    hash_tree_command.add_argument("--path", type=Path, required=True)
    hash_tree_command.add_argument("--node", help="hash a path on its owning inventory node")
    hash_tree_command.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    hash_tree_command.add_argument("--ask-become-pass", action="store_true", help="let Ansible prompt for normal sudo authentication")

    hash_dump_command = lifecycle_commands.add_parser("hash-database-dump", help="hash a canonical PostgreSQL or MySQL logical dump on its owning node")
    hash_dump_command.add_argument("--path", type=Path, required=True)
    hash_dump_command.add_argument("--format", choices={"postgresql-sql-v1", "mysql-sql-v1"}, required=True)
    hash_dump_command.add_argument("--node", required=True)
    hash_dump_command.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    hash_dump_command.add_argument("--ask-become-pass", action="store_true", help="let Ansible prompt for normal sudo authentication")

    for command_name in ("verify-daemonset-scope", "verify-workloads-stopped", "verify-emptydir-policy"):
        verify_kubernetes = lifecycle_commands.add_parser(command_name, help=argparse.SUPPRESS)
        verify_kubernetes.add_argument("--kubeconfig", type=Path, required=True)
        if command_name == "verify-workloads-stopped":
            verify_kubernetes.add_argument("--ignore-daemonsets", action="store_true")

    privilege_probe = lifecycle_commands.add_parser(
        "verify-privilege", help="run an authenticated, read-only become probe on each configured node"
    )
    privilege_probe.add_argument("--inventory", type=Path, default=DEFAULT_ANSIBLE_INVENTORY)
    privilege_probe.add_argument("--ask-become-pass", action="store_true", help="let Ansible prompt for normal sudo authentication")

    args = parser.parse_args(argv)
    if args.command == "operator-ui":
        from gods_mlops.web.app import main as operator_ui_main

        try:
            operator_ui_main()
        except (ValueError, OSError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        return 0
    try:
        if args.command == "check-locks":
            models = load_model_lock(args.model_lock)
            images = load_image_lock(args.image_lock)
            print(f"lock structure valid: {len(models.models)} model revisions, {len(images.images)} images")
            print("model cache readiness: not checked")
            return 0

        if args.command == "prepare-models":
            model_set = load_model_lock(args.model_lock)
            selected = _select_models(model_set.models, args.model_id)
            for model in selected:
                path = prepare_model(model, args.cache_root)
                print(f"prepared and verified: {model.model_id}@{model.revision} -> {path}")
            return 0

        if args.command == "check-models":
            model_set = load_model_lock(args.model_lock)
            selected = _select_models(model_set.models, args.model_id)
            if args.model_id is None:
                validate_all_model_caches(model_set, args.cache_root)
            else:
                for model in selected:
                    validate_model_cache(model, model_cache_path(args.cache_root, model))
            print(f"model files ready: {len(selected)} locked model revisions")
            return 0

        if args.command == "check-training-image":
            return _check_training_image(require_cuda=args.require_cuda)
        if args.command == "lifecycle":
            return _lifecycle_command(args)
    except LockValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (InventoryError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2


def training_preflight_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gods-mlops-training-preflight")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    return _check_training_image(require_cuda=args.require_cuda)


def _select_models(models: tuple, requested_id: str | None) -> tuple:
    if requested_id is None:
        return models
    selected = tuple(model for model in models if model.model_id == requested_id)
    if not selected:
        raise LockValidationError(f"model is not present in the immutable lock: {requested_id}")
    return selected


def _check_training_image(*, require_cuda: bool) -> int:
    import numpy
    import scipy
    import torch
    import torchvision
    from PIL import __version__ as pillow_version
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from transformers import (
        CLIPModel,
        CLIPProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLProcessor,
        RTDetrImageProcessor,
        RTDetrV2ForObjectDetection,
    )
    from transformers import RTDetrV2Config
    from transformers.loss.loss_rt_detr import RTDetrHungarianMatcher

    # Transformers 4.57.1 imports the RT-DETR Hungarian matcher lazily; importing
    # its model class alone does not prove that scipy is present for optimization.
    matcher = RTDetrHungarianMatcher(RTDetrV2Config(num_labels=2))
    matches = matcher(
        outputs={
            "logits": torch.tensor([[[0.1, 0.9], [0.8, 0.2]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]]),
        },
        targets=[
            {"class_labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}
        ],
    )
    if len(matches) != 1 or len(matches[0][0]) != 1:
        raise RuntimeError("RT-DETR Hungarian assignment smoke did not match its synthetic target")

    cuda_available = torch.cuda.is_available()
    report = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": cuda_available,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pillow": pillow_version,
        "evaluation": [COCO.__name__, COCOeval.__name__],
        "model_classes": [
            RTDetrV2ForObjectDetection.__name__,
            RTDetrImageProcessor.__name__,
            RTDetrHungarianMatcher.__name__,
            CLIPModel.__name__,
            CLIPProcessor.__name__,
            Qwen2_5_VLForConditionalGeneration.__name__,
            Qwen2_5_VLProcessor.__name__,
        ],
    }
    print(json.dumps(report, sort_keys=True))
    if require_cuda and not cuda_available:
        print("ERROR: CUDA is required for this training preflight", file=sys.stderr)
        return 1
    return 0


def _lifecycle_command(args: argparse.Namespace) -> int:
    command = args.lifecycle_command
    if command in {
        "plan-reclaim",
        "verify-retained",
        "reclaim",
        "reconnect",
        "purge",
        "verify-purge-targets",
        "check-retained-manifest",
        "export-retained-node",
        "capture-daemonsets",
        "verify-daemonset-snapshot",
        "restore-daemonset-snapshot",
    }:
        inventory = load_inventory(
            args.inventory,
            preflight_path=getattr(args, "preflight", None),
            storage_path=getattr(args, "storage", None),
        )
        plan = plan_reclaim(inventory)
    else:
        inventory = None
        plan = None

    if command == "plan-reclaim":
        return _write_json(plan, getattr(args, "output", None))

    if command == "verify-retained":
        if args.confirm_plan != plan["plan_id"]:
            raise ValueError(f"retained-state confirmation does not match current plan_id {plan['plan_id']}")
        validate_retained_manifest(_read_json(args.manifest), inventory)
        command_args = [
            "ansible-playbook",
            "--inventory",
            str(args.inventory),
            str(DEFAULT_VERIFY_RETAINED_PLAYBOOK),
            "--extra-vars",
            f"gods_recovery_manifest={args.manifest.resolve()}",
            "--extra-vars",
            f"gods_lifecycle_inventory_file={args.inventory.resolve()}",
            "--extra-vars",
            f"gods_reclaim_plan_id={plan['plan_id']}",
        ]
        if args.ask_become_pass:
            command_args.append("--ask-become-pass")
        return _run_ansible(command_args)

    if command == "check-retained-manifest":
        report = validate_retained_manifest(_read_json(args.manifest), inventory)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if command == "export-retained-node":
        manifest = _read_json(args.manifest)
        node_manifest = slice_retained_manifest_for_node(manifest, inventory, args.node)
        print(json.dumps(node_manifest, sort_keys=True))
        return 0

    if command == "capture-daemonsets":
        if args.plan_id != plan["plan_id"]:
            raise ValueError(f"DaemonSet snapshot plan id does not match current plan {plan['plan_id']}")
        output_path = args.output or (
            APP_ROOT / "infra" / "ansible" / "artifacts" / f"reclaim-{plan['plan_id']}-daemonsets.json"
        )
        if output_path.exists():
            snapshot = _read_private_json(output_path)
            capture_status = "reused"
            summary = validate_daemonset_snapshot(
                snapshot,
                confirmation=snapshot.get("snapshot_sha256", ""),
                expected_plan_id=plan["plan_id"],
            )
        else:
            snapshot = build_daemonset_snapshot(
                _kubectl_json(args.kubeconfig, "get", "daemonsets", "--all-namespaces"),
                plan_id=plan["plan_id"],
            )
            output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            output_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            output_path.chmod(0o600)
            capture_status = "captured"
            summary = validate_daemonset_snapshot(
                snapshot,
                confirmation=snapshot["snapshot_sha256"],
                expected_plan_id=plan["plan_id"],
            )
        print(json.dumps({
            "status": capture_status,
            "plan_id": plan["plan_id"],
            "path": str(output_path.resolve()),
            "snapshot_sha256": summary["snapshot_sha256"],
            "objects": summary["objects"],
        }, sort_keys=True))
        return 0

    if command == "verify-daemonset-snapshot":
        if args.confirm_plan != plan["plan_id"]:
            raise ValueError(f"snapshot plan confirmation does not match current plan {plan['plan_id']}")
        summary = validate_daemonset_snapshot(
            _read_private_json(args.snapshot),
            confirmation=args.confirm_snapshot_sha256,
            expected_plan_id=plan["plan_id"],
        )
        print(json.dumps({key: value for key, value in summary.items() if key != "apply_objects"}, sort_keys=True))
        return 0

    if command == "restore-daemonset-snapshot":
        if args.confirm_plan != plan["plan_id"]:
            raise ValueError(f"snapshot plan confirmation does not match current plan {plan['plan_id']}")
        summary = validate_daemonset_snapshot(
            _read_private_json(args.snapshot),
            confirmation=args.confirm_snapshot_sha256,
            expected_plan_id=plan["plan_id"],
        )
        payload = json.dumps({"apiVersion": "v1", "kind": "List", "items": summary["apply_objects"]})
        _run_kubectl(args.kubeconfig, ["apply", "--filename", "-"], stdin=payload)
        for item in summary["objects"]:
            _run_kubectl(
                args.kubeconfig,
                ["rollout", "status", f"daemonset/{item['name']}", "--namespace", item["namespace"], "--timeout=5m"],
            )
        live = _kubectl_json(args.kubeconfig, "get", "daemonsets", "--all-namespaces")
        verified = verify_restored_daemonsets(summary["apply_objects"], live)
        print(json.dumps({
            "status": "restored",
            "plan_id": plan["plan_id"],
            "snapshot_sha256": summary["snapshot_sha256"],
            "objects": verified["objects"],
        }, sort_keys=True))
        return 0

    if command == "reclaim":
        if not plan["plan_valid"]:
            raise ValueError("reclaim plan contains an ownership or protected-path blocker")
        if args.confirm_plan != plan["plan_id"]:
            raise ValueError(f"reclaim confirmation does not match the current plan_id {plan['plan_id']}")
        validate_retained_manifest(_read_json(args.manifest), inventory)
        command_args = [
            "ansible-playbook",
            "--inventory",
            str(args.inventory),
            str(DEFAULT_TEARDOWN_PLAYBOOK),
            "--extra-vars",
            f"gods_recovery_manifest={args.manifest.resolve()}",
            "--extra-vars",
            f"gods_lifecycle_inventory_file={args.inventory.resolve()}",
            "--extra-vars",
            f"gods_reclaim_plan_id={plan['plan_id']}",
        ]
        if args.ask_become_pass:
            command_args.append("--ask-become-pass")
        return _run_ansible(command_args)

    if command == "reconnect":
        validate_retained_manifest(_read_json(args.manifest), inventory)
        if args.confirm_plan != plan["plan_id"]:
            raise ValueError(f"reconnect plan confirmation does not match current plan {plan['plan_id']}")
        snapshot_summary = validate_daemonset_snapshot(
            _read_private_json(args.daemonset_snapshot),
            confirmation=args.confirm_snapshot_sha256,
            expected_plan_id=plan["plan_id"],
        )
        command_args = [
            "ansible-playbook",
            "--inventory",
            str(args.inventory),
            str(DEFAULT_RECONNECT_PLAYBOOK),
            "--extra-vars",
            f"gods_recovery_manifest={args.manifest.resolve()}",
            "--extra-vars",
            f"gods_lifecycle_inventory_file={args.inventory.resolve()}",
            "--extra-vars",
            f"gods_daemonset_snapshot={args.daemonset_snapshot.resolve()}",
            "--extra-vars",
            f"gods_daemonset_snapshot_sha256={snapshot_summary['snapshot_sha256']}",
            "--extra-vars",
            f"gods_reclaim_plan_id={plan['plan_id']}",
        ]
        if args.ask_become_pass:
            command_args.append("--ask-become-pass")
        return _run_ansible(command_args)

    if command == "purge":
        target_document = _read_json(args.targets)
        digest = purge_targets_digest(target_document)
        if not args.confirm_targets:
            print(json.dumps({
                "operation": "purge",
                "dry_run": True,
                "targets": target_document.get("targets", []),
                "target_sha256": digest,
                "reclaim_plan_id": plan["plan_id"],
                "confirmation_required": f"--confirm-targets {digest} --confirm-plan {plan['plan_id']}",
            }, indent=2, sort_keys=True))
            return 2
        normalized = validate_purge_targets(target_document, confirmation=args.confirm_targets)
        if args.confirm_plan != plan["plan_id"]:
            raise ValueError(f"purge reclaim plan confirmation does not match current plan {plan['plan_id']}")
        command_args = [
            "ansible-playbook",
            "--inventory",
            str(args.inventory),
            str(DEFAULT_PURGE_PLAYBOOK),
            "--extra-vars",
            f"@{args.targets.resolve()}",
            "--extra-vars",
            f"gods_purge_target_digest={digest}",
            "--extra-vars",
            f"gods_purge_targets_file={args.targets.resolve()}",
            "--extra-vars",
            f"gods_reclaim_plan_id={plan['plan_id']}",
        ]
        if args.ask_become_pass:
            command_args.append("--ask-become-pass")
        print(json.dumps({"operation": "purge", "target_sha256": digest, "target_count": len(normalized["targets"])}, sort_keys=True))
        return _run_ansible(command_args)

    if command == "verify-purge-targets":
        target_document = _read_json(args.targets)
        normalized = validate_purge_targets(target_document, confirmation=args.confirm_targets)
        if args.confirm_plan != plan["plan_id"]:
            raise ValueError(f"purge reclaim plan confirmation does not match current plan {plan['plan_id']}")
        print(json.dumps({
            "status": "verified",
            "target_sha256": purge_targets_digest(normalized),
            "target_count": len(normalized["targets"]),
            "plan_id": plan["plan_id"],
            "required_retained_paths_by_node": {
                node: requirements["retained_paths"]
                for node, requirements in inventory["requirements_by_node"].items()
            },
        }, sort_keys=True))
        return 0

    if command == "hash-tree":
        if args.node:
            command_args = [
                "ansible-playbook",
                "--inventory",
                str(args.inventory),
                "--limit",
                args.node,
                str(DEFAULT_HASH_PATH_PLAYBOOK),
                "--extra-vars",
                f"gods_hash_path={args.path}",
            ]
            if args.ask_become_pass:
                command_args.append("--ask-become-pass")
            return _run_ansible(command_args)
        print(json.dumps({"path": str(args.path), "sha256": hash_tree(args.path)}, sort_keys=True))
        return 0

    if command == "hash-database-dump":
        command_args = [
            "ansible-playbook",
            "--inventory",
            str(args.inventory),
            "--limit",
            args.node,
            str(DEFAULT_HASH_PATH_PLAYBOOK),
            "--extra-vars",
            f"gods_hash_path={args.path}",
            "--extra-vars",
            f"gods_hash_dump_format={args.format}",
        ]
        if args.ask_become_pass:
            command_args.append("--ask-become-pass")
        return _run_ansible(command_args)

    if command == "verify-daemonset-scope":
        document = _kubectl_json(args.kubeconfig, "get", "daemonsets", "--all-namespaces")
        report = owned_daemonsets(document)
        print(json.dumps(report, sort_keys=True))
        return 0

    if command == "verify-workloads-stopped":
        document = _kubectl_json(args.kubeconfig, "get", "pods", "--all-namespaces")
        remaining = active_workload_pods(document, ignore_daemonsets=args.ignore_daemonsets)
        report = {"status": "stopped" if not remaining else "pending", "remaining": remaining}
        print(json.dumps(report, sort_keys=True))
        return 0 if not remaining else 1

    if command == "verify-emptydir-policy":
        document = _kubectl_json(args.kubeconfig, "get", "pods", "--all-namespaces")
        report = validate_emptydir_policy(document)
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "verified" else 1

    if command == "verify-privilege":
        command_args = [
            "ansible-playbook",
            "--inventory",
            str(args.inventory),
            str(DEFAULT_PRIVILEGE_PROBE),
        ]
        if args.ask_become_pass:
            command_args.append("--ask-become-pass")
        return _run_ansible(command_args)

    return 2


def _run_ansible(command_args: list[str]) -> int:
    try:
        result = subprocess.run(command_args, check=False)
    except FileNotFoundError as exc:
        raise ValueError("ansible-playbook is required for lifecycle execution") from exc
    return result.returncode


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_private_json(path: Path) -> dict:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError(f"private lifecycle snapshot must be a regular mode-0600 file: {path}")
    if metadata.st_uid != os.getuid():
        raise ValueError(f"private lifecycle snapshot must be owned by the invoking user: {path}")
    return _read_json(path)


def _kubectl_json(kubeconfig: Path, *arguments: str) -> dict:
    command = ["kubectl", "--kubeconfig", str(kubeconfig), *arguments, "--output=json"]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ValueError("kubectl is unavailable or timed out while reading the cluster state") from exc
    if result.returncode != 0:
        raise ValueError(f"kubectl read failed: {result.stderr.strip() or 'no details'}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError("kubectl returned an invalid JSON object")
    return value


def _run_kubectl(kubeconfig: Path, arguments: list[str], *, stdin: str | None = None) -> None:
    command = ["kubectl", "--kubeconfig", str(kubeconfig), *arguments]
    try:
        result = subprocess.run(
            command,
            input=stdin,
            check=False,
            capture_output=True,
            text=True,
            timeout=360,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ValueError("kubectl is unavailable or timed out during DaemonSet recovery") from exc
    if result.returncode != 0:
        raise ValueError(f"kubectl operation failed: {result.stderr.strip() or 'no details'}")
    if result.stdout.strip():
        print(result.stdout, end="")


def _write_json(value: dict, output_path: Path | None) -> int:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
        output_path.chmod(0o600)
    else:
        print(serialized, end="")
    return 0
