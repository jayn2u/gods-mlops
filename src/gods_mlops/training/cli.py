"""CPU-controller and GPU-worker entry point for Kubeflow components."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gods-mlops-training")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("run-training", help="submit and wait for published dataset training")
    training.add_argument("--dataset-version", required=True)
    training.add_argument("--model-kind", choices=("detr", "clip"), required=True)
    training.add_argument("--config-version", required=True)
    training.add_argument("--worker-image", required=True)

    preparation = commands.add_parser("run-preparation", help="submit and wait for one typed draft batch")
    preparation.add_argument("--source-selections-json", required=True)
    preparation.add_argument("--model-kind", choices=("detr", "qwen"), required=True)
    preparation.add_argument("--config-version", required=True)
    preparation.add_argument("--worker-image", required=True)

    probe = commands.add_parser(
        "run-probe", help="measure one locked model on Ubuntu through Task 7 and strict-SSH Docker"
    )
    probe.add_argument("--model-kind", choices=("detr", "clip", "qwen"), required=True)
    probe.add_argument("--target-phase", choices=("training", "preparation", "evaluation"))
    probe.add_argument(
        "--training-probe-job-id",
        help="successful measured training-target probe checkpoint required for evaluation profile calibration",
    )
    probe.add_argument("--worker-image", required=True, help="source-matched image already loaded on Ubuntu")
    probe.add_argument("--worker-image-id", required=True, help="exact immutable Docker image ID from Ubuntu")
    probe.add_argument(
        "--model-cache-root",
        default="/data/jayn2u/gods-mlops-model-preparation",
        help="read-only host cache root containing all locked model files",
    )
    probe.add_argument("--timeout-seconds", type=int, default=3600)
    probe.add_argument("--evidence-directory", default="output/task8-real-model-probes")

    commands.add_parser("worker", help="execute the current fenced Kubernetes GPU worker")
    args = parser.parse_args(argv)
    try:
        if args.command == "worker":
            from .worker import run_worker

            return asyncio.run(run_worker())
        if args.command == "run-probe":
            return asyncio.run(_run_probe(args))
        if args.command == "run-training":
            return asyncio.run(_run_training(args))
        return asyncio.run(_run_preparation(args))
    except (KeyError, OSError, RuntimeError, TimeoutError, ValueError) as error:
        reason_code = getattr(error, "reason_code", None)
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": type(error).__name__,
                    **({"reason_code": reason_code} if isinstance(reason_code, str) else {}),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


async def _run_training(args: argparse.Namespace) -> int:
    from .controller import build_controller_from_environment, submit_training

    queue, job_id = await submit_training(
        dataset_version=args.dataset_version,
        model_kind=args.model_kind,
        config_version=args.config_version,
    )
    controller = None
    try:
        controller = build_controller_from_environment(
            worker_image=args.worker_image,
            namespace=os.environ.get("GODS_MLOPS_WORKER_NAMESPACE", "gods-mlops"),
        )
        result = await controller.run(job_id)
        await controller.close()
        controller = None
    finally:
        if controller is not None:
            await controller.close()
        await queue.repository.close()
        await queue.source_registry.close()
    _print_job_result(job_id, result)
    return 0 if result["state"] == "completed" else 1


async def _run_preparation(args: argparse.Namespace) -> int:
    from .controller import build_controller_from_environment, submit_preparation

    try:
        raw = json.loads(args.source_selections_json)
    except json.JSONDecodeError as error:
        raise ValueError("source selections are not valid JSON") from error
    if not isinstance(raw, list) or not raw or any(not isinstance(item, dict) for item in raw):
        raise ValueError("source selections must be a non-empty JSON array of immutable refs")
    queue, job_id = await submit_preparation(
        source_selections=raw,
        model_kind=args.model_kind,
        config_version=args.config_version,
    )
    controller = None
    try:
        controller = build_controller_from_environment(
            worker_image=args.worker_image,
            namespace=os.environ.get("GODS_MLOPS_WORKER_NAMESPACE", "gods-mlops"),
        )
        result = await controller.run(job_id)
        await controller.close()
        controller = None
        if result.get("state") == "completed":
            from .review_handoff import publish_preparation_handoffs

            await publish_preparation_handoffs(queue, job_id)
    finally:
        if controller is not None:
            await controller.close()
        await queue.repository.close()
        await queue.source_registry.close()
    _print_job_result(job_id, result)
    return 0 if result["state"] == "completed" else 1


async def _run_probe(args: argparse.Namespace) -> int:
    from .docker_probe import run_docker_model_probe

    await run_docker_model_probe(
        model_kind=args.model_kind,
        target_phase=args.target_phase,
        worker_image=args.worker_image,
        expected_image_id=args.worker_image_id,
        training_probe_job_id=args.training_probe_job_id,
        model_cache_root=args.model_cache_root,
        timeout_seconds=args.timeout_seconds,
        evidence_directory=args.evidence_directory,
    )
    return 0


def _print_job_result(job_id: str, result: dict[str, Any]) -> None:
    # Never serialize queue profiles, source_refs, environment variables, or lease tokens to KFP logs.
    print(json.dumps({"job_id": job_id, "state": result["state"]}, sort_keys=True))
