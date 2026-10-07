from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from uuid import uuid4
from types import SimpleNamespace
from unittest.mock import patch

from gods_mlops.jobs.checkpoints import CheckpointIdentity
from gods_mlops.jobs.models import EvaluationCheckpointSource, ProbeInput, ProcessIdentity
from gods_mlops.training.claims import WorkerAuthorizationError, WorkerClaim, validate_current_worker_claim, validate_worker_claim


def _source(*, payload: bytes = b"verified training checkpoint") -> EvaluationCheckpointSource:
    training_job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    dataset_version = "dataset-2026-10-01-a1b2c3d4"
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    identity = {
        "job_id": training_job_id,
        "input_kind": "dataset_version",
        "input_id": dataset_version,
        "input_sha256": "a" * 64,
        "phase": "training",
        "model_kind": "clip",
        "config_version": "clip-training-v1",
        "config_sha256": "c" * 64,
        "dataset_version": dataset_version,
    }
    return EvaluationCheckpointSource(
        training_job_id=training_job_id,
        dataset_version=dataset_version,
        model_kind="clip",
        training_manifest_sha256="a" * 64,
        checkpoint_uri=(
            f"s3://gods-mlops/jobs/{training_job_id}/checkpoints/" + "d" * 64 + f"/{checkpoint_sha}.checkpoint"
        ),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_size_bytes=len(payload),
        checkpoint_identity=identity,
        model_revision="locked-clip-revision",
    )


def _evaluation_probe_source(
    checkpoint_sha: str = "b" * 64,
    checkpoint_size_bytes: int = 1024,
    *,
    model_kind: str = "detr",
):
    source_type = _require("gods_mlops.jobs.models", "EvaluationProbeCheckpointSource")
    from gods_mlops.training.contracts import locked_model

    job_id = (
        "d4d6e1a4-d788-4722-8de8-25b9ccfe87bd"
        if model_kind == "detr"
        else "a0320b59-663c-4cdc-b893-086bb970ea60"
    )
    model = locked_model(model_kind)
    input_id = "task8-detr-synthetic-probe-v1" if model_kind == "detr" else "task8-clip-synthetic-probe-v1"
    config_version = (
        "task8-detr-640-microbatch1-probe-v1"
        if model_kind == "detr"
        else "task8-clip-224-microbatch2-explicit-negative-probe-v1"
    )
    config_sha = "7" * 64 if model_kind == "detr" else "8" * 64
    identity = {
        "job_id": job_id,
        "input_kind": "probe_input",
        "input_id": input_id,
        "input_sha256": "3" * 64,
        "phase": "probe",
        "model_kind": model_kind,
        "config_version": config_version,
        "config_sha256": config_sha,
        "dataset_version": None,
    }
    return source_type(
        training_probe_job_id=job_id,
        model_kind=model_kind,
        model_id=model.model_id,
        model_revision=model.revision,
        checkpoint_uri=(
            f"s3://gods-task8-test/jobs/{job_id}/checkpoints/"
            + "0" * 64
            + f"/{checkpoint_sha}.checkpoint"
        ),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_size_bytes=checkpoint_size_bytes,
        checkpoint_identity=identity,
        worker_image_id="sha256:" + "a" * 64,
        source_commit="f" * 40,
        runtime_evidence_sha256="c" * 64,
    )


def _evaluation_probe_claim_fixture(source, *, source_refs_override=None):
    from gods_mlops.datasets.manifest import canonical_json, content_sha256

    probe_id = "task9-detr-evaluation-probe-bbbbbbbbbbbb"
    config_version = "task9-detr-person-coco-evaluation-probe-v1"
    manifest_payload = canonical_json(
        {
            "schema_version": 1,
            "fixture": True,
            "phase": "probe",
            "model_kind": "detr",
            "input_kind": "probe_input",
            "input_id": probe_id,
            "config_version": config_version,
            "items": [],
        }
    )
    probe_input = ProbeInput(
        probe_input_id=probe_id,
        model_kind="detr",
        target_phase="evaluation",
        config_version=config_version,
        manifest_object_key="probe-inputs/task9-evaluation/manifest.json",
        input_sha256=content_sha256(manifest_payload),
        object_size_bytes=len(manifest_payload),
        evaluation_checkpoint_source=source,
    )
    job_id = "e4d82c26-8f0a-4f45-8b88-5fe84302d948"
    token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
    job = {
        "job_id": job_id,
        "state": "running",
        "lease_token": token,
        "lease_generation": 3,
        "phase": "probe",
        "target_phase": "evaluation",
        "input_kind": "probe_input",
        "input_id": probe_id,
        "input_sha256": probe_input.input_sha256,
        "dataset_version": None,
        "model_kind": "detr",
        "config_version": probe_input.config_version,
        "config_sha256": "e" * 64,
        "profile_state_snapshot": "candidate",
        "source_refs": source_refs_override if source_refs_override is not None else probe_input.as_dict(),
    }
    lease = {
        "job_id": job_id,
        "lease_token": token,
        "fencing_token": 3,
        "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
    }
    profile = {
        "phase": "probe",
        "target_phase": "evaluation",
        "model_kind": "detr",
        "config_version": job["config_version"],
        "config_sha256": job["config_sha256"],
        "profile_state": "candidate",
        "config_json": {
            "model_id": source.model_id,
            "model_revision": source.model_revision,
            "input_size": 640,
            "micro_batch": 1,
            "score_threshold": 0.3,
            "max_detections": 100,
            "max_evaluation_frames": 1,
        },
    }
    return job, lease, profile, manifest_payload


def _evaluation_claim_fixture(source: EvaluationCheckpointSource):
    job_id = "e4d82c26-8f0a-4f45-8b88-5fe84302d948"
    token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
    job = {
        "job_id": job_id,
        "state": "running",
        "lease_token": token,
        "lease_generation": 3,
        "phase": "evaluation",
        "target_phase": "evaluation",
        "input_kind": "dataset_version",
        "input_id": source.dataset_version,
        "input_sha256": source.training_manifest_sha256,
        "dataset_version": source.dataset_version,
        "model_kind": source.model_kind,
        "config_version": "clip-evaluation-v1",
        "config_sha256": "e" * 64,
        "profile_state_snapshot": "measured",
        "source_refs": {
            "schema": "gods-mlops-evaluation-source-v1",
            "dataset_version": source.dataset_version,
            "manifest_sha256": source.training_manifest_sha256,
            "target": "both",
            "evaluation_split": "test",
            "checkpoint": source.as_dict(),
        },
    }
    lease = {
        "job_id": job_id,
        "lease_token": token,
        "fencing_token": 3,
        "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
    }
    profile = {
        "phase": "evaluation",
        "target_phase": None,
        "model_kind": source.model_kind,
        "config_version": job["config_version"],
        "config_sha256": job["config_sha256"],
        "profile_state": "measured",
    }
    return job, lease, profile


def _require(module_name: str, symbol: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise AssertionError(f"missing evaluation worker implementation module {module_name}") from error
    value = getattr(module, symbol, None)
    if value is None:
        raise AssertionError(f"{module_name}.{symbol} is part of the evaluation worker contract")
    return value


class EvaluationWorkerTests(unittest.TestCase):
    def test_worker_claim_requires_a_measured_evaluation_profile_and_matching_checkpoint_ref(self) -> None:
        source = _source()
        job, lease, profile = _evaluation_claim_fixture(source)
        claim = WorkerClaim.from_admitted_job(job, lease, image_id="sha256:" + "f" * 64)

        validate_worker_claim(claim, job=job, lease=lease, profile=profile)

        with self.assertRaisesRegex(WorkerAuthorizationError, "checkpoint source"):
            validate_worker_claim(
                claim,
                job={**job, "source_refs": {**job["source_refs"], "checkpoint": {**source.as_dict(), "checkpoint_sha256": "0" * 64}}},
                lease=lease,
                profile=profile,
            )
        with self.assertRaisesRegex(WorkerAuthorizationError, "measured"):
            validate_worker_claim(claim, job=job, lease=lease, profile={**profile, "profile_state": "candidate"})
        forged_refs = {
            **job["source_refs"],
            "baseline": {"model_id": "baseline-alias", "revision": "unverified", "verified": True},
        }
        with self.assertRaisesRegex(WorkerAuthorizationError, "baseline"):
            validate_worker_claim(claim, job={**job, "source_refs": forged_refs}, lease=lease, profile=profile)

    def test_worker_rechecks_current_evaluation_source_before_model_start(self) -> None:
        source = _source()
        job, lease, profile = _evaluation_claim_fixture(source)
        claim = WorkerClaim.from_admitted_job(job, lease, image_id="sha256:" + "f" * 64)

        class Repository:
            async def get_active_lease(self, _gpu_uuid):
                return lease

            async def get_profile(self, **_kwargs):
                return profile

            async def lease_is_current(self, _job_id, _lease_token):
                return True

        class Queue:
            repository = Repository()

            async def get(self, _job_id):
                return job

            async def evaluation_source_block_reasons(self, _job_id):
                return ("late_cross_boundary_link", "dataset_not_evaluation_eligible")

        with self.assertRaisesRegex(WorkerAuthorizationError, "late_cross_boundary_link"):
            asyncio.run(validate_current_worker_claim(Queue(), claim))

    def test_worker_to_report_composition_preserves_raw_current_lineage(self) -> None:
        compose_report = _require("gods_mlops.training.worker", "_build_owned_evaluation_report")
        dataset_version = "dataset-2026-10-01-a1b2c3d4"
        manifest = {
            "schema_version": 1,
            "dataset_version": dataset_version,
            "target": "detr",
            "training": {"ready": True, "reason_codes": []},
            "evaluation": {"eligible": True, "reason_codes": []},
            "split_counts": {
                "test": {"groups": 3, "frames": 20, "positive_frames": 10, "negative_frames": 5}
            },
            "items": [],
        }
        from gods_mlops.datasets.manifest import canonical_json

        manifest_sha = hashlib.sha256(canonical_json(manifest)).hexdigest()
        authority = {
            "dataset_version": dataset_version,
            "training_eligible": True,
            "evaluation_eligible": True,
            "training_reasons": [],
            "evaluation_reasons": [],
            "impacts": [],
        }
        class_mapping = {
            "model_revision": "locked-detr-revision",
            "model_class_index": 17,
            "model_class_name": "person",
            "coco_category_id": 1,
            "coco_category_name": "person",
        }
        claim = SimpleNamespace(
            phase="evaluation",
            target_phase="evaluation",
            model_kind="detr",
            dataset_version=dataset_version,
            image_id="sha256:" + "2" * 64,
        )
        with patch(
            "gods_mlops.evaluation.report.evaluate_detections",
            return_value={
                "status": "complete",
                "metrics": {"ap_50_95": 0.5, "ap_50": 0.8, "ar": 0.6},
                "counts": {"frames": 20, "person_ground_truth": 10},
                "settings": {"score_threshold": 0.8, "max_detections": 200},
                "reasons": [],
            },
        ):
            report = compose_report(
                manifest=manifest,
                prediction_payload={
                    "model_kind": "detr",
                    "model_revision": "locked-detr-revision",
                    "person_class_mapping": class_mapping,
                    "input_sha256": manifest_sha,
                    "drafts": [],
                },
                claim=claim,
                candidate={
                    "model_kind": "detr",
                    "model_revision": "locked-detr-revision",
                    "checkpoint_sha256": "b" * 64,
                    "verified": True,
                },
                baseline=None,
                source={"dataset_version": dataset_version, "manifest_sha256": manifest_sha},
                evaluation_config={
                    "version": "detr-eval-v2",
                    "sha256": "c" * 64,
                    "split": "test",
                    "settings": {"score_threshold": 0.8, "max_detections": 200},
                },
                current_lineage=authority,
                model_revision="locked-detr-revision",
            )

        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["training_ready"])
        self.assertEqual(report["training_reasons"], [])
        self.assertNotIn("dataset_not_training_ready", report["insufficient_reasons"])
        self.assertEqual(report["evaluation_provenance"]["worker_image_id"], claim.image_id)

    def test_evaluation_profile_probe_is_inference_only_and_has_explicit_fixture_identity(self) -> None:
        expected_kind = _require("gods_mlops.training.worker", "_expected_result_kind")
        verification = _require("gods_mlops.training.worker", "_probe_verification")
        from gods_mlops.datasets.manifest import canonical_json

        probe_input = ProbeInput(
            probe_input_id="task9-clip-evaluation-synthetic-probe-v1",
            model_kind="clip",
            target_phase="evaluation",
            config_version="clip-retrieval-evaluation-probe-v1",
            manifest_object_key="probe-inputs/task9/manifest.json",
            input_sha256="e" * 64,
            object_size_bytes=512,
            evaluation_checkpoint_source=_evaluation_probe_source(model_kind="clip"),
        )
        claim = type(
            "ProbeClaim",
            (),
            {
                "phase": "probe",
                "target_phase": "evaluation",
                "model_kind": "clip",
                "input_kind": "probe_input",
                "input_id": probe_input.probe_input_id,
                "input_sha256": probe_input.input_sha256,
                "config_version": probe_input.config_version,
                "dataset_version": None,
            },
        )()

        self.assertTrue(probe_input.as_dict()["fixture"])
        self.assertEqual(expected_kind(claim), "evaluation_probe")
        source_payload = canonical_json(
            {
                "evaluation_probe_checkpoint": {
                    "training_probe_job_id": probe_input.evaluation_checkpoint_source.training_probe_job_id,
                    "checkpoint_sha256": probe_input.evaluation_checkpoint_source.checkpoint_sha256,
                    "model_revision": probe_input.evaluation_checkpoint_source.model_revision,
                }
            }
        )
        measured = verification(
            claim,
            {"hash": hashlib.sha256(source_payload).hexdigest()},
            {
                "model_id": "openai/clip-vit-base-patch16",
                "model_revision": "locked-clip-revision",
                "optimizer_steps": 0,
                "inference_steps": 2,
            },
            None,
            result_artifact_payload=source_payload,
            evaluation_probe_checkpoint_source=probe_input.evaluation_checkpoint_source.as_dict(),
        )
        self.assertTrue(measured["passed"])
        self.assertIsNone(measured["learning_signal_verified"])
        self.assertTrue(measured["evaluation_probe_checkpoint_verified"])

        tampered_payload = (
            b'{"evaluation_probe_checkpoint":{"training_probe_job_id":"'
            + probe_input.evaluation_checkpoint_source.training_probe_job_id.encode()
            + b'","checkpoint_sha256":"'
            + ("0" * 64).encode()
            + b'","model_revision":"locked-clip-revision"}}'
        )
        tampered = verification(
            claim,
            {"hash": hashlib.sha256(tampered_payload).hexdigest()},
            {"optimizer_steps": 0, "inference_steps": 2},
            None,
            result_artifact_payload=tampered_payload,
            evaluation_probe_checkpoint_source=probe_input.evaluation_checkpoint_source.as_dict(),
        )
        self.assertFalse(tampered["passed"])
        self.assertFalse(tampered["evaluation_probe_checkpoint_verified"])

    def test_evaluation_probe_claim_requires_typed_checkpoint_and_rejects_tampering(self) -> None:
        source = _evaluation_probe_source()
        job, lease, profile, _manifest_payload = _evaluation_probe_claim_fixture(source)
        claim = WorkerClaim.from_admitted_job(job, lease, image_id="sha256:" + "f" * 64)

        validate_worker_claim(claim, job=job, lease=lease, profile=profile)

        legacy_refs = {key: value for key, value in job["source_refs"].items() if key != "evaluation_checkpoint_source"}
        with self.assertRaisesRegex(WorkerAuthorizationError, "checkpoint"):
            validate_worker_claim(
                claim,
                job={**job, "source_refs": legacy_refs},
                lease=lease,
                profile=profile,
            )
        tampered_refs = {
            **job["source_refs"],
            "evaluation_checkpoint_source": {
                **source.as_dict(),
                "checkpoint_sha256": "0" * 64,
            },
        }
        with self.assertRaisesRegex(WorkerAuthorizationError, "checkpoint"):
            validate_worker_claim(
                claim,
                job={**job, "source_refs": tampered_refs},
                lease=lease,
                profile=profile,
            )

    def test_evaluation_probe_rechecks_prior_checkpoint_origin_before_owner_start(self) -> None:
        source = _evaluation_probe_source()
        job, lease, profile, manifest_payload = _evaluation_probe_claim_fixture(source)
        claim = WorkerClaim.from_admitted_job(job, lease, image_id="sha256:" + "f" * 64)

        class Repository:
            async def get_active_lease(self, _gpu_uuid):
                return lease

            async def get_profile(self, **_kwargs):
                return profile

            async def lease_is_current(self, _job_id, _lease_token):
                return True

        class Queue:
            repository = Repository()

            async def get(self, _job_id):
                return job

            async def evaluation_probe_source_block_reasons(self, _job_id):
                return ("training_probe_checkpoint_commit_changed",)

        class Objects:
            def read_source(self, *, object_key, sha256_digest, size_bytes):
                self.assertions = (object_key, sha256_digest, size_bytes)
                return manifest_payload

        with self.assertRaisesRegex(WorkerAuthorizationError, "training_probe_checkpoint_commit_changed"):
            asyncio.run(validate_current_worker_claim(Queue(), claim, object_store=Objects()))

    def test_worker_checkpoint_reload_rejects_missing_persisted_runtime_authority(self) -> None:
        from gods_mlops.training.worker import _load_verified_evaluation_probe_checkpoint

        source = _evaluation_probe_source()
        identity = CheckpointIdentity.from_dict(source.checkpoint_identity)
        job = {
            "job_id": source.training_probe_job_id,
            "state": "completed",
            "phase": "probe",
            "target_phase": "training",
            "input_kind": "probe_input",
            "input_id": identity.input_id,
            "input_sha256": identity.input_sha256,
            "dataset_version": None,
            "model_kind": source.model_kind,
            "config_version": identity.config_version,
            "config_sha256": identity.config_sha256,
            "checkpoint_uri": source.checkpoint_uri,
            "checkpoint_sha256": source.checkpoint_sha256,
            "checkpoint_identity": source.checkpoint_identity,
        }
        metadata = {
            "uri": source.checkpoint_uri,
            "sha256": source.checkpoint_sha256,
            "size_bytes": source.checkpoint_size_bytes,
            "identity": source.checkpoint_identity,
        }

        class Repository:
            async def checkpoint_identity(self, _job_id):
                return identity

            async def checkpoint_metadata_for(self, _job_id):
                return metadata

            async def get_profile(self, *, phase, **_kwargs):
                return {"phase": phase}

            async def profile_measurement_for_job(self, _job_id):
                return {"result_state": "succeeded"}

            async def probe_runtime_evidence_for_job(self, _job_id):
                return None

            async def result_artifacts_for(self, _job_id):
                return []

        class Queue:
            repository = Repository()

            async def get(self, _job_id):
                return job

        class Store:
            loaded = False

            def load_uri(self, *_args, **_kwargs):
                self.loaded = True
                raise AssertionError("missing runtime authority must fail before checkpoint S3 load")

        store = Store()
        with self.assertRaisesRegex(WorkerAuthorizationError, "origin is incomplete"):
            asyncio.run(_load_verified_evaluation_probe_checkpoint(Queue(), store, source))
        self.assertFalse(store.loaded)

    def test_worker_checkpoint_reload_uses_authoritative_image_source_record(self) -> None:
        from dataclasses import replace

        from gods_mlops.datasets.manifest import canonical_json
        from gods_mlops.training.worker import _load_verified_evaluation_probe_checkpoint

        source = _evaluation_probe_source()
        identity = CheckpointIdentity.from_dict(source.checkpoint_identity)
        evidence = {
            "event": "task8_real_model_probe_complete",
            "job_id": source.training_probe_job_id,
            "model_kind": source.model_kind,
            "target_phase": "training",
            "config_version": identity.config_version,
            "input_sha256": identity.input_sha256,
            "docker_image_id": source.worker_image_id,
            "image_source_commit": source.source_commit,
            "source_commit": source.source_commit,
        }
        evidence_sha256 = hashlib.sha256(canonical_json(evidence)).hexdigest()
        source = replace(source, runtime_evidence_sha256=evidence_sha256)
        result_artifact = {
            "kind": "model",
            "uri": f"s3://gods-task8-test/jobs/{source.training_probe_job_id}/results/model.artifact",
            "sha256": "c" * 64,
            "size_bytes": 2048,
            "identity": source.checkpoint_identity,
            "object_key": f"jobs/{source.training_probe_job_id}/results/model.artifact",
        }
        authority = {
            "schema": "gods-mlops-probe-runtime-evidence-v1",
            "evidence_sha256": evidence_sha256,
            "evidence_canonical_sha256": evidence_sha256,
            "evidence_projection_sha256": evidence_sha256,
            "evidence": evidence,
            "job_id": source.training_probe_job_id,
            "measurement_id": "0be230aa-c85b-4fa5-9d7a-9f2de81d8b1e",
            "checkpoint_sha256": source.checkpoint_sha256,
            "checkpoint_identity": source.checkpoint_identity,
            "result_artifact": result_artifact,
        }
        job = {
            "job_id": source.training_probe_job_id,
            "state": "completed",
            "phase": "probe",
            "target_phase": "training",
            "profile_state_snapshot": "candidate",
            "input_kind": "probe_input",
            "input_id": identity.input_id,
            "input_sha256": identity.input_sha256,
            "dataset_version": None,
            "model_kind": source.model_kind,
            "config_version": identity.config_version,
            "config_sha256": identity.config_sha256,
            "checkpoint_uri": source.checkpoint_uri,
            "checkpoint_sha256": source.checkpoint_sha256,
            "checkpoint_identity": identity.as_dict(),
        }
        profile_config = {"model_id": source.model_id, "model_revision": source.model_revision}
        probe_profile = {
            "phase": "probe", "target_phase": "training", "model_kind": source.model_kind,
            "config_version": identity.config_version, "config_sha256": identity.config_sha256,
            "profile_state": "candidate", "config_json": profile_config,
        }
        training_profile = {
            **probe_profile, "phase": "training", "target_phase": None, "profile_state": "measured",
        }
        measurement = {
            "measurement_id": authority["measurement_id"],
            "result_state": "succeeded",
            "target_phase": "training",
            "model_kind": source.model_kind,
            "input_sha256": identity.input_sha256,
            "config_sha256": identity.config_sha256,
            "optimizer_steps": 4,
            "checkpoint_resumed": True,
            "checkpoint_sha256": source.checkpoint_sha256,
            "exit_code": 0,
            "verification_details": {
                "passed": True,
                "learning_signal_verified": True,
                "checkpoint_resume_verified": True,
            },
        }
        checkpoint_commit = {
            "uri": source.checkpoint_uri,
            "sha256": source.checkpoint_sha256,
            "size_bytes": source.checkpoint_size_bytes,
            "identity": identity.as_dict(),
        }

        class Repository:
            async def checkpoint_identity(self, _job_id):
                return identity

            async def checkpoint_metadata_for(self, _job_id):
                return checkpoint_commit

            async def get_profile(self, *, phase, **_kwargs):
                return probe_profile if phase == "probe" else training_profile

            async def profile_measurement_for_job(self, _job_id):
                return measurement

            async def probe_runtime_evidence_for_job(self, _job_id):
                return authority

            async def result_artifacts_for(self, _job_id):
                return [result_artifact]

        class Queue:
            repository = Repository()

            async def get(self, _job_id):
                return job

        class Store:
            def load_uri(self, uri, *, expected_identity, expected_sha256, expected_size_bytes):
                self.request = (uri, expected_identity, expected_sha256, expected_size_bytes)
                return SimpleNamespace(
                    identity=expected_identity,
                    sha256=expected_sha256,
                    size_bytes=expected_size_bytes,
                    payload=b"verified checkpoint",
                )

        store = Store()
        verified = asyncio.run(_load_verified_evaluation_probe_checkpoint(Queue(), store, source))
        self.assertEqual(verified.sha256, source.checkpoint_sha256)
        self.assertEqual(store.request[1:], (identity, source.checkpoint_sha256, source.checkpoint_size_bytes))

    def test_evaluation_probe_weight_loader_applies_only_verified_model_state(self) -> None:
        from gods_mlops.training.worker import apply_verified_evaluation_probe_checkpoint_weights

        payload = b"verified training-probe checkpoint"
        source = _evaluation_probe_source(hashlib.sha256(payload).hexdigest(), len(payload))
        checkpoint = {
            "format": "gods-mlops-training-checkpoint-v1",
            "identity": source.checkpoint_identity,
            "model_revision": source.model_revision,
            "model_state_dict": {"trained-weight": b"trained"},
            "optimizer_state_dict": {"step": 4},
        }
        torch_module = ModuleType("torch")
        torch_module.load = lambda stream, **_kwargs: checkpoint if stream.read() == payload else None

        class Model:
            state = None
            strict = None

            def load_state_dict(self, state, *, strict):
                self.state = state
                self.strict = strict

        model = Model()
        config = {
            "_evaluation_probe_checkpoint_source": source.as_dict(),
            "_evaluation_checkpoint_payload": payload,
            "_evaluation_checkpoint_identity": source.checkpoint_identity,
            "_evaluation_checkpoint_sha256": source.checkpoint_sha256,
        }
        with patch.dict(sys.modules, {"torch": torch_module}):
            apply_verified_evaluation_probe_checkpoint_weights(
                model,
                config,
                model_kind="detr",
                model_revision=source.model_revision,
            )

        self.assertEqual(model.state, checkpoint["model_state_dict"])
        self.assertTrue(model.strict)
        self.assertNotIn("optimizer_state_dict", model.state)

    def test_detr_and_clip_evaluation_probe_load_source_weights_before_inference(self) -> None:
        class FakeCuda:
            @staticmethod
            def reset_peak_memory_stats(_device):
                return None

        torch_module = ModuleType("torch")
        torch_module.cuda = FakeCuda()
        torch_module.float32 = "float32"

        class Processor:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

        class DetrModel:
            def __init__(self):
                self.config = SimpleNamespace(id2label={"0": "person"})
                self.weights_loaded = False

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def to(self, _device):
                return self

        class ClipModel:
            def __init__(self):
                self.weights_loaded = False

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def to(self, _device):
                return self

        detr_source = _evaluation_probe_source()
        detr_loaded = []

        def mark_detr(model, *_args, **_kwargs):
            model.weights_loaded = True
            detr_loaded.append(model)

        detr_config = {
            "phase": "probe",
            "target_phase": "evaluation",
            "input_size": 640,
            "micro_batch": 1,
            "_evaluation_probe_checkpoint_source": detr_source.as_dict(),
        }
        detr_identity = {
            "model_id": detr_source.model_id,
            "model_revision": detr_source.model_revision,
            "config_version": "task9-detr-person-coco-evaluation-probe-v1",
        }
        with (
            patch.dict(sys.modules, {
                "torch": torch_module,
                "transformers": SimpleNamespace(
                    RTDetrImageProcessor=Processor,
                    RTDetrV2ForObjectDetection=DetrModel,
                ),
            }),
            patch("gods_mlops.training.detector.load_manifest", return_value={"items": []}),
            patch("gods_mlops.training.detector.validate_manifest_identity", return_value=detr_identity),
            patch("gods_mlops.training.detector.locked_model", return_value=SimpleNamespace(revision=detr_source.model_revision)),
            patch("gods_mlops.training.detector._object_store_if_needed", return_value=None),
            patch("gods_mlops.training.detector._detector_items", return_value=[{"item_id": "frame-1"}]),
            patch("gods_mlops.training.detector.load_rgb_image", return_value=SimpleNamespace(width=640, height=640)),
            patch("gods_mlops.training.detector._output_path", return_value="unused-output"),
            patch("gods_mlops.training.runner_support.cache_directory", return_value="/cache"),
            patch("gods_mlops.training.runner_support.require_cuda", return_value="cuda:0"),
            patch("gods_mlops.training.worker.apply_verified_evaluation_probe_checkpoint_weights", side_effect=mark_detr),
            patch("gods_mlops.training.detector._draft_detections", side_effect=lambda **kwargs: (
                self.assertTrue(kwargs["model"].weights_loaded) or {"status": "succeeded"}
            )),
        ):
            from gods_mlops.training import detector

            detr_result = detector.run(detr_config, "unused-manifest", "unused-output")
        self.assertEqual(detr_result["status"], "succeeded")
        self.assertEqual(len(detr_loaded), 1)

        clip_source = _evaluation_probe_source(model_kind="clip")
        clip_loaded = []

        def mark_clip(model, *_args, **_kwargs):
            model.weights_loaded = True
            clip_loaded.append(model)

        clip_config = {
            "phase": "probe",
            "target_phase": "evaluation",
            "resolution": 224,
            "micro_batch": 2,
            "gradient_accumulation_steps": 1,
            "_evaluation_probe_checkpoint_source": clip_source.as_dict(),
        }
        clip_identity = {
            "model_id": clip_source.model_id,
            "model_revision": clip_source.model_revision,
            "model_kind": "clip",
        }
        with (
            patch.dict(sys.modules, {
                "torch": torch_module,
                "transformers": SimpleNamespace(CLIPModel=ClipModel, CLIPProcessor=Processor),
            }),
            patch("gods_mlops.training.clip.load_manifest", return_value={}),
            patch("gods_mlops.training.clip.validate_manifest_identity", return_value=clip_identity),
            patch("gods_mlops.training.clip.locked_model", return_value=SimpleNamespace(revision=clip_source.model_revision)),
            patch("gods_mlops.training.runner_support.cache_directory", return_value="/cache"),
            patch("gods_mlops.training.runner_support.require_cuda", return_value="cuda:0"),
            patch("gods_mlops.training.clip._pairs_from_manifest", return_value=([{"image_id": "a", "text": "person", "media": {"image_path": "unused"}, "image": object()}, {"image_id": "b", "text": "person", "media": {"image_path": "unused"}, "image": object()}], None)),
            patch("gods_mlops.training.worker.apply_verified_evaluation_probe_checkpoint_weights", side_effect=mark_clip),
            patch("gods_mlops.training.clip._run_clip_evaluation", side_effect=lambda _config, **kwargs: (
                self.assertTrue(kwargs["model"].weights_loaded) or {"status": "succeeded"}
            )),
        ):
            from gods_mlops.training import clip

            clip_result = clip.run(clip_config, "unused-manifest", "unused-output")
        self.assertEqual(clip_result["status"], "succeeded")
        self.assertEqual(len(clip_loaded), 1)

    def test_detr_evaluation_probe_resumes_its_own_cursor_without_optimizer_state(self) -> None:
        from gods_mlops.datasets.manifest import canonical_json

        decode_evaluation_probe_cursor = _require("gods_mlops.evaluation.report", "decode_evaluation_probe_cursor")
        encode_evaluation_probe_cursor = _require("gods_mlops.evaluation.report", "encode_evaluation_probe_cursor")

        source = _evaluation_probe_source()
        job_identity = CheckpointIdentity(
            job_id="e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            input_kind="probe_input",
            input_id="task9-detr-evaluation-probe-" + source.checkpoint_sha256,
            input_sha256="e" * 64,
            phase="probe",
            model_kind="detr",
            config_version="task9-detr-person-coco-evaluation-probe-v1",
            config_sha256="f" * 64,
            dataset_version=None,
        )
        saved_draft = {
            "item_id": "task9-frame-1",
            "image_width": 640,
            "image_height": 640,
            "detections": [],
        }
        cursor_payload = encode_evaluation_probe_cursor(
            identity=job_identity.as_dict(),
            candidate_checkpoint_sha256=source.checkpoint_sha256,
            manifest_sha256=job_identity.input_sha256,
            config_sha256=job_identity.config_sha256,
            next_index=1,
            predictions={"stage": "detection_frames", "drafts": [saved_draft]},
        )
        committed_early = []
        config = {
            "phase": "probe",
            "target_phase": "evaluation",
            "score_threshold": 0.3,
            "input_sha256": job_identity.input_sha256,
            "config_sha256": job_identity.config_sha256,
            "_evaluation_probe_checkpoint_source": source.as_dict(),
            "_evaluation_checkpoint_sha256": source.checkpoint_sha256,
            "_evaluation_job_identity": job_identity.as_dict(),
            "_evaluation_resume_payload": cursor_payload,
            "_commit_result_artifact": lambda *args: (
                committed_early.append(args) or SimpleNamespace(uri="s3://task9/test-result")
            ),
        }
        identity = {
            "model_id": source.model_id,
            "model_revision": source.model_revision,
            "input_id": job_identity.input_id,
            "input_sha256": job_identity.input_sha256,
            "config_version": job_identity.config_version,
        }
        torch_module = ModuleType("torch")
        torch_module.inference_mode = nullcontext
        torch_module.cuda = SimpleNamespace(
            synchronize=lambda: None,
            max_memory_allocated=lambda _device: 0,
            max_memory_reserved=lambda _device: 0,
        )
        with (
            patch.dict(sys.modules, {"torch": torch_module}),
            patch("gods_mlops.training.detector.write_json", side_effect=lambda _path, document: canonical_json(document)),
        ):
            from gods_mlops.training.detector import _draft_detections

            result = _draft_detections(
                config=config,
                identity=identity,
                model=SimpleNamespace(eval=lambda: None),
                processor=object(),
                items=[{"item_id": "task9-frame-1"}],
                images=[SimpleNamespace(width=640, height=640)],
                person_class_id=0,
                person_class_name="person",
                device="cuda:0",
                output_uri="unused",
                output_path=Path("unused"),
                started=0.0,
            )

        document = json.loads(result["result_artifact_payload"])
        self.assertEqual(document["drafts"], [saved_draft])
        self.assertEqual(document["evaluation_probe_checkpoint"]["checkpoint_sha256"], source.checkpoint_sha256)
        self.assertEqual(committed_early, [])
        self.assertEqual(result["resource_measurements"]["optimizer_steps"], 0)
        self.assertEqual(result["resource_measurements"]["inference_steps"], 1)
        decoded = decode_evaluation_probe_cursor(
            cursor_payload,
            expected_identity=job_identity.as_dict(),
            expected_candidate_checkpoint_sha256=source.checkpoint_sha256,
            expected_manifest_sha256=job_identity.input_sha256,
            expected_config_sha256=job_identity.config_sha256,
        )
        self.assertEqual(decoded["identity"]["job_id"], job_identity.job_id)
        self.assertNotEqual(decoded["identity"]["job_id"], source.training_probe_job_id)

    def test_clip_evaluation_probe_yield_cursor_is_separate_from_training_probe_checkpoint(self) -> None:
        decode_evaluation_probe_cursor = _require("gods_mlops.evaluation.report", "decode_evaluation_probe_cursor")
        encode_evaluation_probe_cursor = _require("gods_mlops.evaluation.report", "encode_evaluation_probe_cursor")

        source = _evaluation_probe_source(model_kind="clip")
        job_identity = CheckpointIdentity(
            job_id="e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            input_kind="probe_input",
            input_id="task9-clip-evaluation-probe-" + source.checkpoint_sha256,
            input_sha256="e" * 64,
            phase="probe",
            model_kind="clip",
            config_version="task9-clip-retrieval-evaluation-probe-v1",
            config_sha256="f" * 64,
            dataset_version=None,
        )
        cursor_payload = encode_evaluation_probe_cursor(
            identity=job_identity.as_dict(),
            candidate_checkpoint_sha256=source.checkpoint_sha256,
            manifest_sha256=job_identity.input_sha256,
            config_sha256=job_identity.config_sha256,
            next_index=1,
            predictions={
                "stage": "queries",
                "query_embeddings": {"query-1": [0.5]},
                "crop_embeddings": {},
            },
        )
        config = {
            "phase": "probe",
            "target_phase": "evaluation",
            "evaluation_batch_size": 2,
            "input_sha256": job_identity.input_sha256,
            "config_sha256": job_identity.config_sha256,
            "_evaluation_probe_checkpoint_source": source.as_dict(),
            "_evaluation_checkpoint_sha256": source.checkpoint_sha256,
            "_evaluation_job_identity": job_identity.as_dict(),
            "_evaluation_resume_payload": cursor_payload,
        }
        torch_module = ModuleType("torch")
        with (
            patch.dict(sys.modules, {"torch": torch_module}),
            patch("gods_mlops.training.clip._worker_should_continue", return_value=False),
        ):
            from gods_mlops.training.clip import _run_clip_evaluation

            result = _run_clip_evaluation(
                config,
                model=object(),
                processor=object(),
                queries=[{"query_id": "query-1", "query_text": "red"}, {"query_id": "query-2", "query_text": "blue"}],
                gallery_items=[{"item_id": "crop-1"}, {"item_id": "crop-2"}],
                gallery_images=[object(), object()],
                identity={"model_id": source.model_id, "model_revision": source.model_revision, "input_id": job_identity.input_id, "input_sha256": job_identity.input_sha256},
                model_revision=source.model_revision,
                started=0.0,
            )

        resumed = decode_evaluation_probe_cursor(
            result["checkpoint_payload"],
            expected_identity=job_identity.as_dict(),
            expected_candidate_checkpoint_sha256=source.checkpoint_sha256,
            expected_manifest_sha256=job_identity.input_sha256,
            expected_config_sha256=job_identity.config_sha256,
        )
        self.assertEqual(result["status"], "yielded")
        self.assertNotIn("optimizer_steps", result.get("resource_measurements", {}))
        self.assertNotIn("optimizer_state_dict", resumed["predictions"])
        self.assertEqual(resumed["identity"]["job_id"], job_identity.job_id)
        self.assertNotEqual(resumed["identity"]["job_id"], source.training_probe_job_id)

    def test_worker_loads_only_checkpoint_with_matching_terminal_training_job_and_commit_marker(self) -> None:
        load_checkpoint = _require("gods_mlops.training.worker", "_load_verified_evaluation_checkpoint")
        payload = b"verified training checkpoint"
        source = _source(payload=payload)
        identity = CheckpointIdentity.from_dict(source.checkpoint_identity)
        training_job = {
            "job_id": source.training_job_id,
            "state": "completed",
            "phase": "training",
            "target_phase": "training",
            "input_kind": "dataset_version",
            "input_id": source.dataset_version,
            "input_sha256": source.training_manifest_sha256,
            "dataset_version": source.dataset_version,
            "model_kind": source.model_kind,
            "config_version": identity.config_version,
            "config_sha256": identity.config_sha256,
        }
        metadata = {
            "uri": source.checkpoint_uri,
            "sha256": source.checkpoint_sha256,
            "size_bytes": source.checkpoint_size_bytes,
            "identity": source.checkpoint_identity,
        }

        class Repository:
            async def checkpoint_identity(self, _job_id):
                return identity

            async def checkpoint_metadata_for(self, _job_id):
                return metadata

        class Queue:
            repository = Repository()

            async def get(self, _job_id):
                return training_job

        class Objects:
            def read_source(self, *, object_key, sha256_digest, size_bytes):
                self.object_key = object_key
                if hashlib.sha256(payload).hexdigest() != sha256_digest or len(payload) != size_bytes:
                    raise OSError("immutable checkpoint bytes changed")
                return payload

        from gods_mlops.training.checkpoints import S3CheckpointStore

        objects = Objects()
        store = S3CheckpointStore(objects=objects, bucket="gods-mlops")
        verified = asyncio.run(load_checkpoint(Queue(), store, source))

        self.assertEqual(verified.payload, payload)
        self.assertEqual(verified.sha256, source.checkpoint_sha256)
        self.assertTrue(objects.object_key.endswith(f"/{source.checkpoint_sha256}.checkpoint"))

        training_job["state"] = "failed"
        with self.assertRaisesRegex(WorkerAuthorizationError, "successful training"):
            asyncio.run(load_checkpoint(Queue(), store, source))

        training_job["state"] = "completed"
        metadata["sha256"] = "0" * 64
        with self.assertRaisesRegex(WorkerAuthorizationError, "metadata"):
            asyncio.run(load_checkpoint(Queue(), store, source))

    def test_evaluation_progress_callback_is_absent_unless_first_lease_is_armed(self) -> None:
        build_callback = _require(
            "gods_mlops.training.worker",
            "_evaluation_progress_callback",
        )
        claim = WorkerClaim(
            job_id="e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
            fence=1,
            gpu_uuid="GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
            phase="probe",
            target_phase="evaluation",
            input_kind="probe_input",
            input_id="task9-clip-evaluation-probe-" + "b" * 64,
            input_sha256="a" * 64,
            dataset_version=None,
            model_kind="clip",
            config_version="task9-clip-retrieval-evaluation-probe-v1",
            config_sha256="c" * 64,
            image_id="sha256:" + "d" * 64,
        )
        owner = ProcessIdentity(pid=12345, start_ticks=67890, uid=10001)
        deadline = {
            "job_id": claim.job_id,
            "lease_token": claim.lease_token,
            "fencing_token": claim.fence,
            "controller_invocation_id": "8f8cfb84-98a4-46b8-aabc-778da1d58aa6",
        }

        class Queue:
            def __init__(self):
                self.calls = []

            async def record_evaluation_progress(self, **values):
                self.calls.append(values)
                return {"status": "yield_requested"}

        queue = Queue()

        def bridge(coroutine):
            return asyncio.run(coroutine)

        unarmed = build_callback(
            queue=queue,
            claim=claim,
            armed_request=None,
            owner=owner,
            artifact_deadline=deadline,
            from_runner=bridge,
        )
        self.assertIsNone(unarmed)
        self.assertEqual(queue.calls, [])

        callback = build_callback(
            queue=queue,
            claim=claim,
            armed_request={"arm_event_id": 7, "expected_lease_generation": 1},
            owner=owner,
            artifact_deadline=deadline,
            from_runner=bridge,
        )
        self.assertTrue(callable(callback))
        progress = {
            "phase": "probe",
            "target_phase": "evaluation",
            "stage": "queries",
            "next_index": 2,
            "total_items": 2,
            "completed_batch_count": 1,
            "input_sha256": claim.input_sha256,
            "config_sha256": claim.config_sha256,
            "training_probe_job_id": "a0320b59-663c-4cdc-b893-086bb970ea60",
            "training_source_checkpoint_sha256": "b" * 64,
        }
        result = callback(**progress)
        self.assertEqual(result["status"], "yield_requested")
        self.assertEqual(len(queue.calls), 1)
        self.assertEqual(queue.calls[0]["job_id"], claim.job_id)
        self.assertEqual(queue.calls[0]["lease_token"], claim.lease_token)
        self.assertEqual(queue.calls[0]["fencing_token"], claim.fence)
        self.assertEqual(queue.calls[0]["arm_event_id"], 7)
        self.assertEqual(queue.calls[0]["progress"], progress)
        self.assertIs(queue.calls[0]["owner"], owner)
        self.assertIs(queue.calls[0]["artifact_deadline"], deadline)


if __name__ == "__main__":
    unittest.main()
