from __future__ import annotations

import asyncio
import hashlib
import importlib
import unittest
from uuid import uuid4

from gods_mlops.jobs.checkpoints import CheckpointIdentity
from gods_mlops.jobs.models import EvaluationCheckpointSource, ProbeInput
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

    def test_evaluation_profile_probe_is_inference_only_and_has_explicit_fixture_identity(self) -> None:
        expected_kind = _require("gods_mlops.training.worker", "_expected_result_kind")
        verification = _require("gods_mlops.training.worker", "_probe_verification")
        probe_input = ProbeInput(
            probe_input_id="task9-clip-evaluation-synthetic-probe-v1",
            model_kind="clip",
            target_phase="evaluation",
            config_version="clip-retrieval-evaluation-probe-v1",
            manifest_object_key="probe-inputs/task9/manifest.json",
            input_sha256="e" * 64,
            object_size_bytes=512,
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
        measured = verification(
            claim,
            {"hash": "a" * 64},
            {
                "model_id": "openai/clip-vit-base-patch16",
                "model_revision": "locked-clip-revision",
                "optimizer_steps": 0,
                "inference_steps": 2,
            },
            None,
        )
        self.assertTrue(measured["passed"])
        self.assertIsNone(measured["learning_signal_verified"])

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


if __name__ == "__main__":
    unittest.main()
