from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gods_mlops.datasets.manifest import canonical_json, content_sha256
from gods_mlops.jobs.checkpoints import CheckpointIdentity
from gods_mlops.jobs.models import DatasetTrainingSource, ExecutionProfile, ProbeInput


def _require(module_name: str, symbol: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise AssertionError(f"missing evaluation implementation module {module_name}") from error
    value = getattr(module, symbol, None)
    if value is None:
        raise AssertionError(f"{module_name}.{symbol} is part of the evaluation contract")
    return value


def _checkpoint_source():
    source_type = _require("gods_mlops.jobs.models", "EvaluationCheckpointSource")
    training_job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
    identity = {
        "job_id": training_job_id,
        "input_kind": "dataset_version",
        "input_id": "dataset-2026-10-01-a1b2c3d4",
        "input_sha256": "a" * 64,
        "phase": "training",
        "model_kind": "clip",
        "config_version": "clip-training-v1",
        "config_sha256": "c" * 64,
        "dataset_version": "dataset-2026-10-01-a1b2c3d4",
    }
    return source_type(
        training_job_id=training_job_id,
        dataset_version="dataset-2026-10-01-a1b2c3d4",
        model_kind="clip",
        training_manifest_sha256="a" * 64,
        checkpoint_uri=(
            "s3://gods-mlops/jobs/" + training_job_id + "/checkpoints/" + "d" * 64 + "/" + "b" * 64 + ".checkpoint"
        ),
        checkpoint_sha256="b" * 64,
        checkpoint_size_bytes=2048,
        checkpoint_identity=identity,
        model_revision="immutable-clip-revision",
    )


def _evaluation_probe_checkpoint_source(
    checkpoint_sha: str = "b" * 64,
    *,
    model_kind: str = "detr",
    checkpoint_size_bytes: int = 244_331_513,
):
    source_type = _require("gods_mlops.jobs.models", "EvaluationProbeCheckpointSource")
    from gods_mlops.training.contracts import locked_model

    locked = locked_model(model_kind)
    probe_job_id = (
        "d4d6e1a4-d788-4722-8de8-25b9ccfe87bd"
        if model_kind == "detr"
        else "a0320b59-663c-4cdc-b893-086bb970ea60"
    )
    model_id = locked.model_id
    model_revision = locked.revision
    input_id = "task8-detr-synthetic-probe-v1" if model_kind == "detr" else "task8-clip-synthetic-probe-v1"
    config_version = (
        "task8-detr-640-microbatch1-probe-v1"
        if model_kind == "detr"
        else "task8-clip-224-microbatch2-explicit-negative-probe-v1"
    )
    config_sha = "7" * 64 if model_kind == "detr" else "8" * 64
    worker_image_id = "sha256:" + "a" * 64
    source_commit = "fa092fe485a47932eff1ebcf66f31957ee34ad12"
    runtime_evidence = {
        "event": "task8_real_model_probe_complete",
        "job_id": probe_job_id,
        "model_kind": model_kind,
        "target_phase": "training",
        "config_version": config_version,
        "input_sha256": "3" * 64,
        "docker_image_id": worker_image_id,
        "image_source_commit": source_commit,
        "source_commit": source_commit,
    }
    identity = {
        "job_id": probe_job_id,
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
        training_probe_job_id=probe_job_id,
        model_kind=model_kind,
        model_id=model_id,
        model_revision=model_revision,
        checkpoint_uri=(
            f"s3://gods-task8-test/jobs/{probe_job_id}/checkpoints/"
            + "0" * 64
            + f"/{checkpoint_sha}.checkpoint"
        ),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_size_bytes=checkpoint_size_bytes,
        checkpoint_identity=identity,
        worker_image_id=worker_image_id,
        source_commit=source_commit,
        runtime_evidence_sha256=content_sha256(canonical_json(runtime_evidence)),
    )


def _evaluation_probe_origin(source):
    identity = source.checkpoint_identity
    config = {
        "model_id": source.model_id,
        "model_revision": source.model_revision,
        "input_size": 640,
        "micro_batch": 1,
        "optimizer_steps": 3,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
    }
    job = {
        "job_id": source.training_probe_job_id,
        "state": "completed",
        "phase": "probe",
        "target_phase": "training",
        "profile_state_snapshot": "candidate",
        "input_kind": "probe_input",
        "input_id": identity["input_id"],
        "input_sha256": identity["input_sha256"],
        "dataset_version": None,
        "model_kind": source.model_kind,
        "config_version": identity["config_version"],
        "config_sha256": identity["config_sha256"],
        "checkpoint_uri": source.checkpoint_uri,
        "checkpoint_sha256": source.checkpoint_sha256,
        "checkpoint_identity": identity,
    }
    probe_profile = {
        "phase": "probe",
        "target_phase": "training",
        "model_kind": source.model_kind,
        "config_version": identity["config_version"],
        "config_sha256": identity["config_sha256"],
        "profile_state": "candidate",
        "config_json": config,
    }
    training_profile = {
        **probe_profile,
        "phase": "training",
        "target_phase": None,
        "profile_state": "measured",
    }
    measurement = {
        "measurement_id": "0be230aa-c85b-4fa5-9d7a-9f2de81d8b1e",
        "result_state": "succeeded",
        "target_phase": "training",
        "model_kind": source.model_kind,
        "input_sha256": identity["input_sha256"],
        "config_sha256": identity["config_sha256"],
        "optimizer_steps": 4,
        "checkpoint_resumed": True,
        "checkpoint_sha256": source.checkpoint_sha256,
        "exit_code": 0,
        "measured_at": "2026-10-06T00:00:00+00:00",
        "verification_details": {
            "passed": True,
            "learning_signal_verified": True,
            "checkpoint_resume_verified": True,
        },
    }
    commit = {
        "uri": source.checkpoint_uri,
        "sha256": source.checkpoint_sha256,
        "size_bytes": source.checkpoint_size_bytes,
        "identity": identity,
    }
    runtime_evidence = {
        "event": "task8_real_model_probe_complete",
        "job_id": source.training_probe_job_id,
        "model_kind": source.model_kind,
        "target_phase": "training",
        "config_version": identity["config_version"],
        "input_sha256": identity["input_sha256"],
        "docker_image_id": source.worker_image_id,
        "image_source_commit": source.source_commit,
        "source_commit": source.source_commit,
    }
    return (
        job,
        probe_profile,
        training_profile,
        measurement,
        commit,
        runtime_evidence,
        content_sha256(canonical_json(runtime_evidence)),
    )


def _probe_result_artifact(source):
    return {
        "kind": "model",
        "uri": f"s3://gods-task8-test/jobs/{source.training_probe_job_id}/results/model.artifact",
        "sha256": "c" * 64,
        "size_bytes": 2048,
        "identity": source.checkpoint_identity,
        "object_key": f"jobs/{source.training_probe_job_id}/results/model.artifact",
    }


def _probe_runtime_evidence_record(source):
    _job, _probe, _training, measurement, _commit, evidence, evidence_sha = _evaluation_probe_origin(source)
    artifact = _probe_result_artifact(source)
    return {
        "schema": "gods-mlops-probe-runtime-evidence-v1",
        "evidence_sha256": evidence_sha,
        "evidence_canonical_sha256": hashlib.sha256(canonical_json(evidence)).hexdigest(),
        "evidence_projection_sha256": hashlib.sha256(canonical_json(evidence)).hexdigest(),
        "evidence": evidence,
        "job_id": source.training_probe_job_id,
        "measurement_id": measurement["measurement_id"],
        "checkpoint_sha256": source.checkpoint_sha256,
        "checkpoint_identity": source.checkpoint_identity,
        "result_artifact": {
            key: artifact[key]
            for key in ("kind", "uri", "sha256", "size_bytes", "identity", "object_key")
        },
    }


def _evaluation_probe_queue_case(source=None):
    from gods_mlops.jobs.queue import _enqueue_dedupe_key

    source = source or _evaluation_probe_checkpoint_source()
    probe_input = ProbeInput(
        probe_input_id="task9-detr-evaluation-probe-" + source.checkpoint_sha256,
        model_kind=source.model_kind,
        target_phase="evaluation",
        config_version="task9-detr-person-coco-evaluation-probe-v1",
        manifest_object_key="probe-inputs/task9-evaluation/manifest.json",
        input_sha256="e" * 64,
        object_size_bytes=512,
        evaluation_checkpoint_source=source,
    )
    job = {
        "job_id": "e4d82c26-8f0a-4f45-8b88-5fe84302d948",
        "phase": "probe",
        "target_phase": "evaluation",
        "input_kind": "probe_input",
        "input_id": probe_input.probe_input_id,
        "input_sha256": probe_input.input_sha256,
        "dataset_version": None,
        "model_kind": source.model_kind,
        "config_version": probe_input.config_version,
        "config_sha256": "f" * 64,
        "rerun": False,
        "dedupe_key": None,
        "source_refs": probe_input.as_dict(),
    }
    identity = {
        key: job[key]
        for key in (
            "phase",
            "target_phase",
            "input_kind",
            "input_id",
            "input_sha256",
            "dataset_version",
            "model_kind",
            "config_version",
            "config_sha256",
        )
    }
    job["dedupe_key"] = _enqueue_dedupe_key(
        identity,
        {"evaluation_checkpoint_source": source.as_dict()},
    )
    prior_job, probe_profile, training_profile, measurement, commit, _evidence, _evidence_sha = (
        _evaluation_probe_origin(source)
    )
    runtime_authority = _probe_runtime_evidence_record(source)
    result_artifact = _probe_result_artifact(source)

    class Connection:
        def __init__(self):
            self.prior_job = prior_job
            self.probe_profile = probe_profile
            self.training_profile = training_profile
            self.measurement = measurement
            self.commit = commit
            self.runtime_authority_rows = [{"details": runtime_authority}]
            self.result_artifact_rows = [{"details": result_artifact}]

        async def fetchrow(self, query, *_args):
            if "FROM gods_mlops_jobs" in query:
                return self.prior_job
            if "FROM gods_mlops_resource_profiles" in query:
                return self.probe_profile if "phase='probe'" in query else self.training_profile
            if "FROM gods_mlops_profile_measurements" in query:
                return self.measurement
            if "checkpoint_committed" in query:
                return {"details": self.commit}
            raise AssertionError("unexpected query in probe-source fence")

        async def fetch(self, query, *_args):
            if "probe_runtime_evidence_committed" in query:
                return self.runtime_authority_rows
            if "result_artifact_committed" in query:
                return self.result_artifact_rows
            raise AssertionError("unexpected query in probe-source fence")

    return source, probe_input, job, identity, prior_job, Connection()


class EvaluationCheckpointSourceTests(unittest.TestCase):
    def test_training_and_evaluation_transaction_predicates_share_one_locked_source_reader(self) -> None:
        registry_type = _require("gods_mlops.jobs.sources", "DatasetSourceRegistry")
        _require("gods_mlops.jobs.sources", "_current_dataset_source_in_transaction")
        source = DatasetTrainingSource(
            dataset_version="dataset-2026-10-01-a1b2c3d4",
            target="both",
            manifest_sha256="a" * 64,
            state="published",
            training_ready=True,
            training_reasons=(),
            evaluation_eligible=False,
            evaluation_reasons=("late_cross_boundary_link",),
            invalidated_source_count=0,
            leakage_impact_count=1,
        )
        registry = object.__new__(registry_type)
        connection = object()

        async def read_phase_reasons():
            training = await registry.training_block_reasons_in_transaction(
                connection,
                dataset_version=source.dataset_version,
                model_kind="detr",
            )
            evaluation = await registry.evaluation_block_reasons_in_transaction(
                connection,
                dataset_version=source.dataset_version,
                model_kind="detr",
            )
            return training, evaluation

        with patch(
            "gods_mlops.jobs.sources._current_dataset_source_in_transaction",
            new_callable=AsyncMock,
            return_value=source,
        ) as read_source:
            training_reasons, evaluation_reasons = asyncio.run(read_phase_reasons())

        self.assertEqual(read_source.await_count, 2)
        self.assertEqual(training_reasons, ())
        self.assertIn("late_cross_boundary_link", evaluation_reasons)
        self.assertIn("dataset_not_evaluation_eligible", evaluation_reasons)

    def test_checkpoint_source_round_trips_full_training_identity_and_content_hash(self) -> None:
        source = _checkpoint_source()
        source_type = type(source)

        restored = source_type.from_dict(source.as_dict())

        self.assertEqual(restored, source)
        self.assertEqual(restored.as_dict()["schema"], "gods-mlops-evaluation-checkpoint-v1")
        self.assertEqual(restored.checkpoint_identity["phase"], "training")
        self.assertEqual(restored.checkpoint_sha256, "b" * 64)

    def test_checkpoint_source_rejects_a_mismatched_dataset_or_untrusted_uri(self) -> None:
        source = _checkpoint_source()
        source_type = type(source)
        with self.assertRaisesRegex(ValueError, "dataset"):
            source_type.from_dict({**source.as_dict(), "dataset_version": "another-dataset"})
        with self.assertRaisesRegex(ValueError, "URI|URI"):
            source_type.from_dict({**source.as_dict(), "checkpoint_uri": "https://example.invalid/checkpoint"})

    def test_checkpoint_source_requires_terminal_measured_training_and_matching_commit_marker(self) -> None:
        source = _checkpoint_source()
        training_job = {
            "job_id": source.training_job_id,
            "state": "completed",
            "phase": "training",
            "target_phase": "training",
            "profile_state_snapshot": "measured",
            "input_kind": "dataset_version",
            "input_id": source.dataset_version,
            "input_sha256": source.training_manifest_sha256,
            "dataset_version": source.dataset_version,
            "model_kind": source.model_kind,
            "config_version": "clip-training-v1",
            "config_sha256": "c" * 64,
            "checkpoint_uri": source.checkpoint_uri,
            "checkpoint_sha256": source.checkpoint_sha256,
            "checkpoint_identity": source.checkpoint_identity,
        }
        profile = {
            "phase": "training",
            "model_kind": source.model_kind,
            "config_version": "clip-training-v1",
            "config_sha256": "c" * 64,
            "profile_state": "measured",
            "config_json": {"model_revision": source.model_revision},
        }
        commit = {
            "checkpoint_uri": source.checkpoint_uri,
            "sha256": source.checkpoint_sha256,
            "size_bytes": source.checkpoint_size_bytes,
            "identity": source.checkpoint_identity,
        }

        source.validate_training_origin(training_job, profile, commit)

        with self.assertRaisesRegex(ValueError, "successful training"):
            source.validate_training_origin({**training_job, "state": "failed"}, profile, commit)
        with self.assertRaisesRegex(ValueError, "commit marker"):
            source.validate_training_origin(training_job, profile, {**commit, "sha256": "0" * 64})

    def test_probe_checkpoint_source_accepts_verified_null_dataset_training_probe(self) -> None:
        source = _evaluation_probe_checkpoint_source()
        restored = type(source).from_dict(source.as_dict())
        job, probe_profile, training_profile, measurement, commit, _evidence, _evidence_sha = (
            _evaluation_probe_origin(source)
        )

        self.assertEqual(restored, source)
        self.assertEqual(restored.as_dict()["schema"], "gods-mlops-evaluation-probe-checkpoint-v1")
        self.assertIsNone(restored.checkpoint_identity["dataset_version"])
        source.validate_training_probe_origin(
            job,
            probe_profile,
            training_profile,
            measurement,
            commit,
            _probe_runtime_evidence_record(source),
            _probe_result_artifact(source),
        )

    def test_probe_checkpoint_source_rejects_wrong_origin_profile_measurement_or_evidence(self) -> None:
        source = _evaluation_probe_checkpoint_source()
        job, probe_profile, training_profile, measurement, commit, _evidence, _evidence_sha = (
            _evaluation_probe_origin(source)
        )
        authority = _probe_runtime_evidence_record(source)
        result_artifact = _probe_result_artifact(source)
        invalid_origins = (
            ({**job, "state": "failed"}, probe_profile, training_profile, measurement, commit, authority, result_artifact),
            ({**job, "target_phase": "evaluation"}, probe_profile, training_profile, measurement, commit, authority, result_artifact),
            ({**job, "dataset_version": "forged-dataset"}, probe_profile, training_profile, measurement, commit, authority, result_artifact),
            (job, probe_profile, {**training_profile, "profile_state": "candidate"}, measurement, commit, authority, result_artifact),
            (job, probe_profile, training_profile, {**measurement, "optimizer_steps": 0}, commit, authority, result_artifact),
            (job, probe_profile, training_profile, measurement, {**commit, "size_bytes": 1}, authority, result_artifact),
            (
                job,
                probe_profile,
                training_profile,
                measurement,
                commit,
                {**authority, "evidence": {**authority["evidence"], "docker_image_id": "sha256:" + "c" * 64}},
                result_artifact,
            ),
            (job, probe_profile, training_profile, measurement, commit, {**authority, "evidence_sha256": "0" * 64}, result_artifact),
            (job, probe_profile, training_profile, measurement, commit, {**authority, "result_artifact": {}}, result_artifact),
        )
        for invalid_origin in invalid_origins:
            with self.subTest(invalid_origin=invalid_origin):
                with self.assertRaises(ValueError):
                    source.validate_training_probe_origin(*invalid_origin)

    def test_probe_checkpoint_source_never_relaxes_public_dataset_training_source(self) -> None:
        source = _evaluation_probe_checkpoint_source()
        identity = source.checkpoint_identity
        public_source = _require("gods_mlops.jobs.models", "EvaluationCheckpointSource")

        with self.assertRaisesRegex(ValueError, "dataset/model identity"):
            public_source(
                training_job_id=source.training_probe_job_id,
                dataset_version=identity["input_id"],
                model_kind=source.model_kind,
                training_manifest_sha256=identity["input_sha256"],
                checkpoint_uri=source.checkpoint_uri,
                checkpoint_sha256=source.checkpoint_sha256,
                checkpoint_size_bytes=source.checkpoint_size_bytes,
                checkpoint_identity=identity,
                model_revision=source.model_revision,
            )

    def test_task8_probe_inputs_keep_exact_legacy_ids_hashes_and_wire_shape(self) -> None:
        create_probe_input = _require("gods_mlops.training.probe_setup", "create_probe_input")

        class Objects:
            def __init__(self):
                self.values = {}

            def write_immutable(self, *, object_key, content, sha256_digest, content_type):
                previous = self.values.get(object_key)
                if previous is not None and previous != content:
                    raise AssertionError("legacy Task 8 probe object was rewritten")
                self.values[object_key] = content

        expected = {
            "training": {
                "probe_input_id": "task8-detr-synthetic-probe-v1",
                "config_version": "task8-detr-640-microbatch1-probe-v1",
                "input_sha256": "360eaf37b568049b82504a38980c8ad6a929028a167b640256247ecc639de7d5",
            },
            "preparation": {
                "probe_input_id": "task8-detr-preparation-synthetic-probe-v1",
                "config_version": "task8-detr-640-frame-drafts-preparation-probe-v1",
                "input_sha256": "abd8dcb6d0cd379af0499b5d1015430b57c1fc105f0399a57c2d59f330323a1b",
            },
        }
        for phase, expected_input in expected.items():
            probe = create_probe_input(objects=Objects(), model_kind="detr", target_phase=phase)
            wire = probe.as_dict()
            self.assertEqual(probe.probe_input_id, expected_input["probe_input_id"])
            self.assertEqual(probe.config_version, expected_input["config_version"])
            self.assertEqual(probe.input_sha256, expected_input["input_sha256"])
            self.assertEqual(
                set(wire),
                {
                    "schema", "probe_input_id", "input_kind", "model_kind", "target_phase",
                    "config_version", "manifest_object_key", "input_sha256", "object_size_bytes",
                    "fixture", "dataset_version",
                },
            )
            self.assertEqual(ProbeInput.from_dict(wire).as_dict(), wire)

    def test_probe_input_serializes_optional_evaluation_checkpoint_only_when_present(self) -> None:
        source = _evaluation_probe_checkpoint_source()
        probe = ProbeInput(
            probe_input_id="task9-detr-evaluation-probe-bbbbbbbbbbbb",
            model_kind="detr",
            target_phase="evaluation",
            config_version="task9-detr-person-coco-evaluation-probe-v1",
            manifest_object_key="probe-inputs/task9-evaluation/manifest.json",
            input_sha256="e" * 64,
            object_size_bytes=512,
            evaluation_checkpoint_source=source,
        )

        wire = probe.as_dict()
        self.assertEqual(wire["evaluation_checkpoint_source"], source.as_dict())
        self.assertEqual(ProbeInput.from_dict(wire).as_dict(), wire)

    def test_evaluation_probe_manifest_id_is_checkpoint_specific(self) -> None:
        create_probe_input = _require("gods_mlops.training.probe_setup", "create_probe_input")

        class Objects:
            def __init__(self):
                self.values = {}

            def write_immutable(self, *, object_key, content, sha256_digest, content_type):
                previous = self.values.get(object_key)
                if previous is not None and previous != content:
                    raise AssertionError("evaluation probe attempted to rewrite a frozen manifest")
                self.values[object_key] = content

        first_source = _evaluation_probe_checkpoint_source("b" * 64)
        second_source = _evaluation_probe_checkpoint_source("c" * 64)
        first_store, second_store = Objects(), Objects()
        first = create_probe_input(
            objects=first_store,
            model_kind="detr",
            target_phase="evaluation",
            evaluation_checkpoint_source=first_source,
        )
        second = create_probe_input(
            objects=second_store,
            model_kind="detr",
            target_phase="evaluation",
            evaluation_checkpoint_source=second_source,
        )

        self.assertEqual(first.probe_input_id, "task9-detr-evaluation-probe-" + "b" * 64)
        self.assertEqual(second.probe_input_id, "task9-detr-evaluation-probe-" + "c" * 64)
        self.assertNotEqual(first.manifest_object_key, second.manifest_object_key)
        self.assertNotEqual(first.input_sha256, second.input_sha256)
        self.assertEqual(first.evaluation_checkpoint_source, first_source)
        self.assertEqual(second.evaluation_checkpoint_source, second_source)

    def test_cpu_probe_source_builder_uses_recorded_origin_and_reads_committed_s3_bytes(self) -> None:
        from gods_mlops.jobs.checkpoints import CheckpointIdentity
        from gods_mlops.training.probe_setup import load_evaluation_probe_checkpoint_source

        payload = b"verified training-probe checkpoint bytes"
        source = _evaluation_probe_checkpoint_source(
            hashlib.sha256(payload).hexdigest(),
            checkpoint_size_bytes=len(payload),
        )
        job, probe_profile, training_profile, measurement, commit, _runtime_evidence, _evidence_sha = (
            _evaluation_probe_origin(source)
        )
        identity = CheckpointIdentity.from_dict(source.checkpoint_identity)

        class Repository:
            def __init__(self):
                self.evidence_reads = []

            async def get_job(self, _job_id):
                return job

            async def checkpoint_identity(self, _job_id):
                return identity

            async def checkpoint_metadata_for(self, _job_id):
                return commit

            async def get_profile(self, *, phase, **_kwargs):
                return probe_profile if phase == "probe" else training_profile

            async def profile_measurement_for_job(self, _job_id):
                return measurement

            async def probe_runtime_evidence_for_job(self, job_id):
                self.evidence_reads.append(job_id)
                return _probe_runtime_evidence_record(source)

            async def result_artifacts_for(self, _job_id):
                return [_probe_result_artifact(source)]

        class Objects:
            def __init__(self):
                self.reads = []

            def read_source(self, *, object_key, sha256_digest, size_bytes):
                self.reads.append((object_key, sha256_digest, size_bytes))
                if sha256_digest != hashlib.sha256(payload).hexdigest() or size_bytes != len(payload):
                    raise AssertionError("builder did not verify the checkpoint commit bytes")
                return payload

        objects = Objects()
        repository = Repository()
        resolved = asyncio.run(
            load_evaluation_probe_checkpoint_source(
                repository=repository,
                objects=objects,
                bucket="gods-task8-test",
                training_probe_job_id=source.training_probe_job_id,
                model_kind=source.model_kind,
            )
        )

        self.assertEqual(resolved, source)
        self.assertEqual(len(objects.reads), 1)
        self.assertEqual(objects.reads[0][1:], (source.checkpoint_sha256, len(payload)))
        self.assertEqual(repository.evidence_reads, [source.training_probe_job_id])

    def test_d4d_backfill_rejects_unreviewed_evidence_before_any_repository_write(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path

        from gods_mlops.training import probe_setup

        with TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "detr-d4d.json"
            payload = b'{"job_id":"d4d6e1a4-d788-4722-8de8-25b9ccfe87bd"}'
            evidence_path.write_bytes(payload)
            with (
                patch.object(probe_setup, "_D4D_TRAINING_PROBE_EVIDENCE_PATH", evidence_path),
                patch.object(probe_setup, "_D4D_TRAINING_PROBE_EVIDENCE_SIZE_BYTES", len(payload)),
                patch.object(probe_setup, "_D4D_TRAINING_PROBE_EVIDENCE_SHA256", "0" * 64),
            ):
                with self.assertRaisesRegex(ValueError, "pinned Task 8 record"):
                    probe_setup._load_reviewed_d4d_runtime_evidence()

    def test_d4d_backfill_verifies_pinned_identity_and_checkpoint_result_bytes_before_registration(self) -> None:
        import io
        import tarfile
        from dataclasses import replace
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace

        from gods_mlops.jobs.checkpoints import CheckpointIdentity
        from gods_mlops.training.artifacts import S3ResultArtifactStore
        from gods_mlops.training import probe_setup

        checkpoint_sha = "f59b1eaef140229b9b1e4239c91550744461cfd81456159a4d383ef2ade28f63"
        source = _evaluation_probe_checkpoint_source(checkpoint_sha, checkpoint_size_bytes=244_331_513)
        identity = {
            **source.checkpoint_identity,
            "input_sha256": "360eaf37b568049b82504a38980c8ad6a929028a167b640256247ecc639de7d5",
            "config_sha256": "7aee75e99a4f5b96ffc8c2373048f64dbf6d69ce4bd10949f0b42f12f5eb5a3b",
        }
        source = replace(
            source,
            checkpoint_identity=identity,
            worker_image_id="sha256:ae56b05de4393a9fb84fb266b662eb85311d10775110230f79cbfce58c122c28",
            source_commit="fa092fe485a47932eff1ebcf66f31957ee34ad12",
        )
        job, probe_profile, training_profile, measurement, commit, runtime_evidence, _evidence_sha = (
            _evaluation_probe_origin(source)
        )
        result_metrics = {"status": "succeeded", "resource_measurements": {"optimizer_steps": 4}}
        result_bytes = canonical_json(result_metrics)
        result_tar = io.BytesIO()
        with tarfile.open(fileobj=result_tar, mode="w") as archive:
            info = tarfile.TarInfo("metrics.json")
            info.size = len(result_bytes)
            archive.addfile(info, io.BytesIO(result_bytes))
        result_payload = result_tar.getvalue()
        result_identity = CheckpointIdentity.from_dict(identity)
        prepared = S3ResultArtifactStore(objects=object(), bucket="gods-task8-test").prepare(
            identity=result_identity,
            kind="model",
            payload=result_payload,
            reservation_bytes=len(result_payload) + 1,
        )
        artifact = {
            "kind": prepared.kind,
            "uri": prepared.uri,
            "sha256": prepared.sha256,
            "size_bytes": prepared.size_bytes,
            "identity": identity,
            "object_key": prepared.object_key,
        }
        evidence = {
            **runtime_evidence,
            "image_reference": source.worker_image_id,
            "docker_image_id": source.worker_image_id,
            "image_source_commit": source.source_commit,
            "source_commit": source.source_commit,
            "result_artifact": artifact,
            "measurement": measurement,
            "output": result_metrics,
        }
        evidence_payload = json.dumps(evidence, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        with TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "detr-d4d.json"
            evidence_path.write_bytes(evidence_payload)
            job.update({"input_sha256": identity["input_sha256"], "config_sha256": identity["config_sha256"]})
            measurement.update({"input_sha256": identity["input_sha256"], "config_sha256": identity["config_sha256"]})
            probe_profile.update({"config_sha256": identity["config_sha256"]})
            training_profile.update({"config_sha256": identity["config_sha256"]})

            class Repository:
                registered_payload = None

                async def get_job(self, _job_id):
                    return job

                async def checkpoint_identity(self, _job_id):
                    return result_identity

                async def checkpoint_metadata_for(self, _job_id):
                    return commit

                async def get_profile(self, *, phase, **_kwargs):
                    return probe_profile if phase == "probe" else training_profile

                async def profile_measurement_for_job(self, _job_id):
                    return measurement

                async def result_artifacts_for(self, _job_id):
                    return [artifact]

                async def _record_probe_runtime_evidence(self, payload):
                    self.registered_payload = payload
                    return {"evidence_sha256": hashlib.sha256(payload).hexdigest()}

            class Objects:
                def read_source(self, *, object_key, sha256_digest, size_bytes):
                    self.last_read = (object_key, sha256_digest, size_bytes)
                    return result_payload

            objects = Objects()
            repo = Repository()
            with (
                patch.object(probe_setup, "_D4D_TRAINING_PROBE_EVIDENCE_PATH", evidence_path),
                patch.object(
                    probe_setup,
                    "_D4D_TRAINING_PROBE_EVIDENCE_SHA256",
                    hashlib.sha256(evidence_payload).hexdigest(),
                ),
                patch.object(probe_setup, "_D4D_TRAINING_PROBE_EVIDENCE_SIZE_BYTES", len(evidence_payload)),
                patch(
                    "gods_mlops.training.checkpoints.S3CheckpointStore.load_uri",
                    return_value=SimpleNamespace(
                        identity=result_identity,
                        sha256=checkpoint_sha,
                        size_bytes=244_331_513,
                    ),
                ) as read_checkpoint,
            ):
                record = asyncio.run(
                    probe_setup.backfill_d4d_probe_runtime_evidence(
                        repository=repo,
                        objects=objects,
                        bucket="gods-task8-test",
                    )
                )

        self.assertEqual(repo.registered_payload, evidence_payload)
        self.assertEqual(record["evidence_sha256"], hashlib.sha256(evidence_payload).hexdigest())
        self.assertEqual(objects.last_read, (prepared.object_key, prepared.sha256, prepared.size_bytes))
        self.assertEqual(read_checkpoint.call_args.kwargs["expected_sha256"], checkpoint_sha)
        self.assertEqual(read_checkpoint.call_args.kwargs["expected_size_bytes"], 244_331_513)

    def test_probe_submit_binds_checkpoint_source_into_dedupe_only_for_calibration(self) -> None:
        from gods_mlops.jobs.queue import JobQueue

        source = _evaluation_probe_checkpoint_source()
        profile = {
            "phase": "probe",
            "target_phase": "evaluation",
            "model_kind": "detr",
            "config_version": "task9-detr-person-coco-evaluation-probe-v1",
            "config_sha256": "f" * 64,
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

        class Repository:
            def __init__(self):
                self.enqueued = []

            async def get_profile(self, *, phase, model_kind, config_version):
                self.requested_profile = (phase, model_kind, config_version)
                if config_version == "task8-detr-640-microbatch1-probe-v1":
                    return {
                        "phase": "probe",
                        "target_phase": "training",
                        "model_kind": "detr",
                        "config_version": config_version,
                        "config_sha256": "7" * 64,
                        "profile_state": "candidate",
                    }
                return profile

            async def enqueue(self, **kwargs):
                self.enqueued.append(kwargs)
                return f"job-{len(self.enqueued)}"

        async def submit(probe):
            repository = Repository()
            queue = JobQueue(repository=repository, sources=object())
            job_id = await queue.submit_probe(
                model_kind="detr",
                config_version=profile["config_version"],
                probe_input=probe,
            )
            return job_id, repository.enqueued[0]

        legacy_probe = ProbeInput(
            probe_input_id="task8-detr-synthetic-probe-v1",
            model_kind="detr",
            target_phase="training",
            config_version="task8-detr-640-microbatch1-probe-v1",
            manifest_object_key="probe-inputs/task8-training/manifest.json",
            input_sha256="3" * 64,
            object_size_bytes=512,
        )
        legacy_repository = Repository()
        legacy_queue = JobQueue(repository=legacy_repository, sources=object())
        asyncio.run(
            legacy_queue.submit_probe(
                model_kind="detr",
                config_version=legacy_probe.config_version,
                probe_input=legacy_probe,
            )
        )
        self.assertIsNone(legacy_repository.enqueued[0].get("stable_source_identity"))

        evaluation_probe = ProbeInput(
            probe_input_id="task9-detr-evaluation-probe-bbbbbbbbbbbb",
            model_kind="detr",
            target_phase="evaluation",
            config_version=profile["config_version"],
            manifest_object_key="probe-inputs/task9-evaluation/manifest.json",
            input_sha256="e" * 64,
            object_size_bytes=512,
            evaluation_checkpoint_source=source,
        )
        _, submitted = asyncio.run(submit(evaluation_probe))
        self.assertEqual(
            submitted["stable_source_identity"],
            {"evaluation_checkpoint_source": source.as_dict()},
        )
        second_source = _evaluation_probe_checkpoint_source("c" * 64)
        second_probe = ProbeInput(
            probe_input_id="task9-detr-evaluation-probe-cccccccccccc",
            model_kind="detr",
            target_phase="evaluation",
            config_version=profile["config_version"],
            manifest_object_key="probe-inputs/task9-evaluation-cccc/manifest.json",
            input_sha256="c" * 64,
            object_size_bytes=512,
            evaluation_checkpoint_source=second_source,
        )
        _, second_submission = asyncio.run(submit(second_probe))
        self.assertNotEqual(submitted["stable_source_identity"], second_submission["stable_source_identity"])

    def test_legacy_task8_probe_dedupe_key_is_unchanged_without_calibration_source(self) -> None:
        from gods_mlops.jobs.queue import _enqueue_dedupe_key

        stable_identity = {
            "phase": "probe",
            "target_phase": "training",
            "input_kind": "probe_input",
            "input_id": "task8-detr-synthetic-probe-v1",
            "input_sha256": "360eaf37b568049b82504a38980c8ad6a929028a167b640256247ecc639de7d5",
            "dataset_version": None,
            "model_kind": "detr",
            "config_version": "task8-detr-640-microbatch1-probe-v1",
            "config_sha256": "7aee75e99a4f5b96ffc8c2373048f64dbf6d69ce4bd10949f0b42f12f5eb5a3b",
        }

        self.assertEqual(
            _enqueue_dedupe_key(stable_identity, None),
            "5e1a88c09115058b66290af4755fd75ba1818737cbb3d6ecd5294461386ff236",
        )

    def test_queue_origin_fence_checks_probe_source_and_checkpoint_commit_before_write(self) -> None:
        from gods_mlops.jobs.queue import _evaluation_probe_checkpoint_source_error

        _source, _probe_input, job, _identity, _prior_job, connection = _evaluation_probe_queue_case()
        self.assertIsNone(asyncio.run(_evaluation_probe_checkpoint_source_error(connection, job)))
        self.assertEqual(
            asyncio.run(
                _evaluation_probe_checkpoint_source_error(
                    connection,
                    {**job, "dedupe_key": "0" * 64},
                )
            ),
            "evaluation_probe_checkpoint_source_invalid",
        )
        connection.measurement = {**connection.measurement, "result_state": "failed"}
        self.assertEqual(
            asyncio.run(_evaluation_probe_checkpoint_source_error(connection, job)),
            "evaluation_probe_checkpoint_source_invalid",
        )
        connection.runtime_authority_rows = [
            _probe_runtime_evidence_record(_source),
            _probe_runtime_evidence_record(_source),
        ]
        self.assertEqual(
            asyncio.run(_evaluation_probe_checkpoint_source_error(connection, job)),
            "evaluation_probe_checkpoint_source_invalid",
        )

    def test_queue_origin_fence_normalizes_postgres_uuid_prior_job_id(self) -> None:
        from uuid import UUID

        from gods_mlops.jobs.queue import _evaluation_probe_checkpoint_source_error

        source, _probe_input, job, _identity, _prior_job, connection = _evaluation_probe_queue_case()

        class AsyncpgConnection:
            async def fetchrow(self, query, *args):
                row = await connection.fetchrow(query, *args)
                if "FROM gods_mlops_jobs" in query:
                    return {**row, "job_id": UUID(source.training_probe_job_id)}
                return row

            async def fetch(self, query, *args):
                return await connection.fetch(query, *args)

        self.assertIsNone(
            asyncio.run(_evaluation_probe_checkpoint_source_error(AsyncpgConnection(), job)),
            "a valid asyncpg UUID must compare equal to the canonical source job ID",
        )

    def test_queue_origin_fence_rejects_rehashed_caller_image_source_and_evidence(self) -> None:
        from gods_mlops.jobs.queue import _enqueue_dedupe_key, _evaluation_probe_checkpoint_source_error

        source, probe_input, job, identity, _prior_job, connection = _evaluation_probe_queue_case()
        altered_source = replace(
            source,
            worker_image_id="sha256:" + "9" * 64,
            source_commit="8" * 40,
            runtime_evidence_sha256="7" * 64,
        )
        altered_input = replace(probe_input, evaluation_checkpoint_source=altered_source)
        altered_job = {
            **job,
            "source_refs": altered_input.as_dict(),
            "dedupe_key": _enqueue_dedupe_key(
                identity,
                {"evaluation_checkpoint_source": altered_source.as_dict()},
            ),
        }

        self.assertEqual(
            asyncio.run(_evaluation_probe_checkpoint_source_error(connection, altered_job)),
            "evaluation_probe_checkpoint_source_invalid",
            "the durable origin fence must reject provenance substitution after valid dedupe recomputation",
        )

    def test_queue_origin_fence_requires_exactly_one_authority_event(self) -> None:
        from gods_mlops.jobs.queue import _evaluation_probe_checkpoint_source_error

        _source, _probe_input, job, _identity, _prior_job, connection = _evaluation_probe_queue_case()
        connection.runtime_authority_rows = []
        self.assertEqual(
            asyncio.run(_evaluation_probe_checkpoint_source_error(connection, job)),
            "evaluation_probe_checkpoint_source_invalid",
        )

    def test_probe_runtime_evidence_registration_is_idempotent_and_rejects_divergent_replay(self) -> None:
        from gods_mlops.jobs.checkpoints import CheckpointIdentityError
        from gods_mlops.jobs.queue import PostgresJobQueueRepository

        source = _evaluation_probe_checkpoint_source()
        job, probe_profile, training_profile, measurement, commit, _projection, _sha = _evaluation_probe_origin(source)
        artifact = _probe_result_artifact(source)
        evidence = {
            **_projection,
            "image_reference": source.worker_image_id,
            "container_id": "a" * 64,
            "measurement": measurement,
            "result_artifact": artifact,
            "output": {"resource_measurements": artifact.get("runtime_measurements", {})},
        }
        payload = canonical_json(evidence)

        class Connection:
            def __init__(self):
                self.events = []
                self.writes = []

            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def fetchrow(self, query, *_args):
                if "SELECT * FROM gods_mlops_jobs" in query:
                    return job
                if "FROM gods_mlops_resource_profiles" in query:
                    return probe_profile if "phase='probe'" in query else training_profile
                if "FROM gods_mlops_profile_measurements" in query:
                    return measurement
                if "checkpoint_committed" in query:
                    return {"details": commit}
                return None

            async def fetch(self, query, *args):
                if "result_artifact_committed" in query:
                    return [{"details": artifact}]
                if "event_type=$2" in query and len(args) > 1 and args[1] == "probe_runtime_evidence_committed":
                    return self.events
                return []

            async def execute(self, query, *args):
                self.writes.append(query)
                if len(args) > 1 and args[1] == "probe_runtime_evidence_committed":
                    self.events.append({"details": json.loads(args[-1])})

        connection = Connection()

        class Pool:
            def acquire(self):
                return connection

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

        repository = Repository(database_url="postgresql://unused")
        registered = asyncio.run(repository._record_probe_runtime_evidence(payload))
        canonical_replay = json.dumps(evidence, sort_keys=True, ensure_ascii=False, indent=4).encode("utf-8")
        replayed = asyncio.run(repository._record_probe_runtime_evidence(canonical_replay))
        self.assertEqual(registered, replayed)
        self.assertEqual(len(connection.events), 1)
        self.assertEqual(registered["evidence_sha256"], hashlib.sha256(payload).hexdigest())

        divergent = {
            **evidence,
            "image_source_commit": "8" * 40,
            "source_commit": "8" * 40,
        }
        with self.assertRaisesRegex(CheckpointIdentityError, "different runtime evidence"):
            asyncio.run(repository._record_probe_runtime_evidence(canonical_json(divergent)))

    def test_probe_checkpoint_intent_then_runtime_authority_loss_blocks_s3_commit_and_stale_owner(self) -> None:
        from gods_mlops.jobs.checkpoints import CheckpointIdentity, CheckpointIdentityError, StaleCheckpointOwnerError
        from gods_mlops.jobs.queue import PostgresJobQueueRepository

        source = _evaluation_probe_checkpoint_source()
        prior_job, probe_profile, training_profile, measurement, commit, _evidence, _evidence_sha = (
            _evaluation_probe_origin(source)
        )
        runtime_authority = _probe_runtime_evidence_record(source)
        result_artifact = _probe_result_artifact(source)
        current_config_sha = "f" * 64
        current_config_version = "task9-detr-person-coco-evaluation-probe-v1"
        probe_input = ProbeInput(
            probe_input_id="task9-detr-evaluation-probe-" + source.checkpoint_sha256,
            model_kind="detr",
            target_phase="evaluation",
            config_version=current_config_version,
            manifest_object_key="probe-inputs/task9-evaluation/manifest.json",
            input_sha256="e" * 64,
            object_size_bytes=512,
            evaluation_checkpoint_source=source,
        )
        job = {
            "job_id": "e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            "state": "running",
            "lease_token": "8ad96890-3434-4f07-85bb-8cde17a2b009",
            "lease_generation": 1,
            "phase": "probe",
            "target_phase": "evaluation",
            "input_kind": "probe_input",
            "input_id": probe_input.probe_input_id,
            "input_sha256": probe_input.input_sha256,
            "dataset_version": None,
            "model_kind": "detr",
            "config_version": current_config_version,
            "config_sha256": current_config_sha,
            "rerun": False,
            "source_refs": probe_input.as_dict(),
            "checkpoint_uri": None,
            "checkpoint_sha256": None,
            "checkpoint_identity": None,
        }
        from gods_mlops.jobs.queue import _enqueue_dedupe_key

        stable_identity = {
            "phase": "probe",
            "target_phase": "evaluation",
            "input_kind": "probe_input",
            "input_id": job["input_id"],
            "input_sha256": job["input_sha256"],
            "dataset_version": None,
            "model_kind": "detr",
            "config_version": current_config_version,
            "config_sha256": current_config_sha,
        }
        job["dedupe_key"] = _enqueue_dedupe_key(
            stable_identity,
            {"evaluation_checkpoint_source": source.as_dict()},
        )
        identity = CheckpointIdentity.from_dict(
            {
                "job_id": job["job_id"],
                "input_kind": job["input_kind"],
                "input_id": job["input_id"],
                "input_sha256": job["input_sha256"],
                "phase": job["phase"],
                "model_kind": job["model_kind"],
                "config_version": job["config_version"],
                "config_sha256": job["config_sha256"],
                "dataset_version": None,
            }
        )
        database_now = datetime.now(UTC)
        lease = {
            "job_id": job["job_id"],
            "lease_token": job["lease_token"],
            "fencing_token": 1,
            "expires_at": database_now + timedelta(minutes=1),
        }
        reservation = {"job_id": job["job_id"], "reserved_bytes": 4096, "consumed_bytes": 0, "state": "reserved"}
        changes = {"authority_valid": True, "writes": []}

        class Connection:
            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def fetchrow(self, query, *args):
                if "ingestion_storage_usage" in query:
                    return {"used_bytes": 0}
                if "SELECT * FROM gods_mlops_jobs" in query:
                    return prior_job
                if "FROM gods_mlops_resource_profiles" in query and "checkpoint_reservation_bytes" in query:
                    return {"checkpoint_reservation_bytes": 1024, "result_reservation_bytes": 1024}
                if "FROM gods_mlops_resource_profiles" in query:
                    return probe_profile if "phase='probe'" in query else training_profile
                if "FROM gods_mlops_profile_measurements" in query:
                    return measurement
                if "checkpoint_committed" in query:
                    return {"details": commit}
                if "FROM gods_mlops_artifact_reservations" in query:
                    return reservation
                return None

            async def fetchval(self, query, *_args):
                if "clock_timestamp" in query:
                    return database_now
                return None

            async def fetch(self, query, *_args):
                if "event_type IN" in query:
                    return []
                if "probe_runtime_evidence_committed" in query:
                    return [{"details": runtime_authority}] if changes["authority_valid"] else []
                if "result_artifact_committed" in query:
                    if "SELECT event_type,details" in query:
                        return [{"event_type": "result_artifact_committed", "details": result_artifact}]
                    return [{"details": result_artifact}]
                return []

            async def execute(self, query, *_args):
                changes["writes"].append(query)

        connection = Connection()

        class Pool:
            def acquire(self):
                return connection

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

            async def _lock_job_then_lease(self, _connection, **_kwargs):
                return job, lease

        repository = Repository(database_url="postgresql://unused")
        prepared = SimpleNamespace(
            identity=identity,
            uri=f"s3://gods-task8-test/jobs/{job['job_id']}/results/probe.artifact",
            object_key=f"jobs/{job['job_id']}/results/probe.artifact",
            sha256="1" * 64,
            size_bytes=32,
            metadata_size_bytes=0,
            kind="drafts",
            previous_uri=None,
            previous_sha256=None,
            previous_size_bytes=None,
            previous_metadata_size_bytes=None,
        )
        store = SimpleNamespace(_bucket="gods-task8-test", _prefix="jobs", commit=lambda _item: self.fail("stale result reached S3 commit"))

        operation_id = asyncio.run(
            repository.begin_artifact_write(
                job_id=job["job_id"],
                lease_token=job["lease_token"],
                identity=identity,
                prepared=prepared,
                store=store,
                operation="result",
                runtime_measurements={"optimizer_steps": 0},
            )
        )
        self.assertTrue(operation_id)
        self.assertTrue(any("artifact_write_pending" in write for write in changes["writes"]))
        changes["authority_valid"] = False
        with self.assertRaisesRegex(CheckpointIdentityError, "source changed before result commit"):
            asyncio.run(
                repository.commit_result_artifact(
                    job_id=job["job_id"],
                    lease_token=job["lease_token"],
                    identity=identity,
                    prepared=prepared,
                    store=store,
                    source_registry=object(),
                    operation_id=operation_id,
                    precharged=True,
                    runtime_measurements={"optimizer_steps": 0},
                )
            )

        class StaleRepository(Repository):
            async def _lock_job_then_lease(self, _connection, **_kwargs):
                return job, None

        with self.assertRaisesRegex(StaleCheckpointOwnerError, "no longer owns"):
            asyncio.run(
                StaleRepository(database_url="postgresql://unused").commit_result_artifact(
                    job_id=job["job_id"],
                    lease_token=job["lease_token"],
                    identity=identity,
                    prepared=prepared,
                    store=store,
                    source_registry=object(),
                    operation_id=operation_id,
                    precharged=True,
                    runtime_measurements={"optimizer_steps": 0},
                )
            )

    def test_probe_measurement_fails_closed_without_runtime_authority(self) -> None:
        from gods_mlops.jobs.queue import PostgresJobQueueRepository

        source, _probe_input, job, _identity, prior_job, _connection = _evaluation_probe_queue_case()
        job = {
            **job,
            "state": "running",
            "lease_token": "8ad96890-3434-4f07-85bb-8cde17a2b009",
        }
        lease = {"job_id": job["job_id"], "lease_token": job["lease_token"], "fencing_token": 1}
        probe_profile = {
            "phase": "probe",
            "target_phase": "evaluation",
            "model_kind": source.model_kind,
            "config_version": job["config_version"],
            "profile_state": "candidate",
            "memory_requirement_mib": 36_000,
        }
        _job, training_probe_profile, training_profile, training_measurement, commit, _evidence, _sha = (
            _evaluation_probe_origin(source)
        )
        changes = []

        class Connection:
            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def fetchrow(self, query, *_args):
                if "SELECT * FROM gods_mlops_resource_profiles" in query:
                    return probe_profile
                if "FROM gods_mlops_jobs" in query:
                    return prior_job
                if "FROM gods_mlops_resource_profiles" in query:
                    return training_probe_profile if "phase='probe'" in query else training_profile
                if "FROM gods_mlops_profile_measurements" in query:
                    return training_measurement
                if "checkpoint_committed" in query:
                    return {"details": commit}
                return None

            async def fetch(self, _query, *_args):
                return []

            async def execute(self, query, *_args):
                changes.append(query)

        connection = Connection()

        class Pool:
            def acquire(self):
                return connection

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

            async def _lock_job_then_lease(self, _connection, **_kwargs):
                return job, lease

        with self.assertRaisesRegex(ValueError, "checkpoint source changed before measurement commit"):
            asyncio.run(
                Repository(database_url="postgresql://unused").record_probe_measurement(
                    job_id=job["job_id"],
                    lease_token=job["lease_token"],
                    exit_code=0,
                    peak_allocated_mib=100,
                    peak_reserved_mib=100,
                    optimizer_steps=0,
                    checkpoint_resumed=False,
                    checkpoint_sha256=None,
                    inference_steps=1,
                    verification_details={"passed": True},
                )
            )
        self.assertEqual(changes, [])

    def test_evaluation_profile_probe_and_clip_fixture_can_name_evaluation_phase(self) -> None:
        profile = ExecutionProfile(
            model_kind="clip",
            config_version="clip-evaluation-profile-v1",
            phase="probe",
            target_phase="evaluation",
            memory_requirement_mib=36_000,
            artifact_reservation_bytes=8 * 1024**3,
            config={"model_id": "openai/clip-vit-base-patch16", "model_revision": "locked", "micro_batch": 2},
            candidate=True,
        )
        probe = ProbeInput(
            probe_input_id="clip-evaluation-profile-fixture-v1",
            model_kind="clip",
            target_phase="evaluation",
            config_version=profile.config_version,
            manifest_object_key="probe-inputs/fixture/manifest.json",
            input_sha256="e" * 64,
            object_size_bytes=512,
            evaluation_checkpoint_source=_evaluation_probe_checkpoint_source(model_kind="clip"),
        )

        self.assertEqual(profile.target_phase, "evaluation")
        self.assertEqual(ProbeInput.from_dict(probe.as_dict()).target_phase, "evaluation")

    def test_profile_probe_for_evaluation_has_its_own_inference_only_config(self) -> None:
        candidate_profile = _require("gods_mlops.training.probe_setup", "candidate_profile")
        profile = candidate_profile("clip", target_phase="evaluation")

        self.assertEqual(profile.phase, "probe")
        self.assertEqual(profile.target_phase, "evaluation")
        self.assertTrue(profile.candidate)
        self.assertEqual(profile.config["micro_batch"], 2)
        self.assertNotIn("optimizer_steps", profile.config)
        self.assertNotIn("learning_rate", profile.config)

        detector_profile = candidate_profile("detr", target_phase="evaluation")
        self.assertEqual(detector_profile.config["score_threshold"], 0.3)
        self.assertEqual(detector_profile.config["max_detections"], 100)

    def test_readiness_cli_routes_evaluation_profile_probe_to_task7_runner(self) -> None:
        from gods_mlops.training.cli import main

        with patch("gods_mlops.training.docker_probe.run_docker_model_probe", new_callable=AsyncMock) as run_probe:
            try:
                exit_code = main(
                    [
                        "run-probe",
                        "--model-kind",
                        "clip",
                        "--target-phase",
                        "evaluation",
                        "--training-probe-job-id",
                        "a0320b59-663c-4cdc-b893-086bb970ea60",
                        "--worker-image",
                        "gods-mlops-training:task9",
                        "--worker-image-id",
                        "sha256:" + "f" * 64,
                    ]
                )
            except SystemExit as error:
                self.fail(f"evaluation profile probe was rejected by the CLI parser: {error}")

        self.assertEqual(exit_code, 0)
        run_probe.assert_awaited_once()
        self.assertEqual(run_probe.await_args.kwargs["target_phase"], "evaluation")
        self.assertEqual(
            run_probe.await_args.kwargs["training_probe_job_id"],
            "a0320b59-663c-4cdc-b893-086bb970ea60",
        )

    def test_evaluation_profile_probe_rejects_missing_training_probe_source_before_runtime_setup(self) -> None:
        from gods_mlops.training.docker_probe import run_docker_model_probe

        with self.assertRaisesRegex(ValueError, "training-probe job ID"):
            asyncio.run(
                run_docker_model_probe(
                    model_kind="detr",
                    target_phase="evaluation",
                    worker_image="sha256:" + "f" * 64,
                    expected_image_id="sha256:" + "f" * 64,
                )
            )

    def test_evaluation_job_requires_current_evaluation_eligibility_and_measured_profile(self) -> None:
        job_queue_type = _require("gods_mlops.jobs.queue", "JobQueue")
        submit_evaluation = getattr(job_queue_type, "submit_evaluation", None)
        if submit_evaluation is None:
            self.fail("JobQueue.submit_evaluation is part of the evaluation contract")
        source = _checkpoint_source()
        dataset_source = DatasetTrainingSource(
            dataset_version=source.dataset_version,
            target="both",
            manifest_sha256=source.training_manifest_sha256,
            state="published",
            training_ready=True,
            training_reasons=(),
            evaluation_eligible=True,
            evaluation_reasons=(),
            invalidated_source_count=0,
            leakage_impact_count=0,
        )

        class Sources:
            def __init__(self, current_source):
                self.current_source = current_source

            async def get_training_source(self, _dataset_version):
                return self.current_source

        class Repository:
            profile_state = "measured"
            inserted = None

            async def get_profile(self, *, phase, model_kind, config_version):
                self.requested_profile = (phase, model_kind, config_version)
                return {
                    "phase": "evaluation",
                    "target_phase": None,
                    "model_kind": "clip",
                    "config_version": config_version,
                    "config_sha256": "c" * 64,
                    "profile_state": self.profile_state,
                }

            async def enqueue(self, **values):
                self.inserted = values
                return "e4d82c26-8f0a-4f45-8b88-5fe84302d948"

        repository = Repository()
        sources = Sources(dataset_source)
        queue = job_queue_type(repository=repository, sources=sources)
        job_id = asyncio.run(
            queue.submit_evaluation(
                dataset_version=source.dataset_version,
                model_kind="clip",
                config_version="clip-evaluation-profile-v1",
                checkpoint_source=source,
                evaluation_split="test",
            )
        )

        self.assertEqual(job_id, "e4d82c26-8f0a-4f45-8b88-5fe84302d948")
        self.assertEqual(repository.requested_profile, ("evaluation", "clip", "clip-evaluation-profile-v1"))
        self.assertEqual(repository.inserted["phase"], "evaluation")
        self.assertEqual(repository.inserted["input_id"], source.dataset_version)
        self.assertEqual(
            repository.inserted["stable_source_identity"]["checkpoint"]["checkpoint_sha256"],
            "b" * 64,
        )

        sources.current_source = dataset_source
        repository.profile_state = "measured"
        asyncio.run(
            queue.submit_evaluation(
                dataset_version=source.dataset_version,
                model_kind="clip",
                config_version="clip-evaluation-profile-v1",
                checkpoint_source=source,
                evaluation_split="test",
                baseline_metadata={"model_id": "baseline-alias", "revision": "operator-label"},
            )
        )
        self.assertEqual(repository.inserted["source_refs"]["baseline"], {
            "model_id": "baseline-alias",
            "revision": "operator-label",
            "verified": False,
        })
        self.assertIn("baseline", repository.inserted["stable_source_identity"])

        sources.current_source = replace(
            dataset_source,
            evaluation_eligible=False,
            evaluation_reasons=("late_cross_boundary_link",),
        )
        with self.assertRaisesRegex(ValueError, "late_cross_boundary_link"):
            asyncio.run(
                queue.submit_evaluation(
                    dataset_version=source.dataset_version,
                    model_kind="clip",
                    config_version="clip-evaluation-profile-v1",
                    checkpoint_source=source,
                    evaluation_split="test",
                )
            )

        sources.current_source = dataset_source
        repository.profile_state = "candidate"
        with self.assertRaisesRegex(ValueError, "measured"):
            asyncio.run(
                queue.submit_evaluation(
                    dataset_version=source.dataset_version,
                    model_kind="clip",
                    config_version="clip-evaluation-profile-v1",
                    checkpoint_source=source,
                    evaluation_split="test",
                )
            )

    def test_dedupe_identity_changes_for_a_different_checkpoint_but_not_a_retry(self) -> None:
        dedupe_key = _require("gods_mlops.jobs.queue", "_enqueue_dedupe_key")
        base_identity = {"phase": "evaluation", "input_id": "dataset-1", "config_version": "eval-v1"}
        first_checkpoint = {"checkpoint_sha256": "b" * 64, "checkpoint_uri": "s3://bucket/a"}

        original = dedupe_key(base_identity, first_checkpoint)
        retry = dedupe_key(base_identity, dict(first_checkpoint))
        next_checkpoint = dedupe_key(
            base_identity,
            {"checkpoint_sha256": "d" * 64, "checkpoint_uri": "s3://bucket/b"},
        )

        self.assertEqual(original, retry)
        self.assertNotEqual(original, next_checkpoint)

    def test_transactional_evaluation_source_gate_uses_durable_leakage_and_invalidation_counts(self) -> None:
        registry_type = _require("gods_mlops.jobs.sources", "DatasetSourceRegistry")

        class Connection:
            sql = ""

            async def fetchrow(self, query, *_args):
                self.sql = query
                return {
                    "dataset_version": "dataset-2026-10-01-a1b2c3d4",
                    "target": "both",
                    "manifest_sha256": "a" * 64,
                    "state": "published",
                    "training_ready": True,
                    "training_reasons": "[]",
                    "evaluation_eligible": False,
                    "evaluation_reasons": '["late_cross_boundary_link"]',
                    "invalidated_source_count": 0,
                    "leakage_impact_count": 1,
                }

        connection = Connection()
        registry = object.__new__(registry_type)
        reasons = asyncio.run(
            registry.evaluation_block_reasons_in_transaction(
                connection,
                dataset_version="dataset-2026-10-01-a1b2c3d4",
                model_kind="clip",
            )
        )

        self.assertIn("late_cross_boundary_link", reasons)
        self.assertIn("dataset_not_evaluation_eligible", reasons)
        self.assertIn("FOR SHARE OF version", connection.sql)

    def test_queue_rechecks_evaluation_readiness_for_an_existing_job(self) -> None:
        queue_type = _require("gods_mlops.jobs.queue", "JobQueue")
        current_source = DatasetTrainingSource(
            dataset_version="dataset-2026-10-01-a1b2c3d4",
            target="both",
            manifest_sha256="a" * 64,
            state="published",
            training_ready=True,
            training_reasons=(),
            evaluation_eligible=False,
            evaluation_reasons=("late_cross_boundary_link",),
            invalidated_source_count=0,
            leakage_impact_count=1,
        )

        class Sources:
            async def get_training_source(self, _dataset_version):
                return current_source

        class Repository:
            async def get_job(self, _job_id):
                return {
                    "phase": "evaluation",
                    "dataset_version": current_source.dataset_version,
                    "model_kind": "clip",
                }

        queue = queue_type(repository=Repository(), sources=Sources())
        reasons = asyncio.run(queue.evaluation_source_block_reasons("e4d82c26-8f0a-4f45-8b88-5fe84302d948"))

        self.assertIn("late_cross_boundary_link", reasons)
        self.assertIn("dataset_not_evaluation_eligible", reasons)

    def test_result_commit_rechecks_current_evaluation_status_after_locking_shared_ledger(self) -> None:
        from gods_mlops.jobs.queue import PostgresJobQueueRepository

        error_type = _require("gods_mlops.jobs.queue", "DatasetNotReadyForEvaluationError")
        events = []

        class Connection:
            async def fetchrow(self, query, *_args):
                if "ingestion_storage_usage" in query:
                    events.append("global_ledger_lock")
                    return {"used_bytes": 0}
                events.append("unexpected_database_read")
                return None

            async def execute(self, *_args):
                events.append("database_write")

            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class Pool:
            def acquire(self):
                return Connection()

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

        class SourceRegistry:
            async def evaluation_block_reasons_in_transaction(self, connection, **_kwargs):
                self.connection = connection
                events.append("current_evaluation_source_check")
                return ("late_cross_boundary_link", "dataset_not_evaluation_eligible")

        dataset_version = "dataset-2026-10-01-a1b2c3d4"
        identity = CheckpointIdentity(
            job_id="e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            input_kind="dataset_version",
            input_id=dataset_version,
            input_sha256="a" * 64,
            phase="evaluation",
            model_kind="clip",
            config_version="clip-evaluation-v1",
            config_sha256="c" * 64,
            dataset_version=dataset_version,
        )
        digest = hashlib.sha256(b"evaluation report").hexdigest()
        prepared = type(
            "Prepared",
            (),
            {
                "identity": identity,
                "kind": "evaluation",
                "sha256": digest,
                "size_bytes": len(b"evaluation report"),
                "metadata_size_bytes": 0,
                "object_key": "jobs/evaluation/report.artifact",
                "uri": "s3://gods-mlops/jobs/evaluation/report.artifact",
            },
        )()
        with self.assertRaisesRegex(error_type, "late_cross_boundary_link"):
            asyncio.run(
                Repository(database_url="postgresql://unused:unused@127.0.0.1/unused").begin_artifact_write(
                    job_id=identity.job_id,
                    lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
                    identity=identity,
                    prepared=prepared,
                    store=type("Store", (), {"_bucket": "gods-mlops", "_prefix": "jobs"})(),
                    operation="result",
                    source_registry=SourceRegistry(),
                    runtime_measurements={"optimizer_steps": 0},
                )
            )

        self.assertEqual(events, ["global_ledger_lock", "current_evaluation_source_check"])

    def test_save_checkpoint_facade_rechecks_invalidation_at_commit_after_durable_intent(self) -> None:
        from gods_mlops.jobs.queue import DatasetNotReadyForEvaluationError, JobQueue, PostgresJobQueueRepository

        checkpoint_source = _checkpoint_source()
        job_id = "e4d82c26-8f0a-4f45-8b88-5fe84302d948"
        lease_token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
        identity = CheckpointIdentity(
            job_id=job_id,
            input_kind="dataset_version",
            input_id=checkpoint_source.dataset_version,
            input_sha256=checkpoint_source.training_manifest_sha256,
            phase="evaluation",
            model_kind=checkpoint_source.model_kind,
            config_version="clip-evaluation-v1",
            config_sha256="e" * 64,
            dataset_version=checkpoint_source.dataset_version,
        )
        eval_job = {
            "job_id": job_id,
            "state": "running",
            "lease_token": lease_token,
            "lease_generation": 3,
            "phase": "evaluation",
            "target_phase": "evaluation",
            "input_kind": "dataset_version",
            "input_id": checkpoint_source.dataset_version,
            "input_sha256": checkpoint_source.training_manifest_sha256,
            "dataset_version": checkpoint_source.dataset_version,
            "model_kind": checkpoint_source.model_kind,
            "config_version": identity.config_version,
            "config_sha256": identity.config_sha256,
            "checkpoint_uri": None,
            "checkpoint_sha256": None,
            "checkpoint_identity": {},
            "source_refs": {
                "schema": "gods-mlops-evaluation-source-v1",
                "dataset_version": checkpoint_source.dataset_version,
                "manifest_sha256": checkpoint_source.training_manifest_sha256,
                "target": "both",
                "evaluation_split": "test",
                "checkpoint": checkpoint_source.as_dict(),
            },
        }
        training_job = {
            "job_id": checkpoint_source.training_job_id,
            "state": "completed",
            "phase": "training",
            "target_phase": "training",
            "profile_state_snapshot": "measured",
            "input_kind": "dataset_version",
            "input_id": checkpoint_source.dataset_version,
            "input_sha256": checkpoint_source.training_manifest_sha256,
            "dataset_version": checkpoint_source.dataset_version,
            "model_kind": checkpoint_source.model_kind,
            "config_version": checkpoint_source.checkpoint_identity["config_version"],
            "config_sha256": checkpoint_source.checkpoint_identity["config_sha256"],
            "checkpoint_uri": checkpoint_source.checkpoint_uri,
            "checkpoint_sha256": checkpoint_source.checkpoint_sha256,
            "checkpoint_identity": checkpoint_source.checkpoint_identity,
        }
        training_profile = {
            "phase": "training",
            "model_kind": checkpoint_source.model_kind,
            "config_version": training_job["config_version"],
            "config_sha256": training_job["config_sha256"],
            "profile_state": "measured",
            "config_json": {"model_revision": checkpoint_source.model_revision},
        }
        checkpoint_marker = {
            "checkpoint_uri": checkpoint_source.checkpoint_uri,
            "sha256": checkpoint_source.checkpoint_sha256,
            "size_bytes": checkpoint_source.checkpoint_size_bytes,
            "identity": checkpoint_source.checkpoint_identity,
        }
        eval_profile = {"checkpoint_reservation_bytes": 32_000, "result_reservation_bytes": 32_000}
        reservation = {"state": "reserved", "reserved_bytes": 100_000, "consumed_bytes": 0}

        class Database:
            def __init__(self):
                self.events = []
                self.event_rows = []
                self.source_reads = 0
                self.stale = False

        database = Database()

        class Connection:
            async def fetchrow(self, query, *_args):
                if "JOIN gods_mlops_job_events e" in query:
                    return None
                if "ingestion_storage_usage" in query:
                    database.events.append("global_ledger_lock")
                    return {"used_bytes": 0}
                if "SELECT * FROM gods_mlops_jobs" in query and "FOR UPDATE" in query:
                    database.events.append("job_lock")
                    return eval_job
                if "FROM gods_mlops_gpu_leases" in query and "FOR UPDATE" in query:
                    database.events.append("lease_lock")
                    return None if database.stale else {
                        "job_id": job_id,
                        "lease_token": lease_token,
                        "fencing_token": 3,
                        "expires_at": datetime.now(UTC) + timedelta(minutes=1),
                    }
                if "SELECT state FROM gods_mlops_jobs" in query:
                    return {"state": eval_job["state"]}
                if "FROM gods_mlops_jobs" in query and "checkpoint_uri" not in query:
                    return training_job
                if "WHERE phase = 'training'" in query:
                    return training_profile
                if "checkpoint_committed" in query:
                    return {"details": checkpoint_marker}
                if "SELECT checkpoint_reservation_bytes,result_reservation_bytes" in query:
                    return eval_profile
                if "FROM gods_mlops_artifact_reservations" in query:
                    return reservation
                if "event_type='artifact_write_pending'" in query:
                    return None
                return None

            async def fetch(self, query, *_args):
                if "gods_mlops_job_events" in query:
                    return list(database.event_rows)
                return []

            async def fetchval(self, query, *_args):
                if "clock_timestamp" in query:
                    return datetime.now(UTC)
                if "result_artifact_committed" in query or "checkpoint_prune_pending" in query:
                    return False
                return None

            async def execute(self, query, *args):
                database.events.append("database_write")
                if "'artifact_write_pending'" in query:
                    database.event_rows.append(
                        {"event_type": "artifact_write_pending", "details": json.loads(args[3])}
                    )
                elif "'artifact_write_started'" in query:
                    database.event_rows.append(
                        {
                            "event_type": "artifact_write_started",
                            "fencing_token": args[2],
                            "details": json.loads(args[3]),
                        }
                    )
                elif "'artifact_write_quiescent'" in query:
                    database.event_rows.append(
                        {
                            "event_type": "artifact_write_quiescent",
                            "fencing_token": args[2],
                            "details": json.loads(args[3]),
                        }
                    )

            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class Pool:
            def acquire(self):
                return Connection()

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

            async def get_job(self, _job_id):
                return eval_job

            async def lease_is_current(self, _job_id, _lease_token):
                return not database.stale

            async def get_profile(self, **_kwargs):
                return eval_profile

            async def checkpoint_metadata_for(self, _job_id):
                return None

            async def pending_checkpoint_prunes_for(self, _job_id):
                return []

        class Sources:
            async def evaluation_block_reasons_in_transaction(self, _connection, **_kwargs):
                database.source_reads += 1
                database.events.append("current_evaluation_source_check")
                if database.source_reads == 2:
                    return ("late_cross_boundary_link", "dataset_not_evaluation_eligible")
                return ()

        class Prepared:
            def __init__(self):
                self.identity = identity
                self.sha256 = hashlib.sha256(b"evaluation cursor").hexdigest()
                self.size_bytes = len(b"evaluation cursor")
                self.metadata_size_bytes = 0
                self.object_key = f"jobs/{job_id}/checkpoints/evaluation/{self.sha256}.checkpoint"
                self.uri = f"s3://gods-mlops/{self.object_key}"
                self.previous_uri = None
                self.previous_sha256 = None
                self.previous_size_bytes = None

        class Store:
            _bucket = "gods-mlops"
            _prefix = "jobs"

            def prepare(self, **_kwargs):
                return Prepared()

            def commit(self, _prepared):
                database.events.append("object_commit")
                return SimpleNamespace(
                    identity=identity,
                    sha256=_prepared.sha256,
                    size_bytes=_prepared.size_bytes,
                    uri=_prepared.uri,
                    path=None,
                )

        queue = JobQueue(repository=Repository(database_url="postgresql://unused"), sources=Sources())
        error_type = DatasetNotReadyForEvaluationError
        with self.assertRaisesRegex(error_type, "late_cross_boundary_link"):
            asyncio.run(
                queue.save_checkpoint(
                    store=Store(),
                    job_id=job_id,
                    lease_token=lease_token,
                    identity=identity,
                    payload=b"evaluation cursor",
                )
            )

        self.assertEqual(database.source_reads, 2)
        self.assertEqual(database.events[:4], [
            "global_ledger_lock",
            "current_evaluation_source_check",
            "job_lock",
            "lease_lock",
        ])
        self.assertEqual(database.events[-2:], ["global_ledger_lock", "current_evaluation_source_check"])
        self.assertIn("object_commit", database.events)
        self.assertTrue(any(row["event_type"] == "artifact_write_pending" for row in database.event_rows))
        self.assertTrue(any(row["event_type"] == "artifact_write_started" for row in database.event_rows))
        self.assertTrue(any(row["event_type"] == "artifact_write_quiescent" for row in database.event_rows))
        self.assertFalse(any(row["event_type"] == "checkpoint_committed" for row in database.event_rows))

    def test_save_checkpoint_facade_rejects_stale_lease_at_final_evaluation_commit(self) -> None:
        from gods_mlops.jobs.checkpoints import StaleCheckpointOwnerError
        from gods_mlops.jobs.queue import PostgresJobQueueRepository

        identity = CheckpointIdentity(
            job_id="e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            input_kind="dataset_version",
            input_id="dataset-2026-10-01-a1b2c3d4",
            input_sha256="a" * 64,
            phase="evaluation",
            model_kind="clip",
            config_version="clip-evaluation-v1",
            config_sha256="e" * 64,
            dataset_version="dataset-2026-10-01-a1b2c3d4",
        )
        events = []

        class Connection:
            async def fetchrow(self, query, *_args):
                if "ingestion_storage_usage" in query:
                    events.append("global_ledger_lock")
                    return {"used_bytes": 0}
                if "SELECT * FROM gods_mlops_jobs" in query and "FOR UPDATE" in query:
                    events.append("job_lock")
                    return {"job_id": identity.job_id}
                if "FROM gods_mlops_gpu_leases" in query and "FOR UPDATE" in query:
                    events.append("lease_lock")
                    return None
                return None

            async def fetchval(self, query, *_args):
                if "clock_timestamp" in query:
                    return datetime.now(UTC)
                return None

            def transaction(self):
                return self

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class Pool:
            def acquire(self):
                return Connection()

        class Repository(PostgresJobQueueRepository):
            async def ensure_schema(self):
                return None

            async def _get_pool(self):
                return Pool()

        class Sources:
            async def evaluation_block_reasons_in_transaction(self, _connection, **_kwargs):
                events.append("current_evaluation_source_check")
                return ()

        prepared = type(
            "Prepared",
            (),
            {
                "identity": identity,
                "sha256": "b" * 64,
                "size_bytes": 64,
                "metadata_size_bytes": 0,
                "object_key": "jobs/evaluation/cursor.checkpoint",
                "uri": "s3://gods-mlops/jobs/evaluation/cursor.checkpoint",
            },
        )()
        repository = Repository(database_url="postgresql://unused")
        with self.assertRaises(StaleCheckpointOwnerError):
            asyncio.run(
                repository.commit_checkpoint(
                    job_id=identity.job_id,
                    lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
                    identity=identity,
                    prepared=prepared,
                    store=object(),
                    source_registry=Sources(),
                    precharged=True,
                )
            )
        self.assertEqual(
            events,
            ["global_ledger_lock", "current_evaluation_source_check", "job_lock", "lease_lock"],
        )

    def test_admission_source_reference_is_checked_against_original_job_profile_and_commit_event(self) -> None:
        validate_origin = _require("gods_mlops.jobs.queue", "_evaluation_checkpoint_source_error")
        checkpoint_source = _checkpoint_source()
        training_job = {
            "job_id": checkpoint_source.training_job_id,
            "state": "completed",
            "phase": "training",
            "target_phase": "training",
            "profile_state_snapshot": "measured",
            "input_kind": "dataset_version",
            "input_id": checkpoint_source.dataset_version,
            "input_sha256": checkpoint_source.training_manifest_sha256,
            "dataset_version": checkpoint_source.dataset_version,
            "model_kind": checkpoint_source.model_kind,
            "config_version": "clip-training-v1",
            "config_sha256": "c" * 64,
            "checkpoint_uri": checkpoint_source.checkpoint_uri,
            "checkpoint_sha256": checkpoint_source.checkpoint_sha256,
            "checkpoint_identity": checkpoint_source.checkpoint_identity,
        }
        profile = {
            "phase": "training",
            "model_kind": checkpoint_source.model_kind,
            "config_version": "clip-training-v1",
            "config_sha256": "c" * 64,
            "profile_state": "measured",
            "config_json": {"model_revision": checkpoint_source.model_revision},
        }
        event = {
            "checkpoint_uri": checkpoint_source.checkpoint_uri,
            "sha256": checkpoint_source.checkpoint_sha256,
            "size_bytes": checkpoint_source.checkpoint_size_bytes,
            "identity": checkpoint_source.checkpoint_identity,
        }
        evaluation_job = {
            "phase": "evaluation",
            "source_refs": {
                "schema": "gods-mlops-evaluation-source-v1",
                "checkpoint": checkpoint_source.as_dict(),
            },
        }

        class Connection:
            bad_event = False

            async def fetchrow(self, query, *_args):
                if "gods_mlops_resource_profiles" in query:
                    return profile
                if "gods_mlops_job_events" in query:
                    details = {**event, "sha256": "0" * 64} if self.bad_event else event
                    return {"details": details}
                if "gods_mlops_jobs" in query:
                    return training_job
                raise AssertionError("unexpected SQL in checkpoint origin check")

        connection = Connection()
        self.assertIsNone(asyncio.run(validate_origin(connection, evaluation_job)))
        connection.bad_event = True
        self.assertEqual(asyncio.run(validate_origin(connection, evaluation_job)), "evaluation_checkpoint_source_invalid")

    def test_controller_submission_uses_existing_queue_and_returns_job_identity(self) -> None:
        submit_evaluation = _require("gods_mlops.training.controller", "submit_evaluation")
        checkpoint_source = _checkpoint_source()

        class Queue:
            submission = None

            async def submit_evaluation(self, **values):
                self.submission = values
                return "e4d82c26-8f0a-4f45-8b88-5fe84302d948"

        repository = object()
        queue = Queue()
        with patch(
            "gods_mlops.training.controller._queue_from_environment",
            return_value=(repository, queue),
        ):
            returned_queue, job_id = asyncio.run(
                submit_evaluation(
                    dataset_version=checkpoint_source.dataset_version,
                    model_kind="clip",
                    config_version="clip-evaluation-v1",
                    checkpoint_source=checkpoint_source,
                    evaluation_split="test",
                )
            )

        self.assertIs(returned_queue, queue)
        self.assertEqual(job_id, "e4d82c26-8f0a-4f45-8b88-5fe84302d948")
        self.assertEqual(queue.submission["checkpoint_source"], checkpoint_source)
        self.assertEqual(queue.submission["evaluation_split"], "test")


if __name__ == "__main__":
    unittest.main()
