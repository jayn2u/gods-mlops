from __future__ import annotations

import asyncio
import hashlib
import importlib
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

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


class EvaluationCheckpointSourceTests(unittest.TestCase):
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
