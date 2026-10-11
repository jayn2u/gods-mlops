from __future__ import annotations

import importlib
import json
import unittest
from time import perf_counter
from unittest.mock import patch


def _require(module_name: str, symbol: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise AssertionError(f"missing evaluation runner module {module_name}") from error
    value = getattr(module, symbol, None)
    if value is None:
        raise AssertionError(f"{module_name}.{symbol} is part of the evaluation runner contract")
    return value


class ClipEvaluationInferenceTests(unittest.TestCase):
    def test_clip_evaluation_uses_only_frozen_query_text_and_selected_gallery_ids(self) -> None:
        select_inputs = _require("gods_mlops.training.clip", "clip_evaluation_sources")
        manifest = {
            "items": [
                {"kind": "crop", "item_id": "crop-train", "split": "train"},
                {"kind": "crop", "item_id": "crop-test-a", "split": "test"},
                {"kind": "crop", "item_id": "crop-test-b", "split": "test"},
                {"kind": "crop", "item_id": "crop-validation", "split": "validation"},
            ],
            "evaluation": {
                "gallery_crop_ids": ["crop-test-b", "crop-test-a"],
                "relevance_matrices": [
                    {
                        "query_id": "query-1",
                        "query_text": "frozen human reviewed wording",
                        "status": "complete",
                        "evaluation_eligible": True,
                        "gallery_matches_selection": True,
                        "judgments": [
                            {"crop_id": "crop-test-a", "judgment": "relevant"},
                            {"crop_id": "crop-test-b", "judgment": "not_relevant"},
                        ],
                    }
                ],
            },
        }

        queries, gallery = select_inputs(manifest, evaluation_split="test")

        self.assertEqual(queries, [{"query_id": "query-1", "query_text": "frozen human reviewed wording"}])
        self.assertEqual([item["item_id"] for item in gallery], ["crop-test-a", "crop-test-b"])
        self.assertNotIn("crop-train", [item["item_id"] for item in gallery])

    def test_clip_evaluation_refuses_incomplete_frozen_human_truth(self) -> None:
        select_inputs = _require("gods_mlops.training.clip", "clip_evaluation_sources")
        manifest = {
            "items": [{"kind": "crop", "item_id": "crop-test-a", "split": "test"}],
            "evaluation": {
                "gallery_crop_ids": ["crop-test-a"],
                "relevance_matrices": [
                    {
                        "query_id": "query-1",
                        "query_text": "frozen query",
                        "status": "pending",
                        "evaluation_eligible": False,
                        "gallery_matches_selection": True,
                        "judgments": [],
                    }
                ],
            },
        }

        with self.assertRaisesRegex(ValueError, "human relevance"):
            select_inputs(manifest, evaluation_split="test")

    def test_candidate_embedding_batches_run_eval_only_with_normalized_features(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("locked training image provides torch for model inference coverage")
        encode = _require("gods_mlops.training.clip", "encode_evaluation_batch")

        class Processor:
            def __call__(self, *, text=None, images=None, return_tensors, padding=True, truncation=False):
                if text is not None:
                    return {"input_ids": torch.tensor([[3.0, 4.0], [0.0, 2.0]])}
                return {"pixel_values": torch.tensor([[1.0, 0.0], [1.0, 1.0]])}

        class Model:
            eval_called = False
            text_inference_mode = False
            image_inference_mode = False

            def eval(self):
                self.eval_called = True
                return self

            def get_text_features(self, *, input_ids):
                self.text_inference_mode = torch.is_inference_mode_enabled()
                return input_ids

            def get_image_features(self, *, pixel_values):
                self.image_inference_mode = torch.is_inference_mode_enabled()
                return pixel_values

        model = Model()
        text_embeddings, image_embeddings = encode(
            model,
            Processor(),
            texts=["query-a", "query-b"],
            images=[object(), object()],
            device=torch.device("cpu"),
        )

        self.assertTrue(model.eval_called)
        self.assertTrue(model.text_inference_mode)
        self.assertTrue(model.image_inference_mode)
        self.assertAlmostEqual(text_embeddings[0][0], 0.6)
        self.assertAlmostEqual(text_embeddings[0][1], 0.8)
        self.assertEqual(text_embeddings[1], [0.0, 1.0])
        self.assertAlmostEqual(image_embeddings[1][0], 2 ** -0.5)
        self.assertAlmostEqual(image_embeddings[1][1], 2 ** -0.5)

    def test_clip_eval_yield_resume_keeps_its_own_cursor_not_training_optimizer_state(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("locked training image provides torch for cursor runner coverage")
        run_eval = _require("gods_mlops.training.clip", "_run_clip_evaluation")
        encode = _require("gods_mlops.evaluation.report", "decode_evaluation_cursor")
        identity = {
            "job_id": "e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            "input_kind": "dataset_version",
            "input_id": "dataset-2026-10-01-a1b2c3d4",
            "input_sha256": "a" * 64,
            "phase": "evaluation",
            "model_kind": "clip",
            "config_version": "clip-eval-v1",
            "config_sha256": "c" * 64,
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
        }
        candidate_sha = "b" * 64

        class Processor:
            def __call__(self, *, text=None, images=None, return_tensors, padding=True, truncation=False):
                if text is not None:
                    return {"input_ids": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
                return {"pixel_values": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}

        class Model:
            eval_called = False

            def eval(self):
                self.eval_called = True
                return self

            def get_text_features(self, *, input_ids):
                return input_ids

            def get_image_features(self, *, pixel_values):
                return pixel_values

        model = Model()
        config = {
            "phase": "evaluation",
            "target_phase": "evaluation",
            "evaluation_split": "test",
            "evaluation_batch_size": 2,
            "job_id": identity["job_id"],
            "input_sha256": identity["input_sha256"],
            "config_sha256": identity["config_sha256"],
            "_evaluation_checkpoint_sha256": candidate_sha,
            "_evaluation_job_identity": identity,
            "_device": torch.device("cpu"),
        }
        guard_calls = 0

        def yield_before_gallery() -> bool:
            nonlocal guard_calls
            guard_calls += 1
            return guard_calls == 1

        config["_assert_current"] = yield_before_gallery
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("torch.cuda.max_memory_reserved", return_value=0),
        ):
            yielded = run_eval(
                config,
                model=model,
                processor=Processor(),
                queries=[{"query_id": "query-a", "query_text": "first query"}, {"query_id": "query-b", "query_text": "second query"}],
                gallery_items=[{"item_id": "crop-a"}, {"item_id": "crop-b"}],
                gallery_images=[object(), object()],
                identity={"model_id": "openai/clip-vit-base-patch16", "input_id": identity["input_id"], "input_sha256": identity["input_sha256"]},
                model_revision="locked-clip-revision",
                started=perf_counter(),
            )

        self.assertEqual(yielded["status"], "yielded")
        self.assertTrue(model.eval_called)
        cursor = encode(
            yielded["checkpoint_payload"],
            expected_identity=identity,
            expected_candidate_checkpoint_sha256=candidate_sha,
            expected_manifest_sha256=identity["input_sha256"],
            expected_config_sha256=identity["config_sha256"],
        )
        self.assertEqual(cursor["predictions"]["stage"], "gallery")
        config["_assert_current"] = lambda: True
        config["_evaluation_resume_payload"] = yielded["checkpoint_payload"]
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("torch.cuda.max_memory_reserved", return_value=0),
        ):
            resumed = run_eval(
                config,
                model=model,
                processor=Processor(),
                queries=[{"query_id": "query-a", "query_text": "first query"}, {"query_id": "query-b", "query_text": "second query"}],
                gallery_items=[{"item_id": "crop-a"}, {"item_id": "crop-b"}],
                gallery_images=[object(), object()],
                identity={"model_id": "openai/clip-vit-base-patch16", "input_id": identity["input_id"], "input_sha256": identity["input_sha256"]},
                model_revision="locked-clip-revision",
                started=perf_counter(),
            )

        self.assertEqual(resumed["status"], "succeeded")
        predictions = json.loads(resumed["result_artifact_payload"])
        self.assertEqual(set(predictions["query_embeddings"]), {"query-a", "query-b"})
        self.assertEqual(set(predictions["crop_embeddings"]), {"crop-a", "crop-b"})
        self.assertEqual(resumed["resource_measurements"]["optimizer_steps"], 0)

    def test_clip_evaluation_probe_reports_query_boundary_before_nonempty_cursor_yield(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("locked training image provides torch for cursor runner coverage")
        run_eval = _require("gods_mlops.training.clip", "_run_clip_evaluation")
        decode_cursor = _require("gods_mlops.evaluation.report", "decode_evaluation_probe_cursor")
        source_checkpoint_sha = "b" * 64
        identity = {
            "job_id": "e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            "input_kind": "probe_input",
            "input_id": f"task9-clip-evaluation-probe-{source_checkpoint_sha}",
            "input_sha256": "a" * 64,
            "phase": "probe",
            "model_kind": "clip",
            "config_version": "task9-clip-retrieval-evaluation-probe-v1",
            "config_sha256": "c" * 64,
            "dataset_version": None,
        }

        class Processor:
            def __call__(self, *, text=None, images=None, return_tensors, padding=True, truncation=False):
                if text is not None:
                    return {"input_ids": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
                return {"pixel_values": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}

        class Model:
            def eval(self):
                return self

            def get_text_features(self, *, input_ids):
                return input_ids

            def get_image_features(self, *, pixel_values):
                return pixel_values

        guard_calls = 0
        reported = []

        def yield_before_gallery() -> bool:
            nonlocal guard_calls
            guard_calls += 1
            return guard_calls == 1

        def report_batch(**progress) -> None:
            reported.append(progress)

        config = {
            "phase": "probe",
            "target_phase": "evaluation",
            "evaluation_batch_size": 2,
            "input_sha256": identity["input_sha256"],
            "config_sha256": identity["config_sha256"],
            "_evaluation_probe_checkpoint_source": {
                "training_probe_job_id": "a0320b59-663c-4cdc-b893-086bb970ea60",
                "checkpoint_sha256": source_checkpoint_sha,
                "model_id": "openai/clip-vit-base-patch16",
                "model_revision": "locked-clip-revision",
            },
            "_evaluation_checkpoint_sha256": source_checkpoint_sha,
            "_evaluation_job_identity": identity,
            "_device": torch.device("cpu"),
            "_assert_current": yield_before_gallery,
            "_evaluation_batch_completed": report_batch,
        }

        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("torch.cuda.max_memory_reserved", return_value=0),
        ):
            yielded = run_eval(
                config,
                model=Model(),
                processor=Processor(),
                queries=[
                    {"query_id": "query-a", "query_text": "first query"},
                    {"query_id": "query-b", "query_text": "second query"},
                ],
                gallery_items=[{"item_id": "crop-a"}, {"item_id": "crop-b"}],
                gallery_images=[object(), object()],
                identity={
                    "model_id": "openai/clip-vit-base-patch16",
                    "model_revision": "locked-clip-revision",
                    "input_id": identity["input_id"],
                    "input_sha256": identity["input_sha256"],
                },
                model_revision="locked-clip-revision",
                started=perf_counter(),
            )

        self.assertEqual(len(reported), 1)
        self.assertEqual(
            reported[0],
            {
                "phase": "probe",
                "target_phase": "evaluation",
                "model_kind": "clip",
                "stage": "queries",
                "next_index": 2,
                "total_items": 2,
                "completed_batch_count": 1,
                "input_sha256": identity["input_sha256"],
                "config_sha256": identity["config_sha256"],
                "training_probe_job_id": "a0320b59-663c-4cdc-b893-086bb970ea60",
                "training_source_checkpoint_sha256": source_checkpoint_sha,
            },
        )
        self.assertEqual(yielded["status"], "yielded")
        cursor = decode_cursor(
            yielded["checkpoint_payload"],
            expected_identity=identity,
            expected_candidate_checkpoint_sha256=source_checkpoint_sha,
            expected_manifest_sha256=identity["input_sha256"],
            expected_config_sha256=identity["config_sha256"],
        )
        self.assertEqual(cursor["predictions"]["stage"], "gallery")
        self.assertEqual(cursor["next_index"], 0)
        self.assertEqual(set(cursor["predictions"]["query_embeddings"]), {"query-a", "query-b"})
        self.assertEqual(cursor["predictions"]["crop_embeddings"], {})

        def run_to_completion(progress_callback=None):
            complete_config = dict(config)
            complete_config["_assert_current"] = lambda: True
            if progress_callback is None:
                complete_config.pop("_evaluation_batch_completed", None)
            else:
                complete_config["_evaluation_batch_completed"] = progress_callback
            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
            ):
                return run_eval(
                    complete_config,
                    model=Model(),
                    processor=Processor(),
                    queries=[
                        {"query_id": "query-a", "query_text": "first query"},
                        {"query_id": "query-b", "query_text": "second query"},
                    ],
                    gallery_items=[{"item_id": "crop-a"}, {"item_id": "crop-b"}],
                    gallery_images=[object(), object()],
                    identity={
                        "model_id": "openai/clip-vit-base-patch16",
                        "model_revision": "locked-clip-revision",
                        "input_id": identity["input_id"],
                        "input_sha256": identity["input_sha256"],
                    },
                    model_revision="locked-clip-revision",
                    started=perf_counter(),
                )

        unarmed_result = run_to_completion()
        observed_boundaries = []
        callback_result = run_to_completion(lambda **progress: observed_boundaries.append(progress))
        self.assertEqual(unarmed_result["status"], "succeeded")
        self.assertEqual(unarmed_result["result_artifact_payload"], callback_result["result_artifact_payload"])
        self.assertEqual(unarmed_result["resource_measurements"]["inference_steps"], 2)
        self.assertEqual(callback_result["resource_measurements"]["inference_steps"], 2)
        self.assertEqual([item["stage"] for item in observed_boundaries], ["queries", "gallery"])

    def test_evaluation_probe_dispatch_preserves_armed_progress_through_real_validators(self) -> None:
        import hashlib
        import tempfile
        from pathlib import Path

        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("locked training image provides torch for runner dispatch coverage")

        from gods_mlops.jobs.checkpoints import CheckpointIdentity
        from gods_mlops.jobs.models import (
            EvaluationProbeCheckpointSource,
            ExecutionProfile,
        )
        from gods_mlops.jobs.queue import _normalize_evaluation_progress
        from gods_mlops.training.claims import (
            WorkerAuthorizationError,
            WorkerClaim,
            validate_worker_claim,
        )
        from gods_mlops.training.contracts import (
            locked_model,
            validate_manifest_identity,
        )
        from gods_mlops.training.probe_setup import (
            candidate_profile,
            create_probe_input,
        )

        probe = importlib.import_module("gods_mlops.training.probe")
        clip = importlib.import_module("gods_mlops.training.clip")
        training_data = importlib.import_module("gods_mlops.training.data")
        runner_support = importlib.import_module("gods_mlops.training.runner_support")
        worker = importlib.import_module("gods_mlops.training.worker")
        transformers = importlib.import_module("transformers")

        class MemoryObjects:
            def __init__(self) -> None:
                self.values: dict[str, bytes] = {}

            def write_immutable(self, *, object_key: str, content: bytes, sha256_digest: str, content_type: str) -> None:
                if hashlib.sha256(content).hexdigest() != sha256_digest:
                    raise AssertionError("probe fixture digest does not match its bytes")
                previous = self.values.get(object_key)
                if previous is not None and previous != content:
                    raise AssertionError("an immutable probe fixture was rewritten")
                self.values[object_key] = content

            def read_source(self, *, object_key: str, sha256_digest: str, size_bytes: int) -> bytes:
                payload = self.values[object_key]
                if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != sha256_digest:
                    raise AssertionError("probe fixture read did not match its immutable identity")
                return payload

        model_lock = locked_model("clip")
        training_profile = candidate_profile("clip", target_phase="training")
        training_job_id = "a0320b59-663c-4cdc-b893-086bb970ea60"
        source_sha = "b" * 64
        training_identity = CheckpointIdentity(
            job_id=training_job_id,
            input_kind="probe_input",
            input_id="task8-clip-synthetic-probe-v1",
            input_sha256="1" * 64,
            phase="probe",
            model_kind="clip",
            config_version=training_profile.config_version,
            config_sha256=training_profile.config_sha256,
            dataset_version=None,
        )
        source = EvaluationProbeCheckpointSource(
            training_probe_job_id=training_job_id,
            model_kind="clip",
            model_id=model_lock.model_id,
            model_revision=model_lock.revision,
            checkpoint_uri=(
                f"s3://gods-task8-test/jobs/{training_job_id}/checkpoints/"
                + "2" * 64
                + f"/{source_sha}.checkpoint"
            ),
            checkpoint_sha256=source_sha,
            checkpoint_size_bytes=123,
            checkpoint_identity=training_identity.as_dict(),
            worker_image_id="sha256:" + "3" * 64,
            source_commit="4" * 40,
            runtime_evidence_sha256="5" * 64,
        )
        objects = MemoryObjects()
        probe_input = create_probe_input(
            objects=objects,
            model_kind="clip",
            target_phase="evaluation",
            evaluation_checkpoint_source=source,
        )
        profile = candidate_profile("clip", target_phase="evaluation")
        profile_row = {
            "phase": "probe",
            "target_phase": "evaluation",
            "model_kind": "clip",
            "config_version": profile.config_version,
            "config_sha256": profile.config_sha256,
            "config_json": profile.config,
            "profile_state": "candidate",
        }
        job_id = "e4d82c26-8f0a-4f45-8b88-5fe84302d948"
        lease_token = "8ad96890-3434-4f07-85bb-8cde17a2b009"
        job = {
            "job_id": job_id,
            "state": "running",
            "lease_token": lease_token,
            "lease_generation": 1,
            "phase": "probe",
            "target_phase": "evaluation",
            "input_kind": "probe_input",
            "input_id": probe_input.probe_input_id,
            "input_sha256": probe_input.input_sha256,
            "dataset_version": None,
            "model_kind": "clip",
            "config_version": profile.config_version,
            "config_sha256": profile.config_sha256,
            "source_refs": probe_input.as_dict(),
        }
        lease = {
            "job_id": job_id,
            "lease_token": lease_token,
            "fencing_token": 1,
            "gpu_uuid": "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e",
        }
        claim = WorkerClaim.from_admitted_job(job, lease, image_id="sha256:" + "6" * 64)
        validate_worker_claim(claim, job=job, lease=lease, profile=profile_row)

        unknown_config = {**profile.config, "unknown_profile_option": True}
        unknown_profile = ExecutionProfile(
            model_kind=profile.model_kind,
            config_version=profile.config_version,
            phase=profile.phase,
            target_phase=profile.target_phase,
            memory_requirement_mib=profile.memory_requirement_mib,
            artifact_reservation_bytes=profile.artifact_reservation_bytes,
            config=unknown_config,
            candidate=True,
        )
        with self.assertRaisesRegex(WorkerAuthorizationError, "config hash"):
            validate_worker_claim(
                claim,
                job=job,
                lease=lease,
                profile={**profile_row, "config_json": unknown_profile.config, "config_sha256": unknown_profile.config_sha256},
            )

        eval_identity = CheckpointIdentity(
            job_id=job_id,
            input_kind="probe_input",
            input_id=probe_input.probe_input_id,
            input_sha256=probe_input.input_sha256,
            phase="probe",
            model_kind="clip",
            config_version=profile.config_version,
            config_sha256=profile.config_sha256,
            dataset_version=None,
        )
        config = {
            **profile.config,
            **claim.as_dict(),
            "manifest_size_bytes": probe_input.object_size_bytes,
            "_evaluation_probe_checkpoint_source": source.as_dict(),
            "_evaluation_checkpoint_sha256": source.checkpoint_sha256,
            "_evaluation_job_identity": eval_identity.as_dict(),
        }
        config_sha256 = config["config_sha256"]
        self.assertEqual(config_sha256, profile.config_sha256)
        self.assertNotIn("_evaluation_batch_completed", profile.config)

        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            manifest_path = root_path / "manifest.json"
            manifest_path.write_bytes(objects.values[probe_input.manifest_object_key])
            manifest_uri = str(manifest_path)
            output_uri = str(root_path / "output")

            class Processor:
                def __call__(self, *, text=None, images=None, return_tensors, padding=True, truncation=False):
                    count = len(text if text is not None else images)
                    features = torch.eye(count, 2, dtype=torch.float32)
                    return {"input_ids" if text is not None else "pixel_values": features}

            class Model:
                def to(self, _device):
                    return self

                def eval(self):
                    return self

                def get_text_features(self, *, input_ids):
                    return input_ids

                def get_image_features(self, *, pixel_values):
                    return pixel_values

            model_loads: list[Model] = []
            validator_observations: list[tuple[object, str]] = []

            def load_model(*_args, **_kwargs):
                model = Model()
                model_loads.append(model)
                return model

            def observe_manifest(config_value, manifest, *, probe_manifest_pin=None):
                callback_value = config_value.get("_evaluation_batch_completed")
                self.assertEqual(config_value.get("config_sha256"), config_sha256)
                if callback_value is not None:
                    self.assertTrue(callable(callback_value))
                validator_observations.append((callback_value, config_value["config_sha256"]))
                return validate_manifest_identity(
                    config_value,
                    manifest,
                    probe_manifest_pin=probe_manifest_pin,
                )

            applied_sources = []

            def apply_source(model, config_value, *, model_kind, model_revision):
                self.assertEqual(config_value.get("_evaluation_probe_checkpoint_source"), source.as_dict())
                self.assertEqual(config_value.get("config_sha256"), config_sha256)
                self.assertEqual((model_kind, model_revision), ("clip", model_lock.revision))
                applied_sources.append(source.checkpoint_sha256)

            def run_with(callback=None, *, assert_current=lambda: True):
                runtime_config = dict(config)
                runtime_config["_assert_current"] = assert_current
                if callback is None:
                    runtime_config.pop("_evaluation_batch_completed", None)
                else:
                    runtime_config["_evaluation_batch_completed"] = callback
                return probe.run(runtime_config, manifest_uri, output_uri)

            with (
                patch.object(probe, "validate_manifest_identity", side_effect=observe_manifest),
                patch.object(clip, "validate_manifest_identity", side_effect=observe_manifest),
                patch.object(training_data, "dataset_object_store_from_environment", return_value=objects),
                patch.object(runner_support, "cache_directory", return_value=root_path),
                patch.object(runner_support, "require_cuda", return_value=torch.device("cpu")),
                patch.object(worker, "apply_verified_evaluation_probe_checkpoint_weights", side_effect=apply_source),
                patch.object(transformers.CLIPModel, "from_pretrained", side_effect=load_model),
                patch.object(transformers.CLIPProcessor, "from_pretrained", return_value=Processor()),
                patch.object(torch.cuda, "reset_peak_memory_stats", return_value=None),
                patch.object(torch.cuda, "synchronize", return_value=None),
                patch.object(torch.cuda, "max_memory_allocated", return_value=0),
                patch.object(torch.cuda, "max_memory_reserved", return_value=0),
            ):
                baseline = run_with()
                noop_events = []
                def noop_callback(**event):
                    noop_events.append(event)

                unarmed_with_callback = run_with(noop_callback)
                self.assertEqual(baseline["status"], "succeeded")
                self.assertEqual(
                    baseline["result_artifact_payload"],
                    unarmed_with_callback["result_artifact_payload"],
                )
                self.assertEqual(len(noop_events), 2)

                progress_events = []
                guard_calls = 0

                def yield_after_query() -> bool:
                    nonlocal guard_calls
                    guard_calls += 1
                    return guard_calls == 1

                def armed_callback(**event):
                    progress_events.append(_normalize_evaluation_progress(event))

                armed_result = run_with(
                    armed_callback,
                    assert_current=yield_after_query,
                )
                self.assertEqual(armed_result["status"], "yielded")
                self.assertEqual([event["stage"] for event in progress_events], ["queries"])
                cursor = _require("gods_mlops.evaluation.report", "decode_evaluation_probe_cursor")(
                    armed_result["checkpoint_payload"],
                    expected_identity=eval_identity.as_dict(),
                    expected_candidate_checkpoint_sha256=source.checkpoint_sha256,
                    expected_manifest_sha256=probe_input.input_sha256,
                    expected_config_sha256=profile.config_sha256,
                )
                self.assertEqual(cursor["predictions"]["stage"], "gallery")
                self.assertEqual(cursor["next_index"], 0)
                self.assertEqual(cursor["predictions"]["crop_embeddings"], {})
                self.assertEqual(progress_events[0]["model_kind"], "clip")
                self.assertEqual(set(cursor["predictions"]["query_embeddings"]), {"task8-crop-red", "task8-crop-blue"})
                self.assertEqual(progress_events[0]["config_sha256"], profile.config_sha256)

                before_bad_manifest_models = len(model_loads)
                bad_manifest_config = dict(config)
                bad_manifest_config["_evaluation_batch_completed"] = lambda **_event: None
                bad_manifest_config["_assert_current"] = lambda: True
                bad_manifest_config["input_sha256"] = "f" * 64
                with self.assertRaisesRegex(ValueError, "manifest content hash"):
                    probe.run(bad_manifest_config, manifest_uri, str(root_path / "tampered"))
                self.assertEqual(len(model_loads), before_bad_manifest_models)

            self.assertTrue(validator_observations)
            self.assertTrue(all(observed_hash == profile.config_sha256 for _, observed_hash in validator_observations))
            self.assertEqual(
                sum(callback_value is noop_callback for callback_value, _ in validator_observations),
                2,
            )
            self.assertEqual(
                sum(callback_value is armed_callback for callback_value, _ in validator_observations),
                2,
            )
            self.assertEqual(len(applied_sources), 3)
            self.assertEqual(applied_sources, [source.checkpoint_sha256] * 3)
            self.assertEqual(len(model_loads), 3)


if __name__ == "__main__":
    unittest.main()
