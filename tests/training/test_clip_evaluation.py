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


if __name__ == "__main__":
    unittest.main()
