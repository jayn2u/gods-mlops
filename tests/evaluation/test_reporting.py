from __future__ import annotations

import importlib
import io
import json
import unittest
from unittest.mock import AsyncMock, patch


def _require(module_name: str, symbol: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise AssertionError(f"missing evaluation implementation module {module_name}") from error
    value = getattr(module, symbol, None)
    if value is None:
        raise AssertionError(f"{module_name}.{symbol} is part of the evaluation contract")
    return value


def _request_config() -> dict:
    return {
        "dataset_version": "dataset-2026-10-01-a1b2c3d4",
        "manifest_sha256": "a" * 64,
        "training_job_id": "a0320b59-663c-4cdc-b893-086bb970ea60",
        "checkpoint_sha256": "b" * 64,
        "model_kind": "clip",
        "evaluation_config_version": "clip-retrieval-eval-v1",
        "evaluation_split": "test",
        "baseline": {"model_id": "operator-supplied-baseline-alias", "revision": "external"},
    }


class EvaluationRequestTests(unittest.TestCase):
    def test_request_binds_manifest_checkpoint_and_profile_without_accepting_predictions(self) -> None:
        validate = _require("gods_mlops.evaluation.eligibility", "validate_evaluation_request")
        normalized = validate(
            "s3://gods-mlops/datasets/dataset-2026-10-01-a1b2c3d4/manifest.json",
            "s3://gods-mlops/jobs/a0320b59-663c-4cdc-b893-086bb970ea60/checkpoints/identity/" + "b" * 64 + ".checkpoint",
            _request_config(),
        )

        self.assertEqual(normalized["dataset_version"], "dataset-2026-10-01-a1b2c3d4")
        self.assertEqual(normalized["manifest_sha256"], "a" * 64)
        self.assertEqual(normalized["checkpoint_sha256"], "b" * 64)
        self.assertEqual(normalized["training_job_id"], "a0320b59-663c-4cdc-b893-086bb970ea60")
        self.assertEqual(normalized["baseline"]["model_id"], "operator-supplied-baseline-alias")
        self.assertFalse(normalized["baseline"]["verified"])

        with self.assertRaisesRegex(ValueError, "unsupported"):
            validate(
                "s3://gods-mlops/datasets/dataset-2026-10-01-a1b2c3d4/manifest.json",
                "s3://gods-mlops/jobs/a0320b59-663c-4cdc-b893-086bb970ea60/checkpoints/identity/" + "b" * 64 + ".checkpoint",
                {**_request_config(), "predictions": []},
            )

    def test_request_rejects_unbound_or_non_s3_checkpoint_and_manifest_uris(self) -> None:
        validate = _require("gods_mlops.evaluation.eligibility", "validate_evaluation_request")
        with self.assertRaisesRegex(ValueError, "S3"):
            validate(
                "file:///tmp/manifest.json",
                "s3://gods-mlops/checkpoint",
                _request_config(),
            )
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate(
                "s3://gods-mlops/datasets/dataset-2026-10-01-a1b2c3d4/manifest.json",
                "s3://gods-mlops/jobs/not-the-training-job/checkpoint",
                _request_config(),
            )

    def test_public_evaluate_only_returns_the_owned_worker_result_and_accepts_no_injected_runner(self) -> None:
        evaluate = _require("gods_mlops.evaluation.report", "evaluate")
        _require("gods_mlops.evaluation.report", "_evaluate_with_task7_owned_worker")
        manifest_uri = "s3://gods-mlops/datasets/dataset-2026-10-01-a1b2c3d4/manifest.json"
        checkpoint_uri = (
            "s3://gods-mlops/jobs/a0320b59-663c-4cdc-b893-086bb970ea60/checkpoints/"
            + "d" * 64
            + "/"
            + "b" * 64
            + ".checkpoint"
        )
        owned_result = {"status": "insufficient", "artifact": None, "insufficient_reasons": ["evaluation_profile_not_measured"]}
        with patch(
            "gods_mlops.evaluation.report._evaluate_with_task7_owned_worker",
            new_callable=AsyncMock,
            return_value=owned_result,
        ) as run_owned:
            result = evaluate(manifest_uri, checkpoint_uri, _request_config())

        self.assertEqual(result, owned_result)
        run_owned.assert_awaited_once()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            evaluate(manifest_uri, checkpoint_uri, {**_request_config(), "run_predictions": lambda: []})

    def test_evaluation_cursor_resumes_its_own_position_and_rejects_training_state(self) -> None:
        encode = _require("gods_mlops.evaluation.report", "encode_evaluation_cursor")
        decode = _require("gods_mlops.evaluation.report", "decode_evaluation_cursor")
        identity = {
            "job_id": "e4d82c26-8f0a-4f45-8b88-5fe84302d948",
            "input_kind": "dataset_version",
            "input_id": "dataset-2026-10-01-a1b2c3d4",
            "input_sha256": "a" * 64,
            "phase": "evaluation",
            "model_kind": "detr",
            "config_version": "detr-evaluation-v1",
            "config_sha256": "c" * 64,
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
        }
        payload = encode(
            identity=identity,
            candidate_checkpoint_sha256="b" * 64,
            manifest_sha256="a" * 64,
            config_sha256="c" * 64,
            next_index=1,
            predictions=[{"item_id": "frame-a", "detections": []}],
        )

        restored = decode(
            payload,
            expected_identity=identity,
            expected_candidate_checkpoint_sha256="b" * 64,
            expected_manifest_sha256="a" * 64,
            expected_config_sha256="c" * 64,
        )
        self.assertEqual(restored["next_index"], 1)
        self.assertEqual(restored["predictions"][0]["item_id"], "frame-a")
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            decode(
                payload,
                expected_identity=identity,
                expected_candidate_checkpoint_sha256="d" * 64,
                expected_manifest_sha256="a" * 64,
                expected_config_sha256="c" * 64,
            )
        training_payload = json.dumps(
            {"format": "gods-mlops-training-checkpoint-v1", "optimizer_state_dict": {}}
        ).encode()
        with self.assertRaisesRegex(ValueError, "cursor"):
            decode(
                training_payload,
                expected_identity=identity,
                expected_candidate_checkpoint_sha256="b" * 64,
                expected_manifest_sha256="a" * 64,
                expected_config_sha256="c" * 64,
            )


class EvaluationEligibilityTests(unittest.TestCase):
    def test_training_readiness_and_evaluation_minimums_remain_independent(self) -> None:
        readiness = _require("gods_mlops.evaluation.eligibility", "evaluation_readiness")
        manifest = {
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
            "target": "detr",
            "training": {"ready": True, "reason_codes": []},
            "evaluation": {"eligible": True, "reason_codes": [], "relevance_matrices": []},
            "split_counts": {
                "validation": {"groups": 2, "frames": 20, "positive_frames": 10, "negative_frames": 5},
                "test": {"groups": 2, "frames": 20, "positive_frames": 9, "negative_frames": 5},
            },
        }
        current = {
            "dataset_version": manifest["dataset_version"],
            "training_eligible": True,
            "evaluation_eligible": True,
            "impacts": [],
        }

        result = readiness(manifest, current, model_kind="detr", evaluation_split="test")

        self.assertTrue(result["training_ready"])
        self.assertEqual(result["training_reasons"], [])
        self.assertFalse(result["evaluation_eligible"])
        self.assertIn("insufficient_test_detr_positive_frames", result["evaluation_reasons"])

    def test_late_leakage_overlay_blocks_current_evaluation_but_keeps_training_ready(self) -> None:
        readiness = _require("gods_mlops.evaluation.eligibility", "evaluation_readiness")
        manifest = {
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
            "target": "clip",
            "training": {"ready": True, "reason_codes": []},
            "evaluation": {"eligible": True, "reason_codes": [], "relevance_matrices": []},
            "split_counts": {
                "validation": {"groups": 2, "crop_caption_pairs": 20},
                "test": {"groups": 2, "crop_caption_pairs": 20},
            },
        }
        current = {
            "dataset_version": manifest["dataset_version"],
            "training_eligible": True,
            "evaluation_eligible": False,
            "impacts": [{"reason": "evaluation_split_leakage"}],
        }

        result = readiness(manifest, current, model_kind="clip", evaluation_split="test")

        self.assertTrue(result["training_ready"])
        self.assertFalse(result["evaluation_eligible"])
        self.assertIn("evaluation_split_leakage", result["evaluation_reasons"])

    def test_training_not_ready_does_not_rewrite_a_true_evaluation_flag(self) -> None:
        readiness = _require("gods_mlops.evaluation.eligibility", "evaluation_readiness")
        manifest = {
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
            "target": "clip",
            "training": {"ready": False, "reason_codes": ["insufficient_train_clip_pairs"]},
            "evaluation": {
                "eligible": True,
                "reason_codes": [],
                "gallery_crop_ids": ["crop-a", "crop-b"],
                "relevance_matrices": [{"status": "complete", "evaluation_eligible": True}],
            },
            "split_counts": {"test": {"groups": 1, "crop_caption_pairs": 20}},
        }
        current = {
            "dataset_version": manifest["dataset_version"],
            "training_eligible": False,
            "evaluation_eligible": True,
            "impacts": [],
        }

        result = readiness(manifest, current, model_kind="clip", evaluation_split="test")

        self.assertFalse(result["training_ready"])
        self.assertTrue(result["evaluation_eligible"])
        self.assertIn("insufficient_train_clip_pairs", result["training_reasons"])
        self.assertEqual(result["evaluation_reasons"], [])

    def test_small_internal_artifact_stays_blocked_without_watching_cuhk_and_product_evidence(self) -> None:
        deployment = _require("gods_mlops.evaluation.eligibility", "deployment_eligibility")
        result = deployment(
            {
                "status": "complete",
                "model_kind": "clip",
                "candidate": {"checkpoint_sha256": "b" * 64, "verified": True},
                "baseline": None,
                "metrics": {"hit_rate@5": 0.9, "map": 0.8},
            }
        )

        self.assertFalse(result["eligible"])
        self.assertIn("baseline_evidence_missing", result["reasons"])
        self.assertIn("cuhk_report_missing", result["reasons"])
        self.assertIn("product_retrieval_evidence_missing", result["reasons"])

    def test_inference_weight_loader_validates_prior_training_identity_without_optimizer_restore(self) -> None:
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("locked training image provides torch for checkpoint contract coverage")
        apply_weights = _require("gods_mlops.evaluation.eligibility", "apply_verified_model_weights")
        model = torch.nn.Linear(2, 1, bias=False)
        checkpoint_model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            checkpoint_model.weight.copy_(torch.tensor([[3.0, 4.0]]))
        identity = {
            "job_id": "a0320b59-663c-4cdc-b893-086bb970ea60",
            "input_kind": "dataset_version",
            "input_id": "dataset-2026-10-01-a1b2c3d4",
            "input_sha256": "a" * 64,
            "phase": "training",
            "model_kind": "clip",
            "config_version": "clip-training-v1",
            "config_sha256": "c" * 64,
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
        }
        checkpoint = io.BytesIO()
        torch.save(
            {
                "format": "gods-mlops-training-checkpoint-v1",
                "identity": identity,
                "model_revision": "locked-clip-revision",
                "model_state_dict": checkpoint_model.state_dict(),
            },
            checkpoint,
        )
        apply_weights(
            model,
            {
                "_evaluation_checkpoint_payload": checkpoint.getvalue(),
                "_evaluation_checkpoint_identity": identity,
            },
            model_kind="clip",
            model_revision="locked-clip-revision",
        )

        self.assertTrue(torch.equal(model.weight, torch.tensor([[3.0, 4.0]])))
        with self.assertRaisesRegex(ValueError, "identity"):
            apply_weights(
                model,
                {
                    "_evaluation_checkpoint_payload": checkpoint.getvalue(),
                    "_evaluation_checkpoint_identity": {**identity, "config_sha256": "e" * 64},
                },
                model_kind="clip",
                model_revision="locked-clip-revision",
            )


class EvaluationReportTests(unittest.TestCase):
    def test_report_preserves_candidate_source_config_overlay_and_insufficient_reasons(self) -> None:
        build = _require("gods_mlops.evaluation.report", "build_evaluation_report")
        report = build(
            candidate={
                "training_job_id": "a0320b59-663c-4cdc-b893-086bb970ea60",
                "checkpoint_uri": "s3://gods-mlops/jobs/train/checkpoint",
                "checkpoint_sha256": "b" * 64,
                "checkpoint_identity": {"phase": "training", "model_kind": "clip"},
                "model_id": "sha256:" + "b" * 64,
            },
            baseline={"model_id": "operator-supplied-baseline-alias", "verified": False},
            source={
                "dataset_version": "dataset-2026-10-01-a1b2c3d4",
                "manifest_uri": "s3://gods-mlops/datasets/dataset-2026-10-01-a1b2c3d4/manifest.json",
                "manifest_sha256": "a" * 64,
                "target": "clip",
            },
            evaluation_config={
                "version": "clip-retrieval-eval-v1",
                "sha256": "c" * 64,
                "split": "test",
            },
            current_eligibility={
                "training_eligible": True,
                "evaluation_eligible": False,
                "evaluation_reasons": ["late_cross_boundary_link"],
                "impacts": [{"reason": "evaluation_split_leakage"}],
            },
            metrics={
                "status": "insufficient",
                "metrics": {"hit_rate@1": None, "hit_rate@5": None, "hit_rate@10": None, "map": None},
                "counts": {"queries": 1, "gallery_crops": 2, "relevant_judgments": 0},
                "reasons": ["human_relevance_truth_unresolved"],
            },
            predictions={"query_embeddings": {"query-1": [1.0, 0.0]}},
        )

        self.assertEqual(report["status"], "insufficient")
        self.assertTrue(report["training_ready"])
        self.assertFalse(report["evaluation_eligible"])
        self.assertEqual(report["candidate"]["checkpoint_sha256"], "b" * 64)
        self.assertEqual(report["source"]["manifest_sha256"], "a" * 64)
        self.assertEqual(report["evaluation_config"]["sha256"], "c" * 64)
        self.assertIn("evaluation_split_leakage", report["insufficient_reasons"])
        self.assertIn("human_relevance_truth_unresolved", report["insufficient_reasons"])
        self.assertEqual(report["sample_counts"]["queries"], 1)
        self.assertIn("predictions", report)

    def test_internal_detection_run_can_succeed_while_test_minimums_remain_insufficient(self) -> None:
        create_report = _require("gods_mlops.evaluation.report", "evaluate_prediction_payload")
        manifest = {
            "schema_version": 1,
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
            "target": "detr",
            "training": {"ready": True, "reason_codes": []},
            "evaluation": {"eligible": True, "reason_codes": [], "gallery_crop_ids": [], "relevance_matrices": []},
            "split_counts": {
                "test": {"groups": 1, "frames": 1, "positive_frames": 1, "negative_frames": 0, "crop_caption_pairs": 0}
            },
            "items": [
                {
                    "kind": "frame",
                    "item_kind": "frame",
                    "item_id": "frame-a",
                    "split": "test",
                    "snapshot": {
                        "bbox_annotation": {
                            "result": [
                                {
                                    "from_name": "bbox",
                                    "type": "rectanglelabels",
                                    "original_width": 100,
                                    "original_height": 80,
                                    "value": {
                                        "x": 10,
                                        "y": 10,
                                        "width": 20,
                                        "height": 25,
                                        "rectanglelabels": ["person"],
                                    },
                                }
                            ]
                        }
                    },
                }
            ],
        }
        from gods_mlops.datasets.manifest import canonical_json
        from hashlib import sha256

        manifest_sha = sha256(canonical_json(manifest)).hexdigest()
        class_mapping = {
            "model_revision": "locked-detr-revision",
            "model_class_index": 1,
            "model_class_name": "person",
            "coco_category_id": 1,
            "coco_category_name": "person",
        }
        report = create_report(
            manifest=manifest,
            prediction_payload={
                "model_kind": "detr",
                "model_revision": "locked-detr-revision",
                "person_class_mapping": class_mapping,
                "input_sha256": manifest_sha,
                "drafts": [
                    {
                        "item_id": "frame-a",
                        "image_width": 100,
                        "image_height": 80,
                        "detections": [{"score": 0.9, "bbox_xyxy": [10.0, 8.0, 30.0, 28.0]}],
                    }
                ],
            },
            model_kind="detr",
            evaluation_split="test",
            candidate={
                "checkpoint_sha256": "b" * 64,
                "model_id": "sha256:" + "b" * 64,
                "model_revision": "locked-detr-revision",
            },
            baseline=None,
            source={
                "dataset_version": manifest["dataset_version"],
                "manifest_uri": "s3://gods-mlops/datasets/dataset-2026-10-01-a1b2c3d4/manifest.json",
                "manifest_sha256": manifest_sha,
                "target": "detr",
            },
            evaluation_config={
                "version": "detr-eval-v1",
                "sha256": "c" * 64,
                "settings": {"score_threshold": 0.5, "max_detections": 100},
            },
            current_eligibility={"training_eligible": True, "evaluation_eligible": True, "impacts": []},
            evaluator_provenance={
                "evaluator_revision": "sha256:" + "1" * 64,
                "worker_image_id": "sha256:" + "2" * 64,
                "model_revision": "locked-detr-revision",
                "person_class_mapping": class_mapping,
            },
        )

        self.assertEqual(report["execution_status"], "succeeded")
        self.assertEqual(report["status"], "insufficient")
        self.assertAlmostEqual(report["metrics"]["ap_50_95"], 1.0)
        self.assertIn("insufficient_test_detr_frames", report["insufficient_reasons"])

    def test_clip_prediction_report_uses_complete_human_matrix_over_the_frozen_gallery(self) -> None:
        create_report = _require("gods_mlops.evaluation.report", "evaluate_prediction_payload")
        gallery_ids = [f"crop-{index:02d}" for index in range(20)]
        manifest = {
            "schema_version": 1,
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
            "target": "clip",
            "training": {"ready": True, "reason_codes": []},
            "evaluation": {
                "eligible": True,
                "reason_codes": [],
                "gallery_crop_ids": gallery_ids,
                "relevance_matrices": [
                    {
                        "query_id": "query-1",
                        "query_text": "frozen human reviewed description",
                        "query_sha256": "f" * 64,
                        "status": "complete",
                        "evaluation_eligible": True,
                        "gallery_matches_selection": True,
                        "judgments": [
                            {
                                "crop_id": crop_id,
                                "sha256": "d" * 64,
                                "judgment": "relevant" if crop_id in {"crop-00", "crop-01"} else "not_relevant",
                            }
                            for crop_id in gallery_ids
                        ],
                    }
                ],
            },
            "split_counts": {"test": {"groups": 3, "frames": 0, "positive_frames": 0, "negative_frames": 0, "crop_caption_pairs": 20}},
            "items": [
                {"kind": "crop", "item_kind": "crop", "item_id": crop_id, "split": "test"}
                for crop_id in gallery_ids
            ],
        }
        from gods_mlops.datasets.manifest import canonical_json
        from hashlib import sha256

        manifest_sha = sha256(canonical_json(manifest)).hexdigest()
        report = create_report(
            manifest=manifest,
            prediction_payload={
                "model_kind": "clip",
                "model_revision": "locked-clip-revision",
                "input_sha256": manifest_sha,
                "query_embeddings": {"query-1": [1.0, 0.0]},
                "crop_embeddings": {
                    crop_id: ([1.0, 0.0] if crop_id == "crop-00" else [0.8, 0.6] if crop_id == "crop-01" else [0.0, 1.0])
                    for crop_id in gallery_ids
                },
            },
            model_kind="clip",
            evaluation_split="test",
            candidate={
                "checkpoint_sha256": "b" * 64,
                "model_id": "sha256:" + "b" * 64,
                "model_revision": "locked-clip-revision",
            },
            baseline=None,
            source={
                "dataset_version": manifest["dataset_version"],
                "manifest_uri": "s3://gods-mlops/datasets/dataset-2026-10-01-a1b2c3d4/manifest.json",
                "manifest_sha256": manifest_sha,
                "target": "clip",
            },
            evaluation_config={"version": "clip-eval-v1", "sha256": "c" * 64, "split": "test"},
            current_eligibility={"training_eligible": True, "evaluation_eligible": True, "impacts": []},
            evaluator_provenance={
                "evaluator_revision": "sha256:" + "1" * 64,
                "worker_image_id": "sha256:" + "2" * 64,
                "model_revision": "locked-clip-revision",
            },
        )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["metrics"]["hit_rate@1"], 1.0)
        self.assertEqual(report["metrics"]["map"], 1.0)
        self.assertEqual(report["sample_counts"]["gallery_crops"], 20)

    def test_worker_profile_settings_drive_detr_metrics_and_report_protocol(self) -> None:
        create_report = _require("gods_mlops.evaluation.report", "evaluate_prediction_payload")
        manifest = {
            "schema_version": 1,
            "dataset_version": "dataset-2026-10-01-a1b2c3d4",
            "target": "detr",
            "training": {"ready": True, "reason_codes": []},
            "evaluation": {"eligible": True, "reason_codes": []},
            "split_counts": {
                "test": {"groups": 3, "frames": 20, "positive_frames": 10, "negative_frames": 5}
            },
            "items": [],
        }
        from gods_mlops.datasets.manifest import canonical_json
        from hashlib import sha256

        manifest_sha = sha256(canonical_json(manifest)).hexdigest()
        class_mapping = {
            "model_revision": "locked-detr-revision",
            "model_class_index": 17,
            "model_class_name": "person",
            "coco_category_id": 1,
            "coco_category_name": "person",
        }
        provenance = {
            "evaluator_revision": "sha256:" + "1" * 64,
            "worker_image_id": "sha256:" + "2" * 64,
            "model_revision": "locked-detr-revision",
            "person_class_mapping": class_mapping,
        }
        with patch(
            "gods_mlops.evaluation.report.evaluate_detections",
            return_value={
                "status": "complete",
                "metrics": {"ap_50_95": 0.5, "ap_50": 0.8, "ar": 0.6},
                "counts": {"frames": 20, "person_ground_truth": 10},
                "settings": {"score_threshold": 0.8, "max_detections": 200},
                "reasons": [],
            },
        ) as metric:
            report = create_report(
                manifest=manifest,
                prediction_payload={
                    "model_kind": "detr",
                    "model_revision": "locked-detr-revision",
                    "person_class_mapping": class_mapping,
                    "input_sha256": manifest_sha,
                    "drafts": [],
                },
                model_kind="detr",
                evaluation_split="test",
                candidate={"model_kind": "detr", "model_revision": "locked-detr-revision", "checkpoint_sha256": "b" * 64},
                baseline=None,
                source={"dataset_version": manifest["dataset_version"], "manifest_sha256": manifest_sha},
                evaluation_config={
                    "version": "detr-eval-v2",
                    "sha256": "c" * 64,
                    "split": "test",
                    "settings": {"score_threshold": 0.8, "max_detections": 200},
                },
                current_eligibility={"training_eligible": True, "evaluation_eligible": True, "impacts": []},
                evaluator_provenance=provenance,
            )

        self.assertEqual(metric.call_args.kwargs["score_threshold"], 0.8)
        self.assertEqual(metric.call_args.kwargs["max_detections"], 200)
        self.assertEqual(report["evaluation_config"]["score_threshold"], 0.8)
        self.assertEqual(report["evaluation_config"]["max_detections"], 200)
        self.assertEqual(report["metric_settings"]["score_threshold"], 0.8)
        self.assertEqual(report["metric_settings"]["max_detections"], 200)
        self.assertEqual(report["evaluation_provenance"]["person_class_mapping"]["model_class_index"], 17)

    def test_owned_readback_rejects_missing_or_tampered_evaluator_provenance(self) -> None:
        validate = _require("gods_mlops.evaluation.report", "validate_owned_evaluation_provenance")
        class_mapping = {
            "model_revision": "locked-detr-revision",
            "model_class_index": 17,
            "model_class_name": "person",
            "coco_category_id": 1,
            "coco_category_name": "person",
        }
        report = {
            "model_kind": "detr",
            "candidate": {"model_revision": "locked-detr-revision"},
            "predictions": {
                "model_revision": "locked-detr-revision",
                "person_class_mapping": class_mapping,
            },
            "evaluation_provenance": {
                "evaluator_revision": "sha256:" + "1" * 64,
                "worker_image_id": "sha256:" + "2" * 64,
                "model_revision": "locked-detr-revision",
                "person_class_mapping": class_mapping,
            },
        }

        provenance = validate(
            report,
            expected_evaluator_revision="sha256:" + "1" * 64,
            expected_worker_image_id="sha256:" + "2" * 64,
            expected_model_revision="locked-detr-revision",
        )

        self.assertEqual(provenance["person_class_mapping"]["model_class_index"], 17)
        for bad in (
            {"model_kind": "detr"},
            {
                **report,
                "evaluation_provenance": {
                    **report["evaluation_provenance"],
                    "worker_image_id": "sha256:" + "3" * 64,
                },
            },
            {
                **report,
                "evaluation_provenance": {
                    **report["evaluation_provenance"],
                    "person_class_mapping": {**class_mapping, "coco_category_id": 2},
                },
            },
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate(
                    bad,
                    expected_evaluator_revision="sha256:" + "1" * 64,
                    expected_worker_image_id="sha256:" + "2" * 64,
                    expected_model_revision="locked-detr-revision",
                )


if __name__ == "__main__":
    unittest.main()
