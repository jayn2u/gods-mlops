from __future__ import annotations

import importlib
import unittest


def _require(module_name: str, symbol: str):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise AssertionError(f"missing evaluation implementation module {module_name}") from error
    value = getattr(module, symbol, None)
    if value is None:
        raise AssertionError(f"{module_name}.{symbol} is part of the evaluation contract")
    return value


class DetectionMetricsTests(unittest.TestCase):
    def test_manifest_person_boxes_are_scaled_to_original_frame_coordinates(self) -> None:
        extract = _require("gods_mlops.evaluation.detection", "manifest_person_boxes_xywh")
        item = {
            "snapshot": {
                "bbox_annotation": {
                    "result": [
                        {
                            "from_name": "bbox",
                            "type": "rectanglelabels",
                            "original_width": 200,
                            "original_height": 160,
                            "value": {
                                "x": 10,
                                "y": 5,
                                "width": 20,
                                "height": 25,
                                "rectanglelabels": ["person"],
                            },
                        },
                        {
                            "from_name": "bbox",
                            "type": "rectanglelabels",
                            "original_width": 200,
                            "original_height": 160,
                            "value": {
                                "x": 50,
                                "y": 20,
                                "width": 10,
                                "height": 10,
                                "rectanglelabels": ["car"],
                            },
                        },
                    ]
                }
            }
        }

        boxes = extract(item, image_width=100, image_height=80)

        self.assertEqual(boxes, [[10.0, 4.0, 20.0, 20.0]])

    def test_person_coco_metrics_and_fixed_threshold_counts_include_empty_frames(self) -> None:
        evaluate_detections = _require("gods_mlops.evaluation.detection", "evaluate_detections")
        result = evaluate_detections(
            frames=[
                {
                    "item_id": "frame-a",
                    "width": 100,
                    "height": 80,
                    "person_boxes_xywh": [[10.0, 10.0, 20.0, 20.0]],
                },
                {
                    "item_id": "frame-b",
                    "width": 100,
                    "height": 80,
                    "person_boxes_xywh": [],
                },
            ],
            predictions=[
                {
                    "item_id": "frame-a",
                    "detections": [{"score": 0.9, "bbox_xyxy": [10.0, 10.0, 30.0, 30.0]}],
                },
                {
                    "item_id": "frame-b",
                    "detections": [{"score": 0.8, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}],
                },
            ],
            score_threshold=0.5,
            max_detections=100,
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["reasons"], [])
        self.assertAlmostEqual(result["metrics"]["ap_50_95"], 1.0)
        self.assertAlmostEqual(result["metrics"]["ap50"], 1.0)
        self.assertAlmostEqual(result["metrics"]["ar"], 1.0)
        self.assertEqual(result["counts"]["frames"], 2)
        self.assertEqual(result["counts"]["person_ground_truth"], 1)
        self.assertEqual(result["counts"]["false_positives"], 1)
        self.assertEqual(result["counts"]["false_negatives"], 0)

    def test_zero_detections_are_a_valid_zero_score_when_person_truth_exists(self) -> None:
        evaluate_detections = _require("gods_mlops.evaluation.detection", "evaluate_detections")
        result = evaluate_detections(
            frames=[
                {
                    "item_id": "frame-a",
                    "width": 40,
                    "height": 30,
                    "person_boxes_xywh": [[2.0, 3.0, 10.0, 12.0]],
                }
            ],
            predictions=[],
            score_threshold=0.5,
            max_detections=100,
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["metrics"], {"ap_50_95": 0.0, "ap50": 0.0, "ar": 0.0})
        self.assertEqual(result["counts"]["false_positives"], 0)
        self.assertEqual(result["counts"]["false_negatives"], 1)

    def test_no_person_ground_truth_is_insufficient_instead_of_a_zero_quality_score(self) -> None:
        evaluate_detections = _require("gods_mlops.evaluation.detection", "evaluate_detections")
        result = evaluate_detections(
            frames=[
                {
                    "item_id": "empty-frame",
                    "width": 40,
                    "height": 30,
                    "person_boxes_xywh": [],
                }
            ],
            predictions=[],
            score_threshold=0.5,
            max_detections=100,
        )

        self.assertEqual(result["status"], "insufficient")
        self.assertIn("no_person_ground_truth", result["reasons"])
        self.assertEqual(result["metrics"], {"ap_50_95": None, "ap50": None, "ar": None})

    def test_coco_ap_ignores_the_fixed_count_threshold_and_respects_max_detections(self) -> None:
        evaluate_detections = _require("gods_mlops.evaluation.detection", "evaluate_detections")
        threshold_result = evaluate_detections(
            frames=[
                {
                    "item_id": "frame-a",
                    "width": 40,
                    "height": 30,
                    "person_boxes_xywh": [[2.0, 3.0, 10.0, 12.0]],
                }
            ],
            predictions=[
                {"item_id": "frame-a", "detections": [{"score": 0.9, "bbox_xyxy": [2.0, 3.0, 12.0, 15.0]}]}
            ],
            score_threshold=0.95,
            max_detections=100,
        )
        self.assertAlmostEqual(threshold_result["metrics"]["ap_50_95"], 1.0)
        self.assertEqual(threshold_result["counts"]["false_negatives"], 1)

        detections = [
            {"score": 0.99, "bbox_xyxy": [float(index * 20 + 20), 0.0, float(index * 20 + 30), 10.0]}
            for index in range(100)
        ]
        detections.append({"score": 0.1, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]})
        capped_result = evaluate_detections(
            frames=[
                {
                    "item_id": "frame-a",
                    "width": 2500,
                    "height": 30,
                    "person_boxes_xywh": [[0.0, 0.0, 10.0, 10.0]],
                }
            ],
            predictions=[{"item_id": "frame-a", "detections": detections}],
            score_threshold=0.05,
            max_detections=100,
        )
        self.assertEqual(capped_result["metrics"], {"ap_50_95": 0.0, "ap50": 0.0, "ar": 0.0})
        self.assertEqual(capped_result["counts"]["false_positives"], 100)
        self.assertEqual(capped_result["counts"]["false_negatives"], 1)


class RetrievalMetricsTests(unittest.TestCase):
    def test_cosine_ties_use_crop_id_and_map_uses_all_human_relevant_crops(self) -> None:
        evaluate_retrieval = _require("gods_mlops.evaluation.retrieval", "evaluate_retrieval")
        result = evaluate_retrieval(
            query_embeddings={"query-1": [1.0, 0.0]},
            crop_embeddings={
                "crop-a-negative": [1.0, 0.0],
                "crop-b-positive": [1.0, 0.0],
                "crop-c-positive": [0.8, 0.6],
                "crop-d-negative": [0.0, 1.0],
            },
            gallery_crop_ids=["crop-d-negative", "crop-c-positive", "crop-b-positive", "crop-a-negative"],
            relevance_matrices=[
                {
                    "query_id": "query-1",
                    "status": "complete",
                    "evaluation_eligible": True,
                    "gallery_matches_selection": True,
                    "judgments": [
                        {"crop_id": "crop-a-negative", "judgment": "not_relevant"},
                        {"crop_id": "crop-b-positive", "judgment": "relevant"},
                        {"crop_id": "crop-c-positive", "judgment": "relevant"},
                        {"crop_id": "crop-d-negative", "judgment": "not_relevant"},
                    ],
                }
            ],
        )

        self.assertEqual(result["status"], "complete")
        self.assertAlmostEqual(result["metrics"]["hit_rate@1"], 0.0)
        self.assertAlmostEqual(result["metrics"]["hit_rate@5"], 1.0)
        self.assertAlmostEqual(result["metrics"]["hit_rate@10"], 1.0)
        self.assertAlmostEqual(result["metrics"]["map"], 7.0 / 12.0)
        self.assertEqual(result["counts"]["queries"], 1)
        self.assertEqual(result["counts"]["relevant_judgments"], 2)

    def test_missing_or_uncertain_relevance_truth_makes_retrieval_insufficient(self) -> None:
        evaluate_retrieval = _require("gods_mlops.evaluation.retrieval", "evaluate_retrieval")
        common = {
            "query_embeddings": {"query-1": [1.0, 0.0]},
            "crop_embeddings": {"crop-a": [1.0, 0.0], "crop-b": [0.0, 1.0]},
            "gallery_crop_ids": ["crop-a", "crop-b"],
        }
        cases = [
            ([], "human_relevance_truth_missing"),
            (
                [
                    {
                        "query_id": "query-1",
                        "status": "complete",
                        "evaluation_eligible": False,
                        "gallery_matches_selection": True,
                        "judgments": [
                            {"crop_id": "crop-a", "judgment": "relevant"},
                            {"crop_id": "crop-b", "judgment": "uncertain"},
                        ],
                    }
                ],
                "human_relevance_truth_unresolved",
            ),
        ]
        for matrices, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason):
                result = evaluate_retrieval(**common, relevance_matrices=matrices)
                self.assertEqual(result["status"], "insufficient")
                self.assertIn(expected_reason, result["reasons"])
                self.assertIsNone(result["metrics"]["map"])


if __name__ == "__main__":
    unittest.main()
