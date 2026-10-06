from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from gods_mlops.training.detector import _draft_detections


class _Model:
    def eval(self):
        return self

    def __call__(self, **_inputs):
        return object()


class _Processor:
    last_threshold = None

    def __call__(self, *, images, return_tensors):
        return {"pixel_values": torch.zeros((1, 3, images.height, images.width))}

    def post_process_object_detection(self, _outputs, *, target_sizes, threshold):
        self.last_threshold = threshold
        return [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "boxes": torch.tensor([[10.0, 5.0, 40.0, 35.0]]),
            }
        ]


class DetectorOutputPersistenceTests(unittest.TestCase):
    def test_draft_detections_persists_json_before_returning_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
            ):
                result = _draft_detections(
                    config={"score_threshold": 0.3},
                    identity={
                        "model_id": "PekingU/rtdetr_v2_r18vd",
                        "model_revision": "locked-revision",
                        "config_version": "detr-test-v1",
                        "input_id": "dataset-test-v1",
                        "input_sha256": "a" * 64,
                    },
                    model=_Model(),
                    processor=_Processor(),
                    items=[{"item_id": "frame-1"}],
                    images=[Image.new("RGB", (100, 60))],
                    person_class_id=1,
                    device=torch.device("cpu"),
                    output_uri=directory,
                    output_path=output_path,
                    started=time.perf_counter(),
                )

            persisted_path = output_path / "draft-detections.json"
            self.assertTrue(persisted_path.is_file())
            persisted_bytes = persisted_path.read_bytes()
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["result_artifact_payload"], persisted_bytes)
            draft = json.loads(persisted_bytes)["drafts"][0]
            self.assertEqual(draft["item_id"], "frame-1")
            self.assertEqual((draft["image_width"], draft["image_height"]), (100, 60))
            self.assertEqual(draft["detections"][0]["bbox_xyxy"], [10.0, 5.0, 40.0, 35.0])
            self.assertAlmostEqual(draft["detections"][0]["score"], 0.9)

    def test_evaluation_inference_keeps_all_scores_for_coco_ap_and_records_count_threshold(self) -> None:
        processor = _Processor()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
            ):
                _draft_detections(
                    config={"phase": "evaluation", "target_phase": "evaluation", "score_threshold": 0.5},
                    identity={
                        "model_id": "PekingU/rtdetr_v2_r18vd",
                        "model_revision": "locked-revision",
                        "config_version": "detr-eval-v1",
                        "input_id": "dataset-test-v1",
                        "input_sha256": "a" * 64,
                    },
                    model=_Model(),
                    processor=processor,
                    items=[{"item_id": "frame-1"}],
                    images=[Image.new("RGB", (100, 60))],
                    person_class_id=1,
                    device=torch.device("cpu"),
                    output_uri=directory,
                    output_path=Path(directory),
                    started=time.perf_counter(),
                )
            payload = json.loads((Path(directory) / "draft-detections.json").read_bytes())

        self.assertEqual(processor.last_threshold, 0.0)
        self.assertEqual(payload["score_threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
