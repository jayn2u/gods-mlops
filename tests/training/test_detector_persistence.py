from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
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
                    person_class_name="person",
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
                    person_class_name="person",
                    device=torch.device("cpu"),
                    output_uri=directory,
                    output_path=Path(directory),
                    started=time.perf_counter(),
                )
            payload = json.loads((Path(directory) / "draft-detections.json").read_bytes())

        self.assertEqual(processor.last_threshold, 0.0)
        self.assertEqual(payload["score_threshold"], 0.5)

    def test_detr_evaluation_yield_cursor_resumes_after_completed_frame_without_optimizer_state(self) -> None:
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
        checkpoint_sha = "b" * 64
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            processor = _FrameProcessor(person_class_id=17)
            guard_calls = iter((True, False))
            first_config = {
                "phase": "evaluation",
                "target_phase": "evaluation",
                "score_threshold": 0.3,
                "input_sha256": identity["input_sha256"],
                "config_sha256": identity["config_sha256"],
                "_evaluation_job_identity": identity,
                "_evaluation_checkpoint_sha256": checkpoint_sha,
                "_assert_current": lambda: next(guard_calls),
            }
            items = [{"item_id": "frame-a"}, {"item_id": "frame-b"}]
            images = [Image.new("RGB", (100, 60)), Image.new("RGB", (200, 60))]
            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
            ):
                yielded = _draft_detections(
                    config=first_config,
                    identity={
                        "model_id": "PekingU/rtdetr_v2_r18vd",
                        "model_revision": "locked-detr-revision",
                        "config_version": "detr-evaluation-v1",
                        "input_id": identity["input_id"],
                        "input_sha256": identity["input_sha256"],
                    },
                    model=_Model(),
                    processor=processor,
                    items=items,
                    images=images,
                    person_class_id=17,
                    person_class_name="person",
                    device=torch.device("cpu"),
                    output_uri=directory,
                    output_path=output_path,
                    started=time.perf_counter(),
                )

            self.assertEqual(yielded["status"], "yielded")
            self.assertIsInstance(yielded.get("checkpoint_payload"), bytes)
            decode = __import__("gods_mlops.evaluation.report", fromlist=["decode_evaluation_cursor"]).decode_evaluation_cursor
            cursor = decode(
                yielded["checkpoint_payload"],
                expected_identity=identity,
                expected_candidate_checkpoint_sha256=checkpoint_sha,
                expected_manifest_sha256=identity["input_sha256"],
                expected_config_sha256=identity["config_sha256"],
            )
            self.assertEqual(cursor["next_index"], 1)
            self.assertEqual([draft["item_id"] for draft in cursor["predictions"]["drafts"]], ["frame-a"])
            self.assertNotIn("optimizer_state_dict", cursor["predictions"])

            from gods_mlops.jobs.checkpoints import CheckpointIdentity
            from gods_mlops.jobs.queue import JobQueue

            checkpoint_identity = CheckpointIdentity.from_dict(identity)
            source_authority = object()

            class Repository:
                committed_payload = None
                intent_source_registry = None
                commit_source_registry = None

                async def get_job(self, _job_id):
                    return {"phase": "evaluation", "model_kind": "detr", "config_version": "detr-evaluation-v1"}

                async def checkpoint_identity(self, _job_id):
                    return checkpoint_identity

                async def lease_is_current(self, _job_id, _lease_token):
                    return True

                async def pending_checkpoint_prunes_for(self, _job_id):
                    return []

                async def get_profile(self, **_kwargs):
                    return {"checkpoint_reservation_bytes": 1024}

                async def checkpoint_metadata_for(self, _job_id):
                    return None

                async def begin_artifact_write(self, **kwargs):
                    self.intent_source_registry = kwargs["source_registry"]
                    self.prepared = kwargs["prepared"]
                    return "cursor-write-1"

                async def commit_checkpoint(self, **kwargs):
                    self.commit_source_registry = kwargs["source_registry"]
                    self.committed_payload = kwargs["prepared"].payload
                    return SimpleNamespace(uri=kwargs["prepared"].uri, sha256=kwargs["prepared"].sha256)

            class CursorStore:
                _bucket = "gods-mlops"
                _prefix = "jobs"

                def prepare(self, *, identity, payload, **_kwargs):
                    digest = __import__("hashlib").sha256(payload).hexdigest()
                    key = f"jobs/{identity.job_id}/checkpoints/{digest}.checkpoint"
                    return SimpleNamespace(
                        identity=identity,
                        payload=payload,
                        sha256=digest,
                        size_bytes=len(payload),
                        metadata_size_bytes=0,
                        object_key=key,
                        uri=f"s3://gods-mlops/{key}",
                        previous_uri=None,
                        previous_sha256=None,
                        previous_size_bytes=None,
                        previous_metadata_size_bytes=None,
                    )

            repository = Repository()
            queue = JobQueue(repository=repository, sources=source_authority)
            asyncio.run(
                queue.save_checkpoint(
                    store=CursorStore(),
                    job_id=identity["job_id"],
                    lease_token="8ad96890-3434-4f07-85bb-8cde17a2b009",
                    identity=checkpoint_identity,
                    payload=yielded["checkpoint_payload"],
                )
            )
            self.assertIs(repository.intent_source_registry, source_authority)
            self.assertIs(repository.commit_source_registry, source_authority)
            self.assertEqual(repository.committed_payload, yielded["checkpoint_payload"])

            resumed_processor = _FrameProcessor(person_class_id=17)
            resumed_config = {
                **first_config,
                "_evaluation_resume_payload": yielded["checkpoint_payload"],
                "_assert_current": lambda: True,
            }
            with (
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
            ):
                completed = _draft_detections(
                    config=resumed_config,
                    identity={
                        "model_id": "PekingU/rtdetr_v2_r18vd",
                        "model_revision": "locked-detr-revision",
                        "config_version": "detr-evaluation-v1",
                        "input_id": identity["input_id"],
                        "input_sha256": identity["input_sha256"],
                    },
                    model=_Model(),
                    processor=resumed_processor,
                    items=items,
                    images=images,
                    person_class_id=17,
                    person_class_name="person",
                    device=torch.device("cpu"),
                    output_uri=directory,
                    output_path=output_path,
                    started=time.perf_counter(),
                )

            self.assertEqual(completed["status"], "succeeded")
            self.assertEqual(resumed_processor.seen_sizes, [(200, 60)])
            self.assertEqual(completed["resource_measurements"]["optimizer_steps"], 0)
            payload = json.loads(completed["result_artifact_payload"])
            self.assertEqual([draft["item_id"] for draft in payload["drafts"]], ["frame-a", "frame-b"])
            self.assertEqual(
                payload["person_class_mapping"],
                {
                    "model_revision": "locked-detr-revision",
                    "model_class_index": 17,
                    "model_class_name": "person",
                    "coco_category_id": 1,
                    "coco_category_name": "person",
                },
            )


class _FrameProcessor:
    def __init__(self, *, person_class_id: int):
        self.person_class_id = person_class_id
        self.seen_sizes: list[tuple[int, int]] = []

    def __call__(self, *, images, return_tensors):
        self.seen_sizes.append(images.size)
        return {"pixel_values": torch.zeros((1, 3, images.height, images.width))}

    def post_process_object_detection(self, _outputs, *, target_sizes, threshold):
        return [
            {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([self.person_class_id]),
                "boxes": torch.tensor([[10.0, 5.0, 40.0, 35.0]]),
            }
        ]


if __name__ == "__main__":
    unittest.main()
