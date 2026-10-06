"""Canonical person-only COCO metrics and fixed-threshold error counts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

_IOU_FOR_COUNTS = 0.5


def manifest_person_boxes_xywh(
    item: Mapping[str, Any], *, image_width: int, image_height: int
) -> list[list[float]]:
    """Convert frozen Task 6 person percentage boxes to original-frame COCO xywh."""
    if type(image_width) is not int or image_width <= 0 or type(image_height) is not int or image_height <= 0:
        raise ValueError("original frame dimensions must be positive integers")
    snapshot = item.get("snapshot", {})
    bbox = snapshot.get("bbox_annotation", item.get("bbox_annotation", {})) if isinstance(snapshot, Mapping) else {}
    results = bbox.get("result", []) if isinstance(bbox, Mapping) else []
    boxes: list[list[float]] = []
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("bbox annotation contains a non-object result")
        if result.get("from_name") != "bbox" or result.get("type") != "rectanglelabels":
            continue
        value = result.get("value")
        if not isinstance(value, Mapping) or "person" not in value.get("rectanglelabels", []):
            continue
        source_width = _finite_number(result.get("original_width", image_width), "bbox original width")
        source_height = _finite_number(result.get("original_height", image_height), "bbox original height")
        if source_width <= 0 or source_height <= 0:
            raise ValueError("bbox annotation image dimensions are invalid")
        scale_x, scale_y = image_width / source_width, image_height / source_height
        x = _finite_number(value.get("x"), "bbox x") * source_width / 100.0 * scale_x
        y = _finite_number(value.get("y"), "bbox y") * source_height / 100.0 * scale_y
        width = _finite_number(value.get("width"), "bbox width") * source_width / 100.0 * scale_x
        height = _finite_number(value.get("height"), "bbox height") * source_height / 100.0 * scale_y
        x = min(max(0.0, x), float(image_width))
        y = min(max(0.0, y), float(image_height))
        width = min(max(0.0, width), float(image_width) - x)
        height = min(max(0.0, height), float(image_height) - y)
        if width <= 0 or height <= 0:
            raise ValueError("person bbox has no visible area in the frozen frame")
        boxes.append([x, y, width, height])
    return boxes


def evaluate_detections(
    *,
    frames: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    score_threshold: float,
    max_detections: int,
) -> dict[str, Any]:
    """Evaluate original-coordinate person boxes with pycocotools COCOeval.

    Each frame contains `item_id`, original `width`/`height`, and person-only
    `person_boxes_xywh`. Prediction boxes use the runner's original-coordinate
    `bbox_xyxy` format. COCO AP uses scores at every confidence; the configured
    threshold is applied only to the separate FP/FN counts.
    """
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise ValueError("detection frames must be a sequence")
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise ValueError("detection predictions must be a sequence")
    if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("detection confidence threshold must be between zero and one")
    if type(max_detections) is not int or max_detections < 10:
        raise ValueError("COCO max_detections must be an integer of at least 10")

    normalized_frames = _normalize_frames(frames)
    if not normalized_frames:
        return _insufficient(
            reason="evaluation_frames_missing",
            frames=0,
            person_ground_truth=0,
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
    frame_ids = {item["item_id"] for item in normalized_frames}
    normalized_predictions = _normalize_predictions(predictions, frame_ids)
    ground_truth_count = sum(len(frame["person_boxes_xywh"]) for frame in normalized_frames)
    count_errors = _threshold_counts(
        normalized_frames,
        normalized_predictions,
        threshold=score_threshold,
        max_detections=max_detections,
    )
    counts = {
        "frames": len(normalized_frames),
        "person_ground_truth": ground_truth_count,
        "predictions": sum(len(items) for items in normalized_predictions.values()),
        **count_errors,
    }
    if ground_truth_count == 0:
        return _insufficient(
            reason="no_person_ground_truth",
            frames=len(normalized_frames),
            person_ground_truth=0,
            score_threshold=score_threshold,
            max_detections=max_detections,
            counts=counts,
        )

    metrics = _coco_metrics(
        normalized_frames,
        normalized_predictions,
        max_detections=max_detections,
    )
    return {
        "status": "complete",
        "metrics": metrics,
        "counts": counts,
        "settings": {
            "score_threshold": score_threshold,
            "count_iou_threshold": _IOU_FOR_COUNTS,
            "max_detections": max_detections,
            "coordinates": "original-pixel-xyxy-predictions-and-xywh-ground-truth",
            "class_mapping": {"person": 1},
            "coco_iou_thresholds": [round(0.5 + 0.05 * index, 2) for index in range(10)],
        },
        "reasons": [],
    }


def _normalize_frames(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("each evaluation frame must be an object")
        item_id = _identifier(frame.get("item_id"), "frame item_id")
        if item_id in seen_ids:
            raise ValueError("evaluation frame item IDs must be unique")
        seen_ids.add(item_id)
        width = _dimension(frame.get("width"), "frame width")
        height = _dimension(frame.get("height"), "frame height")
        boxes = frame.get("person_boxes_xywh")
        if not isinstance(boxes, Sequence) or isinstance(boxes, (str, bytes)):
            raise ValueError("frame person_boxes_xywh must be a sequence")
        normalized_boxes = []
        for box in boxes:
            if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
                raise ValueError("person ground-truth boxes must use x, y, width, height")
            x, y, box_width, box_height = (_finite_number(value, "ground-truth coordinate") for value in box)
            if (
                x < 0
                or y < 0
                or box_width <= 0
                or box_height <= 0
                or x + box_width > width
                or y + box_height > height
            ):
                raise ValueError("person ground-truth boxes must fit inside the original frame")
            normalized_boxes.append([x, y, box_width, box_height])
        normalized.append(
            {
                "item_id": item_id,
                "width": width,
                "height": height,
                "person_boxes_xywh": normalized_boxes,
            }
        )
    return normalized


def _normalize_predictions(
    predictions: Sequence[Mapping[str, Any]], frame_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    normalized = {item_id: [] for item_id in frame_ids}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ValueError("each frame prediction must be an object")
        item_id = _identifier(prediction.get("item_id"), "prediction item_id")
        if item_id not in frame_ids:
            raise ValueError("prediction references a frame outside the frozen evaluation set")
        detections = prediction.get("detections")
        if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
            raise ValueError("frame detections must be a sequence")
        for detection in detections:
            if not isinstance(detection, Mapping):
                raise ValueError("each person detection must be an object")
            score = _finite_number(detection.get("score"), "detection score")
            if not 0 <= score <= 1:
                raise ValueError("detection score must be between zero and one")
            box = detection.get("bbox_xyxy")
            if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
                raise ValueError("person detections must use original-coordinate xyxy boxes")
            x1, y1, x2, y2 = (_finite_number(value, "detection coordinate") for value in box)
            if x2 <= x1 or y2 <= y1:
                raise ValueError("person detection boxes must have positive area")
            normalized[item_id].append({"score": score, "bbox_xyxy": [x1, y1, x2, y2]})
    return normalized


def _threshold_counts(
    frames: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    threshold: float,
    max_detections: int,
) -> dict[str, int]:
    true_positives = false_positives = false_negatives = 0
    for frame in frames:
        boxes = frame["person_boxes_xywh"]
        candidates = [item for item in predictions[frame["item_id"]] if item["score"] >= threshold]
        candidates.sort(key=lambda item: (-item["score"], item["bbox_xyxy"]))
        candidates = candidates[:max_detections]
        unmatched = set(range(len(boxes)))
        for candidate in candidates:
            candidate_xywh = _xyxy_to_xywh(candidate["bbox_xyxy"])
            match = max(
                unmatched,
                key=lambda index: _iou(candidate_xywh, boxes[index]),
                default=None,
            )
            if match is None or _iou(candidate_xywh, boxes[match]) < _IOU_FOR_COUNTS:
                false_positives += 1
            else:
                unmatched.remove(match)
                true_positives += 1
        false_negatives += len(unmatched)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def _coco_metrics(
    frames: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError("canonical person COCO evaluation requires locked pycocotools") from error

    with redirect_stdout(StringIO()):
        return _run_coco_evaluation(
            frames,
            predictions,
            max_detections=max_detections,
            coco_type=COCO,
            coco_evaluator_type=COCOeval,
        )


def _run_coco_evaluation(
    frames: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    *,
    max_detections: int,
    coco_type,
    coco_evaluator_type,
) -> dict[str, float]:
    sorted_frames = sorted(frames, key=lambda item: item["item_id"])
    image_ids = {frame["item_id"]: index for index, frame in enumerate(sorted_frames, start=1)}
    image_rows = [
        {"id": image_ids[frame["item_id"]], "width": frame["width"], "height": frame["height"]}
        for frame in sorted_frames
    ]
    ground_truth_annotations = []
    detection_rows = []
    annotation_id = 1
    for frame in sorted_frames:
        image_id = image_ids[frame["item_id"]]
        for box in frame["person_boxes_xywh"]:
            ground_truth_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        for detection in predictions[frame["item_id"]]:
            detection_rows.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": _xyxy_to_xywh(detection["bbox_xyxy"]),
                    "score": detection["score"],
                }
            )

    ground_truth = coco_type()
    ground_truth.dataset = {
        "info": {"description": "person-only immutable Gods MLOps evaluation"},
        "images": image_rows,
        "categories": [{"id": 1, "name": "person"}],
        "annotations": ground_truth_annotations,
    }
    ground_truth.createIndex()
    if detection_rows:
        detection_results = ground_truth.loadRes(detection_rows)
    else:
        detection_results = coco_type()
        detection_results.dataset = {
            "info": {"description": "empty person-only prediction set"},
            "images": image_rows,
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [],
        }
        detection_results.createIndex()

    evaluator = coco_evaluator_type(ground_truth, detection_results, "bbox")
    evaluator.params.imgIds = [row["id"] for row in image_rows]
    evaluator.params.catIds = [1]
    evaluator.params.maxDets = [1, 10, max_detections]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    ap = _mean_valid(precision[:, :, 0, 0, -1])
    ap50 = _mean_valid(precision[0, :, 0, 0, -1])
    ar = _mean_valid(recall[:, 0, 0, -1])
    return {"ap_50_95": ap, "ap50": ap50, "ar": ar}


def _mean_valid(values) -> float:
    flattened = [float(value) for value in values.flat if value > -1]
    return sum(flattened) / len(flattened) if flattened else 0.0


def _insufficient(
    *,
    reason: str,
    frames: int,
    person_ground_truth: int,
    score_threshold: float,
    max_detections: int,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    if counts is None:
        counts = {
            "frames": frames,
            "person_ground_truth": person_ground_truth,
            "predictions": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
    return {
        "status": "insufficient",
        "metrics": {"ap_50_95": None, "ap50": None, "ar": None},
        "counts": counts,
        "settings": {
            "score_threshold": score_threshold,
            "count_iou_threshold": _IOU_FOR_COUNTS,
            "max_detections": max_detections,
            "coordinates": "original-pixel-xyxy-predictions-and-xywh-ground-truth",
            "class_mapping": {"person": 1},
            "coco_iou_thresholds": [round(0.5 + 0.05 * index, 2) for index in range(10)],
        },
        "reasons": [reason],
    }


def _xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    left_x2, left_y2 = left[0] + left[2], left[1] + left[3]
    right_x2, right_y2 = right[0] + right[2], right[1] + right[3]
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _dimension(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result
