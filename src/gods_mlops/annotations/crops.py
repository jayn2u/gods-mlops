"""Create revision-bound person crops and retain their source lineage."""

from __future__ import annotations

import io
import math
from functools import partial
from hashlib import sha256
from typing import Any
from uuid import UUID

import anyio
from PIL import Image

from gods_mlops.ingestion.storage import S3SampleStore

from .models import ReviewAssignmentConflictError
from .storage import PostgresAnnotationRepository


class CropService:
    def __init__(self, *, repository: PostgresAnnotationRepository, objects: S3SampleStore) -> None:
        self._repository = repository
        self._objects = objects

    async def create_crop(self, frame_id: str, bbox_revision: str) -> dict[str, Any]:
        """Create all person crops for one immutable bbox annotation revision."""
        sample_id = UUID(frame_id)
        bbox_revision_id = UUID(bbox_revision)
        existing = await self._repository.crops_for_revision(
            sample_id=sample_id,
            bbox_revision=bbox_revision_id,
        )
        if existing and not any(crop["parent_available"] for crop in existing):
            if all(crop["state"] == "ready" and crop["crop_set_ready"] for crop in existing):
                return _crop_batch(sample_id, bbox_revision_id, existing)
            raise ReviewAssignmentConflictError("the expired frame cannot repair an incomplete crop")

        source = await self._repository.bbox_crop_source(
            sample_id=sample_id,
            bbox_revision=bbox_revision_id,
        )
        frame_bytes = await anyio.to_thread.run_sync(
            partial(
                self._objects.read_object,
                object_key=source["object_key"],
                expected_sha256=source["sha256"],
            )
        )
        specs = await anyio.to_thread.run_sync(
            partial(extract_person_crops, frame_bytes=frame_bytes, result=source["result"])
        )
        if not specs:
            if existing:
                raise ReviewAssignmentConflictError("bbox result no longer matches its immutable crop set")
            return _crop_batch(sample_id, bbox_revision_id, [])

        reserved = await self._repository.reserve_crops(
            sample_id=sample_id,
            bbox_revision=bbox_revision_id,
            specs=specs,
            source=source,
        )
        for crop, spec in zip(reserved, specs, strict=True):
            try:
                await anyio.to_thread.run_sync(
                    partial(
                        self._objects.ensure_object,
                        object_key=crop["object_key"],
                        image=spec["image"],
                        expected_sha256=crop["sha256"],
                    )
                )
            except Exception as upload_error:
                try:
                    await self._reconcile_expired_crop_write(crop)
                except Exception as cleanup_error:
                    raise cleanup_error from upload_error
                raise
            try:
                await self._repository.mark_crop_ready(UUID(crop["crop_id"]))
            except ReviewAssignmentConflictError as state_error:
                try:
                    await self._reconcile_expired_crop_write(crop)
                except Exception as cleanup_error:
                    raise cleanup_error from state_error
                raise
        return _crop_batch(
            sample_id,
            bbox_revision_id,
            await self._repository.crops_for_revision(
                sample_id=sample_id,
                bbox_revision=bbox_revision_id,
            ),
        )

    async def _reconcile_expired_crop_write(self, crop: dict[str, Any]) -> None:
        """Remove ambiguous late bytes only after expiry fenced the crop row."""
        rows = await self._repository.crops_for_revision(
            sample_id=UUID(crop["sample_id"]),
            bbox_revision=UUID(crop["bbox_revision"]),
        )
        current = next((item for item in rows if item["crop_id"] == crop["crop_id"]), None)
        if current is None or current["state"] not in {"purge_pending", "deleted"}:
            return
        try:
            await anyio.to_thread.run_sync(
                partial(self._objects.delete_object, crop["object_key"])
            )
        except Exception as error:  # noqa: BLE001 - retain a quota-safe retry marker
            await self._repository.mark_crop_cleanup_pending(UUID(crop["crop_id"]))
            await self._repository.record_retention_event(
                sample_id=UUID(crop["sample_id"]),
                action="crop_expiry_deferred",
                reason="late_crop_writer_cleanup_failed",
                details={"crop_id": crop["crop_id"], "error_type": type(error).__name__},
            )
            raise


def extract_person_crops(*, frame_bytes: bytes, result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interpret Label Studio percentage rectangles against the exact frame dimensions."""
    with Image.open(io.BytesIO(frame_bytes)) as source:
        source.load()
        width, height = source.size
        if width <= 0 or height <= 0:
            raise ValueError("source frame has invalid dimensions")
        rgb = source.convert("RGB")
        crops: list[dict[str, Any]] = []
        for result_index, region in enumerate(result):
            if region.get("from_name") != "bbox" or region.get("type") != "rectanglelabels":
                continue
            value = region.get("value")
            if not isinstance(value, dict):
                raise ValueError("bbox result is missing rectangle values")
            labels = value.get("rectanglelabels")
            if not isinstance(labels, list) or "person" not in labels:
                continue
            if region.get("original_width") != width or region.get("original_height") != height:
                raise ValueError("bbox dimensions do not match the source frame")
            x = _coordinate(value.get("x"))
            y = _coordinate(value.get("y"))
            box_width = _coordinate(value.get("width"))
            box_height = _coordinate(value.get("height"))
            if box_width <= 0 or box_height <= 0:
                raise ValueError("bbox width and height must be positive")
            left = max(0, math.floor(x * width / 100))
            top = max(0, math.floor(y * height / 100))
            right = min(width, math.ceil((x + box_width) * width / 100))
            bottom = min(height, math.ceil((y + box_height) * height / 100))
            if right <= left or bottom <= top:
                raise ValueError("bbox does not overlap the source frame")
            region_id = region.get("id")
            if not isinstance(region_id, str) or not region_id:
                region_id = f"region-{result_index}"
            output = io.BytesIO()
            rgb.crop((left, top, right, bottom)).save(
                output,
                format="JPEG",
                quality=92,
                optimize=True,
            )
            data = output.getvalue()
            crops.append(
                {
                    "region_index": result_index,
                    "region_id": region_id,
                    "image": data,
                    "sha256": sha256(data).hexdigest(),
                    "width": right - left,
                    "height": bottom - top,
                }
            )
        return crops


def _coordinate(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("bbox coordinates must be finite numbers")
    if value < 0 or value > 100:
        raise ValueError("bbox coordinates must use 0 to 100 percent units")
    return float(value)


def _crop_batch(sample_id: UUID, bbox_revision: UUID, crops: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "frame_id": str(sample_id),
        "bbox_revision": str(bbox_revision),
        "crop_count": len(crops),
        "crop_ids": [crop["crop_id"] for crop in crops],
        "crop_set_ready": all(crop["crop_set_ready"] for crop in crops),
        "crops": crops,
    }
    if len(crops) == 1:
        result["crop_id"] = crops[0]["crop_id"]
    return result
