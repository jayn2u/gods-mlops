from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import threading
from datetime import date, datetime, timedelta, timezone
from inspect import iscoroutinefunction, signature
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import asyncpg
import pytest
from PIL import Image

from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.ingestion.storage import GLOBAL_OBJECT_LIMIT, PostgresIngestionRepository, S3SampleStore


def _module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("gods_mlops.datasets"):
            pytest.fail(f"dataset publication implementation is missing: {error}")
        raise


def test_publish_and_invalidation_functions_expose_the_approved_contracts() -> None:
    publisher_module = _module("gods_mlops.datasets.publish")
    deletion_module = _module("gods_mlops.datasets.deletion")

    publish = publisher_module.publish_dataset
    invalidate = deletion_module.invalidate_sample
    assert iscoroutinefunction(publish)
    assert iscoroutinefunction(invalidate)
    assert list(signature(publish).parameters)[:2] == ["selection", "config_version"]
    assert list(signature(invalidate).parameters)[0] == "sample_id"
    assert hasattr(publisher_module, "DatasetPublisher")
    assert hasattr(publisher_module, "DatasetObjectStore")


def test_selection_must_pin_current_annotation_revisions() -> None:
    module = _module("gods_mlops.datasets.publish")
    sample_id, crop_id = str(uuid4()), str(uuid4())

    with pytest.raises(ValueError, match="bbox_revisions must pin"):
        module._normalize_selection({"target": "detr", "sample_ids": [sample_id]})
    with pytest.raises(ValueError, match="caption_revisions must pin"):
        module._normalize_selection({"target": "clip", "crop_ids": [crop_id]})


def test_sparse_evaluation_splits_do_not_block_ready_training_inputs() -> None:
    module = _module("gods_mlops.datasets.publish")
    counts = {
        "train": {"frames": 20, "positive_frames": 10, "negative_frames": 10, "crop_caption_pairs": 20},
        "validation": {"frames": 19, "positive_frames": 9, "negative_frames": 10, "crop_caption_pairs": 19},
        "test": {"frames": 20, "positive_frames": 10, "negative_frames": 10, "crop_caption_pairs": 20},
    }

    training_reasons, evaluation_reasons = module._readiness_reasons("detr", counts)

    assert training_reasons == []
    assert evaluation_reasons == [
        "insufficient_validation_detr_frames",
        "insufficient_validation_detr_positive_frames",
    ]


def test_training_side_minimums_remain_hard_publication_gates() -> None:
    module = _module("gods_mlops.datasets.publish")
    counts = {
        "train": {"frames": 19, "positive_frames": 10, "negative_frames": 9, "crop_caption_pairs": 20},
        "validation": {"frames": 20, "positive_frames": 10, "negative_frames": 10, "crop_caption_pairs": 20},
        "test": {"frames": 20, "positive_frames": 10, "negative_frames": 10, "crop_caption_pairs": 20},
    }

    training_reasons, evaluation_reasons = module._readiness_reasons("detr", counts)

    assert training_reasons == ["insufficient_train_detr_frames"]
    assert evaluation_reasons == []


def test_unready_human_truth_is_not_freezable_but_uncertain_truth_stays_explicit() -> None:
    module = _module("gods_mlops.datasets.publish")
    gallery = [
        {"crop_id": str(uuid4()), "sha256": "1" * 64},
        {"crop_id": str(uuid4()), "sha256": "2" * 64},
        {"crop_id": str(uuid4()), "sha256": "3" * 64},
    ]
    provenance = {
        "source": "label_studio",
        "label_studio_annotation_id": 5,
        "reviewer_id": "human-reviewer",
    }
    positive_and_unknown = [
        {"crop_id": gallery[0]["crop_id"], "judgment": "relevant"},
        {"crop_id": gallery[1]["crop_id"], "judgment": "uncertain"},
    ]
    complete_but_uncertain = [
        {"crop_id": gallery[0]["crop_id"], "judgment": "relevant"},
        {"crop_id": gallery[1]["crop_id"], "judgment": "uncertain"},
        {"crop_id": gallery[2]["crop_id"], "judgment": "not_relevant"},
    ]

    assert module._complete_human_matrix(gallery, positive_and_unknown, provenance) is False
    assert module._complete_human_matrix(gallery, complete_but_uncertain, provenance) is True


def _settings() -> dict[str, str] | None:
    names = (
        "GODS_MLOPS_TEST_DATABASE_URL",
        "GODS_MLOPS_TEST_S3_ENDPOINT",
        "GODS_MLOPS_TEST_S3_ACCESS_KEY",
        "GODS_MLOPS_TEST_S3_SECRET_KEY",
        "GODS_MLOPS_TEST_S3_BUCKET",
    )
    values = {name: os.environ.get(name, "") for name in names}
    return values if all(values.values()) else None


def _seed_sources(
    *,
    target: str,
    count_per_day: int = 20,
    capture_days: tuple[int, ...] = (1, 2, 3),
    camera_id: UUID | None = None,
) -> tuple[list[dict], PostgresAnnotationRepository]:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]

    async def exercise() -> tuple[list[dict], PostgresAnnotationRepository]:
        ingestion = PostgresIngestionRepository(database_url=database_url)
        await ingestion.ensure_schema()
        annotations = PostgresAnnotationRepository(database_url=database_url)
        await annotations.ensure_schema()
        objects = S3SampleStore(
            endpoint_url=settings["GODS_MLOPS_TEST_S3_ENDPOINT"],
            access_key=settings["GODS_MLOPS_TEST_S3_ACCESS_KEY"],
            secret_key=settings["GODS_MLOPS_TEST_S3_SECRET_KEY"],
            bucket=settings["GODS_MLOPS_TEST_S3_BUCKET"],
            region="us-east-1",
        )
        source_camera_id = camera_id or uuid4()
        task_base = uuid4().int % 100_000_000
        now = datetime.now(timezone.utc)
        records: list[dict] = []
        accounted_bytes = 0
        connection = await asyncpg.connect(database_url)
        try:
            for day in capture_days:
                for offset in range(count_per_day):
                    index = (day - 1) * count_per_day + offset
                    sample_id = uuid4()
                    frame_bytes = _jpeg(index)
                    frame_sha = hashlib.sha256(frame_bytes).hexdigest()
                    frame_key = f"samples/{source_camera_id}/{sample_id}/{frame_sha}.jpg"
                    objects.ensure_object(
                        object_key=frame_key,
                        image=frame_bytes,
                        expected_sha256=frame_sha,
                    )
                    local_capture = datetime(2026, 10, day, 12, offset % 60, tzinfo=ZoneInfo("Asia/Seoul"))
                    captured_at = local_capture.astimezone(timezone.utc)
                    await connection.execute(
                        """
                        INSERT INTO ingestion_samples (
                            sample_id, camera_id, capture_day, captured_at_utc, reason,
                            sha256, model_revision, processor_revision, object_key,
                            object_size_bytes, receipt_id, state, selected, received_at,
                            retention_until
                        ) VALUES ($1, $2, $3, $4, 'periodic', $5, 'detector-r1', 'processor-r1',
                                  $6, $7, $8, 'received', FALSE, $4, $9)
                        """,
                        sample_id,
                        source_camera_id,
                        date(2026, 10, day),
                        captured_at,
                        frame_sha,
                        frame_key,
                        len(frame_bytes),
                        uuid4(),
                        now + timedelta(days=7),
                    )
                    accounted_bytes += len(frame_bytes)

                    bbox_revision = uuid4()
                    bbox_assignment = uuid4()
                    has_person = offset < 10
                    bbox_result = (
                        [
                            {
                                "from_name": "bbox",
                                "type": "rectanglelabels",
                                "original_width": 20,
                                "original_height": 20,
                                "value": {
                                    "x": 10,
                                    "y": 10,
                                    "width": 50,
                                    "height": 50,
                                    "rectanglelabels": ["person"],
                                },
                            }
                        ]
                        if has_person
                        else []
                    )
                    bbox_json = json.dumps(bbox_result, sort_keys=True, separators=(",", ":"))
                    bbox_sha = hashlib.sha256(bbox_json.encode("utf-8")).hexdigest()
                    bbox_task_id = 1_000_000 + task_base + index
                    await connection.execute(
                        """
                        INSERT INTO review_assignments (
                            assignment_id, revision, sample_id, stage, bbox_revision,
                            label_studio_task_id, media_object_key, required_bytes, state, finished_at
                        ) VALUES ($1, $2, $3, 'bbox', NULL, $4, $5, $6, 'finalized', now())
                        """,
                        uuid4(),
                        bbox_assignment,
                        sample_id,
                        bbox_task_id,
                        frame_key,
                        len(frame_bytes),
                    )
                    await connection.execute(
                        """
                        INSERT INTO annotation_revisions (
                            annotation_revision_id, sample_id, assignment_revision, stage,
                            label_studio_task_id, label_studio_annotation_id, result,
                            result_sha256, provenance, submitted_at
                        ) VALUES ($1, $2, $3, 'bbox', $4, $5, $6::jsonb, $7, $8::jsonb, $9)
                        """,
                        bbox_revision,
                        sample_id,
                        bbox_assignment,
                        bbox_task_id,
                        1_000_000_000 + task_base + index,
                        bbox_json,
                        bbox_sha,
                        json.dumps({"source": "task6-test-fixture", "reviewer_id": "test-reviewer"}),
                        captured_at,
                    )
                    await connection.execute(
                        """
                        INSERT INTO sample_annotation_heads (
                            sample_id, current_bbox_assignment_revision, latest_bbox_revision
                        ) VALUES ($1, NULL, $2)
                        """,
                        sample_id,
                        bbox_revision,
                    )

                    record = {
                        "sample_id": str(sample_id),
                        "camera_id": str(source_camera_id),
                        "capture_day": date(2026, 10, day),
                        "captured_at_utc": captured_at,
                        "frame_object_key": frame_key,
                        "frame_sha256": frame_sha,
                        "frame_size_bytes": len(frame_bytes),
                        "bbox_revision": str(bbox_revision),
                        "bbox_sha256": bbox_sha,
                        "positive": has_person,
                    }
                    if target == "clip":
                        crop_id = uuid4()
                        crop_bytes = _jpeg(index + 100)
                        crop_sha = hashlib.sha256(crop_bytes).hexdigest()
                        crop_key = f"crops/{sample_id}/{bbox_revision}/{crop_id}.jpg"
                        objects.ensure_object(object_key=crop_key, image=crop_bytes, expected_sha256=crop_sha)
                        caption_revision = uuid4()
                        caption_assignment = uuid4()
                        caption_task_id = 200_000_000 + task_base + index
                        caption_text = f"person in jacket {index}"
                        caption_result = [{
                            "from_name": "caption",
                            "to_name": "image",
                            "type": "textarea",
                            "value": {"text": [caption_text]},
                        }]
                        caption_json = json.dumps(caption_result, sort_keys=True, separators=(",", ":"))
                        caption_sha = hashlib.sha256(caption_json.encode("utf-8")).hexdigest()
                        await connection.execute(
                            """
                            INSERT INTO review_assignments (
                                assignment_id, revision, sample_id, stage, bbox_revision,
                                label_studio_task_id, media_object_key, required_bytes, state, finished_at
                            ) VALUES ($1, $2, $3, 'caption', $4, $5, $6, $7, 'finalized', now())
                            """,
                            uuid4(),
                            caption_assignment,
                            sample_id,
                            str(bbox_revision),
                            caption_task_id,
                            crop_key,
                            len(crop_bytes),
                        )
                        await connection.execute(
                            """
                            INSERT INTO annotation_revisions (
                                annotation_revision_id, sample_id, assignment_revision, stage,
                                label_studio_task_id, label_studio_annotation_id, result,
                                result_sha256, provenance, submitted_at
                            ) VALUES ($1, $2, $3, 'caption', $4, $5, $6::jsonb, $7, $8::jsonb, $9)
                            """,
                            caption_revision,
                            sample_id,
                            caption_assignment,
                            caption_task_id,
                            1_200_000_000 + task_base + index,
                            caption_json,
                            caption_sha,
                            json.dumps({"source": "task6-test-fixture", "reviewer_id": "test-reviewer"}),
                            captured_at,
                        )
                        await connection.execute(
                            """
                            INSERT INTO annotation_crops (
                                crop_id, sample_id, bbox_revision, region_index, region_id,
                                object_key, sha256, object_size_bytes, state, caption_state,
                                caption_revision_id, provenance, crop_set_ready
                            ) VALUES ($1, $2, $3, 0, 'region-0', $4, $5, $6, 'ready', 'reviewed',
                                      $7, $8::jsonb, TRUE)
                            """,
                            crop_id,
                            sample_id,
                            bbox_revision,
                            crop_key,
                            crop_sha,
                            len(crop_bytes),
                            caption_revision,
                            json.dumps({
                                "frame_sample_id": str(sample_id),
                                "frame_object_key": frame_key,
                                "frame_sha256": frame_sha,
                                "bbox_revision": str(bbox_revision),
                                "bbox_result_sha256": bbox_sha,
                            }),
                        )
                        record.update(
                            {
                                "crop_id": str(crop_id),
                                "crop_object_key": crop_key,
                                "crop_sha256": crop_sha,
                                "crop_size_bytes": len(crop_bytes),
                                "caption_revision": str(caption_revision),
                                "caption_sha256": caption_sha,
                                "caption_text": caption_text,
                            }
                        )
                        accounted_bytes += len(crop_bytes)
                    records.append(record)

            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = used_bytes + $1 WHERE singleton = TRUE",
                accounted_bytes,
            )
        finally:
            await connection.close()
            await ingestion.close()
            await annotations.close()
        return records, PostgresAnnotationRepository(database_url=database_url)

    return asyncio.run(exercise())


def _jpeg(seed: int) -> bytes:
    from io import BytesIO

    image = Image.new("RGB", (20, 20), (seed % 255, (seed * 3) % 255, (seed * 7) % 255))
    output = BytesIO()
    image.save(output, format="JPEG", quality=86)
    return output.getvalue()


def _make_dataset_objects(settings: dict[str, str], store_type=None):
    module = _module("gods_mlops.datasets.publish")
    factory = store_type or module.DatasetObjectStore
    return factory(
        endpoint_url=settings["GODS_MLOPS_TEST_S3_ENDPOINT"],
        access_key=settings["GODS_MLOPS_TEST_S3_ACCESS_KEY"],
        secret_key=settings["GODS_MLOPS_TEST_S3_SECRET_KEY"],
        bucket=settings["GODS_MLOPS_TEST_S3_BUCKET"],
        region="us-east-1",
    )


def _selection(records: list[dict], target: str) -> dict:
    if target == "detr":
        return {
            "target": "detr",
            "sample_ids": [item["sample_id"] for item in records],
            "bbox_revisions": {item["sample_id"]: item["bbox_revision"] for item in records},
        }
    return {
        "target": "clip",
        "crop_ids": [item["crop_id"] for item in records],
        "caption_revisions": {item["crop_id"]: item["caption_revision"] for item in records},
        "evaluation_gallery_crop_ids": [],
        "relevance_reviews": [],
    }


def _read_manifest(settings: dict[str, str], key: str) -> tuple[dict, bytes]:
    import boto3
    from botocore.client import Config

    client = boto3.client(
        "s3",
        endpoint_url=settings["GODS_MLOPS_TEST_S3_ENDPOINT"],
        aws_access_key_id=settings["GODS_MLOPS_TEST_S3_ACCESS_KEY"],
        aws_secret_access_key=settings["GODS_MLOPS_TEST_S3_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    response = client.get_object(Bucket=settings["GODS_MLOPS_TEST_S3_BUCKET"], Key=key)
    body = response["Body"].read()
    return json.loads(body), body


def test_publish_is_retryable_without_double_count_and_deletion_keeps_manifest_history() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    records, annotations = _seed_sources(target="detr")

    class LostAckStore(module.DatasetObjectStore):
        lost = False

        def write_immutable(self, **kwargs) -> None:
            super().write_immutable(**kwargs)
            if not self.lost:
                self.lost = True
                raise OSError("injected lost immutable-object acknowledgement")

    async def exercise() -> None:
        publisher = module.DatasetPublisher(
            database_url=settings["GODS_MLOPS_TEST_DATABASE_URL"],
            objects=_make_dataset_objects(settings, LostAckStore),
        )
        ingestion = PostgresIngestionRepository(database_url=settings["GODS_MLOPS_TEST_DATABASE_URL"])
        selection = _selection(records, "detr")
        try:
            initial_bytes = await ingestion.storage_bytes()
            with pytest.raises(OSError, match="lost immutable-object acknowledgement"):
                await publisher.publish_dataset(selection, "config-v1")
            reserved_bytes = await ingestion.storage_bytes()
            assert reserved_bytes > initial_bytes

            # Resume from a new repository instance using only the durable input identity.
            await publisher.close()
            publisher = module.DatasetPublisher(
                database_url=settings["GODS_MLOPS_TEST_DATABASE_URL"],
                objects=_make_dataset_objects(settings),
            )
            published = await publisher.publish_dataset(selection, "config-v1")
            assert published["state"] == "published"
            assert published["training_ready"] is True
            assert published["evaluation_eligible"] is True
            assert published["dataset_version"]
            assert len(published["manifest_hash"]) == 64
            assert await ingestion.storage_bytes() == reserved_bytes

            manifest, raw = _read_manifest(settings, published["manifest_object_key"])
            assert hashlib.sha256(raw).hexdigest() == published["manifest_hash"]
            assert manifest["dataset_version"] == published["dataset_version"]
            assert manifest["config"]["version"] == "config-v1"
            assert manifest["split_counts"]["train"]["frames"] == 20
            assert manifest["code"]["sha256"]

            linked_ids = [records[0]["sample_id"], records[-1]["sample_id"]]
            connection = await asyncpg.connect(settings["GODS_MLOPS_TEST_DATABASE_URL"])
            try:
                prior_splits = await connection.fetch(
                    "SELECT sample_id, split, group_id FROM dataset_sample_splits WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id",
                    [UUID(sample_id) for sample_id in linked_ids],
                )
            finally:
                await connection.close()

            await publisher.register_model_lineage(
                model_id="task7-model-registration-seam",
                dataset_version=published["dataset_version"],
            )
            late_selection = {
                **selection,
                "event_links": [{"link_id": "late-train-test-link", "sample_ids": linked_ids}],
            }
            blocked = await publisher.publish_dataset(late_selection, "config-v1")
            assert blocked["state"] == "blocked"
            assert "late_cross_boundary_link" in blocked["training_reasons"]

            # A new repository instance reads the persisted block and evaluation overlay.
            await publisher.close()
            publisher = module.DatasetPublisher(
                database_url=settings["GODS_MLOPS_TEST_DATABASE_URL"],
                objects=_make_dataset_objects(settings),
            )
            blocked_again = await publisher.publish_dataset(late_selection, "config-v1")
            assert blocked_again == blocked
            assert await ingestion.storage_bytes() == reserved_bytes
            await publisher.register_model_lineage(
                model_id="model-registered-after-leak",
                dataset_version=published["dataset_version"],
            )
            await publisher.register_model_lineage(
                model_id="model-registered-after-leak",
                dataset_version=published["dataset_version"],
            )

            connection = await asyncpg.connect(settings["GODS_MLOPS_TEST_DATABASE_URL"])
            try:
                current_splits = await connection.fetch(
                    "SELECT sample_id, split, group_id FROM dataset_sample_splits WHERE sample_id = ANY($1::uuid[]) ORDER BY sample_id",
                    [UUID(sample_id) for sample_id in linked_ids],
                )
                normalized = module._normalize_selection(late_selection)
                input_sha = hashlib.sha256(
                    json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    + b"\nconfig-v1\n"
                    + module.dataset_code_sha256().encode("ascii")
                ).hexdigest()
                leak_count = await connection.fetchval(
                    "SELECT count(*) FROM dataset_split_leakage_impacts WHERE input_sha256 = $1",
                    input_sha,
                )
                leak_sample_count = await connection.fetchval(
                    """
                    SELECT jsonb_array_length(sample_ids)
                    FROM dataset_split_leakage_impacts WHERE input_sha256 = $1
                    """,
                    input_sha,
                )
                row = await connection.fetchrow(
                    """
                    SELECT state, training_ready, evaluation_eligible, evaluation_reasons,
                           manifest_sha256
                    FROM dataset_versions WHERE dataset_version = $1
                    """,
                    published["dataset_version"],
                )
                overlay_count = await connection.fetchval(
                    "SELECT count(*) FROM dataset_version_leakage_impacts WHERE dataset_version = $1",
                    published["dataset_version"],
                )
                model_leak_count = await connection.fetchval(
                    """
                    SELECT count(*) FROM dataset_model_impacts
                    WHERE dataset_version = $1 AND reason = 'evaluation_split_leakage'
                    """,
                    published["dataset_version"],
                )
                late_model_impact_count = await connection.fetchval(
                    """
                    SELECT count(*) FROM dataset_model_impacts
                    WHERE model_id = 'model-registered-after-leak'
                      AND dataset_version = $1 AND reason = 'evaluation_split_leakage'
                    """,
                    published["dataset_version"],
                )
                assert current_splits == prior_splits
                assert leak_count == 1
                assert overlay_count == 1
                assert model_leak_count >= 1
                assert late_model_impact_count == leak_sample_count
                assert row["state"] == "published"
                assert row["training_ready"] is True
                assert row["evaluation_eligible"] is False
                assert "late_cross_boundary_link" in json.loads(row["evaluation_reasons"])
                assert row["manifest_sha256"].strip() == published["manifest_hash"]
            finally:
                await connection.close()

            sample_id = records[0]["sample_id"]
            impact = await publisher.invalidate_sample(sample_id)
            assert published["dataset_version"] in impact["datasets"]
            assert impact["models"] == [
                "model-registered-after-leak",
                "task7-model-registration-seam",
            ]
            assert impact["block_training"] is True
            assert impact["block_evaluation"] is True
            late_deleted_model = await publisher.register_model_lineage(
                model_id="model-registered-after-deletion",
                dataset_version=published["dataset_version"],
            )
            assert late_deleted_model["training_eligible"] is False
            assert late_deleted_model["evaluation_eligible"] is False
            deletion_impacts = [
                impact
                for impact in late_deleted_model["impacts"]
                if impact["reason"] == "source_sample_explicitly_invalidated"
            ]
            assert [impact["sample_id"] for impact in deletion_impacts] == [sample_id]
            after_invalidation = await publisher.publish_dataset(selection, "config-v1")
            assert after_invalidation["state"] == "invalidated"
            assert after_invalidation["manifest_hash"] == published["manifest_hash"]
            assert await ingestion.storage_bytes() == reserved_bytes
        finally:
            await publisher.close()
            await ingestion.close()
            await annotations.close()

    asyncio.run(exercise())


def test_bbox_edit_after_source_freeze_keeps_the_original_label_revision() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    records, annotations = _seed_sources(target="detr")

    class PauseFirstWrite(module.DatasetObjectStore):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.entered = threading.Event()
            self.release = threading.Event()
            self.paused = False

        def write_immutable(self, **kwargs) -> None:
            if not self.paused:
                self.paused = True
                self.entered.set()
                if not self.release.wait(timeout=15):
                    raise TimeoutError("test did not release the dataset upload")
            super().write_immutable(**kwargs)

    async def exercise() -> None:
        store = _make_dataset_objects(settings, PauseFirstWrite)
        publisher = module.DatasetPublisher(
            database_url=settings["GODS_MLOPS_TEST_DATABASE_URL"],
            objects=store,
        )
        selection = _selection(records, "detr")
        source = records[0]
        try:
            publish_task = asyncio.create_task(publisher.publish_dataset(selection, "bbox-race-v1"))
            assert await asyncio.to_thread(store.entered.wait, 10) is True
            assignment = await annotations.create_review_assignment(
                sample_id=UUID(source["sample_id"]),
                stage="bbox",
                label_studio_task_id=None,
                bbox_revision="operator-edit-draft",
                media_object_key=source["frame_object_key"],
                required_bytes=source["frame_size_bytes"],
            )
            assert assignment.state == "provisioning"
            store.release.set()

            published = await publish_task
            assert published["state"] == "published"
            manifest, raw = _read_manifest(settings, published["manifest_object_key"])
            assert hashlib.sha256(raw).hexdigest() == published["manifest_hash"]
            item = next(item for item in manifest["items"] if item["sample_id"] == source["sample_id"])
            assert item["snapshot"]["annotation"]["revision_id"] == source["bbox_revision"]
            assert item["snapshot"]["annotation"]["sha256"] == source["bbox_sha256"]
            assert item["snapshot"]["has_person"] is True
        finally:
            store.release.set()
            await publisher.close()
            await annotations.close()

    asyncio.run(exercise())


def test_new_caption_revision_gets_new_version_and_old_selection_keeps_its_manifest() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    records, annotations = _seed_sources(target="clip")

    class SubmittedCaption:
        def __init__(self, task: dict) -> None:
            self.task = task

        async def get_task(self, task_id: int) -> dict:
            assert task_id == self.task["id"]
            return self.task

    async def exercise() -> None:
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        ingestion = PostgresIngestionRepository(database_url=database_url)
        publisher = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        old_selection = _selection(records, "clip")
        try:
            original = await publisher.publish_dataset(old_selection, "config-caption-r1")
            assert original["state"] == "published"
            original_bytes = await ingestion.storage_bytes()

            edited = records[0]
            task_id = uuid4().int % 2_000_000_000 + 1
            annotation_id = uuid4().int % 2_000_000_000 + 1
            assignment = await annotations.create_review_assignment(
                sample_id=UUID(edited["sample_id"]),
                stage="caption",
                label_studio_task_id=task_id,
                bbox_revision=edited["bbox_revision"],
                media_object_key=edited["crop_object_key"],
                required_bytes=edited["crop_size_bytes"],
                expected_caption_revision_id=edited["caption_revision"],
            )
            replacement_text = "person wearing a green coat"
            task = {
                "id": task_id,
                "annotations": [
                    {
                        "id": annotation_id,
                        "was_cancelled": False,
                        "completed_by": {"id": 72, "email": "caption-reviewer@example.invalid"},
                        "result": [
                            {
                                "from_name": "caption",
                                "to_name": "image",
                                "type": "textarea",
                                "value": {"text": [replacement_text]},
                            }
                        ],
                    }
                ],
            }
            revised_caption = await AnnotationService(
                repository=annotations,
                label_studio=SubmittedCaption(task),
            ).finalize_annotation(edited["sample_id"], assignment.revision)
            new_selection = _selection(records, "clip")
            new_selection["caption_revisions"][edited["crop_id"]] = revised_caption["annotation_revision_id"]

            replacement = await publisher.publish_dataset(new_selection, "config-caption-r1")
            assert replacement["state"] == "published"
            assert replacement["dataset_version"] != original["dataset_version"]
            replacement_manifest, replacement_raw = _read_manifest(settings, replacement["manifest_object_key"])
            assert hashlib.sha256(replacement_raw).hexdigest() == replacement["manifest_hash"]
            new_item = next(
                item for item in replacement_manifest["items"]
                if item["snapshot"]["crop"]["crop_id"] == edited["crop_id"]
            )
            assert new_item["snapshot"]["caption"]["revision_id"] == revised_caption["annotation_revision_id"]
            assert new_item["snapshot"]["caption"]["text"] == replacement_text
            after_replacement_bytes = await ingestion.storage_bytes()
            assert after_replacement_bytes > original_bytes

            old_retry = await publisher.publish_dataset(old_selection, "config-caption-r1")
            assert old_retry["dataset_version"] == original["dataset_version"]
            assert old_retry["manifest_hash"] == original["manifest_hash"]
            old_manifest, old_raw = _read_manifest(settings, old_retry["manifest_object_key"])
            assert hashlib.sha256(old_raw).hexdigest() == original["manifest_hash"]
            old_item = next(
                item for item in old_manifest["items"]
                if item["snapshot"]["crop"]["crop_id"] == edited["crop_id"]
            )
            assert old_item["snapshot"]["caption"]["revision_id"] == edited["caption_revision"]
            assert await ingestion.storage_bytes() == after_replacement_bytes
        finally:
            await publisher.close()
            await ingestion.close()
            await annotations.close()

    asyncio.run(exercise())


def test_new_camera_after_authority_gets_no_new_test_assignments() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    established, established_annotations = _seed_sources(target="detr")
    new_camera, new_annotations = _seed_sources(target="detr")

    async def exercise() -> None:
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        first = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        try:
            assert (await first.publish_dataset(_selection(established, "detr"), "new-camera-authority-v1"))["state"] == "published"
        finally:
            await first.close()
            await established_annotations.close()
        second = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        try:
            new_version = await second.publish_dataset(_selection(new_camera, "detr"), "new-camera-authority-v2")
            assert new_version["state"] == "published"
            assert new_version["training_ready"] is True
            connection = await asyncpg.connect(database_url)
            try:
                splits = await connection.fetch(
                    "SELECT split FROM dataset_sample_splits WHERE sample_id = ANY($1::uuid[])",
                    [UUID(item["sample_id"]) for item in new_camera],
                )
                assert len(splits) == len(new_camera)
                assert {row["split"] for row in splits} <= {"train", "validation"}
            finally:
                await connection.close()
            assert new_version["evaluation_eligible"] is False
        finally:
            await second.close()
            await new_annotations.close()

    asyncio.run(exercise())


def test_disjoint_samples_on_historical_camera_day_inherit_without_old_ids_in_selection() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    existing, existing_annotations = _seed_sources(target="detr")
    added, added_annotations = _seed_sources(
        target="detr", count_per_day=20, capture_days=(1,), camera_id=UUID(existing[0]["camera_id"])
    )

    async def exercise() -> None:
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        first = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        try:
            assert (await first.publish_dataset(_selection(existing, "detr"), "same-day-authority-v1"))["state"] == "published"
        finally:
            await first.close()
            await existing_annotations.close()
        second = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        try:
            # This selection contains only new sample IDs from an already assigned camera/date.
            result = await second.publish_dataset(_selection(added, "detr"), "same-day-authority-v2")
            assert result["state"] == "published"
            assert result["training_ready"] is True
            assert result["evaluation_eligible"] is False
            connection = await asyncpg.connect(database_url)
            try:
                splits = await connection.fetch(
                    "SELECT split FROM dataset_sample_splits WHERE sample_id = ANY($1::uuid[])",
                    [UUID(item["sample_id"]) for item in added],
                )
                assert len(splits) == len(added)
                assert {row["split"] for row in splits} == {"train"}
            finally:
                await connection.close()
        finally:
            await second.close()
            await added_annotations.close()

    asyncio.run(exercise())


def test_historical_event_edge_expands_when_old_member_and_edge_are_omitted() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    camera_a, annotations_a = _seed_sources(target="detr")
    camera_b, annotations_b = _seed_sources(target="detr")
    event_id = "persisted-event-edge"
    new_day, annotations_new = _seed_sources(
        target="detr", capture_days=(4,), camera_id=UUID(camera_b[0]["camera_id"])
    )

    async def exercise() -> None:
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        first = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        initial = _selection(camera_a + camera_b, "detr")
        initial["event_links"] = [
            {"link_id": event_id, "sample_ids": [camera_a[0]["sample_id"], camera_b[0]["sample_id"]]}
        ]
        try:
            version = await first.publish_dataset(initial, "event-edge-authority-v1")
            assert version["state"] == "published"
            base_manifest, _ = _read_manifest(settings, version["manifest_object_key"])
            base_member_id = camera_b[0]["sample_id"]
            base_item = next(item for item in base_manifest["items"] if item["sample_id"] == base_member_id)
        finally:
            await first.close()
            await annotations_a.close()
            await annotations_b.close()

        second = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        selection = _selection(new_day, "detr")
        # The follow-up links the new frame to A's historical member, omitting B's historical member.
        selection["event_links"] = [
            {"link_id": event_id, "sample_ids": [new_day[0]["sample_id"], camera_a[0]["sample_id"]]}
        ]
        try:
            result = await second.publish_dataset(selection, "event-edge-authority-v2")
            assert result["state"] == "published"
            manifest, raw = _read_manifest(settings, result["manifest_object_key"])
            assert hashlib.sha256(raw).hexdigest() == result["manifest_hash"]
            new_item = next(item for item in manifest["items"] if item["sample_id"] == new_day[0]["sample_id"])
            assert new_item["component_id"] == base_item["component_id"]
            assert new_item["split"] == base_item["split"] == "train"
            connection = await asyncpg.connect(database_url)
            try:
                assert await connection.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM dataset_group_link_members
                        WHERE link_kind = 'event' AND link_id = $1 AND sample_id = $2
                    )
                    """,
                    event_id,
                    UUID(base_member_id),
                ) is True
            finally:
                await connection.close()
        finally:
            await second.close()
            await annotations_new.close()

    asyncio.run(exercise())


def test_explicit_source_invalidation_before_first_publication_is_a_durable_tombstone() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    records, annotations = _seed_sources(target="detr")

    async def exercise() -> None:
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        publisher = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        sample_id = records[0]["sample_id"]
        try:
            result = await publisher.invalidate_sample(sample_id)
            assert result["datasets"] == []
            assert result["models"] == []
            assert result["block_training"] is True
            assert result["block_evaluation"] is True

            for config_version in ("config-before-delete", "config-after-delete"):
                blocked = await publisher.publish_dataset(_selection(records, "detr"), config_version)
                assert blocked["state"] == "blocked"
                assert "source_explicitly_invalidated" in blocked["training_reasons"]
            connection = await asyncpg.connect(database_url)
            try:
                tombstone = await connection.fetchrow(
                    "SELECT reason FROM dataset_source_invalidations WHERE sample_id = $1",
                    UUID(sample_id),
                )
                assert tombstone["reason"] == "sample_explicitly_invalidated"
                assert await connection.fetchval("SELECT count(*) FROM dataset_versions") == 0
            finally:
                await connection.close()
        finally:
            await publisher.close()
            await annotations.close()

    asyncio.run(exercise())


def test_inflight_publication_rechecks_source_tombstone_before_ready_state() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    records, annotations = _seed_sources(target="detr")

    class PauseFirstWrite(module.DatasetObjectStore):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.entered = threading.Event()
            self.release = threading.Event()
            self.paused = False

        def write_immutable(self, **kwargs) -> None:
            if not self.paused:
                self.paused = True
                self.entered.set()
                if not self.release.wait(timeout=15):
                    raise TimeoutError("test did not release the dataset upload")
            super().write_immutable(**kwargs)

    async def exercise() -> None:
        store = _make_dataset_objects(settings, PauseFirstWrite)
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        publisher = module.DatasetPublisher(database_url=database_url, objects=store)
        selection = _selection(records, "detr")
        try:
            pending = asyncio.create_task(publisher.publish_dataset(selection, "inflight-delete-v1"))
            assert await asyncio.to_thread(store.entered.wait, 10) is True
            invalidation = await publisher.invalidate_sample(records[0]["sample_id"])
            assert invalidation["block_training"] is True
            store.release.set()

            finished = await pending
            assert finished["state"] == "invalidated"
            assert finished["training_ready"] is False
            assert finished["evaluation_eligible"] is False
            assert finished["manifest_hash"] is None

            changed_selection = _selection(records, "detr")
            blocked = await publisher.publish_dataset(changed_selection, "inflight-delete-v2")
            assert blocked["state"] == "blocked"
            assert "source_explicitly_invalidated" in blocked["training_reasons"]
        finally:
            store.release.set()
            await publisher.close()
            await annotations.close()

    asyncio.run(exercise())


def test_late_cross_split_link_during_upload_overlay_survives_finalize() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    records, annotations = _seed_sources(target="detr")

    class PauseFirstWrite(module.DatasetObjectStore):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.entered = threading.Event()
            self.release = threading.Event()
            self.paused = False

        def write_immutable(self, **kwargs) -> None:
            if not self.paused:
                self.paused = True
                self.entered.set()
                if not self.release.wait(timeout=15):
                    raise TimeoutError("test did not release the pending dataset upload")
            super().write_immutable(**kwargs)

    async def exercise() -> None:
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        store = _make_dataset_objects(settings, PauseFirstWrite)
        publisher = module.DatasetPublisher(database_url=database_url, objects=store)
        other_publisher = module.DatasetPublisher(
            database_url=database_url,
            objects=_make_dataset_objects(settings),
        )
        selection = _selection(records, "detr")
        pending_task = None
        try:
            pending_task = asyncio.create_task(publisher.publish_dataset(selection, "link-race-v1"))
            assert await asyncio.to_thread(store.entered.wait, 10) is True
            late_selection = {
                **selection,
                "event_links": [{
                    "link_id": "late-link-during-upload",
                    "sample_ids": [records[0]["sample_id"], records[-1]["sample_id"]],
                }],
            }
            blocked = await other_publisher.publish_dataset(late_selection, "link-race-v2")
            assert blocked["state"] == "blocked"
            store.release.set()

            finalized = await pending_task
            assert finalized["state"] == "published"
            assert finalized["training_ready"] is True
            assert finalized["evaluation_eligible"] is False
            assert finalized["manifest_hash"]
            connection = await asyncpg.connect(database_url)
            try:
                row = await connection.fetchrow(
                    """
                    SELECT state, training_ready, evaluation_eligible, evaluation_reasons, manifest_sha256
                    FROM dataset_versions WHERE dataset_version = $1
                    """,
                    finalized["dataset_version"],
                )
                overlay_count = await connection.fetchval(
                    "SELECT count(*) FROM dataset_version_leakage_impacts WHERE dataset_version = $1",
                    finalized["dataset_version"],
                )
                assert row["state"] == "published"
                assert row["training_ready"] is True
                assert row["evaluation_eligible"] is False
                assert "late_cross_boundary_link" in json.loads(row["evaluation_reasons"])
                assert row["manifest_sha256"].strip() == finalized["manifest_hash"]
                assert overlay_count == 1
            finally:
                await connection.close()
        finally:
            store.release.set()
            if pending_task is not None and not pending_task.done():
                await pending_task
            await publisher.close()
            await other_publisher.close()
            await annotations.close()

    asyncio.run(exercise())


def test_clip_crop_survives_expired_parent_and_missing_truth_is_reported_separately() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    records, annotations = _seed_sources(target="clip")

    async def exercise() -> None:
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        ingestion = PostgresIngestionRepository(database_url=database_url)
        source_objects = S3SampleStore(
            endpoint_url=settings["GODS_MLOPS_TEST_S3_ENDPOINT"],
            access_key=settings["GODS_MLOPS_TEST_S3_ACCESS_KEY"],
            secret_key=settings["GODS_MLOPS_TEST_S3_SECRET_KEY"],
            bucket=settings["GODS_MLOPS_TEST_S3_BUCKET"],
            region="us-east-1",
        )
        publisher = module.DatasetPublisher(
            database_url=database_url,
            objects=_make_dataset_objects(settings),
        )
        expired = records[0]
        connection = await asyncpg.connect(database_url)
        try:
            await annotations.adopt_for_dataset(
                dataset_version="task5-expiry-contract-fixture",
                target="clip",
                sample_id=UUID(expired["sample_id"]),
                bbox_revision=UUID(expired["bbox_revision"]),
                crop_id=UUID(expired["crop_id"]),
                caption_revision=UUID(expired["caption_revision"]),
            )
            await connection.execute(
                "UPDATE ingestion_samples SET retention_until = now() - interval '1 minute' WHERE sample_id = $1",
                UUID(expired["sample_id"]),
            )
        finally:
            await connection.close()
        claims = await ingestion.claim_expired(now=datetime.now(timezone.utc))
        claim = next(item for item in claims if str(item.sample_id) == expired["sample_id"])
        source_objects.delete_object(claim.object_key)
        assert await ingestion.finish_expiry(UUID(expired["sample_id"])) is True
        with pytest.raises(FileNotFoundError):
            source_objects.read_object(
                object_key=expired["frame_object_key"],
                expected_sha256=expired["frame_sha256"],
            )

        crop_rows = await annotations.crops_for_revision(
            sample_id=UUID(expired["sample_id"]),
            bbox_revision=UUID(expired["bbox_revision"]),
        )
        state = next(item for item in crop_rows if item["crop_id"] == expired["crop_id"])
        assert state["parent_available"] is False
        assert state["parent_regenerable"] is False
        selection = _selection(records, "clip")
        try:
            published = await publisher.publish_dataset(selection, "config-v1")
            assert published["state"] == "published"
            assert published["training_ready"] is True
            assert published["evaluation_eligible"] is False
            assert "human_relevance_truth_missing" in published["evaluation_reasons"]
            manifest, raw = _read_manifest(settings, published["manifest_object_key"])
            assert hashlib.sha256(raw).hexdigest() == published["manifest_hash"]
            expired_item = next(
                item for item in manifest["items"] if item["snapshot"]["crop"]["crop_id"] == expired["crop_id"]
            )
            assert expired_item["snapshot"]["crop"]["parent_available"] is False
            assert expired_item["snapshot"]["crop"]["regeneration_available"] is False
            assert all(item["object"]["key"] != expired["frame_object_key"] for item in manifest["items"])
        finally:
            await publisher.close()
            await ingestion.close()
            await annotations.close()

    asyncio.run(exercise())


def test_complete_relevance_freeze_rolls_back_with_shared_quota_failure() -> None:
    settings = _settings()
    if settings is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    module = _module("gods_mlops.datasets.publish")
    records, annotations = _seed_sources(target="clip")

    async def exercise() -> None:
        database_url = settings["GODS_MLOPS_TEST_DATABASE_URL"]
        ingestion = PostgresIngestionRepository(database_url=database_url)
        publisher = module.DatasetPublisher(database_url=database_url, objects=_make_dataset_objects(settings))
        gallery_records = records[-20:]
        gallery = [{"crop_id": item["crop_id"], "sha256": item["crop_sha256"]} for item in gallery_records]
        review = await annotations.prepare_relevance_review(
            query_id="night-shift-query",
            query_text="a person wearing a red jacket",
            query_revision="caption-query-r1",
            gallery=gallery,
        )
        relevance = {
            "review_id": review["review_id"],
            "judgments": [
                {"crop_id": item["crop_id"], "judgment": "relevant" if index == 0 else "not_relevant"}
                for index, item in enumerate(gallery_records)
            ],
            "provenance": {
                "source": "label_studio",
                "label_studio_annotation_id": 987654,
                "reviewer_id": "task6-human-fixture",
            },
        }
        selection = _selection(records, "clip")
        selection["evaluation_gallery_crop_ids"] = [item["crop_id"] for item in gallery_records]
        selection["relevance_reviews"] = [relevance]
        connection = await asyncpg.connect(database_url)
        before_bytes = await ingestion.storage_bytes()
        try:
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = $1 WHERE singleton = TRUE",
                GLOBAL_OBJECT_LIMIT - 1,
            )
            with pytest.raises(module.DatasetQuotaExceededError):
                await publisher.publish_dataset(selection, "config-v1")
            review_state = await connection.fetchval(
                "SELECT state FROM relevance_matrix_review_drafts WHERE review_id = $1",
                UUID(review["review_id"]),
            )
            assert review_state == "pending"
            normalized = module._normalize_selection(selection)
            input_sha = hashlib.sha256(
                json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\nconfig-v1\n"
                + module.dataset_code_sha256().encode("ascii")
            ).hexdigest()
            dataset_version = f"dataset-{input_sha[:24]}"
            assert await connection.fetchval(
                "SELECT count(*) FROM dataset_versions WHERE input_sha256 = $1",
                input_sha,
            ) == 0
            assert await connection.fetchval(
                "SELECT count(*) FROM dataset_adoptions WHERE dataset_version = $1",
                dataset_version,
            ) == 0
            await connection.execute(
                "UPDATE ingestion_storage_usage SET used_bytes = $1 WHERE singleton = TRUE",
                before_bytes,
            )
        finally:
            await connection.close()

        result = await publisher.publish_dataset(selection, "config-v1")
        assert result["evaluation_eligible"] is True
        frozen = await annotations.relevance_revision(result["relevance_revision_ids"][0])
        assert frozen["status"] == "complete"
        assert frozen["evaluation_eligible"] is True
        assert len(frozen["judgments"]) == len(gallery_records)
        await publisher.close()
        await ingestion.close()
        await annotations.close()

    asyncio.run(exercise())
