from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
from uuid import uuid4

import pytest

from gods_mlops.annotations.service import AnnotationService
from gods_mlops.annotations.storage import PostgresAnnotationRepository
from gods_mlops.datasets.publish import DatasetPublicationError, DatasetPublisher
from gods_mlops.datasets.deletion import preview_invalidate_sample
from gods_mlops.ingestion.service import IngestionService
from gods_mlops.ingestion.storage import PostgresIngestionRepository


class _Acquire:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class _Pool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _Acquire(self.connection)


class _ListingConnection:
    def __init__(self, *, samples=(), reviews=(), publications=(), bbox_revisions=(), captions=()):
        self.samples = list(samples)
        self.reviews = list(reviews)
        self.publications = list(publications)
        self.bbox_revisions = list(bbox_revisions)
        self.captions = list(captions)
        self.fetch_queries = []

    async def fetch(self, query, *args):
        self.fetch_queries.append(query)
        if "FROM annotation_crops AS crop" in query:
            if "sample.state IN ('received', 'purge_pending', 'expired')" in query:
                return [
                    row
                    for row in self.captions
                    if row.get("parent_sample_state") in {"received", "purge_pending", "expired"}
                ]
            return [row for row in self.captions if row.get("parent_sample_state", "received") == "received"]
        if "FROM ingestion_samples" in query:
            return self.samples
        if "FROM review_assignments" in query:
            return self.reviews
        if "FROM dataset_versions" in query:
            return self.publications
        if "FROM sample_annotation_heads" in query:
            return self.bbox_revisions
        if "FROM annotation_crops" in query:
            return self.captions
        raise AssertionError(f"unexpected listing query: {query}")


def test_ingestion_service_lists_candidate_status_without_object_store_details() -> None:
    async def exercise() -> None:
        sample_id = uuid4()
        camera_id = uuid4()
        repository = PostgresIngestionRepository(database_url="postgresql://unused")
        repository._pool = _Pool(
            _ListingConnection(
                samples=[
                    {
                        "sample_id": sample_id,
                        "camera_id": camera_id,
                        "captured_at_utc": datetime(2026, 10, 7, tzinfo=UTC),
                        "reason": "operator",
                        "sha256": "a" * 64,
                        "state": "received",
                        "selected": False,
                        "object_size_bytes": 128,
                        "last_failure_code": None,
                        "received_at": datetime(2026, 10, 7, tzinfo=UTC),
                        "retention_until": None,
                    }
                ]
            )
        )
        service = IngestionService(repository=repository, objects=object(), collection_gate=None)

        samples = await service.list_samples(limit=10, state="received")

        assert samples == [
            {
                "sample_id": str(sample_id),
                "camera_id": str(camera_id),
                "captured_at_utc": "2026-10-07T00:00:00+00:00",
                "reason": "operator",
                "sha256": "a" * 64,
                "state": "received",
                "selected": False,
                "object_size_bytes": 128,
                "last_failure_code": None,
                "received_at": "2026-10-07T00:00:00+00:00",
                "retention_until": None,
            }
        ]
        assert "object_key" not in samples[0]

    asyncio.run(exercise())


def test_annotation_service_lists_review_state_and_keeps_label_provenance_explicit() -> None:
    async def exercise() -> None:
        sample_id, revision = uuid4(), uuid4()
        provenance = {
            "source": "label_studio",
            "label_studio_annotation_id": 88,
            "completed_by": {"id": 5, "email": "reviewer@example.invalid"},
        }
        repository = PostgresAnnotationRepository(database_url="postgresql://unused")
        repository._pool = _Pool(
            _ListingConnection(
                reviews=[
                    {
                        "sample_id": sample_id,
                        "revision": revision,
                        "stage": "bbox",
                        "assignment_state": "finalized",
                        "label_studio_task_id": 44,
                        "project_id": 7,
                        "created_at": datetime(2026, 10, 7, tzinfo=UTC),
                        "finished_at": datetime(2026, 10, 7, tzinfo=UTC),
                        "annotation_revision_id": uuid4(),
                        "provenance": json.dumps(provenance),
                    }
                ]
            )
        )
        service = AnnotationService(repository=repository, label_studio=object())

        reviews = await service.list_reviews(limit=10)

        assert len(reviews) == 1
        assert reviews[0]["sample_id"] == str(sample_id)
        assert reviews[0]["revision"] == str(revision)
        assert reviews[0]["state"] == "finalized"
        assert reviews[0]["provenance"] == provenance
        assert reviews[0]["human_review_recorded"] is True

    asyncio.run(exercise())


def test_dataset_publication_selection_pins_current_review_revisions_and_lists_gates() -> None:
    async def exercise() -> None:
        sample_id, bbox_revision = uuid4(), uuid4()
        dataset_publisher = DatasetPublisher(database_url="postgresql://unused", objects=object())
        dataset_publisher._schema_ready = True
        dataset_publisher._pool = _Pool(
            _ListingConnection(
                bbox_revisions=[
                    {
                        "sample_id": sample_id,
                        "bbox_revision": bbox_revision,
                        "current_bbox_assignment_revision": None,
                    }
                ],
                publications=[
                    {
                        "dataset_version": "dataset-123",
                        "manifest_sha256": "b" * 64,
                        "manifest_object_key": "datasets/dataset-123/manifest.json",
                        "split_counts": {"train": {"frames": 20}},
                        "state": "published",
                        "training_ready": True,
                        "training_reasons": [],
                        "evaluation_eligible": False,
                        "evaluation_reasons": ["human_relevance_truth_missing"],
                        "relevance_revision_ids": [],
                        "published_at": datetime(2026, 10, 7, tzinfo=UTC),
                    }
                ],
            )
        )

        selection = await dataset_publisher.publication_selection(
            target="detr",
            sample_ids=[str(sample_id)],
        )
        publications = await dataset_publisher.list_publications(limit=5)

        assert selection["target"] == "detr"
        assert selection["sample_ids"] == [str(sample_id)]
        assert selection["bbox_revisions"] == {str(sample_id): str(bbox_revision)}
        assert selection["crop_ids"] == []
        assert publications[0]["training_ready"] is True
        assert publications[0]["evaluation_eligible"] is False
        assert publications[0]["evaluation_reasons"] == ["human_relevance_truth_missing"]

    asyncio.run(exercise())


def test_dataset_selection_rejects_samples_without_a_finalized_review_revision() -> None:
    async def exercise() -> None:
        dataset_publisher = DatasetPublisher(database_url="postgresql://unused", objects=object())
        dataset_publisher._schema_ready = True
        dataset_publisher._pool = _Pool(_ListingConnection(bbox_revisions=[]))

        with pytest.raises(DatasetPublicationError, match="finalized bbox review"):
            await dataset_publisher.publication_selection(target="detr", sample_ids=[str(uuid4())])

    asyncio.run(exercise())


def test_dataset_candidate_list_distinguishes_reviewed_samples_and_crops() -> None:
    async def exercise() -> None:
        sample_id = uuid4()
        crop_id = uuid4()
        bbox_revision = uuid4()
        caption_revision = uuid4()
        connection = _ListingConnection(
            samples=[
                {
                    "sample_id": sample_id,
                    "camera_id": uuid4(),
                    "captured_at_utc": datetime(2026, 10, 7, tzinfo=UTC),
                    "reason": "operator",
                    "state": "received",
                    "selected": False,
                    "object_size_bytes": 128,
                    "latest_bbox_revision": bbox_revision,
                    "current_bbox_assignment_revision": None,
                }
            ],
            captions=[
                {
                    "crop_id": crop_id,
                    "sample_id": sample_id,
                    "bbox_revision": bbox_revision,
                    "caption_revision_id": caption_revision,
                    "state": "ready",
                    "caption_state": "reviewed",
                    "sha256": "a" * 64,
                    "parent_sample_state": "expired",
                }
            ],
        )
        dataset_publisher = DatasetPublisher(database_url="postgresql://unused", objects=object())
        dataset_publisher._schema_ready = True
        dataset_publisher._pool = _Pool(connection)

        candidates = await dataset_publisher.list_publication_candidates(limit=10)

        assert candidates["samples"][0]["sample_id"] == str(sample_id)
        assert candidates["samples"][0]["bbox_reviewed"] is True
        assert candidates["crops"] == [
            {
                "crop_id": str(crop_id),
                "sample_id": str(sample_id),
                "bbox_revision": str(bbox_revision),
                "caption_revision_id": str(caption_revision),
                "state": "ready",
                "caption_state": "reviewed",
                "sha256": "a" * 64,
            }
        ]
        assert any(
            "sample.state IN ('received', 'purge_pending', 'expired')" in query
            for query in connection.fetch_queries
        )

    asyncio.run(exercise())


def test_invalidation_preview_reports_impact_without_mutating_source_state() -> None:
    class Publisher:
        def __init__(self):
            self.preview_calls = []
            self.invalidate_calls = []

        async def preview_sample_invalidation(self, sample_id):
            self.preview_calls.append(sample_id)
            return {
                "sample_id": sample_id,
                "physical_deletion": False,
                "datasets": ["dataset-123"],
                "models": ["model-current"],
                "active_review_count": 1,
                "active_job_count": 2,
                "block_training": True,
                "block_evaluation": True,
            }

        async def invalidate_sample(self, sample_id):
            self.invalidate_calls.append(sample_id)
            raise AssertionError("preview must not invalidate")

    async def exercise() -> None:
        publisher = Publisher()
        sample_id = str(uuid4())

        impact = await preview_invalidate_sample(sample_id, publisher=publisher)

        assert impact["physical_deletion"] is False
        assert impact["datasets"] == ["dataset-123"]
        assert impact["models"] == ["model-current"]
        assert impact["active_review_count"] == 1
        assert impact["active_job_count"] == 2
        assert publisher.preview_calls == [sample_id]
        assert publisher.invalidate_calls == []

    asyncio.run(exercise())
