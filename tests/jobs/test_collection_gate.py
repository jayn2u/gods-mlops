from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
import time
from functools import partial
from urllib import error as urllib_error
from urllib import request as urllib_request
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import FastAPI
import uvicorn
from conftest import seed_training_ready_dataset

from gods_mlops.ingestion.collection_gate import (
    CollectionPausedError,
    CollectionStorageGate,
)
from gods_mlops.ingestion.schemas import CandidateMetadata, CandidateReason
from gods_mlops.ingestion.service import IngestionService
from gods_mlops.ingestion.storage import PostgresIngestionRepository
from gods_mlops.ingestion.routes import build_ingestion_router
from gods_mlops.jobs.admission import UBUNTU_STORAGE_MIN_FREE_BYTES
from gods_mlops.jobs.models import ResourceObservation
from gods_mlops.jobs.queue import PostgresJobQueueRepository

GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"


class MemorySampleStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def ensure_object(self, *, object_key: str, image: bytes, expected_sha256: str) -> None:
        assert hashlib.sha256(image).hexdigest() == expected_sha256
        self.objects.setdefault(object_key, image)

    def ready(self) -> None:
        return None

    def delete_object(self, object_key: str) -> None:
        self.objects.pop(object_key, None)


def _candidate(image: bytes = b"bounded candidate frame") -> tuple[CandidateMetadata, bytes]:
    return CandidateMetadata(
        sample_id=uuid4(),
        camera_id=uuid4(),
        captured_at_utc="2026-10-05T12:00:00Z",
        reason=CandidateReason.OPERATOR,
        sha256=hashlib.sha256(image).hexdigest(),
        model_revision="detector-test-revision",
        processor_revision="processor-test-revision",
    ), image


def _observation(*, available_bytes: int, observed_at: datetime | None = None) -> ResourceObservation:
    return ResourceObservation(
        observation_id=str(uuid4()),
        node_id="ubuntu",
        hostname="ubuntu",
        host_identity=HOST_IDENTITY,
        gpu_name="NVIDIA RTX A6000",
        gpu_uuid=GPU_UUID,
        free_mib=48_000,
        total_mib=49_140,
        gpu_processes=(),
        gpu_process_list_complete=True,
        process_table=(),
        process_table_complete=True,
        storage_path=STORAGE_PATH,
        filesystem_identity=FILESYSTEM_IDENTITY,
        filesystem_available_bytes=available_bytes,
        observed_at=observed_at or datetime.now(UTC),
    )


async def _gate(database_url: str) -> tuple[PostgresJobQueueRepository, CollectionStorageGate]:
    await seed_training_ready_dataset(database_url)
    repository = PostgresJobQueueRepository(database_url=database_url)
    await repository.ensure_schema()
    return repository, CollectionStorageGate(
        repository=repository,
        expected_host_identity=HOST_IDENTITY,
        expected_gpu_uuid=GPU_UUID,
        expected_filesystem_identity=FILESYSTEM_IDENTITY,
        expected_storage_path=STORAGE_PATH,
    )


def test_new_receipt_without_current_ubuntu_storage_observation_pauses_before_reservation(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        ingestion = PostgresIngestionRepository(database_url=task7_database_url)
        await ingestion.ensure_schema()
        resource_repository, gate = await _gate(task7_database_url)
        service = IngestionService(
            repository=ingestion,
            objects=MemorySampleStore(),
            collection_gate=gate,
        )
        metadata, image = _candidate()
        before = await ingestion.storage_bytes()
        with pytest.raises(CollectionPausedError) as paused:
            await service.receive(metadata, image)
        assert paused.value.reason_code == "ubuntu_observation_unavailable"
        assert paused.value.retryable is True
        assert paused.value.as_detail() == {
            "code": "collection_paused",
            "reason": "ubuntu_observation_unavailable",
            "retryable": True,
        }
        assert await ingestion.count_sample(metadata.sample_id) == 0
        assert await ingestion.storage_bytes() == before
        await ingestion.close()
        await resource_repository.close()

    asyncio.run(exercise())


def test_filesystem_headroom_pause_is_distinct_and_does_not_charge_the_ledger(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        ingestion = PostgresIngestionRepository(database_url=task7_database_url)
        await ingestion.ensure_schema()
        resource_repository, gate = await _gate(task7_database_url)
        await resource_repository.record_observation(
            _observation(available_bytes=UBUNTU_STORAGE_MIN_FREE_BYTES - 1)
        )
        service = IngestionService(
            repository=ingestion,
            objects=MemorySampleStore(),
            collection_gate=gate,
        )
        metadata, image = _candidate()
        with pytest.raises(CollectionPausedError) as paused:
            await service.receive(metadata, image)
        assert paused.value.reason_code == "ubuntu_filesystem_headroom_below_minimum"
        assert await ingestion.count_sample(metadata.sample_id) == 0
        assert await ingestion.storage_bytes() == 0
        await ingestion.close()
        await resource_repository.close()

    asyncio.run(exercise())


def test_lost_ack_duplicate_receipt_survives_a_stale_observer_gate(task7_database_url: str) -> None:
    async def exercise() -> None:
        ingestion = PostgresIngestionRepository(database_url=task7_database_url)
        await ingestion.ensure_schema()
        resource_repository, gate = await _gate(task7_database_url)
        observation = _observation(available_bytes=2 * UBUNTU_STORAGE_MIN_FREE_BYTES)
        await resource_repository.record_observation(observation)
        objects = MemorySampleStore()
        first_service = IngestionService(repository=ingestion, objects=objects)
        metadata, image = _candidate()
        receipt = await first_service.receive(metadata, image)
        charged = await ingestion.storage_bytes()

        await resource_repository.record_observation_failure(
            node_id="ubuntu", failure_code="observer_unreachable"
        )
        gated_retry_service = IngestionService(
            repository=ingestion,
            objects=objects,
            collection_gate=gate,
        )
        retry = await gated_retry_service.receive(metadata, image)
        assert retry == receipt
        assert await ingestion.storage_bytes() == charged
        assert await ingestion.count_sample(metadata.sample_id) == 1
        await ingestion.close()
        await resource_repository.close()

    asyncio.run(exercise())


def test_receiver_http_response_surfaces_pause_reason_and_retryability() -> None:
    token = "task7-collection-gate-http-test-token-012345"

    class PausedReceiver:
        async def receive(self, _metadata, _image):
            raise CollectionPausedError(
                "ubuntu_filesystem_headroom_below_minimum",
                details={"available_bytes": 1, "minimum_bytes": UBUNTU_STORAGE_MIN_FREE_BYTES},
            )

    app = FastAPI()
    app.include_router(build_ingestion_router(service=PausedReceiver(), bearer_token=token))
    image = b"route gate test bytes"
    metadata = CandidateMetadata(
        sample_id=uuid4(),
        camera_id=uuid4(),
        captured_at_utc=datetime.now(UTC),
        reason="operator",
        sha256=hashlib.sha256(image).hexdigest(),
        model_revision="detector-test-revision",
        processor_revision="processor-test-revision",
    )
    boundary = "gods-task7-collection-pause"
    fields = {
        "sample_id": str(metadata.sample_id),
        "camera_id": str(metadata.camera_id),
        "captured_at_utc": metadata.captured_at_utc.isoformat(),
        "reason": metadata.reason.value,
        "sha256": metadata.sha256,
        "model_revision": metadata.model_revision,
        "processor_revision": metadata.processor_revision,
    }
    parts = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode(),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="image"; filename="candidate.jpg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            image,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    body = b"".join(parts)

    with socket.socket() as listen_socket:
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind(("127.0.0.1", 0))
        listen_socket.listen(128)
        listen_socket.setblocking(False)
        port = int(listen_socket.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(app, log_level="critical", access_log=False))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listen_socket]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.025)
            assert server.started
            request = urllib_request.Request(
                f"http://127.0.0.1:{port}/api/samples",
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                method="POST",
            )
            with pytest.raises(urllib_error.HTTPError) as response:
                urllib_request.urlopen(request, timeout=10)
            assert response.value.code == 503
            assert response.value.headers.get("Retry-After") == "5"
            detail = json.loads(response.value.read())["detail"]
            assert detail["code"] == "collection_paused"
            assert detail["reason"] == "ubuntu_filesystem_headroom_below_minimum"
            assert detail["retryable"] is True
        finally:
            server.should_exit = True
            thread.join(timeout=10)
