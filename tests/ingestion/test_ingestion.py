from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import threading
import time
from urllib import error as urllib_error
from urllib import request as urllib_request
from functools import partial
from uuid import UUID, uuid4

import anyio
import asyncpg
import pytest

from gods_mlops.ingestion.schemas import (
    CandidateMetadata,
    GlobalObjectLimitError,
    SampleConflictError,
    SampleExpiredError,
    SampleReceipt,
    SampleStorageError,
)
from gods_mlops.ingestion.app import IngestionSettings, build_app
from gods_mlops.ingestion.service import IngestionService
from gods_mlops.ingestion.schemas import CandidateReason
from gods_mlops.ingestion.storage import PostgresIngestionRepository, S3SampleStore


def _configured_backends(
    *,
    max_object_bytes: int = 1024**4,
) -> tuple[PostgresIngestionRepository, S3SampleStore, str] | None:
    database_url = os.environ.get("GODS_MLOPS_TEST_DATABASE_URL")
    endpoint_url = os.environ.get("GODS_MLOPS_TEST_S3_ENDPOINT")
    access_key = os.environ.get("GODS_MLOPS_TEST_S3_ACCESS_KEY")
    secret_key = os.environ.get("GODS_MLOPS_TEST_S3_SECRET_KEY")
    bucket = os.environ.get("GODS_MLOPS_TEST_S3_BUCKET")
    if not all((database_url, endpoint_url, access_key, secret_key, bucket)):
        return None
    repository = PostgresIngestionRepository(
        database_url=database_url,
        max_object_bytes=max_object_bytes,
    )
    objects = S3SampleStore(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region="us-east-1",
    )
    return repository, objects, database_url


def test_lost_ack_is_idempotent() -> None:
    configured = _configured_backends()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    repository, objects, _database_url = configured
    service = IngestionService(repository=repository, objects=objects)
    image = b"small deterministic jpeg fixture"

    metadata = CandidateMetadata(
        sample_id=uuid4(),
        camera_id=uuid4(),
        captured_at_utc="2026-10-05T00:00:00Z",
        reason="periodic",
        sha256=hashlib.sha256(image).hexdigest(),
        model_revision="runtime-detector-abcdef",
        processor_revision="runtime-processor-123456",
    )

    async def exercise() -> None:
        await repository.ensure_schema()
        first = await service.receive(metadata, image)
        retry = await service.receive(metadata, image)
        assert retry == first
        assert await repository.count_sample(metadata.sample_id) == 1
        assert objects.read_verified(metadata, image) is True
        await repository.close()

    asyncio.run(exercise())


def test_conflicting_bytes_for_sample_id_are_rejected() -> None:
    configured = _configured_backends()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    repository, objects, _database_url = configured
    service = IngestionService(repository=repository, objects=objects)
    first_bytes = b"candidate bytes A"
    second_bytes = b"candidate bytes B"
    sample_id = uuid4()
    base = {
        "sample_id": sample_id,
        "camera_id": uuid4(),
        "captured_at_utc": "2026-10-05T00:00:00Z",
        "reason": "operator",
        "model_revision": "runtime-detector-abcdef",
        "processor_revision": "runtime-processor-123456",
    }
    first = CandidateMetadata(**base, sha256=hashlib.sha256(first_bytes).hexdigest())
    conflict = CandidateMetadata(**base, sha256=hashlib.sha256(second_bytes).hexdigest())

    async def exercise() -> None:
        await repository.ensure_schema()
        _receipt = await service.receive(first, first_bytes)
        with pytest.raises(SampleConflictError):
            _receipt = await service.receive(conflict, second_bytes)
        assert await repository.count_sample(sample_id) == 1
        await repository.close()

    asyncio.run(exercise())


def test_metadata_commit_failure_after_s3_write_is_repaired() -> None:
    configured = _configured_backends()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    _repository, objects, database_url = configured

    class FailFirstReceiptRepository(PostgresIngestionRepository):
        failed_once = False

        async def mark_received(self, sample_id: UUID) -> SampleReceipt:
            if not self.failed_once:
                self.failed_once = True
                raise RuntimeError("injected metadata commit failure")
            return await super().mark_received(sample_id)

    flaky_repository = FailFirstReceiptRepository(database_url=database_url)
    service = IngestionService(repository=flaky_repository, objects=objects)
    image = b"image object verifies before receipt metadata"
    metadata = CandidateMetadata(
        sample_id=uuid4(),
        camera_id=uuid4(),
        captured_at_utc="2026-10-05T00:00:00Z",
        reason="low_confidence",
        sha256=hashlib.sha256(image).hexdigest(),
        model_revision="runtime-detector-abcdef",
        processor_revision="runtime-processor-123456",
    )

    async def exercise() -> None:
        await flaky_repository.ensure_schema()
        before_bytes = await flaky_repository.storage_bytes()
        with pytest.raises(SampleStorageError):
            _receipt = await service.receive(metadata, image)
        assert objects.read_verified(metadata, image) is True
        assert await flaky_repository.daily_count(
            metadata.camera_id,
            metadata.captured_at_utc.date(),
        ) == 1
        receipt = await service.receive(metadata, image)
        assert receipt.sample_id == metadata.sample_id
        assert receipt.sha256 == metadata.sha256
        assert await flaky_repository.storage_bytes() == before_bytes + len(image)
        await flaky_repository.close()

    asyncio.run(exercise())


def test_s3_write_that_loses_its_response_is_repaired_without_duplicate_storage() -> None:
    configured = _configured_backends()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    repository, objects, _database_url = configured

    class FailAfterFirstPutStore(S3SampleStore):
        failed_once = False

        def ensure_object(self, *, object_key: str, image: bytes, expected_sha256: str) -> None:
            super().ensure_object(
                object_key=object_key,
                image=image,
                expected_sha256=expected_sha256,
            )
            if not self.failed_once:
                self.failed_once = True
                raise OSError("injected lost S3 write response")

    flaky_objects = FailAfterFirstPutStore(
        endpoint_url=os.environ["GODS_MLOPS_TEST_S3_ENDPOINT"],
        access_key=os.environ["GODS_MLOPS_TEST_S3_ACCESS_KEY"],
        secret_key=os.environ["GODS_MLOPS_TEST_S3_SECRET_KEY"],
        bucket=os.environ["GODS_MLOPS_TEST_S3_BUCKET"],
        region="us-east-1",
    )
    service = IngestionService(repository=repository, objects=flaky_objects)
    image = b"S3 persisted but first response was lost"
    metadata = CandidateMetadata(
        sample_id=uuid4(),
        camera_id=uuid4(),
        captured_at_utc="2026-10-05T00:00:00Z",
        reason="operator",
        sha256=hashlib.sha256(image).hexdigest(),
        model_revision="runtime-detector-abcdef",
        processor_revision="runtime-processor-123456",
    )

    async def exercise() -> None:
        await repository.ensure_schema()
        before_bytes = await repository.storage_bytes()
        with pytest.raises(SampleStorageError):
            _receipt = await service.receive(metadata, image)
        assert flaky_objects.read_verified(metadata, image) is True
        receipt = await service.receive(metadata, image)
        assert receipt.sample_id == metadata.sample_id
        assert await repository.count_sample(metadata.sample_id) == 1
        assert await repository.daily_count(metadata.camera_id, metadata.captured_at_utc.date()) == 1
        assert await repository.storage_bytes() == before_bytes + len(image)
        await repository.close()

    asyncio.run(exercise())


def test_global_object_cap_is_checked_after_sample_id_idempotency() -> None:
    image = b"larger than injected object limit"
    configured = _configured_backends(max_object_bytes=len(image) - 1)
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    repository, objects, _database_url = configured
    service = IngestionService(repository=repository, objects=objects)
    metadata = CandidateMetadata(
        sample_id=uuid4(),
        camera_id=uuid4(),
        captured_at_utc="2026-10-05T00:00:00Z",
        reason="operator",
        sha256=hashlib.sha256(image).hexdigest(),
        model_revision="runtime-detector-abcdef",
        processor_revision="runtime-processor-123456",
    )

    async def exercise() -> None:
        await repository.ensure_schema()
        before_bytes = await repository.storage_bytes()
        with pytest.raises(GlobalObjectLimitError):
            _receipt = await service.receive(metadata, image)
        with pytest.raises(GlobalObjectLimitError):
            _receipt = await service.receive(metadata, image)
        assert await repository.count_sample(metadata.sample_id) == 1
        assert await repository.daily_count(metadata.camera_id, metadata.captured_at_utc.date()) == 1
        assert await repository.storage_bytes() == before_bytes
        assert objects.read_verified(metadata, image) is False
        await repository.close()

    asyncio.run(exercise())


def test_retention_deletes_unselected_object_but_keeps_idempotency_tombstone() -> None:
    configured = _configured_backends()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    repository, objects, database_url = configured
    service = IngestionService(repository=repository, objects=objects)
    image = b"unselected candidate expires after seven days"
    metadata = CandidateMetadata(
        sample_id=uuid4(),
        camera_id=uuid4(),
        captured_at_utc="2026-10-05T00:00:00Z",
        reason="periodic",
        sha256=hashlib.sha256(image).hexdigest(),
        model_revision="runtime-detector-abcdef",
        processor_revision="runtime-processor-123456",
    )

    async def exercise() -> None:
        await repository.ensure_schema()
        prior_usage = await repository.storage_bytes()
        original = await service.receive(metadata, image)
        connection = await asyncpg.connect(database_url)
        try:
            await connection.execute(
                "UPDATE ingestion_samples SET retention_until = now() WHERE sample_id = $1",
                metadata.sample_id,
            )
        finally:
            await connection.close()
        claimed = await repository.claim_expired()
        assert any(item.sample_id == metadata.sample_id for item in claimed)
        expired_sample = next(item for item in claimed if item.sample_id == metadata.sample_id)
        await anyio.to_thread.run_sync(partial(objects.delete_object, expired_sample.object_key))
        assert await repository.finish_expiry(metadata.sample_id) is True
        with pytest.raises(SampleExpiredError):
            _replay = await service.receive(metadata, image)
        assert objects.read_verified(metadata, image) is False
        assert await repository.storage_bytes() == prior_usage
        assert await repository.count_sample(metadata.sample_id) == 1
        assert original.sample_id == metadata.sample_id
        await repository.close()

    asyncio.run(exercise())


def test_http_receiver_auth_idempotency_and_conflicting_bytes() -> None:
    configured = _configured_backends()
    if configured is None:
        pytest.skip("isolated PostgreSQL and S3 test endpoints are not configured")
    _repository, _objects, database_url = configured
    token = "task4-http-test-token-with-at-least-32-chars"
    settings = IngestionSettings(
        database_url=database_url,
        s3_endpoint_url=os.environ["GODS_MLOPS_TEST_S3_ENDPOINT"],
        s3_access_key=os.environ["GODS_MLOPS_TEST_S3_ACCESS_KEY"],
        s3_secret_key=os.environ["GODS_MLOPS_TEST_S3_SECRET_KEY"],
        s3_bucket=os.environ["GODS_MLOPS_TEST_S3_BUCKET"],
        s3_region="us-east-1",
        bearer_token=token,
    )
    app = build_app(settings)
    import uvicorn

    def post_sample(endpoint: str, *, authorization: str | None, body: bytes, boundary: str) -> tuple[int, bytes]:
        request = urllib_request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                **({"Authorization": authorization} if authorization is not None else {}),
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=10) as response:
                return response.status, response.read()
        except urllib_error.HTTPError as error:
            return error.code, error.read()

    with socket.socket() as listen_socket:
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_socket.bind(("127.0.0.1", 0))
        listen_socket.listen(128)
        listen_socket.setblocking(False)
        port = int(listen_socket.getsockname()[1])
        server = uvicorn.Server(
            uvicorn.Config(app, log_level="critical", access_log=False, lifespan="on")
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [listen_socket]},
            daemon=True,
        )
        thread.start()
        try:
            deadline = time.monotonic() + 10
            while not server.started and time.monotonic() < deadline:
                time.sleep(0.025)
            assert server.started, "candidate API failed to start against isolated dependencies"
            endpoint = f"http://127.0.0.1:{port}/api/samples"
            image = b"jpeg-like bytes sent through multipart HTTP"
            sample_id = uuid4()
            boundary = "gods-task4-test-boundary"
            common = {
                "sample_id": str(sample_id),
                "camera_id": str(uuid4()),
                "captured_at_utc": "2026-10-05T00:00:00+00:00",
                "reason": CandidateReason.OPERATOR.value,
                "model_revision": "runtime-detector-abcdef",
                "processor_revision": "runtime-processor-123456",
            }

            def request_body(payload: bytes) -> bytes:
                fields = {
                    **common,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                chunks = []
                for name, value in fields.items():
                    chunks.extend(
                        (
                            f"--{boundary}\r\n".encode(),
                            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                            value.encode(),
                            b"\r\n",
                        )
                    )
                chunks.extend(
                    (
                        f"--{boundary}\r\n".encode(),
                        b'Content-Disposition: form-data; name="image"; filename="sample.jpg"\r\n',
                        b"Content-Type: image/jpeg\r\n\r\n",
                        payload,
                        b"\r\n",
                        f"--{boundary}--\r\n".encode(),
                    )
                )
                return b"".join(chunks)

            body = request_body(image)
            assert post_sample(endpoint, authorization=None, body=body, boundary=boundary)[0] == 401
            first_status, first_body = post_sample(
                endpoint,
                authorization=f"Bearer {token}",
                body=body,
                boundary=boundary,
            )
            replay_status, replay_body = post_sample(
                endpoint,
                authorization=f"Bearer {token}",
                body=body,
                boundary=boundary,
            )
            assert first_status == 201
            assert replay_status == 201
            assert json.loads(replay_body) == json.loads(first_body)

            conflicting_image = b"different bytes for the same sample ID"
            conflicting_status, _body = post_sample(
                endpoint,
                authorization=f"Bearer {token}",
                body=request_body(conflicting_image),
                boundary=boundary,
            )
            assert conflicting_status == 409
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive(), "candidate API test server did not stop cleanly"
