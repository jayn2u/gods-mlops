"""PostgreSQL metadata and S3-compatible full-frame object adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from typing import Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import asyncpg
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from .schemas import (
    CandidateMetadata,
    DailySampleLimitError,
    GlobalObjectLimitError,
    SampleConflictError,
    SampleReceipt,
)

_DAILY_LIMIT = 2_000
GLOBAL_OBJECT_LIMIT = 1024**4
_GLOBAL_OBJECT_LIMIT = GLOBAL_OBJECT_LIMIT
_RETENTION_DAYS = 7
_CAPTURE_TIMEZONE = ZoneInfo("Asia/Seoul")

_CREATE_TABLES = f"""
CREATE TABLE IF NOT EXISTS ingestion_samples (
    sample_id UUID PRIMARY KEY,
    camera_id UUID NOT NULL,
    capture_day DATE NOT NULL,
    captured_at_utc TIMESTAMPTZ NOT NULL,
    reason TEXT NOT NULL CHECK (reason IN ('periodic', 'low_confidence', 'operator')),
    sha256 CHAR(64) NOT NULL CHECK (sha256 ~ '^[0-9a-f]{{64}}$'),
    model_revision VARCHAR(255) NOT NULL,
    processor_revision VARCHAR(255) NOT NULL,
    object_key TEXT NOT NULL,
    object_size_bytes BIGINT NOT NULL CHECK (object_size_bytes > 0),
    receipt_id UUID NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN ('pending', 'received', 'storage_limited', 'purge_pending', 'expired')),
    selected BOOLEAN NOT NULL DEFAULT FALSE,
    last_failure_code TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    received_at TIMESTAMPTZ,
    retention_until TIMESTAMPTZ
);
ALTER TABLE ingestion_samples ADD COLUMN IF NOT EXISTS object_size_bytes BIGINT NOT NULL DEFAULT 1;
ALTER TABLE ingestion_samples ADD COLUMN IF NOT EXISTS selected BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE ingestion_samples ADD COLUMN IF NOT EXISTS retention_until TIMESTAMPTZ;
ALTER TABLE ingestion_samples DROP CONSTRAINT IF EXISTS ingestion_samples_state_check;
ALTER TABLE ingestion_samples ADD CONSTRAINT ingestion_samples_state_check
    CHECK (state IN ('pending', 'received', 'storage_limited', 'purge_pending', 'expired'));
ALTER TABLE ingestion_samples DROP CONSTRAINT IF EXISTS ingestion_samples_sha256_check;
ALTER TABLE ingestion_samples ADD CONSTRAINT ingestion_samples_sha256_check
    CHECK (sha256 ~ '^[0-9a-f]{{64}}$');
CREATE INDEX IF NOT EXISTS ix_ingestion_samples_camera_day
    ON ingestion_samples (camera_id, capture_day);
CREATE TABLE IF NOT EXISTS ingestion_daily_usage (
    camera_id UUID NOT NULL,
    capture_day DATE NOT NULL,
    accepted_count INTEGER NOT NULL CHECK (accepted_count BETWEEN 1 AND 2000),
    PRIMARY KEY (camera_id, capture_day)
);
CREATE TABLE IF NOT EXISTS ingestion_storage_usage (
    singleton BOOLEAN PRIMARY KEY CHECK (singleton),
    used_bytes BIGINT NOT NULL CHECK (used_bytes BETWEEN 0 AND {_GLOBAL_OBJECT_LIMIT})
);
ALTER TABLE ingestion_samples DROP CONSTRAINT IF EXISTS ingestion_samples_object_size_bytes_check;
ALTER TABLE ingestion_samples ADD CONSTRAINT ingestion_samples_object_size_bytes_check
    CHECK (object_size_bytes > 0);
INSERT INTO ingestion_storage_usage (singleton, used_bytes)
VALUES (TRUE, 0)
ON CONFLICT (singleton) DO NOTHING;
"""


@dataclass(frozen=True, slots=True)
class SampleReservation:
    """Persisted immutable sample identity, whether pending or complete."""

    sample_id: UUID
    camera_id: UUID
    sha256: str
    object_key: str
    receipt_id: UUID
    state: str
    created: bool
    object_size_bytes: int


@dataclass(frozen=True, slots=True)
class ExpiredSample:
    """One claimed object deletion whose metadata tombstone remains durable."""

    sample_id: UUID
    object_key: str
    object_size_bytes: int


class PostgresIngestionRepository:
    """Reserve daily quota and sample identity in durable PostgreSQL transactions."""

    def __init__(
        self,
        *,
        database_url: str,
        max_samples_per_camera_day: int = _DAILY_LIMIT,
        max_object_bytes: int = _GLOBAL_OBJECT_LIMIT,
    ) -> None:
        if not 1 <= max_samples_per_camera_day <= _DAILY_LIMIT:
            raise ValueError("max_samples_per_camera_day must be between 1 and 2000")
        if not 1 <= max_object_bytes <= _GLOBAL_OBJECT_LIMIT:
            raise ValueError("max_object_bytes must be between 1 and 1 TiB")
        self._database_url = database_url
        self._daily_limit = max_samples_per_camera_day
        self._max_object_bytes = max_object_bytes
        self._pool: asyncpg.Pool | None = None

    async def ensure_schema(self) -> None:
        """Create only the receiver-owned tables; existing databases remain isolated."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(_CREATE_TABLES)
        # Annotation state shares this retained database and its object usage ledger.
        from gods_mlops.annotations.storage import PostgresAnnotationRepository

        annotations = PostgresAnnotationRepository(database_url=self._database_url)
        try:
            await annotations.ensure_schema()
        finally:
            await annotations.close()

    async def reserve(
        self,
        metadata: CandidateMetadata,
        object_key: str,
        object_size_bytes: int,
    ) -> SampleReservation:
        """Record an immutable ID before object I/O, charging every distinct attempt."""
        if object_size_bytes <= 0:
            raise ValueError("object_size_bytes must be positive")
        pool = await self._get_pool()
        capture_day = metadata.captured_at_utc.astimezone(_CAPTURE_TIMEZONE).date()
        async with pool.acquire() as connection:
            async with connection.transaction():
                receipt_id = uuid4()
                inserted = await connection.fetchrow(
                    """
                    INSERT INTO ingestion_samples (
                        sample_id, camera_id, capture_day, captured_at_utc, reason, sha256,
                        model_revision, processor_revision, object_key, object_size_bytes, receipt_id, state
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'pending')
                    ON CONFLICT (sample_id) DO NOTHING
                    RETURNING sample_id, camera_id, sha256, object_key, receipt_id, state, object_size_bytes
                    """,
                    metadata.sample_id,
                    metadata.camera_id,
                    capture_day,
                    metadata.captured_at_utc,
                    metadata.reason.value,
                    metadata.sha256,
                    metadata.model_revision,
                    metadata.processor_revision,
                    object_key,
                    object_size_bytes,
                    receipt_id,
                )
                if inserted is not None:
                    count_row = await connection.fetchrow(
                        """
                        INSERT INTO ingestion_daily_usage (camera_id, capture_day, accepted_count)
                        VALUES ($1, $2, 1)
                        ON CONFLICT (camera_id, capture_day) DO UPDATE
                          SET accepted_count = ingestion_daily_usage.accepted_count + 1
                          WHERE ingestion_daily_usage.accepted_count < $3
                        RETURNING accepted_count
                        """,
                        metadata.camera_id,
                        capture_day,
                        self._daily_limit,
                    )
                    if count_row is None:
                        raise DailySampleLimitError("camera daily candidate limit reached")
                    storage_row = await connection.fetchrow(
                        """
                        INSERT INTO ingestion_storage_usage (singleton, used_bytes)
                        VALUES (TRUE, $1)
                        ON CONFLICT (singleton) DO UPDATE
                          SET used_bytes = ingestion_storage_usage.used_bytes + $1
                          WHERE ingestion_storage_usage.used_bytes + $1 <= $2
                        RETURNING used_bytes
                        """,
                        object_size_bytes,
                        self._max_object_bytes,
                    )
                    if storage_row is None:
                        await connection.execute(
                            "UPDATE ingestion_samples SET state = 'storage_limited', last_failure_code = 'object_store_limit' WHERE sample_id = $1",
                            metadata.sample_id,
                        )
                        limited = await connection.fetchrow(
                            """
                            SELECT sample_id, camera_id, sha256, object_key, receipt_id, state,
                                   object_size_bytes
                            FROM ingestion_samples WHERE sample_id = $1
                            """,
                            metadata.sample_id,
                        )
                        return _reservation(limited, created=True)
                    return _reservation(inserted, created=True)

                existing = await connection.fetchrow(
                    """
                    SELECT sample_id, camera_id, sha256, object_key, receipt_id, state,
                           object_size_bytes
                    FROM ingestion_samples WHERE sample_id = $1
                    """,
                    metadata.sample_id,
                )
                if existing is None:
                    raise RuntimeError("sample reservation disappeared during conflict handling")
                if existing["sha256"] != metadata.sha256:
                    raise SampleConflictError("sample ID is already bound to different bytes")
                return _reservation(existing, created=False)

    async def mark_received(self, sample_id: UUID) -> SampleReceipt:
        """Commit a receipt after S3 verification; repeat commits return the same receipt."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                record = await connection.fetchrow(
                    """
                    UPDATE ingestion_samples
                    SET state = 'received', last_failure_code = NULL,
                        received_at = COALESCE(received_at, now()),
                        retention_until = COALESCE(retention_until, now() + make_interval(days => $2))
                    WHERE sample_id = $1 AND state IN ('pending', 'received')
                    RETURNING sample_id, sha256, receipt_id, object_key
                    """,
                    sample_id,
                    _RETENTION_DAYS,
                )
                if record is None:
                    raise RuntimeError("cannot receipt an unreserved sample")
                return SampleReceipt(
                    sample_id=record["sample_id"],
                    sha256=record["sha256"].strip(),
                    receipt_id=record["receipt_id"],
                    object_key=record["object_key"],
                )

    async def record_failure(self, sample_id: UUID, code: str) -> None:
        """Retain a bounded operational failure code for later repair attempts."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                "UPDATE ingestion_samples SET last_failure_code = $2 WHERE sample_id = $1 AND state = 'pending'",
                sample_id,
                code[:64],
            )

    async def count_sample(self, sample_id: UUID) -> int:
        """Return a testable durable row count without exposing data to production callers."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            count = await connection.fetchval(
                "SELECT count(*) FROM ingestion_samples WHERE sample_id = $1",
                sample_id,
            )
        return int(count)

    async def daily_count(self, camera_id: UUID, capture_day: date) -> int:
        """Return the distinct per-camera daily reservation count."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            count = await connection.fetchval(
                "SELECT accepted_count FROM ingestion_daily_usage WHERE camera_id = $1 AND capture_day = $2",
                camera_id,
                capture_day,
            )
        return int(count or 0)

    async def storage_bytes(self) -> int:
        """Return bytes reserved for candidate objects under the global S3 limit."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            count = await connection.fetchval(
                "SELECT used_bytes FROM ingestion_storage_usage WHERE singleton = TRUE"
            )
        return int(count or 0)

    async def claim_expired(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> tuple[ExpiredSample, ...]:
        """Claim unselected seven-day objects for retryable deletion, retaining DB tombstones."""
        claim_time = now or datetime.now(timezone.utc)
        if claim_time.tzinfo is None:
            raise ValueError("retention claim time must be timezone-aware")
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                rows = await connection.fetch(
                    """
                    SELECT sample.sample_id, sample.object_key, sample.object_size_bytes, sample.state
                    FROM ingestion_samples AS sample
                    WHERE (sample.state = 'purge_pending'
                       OR (sample.state = 'received' AND sample.retention_until <= $1))
                      AND sample.selected = FALSE
                      AND NOT EXISTS (
                          SELECT 1 FROM review_assignments AS assignment
                          WHERE assignment.sample_id = sample.sample_id
                            AND assignment.stage = 'bbox'
                            AND assignment.state IN ('active', 'provisioning')
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM dataset_adoptions AS adoption
                          WHERE adoption.sample_id = sample.sample_id AND adoption.target = 'detr'
                      )
                    ORDER BY COALESCE(sample.retention_until, sample.created_at), sample.sample_id
                    LIMIT $2
                    FOR UPDATE SKIP LOCKED
                    """,
                    claim_time,
                    limit,
                )
                samples: list[ExpiredSample] = []
                for row in rows:
                    if row["state"] == "received":
                        await connection.execute(
                            "UPDATE ingestion_samples SET state = 'purge_pending' WHERE sample_id = $1",
                            row["sample_id"],
                        )
                    samples.append(
                        ExpiredSample(
                            sample_id=row["sample_id"],
                            object_key=row["object_key"],
                            object_size_bytes=row["object_size_bytes"],
                        )
                    )
                return tuple(samples)

    async def finish_expiry(self, sample_id: UUID) -> bool:
        """Tombstone one deleted object and release its reserved bytes exactly once."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                deleted = await connection.fetchrow(
                    """
                    UPDATE ingestion_samples
                    SET state = 'expired'
                    WHERE sample_id = $1 AND state = 'purge_pending' AND selected = FALSE
                    RETURNING object_size_bytes
                    """,
                    sample_id,
                )
                if deleted is None:
                    return False
                await connection.execute(
                    """
                    UPDATE annotation_crops
                    SET parent_available = FALSE, regeneration_available = FALSE, updated_at = now()
                    WHERE sample_id = $1
                    """,
                    sample_id,
                )
                usage = await connection.fetchrow(
                    """
                    UPDATE ingestion_storage_usage
                    SET used_bytes = used_bytes - $1
                    WHERE singleton = TRUE AND used_bytes >= $1
                    RETURNING used_bytes
                    """,
                    deleted["object_size_bytes"],
                )
                if usage is None:
                    raise RuntimeError("object quota accounting is inconsistent during retention")
                return True

    async def sample_state(self, sample_id: UUID) -> str | None:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            return await connection.fetchval(
                "SELECT state FROM ingestion_samples WHERE sample_id = $1", sample_id
            )

    async def ready(self) -> None:
        """Check that PostgreSQL accepts a process-owned query."""
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute("SELECT 1")

    async def close(self) -> None:
        """Close the process-owned async PostgreSQL pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._database_url,
                min_size=1,
                max_size=8,
                command_timeout=10,
            )
        return self._pool


class S3SampleStore:
    """Write and read back objects through an S3-compatible endpoint."""

    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str,
    ) -> None:
        if not endpoint_url.startswith(("http://", "https://")) or "@" in endpoint_url:
            raise ValueError("S3 endpoint must be an HTTP(S) URL without embedded credentials")
        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(
                signature_version="s3v4",
                connect_timeout=2,
                read_timeout=10,
                retries={"mode": "standard", "total_max_attempts": 3},
                s3={"addressing_style": "path"},
            ),
        )

    def ready(self) -> None:
        """Verify the configured bucket can be reached with the active credentials."""
        self._client.head_bucket(Bucket=self._bucket)

    def ensure_object(self, *, object_key: str, image: bytes, expected_sha256: str) -> None:
        """Upload if absent or corrupt, then read back and hash the stored object."""
        if not self._matches(object_key, len(image), expected_sha256):
            self._client.put_object(
                Bucket=self._bucket,
                Key=object_key,
                Body=image,
                ContentType="image/jpeg",
                Metadata={"sha256": expected_sha256},
            )
        if not self._matches(object_key, len(image), expected_sha256):
            raise OSError("S3 object failed read-after-write verification")

    def read_verified(self, metadata: CandidateMetadata, image: bytes) -> bool:
        """Expose a narrow integration-test assertion for the stored object bytes."""
        key = sample_object_key(metadata)
        return self._matches(key, len(image), metadata.sha256)

    def delete_object(self, object_key: str) -> None:
        """Delete one receiver-owned object idempotently after metadata is purge-pending."""
        self._client.delete_object(Bucket=self._bucket, Key=object_key)

    def read_object(self, *, object_key: str, expected_sha256: str) -> bytes:
        """Read one receiver-owned object and verify its immutable source digest."""
        try:
            response: dict[str, Any] = self._client.get_object(
                Bucket=self._bucket,
                Key=object_key,
            )
        except ClientError as error:
            response_code = str(error.response.get("Error", {}).get("Code", ""))
            if response_code in {"NoSuchKey", "NotFound", "404"}:
                raise FileNotFoundError(f"object is missing: {object_key}") from error
            raise
        body = response["Body"]
        digest = sha256()
        chunks: list[bytes] = []
        size = 0
        while chunk := body.read(1024 * 1024):
            size += len(chunk)
            if size > 20 * 1024 * 1024:
                body.close()
                raise OSError("source frame exceeds the 20 MiB candidate limit")
            digest.update(chunk)
            chunks.append(chunk)
        body.close()
        if digest.hexdigest() != expected_sha256:
            raise OSError("source object failed SHA-256 verification")
        return b"".join(chunks)

    def _matches(self, object_key: str, expected_length: int, expected_sha256: str) -> bool:
        try:
            response: dict[str, Any] = self._client.get_object(
                Bucket=self._bucket,
                Key=object_key,
            )
        except ClientError as error:
            response_code = str(error.response.get("Error", {}).get("Code", ""))
            if response_code in {"NoSuchKey", "NotFound", "404"}:
                return False
            raise
        body = response["Body"]
        digest = sha256()
        length = 0
        while chunk := body.read(1024 * 1024):
            digest.update(chunk)
            length += len(chunk)
        body.close()
        return length == expected_length and digest.hexdigest() == expected_sha256


def sample_object_key(metadata: CandidateMetadata) -> str:
    """Return an immutable key whose path is scoped to the camera and sample ID."""
    return f"samples/{metadata.camera_id}/{metadata.sample_id}/{metadata.sha256}.jpg"


def _reservation(record: Any, *, created: bool) -> SampleReservation:
    return SampleReservation(
        sample_id=record["sample_id"],
        camera_id=record["camera_id"],
        sha256=record["sha256"].strip(),
        object_key=record["object_key"],
        receipt_id=record["receipt_id"],
        state=record["state"],
        created=created,
        object_size_bytes=record["object_size_bytes"],
    )
