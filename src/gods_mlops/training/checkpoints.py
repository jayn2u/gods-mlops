"""S3 checkpoint adapter backed by Task 6's verified immutable object writer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

from gods_mlops.datasets.manifest import canonical_json
from gods_mlops.jobs.checkpoints import CheckpointIdentity, CheckpointIntegrityError, VerifiedCheckpoint


@dataclass(frozen=True, slots=True)
class PreparedS3Checkpoint:
    identity: CheckpointIdentity
    bucket: str
    object_key: str
    sha256: str
    size_bytes: int
    created_at: datetime
    payload: bytes
    previous_uri: str | None
    previous_sha256: str | None
    previous_size_bytes: int | None

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.object_key}"

    @property
    def metadata_size_bytes(self) -> int:
        # Identity and size are atomically committed in the Task 7 PostgreSQL event/row.
        return 0


class S3CheckpointStore:
    """Keep optimizer/model checkpoint bytes in the shared S3 object lake."""

    def __init__(self, *, objects: Any, bucket: str, prefix: str = "jobs") -> None:
        if not bucket or "/" in bucket:
            raise ValueError("S3 checkpoint bucket is invalid")
        self._objects = objects
        self._bucket = bucket
        self._prefix = _safe_prefix(prefix)

    def prepare(
        self,
        *,
        identity: CheckpointIdentity,
        payload: bytes,
        reservation_bytes: int,
        replacement_reservation_bytes: int | None = None,
        previous_uri: str | None = None,
        previous_sha256: str | None = None,
        previous_size_bytes: int | None = None,
    ) -> PreparedS3Checkpoint:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("checkpoint payload must contain bytes")
        if reservation_bytes <= 0 or len(payload) > reservation_bytes:
            raise ValueError("checkpoint payload exceeds the active result reservation")
        if replacement_reservation_bytes is not None and len(payload) > replacement_reservation_bytes:
            raise ValueError("checkpoint payload exceeds its atomic replacement reservation")
        identity_sha = sha256(canonical_json(identity.as_dict())).hexdigest()
        digest = sha256(payload).hexdigest()
        key = f"{self._prefix}/{identity.job_id}/checkpoints/{identity_sha}/{digest}.checkpoint"
        return PreparedS3Checkpoint(
            identity=identity,
            bucket=self._bucket,
            object_key=key,
            sha256=digest,
            size_bytes=len(payload),
            created_at=datetime.now(UTC),
            payload=payload,
            previous_uri=previous_uri,
            previous_sha256=previous_sha256,
            previous_size_bytes=previous_size_bytes,
        )

    def commit(self, prepared: PreparedS3Checkpoint) -> VerifiedCheckpoint:
        if sha256(prepared.payload).hexdigest() != prepared.sha256:
            raise CheckpointIntegrityError("prepared checkpoint bytes changed before S3 commit")
        self._objects.write_immutable(
            object_key=prepared.object_key,
            content=prepared.payload,
            sha256_digest=prepared.sha256,
            content_type="application/octet-stream",
        )
        stored = self._objects.read_source(
            object_key=prepared.object_key,
            sha256_digest=prepared.sha256,
            size_bytes=prepared.size_bytes,
        )
        if stored != prepared.payload:
            raise CheckpointIntegrityError("S3 checkpoint failed read-after-write verification")
        return VerifiedCheckpoint(
            identity=prepared.identity,
            path=None,
            sha256=prepared.sha256,
            size_bytes=prepared.size_bytes,
            created_at=prepared.created_at,
            payload=stored,
            uri=f"s3://{self._bucket}/{prepared.object_key}",
        )

    def load_uri(
        self,
        uri: str,
        *,
        expected_identity: CheckpointIdentity,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> VerifiedCheckpoint:
        bucket, key = _parse_uri(uri)
        if bucket != self._bucket or not key.startswith(
            f"{self._prefix}/{expected_identity.job_id}/checkpoints/"
        ):
            raise CheckpointIntegrityError("checkpoint URI is outside the immutable job prefix")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise CheckpointIntegrityError("checkpoint SHA-256 is invalid")
        payload = self._objects.read_source(
            object_key=key,
            sha256_digest=expected_sha256,
            size_bytes=expected_size_bytes,
        )
        return VerifiedCheckpoint(
            identity=expected_identity,
            path=None,
            sha256=expected_sha256,
            size_bytes=expected_size_bytes,
            created_at=datetime.now(UTC),
            payload=payload,
            uri=uri,
        )

    def prune_previous(self, prepared: PreparedS3Checkpoint) -> None:
        if not prepared.previous_uri or prepared.previous_uri == prepared.uri:
            return
        if prepared.previous_sha256 is None or prepared.previous_size_bytes is None:
            raise CheckpointIntegrityError("previous checkpoint deletion has no verified size and SHA-256")
        self.prune_uri(
            prepared.previous_uri,
            sha256_digest=prepared.previous_sha256,
            size_bytes=prepared.previous_size_bytes,
            job_id=prepared.identity.job_id,
        )

    def prune_uri(self, uri: str, *, sha256_digest: str, size_bytes: int, job_id: str) -> None:
        bucket, key = _parse_uri(uri)
        allowed_prefix = f"{self._prefix}/{job_id}/checkpoints/"
        if bucket != self._bucket or not key.startswith(allowed_prefix):
            raise CheckpointIntegrityError("previous checkpoint URI is outside the immutable job prefix")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256_digest) or size_bytes <= 0:
            raise CheckpointIntegrityError("previous checkpoint deletion identity is invalid")
        if not key.endswith(f"/{sha256_digest}.checkpoint"):
            raise CheckpointIntegrityError("previous checkpoint key does not match its content identity")
        try:
            self._objects.read_source(
                object_key=key,
                sha256_digest=sha256_digest,
                size_bytes=size_bytes,
            )
        except FileNotFoundError:
            return
        self._objects.delete_object(object_key=key)
        try:
            self._objects.read_source(
                object_key=key, sha256_digest=sha256_digest, size_bytes=size_bytes
            )
        except FileNotFoundError:
            return
        raise CheckpointIntegrityError("previous checkpoint still exists after deletion")


def _parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key or ".." in key.split("/"):
        raise CheckpointIntegrityError("checkpoint URI is not a safe S3 object URI")
    return parsed.netloc, key


def _safe_prefix(value: str) -> str:
    parts = value.strip("/").split("/")
    if not parts or any(
        not part or part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", part)
        for part in parts
    ):
        raise ValueError("S3 checkpoint prefix must contain safe relative segments")
    return "/".join(parts)
