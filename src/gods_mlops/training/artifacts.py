"""Immutable Task 8 result artifacts written through the existing S3 object API."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from gods_mlops.datasets.manifest import canonical_json

_KIND = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class ResultArtifactIntegrityError(RuntimeError):
    """A result artifact does not match its reserved immutable identity."""


@dataclass(frozen=True, slots=True)
class PreparedResultArtifact:
    identity: Any
    kind: str
    sha256: str
    size_bytes: int
    object_key: str | None
    uri: str | None
    payload: bytes
    temporary_path: Path | None = None
    final_path: Path | None = None


@dataclass(frozen=True, slots=True)
class VerifiedResultArtifact:
    identity: Any
    kind: str
    sha256: str
    size_bytes: int
    uri: str
    object_key: str | None = None
    path: Path | None = None


class S3ResultArtifactStore:
    """Publish a hashed result bundle through Task 6's immutable S3 writer."""

    def __init__(self, *, objects: Any, bucket: str, prefix: str = "jobs") -> None:
        if not bucket or "/" in bucket:
            raise ValueError("S3 result bucket is invalid")
        self._objects = objects
        self._bucket = bucket
        self._prefix = _safe_prefix(prefix)

    def prepare(
        self,
        *,
        identity: Any,
        kind: str,
        payload: bytes,
        reservation_bytes: int,
    ) -> PreparedResultArtifact:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("result artifact must contain bytes")
        if reservation_bytes <= 0 or len(payload) > reservation_bytes:
            raise ValueError("result artifact exceeds its active versioned reservation")
        if not _KIND.fullmatch(kind):
            raise ValueError("result artifact kind is invalid")
        identity_hash = sha256(canonical_json(identity.as_dict())).hexdigest()
        digest = sha256(payload).hexdigest()
        object_key = f"{self._prefix}/{identity.job_id}/results/{identity_hash}/{kind}-{digest}.artifact"
        return PreparedResultArtifact(
            identity=identity,
            kind=kind,
            sha256=digest,
            size_bytes=len(payload),
            object_key=object_key,
            uri=f"s3://{self._bucket}/{object_key}",
            payload=payload,
        )

    def commit(self, prepared: PreparedResultArtifact) -> VerifiedResultArtifact:
        assert prepared.object_key is not None and prepared.uri is not None
        self._objects.write_immutable(
            object_key=prepared.object_key,
            content=prepared.payload,
            sha256_digest=prepared.sha256,
            content_type="application/octet-stream",
        )
        verified = self._objects.read_source(
            object_key=prepared.object_key,
            sha256_digest=prepared.sha256,
            size_bytes=prepared.size_bytes,
        )
        if verified != prepared.payload:
            raise ResultArtifactIntegrityError("S3 result artifact failed read-after-write verification")
        return VerifiedResultArtifact(
            identity=prepared.identity,
            kind=prepared.kind,
            sha256=prepared.sha256,
            size_bytes=prepared.size_bytes,
            uri=prepared.uri,
            object_key=prepared.object_key,
        )

    def verify_committed(self, details: dict[str, Any], *, expected_identity: Any) -> VerifiedResultArtifact:
        """Read and verify one already committed S3 result without publishing new bytes."""
        if not isinstance(details, dict) or details.get("identity") != expected_identity.as_dict():
            raise ResultArtifactIntegrityError("committed result identity differs from the current job")
        kind = details.get("kind")
        digest = details.get("sha256")
        size_bytes = details.get("size_bytes")
        if not isinstance(kind, str) or not _KIND.fullmatch(kind):
            raise ResultArtifactIntegrityError("committed result kind is invalid")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ResultArtifactIntegrityError("committed result SHA-256 is invalid")
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ResultArtifactIntegrityError("committed result size is invalid")
        identity_hash = sha256(canonical_json(expected_identity.as_dict())).hexdigest()
        object_key = f"{self._prefix}/{expected_identity.job_id}/results/{identity_hash}/{kind}-{digest}.artifact"
        uri = f"s3://{self._bucket}/{object_key}"
        if details.get("object_key") != object_key or details.get("uri") != uri:
            raise ResultArtifactIntegrityError("committed result URI differs from its immutable job key")
        self._objects.read_source(
            object_key=object_key,
            sha256_digest=digest,
            size_bytes=size_bytes,
        )
        return VerifiedResultArtifact(
            identity=expected_identity,
            kind=kind,
            sha256=digest,
            size_bytes=size_bytes,
            uri=uri,
            object_key=object_key,
        )


class FileResultArtifactStore:
    """Local atomic adapter for bounded Task 8 probes and isolated unit tests."""

    def __init__(self, *, root: str | Path) -> None:
        self._root = Path(root).resolve()

    def prepare(
        self,
        *,
        identity: Any,
        kind: str,
        payload: bytes,
        reservation_bytes: int,
    ) -> PreparedResultArtifact:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("result artifact must contain bytes")
        if reservation_bytes <= 0 or len(payload) > reservation_bytes:
            raise ValueError("result artifact exceeds its active versioned reservation")
        if not _KIND.fullmatch(kind):
            raise ValueError("result artifact kind is invalid")
        identity_hash = sha256(canonical_json(identity.as_dict())).hexdigest()
        digest = sha256(payload).hexdigest()
        directory = self._job_directory(identity.job_id)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        final = directory / f"{identity_hash}-{kind}-{digest}.artifact"
        if final.is_file():
            current = final.read_bytes()
            if len(current) != len(payload) or sha256(current).hexdigest() != digest:
                raise ResultArtifactIntegrityError("immutable result artifact key contains different bytes")
            temporary = None
        else:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".partial", dir=directory)
            temporary = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb", closefd=True) as target:
                    target.write(payload)
                    target.flush()
                    os.fsync(target.fileno())
            except Exception:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                temporary.unlink(missing_ok=True)
                raise
        return PreparedResultArtifact(
            identity=identity,
            kind=kind,
            sha256=digest,
            size_bytes=len(payload),
            object_key=None,
            uri=None,
            payload=payload,
            temporary_path=temporary,
            final_path=final,
        )

    def commit(self, prepared: PreparedResultArtifact) -> VerifiedResultArtifact:
        assert prepared.final_path is not None
        if prepared.temporary_path is not None:
            if prepared.final_path.exists():
                current = prepared.final_path.read_bytes()
                if len(current) != prepared.size_bytes or sha256(current).hexdigest() != prepared.sha256:
                    raise ResultArtifactIntegrityError("immutable result artifact key contains different bytes")
                prepared.temporary_path.unlink(missing_ok=True)
            else:
                os.replace(prepared.temporary_path, prepared.final_path)
        current = prepared.final_path.read_bytes()
        if len(current) != prepared.size_bytes or sha256(current).hexdigest() != prepared.sha256:
            raise ResultArtifactIntegrityError("result artifact failed read-after-write verification")
        return VerifiedResultArtifact(
            identity=prepared.identity,
            kind=prepared.kind,
            sha256=prepared.sha256,
            size_bytes=prepared.size_bytes,
            uri=prepared.final_path.resolve().as_uri(),
            path=prepared.final_path,
        )

    def discard(self, prepared: PreparedResultArtifact) -> None:
        if prepared.temporary_path is not None:
            prepared.temporary_path.unlink(missing_ok=True)

    def _job_directory(self, job_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", job_id):
            raise ValueError("result artifact job ID is unsafe")
        directory = self._root / job_id
        if directory.resolve(strict=False).parent != self._root:
            raise ValueError("result artifact path escapes its root")
        return directory


def _safe_prefix(value: str) -> str:
    parts = value.strip("/").split("/")
    if not parts or any(not part or part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in parts):
        raise ValueError("S3 artifact prefix must contain only safe relative segments")
    return "/".join(parts)
