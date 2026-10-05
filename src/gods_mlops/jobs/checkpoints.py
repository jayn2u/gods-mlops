"""Hash-verified, atomically committed checkpoints scoped to immutable job input."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

CHECKPOINT_INTERVAL_SECONDS = 300
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CheckpointIntegrityError(RuntimeError):
    """A committed checkpoint's bytes or metadata no longer match their hashes."""


class CheckpointIdentityError(ValueError):
    """A checkpoint does not belong to the exact job, input, model, and config."""


class StaleCheckpointOwnerError(RuntimeError):
    """A previous lease fence no longer owns the job and cannot publish metadata."""


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    job_id: str
    input_kind: str
    input_id: str
    input_sha256: str
    phase: str
    model_kind: str
    config_version: str
    config_sha256: str
    dataset_version: str | None

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.job_id):
            raise ValueError("checkpoint job_id is not a safe identifier")
        if self.input_kind not in {"dataset_version", "annotation_batch", "probe_input", "checkpoint"}:
            raise ValueError("checkpoint input_kind is unsupported")
        if not self.input_id or len(self.input_id) > 255:
            raise ValueError("checkpoint input_id must contain 1 to 255 characters")
        if self.phase not in {"preparation", "training", "evaluation", "probe"}:
            raise ValueError("checkpoint phase is unsupported")
        if self.model_kind not in {"detr", "clip", "qwen"}:
            raise ValueError("checkpoint model_kind is unsupported")
        if not self.config_version or len(self.config_version) > 255:
            raise ValueError("checkpoint config_version must contain 1 to 255 characters")
        if not _SHA256.fullmatch(self.input_sha256) or not _SHA256.fullmatch(self.config_sha256):
            raise ValueError("checkpoint input and config identities must be SHA-256 digests")
        if self.input_kind == "dataset_version" and self.dataset_version != self.input_id:
            raise ValueError("dataset checkpoint identity must pin its dataset version")
        if self.input_kind != "dataset_version" and self.dataset_version is not None:
            raise ValueError("non-dataset checkpoint identity cannot claim a dataset version")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "job_id": self.job_id,
            "input_kind": self.input_kind,
            "input_id": self.input_id,
            "input_sha256": self.input_sha256,
            "phase": self.phase,
            "model_kind": self.model_kind,
            "config_version": self.config_version,
            "config_sha256": self.config_sha256,
            "dataset_version": self.dataset_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CheckpointIdentity":
        return cls(
            job_id=str(value["job_id"]),
            input_kind=str(value["input_kind"]),
            input_id=str(value["input_id"]),
            input_sha256=str(value["input_sha256"]),
            phase=str(value["phase"]),
            model_kind=str(value["model_kind"]),
            config_version=str(value["config_version"]),
            config_sha256=str(value["config_sha256"]),
            dataset_version=(str(value["dataset_version"]) if value.get("dataset_version") is not None else None),
        )


@dataclass(frozen=True, slots=True)
class VerifiedCheckpoint:
    identity: CheckpointIdentity
    path: Path | None
    sha256: str
    size_bytes: int
    created_at: datetime
    payload: bytes
    uri: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedCheckpoint:
    identity: CheckpointIdentity
    data_path: Path
    metadata_path: Path
    metadata_bytes: bytes
    sha256: str
    size_bytes: int
    metadata_size_bytes: int
    created_at: datetime


class FileCheckpointStore:
    """Keep payload bytes immutable and publish verified metadata as the commit marker."""

    def __init__(self, *, root: str | Path) -> None:
        self._root = Path(root).resolve()

    def save(
        self,
        *,
        identity: CheckpointIdentity,
        payload: bytes,
        reservation_bytes: int,
    ) -> VerifiedCheckpoint:
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("checkpoint payload must contain bytes")
        if reservation_bytes <= 0:
            raise ValueError("checkpoint artifact reservation must be positive")
        prepared = self.prepare(
            identity=identity,
            payload=payload,
            reservation_bytes=reservation_bytes,
        )
        result = self.commit(prepared)
        self.prune_previous(prepared)
        return result

    def prepare(
        self,
        *,
        identity: CheckpointIdentity,
        payload: bytes,
        reservation_bytes: int,
        replacement_reservation_bytes: int | None = None,
        previous_uri: str | None = None,
    ) -> PreparedCheckpoint:
        """Write and verify payload bytes without publishing a resume marker."""
        if not isinstance(payload, bytes) or not payload:
            raise ValueError("checkpoint payload must contain bytes")
        if reservation_bytes <= 0:
            raise ValueError("checkpoint artifact reservation must be positive")
        digest = sha256(payload).hexdigest()
        identity_hash = sha256(_canonical_json(identity.as_dict()).encode("utf-8")).hexdigest()
        created_at = datetime.now(UTC)
        directory = self._job_directory(identity.job_id)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        data_path = directory / f"{identity_hash}-{digest}.checkpoint"
        metadata_path = directory / f"{identity_hash}-{digest}.json"
        metadata = {
            "format": "gods-mlops-checkpoint-v1",
            "identity": identity.as_dict(),
            "checkpoint_file": data_path.name,
            "sha256": digest,
            "size_bytes": len(payload),
            "created_at": created_at.isoformat(),
        }
        encoded_metadata = _canonical_json(metadata).encode("utf-8")
        if len(payload) + len(encoded_metadata) > reservation_bytes:
            raise ValueError("checkpoint payload and metadata exceed the artifact reservation")
        replacement_limit = replacement_reservation_bytes or reservation_bytes
        identity_prefix = f"{identity_hash}-"
        existing_bytes = sum(
            item.stat().st_size
            for item in directory.iterdir()
            if item.is_file()
            and item.name.startswith(identity_prefix)
            and item.suffix in {".checkpoint", ".json"}
        )
        additional_payload_bytes = 0 if data_path.exists() else len(payload)
        if existing_bytes + additional_payload_bytes + len(encoded_metadata) > replacement_limit:
            raise ValueError("atomic checkpoint replacement exceeds its versioned reservation")

        if data_path.exists():
            existing = data_path.read_bytes()
            if len(existing) != len(payload) or sha256(existing).hexdigest() != digest:
                raise CheckpointIntegrityError("immutable checkpoint key contains different bytes")
        else:
            self._atomic_write(data_path, payload)
        verified_bytes = data_path.read_bytes()
        if len(verified_bytes) != len(payload) or sha256(verified_bytes).hexdigest() != digest:
            raise CheckpointIntegrityError("checkpoint data failed SHA-256 read-after-write verification")
        return PreparedCheckpoint(
            identity=identity,
            data_path=data_path,
            metadata_path=metadata_path,
            metadata_bytes=encoded_metadata,
            sha256=digest,
            size_bytes=len(payload),
            metadata_size_bytes=len(encoded_metadata),
            created_at=created_at,
        )

    def commit(self, prepared: PreparedCheckpoint) -> VerifiedCheckpoint:
        """Publish the atomic metadata commit marker after verifying payload bytes."""
        current = prepared.data_path.read_bytes()
        if len(current) != prepared.size_bytes or sha256(current).hexdigest() != prepared.sha256:
            raise CheckpointIntegrityError("prepared checkpoint data changed before metadata commit")
        self._atomic_write(prepared.metadata_path, prepared.metadata_bytes)
        result = self.load(prepared.identity.job_id, expected_identity=prepared.identity)
        if result is None or result.sha256 != prepared.sha256:
            raise CheckpointIntegrityError("checkpoint metadata failed read-after-write verification")
        return result

    def prune_previous(self, prepared: PreparedCheckpoint) -> None:
        """Remove older same-identity checkpoints only after the replacement is committed."""
        directory = prepared.data_path.parent
        identity_prefix = prepared.data_path.name.split("-", 1)[0]
        for pattern in (f"{identity_prefix}-*.checkpoint", f"{identity_prefix}-*.json"):
            for path in directory.glob(pattern):
                if path not in {prepared.data_path, prepared.metadata_path}:
                    path.unlink(missing_ok=True)

    def checkpoint_identity_hash(self, identity: CheckpointIdentity) -> str:
        return sha256(_canonical_json(identity.as_dict()).encode("utf-8")).hexdigest()

    def load(
        self,
        job_id: str,
        *,
        expected_identity: CheckpointIdentity,
    ) -> VerifiedCheckpoint | None:
        if job_id != expected_identity.job_id:
            raise CheckpointIdentityError("requested job ID does not match checkpoint identity")
        directory = self._job_directory(job_id)
        if not directory.is_dir():
            return None
        candidates: list[tuple[datetime, VerifiedCheckpoint]] = []
        for metadata_path in sorted(directory.glob("*.json")):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("format") != "gods-mlops-checkpoint-v1":
                    raise CheckpointIntegrityError("checkpoint metadata format is invalid")
                identity = CheckpointIdentity.from_dict(metadata["identity"])
                if identity != expected_identity:
                    continue
                filename = metadata.get("checkpoint_file")
                if not isinstance(filename, str) or Path(filename).name != filename:
                    raise CheckpointIntegrityError("checkpoint metadata path is invalid")
                digest = metadata.get("sha256")
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise CheckpointIntegrityError("checkpoint metadata SHA-256 is invalid")
                size_bytes = int(metadata["size_bytes"])
                created_at = datetime.fromisoformat(metadata["created_at"])
                path = directory / filename
                if not path.is_file():
                    raise CheckpointIntegrityError("committed checkpoint payload is missing")
                payload = path.read_bytes()
                if len(payload) != size_bytes or sha256(payload).hexdigest() != digest:
                    raise CheckpointIntegrityError("checkpoint payload SHA-256 or size does not match metadata")
                candidates.append(
                    (
                        created_at,
                        VerifiedCheckpoint(
                            identity=identity,
                            path=path,
                            sha256=digest,
                            size_bytes=size_bytes,
                            created_at=created_at,
                            payload=payload,
                            uri=path.resolve().as_uri(),
                        ),
                    )
                )
            except CheckpointIntegrityError:
                raise
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                raise CheckpointIntegrityError("checkpoint metadata is incomplete or unreadable") from error
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _job_directory(self, job_id: str) -> Path:
        if not _IDENTIFIER.fullmatch(job_id):
            raise ValueError("checkpoint job_id is not a safe identifier")
        directory = self._root / job_id
        if directory.parent != self._root or directory.resolve(strict=False).parent != self._root:
            raise ValueError("checkpoint path escapes the configured root")
        return directory

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
        temp_path = Path(temp_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temp_path.unlink(missing_ok=True)
            raise


def checkpoint_due(last_saved_at: datetime | None, now: datetime) -> bool:
    """Return whether the five-minute checkpoint interval has elapsed."""
    if now.tzinfo is None or (last_saved_at is not None and last_saved_at.tzinfo is None):
        raise ValueError("checkpoint times must be timezone-aware")
    return last_saved_at is None or (now - last_saved_at).total_seconds() >= CHECKPOINT_INTERVAL_SECONDS


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
