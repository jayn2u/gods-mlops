"""Typed identities shared by the queue, admission, and worker adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    """One immutable model/config resource profile.

    Profiles are first registered as measurement candidates. Training and
    evaluation admission will only use a profile backed by a successful probe.
    """

    model_kind: str
    config_version: str
    phase: str
    memory_requirement_mib: int
    artifact_reservation_bytes: int
    config: dict[str, Any]
    target_phase: str | None = None
    candidate: bool = True
    measurement_id: str | None = None
    oom_alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.model_kind not in {"detr", "clip", "qwen"}:
            raise ValueError("model_kind must be detr, clip, or qwen")
        if self.phase not in {"preparation", "training", "evaluation", "probe"}:
            raise ValueError("unsupported GPU job phase")
        if self.model_kind == "qwen" and self.phase in {"training", "evaluation"}:
            raise ValueError("the fixed Qwen caption model is inference-only")
        if self.phase == "probe":
            target_phase = self.target_phase or ("preparation" if self.model_kind == "qwen" else "training")
            if target_phase not in {"preparation", "training", "evaluation"}:
                raise ValueError("probe target_phase must be preparation, training, or evaluation")
            if target_phase == "training" and self.model_kind == "qwen":
                raise ValueError("the fixed Qwen caption model is an inference target, not a training target")
            if self.model_kind == "qwen" and target_phase != "preparation":
                raise ValueError("the fixed Qwen caption model only supports preparation inference probes")
            object.__setattr__(self, "target_phase", target_phase)
        elif self.target_phase is not None:
            raise ValueError("target_phase is only valid for a measurement probe")
        if not self.config_version.strip() or len(self.config_version) > 255:
            raise ValueError("config_version must contain 1 to 255 characters")
        if not 1 <= self.memory_requirement_mib <= 49_140:
            raise ValueError("memory_requirement_mib is outside the supported A6000 profile")
        if 3 <= self.artifact_reservation_bytes <= 1024**4:
            pass
        else:
            raise ValueError("artifact reservation must be between 3 bytes and 1 TiB")
        if len(self.oom_alternatives) > 2 or len(set(self.oom_alternatives)) != len(self.oom_alternatives):
            raise ValueError("at most two distinct OOM alternatives may be configured")
        if self.candidate and self.measurement_id is not None:
            raise ValueError("candidate profiles cannot claim a measurement")
        if not self.candidate and not self.measurement_id:
            raise ValueError("measured profiles require a successful measurement ID")

    @property
    def config_sha256(self) -> str:
        payload = json.dumps(self.config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(payload.encode("utf-8")).hexdigest()

    @property
    def checkpoint_reservation_bytes(self) -> int:
        """Reserve one third for each checkpoint version during atomic replacement."""
        return max(1, self.artifact_reservation_bytes // 3)

    @property
    def result_reservation_bytes(self) -> int:
        """Leave a distinct one-third bound for the final model and result files."""
        return max(1, self.artifact_reservation_bytes // 3)


@dataclass(frozen=True, slots=True)
class DatasetTrainingSource:
    """Current mutable readiness state over an immutable published manifest."""

    dataset_version: str
    target: str
    manifest_sha256: str
    state: str
    training_ready: bool
    training_reasons: tuple[str, ...]
    evaluation_eligible: bool
    evaluation_reasons: tuple[str, ...]
    invalidated_source_count: int
    leakage_impact_count: int

    def training_block_reasons(self) -> tuple[str, ...]:
        reasons = set(self.training_reasons)
        if self.state != "published":
            reasons.add(f"dataset_state_{self.state}")
        if self.invalidated_source_count:
            reasons.add("source_sample_explicitly_invalidated")
        if not self.training_ready:
            reasons.add("dataset_not_training_ready")
        return tuple(sorted(reasons))

    def evaluation_block_reasons(self) -> tuple[str, ...]:
        reasons = set(self.evaluation_reasons)
        if self.invalidated_source_count:
            reasons.add("source_sample_explicitly_invalidated")
        if self.leakage_impact_count:
            reasons.add("late_cross_boundary_link")
        if not self.evaluation_eligible:
            reasons.add("dataset_not_evaluation_eligible")
        return tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class ProbeInput:
    """Immutable S3 manifest for a model-readiness fixture, separate from a dataset."""

    probe_input_id: str
    model_kind: str
    target_phase: str
    config_version: str
    manifest_object_key: str
    input_sha256: str
    object_size_bytes: int
    fixture: bool = True

    def __post_init__(self) -> None:
        import re

        if not self.probe_input_id.strip() or len(self.probe_input_id) > 255:
            raise ValueError("probe_input_id must contain 1 to 255 characters")
        if self.model_kind not in {"detr", "clip", "qwen"}:
            raise ValueError("probe model_kind must be detr, clip, or qwen")
        if self.target_phase not in {"preparation", "training"}:
            raise ValueError("model readiness probe target_phase is unsupported")
        if self.model_kind == "qwen" and self.target_phase != "preparation":
            raise ValueError("Qwen readiness probes are inference-only")
        if self.model_kind == "clip" and self.target_phase != "training":
            raise ValueError("CLIP readiness probes require their training target")
        if not self.config_version.strip() or len(self.config_version) > 255:
            raise ValueError("probe config_version must contain 1 to 255 characters")
        if (
            not self.manifest_object_key
            or self.manifest_object_key.startswith("/")
            or ".." in self.manifest_object_key.split("/")
            or "\\" in self.manifest_object_key
        ):
            raise ValueError("probe manifest key must be a safe relative object key")
        if not re.fullmatch(r"[0-9a-f]{64}", self.input_sha256):
            raise ValueError("probe manifest SHA-256 must be lowercase hexadecimal")
        if self.object_size_bytes <= 0:
            raise ValueError("probe manifest object size must be positive")
        if self.fixture is not True:
            raise ValueError("model readiness probe input must be explicitly marked as a fixture")

    @property
    def input_kind(self) -> str:
        return "probe_input"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "gods-mlops-probe-input-v1",
            "probe_input_id": self.probe_input_id,
            "input_kind": self.input_kind,
            "model_kind": self.model_kind,
            "target_phase": self.target_phase,
            "config_version": self.config_version,
            "manifest_object_key": self.manifest_object_key,
            "input_sha256": self.input_sha256,
            "object_size_bytes": self.object_size_bytes,
            "fixture": self.fixture,
            "dataset_version": None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProbeInput":
        if value.get("schema") != "gods-mlops-probe-input-v1" or value.get("input_kind") != "probe_input":
            raise ValueError("probe input reference schema is unsupported")
        if value.get("dataset_version") is not None:
            raise ValueError("probe input cannot claim a published dataset version")
        return cls(
            probe_input_id=str(value["probe_input_id"]),
            model_kind=str(value["model_kind"]),
            target_phase=str(value["target_phase"]),
            config_version=str(value["config_version"]),
            manifest_object_key=str(value["manifest_object_key"]),
            input_sha256=str(value["input_sha256"]),
            object_size_bytes=int(value["object_size_bytes"]),
            fixture=value.get("fixture") is True,
        )

    def verify(self, object_store: Any) -> dict[str, Any]:
        import json

        from gods_mlops.datasets.manifest import canonical_json, content_sha256

        payload = object_store.read_source(
            object_key=self.manifest_object_key,
            sha256_digest=self.input_sha256,
            size_bytes=self.object_size_bytes,
        )
        if len(payload) != self.object_size_bytes or content_sha256(payload) != self.input_sha256:
            raise ValueError("probe manifest blob failed frozen size or SHA-256 verification")
        try:
            manifest = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("probe manifest object is not valid UTF-8 JSON") from error
        expected = {
            "schema_version": 1,
            "fixture": True,
            "phase": "probe",
            "model_kind": self.model_kind,
            "input_kind": self.input_kind,
            "input_id": self.probe_input_id,
            "config_version": self.config_version,
        }
        if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("probe manifest content differs from its typed immutable reference")
        if content_sha256(canonical_json(manifest)) != self.input_sha256:
            raise ValueError("probe manifest canonical content hash differs from its typed reference")
        return manifest


@dataclass(frozen=True, slots=True)
class AnnotationSourceSelection:
    """Caller-selected immutable frame or completed crop for pre-publication GPU work."""

    item_kind: str
    item_id: str
    sha256: str
    revision_id: str | None = None

    def __post_init__(self) -> None:
        from uuid import UUID

        if self.item_kind not in {"frame", "crop"}:
            raise ValueError("annotation input kind must be frame or crop")
        try:
            UUID(self.item_id)
            if self.revision_id is not None:
                UUID(self.revision_id)
        except ValueError as error:
            raise ValueError("annotation source IDs must be UUIDs") from error
        if self.item_kind == "frame" and self.revision_id is not None:
            raise ValueError("frame source references do not carry a crop revision")
        if self.item_kind == "crop" and self.revision_id is None:
            raise ValueError("crop source references must pin a bbox revision")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("annotation source SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True, slots=True)
class ImmutableAnnotationItem:
    item_kind: str
    item_id: str
    sample_id: str
    sha256: str
    object_key: str
    object_size_bytes: int
    revision_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_kind": self.item_kind,
            "item_id": self.item_id,
            "sample_id": self.sample_id,
            "sha256": self.sha256,
            "object_key": self.object_key,
            "object_size_bytes": self.object_size_bytes,
            "revision_id": self.revision_id,
        }


@dataclass(frozen=True, slots=True)
class AnnotationPreparationBatch:
    """Stable immutable references for GPU draft work before any dataset is published."""

    batch_id: str
    input_sha256: str
    items: tuple[ImmutableAnnotationItem, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "input_sha256": self.input_sha256,
            "items": [item.as_dict() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """PID identity including the kernel start time, which survives PID reuse."""

    pid: int
    start_ticks: int
    uid: int | None = None
    cgroup_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.pid <= 0 or self.start_ticks <= 0:
            raise ValueError("process identity needs a positive PID and start time")
        if any(
            not isinstance(path, str)
            or not path.startswith("/")
            or "\x00" in path
            or len(path) > 4096
            for path in self.cgroup_paths
        ):
            raise ValueError("process cgroup identities must be absolute bounded paths")

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "start_ticks": self.start_ticks,
            "uid": self.uid,
            "cgroup_paths": list(self.cgroup_paths),
        }


@dataclass(frozen=True, slots=True)
class ResourceObservation:
    """One immutable sample from the Ubuntu A6000 and storage observer."""

    observation_id: str
    node_id: str
    hostname: str
    host_identity: str
    gpu_name: str
    gpu_uuid: str
    free_mib: int
    total_mib: int
    gpu_processes: tuple[ProcessIdentity, ...]
    gpu_process_list_complete: bool
    process_table: tuple[ProcessIdentity, ...]
    process_table_complete: bool
    storage_path: str
    filesystem_identity: str
    filesystem_available_bytes: int
    observed_at: Any

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ResourceObservation":
        from datetime import UTC, datetime

        def process(item: Any) -> ProcessIdentity:
            if isinstance(item, ProcessIdentity):
                return item
            if not isinstance(item, dict):
                raise ValueError("process observations must be objects")
            return ProcessIdentity(
                pid=int(item["pid"]),
                start_ticks=int(item["start_ticks"]),
                uid=int(item["uid"]) if item.get("uid") is not None else None,
                cgroup_paths=tuple(str(path) for path in item.get("cgroup_paths", [])),
            )

        if not isinstance(value, dict):
            raise ValueError("resource observation must be an object")
        observed_at = value.get("observed_at")
        if isinstance(observed_at, str):
            observed_at = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        if not isinstance(observed_at, datetime) or observed_at.tzinfo is None:
            raise ValueError("observed_at must be a timezone-aware timestamp")
        return cls(
            observation_id=str(value["observation_id"]),
            node_id=str(value["node_id"]),
            hostname=str(value["hostname"]),
            host_identity=str(value["host_identity"]),
            gpu_name=str(value["gpu_name"]),
            gpu_uuid=str(value["gpu_uuid"]),
            free_mib=int(value["free_mib"]),
            total_mib=int(value["total_mib"]),
            gpu_processes=tuple(process(item) for item in value.get("gpu_processes", [])),
            gpu_process_list_complete=value.get("gpu_process_list_complete") is True,
            process_table=tuple(process(item) for item in value.get("process_table", [])),
            process_table_complete=value.get("process_table_complete") is True,
            storage_path=str(value["storage_path"]),
            filesystem_identity=str(value["filesystem_identity"]),
            filesystem_available_bytes=int(value["filesystem_available_bytes"]),
            observed_at=observed_at.astimezone(UTC),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "node_id": self.node_id,
            "hostname": self.hostname,
            "host_identity": self.host_identity,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "free_mib": self.free_mib,
            "total_mib": self.total_mib,
            "gpu_processes": [item.as_dict() for item in self.gpu_processes],
            "gpu_process_list_complete": self.gpu_process_list_complete,
            "process_table": [item.as_dict() for item in self.process_table],
            "process_table_complete": self.process_table_complete,
            "storage_path": self.storage_path,
            "filesystem_identity": self.filesystem_identity,
            "filesystem_available_bytes": self.filesystem_available_bytes,
            "observed_at": self.observed_at.isoformat(),
        }
