from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from gods_mlops.evaluation.report import load_operator_evaluation_report


class _ReportRepository:
    def __init__(self, artifacts, identity):
        self.artifacts = artifacts
        self.identity = identity

    async def result_artifacts_for(self, job_id):
        return self.artifacts

    async def checkpoint_identity(self, job_id):
        return self.identity


class _Queue:
    def __init__(self, job, artifacts=(), identity=None, source_reasons=()):
        self.job = job
        self.repository = _ReportRepository(list(artifacts), identity)
        self.source_reasons = tuple(source_reasons)

    async def get(self, job_id):
        assert job_id == self.job["job_id"]
        return self.job

    async def evaluation_source_block_reasons(self, job_id):
        return self.source_reasons


class _Identity:
    def __init__(self, job_id):
        self.job_id = job_id

    def as_dict(self):
        return {"job_id": self.job_id, "phase": "evaluation"}


class _ResultStore:
    def __init__(self, payload):
        self.payload = payload
        self.reads = []

    def read_committed(self, details, *, expected_identity):
        self.reads.append((details, expected_identity))
        return object(), self.payload


def _evaluation_job():
    job_id = str(uuid4())
    return {
        "job_id": job_id,
        "phase": "evaluation",
        "state": "completed",
        "dataset_version": "dataset-123",
        "input_sha256": "a" * 64,
        "model_kind": "clip",
        "config_version": "eval-config-v1",
        "config_sha256": "b" * 64,
        "source_refs": {"checkpoint": {"checkpoint_sha256": "c" * 64}},
    }


def _evaluation_report(job):
    return {
        "schema_version": 1,
        "execution_status": "succeeded",
        "status": "complete",
        "model_kind": "clip",
        "candidate": {"checkpoint_sha256": "c" * 64},
        "baseline": {"verified": True},
        "source": {"dataset_version": "dataset-123", "manifest_sha256": "a" * 64},
        "evaluation_config": {"version": "eval-config-v1", "sha256": "b" * 64},
        "evaluation_job": {
            "job_id": job["job_id"],
            "config_version": "eval-config-v1",
            "config_sha256": "b" * 64,
        },
        "metrics": {"recall_at_5": 0.72},
    }


def test_missing_report_is_explicit_and_never_synthesized() -> None:
    async def exercise() -> None:
        job = _evaluation_job()
        queue = _Queue(job, artifacts=[], identity=_Identity(job["job_id"]))
        store = _ResultStore(b"")

        result = await load_operator_evaluation_report(job["job_id"], queue=queue, result_store=store)

        assert result == {
            "availability": "not_available",
            "job_id": job["job_id"],
            "reason_code": "evaluation_result_artifact_missing",
            "report": None,
            "deployment_eligibility": None,
            "current_source_reasons": [],
        }
        assert store.reads == []

    asyncio.run(exercise())


def test_committed_report_uses_existing_release_gate_and_current_source_overlay() -> None:
    async def exercise() -> None:
        job = _evaluation_job()
        artifact = {"kind": "evaluation", "sha256": "d" * 64, "size_bytes": 42}
        payload = json.dumps(_evaluation_report(job)).encode("utf-8")
        queue = _Queue(
            job,
            artifacts=[artifact],
            identity=_Identity(job["job_id"]),
            source_reasons=("source_sample_explicitly_invalidated",),
        )
        store = _ResultStore(payload)

        result = await load_operator_evaluation_report(job["job_id"], queue=queue, result_store=store)

        assert result["availability"] == "available"
        assert result["report"]["metrics"] == {"recall_at_5": 0.72}
        assert result["current_source_reasons"] == ["source_sample_explicitly_invalidated"]
        gate = result["deployment_eligibility"]
        assert gate["eligible"] is False
        assert gate["status"] == "blocked"
        assert "cuhk_report_missing" in gate["reasons"]
        assert "trusted_quality_policy_missing" in gate["reasons"]
        assert "source_sample_explicitly_invalidated" in gate["reasons"]
        assert len(store.reads) == 1

    asyncio.run(exercise())
