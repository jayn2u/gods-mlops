"""Narrow PostgreSQL regression for the Task 11 CLIP evaluation progress fence.

The exact loopback database URL is injected only by the private owned-15441 gate.
This test never starts containers, opens S3, or reads checkpoint payload bytes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import replace
from datetime import timedelta
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg
import pytest

os.environ.setdefault(
    "GODS_MLOPS_MODEL_LOCK",
    "/mnt/data/gods-mlops-clip-runtime-50d93e9/models/lock.json",
)

from gods_mlops.jobs.models import (
    EvaluationProbeCheckpointSource,
    ProbeInput,
    ProcessIdentity,
)
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry
from gods_mlops.training.contracts import locked_model
from gods_mlops.training.probe_setup import candidate_profile

TRAINING_PROBE_JOB_ID = "40d6c455-f8b3-4cab-a21b-6627f0f885fd"
TRAINING_CHECKPOINT_SHA256 = "eb6e7dbff626c0907a2e0aa29eb69ffe67b52c52aa7c9da8b94dd8d01bbc16c5"
TRAINING_CHECKPOINT_SIZE_BYTES = 1_795_903_071
TRAINING_MODEL_SHA256 = "64da9172afec046e0949b9564d3df8b484704e947748cdaac4096ecc67192464"
TRAINING_RUNTIME_EVIDENCE_SHA256 = "4c8e272457fa5ec255793298227e6cdca446cb91f8deeb523f2eaaab8384ca81"
TRAINING_IMAGE_ID = "sha256:7d209472d5b861404fce0ee8b88f7ff1d495b8e513cfe5da1bf0137f351568f7"
SOURCE_COMMIT = "50d93e97ccca6142602c47345f53a8a825ffe6b2"
DATABASE_NAME = "gods_mlopsgpu_task11_clip50_ordinary_20261007_26b7e1f07485"
DATABASE_URL_ENV = "GODS_MLOPS_TASK11_EVAL_PROGRESS_NATIVE_DATABASE_URL"


def _native_database_url() -> str:
    value = os.environ.get(DATABASE_URL_ENV)
    if not value:
        pytest.skip("owned PostgreSQL 15441 native gate is not enabled")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        pytest.fail("native evaluation progress gate received an invalid protected database target")
    if (
        parsed.scheme != "postgresql"
        or parsed.hostname != "127.0.0.1"
        or port != 15441
        or parsed.path != f"/{DATABASE_NAME}"
    ):
        pytest.fail("native evaluation progress gate refused a non-owned PostgreSQL target")
    return value


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


async def _training_source(connection) -> EvaluationProbeCheckpointSource:
    training_job = await connection.fetchrow(
        "SELECT * FROM gods_mlops_jobs WHERE job_id=$1::uuid",
        TRAINING_PROBE_JOB_ID,
    )
    checkpoint_event = await connection.fetchrow(
        """SELECT details FROM gods_mlops_job_events
           WHERE job_id=$1::uuid AND event_type='checkpoint_committed'
           ORDER BY event_id DESC LIMIT 1""",
        TRAINING_PROBE_JOB_ID,
    )
    runtime_event = await connection.fetchrow(
        """SELECT details FROM gods_mlops_job_events
           WHERE job_id=$1::uuid AND event_type='probe_runtime_evidence_committed'
           ORDER BY event_id DESC LIMIT 1""",
        TRAINING_PROBE_JOB_ID,
    )
    model_event = await connection.fetchrow(
        """SELECT details FROM gods_mlops_job_events
           WHERE job_id=$1::uuid AND event_type='result_artifact_committed'
           ORDER BY event_id DESC LIMIT 1""",
        TRAINING_PROBE_JOB_ID,
    )
    if training_job is None or checkpoint_event is None or runtime_event is None or model_event is None:
        pytest.fail("native gate could not find the pinned training authority rows")
    checkpoint = _json(checkpoint_event["details"])
    runtime = _json(runtime_event["details"])
    model_result = _json(model_event["details"])
    evidence = runtime.get("evidence")
    if not isinstance(evidence, dict):
        pytest.fail("native gate found malformed committed training runtime evidence")
    if str(model_result.get("sha256", "")).strip() != TRAINING_MODEL_SHA256:
        pytest.fail("native gate model artifact identity changed")
    model = locked_model("clip")
    source = EvaluationProbeCheckpointSource(
        training_probe_job_id=TRAINING_PROBE_JOB_ID,
        model_kind="clip",
        model_id=model.model_id,
        model_revision=model.revision,
        checkpoint_uri=str(checkpoint.get("uri", checkpoint.get("checkpoint_uri", ""))),
        checkpoint_sha256=str(checkpoint.get("sha256", "")).strip(),
        checkpoint_size_bytes=int(checkpoint.get("size_bytes", 0)),
        checkpoint_identity=_json(training_job["checkpoint_identity"]),
        worker_image_id=str(evidence.get("docker_image_id", "")),
        source_commit=str(evidence.get("source_commit", "")),
        runtime_evidence_sha256=str(runtime.get("evidence_sha256", "")),
    )
    if (
        source.checkpoint_sha256 != TRAINING_CHECKPOINT_SHA256
        or source.checkpoint_size_bytes != TRAINING_CHECKPOINT_SIZE_BYTES
        or source.worker_image_id != TRAINING_IMAGE_ID
        or source.source_commit != SOURCE_COMMIT
        or source.runtime_evidence_sha256 != TRAINING_RUNTIME_EVIDENCE_SHA256
    ):
        pytest.fail("native gate training checkpoint/image/evidence identity changed")
    return source


def test_native_arm_consume_replay_and_expired_lease_refusal() -> None:
    database_url = _native_database_url()

    async def exercise() -> None:
        connection = await asyncpg.connect(database_url)
        repository = PostgresJobQueueRepository(database_url=database_url)
        sources = DatasetSourceRegistry(database_url=database_url)
        queue = JobQueue(repository=repository, sources=sources)
        job_id = None
        profile_version = f"task11-clip-eval-progress-native-{uuid4().hex[:12]}"
        profile_created = False
        token = str(uuid4())
        gpu_uuid = f"GPU-TEST-EVAL-PROGRESS-{uuid4().hex}"
        owner = ProcessIdentity(pid=1_900_000_000, start_ticks=1_800_000_000, uid=10001)
        try:
            schema_version = await connection.fetchval(
                "SELECT max(version) FROM gods_mlops_schema_migrations"
            )
            if int(schema_version or 0) != 16:
                pytest.fail("native evaluation progress gate requires the already-migrated schema version 16")
            # Do not let repository setup attempt bootstrap/schema work in this
            # native fixture; the protected gate has already checked version 16.
            repository._schema_ready = True
            # The safe gate verified this existing schema. Keep repository
            # helpers from applying migrations while the test writes fixtures.
            repository._schema_ready = True

            source = await _training_source(connection)
            base_profile = candidate_profile("clip", target_phase="evaluation")
            profile = replace(base_profile, config_version=profile_version)
            existing_profile = await connection.fetchrow(
                """SELECT profile_state,config_sha256 FROM gods_mlops_resource_profiles
                   WHERE phase='probe' AND model_kind='clip' AND config_version=$1""",
                profile_version,
            )
            if existing_profile is not None:
                pytest.fail("native gate test profile version unexpectedly already exists")
            await queue.register_profile(profile)
            profile_created = True

            probe_input_id = f"task11-clip-progress-native-{uuid4().hex}"
            probe_input = ProbeInput(
                probe_input_id=probe_input_id,
                model_kind="clip",
                target_phase="evaluation",
                config_version=profile_version,
                manifest_object_key=f"probe-inputs/{probe_input_id}/manifest.json",
                input_sha256=hashlib.sha256(probe_input_id.encode("utf-8")).hexdigest(),
                object_size_bytes=256,
                evaluation_checkpoint_source=source,
            )
            submitted_after = await repository.artifact_database_clock()
            job_id = await queue.submit_probe(
                model_kind="clip",
                config_version=profile_version,
                probe_input=probe_input,
            )
            target = {
                "phase": "probe",
                "target_phase": "evaluation",
                "model_kind": "clip",
                "stage": "queries",
                "next_index": 2,
                "total_items": 2,
                "completed_batch_count": 1,
                "input_sha256": probe_input.input_sha256,
                "config_sha256": profile.config_sha256,
                "training_probe_job_id": source.training_probe_job_id,
                "training_source_checkpoint_sha256": source.checkpoint_sha256,
                "submitted_after": submitted_after.isoformat(),
            }
            armed = await queue.request_yield(
                job_id,
                "task11_clip_evaluation_cursor",
                after_progress=target,
            )
            if armed is None or armed.get("pending_first_attempt") is not True:
                pytest.fail("native gate could not arm the new unleased evaluation probe")

            database_now = await repository.artifact_database_clock()
            expires_at = database_now + timedelta(minutes=2)
            async with connection.transaction():
                await connection.execute(
                    """INSERT INTO gods_mlops_gpu_leases (
                           gpu_uuid,job_id,lease_token,fencing_token,memory_requirement_mib,
                           expires_at,granted_observation_id
                       ) VALUES ($1,$2::uuid,$3::uuid,1,36000,$4,$5::uuid)""",
                    gpu_uuid,
                    job_id,
                    token,
                    expires_at,
                    str(uuid4()),
                )
                await connection.execute(
                    """UPDATE gods_mlops_jobs SET state='running',lease_token=$2::uuid,
                           lease_generation=1,lease_expires_at=$3,updated_at=now()
                       WHERE job_id=$1::uuid""",
                    job_id,
                    token,
                    expires_at,
                )
            await repository.bind_process(job_id, token, owner)
            deadline = await repository.bind_worker_artifact_deadline(
                job_id=job_id,
                lease_token=token,
                fencing_token=1,
                controller_invocation_id=str(uuid4()),
                candidate_deadline_at=(await repository.artifact_database_clock()) + timedelta(minutes=5),
            )
            pending = await queue.pending_evaluation_progress_yield(
                job_id=job_id,
                lease_token=token,
                fencing_token=1,
                input_sha256=probe_input.input_sha256,
                config_sha256=profile.config_sha256,
                training_source_checkpoint_sha256=source.checkpoint_sha256,
            )
            if pending is None or pending["arm_event_id"] != armed["arm_event_id"]:
                pytest.fail("native gate could not find its exact first-generation arm")

            progress = {
                "phase": "probe",
                "target_phase": "evaluation",
                "model_kind": "clip",
                "stage": "queries",
                "next_index": 2,
                "total_items": 2,
                "completed_batch_count": 1,
                "input_sha256": probe_input.input_sha256,
                "config_sha256": profile.config_sha256,
                "training_probe_job_id": source.training_probe_job_id,
                "training_source_checkpoint_sha256": source.checkpoint_sha256,
            }

            expired_at = (await repository.artifact_database_clock()) - timedelta(seconds=1)
            await connection.execute(
                "UPDATE gods_mlops_gpu_leases SET expires_at=$2 WHERE job_id=$1::uuid",
                job_id,
                expired_at,
            )
            await connection.execute(
                "UPDATE gods_mlops_jobs SET lease_expires_at=$2 WHERE job_id=$1::uuid",
                job_id,
                expired_at,
            )
            with pytest.raises(RuntimeError):
                await queue.record_evaluation_progress(
                    job_id=job_id,
                    lease_token=token,
                    fencing_token=1,
                    arm_event_id=armed["arm_event_id"],
                    progress=progress,
                    owner=owner,
                    artifact_deadline=deadline,
                )
            expired_events = await connection.fetch(
                """SELECT event_type FROM gods_mlops_job_events WHERE job_id=$1::uuid
                   AND event_type IN ('evaluation_batch_completed','yield_requested')""",
                job_id,
            )
            if expired_events:
                pytest.fail("expired lease published evaluation progress or yield")

            expires_at = (await repository.artifact_database_clock()) + timedelta(minutes=2)
            await connection.execute(
                "UPDATE gods_mlops_gpu_leases SET expires_at=$2 WHERE job_id=$1::uuid",
                job_id,
                expires_at,
            )
            await connection.execute(
                "UPDATE gods_mlops_jobs SET lease_expires_at=$2 WHERE job_id=$1::uuid",
                job_id,
                expires_at,
            )
            triggered = await queue.record_evaluation_progress(
                job_id=job_id,
                lease_token=token,
                fencing_token=1,
                arm_event_id=armed["arm_event_id"],
                progress=progress,
                owner=owner,
                artifact_deadline=deadline,
            )
            replay = await queue.record_evaluation_progress(
                job_id=job_id,
                lease_token=token,
                fencing_token=1,
                arm_event_id=armed["arm_event_id"],
                progress=progress,
                owner=owner,
                artifact_deadline=deadline,
            )
            if triggered.get("status") != "yield_requested" or replay.get("idempotent_replay") is not True:
                pytest.fail("native progress consume/replay did not preserve its exact event identity")
            row = await connection.fetchrow(
                "SELECT state,checkpoint_sha256 FROM gods_mlops_jobs WHERE job_id=$1::uuid",
                job_id,
            )
            if row is None or row["state"] != "yield_requested" or row["checkpoint_sha256"] is not None:
                pytest.fail("native progress consume was mistaken for a committed cursor")
            events = await connection.fetch(
                """SELECT event_type,fencing_token,details FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type IN
                     ('evaluation_batch_completed','yield_requested') ORDER BY event_id""",
                job_id,
            )
            if len(events) != 2 or any(int(event["fencing_token"] or 0) != 1 for event in events):
                pytest.fail("native progress and yield events do not share the expected first fence")
        finally:
            if job_id is not None:
                async with connection.transaction():
                    await connection.execute(
                        "DELETE FROM gods_mlops_worker_artifact_deadlines WHERE job_id=$1::uuid",
                        job_id,
                    )
                    await connection.execute(
                        "DELETE FROM gods_mlops_gpu_leases WHERE job_id=$1::uuid",
                        job_id,
                    )
                    await connection.execute(
                        "DELETE FROM gods_mlops_artifact_reservations WHERE job_id=$1::uuid",
                        job_id,
                    )
                    await connection.execute(
                        "DELETE FROM gods_mlops_job_events WHERE job_id=$1::uuid",
                        job_id,
                    )
                    await connection.execute(
                        "DELETE FROM gods_mlops_jobs WHERE job_id=$1::uuid",
                        job_id,
                    )
            if profile_created:
                await connection.execute(
                    """DELETE FROM gods_mlops_resource_profiles
                       WHERE phase='probe' AND model_kind='clip' AND config_version=$1
                         AND measurement_id IS NULL""",
                    profile_version,
                )
            await connection.close()
            await repository.close()
            await sources.close()

    asyncio.run(exercise())
