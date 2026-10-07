"""Native lock-wait regression; the private gate supplies a disposable DB only."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import asyncpg
import pytest

os.environ.setdefault(
    "GODS_MLOPS_MODEL_LOCK",
    "/mnt/data/gods-mlops-clip-runtime-50d93e9/models/lock.json",
)

from gods_mlops.datasets.manifest import canonical_json
from gods_mlops.jobs.checkpoints import CheckpointIdentity
from gods_mlops.jobs.models import (
    EvaluationProbeCheckpointSource,
    ExecutionProfile,
    ProbeInput,
    ProcessIdentity,
)
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry
from gods_mlops.training.contracts import locked_model
from gods_mlops.training.probe_setup import candidate_profile

DATABASE_URL_ENV = "GODS_MLOPS_TASK11_EVAL_PROGRESS_NATIVE_DATABASE_URL"
DATABASE_NAME_ENV = "GODS_MLOPS_TASK11_EVAL_PROGRESS_NATIVE_DATABASE_NAME"
NATIVE_MARKER_ENV = "GODS_MLOPS_TASK11_EVAL_PROGRESS_NATIVE_MARKER"
EXPECTED_COMMIT_ENV = "GODS_MLOPS_TASK11_EVAL_PROGRESS_EXPECTED_COMMIT"
EXPECTED_TREE_ENV = "GODS_MLOPS_TASK11_EVAL_PROGRESS_EXPECTED_TREE"
EXPECTED_TEST_SHA_ENV = "GODS_MLOPS_TASK11_EVAL_PROGRESS_EXPECTED_TEST_SHA256"
EXPECTED_MODULE_ENV = "GODS_MLOPS_TASK11_EVAL_PROGRESS_EXPECTED_MODULE"
WRITER_APPLICATION_NAME = "gods-mlops-eval-progress-writer"
TEST_MODULE_NAME = "tests.jobs.test_evaluation_progress_queue_native"
TEST_FILE = Path(__file__).resolve()
SOURCE_ROOT = TEST_FILE.parents[2]


def _safe_identity_error(code: str) -> None:
    pytest.fail(code)


def _assert_executed_source_identity() -> None:
    expected_commit = os.environ.get(EXPECTED_COMMIT_ENV, "")
    expected_tree = os.environ.get(EXPECTED_TREE_ENV, "")
    expected_test_sha = os.environ.get(EXPECTED_TEST_SHA_ENV, "")
    expected_module = os.environ.get(EXPECTED_MODULE_ENV, "")
    if not all((expected_commit, expected_tree, expected_test_sha, expected_module)):
        _safe_identity_error("native gate source pin is incomplete")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=SOURCE_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=SOURCE_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=SOURCE_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        _safe_identity_error("native gate source identity check failed")
    test_sha = hashlib.sha256(TEST_FILE.read_bytes()).hexdigest()
    module = sys.modules.get(__name__)
    module_file = Path(getattr(module, "__file__", "")).resolve()
    module_origin = Path(getattr(getattr(module, "__spec__", None), "origin", "")).resolve()
    if (
        commit != expected_commit
        or tree != expected_tree
        or status
        or test_sha != expected_test_sha
        or expected_module != TEST_MODULE_NAME
        or module_file != TEST_FILE
        or module_origin != TEST_FILE
        or not __name__.endswith("test_evaluation_progress_queue_native")
    ):
        _safe_identity_error("native gate executed source or test differs from its reviewed pin")


def _native_database_url() -> tuple[str, str, str]:
    value = os.environ.get(DATABASE_URL_ENV, "")
    database_name = os.environ.get(DATABASE_NAME_ENV, "")
    marker = os.environ.get(NATIVE_MARKER_ENV, "")
    if not value:
        pytest.skip("owned PostgreSQL 15441 native gate is not enabled")
    if not re_fullmatch_database_name(database_name) or not re.fullmatch(r"[0-9a-f]{32}", marker):
        _safe_identity_error("native gate disposable database name is invalid")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        _safe_identity_error("native gate database target is invalid")
    if (
        parsed.scheme != "postgresql"
        or parsed.hostname != "127.0.0.1"
        or port != 15441
        or parsed.path != f"/{database_name}"
        or parsed.fragment
    ):
        _safe_identity_error("native gate refused a non-owned disposable PostgreSQL target")
    if dict(parse_query_pairs(parsed.query)).get("application_name") != WRITER_APPLICATION_NAME:
        _safe_identity_error("native gate writer application identity is missing")
    return value, database_name, marker


def re_fullmatch_database_name(value: str) -> bool:
    return re.fullmatch(r"gods_task11_eval_progress_[0-9a-f]{12}", value) is not None


def parse_query_pairs(value: str):
    from urllib.parse import parse_qsl

    return parse_qsl(value, keep_blank_values=True)


async def _seed_synthetic_training_authority(
    connection: asyncpg.Connection,
    queue: JobQueue,
) -> EvaluationProbeCheckpointSource:
    probe_profile = candidate_profile("clip", target_phase="training")
    await queue.register_profile(probe_profile)
    measurement_id = str(uuid4())
    measured_profile = ExecutionProfile(
        model_kind=probe_profile.model_kind,
        config_version=probe_profile.config_version,
        phase="training",
        memory_requirement_mib=probe_profile.memory_requirement_mib,
        artifact_reservation_bytes=probe_profile.artifact_reservation_bytes,
        config=probe_profile.config,
        candidate=False,
        measurement_id=measurement_id,
    )
    await connection.execute(
        """INSERT INTO gods_mlops_resource_profiles (
               phase,target_phase,model_kind,config_version,config_sha256,
               memory_requirement_mib,artifact_reservation_bytes,
               checkpoint_reservation_bytes,result_reservation_bytes,config_json,
               profile_state,measurement_id,oom_alternatives
           ) VALUES ('training',NULL,$1,$2,$3,$4,$5,$6,$7,$8::jsonb,'measured',$9::uuid,$10::jsonb)""",
        measured_profile.model_kind,
        measured_profile.config_version,
        measured_profile.config_sha256,
        measured_profile.memory_requirement_mib,
        measured_profile.artifact_reservation_bytes,
        measured_profile.checkpoint_reservation_bytes,
        measured_profile.result_reservation_bytes,
        canonical_json(measured_profile.config),
        measurement_id,
        canonical_json([]),
    )

    training_input_id = f"task11-native-training-{uuid4().hex}"
    training_input = ProbeInput(
        probe_input_id=training_input_id,
        model_kind="clip",
        target_phase="training",
        config_version=probe_profile.config_version,
        manifest_object_key=f"probe-inputs/{training_input_id}/manifest.json",
        input_sha256="1" * 64,
        object_size_bytes=128,
    )
    training_job_id = await queue.submit_probe(
        model_kind="clip",
        config_version=probe_profile.config_version,
        probe_input=training_input,
    )
    training_identity = CheckpointIdentity(
        job_id=training_job_id,
        input_kind="probe_input",
        input_id=training_input.probe_input_id,
        input_sha256=training_input.input_sha256,
        phase="probe",
        model_kind="clip",
        config_version=probe_profile.config_version,
        config_sha256=probe_profile.config_sha256,
        dataset_version=None,
    )
    checkpoint_sha256 = "b" * 64
    checkpoint_uri = (
        f"s3://task11-native-fixture/jobs/{training_job_id}/checkpoints/"
        + "c" * 64
        + f"/{checkpoint_sha256}.checkpoint"
    )
    model = locked_model("clip")
    worker_image_id = "sha256:" + "9" * 64
    source_commit = "f" * 40
    runtime_evidence_sha256 = "d" * 64
    source = EvaluationProbeCheckpointSource(
        training_probe_job_id=training_job_id,
        model_kind="clip",
        model_id=model.model_id,
        model_revision=model.revision,
        checkpoint_uri=checkpoint_uri,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size_bytes=123,
        checkpoint_identity=training_identity.as_dict(),
        worker_image_id=worker_image_id,
        source_commit=source_commit,
        runtime_evidence_sha256=runtime_evidence_sha256,
    )
    checkpoint_commit = {
        "uri": checkpoint_uri,
        "checkpoint_uri": checkpoint_uri,
        "sha256": checkpoint_sha256,
        "size_bytes": source.checkpoint_size_bytes,
        "identity": training_identity.as_dict(),
    }
    result_artifact = {
        "kind": "model",
        "uri": f"s3://task11-native-fixture/jobs/{training_job_id}/results/model.tar",
        "sha256": "8" * 64,
        "size_bytes": 4096,
        "identity": training_identity.as_dict(),
        "object_key": f"jobs/{training_job_id}/results/model.tar",
    }
    evidence = {
        "event": "task8_real_model_probe_complete",
        "job_id": training_job_id,
        "model_kind": "clip",
        "target_phase": "training",
        "config_version": probe_profile.config_version,
        "input_sha256": training_input.input_sha256,
        "docker_image_id": worker_image_id,
        "image_source_commit": source_commit,
        "source_commit": source_commit,
    }
    evidence_sha256 = hashlib.sha256(canonical_json(evidence).encode("utf-8")).hexdigest()
    runtime_authority = {
        "schema": "gods-mlops-probe-runtime-evidence-v1",
        "evidence_sha256": runtime_evidence_sha256,
        "evidence_canonical_sha256": evidence_sha256,
        "evidence_projection_sha256": evidence_sha256,
        "evidence": evidence,
        "job_id": training_job_id,
        "measurement_id": measurement_id,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity": training_identity.as_dict(),
        "result_artifact": result_artifact,
    }
    verification = {
        "passed": True,
        "learning_signal_verified": True,
        "checkpoint_resume_verified": True,
    }
    await connection.execute(
        """INSERT INTO gods_mlops_profile_measurements (
               measurement_id,job_id,input_sha256,config_sha256,model_kind,target_phase,
               result_state,peak_allocated_mib,peak_reserved_mib,optimizer_steps,
               checkpoint_resumed,verification_details,checkpoint_sha256,exit_code
           ) VALUES ($1::uuid,$2::uuid,$3,$4,'clip','training','succeeded',1,1,3,TRUE,$5::jsonb,$6,0)""",
        measurement_id,
        training_job_id,
        training_input.input_sha256,
        probe_profile.config_sha256,
        canonical_json(verification),
        checkpoint_sha256,
    )
    await connection.execute(
        """UPDATE gods_mlops_jobs SET state='completed',checkpoint_uri=$2,
               checkpoint_sha256=$3,checkpoint_identity=$4::jsonb,completed_at=now()
           WHERE job_id=$1::uuid""",
        training_job_id,
        checkpoint_uri,
        checkpoint_sha256,
        canonical_json(training_identity.as_dict()),
    )
    for event_type, details in (
        ("checkpoint_committed", checkpoint_commit),
        ("result_artifact_committed", result_artifact),
        ("probe_runtime_evidence_committed", runtime_authority),
    ):
        await connection.execute(
            """INSERT INTO gods_mlops_job_events (
                   job_id,event_type,state,fencing_token,details
               ) VALUES ($1::uuid,$2,'completed',1,$3::jsonb)""",
            training_job_id,
            event_type,
            canonical_json(details),
        )
    return source


async def _create_armed_evaluation_job(
    queue: JobQueue,
    repository: PostgresJobQueueRepository,
    source: EvaluationProbeCheckpointSource,
) -> tuple[str, dict, ProbeInput, object]:
    profile = candidate_profile("clip", target_phase="evaluation")
    await queue.register_profile(profile)
    probe_input_id = f"task9-clip-evaluation-probe-{source.checkpoint_sha256}"
    probe_input = ProbeInput(
        probe_input_id=probe_input_id,
        model_kind="clip",
        target_phase="evaluation",
        config_version=profile.config_version,
        manifest_object_key=f"probe-inputs/{probe_input_id}/manifest.json",
        input_sha256=hashlib.sha256(probe_input_id.encode("utf-8")).hexdigest(),
        object_size_bytes=128,
        evaluation_checkpoint_source=source,
    )
    submitted_after = await repository.artifact_database_clock()
    job_id = await queue.submit_probe(
        model_kind="clip",
        config_version=profile.config_version,
        probe_input=probe_input,
    )
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
        "submitted_after": submitted_after.isoformat(),
    }
    armed = await queue.request_yield(
        job_id,
        "task11_clip_evaluation_cursor",
        after_progress=progress,
    )
    if armed is None or armed.get("pending_first_attempt") is not True:
        _safe_identity_error("native fixture could not arm its generation-zero probe")
    return job_id, profile.config_sha256, probe_input, armed


async def _grant_first_fence(
    connection: asyncpg.Connection,
    repository: PostgresJobQueueRepository,
    job_id: str,
    profile,
    *,
    lifetime_seconds: int,
) -> tuple[str, ProcessIdentity, dict, datetime]:
    token = str(uuid4())
    gpu_uuid = f"GPU-NATIVE-EVAL-PROGRESS-{uuid4().hex}"
    owner = ProcessIdentity(pid=1_900_000_000, start_ticks=1_800_000_000, uid=10001)
    now = await repository.artifact_database_clock()
    expires_at = now + timedelta(seconds=lifetime_seconds)
    async with connection.transaction():
        await connection.execute(
            """INSERT INTO gods_mlops_gpu_leases (
                   gpu_uuid,job_id,lease_token,fencing_token,memory_requirement_mib,
                   expires_at,granted_observation_id
               ) VALUES ($1,$2::uuid,$3::uuid,1,$4,$5,$6::uuid)""",
            gpu_uuid,
            job_id,
            token,
            profile.memory_requirement_mib,
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
        candidate_deadline_at=expires_at,
    )
    return token, owner, deadline, expires_at


async def _wait_for_profile_lock_wait(connection: asyncpg.Connection) -> None:
    loop = asyncio.get_running_loop()
    timeout_at = loop.time() + 10
    while loop.time() < timeout_at:
        blocked = await connection.fetchval(
            """SELECT EXISTS (
                   SELECT 1 FROM pg_stat_activity
                   WHERE datname=current_database()
                     AND application_name=$1
                     AND wait_event_type='Lock'
                     AND query ILIKE '%gods_mlops_resource_profiles%'
               )""",
            WRITER_APPLICATION_NAME,
        )
        if blocked:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("native writer did not block on the locked profile row")


def test_native_profile_lock_wait_expiry_rolls_back_progress() -> None:
    database_url, _database_name, expected_marker = _native_database_url()
    _assert_executed_source_identity()

    async def exercise() -> None:
        connection = await asyncpg.connect(database_url, timeout=5)
        repository = PostgresJobQueueRepository(database_url=database_url)
        sources = DatasetSourceRegistry(database_url=database_url)
        queue = JobQueue(repository=repository, sources=sources)
        blocker = await asyncpg.connect(
            database_url,
            timeout=5,
            server_settings={"application_name": "gods-mlops-eval-progress-profile-lock"},
        )
        blocker_transaction = None
        record_task = None
        try:
            database_identity = await connection.fetchrow(
                """SELECT current_database() AS name,current_user AS owner,
                          current_setting('server_version_num')::int AS version"""
            )
            if (
                database_identity is None
                or database_identity["name"] != _database_name
                or database_identity["owner"] != "postgres"
                or int(database_identity["version"]) // 10_000 != 16
            ):
                _safe_identity_error("native fixture database identity changed")
            marker = await connection.fetchval(
                "SELECT marker FROM public.task11_eval_progress_gate_identity LIMIT 1"
            )
            if marker != expected_marker:
                _safe_identity_error("native fixture database marker differs from the protected gate")
            await queue.ensure_schema()
            schema_version = await connection.fetchval(
                "SELECT max(version) FROM gods_mlops_schema_migrations"
            )
            if int(schema_version or 0) != 16:
                _safe_identity_error("native fixture schema migration version is not 16")

            source = await _seed_synthetic_training_authority(connection, queue)
            job_id, _config_sha256, probe_input, armed = await _create_armed_evaluation_job(
                queue,
                repository,
                source,
            )
            profile = candidate_profile("clip", target_phase="evaluation")
            token, owner, deadline, expires_at = await _grant_first_fence(
                connection,
                repository,
                job_id,
                profile,
                lifetime_seconds=8,
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
                _safe_identity_error("native fixture could not read its armed first fence")
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
            blocker_transaction = blocker.transaction()
            await blocker_transaction.start()
            await blocker.fetchrow(
                """SELECT config_version FROM gods_mlops_resource_profiles
                   WHERE phase='probe' AND model_kind='clip' AND config_version=$1
                   FOR UPDATE""",
                profile.config_version,
            )
            record_task = asyncio.create_task(
                queue.record_evaluation_progress(
                    job_id=job_id,
                    lease_token=token,
                    fencing_token=1,
                    arm_event_id=armed["arm_event_id"],
                    progress=progress,
                    owner=owner,
                    artifact_deadline=deadline,
                )
            )
            await _wait_for_profile_lock_wait(connection)
            while await repository.artifact_database_clock() < expires_at:
                await asyncio.sleep(0.05)
            await blocker_transaction.rollback()
            blocker_transaction = None
            try:
                await asyncio.wait_for(record_task, timeout=10)
            except (RuntimeError, TimeoutError):
                pass
            else:
                _safe_identity_error("expired profile-lock wait published evaluation progress")
            record_task = None

            late_events = await connection.fetch(
                """SELECT event_type FROM gods_mlops_job_events
                   WHERE job_id=$1::uuid AND event_type IN
                     ('evaluation_batch_completed','yield_requested')""",
                job_id,
            )
            job = await connection.fetchrow(
                "SELECT state,checkpoint_sha256 FROM gods_mlops_jobs WHERE job_id=$1::uuid",
                job_id,
            )
            if late_events or job is None or job["state"] != "running" or job["checkpoint_sha256"] is not None:
                _safe_identity_error("expired profile-lock wait left progress, yield, or cursor state")
        finally:
            if blocker_transaction is not None:
                await blocker_transaction.rollback()
            if record_task is not None and not record_task.done():
                record_task.cancel()
                await asyncio.gather(record_task, return_exceptions=True)
            await blocker.close()
            await repository.close()
            await sources.close()
            await connection.close()

    asyncio.run(exercise())
