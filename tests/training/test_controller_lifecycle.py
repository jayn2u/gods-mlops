from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from gods_mlops.jobs.models import ResourceObservation
from gods_mlops.training.claims import WorkerAuthorizationError
from gods_mlops.training.controller import TrainingController


GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
JOB_ID = "a0320b59-663c-4cdc-b893-086bb970ea60"
TOKEN_1 = "8ad96890-3434-4f07-85bb-8cde17a2b009"
TOKEN_2 = "9bd96890-3434-4f07-85bb-8cde17a2b009"


def _observation() -> ResourceObservation:
    return ResourceObservation(
        observation_id=str(uuid4()),
        node_id="ubuntu",
        hostname="ubuntu",
        host_identity="machine-sha256:controller-test",
        gpu_name="NVIDIA RTX A6000",
        gpu_uuid=GPU_UUID,
        free_mib=48_000,
        total_mib=49_140,
        gpu_processes=(),
        gpu_process_list_complete=True,
        process_table=(),
        process_table_complete=True,
        storage_path="/data",
        filesystem_identity="ext4:uuid=controller-test",
        filesystem_available_bytes=2**40,
        observed_at=datetime.now(UTC),
    )


def _job(state: str, lease_token: str | None) -> dict:
    return {
        "job_id": JOB_ID,
        "state": state,
        "lease_token": lease_token,
        "phase": "training",
        "model_kind": "detr",
        "config_version": "detr-trained-v1",
    }


def _lease(token: str, generation: int) -> dict:
    return {
        "job_id": JOB_ID,
        "lease_token": token,
        "fencing_token": generation,
        "gpu_uuid": GPU_UUID,
        "owner_pid": 7312,
        "owner_start_ticks": 238191,
        "owner_uid": 10001,
    }


class _Queue:
    def __init__(self, job: dict) -> None:
        self.job = job
        self.source_registry = SimpleNamespace(close=lambda: None)

    async def get(self, job_id: str) -> dict:
        assert job_id == JOB_ID
        return dict(self.job)


class _Repository:
    def __init__(self, job: dict, lease: dict | None) -> None:
        self.job = job
        self.lease = lease

    async def get_active_lease(self, gpu_uuid: str) -> dict | None:
        assert gpu_uuid == GPU_UUID
        return self.lease

    async def get_profile(self, *, phase: str, model_kind: str, config_version: str) -> dict:
        return {
            "phase": phase,
            "model_kind": model_kind,
            "config_version": config_version,
            "profile_state": "measured",
            "config_sha256": "2" * 64,
        }


class _Adapter:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.read: list[str] = []

    def ensure_worker(self, *, job: dict, lease: dict, profile: dict):
        if job["state"] != "running":
            raise WorkerAuthorizationError("worker job is not currently running")
        self.created.append(str(lease["lease_token"]))
        return SimpleNamespace(metadata=SimpleNamespace(uid="job-uid-running"))

    def read_existing_worker(self, *, job: dict, lease: dict, profile: dict):
        self.read.append(str(lease["lease_token"]))
        return SimpleNamespace(metadata=SimpleNamespace(uid=f"job-uid-{lease['fencing_token']}"))


class _Admission:
    expected_gpu_uuid = GPU_UUID

    def __init__(self, job: dict, repository: _Repository) -> None:
        self.job = job
        self.repository = repository
        self.calls = 0

    async def admit(self, job_id: str, observation: ResourceObservation) -> dict:
        assert job_id == JOB_ID
        self.calls += 1
        self.job.update(state="running", lease_token=TOKEN_2, lease_generation=2)
        self.repository.lease = _lease(TOKEN_2, 2)
        return dict(self.job)


class _Observer:
    def observe(self) -> ResourceObservation:
        return _observation()


class _Monitor:
    def __init__(self, job: dict, repository: _Repository) -> None:
        self.job = job
        self.repository = repository
        self.calls = 0

    async def observe(self, observation: ResourceObservation) -> dict:
        self.calls += 1
        if self.job["state"] == "completed":
            self.repository.lease = None
        elif self.job["state"] == "yield_requested" and self.calls == 2:
            self.job.update(state="waiting_gpu", lease_token=None)
            self.repository.lease = None
        elif self.job["state"] == "running":
            self.job.update(state="completed")
            self.repository.lease = None
        return dict(self.job)


def _controller(job: dict, *, repository: _Repository, adapter: _Adapter, admission: _Admission, monitor: _Monitor):
    queue = _Queue(job)
    now = [0.0]

    async def sleep(seconds: float) -> None:
        now[0] += seconds

    controller = TrainingController(
        queue=queue,
        repository=repository,
        admission=admission,
        monitor=monitor,
        observer=_Observer(),
        worker_adapter=adapter,
        core_api=SimpleNamespace(),
        namespace="gods-mlops",
        sleep=sleep,
        clock=lambda: now[0],
    )
    return controller


def test_completed_job_with_live_lease_reconciles_existing_worker_without_launching() -> None:
    async def exercise() -> None:
        job = _job("completed", TOKEN_1)
        repository = _Repository(job, _lease(TOKEN_1, 1))
        adapter = _Adapter()
        admission = _Admission(job, repository)
        monitor = _Monitor(job, repository)
        controller = _controller(
            job, repository=repository, adapter=adapter, admission=admission, monitor=monitor
        )

        completed = await controller.run(JOB_ID, timeout_seconds=10)

        assert completed["state"] == "completed"
        assert adapter.created == []
        assert adapter.read == [TOKEN_1]
        assert monitor.calls == 1
        assert repository.lease is None

    asyncio.run(exercise())


def test_yielded_live_owner_is_monitored_before_a_new_fence_can_launch() -> None:
    async def exercise() -> None:
        job = _job("yield_requested", TOKEN_1)
        job["lease_generation"] = 1
        repository = _Repository(job, _lease(TOKEN_1, 1))
        adapter = _Adapter()
        admission = _Admission(job, repository)
        monitor = _Monitor(job, repository)
        controller = _controller(
            job, repository=repository, adapter=adapter, admission=admission, monitor=monitor
        )

        completed = await controller.run(JOB_ID, timeout_seconds=25)

        assert completed["state"] == "completed"
        assert adapter.read[:2] == [TOKEN_1, TOKEN_1]
        assert adapter.created == [TOKEN_2]
        assert admission.calls == 1
        assert monitor.calls == 3
        assert repository.lease is None

    asyncio.run(exercise())


def test_controller_binds_one_authority_deadline_after_admission_and_passes_it_to_worker() -> None:
    async def exercise() -> None:
        job = _job("waiting_gpu", None)
        repository = _Repository(job, None)
        db_now = datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
        monotonic_now = [0.0]
        deadline_records = []

        class DeadlineRepository(_Repository):
            async def artifact_database_clock(self):
                monotonic_now[0] += 2.0
                return db_now

            async def bind_worker_artifact_deadline(
                self,
                *,
                job_id,
                lease_token,
                fencing_token,
                controller_invocation_id,
                candidate_deadline_at,
            ):
                record = {
                    "job_id": job_id,
                    "lease_token": lease_token,
                    "fencing_token": fencing_token,
                    "controller_invocation_id": controller_invocation_id,
                    "artifact_deadline_at": candidate_deadline_at,
                }
                deadline_records.append(record)
                return record

        class DeadlineAdapter(_Adapter):
            def __init__(self):
                super().__init__()
                self.deadlines = []

            def ensure_worker(self, *, job, lease, profile, artifact_deadline=None):
                self.deadlines.append(artifact_deadline)
                return super().ensure_worker(job=job, lease=lease, profile=profile)

        repository = DeadlineRepository(job, None)
        adapter = DeadlineAdapter()
        admission = _Admission(job, repository)
        monitor = _Monitor(job, repository)
        controller = TrainingController(
            queue=_Queue(job),
            repository=repository,
            admission=admission,
            monitor=monitor,
            observer=_Observer(),
            worker_adapter=adapter,
            core_api=SimpleNamespace(),
            namespace="gods-mlops",
            sleep=lambda _seconds: asyncio.sleep(0),
            clock=lambda: monotonic_now[0],
        )

        completed = await controller.run(JOB_ID, timeout_seconds=100)

        assert completed["state"] == "completed"
        assert len(deadline_records) == 1
        assert UUID(deadline_records[0]["controller_invocation_id"])
        assert deadline_records[0]["artifact_deadline_at"] == db_now + timedelta(seconds=98)
        assert adapter.deadlines == deadline_records

    asyncio.run(exercise())
