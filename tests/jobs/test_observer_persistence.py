from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from gods_mlops.jobs.models import ResourceObservation
from gods_mlops.jobs.queue import ObservationReplayError, PostgresJobQueueRepository

GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data/jayn2u/gods-mlops"


def _observation(at: datetime, **overrides) -> ResourceObservation:
    return ResourceObservation.from_dict({
        "observation_id": str(overrides.pop("observation_id", uuid4())),
        "node_id": "ubuntu",
        "hostname": "ubuntu",
        "host_identity": HOST_IDENTITY,
        "gpu_name": "NVIDIA RTX A6000",
        "gpu_uuid": GPU_UUID,
        "free_mib": 48_000,
        "total_mib": 49_140,
        "gpu_processes": [],
        "gpu_process_list_complete": True,
        "process_table": [],
        "process_table_complete": True,
        "storage_path": STORAGE_PATH,
        "filesystem_identity": FILESYSTEM_IDENTITY,
        "filesystem_available_bytes": 2 * 1024**4,
        "observed_at": at.isoformat(),
    } | overrides)


async def _repository(database_url: str) -> PostgresJobQueueRepository:
    repository = PostgresJobQueueRepository(
        database_url=database_url,
        expected_host_identity=HOST_IDENTITY,
        expected_gpu_uuid=GPU_UUID,
        expected_filesystem_identity=FILESYSTEM_IDENTITY,
        expected_storage_path=STORAGE_PATH,
    )
    await repository.ensure_schema()
    return repository


def test_direct_observer_persistence_rejects_bad_history_and_restarts_idle_window(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        start = datetime.now(UTC)
        repository = await _repository(task7_database_url)
        for offset in range(0, 31, 5):
            at = start + timedelta(seconds=offset)
            await repository.record_observation(_observation(at), received_at=at)
        state = await repository.get_observation_state("ubuntu")
        assert state["idle_observation_count"] == 7

        bad_observations = [
            _observation(start + timedelta(seconds=35), gpu_process_list_complete=False),
            _observation(start + timedelta(seconds=40), process_table_complete=False),
            _observation(start + timedelta(seconds=45), host_identity="machine-sha256:wrong"),
            _observation(start + timedelta(seconds=50), gpu_uuid="GPU-wrong"),
            _observation(start + timedelta(seconds=55), filesystem_identity="ext4:uuid=wrong"),
            _observation(start + timedelta(seconds=60), storage_path="/other/path"),
            _observation(start - timedelta(minutes=1)),
        ]
        for index, bad in enumerate(bad_observations, start=35):
            with pytest.raises(ValueError):
                await repository.record_observation(
                    bad,
                    received_at=start + timedelta(seconds=index),
                )
            state = await repository.get_observation_state("ubuntu")
            assert state["failure_code"] is not None
            assert state["idle_since"] is None
            assert state["idle_observation_count"] == 0

        await repository.close()
        # The 30-second series must not reappear after a producer/repository restart.
        repository = await _repository(task7_database_url)
        current = _observation(start + timedelta(seconds=75))
        reset = await repository.record_observation(
            current,
            received_at=start + timedelta(seconds=75),
        )
        assert reset["idle_observation_count"] == 1
        assert reset["idle_since"] == current.observed_at.isoformat()

        with pytest.raises(ObservationReplayError):
            await repository.record_observation(
                current,
                received_at=start + timedelta(seconds=80),
            )
        replay_reset = await repository.get_observation_state("ubuntu")
        assert replay_reset["failure_code"] is not None
        assert replay_reset["idle_since"] is None
        assert replay_reset["idle_observation_count"] == 0
        await repository.close()

    asyncio.run(exercise())
