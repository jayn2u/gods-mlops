from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from gods_mlops.ingestion.collection_gate import CollectionStorageGate
from gods_mlops.jobs import observer as observer_module
from gods_mlops.jobs.models import ResourceObservation
from gods_mlops.jobs.queue import PostgresJobQueueRepository

NODE_ID = "ubuntu"
HOST_IDENTITY = "machine-sha256:task7-test-ubuntu"
GPU_UUID = "GPU-e5fd41ed-1688-8aca-3cd4-7904d53d764e"
FILESYSTEM_IDENTITY = "ext4:uuid=task7-test-data"
STORAGE_PATH = "/data"


def _observation() -> ResourceObservation:
    return ResourceObservation(
        observation_id=str(uuid4()),
        node_id=NODE_ID,
        hostname="ubuntu",
        host_identity=HOST_IDENTITY,
        gpu_name="NVIDIA RTX A6000",
        gpu_uuid=GPU_UUID,
        free_mib=48_000,
        total_mib=49_140,
        gpu_processes=(),
        gpu_process_list_complete=True,
        process_table=(),
        process_table_complete=True,
        storage_path=STORAGE_PATH,
        filesystem_identity=FILESYSTEM_IDENTITY,
        filesystem_available_bytes=2 * 1024**4,
        observed_at=datetime.now(UTC),
    )


def _set_observer_environment(monkeypatch, database_url: str) -> None:
    values = {
        "GODS_MLOPS_DATABASE_URL": database_url,
        "GODS_MLOPS_UBUNTU_NODE_ID": NODE_ID,
        "GODS_MLOPS_UBUNTU_HOST_IDENTITY": HOST_IDENTITY,
        "GODS_MLOPS_UBUNTU_GPU_UUID": GPU_UUID,
        "GODS_MLOPS_UBUNTU_FILESYSTEM_IDENTITY": FILESYSTEM_IDENTITY,
        "GODS_MLOPS_UBUNTU_STORAGE_PATH": STORAGE_PATH,
        "GODS_MLOPS_UBUNTU_SSH_TARGET": "jayn2u@203.0.113.179",
        "GODS_MLOPS_UBUNTU_SSH_PORT": "2222",
        "GODS_MLOPS_UBUNTU_SSH_IDENTITY_FILE": "/run/ubuntu-ssh/id_ed25519",
        "GODS_MLOPS_UBUNTU_SSH_KNOWN_HOSTS": "/run/ubuntu-ssh/known_hosts",
        "GODS_MLOPS_UBUNTU_SSH_TIMEOUT_SECONDS": "10",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_observer_main_persists_its_pinned_sample_for_the_receiver_gate(
    task7_database_url: str,
    monkeypatch,
) -> None:
    _set_observer_environment(monkeypatch, task7_database_url)

    async def initialize_owned_schema() -> None:
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        try:
            await repository.ensure_schema()
        finally:
            await repository.close()

    asyncio.run(initialize_owned_schema())
    sample = _observation()
    construction: list[dict[str, object]] = []
    consumed: list[ResourceObservation] = []

    class StubObserver:
        def __init__(self, **settings) -> None:
            construction.append(settings)

        def observe(self) -> ResourceObservation:
            return sample

    monkeypatch.setattr(observer_module, "UbuntuResourceObserver", StubObserver)

    async def persist_once(repository, observer) -> None:
        try:
            produced = await observer_module.observe_once_and_persist(
                repository=repository,
                observer=observer,
            )
            gate = CollectionStorageGate(
                repository=repository,
                expected_node_id=NODE_ID,
                expected_host_identity=HOST_IDENTITY,
                expected_gpu_uuid=GPU_UUID,
                expected_filesystem_identity=FILESYSTEM_IDENTITY,
                expected_storage_path=STORAGE_PATH,
                min_free_bytes=1024**4,
            )
            consumed.append(await gate.ensure_available())
            assert produced.observation_id == sample.observation_id
        finally:
            await repository.close()

    monkeypatch.setattr(observer_module, "_run_and_close", persist_once)
    failure: Exception | None = None
    try:
        observer_module.main()
    except Exception as error:  # noqa: BLE001 - retain the startup failure as a real assertion
        failure = error

    assert failure is None, f"configured observer startup could not persist for M: {failure}"
    assert len(consumed) == 1
    assert consumed[0].observation_id == sample.observation_id
    assert construction[0]["node_id"] == NODE_ID
    assert construction[0]["host_identity"] == HOST_IDENTITY
    assert construction[0]["gpu_uuid"] == GPU_UUID
    assert construction[0]["filesystem_identity"] == FILESYSTEM_IDENTITY
    assert construction[0]["storage_path"] == STORAGE_PATH


def test_observer_main_fails_closed_when_any_trust_pin_is_missing(
    monkeypatch,
) -> None:
    _set_observer_environment(monkeypatch, "postgresql://gods_task7:unused@127.0.0.1:15438/postgres")
    monkeypatch.delenv("GODS_MLOPS_UBUNTU_FILESYSTEM_IDENTITY")

    async def no_op(repository, _observer) -> None:
        await repository.close()

    monkeypatch.setattr(observer_module, "_run_and_close", no_op)
    failure: Exception | None = None
    try:
        observer_module.main()
    except Exception as error:  # noqa: BLE001 - startup must name the missing pin
        failure = error

    assert isinstance(failure, ValueError)
    assert "GODS_MLOPS_UBUNTU_FILESYSTEM_IDENTITY" in str(failure)
