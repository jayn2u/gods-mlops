from __future__ import annotations

import asyncio

from conftest import seed_training_ready_dataset
from gods_mlops.jobs.models import ExecutionProfile
from gods_mlops.jobs.queue import JobQueue, PostgresJobQueueRepository
from gods_mlops.jobs.sources import DatasetSourceRegistry

def test_submission_is_durable_deduplicated_and_explicit_rerun_gets_a_new_id(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        dataset_version = await seed_training_ready_dataset(task7_database_url)
        first_repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await first_repository.ensure_schema()
        sources = DatasetSourceRegistry(database_url=task7_database_url)
        queue = JobQueue(repository=first_repository, sources=sources)
        await queue.register_profile(
            ExecutionProfile(
                model_kind="detr",
                config_version="detr-config-v1",
                phase="training",
                memory_requirement_mib=8_192,
                artifact_reservation_bytes=64 * 1024**2,
                config={"input_size": 640, "micro_batch": 1},
                candidate=True,
            )
        )
        first = await queue.submit(dataset_version, "detr", "detr-config-v1")
        await queue.close()
        await first_repository.close()

        second_repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await second_repository.ensure_schema()
        restarted_queue = JobQueue(repository=second_repository, sources=sources)
        automatic_retry = await restarted_queue.submit(dataset_version, "detr", "detr-config-v1")
        rerun = await restarted_queue.submit(dataset_version, "detr", "detr-config-v1", rerun=True)
        assert automatic_retry == first
        assert rerun != first
        assert (await restarted_queue.get(first))["input_kind"] == "dataset_version"
        await restarted_queue.close()
        await second_repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_training_submission_accepts_training_ready_source_with_ineligible_evaluation(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        dataset_version = await seed_training_ready_dataset(task7_database_url)
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        sources = DatasetSourceRegistry(database_url=task7_database_url)
        source = await sources.get_training_source(dataset_version)
        assert source.training_ready is True
        assert source.evaluation_eligible is False
        queue = JobQueue(repository=repository, sources=sources)
        await queue.register_profile(
            ExecutionProfile(
                model_kind="detr",
                config_version="detr-config-v1",
                phase="training",
                memory_requirement_mib=8_192,
                artifact_reservation_bytes=64 * 1024**2,
                config={"input_size": 640, "micro_batch": 1},
                candidate=True,
            )
        )
        job_id = await queue.submit(dataset_version, "detr", "detr-config-v1")
        assert (await queue.get(job_id))["state"] == "queued"
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_probe_and_published_training_never_deduplicate_across_phases(
    task7_database_url: str,
) -> None:
    async def exercise() -> None:
        dataset_version = await seed_training_ready_dataset(task7_database_url)
        repository = PostgresJobQueueRepository(database_url=task7_database_url)
        await repository.ensure_schema()
        sources = DatasetSourceRegistry(database_url=task7_database_url)
        source = await sources.get_training_source(dataset_version)
        queue = JobQueue(repository=repository, sources=sources)
        await queue.register_profile(
            ExecutionProfile(
                model_kind="detr",
                config_version="same-config-v1",
                phase="training",
                memory_requirement_mib=8_192,
                artifact_reservation_bytes=32 * 1024**2,
                config={"input_size": 640},
                candidate=True,
            )
        )
        await queue.register_profile(
            ExecutionProfile(
                model_kind="detr",
                config_version="same-config-v1",
                phase="probe",
                memory_requirement_mib=8_192,
                artifact_reservation_bytes=32 * 1024**2,
                config={"input_size": 640},
                candidate=True,
            )
        )
        training_id = await queue.submit(dataset_version, "detr", "same-config-v1")
        probe_id = await queue.submit_probe(
            probe_input_id=dataset_version,
            input_sha256=source.manifest_sha256,
            model_kind="detr",
            config_version="same-config-v1",
        )
        assert training_id != probe_id
        assert (await queue.get(training_id))["phase"] == "training"
        assert (await queue.get(probe_id))["phase"] == "probe"
        await queue.close()
        await repository.close()
        await sources.close()

    asyncio.run(exercise())


def test_versioned_profile_keeps_swap_and_result_room_inside_artifact_reservation() -> None:
    profile = ExecutionProfile(
        model_kind="detr",
        config_version="bounded-artifacts-v1",
        phase="training",
        memory_requirement_mib=8_192,
        artifact_reservation_bytes=3_000,
        config={"input_size": 640},
        candidate=True,
    )
    assert profile.checkpoint_reservation_bytes > 0
    assert profile.result_reservation_bytes > 0
    assert 2 * profile.checkpoint_reservation_bytes + profile.result_reservation_bytes <= profile.artifact_reservation_bytes


def test_fixed_qwen_caption_profile_is_inference_only() -> None:
    import pytest

    with pytest.raises(ValueError, match="Qwen.*inference"):
        ExecutionProfile(
            model_kind="qwen",
            config_version="qwen-training-forbidden-v1",
            phase="training",
            memory_requirement_mib=16_384,
            artifact_reservation_bytes=32 * 1024**2,
            config={"purpose": "caption"},
            candidate=True,
        )
