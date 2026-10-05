from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest


def _plan_splits(*args, **kwargs):
    try:
        from gods_mlops.datasets.split import plan_splits
    except ModuleNotFoundError as error:
        pytest.fail(f"dataset split implementation is missing: {error}")
    return plan_splits(*args, **kwargs)


def _sample(
    sample_id: str,
    day: int,
    *,
    hour: int = 12,
    minute: int = 0,
    camera_id: str = "camera-a",
) -> dict:
    return {
        "sample_id": sample_id,
        "camera_id": camera_id,
        "capture_day": date(2026, 10, day),
        # Seoul is UTC+9. The local time is used for the midnight exclusion window.
        "captured_at_utc": datetime(
            2026, 10, day, hour, minute, tzinfo=ZoneInfo("Asia/Seoul")
        ).astimezone(timezone.utc),
    }


def test_midnight_window_is_excluded_only_at_a_different_split_boundary() -> None:
    samples = [
        _sample("s1", 1),
        _sample("s2", 2, hour=0, minute=4),
        _sample("s3", 3),
        _sample("s4-midnight", 4, hour=0, minute=4),
        _sample("s4-noon", 4),
        _sample("s5", 5),
    ]

    plan = _plan_splits(samples, event_links=[], clip_links=[])

    assert [item["sample_id"] for item in plan["excluded"]] == ["s4-midnight"]
    assert set(plan["assignments"]) == {"s1", "s2", "s3", "s4-noon", "s5"}
    assert plan["assignments"]["s2"]["split"] == "train"
    assert plan["assignments"]["s4-noon"]["split"] == "validation"
    # Five independent camera/date groups split chronologically as 3:1:1.
    assert plan["group_counts"] == {"train": 3, "validation": 1, "test": 1}


def test_event_and_clip_links_form_indivisible_components() -> None:
    samples = [_sample(f"s{day}", day) for day in range(1, 7)]

    plan = _plan_splits(
        samples,
        event_links=[{"link_id": "overnight-event", "sample_ids": ["s1", "s6"]}],
        clip_links=[{"link_id": "camera-clip", "sample_ids": ["s2", "s3"]}],
    )

    assignments = plan["assignments"]
    assert assignments["s1"]["group_id"] == assignments["s6"]["group_id"]
    assert assignments["s1"]["split"] == assignments["s6"]["split"]
    assert assignments["s2"]["group_id"] == assignments["s3"]["group_id"]
    assert assignments["s2"]["split"] == assignments["s3"]["split"]
    # The two linked pairs merge six dates into four independent components.
    assert plan["group_counts"] == {"train": 2, "validation": 1, "test": 1}


def test_existing_test_assignments_stay_fixed_and_new_groups_avoid_test() -> None:
    samples = [_sample(f"s{day}", day) for day in range(1, 6)]
    existing = {
        "s1": {"split": "test", "group_id": "old-test"},
        "s2": {"split": "train", "group_id": "old-train"},
    }

    plan = _plan_splits(samples, event_links=[], clip_links=[], prior_assignments=existing)

    assert plan["assignments"]["s1"] == existing["s1"]
    assert plan["assignments"]["s2"] == existing["s2"]
    assert {plan["assignments"][sample_id]["split"] for sample_id in ("s3", "s4", "s5")} <= {
        "train",
        "validation",
    }


def test_late_link_across_fixed_splits_blocks_and_reports_leakage() -> None:
    samples = [_sample("train-sample", 1), _sample("test-sample", 2)]
    existing = {
        "train-sample": {"split": "train", "group_id": "train-group"},
        "test-sample": {"split": "test", "group_id": "test-group"},
    }

    plan = _plan_splits(
        samples,
        event_links=[{"link_id": "late-event-link", "sample_ids": ["train-sample", "test-sample"]}],
        clip_links=[],
        prior_assignments=existing,
    )

    assert plan["blocked"] is True
    assert plan["leakage_impacts"] == [
        {
            "link_ids": ["event:late-event-link"],
            "sample_ids": ["test-sample", "train-sample"],
            "splits": ["test", "train"],
        }
    ]
    assert plan["assignments"]["test-sample"] == existing["test-sample"]


def test_capture_day_must_match_declared_seoul_time_semantics() -> None:
    sample = _sample("wrong-day", 1)
    sample["captured_at_utc"] = datetime(2026, 10, 2, 3, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="capture_day does not match Asia/Seoul"):
        _plan_splits([sample], event_links=[], clip_links=[])


def test_initial_split_blocks_before_three_independent_day_groups() -> None:
    samples = [_sample("s1", 1), _sample("s2", 2)]

    with pytest.raises(ValueError, match="at least three independent camera/date groups"):
        _plan_splits(samples, event_links=[], clip_links=[])


def test_initial_partitions_are_allocated_per_camera() -> None:
    samples = [
        _sample(f"{camera}-d{day}", day, camera_id=camera)
        for camera in ("camera-a", "camera-b")
        for day in (1, 2, 3)
    ]

    plan = _plan_splits(samples, event_links=[], clip_links=[])

    for camera in ("camera-a", "camera-b"):
        camera_splits = {
            plan["assignments"][sample["sample_id"]]["split"]
            for sample in samples
            if sample["camera_id"] == camera
        }
        assert camera_splits == {"train", "validation", "test"}


def test_cross_camera_component_that_breaks_a_camera_partition_is_blocked() -> None:
    samples = [
        _sample(f"{camera}-d{day}", day, camera_id=camera)
        for camera in ("camera-a", "camera-b")
        for day in (1, 2, 3)
    ]

    plan = _plan_splits(
        samples,
        event_links=[{"link_id": "cross-camera-incident", "sample_ids": ["camera-a-d3", "camera-b-d1"]}],
        clip_links=[],
    )

    assert plan["blocked"] is True
    assert plan["block_reasons"] == ["cross_camera_component_prevents_independent_partitions"]


def test_late_cross_split_component_with_new_member_keeps_fixed_rows_without_stopiteration() -> None:
    samples = [
        _sample("train-old", 1),
        _sample("test-old", 2),
        _sample("new-member", 3),
    ]
    existing = {
        "train-old": {"split": "train", "group_id": "train-group"},
        "test-old": {"split": "test", "group_id": "test-group"},
    }

    plan = _plan_splits(
        samples,
        event_links=[{"link_id": "late-link-with-new-member", "sample_ids": ["train-old", "test-old", "new-member"]}],
        clip_links=[],
        prior_assignments=existing,
    )

    assert plan["blocked"] is True
    assert plan["assignments"]["train-old"] == existing["train-old"]
    assert plan["assignments"]["test-old"] == existing["test-old"]
    assert "new-member" not in plan["assignments"]
