"""Deterministic camera/date component splitting for immutable datasets."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from hashlib import sha256
from typing import Any, Iterable
from zoneinfo import ZoneInfo

_CAPTURE_TIMEZONE = ZoneInfo("Asia/Seoul")
_SPLITS = ("train", "validation", "test")
_MIDNIGHT_WINDOW = timedelta(minutes=5)


def plan_splits(
    samples: list[dict[str, Any]],
    *,
    event_links: list[dict[str, Any]],
    clip_links: list[dict[str, Any]],
    prior_assignments: dict[str, dict[str, str]] | None = None,
    has_authority: bool | None = None,
) -> dict[str, Any]:
    """Assign camera/date connected components while preserving historical splits.

    ``captured_at_utc`` is interpreted in UTC and ``capture_day`` must already be
    the corresponding Asia/Seoul calendar day. Samples within five minutes of a
    local midnight are excluded only when adjacent camera/date groups land in
    different splits.
    """
    prior = prior_assignments or {}
    normalized = _normalize_samples(samples)
    selected_ids = set(normalized)
    parent: dict[str, str] = {sample_id: sample_id for sample_id in selected_ids}
    all_links = _normalize_links(event_links, "event") + _normalize_links(clip_links, "clip")
    for assignment_id, assignment in prior.items():
        if not isinstance(assignment, dict) or assignment.get("split") not in _SPLITS:
            raise ValueError(f"invalid prior split assignment for {assignment_id}")
        parent.setdefault(str(assignment_id), str(assignment_id))
    for link in all_links:
        for sample_id in link["sample_ids"]:
            parent.setdefault(sample_id, sample_id)
        for sample_id in link["sample_ids"][1:]:
            _union(parent, link["sample_ids"][0], sample_id)

    camera_dates: dict[tuple[str, date], list[str]] = defaultdict(list)
    for sample_id, sample in normalized.items():
        camera_dates[(sample["camera_id"], sample["capture_day"])].append(sample_id)
    for sample_ids in camera_dates.values():
        for sample_id in sample_ids[1:]:
            _union(parent, sample_ids[0], sample_id)

    components: dict[str, list[str]] = defaultdict(list)
    for sample_id in parent:
        components[_find(parent, sample_id)].append(sample_id)
    selected_components = [
        sorted(component)
        for component in components.values()
        if selected_ids.intersection(component)
    ]
    selected_components.sort(key=lambda group: _component_order(group, normalized))

    has_history = bool(prior) if has_authority is None else (has_authority or bool(prior))
    fixed_by_root: dict[str, dict[str, str]] = {}
    component_split: dict[str, str] = {}
    component_group_id: dict[str, str] = {}
    leakage_impacts: list[dict[str, Any]] = []

    for component in selected_components:
        root = _find(parent, component[0])
        old = [(sample_id, prior[sample_id]) for sample_id in component if sample_id in prior]
        old_splits = {assignment["split"] for _, assignment in old}
        if len(old_splits) > 1:
            crossing_links = sorted(
                {
                    link["link_id"]
                    for link in all_links
                    if len(
                        {
                            prior[sample_id]["split"]
                            for sample_id in link["sample_ids"]
                            if sample_id in prior
                        }
                    )
                    > 1
                    and set(link["sample_ids"]).intersection(component)
                }
            )
            if not crossing_links:
                crossing_links = [
                    f"camera-date:{normalized[sample_id]['camera_id']}:{normalized[sample_id]['capture_day'].isoformat()}"
                    for sample_id in component
                    if sample_id in normalized
                ][:1]
            leakage_impacts.append(
                {
                    "link_ids": crossing_links,
                    "sample_ids": sorted(component),
                    "splits": sorted(old_splits),
                }
            )
            fixed_by_root[root] = {sample_id: prior[sample_id]["split"] for sample_id, _ in old}
            component_group_id[root] = _group_id(component)
        elif old_splits:
            split_name = next(iter(old_splits))
            group_ids = sorted({assignment["group_id"] for _, assignment in old if assignment.get("group_id")})
            component_split[root] = split_name
            component_group_id[root] = group_ids[0] if len(group_ids) == 1 else _group_id(component)
        else:
            component_group_id[root] = _group_id(component)

    new_roots = [
        _find(parent, component[0])
        for component in selected_components
        if not any(sample_id in prior for sample_id in component)
    ]
    if not has_history:
        camera_roots: dict[str, list[str]] = defaultdict(list)
        for (camera_id, _capture_day), sample_ids in camera_dates.items():
            roots = {_find(parent, sample_id) for sample_id in sample_ids}
            camera_roots[camera_id].extend(roots)
        for camera_id, roots in camera_roots.items():
            camera_roots[camera_id] = sorted(
                set(roots),
                key=lambda root: min(
                    (normalized[sample_id]["capture_day"], _group_id(components[root]))
                    for sample_id in components[root]
                    if sample_id in normalized and normalized[sample_id]["camera_id"] == camera_id
                ),
            )
            if len(camera_roots[camera_id]) < 3:
                raise ValueError("initial split requires at least three independent camera/date groups")
        initial_assignments = _initial_per_camera_assignments(camera_roots)
        if initial_assignments is None:
            return {
                "assignments": {},
                "component_by_sample": {},
                "excluded": [],
                "group_counts": {split_name: 0 for split_name in _SPLITS},
                "blocked": True,
                "block_reasons": ["cross_camera_component_prevents_independent_partitions"],
                "leakage_impacts": [],
                "timezone": "Asia/Seoul",
                "target_ratio": {"train": 0.6, "validation": 0.2, "test": 0.2},
            }
        component_split.update(initial_assignments)
    else:
        train_count = (3 * len(new_roots) + 3) // 4
        new_splits = ["train"] * train_count + ["validation"] * (len(new_roots) - train_count)
        component_split.update(zip(new_roots, new_splits, strict=True))

    sample_component: dict[str, str] = {}
    for component in selected_components:
        root = _find(parent, component[0])
        for sample_id in component:
            sample_component[sample_id] = root

    exclusions: list[dict[str, str]] = []
    assignments: dict[str, dict[str, str]] = {}
    blocked_roots = {
        root
        for root, fixed_assignments in fixed_by_root.items()
        if fixed_assignments
    }

    for sample_id, sample in normalized.items():
        root = sample_component[sample_id]
        if root in blocked_roots:
            if sample_id in prior:
                assignments[sample_id] = dict(prior[sample_id])
            continue
        split_name = prior.get(sample_id, {}).get("split", component_split[root])
        if _crosses_split_midnight(sample, split_name, camera_dates, sample_component, component_split, prior):
            exclusions.append(
                {
                    "sample_id": sample_id,
                    "reason": "midnight_split_boundary",
                    "capture_day": sample["capture_day"].isoformat(),
                    "timezone": "Asia/Seoul",
                }
            )
            continue
        assignments[sample_id] = {
            "split": split_name,
            "group_id": prior.get(sample_id, {}).get("group_id", component_group_id[root]),
        }

    group_counts = {split_name: 0 for split_name in _SPLITS}
    groups_by_split: dict[str, set[str]] = {split_name: set() for split_name in _SPLITS}
    for sample_id, assignment in assignments.items():
        groups_by_split[assignment["split"]].add(component_group_id[sample_component[sample_id]])
    for split_name in _SPLITS:
        group_counts[split_name] = len(groups_by_split[split_name])

    return {
        "assignments": assignments,
        "component_by_sample": {
            sample_id: component_group_id[sample_component[sample_id]]
            for sample_id in assignments
        },
        "excluded": sorted(exclusions, key=lambda item: item["sample_id"]),
        "group_counts": group_counts,
        "blocked": bool(leakage_impacts),
        "block_reasons": ["late_cross_boundary_link"] if leakage_impacts else [],
        "has_authority": has_history,
        "leakage_impacts": leakage_impacts,
        "timezone": "Asia/Seoul",
        "target_ratio": {"train": 0.6, "validation": 0.2, "test": 0.2},
    }


def _normalize_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for raw in samples:
        if not isinstance(raw, dict):
            raise ValueError("each sample must be an object")
        sample_id = str(raw.get("sample_id", ""))
        camera_id = str(raw.get("camera_id", ""))
        captured = raw.get("captured_at_utc")
        capture_day = raw.get("capture_day")
        if not sample_id or not camera_id or not isinstance(captured, datetime) or not isinstance(capture_day, date):
            raise ValueError("sample requires an ID, camera, capture_day, and captured_at_utc")
        if captured.tzinfo is None or captured.utcoffset() is None:
            raise ValueError("captured_at_utc must be timezone-aware")
        local_time = captured.astimezone(_CAPTURE_TIMEZONE)
        if local_time.date() != capture_day:
            raise ValueError("capture_day does not match Asia/Seoul captured_at_utc")
        if sample_id in normalized:
            raise ValueError(f"duplicate sample ID {sample_id}")
        normalized[sample_id] = {
            "sample_id": sample_id,
            "camera_id": camera_id,
            "capture_day": capture_day,
            "captured_at_utc": captured,
            "local_time": local_time.timetz().replace(tzinfo=None),
        }
    return normalized


def _normalize_links(links: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for link in links:
        if not isinstance(link, dict):
            raise ValueError(f"{kind} link must be an object")
        link_id = str(link.get("link_id", ""))
        sample_ids = sorted({str(sample_id) for sample_id in link.get("sample_ids", [])})
        if not link_id or not sample_ids:
            raise ValueError(f"{kind} link requires an ID and sample IDs")
        identity = f"{kind}:{link_id}"
        if identity in identifiers:
            raise ValueError(f"duplicate {kind} link ID {link_id}")
        identifiers.add(identity)
        normalized.append({"link_id": identity, "sample_ids": sample_ids})
    return normalized


def _initial_per_camera_assignments(camera_roots: dict[str, list[str]]) -> dict[str, str] | None:
    """Find chronological, non-empty per-camera partitions consistent across links."""
    cameras_by_root: dict[str, set[str]] = defaultdict(set)
    for camera_id, roots in camera_roots.items():
        for root in roots:
            cameras_by_root[root].add(camera_id)

    shared_roots = {
        camera_id: sorted(root for root in roots if len(cameras_by_root[root]) > 1)
        for camera_id, roots in camera_roots.items()
    }
    options: dict[str, list[tuple[float, dict[str, str]]]] = {}
    best_counts: dict[str, dict[tuple[str, ...], tuple[int, int]]] = {}
    for camera_id, roots in camera_roots.items():
        camera_options: dict[tuple[str, ...], tuple[float, int, int]] = {}
        group_count = len(roots)
        root_positions = {root: index for index, root in enumerate(roots)}
        for train_count in range(1, group_count - 1):
            for validation_count in range(1, group_count - train_count):
                test_count = group_count - train_count - validation_count
                if test_count < 1:
                    continue
                labels = (
                    ["train"] * train_count
                    + ["validation"] * validation_count
                    + ["test"] * test_count
                )
                score = sum(
                    (observed - target) ** 2
                    for observed, target in zip(
                        (train_count / group_count, validation_count / group_count, test_count / group_count),
                        (0.6, 0.2, 0.2),
                        strict=True,
                    )
                )
                signature = tuple(labels[root_positions[root]] for root in shared_roots[camera_id])
                previous = camera_options.get(signature)
                if previous is None or score < previous[0]:
                    camera_options[signature] = (score, train_count, validation_count)
        best_counts[camera_id] = {
            signature: (train_count, validation_count)
            for signature, (_score, train_count, validation_count) in camera_options.items()
        }
        options[camera_id] = sorted(
            (
                (score, dict(zip(shared_roots[camera_id], signature, strict=True)))
                for signature, (score, _train_count, _validation_count) in camera_options.items()
            ),
            key=lambda item: item[0],
        )

    camera_order = sorted(
        camera_roots,
        key=lambda camera_id: sum(
            len(option) for _, option in options[camera_id]
        ),
    )

    def choose(index: int, chosen: dict[str, str]) -> dict[str, str] | None:
        if index == len(camera_order):
            return chosen
        camera_id = camera_order[index]
        for _, assignment in options[camera_id]:
            if any(root in chosen and chosen[root] != split for root, split in assignment.items()):
                continue
            result = choose(index + 1, {**chosen, **assignment})
            if result is not None:
                return result
        return None

    chosen = choose(0, {})
    if chosen is None:
        return None
    assignments: dict[str, str] = {}
    for camera_id, roots in camera_roots.items():
        signature = tuple(chosen[root] for root in shared_roots[camera_id])
        train_count, validation_count = best_counts[camera_id][signature]
        labels = (
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * (len(roots) - train_count - validation_count)
        )
        assignments.update(dict(zip(roots, labels, strict=True)))
    return assignments


def _component_order(group: list[str], samples: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    selected = [samples[sample_id] for sample_id in group if sample_id in samples]
    return (
        min(item["camera_id"] for item in selected),
        min(item["capture_day"] for item in selected),
        _group_id(group),
    )


def _crosses_split_midnight(
    sample: dict[str, Any],
    split_name: str,
    camera_dates: dict[tuple[str, date], list[str]],
    sample_component: dict[str, str],
    component_split: dict[str, str],
    prior: dict[str, dict[str, str]],
) -> bool:
    local_time: time = sample["local_time"]
    day_start = local_time <= time(0, 5)
    day_end = local_time >= time(23, 55)
    if not day_start and not day_end:
        return False
    adjacent_days = []
    if day_start:
        adjacent_days.append(sample["capture_day"] - timedelta(days=1))
    if day_end:
        adjacent_days.append(sample["capture_day"] + timedelta(days=1))
    for adjacent_day in adjacent_days:
        neighbor_ids = camera_dates.get((sample["camera_id"], adjacent_day), [])
        for neighbor_id in neighbor_ids:
            neighbor_split = prior.get(neighbor_id, {}).get("split")
            if neighbor_split is None:
                root = sample_component.get(neighbor_id)
                neighbor_split = component_split.get(root or "")
            if neighbor_split is not None and neighbor_split != split_name:
                return True
    return False


def _group_id(sample_ids: Iterable[str]) -> str:
    digest = sha256("\n".join(sorted(sample_ids)).encode("utf-8")).hexdigest()[:20]
    return f"group-{digest}"


def _find(parent: dict[str, str], sample_id: str) -> str:
    root = sample_id
    while parent[root] != root:
        root = parent[root]
    while parent[sample_id] != sample_id:
        next_id = parent[sample_id]
        parent[sample_id] = root
        sample_id = next_id
    return root


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root, right_root = _find(parent, left), _find(parent, right)
    if left_root == right_root:
        return
    first, second = sorted((left_root, right_root))
    parent[second] = first
