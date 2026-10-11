"""Human-reviewed CLIP cosine retrieval metrics for a frozen gallery."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_JUDGMENTS = {"relevant", "not_relevant", "uncertain"}


def evaluate_retrieval(
    *,
    query_embeddings: Mapping[str, Sequence[float]],
    crop_embeddings: Mapping[str, Sequence[float]],
    gallery_crop_ids: Sequence[str],
    relevance_matrices: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute hit@1/5/10 and mAP only from complete frozen human matrices."""
    if not isinstance(query_embeddings, Mapping) or not isinstance(crop_embeddings, Mapping):
        raise ValueError("CLIP query and crop embeddings must be ID-keyed mappings")
    if not isinstance(gallery_crop_ids, Sequence) or isinstance(gallery_crop_ids, (str, bytes)):
        raise ValueError("the frozen gallery must be a sequence of crop IDs")
    if not isinstance(relevance_matrices, Sequence) or isinstance(relevance_matrices, (str, bytes)):
        raise ValueError("frozen relevance matrices must be a sequence")

    gallery_ids = [_identifier(item, "gallery crop ID") for item in gallery_crop_ids]
    if len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("the frozen CLIP gallery must not contain duplicate crop IDs")
    if not gallery_ids:
        return _insufficient("evaluation_gallery_missing", queries=0, gallery=0)
    missing_gallery_embeddings = sorted(set(gallery_ids) - set(crop_embeddings))
    if missing_gallery_embeddings:
        return _insufficient("gallery_embedding_missing", queries=len(query_embeddings), gallery=len(gallery_ids))

    normalized_queries, query_error = _normalize_embeddings(query_embeddings)
    if query_error:
        return _insufficient(query_error, queries=len(query_embeddings), gallery=len(gallery_ids))
    normalized_crops, crop_error = _normalize_embeddings(
        {crop_id: crop_embeddings[crop_id] for crop_id in gallery_ids}
    )
    if crop_error:
        return _insufficient(crop_error, queries=len(query_embeddings), gallery=len(gallery_ids))

    matrices_by_query: dict[str, Mapping[str, Any]] = {}
    for matrix in relevance_matrices:
        if not isinstance(matrix, Mapping):
            return _insufficient("human_relevance_truth_invalid", queries=len(query_embeddings), gallery=len(gallery_ids))
        query_id = _identifier(matrix.get("query_id"), "relevance query ID")
        if query_id in matrices_by_query:
            return _insufficient("human_relevance_truth_invalid", queries=len(query_embeddings), gallery=len(gallery_ids))
        matrices_by_query[query_id] = matrix

    reasons: set[str] = set()
    all_query_ids = set(normalized_queries)
    for query_id in all_query_ids:
        if query_id not in matrices_by_query:
            reasons.add("human_relevance_truth_missing")
    for query_id in set(matrices_by_query) - all_query_ids:
        reasons.add("query_embedding_missing")
    if not all_query_ids:
        reasons.add("evaluation_queries_missing")
    if reasons:
        return _insufficient(sorted(reasons)[0], queries=len(all_query_ids), gallery=len(gallery_ids))

    relevance: dict[str, set[str]] = {}
    for query_id, matrix in matrices_by_query.items():
        if matrix.get("status") != "complete" or matrix.get("evaluation_eligible") is not True:
            reasons.add("human_relevance_truth_unresolved")
            continue
        if matrix.get("gallery_matches_selection") is False:
            reasons.add("human_relevance_gallery_mismatch")
            continue
        judgments = matrix.get("judgments")
        if not isinstance(judgments, Sequence) or isinstance(judgments, (str, bytes)):
            reasons.add("human_relevance_truth_missing_or_incomplete")
            continue
        judgments_by_crop: dict[str, str] = {}
        for judgment in judgments:
            if not isinstance(judgment, Mapping):
                reasons.add("human_relevance_truth_unresolved")
                continue
            crop_id = judgment.get("crop_id")
            value = judgment.get("judgment")
            if not isinstance(crop_id, str) or value not in _JUDGMENTS or crop_id in judgments_by_crop:
                reasons.add("human_relevance_truth_unresolved")
                continue
            judgments_by_crop[crop_id] = str(value)
        if set(judgments_by_crop) != set(gallery_ids):
            reasons.add("human_relevance_truth_missing_or_incomplete")
            continue
        values = set(judgments_by_crop.values())
        if "uncertain" in values:
            reasons.add("human_relevance_truth_unresolved")
            continue
        if "relevant" not in values:
            reasons.add("human_relevance_positive_missing")
            continue
        if "not_relevant" not in values:
            reasons.add("human_relevance_negative_missing")
            continue
        relevance[query_id] = {
            crop_id for crop_id, value in judgments_by_crop.items() if value == "relevant"
        }
    if reasons:
        return _insufficient(sorted(reasons)[0], queries=len(all_query_ids), gallery=len(gallery_ids))

    hits = {1: [], 5: [], 10: []}
    average_precisions = []
    relevant_judgments = 0
    for query_id in sorted(all_query_ids):
        query = normalized_queries[query_id]
        ranked = sorted(
            gallery_ids,
            key=lambda crop_id: (-_cosine(query, normalized_crops[crop_id]), crop_id),
        )
        relevant = relevance[query_id]
        relevant_judgments += len(relevant)
        for cutoff in hits:
            hits[cutoff].append(float(any(crop_id in relevant for crop_id in ranked[:cutoff])))
        found = 0
        precision_at_relevant_ranks = []
        for rank, crop_id in enumerate(ranked, start=1):
            if crop_id in relevant:
                found += 1
                precision_at_relevant_ranks.append(found / rank)
        average_precisions.append(sum(precision_at_relevant_ranks) / len(relevant))

    return {
        "status": "complete",
        "metrics": {
            "hit_rate@1": sum(hits[1]) / len(hits[1]),
            "hit_rate@5": sum(hits[5]) / len(hits[5]),
            "hit_rate@10": sum(hits[10]) / len(hits[10]),
            "map": sum(average_precisions) / len(average_precisions),
        },
        "counts": {
            "queries": len(all_query_ids),
            "gallery_crops": len(gallery_ids),
            "relevant_judgments": relevant_judgments,
        },
        "settings": {
            "similarity": "cosine",
            "tie_break": "ascending_crop_id",
            "relevance_source": "complete_frozen_human_matrix",
        },
        "reasons": [],
    }


def _normalize_embeddings(
    embeddings: Mapping[str, Sequence[float]],
) -> tuple[dict[str, list[float]], str | None]:
    normalized: dict[str, list[float]] = {}
    dimension: int | None = None
    for identifier, vector in embeddings.items():
        if not isinstance(identifier, str) or not identifier.strip():
            return {}, "embedding_identity_invalid"
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)) or not vector:
            return {}, "embedding_invalid"
        try:
            values = [float(item) for item in vector]
        except (TypeError, ValueError):
            return {}, "embedding_invalid"
        if any(not math.isfinite(item) for item in values):
            return {}, "embedding_invalid"
        if dimension is None:
            dimension = len(values)
        elif len(values) != dimension:
            return {}, "embedding_dimension_mismatch"
        norm = math.sqrt(math.fsum(item * item for item in values))
        if norm == 0 or not math.isfinite(norm):
            return {}, "embedding_zero_or_nonfinite_norm"
        normalized[identifier] = [item / norm for item in values]
    return normalized, None


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _insufficient(reason: str, *, queries: int, gallery: int) -> dict[str, Any]:
    return {
        "status": "insufficient",
        "metrics": {
            "hit_rate@1": None,
            "hit_rate@5": None,
            "hit_rate@10": None,
            "map": None,
        },
        "counts": {"queries": queries, "gallery_crops": gallery, "relevant_judgments": 0},
        "settings": {
            "similarity": "cosine",
            "tie_break": "ascending_crop_id",
            "relevance_source": "complete_frozen_human_matrix",
        },
        "reasons": [reason],
    }
