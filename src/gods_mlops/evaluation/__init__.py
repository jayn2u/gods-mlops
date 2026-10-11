"""Immutable, provenance-aware model evaluation helpers."""

from .detection import evaluate_detections
from .retrieval import evaluate_retrieval

__all__ = ["evaluate_detections", "evaluate_retrieval"]
