"""Evaluation and ablations."""

from __future__ import annotations

from .goldset import (
    GoldQuestion,
    GoldSetBuilder,
    goldset_summary,
    load_goldset,
    save_goldset,
)
from .metrics import (
    RefusalMetrics,
    RetrievalMetrics,
    evaluate_refusal,
    evaluate_retrieval,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    set_recall_at_k,
)
from .runner import ABLATIONS, AblationArm, EvalResult, run_ablations, run_arm

__all__ = [
    "GoldQuestion",
    "GoldSetBuilder",
    "goldset_summary",
    "load_goldset",
    "save_goldset",
    "RetrievalMetrics",
    "RefusalMetrics",
    "evaluate_retrieval",
    "evaluate_refusal",
    "recall_at_k",
    "set_recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "percentile",
    "AblationArm",
    "ABLATIONS",
    "EvalResult",
    "run_arm",
    "run_ablations",
]
