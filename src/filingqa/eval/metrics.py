"""Retrieval and answer metrics.

Definitions matter more than the code here, because "recall@10" means at least
three different things in the RAG literature and the difference is worth
several points.

**recall@k** - fraction of questions for which *at least one* passage graded
relevance >= 2 appears in the top k. This is the hit-rate definition, and it is
the right one for this product: the answer needs one good source, not all of
them. The stricter set-recall definition (fraction of *all* gold passages
retrieved) is reported separately as `set_recall@k` because it is the number
that degrades when chunking is bad, and hiding it would flatter the chunker.

**MRR** - mean reciprocal rank of the first relevant passage. Sensitive to
whether the best passage is at rank 1 or rank 4, which matters when a
generator only reads the top few.

**nDCG@k** - uses the graded labels rather than collapsing them to binary, so
retrieving the passage that directly answers (grade 3) scores above one that
merely supports (grade 2).

For refusal, precision and recall are reported from the perspective of
*refusing*: recall is the share of unanswerable questions correctly refused,
precision the share of refusals that were correct. Both are needed. A system
that refuses everything scores perfect refusal recall and is useless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..models import Answer, Scored
from .goldset import GoldQuestion

MIN_RELEVANT_GRADE = 2


def recall_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    gold = {c for c, g in relevance.items() if g >= MIN_RELEVANT_GRADE}
    if not gold:
        return 0.0
    return 1.0 if gold & set(retrieved[:k]) else 0.0


def set_recall_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    gold = {c for c, g in relevance.items() if g >= MIN_RELEVANT_GRADE}
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def precision_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    gold = {c for c, g in relevance.items() if g >= MIN_RELEVANT_GRADE}
    return len(gold & set(retrieved[:k])) / k


def reciprocal_rank(retrieved: list[str], relevance: dict[str, int]) -> float:
    gold = {c for c, g in relevance.items() if g >= MIN_RELEVANT_GRADE}
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    if not relevance:
        return 0.0
    gains = [relevance.get(cid, 0) for cid in retrieved[:k]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalMetrics:
    n: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    set_recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    ndcg: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "questions": self.n,
            "recall_at_k": {str(k): round(v, 4) for k, v in sorted(self.recall.items())},
            "set_recall_at_k": {str(k): round(v, 4) for k, v in sorted(self.set_recall.items())},
            "precision_at_k": {str(k): round(v, 4) for k, v in sorted(self.precision.items())},
            "ndcg_at_k": {str(k): round(v, 4) for k, v in sorted(self.ndcg.items())},
            "mrr": round(self.mrr, 4),
        }


def evaluate_retrieval(
    results: dict[str, list[Scored]],
    questions: list[GoldQuestion],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> RetrievalMetrics:
    answerable = [q for q in questions if q.answerable and q.gold_chunks]
    metrics = RetrievalMetrics(n=len(answerable))
    if not answerable:
        return metrics

    for k in ks:
        metrics.recall[k] = sum(
            recall_at_k([s.chunk.chunk_id for s in results.get(q.question_id, [])], q.relevance, k)
            for q in answerable
        ) / len(answerable)
        metrics.set_recall[k] = sum(
            set_recall_at_k(
                [s.chunk.chunk_id for s in results.get(q.question_id, [])], q.relevance, k
            )
            for q in answerable
        ) / len(answerable)
        metrics.precision[k] = sum(
            precision_at_k(
                [s.chunk.chunk_id for s in results.get(q.question_id, [])], q.relevance, k
            )
            for q in answerable
        ) / len(answerable)
        metrics.ndcg[k] = sum(
            ndcg_at_k([s.chunk.chunk_id for s in results.get(q.question_id, [])], q.relevance, k)
            for q in answerable
        ) / len(answerable)

    metrics.mrr = sum(
        reciprocal_rank([s.chunk.chunk_id for s in results.get(q.question_id, [])], q.relevance)
        for q in answerable
    ) / len(answerable)
    return metrics


@dataclass
class RefusalMetrics:
    true_refusals: int = 0
    false_refusals: int = 0
    missed_refusals: int = 0
    true_answers: int = 0

    @property
    def precision(self) -> float:
        total = self.true_refusals + self.false_refusals
        return self.true_refusals / total if total else 0.0

    @property
    def recall(self) -> float:
        total = self.true_refusals + self.missed_refusals
        return self.true_refusals / total if total else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def answer_rate(self) -> float:
        """Share of answerable questions actually answered.

        The counterweight to refusal recall. Optimising refusal alone drives
        this to zero and the product stops existing.
        """
        total = self.true_answers + self.false_refusals
        return self.true_answers / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctly_refused": self.true_refusals,
            "wrongly_refused": self.false_refusals,
            "missed_refusals": self.missed_refusals,
            "correctly_answered": self.true_answers,
            "refusal_precision": round(self.precision, 4),
            "refusal_recall": round(self.recall, 4),
            "refusal_f1": round(self.f1, 4),
            "answer_rate_on_answerable": round(self.answer_rate, 4),
        }


def evaluate_refusal(answers: dict[str, Answer], questions: list[GoldQuestion]) -> RefusalMetrics:
    metrics = RefusalMetrics()
    for question in questions:
        answer = answers.get(question.question_id)
        if answer is None:
            continue
        refused = not answer.answered
        if question.answerable:
            if refused:
                metrics.false_refusals += 1
            else:
                metrics.true_answers += 1
        else:
            if refused:
                metrics.true_refusals += 1
            else:
                metrics.missed_refusals += 1
    return metrics


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
    return ordered[index]
