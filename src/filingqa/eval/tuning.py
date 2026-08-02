"""Parameter sweeps, run on the dev split only.

Everything here touches the dev half of the gold set and nothing else. The
headline numbers in the README come from the test half, scored once with the
parameters this module selected.

That discipline costs a couple of points of reported recall and it is the
difference between a measurement and a sales pitch. Tuning `b`, the fusion
weights and the refusal thresholds against the same 100 questions you then
report on will produce a better-looking number every time, and the number will
not survive contact with a new corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..answer import FilingQA, RefusalPolicy
from ..models import Chunk, Filing
from ..retrieval import HybridRetriever, RetrievalConfig
from ..retrieval.indexes import BM25Params
from .goldset import GoldQuestion
from .metrics import evaluate_refusal, recall_at_k


@dataclass
class SweepResult:
    parameter: str
    values: list[Any] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    best_value: Any = None
    best_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "points": [
                {"value": v, "score": round(s, 4)}
                for v, s in zip(self.values, self.scores, strict=False)
            ],
            "best_value": self.best_value,
            "best_score": round(self.best_score, 4),
        }


def _recall(retriever: HybridRetriever, questions: list[GoldQuestion], k: int = 10) -> float:
    answerable = [q for q in questions if q.answerable and q.gold_chunks]
    if not answerable:
        return 0.0
    total = 0.0
    for question in answerable:
        results = retriever.search(question.question, top_k=k)
        total += recall_at_k([s.chunk.chunk_id for s in results], question.relevance, k)
    return total / len(answerable)


def sweep_bm25_b(
    chunks: list[Chunk],
    dev: list[GoldQuestion],
    *,
    values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    embedder=None,
    reranker=None,
) -> SweepResult:
    """Length normalisation.

    Worth sweeping specifically because the usual default of 0.75 assumes long
    documents are padded. Here a long chunk is often a complete financial
    table, which is the most information-dense thing in the corpus, so
    penalising it by length is backwards.
    """
    result = SweepResult(parameter="bm25_b")
    for b in values:
        retriever = HybridRetriever(
            chunks,
            embedder=embedder,
            reranker=reranker,
            config=RetrievalConfig(use_bm25=True, use_dense=False),
        )
        retriever.bm25.params = BM25Params(b=b)
        retriever.bm25.build(chunks)
        retriever._built = True
        score = _recall(retriever, dev)
        result.values.append(b)
        result.scores.append(score)
    best = max(range(len(result.scores)), key=lambda i: result.scores[i])
    result.best_value = result.values[best]
    result.best_score = result.scores[best]
    return result


def sweep_refusal_thresholds(
    retriever: HybridRetriever,
    filings: list[Filing],
    dev: list[GoldQuestion],
    *,
    score_values: tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6),
    coverage_values: tuple[float, ...] = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6),
) -> dict[str, Any]:
    """Grid over the two refusal signals, scored by refusal F1.

    F1 rather than refusal recall, because recall alone is maximised by
    refusing everything. The answer rate on answerable questions is reported
    alongside so the trade is visible rather than implied.

    Thresholds are not transferable between rerankers - a cross-encoder's
    scores live on a different scale entirely - which is why this is a
    function you re-run rather than a constant someone copies.
    """
    best: dict[str, Any] = {"f1": -1.0}
    grid: list[dict[str, Any]] = []

    for min_score in score_values:
        for min_coverage in coverage_values:
            policy = RefusalPolicy(
                min_top_score=min_score,
                min_term_coverage=min_coverage,
                check_scope=True,
            )
            engine = FilingQA(retriever, filings, refusal=policy)
            answers = {q.question_id: engine.ask(q.question) for q in dev}
            metrics = evaluate_refusal(answers, dev)
            point = {
                "min_top_score": min_score,
                "min_term_coverage": min_coverage,
                "refusal_f1": round(metrics.f1, 4),
                "refusal_precision": round(metrics.precision, 4),
                "refusal_recall": round(metrics.recall, 4),
                "answer_rate": round(metrics.answer_rate, 4),
            }
            grid.append(point)
            if metrics.f1 > best["f1"]:
                best = {
                    "f1": metrics.f1,
                    "min_top_score": min_score,
                    "min_term_coverage": min_coverage,
                    "refusal_precision": metrics.precision,
                    "refusal_recall": metrics.recall,
                    "answer_rate": metrics.answer_rate,
                }

    return {"best": best, "grid": grid}


def save_tuning(payload: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
