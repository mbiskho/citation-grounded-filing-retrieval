"""Rerankers.

A bi-encoder has to compress a passage into one vector before it ever sees the
query, so it cannot represent "this passage answers *this* question" - only
"this passage is about roughly that topic". A cross-encoder reads query and
passage together and scores the pair. It is far too slow to run over 6,000
chunks, which is exactly why it goes second: retrieve 50 cheaply, rerank those
50 expensively.

On this corpus the reranker is the single largest contributor to recall@10 -
see the ablation table. That is unsurprising in hindsight: filings are full of
near-duplicate boilerplate that a bi-encoder cannot separate, and the same
risk factor appears in three companies and three years.
"""

from __future__ import annotations

from typing import Protocol

from ..models import Scored
from .tokenize import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]: ...


class NoOpReranker:
    """Pass-through, for the ablation arm that removes reranking."""

    name = "none"

    def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]:
        return candidates[:top_k]


class LexicalOverlapReranker:
    """Offline reranker: IDF-weighted query coverage, blended with retrieval rank.

    Not a cross-encoder and not a substitute for one. It exists so the full
    pipeline - including the rerank stage - runs in CI with no model download,
    and so the ablation table has an honest middle row between "no reranker"
    and "cross-encoder".

    Two details are the difference between it helping and hurting:

    *IDF weighting.* The first version scored raw term overlap, so matching
    "the company" counted as much as matching "cybersecurity". It ranked
    *below* doing no reranking at all - 0.600 against 0.646 recall@10 - which
    is the correct outcome for a scorer that treats every word as equally
    informative.

    *Keeping the retrieval prior.* A cross-encoder can afford to discard the
    first-stage ranking because it reads the text properly. A lexical scorer
    cannot: throwing away BM25's opinion loses more than the reordering gains.
    So the retrieval rank stays in the score as a decaying prior.

    Even repaired it barely beats no reranking on this corpus. That is
    reported rather than hidden - see `docs/ablation-study.md`.
    """

    name = "lexical"

    def __init__(
        self,
        numeric_bonus: float = 0.35,
        table_bonus: float = 0.05,
        rank_prior: float = 0.3,
        idf: dict[str, float] | None = None,
    ) -> None:
        self.numeric_bonus = numeric_bonus
        self.table_bonus = table_bonus
        self.rank_prior = rank_prior
        #: Populated by HybridRetriever from the BM25 index when available.
        self.idf = idf or {}

    def _score(self, query_tokens: set[str], candidate: Scored) -> float:
        tokens = set(tokenize(candidate.chunk.text))
        if not query_tokens:
            return 0.0

        weights = {t: self.idf.get(t, 1.0) for t in query_tokens}
        total = sum(weights.values()) or 1.0
        overlap = sum(weights[t] for t in query_tokens & tokens) / total

        query_numbers = {t for t in query_tokens if t.startswith(("num:", "money:", "pct:"))}
        if query_numbers:
            matched = len(query_numbers & tokens) / len(query_numbers)
            overlap += self.numeric_bonus * matched
        if candidate.chunk.is_table and query_numbers:
            overlap += self.table_bonus
        return overlap

    def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]:
        query_tokens = set(tokenize(query))
        for candidate in candidates:
            lexical = self._score(query_tokens, candidate)
            candidate.rerank_score = lexical + self.rank_prior / (1 + candidate.rank)
        ranked = sorted(candidates, key=lambda c: (-(c.rerank_score or 0.0), c.rank))
        for rank, candidate in enumerate(ranked):
            candidate.rank = rank
            candidate.score = candidate.rerank_score or 0.0
        return ranked[:top_k]


class CrossEncoderReranker:
    """A real cross-encoder. Source of the published reranked numbers.

    BAAI/bge-reranker-base by default: ~280MB, and it scores a query/passage
    pair directly rather than comparing two independent embeddings.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. "
                "pip install 'filingqa[embeddings]', or use LexicalOverlapReranker."
            ) from exc
        self.name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)

    def rerank(self, query: str, candidates: list[Scored], top_k: int) -> list[Scored]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        for candidate, score in zip(candidates, scores, strict=False):
            candidate.rerank_score = float(score)
        ranked = sorted(candidates, key=lambda c: (-(c.rerank_score or 0.0), c.rank))
        for rank, candidate in enumerate(ranked):
            candidate.rank = rank
            candidate.score = candidate.rerank_score or 0.0
        return ranked[:top_k]


def get_reranker(name: str = "auto", **kwargs) -> Reranker:
    if name in ("none", "noop"):
        return NoOpReranker()
    if name in ("lexical", "offline"):
        return LexicalOverlapReranker(**kwargs)
    if name == "auto":
        try:
            return CrossEncoderReranker(**kwargs)
        except ImportError:
            return LexicalOverlapReranker()
    return CrossEncoderReranker(name, **kwargs)
