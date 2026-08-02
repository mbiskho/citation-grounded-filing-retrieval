"""Hybrid retrieval pipeline.

    query -> BM25 (top 50) ─┐
                            ├─ RRF ─> top 50 ─> reranker ─> top k
             dense (top 50) ┘

Each stage is switchable so the ablation study can remove exactly one thing at
a time and attribute the difference. That is the only reason to build it this
way rather than as one function.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Chunk, Scored
from .embedders import (
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)
from .expansion import ExpansionResult, QueryExpander
from .indexes import BM25Index, BM25Params, DenseIndex, reciprocal_rank_fusion, to_scored
from .rerank import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    NoOpReranker,
    Reranker,
    get_reranker,
)

__all__ = [
    "HybridRetriever",
    "RetrievalConfig",
    "BM25Index",
    "BM25Params",
    "DenseIndex",
    "Embedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "get_embedder",
    "Reranker",
    "NoOpReranker",
    "LexicalOverlapReranker",
    "CrossEncoderReranker",
    "get_reranker",
    "reciprocal_rank_fusion",
    "QueryExpander",
    "ExpansionResult",
]


@dataclass
class RetrievalConfig:
    use_bm25: bool = True
    use_dense: bool = True
    #: Candidates pulled from each retriever before fusion. Larger costs
    #: rerank time linearly; smaller caps the ceiling on recall.
    candidates_per_retriever: int = 50
    rerank_top_k: int = 50
    top_k: int = 10
    rrf_k: float = 60.0
    #: Fusion weights (bm25, dense). Equal by default - an asymmetric default
    #: would be a tuned constant masquerading as a principle.
    weights: tuple[float, float] = (1.0, 1.0)
    #: Restrict to a company/year when the question names one. Filters are a
    #: precision win and a recall risk: a wrong filter is an unrecoverable
    #: miss, so only explicit matches apply.
    allow_filters: bool = True
    #: Rewrite the query into filing vocabulary before retrieval. The largest
    #: single recall win on this corpus - see the ablation table.
    expand_query: bool = True


@dataclass
class RetrievalTrace:
    """What each stage contributed. Populated on every query, cheap to keep."""

    bm25_ms: float = 0.0
    dense_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    bm25_hits: int = 0
    dense_hits: int = 0
    fused_hits: int = 0
    filters: dict = field(default_factory=dict)
    expanded_query: str = ""
    expansion_terms: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "expanded_query": self.expanded_query,
            "expansion_terms": self.expansion_terms,
            "bm25_ms": round(self.bm25_ms, 2),
            "dense_ms": round(self.dense_ms, 2),
            "fusion_ms": round(self.fusion_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "bm25_hits": self.bm25_hits,
            "dense_hits": self.dense_hits,
            "fused_hits": self.fused_hits,
            "filters": self.filters,
        }


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.chunks = chunks
        self.config = config or RetrievalConfig()
        self.embedder = embedder or get_embedder("auto")
        self.reranker = reranker or get_reranker("auto")
        self.expander = QueryExpander()

        # Fusing a weak retriever into a strong one makes the strong one
        # worse. With the offline hashing embedder, dense retrieval scores
        # recall@10 0.405 against BM25's 0.643, and fusing them lands at
        # 0.595 - five points *below* not using it at all. So when the
        # embedder is the no-download fallback, dense fusion defaults off.
        #
        # An explicitly-constructed RetrievalConfig always wins; this only
        # changes what happens when the caller expressed no preference, and
        # it flips back automatically once a real embedding model is present.
        if config is None and getattr(self.embedder, "name", "") == "hashing":
            self.config.use_dense = False
        self.bm25 = BM25Index()
        self.dense = DenseIndex(self.embedder)
        self._index_of = {c.chunk_id: i for i, c in enumerate(chunks)}
        self._built = False

    # -- build -------------------------------------------------------------

    def build(self, *, verbose: bool = True) -> HybridRetriever:
        if self.config.use_bm25:
            started = time.perf_counter()
            self.bm25.build(self.chunks)
            self._share_idf()
            if verbose:
                print(
                    f"  bm25   {len(self.bm25.postings):>8,} terms   "
                    f"{(time.perf_counter() - started):.1f}s"
                )
        if self.config.use_dense:
            started = time.perf_counter()
            self.dense.build(self.chunks)
            if verbose:
                print(
                    f"  dense  {self.dense.matrix.shape}  {self.embedder.name}   "
                    f"{(time.perf_counter() - started):.1f}s"
                )
        self._built = True
        return self

    def _share_idf(self) -> None:
        """Lend the BM25 IDF table to a reranker that can use it.

        Recomputing corpus statistics inside the reranker would be wasteful
        and, worse, would let the two components drift apart as the corpus
        changes.
        """
        if hasattr(self.reranker, "idf") and not getattr(self.reranker, "idf", None):
            self.reranker.idf = self.bm25.idf

    # -- query -------------------------------------------------------------

    def _filters(self, query: str) -> dict:
        """Company and fiscal year mentioned explicitly in the question.

        Only exact, unambiguous mentions count. A filter that fires wrongly
        removes the answer from the candidate set entirely, and no amount of
        downstream reranking recovers from that.
        """
        if not self.config.allow_filters:
            return {}
        filters: dict = {}
        lowered = query.lower()

        aliases = {
            "AAPL": ["apple", "aapl"],
            "JPM": ["jpmorgan", "jp morgan", "jpm", "chase"],
            "F": ["ford motor", "ford"],
        }
        matched = [
            ticker for ticker, names in aliases.items() if any(name in lowered for name in names)
        ]
        # Two companies named means a comparison question; filtering to one
        # would guarantee a wrong answer.
        if len(matched) == 1:
            filters["company"] = matched[0]

        years = {
            c.fiscal_year
            for c in self.chunks
            if str(c.fiscal_year) in query or f"fy{c.fiscal_year}" in lowered
        }
        if len(years) == 1:
            filters["fiscal_year"] = years.pop()
        return filters

    def _passes(self, chunk: Chunk, filters: dict) -> bool:
        if "company" in filters and chunk.company != filters["company"]:
            return False
        return not ("fiscal_year" in filters and chunk.fiscal_year != filters["fiscal_year"])

    def search(self, query: str, *, top_k: int | None = None) -> list[Scored]:
        """Ranked passages for a query."""
        return self.search_with_trace(query, top_k=top_k)[0]

    def search_with_trace(
        self, query: str, *, top_k: int | None = None
    ) -> tuple[list[Scored], RetrievalTrace]:
        """Ranked passages plus per-stage timings and decisions.

        Kept as a separate method rather than a `trace=True` flag: a function
        whose return type depends on an argument value is awkward to type and
        awkward to call, and every caller here knows statically which one it
        wants.
        """
        if not self._built:
            raise RuntimeError("call build() before search()")

        top_k = top_k or self.config.top_k
        record = RetrievalTrace()
        overall = time.perf_counter()

        # Filters and reranking read the *original* question - expansion adds
        # filing vocabulary that would confuse entity detection and dilute the
        # rerank signal. Only the two retrievers see the expanded form.
        original = query
        if self.config.expand_query:
            expansion = self.expander.expand(query)
            query = expansion.expanded
            record.expanded_query = expansion.expanded
            record.expansion_terms = expansion.added

        n = self.config.candidates_per_retriever
        rankings: list[list[tuple[int, float]]] = []
        bm25_scores: dict[int, float] = {}
        dense_scores: dict[int, float] = {}

        if self.config.use_bm25:
            started = time.perf_counter()
            hits = self.bm25.search(query, top_k=n)
            record.bm25_ms = (time.perf_counter() - started) * 1000
            record.bm25_hits = len(hits)
            bm25_scores = dict(hits)
            rankings.append(hits)

        if self.config.use_dense:
            started = time.perf_counter()
            hits = self.dense.search(query, top_k=n)
            record.dense_ms = (time.perf_counter() - started) * 1000
            record.dense_hits = len(hits)
            dense_scores = dict(hits)
            rankings.append(hits)

        if not rankings:
            raise ValueError("both retrievers are disabled")

        started = time.perf_counter()
        if len(rankings) == 1:
            fused = rankings[0]
        else:
            weights = list(self.config.weights)[: len(rankings)]
            fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k, weights=weights)
        record.fusion_ms = (time.perf_counter() - started) * 1000

        filters = self._filters(original)
        record.filters = filters
        if filters:
            filtered = [(i, s) for i, s in fused if self._passes(self.chunks[i], filters)]
            # A filter that removes everything is a filter that misfired.
            if filtered:
                fused = filtered
            else:
                record.filters = {**filters, "applied": False}
        record.fused_hits = len(fused)

        candidates = to_scored(
            fused[: self.config.rerank_top_k],
            self.chunks,
            bm25=bm25_scores,
            dense=dense_scores,
        )

        started = time.perf_counter()
        results = self.reranker.rerank(original, candidates, top_k)
        record.rerank_ms = (time.perf_counter() - started) * 1000
        record.total_ms = (time.perf_counter() - overall) * 1000

        return results, record

    # -- persistence -------------------------------------------------------

    def save(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self.chunks)
        )
        if self.config.use_bm25:
            self.bm25.save(directory / "bm25.pkl")
        if self.config.use_dense:
            self.dense.save(directory / "dense.npz")
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "embedder": self.embedder.name,
                    "dim": getattr(self.embedder, "dim", None),
                    "reranker": self.reranker.name,
                    "chunks": len(self.chunks),
                },
                indent=2,
            )
        )

    @classmethod
    def load(
        cls,
        directory: Path | str,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> HybridRetriever:
        directory = Path(directory)
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (directory / "chunks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        retriever = cls(chunks, embedder=embedder, reranker=reranker, config=config)
        if (directory / "bm25.pkl").exists():
            retriever.bm25 = BM25Index.load(directory / "bm25.pkl")
        if (directory / "dense.npz").exists():
            retriever.dense.load(directory / "dense.npz")
        retriever._built = True
        return retriever


def load_chunks(path: Path | str) -> list[Chunk]:
    return [
        Chunk.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
