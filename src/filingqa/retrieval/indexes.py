"""The two indexes and how their results are combined.

BM25 and a dense retriever fail in opposite directions, which is the entire
argument for running both:

* **BM25 misses paraphrase.** A question about "revenue" does not match a
  filing that says "net sales". Apple says "net sales". This is not an edge
  case; it is most of the corpus.
* **Dense retrieval misses exact tokens.** Ask for a specific figure and the
  embedding will happily return a semantically similar table with different
  numbers, which is the worst possible failure here - it looks right.

Fusion is Reciprocal Rank Fusion rather than a weighted score blend. Score
blending requires the two scorers to be on comparable scales, which they are
not: BM25 is unbounded and corpus-dependent, cosine similarity sits in a
narrow band near the top. Normalising either one introduces a tuning knob that
has to be re-fit whenever the corpus changes. RRF only uses rank, so it has
one parameter and it does not drift.
"""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import Chunk, Scored
from .tokenize import tokenize

# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------


@dataclass
class BM25Params:
    k1: float = 1.2
    #: Length normalisation. Lowered from the usual 0.75 because table chunks
    #: are legitimately long - a full balance sheet is not "padded" and should
    #: not be penalised as though it were.
    b: float = 0.5


class BM25Index:
    """Okapi BM25 over the financial tokeniser.

    Implemented directly rather than pulled from a library because the
    tokeniser is the interesting part: money, percentages and negative numbers
    in parentheses have to survive into the postings, and a library tokeniser
    shreds them.
    """

    def __init__(self, params: BM25Params | None = None) -> None:
        self.params = params or BM25Params()
        self.chunk_ids: list[str] = []
        self.doc_lengths: np.ndarray = np.zeros(0)
        self.avg_length: float = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}

    def build(self, chunks: list[Chunk]) -> BM25Index:
        self.chunk_ids = [c.chunk_id for c in chunks]
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(len(chunks), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            lengths[i] = len(tokens)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                postings[token].append((i, count))

        self.postings = dict(postings)
        self.doc_lengths = lengths
        self.avg_length = float(lengths.mean()) if len(lengths) else 0.0

        total = len(chunks)
        self.idf = {
            token: math.log(1 + (total - len(plist) + 0.5) / (len(plist) + 0.5))
            for token, plist in self.postings.items()
        }
        return self

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens or not self.chunk_ids:
            return []

        k1, b = self.params.k1, self.params.b
        scores = np.zeros(len(self.chunk_ids), dtype=np.float32)

        for token in set(tokens):
            plist = self.postings.get(token)
            if not plist:
                continue
            idf = self.idf.get(token, 0.0)
            for index, count in plist:
                norm = 1 - b + b * (self.doc_lengths[index] / max(self.avg_length, 1e-9))
                scores[index] += idf * (count * (k1 + 1)) / (count + k1 * norm)

        top = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        ranked = sorted(((int(i), float(scores[i])) for i in top), key=lambda x: -x[1])
        return [(i, s) for i, s in ranked if s > 0]

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "params": self.params,
                    "chunk_ids": self.chunk_ids,
                    "doc_lengths": self.doc_lengths,
                    "avg_length": self.avg_length,
                    "postings": self.postings,
                    "idf": self.idf,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path | str) -> BM25Index:
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        index = cls(state["params"])
        index.chunk_ids = state["chunk_ids"]
        index.doc_lengths = state["doc_lengths"]
        index.avg_length = state["avg_length"]
        index.postings = state["postings"]
        index.idf = state["idf"]
        return index


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------


class DenseIndex:
    """Exact cosine search over a normalised matrix.

    Flat and exhaustive on purpose. This corpus is ~6k chunks; a brute-force
    matmul answers in under a millisecond, and an approximate index would add
    a dependency, a build step and a recall/latency knob to tune - all to
    optimise something that is not the bottleneck. `docs/architecture.md`
    covers where that stops being true.
    """

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.chunk_ids: list[str] = []
        self.matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def build(self, chunks: list[Chunk]) -> DenseIndex:
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.matrix = self.embedder.encode_documents([c.text for c in chunks])
        return self

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        if not len(self.chunk_ids):
            return []
        vector = self.embedder.encode_query(query)
        scores = self.matrix @ vector
        top = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        return sorted(((int(i), float(scores[i])) for i in top), key=lambda x: -x[1])

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, matrix=self.matrix, chunk_ids=np.array(self.chunk_ids, dtype=object)
        )

    def load(self, path: Path | str) -> DenseIndex:
        payload = np.load(path, allow_pickle=True)
        self.matrix = payload["matrix"]
        self.chunk_ids = list(payload["chunk_ids"])
        return self


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    *,
    k: float = 60.0,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """Combine ranked lists by rank, not by score.

    `k=60` is the value from the original RRF paper and it is not sensitive -
    anything in 20-100 behaves similarly. Its job is to flatten the difference
    between rank 1 and rank 2 so a single retriever cannot dominate on
    confidence alone.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=False):
        for rank, (index, _score) in enumerate(ranking):
            fused[index] += weight / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])


def to_scored(
    indices: list[tuple[int, float]],
    chunks: list[Chunk],
    *,
    bm25: dict[int, float] | None = None,
    dense: dict[int, float] | None = None,
) -> list[Scored]:
    results = []
    for rank, (index, score) in enumerate(indices):
        results.append(
            Scored(
                chunk=chunks[index],
                score=score,
                bm25_score=(bm25 or {}).get(index),
                dense_score=(dense or {}).get(index),
                rank=rank,
            )
        )
    return results
