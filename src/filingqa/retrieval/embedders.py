"""Embedders.

Three implementations behind one interface:

* `SentenceTransformerEmbedder` - the real one. Published numbers come from
  this. Default model is BAAI/bge-small-en-v1.5: 384 dimensions, ~130MB, and
  it handles the retrieval-query/passage asymmetry properly, which matters
  more than raw benchmark rank for this kind of corpus.
* `HashingEmbedder` - dependency-free, deterministic, no download. Exists so
  tests and CI run in seconds without pulling 400MB of torch, and so anyone
  can clone the repo and get a working system immediately. It is a genuinely
  weaker retriever and the README says so with numbers.
* `OpenAIEmbedder` - for when you want text-embedding-3 quality and are happy
  to pay per query.

The interface is deliberately tiny. Anything richer and swapping embedders
stops being a one-line change, which is the thing the ablation study needs.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

import numpy as np

from .tokenize import char_ngrams, tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class HashingEmbedder:
    """Hashed character n-grams and word tokens, IDF-weighted.

    Not a neural model and not pretending to be. It is a sparse lexical
    signature projected into a fixed-width dense vector, so it captures
    surface overlap and morphology but nothing about meaning: it will not
    connect "revenue" to "net sales", which is the single most common
    paraphrase in this domain.

    It earns its place by being deterministic and instant. Every test in this
    repository runs against it, so the retrieval logic is exercised on every
    push without a model download, and the ablation table shows exactly what
    it costs.
    """

    name = "hashing"

    def __init__(self, dim: int = 512, ngram: int = 4, seed: int = 17) -> None:
        self.dim = dim
        self.ngram = ngram
        self.seed = seed
        self._idf: dict[str, float] = {}
        self._fitted = False

    def _features(self, text: str) -> list[str]:
        return tokenize(text) + char_ngrams(text, self.ngram)

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(f"{self.seed}:{feature}".encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        # Signed hashing: half the features get -1, which keeps collisions
        # from systematically inflating similarity.
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def fit(self, texts: list[str]) -> HashingEmbedder:
        """Learn IDF over the corpus.

        Without it, boilerplate that appears in every filing dominates the
        vector and everything looks similar to everything.
        """
        document_frequency: dict[str, int] = {}
        for text in texts:
            for feature in set(self._features(text)):
                document_frequency[feature] = document_frequency.get(feature, 0) + 1
        total = max(1, len(texts))
        self._idf = {
            feature: math.log(1 + total / (1 + count))
            for feature, count in document_frequency.items()
        }
        self._fitted = True
        return self

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        counts: dict[str, int] = {}
        for feature in self._features(text):
            counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            weight = (1 + math.log(count)) * self._idf.get(feature, 1.0)
            index, sign = self._bucket(feature)
            vector[index] += sign * weight
        return vector

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return _l2_normalise(np.vstack([self._encode_one(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return _l2_normalise(self._encode_one(text)[None, :])[0]


class SentenceTransformerEmbedder:
    """A real bi-encoder. The source of the published retrieval numbers.

    BGE models expect an instruction prefix on the query side and none on the
    passage side. Skipping the prefix costs several points of recall and is
    invisible unless you read the model card, so it is applied here rather
    than left to the caller.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 64,
        query_prefix: str | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. "
                "pip install 'filingqa[embeddings]', or use HashingEmbedder."
            ) from exc

        self.model_name = model_name
        self.name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        if query_prefix is None:
            query_prefix = (
                "Represent this sentence for searching relevant passages: "
                if "bge" in model_name.lower()
                else ""
            )
        self.query_prefix = query_prefix

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 2000,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            self.query_prefix + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.astype(np.float32)


class OpenAIEmbedder:
    """OpenAI embeddings. Costs money per call; priced into the eval report."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        batch_size: int = 256,
    ) -> None:
        self.name = model
        self.model = model
        self.batch_size = batch_size
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.dim = 1536 if "small" in model else 3072
        #: USD per 1M tokens, for the cost column in the eval report.
        self.price_per_mtok = 0.02 if "small" in model else 0.13
        self.tokens_used = 0

    def _post(self, inputs: list[str]) -> np.ndarray:
        import json
        import urllib.request

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self.model, "input": inputs}).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read())
        self.tokens_used += payload.get("usage", {}).get("total_tokens", 0)
        return np.array([d["embedding"] for d in payload["data"]], dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        batches = [
            self._post(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        return _l2_normalise(np.vstack(batches))

    def encode_query(self, text: str) -> np.ndarray:
        return _l2_normalise(self._post([text]))[0]

    @property
    def cost_usd(self) -> float:
        return self.tokens_used * self.price_per_mtok / 1_000_000


def get_embedder(name: str = "auto", **kwargs) -> Embedder:
    """Build an embedder by name.

    `auto` prefers the real model and silently degrades to hashing when
    sentence-transformers is absent - which is what makes `git clone && make
    demo` work on a machine with no GPU, no torch and no patience.
    """
    if name in ("hashing", "offline"):
        return HashingEmbedder(**kwargs)
    if name == "openai":
        return OpenAIEmbedder(**kwargs)
    if name == "auto":
        try:
            return SentenceTransformerEmbedder(**kwargs)
        except ImportError:
            return HashingEmbedder()
    return SentenceTransformerEmbedder(name, **kwargs)
