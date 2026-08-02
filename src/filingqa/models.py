"""Core data types.

The design decision everything else follows from: **a normalised plain-text
rendering of each filing is the single source of truth**, and every other
object refers to it by character offset.

That means a citation is not a label the model produced and we hope is right.
It is a `(doc_id, start, end)` triple that can be sliced out of the source text
and compared against what the answer claims it says. Verification is a string
comparison, not a judgement call - which is what makes "zero uncited claims"
something you can check rather than assert.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------
# Filings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FilingRef:
    """Enough to locate a filing on EDGAR and name it in a citation."""

    cik: str
    company: str
    ticker: str
    form: str
    fiscal_year: int
    filing_date: str
    accession: str
    primary_document: str

    @property
    def doc_id(self) -> str:
        return f"{self.ticker.lower()}-{self.fiscal_year}-{self.form.replace('-', '').lower()}"

    @property
    def url(self) -> str:
        acc = self.accession.replace("-", "")
        cik = self.cik.lstrip("0")
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{self.primary_document}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "doc_id": self.doc_id, "url": self.url}


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"


@dataclass
class Block:
    """One structural unit of a filing, with its span in the rendered text.

    `start`/`end` index into `Filing.text`. They are assigned during rendering
    and are the anchor for every downstream citation, so nothing may mutate
    `Filing.text` after blocks are built.
    """

    block_type: BlockType
    text: str
    start: int
    end: int
    item: str = ""  # "Item 1A", "Item 7", ... - empty before the front matter
    #: Tables only: rows of normalised cells, kept so numeric retrieval can
    #: match on a cell rather than on a flattened string.
    rows: list[list[str]] = field(default_factory=list)
    #: Tables only: the nearest preceding heading or caption.
    caption: str = ""

    @property
    def is_table(self) -> bool:
        return self.block_type is BlockType.TABLE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "block_type": self.block_type.value,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "item": self.item,
        }
        if self.is_table:
            payload["rows"] = self.rows
            payload["caption"] = self.caption
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Block:
        return cls(
            block_type=BlockType(data["block_type"]),
            text=data["text"],
            start=data["start"],
            end=data["end"],
            item=data.get("item", ""),
            rows=data.get("rows", []),
            caption=data.get("caption", ""),
        )


@dataclass
class Filing:
    """A parsed filing: the rendered text plus the blocks that index into it."""

    ref: FilingRef
    text: str
    blocks: list[Block]

    @property
    def doc_id(self) -> str:
        return self.ref.doc_id

    @property
    def checksum(self) -> str:
        """Guards against citations drifting when a filing is re-parsed.

        A stored citation is an offset into a specific rendering. If the parser
        changes and the text shifts by one character, every citation silently
        points somewhere slightly wrong. Comparing checksums turns that into a
        loud failure.
        """
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]

    def blocks_for_item(self, item: str) -> list[Block]:
        return [b for b in self.blocks if b.item == item]

    @property
    def items(self) -> list[str]:
        seen: list[str] = []
        for b in self.blocks:
            if b.item and b.item not in seen:
                seen.append(b.item)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.to_dict(),
            "checksum": self.checksum,
            "text": self.text,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Filing:
        ref_data = {k: v for k, v in data["ref"].items() if k in FilingRef.__dataclass_fields__}
        filing = cls(
            ref=FilingRef(**ref_data),
            text=data["text"],
            blocks=[Block.from_dict(b) for b in data["blocks"]],
        )
        stored = data.get("checksum")
        if stored and stored != filing.checksum:
            raise ValueError(
                f"{filing.doc_id}: text checksum {filing.checksum} does not match stored "
                f"{stored}; citations built against the old rendering are invalid"
            )
        return filing


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


@dataclass
class Chunk:
    """A retrievable unit. Carries its own provenance.

    There is no separate "metadata store": if a chunk can be retrieved, it can
    be cited, because the span travels with it.
    """

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    item: str = ""
    is_table: bool = False
    caption: str = ""
    company: str = ""
    fiscal_year: int = 0

    @property
    def citation_label(self) -> str:
        """What a reader sees, e.g. 'AAPL 2024 10-K, Item 7'."""
        base = f"{self.company} {self.fiscal_year} 10-K"
        return f"{base}, {self.item}" if self.item else base

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


@dataclass
class Scored:
    """A chunk with the scores that got it here.

    Component scores are kept rather than collapsed, because the ablation study
    needs to attribute a result to BM25, the dense retriever or the reranker,
    and because "why did this rank first" is the most common debugging question.
    """

    chunk: Chunk
    score: float
    bm25_score: float | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "score": round(self.score, 6),
            "bm25_score": None if self.bm25_score is None else round(self.bm25_score, 6),
            "dense_score": None if self.dense_score is None else round(self.dense_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
            "rank": self.rank,
            "item": self.chunk.item,
            "is_table": self.chunk.is_table,
        }


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


@dataclass
class Citation:
    """A verifiable pointer into a filing.

    `quote` is what the answer claims the filing says. `verified` is whether
    that claim survived being checked against the actual span. An unverified
    citation is worse than no citation - it looks like provenance and is not.
    """

    doc_id: str
    start: int
    end: int
    quote: str
    label: str = ""
    chunk_id: str = ""
    verified: bool = False
    verification_note: str = ""

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Verdict(str, Enum):
    ANSWERED = "answered"
    REFUSED_LOW_CONFIDENCE = "refused_low_confidence"
    REFUSED_NO_CITATION = "refused_no_citation"
    REFUSED_OUT_OF_CORPUS = "refused_out_of_corpus"

    @property
    def is_refusal(self) -> bool:
        return self is not Verdict.ANSWERED


@dataclass
class Answer:
    """The product. An answer with no verified citation is a refusal."""

    question: str
    verdict: Verdict
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[Scored] = field(default_factory=list)
    confidence: float = 0.0
    refusal_reason: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    generator: str = ""
    #: Per-stage retrieval timings and decisions, for the demo and debugging.
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.verdict is Verdict.ANSWERED

    @property
    def all_citations_verified(self) -> bool:
        return bool(self.citations) and all(c.verified for c in self.citations)

    @property
    def uncited_sentences(self) -> list[str]:
        """Sentences in the answer carrying no citation marker.

        The headline claim of this project is that this list is empty for every
        answered question. It is computed from the rendered text rather than
        tracked during generation, so it cannot be quietly satisfied by
        bookkeeping.
        """
        if not self.answered:
            return []
        out = []
        for sentence in split_sentences(self.text):
            if _TRIVIAL.match(sentence):
                continue
            if not CITE_MARKER.search(sentence):
                out.append(sentence)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "verdict": self.verdict.value,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "refusal_reason": self.refusal_reason,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 8),
            "generator": self.generator,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved": [s.to_dict() for s in self.retrieved[:10]],
        }


#: Inline citation marker, e.g. "[1]" or "[2, 3]".
CITE_MARKER = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")

#: Sentences that carry no claim and therefore need no citation.
_TRIVIAL = re.compile(r"^\s*(?:\W*|(?:based on|according to)\b.{0,40})\s*$", re.IGNORECASE)

#: A sentence boundary is terminal punctuation followed by a capital or an
#: opening paren. A "[" is deliberately *not* a boundary trigger: the only
#: brackets in generated text are our own citation markers, and treating one
#: as the start of a sentence detaches the marker from the claim it supports.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    """Sentence split that does not trip over financial prose.

    Abbreviations like "U.S.", "Inc.", "No. 3" and decimals like "1.5 billion"
    all look like sentence ends to a naive splitter, and a false split creates
    a fragment with no citation marker - which would show up as a phantom
    uncited claim. Protecting them first is cheaper than explaining the
    resulting metric.
    """
    if not text.strip():
        return []
    protected = text
    for pattern, replacement in _PROTECTED:
        protected = pattern.sub(replacement, protected)
    parts = _SENTENCE_END.split(protected)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


#: Sentinel standing in for a protected full stop. Must not appear in filing
#: text, and must survive being written to and read back from JSON.
_DOT = ""

_PROTECTED = [
    (
        re.compile(r"\b(U\.S|U\.K|E\.U|Inc|Corp|Ltd|Co|No|Nos|Jr|Sr|St|vs|approx|Dr|Mr|Ms)\."),
        "\\1" + _DOT,
    ),
    (re.compile(r"(\d)\.(\d)"), "\\1" + _DOT + "\\2"),
    (re.compile(r"\b([A-Z])\.(?=[A-Z]\.)"), "\\1" + _DOT),
]


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
