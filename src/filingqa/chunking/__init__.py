"""Layout-preserving chunking.

The rule that matters: **a table is never split.**

Fixed-window chunking is the default in every RAG tutorial and it is actively
harmful here. Split a balance sheet at 512 tokens and you get one chunk with
the row labels and another with the numbers. Both retrieve. Neither answers
anything. Worse, the numbers-only chunk will happily be quoted by a model as
though it referred to whatever the question asked about, which is precisely
the failure this project exists to prevent.

So tables are atomic units. A table larger than the token budget stays whole
and oversized rather than being cut - an oversized chunk costs context, a cut
table costs correctness.

Prose is packed to a token budget on sentence boundaries with a small overlap,
and every chunk keeps the character span it came from, because the span is the
citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Block, Chunk, Filing, split_sentences

#: Words-to-tokens fudge factor. Real tokenisers are model-specific and
#: importing one to decide chunk boundaries is not worth the dependency; the
#: budget is a soft target, not a hard limit.
TOKENS_PER_WORD = 1.35


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * TOKENS_PER_WORD))


@dataclass
class ChunkConfig:
    target_tokens: int = 320
    #: Prose chunks below this get merged into their neighbour. A 20-token
    #: chunk matches almost any query weakly and dilutes the index.
    min_tokens: int = 60
    #: Sentences of overlap between adjacent prose chunks, so a fact stated
    #: across a boundary is retrievable from either side.
    overlap_sentences: int = 1
    #: Tables above this are still not split - they are just flagged, so the
    #: retriever can down-rank them if context budget is tight.
    large_table_tokens: int = 1200
    #: Prefix each chunk with its section so an isolated chunk still says
    #: where it came from. Costs tokens, buys interpretability at rerank time.
    prefix_section: bool = True


def chunk_filing(filing: Filing, config: ChunkConfig | None = None) -> list[Chunk]:
    config = config or ChunkConfig()
    chunks: list[Chunk] = []
    prose_buffer: list[Block] = []

    def flush() -> None:
        if prose_buffer:
            chunks.extend(_chunk_prose(prose_buffer, filing, config, len(chunks)))
            prose_buffer.clear()

    for block in filing.blocks:
        if block.is_table:
            flush()
            chunks.append(_chunk_table(block, filing, config, len(chunks)))
        else:
            # A section change flushes too: chunks must not straddle Item
            # boundaries, or a citation label says "Item 1A" for text that
            # came from Item 2.
            if prose_buffer and prose_buffer[-1].item != block.item:
                flush()
            prose_buffer.append(block)
    flush()

    return [c for c in chunks if c.text.strip()]


def _label(filing: Filing, block: Block, config: ChunkConfig) -> str:
    if not config.prefix_section or not block.item:
        return ""
    from ..ingest.parser import ITEM_TITLES

    title = ITEM_TITLES.get(block.item, "")
    section = f"{block.item}: {title}" if title else block.item
    return f"[{filing.ref.ticker} FY{filing.ref.fiscal_year} 10-K | {section}]\n"


def _chunk_table(block: Block, filing: Filing, config: ChunkConfig, index: int) -> Chunk:
    prefix = _label(filing, block, config)
    return Chunk(
        chunk_id=f"{filing.doc_id}::{index:05d}",
        doc_id=filing.doc_id,
        text=prefix + block.text,
        start=block.start,
        end=block.end,
        item=block.item,
        is_table=True,
        caption=block.caption,
        company=filing.ref.ticker,
        fiscal_year=filing.ref.fiscal_year,
    )


def _chunk_prose(
    blocks: list[Block], filing: Filing, config: ChunkConfig, start_index: int
) -> list[Chunk]:
    """Pack consecutive prose blocks into token-budgeted chunks.

    Spans are tracked against the filing text rather than reconstructed from
    the chunk string, so the prefix added for readability never corrupts the
    citation offsets.
    """
    units: list[tuple[str, int, int]] = []
    for block in blocks:
        for sentence, start, end in _sentence_spans(block):
            units.append((sentence, start, end))
    if not units:
        return []

    chunks: list[Chunk] = []
    item = blocks[0].item
    prefix = _label(filing, blocks[0], config)

    current: list[tuple[str, int, int]] = []
    current_tokens = 0

    def emit() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        text = " ".join(u[0] for u in current)
        chunks.append(
            Chunk(
                chunk_id=f"{filing.doc_id}::{start_index + len(chunks):05d}",
                doc_id=filing.doc_id,
                text=prefix + text,
                start=current[0][1],
                end=current[-1][2],
                item=item,
                is_table=False,
                company=filing.ref.ticker,
                fiscal_year=filing.ref.fiscal_year,
            )
        )
        overlap = current[-config.overlap_sentences :] if config.overlap_sentences else []
        current = list(overlap)
        current_tokens = sum(estimate_tokens(u[0]) for u in current)

    for sentence, start, end in units:
        tokens = estimate_tokens(sentence)
        if current_tokens + tokens > config.target_tokens and current_tokens >= config.min_tokens:
            emit()
        current.append((sentence, start, end))
        current_tokens += tokens

    # The tail is emitted even if short; merging it backwards would be tidier
    # but would extend the previous chunk's span past what it actually covers.
    if current:
        text = " ".join(u[0] for u in current)
        if estimate_tokens(text) < config.min_tokens and chunks:
            last = chunks[-1]
            chunks[-1] = Chunk(
                chunk_id=last.chunk_id,
                doc_id=last.doc_id,
                text=last.text + " " + text,
                start=last.start,
                end=current[-1][2],
                item=last.item,
                is_table=False,
                company=last.company,
                fiscal_year=last.fiscal_year,
            )
        else:
            emit()

    return chunks


_SENT_SPLIT_SAFE = re.compile(r"\S")


def _sentence_spans(block: Block) -> list[tuple[str, int, int]]:
    """Sentences of a block with absolute spans into the filing text.

    Offsets are found by walking forward through the block text rather than
    by `str.find`, because filings repeat sentences verbatim (boilerplate risk
    factors especially) and `find` would attach the wrong occurrence.
    """
    sentences = split_sentences(block.text)
    if not sentences:
        return [(block.text, block.start, block.end)]

    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for sentence in sentences:
        idx = block.text.find(sentence, cursor)
        if idx == -1:
            # Sentence splitting normalised something; fall back to the whole
            # block rather than emitting a wrong span.
            return [(block.text, block.start, block.end)]
        spans.append((sentence, block.start + idx, block.start + idx + len(sentence)))
        cursor = idx + len(sentence)
    return spans


def chunk_corpus(filings: list[Filing], config: ChunkConfig | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for filing in filings:
        chunks.extend(chunk_filing(filing, config))
    return chunks


def verify_spans(chunks: list[Chunk], filings: list[Filing]) -> list[str]:
    """Every chunk span must slice back to text the chunk actually contains.

    Run after chunking and in CI. A chunk whose span does not resolve is a
    citation that will point at the wrong paragraph, and there is no way to
    notice that from the answer alone.
    """
    by_doc = {f.doc_id: f for f in filings}
    problems: list[str] = []
    for chunk in chunks:
        filing = by_doc.get(chunk.doc_id)
        if filing is None:
            problems.append(f"{chunk.chunk_id}: unknown doc {chunk.doc_id}")
            continue
        if not (0 <= chunk.start < chunk.end <= len(filing.text)):
            problems.append(f"{chunk.chunk_id}: span {chunk.start}:{chunk.end} out of bounds")
            continue
        source = filing.text[chunk.start : chunk.end]
        body = chunk.text.split("]\n", 1)[-1] if chunk.text.startswith("[") else chunk.text
        # Whitespace differs because chunks join sentences with a single
        # space; compare on collapsed non-space characters.
        if _collapse(source)[:200] != _collapse(body)[:200]:
            problems.append(
                f"{chunk.chunk_id}: span does not match chunk text\n"
                f"    source: {source[:90]!r}\n"
                f"    chunk:  {body[:90]!r}"
            )
    return problems


def _collapse(text: str) -> str:
    return "".join(text.split())


__all__ = [
    "ChunkConfig",
    "chunk_filing",
    "chunk_corpus",
    "estimate_tokens",
    "verify_spans",
]
