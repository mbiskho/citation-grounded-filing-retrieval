from __future__ import annotations

import pytest

from filingqa.models import Block, BlockType, Chunk, Filing, FilingRef

SAMPLE_HTML = """
<html><body>
<table><tr><td>Table of Contents</td></tr></table>
<h1>Item 1. Business</h1>
<p>The Company designs, manufactures and markets smartphones and personal computers.
The Company's fiscal year is the 52- or 53-week period that ends in September.</p>
<p>Apple Inc. | 2024 Form 10-K | 7</p>
<h2>Segment Operating Performance</h2>
<table>
  <tr><th>Segment</th><th></th><th>2024</th><th></th><th>2023</th></tr>
  <tr><td>Americas</td><td>$</td><td>167,045</td><td>$</td><td>162,560</td></tr>
  <tr><td>Europe</td><td>$</td><td>101,328</td><td>$</td><td>94,294</td></tr>
</table>
<p>37 Parts I and II Item 1A. Risk Factors.</p>
<p>The Company's business is subject to a variety of risks including supply chain
concentration and cybersecurity threats. A breach could harm the Company.</p>
</body></html>
"""


@pytest.fixture
def ref():
    return FilingRef(
        cik="0000320193",
        company="AAPL",
        ticker="AAPL",
        form="10-K",
        fiscal_year=2024,
        filing_date="2024-11-01",
        accession="0000320193-24-000123",
        primary_document="aapl-20240928.htm",
    )


@pytest.fixture
def sample_html():
    return SAMPLE_HTML


@pytest.fixture
def filing(ref):
    """A small hand-built filing with known offsets."""
    text = (
        "Item 1. Business\n\n"
        "The Company designs and sells smartphones. Revenue grew in fiscal 2024.\n\n"
        "Item 1A. Risk Factors\n\n"
        "Supply chain concentration is a material risk to the Company."
    )
    blocks = []
    cursor = 0
    for raw, item, kind in [
        ("Item 1. Business", "Item 1", BlockType.HEADING),
        (
            "The Company designs and sells smartphones. Revenue grew in fiscal 2024.",
            "Item 1",
            BlockType.PARAGRAPH,
        ),
        ("Item 1A. Risk Factors", "Item 1A", BlockType.HEADING),
        (
            "Supply chain concentration is a material risk to the Company.",
            "Item 1A",
            BlockType.PARAGRAPH,
        ),
    ]:
        blocks.append(
            Block(block_type=kind, text=raw, start=cursor, end=cursor + len(raw), item=item)
        )
        cursor += len(raw) + 2
    return Filing(ref=ref, text=text, blocks=blocks)


@pytest.fixture
def chunks(filing):
    return [
        Chunk(
            chunk_id=f"{filing.doc_id}::{i:05d}",
            doc_id=filing.doc_id,
            text=block.text,
            start=block.start,
            end=block.end,
            item=block.item,
            company=filing.ref.ticker,
            fiscal_year=filing.ref.fiscal_year,
        )
        for i, block in enumerate(filing.blocks)
    ]
