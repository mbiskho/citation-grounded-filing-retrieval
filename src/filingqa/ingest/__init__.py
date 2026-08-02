"""Filing ingestion: fetch from EDGAR, parse to offset-carrying blocks."""

from __future__ import annotations

import json
from pathlib import Path

from ..models import Filing, FilingRef
from .edgar import DEFAULT_COMPANIES, Company, EdgarClient, default_corpus
from .parser import parse_filing

__all__ = [
    "EdgarClient",
    "Company",
    "DEFAULT_COMPANIES",
    "default_corpus",
    "parse_filing",
    "save_filing",
    "load_filing",
    "load_corpus",
]


def save_filing(filing: Filing, parsed_dir: Path | str = "data/parsed") -> Path:
    path = Path(parsed_dir) / f"{filing.doc_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(filing.to_dict(), ensure_ascii=False))
    return path


def load_filing(doc_id: str, parsed_dir: Path | str = "data/parsed") -> Filing:
    path = Path(parsed_dir) / f"{doc_id}.json"
    return Filing.from_dict(json.loads(path.read_text()))


def load_corpus(parsed_dir: Path | str = "data/parsed") -> list[Filing]:
    directory = Path(parsed_dir)
    if not directory.exists():
        raise FileNotFoundError(f"no parsed filings in {directory}. Run `filingqa ingest` first.")
    return [Filing.from_dict(json.loads(p.read_text())) for p in sorted(directory.glob("*.json"))]


def ingest(
    refs: list[FilingRef],
    client: EdgarClient,
    *,
    parsed_dir: Path | str = "data/parsed",
    verbose: bool = True,
) -> list[Filing]:
    filings = []
    for ref in refs:
        html = client.fetch_document(ref)
        filing = parse_filing(html, ref)
        save_filing(filing, parsed_dir)
        filings.append(filing)
        if verbose:
            tables = sum(1 for b in filing.blocks if b.is_table)
            print(
                f"  {filing.doc_id:<16} {len(filing.text):>8,} chars  "
                f"{len(filing.blocks):>5} blocks  {tables:>4} tables  "
                f"items: {len(filing.items)}"
            )
    return filings
