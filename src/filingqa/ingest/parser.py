"""Table-aware filing parser.

This is the part of the project that took the longest, and the reason is worth
stating plainly: **financial filings are mostly tables, and naive HTML-to-text
shreds them.**

Run `soup.get_text()` on a 10-K and a balance sheet becomes a stream of
disconnected numbers. Retrieval then looks fine on prose - risk factors,
MD&A narrative - and falls apart on anything numeric, which is most of what
anyone actually wants from a filing.

Three specific problems, all of which produce plausible-looking garbage:

1. **Inline XBRL splits every cell.** "$2,755" arrives as three sibling cells:
   "$", "2,755", and often an empty one. Flattening naively yields
   "$ 2,755 $ 3,089" with no way to tell which year is which.
2. **Layout tables are not data tables.** Filings use `<table>` for page
   headers, spacers and section dividers. Treating those as data floods the
   index with junk that matches numeric queries.
3. **Column headers live in a different row from their values**, sometimes two
   rows up, sometimes in a colspan. A row extracted without its header is a
   sequence of numbers with no referent, which is exactly the kind of chunk
   that produces a confident wrong answer.

The output is a normalised plain-text rendering plus blocks carrying character
offsets into it. That rendering is the citation substrate: everything
downstream points into it, so it must be built once and never mutated.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

from ..models import Block, BlockType, Filing, FilingRef

try:  # pragma: no cover - import guard
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "beautifulsoup4 is required for parsing; pip install filingqa[ingest]"
    ) from exc


# --------------------------------------------------------------------------
# Item section detection
# --------------------------------------------------------------------------

#: 10-K items we care about. Others exist; these are the ones that carry
#: content analysts ask questions about.
ITEM_PATTERNS = [
    (r"item\s+1a\b", "Item 1A"),  # Risk Factors - before Item 1, longest match first
    (r"item\s+1b\b", "Item 1B"),
    (r"item\s+1c\b", "Item 1C"),  # Cybersecurity, added FY2023
    (r"item\s+1\b", "Item 1"),  # Business
    (r"item\s+2\b", "Item 2"),  # Properties
    (r"item\s+3\b", "Item 3"),  # Legal Proceedings
    (r"item\s+5\b", "Item 5"),  # Market for Registrant's Common Equity
    (r"item\s+7a\b", "Item 7A"),  # Quantitative and Qualitative Disclosures
    (r"item\s+7\b", "Item 7"),  # MD&A
    (r"item\s+8\b", "Item 8"),  # Financial Statements
    (r"item\s+9a\b", "Item 9A"),  # Controls and Procedures
    (r"item\s+10\b", "Item 10"),
    (r"item\s+11\b", "Item 11"),
    (r"item\s+12\b", "Item 12"),
]
_ITEM_RE = [(re.compile(p, re.IGNORECASE), label) for p, label in ITEM_PATTERNS]

ITEM_TITLES = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 1B": "Unresolved Staff Comments",
    "Item 1C": "Cybersecurity",
    "Item 2": "Properties",
    "Item 3": "Legal Proceedings",
    "Item 5": "Market for Common Equity",
    "Item 7": "Management's Discussion and Analysis",
    "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "Item 8": "Financial Statements and Supplementary Data",
    "Item 9A": "Controls and Procedures",
    "Item 10": "Directors and Executive Officers",
    "Item 11": "Executive Compensation",
    "Item 12": "Security Ownership",
}

#: A heading is short. The table of contents also matches item patterns, so
#: length is what separates "Item 1A. Risk Factors" as a section start from a
#: sentence that happens to mention Item 1A.
MAX_HEADING_CHARS = 120

_WS = re.compile(r"[ \t   ]+")
_MULTINEWLINE = re.compile(r"\n{3,}")
_CURRENCY_ONLY = re.compile(r"^[$€£¥]$")
_NUMERIC = re.compile(r"^\(?[-+]?[\d,]+\.?\d*\)?%?$")


def _clean(text: str) -> str:
    return _WS.sub(" ", text.replace("­", "")).strip()


#: Page headers and footers. Every filer stamps these on each page and the
#: HTML gives them no distinguishing markup, so they arrive looking exactly
#: like body prose. Left in, they get chunked, retrieved and - worst of all -
#: quoted, so an answer ends up citing "Apple Inc. | 2024 Form 10-K | 7" as
#: though it said something.
_PAGE_FURNITURE = [
    re.compile(r"^\s*\d{1,4}\s*$"),
    re.compile(r"^.{0,60}\|\s*\d{4}\s*Form\s+10-K\s*\|\s*\d{1,4}\s*$", re.IGNORECASE),
    re.compile(r"^.{0,40}\s+20\d{2}\s+Form\s+10-K\s+\d{1,4}$", re.IGNORECASE),
    re.compile(r"^\s*(?:table of contents|index)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Part\s+[IVX]+(?:\s*(?:,|and)\s*[IVX]+)*\s*$", re.IGNORECASE),
    re.compile(r"^\s*\(?continued\)?\s*$", re.IGNORECASE),
    re.compile(
        r"^\s*(?:JPMorgan\s+Chase\s*&\s*Co\.?|Apple\s+Inc\.?|Ford\s+Motor\s+Company)\s*$",
        re.IGNORECASE,
    ),
]


def is_page_furniture(text: str) -> bool:
    """Whether a block is a running header, footer or page number."""
    candidate = text.strip()
    if len(candidate) > 90:
        return False
    return any(pattern.match(candidate) for pattern in _PAGE_FURNITURE)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def normalise_row(cells: list[str]) -> list[str]:
    """Repair a row of inline-XBRL cells.

    Merges a lone currency symbol into the number that follows it, joins a
    trailing "%" or ")" onto its value, and drops empties. Without this, a
    balance sheet row reads "$ 2,755 $ 3,089" and there is no way to associate
    a figure with its column.
    """
    cleaned = [_clean(c) for c in cells]
    out: list[str] = []
    i = 0
    while i < len(cleaned):
        cell = cleaned[i]
        if not cell:
            i += 1
            continue
        if _CURRENCY_ONLY.match(cell) and i + 1 < len(cleaned):
            nxt = cleaned[i + 1]
            if nxt and _NUMERIC.match(nxt):
                out.append(f"{cell}{nxt}")
                i += 2
                continue
        if cell == "%" and out:
            out[-1] = f"{out[-1]}%"
            i += 1
            continue
        if cell in (")", ")%") and out:
            out[-1] = f"{out[-1]}{cell}"
            i += 1
            continue
        out.append(cell)
        i += 1
    return out


def is_layout_table(rows: list[list[str]]) -> bool:
    """Whether a `<table>` is page furniture rather than data.

    Filings use tables for page headers, spacers and rule lines. Indexing them
    fills the corpus with chunks that match numeric queries and say nothing.
    """
    if not rows:
        return True
    non_empty = [r for r in rows if r]
    if not non_empty:
        return True
    # A single row, or a single column, is almost never a real data table.
    if len(non_empty) < 2:
        return True
    if max(len(r) for r in non_empty) < 2:
        return True
    # Mostly empty cells with no numbers: a spacer.
    cells = [c for r in non_empty for c in r]
    if not cells:
        return True
    numeric = sum(1 for c in cells if _NUMERIC.match(c) or _CURRENCY_ONLY.match(c[:1]))
    # No numbers and almost no text: a spacer or a rule line.
    return numeric == 0 and sum(len(c) for c in cells) < 60


def render_table(rows: list[list[str]], caption: str = "") -> str:
    """Flatten a table to text that survives both BM25 and an embedder.

    Each row is emitted with its header prefix, so a row retrieved in isolation
    still says what its numbers refer to. This is deliberately redundant - the
    same header text repeats on every row - because a chunk that is retrieved
    without its header is a chunk that produces a confidently wrong answer.
    """
    if not rows:
        return ""
    header, body = _split_header(rows)
    lines: list[str] = []
    if caption:
        lines.append(caption)
    if header:
        lines.append(" | ".join(header))
    for row in body:
        if not row:
            continue
        if header and len(row) > 1:
            label, *values = row
            paired = []
            # Align values to headers from the right: the label column often
            # spans several header cells, so left-alignment mismatches.
            offset = len(header) - len(values)
            for i, value in enumerate(values):
                head = header[offset + i] if 0 <= offset + i < len(header) else ""
                paired.append(f"{head} {value}".strip() if head else value)
            lines.append(f"{label}: " + "; ".join(paired))
        else:
            lines.append(" | ".join(row))
    return "\n".join(lines)


_YEAR_CELL = re.compile(r"^(?:FY\s?)?(?:19|20)\d{2}$", re.IGNORECASE)


def _is_data_number(cell: str) -> bool:
    """Numeric, and not a year used as a column label.

    This distinction decides whether tables get headers at all. The most
    common header row in a filing is a list of fiscal years - "Segment | 2024
    | 2023" - and a plain numeric test classifies two of those three cells as
    data, so the row is treated as data and the table renders with no headers
    at all. Every row then becomes a sequence of numbers with no referent,
    which is exactly the failure the header-per-row design exists to prevent.
    """
    return bool(_NUMERIC.match(cell)) and not _YEAR_CELL.match(cell)


def _split_header(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """First row that looks like headers, and everything after it."""
    for i, row in enumerate(rows[:3]):
        if not row:
            continue
        if len(row) >= 2 and not any(_is_data_number(c) for c in row):
            return row, rows[i + 1 :]
    return [], rows


def extract_rows(table: Any) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        normalised = normalise_row(cells)
        if normalised:
            rows.append(normalised)
    return rows


# --------------------------------------------------------------------------
# Document parsing
# --------------------------------------------------------------------------


def _looks_like_heading(text: str, tag_name: str) -> bool:
    if len(text) > MAX_HEADING_CHARS:
        return False
    if tag_name in ("h1", "h2", "h3", "h4"):
        return True
    if not text or text[-1] in ".,;:":
        return False
    words = text.split()
    if len(words) > 14:
        return False
    # ALL CAPS or Title Case short lines are headings in filings.
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
        return True
    return bool(re.match(r"^(item|part)\s", text, re.IGNORECASE))


#: Running headers and page numbers that sit in front of a section heading.
#: JPMorgan renders "Item 1B" as "37 Parts I and II Item 1B. Unresolved Staff
#: Comments." - the page number and the part banner are part of the same text
#: node, so the heading does not start with "Item" at all.
_RUNNING_HEADER = re.compile(
    r"^(?:\s*\d{1,4}\s+|\s*Parts?\s+[IVX]+(?:\s*(?:,|and)\s*[IVX]+)*\s+|\s*JPMorgan\s+Chase\s*&\s*Co\.?\s*)+",
    re.IGNORECASE,
)

_ITEM_START = re.compile(r"^item\s+(\d{1,2}[a-c]?)\s*[.:—-]?\s*(.*)$", re.IGNORECASE | re.DOTALL)

#: A table-of-contents row pairs a title with a page number.
_TOC_ROW = re.compile(r"\bpage\b\s*[:\d]|\.{4,}|\|\s*\d+\s*$", re.IGNORECASE)


def strip_running_header(text: str) -> str:
    """Remove page numbers and part banners from the front of a heading."""
    return _RUNNING_HEADER.sub("", text, count=1).strip()


def detect_item(text: str, *, is_table: bool = False) -> str:
    """Which 10-K item this block starts, or empty if it starts none.

    Runs on *any* block, not just ones that look like headings. Three filers
    into this corpus it became clear there is no reliable visual signature:
    Apple emits "Item 1A. Risk Factors" as a styled heading, Ford as a bare
    paragraph, JPMorgan as body text with a page number glued to the front.
    Requiring a heading found 13 sections in Apple, 7 in Ford and 0 in
    JPMorgan - and zero is the kind of number that looks like an empty
    company rather than a broken parser.

    Tables are excluded outright, which is what handles the table of contents:
    every filer renders its ToC as a table, and its rows match item patterns
    perfectly.
    """
    if is_table:
        return ""

    candidate = strip_running_header(text)
    match = _ITEM_START.match(candidate)
    if not match:
        return ""

    number, remainder = match.group(1).lower(), match.group(2).strip()

    # A ToC row that escaped the table check: "Item 1A. Risk Factors ... 10-37"
    if _TOC_ROW.search(remainder[:200]):
        return ""

    label = next(
        (lab for pattern, lab in _ITEM_RE if pattern.match(f"item {number}")),
        "",
    )
    if not label:
        return ""

    # The heading may run straight into the section body, so only the opening
    # of the remainder is checked. Requiring the whole block to be short would
    # miss every filer that does not put the heading in its own element.
    opening = remainder[:MAX_HEADING_CHARS]
    if not opening:
        return label
    expected = ITEM_TITLES.get(label, "").lower()
    if expected and opening.lower().startswith(expected[:12]):
        return label
    # Otherwise accept it only if the remainder opens like a title rather than
    # mid-sentence - guards against prose that mentions "Item 7" in passing.
    return label if re.match(r"^[A-Z“(]", opening) else ""


def parse_filing(html: str, ref: FilingRef) -> Filing:
    """HTML to a `Filing` with a rendered text body and offset-carrying blocks."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    body = soup.body or soup
    raw_blocks: list[tuple[BlockType, str, list[list[str]], str]] = []
    last_heading = ""

    for element in body.find_all(["p", "div", "span", "td", "table", "h1", "h2", "h3", "h4"]):
        if element.name == "table":
            rows = extract_rows(element)
            if is_layout_table(rows):
                continue
            rendered = render_table(rows, caption=last_heading)
            if rendered.strip():
                raw_blocks.append((BlockType.TABLE, rendered, rows, last_heading))
            element.decompose()  # consumed; do not re-emit its cells as prose
            continue

        # Only take elements whose text is their own, not inherited from
        # children, or every paragraph is emitted once per ancestor.
        own = "".join(c for c in element.strings if c.parent is element)
        text = _clean(own)
        if not text or len(text) < 2:
            continue

        # Dropped before any block is built, so page furniture occupies no
        # span and can never be cited.
        if is_page_furniture(text):
            continue

        if _looks_like_heading(text, element.name):
            raw_blocks.append((BlockType.HEADING, text, [], ""))
            last_heading = text
        else:
            raw_blocks.append((BlockType.PARAGRAPH, text, [], ""))

    return _assemble(raw_blocks, ref)


def _assemble(
    raw_blocks: list[tuple[BlockType, str, list[list[str]], str]], ref: FilingRef
) -> Filing:
    """Build the rendered text and assign every block its span.

    Offsets are computed while writing the text, not searched for afterwards.
    Searching would silently attach a block to the wrong occurrence of a
    repeated string, and filings repeat strings constantly.
    """
    parts: list[str] = []
    blocks: list[Block] = []
    cursor = 0
    current_item = ""

    merged = _merge_adjacent_paragraphs(raw_blocks)

    for block_type, text, rows, caption in merged:
        item = detect_item(text, is_table=block_type is BlockType.TABLE)
        if item:
            current_item = item

        chunk_text = text
        start = cursor
        end = start + len(chunk_text)
        parts.append(chunk_text)
        cursor = end + 2  # the "\n\n" join below

        blocks.append(
            Block(
                block_type=block_type,
                text=chunk_text,
                start=start,
                end=end,
                item=current_item,
                rows=rows,
                caption=caption,
            )
        )

    text = "\n\n".join(parts)
    text = _MULTINEWLINE.sub("\n\n", text)

    filing = Filing(ref=ref, text=text, blocks=blocks)
    _assert_offsets(filing)
    return filing


def _merge_adjacent_paragraphs(
    raw: list[tuple[BlockType, str, list[list[str]], str]],
) -> list[tuple[BlockType, str, list[list[str]], str]]:
    """Join fragments that HTML split but a reader would read as one sentence.

    Filings wrap sentences across `<span>` boundaries for formatting, which
    produces two-word "paragraphs". Left alone these become useless chunks and
    they corrupt sentence-level citation spans.
    """
    out: list[tuple[BlockType, str, list[list[str]], str]] = []
    for block in raw:
        block_type, text, rows, caption = block
        if (
            out
            and block_type is BlockType.PARAGRAPH
            and out[-1][0] is BlockType.PARAGRAPH
            and not out[-1][1].endswith((".", ":", ";", "?", "!"))
            and len(out[-1][1]) < 400
        ):
            prev = out[-1]
            out[-1] = (prev[0], f"{prev[1]} {text}".strip(), prev[2], prev[3])
        else:
            out.append(block)
    return out


def _assert_offsets(filing: Filing) -> None:
    """Every block must slice back to its own text.

    This is cheap and it is the difference between citations that work and
    citations that are off by a few characters in a way nobody notices until
    an analyst clicks one.
    """
    for block in filing.blocks:
        actual = filing.text[block.start : block.end]
        if actual != block.text:
            raise AssertionError(
                f"{filing.doc_id}: block offset mismatch at {block.start}:{block.end}\n"
                f"  expected: {block.text[:80]!r}\n"
                f"  actual:   {actual[:80]!r}"
            )
