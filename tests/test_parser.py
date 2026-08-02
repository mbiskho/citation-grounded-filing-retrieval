"""Parser tests.

Heavy on tables and item detection, because those are the two things that
silently produce a system which looks fine on prose and is wrong on numbers.
"""

from __future__ import annotations

import pytest

from filingqa.ingest.parser import (
    detect_item,
    is_layout_table,
    is_page_furniture,
    normalise_row,
    parse_filing,
    render_table,
    strip_running_header,
)

pytest.importorskip("bs4")


# -- inline XBRL cell repair ----------------------------------------------


def test_currency_symbol_merges_into_the_number():
    """Inline XBRL emits "$" and "2,755" as separate cells.

    Left split, a balance sheet row reads "$ 2,755 $ 3,089" and there is no
    way to tell which figure belongs to which year.
    """
    assert normalise_row(["Investment portfolio", "$", "2,755", "$", "3,089"]) == [
        "Investment portfolio",
        "$2,755",
        "$3,089",
    ]


def test_empty_cells_are_dropped():
    assert normalise_row(["Revenue", "", "  ", "100"]) == ["Revenue", "100"]


def test_trailing_percent_joins_its_value():
    assert normalise_row(["Growth", "12.5", "%"]) == ["Growth", "12.5%"]


def test_closing_paren_joins_its_value():
    assert normalise_row(["Loss", "(1,234", ")"]) == ["Loss", "(1,234)"]


# -- layout vs data tables -------------------------------------------------


def test_single_row_table_is_layout():
    assert is_layout_table([["Table of Contents"]])


def test_single_column_table_is_layout():
    assert is_layout_table([["a"], ["b"], ["c"]])


def test_empty_table_is_layout():
    assert is_layout_table([])


def test_real_data_table_is_kept():
    rows = [["Segment", "2024", "2023"], ["Americas", "$167,045", "$162,560"]]
    assert not is_layout_table(rows)


# -- table rendering -------------------------------------------------------


def test_every_row_carries_its_header():
    """A row retrieved without its header is numbers with no referent.

    The repetition is deliberate: chunks get retrieved individually, so each
    row has to stand alone.
    """
    rendered = render_table([["Segment", "2024", "2023"], ["Americas", "$167,045", "$162,560"]])
    assert "Americas: 2024 $167,045; 2023 $162,560" in rendered


def test_caption_is_included():
    rendered = render_table([["a", "b"], ["1", "2"]], caption="Segment Performance")
    assert rendered.startswith("Segment Performance")


def test_values_align_to_headers_from_the_right():
    """The label column often spans several header cells."""
    rendered = render_table([["Item", "FY24", "FY23"], ["Revenue", "100", "90"]])
    assert "FY24 100" in rendered and "FY23 90" in rendered


def test_column_headers_that_are_years():
    """Years are numeric, and that silently disabled headers everywhere.

    The most common header row in a filing is a list of fiscal years. A plain
    "is this cell numeric" test classified two of three cells as data, so the
    row was not recognised as a header, so the table rendered with no headers
    at all - and every row became a sequence of numbers with no referent,
    which is the exact failure the header-per-row design exists to prevent.
    """
    rendered = render_table([["Segment", "2024", "2023"], ["Americas", "$1", "$2"]])
    assert "2024 $1" in rendered
    assert "2023 $2" in rendered

    # A row of genuine data values must still not be mistaken for a header.
    data_only = render_table([["Americas", "167045", "162560"], ["Europe", "1", "2"]])
    assert "Americas" in data_only


# -- item detection --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Item 1. Business", "Item 1"),
        ("Item 1A. Risk Factors", "Item 1A"),
        ("ITEM 7. Management's Discussion and Analysis", "Item 7"),
        ("Item 7A. Quantitative and Qualitative Disclosures", "Item 7A"),
        ("Item 9A. Controls and Procedures", "Item 9A"),
    ],
)
def test_item_headings_are_detected(text, expected):
    assert detect_item(text) == expected


def test_item_1a_is_not_swallowed_by_item_1():
    """Longest match must win or every 1A/1B/1C section is labelled Item 1."""
    assert detect_item("Item 1A. Risk Factors") == "Item 1A"
    assert detect_item("Item 1C. Cybersecurity") == "Item 1C"


def test_running_header_is_stripped_before_matching():
    """JPMorgan glues the page number and part banner onto the heading."""
    assert strip_running_header("37 Parts I and II Item 1B.") == "Item 1B."
    assert detect_item("37 Parts I and II Item 1B. Unresolved Staff Comments.") == "Item 1B"


def test_table_of_contents_rows_are_not_item_starts():
    assert detect_item("Item 1A. Risk Factors ... Page 10-37") == ""
    assert detect_item("Item 1A.: Part I Risk Factors.; Page 10-37") == ""


def test_tables_never_start_an_item():
    """Every filer renders its ToC as a table and every row matches."""
    assert detect_item("Item 1. | Business | 1", is_table=True) == ""


def test_prose_mentioning_an_item_is_not_a_heading():
    assert detect_item("Item 7 discusses this in more detail below.") == ""


# -- page furniture --------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "42",
        "Apple Inc. | 2024 Form 10-K | 7",
        "Table of Contents",
        "Part II",
        "(continued)",
    ],
)
def test_page_furniture_is_recognised(text):
    assert is_page_furniture(text)


def test_real_prose_is_not_furniture():
    assert not is_page_furniture(
        "The Company designs, manufactures and markets smartphones and personal computers."
    )


# -- end to end ------------------------------------------------------------


def test_parse_assigns_items_and_keeps_offsets(sample_html, ref):
    filing = parse_filing(sample_html, ref)

    assert "Item 1" in filing.items
    assert "Item 1A" in filing.items
    # _assert_offsets runs inside parse_filing, but check explicitly too:
    for block in filing.blocks:
        assert filing.text[block.start : block.end] == block.text


def test_parse_extracts_tables_with_merged_currency(sample_html, ref):
    filing = parse_filing(sample_html, ref)
    tables = [b for b in filing.blocks if b.is_table]
    assert tables
    assert any("$167,045" in t.text for t in tables)


def test_page_footer_never_reaches_the_text(sample_html, ref):
    """If furniture is in the text it can be chunked, retrieved and quoted."""
    filing = parse_filing(sample_html, ref)
    assert "2024 Form 10-K | 7" not in filing.text


def test_checksum_detects_a_changed_rendering(filing):
    from filingqa.models import Filing

    payload = filing.to_dict()
    payload["text"] = payload["text"] + " tampered"
    with pytest.raises(ValueError, match="checksum"):
        Filing.from_dict(payload)
