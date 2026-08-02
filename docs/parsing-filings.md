# Parsing filings

The project brief said table parsing would take a week I had not planned for.
It did. This is what was actually in that week, because none of it is visible
in the finished pipeline and all of it changes the numbers.

## The core problem

Run `BeautifulSoup(html).get_text()` on a 10-K and you get a stream of
disconnected numbers where the financial statements were. Retrieval then looks
fine on prose — risk factors, MD&A narrative — and falls apart on anything
numeric, which is most of what anyone wants from a filing.

Worse, it fails *quietly*. A numbers-only chunk still retrieves, and a
generator will happily quote it as though it referred to whatever was asked.

## Five things that break, in the order I hit them

### 1. Inline XBRL splits every cell

`$2,755` arrives as three sibling `<td>`s: `"$"`, `"2,755"`, and usually an
empty one. Flatten naively and a row reads:

```
Investment portfolio $ 2,755 $ 3,089
```

There is no way to tell which figure belongs to which year. `normalise_row`
merges a lone currency symbol into the number that follows it, joins trailing
`%` and `)` onto their values, and drops empties:

```python
normalise_row(["Investment portfolio", "$", "2,755", "$", "3,089"])
# ["Investment portfolio", "$2,755", "$3,089"]
```

### 2. Layout tables are not data tables

Filings use `<table>` for page headers, spacers and rule lines. Indexing them
floods the corpus with chunks that match numeric queries and say nothing.
`is_layout_table` rejects tables with fewer than two rows, fewer than two
columns, or no numeric content and almost no text.

### 3. Column headers are numeric

This one cost the most and was the least obvious. The most common header row
in a filing is a list of fiscal years:

```
Segment | 2024 | 2023
```

A plain "is this cell numeric" test classifies two of those three cells as
data, so the row is not recognised as a header, so the table renders with no
headers at all — and the entire point of repeating headers on every row is
lost. Every row becomes numbers with no referent.

The fix is one predicate: a cell is *data* if it is numeric **and not a
four-digit year**. Caught by a test that asserts the rendered output contains
`Americas: 2024 $167,045; 2023 $162,560`, which it did not for most of the
build.

### 4. Every filer marks up headings differently

Item detection is how a citation gets to say "Item 1A: Risk Factors" instead
of "somewhere in a 200-page document". Three filers, three conventions:

| filer | how `Item 1A` appears |
|---|---|
| Apple | styled heading element, no trailing period |
| Ford | bare paragraph, indistinguishable from body text |
| JPMorgan | body text with page number and part banner glued on: `37 Parts I and II Item 1A. Risk Factors.` |

The first version only looked at things that "looked like headings" — short,
title-cased, no trailing full stop. Results: **13 sections in Apple, 7 in
Ford, 0 in JPMorgan.**

Zero is the kind of number that looks like an empty company rather than a
broken parser, which is exactly why the corpus has three filers from three
sectors in it. One filer and this ships broken.

The working version runs on every block, strips running headers and page
numbers first, and validates the remainder against the known item title.

### 5. The table of contents matches everything

Every ToC row is `Item 1A. Risk Factors ... 5`, which matches any item
pattern perfectly. An earlier version tried to detect "am I inside the ToC"
by counting how many item headings appeared nearby — fragile and filer
specific.

The actual answer is one line: **tables never start a section.** Every filer
renders its ToC as a table. The heuristic went away entirely.

## Page furniture

Running headers and footers — `Apple Inc. | 2024 Form 10-K | 7`, bare page
numbers, `Part II`, `(continued)` — have no distinguishing markup and arrive
looking like body prose.

Left in, they get chunked, retrieved, and *quoted*, so an answer ends up
citing a page footer as though it said something. This actually happened
during the build; the fix is to drop them before any block is built, so they
occupy no span and cannot be cited.

## Offsets are the contract

Everything downstream — chunks, citations, the highlighted span in the demo —
is a character offset into the rendered text. Two invariants protect that:

**Offsets are assigned while writing the text, never searched for
afterwards.** `str.find` would silently attach a block to the wrong occurrence
of a repeated string, and filings repeat boilerplate constantly.

**Every block is asserted to slice back to its own text** at parse time, and
every chunk at index time. A citation that is off by a few characters is
invisible until an analyst clicks one.

There is also a checksum on the rendered text. A stored citation is an offset
into one specific rendering; if the parser changes and the text shifts by a
character, every stored citation quietly points somewhere slightly wrong.
Comparing checksums turns that into a loud failure at load time.

## What is still wrong

- **Multi-level column headers** — a table with a spanning header above
  per-year columns loses the top level.
- **Footnote markers** attach to values (`1,234(a)`), which the number
  normaliser mostly strips but not always.
- **Statement-level structure is lost.** A balance sheet becomes rows of text
  rather than a typed record with a period and a unit. This is the main reason
  numeric recall (0.545) trails prose (0.708), and fixing it properly means
  indexing cells as structured records rather than flattening to strings.
