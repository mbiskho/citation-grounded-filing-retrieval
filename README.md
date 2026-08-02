# Citation-Grounded Filing Retrieval

**Answers questions about 10-Ks and shows you the page.**

RAG over SEC filings where the citation is the product and the answer is just
how you get there. Every claim carries a character-level span into the filing,
every span is verified against the source before the answer is returned, and
the system refuses when the corpus does not cover the question.

An unsourced number is useless in a credit memo. So an answer that cannot be
sourced is not returned.

```bash
pip install -e ".[ingest]"
filingqa ingest          # 9 filings from EDGAR: Apple, JPMorgan, Ford x 3 years
filingqa ask "What supplier risks does Apple flag in FY2024?"
filingqa app             # Streamlit demo with clickable, highlighted citations
```

---

## Results

100-question gold set, **scored on a held-out test split of 63 questions**
(42 answerable, 21 not). Parameters and refusal thresholds were tuned on the
other 37. Full report: [`evaluation/results/report-test.md`](evaluation/results/report-test.md).

| metric | result | target | |
|---|---:|---:|---|
| **Citation accuracy** | **1.000** (197/197) | — | ✅ |
| **Answers with an uncited claim** | **0** | 0 | ✅ |
| **Refusal precision** | **1.000** | — | ✅ |
| **Answer rate on answerable questions** | **1.000** | — | ✅ |
| **p95 latency** | **28 ms** | — | ✅ |
| **Cost per query** | **$0.00** | — | ✅ |
| recall@10 | 0.643 | 0.85 | ❌ |
| refusal recall | 0.619 | 0.90 | ❌ |

**Two of the three targets I set were missed, and the misses are the
interesting part.** They are analysed below rather than smoothed over.

MRR 0.423 · nDCG@10 0.323 · recall@5 0.548.

### Where retrieval works and where it does not

| question type | n | recall@10 |
|---|---:|---:|
| prose / topic | 24 | 0.708 |
| cross-filing comparison | 7 | 0.571 |
| numeric / table | 11 | 0.545 |

Tables are the weak spot, exactly as expected, and that gap survived a week of
work on the parser.

---

## Read this before quoting the numbers

**The published numbers come from the dependency-free retrieval stack**, not
from a neural embedder. `sentence-transformers` and a BGE cross-encoder
reranker are implemented and wired in (`--embedder auto`), but the build
environment could not install torch, so I did not benchmark them and will not
report numbers I did not measure. Re-running on the neural stack is one
command:

```bash
pip install -e ".[embeddings]"
filingqa calibrate && filingqa evaluate --split test
```

I would expect a meaningful recall gain there — the failures below are mostly
paraphrase failures, which is precisely what a bi-encoder fixes. But expecting
is not measuring.

**The gold set is template-generated, not expert-annotated.** I am not a credit
analyst. What makes it usable is that the relevance labels come from
*structure* — a passage is relevant because it is in the right filing, the
right Item, and contains the topic's vocabulary — never from similarity to the
question. The system cannot be rewarded for agreeing with itself. The known
bias is lexical overlap between question and passage, which flatters BM25;
that is why the ablation table reports a BM25-only floor. Method in full:
[`docs/evaluation-methodology.md`](docs/evaluation-methodology.md).

---

## Why recall@10 is 0.643 and not 0.85

Three causes, in order of size.

**1. Vocabulary mismatch, partly fixed.** Analysts and filings use different
words for the same thing. Apple says "net sales"; nobody asks for "net sales".
A hand-built synonym layer that rewrites the query into filing vocabulary is
worth **+17 points** on its own — the single largest component in the ablation.
It does not close the gap because a hand-built table cannot anticipate every
paraphrase. This is the failure a neural embedder is supposed to absorb.

**2. Tables.** 0.545 against 0.708 on prose. Filings are mostly tables and the
retrieval problem there is genuinely different: the answer is a cell, the
context is a header two rows up, and lexical overlap between "what was total
revenue" and a rendered balance sheet is thin. The parser work — merging split
currency cells, keeping tables atomic, repeating headers on every row — moved
this a long way from where it started, and it is still the worst category.

**3. Small-sample variance.** The same configuration scores 0.826 on dev and
0.643 on test. With 42 answerable test questions, one question is 2.4 points.
The dev/test gap is a caution about how precisely any of these numbers should
be read, and it is the reason I am reporting the held-out figure rather than
the flattering one.

---

## Why refusal recall is 0.619, and why I think that is the real finding

Refusal breaks cleanly into three classes:

| unanswerable because… | refused | how |
|---|---:|---|
| the company is not in the corpus | **10/10 (100%)** | exact scope check |
| the fiscal year is not in the corpus | **12/12 (100%)** | exact scope check |
| a 10-K structurally does not contain it | **0/13 (0%)** | nothing works |

The third class is questions like *"What price target do analysts have for
Apple?"* or *"What was the share price on the last day of FY2024?"* — the
company is in the corpus, the year is in the corpus, the vocabulary is
financial. Retrieval returns ten confident-looking passages about Apple and
about pricing.

I tried to catch these with confidence thresholds and swept them properly on
dev. **The sweep chose to switch them off.** The reason is in the
distributions:

| question class | mean best-support score |
|---|---:|
| answerable | 0.720 |
| unanswerable by topic | 0.698 |

Those populations are not separated, so no threshold separates them. Every
setting that caught more bad questions refused more good ones, and refusal F1
peaked with both gates at zero.

**The conclusion I take from this is that calibrated refusal in RAG needs
structured knowledge of what the corpus contains, not a confidence score.**
The scope check is exact — a company or a fiscal year is a fact, not an
estimate — and it gets 100% on the two classes it covers and 0% on the one it
does not. Closing the third class needs a model of what a 10-K *is*, which is
a different piece of work than tuning a threshold.

The gates remain in the code, calibrated to off, with the sweep committed at
[`evaluation/results/refusal-calibration.json`](evaluation/results/refusal-calibration.json).
A cross-encoder produces far better-separated scores and might make them earn
their place.

---

## The citation guarantee

This is the part that works, and it is structural rather than aspirational.

A normalised plain-text rendering of each filing is the single source of
truth. Every block, chunk and citation refers to it by character offset. So a
citation is a `(doc_id, start, end)` triple that can be sliced out and
compared against what the answer claims it says. Verification is a string
comparison, not a judgement.

Every citation is checked three ways before an answer is returned:

1. the span resolves inside a filing that exists,
2. the quote matches the span (whitespace- and quote-normalised, with a token
   fallback for elided passages),
3. the span is neither degenerate nor a whole section.

Any failure downgrades the answer to a refusal. Any sentence making a claim
without a marker downgrades it too. **197 of 197 citations verified; 0 answers
contained an uncited claim, and no answerable question was refused.**

The default generator is extractive — it selects sentences from retrieved
spans and quotes them verbatim. In a consumer product that would be a
limitation. In a credit memo it is closer to the requirement: what gets pasted
into the document is the filing's own words with a pointer to the page, and a
fluent paraphrase is the part nobody can check. An `LLMGenerator` is included
for prose answers and runs through exactly the same verification.

---

## Ablations

Held-out test split. Each arm changes one thing.

| arm | recall@10 | recall@5 | MRR | nDCG@10 | p95 |
|---|---:|---:|---:|---:|---:|
| BM25, no expansion (floor) | 0.476 | 0.405 | 0.305 | 0.198 | 28 ms |
| **BM25 + query expansion** (shipped default) | **0.643** | **0.548** | **0.423** | **0.323** | 28 ms |
| Dense only (offline embedder) | 0.476 | 0.405 | 0.333 | 0.180 | 8 ms |
| Hybrid RRF, no reranker | 0.619 | 0.548 | 0.405 | 0.286 | 25 ms |
| BM25 + reranker | 0.619 | 0.571 | 0.404 | 0.298 | 62 ms |
| Hybrid + reranker, no metadata filters | 0.429 | 0.381 | 0.271 | 0.183 | 61 ms |
| Hybrid + reranker, no query expansion | 0.476 | 0.452 | 0.271 | 0.183 | 76 ms |
| Hybrid + reranker (neural-stack default) | 0.619 | 0.524 | 0.390 | 0.281 | 54 ms |

Three things worth saying out loud:

**Query expansion is the biggest single lever** (+17 points), larger than
fusion and reranking combined.

**With a weak embedder, adding dense retrieval makes things worse.** Dense
alone scores 0.476; fusing it into BM25 drops the result from 0.643 to 0.619.
Fusing a weak retriever into a strong one dilutes the strong one. So the
system ships with dense fusion *off* when the offline embedder is in use, and
flips it back on automatically when a real model is present — the default
follows the measurement rather than the architecture diagram.

**Metadata filters matter more than reranking.** Removing company/year
filtering costs 19 points; removing the reranker costs nothing measurable
here.

---

## How it works

```
EDGAR ──► parse ──► chunk ──► index ──► retrieve ──► generate ──► verify
          │          │                    │                        │
   table-aware   tables are          BM25 + expansion         every span
   offsets kept  never split         (+ dense, + rerank)      checked, or
                                                              it's a refusal
```

**Ingestion** pulls 10-Ks from EDGAR with the required User-Agent and a rate
limiter, and caches everything on disk. Parsing is where the time went: inline
XBRL splits `$2,755` into three cells, filings use `<table>` for page
furniture, and column headers live two rows from their values. Item detection
had to work on three filers that mark up section headings three different ways
— Apple as styled headings, Ford as bare paragraphs, JPMorgan as body text
with a page number glued to the front. Requiring a "heading" found 13 sections
in Apple, 7 in Ford and **0** in JPMorgan.

**Chunking** never splits a table. A balance sheet cut at 512 tokens gives you
one chunk of labels and one of numbers; both retrieve, neither answers, and
the numbers-only chunk is exactly the kind of context that produces a
confident wrong answer.

**Retrieval** is BM25 over a tokeniser that keeps money, percentages and
parenthesised negatives intact, plus optional dense retrieval fused by
reciprocal rank fusion, plus optional reranking, plus metadata filters that
only fire on unambiguous mentions.

**Answering** refuses in three independent places: before generation on scope,
during generation when nothing is worth quoting, and after generation when a
citation fails verification.

---

## Corpus

Nine 10-Ks — Apple, JPMorgan Chase and Ford, FY2023–FY2025 — chosen for parser
diversity rather than name recognition. A tech company, a bank and an
industrial have genuinely different filing structures, and JPMorgan's is
mostly tables. 5,845 chunks, ~7.6M characters.

Parsed filings are committed so the repo works offline; raw HTML is not.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the pieces fit and why
- [`docs/parsing-filings.md`](docs/parsing-filings.md) — the table problem in detail
- [`docs/evaluation-methodology.md`](docs/evaluation-methodology.md) — what the numbers do and do not show
- [`docs/ablation-study.md`](docs/ablation-study.md) — component-by-component analysis
- [`docs/refusal.md`](docs/refusal.md) — why confidence thresholds do not work

---

## Development

```bash
make install    # pip install -e ".[dev,ingest]"
make test       # 83 tests, no network
make lint type
make eval       # regenerate the gold set, calibrate on dev, score on test
```

CI runs the suite on 3.10/3.11/3.12 and re-runs the evaluation against the
committed corpus, failing the build if citation accuracy drops below 1.0 or an
uncited claim appears.

---

## What I would do next

- **Benchmark the neural stack.** The largest known unknown in this repo.
- **Close the third refusal class** with a taxonomy of what a 10-K contains,
  rather than another threshold.
- **Table-native retrieval** — index cells with their row and column headers
  as structured records instead of flattening to text.
- **Year-over-year diff mode** for risk factors, which the corpus already
  supports and the comparison questions already half-measure.

MIT licensed. The methodology is the part most worth arguing with, and the
whole evaluation re-runs in about a minute.
