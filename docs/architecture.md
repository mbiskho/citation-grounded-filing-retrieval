# Architecture

```
                   EDGAR
                     │  cached HTML, rate-limited, User-Agent required
                     ▼
              ingest/parser.py ──────► Filing
                     │                  ├─ text   (the citation substrate)
                     │                  └─ blocks (offsets into text)
                     ▼
                chunking/  ───────────► Chunk[]  (span + provenance)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   BM25Index                 DenseIndex
   (financial tokeniser)     (pluggable embedder)
        └────────────┬────────────┘
                     ▼
              reciprocal rank fusion
                     ▼
                  reranker
                     ▼
              answer/  ──► generate ──► verify ──► Answer | Refusal
```

## The one decision everything follows from

**A normalised plain-text rendering of each filing is the single source of
truth, and every other object refers to it by character offset.**

That makes a citation a `(doc_id, start, end)` triple which can be sliced out
of the source and compared against what the answer claims it says.
Verification becomes a string comparison rather than a judgement call — which
is the difference between "zero uncited claims" being a measurement and being
a slogan.

Two invariants protect it:

- `Filing.text` is written once at parse time and never mutated. Offsets are
  assigned while writing, never located afterwards with `str.find`, because
  filings repeat boilerplate and `find` would attach a block to the wrong
  occurrence.
- `Filing.checksum` is compared on load. A stored citation is an offset into
  one specific rendering; if the parser changes, comparing checksums turns a
  silent misalignment into a loud failure.

## Components

| module | responsibility |
|---|---|
| `ingest/edgar.py` | EDGAR client: rate limiting, User-Agent, disk cache, shard paging |
| `ingest/parser.py` | HTML → rendered text + offset-carrying blocks; table repair; item detection |
| `chunking/` | token-budgeted prose chunks, atomic tables, spans preserved |
| `retrieval/tokenize.py` | financial tokeniser — money, percentages, parenthesised negatives |
| `retrieval/expansion.py` | query rewriting into filing vocabulary |
| `retrieval/indexes.py` | BM25, dense, reciprocal rank fusion |
| `retrieval/rerank.py` | cross-encoder, lexical fallback, no-op |
| `answer/scope.py` | corpus coverage checks — the refusal mechanism that works |
| `answer/generators.py` | extractive (default) and LLM generation |
| `answer/citations.py` | span verification and answer downgrade |
| `eval/` | gold set, metrics, ablation runner, dev-split tuning |

## Design notes

### Why the pipeline is switchable at every stage

Each retrieval component can be turned off independently. That exists for the
ablation study: the only way to say what a component is worth is to remove
exactly one thing and re-measure. It paid for itself — the ablation is what
revealed that dense fusion with a weak embedder is actively harmful, which is
now the shipped default's reason for being what it is.

### Why the default configuration is chosen by measurement

`HybridRetriever` disables dense fusion when the embedder is the offline
hashing fallback, because that configuration measured 0.643 against 0.619 for
the hybrid. An explicitly-constructed `RetrievalConfig` always wins; the rule
only applies when the caller expressed no preference, and it reverses
automatically once a real embedding model is present.

Shipping the more sophisticated default when the simpler one measured better
would be choosing the architecture diagram over the evidence.

### Why the dense index is flat

~6,000 chunks. A brute-force matmul answers in under a millisecond. An
approximate index would add a dependency, a build step and a recall/latency
knob to tune, all to optimise something that is not the bottleneck. At roughly
100k chunks that stops being true and HNSW or pgvector becomes worth the
complexity — the `DenseIndex` interface is small enough to swap.

### Why fusion is RRF rather than weighted scores

Score blending requires the two scorers to be on comparable scales. BM25 is
unbounded and corpus-dependent; cosine similarity sits in a narrow band near
the top. Normalising either introduces a constant that has to be re-fit
whenever the corpus changes. RRF uses only rank, has one insensitive
parameter, and does not drift.

### Why the default generator is extractive

It selects sentences from retrieved spans and quotes them verbatim. In a
consumer product that is a limitation; in a credit memo it is closer to the
requirement, because what gets pasted into the document is the filing's own
words and a fluent paraphrase is the part nobody can check.

It also makes the headline claim structural: if every sentence *is* a quoted
span, zero uncited claims cannot fail for that path. The LLM path can fail it,
which is why the same verification runs over both and why the LLM path is not
the default.

### Why the embedder interface is tiny

`encode_documents`, `encode_query`, `name`, `dim`. Anything richer and
swapping embedders stops being a one-line change — which is exactly what the
ablation study needs it to be.

## Extension points

| I want to… | Do this |
|---|---|
| use a real embedding model | `pip install -e ".[embeddings]"`; `--embedder auto` picks it up |
| add a company | append to `DEFAULT_COMPANIES`, re-run `filingqa ingest` |
| support 10-Qs or 8-Ks | `list_filings(form="10-Q")`; item patterns need extending |
| swap the vector store | implement `build`/`search`/`save`/`load` like `DenseIndex` |
| add domain synonyms | extend `SYNONYMS` in `retrieval/expansion.py` |
| change the quality bar | subclass `Generator`, keep `verify_answer` |

## What is deliberately absent

- **An agent loop.** Multi-hop retrieval over filings is a real need and a
  different project. Every hop multiplies the citation surface, and the
  citation guarantee is the thing worth protecting.
- **Answer caching.** The corpus is static, so this would be easy and would
  hide latency regressions during development.
- **A vector database service.** Nine filings fit in memory. Introducing
  Postgres + pgvector here would be infrastructure for its own sake; the
  interface is ready for it when the corpus is not.
