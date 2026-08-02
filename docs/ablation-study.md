# Ablation study

Held-out test split: 63 questions, 42 answerable. Every arm changes exactly
one thing and sees the same questions in the same order. Regenerate with
`filingqa evaluate --split test`.

| arm | recall@10 | recall@5 | set-recall@10 | MRR | nDCG@10 | p95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25, no expansion | 0.476 | 0.405 | 0.188 | 0.305 | 0.198 | 28 ms |
| **BM25 + query expansion** | **0.643** | **0.548** | **0.356** | **0.423** | **0.323** | 28 ms |
| Dense only | 0.476 | 0.405 | 0.196 | 0.333 | 0.180 | 8 ms |
| Hybrid RRF, no reranker | 0.619 | 0.548 | 0.321 | 0.405 | 0.286 | 25 ms |
| BM25 + reranker | 0.619 | 0.571 | 0.318 | 0.404 | 0.298 | 62 ms |
| Hybrid + rerank, no filters | 0.429 | 0.381 | 0.220 | 0.271 | 0.183 | 61 ms |
| Hybrid + rerank, no expansion | 0.476 | 0.452 | 0.185 | 0.271 | 0.183 | 76 ms |
| Hybrid + reranker (all on) | 0.619 | 0.524 | 0.328 | 0.390 | 0.281 | 54 ms |

---

## 1. Query expansion: +17 points, the biggest lever

`0.476 → 0.643`. Larger than fusion and reranking put together.

The reason is domain-specific and not subtle. Analysts and filings use
different words for the same thing:

| the question says | Apple's 10-K says |
|---|---|
| revenue | net sales |
| top-line | net sales |
| R&D | research and development |
| headcount | full-time equivalent employees |
| buyback | repurchase of common stock |

No amount of reranking repairs a candidate set that never contained the right
passage. A hand-built synonym table that rewrites the query into filing
vocabulary before retrieval fixes the ones it knows about.

It is also the component I trust least to generalise. The table is hand-built,
so it is narrow and auditable but it will not anticipate every paraphrase, and
a pharma or energy corpus needs a different one. This is precisely the gap a
neural bi-encoder is supposed to absorb, which is the strongest argument for
benchmarking the neural stack.

The expanded query goes only to the retrievers. Filters and reranking see the
original — expansion adds vocabulary that would confuse entity detection and
dilute the rerank signal.

## 2. Dense retrieval: negative, with a weak embedder

Dense alone: **0.476**. BM25 alone: **0.643**. Fused: **0.619**.

Adding the dense retriever to BM25 made the result *worse*. That is not a bug
in the fusion — it is what fusion does. RRF averages rank positions, so
blending a retriever that is 17 points weaker pulls good candidates down.

The honest conclusion for this configuration: **with the offline hashing
embedder, the vector index is not worth having.** The system ships with dense
fusion off in that case and turns it back on automatically once a real
embedding model is present.

Two caveats worth stating. The hashing embedder is a lexical signature
projected into a dense vector — it captures surface overlap and morphology and
nothing about meaning, so it cannot connect "revenue" to "net sales", which is
the single most common paraphrase here. A real bi-encoder plausibly reverses
this entire row. And the gold set's lexical bias flatters BM25 specifically,
so the true gap is probably narrower than 17 points.

## 3. Reranking: no measurable benefit here

`hybrid-no-rerank` 0.619 vs `full-hybrid` 0.619 at recall@10. The reranker
moves recall@5 slightly (0.548 → 0.571 on the BM25 arm) and costs 2–3× the
latency.

That is a fair result for the *lexical* reranker, which is a fallback, not a
cross-encoder. It scores IDF-weighted query coverage and blends in the
retrieval rank — it cannot read the passage, so there is a hard ceiling on how
much it can reorder correctly.

An earlier version scored raw term overlap and ranked **below no reranking at
all** (0.600 vs 0.646 on an earlier corpus build). Two fixes made it
break-even: weighting by IDF so "the company" stops counting as much as
"cybersecurity", and keeping the retrieval rank as a decaying prior instead of
discarding the first stage entirely. A cross-encoder can afford to throw away
the first-stage ranking; a lexical scorer cannot.

The cross-encoder path (`BAAI/bge-reranker-base`) is implemented and
unmeasured.

## 4. Metadata filters: −19 points when removed

`0.619 → 0.429`. The largest single drop of any removal.

Filtering to the company and fiscal year the question names removes eight of
nine filings from contention. On a corpus with three companies × three years
of *highly repetitive* boilerplate — the same risk factor appears nine times
with minor edits — that is most of the retrieval problem solved before ranking
starts.

Filters are also the riskiest component, because a wrong filter is an
unrecoverable miss: the answer is gone before ranking. Three rules keep them
safe:

- Only unambiguous mentions fire a filter.
- Two companies named means a comparison question, so no company filter.
- A filter that empties the candidate set is dropped and recorded as
  `applied: false`.

## 5. Set-recall tells a harsher story

Set-recall@10 — the fraction of *all* gold passages retrieved — is 0.356 for
the shipped configuration against 0.643 hit-rate recall.

The system reliably finds *a* relevant passage and reliably misses most of the
others. For single-fact questions that is fine, and it is what the product
needs. For "summarise everything this filing says about liquidity" it is not,
and the comparison questions (0.571) are where it starts to show.

Reporting only the hit-rate definition would have hidden this, which is why
both are in the results file.

## What this adds up to

For this corpus, on this stack, the ranking of what mattered was:

1. **metadata filters** (−19 when removed)
2. **query expansion** (+17 when added)
3. **the reranker** (≈0)
4. **dense retrieval** (−2, actively harmful)

That ordering is close to the inverse of where the effort usually goes in a
RAG project. Two of the four are domain knowledge — knowing that filings are
partitioned by company and period, and knowing that "revenue" means "net
sales" — rather than model choice.

The result to be most careful with is the dense row. It says the *offline
fallback* embedder is not worth fusing. It does not say vector search is
useless on filings, and I would not quote it as though it did.
