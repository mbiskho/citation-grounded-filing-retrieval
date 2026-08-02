# Retrieval quality report

Questions: **37** (23 answerable, 14 unanswerable).

## Headline

- **recall@10** 0.826
- **MRR** 0.579
- **nDCG@10** 0.374
- **refusal recall** 0.643 (precision 1.000)
- **answer rate on answerable questions** 1.000
- **citation accuracy** 1.000 over 112 citations
- **answers containing an uncited claim** 0
- **latency** p50 15ms / p95 22ms

## Ablations

| arm | recall@10 | recall@5 | MRR | nDCG@10 | refusal recall | p95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25, no expansion | 0.783 | 0.652 | 0.454 | 0.259 | 0.643 | 23 ms |
| **BM25 + query expansion (shipped default on the offline stack)** | 0.826 | 0.696 | 0.579 | 0.374 | 0.643 | 22 ms |
| Dense only | 0.478 | 0.478 | 0.275 | 0.215 | 0.643 | 6 ms |
| Hybrid (RRF), no reranker | 0.739 | 0.696 | 0.495 | 0.370 | 0.643 | 25 ms |
| BM25 + reranker | 0.783 | 0.609 | 0.561 | 0.357 | 0.643 | 37 ms |
| Hybrid + reranker, no metadata filters | 0.696 | 0.522 | 0.432 | 0.268 | 0.643 | 78 ms |
| Full pipeline, no query expansion | 0.652 | 0.565 | 0.433 | 0.249 | 0.643 | 51 ms |
| Hybrid + reranker (default once a neural embedder is present) | 0.783 | 0.652 | 0.515 | 0.368 | 0.643 | 46 ms |

- **BM25, no expansion** — Keyword search alone, without the financial synonym rewrite. The floor.
- **BM25 + query expansion (shipped default on the offline stack)** — Keyword search with the financial tokeniser and synonym rewrite. The best-measured configuration when no neural model is installed, and therefore the default in that case.
- **Dense only** — Bi-encoder retrieval alone. No keyword matching, no reranker.
- **Hybrid (RRF), no reranker** — BM25 and dense fused by reciprocal rank fusion.
- **BM25 + reranker** — Isolates the reranker's contribution without the dense retriever.
- **Hybrid + reranker, no metadata filters** — Full pipeline with company/year filtering disabled.
- **Full pipeline, no query expansion** — Everything except the financial synonym rewrite.
- **Hybrid + reranker (default once a neural embedder is present)** — Everything on. Measured worse than BM25 alone with the offline embedder, which is why it is not the default in that case.

## Where it works and where it does not

| question type | n | recall@10 |
|---|---:|---:|
| numeric / table | 7 | 0.857 |
| prose / topic | 16 | 0.812 |

## Refusals by type

| verdict | count |
|---|---:|
| refused_out_of_corpus | 9 |

