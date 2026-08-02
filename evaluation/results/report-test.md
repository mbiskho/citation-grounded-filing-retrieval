# Retrieval quality report

Questions: **63** (42 answerable, 21 unanswerable).

## Headline

- **recall@10** 0.643
- **MRR** 0.423
- **nDCG@10** 0.323
- **refusal recall** 0.619 (precision 1.000)
- **answer rate on answerable questions** 1.000
- **citation accuracy** 1.000 over 197 citations
- **answers containing an uncited claim** 0
- **latency** p50 17ms / p95 29ms

## Ablations

| arm | recall@10 | recall@5 | MRR | nDCG@10 | refusal recall | p95 |
|---|---:|---:|---:|---:|---:|---:|
| BM25, no expansion | 0.476 | 0.405 | 0.305 | 0.198 | 0.619 | 41 ms |
| **BM25 + query expansion (shipped default on the offline stack)** | 0.643 | 0.548 | 0.423 | 0.323 | 0.619 | 29 ms |
| Dense only | 0.476 | 0.405 | 0.333 | 0.180 | 0.619 | 9 ms |
| Hybrid (RRF), no reranker | 0.619 | 0.548 | 0.405 | 0.286 | 0.619 | 28 ms |
| BM25 + reranker | 0.619 | 0.571 | 0.404 | 0.298 | 0.619 | 71 ms |
| Hybrid + reranker, no metadata filters | 0.429 | 0.381 | 0.271 | 0.183 | 0.619 | 177 ms |
| Full pipeline, no query expansion | 0.476 | 0.452 | 0.271 | 0.183 | 0.619 | 83 ms |
| Hybrid + reranker (default once a neural embedder is present) | 0.619 | 0.524 | 0.390 | 0.281 | 0.619 | 118 ms |

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
| prose / topic | 24 | 0.708 |
| cross-filing comparison | 7 | 0.571 |
| numeric / table | 11 | 0.545 |

## Refusals by type

| verdict | count |
|---|---:|
| refused_out_of_corpus | 13 |

