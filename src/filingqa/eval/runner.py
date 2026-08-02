"""Ablation runner.

The point of an ablation is to answer "what is each component actually worth",
and the only way that means anything is if exactly one thing changes per arm
and every arm sees the same questions.

The arms are ordered so the table reads as an argument:

    bm25 only          - would a keyword search have been enough?
    dense only         - would embeddings alone have been enough?
    hybrid, no rerank  - what does fusion add over the better single retriever?
    hybrid + rerank    - what does the reranker add on top?
    ...plus chunking and filtering variants.

If the full system does not beat `bm25 only` by a wide margin, the honest
conclusion is that the corpus did not need a vector database, and that
conclusion should be reachable from the table rather than argued around.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..answer import FilingQA, RefusalPolicy
from ..answer.citations import citation_report
from ..models import Answer, Chunk, Filing, Scored
from ..retrieval import HybridRetriever, RetrievalConfig, get_embedder, get_reranker
from ..retrieval.rerank import NoOpReranker
from .goldset import GoldQuestion
from .metrics import evaluate_refusal, evaluate_retrieval, percentile


@dataclass
class AblationArm:
    key: str
    label: str
    description: str
    config: RetrievalConfig
    reranker: str = "auto"
    #: Set on the arm that represents the shipped system.
    is_full_system: bool = False


ABLATIONS: list[AblationArm] = [
    AblationArm(
        key="bm25-plain",
        label="BM25, no expansion",
        description="Keyword search alone, without the financial synonym rewrite. The floor.",
        config=RetrievalConfig(use_bm25=True, use_dense=False, expand_query=False),
        reranker="none",
    ),
    AblationArm(
        key="bm25-only",
        label="BM25 + query expansion (shipped default on the offline stack)",
        description=(
            "Keyword search with the financial tokeniser and synonym rewrite. "
            "The best-measured configuration when no neural model is installed, "
            "and therefore the default in that case."
        ),
        config=RetrievalConfig(use_bm25=True, use_dense=False),
        reranker="none",
        is_full_system=True,
    ),
    AblationArm(
        key="dense-only",
        label="Dense only",
        description="Bi-encoder retrieval alone. No keyword matching, no reranker.",
        config=RetrievalConfig(use_bm25=False, use_dense=True),
        reranker="none",
    ),
    AblationArm(
        key="hybrid-no-rerank",
        label="Hybrid (RRF), no reranker",
        description="BM25 and dense fused by reciprocal rank fusion.",
        config=RetrievalConfig(use_bm25=True, use_dense=True),
        reranker="none",
    ),
    AblationArm(
        key="bm25-rerank",
        label="BM25 + reranker",
        description="Isolates the reranker's contribution without the dense retriever.",
        config=RetrievalConfig(use_bm25=True, use_dense=False),
        reranker="auto",
    ),
    AblationArm(
        key="hybrid-no-filters",
        label="Hybrid + reranker, no metadata filters",
        description="Full pipeline with company/year filtering disabled.",
        config=RetrievalConfig(use_bm25=True, use_dense=True, allow_filters=False),
        reranker="auto",
    ),
    AblationArm(
        key="no-expansion",
        label="Full pipeline, no query expansion",
        description="Everything except the financial synonym rewrite.",
        config=RetrievalConfig(use_bm25=True, use_dense=True, expand_query=False),
        reranker="auto",
    ),
    AblationArm(
        key="full-hybrid",
        label="Hybrid + reranker (default once a neural embedder is present)",
        description=(
            "Everything on. Measured worse than BM25 alone with the offline "
            "embedder, which is why it is not the default in that case."
        ),
        config=RetrievalConfig(use_bm25=True, use_dense=True, allow_filters=True),
        reranker="auto",
    ),
]


@dataclass
class EvalResult:
    arm: str
    label: str
    description: str
    embedder: str = ""
    reranker: str = ""
    retrieval: dict[str, Any] = field(default_factory=dict)
    refusal: dict[str, Any] = field(default_factory=dict)
    citations: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    is_full_system: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_arm(
    arm: AblationArm,
    chunks: list[Chunk],
    filings: list[Filing],
    questions: list[GoldQuestion],
    *,
    embedder_name: str = "auto",
    refusal: RefusalPolicy | None = None,
    verbose: bool = True,
    prebuilt: HybridRetriever | None = None,
    reranker=None,
) -> tuple[EvalResult, dict[str, Answer]]:
    """Run one ablation arm over the whole gold set."""
    embedder = prebuilt.embedder if prebuilt else get_embedder(embedder_name)
    reranker = reranker or get_reranker(arm.reranker)

    retriever = HybridRetriever(chunks, embedder=embedder, reranker=reranker, config=arm.config)
    if prebuilt is not None:
        # Index construction is by far the slowest step and it does not depend
        # on which arm is running, so it is built once and shared. Each arm
        # still gets its own config and reranker.
        retriever.bm25 = prebuilt.bm25
        retriever.dense = prebuilt.dense
        retriever._built = True
        retriever._share_idf()
    else:
        retriever.build(verbose=False)

    engine = FilingQA(retriever, filings, refusal=refusal or RefusalPolicy())

    retrieved: dict[str, list[Scored]] = {}
    answers: dict[str, Answer] = {}
    latencies: list[float] = []

    for question in questions:
        started = time.perf_counter()
        answer = engine.ask(question.question, top_k=20)
        latencies.append((time.perf_counter() - started) * 1000)
        answers[question.question_id] = answer
        retrieved[question.question_id] = answer.retrieved

    retrieval_metrics = evaluate_retrieval(retrieved, questions)
    refusal_metrics = evaluate_refusal(answers, questions)

    result = EvalResult(
        arm=arm.key,
        label=arm.label,
        description=arm.description,
        embedder=getattr(embedder, "name", "?"),
        reranker=getattr(reranker, "name", "?"),
        retrieval=retrieval_metrics.to_dict(),
        refusal=refusal_metrics.to_dict(),
        citations=citation_report(list(answers.values())),
        latency_ms={
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "p99": round(percentile(latencies, 0.99), 2),
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        },
        is_full_system=arm.is_full_system,
    )

    if verbose:
        r10 = result.retrieval["recall_at_k"].get("10", 0)
        print(
            f"  {arm.key:<22} recall@10 {r10:>6.3f}  "
            f"mrr {result.retrieval['mrr']:>6.3f}  "
            f"refusal-r {result.refusal['refusal_recall']:>5.2f}  "
            f"p95 {result.latency_ms['p95']:>7.1f}ms",
            flush=True,
        )
    return result, answers


def run_ablations(
    chunks: list[Chunk],
    filings: list[Filing],
    questions: list[GoldQuestion],
    *,
    arms: list[AblationArm] | None = None,
    embedder_name: str = "auto",
    verbose: bool = True,
) -> tuple[list[EvalResult], dict[str, Answer]]:
    arms = arms or ABLATIONS

    if verbose:
        print("building shared indexes...", flush=True)
    shared = HybridRetriever(
        chunks,
        embedder=get_embedder(embedder_name),
        reranker=NoOpReranker(),
        config=RetrievalConfig(),
    ).build(verbose=verbose)

    results: list[EvalResult] = []
    full_answers: dict[str, Answer] = {}
    for arm in arms:
        result, answers = run_arm(
            arm,
            chunks,
            filings,
            questions,
            embedder_name=embedder_name,
            verbose=verbose,
            prebuilt=shared,
        )
        results.append(result)
        if arm.is_full_system:
            full_answers = answers
    return results, full_answers


def save_results(
    results: list[EvalResult],
    path: Path | str,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    import platform
    from datetime import datetime, timezone

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        **(extra or {}),
        "arms": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
