"""The engine: retrieve, decide whether to answer, generate, verify.

The refusal path gets more design attention here than the retrieval path, and
that ordering is deliberate. In finance the expensive failure is not a miss -
an analyst who gets nothing goes and reads the filing, which is what they were
going to do anyway. The expensive failure is a confident answer with no filing
behind it, because that one gets pasted into a credit memo and nobody checks
it until it matters.

So there are three independent chances to refuse:

1. **Before generating** - retrieval confidence is too low.
2. **During generation** - the generator finds nothing worth quoting, or the
   model declares the evidence insufficient.
3. **After generating** - a citation fails verification, or a sentence makes a
   claim with nothing attached.

Only an answer that survives all three is returned as an answer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..models import Answer, Filing, Verdict
from ..retrieval import HybridRetriever
from ..retrieval.tokenize import tokenize
from .citations import citation_report, normalise, verify_answer, verify_citation
from .generators import (
    ExtractiveGenerator,
    Generator,
    LLMGenerator,
    get_generator,
)
from .scope import CorpusScope, ScopeVerdict

__all__ = [
    "FilingQA",
    "RefusalPolicy",
    "CorpusScope",
    "ScopeVerdict",
    "Generator",
    "ExtractiveGenerator",
    "LLMGenerator",
    "get_generator",
    "verify_answer",
    "verify_citation",
    "citation_report",
    "normalise",
]


@dataclass
class RefusalPolicy:
    """When to decline before spending anything on generation.

    **The defaults below are calibrated, and the calibration says the soft
    signals do not work on this corpus.**

    A grid sweep over both thresholds on the dev split (see
    `evaluation/results/refusal-calibration.json`) maximises refusal F1 at
    zero for both - that is, with the confidence gates switched off entirely.
    Any positive threshold refused more genuine questions than bad ones.

    The reason is visible in the score distributions. Best answer-support
    score averages 0.720 on answerable questions and 0.698 on questions a
    10-K structurally cannot answer, like "what price target do analysts
    have". Those two distributions overlap almost completely, because a
    question about Apple and pricing retrieves passages about Apple and
    pricing whether or not the filing contains the answer. A threshold cannot
    separate populations that are not separated.

    What does work is the exact scope check: a company or fiscal year outside
    the corpus is a fact, not a confidence estimate. That catches 100% of
    those two failure classes and 0% of the third, which is exactly what the
    refusal numbers in the README show.

    The gates are kept because a cross-encoder reranker produces far better
    separated scores and would likely make them earn their place - but they
    ship calibrated to off rather than set to a plausible-looking constant.
    """

    #: Reranker score of the top hit, below which nothing is worth quoting.
    #: Calibrated to 0.0 - see the class docstring.
    min_top_score: float = 0.0
    #: Fraction of substantive query terms appearing in the top results.
    #: Calibrated to 0.0: common financial vocabulary means a question the
    #: corpus cannot answer still scores near 1.0.
    min_term_coverage: float = 0.0
    #: How many top results to compute coverage over.
    coverage_depth: int = 5
    #: Refuse when the question names a company or year the corpus does not
    #: hold. Exact rather than heuristic, and the highest-precision signal
    #: available - see `scope.py`.
    check_scope: bool = True
    #: Set False to measure the pipeline with refusal disabled.
    enabled: bool = True


def _term_coverage(question: str, texts: list[str]) -> float:
    query_terms = {t for t in tokenize(question) if not t.startswith(("num:", "money:", "pct:"))}
    if not query_terms:
        return 1.0
    pool: set[str] = set()
    for text in texts:
        pool |= set(tokenize(text))
    return len(query_terms & pool) / len(query_terms)


class FilingQA:
    """End-to-end question answering over a filing corpus."""

    def __init__(
        self,
        retriever: HybridRetriever,
        filings: list[Filing] | dict[str, Filing],
        *,
        generator: Generator | None = None,
        refusal: RefusalPolicy | None = None,
        top_k: int = 10,
    ) -> None:
        self.retriever = retriever
        self.filings = filings if isinstance(filings, dict) else {f.doc_id: f for f in filings}
        self.generator = generator or ExtractiveGenerator()
        self.refusal = refusal or RefusalPolicy()
        self.top_k = top_k
        self.scope = CorpusScope.from_chunks(retriever.chunks)

    def ask(self, question: str, *, top_k: int | None = None) -> Answer:
        started = time.perf_counter()

        # Scope first: it is exact, it costs microseconds, and it is the only
        # check that reliably catches "right kind of question, wrong corpus".
        # Running it before retrieval also means an out-of-corpus question
        # never reaches a paid generator.
        if self.refusal.enabled and self.refusal.check_scope:
            verdict = self.scope.check(question)
            if not verdict.in_scope:
                return self._refuse(
                    question,
                    Verdict.REFUSED_OUT_OF_CORPUS,
                    verdict.reason,
                    started,
                    [],
                    0.0,
                )

        results, trace = self.retriever.search_with_trace(question, top_k=top_k or self.top_k)

        if not results:
            return self._refuse(
                question,
                Verdict.REFUSED_OUT_OF_CORPUS,
                "retrieval returned nothing for this question",
                started,
                results,
                0.0,
            )

        top_score = results[0].score
        coverage = _term_coverage(
            question, [r.chunk.text for r in results[: self.refusal.coverage_depth]]
        )
        # Confidence is the weaker of the two signals rather than their
        # average. A question can retrieve a strong-looking chunk about the
        # wrong subject, and averaging lets a high score paper over the term
        # coverage that would have caught it.
        confidence = min(_squash(top_score, self.refusal.min_top_score), coverage)

        if self.refusal.enabled:
            if top_score < self.refusal.min_top_score:
                return self._refuse(
                    question,
                    Verdict.REFUSED_LOW_CONFIDENCE,
                    f"top retrieval score {top_score:.3f} below threshold "
                    f"{self.refusal.min_top_score:.3f}",
                    started,
                    results,
                    confidence,
                )
            if coverage < self.refusal.min_term_coverage:
                return self._refuse(
                    question,
                    Verdict.REFUSED_OUT_OF_CORPUS,
                    f"only {coverage:.0%} of question terms appear in the corpus; "
                    "this looks like a question the filings do not cover",
                    started,
                    results,
                    confidence,
                )

        answer = self.generator.generate(question, results, self.filings)
        answer.confidence = confidence
        answer = verify_answer(answer, self.filings)
        answer.latency_ms = (time.perf_counter() - started) * 1000
        answer.retrieved = results
        answer.trace = trace.to_dict()
        return answer

    def _refuse(
        self,
        question: str,
        verdict: Verdict,
        reason: str,
        started: float,
        results,
        confidence: float,
    ) -> Answer:
        return Answer(
            question=question,
            verdict=verdict,
            refusal_reason=reason,
            retrieved=results,
            confidence=confidence,
            latency_ms=(time.perf_counter() - started) * 1000,
            generator=self.generator.name,
        )

    def ask_many(self, questions: list[str]) -> list[Answer]:
        return [self.ask(q) for q in questions]


def _squash(score: float, threshold: float) -> float:
    """Map a raw retriever score onto 0-1 for reporting.

    Reranker scores are not probabilities and differ by an order of magnitude
    between the lexical and cross-encoder rerankers, so a single absolute
    scale would be meaningless. This anchors on the configured threshold: at
    threshold the confidence is 0.5, and it rises from there.
    """
    if threshold <= 0:
        return 1.0
    ratio = score / (2 * threshold)
    return max(0.0, min(1.0, ratio))
