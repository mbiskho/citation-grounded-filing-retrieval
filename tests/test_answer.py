"""Tests for the part that matters: citations and refusal."""

from __future__ import annotations

import pytest

from filingqa.answer import FilingQA, RefusalPolicy
from filingqa.answer.citations import (
    MAX_SPAN_CHARS,
    MIN_SPAN_CHARS,
    citation_report,
    normalise,
    verify_answer,
    verify_citation,
)
from filingqa.answer.generators import ExtractiveGenerator
from filingqa.answer.scope import CorpusScope
from filingqa.models import Answer, Citation, Scored, Verdict, split_sentences
from filingqa.retrieval import HashingEmbedder, HybridRetriever
from filingqa.retrieval.rerank import NoOpReranker

# -- sentence splitting ----------------------------------------------------


def test_abbreviations_do_not_split_sentences():
    """Splitting is biased toward under-splitting, on purpose.

    "…outside the U.S. As a result…" is genuinely two sentences, and this
    keeps it as one, because "U.S." at a sentence end is indistinguishable
    from "U.S." mid-sentence without far more machinery.

    The bias is chosen deliberately. An over-split produces a fragment
    carrying no citation marker, which the uncited-claim check would report
    as an ungrounded claim - a false alarm on the project's headline metric.
    An under-split produces one longer quote that still carries its marker
    and still verifies against the source. Wrong in the cheap direction.
    """
    text = "Facilities are located outside the U.S. As a result, performance depends on demand."
    assert len(split_sentences(text)) == 1

    # A plain sentence boundary still splits.
    assert len(split_sentences("Revenue rose. Costs fell.")) == 2


def test_decimals_do_not_split_sentences():
    assert len(split_sentences("Revenue grew 1.5 percent this year.")) == 1


def test_citation_marker_does_not_start_a_new_sentence():
    """ "...conditions. [1] Apple" must not split before the marker."""
    parts = split_sentences("Revenue rose [1]. Costs fell [2].")
    assert len(parts) == 2
    assert all("[" in p for p in parts)


# -- citation verification -------------------------------------------------


def test_valid_citation_verifies(filing):
    filings = {filing.doc_id: filing}
    block = filing.blocks[1]
    citation = Citation(doc_id=filing.doc_id, start=block.start, end=block.end, quote=block.text)
    assert verify_citation(citation, filings).verified


def test_fabricated_quote_fails(filing):
    """The whole claim rests on this: a quote that is not in the span is caught."""
    filings = {filing.doc_id: filing}
    block = filing.blocks[1]
    citation = Citation(
        doc_id=filing.doc_id,
        start=block.start,
        end=block.end,
        quote="Revenue tripled and margins doubled in every segment.",
    )
    result = verify_citation(citation, filings)
    assert not result.verified
    assert "does not match" in result.verification_note


def test_out_of_bounds_span_fails(filing):
    citation = Citation(doc_id=filing.doc_id, start=0, end=999_999, quote="x" * 40)
    assert not verify_citation(citation, {filing.doc_id: filing}).verified


def test_unknown_document_fails(filing):
    citation = Citation(doc_id="nope-2024-10k", start=0, end=50, quote="x" * 40)
    result = verify_citation(citation, {filing.doc_id: filing})
    assert not result.verified and "unknown document" in result.verification_note


def test_span_too_short_fails(filing):
    citation = Citation(doc_id=filing.doc_id, start=0, end=MIN_SPAN_CHARS - 1, quote="Item")
    assert not verify_citation(citation, {filing.doc_id: filing}).verified


def test_whole_section_span_fails(filing):
    """A citation pointing at a whole section is not provenance for a claim."""
    long_filing = filing
    long_filing.text = "x " * (MAX_SPAN_CHARS + 500)
    citation = Citation(doc_id=filing.doc_id, start=0, end=MAX_SPAN_CHARS + 100, quote="x " * 50)
    result = verify_citation(citation, {filing.doc_id: long_filing})
    assert not result.verified
    assert "section" in result.verification_note


def test_typographic_quotes_do_not_break_verification():
    assert normalise("the Company’s risk") == normalise("the Company's risk")


def test_answer_with_a_failed_citation_becomes_a_refusal(filing):
    answer = Answer(
        question="q",
        verdict=Verdict.ANSWERED,
        text="Revenue tripled [1].",
        citations=[
            Citation(doc_id=filing.doc_id, start=0, end=40, quote="completely invented text")
        ],
    )
    verified = verify_answer(answer, {filing.doc_id: filing})
    assert verified.verdict is Verdict.REFUSED_NO_CITATION


def test_answer_with_no_citations_becomes_a_refusal(filing):
    answer = Answer(question="q", verdict=Verdict.ANSWERED, text="Revenue tripled.")
    assert verify_answer(answer, {filing.doc_id: filing}).verdict is Verdict.REFUSED_NO_CITATION


def test_uncited_sentence_becomes_a_refusal(filing):
    block = filing.blocks[1]
    answer = Answer(
        question="q",
        verdict=Verdict.ANSWERED,
        text=f"{block.text} [1]. Margins also improved substantially this year.",
        citations=[
            Citation(doc_id=filing.doc_id, start=block.start, end=block.end, quote=block.text)
        ],
    )
    result = verify_answer(answer, {filing.doc_id: filing})
    assert result.verdict is Verdict.REFUSED_NO_CITATION
    assert "no citation" in result.refusal_reason


def test_citation_report_counts(filing):
    answer = Answer(question="q", verdict=Verdict.ANSWERED, text="x [1].", citations=[])
    report = citation_report([answer])
    assert report["answers"] == 1


# -- scope -----------------------------------------------------------------


@pytest.fixture
def scope(chunks):
    return CorpusScope.from_chunks(chunks)


def test_company_outside_the_corpus_is_refused(scope):
    verdict = scope.check("What was Tesla's revenue in FY2024?")
    assert not verdict.in_scope and "TSLA" in verdict.reason


def test_year_outside_the_corpus_is_refused(scope):
    verdict = scope.check("What did Apple report in fiscal 2019?")
    assert not verdict.in_scope and "2019" in verdict.reason


def test_in_corpus_question_is_in_scope(scope):
    assert scope.check("What are Apple's risks in FY2024?").in_scope


def test_comparison_with_one_known_company_stays_in_scope(scope):
    """Retrieval can still contribute half the answer, so this is not refused
    here - it falls through to the confidence checks."""
    assert scope.check("Compare Apple and Tesla on supply chain risk in FY2024").in_scope


def test_longest_alias_wins(scope):
    assert "GM" in scope.detect_companies("How does General Motors describe competition?")


# -- extractive generation -------------------------------------------------


def test_extractive_answers_are_quotes(chunks, filing):
    """Every sentence is a span from the filing, so the claim cannot fail
    for the extractive path by construction."""
    generator = ExtractiveGenerator()
    results = [Scored(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)]
    answer = generator.generate("supply chain concentration risk", results)

    assert answer.answered
    verified = verify_answer(answer, {filing.doc_id: filing})
    assert verified.answered
    assert verified.all_citations_verified
    assert verified.uncited_sentences == []


def test_extractive_refuses_when_nothing_matches(chunks):
    generator = ExtractiveGenerator(min_sentence_score=0.99)
    results = [Scored(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)]
    answer = generator.generate("photosynthesis in marine algae", results)
    assert not answer.answered


def test_table_chunks_are_quoted_by_row(filing):
    """Quoting a whole table produced 5,000-char spans that failed
    verification and turned answerable questions into refusals."""
    from filingqa.models import Chunk

    table_text = "Segment | 2024 | 2023\nAmericas: 2024 $167,045; 2023 $162,560\nEurope: 2024 $101,328; 2023 $94,294"
    filing.text = filing.text + "\n\n" + table_text
    chunk = Chunk(
        chunk_id="t1",
        doc_id=filing.doc_id,
        text=table_text,
        start=len(filing.text) - len(table_text),
        end=len(filing.text),
        item="Item 7",
        is_table=True,
        company="AAPL",
        fiscal_year=2024,
    )
    generator = ExtractiveGenerator(min_sentence_score=0.0)
    answer = generator.generate("Americas segment 2024", [Scored(chunk=chunk, score=1.0)])
    assert answer.citations
    assert all(c.end - c.start < 200 for c in answer.citations)
    assert verify_answer(answer, {filing.doc_id: filing}).all_citations_verified


# -- engine ----------------------------------------------------------------


@pytest.fixture
def engine(chunks, filing):
    retriever = HybridRetriever(
        chunks, embedder=HashingEmbedder(dim=128), reranker=NoOpReranker()
    ).build(verbose=False)
    return FilingQA(retriever, [filing])


def test_engine_answers_an_in_corpus_question(engine):
    answer = engine.ask("What supply chain risks does Apple describe?")
    assert answer.answered
    assert answer.all_citations_verified
    assert answer.uncited_sentences == []


def test_engine_refuses_out_of_corpus_before_retrieving(engine):
    answer = engine.ask("What was Tesla's revenue in FY2024?")
    assert answer.verdict is Verdict.REFUSED_OUT_OF_CORPUS
    assert answer.retrieved == [], "scope check must run before retrieval, not after"


def test_refusal_can_be_disabled_for_measurement(chunks, filing):
    retriever = HybridRetriever(
        chunks, embedder=HashingEmbedder(dim=128), reranker=NoOpReranker()
    ).build(verbose=False)
    engine = FilingQA(retriever, [filing], refusal=RefusalPolicy(enabled=False))
    assert engine.ask("What was Tesla's revenue?").retrieved
