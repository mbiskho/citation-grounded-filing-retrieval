"""Citation verification.

The claim this project makes is "zero uncited claims". That is only worth
anything if "cited" means something stronger than "the model wrote a number in
brackets".

So every citation is checked against the source text three ways:

1. **The span resolves.** `(doc_id, start, end)` must be in bounds for a
   filing that exists in the corpus.
2. **The quote matches the span.** What the answer says the filing says must
   be what the filing says, compared on normalised whitespace and quote
   characters.
3. **The span is not empty or degenerate.** A zero-width or whole-document
   citation is not provenance.

A citation that fails any of these is marked unverified, and an answer with an
unverified citation is downgraded to a refusal. Failing loudly is the point:
a wrong citation is worse than no citation because it carries the appearance
of provenance.
"""

from __future__ import annotations

import re
import unicodedata

from ..models import Answer, Citation, Filing, Verdict

#: Below this, a span is too short to establish anything.
MIN_SPAN_CHARS = 12
#: Above this, a "citation" is really just pointing at the document.
MAX_SPAN_CHARS = 4000

_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Whitespace- and quote-insensitive form for comparison.

    Filings use typographic quotes and non-breaking spaces inconsistently, and
    chunking joins sentences with a single space. None of that changes what
    the filing says, so none of it should fail a citation check.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return _WS.sub(" ", folded).strip().lower()


def verify_citation(citation: Citation, filings: dict[str, Filing]) -> Citation:
    """Check one citation against the corpus. Returns it with `verified` set."""
    filing = filings.get(citation.doc_id)
    if filing is None:
        citation.verified = False
        citation.verification_note = f"unknown document {citation.doc_id!r}"
        return citation

    if not (0 <= citation.start < citation.end <= len(filing.text)):
        citation.verified = False
        citation.verification_note = (
            f"span {citation.start}:{citation.end} out of bounds for "
            f"{citation.doc_id} (length {len(filing.text)})"
        )
        return citation

    span_length = citation.end - citation.start
    if span_length < MIN_SPAN_CHARS:
        citation.verified = False
        citation.verification_note = f"span of {span_length} chars is too short to cite"
        return citation
    if span_length > MAX_SPAN_CHARS:
        citation.verified = False
        citation.verification_note = f"span of {span_length} chars points at a section, not a claim"
        return citation

    source = normalise(filing.slice(citation.start, citation.end))
    quoted = normalise(citation.quote)

    if not quoted:
        citation.verified = False
        citation.verification_note = "citation carries no quote to check"
        return citation

    if quoted in source or source in quoted:
        citation.verified = True
        citation.verification_note = ""
        return citation

    # Substring failed. Fall back to token containment, which tolerates the
    # ellipsis a generator inserts when quoting a long passage, but still
    # requires the words to actually be there.
    source_tokens = set(source.split())
    quote_tokens = [t for t in quoted.split() if len(t) > 2]
    if quote_tokens:
        coverage = sum(1 for t in quote_tokens if t in source_tokens) / len(quote_tokens)
        if coverage >= 0.9:
            citation.verified = True
            citation.verification_note = f"token match at {coverage:.0%}"
            return citation
        citation.verification_note = f"quote does not match span (token overlap {coverage:.0%})"
    else:
        citation.verification_note = "quote has no substantive tokens"

    citation.verified = False
    return citation


def verify_answer(answer: Answer, filings: dict[str, Filing]) -> Answer:
    """Verify every citation and downgrade the answer if any fails.

    Ordering matters: an answer is only allowed to stay `ANSWERED` if it has
    at least one citation, every citation verifies, and no sentence making a
    claim is left uncited.
    """
    answer.citations = [verify_citation(c, filings) for c in answer.citations]

    if not answer.answered:
        return answer

    if not answer.citations:
        answer.verdict = Verdict.REFUSED_NO_CITATION
        answer.refusal_reason = "the generated answer carried no citations"
        return answer

    failed = [c for c in answer.citations if not c.verified]
    if failed:
        answer.verdict = Verdict.REFUSED_NO_CITATION
        answer.refusal_reason = (
            f"{len(failed)} of {len(answer.citations)} citations failed verification: "
            + "; ".join(c.verification_note for c in failed[:3])
        )
        return answer

    uncited = answer.uncited_sentences
    if uncited:
        answer.verdict = Verdict.REFUSED_NO_CITATION
        answer.refusal_reason = (
            f"{len(uncited)} sentence(s) made a claim with no citation: {uncited[0][:120]!r}"
        )
    return answer


def citation_report(answers: list[Answer]) -> dict:
    """Corpus-level citation health. The numbers the README publishes."""
    answered = [a for a in answers if a.answered]
    citations = [c for a in answered for c in a.citations]
    uncited = [s for a in answered for s in a.uncited_sentences]

    return {
        "answers": len(answers),
        "answered": len(answered),
        "refused": len(answers) - len(answered),
        "total_citations": len(citations),
        "verified_citations": sum(1 for c in citations if c.verified),
        "citation_accuracy": (
            sum(1 for c in citations if c.verified) / len(citations) if citations else 0.0
        ),
        "citations_per_answer": len(citations) / len(answered) if answered else 0.0,
        "answers_with_uncited_claims": sum(1 for a in answered if a.uncited_sentences),
        "uncited_claim_count": len(uncited),
    }
