"""Answer generators.

Two, and the default is the boring one on purpose.

`ExtractiveGenerator` does not write prose. It selects sentences out of the
retrieved spans and returns them verbatim with their citations attached. That
sounds like a limitation and in a consumer product it would be. In a credit
memo it is closer to the requirement: what an analyst needs to paste into a
document is the filing's own words with a pointer to the page, and a fluent
paraphrase is a liability, because the paraphrase is the part nobody can
check.

It also makes the headline claim structural rather than aspirational: if every
sentence in the answer *is* a quoted span, "zero uncited claims" cannot fail
for the extractive path. The LLM path can fail it, which is why the same
verification runs over both.

`LLMGenerator` produces readable prose with bracketed citation markers, then
gets verified against the source exactly like anything else. It is better to
read and it is the one that can hallucinate, so it is not the default.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

from ..models import Answer, Chunk, Citation, Filing, Scored, Verdict, split_sentences
from ..retrieval.tokenize import tokenize


class Generator(Protocol):
    name: str

    def generate(
        self,
        question: str,
        results: list[Scored],
        filings: dict[str, Filing] | None = None,
    ) -> Answer: ...


# --------------------------------------------------------------------------
# Extractive
# --------------------------------------------------------------------------


def _sentence_spans_in_chunk(
    chunk: Chunk, filings: dict[str, Filing] | None = None
) -> list[tuple[str, int, int]]:
    """Quotable units of a chunk with absolute spans into the filing text.

    **Units are cut from the source text, not from `chunk.text`.** That
    distinction is the whole correctness argument here, and getting it wrong
    is subtle:

    A prose chunk is built by joining sentences from several blocks with a
    single space, while the source separates those blocks with "\\n\\n". So an
    offset computed by searching inside the reconstructed chunk drifts by one
    character for every block boundary the chunk crosses. The resulting
    citation is *nearly* right - off by two or three characters - which
    verification catches as an 88% token overlap and reports as a refusal.
    An answerable question is lost to an arithmetic error.

    Slicing the filing and splitting that removes the class of bug entirely:
    the quote is, by construction, exactly the characters the span points at.

    Falls back to the chunk text when no filing is supplied, so the generator
    still works standalone (in tests, say) - just with the caveat above.

    **Tables are split by row, not by sentence.** A rendered table contains no
    sentence-ending punctuation, so a sentence splitter returns the entire
    table as one unit - which then gets quoted whole. That produced 5,000-plus
    character "quotes" that failed span verification for pointing at a section
    rather than a claim. Row-level spans are also what an analyst wants: the
    citation should land on the line carrying the number.
    """
    filing = (filings or {}).get(chunk.doc_id)
    if filing is not None:
        body = filing.text[chunk.start : chunk.end]
        offset = chunk.start
    else:
        body = chunk.text
        offset = chunk.start
        if body.startswith("[") and "]\n" in body:
            # The section prefix is display furniture and occupies no space in
            # the source document.
            body = body.split("]\n", 1)[1]

    units = (
        [line for line in body.split("\n") if line.strip()]
        if chunk.is_table
        else split_sentences(body)
    )

    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for unit in units:
        index = body.find(unit, cursor)
        if index == -1:
            continue
        spans.append((unit, offset + index, offset + index + len(unit)))
        cursor = index + len(unit)
    return spans


def _mark(sentence: str, index: int) -> str:
    """Attach a citation marker inside the sentence, before its full stop.

    "...fell 3% [1]." rather than "...fell 3%. [1]". Not a style preference:
    a marker after the terminal punctuation is, to any sentence splitter, the
    first token of the *next* sentence - so the claim it supports reads as
    uncited and the following sentence reads as doubly cited.
    """
    stripped = sentence.rstrip()
    if stripped and stripped[-1] in ".!?":
        return f"{stripped[:-1].rstrip()} [{index}]{stripped[-1]}"
    return f"{stripped} [{index}]"


class ExtractiveGenerator:
    """Selects the best-supported sentences and quotes them.

    Sentence scoring is query-term coverage with a bonus for numeric matches
    and a small bonus for being early in a retrieved chunk (filings front-load
    the claim and follow with qualifications).
    """

    name = "extractive"

    def __init__(
        self,
        *,
        max_sentences: int = 4,
        min_sentence_score: float = 0.12,
        numeric_bonus: float = 0.4,
        max_per_chunk: int = 2,
    ) -> None:
        self.max_sentences = max_sentences
        self.min_sentence_score = min_sentence_score
        self.numeric_bonus = numeric_bonus
        self.max_per_chunk = max_per_chunk

    def _score_sentence(self, query_tokens: set[str], sentence: str) -> float:
        tokens = set(tokenize(sentence))
        if not query_tokens or not tokens:
            return 0.0
        coverage = len(query_tokens & tokens) / len(query_tokens)
        query_numbers = {t for t in query_tokens if t.startswith(("num:", "money:", "pct:"))}
        if query_numbers and query_numbers & tokens:
            coverage += self.numeric_bonus
        # A sentence with no numbers at all is a weak answer to a numeric
        # question, and filings are full of prose that mentions the right
        # words without the figure.
        if query_numbers and not any(t.startswith("num:") for t in tokens):
            coverage *= 0.6
        return coverage

    def generate(
        self,
        question: str,
        results: list[Scored],
        filings: dict[str, Filing] | None = None,
    ) -> Answer:
        query_tokens = set(tokenize(question))
        scored: list[tuple[float, str, Chunk, int, int]] = []

        for position, result in enumerate(results):
            per_chunk = 0
            for order, (sentence, start, end) in enumerate(
                _sentence_spans_in_chunk(result.chunk, filings)
            ):
                if len(sentence) < 25:
                    continue
                score = self._score_sentence(query_tokens, sentence)
                # Retrieval rank and within-chunk position both matter, but
                # only as tie-breakers - a strong sentence at rank 8 should
                # still beat a weak one at rank 1.
                score += 0.05 / (1 + position) + 0.02 / (1 + order)
                if score < self.min_sentence_score:
                    continue
                scored.append((score, sentence, result.chunk, start, end))
                per_chunk += 1
                if per_chunk >= self.max_per_chunk:
                    break

        scored.sort(key=lambda x: -x[0])
        selected = scored[: self.max_sentences]

        if not selected:
            return Answer(
                question=question,
                verdict=Verdict.REFUSED_LOW_CONFIDENCE,
                refusal_reason="no retrieved sentence matched the question well enough to quote",
                retrieved=results,
                generator=self.name,
            )

        # Restore document order so the answer reads coherently rather than in
        # descending score order.
        selected.sort(key=lambda x: (x[2].doc_id, x[3]))

        citations: list[Citation] = []
        parts: list[str] = []
        for index, (_score, sentence, chunk, start, end) in enumerate(selected, start=1):
            citations.append(
                Citation(
                    doc_id=chunk.doc_id,
                    start=start,
                    end=end,
                    quote=sentence,
                    label=chunk.citation_label,
                    chunk_id=chunk.chunk_id,
                )
            )
            # The marker goes on every sentence the *answer-level* splitter
            # will see, not just at the end of the extracted span. A quoted
            # passage sometimes re-splits into two sentences when read back -
            # a colon followed by a capital, an embedded list - and the
            # trailing half would then look like an uncited claim. Aligning
            # the two splits here keeps the uncited-claim metric measuring
            # hallucination rather than punctuation.
            marked = [_mark(part, index) for part in split_sentences(sentence)] or [
                _mark(sentence, index)
            ]
            parts.append(" ".join(marked))

        return Answer(
            question=question,
            verdict=Verdict.ANSWERED,
            text=" ".join(parts),
            citations=citations,
            retrieved=results,
            generator=self.name,
        )


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You answer questions about SEC filings for financial analysts.

Rules, in order of importance:

1. Every sentence containing a fact, figure or characterisation MUST end with
   a citation marker like [1] or [2, 3] referring to the numbered sources.
2. Quote figures exactly as they appear in the source. Do not convert units,
   round, or compute values that are not stated.
3. If the sources do not answer the question, reply with exactly:
   INSUFFICIENT_EVIDENCE
   Do not partially answer. Do not speculate. An unsourced number is worse
   than no answer.
4. Be brief. Three sentences at most.

Return JSON only:
{"answer": "...", "used_sources": [1, 2]}
or
{"answer": "INSUFFICIENT_EVIDENCE", "used_sources": []}"""


class LLMGenerator:
    """Prose answers with enforced citation markers.

    Everything it returns is verified against the source spans afterwards, so
    a fabricated quote becomes a refusal rather than an answer. The prompt
    asks for citations; the verifier is what makes them true.
    """

    name = "llm"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        max_context_chunks: int = 8,
        temperature: float = 0.0,
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.model = model
        self.name = f"llm:{model}"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.max_context_chunks = max_context_chunks
        self.temperature = temperature
        self.base_url = base_url
        #: USD per 1M tokens (input, output) for the cost column.
        self.pricing = (0.15, 0.60) if "mini" in model else (2.50, 10.00)
        self.cost_usd = 0.0

    def _context(self, results: list[Scored]) -> tuple[str, list[Scored]]:
        used = results[: self.max_context_chunks]
        blocks = []
        for index, result in enumerate(used, start=1):
            blocks.append(f"[{index}] {result.chunk.citation_label}\n{result.chunk.text}")
        return "\n\n".join(blocks), used

    def _call(self, question: str, context: str) -> dict:
        import urllib.request

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"SOURCES\n\n{context}\n\nQUESTION\n{question}"},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read())

        usage = body.get("usage", {})
        self.cost_usd += (
            usage.get("prompt_tokens", 0) * self.pricing[0]
            + usage.get("completion_tokens", 0) * self.pricing[1]
        ) / 1_000_000
        return json.loads(body["choices"][0]["message"]["content"])

    def generate(
        self,
        question: str,
        results: list[Scored],
        filings: dict[str, Filing] | None = None,
    ) -> Answer:
        context, used = self._context(results)
        try:
            payload = self._call(question, context)
        except Exception as exc:  # noqa: BLE001 - a provider failure is a refusal
            return Answer(
                question=question,
                verdict=Verdict.REFUSED_LOW_CONFIDENCE,
                refusal_reason=f"generator unavailable: {exc}",
                retrieved=results,
                generator=self.name,
                cost_usd=self.cost_usd,
            )

        text = (payload.get("answer") or "").strip()
        if not text or text.upper().startswith("INSUFFICIENT_EVIDENCE"):
            return Answer(
                question=question,
                verdict=Verdict.REFUSED_OUT_OF_CORPUS,
                refusal_reason="the model judged the retrieved sources insufficient",
                retrieved=results,
                generator=self.name,
                cost_usd=self.cost_usd,
            )

        citations = self._citations_for(text, used)
        return Answer(
            question=question,
            verdict=Verdict.ANSWERED,
            text=text,
            citations=citations,
            retrieved=results,
            generator=self.name,
            cost_usd=self.cost_usd,
        )

    @staticmethod
    def _citations_for(text: str, used: list[Scored]) -> list[Citation]:
        """Turn [n] markers into spans.

        The model cites a *source*, not a character range - asking it for
        offsets produces confident fiction. So the marker selects the chunk,
        and the quote recorded is the chunk's own text, which is what gets
        verified.
        """
        citations: list[Citation] = []
        seen: set[int] = set()
        for marker in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
            for raw in marker.group(1).split(","):
                index = int(raw.strip())
                if index in seen or not (1 <= index <= len(used)):
                    continue
                seen.add(index)
                chunk = used[index - 1].chunk
                body = chunk.text.split("]\n", 1)[-1] if chunk.text.startswith("[") else chunk.text
                citations.append(
                    Citation(
                        doc_id=chunk.doc_id,
                        start=chunk.start,
                        end=chunk.end,
                        quote=body[:MAX_QUOTE_CHARS],
                        label=chunk.citation_label,
                        chunk_id=chunk.chunk_id,
                    )
                )
        return citations


MAX_QUOTE_CHARS = 3000


def get_generator(name: str = "extractive", **kwargs) -> Generator:
    if name in ("extractive", "offline"):
        return ExtractiveGenerator(**kwargs)
    if name in ("llm", "openai"):
        return LLMGenerator(**kwargs)
    raise KeyError(f"unknown generator {name!r}")
