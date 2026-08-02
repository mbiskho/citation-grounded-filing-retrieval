# Refusal

> I spent longer on refusal than on retrieval. In finance the expensive
> failure is not a miss — it is a confident answer with no filing behind it.

That was the plan. The result is more specific and more useful than the plan:
**one of the two refusal mechanisms works perfectly and the other does not
work at all**, and the reason generalises beyond this project.

## The three failure classes

| unanswerable because… | n | refused | mechanism |
|---|---:|---:|---|
| the company is not in the corpus | 10 | **100%** | exact scope check |
| the fiscal year is not in the corpus | 12 | **100%** | exact scope check |
| a 10-K structurally does not contain it | 13 | **0%** | none found |

Overall refusal recall 0.619, precision 0.929.

## What works: exact scope checks

A question naming Tesla, against a corpus of Apple, JPMorgan and Ford, is
unanswerable as a matter of fact. Same for FY2019 against a FY2023–FY2025
corpus. These are not confidence estimates — they are lookups.

The check runs **before retrieval**, which also means an out-of-corpus
question never reaches a paid generator.

The company side uses a gazetteer of ~40 recognisable public companies. That
is narrow by design: entity linking over free text is its own project, and a
wrong link causes a *false* refusal, which is a worse experience than a miss.
A question about an obscure private company will not be caught here and falls
through to the confidence checks — which, as below, will not catch it either.

The period side is exact rather than heuristic: parse years from the question,
compare against the corpus range.

## What does not work: confidence thresholds

The third class is questions like:

- *"What price target do analysts have for Apple stock?"*
- *"What was Apple's share price on the last day of FY2024?"*
- *"What did Ford's CEO say on the FY2025 earnings call?"*

The company is in the corpus. The year is in the corpus. The vocabulary is
financial. Retrieval returns ten passages about Apple and about pricing, and
they look exactly like the passages returned for a real question.

I built two signals and swept both properly on the dev split:

- **top retrieval score** — the reranker's score for the best hit
- **term coverage** — fraction of substantive query terms appearing anywhere
  in the top results

**The sweep selected zero for both.** Every positive threshold refused more
answerable questions than unanswerable ones; refusal F1 peaked with the gates
switched off. The grid is committed at
[`../evaluation/results/refusal-calibration.json`](../evaluation/results/refusal-calibration.json).

Here is why, measured on dev:

| question class | mean best-support | median | min | max |
|---|---:|---:|---:|---:|
| answerable | 0.720 | 0.733 | 0.511 | 0.956 |
| unanswerable by topic | 0.698 | 0.650 | 0.571 | 0.844 |
| unanswerable by company | 0.580 | 0.686 | 0.120 | 0.829 |
| unanswerable by year | 0.636 | 0.686 | 0.286 | 1.025 |

The answerable and topic-unanswerable distributions overlap almost completely.
**No threshold separates populations that are not separated.**

Term coverage fails for a related reason: financial vocabulary is common. Ask
"what is the capital of France" of a bank's 10-K and both "capital" and
"France" appear all over it, so coverage reads near 1.0 on a question the
corpus has nothing to say about.

## The conclusion

**Calibrated refusal in RAG needs structured knowledge of what the corpus
contains, not a confidence score.**

A retriever's confidence measures *how well the query matches the index*. That
is a different question from *whether the corpus contains the answer*, and the
two come apart precisely when the question is on-topic but out-of-scope —
which is the case that matters, because it is the one that produces a
confident wrong answer.

The scope check works because it answers the second question directly. Closing
the third class needs the same move: a model of what a 10-K contains, so
"analyst price target" and "share price on a date" can be rejected as
categories rather than scored as queries.

That is a taxonomy problem, not a threshold problem. It is the next thing I
would build, and it is deliberately not faked here with a keyword list matched
against my own question templates — that would score 100% and mean nothing.

## The gates stay in the code

Calibrated to off, not deleted, for two reasons.

A cross-encoder reranker produces much better-separated scores than the
lexical fallback used for these measurements, and might make a score threshold
viable. And on a different corpus — narrower vocabulary, less boilerplate —
term coverage could genuinely work.

`filingqa calibrate` re-runs the sweep. The thresholds are a function of the
corpus and the reranker, not constants to copy.

## The other two refusal points

Scope is the first of three independent chances to refuse:

1. **Before generation** — scope check, and the confidence gates when enabled.
2. **During generation** — no retrieved sentence scores well enough to quote,
   or the LLM generator returns `INSUFFICIENT_EVIDENCE`.
3. **After generation** — a citation fails verification, or a sentence makes a
   claim with no marker attached.

The third is what makes the citation guarantee hold: an answer that cannot be
sourced is downgraded to a refusal rather than returned with a bad footnote.
On the test split it fired once, and that once was a genuine catch.

Failing at stage 3 is expensive — the work is already done — but it is the
last line between a plausible sentence and a credit memo.
