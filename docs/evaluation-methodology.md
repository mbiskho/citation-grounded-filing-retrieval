# Evaluation methodology

## How the gold set was built

100 questions. **Template-generated from the corpus, with programmatic
relevance labels, spot-checked by hand.** They are not expert annotations, and
I am not a credit analyst. Claiming otherwise would be the exact failure mode
this project argues against.

Regenerate with `filingqa goldset` — deterministic given the seed.

### Answerable questions (65)

Three families:

**Topic questions (40).** Ten topics — cybersecurity, supply chain, climate,
interest-rate sensitivity, legal proceedings, competition, human capital,
controls, liquidity, segments — crossed with three companies and three years.
Each topic has a defining vocabulary and a target Item. A question is only
kept if the target filing has at least two passages matching the vocabulary,
because a question with a one-passage gold set measures the label rather than
the retriever.

**Numeric / table questions (18).** Asked against tables only, using narrower
vocabulary (revenue, property, debt, share repurchases).

**Cross-filing comparisons (7).** Same topic across two years or two
companies. Included specifically because they are where metadata filtering
hurts: filter to one company and half the answer becomes unreachable.

### Unanswerable questions (35)

The part I trust most, because they need no judgement:

- **10 out-of-corpus companies** — Tesla, Microsoft, Nvidia and so on. Real
  companies, plausible questions, not in the index.
- **12 out-of-range fiscal years** — FY2021, FY2022, FY2026, FY2027 against a
  FY2023–FY2025 corpus.
- **13 structurally unanswerable** — analyst price targets, share price on a
  date, EPO patent counts, earnings-call content. The company and year *are*
  in the corpus; a 10-K simply does not contain the answer.

That last group is where the system fails, and it is in the set precisely
because it is hard. See [`refusal.md`](refusal.md).

## Relevance labels

Graded 1–3. Recall is measured at **≥ 2**.

- **3** — the passage best matching the topic vocabulary in the target filing
  and section.
- **2** — any other passage in the target filing and section containing at
  least two of the topic's defining terms (one, for the narrower table
  vocabularies).
- **1** — matches a single term. Not counted as relevant.

**The labels are structural, never similarity-based.** A passage is relevant
because of where it sits and what vocabulary it contains — never because a
retriever ranked it highly. This is the property that stops the eval from
rewarding the system for agreeing with itself, which is the standard failure
of an LLM-generated benchmark.

### A bug worth recording

The first version capped gold sets at the top N passages by term count. That
produced a subtle, self-serving error: a passage that genuinely supported the
answer but ranked 13th by term frequency was labelled irrelevant, so a
retriever that found it was scored as **wrong**.

Uncapping the labels raised measured recall by several points without a single
line of retrieval changing. Gold sets went from a mean of 4.7 passages to 8.4
— still only 0.14% of the corpus, so retrieval is not trivial — and stopped
punishing correct behaviour.

## The known bias

**Lexical overlap between question and gold passage.** A generated question
about "cybersecurity risk" contains the word "cybersecurity", and the passages
labelled relevant contain it too. That flatters lexical retrieval.

This is disclosed rather than corrected, and the ablation table is the control:
a BM25-only floor arm is reported so a reader can see how much of the result
is available from keyword matching alone. On the test split that floor is
0.476 against 0.643 for the shipped configuration, so lexical overlap is doing
a lot of work — but not all of it.

A gold set written by an analyst who had never seen the filings would be
better. It would also not be reproducible by anyone cloning the repo.

## Dev / test split

**40% dev, 60% test, stratified by category and answerability.**

Everything tuned — BM25 length normalisation, refusal thresholds, fusion
weights — was tuned on dev. The headline numbers come from test, scored once.

Stratification matters here because the categories are unevenly sized: a
random split can put every table question in dev and leave the test set unable
to measure the hardest category.

The split is deterministic and derived from the category name, so it is a
property of the gold set rather than of when it was generated.

**The dev/test gap is large and worth taking seriously.** The same
configuration scores 0.826 on dev and 0.643 on test. With 42 answerable test
questions, one question is 2.4 points. Any comparison in this repo smaller
than about 5 points is noise.

## Metric definitions

"recall@10" means at least three different things in the RAG literature and
the difference is worth several points.

**recall@k** — fraction of questions where *at least one* passage graded ≥ 2
appears in the top k. The hit-rate definition, and the right one for this
product: an answer needs one good source, not all of them.

**set_recall@k** — fraction of *all* gold passages retrieved. Reported
separately because it is the number that degrades when chunking is bad, and
hiding it would flatter the chunker.

**MRR** — mean reciprocal rank of the first relevant passage. Sensitive to
whether the best passage is at rank 1 or rank 4, which matters when a
generator only reads the top few.

**nDCG@k** — uses graded labels rather than collapsing to binary, so a passage
that directly answers outranks one that merely supports.

**Refusal precision / recall** — from the perspective of refusing. Recall is
the share of unanswerable questions correctly refused; precision is the share
of refusals that were correct. Both are needed: a system that refuses
everything has perfect refusal recall and no product. `answer_rate` on
answerable questions is reported alongside for exactly that reason.

## What the numbers do not show

- **Answer quality.** Retrieval and citation integrity are measured. Whether
  the quoted sentence is the *best* sentence is not, and would need human
  judgement.
- **Real API embeddings.** The published run uses the offline stack. See the
  README.
- **Anything beyond three companies and three years.** Nine filings from three
  sectors is enough to catch parser assumptions that only hold for one filer —
  which it did — and nowhere near enough to claim the numbers generalise.
