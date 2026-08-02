"""Gold evaluation set.

**How this was built, stated plainly, because it changes how the numbers
should be read.**

The questions are template-generated from the corpus and the relevance labels
are assigned programmatically. They are not expert annotations. I am not a
credit analyst and pretending 100 hand-authored questions came from one would
be the kind of claim this whole project exists to argue against.

What that buys, and what it costs:

*Buys* - the labels are derived from **structure**, not from similarity to the
question. A passage is relevant because it sits in the right filing, the right
Item, and mentions the topic; never because a retriever scored it highly. So
the eval cannot silently reward the system for agreeing with itself, which is
the standard failure of an LLM-generated benchmark.

*Costs* - a known lexical bias. A generated question about "cybersecurity risk"
contains the word "cybersecurity", and the passages labelled relevant contain
it too. That flatters BM25 specifically, which is why the ablation table
reports a BM25-only arm: if lexical overlap were doing all the work, BM25
alone would match the full pipeline. It does not, and the gap is the honest
measure of what the rest contributes.

Unanswerable questions are the part I trust most, because they need no
judgement: a question about a company not in the corpus, or a fiscal year
outside its range, is unanswerable as a matter of fact rather than opinion.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from ..models import Chunk

# --------------------------------------------------------------------------
# Question types
# --------------------------------------------------------------------------


@dataclass
class GoldQuestion:
    question_id: str
    question: str
    category: str
    answerable: bool
    #: chunk_id -> graded relevance. 3 = directly answers, 2 = supports,
    #: 1 = related context. Recall is measured at >= 2 by default.
    relevance: dict[str, int] = field(default_factory=dict)
    company: str = ""
    fiscal_year: int = 0
    item: str = ""
    #: Why this question is unanswerable, for the refusal report.
    unanswerable_reason: str = ""
    notes: str = ""
    #: "dev" or "test". Thresholds and parameters are tuned on dev only; the
    #: published headline numbers come from test. Without this the reported
    #: recall is the recall of a system tuned on the questions it is being
    #: scored against, which is not a measurement.
    split: str = "test"

    @property
    def gold_chunks(self) -> set[str]:
        return {cid for cid, grade in self.relevance.items() if grade >= 2}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldQuestion:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


COMPANY_NAMES = {"AAPL": "Apple", "JPM": "JPMorgan Chase", "F": "Ford"}


class Topic(TypedDict):
    """A prose topic: what makes a passage about it, and how to ask."""

    key: str
    item: str
    terms: list[str]
    questions: list[str]


class TableTopic(TypedDict):
    """A numeric topic, asked against tables only."""

    key: str
    terms: list[str]
    questions: list[str]


#: Topics with the vocabulary that identifies a passage as being about them.
#: The trigger terms are what assign relevance; the question phrasing is
#: deliberately *not* built from them verbatim, to keep some distance between
#: the query and the labelled passages.
TOPICS: list[Topic] = [
    {
        "key": "cybersecurity",
        "item": "Item 1A",
        "terms": ["cyber", "cybersecurity", "information security", "data breach", "hacking"],
        "questions": [
            "What does {company} identify as its principal information-security exposures in FY{year}?",
            "How does {company} describe the threat of a security breach in its FY{year} annual report?",
        ],
    },
    {
        "key": "supply_chain",
        "item": "Item 1A",
        "terms": ["supply chain", "supplier", "component", "outsourcing partner", "manufacturing"],
        "questions": [
            "What supplier and manufacturing dependencies does {company} flag as a risk in FY{year}?",
            "How concentrated is {company}'s manufacturing base according to its FY{year} filing?",
        ],
    },
    {
        "key": "climate",
        "item": "Item 1A",
        "terms": ["climate", "greenhouse gas", "emissions", "environmental regulation", "carbon"],
        "questions": [
            "What environmental and climate-related risks does {company} disclose for FY{year}?",
            "How does {company} describe the regulatory pressure around emissions in FY{year}?",
        ],
    },
    {
        "key": "interest_rate",
        "item": "Item 7A",
        "terms": ["interest rate", "basis points", "yield", "rate sensitivity", "net interest"],
        "questions": [
            "How sensitive is {company} to a change in interest rates as of FY{year}?",
            "What does {company} disclose about interest-rate exposure in FY{year}?",
        ],
    },
    {
        "key": "legal",
        "item": "Item 3",
        "terms": ["legal proceeding", "litigation", "lawsuit", "court", "claim", "antitrust"],
        "questions": [
            "What material legal proceedings was {company} involved in during FY{year}?",
            "Does {company}'s FY{year} filing disclose any significant litigation?",
        ],
    },
    {
        "key": "competition",
        "item": "Item 1",
        "terms": ["competition", "competitive", "competitors", "market share", "pricing pressure"],
        "questions": [
            "How does {company} characterise competition in its markets in FY{year}?",
            "What competitive pressures does {company} describe in its FY{year} business section?",
        ],
    },
    {
        "key": "human_capital",
        "item": "Item 1",
        "terms": ["human capital", "employees", "workforce", "talent", "headcount"],
        "questions": [
            "What does {company} report about its workforce in FY{year}?",
            "How does {company} describe talent and retention in its FY{year} filing?",
        ],
    },
    {
        "key": "controls",
        "item": "Item 9A",
        "terms": [
            "internal control",
            "disclosure controls",
            "financial reporting",
            "effectiveness",
        ],
        "questions": [
            "What conclusion did {company} management reach about disclosure controls in FY{year}?",
            "Were {company}'s internal controls over financial reporting effective in FY{year}?",
        ],
    },
    {
        "key": "liquidity",
        "item": "Item 7",
        "terms": [
            "liquidity",
            "cash flow",
            "capital resources",
            "credit facility",
            "commercial paper",
        ],
        "questions": [
            "How does {company} describe its liquidity position in FY{year}?",
            "What sources of capital does {company} rely on according to its FY{year} MD&A?",
        ],
    },
    {
        "key": "segments",
        "item": "Item 7",
        "terms": ["segment", "reportable segment", "operating segment", "business segment"],
        "questions": [
            "How did {company}'s reporting segments perform in FY{year}?",
            "What segments does {company} report results for in FY{year}?",
        ],
    },
]

#: Numeric questions asked against tables. The trigger terms pick the table;
#: the question asks for the figure in analyst language.
TABLE_TOPICS: list[TableTopic] = [
    {
        "key": "revenue_table",
        "terms": ["net sales", "total revenue", "revenues", "total net revenue"],
        "questions": [
            "What total revenue did {company} report for FY{year}?",
            "What were {company}'s top-line results in FY{year}?",
        ],
    },
    {
        "key": "property_table",
        "terms": ["property", "equipment", "depreciation", "land and buildings"],
        "questions": [
            "What does {company} report for property and equipment in FY{year}?",
        ],
    },
    {
        "key": "debt_table",
        "terms": ["term debt", "long-term debt", "notes payable", "maturities"],
        "questions": [
            "What is the maturity profile of {company}'s debt as of FY{year}?",
            "How much term debt did {company} have outstanding in FY{year}?",
        ],
    },
    {
        "key": "shares_table",
        "terms": ["shares", "repurchase", "common stock", "dividends declared"],
        "questions": [
            "What did {company} report about share repurchases in FY{year}?",
        ],
    },
]

#: Companies that exist and are plausible to ask about, but are not indexed.
OUT_OF_CORPUS_COMPANIES = [
    ("Tesla", "TSLA"),
    ("Microsoft", "MSFT"),
    ("Amazon", "AMZN"),
    ("General Motors", "GM"),
    ("Goldman Sachs", "GS"),
    ("Nvidia", "NVDA"),
    ("Walmart", "WMT"),
    ("Boeing", "BA"),
    ("Intel", "INTC"),
    ("Netflix", "NFLX"),
]

#: Questions that are financial, plausible, and simply not in a 10-K.
OUT_OF_SCOPE_TOPICS = [
    "What price target do analysts have for {company} stock?",
    "What was {company}'s share price on the last day of FY{year}?",
    "How many patents did {company} file with the EPO in FY{year}?",
    "What is the median employee commute time at {company}?",
    "What did {company}'s CEO say on the FY{year} earnings call?",
    "How does {company}'s valuation multiple compare to its peers?",
]


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def _matches(chunk: Chunk, terms: list[str]) -> int:
    text = chunk.text.lower()
    return sum(1 for term in terms if term in text)


def _grade(hits: int, is_best: bool, *, min_terms: int = 2) -> int:
    """Graded relevance from topic-term coverage.

    Uncapped on purpose. An earlier version labelled only the top N passages
    by term count, which produced a subtle and self-serving bug: a passage
    that genuinely supported the answer but ranked 13th by term frequency was
    labelled irrelevant, so a retriever that found it was scored as *wrong*.
    That is measuring the label, not the system.

    The rule now is structural and complete: every passage in the target
    filing and section containing at least `min_terms` of the topic's defining
    vocabulary is relevant. Gold sets get larger - which is disclosed in the
    methodology, since it makes recall easier - but they stop punishing
    correct retrievals.
    """
    if is_best:
        return 3
    return 2 if hits >= min_terms else 1


class GoldSetBuilder:
    """Builds the gold set from the indexed corpus.

    Deterministic given a seed, so the published numbers refer to a specific
    set of questions that anyone can regenerate.
    """

    def __init__(self, chunks: list[Chunk], *, seed: int = 11) -> None:
        self.chunks = chunks
        self.rng = random.Random(seed)
        self.companies = sorted({c.company for c in chunks if c.company})
        self.years = sorted({c.fiscal_year for c in chunks if c.fiscal_year})

    def _candidates(
        self, company: str, year: int, item: str, terms: list[str], tables: bool | None
    ):
        pool = [
            c
            for c in self.chunks
            if c.company == company
            and c.fiscal_year == year
            and (item == "" or c.item == item)
            and (tables is None or c.is_table == tables)
        ]
        scored = [(c, _matches(c, terms)) for c in pool]
        return sorted([(c, h) for c, h in scored if h > 0], key=lambda x: -x[1])

    def build_topic_questions(self, limit: int) -> list[GoldQuestion]:
        questions: list[GoldQuestion] = []
        combos = [
            (topic, company, year)
            for topic in TOPICS
            for company in self.companies
            for year in self.years
        ]
        self.rng.shuffle(combos)

        for topic, company, year in combos:
            if len(questions) >= limit:
                break
            hits = self._candidates(company, year, topic["item"], topic["terms"], tables=False)
            # Fewer than two supporting passages means the filing barely
            # covers this topic, and a question about it would be measuring
            # the label rather than the retriever.
            if len(hits) < 2:
                continue

            relevance = {
                chunk.chunk_id: _grade(count, index == 0)
                for index, (chunk, count) in enumerate(hits)
            }
            if sum(1 for g in relevance.values() if g >= 2) < 2:
                continue

            template = self.rng.choice(topic["questions"])
            questions.append(
                GoldQuestion(
                    question_id=f"t-{len(questions):03d}",
                    question=template.format(company=COMPANY_NAMES[company], year=year),
                    category=topic["key"],
                    answerable=True,
                    relevance=relevance,
                    company=company,
                    fiscal_year=year,
                    item=topic["item"],
                    notes=f"{len(hits)} passages in {topic['item']} match the topic vocabulary",
                )
            )
        return questions

    def build_table_questions(self, limit: int) -> list[GoldQuestion]:
        questions: list[GoldQuestion] = []
        combos = [
            (topic, company, year)
            for topic in TABLE_TOPICS
            for company in self.companies
            for year in self.years
        ]
        self.rng.shuffle(combos)

        for topic, company, year in combos:
            if len(questions) >= limit:
                break
            hits = self._candidates(company, year, "", topic["terms"], tables=True)
            if not hits:
                continue
            # Table topics use a narrower vocabulary than prose topics - a
            # revenue table says "net sales" once, not four times - so one
            # matching term is enough to mark a table relevant.
            relevance = {
                chunk.chunk_id: _grade(count, index == 0, min_terms=1)
                for index, (chunk, count) in enumerate(hits)
            }
            if sum(1 for g in relevance.values() if g >= 2) < 1:
                continue

            template = self.rng.choice(topic["questions"])
            questions.append(
                GoldQuestion(
                    question_id=f"n-{len(questions):03d}",
                    question=template.format(company=COMPANY_NAMES[company], year=year),
                    category=topic["key"],
                    answerable=True,
                    relevance=relevance,
                    company=company,
                    fiscal_year=year,
                    notes="numeric question; gold passages are tables",
                )
            )
        return questions

    def build_comparison_questions(self, limit: int) -> list[GoldQuestion]:
        """Cross-filing questions: same topic, two companies or two years.

        These are where a naive filter hurts most - restrict to one company
        and half the answer is unreachable - so they are worth measuring
        separately.
        """
        questions: list[GoldQuestion] = []
        topics = [
            t
            for t in TOPICS
            if t["key"] in ("liquidity", "competition", "cybersecurity", "segments")
        ]

        for topic in topics:
            if len(questions) >= limit:
                break
            # Same company, two years.
            company = self.rng.choice(self.companies)
            year_a, year_b = self.years[0], self.years[-1]
            relevance: dict[str, int] = {}
            for year in (year_a, year_b):
                hits = self._candidates(company, year, topic["item"], topic["terms"], tables=None)
                for index, (chunk, count) in enumerate(hits):
                    relevance[chunk.chunk_id] = _grade(count, index == 0)
            if sum(1 for g in relevance.values() if g >= 2) < 2:
                continue
            questions.append(
                GoldQuestion(
                    question_id=f"c-{len(questions):03d}",
                    question=(
                        f"How did {COMPANY_NAMES[company]}'s disclosure on "
                        f"{topic['key'].replace('_', ' ')} change between FY{year_a} and FY{year_b}?"
                    ),
                    category=f"compare_years_{topic['key']}",
                    answerable=True,
                    relevance=relevance,
                    company=company,
                    item=topic["item"],
                    notes="spans two fiscal years",
                )
            )

            if len(questions) >= limit:
                break
            # Two companies, same year.
            year = self.rng.choice(self.years)
            pair = self.rng.sample(self.companies, 2)
            relevance = {}
            for comp in pair:
                hits = self._candidates(comp, year, topic["item"], topic["terms"], tables=None)
                for index, (chunk, count) in enumerate(hits):
                    relevance[chunk.chunk_id] = _grade(count, index == 0)
            if sum(1 for g in relevance.values() if g >= 2) < 2:
                continue
            questions.append(
                GoldQuestion(
                    question_id=f"c-{len(questions):03d}",
                    question=(
                        f"Compare how {COMPANY_NAMES[pair[0]]} and {COMPANY_NAMES[pair[1]]} "
                        f"describe {topic['key'].replace('_', ' ')} in FY{year}."
                    ),
                    category=f"compare_companies_{topic['key']}",
                    answerable=True,
                    relevance=relevance,
                    fiscal_year=year,
                    item=topic["item"],
                    notes="spans two companies",
                )
            )
        return questions

    def build_unanswerable(self, limit: int) -> list[GoldQuestion]:
        questions: list[GoldQuestion] = []
        year = self.years[-1]

        # 1. Companies not in the corpus.
        for name, ticker in OUT_OF_CORPUS_COMPANIES:
            if len(questions) >= limit:
                break
            template = self.rng.choice(
                [
                    "What total revenue did {name} report for FY{year}?",
                    "What risk factors does {name} disclose about competition in FY{year}?",
                    "How does {name} describe its liquidity position in FY{year}?",
                ]
            )
            questions.append(
                GoldQuestion(
                    question_id=f"u-{len(questions):03d}",
                    question=template.format(name=name, year=year),
                    category="unanswerable_company",
                    answerable=False,
                    unanswerable_reason=f"{ticker} is not in the corpus",
                )
            )

        # 2. Fiscal years outside the corpus range.
        missing_years = [
            self.years[0] - 2,
            self.years[0] - 1,
            self.years[-1] + 1,
            self.years[-1] + 2,
        ]
        for missing in missing_years:
            for company in self.companies:
                if len(questions) >= limit:
                    break
                questions.append(
                    GoldQuestion(
                        question_id=f"u-{len(questions):03d}",
                        question=(
                            f"What did {COMPANY_NAMES[company]} report as total revenue "
                            f"in fiscal {missing}?"
                        ),
                        category="unanswerable_year",
                        answerable=False,
                        company=company,
                        fiscal_year=missing,
                        unanswerable_reason=(
                            f"corpus covers FY{self.years[0]}-FY{self.years[-1]}, not {missing}"
                        ),
                    )
                )

        # 3. Questions a 10-K structurally does not answer. These are the hard
        #    ones - the company and year are in the corpus, so scope detection
        #    cannot help and the confidence signals have to carry it.
        for template in OUT_OF_SCOPE_TOPICS:
            for company in self.companies:
                if len(questions) >= limit:
                    break
                questions.append(
                    GoldQuestion(
                        question_id=f"u-{len(questions):03d}",
                        question=template.format(company=COMPANY_NAMES[company], year=year),
                        category="unanswerable_topic",
                        answerable=False,
                        company=company,
                        fiscal_year=year,
                        unanswerable_reason="a 10-K does not contain this information",
                    )
                )

        return questions[:limit]

    def build(
        self,
        *,
        topic: int = 40,
        table: int = 18,
        comparison: int = 8,
        unanswerable: int = 35,
    ) -> list[GoldQuestion]:
        """100 questions: 65 answerable, 35 not.

        The comparison builder yields fewer than requested when a topic lacks
        supporting passages in both filings being compared - a question whose
        gold set is one-sided measures nothing - so the totals are checked in
        `tests/test_goldset.py` rather than assumed here.
        """
        questions = (
            self.build_topic_questions(topic)
            + self.build_table_questions(table)
            + self.build_comparison_questions(comparison)
            + self.build_unanswerable(unanswerable)
        )
        for index, question in enumerate(questions):
            question.question_id = f"q{index:03d}"

        self._assign_splits(questions)
        return questions

    @staticmethod
    def _assign_splits(questions: list[GoldQuestion], dev_fraction: float = 0.4) -> None:
        """Stratified dev/test split.

        Stratified by category *and* answerability, because the categories are
        unevenly sized: a random split can easily put every table question in
        dev and leave the test set unable to measure the thing that was
        hardest. Deterministic, so the split is a property of the gold set
        rather than of when it was generated.
        """
        buckets: dict[tuple[str, bool], list[GoldQuestion]] = {}
        for question in questions:
            buckets.setdefault((question.category, question.answerable), []).append(question)

        for (category, _), group in sorted(buckets.items()):
            rng = random.Random(f"split:{category}")
            ordered = sorted(group, key=lambda q: q.question_id)
            rng.shuffle(ordered)
            cutoff = round(len(ordered) * dev_fraction)
            for question in ordered[:cutoff]:
                question.split = "dev"
            for question in ordered[cutoff:]:
                question.split = "test"


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_goldset(questions: list[GoldQuestion], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(q.to_dict(), ensure_ascii=False) for q in questions) + "\n"
    )
    return path


def load_goldset(path: Path | str) -> list[GoldQuestion]:
    return [
        GoldQuestion.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def goldset_summary(questions: list[GoldQuestion]) -> dict[str, Any]:
    answerable = [q for q in questions if q.answerable]
    categories: dict[str, int] = {}
    for question in questions:
        categories[question.category] = categories.get(question.category, 0) + 1
    gold_counts = [len(q.gold_chunks) for q in answerable]
    return {
        "total": len(questions),
        "answerable": len(answerable),
        "unanswerable": len(questions) - len(answerable),
        "dev": sum(1 for q in questions if q.split == "dev"),
        "test": sum(1 for q in questions if q.split == "test"),
        "categories": dict(sorted(categories.items())),
        "mean_gold_chunks": round(sum(gold_counts) / len(gold_counts), 2) if gold_counts else 0,
        "min_gold_chunks": min(gold_counts) if gold_counts else 0,
    }
