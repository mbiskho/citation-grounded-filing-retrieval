"""Query expansion over financial vocabulary.

The largest single source of retrieval failure on this corpus is not ranking.
It is that analysts and filings use different words for the same thing:

    analyst says          filing says
    ------------          -----------
    revenue               net sales           (Apple)
    revenue               total net revenue   (JPMorgan)
    top line              net sales / revenues
    R&D                   research and development
    headcount             full-time equivalent employees
    buyback               repurchase of common stock
    debt maturity         contractual maturities of term debt

No amount of reranking fixes a candidate set that never contained the right
passage, and a lexical retriever cannot bridge these on its own. A neural
bi-encoder bridges some of them - that is most of what it buys you here - but
even then, "top-line" is analyst jargon that appears nowhere in a 10-K.

So expansion is a first-class component rather than a hack: the query is
rewritten to include the filing's own vocabulary before it reaches either
retriever. The mapping is hand-built, which makes it narrow and auditable.
It is also the component most likely to need extending for a new sector -
a pharma corpus needs a different table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: analyst phrasing -> phrasings that actually appear in filings.
#: Bidirectional in effect: expanding "revenue" to "net sales" also helps a
#: query that already said "net sales", because the expansion is additive.
SYNONYMS: dict[str, list[str]] = {
    "revenue": ["net sales", "total revenue", "revenues", "total net revenue"],
    "revenues": ["net sales", "total revenue"],
    "top line": ["net sales", "total revenue", "revenues"],
    "top-line": ["net sales", "total revenue", "revenues"],
    "sales": ["net sales", "revenue"],
    "profit": ["net income", "operating income", "earnings"],
    "bottom line": ["net income", "earnings per share"],
    "earnings": ["net income", "earnings per share"],
    "eps": ["earnings per share"],
    "margin": ["gross margin", "operating margin"],
    "r&d": ["research and development"],
    "rd": ["research and development"],
    "capex": ["capital expenditures", "payments for acquisition of property"],
    "buyback": ["repurchase", "share repurchase", "treasury stock"],
    "buybacks": ["repurchase", "share repurchase"],
    "dividend": ["dividends declared", "dividends paid"],
    "headcount": ["employees", "full-time equivalent", "workforce"],
    "workforce": ["employees", "human capital"],
    "debt": ["term debt", "long-term debt", "notes payable", "borrowings"],
    "leverage": ["debt", "capital structure", "borrowings"],
    "liquidity": ["cash and cash equivalents", "capital resources", "credit facility"],
    "cash": ["cash and cash equivalents", "cash flow"],
    "guidance": ["outlook", "expectations"],
    "supply chain": ["supplier", "outsourcing partners", "component"],
    "cyber": ["cybersecurity", "information security", "security breach"],
    "cybersecurity": ["information security", "security breach", "cyber attack"],
    "climate": ["greenhouse gas", "emissions", "environmental"],
    "esg": ["environmental", "sustainability", "greenhouse gas"],
    "litigation": ["legal proceedings", "lawsuit", "claims"],
    "lawsuit": ["legal proceedings", "litigation"],
    "competition": ["competitive", "competitors"],
    "segment": ["reportable segment", "operating segment"],
    "segments": ["reportable segments", "operating segments"],
    "interest rate": ["interest rate risk", "basis points", "rate sensitivity"],
    "controls": ["internal control over financial reporting", "disclosure controls"],
    "auditor": ["independent registered public accounting firm"],
    "properties": ["property, plant and equipment", "facilities"],
    "ppe": ["property, plant and equipment"],
    "goodwill": ["goodwill and intangible assets"],
    "tax": ["provision for income taxes", "effective tax rate"],
    "pension": ["postretirement benefit", "defined benefit"],
    "outlook": ["expectations", "forward-looking"],
    "risk": ["risk factors"],
}

#: Phrases that add nothing to retrieval and dilute the query. Financial
#: questions are full of them: "according to the filing", "as of FY2024".
NOISE = [
    r"\baccording to (?:the |its )?(?:filing|report|10-k|annual report)\b",
    r"\bin (?:its |the )?(?:fy\s?\d{4}|fiscal \d{4})? ?(?:filing|annual report|10-k)\b",
    r"\bwhat (?:does|did|do)\b",
    r"\bhow (?:does|did|do)\b",
    r"\bcan you tell me\b",
    r"\bplease\b",
]
_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE]


@dataclass
class ExpansionResult:
    original: str
    expanded: str
    added: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.expanded


class QueryExpander:
    """Rewrites a question into the filing's vocabulary.

    Expansion is additive - the original wording is always kept - because the
    synonym table is hand-built and therefore wrong sometimes. Replacing the
    query would let one bad mapping destroy a question; adding to it costs a
    little precision at worst.
    """

    def __init__(
        self,
        synonyms: dict[str, list[str]] | None = None,
        *,
        max_expansions: int = 6,
        strip_noise: bool = True,
    ) -> None:
        self.synonyms = synonyms or SYNONYMS
        self.max_expansions = max_expansions
        self.strip_noise = strip_noise
        # Longest keys first so "supply chain" wins over "chain", and
        # "interest rate" over "rate".
        self._keys = sorted(self.synonyms, key=len, reverse=True)

    def expand(self, query: str) -> ExpansionResult:
        cleaned = query
        if self.strip_noise:
            for pattern in _NOISE_RE:
                cleaned = pattern.sub(" ", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        lowered = f" {cleaned.lower()} "
        added: list[str] = []
        for key in self._keys:
            if len(added) >= self.max_expansions:
                break
            if re.search(rf"(?<![\w]){re.escape(key)}(?![\w])", lowered):
                for synonym in self.synonyms[key]:
                    if len(added) >= self.max_expansions:
                        break
                    if synonym.lower() not in lowered and synonym not in added:
                        added.append(synonym)

        expanded = cleaned if not added else f"{cleaned} {' '.join(added)}"
        return ExpansionResult(original=query, expanded=expanded, added=added)

    def __call__(self, query: str) -> str:
        return self.expand(query).expanded
