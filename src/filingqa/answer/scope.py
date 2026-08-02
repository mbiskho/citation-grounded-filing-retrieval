"""Out-of-corpus detection.

The refusal that matters most is the one for a question the corpus simply
cannot answer: "What was Tesla's 2024 revenue?" against a corpus of Apple,
JPMorgan and Ford. Retrieval will still return ten chunks - it always returns
ten chunks - and they will be about revenue, and they will look plausible.

Generic confidence signals do not catch this. Term-coverage in particular is
useless here: ask "what is the capital of France" and both "capital" and
"France" appear all over a bank's 10-K, so coverage reads 100% on a question
the corpus has nothing to say about.

So scope detection is explicit and narrow rather than clever:

* **Entity check.** A recognised company that is not in the corpus is an
  immediate refusal. This uses a gazetteer, which means it only catches
  companies it knows about. That is a real limit and it is stated in the docs
  rather than hidden - a question about an obscure private company will not be
  caught here and has to fall through to the confidence checks.
* **Period check.** A fiscal year outside the corpus range is an immediate
  refusal. This one is exact, not heuristic, and it catches the most common
  realistic failure: asking a three-year corpus about year four.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Companies the system recognises by name but may not hold filings for.
#: Deliberately a fixed list: entity linking over free text is its own
#: project, and a wrong link here causes a false refusal, which is a worse
#: user experience than a miss.
KNOWN_COMPANIES = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "jpmorgan chase": "JPM",
    "jpm": "JPM",
    "ford": "F",
    "ford motor": "F",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "citigroup": "C",
    "wells fargo": "WFC",
    "general motors": "GM",
    "gm": "GM",
    "toyota": "TM",
    "rivian": "RIVN",
    "lucid": "LCID",
    "stellantis": "STLA",
    "boeing": "BA",
    "intel": "INTC",
    "walmart": "WMT",
    "exxon": "XOM",
    "chevron": "CVX",
    "pfizer": "PFE",
    "johnson & johnson": "JNJ",
    "berkshire": "BRK",
    "coca-cola": "KO",
    "pepsico": "PEP",
    "disney": "DIS",
    "starbucks": "SBUX",
    "nike": "NKE",
}

_YEAR = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class ScopeVerdict:
    in_scope: bool
    reason: str = ""
    detected_companies: tuple[str, ...] = ()
    detected_years: tuple[int, ...] = ()


class CorpusScope:
    """What the corpus actually covers, and whether a question falls inside it."""

    def __init__(self, companies: set[str], years: set[int]) -> None:
        self.companies = companies
        self.years = years

    @classmethod
    def from_chunks(cls, chunks) -> CorpusScope:
        return cls(
            companies={c.company for c in chunks if c.company},
            years={c.fiscal_year for c in chunks if c.fiscal_year},
        )

    def detect_companies(self, question: str) -> list[str]:
        lowered = f" {question.lower()} "
        found: list[str] = []
        # Longest alias first, so "ford motor" is not also matched as "ford"
        # and "jp morgan" does not slip past as two unrelated tokens.
        for alias in sorted(KNOWN_COMPANIES, key=len, reverse=True):
            if re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", lowered):
                ticker = KNOWN_COMPANIES[alias]
                if ticker not in found:
                    found.append(ticker)
        return found

    def detect_years(self, question: str) -> list[int]:
        return sorted({int(m.group()) for m in _YEAR.finditer(question)})

    def check(self, question: str) -> ScopeVerdict:
        companies = self.detect_companies(question)
        years = self.detect_years(question)

        unknown = [t for t in companies if t not in self.companies]
        if unknown and len(unknown) == len(companies):
            # Every company named is outside the corpus. If even one is inside
            # it is a comparison question and retrieval can still contribute,
            # so it is not refused here.
            return ScopeVerdict(
                False,
                f"the corpus does not contain filings for {', '.join(unknown)}; "
                f"it covers {', '.join(sorted(self.companies))}",
                tuple(companies),
                tuple(years),
            )

        if years and self.years:
            out_of_range = [y for y in years if y not in self.years]
            if out_of_range and len(out_of_range) == len(years):
                low, high = min(self.years), max(self.years)
                return ScopeVerdict(
                    False,
                    f"the corpus covers fiscal years {low}-{high}; "
                    f"no filing is present for {', '.join(map(str, out_of_range))}",
                    tuple(companies),
                    tuple(years),
                )

        return ScopeVerdict(True, "", tuple(companies), tuple(years))
