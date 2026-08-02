"""Tokenisation for financial text.

A stock tokeniser splits on non-alphanumerics, which destroys exactly the
tokens that matter in a filing:

    "$2,755"   -> ["2", "755"]
    "10.5%"    -> ["10", "5"]
    "(1,234)"  -> ["1", "234"]
    "FY2024"   -> ["fy", "2024"]

The "2" and "755" then match every other number in the corpus, so a query for
a specific figure retrieves noise. Since a large fraction of the questions
anyone asks a 10-K are numeric, this is not a detail.

Negative numbers are the subtle one: filings write them in parentheses, so
"(1,234)" means -1234. Dropping the parentheses turns a loss into a profit.
The sign is preserved in a normalised form so the two are distinguishable.
"""

from __future__ import annotations

import re

_MONEY = re.compile(r"[$€£¥]\s?\(?-?[\d,]+(?:\.\d+)?\)?")
_PERCENT = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?\s?%")
_PAREN_NUM = re.compile(r"\(\s?[\d,]+(?:\.\d+)?\s?\)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

#: Words that appear in nearly every filing sentence and carry no signal.
#: Deliberately short - aggressive stopword lists remove terms like "risk"
#: and "material" that are genuinely discriminative in this domain.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "at",
        "by",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "from",
        "we",
        "our",
        "us",
        "their",
        "his",
        "her",
        "they",
        "them",
        "he",
        "she",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "but",
        "if",
        "then",
        "than",
        "so",
        "such",
        "which",
        "who",
        "whom",
    ]
)


def normalise_number(token: str) -> str:
    """Canonical form for a numeric token, sign preserved.

    "(1,234)" and "-1234" both become "num:-1234" so a query for one finds the
    other, while "1234" stays distinct from its negative.
    """
    raw = token.strip()
    negative = raw.startswith("(") and raw.endswith(")")
    digits = re.sub(r"[^\d.\-]", "", raw)
    if not digits or digits in ("-", "."):
        return ""
    try:
        value = float(digits)
    except ValueError:
        return ""
    if negative:
        value = -abs(value)
    if value == int(value):
        return f"num:{int(value)}"
    return f"num:{value:g}"


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Tokens for BM25: money, percentages and numbers survive intact."""
    tokens: list[str] = []
    consumed: list[tuple[int, int]] = []

    def take(pattern: re.Pattern[str], prefix: str) -> None:
        for match in pattern.finditer(text):
            if any(s <= match.start() < e for s, e in consumed):
                continue
            consumed.append(match.span())
            value = match.group()
            if prefix == "money":
                number = normalise_number(value)
                if number:
                    tokens.append(f"money:{number[4:]}")
                    tokens.append(number)
            elif prefix == "pct":
                number = normalise_number(value.replace("%", ""))
                if number:
                    tokens.append(f"pct:{number[4:]}")
                    tokens.append(number)
            else:
                number = normalise_number(value)
                if number:
                    tokens.append(number)

    # Order matters: money and percent patterns overlap the bare-number one,
    # so they claim their spans first.
    take(_MONEY, "money")
    take(_PERCENT, "pct")
    take(_PAREN_NUM, "num")
    take(_NUMBER, "num")

    for match in _WORD.finditer(text):
        if any(s <= match.start() < e for s, e in consumed):
            continue
        word = match.group().lower()
        if not keep_stopwords and word in STOPWORDS:
            continue
        if len(word) < 2:
            continue
        tokens.append(word)

    return tokens


def char_ngrams(text: str, n: int = 4) -> list[str]:
    """Character n-grams, used by the offline hashing embedder."""
    cleaned = re.sub(r"\s+", " ", text.lower())
    return [cleaned[i : i + n] for i in range(max(0, len(cleaned) - n + 1))]
