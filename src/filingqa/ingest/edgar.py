"""SEC EDGAR client.

Two things EDGAR will punish you for and both are in here:

* **The User-Agent.** EDGAR blocks requests without a descriptive one
  containing contact details. This is stated in their access policy and
  enforced with a 403, not a warning.
* **The rate limit.** 10 requests/second, and they mean it. A tight loop over
  a company's filing history gets the IP throttled, so every request goes
  through a shared minimum-interval gate.

Everything is cached on disk. A 10-K is 1-2MB of HTML and re-downloading it
because a parser changed is rude to a public service that costs nothing to
use.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..models import FilingRef

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
MIN_INTERVAL_S = 0.15  # ~7 req/s, comfortably under EDGAR's limit


@dataclass
class Company:
    cik: str
    ticker: str
    name: str

    @property
    def padded_cik(self) -> str:
        return self.cik.zfill(10)


#: The default corpus. Chosen for parser diversity rather than name
#: recognition: a tech company, a bank and an industrial have genuinely
#: different filing structures, and a retriever tuned on one of them will
#: quietly fail on the others. The bank in particular is mostly tables.
DEFAULT_COMPANIES = [
    Company(cik="320193", ticker="AAPL", name="Apple Inc."),
    Company(cik="19617", ticker="JPM", name="JPMorgan Chase & Co."),
    Company(cik="37996", ticker="F", name="Ford Motor Company"),
]


class EdgarClient:
    def __init__(
        self,
        *,
        user_agent: str = "filingqa-research contact@example.com",
        cache_dir: Path | str = "data/raw",
        min_interval_s: float = MIN_INTERVAL_S,
        timeout: float = 60.0,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError(
                "EDGAR requires a User-Agent with contact details, e.g. "
                "'my-project you@example.com'. Requests without one get a 403."
            )
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.min_interval_s = min_interval_s
        self.timeout = timeout
        self._last_request = 0.0

    # -- transport ---------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request = time.monotonic()

    def _fetch(self, url: str) -> bytes:
        self._throttle()
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    import gzip

                    payload = gzip.decompress(payload)
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                raise RuntimeError(
                    f"EDGAR returned 403 for {url}. This is almost always the User-Agent: "
                    "it must identify you and include an email address."
                ) from exc
            raise

    def _cached(self, key: str, url: str) -> bytes:
        path = self.cache_dir / key
        if path.exists():
            return path.read_bytes()
        payload = self._fetch(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return payload

    # -- API ---------------------------------------------------------------

    def list_filings(
        self, company: Company, *, form: str = "10-K", limit: int = 5
    ) -> list[FilingRef]:
        """Most recent filings of a given form, newest first.

        Pages into the archived shards when needed. The `recent` block holds
        roughly the last thousand filings of *any* type, which for a heavy
        filer like a large bank is a few months of 8-Ks and prospectuses - its
        10-Ks fall off the end. Reading only `recent` silently returns one
        annual report for JPMorgan and three for Apple, and the corpus ends up
        lopsided in a way that is easy to miss.
        """
        payload = self._cached(
            f"submissions-{company.padded_cik}.json",
            SUBMISSIONS_URL.format(cik=company.padded_cik),
        )
        data = json.loads(payload)

        refs = self._refs_from_block(data["filings"]["recent"], company, form)
        for shard in data["filings"].get("files", []):
            if len(refs) >= limit:
                break
            shard_payload = self._cached(
                f"submissions-{shard['name']}",
                f"https://data.sec.gov/submissions/{shard['name']}",
            )
            refs.extend(self._refs_from_block(json.loads(shard_payload), company, form))

        refs.sort(key=lambda r: r.filing_date, reverse=True)
        return refs[:limit]

    @staticmethod
    def _refs_from_block(block: dict, company: Company, form: str) -> list[FilingRef]:
        refs: list[FilingRef] = []
        report_dates = block.get("reportDate", [])
        for i, form_type in enumerate(block.get("form", [])):
            if form_type != form:
                continue
            primary = block["primaryDocument"][i]
            if not primary.endswith((".htm", ".html")):
                continue  # pre-2001 filings are raw .txt; the parser expects HTML
            filing_date = block["filingDate"][i]
            refs.append(
                FilingRef(
                    cik=company.padded_cik,
                    company=company.ticker,
                    ticker=company.ticker,
                    form=form_type,
                    # A 10-K filed in Oct 2025 covers FY2025; one filed in
                    # Feb 2025 covers FY2024. Using the filing year directly
                    # would mislabel every calendar-year filer in the corpus.
                    fiscal_year=_fiscal_year(
                        filing_date, report_dates[i] if i < len(report_dates) else ""
                    ),
                    filing_date=filing_date,
                    accession=block["accessionNumber"][i],
                    primary_document=primary,
                )
            )
        return refs

    def fetch_document(self, ref: FilingRef) -> str:
        raw = self._cached(f"{ref.doc_id}.html", ref.url)
        return raw.decode("utf-8", errors="ignore")


def _fiscal_year(filing_date: str, report_date: str) -> int:
    """Fiscal year the filing covers, taken from the period-of-report date."""
    source = report_date or filing_date
    return int(source[:4])


def default_corpus(client: EdgarClient, *, years: int = 3) -> list[FilingRef]:
    """The committed corpus: three companies, three years each."""
    refs: list[FilingRef] = []
    for company in DEFAULT_COMPANIES:
        refs.extend(client.list_filings(company, form="10-K", limit=years))
    return refs
