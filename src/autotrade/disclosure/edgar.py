"""Read-only SEC EDGAR client (company tickers, submissions, insider-transactions data sets).

Endpoints (no API key; SEC requires a descriptive `User-Agent` carrying a contact, kept in
`SEC_USER_AGENT` in .env; at most 10 requests/s -> min_interval 0.12 s):
 - tickers      https://www.sec.gov/files/company_tickers.json  {"0": {cik_str, ticker, title}}
 - submissions  https://data.sec.gov/submissions/CIK##########.json  (`filings.recent` is
   column-oriented; `filings.files` lists older pages, each fetched from
   https://data.sec.gov/submissions/<name> with the same column-oriented shape at top level)
 - insider data sets
   https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{year}q{q}_form345.zip

ASSUMPTIONS (not verifiable offline; the controller checks them against real data):
 - A1 `acceptanceDateTime` ("2024-03-01T16:05:22.000Z") carries a "Z" but is Eastern wall-clock
   time (EDGAR rolls `filingDate` to the next business day for acceptances after 17:30 ET,
   which only lines up if the value is Eastern). `parse_acceptance` is the ONE place that
   encodes this; if it turns out to be true UTC, change it there only.
 - A2 The zip holds NONDERIV_TRANS.tsv (tab separated) with ACCESSION_NUMBER, TRANS_CODE,
   TRANS_SHARES, TRANS_PRICEPERSHARE; a missing column raises EdgarError("insider layout").
   The Form 4 XML fallback (primaryDocument per filing) is NOT implemented: follow-up if A2
   fails at runtime.
 - A3 An insider data set not yet published answers 404: `EdgarNotFound` (ruling R14: 403 and
   every other failure is a plain EdgarError, never "not published", since SEC also answers
   rate excess with 403).

Errors never carry a URL or the User-Agent: only class names and status codes.
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import date

import httpx
import pandas as pd
from dotenv import load_dotenv

__all__ = ["EdgarError", "EdgarRateLimited", "EdgarNotFound", "EdgarClient",
           "make_edgar_client_from_env", "parse_acceptance"]

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS = "https://data.sec.gov/submissions/"
INSIDER_URL = ("https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/"
               "{year}q{q}_form345.zip")
SUBMISSION_FIELDS = ("accessionNumber", "filingDate", "acceptanceDateTime", "form", "items")
INSIDER_COLS = ["accession", "code", "shares", "price"]
NY = "America/New_York"


class EdgarError(Exception):
    """EDGAR failure: a status and message only, never a URL or the User-Agent."""

    def __init__(self, status: str, message: str):
        super().__init__(status, message)
        self.status = status
        self.message = message

    def __str__(self) -> str:
        return f"EDGAR {self.status}: {self.message}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(status={self.status!r}, message={self.message!r})"


class EdgarRateLimited(EdgarError):
    pass


class EdgarNotFound(EdgarError):
    pass


def parse_acceptance(value) -> pd.Timestamp:
    """EDGAR `acceptanceDateTime` -> tz-aware UTC Timestamp (NaT when empty/unparsable).

    ASSUMPTION A1: the wall time in the string (its "Z" notwithstanding) is America/New_York
    time. Ambiguous (fall-back hour) times take the first occurrence, non-existent
    (spring-forward) times shift forward."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
        return pd.NaT
    try:
        ts = pd.Timestamp(str(value).strip())
    except (ValueError, TypeError):
        return pd.NaT
    ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(NY, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")


class EdgarClient:
    def __init__(self, user_agent: str, http: httpx.Client | None = None,
                 min_interval: float = 0.12):
        self._ua = user_agent
        self._http = http if http is not None else httpx.Client(timeout=60.0)
        self._min_interval = min_interval
        self._last = 0.0

    def _get(self, url: str, missing_ok: bool = False) -> httpx.Response:
        wait = self._last + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self._http.get(url, headers={"User-Agent": self._ua,
                                                "Accept-Encoding": "gzip, deflate"})
        except httpx.HTTPError as e:
            self._last = time.monotonic()
            raise EdgarError("http", type(e).__name__) from None
        self._last = time.monotonic()
        code = resp.status_code
        if code == 429:
            raise EdgarRateLimited("429", "rate limited")
        if missing_ok and code == 404:
            raise EdgarNotFound("404", "not available")
        if code != 200:
            raise EdgarError("http", f"HTTP {code}")
        return resp

    def _json(self, url: str):
        try:
            return self._get(url).json()
        except ValueError:
            raise EdgarError("http", "invalid JSON") from None

    def tickers(self) -> pd.DataFrame:
        body = self._json(TICKERS_URL)
        if not isinstance(body, dict):
            raise EdgarError("http", "unexpected tickers body")
        rows = []
        for r in body.values():
            try:
                rows.append({"ticker": str(r["ticker"]).upper(), "cik": f"{int(r['cik_str']):010d}"})
            except (KeyError, TypeError, ValueError):
                raise EdgarError("http", "unexpected tickers body") from None
        return pd.DataFrame(rows, columns=["ticker", "cik"])

    @staticmethod
    def _columns(block) -> list[dict]:
        if not isinstance(block, dict):
            raise EdgarError("http", "unexpected submissions body")
        n = len(block.get("accessionNumber") or [])
        out = []
        cols = {k: (block.get(k) if len(block.get(k) or []) == n else [None] * n)
                for k in SUBMISSION_FIELDS}
        return [{k: cols[k][i] for k in SUBMISSION_FIELDS} for i in range(n)]

    def submissions(self, cik: str, since: date | None = None) -> list[dict]:
        """recent + every older page, merged. `since`: skip older pages that end before it."""
        body = self._json(f"{SUBMISSIONS}CIK{int(cik):010d}.json")
        if not isinstance(body, dict) or not isinstance(body.get("filings"), dict):
            raise EdgarError("http", "unexpected submissions body")
        rows = self._columns(body["filings"].get("recent"))
        for f in body["filings"].get("files") or []:
            if since is not None and str(f.get("filingTo") or "9999")[:10] < since.isoformat():
                continue
            rows.extend(self._columns(self._json(f"{SUBMISSIONS}{f['name']}")))
        return rows

    def insider_quarter(self, year: int, q: int) -> pd.DataFrame:
        """NONDERIV_TRANS of one quarterly data set: accession, code, shares, price."""
        content = self._get(INSIDER_URL.format(year=year, q=q), missing_ok=True).content
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                with z.open("NONDERIV_TRANS.tsv") as f:
                    df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False,
                                     usecols=["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES",
                                              "TRANS_PRICEPERSHARE"])
        except (zipfile.BadZipFile, KeyError, ValueError):
            raise EdgarError("insider layout", "unreadable NONDERIV_TRANS.tsv") from None
        out = pd.DataFrame({"accession": df["ACCESSION_NUMBER"].str.strip(),
                            "code": df["TRANS_CODE"].str.strip().str.upper(),
                            "shares": pd.to_numeric(df["TRANS_SHARES"], errors="coerce"),
                            "price": pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")})
        return out[INSIDER_COLS]


def insider_buy_usd(trans: pd.DataFrame) -> pd.Series:
    """accession -> sum(shares x price) over code "P" rows with both numbers present."""
    p = trans[trans["code"] == "P"]
    amt = (p["shares"] * p["price"]).dropna()
    return amt.groupby(p.loc[amt.index, "accession"]).sum().astype(float)


def make_edgar_client_from_env() -> EdgarClient:
    from autotrade.config import env_path, repo_root
    from autotrade.pipeline import LabError

    load_dotenv(env_path("AUTOTRADE_DOTENV", repo_root() / ".env"), override=False)
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        raise LabError("SEC_USER_AGENT missing in .env")
    return EdgarClient(ua)
