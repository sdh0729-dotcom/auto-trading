"""Supplement delisted KR stocks' daily bars from pykrx (adjusted prices).

Which stocks were delisted: Toss /stocks/all returns no DELISTED KR symbols, and pykrx's
market-wide ticker lists no longer work (KRX site change), so the list comes from KIND's
delisting table (kind.krx.co.kr, KOSPI + KOSDAQ, KONEX left out). KIND gives a 5-character
company code; the common share's ticker is that code + "0".

Adjusted-price mixing: pykrx rows are split/dividend-adjusted. A symbol is saved with
source="pykrx" only as a whole file; we never splice pykrx rows into a symbol that already
has data from another source.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import httpx
import pandas as pd

from autotrade.data.quality import clean_daily, hard_issues
from autotrade.data.store import data_root, load_daily, load_meta, save_daily

PYKRX_COLUMNS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
CALL_INTERVAL_S = 0.5
log = logging.getLogger(__name__)


KIND_URL = "https://kind.krx.co.kr/investwarn/delcompany.do"
KIND_BOARDS = {"유가증권": "KOSPI", "코스닥": "KOSDAQ"}
_KIND_ROW = re.compile(
    r"alt='(유가증권|코스닥|코넥스)'.*?companysummary_open\('(\w+)'\).*?title='([^']*)'"
    r".*?<td class=\"txc\">(\d{4}-\d{2}-\d{2})</td>", re.S)
_KIND_TOTAL = re.compile(r'<td class="first txc" scope="row">(\d+)</td>')
KIND_META_COLUMNS = ["name", "security_type", "product", "is_common_share", "delisted",
                     "source", "market_board", "isin"]


class KindError(Exception):
    """KIND's delisting table could not be fetched or parsed completely."""


def _kind_html(since: date, until: date) -> str:
    form = {"method": "searchDelCompanySub", "currentPageSize": "5000", "pageIndex": "1",
            "orderMode": "2", "orderStat": "D", "marketType": "", "searchCorpName": "",
            "fromDate": since.isoformat(), "toDate": until.isoformat()}
    r = httpx.post(KIND_URL, data=form, timeout=60.0, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def parse_kind(html: str) -> pd.DataFrame:
    """KOSPI/KOSDAQ rows of a KIND delisting page, indexed by ticker (code + "0"), with
    name, market_board and delisted_on. Raises KindError when the row count differs from the
    page's own numbering (a truncated or changed page)."""
    rows = _KIND_ROW.findall(html)
    numbers = [int(n) for n in _KIND_TOTAL.findall(html)]
    if not numbers or len(rows) != max(numbers):
        raise KindError(f"KIND page parsed {len(rows)} rows, numbered up to "
                        f"{max(numbers) if numbers else 'none'}")
    out = [{"symbol": code + "0", "name": name.strip(), "market_board": KIND_BOARDS[board],
            "delisted_on": day} for board, code, name, day in rows if board in KIND_BOARDS]
    df = pd.DataFrame(out, columns=["symbol", "name", "market_board", "delisted_on"])
    return df.drop_duplicates("symbol", keep="first").set_index("symbol")


def kind_delisted(since: date, until: date | None = None) -> pd.DataFrame:
    return parse_kind(_kind_html(since, until or date.today()))


def add_delisted_meta(meta: pd.DataFrame, kind: pd.DataFrame) -> pd.DataFrame:
    """`meta` plus the KIND-delisted common shares it does not already list (delisted=True,
    product stock, source 'kind')."""
    new = kind.loc[~kind.index.isin(meta.index)]
    if new.empty:
        return meta
    rows = pd.DataFrame({"name": new["name"], "security_type": "STOCK", "product": "stock",
                         "is_common_share": True, "delisted": True, "source": "kind",
                         "market_board": new["market_board"], "isin": None},
                        index=new.index)[KIND_META_COLUMNS]
    return pd.concat([meta, rows]).sort_index()


@dataclass
class SupplementReport:
    fetched: int = 0
    empty: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _pykrx_ohlcv(start_str: str, end_str: str, symbol: str, adjusted: bool) -> pd.DataFrame:
    from pykrx import stock

    return stock.get_market_ohlcv(start_str, end_str, symbol, adjusted=adjusted)


def fetch_delisted_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    raw = _pykrx_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), symbol, adjusted=True)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "source"])
    df = raw.rename(columns=PYKRX_COLUMNS)[list(PYKRX_COLUMNS.values())].copy()
    df = clean_daily(df)            # suspension days (O/H/L=0, volume=0) are not real bars
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df.index.name = "date"
    df["source"] = "pykrx"
    return df


def supplement_delisted_kr(start: date, symbols: list[str] | None = None) -> SupplementReport:
    """Supplement every delisted symbol in the KR meta, or only those of `symbols`."""
    meta = load_meta("kr")
    delisted = [str(s) for s in meta.index[meta["delisted"].astype(bool)]]
    symbols = delisted if symbols is None else [s for s in delisted if s in set(symbols)]
    end = date.today()
    report = SupplementReport()
    for i, sym in enumerate(symbols):
        existing = load_daily("kr", sym)
        if existing is not None and not existing["source"].eq("pykrx").all():
            report.skipped.append(sym)
            continue  # never replace/mix with another source's (differently adjusted) data
        if i:
            time.sleep(CALL_INTERVAL_S)
        try:
            df = fetch_delisted_daily(sym, start, end)
            if df.empty:
                report.empty.append(sym)
                continue
            if hard_issues(df):
                report.failed.append(sym)
                continue
            save_daily("kr", sym, df)
            report.fetched += 1
        except Exception as e:      # noqa: BLE001 - pykrx/KRX errors carry no secrets
            log.warning("pykrx delisted %s failed: %s: %s", sym, type(e).__name__, e)
            report.failed.append(sym)
    return report


def supplement_record_path(market: str = "kr") -> Path:
    return data_root() / market / "supplement.json"


def save_supplement_record(report: SupplementReport, since: date, requested: int,
                           market: str = "kr", day: date | None = None) -> Path:
    """What the last `lab collect` supplement actually did (counts, never symbols' data):
    {date, since, requested, fetched, empty, failed, skipped}. Atomic."""
    path = supplement_record_path(market)
    rec = {"date": (day or date.today()).isoformat(), "since": since.isoformat(),
           "requested": int(requested), "fetched": int(report.fetched),
           "empty": len(report.empty), "failed": len(report.failed),
           "skipped": len(report.skipped)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_supplement_record(market: str = "kr") -> dict | None:
    """The last supplement record, or None when missing or unreadable."""
    try:
        rec = json.loads(supplement_record_path(market).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None
