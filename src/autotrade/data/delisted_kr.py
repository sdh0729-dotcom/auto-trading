"""Supplement delisted KR stocks' daily bars from pykrx (adjusted prices).

Adjusted-price mixing: pykrx rows are split/dividend-adjusted. A symbol is saved with
source="pykrx" only as a whole file; we never splice pykrx rows into a symbol that already
has data from another source.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from autotrade.data.quality import clean_daily, hard_issues
from autotrade.data.store import data_root, load_daily, load_meta, save_daily

PYKRX_COLUMNS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
CALL_INTERVAL_S = 0.5
log = logging.getLogger(__name__)


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
