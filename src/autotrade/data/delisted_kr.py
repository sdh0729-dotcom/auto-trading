"""Supplement delisted KR stocks' daily bars from pykrx (adjusted prices).

Adjusted-price mixing: pykrx rows are split/dividend-adjusted. A symbol is saved with
source="pykrx" only as a whole file; we never splice pykrx rows into a symbol that already
has data from another source.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from autotrade.data.quality import hard_issues
from autotrade.data.store import load_daily, load_meta, save_daily

PYKRX_COLUMNS = {"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
CALL_INTERVAL_S = 0.5


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
    # Suspension days come back as O/H/L=0, close=previous close, volume=0: not real bars.
    suspended = (df["volume"] == 0) & (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    df = df[~suspended]
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
        except Exception:
            report.failed.append(sym)
    return report
