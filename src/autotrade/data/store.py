from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd

from autotrade.types import META_COLUMNS, Market, Panel

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
DAILY_COLUMNS = PRICE_COLUMNS + ["source"]


def data_root() -> Path:
    env = os.environ.get("AUTOTRADE_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "data"


def _daily_dir(market: Market) -> Path:
    return data_root() / market / "1d"


def _meta_path(market: Market) -> Path:
    return data_root() / market / "meta.parquet"


def save_daily(market: Market, symbol: str, df: pd.DataFrame) -> None:
    out = df.copy()
    if "source" not in out.columns:
        out["source"] = "toss"
    out = out[DAILY_COLUMNS]
    out.index = pd.DatetimeIndex(out.index, name="date")
    path = _daily_dir(market) / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)


def load_daily(market: Market, symbol: str) -> pd.DataFrame | None:
    path = _daily_dir(market) / f"{symbol}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_meta(market: Market, df: pd.DataFrame) -> None:
    path = _meta_path(market)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_meta(market: Market) -> pd.DataFrame:
    return pd.read_parquet(_meta_path(market))


def _stored_symbols(market: Market) -> list[str]:
    d = _daily_dir(market)
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def trading_dates(market: Market, start: date, end: date) -> pd.DatetimeIndex:
    """Union of stored daily dates in [start, end] (the index `load_panel` would build),
    read without loading any price column."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    dates: set = set()
    for sym in _stored_symbols(market):
        idx = pd.read_parquet(_daily_dir(market) / f"{sym}.parquet", columns=["source"]).index
        dates.update(idx[(idx >= s) & (idx <= e)])
    return pd.DatetimeIndex(sorted(dates), name="date")


def load_panel(market: Market, symbols: list[str] | None, start: date, end: date) -> Panel:
    """Wide OHLCV panel over [start, end] (inclusive). Days a symbol lacks are NaN."""
    syms = symbols if symbols is not None else _stored_symbols(market)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames: dict[str, pd.DataFrame] = {}
    for sym in syms:
        df = load_daily(market, sym)
        if df is None:
            continue
        frames[sym] = df.loc[(df.index >= s) & (df.index <= e)]
    cols = list(frames)
    index = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))),
                             name="date") if frames else pd.DatetimeIndex([], name="date")

    def wide(col: str) -> pd.DataFrame:
        return pd.DataFrame({c: frames[c][col] for c in cols}, index=index, columns=cols,
                            dtype=float)

    try:
        meta = load_meta(market).reindex(cols)
    except FileNotFoundError:
        meta = pd.DataFrame(index=pd.Index(cols, name="symbol"))
    for c in META_COLUMNS:
        if c not in meta.columns:
            meta[c] = pd.Series(dtype=object)
    meta = meta[META_COLUMNS + [c for c in meta.columns if c not in META_COLUMNS]]
    return Panel(wide("open"), wide("high"), wide("low"), wide("close"), wide("volume"), meta)
