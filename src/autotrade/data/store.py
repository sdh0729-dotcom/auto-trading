from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from autotrade.types import META_COLUMNS, Market, Panel

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
DAILY_COLUMNS = PRICE_COLUMNS + ["source"]
CALENDAR_WINDOW_DAYS = 10     # a symbol counts as active on d if it has a bar within +-10 days
CALENDAR_MIN_SHARE = 0.5      # d is a trading day if >= 50% of the active symbols have a bar


def data_root() -> Path:
    from autotrade.config import env_path, repo_root

    return env_path("AUTOTRADE_DATA", repo_root() / "data")


def _daily_dir(market: Market) -> Path:
    return data_root() / market / "1d"


def _meta_path(market: Market) -> Path:
    return data_root() / market / "meta.parquet"


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file next to `path`, then os.replace: readers see the old file or the
    complete new one, never a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_daily(market: Market, symbol: str, df: pd.DataFrame) -> None:
    """Atomic. `df.attrs` (e.g. `history_from`, see data.collect) are stored with the file."""
    out = df.copy()
    if "source" not in out.columns:
        out["source"] = "toss"
    out = out[DAILY_COLUMNS]
    out.index = pd.DatetimeIndex(out.index, name="date")
    out.attrs = dict(df.attrs)
    write_parquet_atomic(out, _daily_dir(market) / f"{symbol}.parquet")


def load_daily(market: Market, symbol: str) -> pd.DataFrame | None:
    path = _daily_dir(market) / f"{symbol}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_meta(market: Market, df: pd.DataFrame) -> None:
    write_parquet_atomic(df, _meta_path(market))


def load_meta(market: Market) -> pd.DataFrame:
    return pd.read_parquet(_meta_path(market))


def _stored_symbols(market: Market) -> list[str]:
    d = _daily_dir(market)
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def calendar(market: Market) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """(trading dates, dropped dates) over all stored daily files, read without any price.

    A stored date is a trading day when at least CALENDAR_MIN_SHARE of the symbols that have
    any bar within +-CALENDAR_WINDOW_DAYS calendar days have a bar on it; other dates (a
    stray bar on a holiday/weekend) are dropped so they cannot punch NaN holes into every
    other symbol's rolling statistics. Uses only which dates exist, never prices."""
    per_symbol = []
    for sym in _stored_symbols(market):
        idx = pd.read_parquet(_daily_dir(market) / f"{sym}.parquet", columns=["source"]).index
        per_symbol.append(np.unique(idx.values.astype("datetime64[D]")))
    empty = pd.DatetimeIndex([], name="date")
    if not per_symbol:
        return empty, empty
    union = np.unique(np.concatenate(per_symbol))
    w = np.timedelta64(CALENDAR_WINDOW_DAYS, "D")
    active = np.zeros(len(union))
    present = np.zeros(len(union))
    for d in per_symbol:
        active += np.searchsorted(d, union + w, "right") > np.searchsorted(d, union - w, "left")
        present += np.isin(union, d, assume_unique=True)
    keep = present >= CALENDAR_MIN_SHARE * active

    def ix(a):
        return pd.DatetimeIndex(a.astype("datetime64[ns]"), name="date")
    return ix(union[keep]), ix(union[~keep])


def trading_dates(market: Market, start: date, end: date) -> pd.DatetimeIndex:
    """Trading days (see `calendar`) in [start, end]: the index `load_panel` builds."""
    kept, _ = calendar(market)
    return kept[(kept >= pd.Timestamp(start)) & (kept <= pd.Timestamp(end))]


def load_panel(market: Market, symbols: list[str] | None, start: date, end: date) -> Panel:
    """Wide OHLCV panel over [start, end] (inclusive) on the market's trading calendar
    (dates dropped by `calendar` are left out). Days a symbol lacks are NaN."""
    syms = symbols if symbols is not None else _stored_symbols(market)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames: dict[str, pd.DataFrame] = {}
    for sym in syms:
        df = load_daily(market, sym)
        if df is None:
            continue
        frames[sym] = df.loc[(df.index >= s) & (df.index <= e)]
    cols = list(frames)
    index = pd.DatetimeIndex([], name="date")
    if frames:
        have = pd.DatetimeIndex(sorted(set().union(*(f.index for f in frames.values()))))
        index = trading_dates(market, start, end)
        index = index[index.isin(have)]

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
