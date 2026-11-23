"""Month-partitioned 1-minute bar store. Timestamps are bar END times, tz-aware (market tz)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from autotrade.data.store import data_root, write_parquet_atomic
from autotrade.types import TZ, Market

MINUTE_COLUMNS = ["open", "high", "low", "close", "volume"]


def minute_dir(market: Market, symbol: str) -> Path:
    return data_root() / market / "1m" / symbol


def _empty(market: Market) -> pd.DataFrame:
    return pd.DataFrame(columns=MINUTE_COLUMNS,
                        index=pd.DatetimeIndex([], name="ts", tz=TZ[market]),
                        dtype=float)


def regular_session(df: pd.DataFrame, market_cfg) -> pd.DataFrame:
    """Keep bars whose END time lies in their local day's regular session, open < ts <= close,
    with the day's ACTUAL hours (data/sessions.py: late opens, exam days, half-days); bars of
    a day that is not a session are dropped."""
    if df.empty:
        return df
    from autotrade.data.sessions import session_mask
    return df[session_mask(df.index, market_cfg.market, market_cfg)]


def save_minutes(market: Market, symbol: str, df: pd.DataFrame,
                 newest_first: bool = False) -> None:
    """Merge `df` into the month files, one atomic file at a time. An interrupted call keeps
    the stored range gap-free only if the month order matches where `df` sits: oldest first
    for bars newer than the stored ones (default), `newest_first` for older ones."""
    if df.empty:
        return
    new = df[MINUTE_COLUMNS].copy()
    new.index = pd.DatetimeIndex(new.index, name="ts")
    tz = TZ[market]
    new.index = new.index.tz_localize(tz) if new.index.tz is None else new.index.tz_convert(tz)
    d = minute_dir(market, symbol)
    months = new.index.strftime("%Y-%m")
    for month in sorted(set(months), reverse=newest_first):
        part = new[months == month]
        path = d / f"{month}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            part = pd.concat([old, part])
        part = part[~part.index.duplicated(keep="last")].sort_index()
        write_parquet_atomic(part, path)


def load_minutes(market: Market, symbol: str, start: date, end: date,
                 regular_only: bool = True) -> pd.DataFrame:
    d = minute_dir(market, symbol)
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    first, last = f"{start:%Y-%m}", f"{end:%Y-%m}"
    frames = [pd.read_parquet(p) for p in files if first <= p.stem <= last]
    if not frames:
        return _empty(market)
    df = pd.concat(frames).sort_index()
    tz = TZ[market]
    local_day = df.index.tz_convert(tz).tz_localize(None).normalize()
    df = df[(local_day >= pd.Timestamp(start)) & (local_day <= pd.Timestamp(end))]
    if regular_only:
        from autotrade.config import load_market_raw
        df = regular_session(df, load_market_raw(market))
    return df


def minute_bounds(market: Market, symbol: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    d = minute_dir(market, symbol)
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return None
    lo = pd.read_parquet(files[0]).index.min()
    hi = pd.read_parquet(files[-1]).index.max()
    return lo, hi


def minute_calendar(market: Market) -> pd.DatetimeIndex:
    """Trading days (tz-naive local midnight) with regular-session bars of the market's
    benchmark symbol, from the market's `minute_start`: the day-trade pocket's calendar."""
    from autotrade.config import load_market_raw
    mc = load_market_raw(market)
    if mc.minute_start is None or mc.benchmark is None:
        return pd.DatetimeIndex([], name="date")
    df = load_minutes(market, mc.benchmark, mc.minute_start, date(2100, 1, 1))
    if df.empty:
        return pd.DatetimeIndex([], name="date")
    days = df.index.tz_convert(TZ[market]).tz_localize(None).normalize().unique()
    return pd.DatetimeIndex(days.sort_values(), name="date")
