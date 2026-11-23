"""Month-partitioned 1-minute bar store. Timestamps are bar END times, tz-aware (market tz)."""
from __future__ import annotations

from datetime import date, time
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


def _parse(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def regular_session(df: pd.DataFrame, market_cfg) -> pd.DataFrame:
    """Keep bars whose local end time t satisfies session_open < t <= session_close."""
    if df.empty:
        return df
    local = df.index.tz_convert(TZ[market_cfg.market])
    t = pd.Index(local.time)
    keep = (t > _parse(market_cfg.session_open)) & (t <= _parse(market_cfg.session_close))
    return df[keep]


def save_minutes(market: Market, symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    new = df[MINUTE_COLUMNS].copy()
    new.index = pd.DatetimeIndex(new.index, name="ts")
    tz = TZ[market]
    new.index = new.index.tz_localize(tz) if new.index.tz is None else new.index.tz_convert(tz)
    d = minute_dir(market, symbol)
    months = new.index.strftime("%Y-%m")
    for month in sorted(set(months)):
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
