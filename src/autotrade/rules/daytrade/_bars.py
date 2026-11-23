"""Shared helpers for END-stamped 1-minute bar slicing (private to rules/daytrade)."""
from __future__ import annotations

import pandas as pd

from autotrade.types import TZ


def _market_of(bars: pd.DataFrame) -> str:
    tz = str(bars.index.tz)
    for m, name in TZ.items():
        if name == tz:
            return m
    raise ValueError(f"unknown bar timezone {tz}")


def _day_slices(bars: pd.DataFrame):
    """(local day, bars of that day) for END-stamped tz-aware bars."""
    local = bars.index.tz_localize(None)
    day = local.normalize()
    for d, idx in bars.groupby(day).indices.items():
        yield pd.Timestamp(d), bars.iloc[idx]


def _open_ts(day: pd.Timestamp, session_open: str, tz) -> pd.Timestamp:
    return pd.Timestamp(f"{day:%Y-%m-%d} {session_open}", tz=tz)


def _last_at_or_before(dbars: pd.DataFrame, ts: pd.Timestamp):
    sub = dbars[dbars.index <= ts]
    return sub.iloc[-1:] if len(sub) else None
