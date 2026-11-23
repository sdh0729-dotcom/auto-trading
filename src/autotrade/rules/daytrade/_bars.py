"""Shared helpers for END-stamped 1-minute bar slicing (private to rules/daytrade)."""
from __future__ import annotations

import pandas as pd

from autotrade.data.sessions import sessions
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


def _session_days(bars: pd.DataFrame):
    """(local day, bars of that day, session) for the days of END-stamped tz-aware bars that
    are exchange sessions; `session` has the day's ACTUAL open / close / auction (tz-aware,
    auction NaT without a closing auction; data/sessions.py), so first-minutes windows start
    at a late open (KR Jan 2 / exam day 10:00) and close-relative times follow the close."""
    if bars.empty:
        return
    local = bars.index.tz_localize(None).normalize()
    tab = sessions(_market_of(bars), local.min(), local.max())
    for d, db in _day_slices(bars):
        if d in tab.index:
            yield d, db, tab.loc[d]


def _continuous_end(session) -> pd.Timestamp:
    """End of continuous trading: the closing-auction start (KR) or the close (US)."""
    return session["auction"] if pd.notna(session["auction"]) else session["close"]


def _last_at_or_before(dbars: pd.DataFrame, ts: pd.Timestamp):
    sub = dbars[dbars.index <= ts]
    return sub.iloc[-1:] if len(sub) else None


def _first_minutes(dbars: pd.DataFrame, open_ts: pd.Timestamp, minutes: int):
    """The bars ending in (open, open + minutes] when ALL `minutes` of them exist, else None:
    an incomplete window (data hole) has no first-minutes volume, never a partial sum."""
    w = dbars[(dbars.index > open_ts) & (dbars.index <= open_ts + pd.Timedelta(minutes=minutes))]
    return w if len(w) == minutes else None


def _prior_valid_mean(s: pd.Series, window: int) -> pd.Series:
    """Per row: mean of the `window` previous VALID (non-NaN) values, excluding the row itself.
    NaN rows get NaN and are skipped by later rows (a late open or a data hole does not blank
    the next `window` days); fewer than `window` prior values -> NaN; a zero mean -> NaN (so a
    relative-volume ratio or a multiple-of-average filter can never pass on it)."""
    v = s.dropna()
    m = v.shift(1).rolling(window, min_periods=window).mean()
    return m.where(m > 0).reindex(s.index)
