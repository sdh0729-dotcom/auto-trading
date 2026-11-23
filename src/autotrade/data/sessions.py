"""Per-day regular sessions: the exchange's real open / close for every trading day.

The day-trade lab trims, windows and exits on each day's ACTUAL session, not on the usual
hours: KRX opens at 10:00 on the first trading day of the year and trades 10:00-16:30 on the
college-entrance-exam day; NYSE closes at 13:00 on half-days. Sessions come from the
`exchange_calendars` package (`MarketConfig.exchange_calendar`: XKRX / XNYS), corrected by the
market config's `session_overrides` (sourced dates the package gets wrong). A day that is not
a session has no regular bars at all.

KR closing auction: it starts `session_close - closing_auction_start` minutes (10 in the
config: 15:20 for 15:30) before the day's close, so 16:20 on an exam day; bars in between are
the auction (no continuous trading), and the bar ending at the close carries the auction price.
"""
from __future__ import annotations

import warnings
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

from autotrade.types import TZ, Market

SESSION_COLUMNS = ["open", "close", "auction"]
_CAL_START = "2000-01-01"


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


@lru_cache(maxsize=None)
def _base(code: str, tz: str) -> pd.DataFrame:
    """exchange_calendars schedule as local tz-aware open / close per local session day."""
    import exchange_calendars as xc
    with warnings.catch_warnings():
        # exchange_calendars 4.13.3 reads Series.values of tz-aware data (deprecated in
        # pandas 3.1); a third-party internal, harmless for the schedule it builds
        warnings.filterwarnings("ignore", message="Series.values returning an ndarray",
                                module="pandas")
        sched = xc.get_calendar(code, start=_CAL_START).schedule
    out = pd.DataFrame({"open": sched["open"].dt.tz_convert(tz),
                        "close": sched["close"].dt.tz_convert(tz)})
    out.index = pd.DatetimeIndex(out.index.tz_localize(None) if out.index.tz is not None
                                 else out.index, name="day")
    return out


def auction_lead_minutes(mc) -> int | None:
    """Minutes between the closing-auction start and the close (KR 10), None if no auction."""
    if not mc.closing_auction_start or not mc.session_close:
        return None
    return _minutes(mc.session_close) - _minutes(mc.closing_auction_start)


def sessions(market: Market, start: date | pd.Timestamp, end: date | pd.Timestamp,
             mc=None) -> pd.DataFrame:
    """Sessions with a local day in [start, end]: index `day` (tz-naive local midnight),
    columns open / close / auction (tz-aware local; auction NaT without a closing auction).
    `mc` defaults to the market's raw config."""
    if mc is None:
        from autotrade.config import load_market_raw
        mc = load_market_raw(market)
    if mc.exchange_calendar is None:
        raise ValueError(f"{market} config has no exchange_calendar")
    tz = TZ[market]
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    base = _base(mc.exchange_calendar, tz)
    tab = base[(base.index >= lo) & (base.index <= hi)].copy()
    for d, (o, c) in (mc.session_overrides or {}).items():
        day = pd.Timestamp(d)
        if lo <= day <= hi:
            tab.loc[day, ["open", "close"]] = [pd.Timestamp(f"{day:%Y-%m-%d} {o}", tz=tz),
                                               pd.Timestamp(f"{day:%Y-%m-%d} {c}", tz=tz)]
    tab = tab.sort_index()
    lead = auction_lead_minutes(mc)
    tab["auction"] = (tab["close"] - pd.Timedelta(minutes=lead)) if lead is not None \
        else pd.Series(pd.NaT, index=tab.index, dtype=f"datetime64[ns, {tz}]")
    return tab[SESSION_COLUMNS]


def session_mask(index: pd.DatetimeIndex, market: Market, mc=None) -> np.ndarray:
    """True for END-stamped bars inside their local day's session: open < ts <= close."""
    if not len(index):
        return np.zeros(0, dtype=bool)
    local = index.tz_convert(TZ[market])
    day = local.tz_localize(None).normalize()
    tab = sessions(market, day.min(), day.max(), mc)
    if not len(tab):
        return np.zeros(len(index), dtype=bool)
    pos = tab.index.get_indexer(day)
    found = pos >= 0
    p = np.where(found, pos, 0)
    ts = local.as_unit("ns").asi8
    o = pd.DatetimeIndex(tab["open"]).as_unit("ns").asi8
    c = pd.DatetimeIndex(tab["close"]).as_unit("ns").asi8
    return found & (ts > o[p]) & (ts <= c[p])
