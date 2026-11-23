"""Intraday executor: per-symbol day-trade candidates -> same-day trades on 1-minute bars.

Bars are END-stamped: a signal from the bar ending `signal_ts` fills at the open of the next
existing bar (never the signal bar). Prices are raw bar prices; costs come later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrade.types import TRADE_COLUMNS

CANDIDATE_COLUMNS = ["day", "signal_ts", "stop_pct", "target_pct", "exit_ts", "confidence"]


@dataclass(frozen=True)
class VIConfig:
    """Placeholder for KR volatility-interruption halts (fields arrive with Task 4)."""


@dataclass(frozen=True)
class SessionRules:
    last_entry: str | None      # local "HH:MM" bar-END time; entry bar must end at or before it
    exit_lead_minutes: int      # session exit = last bar ending <= last bar end - lead
    vi: VIConfig | None = None  # not used yet


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def _walk(o, h, l, c, i0, i_end, i_sig, stop_pct, target_pct):
    """Walk bars i0..i_end; return (exit_idx, exit_price, reason). Entry at o[i0]."""
    entry = o[i0]
    stop = entry * (1.0 - stop_pct)
    target = entry * (1.0 + target_pct) if np.isfinite(target_pct) else None
    for i in range(i0, i_end + 1):
        if o[i] <= stop:
            return i, o[i], "stop"
        if l[i] <= stop:
            return i, stop, "stop"
        if target is not None:
            if o[i] >= target:
                return i, o[i], "target"
            if h[i] >= target:
                return i, target, "target"
        if i_sig is not None and i == i_sig:
            return i, c[i], "signal"
    return i_end, c[i_end], "session_end"


def execute(symbol: str, bars: pd.DataFrame, cands: pd.DataFrame,
            rules: SessionRules) -> pd.DataFrame:
    rows: list[dict] = []
    if len(cands) and len(bars):
        idx = bars.index
        local_day = pd.DatetimeIndex(idx.normalize().tz_localize(None))
        cands = cands.sort_values("signal_ts", kind="stable")
        last_entry = _minutes(rules.last_entry) if rules.last_entry else None
        for day, group in cands.groupby(cands["day"].dt.normalize(), sort=True):
            sel = np.flatnonzero(local_day == day)
            if len(sel) == 0:
                continue
            ts = idx[sel]
            o, h, l, c = (bars[k].to_numpy(float)[sel] for k in ("open", "high", "low", "close"))
            ends = ts.as_unit("ns").asi8
            lead = pd.Timedelta(minutes=rules.exit_lead_minutes).value
            i_end = int(np.searchsorted(ends, ends[-1] - lead, side="right")) - 1
            free_after = None   # exit bar end (ns) of the open/last trade
            for r in group.itertuples(index=False):
                sig = pd.Timestamp(r.signal_ts)
                if sig.tz_convert(ts.tz).normalize().tz_localize(None) != day:
                    raise ValueError(f"signal_ts {sig} does not belong to day {day.date()}")
                if free_after is not None and sig.value < free_after:
                    continue
                i0 = int(np.searchsorted(ends, sig.value, side="right"))
                if i0 >= len(ts) or i0 > i_end:
                    continue
                if last_entry is not None:
                    t = ts[i0]
                    if t.hour * 60 + t.minute > last_entry:
                        continue
                i_sig = None
                if pd.notna(r.exit_ts):
                    i_sig = max(int(np.searchsorted(ends, pd.Timestamp(r.exit_ts).value,
                                                    side="right")) - 1, i0)
                    i_sig = min(i_sig, i_end)
                i, px, why = _walk(o, h, l, c, i0, i_end, i_sig, float(r.stop_pct),
                                   float(r.target_pct))
                free_after = ends[i]
                rows.append({"symbol": symbol, "entry_date": ts[i0], "entry_price": o[i0],
                             "exit_date": ts[i], "exit_price": float(px), "exit_reason": why,
                             "stop_pct": float(r.stop_pct), "confidence": float(r.confidence)})
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)
