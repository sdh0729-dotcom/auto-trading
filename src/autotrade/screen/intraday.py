"""Intraday executor: per-symbol day-trade candidates -> same-day trades on 1-minute bars.

Bars are END-stamped: a signal from the bar ending `signal_ts` fills at the open of the next
existing bar (never the signal bar). Prices are raw bar prices; costs come later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrade.data.sessions import sessions
from autotrade.types import TRADE_COLUMNS, TZ, Product, VIConfig

__all__ = ["CANDIDATE_COLUMNS", "SessionRules", "VIConfig", "execute", "vi_halts"]

CANDIDATE_COLUMNS = ["day", "signal_ts", "stop_pct", "target_pct", "exit_ts", "confidence"]


@dataclass(frozen=True)
class SessionRules:
    """Times relative to each day's ACTUAL session (data/sessions.py), so late opens, KR exam
    days (close 16:30) and US half-days (close 13:00) get their own cut-offs."""
    last_entry_lead: int | None  # entry bar must end <= the day's close - this many minutes
                                 # (KR 10 = closing-auction start: 15:20, 16:20 on exam days)
    exit_lead_minutes: int       # session exit = last bar ending <= the day's close - lead
    vi: VIConfig | None = None   # KR volatility-interruption auctions; None = no VI


def _market_of_tz(tz) -> str:
    name = str(tz)
    for m, z in TZ.items():
        if z == name:
            return m
    raise ValueError(f"unknown bar timezone {name}")


_EPS = 1e-12


def _vi_mask(ends, o, h, l, c, prev_close, dynamic_pct, cfg):
    n = len(ends)
    halted = np.zeros(n, bool)
    halt_ns = pd.Timedelta(minutes=cfg.halt_minutes).value
    ref, last_close = float(prev_close), np.nan
    up, dn = 1.0 + cfg.static_pct - _EPS, 1.0 - cfg.static_pct + _EPS
    dup, ddn = 1.0 + dynamic_pct - _EPS, 1.0 - dynamic_pct + _EPS
    i = 0
    while i < n:
        static = h[i] >= ref * up or l[i] <= ref * dn
        dyn = np.isfinite(last_close) and (h[i] >= last_close * dup or l[i] <= last_close * ddn)
        if not (static or dyn):
            last_close = c[i]
            i += 1
            continue
        j = i + 1
        while j < n and ends[j] <= ends[i] + halt_ns:
            halted[j] = True
            j += 1
        if static and j < n:
            ref = o[j]              # a static VI resets the reference to the auction end price
        last_close = np.nan         # no dynamic check on the first bar after an auction
        i = j
    return halted


def vi_halts(bars_day: pd.DataFrame, prev_close: float, dynamic_pct: float,
             cfg: VIConfig) -> pd.Series:
    """True for the bars (END-stamped) inside a VI auction, for ONE day of bars.

    A bar triggers when its high/low reaches +-static_pct of the static reference (previous
    daily close; after a static VI it resets to the open of the first bar after the auction)
    or +-dynamic_pct of the previous existing bar's close. A trigger bar ending at T is
    itself tradable; bars ending in (T, T + halt_minutes] are halted. The first bar of the
    day has no dynamic check, nor does the first bar after an auction.
    """
    ends = bars_day.index.as_unit("ns").asi8
    o, h, l, c = (bars_day[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    return pd.Series(_vi_mask(ends, o, h, l, c, prev_close, dynamic_pct, cfg),
                     index=bars_day.index, dtype=bool)


def _walk(o, h, l, c, i0, i_end, i_sig, stop_pct, target_pct, halted=None):
    """Walk bars i0..i_end; return (exit_idx, exit_price, reason). Entry at o[i0].

    Fills on a halted bar are deferred to the open of the first non-halted bar: a pending
    stop fills there even below the stop, a pending target only if that open >= target
    (else the walk continues), a pending signal exit fills there.
    """
    entry = o[i0]
    stop = entry * (1.0 - stop_pct)
    target = entry * (1.0 + target_pct) if np.isfinite(target_pct) else None
    p_stop = p_target = p_sig = False
    for i in range(i0, i_end + 1):
        if halted is not None and halted[i]:
            p_stop = p_stop or o[i] <= stop or l[i] <= stop
            if target is not None:
                p_target = p_target or o[i] >= target or h[i] >= target
            p_sig = p_sig or (i_sig is not None and i == i_sig)
            continue
        if p_stop:
            return i, o[i], "stop"
        if p_target and o[i] >= target:
            return i, o[i], "target"
        if p_sig:
            return i, o[i], "session_end" if i >= i_end else "signal"
        p_target = False
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
            return i, c[i], "session_end" if i >= i_end else "signal"
    return i_end, c[i_end], "session_end"


def execute(symbol: str, bars: pd.DataFrame, cands: pd.DataFrame, rules: SessionRules,
            *, prev_close: float | pd.Series | None = None, dynamic_pct: float | None = None,
            product: Product = "stock") -> pd.DataFrame:
    """Turn day-trade candidates into same-day trades on END-stamped 1-minute bars.

    Only bars inside the day's actual session (open < end <= close, data/sessions.py; the
    market follows the bars' timezone) are used: a day that is not a session has no trades,
    pre-open and post-close bars never fill. The session exit is the last bar ending at or
    before close - exit_lead_minutes; entries need an entry bar ending at or before
    close - last_entry_lead.

    VI (only when rules.vi is set, product is not an exempt ETF and a previous close is
    known): `prev_close` is a float (applies to every day) or a Series indexed by LOCAL
    day (tz-naive midnight Timestamps) with the previous daily close; a day missing from
    the Series has no VI. `dynamic_pct` defaults to vi.dynamic_pct_other (KOSPI200
    membership is unknown). Entries on a halted bar move to the open of the first
    non-halted bar (still subject to last_entry); exits are deferred as in `_walk`. The
    session-exit bar is never halted.
    """
    rows: list[dict] = []
    if len(cands) and len(bars):
        idx = bars.index
        ends_all = idx.as_unit("ns").asi8
        day_all = idx.normalize().tz_localize(None).as_unit("ns").asi8
        O, H, L, C = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        cday = pd.DatetimeIndex(cands["day"]).normalize()
        sess = sessions(_market_of_tz(idx.tz), cday.min(), cday.max())
        lead = pd.Timedelta(minutes=rules.exit_lead_minutes).value
        entry_lead = (pd.Timedelta(minutes=rules.last_entry_lead).value
                      if rules.last_entry_lead is not None else None)
        vi = rules.vi
        vi_on = vi is not None and prev_close is not None and not (
            product == "etf" and vi.etf_exempt)
        dyn = vi.dynamic_pct_other if (vi_on and dynamic_pct is None) else dynamic_pct
        cands = cands.sort_values("signal_ts", kind="stable")
        for day, group in cands.groupby(cands["day"].dt.normalize(), sort=True):
            if day not in sess.index:
                continue                    # not a session: no regular bars
            s_open = sess.at[day, "open"].value
            s_close = sess.at[day, "close"].value
            lo = int(np.searchsorted(day_all, day.value, side="left"))
            hi = int(np.searchsorted(day_all, day.value, side="right"))
            lo += int(np.searchsorted(ends_all[lo:hi], s_open, side="right"))
            hi = lo + int(np.searchsorted(ends_all[lo:hi], s_close, side="right"))
            if hi == lo:
                continue
            ts = idx[lo:hi]
            o, h, l, c, ends = O[lo:hi], H[lo:hi], L[lo:hi], C[lo:hi], ends_all[lo:hi]
            i_end = int(np.searchsorted(ends, s_close - lead, side="right")) - 1
            if i_end < 0:
                continue
            last_entry = s_close - entry_lead if entry_lead is not None else None
            halted = None
            if vi_on:
                pc = prev_close.get(day) if isinstance(prev_close, pd.Series) else prev_close
                if pc is not None and np.isfinite(pc):
                    halted = _vi_mask(ends, o, h, l, c, pc, dyn, vi)
                    halted[i_end:] = False      # the closing auction is never halted
            free_after = None   # exit bar end (ns) of the open/last trade
            for r in group.itertuples(index=False):
                sig = pd.Timestamp(r.signal_ts)
                if sig.tz_convert(ts.tz).normalize().tz_localize(None) != day:
                    raise ValueError(f"signal_ts {sig} does not belong to day {day.date()}")
                if free_after is not None and sig.value < free_after:
                    continue
                i0 = int(np.searchsorted(ends, sig.value, side="right"))
                if halted is not None:
                    while i0 < len(ts) and halted[i0]:
                        i0 += 1
                if i0 >= len(ts) or i0 > i_end:
                    continue
                if last_entry is not None and ends[i0] > last_entry:
                    continue
                i_sig = None
                if pd.notna(r.exit_ts):
                    i_sig = max(int(np.searchsorted(ends, pd.Timestamp(r.exit_ts).value,
                                                    side="right")) - 1, i0)
                    i_sig = min(i_sig, i_end)
                i, px, why = _walk(o, h, l, c, i0, i_end, i_sig, float(r.stop_pct),
                                   float(r.target_pct), halted)
                free_after = ends[i]
                rows.append({"symbol": symbol, "entry_date": ts[i0], "entry_price": o[i0],
                             "exit_date": ts[i], "exit_price": float(px), "exit_reason": why,
                             "stop_pct": float(r.stop_pct), "confidence": float(r.confidence)})
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)
