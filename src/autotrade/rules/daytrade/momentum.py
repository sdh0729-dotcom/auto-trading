"""Momentum families (spec 7.5 #6, #8): intraday momentum and close momentum/reversal.

Bars are END-stamped. Both families signal at a late-session bar (KR 14:50, US 15:30 ET) and
hold to the close: KR exits at the bar ending 15:20 (last continuous-session bar before the
closing auction, via `exit_ts`), US at session end (`exit_ts` NaT).

Lab choices (not in the spec): a protective stop of `STOP_PCT` = 2% of the entry for both
families (a crash exit only; the time exit is the real exit; not in the grid so it does not
multiply trials); confidence = min(1, |move| / 0.01) where move is the signal return.

Returns are measured from the previous close (`daily.close.shift(1)`: strictly before day d,
since the panel includes d's own row). A signal/measurement bar that is missing falls back to
the last bar of that day ending at or before the target time, never a later one.

Universe: `select` trusts the point-in-time `eligible` mask (leveraged/inverse ETFs are
already out) and keeps index ETFs (`MarketConfig.always_include`); `intraday_mom` also keeps
eligible ETFs (`daily.meta["product"] == "etf"`). `close_mom` trades index ETFs only. The
market is inferred from the bars' timezone and recorded in `day_features` for `select`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.config import load_market_raw
from autotrade.rules.daytrade.base import DAY_SYMBOL, IntradayRule, candidate_frame, empty_candidates
from autotrade.rules.daytrade._bars import _day_slices, _last_at_or_before, _market_of, _open_ts
from autotrade.types import Panel

STOP_PCT = 0.02
FIRST_MINUTES = 30
SIGNAL_TIME = {"kr": "14:50", "us": "15:30"}
KR_EXIT_TIME = "15:20"
CONF_SCALE = 0.01


def _market_features(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["market"], index=pd.DatetimeIndex([], name="day"))
    market = _market_of(bars)
    days = pd.DatetimeIndex([d for d, _ in _day_slices(bars)], name="day")
    return pd.DataFrame({"market": market}, index=days)


def _select(day_features: pd.DataFrame, daily: Panel, eligible: pd.DataFrame,
            include_etfs: bool) -> pd.MultiIndex:
    empty = pd.MultiIndex.from_arrays([[], []], names=DAY_SYMBOL)
    if day_features.empty:
        return empty
    index_etfs = set(load_market_raw(day_features["market"].iloc[0]).always_include)
    etfs = set(daily.meta.index[daily.meta["product"] == "etf"]) if include_etfs else set()
    ok = index_etfs | etfs
    stacked = eligible.stack()
    pairs = [(d, s) for (d, s), v in stacked.items() if v and s in ok]
    return pd.MultiIndex.from_tuples(pairs, names=DAY_SYMBOL) if pairs else empty


def _row(day, ts, move, market, d_close_ts=None):
    return {"day": day, "signal_ts": ts, "stop_pct": STOP_PCT, "target_pct": np.nan,
            "exit_ts": d_close_ts if d_close_ts is not None else pd.NaT,
            "confidence": float(min(1.0, abs(move) / CONF_SCALE))}


class _Momentum(IntradayRule):
    def day_features(self, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        return _market_features(bars)

    def _scan(self, symbol, bars, days, daily, accept):
        """accept(first30_ret, signal_ret) -> bool; shared signal-bar / exit plumbing."""
        if bars.empty or not len(days) or symbol not in daily.close.columns:
            return empty_candidates()
        market = _market_of(bars)
        sess = load_market_raw(market).session_open
        prev = daily.close[symbol].shift(1)
        want = set(pd.DatetimeIndex(days))
        rows = []
        for d, db in _day_slices(bars):
            if d not in want:
                continue
            pc = prev.get(d, np.nan)
            if not np.isfinite(pc) or pc <= 0:
                continue
            o = _open_ts(d, sess, bars.index.tz)
            first = _last_at_or_before(db, o + pd.Timedelta(minutes=FIRST_MINUTES))
            if first is None or not first.index[0] > o:
                r30 = np.nan
            else:
                r30 = float(first["close"].iloc[0]) / pc - 1.0
            sig_target = pd.Timestamp(f"{d:%Y-%m-%d} {SIGNAL_TIME[market]}", tz=bars.index.tz)
            sig = _last_at_or_before(db, sig_target)
            if sig is None:
                continue
            ret = float(sig["close"].iloc[0]) / pc - 1.0
            if not accept(r30, ret):
                continue
            exit_ts = (pd.Timestamp(f"{d:%Y-%m-%d} {KR_EXIT_TIME}", tz=bars.index.tz)
                       if market == "kr" else pd.NaT)
            move = r30 if self._uses_first30 else ret
            rows.append(_row(d, sig.index[0], move, market, exit_ts))
        return candidate_frame(rows)


class IntradayMomentum(_Momentum):
    """Gao et al.: first-30-minute return (prev close -> close of the bar ending open+30min)
    above `threshold` predicts the last 30 minutes; enter at the signal bar, exit as above."""
    family = "intraday_mom"
    PARAM_KEYS = ("threshold",)
    _uses_first30 = True

    @classmethod
    def grid(cls) -> list[dict]:
        return [{"threshold": t} for t in (0.0, 0.002, 0.005)]

    def select(self, day_features, daily, eligible):
        return _select(day_features, daily, eligible, include_etfs=True)

    def candidates(self, symbol, bars, days, daily):
        th = self.params["threshold"]
        return self._scan(symbol, bars, days, daily,
                          lambda r30, ret: np.isfinite(r30) and r30 > th)


class CloseMomentum(_Momentum):
    """Baltussen et al.: sign of the return prev close -> signal bar. `direction` momentum goes
    long after an up move, reversal after a down move; |move| must be >= `min_move` (and > 0)."""
    family = "close_mom"
    PARAM_KEYS = ("direction", "min_move")
    _uses_first30 = False

    @classmethod
    def grid(cls) -> list[dict]:
        return [{"direction": d, "min_move": m}
                for d in ("momentum", "reversal") for m in (0.0, 0.005)]

    def select(self, day_features, daily, eligible):
        return _select(day_features, daily, eligible, include_etfs=False)

    def candidates(self, symbol, bars, days, daily):
        sign = 1.0 if self.params["direction"] == "momentum" else -1.0
        mm = self.params["min_move"]
        return self._scan(symbol, bars, days, daily,
                          lambda r30, ret: sign * ret > 0 and abs(ret) >= mm)
