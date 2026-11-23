"""Opening-range breakout families (spec 7.5 #0, #1).

Bars are END-stamped. The opening range of `or_minutes` is made of the bars ending in
(open, open + or_minutes], with the day's ACTUAL open (data/sessions.py: 10:00 on KR late-open
days); it is complete only at the bar ending open + or_minutes, so a
breakout signal is a LATER bar whose high exceeds the range high (strictly). The executor
enters at the next bar's open.

Entry proxy for stops: the RANGE HIGH (the breakout level); `stop_pct = distance / range_high`.
Daily-derived inputs for day d (ATR14, prior close) use rows strictly before d (shift 1).
First-5-minute volume `v5` exists only when all 5 bars of (open, open + 5] do (else NaN);
`mean_v5` is the mean of the 14 previous days WITH a v5 (excluding d), NaN when it is 0.

Market: a rule's params must not carry it (rule_id hashes params), so it is inferred from the
bars' timezone (`autotrade.types.TZ`); `day_features` records it in a `market` column so
`select` (which sees no bars) can read it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.config import load_market_raw
from autotrade.rules.daytrade.base import DAY_SYMBOL, IntradayRule, candidate_frame, empty_candidates
from autotrade.rules.daytrade._bars import (_first_minutes, _market_of, _prior_valid_mean,
                                           _session_days)
from autotrade.rules.indicators import atr
from autotrade.types import Panel

TOP_N = 20
MIN_REL_VOLUME = 1.0
RV_WINDOW = 14
FIRST_MINUTES = 5
_MAX_STOP_PCT = 0.99


def _opening_range(dbars: pd.DataFrame, open_ts: pd.Timestamp, minutes: int):
    """(high, low, up, day_open) of the completed range, or None if its last bar is missing."""
    end = open_ts + pd.Timedelta(minutes=minutes)
    if end not in dbars.index:
        return None
    rng = dbars[(dbars.index > open_ts) & (dbars.index <= end)]
    return (float(rng["high"].max()), float(rng["low"].min()),
            float(rng["close"].iloc[-1]) > float(rng["open"].iloc[0]),
            float(rng["open"].iloc[0]))


def _breakout(dbars: pd.DataFrame, end: pd.Timestamp, level: float):
    """First bar ending after `end` whose high exceeds `level`."""
    later = dbars[dbars.index > end]
    hit = later.index[(later["high"] > level).to_numpy()]
    return hit[0] if len(hit) else None


def _prior_atr(daily: Panel, symbol: str) -> pd.Series:
    if symbol not in daily.close.columns:
        return pd.Series(dtype=float)
    return atr(daily, RV_WINDOW)[symbol].shift(1)


class OrbStocksInPlay(IntradayRule):
    """Stocks-in-play ORB: top-20 relative first-5-minute volume, long only on up opens.

    `select` TRUSTS `eligible` (the point-in-time universe mask from universe.build) for the
    price, 14-day average volume and KR tick-cost filters; it adds only the ATR floor
    (MarketConfig.orb_min_atr / orb_min_atr_pct), rel volume >= 1.0 and the top-20 ranking.
    """
    family = "orb_sip"
    PARAM_KEYS = ("stop_atr", "or_minutes")

    @classmethod
    def grid(cls) -> list[dict]:
        # 0.10 ATR = the paper's version (a); the others are the wide-stop variant (b)
        return [{"stop_atr": s, "or_minutes": 5} for s in (0.10, 0.5, 0.75, 1.0)]

    def day_features(self, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        cols = ["v5", "up5", "mean_v5", "rel_vol", "market"]
        if bars.empty:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="day"))
        market = _market_of(bars)
        days, rows = [], []
        for d, db, sess in _session_days(bars):
            rng = _first_minutes(db, sess["open"], FIRST_MINUTES)
            days.append(d)
            if rng is None:                  # incomplete first 5 minutes: no v5, not up
                rows.append((np.nan, False))
            else:
                rows.append((float(rng["volume"].sum()),
                             bool(rng["close"].iloc[-1] > rng["open"].iloc[0])))
        out = pd.DataFrame(rows, index=pd.DatetimeIndex(days, name="day"),
                           columns=["v5", "up5"])
        out["mean_v5"] = _prior_valid_mean(out["v5"], RV_WINDOW)
        out["rel_vol"] = out["v5"] / out["mean_v5"]
        out["market"] = market
        return out[cols]

    def select(self, day_features: pd.DataFrame, daily: Panel,
               eligible: pd.DataFrame) -> pd.MultiIndex:
        empty = pd.MultiIndex.from_arrays([[], []], names=DAY_SYMBOL)
        if day_features.empty:
            return empty
        market = load_market_raw(day_features["market"].iloc[0])
        stacked = eligible.stack()
        elig = stacked[stacked.astype(bool)].index
        f = day_features.reindex(elig.set_names(DAY_SYMBOL))
        f = f[f["rel_vol"] >= MIN_REL_VOLUME]
        if f.empty:
            return empty
        prev_close = daily.close.shift(1)
        atr14 = atr(daily, RV_WINDOW).shift(1)
        keep = []
        for (d, s), _ in f.iterrows():
            if d not in prev_close.index or s not in prev_close.columns:
                keep.append(False)
                continue
            px, a = prev_close.at[d, s], atr14.at[d, s]
            ok = bool(np.isfinite(px) and np.isfinite(a))
            if ok and market.orb_min_atr is not None:
                ok = a > market.orb_min_atr
            if ok and market.orb_min_atr_pct is not None:
                ok = a > market.orb_min_atr_pct * px
            keep.append(ok)
        f = f[np.array(keep, dtype=bool)]
        top = f.groupby(level="day")["rel_vol"].rank(method="first", ascending=False) <= TOP_N
        return f.index[top.to_numpy()]

    def candidates(self, symbol: str, bars: pd.DataFrame, days: pd.DatetimeIndex,
                   daily: Panel) -> pd.DataFrame:
        if bars.empty or not len(days):
            return empty_candidates()
        a = _prior_atr(daily, symbol)
        want = set(pd.DatetimeIndex(days))
        rows = []
        for d, db, sess in _session_days(bars):
            if d not in want:
                continue
            o = sess["open"]
            r = _opening_range(db, o, self.params["or_minutes"])
            atr_d = a.get(d, np.nan)
            if r is None or not r[2] or not np.isfinite(atr_d):
                continue                         # only up opens
            hi = r[0]
            end = o + pd.Timedelta(minutes=self.params["or_minutes"])
            ts = _breakout(db, end, hi)
            if ts is None:
                continue
            stop = min(self.params["stop_atr"] * atr_d / hi, _MAX_STOP_PCT)
            rows.append({"day": d, "signal_ts": ts, "stop_pct": stop, "target_pct": np.nan,
                         "exit_ts": pd.NaT, "confidence": 1.0})
        return candidate_frame(rows)


class Orb(IntradayRule):
    """General opening-range breakout on every eligible symbol-day."""
    family = "orb"
    PARAM_KEYS = ("or_minutes", "stop", "target_r")

    @classmethod
    def grid(cls) -> list[dict]:
        return [{"or_minutes": m, "stop": s, "target_r": t}
                for m in (5, 15, 30) for s in ("or_low", 0.5, 1.0) for t in (np.nan, 2.0)]

    def candidates(self, symbol: str, bars: pd.DataFrame, days: pd.DatetimeIndex,
                   daily: Panel) -> pd.DataFrame:
        if bars.empty or not len(days):
            return empty_candidates()
        a = _prior_atr(daily, symbol)
        want = set(pd.DatetimeIndex(days))
        stop_k, target_r = self.params["stop"], self.params["target_r"]
        rows = []
        for d, db, sess in _session_days(bars):
            if d not in want:
                continue
            o = sess["open"]
            end = o + pd.Timedelta(minutes=self.params["or_minutes"])
            r = _opening_range(db, o, self.params["or_minutes"])
            if r is None:
                continue
            hi, lo = r[0], r[1]
            if stop_k == "or_low":
                dist = hi - lo                   # stop at the range low, entry proxy = range high
            else:
                atr_d = a.get(d, np.nan)
                if not np.isfinite(atr_d):
                    continue
                dist = stop_k * atr_d
            stop = min(dist / hi, _MAX_STOP_PCT)
            if not stop > 0:
                continue
            ts = _breakout(db, end, hi)
            if ts is None:
                continue
            tgt = np.nan if not np.isfinite(target_r) else target_r * stop
            rows.append({"day": d, "signal_ts": ts, "stop_pct": stop, "target_pct": tgt,
                         "exit_ts": pd.NaT, "confidence": 1.0})
        return candidate_frame(rows)
