"""Gap and KR opening-overreaction families (spec 7.5 #4, #5).

Bars are END-stamped; minute windows start at the day's ACTUAL session open
(data/sessions.py: 10:00 on KR late-open days). The day open is the open of the day's first bar; the gap is
`open / prev_close - 1` with prev_close = `daily.close.shift(1)` (strictly before day d, as the
panel includes d's own row). The signal bar of `confirm` minutes is the last bar ending at or
before open + confirm (it must end after the open); the executor fills at the NEXT bar.

Lab choices (not in the spec): confidence = min(1, gap / 0.08) for the gap families and
min(1, |drop| / 0.10) for `kr_open_rev`; `gap_go` stop = (signal close - lowest low from the
open through the signal bar) / signal close (day skipped if <= 0); `gap_fade` target =
actual gap / 2 (fill half the gap). `kr_open_rev`: first-30-minute drop = prev close -> close of
the bar ending open + 30 min; volume filter compares the first-30-minute volume with the mean of
that symbol's first-30-minute volumes on the 14 PRIOR days with data (computed in
`day_features`, filtered in `select`); exit_ts = signal_ts + `hold` minutes. KR only; the market
is inferred from the bars' timezone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.rules.daytrade._bars import _last_at_or_before, _market_of, _session_days
from autotrade.rules.daytrade.base import DAY_SYMBOL, IntradayRule, candidate_frame, empty_candidates
from autotrade.types import Panel

GAP_CONF_SCALE = 0.08
DROP_CONF_SCALE = 0.10
FIRST_MINUTES = 30
VOL_WINDOW = 14
VOL_MULT = 2.0


def _signal_bar(dbars: pd.DataFrame, o: pd.Timestamp, minutes: int):
    sig = _last_at_or_before(dbars, o + pd.Timedelta(minutes=minutes))
    return sig if sig is not None and sig.index[0] > o else None


def _scan_days(symbol, bars, days, daily, minutes, build, market_only=None):
    """Shared per-day plumbing: for each wanted day with a finite prev close and a signal bar
    (open + `minutes`), `build(d, db, sig, pc)` returns a candidate dict or None."""
    if bars.empty or not len(days) or symbol not in daily.close.columns:
        return empty_candidates()
    market = _market_of(bars)
    if market_only is not None and market != market_only:
        return empty_candidates()
    prev = daily.close[symbol].shift(1)
    want = set(pd.DatetimeIndex(days))
    rows = []
    for d, db, sess in _session_days(bars):
        if d not in want:
            continue
        pc = prev.get(d, np.nan)
        if not np.isfinite(pc) or pc <= 0:
            continue
        sig = _signal_bar(db, sess["open"], minutes)
        if sig is None:
            continue
        row = build(d, db, sig, pc)
        if row is not None:
            rows.append(row)
    return candidate_frame(rows)


class _Gap(IntradayRule):
    def _scan(self, symbol, bars, days, daily, build):
        """build(d, db, sig, gap) -> dict|None, gap = day open / prev close - 1."""
        return _scan_days(symbol, bars, days, daily, self.params["confirm"],
                          lambda d, db, sig, pc: build(d, db, sig, float(db["open"].iloc[0]) / pc - 1.0))


class GapGo(_Gap):
    """Gap up >= `gap` whose signal bar (open + `confirm` min) closes above the day open: long at
    the next bar, stop at the day low so far, exit at session end."""
    family = "gap_go"
    PARAM_KEYS = ("gap", "confirm")

    @classmethod
    def grid(cls) -> list[dict]:
        return [{"gap": g, "confirm": c} for g in (0.02, 0.04) for c in (5, 15)]

    def candidates(self, symbol, bars, days, daily):
        th = self.params["gap"]

        def build(d, db, sig, gap):
            if gap < th or not float(sig["close"].iloc[0]) > float(db["open"].iloc[0]):
                return None
            close = float(sig["close"].iloc[0])
            low = float(db.loc[db.index <= sig.index[0], "low"].min())
            stop = (close - low) / close
            if stop <= 0:
                return None
            return {"day": d, "signal_ts": sig.index[0], "stop_pct": stop, "target_pct": np.nan,
                    "exit_ts": pd.NaT, "confidence": float(min(1.0, gap / GAP_CONF_SCALE))}
        return self._scan(symbol, bars, days, daily, build)


class GapFade(_Gap):
    """Gap down <= -`gap`: long after `confirm` minutes if the signal bar closes above the first
    bar's low; stop `stop_pct`, target = half the actual gap, exit at session end."""
    family = "gap_fade"
    PARAM_KEYS = ("gap", "confirm", "stop_pct")

    @classmethod
    def grid(cls) -> list[dict]:
        return [{"gap": g, "confirm": c, "stop_pct": s}
                for g in (0.02, 0.04) for c in (5, 15) for s in (0.01, 0.02)]

    def candidates(self, symbol, bars, days, daily):
        th, stop = self.params["gap"], self.params["stop_pct"]

        def build(d, db, sig, gap):
            if gap > -th or not float(sig["close"].iloc[0]) > float(db["low"].iloc[0]):
                return None
            return {"day": d, "signal_ts": sig.index[0], "stop_pct": stop,
                    "target_pct": abs(gap) / 2.0, "exit_ts": pd.NaT,
                    "confidence": float(min(1.0, abs(gap) / GAP_CONF_SCALE))}
        return self._scan(symbol, bars, days, daily, build)


class OpenReversal(IntradayRule):
    """KR only. First-30-minute drop <= -`drop` on >= 2x the 14-prior-day mean first-30-minute
    volume: long at the next bar, exit `hold` minutes after the signal (or session end)."""
    family = "kr_open_rev"
    PARAM_KEYS = ("drop", "hold", "stop_pct")

    @classmethod
    def grid(cls) -> list[dict]:
        return [{"drop": d, "hold": h, "stop_pct": s}
                for d in (0.03, 0.05) for h in (60, 120) for s in (0.02, 0.03)]

    def day_features(self, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        cols = ["market", "vol30", "avg14"]
        if bars.empty:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="day"))
        market = _market_of(bars)
        days, vols = [], []
        for d, db, sess in _session_days(bars):
            o = sess["open"]
            first = db[(db.index > o) & (db.index <= o + pd.Timedelta(minutes=FIRST_MINUTES))]
            days.append(d)
            vols.append(float(first["volume"].sum()) if len(first) else np.nan)
        vol = pd.Series(vols, index=pd.DatetimeIndex(days, name="day"))
        avg = vol.shift(1).rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean()
        return pd.DataFrame({"market": market, "vol30": vol, "avg14": avg})

    def select(self, day_features, daily, eligible):
        empty = pd.MultiIndex.from_arrays([[], []], names=DAY_SYMBOL)
        if day_features.empty:
            return empty
        f = day_features[day_features["market"] == "kr"]
        hot = f[(f["vol30"] >= VOL_MULT * f["avg14"]).to_numpy()]
        stacked = eligible.stack()
        ok = set(stacked.index[stacked.astype(bool).to_numpy()])
        pairs = [p for p in hot.index if p in ok]
        return pd.MultiIndex.from_tuples(pairs, names=DAY_SYMBOL) if pairs else empty

    def candidates(self, symbol, bars, days, daily):
        def build(d, db, sig, pc):
            ret = float(sig["close"].iloc[0]) / pc - 1.0
            if ret > -self.params["drop"]:
                return None
            ts = sig.index[0]
            return {"day": d, "signal_ts": ts, "stop_pct": self.params["stop_pct"],
                    "target_pct": np.nan,
                    "exit_ts": ts + pd.Timedelta(minutes=self.params["hold"]),
                    "confidence": float(min(1.0, abs(ret) / DROP_CONF_SCALE))}
        return _scan_days(symbol, bars, days, daily, FIRST_MINUTES, build, market_only="kr")
