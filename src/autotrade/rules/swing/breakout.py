from __future__ import annotations

import itertools

from autotrade.rules.base import BaseRule, atr_stop, finalize
from autotrade.rules.indicators import atr, rolling_high, rolling_low
from autotrade.types import Panel, Signals

N = [20, 40, 55]   # breakout lookback
M = [5, 10, 20]    # exit low lookback
K = [1.5, 2.0, 3.0]  # ATR stop multiple
MAX_HOLD = 20


class Breakout(BaseRule):
    family = "breakout"
    PARAM_KEYS = ("n", "m", "k")

    @classmethod
    def grid(cls):
        return [{"n": n, "m": m, "k": k} for n, m, k in itertools.product(N, M, K)]

    def signals(self, panel: Panel) -> Signals:
        n, m, k = (self.params[x] for x in self.PARAM_KEYS)
        prev_high = rolling_high(panel.high, n).shift(1)
        entries = panel.close > prev_high
        exits = panel.close < rolling_low(panel.low, m)
        conf = (panel.close - prev_high) / atr(panel, 14)
        return finalize(panel, entries, exits, atr_stop(panel, k), conf,
                        "next_open", "next_open", MAX_HOLD)
