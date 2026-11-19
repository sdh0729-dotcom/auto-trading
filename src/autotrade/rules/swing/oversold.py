from __future__ import annotations

import itertools

from autotrade.rules.base import BaseRule, atr_stop, finalize
from autotrade.rules.indicators import rsi, sma
from autotrade.types import Panel, Signals

X = [5, 10, 15, 20]  # RSI(2) threshold
K = [1.5, 2.0, 3.0]
MAX_HOLD = 5


class Oversold(BaseRule):
    family = "oversold"
    PARAM_KEYS = ("x", "k")

    @classmethod
    def grid(cls):
        return [{"x": x, "k": k} for x, k in itertools.product(X, K)]

    def signals(self, panel: Panel) -> Signals:
        x, k = self.params["x"], self.params["k"]
        c = panel.close
        r2 = rsi(c, 2)
        entries = (r2 < x) & (c > sma(c, 200))
        exits = c > sma(c, 5)
        conf = (x - r2) / x
        return finalize(panel, entries, exits, atr_stop(panel, k), conf,
                        "next_open", "next_open", MAX_HOLD)
