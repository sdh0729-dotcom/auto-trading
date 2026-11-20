from __future__ import annotations

import itertools

import pandas as pd

from autotrade.rules.base import BaseRule, atr_stop, finalize
from autotrade.rules.indicators import sma
from autotrade.types import Panel, Signals

S = [10, 20, 50]
L = [50, 100, 200]
K = [2.0, 3.0]
MAX_HOLD = 20


class MaTrend(BaseRule):
    """Pullback recovery: sma_s > sma_l, close_{t-1} < sma_s_{t-1}, close_t > sma_s_t."""
    family = "ma_trend"
    PARAM_KEYS = ("s", "l", "k")

    @classmethod
    def grid(cls):
        return [{"s": s, "l": l, "k": k} for s, l, k in itertools.product(S, L, K) if s < l]

    def signals(self, panel: Panel, eligible: pd.DataFrame | None = None) -> Signals:
        s, l, k = (self.params[x] for x in self.PARAM_KEYS)
        c = panel.close
        sma_s, sma_l = sma(c, s), sma(c, l)
        entries = (sma_s > sma_l) & (c.shift(1) < sma_s.shift(1)) & (c > sma_s)
        exits = c < sma_l
        vol20 = c.pct_change().rolling(20).std()
        conf = (sma_s / sma_l - 1) / vol20
        return finalize(panel, entries, exits, atr_stop(panel, k), conf,
                        "next_open", "next_open", MAX_HOLD)
