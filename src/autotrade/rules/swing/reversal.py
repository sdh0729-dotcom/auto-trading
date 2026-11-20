from __future__ import annotations

import itertools

import pandas as pd

from autotrade.rules.base import BaseRule, atr_stop, finalize, in_universe
from autotrade.types import Panel, Signals

LOOKBACK = [5, 10, 15, 20]
TOP_K = [1, 2, 3]
HOLD = [2, 5, 10]
K = [2.0]


class Reversal(BaseRule):
    """Daily: top-k biggest losers over `lookback` days (loss > 0) among the day's universe
    (`eligible`, all symbols without it). Exit by max_hold_days = hold."""
    family = "reversal"
    PARAM_KEYS = ("lookback", "top_k", "hold", "k")

    @classmethod
    def grid(cls):
        return [{"lookback": lb, "top_k": tk, "hold": h, "k": k}
                for lb, tk, h, k in itertools.product(LOOKBACK, TOP_K, HOLD, K)]

    def signals(self, panel: Panel, eligible: pd.DataFrame | None = None) -> Signals:
        lb, tk, hold, k = (self.params[x] for x in self.PARAM_KEYS)
        c = panel.close
        loss = (-(c / c.shift(lb) - 1)).where(in_universe(eligible, c))
        rank = loss.rank(axis=1, ascending=False, method="first")
        entries = (rank <= tk) & (loss > 0)
        exits = entries & False
        conf = (tk - rank + 1) / tk
        return finalize(panel, entries, exits, atr_stop(panel, k), conf,
                        "next_open", "next_open", hold)
