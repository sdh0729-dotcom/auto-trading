from __future__ import annotations

import itertools

import numpy as np

import pandas as pd

from autotrade.rules.base import BaseRule, atr_stop, finalize, in_universe
from autotrade.types import EqualWeight, Panel, Signals

LOOKBACK = [21, 63]
REBALANCE = [10, 20]  # trading days (global cap: hold <= 20, so not calendar-monthly)
TOP_K = [1, 2, 3, 4]
K = [2.0]


class Momentum(BaseRule):
    """Every `rebalance`-th row: enter top-k by lookback return among the day's universe
    (`eligible`, all symbols without it); exit everything outside top-k.

    max_hold_days = rebalance. Rebalance rows are positional (i % rebalance == 0).
    Sized equal-weight over top_k slots (pocket per-position cap applies: single stocks).
    """
    family = "momentum"
    PARAM_KEYS = ("lookback", "rebalance", "top_k", "k")

    @classmethod
    def grid(cls):
        return [{"lookback": lb, "rebalance": rb, "top_k": tk, "k": k}
                for lb, rb, tk, k in itertools.product(LOOKBACK, REBALANCE, TOP_K, K)]

    def signals(self, panel: Panel, eligible: pd.DataFrame | None = None) -> Signals:
        lb, rb, tk, k = (self.params[x] for x in self.PARAM_KEYS)
        c = panel.close
        ret = (c / c.shift(lb) - 1).where(in_universe(eligible, c))
        rank = ret.rank(axis=1, ascending=False, method="first")
        is_reb = np.arange(len(c)) % rb == 0
        reb = c.copy().astype(bool)
        reb.loc[:, :] = is_reb[:, None]
        top = (rank <= tk) & reb
        exits = reb & ~(rank <= tk)
        conf = (tk - rank + 1) / tk
        return finalize(panel, top, exits, atr_stop(panel, k), conf,
                        "next_open", "next_open", rb, equal_weight=EqualWeight(tk))
