from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from autotrade.rules.base import BaseRule, atr_stop, finalize, in_universe
from autotrade.types import EqualWeight, Panel, Signals

LOOKBACK = [63, 126, 252]
REBALANCE = [10, 20]   # trading days (global cap: hold <= 20)
TOP_K = [1, 2, 3]
K = [3.0]              # wide ATR stop: a crash exit, not a trade-level stop


class EtfRotation(BaseRule):
    """ETF dual momentum. Every `rebalance`-th row: rank the day's universe ETFs (meta
    product == "etf"; leveraged/inverse are already outside the universe) by lookback
    return, enter the top-k whose return is > 0 (absolute momentum: otherwise that slot stays
    in cash), exit everything else. Equal weight over top_k slots, up to 100% per ETF.
    max_hold_days = rebalance; rebalance rows are positional like momentum."""
    family = "etf_rotation"
    PARAM_KEYS = ("lookback", "rebalance", "top_k", "k")

    @classmethod
    def grid(cls):
        return [{"lookback": lb, "rebalance": rb, "top_k": tk, "k": k}
                for lb, rb, tk, k in itertools.product(LOOKBACK, REBALANCE, TOP_K, K)]

    def signals(self, panel: Panel, eligible: pd.DataFrame | None = None) -> Signals:
        lb, rb, tk, k = (self.params[x] for x in self.PARAM_KEYS)
        c = panel.close
        is_etf = panel.meta["product"].reindex(c.columns).eq("etf").to_numpy()
        ret = (c / c.shift(lb) - 1).where(in_universe(eligible, c))
        ret = ret.where(np.broadcast_to(is_etf, ret.shape)).where(ret > 0)
        rank = ret.rank(axis=1, ascending=False, method="first")
        is_reb = np.arange(len(c)) % rb == 0
        reb = pd.DataFrame(np.broadcast_to(is_reb[:, None], c.shape), index=c.index,
                           columns=c.columns)
        top = (rank <= tk) & reb
        exits = reb & ~(rank <= tk)
        conf = (tk - rank + 1) / tk
        return finalize(panel, top, exits, atr_stop(panel, k), conf, "next_open", "next_open",
                        rb, equal_weight=EqualWeight(tk, max_weight=1.0))
