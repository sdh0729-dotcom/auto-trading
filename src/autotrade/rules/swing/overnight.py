"""Overnight ETF: buy close, sell next open. max_hold_days = 1 (spec exception: one-night hold,
state this on the rule card)."""
from __future__ import annotations

import numpy as np

from autotrade.rules.base import BaseRule
from autotrade.types import Panel, Signals

STOP = 0.05
CONF = 0.5


class Overnight(BaseRule):
    family = "overnight"
    PARAM_KEYS = ()

    @classmethod
    def grid(cls):
        return [{}]

    def signals(self, panel: Panel) -> Signals:
        c = panel.close
        etf = (panel.meta["product"].reindex(c.columns) == "etf").to_numpy()
        mask = np.broadcast_to(etf, c.shape)
        entries = c.copy().astype(bool)
        entries.loc[:, :] = mask
        exits = c.copy().astype(bool)
        exits.loc[:, :] = True
        stop = c.copy() * np.nan
        stop = stop.mask(entries, STOP)
        conf = stop.mask(entries, CONF)
        return Signals(entries=entries, exits=exits, stop_pct=stop, confidence=conf,
                       entry_at="close", exit_at="next_open", max_hold_days=1)
