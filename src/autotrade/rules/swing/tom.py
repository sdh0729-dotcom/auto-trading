from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from autotrade.rules.base import BaseRule, atr_stop, finalize
from autotrade.types import Panel, Signals

N = [1, 2, 3, 5]   # entry: n trading days before month's last trading day
M = [1, 2, 3, 5]   # exit: m-th trading day of next month
K = [2.0, 3.0]


def final_month_complete(panel: Panel) -> bool:
    """Is the panel's last row its month's last trading day? Per `panel.calendar` when it
    has a date after that row (the next trading date lies in a later month); otherwise only
    a last row on the month's last calendar day proves it."""
    idx = panel.close.index
    if len(idx) == 0:
        return False
    last = idx[-1]
    cal = panel.calendar
    if cal is not None:
        later = cal[cal > last]
        if len(later):
            return later[0].to_period("M") != last.to_period("M")
    return (last + pd.Timedelta(days=1)).month != last.month


class Tom(BaseRule):
    """Turn-of-month. Month boundaries come from the panel's own index (calendar only, no prices).

    The panel's final month gets no entry unless its last row is that month's last trading
    day (`final_month_complete`): a panel ending mid-month must not treat its last row as a
    month end (that would be an entry the longer history does not have).

    max_hold_days = n + m (<= 20 for the grid).
    """
    family = "tom"
    PARAM_KEYS = ("n", "m", "k")

    @classmethod
    def grid(cls):
        return [{"n": n, "m": m, "k": k} for n, m, k in itertools.product(N, M, K)]

    def signals(self, panel: Panel, eligible: pd.DataFrame | None = None) -> Signals:
        n, m, k = (self.params[x] for x in self.PARAM_KEYS)
        c = panel.close
        months = np.asarray(c.index.to_period("M").astype(str))
        starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
        ends = np.r_[starts[1:] - 1, len(months) - 1]
        e_pos = np.zeros(len(c), bool)
        x_pos = np.zeros(len(c), bool)
        complete = final_month_complete(panel)
        for j, last in enumerate(ends):
            if j == len(ends) - 1 and not complete:
                continue          # month end unknown: n days before it is not knowable yet
            if last - n >= 0:
                e_pos[last - n] = True
        for i, first in enumerate(starts):
            if i == 0:
                continue  # no preceding month end in panel
            nxt_end = ends[i]
            if first + m - 1 <= nxt_end:
                x_pos[first + m - 1] = True
        entries = c.copy().astype(bool)
        entries.loc[:, :] = e_pos[:, None]
        exits = c.copy().astype(bool)
        exits.loc[:, :] = x_pos[:, None]
        conf = c.copy() * 0 + 0.5
        return finalize(panel, entries, exits, atr_stop(panel, k), conf,
                        "close", "close", n + m)
