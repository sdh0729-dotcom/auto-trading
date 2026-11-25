"""`disclosure` swing family (spec 2026-11-24-disclosure-lab-design.md §7 #1, §10-1 #2).

Entry flag at t (entry at t+1's open, `entry_at="next_open"`) where ALL of:
 - an event of `bundle` (`bundle_a`, `bundle_b` or `positive`, see events.EVENT_COLUMNS) has
   signal_date t (`event_frame`); with `sized=True` only events whose size passes the table's
   minimum (`size_ok`; an unknown size never passes -> no entry);
 - the confirmation holds at t: `none` -> always (filing alone); `ma20` -> close_t > SMA20 of
   close over the 20 rows ending at and including t; `high20` -> close_t > max(high) over the
   20 rows BEFORE t (t's own high excluded);
 - the symbol is in the point-in-time universe (`eligible`, all True without one).
No rule exit: a trade ends at the ATR stop (`atr_stop(panel, k)`, k x ATR14 / close) or after
`hold` trading days (`max_hold_days`), exit at the next open.
Confidence at t: 1.0 where an event of `bundle` with signal date t passes its size minimum
(`size_ok`; always the case on the `sized=True` path), else 0.5.

Grid = bundle {bundle_a, bundle_b, positive} x hold {5, 10, 20} x k {2, 3} x confirm {none,
ma20, high20} x sized {True, False} = 108 for every bundle; a bundle without a size criterion
(size_ok always True) duplicates its sized axis, accepted: it only makes DSR stricter.
"""
from __future__ import annotations

import itertools

import pandas as pd

from autotrade.disclosure.events import event_frame
from autotrade.rules.base import BaseRule, atr_stop, finalize, in_universe
from autotrade.rules.indicators import rolling_high, sma
from autotrade.types import Panel, Signals

BUNDLES = ["bundle_a", "bundle_b", "positive"]
HOLD = [5, 10, 20]
K = [2.0, 3.0]
CONFIRM = ["none", "ma20", "high20"]
SIZED = [True, False]
CONFIRM_DAYS = 20


class Disclosure(BaseRule):
    family = "disclosure"
    PARAM_KEYS = ("bundle", "hold", "k", "confirm", "sized")

    @classmethod
    def grid(cls):
        return [{"bundle": b, "hold": h, "k": k, "confirm": c, "sized": s}
                for b, h, k, c, s in itertools.product(BUNDLES, HOLD, K, CONFIRM, SIZED)]

    def signals(self, panel: Panel, eligible: pd.DataFrame | None = None) -> Signals:
        if panel.events is None:
            raise ValueError("disclosure rules need Panel.events")
        bundle, hold, k, confirm, sized = (self.params[x] for x in self.PARAM_KEYS)
        if bundle not in BUNDLES or confirm not in CONFIRM:
            raise ValueError(f"disclosure: bad bundle / confirm {bundle!r} / {confirm!r}")
        close = panel.close
        passing = event_frame(panel, bundle, min_size=True)
        events = passing if sized else event_frame(panel, bundle, min_size=False)
        if confirm == "none":
            ok = pd.DataFrame(True, index=close.index, columns=close.columns)
        elif confirm == "ma20":
            ok = close > sma(close, CONFIRM_DAYS)
        else:
            ok = close > rolling_high(panel.high, CONFIRM_DAYS).shift(1)
        entries = events & ok & in_universe(eligible, close)
        exits = pd.DataFrame(False, index=close.index, columns=close.columns)
        conf = passing.astype(float) * 0.5 + 0.5
        return finalize(panel, entries, exits, atr_stop(panel, k), conf,
                        "next_open", "next_open", int(hold))
