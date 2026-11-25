"""`_negf` variants (spec 2026-11-24-disclosure-lab-design.md §7 #2-#3): a price family that
skips entries after bad news.

`with_negative_filter(base, window)` returns a subclass of `base` (family `<base>_negf`, same
PARAM_KEYS and grid, so rule ids carry the new family name and never collide with the base's):
 - entries &= ~recent_negative(panel, window): no entry while a direction -1 event's signal
   date lies within the last `window` rows including t (every -1 event extends the block);
 - exits |= negative_on(panel): a held position exits at the next open when a -1 event's
   signal date is t (`exit_at="next_open"` of the base signals).
Everything else (stop, confidence, entry_at / exit_at, max_hold_days, equal_weight) is the
base's. Both masks use events with signal_date <= t only. Needs Panel.events (ValueError
otherwise). The default window 20 equals negative_window of config/events/*.yaml (tested).
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

from autotrade.disclosure.events import negative_on, recent_negative
from autotrade.rules.base import BaseRule
from autotrade.types import Panel, Signals

NEGATIVE_WINDOW = 20


def with_negative_filter(base: type[BaseRule], window: int = NEGATIVE_WINDOW) -> type[BaseRule]:
    def signals(self, panel: Panel, eligible: pd.DataFrame | None = None) -> Signals:
        if panel.events is None:
            raise ValueError(f"{self.family} rules need Panel.events")
        s = base.signals(self, panel, eligible)
        blocked = recent_negative(panel, window).reindex_like(s.entries).fillna(False)
        forced = negative_on(panel).reindex_like(s.exits).fillna(False)
        return replace(s, entries=s.entries & ~blocked.astype(bool),
                       exits=s.exits | forced.astype(bool))

    return type(f"{base.__name__}Negf", (base,),
                {"family": f"{base.family}_negf", "signals": signals,
                 "__doc__": f"{base.__name__} that skips entries after negative events."})
