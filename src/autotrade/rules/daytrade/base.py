"""Base class of the intraday (day-trade) rule families.

A rule turns one symbol's END-stamped 1-minute bars into day-trade candidates; the
executor (`autotrade.screen.intraday.execute`) fills them. `rule_id`, PARAM_KEYS validation
and `grid()` are shared with the swing rules through `BaseRule`.
"""
from __future__ import annotations

import pandas as pd

from autotrade.rules.base import BaseRule
from autotrade.types import Panel

DAY_SYMBOL = ["day", "symbol"]


class IntradayRule(BaseRule):
    """Subclasses set `family`, `PARAM_KEYS` and implement `grid()` and `candidates()`.
    `day_features` (params-independent, one frame per symbol) and `select` (may depend on
    params) are optional: by default no features exist and every eligible symbol-day trades.
    """

    def day_features(self, symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
        """Per-day features of one symbol (index: tz-naive local day) from its bars. Used
        only by `select`; must not depend on `self.params` (computed once per family)."""
        return pd.DataFrame(index=pd.DatetimeIndex([], name="day"))

    def select(self, day_features: pd.DataFrame, daily: Panel,
               eligible: pd.DataFrame) -> pd.MultiIndex:
        """(day, symbol) pairs to trade. `day_features` stacks every symbol's frame under a
        (day, symbol) MultiIndex. Default: every eligible symbol-day."""
        stacked = eligible.stack()
        stacked = stacked[stacked.astype(bool)]
        return pd.MultiIndex.from_tuples(stacked.index.tolist(), names=DAY_SYMBOL)

    def candidates(self, symbol: str, bars: pd.DataFrame, days: pd.DatetimeIndex,
                   daily: Panel) -> pd.DataFrame:
        """Candidates (CANDIDATE_COLUMNS) for the selected local `days` of one symbol."""
        raise NotImplementedError

    @classmethod
    def needs_features(cls) -> bool:
        return cls.day_features is not IntradayRule.day_features


def candidate_frame(rows: list[dict] | None = None) -> pd.DataFrame:
    """Rows -> a CANDIDATE_COLUMNS frame. The import is lazy: `autotrade.screen` imports the
    registry, which imports the rule modules (import cycle otherwise)."""
    from autotrade.screen.intraday import CANDIDATE_COLUMNS
    return pd.DataFrame(rows or [], columns=CANDIDATE_COLUMNS)


def empty_candidates() -> pd.DataFrame:
    return candidate_frame()
