from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

Market = Literal["kr", "us"]
Pocket = Literal["swing", "daytrade"]
Scenario = Literal["real", "paper"]
Product = Literal["stock", "etf"]

TRADE_COLUMNS = ["symbol", "entry_date", "entry_price", "exit_date", "exit_price",
                 "exit_reason", "stop_pct", "confidence"]
EXIT_REASONS = ("signal", "stop", "gap_stop", "max_hold", "period_end", "data_end")
META_COLUMNS = ["name", "security_type", "product", "is_common_share", "delisted", "source",
                "leverage_factor", "english_name"]   # last two: ETFs from /api/v1/stocks


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    meta: pd.DataFrame
    # the market's trading dates (dates only, no prices; may run past the panel's last row,
    # e.g. to tell whether that row is a month's last trading day); None if unknown
    calendar: pd.DatetimeIndex | None = None

    def slice(self, start: date, end: date) -> "Panel":
        """Inclusive on both ends; meta and calendar unchanged."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        f = lambda df: df.loc[(df.index >= s) & (df.index <= e)]
        return Panel(f(self.open), f(self.high), f(self.low), f(self.close),
                     f(self.volume), self.meta, self.calendar)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


@dataclass(frozen=True)
class EqualWeight:
    """Rotation sizing: each entry gets equity / slots (prior-close equity), capped at
    max_weight of equity (None: the pocket's max_position_weight), independent of the stop
    and confidence. Without it a rule is sized by risk (risk_per_trade / stop)."""
    slots: int
    max_weight: float | None = None


@dataclass
class Signals:
    entries: pd.DataFrame
    exits: pd.DataFrame
    stop_pct: pd.DataFrame
    confidence: pd.DataFrame
    entry_at: Literal["next_open", "close"]
    exit_at: Literal["next_open", "close"]
    max_hold_days: int
    equal_weight: EqualWeight | None = None
