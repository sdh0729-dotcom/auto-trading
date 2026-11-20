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
META_COLUMNS = ["name", "security_type", "product", "is_common_share", "delisted", "source"]


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    meta: pd.DataFrame

    def slice(self, start: date, end: date) -> "Panel":
        """Inclusive on both ends; meta unchanged."""
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        f = lambda df: df.loc[(df.index >= s) & (df.index <= e)]
        return Panel(f(self.open), f(self.high), f(self.low), f(self.close),
                     f(self.volume), self.meta)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


@dataclass
class Signals:
    entries: pd.DataFrame
    exits: pd.DataFrame
    stop_pct: pd.DataFrame
    confidence: pd.DataFrame
    entry_at: Literal["next_open", "close"]
    exit_at: Literal["next_open", "close"]
    max_hold_days: int
