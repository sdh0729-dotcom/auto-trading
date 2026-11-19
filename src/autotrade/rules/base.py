from __future__ import annotations

import hashlib
import json
from typing import ClassVar, Protocol

import pandas as pd

from autotrade.rules.indicators import atr
from autotrade.types import Panel, Signals


class Rule(Protocol):
    family: ClassVar[str]
    params: dict

    def signals(self, panel: Panel) -> Signals: ...

    @classmethod
    def grid(cls) -> list[dict]: ...

    @property
    def rule_id(self) -> str: ...


class BaseRule:
    family: ClassVar[str]
    PARAM_KEYS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, params: dict):
        if set(params) != set(self.PARAM_KEYS):
            raise ValueError(f"{self.family}: params must be exactly {sorted(self.PARAM_KEYS)}, "
                             f"got {sorted(params)}")
        self.params = dict(params)

    @property
    def rule_id(self) -> str:
        digest = hashlib.sha1(json.dumps(self.params, sort_keys=True).encode()).hexdigest()
        return f"{self.family}:{digest[:8]}"

    @classmethod
    def grid(cls) -> list[dict]:
        raise NotImplementedError


def atr_stop(panel: Panel, k: float) -> pd.DataFrame:
    """stop_pct = k * ATR14 / close (uses data through t only)."""
    return k * atr(panel, 14) / panel.close


def finalize(panel: Panel, entries: pd.DataFrame, exits: pd.DataFrame, stop: pd.DataFrame,
             conf: pd.DataFrame, entry_at: str, exit_at: str, max_hold_days: int) -> Signals:
    """Drop entries lacking a valid stop (e.g. ATR warm-up), force bool dtypes, clip confidence."""
    entries = (entries.fillna(False).astype(bool)) & (stop > 0)
    exits = exits.fillna(False).astype(bool)
    return Signals(entries=entries, exits=exits, stop_pct=stop,
                   confidence=conf.clip(0, 1), entry_at=entry_at, exit_at=exit_at,
                   max_hold_days=max_hold_days)
