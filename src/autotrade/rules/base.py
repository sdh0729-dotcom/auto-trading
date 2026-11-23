from __future__ import annotations

import hashlib
import json
from typing import ClassVar, Protocol

import pandas as pd

from autotrade.rules.indicators import atr
from autotrade.types import EqualWeight, Panel, Signals


class Rule(Protocol):
    family: ClassVar[str]
    params: dict

    def signals(self, panel: Panel, eligible: pd.DataFrame | None = None) -> Signals:
        """`eligible`: the point-in-time universe mask (date x symbol, from data through t-1).
        Cross-sectional families rank only within it; the others ignore it."""
        ...

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


def in_universe(eligible: pd.DataFrame | None, like: pd.DataFrame) -> pd.DataFrame:
    """Bool mask aligned to `like`: the eligibility mask, or all True without one."""
    if eligible is None:
        return pd.DataFrame(True, index=like.index, columns=like.columns)
    return eligible.reindex(index=like.index, columns=like.columns).eq(True)


def atr_stop(panel: Panel, k: float) -> pd.DataFrame:
    """stop_pct = k * ATR14 / close (uses data through t only)."""
    return k * atr(panel, 14) / panel.close


def finalize(panel: Panel, entries: pd.DataFrame, exits: pd.DataFrame, stop: pd.DataFrame,
             conf: pd.DataFrame, entry_at: str, exit_at: str, max_hold_days: int,
             equal_weight: EqualWeight | None = None) -> Signals:
    """Drop entries lacking a valid stop (e.g. ATR warm-up), force bool dtypes, clip confidence."""
    entries = (entries.fillna(False).astype(bool)) & (stop > 0)
    exits = exits.fillna(False).astype(bool)
    return Signals(entries=entries, exits=exits, stop_pct=stop,
                   confidence=conf.clip(0, 1), entry_at=entry_at, exit_at=exit_at,
                   max_hold_days=max_hold_days, equal_weight=equal_weight)
