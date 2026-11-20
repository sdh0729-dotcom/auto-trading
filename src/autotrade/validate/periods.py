"""Persisted period split per pocket x market (research/periods-{pocket}-{market}.json).

The split is computed once, on the first `lab screen` for that pocket x market, from the
trading dates stored at that moment (`data_cutoff` = the last one) and is read by every later
command, so data collected afterwards never moves a boundary (spec 7.8). `new_cycle` starts a
re-validation cycle (spec 7.9): train / valid stay, the new sealed period is the data after
the previous sealed end, allowed only with >= MIN_NEW_CYCLE_DAYS new trading days.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from autotrade.validate.split import Periods, split_periods

_VERSION = 1
MIN_NEW_CYCLE_DAYS = 63          # ~3 months of trading days (spec 7.9)


class PeriodError(Exception):
    """A period file is unusable or a new cycle's preconditions are not met."""


@dataclass(frozen=True)
class PeriodRecord:
    pocket: str
    market: str
    periods: Periods
    data_cutoff: date                # last stored trading date when the sealed period was set
    created_at: str                  # ISO-8601 UTC of the first split
    cycle: int = 1
    validate_runs: int = 0           # `lab validate` runs on these train/valid periods
    history: tuple = ()              # earlier cycles: (sealed start, sealed end, data cutoff)


def new_record(pocket: str, market: str, dates: pd.DatetimeIndex, fractions) -> PeriodRecord:
    periods = split_periods(dates, fractions)
    return PeriodRecord(pocket, market, periods, pd.Timestamp(dates[-1]).date(),
                        datetime.now(timezone.utc).isoformat(timespec="seconds"))


def new_cycle(rec: PeriodRecord, dates: pd.DatetimeIndex) -> PeriodRecord:
    """Next cycle: sealed = every trading date after the current sealed end."""
    idx = pd.DatetimeIndex(dates)
    new = idx[idx > pd.Timestamp(rec.periods.sealed[1])]
    if len(new) < MIN_NEW_CYCLE_DAYS:
        raise PeriodError(f"a new cycle needs >= {MIN_NEW_CYCLE_DAYS} trading days after the "
                          f"sealed end {rec.periods.sealed[1]}; only {len(new)} stored")
    sealed = (new[0].date(), new[-1].date())
    old = (rec.periods.sealed[0].isoformat(), rec.periods.sealed[1].isoformat(),
           rec.data_cutoff.isoformat())
    return replace(rec, periods=replace(rec.periods, sealed=sealed), data_cutoff=sealed[1],
                   cycle=rec.cycle + 1, history=rec.history + (old,))


def _iso(p: tuple[date, date]) -> list[str]:
    return [p[0].isoformat(), p[1].isoformat()]


def _dates(v) -> tuple[date, date]:
    return date.fromisoformat(v[0]), date.fromisoformat(v[1])


def save_record(path: Path, rec: PeriodRecord) -> None:
    """Atomic (temp file + os.replace)."""
    path = Path(path)
    data = {"version": _VERSION, "pocket": rec.pocket, "market": rec.market,
            "train": _iso(rec.periods.train), "valid": _iso(rec.periods.valid),
            "sealed": _iso(rec.periods.sealed), "data_cutoff": rec.data_cutoff.isoformat(),
            "created_at": rec.created_at, "cycle": rec.cycle,
            "validate_runs": rec.validate_runs, "history": [list(h) for h in rec.history]}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_record(path: Path) -> PeriodRecord | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != _VERSION:
            raise PeriodError(f"unrecognised period file format: {path}")
        periods = Periods(_dates(d["train"]), _dates(d["valid"]), _dates(d["sealed"]))
        return PeriodRecord(d["pocket"], d["market"], periods,
                            date.fromisoformat(d["data_cutoff"]), d["created_at"],
                            int(d["cycle"]), int(d["validate_runs"]),
                            tuple(tuple(h) for h in d["history"]))
    except (KeyError, TypeError, ValueError) as e:
        raise PeriodError(f"invalid period file {path}: {e}") from e
