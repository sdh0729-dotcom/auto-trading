"""Train / valid / sealed period split and expanding walk-forward windows."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Periods:
    """Inclusive (start, end) trading dates of each contiguous, non-overlapping part."""
    train: tuple[date, date]
    valid: tuple[date, date]
    sealed: tuple[date, date]


def split_periods(dates: pd.DatetimeIndex, fractions=(0.6, 0.2, 0.2)) -> Periods:
    """Split the trading-date index by position. Boundaries are rounded cumulative positions,
    so each part's day count is within 1 of fraction * len(dates)."""
    f = tuple(float(v) for v in fractions)
    if len(f) != 3 or min(f) <= 0 or not np.isclose(sum(f), 1.0):
        raise ValueError(f"fractions must be 3 positive numbers summing to 1, got {fractions}")
    idx = pd.DatetimeIndex(dates)
    if not idx.is_monotonic_increasing or not idx.is_unique:
        raise ValueError("dates must be strictly increasing")
    n = len(idx)
    b1, b2 = round(n * f[0]), round(n * (f[0] + f[1]))
    if not 0 < b1 < b2 < n:
        raise ValueError(f"{n} dates is too few to split by {fractions}")

    def part(i: int, j: int) -> tuple[date, date]:
        return idx[i].date(), idx[j - 1].date()

    return Periods(part(0, b1), part(b1, b2), part(b2, n))


def walk_forward_windows(start: date, end: date, n: int = 4
                         ) -> list[tuple[date, date, date, date]]:
    """Expanding windows over calendar days [start, end]: cut into n + 1 consecutive segments
    of (near-)equal length; window i = (start, end of seg i, start of seg i+1, end of seg i+1).
    Test segments are contiguous and non-overlapping; the last one ends on `end`."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    total = (end - start).days + 1
    if total < n + 1:
        raise ValueError(f"[{start}, {end}] is too short for {n + 1} segments")
    cuts = [start + timedelta(days=k * total // (n + 1)) for k in range(n + 2)]
    seg = [(cuts[k], cuts[k + 1] - timedelta(days=1)) for k in range(n + 1)]
    return [(start, seg[i][1], seg[i + 1][0], seg[i + 1][1]) for i in range(n)]
