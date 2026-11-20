"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation
(Bailey, Borwein, Lopez de Prado, Zhu 2015)."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _block_sharpe(sums: np.ndarray, sqs: np.ndarray, n: int) -> np.ndarray:
    """mean/std(ddof=1) per (combination, column) from aggregated sums; NaN/zero-std -> -inf
    so a candidate with no dispersion (e.g. never trades) ranks lowest."""
    mean = sums / n
    var = (sqs - n * mean ** 2) / (n - 1)
    var = np.where(var > 1e-12 * sqs / n, var, np.nan)   # rounding noise of a constant column
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = mean / np.sqrt(var)
    return np.where(np.isfinite(sr), sr, -np.inf)


def pbo_cscv(perf: pd.DataFrame, n_splits: int = 16) -> float:
    """perf: time (rows) x candidates (columns) of per-period returns, no NaN.

    Rows are cut into `n_splits` contiguous equal blocks (remainder rows at the end dropped).
    For every choice of n_splits/2 blocks as in-sample (the rest out-of-sample) the column
    with the best in-sample Sharpe (mean/std, ddof=1) is picked; its out-of-sample relative
    rank w = rank / (N + 1) (rank 1 = worst of N, ties averaged) gives the logit
    lambda = ln(w / (1 - w)). PBO = share of combinations with lambda <= 0.
    """
    if n_splits < 2 or n_splits % 2:
        raise ValueError(f"n_splits must be an even number >= 2, got {n_splits}")
    x = np.asarray(perf, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("perf needs at least 2 candidate columns")
    if np.isnan(x).any():
        raise ValueError("perf contains NaN; fill or drop before calling pbo_cscv")
    size = x.shape[0] // n_splits
    if size < 1 or size * n_splits // 2 < 2:
        raise ValueError(f"{x.shape[0]} rows is too few for {n_splits} splits")
    blocks = x[: size * n_splits].reshape(n_splits, size, x.shape[1])
    b_sum, b_sq = blocks.sum(axis=1), (blocks ** 2).sum(axis=1)   # (S, N)

    combos = list(combinations(range(n_splits), n_splits // 2))
    is_mask = np.zeros((len(combos), n_splits))
    for k, c in enumerate(combos):
        is_mask[k, list(c)] = 1.0
    oos_mask = 1.0 - is_mask
    n_half = size * n_splits // 2
    is_sr = _block_sharpe(is_mask @ b_sum, is_mask @ b_sq, n_half)
    oos_sr = _block_sharpe(oos_mask @ b_sum, oos_mask @ b_sq, n_half)

    rows = np.arange(len(combos))
    best = np.argmax(is_sr, axis=1)
    chosen = oos_sr[rows, best][:, None]
    below = (oos_sr < chosen).sum(axis=1)
    ties = (oos_sr == chosen).sum(axis=1)               # includes the chosen column itself
    rank = below + (ties + 1) / 2.0
    omega = rank / (x.shape[1] + 1)
    lam = np.log(omega / (1.0 - omega))
    return float(np.mean(lam <= 0))
