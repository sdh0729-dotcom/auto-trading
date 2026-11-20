"""Validation statistics.

Degenerate inputs (too few observations, zero dispersion) return NaN rather than a number,
so that any `metric >= threshold` check in the verdict fails closed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def t_stat(x: np.ndarray) -> float:
    """One-sample t of the mean against 0: mean / (std(ddof=1) / sqrt(n)).
    NaN for n < 2 or zero standard deviation."""
    a = _arr(x)
    if len(a) < 2:
        return float("nan")
    sd = a.std(ddof=1)
    if not sd > 0:
        return float("nan")
    return float(a.mean() / (sd / np.sqrt(len(a))))


def profit_factor(pnl) -> float:
    """Gross profit / gross loss. inf when there are profits and no losses, NaN when there are
    neither (no trades or all zero)."""
    a = _arr(pnl)
    gain = a[a > 0].sum()
    loss = -a[a < 0].sum()
    if loss > 0:
        return float(gain / loss)
    return float("inf") if gain > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a positive fraction of the running peak.
    NaN for an empty series."""
    a = _arr(equity)
    if len(a) == 0:
        return float("nan")
    peak = np.maximum.accumulate(a)
    return float(np.max(1.0 - a / peak))


def sharpe(returns, periods_per_year: int = 252) -> float:
    """mean / std(ddof=1) * sqrt(periods_per_year). periods_per_year=1 gives the
    per-observation Sharpe used by the DSR. NaN for n < 2 or zero std."""
    a = _arr(returns)
    if len(a) < 2:
        return float("nan")
    sd = a.std(ddof=1)
    if not sd > 0:
        return float("nan")
    return float(a.mean() / sd * np.sqrt(periods_per_year))


def expected_max_sharpe(n_trials: int, sr_var_trials: float) -> float:
    """SR0: expected maximum of n_trials Sharpe estimates with true SR 0 and the given variance
    (Bailey & Lopez de Prado 2014). 0 for a single trial."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if sr_var_trials < 0:
        raise ValueError(f"sr_var_trials must be >= 0, got {sr_var_trials}")
    if n_trials == 1:
        return 0.0
    g = EULER_GAMMA
    return float(np.sqrt(sr_var_trials) * ((1 - g) * norm.ppf(1 - 1.0 / n_trials)
                                           + g * norm.ppf(1 - 1.0 / (n_trials * np.e))))


def deflated_sharpe_ratio(sr: float, n_trials: int, n_obs: int, skew: float, kurt: float,
                          sr_var_trials: float) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), a probability in [0, 1].

    All Sharpe inputs are NON-annualized, per observation: `sr` is the strategy's Sharpe per
    return observation, `n_obs` the number of return observations, `skew` and `kurt` the
    returns' skewness and non-excess kurtosis (normal = 3), `sr_var_trials` the variance of
    the per-observation Sharpe across all `n_trials` trials.

        SR0 = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))   (0 when N = 1)
        DSR = Phi((sr - SR0) * sqrt(n_obs - 1) / sqrt(1 - skew*sr + (kurt - 1)/4 * sr^2))

    NaN when n_obs < 2, any input is non-finite, or the variance term is not positive.
    """
    sr0 = expected_max_sharpe(n_trials, sr_var_trials)
    if n_obs < 2 or not all(np.isfinite([sr, skew, kurt, sr_var_trials])):
        return float("nan")
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if not denom > 0:
        return float("nan")
    return float(norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom)))


def profit_without_top(pnl, n: int = 5) -> float:
    """Total pnl after removing the n largest trades (0.0 when there are n trades or fewer)."""
    a = np.sort(_arr(pnl))
    return float(a[:-n].sum()) if n > 0 else float(a.sum())
