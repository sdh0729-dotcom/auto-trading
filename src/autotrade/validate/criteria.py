"""Pass/fail verdict on the sealed sample."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from autotrade.config import Criteria
from autotrade.sizing.portfolio import PortfolioResult
from autotrade.validate.stats import (deflated_sharpe_ratio, max_drawdown, profit_factor,
                                      profit_without_top, sharpe, t_stat)

CHECKS = ("t_stat", "dsr", "pbo", "top_removed", "profit_factor", "mdd", "valid_positive",
          "sealed_positive", "min_trades")


@dataclass(frozen=True)
class Verdict:
    passed: bool
    failed: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def period_equity(result: PortfolioResult, calendar: pd.DatetimeIndex, start: date, end: date,
                  initial_capital: float) -> pd.Series:
    """Portfolio equity on every trading day of `calendar` within [start, end]: idle days carry
    the last marked value forward, days before the first mark hold `initial_capital` (the same
    currency-converted capital the portfolio was run with)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    cal = pd.DatetimeIndex(calendar)
    days = cal[(cal >= s) & (cal <= e)]
    eq = result.equity.astype(float)
    eq.index = pd.DatetimeIndex(eq.index)
    inside = eq.index[(eq.index >= s) & (eq.index <= e)]
    if not inside.isin(days).all():
        raise ValueError("equity has dates in the period that are not calendar trading days")
    eq = eq[eq.index <= e]
    out = eq.reindex(eq.index.union(days)).ffill().reindex(days).fillna(initial_capital)
    out.name = "equity"
    return out


def evaluate(sealed: PortfolioResult, valid: PortfolioResult, pbo: float, n_trials: int,
             sr_var_trials: float, c: Criteria, *, calendar: pd.DatetimeIndex,
             period: tuple[date, date], initial_capital: float) -> Verdict:
    """Apply the sealed-sample pass criteria.

    Sealed equity is rebuilt here with `period_equity(sealed, calendar, *period,
    initial_capital)` (period = Periods.sealed), so MDD and Sharpe/DSR include idle days.
    Daily returns = pct_change of that series. `sr_var_trials` must be the variance of the
    per-observation (daily, non-annualized) Sharpe across the `n_trials` trials.

    t_stat: sealed net_r; profit_factor / top_removed / sealed_positive / min_trades: sealed
    fills; valid_positive: valid fills' net_pnl sum. NaN metrics fail their check.
    `failed` lists the failing names in CHECKS order.
    """
    pnl = sealed.fills["net_pnl"].to_numpy(dtype=float)
    eq = period_equity(sealed, calendar, period[0], period[1], initial_capital)
    rets = eq.pct_change().dropna().to_numpy(dtype=float)
    sr = sharpe(rets, periods_per_year=1)
    sk = float(skew(rets)) if len(rets) > 2 and np.isfinite(sr) else float("nan")
    ku = float(kurtosis(rets, fisher=False)) if len(rets) > 3 and np.isfinite(sr) else float("nan")
    m = {
        "n_trades": int(len(pnl)),
        "t_stat": t_stat(sealed.fills["net_r"].to_numpy(dtype=float)),
        "dsr": deflated_sharpe_ratio(sr, n_trials, len(rets), sk, ku, sr_var_trials),
        "pbo": float(pbo),
        "top_removed": profit_without_top(pnl, c.top_removed_n),
        "profit_factor": profit_factor(pnl),
        "mdd": max_drawdown(eq),
        "valid_net_pnl": float(valid.fills["net_pnl"].astype(float).sum()),
        "sealed_net_pnl": float(pnl.sum()),
        "sharpe": sharpe(rets),
        "sr": sr,
        "skew": sk,
        "kurt": ku,
        "n_obs": int(len(rets)),
        "n_trials": int(n_trials),
        "sr_var_trials": float(sr_var_trials),
    }
    ok = {
        "t_stat": m["t_stat"] >= c.t_min,
        "dsr": m["dsr"] >= c.dsr_min,
        "pbo": m["pbo"] <= c.pbo_max,
        "top_removed": m["top_removed"] > 0,
        "profit_factor": m["profit_factor"] >= c.pf_min,
        "mdd": m["mdd"] <= c.mdd_max,
        "valid_positive": m["valid_net_pnl"] > 0,
        "sealed_positive": m["sealed_net_pnl"] > 0,
        "min_trades": m["n_trades"] >= c.min_trades,
    }
    failed = [name for name in CHECKS if not ok[name]]
    return Verdict(passed=not failed, failed=failed, metrics=m)
