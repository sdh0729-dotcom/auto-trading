from datetime import date

import numpy as np
import pandas as pd
import pytest

from autotrade.config import Criteria
from autotrade.sizing.portfolio import FILL_COLUMNS, PortfolioResult
from autotrade.validate.criteria import Verdict, evaluate, period_equity

CRIT = Criteria(t_min=2.0, dsr_min=0.95, pbo_max=0.5, top_removed_n=5, pf_min=1.2,
                mdd_max=0.15, min_trades=100, max_final_candidates=3, cost_r_max=0.15)
CAL = pd.bdate_range("2023-01-02", periods=300, name="date")
SEALED = (CAL[0].date(), CAL[-1].date())
CAP = 1_000_000.0
NAMES = ["t_stat", "dsr", "pbo", "top_removed", "profit_factor", "mdd", "valid_positive",
         "sealed_positive", "min_trades"]


def fills(pnl, r=None) -> pd.DataFrame:
    pnl = np.asarray(pnl, dtype=float)
    r = pnl / 1000.0 if r is None else np.asarray(r, dtype=float)
    n = len(pnl)
    d = CAL[np.arange(n) % (len(CAL) - 1)]
    df = pd.DataFrame({"symbol": "A", "entry_date": d, "entry_price": 100.0, "exit_date": d,
                       "exit_price": 100.0, "exit_reason": "signal", "stop_pct": 0.05,
                       "confidence": 1.0, "qty": 1.0, "net_pnl": pnl, "net_r": r})
    return df[FILL_COLUMNS]


def smooth_equity(drift=0.002, noise=0.004, seed=0, index=CAL) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, noise, len(index))
    return pd.Series(CAP * np.cumprod(1 + rets), index=index, name="equity")


def good_pnl(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(1500.0, 3000.0, n)  # PF ~ 3, t ~ 5, net clearly positive


def result(pnl, equity=None, r=None) -> PortfolioResult:
    eq = smooth_equity() if equity is None else equity
    return PortfolioResult(eq, fills(pnl, r), pd.DataFrame())


def run(sealed, valid=None, pbo=0.2, n_trials=100, sr_var_trials=0.001, c=CRIT):
    valid = valid or result(good_pnl(seed=9))
    return evaluate(sealed, valid, pbo=pbo, n_trials=n_trials, sr_var_trials=sr_var_trials,
                    c=c, calendar=CAL, period=SEALED, initial_capital=CAP)


def test_criteria_all_pass():
    v = run(result(good_pnl()))
    assert isinstance(v, Verdict)
    assert v.passed and v.failed == [], v
    for k in ["t_stat", "dsr", "pbo", "top_removed", "profit_factor", "mdd", "valid_net_pnl",
              "sealed_net_pnl", "n_trades", "sharpe", "sr", "skew", "kurt", "n_obs",
              "n_trials", "sr_var_trials"]:
        assert k in v.metrics
    assert v.metrics["n_trades"] == 120 and v.metrics["n_obs"] == len(CAL) - 1


def test_criteria_each_failure_reported():
    pnl = np.tile([1100.0, -1000.0], 60)        # PF 1.1, still net positive
    sealed_bad_pf = result(pnl, r=np.tile([1.0, -0.2], 60))
    v = run(sealed_bad_pf)
    assert not v.passed and "profit_factor" in v.failed
    assert v.metrics["profit_factor"] == pytest.approx(1.1)
    assert set(v.failed) <= set(NAMES)


def test_criteria_failed_names_exhaustive():
    """Everything bad at once -> every name reported, in the documented order."""
    pnl = np.concatenate([[1e6] * 5, -np.ones(94)])     # 99 trades, top-5 carried
    eq = smooth_equity(drift=-0.002)                     # falling equity, MDD >> 15%
    v = run(result(pnl, eq, r=np.zeros(99)), valid=result(-np.ones(10)), pbo=0.9,
            n_trials=1000, sr_var_trials=0.5)
    assert v.failed == [n for n in NAMES if n not in ("profit_factor", "sealed_positive")]
    v2 = run(result(-np.ones(99), eq), valid=result(-np.ones(10)), pbo=0.9)
    assert v2.failed == NAMES


def test_criteria_min_trades():
    assert "min_trades" in run(result(good_pnl(99))).failed
    assert "min_trades" not in run(result(good_pnl(100))).failed


def test_criteria_top5_removed():
    pnl = np.concatenate([[50_000.0] * 5, np.tile([900.0, -1000.0], 60)])
    v = run(result(pnl))
    assert "top_removed" in v.failed and "sealed_positive" not in v.failed
    assert v.metrics["top_removed"] < 0


def test_criteria_thresholds_inclusive():
    assert "pbo" not in run(result(good_pnl()), pbo=0.5).failed
    assert "pbo" in run(result(good_pnl()), pbo=0.51).failed


def test_criteria_mdd_valid_dsr_t():
    eq = smooth_equity()
    eq.iloc[150:] *= 0.8                                   # a 20% hole
    assert "mdd" in run(result(good_pnl(), eq)).failed
    assert "valid_positive" in run(result(good_pnl()), valid=result(-good_pnl())).failed
    assert "dsr" in run(result(good_pnl()), n_trials=10_000, sr_var_trials=1.0).failed
    noise = np.random.default_rng(5).normal(0, 1, 120)
    assert "t_stat" in run(result(good_pnl(), r=noise - noise.mean())).failed


def test_period_equity_fills_idle_days():
    eq = pd.Series([1.1e6, 1.2e6, 1.15e6], index=CAL[[10, 11, 15]], name="equity")
    pe = period_equity(PortfolioResult(eq, fills([]), pd.DataFrame()), CAL, CAL[5].date(),
                       CAL[20].date(), CAP)
    assert list(pe.index) == list(CAL[5:21])
    assert (pe.loc[CAL[5]:CAL[9]] == CAP).all()
    assert pe.loc[CAL[12]] == 1.2e6 and pe.loc[CAL[14]] == 1.2e6
    assert (pe.loc[CAL[15]:] == 1.15e6).all()


def test_period_equity_empty_and_misaligned():
    empty = PortfolioResult(pd.Series(dtype=float, index=pd.DatetimeIndex([])), fills([]),
                            pd.DataFrame())
    pe = period_equity(empty, CAL, SEALED[0], SEALED[1], CAP)
    assert len(pe) == len(CAL) and (pe == CAP).all()
    off = pd.Series([1.0], index=pd.DatetimeIndex([pd.Timestamp("2023-01-07")]))  # Saturday
    with pytest.raises(ValueError):
        period_equity(PortfolioResult(off, fills([]), pd.DataFrame()), CAL, SEALED[0],
                      SEALED[1], CAP)


def test_evaluate_uses_calendar_not_trading_days_only():
    """Equity recorded only on 50 trading days still yields one return per calendar day."""
    eq = smooth_equity(index=CAL[100:150])
    v = run(result(good_pnl(), eq))
    assert v.metrics["n_obs"] == len(CAL) - 1
