import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from autotrade.validate.stats import (deflated_sharpe_ratio, max_drawdown, profit_factor,
                                      profit_without_top, sharpe, t_stat)


def test_t_stat_known():
    assert t_stat(np.array([1., 2., 3., 4.])) == pytest.approx(
        2.5 / (np.std([1, 2, 3, 4], ddof=1) / 2))


def test_t_stat_degenerate_is_nan():
    assert np.isnan(t_stat(np.array([1.0])))
    assert np.isnan(t_stat(np.array([])))
    assert np.isnan(t_stat(np.array([2.0, 2.0, 2.0])))  # zero dispersion: undefined


def test_pf_and_mdd():
    assert profit_factor(np.array([10., -5., 20., -10.])) == pytest.approx(2.0)
    assert profit_factor(pd.Series([3., 1.])) == np.inf
    assert np.isnan(profit_factor(np.array([])))
    eq = pd.Series([100., 120., 90., 130., 65.])
    assert max_drawdown(eq) == pytest.approx(0.5)  # 130 -> 65
    assert max_drawdown(pd.Series([1., 2., 3.])) == 0.0
    assert max_drawdown(pd.Series([100., 80., 90.])) == pytest.approx(0.2)


def test_sharpe_known():
    r = pd.Series([0.01, 0.02, -0.01, 0.0])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert sharpe(r, periods_per_year=1) == pytest.approx(r.mean() / r.std(ddof=1))
    assert np.isnan(sharpe(pd.Series([0.01])))


def test_profit_without_top():
    pnl = np.arange(1.0, 11.0)  # 1..10, top five are 6..10
    assert profit_without_top(pnl, 5) == pytest.approx(15.0)
    assert profit_without_top(np.array([100., 50., -1., -2.]), 2) == pytest.approx(-3.0)
    assert profit_without_top(np.array([5., 4.]), 5) == 0.0


def test_dsr_drops_with_more_trials():
    a = deflated_sharpe_ratio(1.5, 1, 1000, 0, 3, 0.25)
    b = deflated_sharpe_ratio(1.5, 500, 1000, 0, 3, 0.25)
    assert 0 <= b < a <= 1


def _dsr_reference(sr, n, t, skew, kurt, var):
    """Independent re-statement of Bailey & Lopez de Prado (2014), eqs. for SR0 and DSR."""
    emc = 0.5772156649
    if n == 1:
        sr0 = 0.0
    else:
        sr0 = np.sqrt(var) * ((1 - emc) * norm.ppf(1 - 1.0 / n)
                              + emc * norm.ppf(1 - 1.0 / (n * np.e)))
    return norm.cdf((sr - sr0) * np.sqrt(t - 1)
                    / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2))


@pytest.mark.parametrize("args", [
    (0.1, 100, 1250, -0.5, 5.0, 0.002),
    (0.05, 20, 500, 0.3, 3.0, 0.0005),
    (0.2, 1, 250, 0.0, 3.0, 0.01),
    (0.08, 1000, 2000, -1.0, 8.0, 0.0001),
])
def test_dsr_reference_value(args):
    assert deflated_sharpe_ratio(*args) == pytest.approx(_dsr_reference(*args), rel=1e-12)


def test_dsr_paper_numerical_example():
    # Bailey & Lopez de Prado (2014) worked example: annualized SR 2.5 over 5y daily (T=1250),
    # N=100 trials, annualized V[SR]=1/2, skew -3, kurtosis 10 -> SR0 ~ 0.1132, DSR ~ 0.9004.
    sr = 2.5 / np.sqrt(250)
    assert deflated_sharpe_ratio(sr, 100, 1250, -3.0, 10.0, 0.5 / 250) == pytest.approx(
        0.9004, abs=1e-3)


def test_dsr_rejects_bad_inputs():
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.1, 0, 100, 0, 3, 0.1)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(0.1, 10, 100, 0, 3, -0.1)
    assert np.isnan(deflated_sharpe_ratio(0.1, 10, 1, 0, 3, 0.1))
