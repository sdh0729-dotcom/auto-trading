import numpy as np
import pandas as pd

from autotrade.rules.indicators import atr, rolling_high, rolling_low, rsi, sma
from tests.fixtures import make_panel


def test_indicators_no_lookahead():
    p = make_panel(100, ["A"], seed=2)
    a1 = atr(p, 14)
    p.high.iloc[60:] *= 2
    a2 = atr(p, 14)
    pd.testing.assert_series_equal(a1["A"].iloc[:60], a2["A"].iloc[:60])


def test_rsi_known_values():
    close = pd.DataFrame({"A": np.arange(1.0, 30.0)})
    r = rsi(close, 2)
    assert (r["A"].iloc[2:] == 100).all()
    assert r["A"].iloc[:1].isna().all()
    down = rsi(pd.DataFrame({"A": np.arange(30.0, 1.0, -1.0)}), 2)
    assert (down["A"].iloc[2:] == 0).all()


def test_atr_is_simple_mean_of_true_range_with_prior_close():
    p = make_panel(30, ["A"], seed=1)
    h, l, c = p.high["A"], p.low["A"], p.close["A"]
    i = 20
    trs = [max(h.iloc[j] - l.iloc[j], abs(h.iloc[j] - c.iloc[j - 1]), abs(l.iloc[j] - c.iloc[j - 1]))
           for j in range(i - 13, i + 1)]
    assert atr(p, 14)["A"].iloc[i] == np.float64(np.mean(trs)) or abs(atr(p, 14)["A"].iloc[i] - np.mean(trs)) < 1e-9


def test_sma_rolling_window_values():
    df = pd.DataFrame({"A": [1.0, 2, 3, 4, 5]})
    assert sma(df, 3)["A"].iloc[-1] == 4.0
    assert rolling_high(df, 3)["A"].iloc[-1] == 5.0
    assert rolling_low(df, 3)["A"].iloc[-1] == 3.0
    assert sma(df, 3)["A"].iloc[:2].isna().all()
