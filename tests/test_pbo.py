import numpy as np
import pandas as pd
import pytest

from autotrade.validate.pbo import pbo_cscv


def test_pbo_random_near_half():
    rng = np.random.default_rng(0)
    perf = pd.DataFrame(rng.normal(0, 1, (1000, 20)))
    assert 0.3 < pbo_cscv(perf) < 0.7


def test_pbo_dominant_strategy_low():
    rng = np.random.default_rng(1)
    perf = pd.DataFrame(rng.normal(0, 1, (1000, 20)))
    perf[0] += 0.5
    assert pbo_cscv(perf) < 0.1


# Two blocks of two rows (S=2): the two combinations are IS=block0 and IS=block1.
CONSISTENT = pd.DataFrame({"A": [1.0, 1.2, 1.0, 1.2], "B": [0.1, -0.1, 0.1, 0.3]})
FLIPPED = pd.DataFrame({"A": [1.0, 1.2, -1.0, -1.2], "B": [0.1, -0.1, 0.1, 0.3]})


def test_pbo_hand_cases():
    assert pbo_cscv(CONSISTENT, n_splits=2) == 0.0  # A best in-sample and out-of-sample
    assert pbo_cscv(FLIPPED, n_splits=2) == 1.0     # IS winner is always the OOS loser


def test_pbo_drops_remainder_rows():
    extra = pd.DataFrame({"A": [-1000.0], "B": [1000.0]})
    assert pbo_cscv(pd.concat([CONSISTENT, extra], ignore_index=True), n_splits=2) == 0.0


def test_pbo_rejects_bad_input():
    with pytest.raises(ValueError):
        pbo_cscv(CONSISTENT, n_splits=3)       # odd
    with pytest.raises(ValueError):
        pbo_cscv(CONSISTENT[["A"]], n_splits=2)  # one candidate
    with pytest.raises(ValueError):
        pbo_cscv(CONSISTENT, n_splits=8)       # fewer rows than blocks
    with pytest.raises(ValueError):
        pbo_cscv(CONSISTENT.assign(A=[1.0, np.nan, 1.0, 1.2]), n_splits=2)


def test_pbo_constant_column_never_selected():
    # A constant column has zero dispersion; rounding noise must not make its Sharpe huge.
    assert pbo_cscv(FLIPPED.assign(C=0.1), n_splits=2) == 1.0
