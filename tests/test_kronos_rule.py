import numpy as np
import pandas as pd

from autotrade.rules.registry import SWING_FAMILIES
from autotrade.rules.swing import kronos as kr
from autotrade.rules.swing.kronos import KronosDaily
from tests.fixtures import make_panel

PARAMS = {"horizon": 5, "threshold": 0.01, "k": 2.0}
WARM = 30


def _fixed(paths_fn):
    def fake(history, n_paths):
        return paths_fn(history, n_paths)
    return fake


def test_grid_small_and_valid():
    g = KronosDaily.grid()
    assert 1 <= len(g) <= 12
    assert all(set(x) == {"horizon", "threshold", "k"} and x["horizon"] <= 20 for x in g)


def test_not_in_registry():
    assert "kronos" not in SWING_FAMILIES


def test_entry_and_confidence(monkeypatch):
    p = make_panel(60, ["A", "B"], seed=1)

    def paths(hist, n):
        last = hist["close"].iloc[-1]
        if hist.attrs["symbol"] == "A":   # mean +5%, tiny dispersion
            r = 1.05 + np.array([-0.001, 0.001] * (n // 2))
        else:                             # mean -5%
            r = 0.95 + np.zeros(n)
        return np.tile((last * r)[:, None], (1, 5))

    monkeypatch.setattr(kr, "_predict_paths", _fixed(paths))
    s = KronosDaily(PARAMS).signals(p)
    assert s.entries["A"].iloc[WARM:].all()
    assert not s.entries["B"].any()
    assert (s.confidence["A"].iloc[WARM:] > 0.9).all()
    assert s.entry_at == "next_open" and s.exit_at == "next_open" and s.max_hold_days == 5
    assert ((s.confidence >= 0) & (s.confidence <= 1) | s.confidence.isna()).all().all()


def test_confidence_drops_with_dispersion(monkeypatch):
    p = make_panel(60, ["A"], seed=2)
    rng = np.random.default_rng(0)

    def noisy(hist, n):
        last = hist["close"].iloc[-1]
        r = 1.05 + rng.normal(0, 0.2, n)
        return np.tile((last * r)[:, None], (1, 5))

    monkeypatch.setattr(kr, "_predict_paths", noisy)
    s = KronosDaily({**PARAMS, "threshold": -1.0}).signals(p)
    c = s.confidence["A"].dropna()
    assert len(c) and (c < 0.9).all()


def test_signal_at_t_uses_only_history_through_t(monkeypatch):
    p = make_panel(80, ["A"], seed=3)
    seen = []

    def spy(hist, n):
        seen.append((hist.index[-1], len(hist)))
        last = hist["close"].iloc[-1]
        return np.tile(np.full((n, 1), last * 1.05), (1, 5))

    monkeypatch.setattr(kr, "_predict_paths", spy)
    s1 = KronosDaily(PARAMS).signals(p)
    assert all(ts <= p.close.index[-1] for ts, _ in seen)
    # each call's history ends at its own date: length == position+1
    pos = {d: i + 1 for i, d in enumerate(p.close.index)}
    assert all(pos[ts] == n for ts, n in seen)

    p.close.iloc[60:] *= 3
    p.open.iloc[60:] *= 3
    p.high.iloc[60:] *= 3
    p.low.iloc[60:] *= 3
    s2 = KronosDaily(PARAMS).signals(p)
    pd.testing.assert_frame_equal(s1.entries.iloc[:60], s2.entries.iloc[:60])
    pd.testing.assert_frame_equal(s1.confidence.iloc[:60], s2.confidence.iloc[:60])


