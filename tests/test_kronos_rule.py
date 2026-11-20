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




# --- C2: seeded sampling and the probe's data window ---------------------------------------
import importlib.util  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from autotrade.rules.swing import _kronos_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_predict_paths_passes_a_fixed_seed(monkeypatch):
    seen = {}

    def fake(history, n_paths, horizon, model_name, seed):
        seen["seed"] = seed
        return np.zeros((n_paths, horizon))
    monkeypatch.setattr(_kronos_model, "sample_paths", fake)
    hist = make_panel(40, ["A"], seed=1).close.rename(columns={"A": "close"})
    hist.attrs["horizon"] = 5
    kr._predict_paths(hist, 4)
    assert seen["seed"] == kr.SEED


def test_sample_paths_is_reproducible_for_a_seed(monkeypatch):
    import torch

    class FakePredictor:
        def predict(self, df, x_timestamp, y_timestamp, pred_len, **kw):
            noise = torch.rand(pred_len).numpy() + np.random.random(pred_len)
            return pd.DataFrame({"close": df["close"].iloc[-1] * (1 + noise)})
    monkeypatch.setattr(_kronos_model, "_predictor", lambda name: FakePredictor())
    p = make_panel(40, ["A"], seed=1)
    hist = pd.DataFrame({k: getattr(p, k)["A"] for k in ("open", "high", "low", "close",
                                                           "volume")})
    a = _kronos_model.sample_paths(hist, 3, 5, "m", seed=7)
    b = _kronos_model.sample_paths(hist, 3, 5, "m", seed=7)
    c = _kronos_model.sample_paths(hist, 3, 5, "m", seed=8)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def _probe():
    spec = importlib.util.spec_from_file_location("kronos_probe", ROOT / "scripts" /
                                                  "kronos_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def probe_store(tmp_path, monkeypatch):
    from autotrade.data import store
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("AUTOTRADE_RESEARCH", str(tmp_path / "research"))
    p = make_panel(900, ["CHEAP", "PRICEY", "MID"], seed=4)
    vol = {"CHEAP": 5e6, "PRICEY": 1e3, "MID": 1e5}            # PRICEY: high price, tiny ADV
    px = {"CHEAP": 1.0, "PRICEY": 50.0, "MID": 2.0}
    for sym in p.close.columns:
        df = pd.DataFrame({k: getattr(p, k)[sym] * px[sym] for k in ("open", "high", "low",
                                                                       "close")})
        df["volume"] = vol[sym]
        store.save_daily("kr", sym, df)
    store.save_meta("kr", p.meta)
    return p


def test_probe_refuses_real_data_without_periods(probe_store):
    with pytest.raises(SystemExit, match="lab screen"):
        _probe().get_panel("kr", 2, 100)


def test_probe_measures_on_train_period_and_picks_by_adv(probe_store):
    from autotrade.pipeline import periods_path
    from autotrade.validate.periods import new_record, save_record
    rec = new_record("swing", "kr", probe_store.close.index, (0.6, 0.2, 0.2))
    save_record(periods_path("swing", "kr"), rec)
    panel, cols, synthetic = _probe().get_panel("kr", 2, 100)
    assert not synthetic
    assert panel.close.index.max() == pd.Timestamp(rec.periods.train[1])
    assert cols == ["CHEAP", "MID"]                     # by 20-day ADV, not by price


def test_probe_synthetic_fallback_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "empty"))
    panel, cols, synthetic = _probe().get_panel("kr", 3, 50)
    assert synthetic and len(cols) == 3
