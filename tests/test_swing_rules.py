import pandas as pd
import pytest

from autotrade.rules.registry import SWING_FAMILIES
from autotrade.rules.swing.breakout import Breakout
from tests.fixtures import make_panel


@pytest.mark.parametrize("family", SWING_FAMILIES)
def test_signals_shape_and_ranges(family):
    p = make_panel(400, ["A", "B", "C"], seed=3)
    for params in SWING_FAMILIES[family].grid()[:3]:
        s = SWING_FAMILIES[family](params).signals(p)
        assert s.entries.shape == p.close.shape and s.entries.dtypes.eq(bool).all()
        assert s.exits.shape == p.close.shape and s.exits.dtypes.eq(bool).all()
        assert ((s.confidence >= 0) & (s.confidence <= 1) | s.confidence.isna()).all().all()
        assert ((s.stop_pct > 0) | ~s.entries).all().all()
        assert 1 <= s.max_hold_days <= 20


@pytest.mark.parametrize("family", SWING_FAMILIES)
def test_signal_at_t_ignores_future(family):
    p = make_panel(400, ["A", "B", "C"], seed=4)
    r = SWING_FAMILIES[family](SWING_FAMILIES[family].grid()[0])
    s1 = r.signals(p)
    p.close.iloc[300:] *= 3
    p.high.iloc[300:] *= 3
    s2 = r.signals(p)
    pd.testing.assert_frame_equal(s1.entries.iloc[:300], s2.entries.iloc[:300])
    pd.testing.assert_frame_equal(s1.exits.iloc[:300], s2.exits.iloc[:300])


@pytest.mark.parametrize("family", SWING_FAMILIES)
def test_families_emit_some_entries(family):
    p = make_panel(600, ["A", "B", "C", "D"], seed=7, products={s: "etf" for s in "ABCD"})
    hits = [SWING_FAMILIES[family](g).signals(p).entries.any().any() for g in SWING_FAMILIES[family].grid()]
    assert any(hits)


def test_overnight_only_etf():
    p = make_panel(50, ["069500", "A"], seed=5, products={"069500": "etf", "A": "stock"})
    s = SWING_FAMILIES["overnight"]({}).signals(p)
    assert not s.entries["A"].any() and s.entries["069500"].any()
    assert (s.stop_pct["069500"][s.entries["069500"]] == 0.05).all()
    assert (s.confidence["069500"][s.entries["069500"]] == 0.5).all()
    assert s.max_hold_days == 1 and s.entry_at == "close" and s.exit_at == "next_open"


def test_grid_size_capped():
    assert all(len(f.grid()) <= 50 for f in SWING_FAMILIES.values())


def test_registry_has_all_families():
    assert set(SWING_FAMILIES) == {"breakout", "ma_trend", "oversold", "momentum",
                                   "reversal", "overnight", "tom"}


def test_rule_id_stable():
    assert Breakout({"n": 20, "m": 10, "k": 2}).rule_id == Breakout({"k": 2, "m": 10, "n": 20}).rule_id
    assert Breakout({"n": 20, "m": 10, "k": 2}).rule_id != Breakout({"n": 40, "m": 10, "k": 2}).rule_id
    assert Breakout({"n": 20, "m": 10, "k": 2}).rule_id.startswith("breakout:")


def test_unknown_or_missing_params_rejected():
    with pytest.raises(ValueError):
        Breakout({"n": 20, "m": 10})
    with pytest.raises(ValueError):
        Breakout({"n": 20, "m": 10, "k": 2, "z": 1})


def test_breakout_entry_semantics():
    p = make_panel(100, ["A"], seed=1)
    p.close.iloc[50, 0] = p.close["A"].iloc[:50].max() * 1.5
    s = SWING_FAMILIES["breakout"]({"n": 20, "m": 10, "k": 2}).signals(p)
    assert s.entries["A"].iloc[50]
    assert s.entry_at == "next_open" and s.exit_at == "next_open"


def test_momentum_rebalance_dates_and_exits():
    p = make_panel(300, ["A", "B", "C", "D"], seed=8)
    s = SWING_FAMILIES["momentum"]({"lookback": 21, "rebalance": 10, "top_k": 2, "k": 2}).signals(p)
    off = [i for i in range(len(p.close)) if i % 10 != 0]
    assert not s.entries.iloc[off].any().any() and not s.exits.iloc[off].any().any()
    assert s.max_hold_days == 10
    reb = s.entries.iloc[30::10]
    assert (reb.sum(axis=1) <= 2).all()
    assert not (s.entries & s.exits).any().any()


def test_reversal_picks_biggest_losers():
    p = make_panel(100, ["A", "B", "C"], seed=2)
    p.close.iloc[60:, 0] *= 0.5
    s = SWING_FAMILIES["reversal"]({"lookback": 5, "top_k": 1, "hold": 5, "k": 2}).signals(p)
    assert s.entries["A"].iloc[60] and s.entries.iloc[60].sum() == 1
    assert s.max_hold_days == 5


def test_tom_calendar_offsets():
    p = make_panel(120, ["A"], seed=3)
    s = SWING_FAMILIES["tom"]({"n": 2, "m": 3, "k": 2}).signals(p)
    idx = p.close.index
    months = pd.Series(idx.to_period("M"), index=idx)
    last_pos = [int(months.reset_index(drop=True)[months.reset_index(drop=True) == m].index[-1])
                for m in months.unique()]
    for lp in last_pos[1:-1]:
        assert s.entries["A"].iloc[lp - 2]
    ent = s.entries["A"]
    assert ent.sum() <= len(last_pos)
    assert s.entry_at == "close" and s.exit_at == "close" and s.max_hold_days == 5
    ex_pos = [i for i, v in enumerate(s.exits["A"]) if v]
    first_pos = [int(months.reset_index(drop=True)[months.reset_index(drop=True) == m].index[0])
                 for m in months.unique()]
    assert ex_pos == [fp + 2 for fp in first_pos[1:]]


def test_breakout_exit_fires_on_break_below_prior_m_day_low():
    p = make_panel(100, ["A"], seed=1)
    for df in (p.open, p.high, p.low, p.close):
        df["A"] = 100.0
    p.low.iloc[:, 0] = 99.0
    p.low.iloc[40, 0] = 90.0       # low 40 bars ago: inside m=20 window at t=50? 50-20=30..49 -> yes
    p.close.iloc[50, 0] = 95.0     # below prior 5-day low (99) ; above 90
    p.low.iloc[50, 0] = 94.0       # realistic bar: low <= close
    s5 = SWING_FAMILIES["breakout"]({"n": 20, "m": 5, "k": 2}).signals(p)
    s20 = SWING_FAMILIES["breakout"]({"n": 20, "m": 20, "k": 2}).signals(p)
    assert s5.exits["A"].tolist() == [i == 50 for i in range(100)]
    assert not s20.exits["A"].any()   # prior 20-day low is 90 -> 95 is not below it
