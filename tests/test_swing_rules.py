import numpy as np
import pandas as pd
import pytest

from autotrade.rules.registry import SWING_FAMILIES
from autotrade.rules.swing.breakout import Breakout
from autotrade.types import EqualWeight
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
                                   "reversal", "overnight", "tom", "etf_rotation"}


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


# --- I4: momentum / reversal rank within the day's universe ---------------------------------
import numpy as np  # noqa: E402

from autotrade.config import load_lab, load_market_raw, load_pocket  # noqa: E402
from autotrade.universe import eligible_mask  # noqa: E402


def test_momentum_ranks_only_eligible_symbols():
    p = make_panel(300, ["A", "B", "C", "D"], seed=8)
    p.close.iloc[:, 0] = 100 * np.exp(np.linspace(0, 3, 300))     # A: by far the best return
    rule = SWING_FAMILIES["momentum"]({"lookback": 21, "rebalance": 10, "top_k": 1, "k": 2})
    assert rule.signals(p).entries["A"].iloc[30::10].all()
    elig = pd.DataFrame(True, index=p.close.index, columns=p.close.columns)
    elig["A"] = False                                              # A outside the universe
    s = rule.signals(p, eligible=elig)
    assert not s.entries["A"].any()
    assert (s.entries.iloc[30::10].sum(axis=1) == 1).all()         # next best takes rank 1
    assert (s.confidence.to_numpy()[s.entries.to_numpy()] == 1.0).all()


def test_reversal_ranks_only_eligible_symbols():
    p = make_panel(100, ["A", "B", "C"], seed=2)
    p.close.iloc[60:, 0] *= 0.5                                    # A: biggest loser at 60
    rule = SWING_FAMILIES["reversal"]({"lookback": 5, "top_k": 1, "hold": 5, "k": 2})
    elig = pd.DataFrame(True, index=p.close.index, columns=p.close.columns)
    elig.iloc[60, 0] = False
    s = rule.signals(p, eligible=elig)
    assert not s.entries["A"].iloc[60]
    loss = -(p.close.iloc[60] / p.close.iloc[55] - 1)
    others = loss.drop("A")
    if (others > 0).any():
        assert s.entries.iloc[60].sum() == 1 and s.entries.iloc[60][others.idxmax()]


@pytest.mark.parametrize("family", sorted(set(SWING_FAMILIES) - {"momentum", "reversal", "etf_rotation"}))
def test_other_families_ignore_eligible(family):
    p = make_panel(300, ["A", "B", "C"], seed=5, products={s: "etf" for s in "ABC"})
    rule = SWING_FAMILIES[family](SWING_FAMILIES[family].grid()[0])
    elig = pd.DataFrame(np.random.default_rng(0).random(p.close.shape) > 0.5,
                        index=p.close.index, columns=p.close.columns)
    a, b = rule.signals(p), rule.signals(p, eligible=elig)
    pd.testing.assert_frame_equal(a.entries, b.entries)
    pd.testing.assert_frame_equal(a.exits, b.exits)


@pytest.mark.parametrize("family", ["momentum", "reversal", "etf_rotation"])
def test_ranking_with_universe_ignores_future(family):
    p = make_panel(400, [f"S{i}" for i in range(6)], seed=4,
                   products={f"S{i}": "etf" for i in range(3)})
    lab = load_lab()
    rule = SWING_FAMILIES[family](SWING_FAMILIES[family].grid()[0])
    mc, pc = load_market_raw("us"), load_pocket("swing")
    s1 = rule.signals(p, eligible=eligible_mask(p, mc, pc, lab, "paper", 1400.0))
    for f in (p.close, p.high, p.volume):
        f.iloc[300:] *= 3
    s2 = rule.signals(p, eligible=eligible_mask(p, mc, pc, lab, "paper", 1400.0))
    pd.testing.assert_frame_equal(s1.entries.iloc[:300], s2.entries.iloc[:300])
    pd.testing.assert_frame_equal(s1.exits.iloc[:300], s2.exits.iloc[:300])


# ---------------------------------------------------------------- P7: tom and the panel end
def _tom_entries(p, n=2):
    return SWING_FAMILIES["tom"]({"n": n, "m": 3, "k": 2}).signals(p).entries["A"]


@pytest.mark.parametrize("with_calendar", [False, True])
def test_tom_truncated_panel_creates_no_entry_the_full_panel_lacks(with_calendar):
    from dataclasses import replace
    full = make_panel(300, ["A"], seed=3)
    if with_calendar:
        full = replace(full, calendar=full.close.index)
    for n in (1, 2, 5):
        ref = _tom_entries(full, n)
        for t in range(150, 300):
            cut = full.slice(full.close.index[0], full.close.index[t])
            got = _tom_entries(cut, n)
            extra = got & ~ref.loc[got.index]
            assert not extra.any(), (n, full.close.index[t], list(extra[extra].index))


def test_tom_final_month_needs_a_known_month_end():
    from dataclasses import replace
    full = make_panel(300, ["A"], seed=3)
    idx = full.close.index
    ref = _tom_entries(full)
    jan = idx[idx.to_period("M") == pd.Period("2016-01")]
    mar = idx[idx.to_period("M") == pd.Period("2016-03")]
    assert jan[-1] == pd.Timestamp("2016-01-29") and mar[-1] == pd.Timestamp("2016-03-31")
    # Jan 29 (Fri) is the month's last trading day, but only the calendar can tell
    cut = full.slice(idx[0], jan[-1])
    assert not _tom_entries(cut).any()
    with_cal = replace(cut, calendar=idx)
    assert _tom_entries(with_cal).equals(ref.loc[cut.close.index]) and ref.loc[jan].any()
    # Mar 31 is the month's last calendar day: complete even without a calendar
    cut = full.slice(idx[0], mar[-1])
    assert _tom_entries(cut).equals(ref.loc[cut.close.index]) and ref.loc[mar].any()
    # a calendar that ends at the panel end tells nothing more
    mid = full.slice(idx[0], pd.Timestamp("2016-02-25"))
    assert not _tom_entries(replace(mid, calendar=mid.close.index)).loc["2016-02"].any()


def test_rotation_families_size_equal_weight_and_others_by_risk():
    p = make_panel(300, ["A", "B", "C", "D"], seed=8, products={s: "etf" for s in "ABCD"})
    mom = SWING_FAMILIES["momentum"]({"lookback": 21, "rebalance": 10, "top_k": 3, "k": 2})
    assert mom.signals(p).equal_weight == EqualWeight(slots=3, max_weight=None)  # pocket cap
    rot = SWING_FAMILIES["etf_rotation"]({"lookback": 63, "rebalance": 20, "top_k": 2, "k": 3})
    assert rot.signals(p).equal_weight == EqualWeight(slots=2, max_weight=1.0)
    assert Breakout(Breakout.grid()[0]).signals(p).equal_weight is None


def test_etf_rotation_holds_only_etfs_with_a_positive_lookback_return():
    p = make_panel(400, ["E1", "E2", "E3", "S"], seed=9,
                   products={"E1": "etf", "E2": "etf", "E3": "etf", "S": "stock"})
    n = len(p.close)
    p.close["S"] = 100 * np.exp(np.linspace(0, 4, n))      # best return, but a stock
    p.close["E1"] = 100 * np.exp(np.linspace(0, 1, n))     # best ETF
    p.close["E2"] = 100 * np.exp(np.linspace(0, -1, n))    # falling: absolute momentum < 0
    p.close["E3"] = 100 * np.exp(np.linspace(0, 0.5, n))
    rule = SWING_FAMILIES["etf_rotation"]({"lookback": 63, "rebalance": 20, "top_k": 2, "k": 3})
    s = rule.signals(p)
    assert not s.entries["S"].any() and not s.entries["E2"].any()
    reb = s.entries.iloc[80::20]
    assert reb["E1"].all() and reb["E3"].all()
    assert s.max_hold_days == 20 and s.entry_at == "next_open"


def test_etf_rotation_goes_to_cash_when_every_etf_falls():
    p = make_panel(300, ["E1", "E2"], seed=9, products={"E1": "etf", "E2": "etf"})
    n = len(p.close)
    p.close["E1"] = 100 * np.exp(np.linspace(0, -1, n))
    p.close["E2"] = 100 * np.exp(np.linspace(0, -0.5, n))
    rule = SWING_FAMILIES["etf_rotation"]({"lookback": 63, "rebalance": 10, "top_k": 1, "k": 3})
    assert not rule.signals(p).entries.any().any()
