"""Task 6: `_negf` negative-event filter variants + no-look-ahead regressions."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autotrade.disclosure import events as E
from autotrade.rules.registry import EVENT_FAMILIES, SWING_FAMILIES
from autotrade.rules.swing.breakout import Breakout
from autotrade.rules.swing.disclosure import Disclosure
from autotrade.rules.swing.ma_trend import MaTrend
from autotrade.rules.swing.momentum import Momentum
from autotrade.rules.swing.negf import with_negative_filter
from autotrade.rules.swing.reversal import Reversal
from autotrade.types import EqualWeight
from tests.fixtures import make_panel

BASES = {"breakout": Breakout, "ma_trend": MaTrend, "momentum": Momentum,
         "reversal": Reversal}
SYMS = ["A", "B", "C", "D"]


def ev(p, rows):
    """rows: (symbol, row index, direction)."""
    return pd.DataFrame([{"symbol": s, "signal_date": p.close.index[i], "event": "e",
                          "direction": d, "size": np.nan, "size_ok": True, "bundle_a": d == 1,
                          "bundle_b": False, "positive": d == 1, "filing_id": f"x{n}"}
                         for n, (s, i, d) in enumerate(rows)], columns=E.EVENT_COLUMNS)


def panel(rows=(), n=300, seed=5):
    p = make_panel(n, SYMS, seed=seed)
    return replace(p, events=ev(p, rows))


def first_entry(base_rule, p):
    s = base_rule.signals(p)
    pos = np.argwhere(s.entries.to_numpy())
    return s, pos


def pick(family="breakout"):
    base = BASES[family]
    return base(base.grid()[0])


def neg_rule(family="breakout"):
    base = BASES[family]
    return SWING_FAMILIES[f"{family}_negf"](base.grid()[0])


def test_registry_and_ids():
    for f, base in BASES.items():
        cls = SWING_FAMILIES[f"{f}_negf"]
        assert cls.family == f"{f}_negf" and issubclass(cls, base)
        assert cls.PARAM_KEYS == base.PARAM_KEYS and cls.grid() == base.grid()
        assert f"{f}_negf" in EVENT_FAMILIES
        assert cls(base.grid()[0]).rule_id.startswith(f"{f}_negf:")
        assert cls(base.grid()[0]).rule_id != base(base.grid()[0]).rule_id
    assert len(EVENT_FAMILIES) == 5


def test_negf_rule_ids_differ_from_base():
    r = SWING_FAMILIES["breakout_negf"](Breakout.grid()[0])
    assert r.rule_id.startswith("breakout_negf:")
    assert not Breakout(Breakout.grid()[0]).rule_id.startswith("breakout_negf")


def test_negative_window_matches_config():
    for m in ("kr", "us"):
        assert E.load_table(m).negative_window == 20


def test_requires_events():
    p = make_panel(100, SYMS, seed=5)
    with pytest.raises(ValueError, match="Panel.events"):
        neg_rule().signals(p)


def test_without_negative_events_equals_base():
    plain = make_panel(300, SYMS, seed=5)
    p = panel([("A", 50, 1), ("B", 80, 1)])
    for f in BASES:
        a, b = pick(f).signals(plain), neg_rule(f).signals(p)
        for name in ("entries", "exits", "stop_pct", "confidence"):
            pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))
        assert (a.entry_at, a.exit_at, a.max_hold_days, a.equal_weight) == \
               (b.entry_at, b.exit_at, b.max_hold_days, b.equal_weight)


def _entry_cell(f="breakout"):
    p0 = panel()
    s, pos = first_entry(pick(f), p0)
    assert len(pos), "fixture produced no entries"
    return p0, pos


def test_entry_blocked_for_20_rows_after_negative_and_allowed_on_21():
    p0, pos = _entry_cell()
    # find an entry cell with another base entry at row r+k for k in 0..19 is not guaranteed:
    # drive it directly by placing the negative event so that a known entry lands at offsets
    base = pick().signals(p0).entries
    r, c = pos[len(pos) // 2]
    sym = base.columns[c]
    for off, blocked in [(0, True), (10, True), (19, True), (20, False)]:
        # negative event at row r-off: entry at r is `off` rows after it
        if r - off < 0:
            continue
        p = panel([(sym, r - off, -1)])
        got = neg_rule().signals(p).entries.iloc[r, c]
        assert got == (not blocked), off
    # other symbols are unaffected
    p = panel([(sym, r, -1)])
    other = neg_rule().signals(p).entries.drop(columns=sym)
    pd.testing.assert_frame_equal(other, base.drop(columns=sym))


def test_repeated_negatives_extend_the_block():
    p0, pos = _entry_cell()
    base = pick().signals(p0).entries
    r, c = pos[len(pos) // 2]
    sym = base.columns[c]
    if r < 25:
        pytest.skip("entry too early")
    p = panel([(sym, r - 25, -1), (sym, r - 10, -1)])    # the first alone would not block
    assert not neg_rule().signals(p).entries.iloc[r, c]
    p = panel([(sym, r - 25, -1)])
    assert neg_rule().signals(p).entries.iloc[r, c]


def test_negative_event_forces_exit_next_open():
    p = panel([("A", 100, -1)])
    s = neg_rule().signals(p)
    assert s.exits.iloc[100, 0] and s.exit_at == "next_open"
    base = pick().signals(panel())
    assert not s.exits.drop(index=p.close.index[100], errors="ignore").iloc[:, 0].eq(
        base.exits.iloc[:, 0]).all() or base.exits.iloc[100, 0]
    # only that cell was added
    diff = s.exits & ~base.exits
    assert diff.sum().sum() == (0 if base.exits.iloc[100, 0] else 1)
    assert s.exits.shape == base.exits.shape and list(s.exits.columns) == SYMS


def test_positive_events_do_not_block_or_exit():
    p0, pos = _entry_cell()
    r, c = pos[0]
    p = panel([(SYMS[c], max(r - 1, 0), 1), (SYMS[c], r, 0)])
    s = neg_rule().signals(p)
    pd.testing.assert_frame_equal(s.entries, pick().signals(p0).entries)


def test_momentum_negf_keeps_equal_weight():
    p = panel([("A", 100, -1)])
    r = SWING_FAMILIES["momentum_negf"](Momentum.grid()[3])
    s = r.signals(p)
    assert s.equal_weight == EqualWeight(Momentum.grid()[3]["top_k"])
    assert s.max_hold_days == Momentum.grid()[3]["rebalance"]


@pytest.mark.parametrize("f", list(BASES))
def test_next_open_everywhere(f):
    s = neg_rule(f).signals(panel([("A", 100, -1)]))
    assert s.entry_at == "next_open" and s.exit_at == "next_open"


def test_all_excludes_and_events_includes_negf():
    non_event = set(SWING_FAMILIES) - EVENT_FAMILIES
    assert not any(n.endswith("_negf") for n in non_event)
    assert {f"{f}_negf" for f in BASES} <= EVENT_FAMILIES


# ---------------------------------------------------------------- no look-ahead
def _tampered(p, t, rng):
    """Same panel up to row t; prices and events strictly after t replaced."""
    n = len(p.close)
    def scramble(df):
        out = df.copy()
        out.iloc[t + 1:] = out.iloc[t + 1:].to_numpy() * rng.uniform(0.5, 1.5, out.iloc[t + 1:].shape)
        return out
    ev_ = p.events
    keep = ev_[ev_["signal_date"] <= p.close.index[t]]
    extra = ev(p, [(s, int(i), d) for s, i, d in
                   zip(rng.choice(SYMS, 10), rng.integers(t + 1, n, 10),
                       rng.choice([-1, 1], 10))])
    return replace(p, close=scramble(p.close), open=scramble(p.open), high=scramble(p.high),
                   low=scramble(p.low), volume=scramble(p.volume),
                   events=pd.concat([keep, extra], ignore_index=True))


def _assert_same_through(a, b, t):
    for name in ("entries", "exits", "stop_pct", "confidence"):
        pd.testing.assert_frame_equal(getattr(a, name).iloc[:t + 1], getattr(b, name).iloc[:t + 1],
                                      check_exact=False, rtol=1e-12)


@pytest.mark.parametrize("f", list(BASES))
def test_negf_no_look_ahead(f):
    rng = np.random.default_rng(1)
    p = panel([("A", 60, -1), ("B", 120, -1), ("C", 150, 1), ("A", 200, -1)])
    t = 160
    q = _tampered(p, t, rng)
    _assert_same_through(neg_rule(f).signals(p), neg_rule(f).signals(q), t)


@pytest.mark.parametrize("confirm", ["none", "ma20", "high20"])
def test_disclosure_no_look_ahead(confirm):
    rng = np.random.default_rng(2)
    p = panel([("A", 60, 1), ("B", 120, 1), ("C", 150, 1), ("D", 159, 1), ("A", 200, 1)])
    r = Disclosure({"bundle": "positive", "hold": 5, "k": 2.0, "confirm": confirm,
                    "sized": False})
    t = 160
    _assert_same_through(r.signals(p), r.signals(_tampered(p, t, rng)), t)
