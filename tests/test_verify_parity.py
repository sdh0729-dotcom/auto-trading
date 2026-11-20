from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from autotrade.config import load_market_raw, load_pocket
from autotrade.costs import CostModel
from autotrade.rules.registry import SWING_FAMILIES
from autotrade.screen.simulate import simulate
from autotrade.sizing import run_portfolio
from autotrade.types import TRADE_COLUMNS, Panel, Signals
from autotrade.verify import parity_report, run_nautilus
from tests.fixtures import make_panel

START, END = date(2016, 3, 1), date(2017, 1, 31)   # END < panel end: signals cut at END
KR = replace(load_market_raw("kr"), commission_rate=0.00015)
US = replace(load_market_raw("us"), commission_rate=0.0002)
CM = CostModel(KR, lambda s: "etf" if s == "069500" else "stock")
SWING = load_pocket("swing")


def all_true_mask(p: Panel) -> pd.DataFrame:
    return pd.DataFrame(True, index=p.close.index, columns=p.close.columns)


def timing(rule, p: Panel) -> dict:
    sig = rule.signals(p)
    return {"entry_at": sig.entry_at, "exit_at": sig.exit_at}


@pytest.mark.parametrize("family", ["breakout", "oversold", "overnight", "tom"])
def test_parity_with_simulator(family):
    p = make_panel(300, ["A", "B", "069500"], seed=7, products={"069500": "etf"})
    rule = SWING_FAMILIES[family](SWING_FAMILIES[family].grid()[0])
    elig = all_true_mask(p)
    ours = run_portfolio(simulate(p, rule.signals(p), elig, START, END), p, CM, SWING,
                         5_000_000, 1.0, False, **timing(rule, p)).fills
    naut = run_nautilus(rule, p, elig, START, END, CM, SWING, 5_000_000, 1.0, False)
    assert len(ours) > 0
    assert len(naut) == len(ours)
    assert np.allclose(naut["net_pnl"].sum(), ours["net_pnl"].sum(), rtol=0.01)
    assert (naut["exit_reason"].values == ours["exit_reason"].values).mean() >= 0.95
    rep = parity_report(ours, naut)
    assert rep["matched_share"] == 1.0
    assert rep["exit_reason_agreement"] == 1.0
    assert list(naut.columns) == TRADE_COLUMNS + ["qty", "net_pnl"]
    # Nautilus' own cash account agrees with the trade ledger the strategy reports
    # (Money rounds each fill's notional and fee to whole won: <= 1 won per trade)
    assert naut.attrs["final_balance"] == pytest.approx(5_000_000 + naut["net_pnl"].sum(),
                                                       abs=len(naut))


def test_parity_us_fractional():
    p = make_panel(300, ["A", "B", "SPY"], seed=11, products={"SPY": "etf"})
    cm = CostModel(US, lambda s: "etf" if s == "SPY" else "stock")
    rule = SWING_FAMILIES["breakout"](SWING_FAMILIES["breakout"].grid()[0])
    elig = all_true_mask(p)
    ours = run_portfolio(simulate(p, rule.signals(p), elig, START, END), p, cm, SWING,
                         5_000_000, 0.00075, True, **timing(rule, p)).fills
    naut = run_nautilus(rule, p, elig, START, END, cm, SWING, 5_000_000, 0.00075, True)
    assert len(ours) > 0 and len(naut) == len(ours)
    assert (naut["qty"] % 1 != 0).any()                       # fractional shares
    # USD Money rounds cash to cents per fill -> sizing equity differs by cents
    np.testing.assert_allclose(naut["qty"], ours["qty"], rtol=1e-4)
    assert np.allclose(naut["net_pnl"].sum(), ours["net_pnl"].sum(), rtol=0.01)
    assert (naut["exit_reason"].values == ours["exit_reason"].values).all()


class FixedRule:
    """Test rule that replays hand-written signal rows (by integer row)."""
    family = "fixed"
    params: dict = {}

    def __init__(self, entries: dict, exits: dict, conf: dict, entry_at="close",
                 exit_at="close", max_hold=20):
        self.e, self.x, self.c = entries, exits, conf
        self.entry_at, self.exit_at, self.max_hold = entry_at, exit_at, max_hold

    def signals(self, panel: Panel) -> Signals:
        idx, cols = panel.close.index, panel.close.columns
        ent = pd.DataFrame(False, index=idx, columns=cols)
        ext = pd.DataFrame(False, index=idx, columns=cols)
        for s, rows in self.e.items():
            ent.iloc[rows, cols.get_loc(s)] = True
        for s, rows in self.x.items():
            ext.iloc[rows, cols.get_loc(s)] = True
        conf = pd.DataFrame({s: float(self.c[s]) for s in cols}, index=idx)
        stop = pd.DataFrame(0.05, index=idx, columns=cols)
        return Signals(ent, ext, stop, conf, self.entry_at, self.exit_at, self.max_hold)


def flat_panel(symbols, n=30) -> Panel:
    idx = pd.bdate_range("2024-01-01", periods=n, name="date")
    f = pd.DataFrame(10_000.0, index=idx, columns=symbols)
    meta = pd.DataFrame({"name": "n", "security_type": "STOCK", "product": "stock",
                         "is_common_share": True, "delisted": False, "source": "t"},
                        index=pd.Index(symbols, name="symbol"))
    return Panel(f, f, f, f, f * 0 + 1e6, meta)


def test_skipped_candidate_blocks_symbol_until_its_would_be_exit():
    p = flat_panel(["A", "B"])
    one_slot = replace(SWING, max_positions=1)
    rule = FixedRule(entries={"A": [2], "B": [2, 11, 14]}, exits={"A": [10], "B": [12, 18]},
                     conf={"A": 0.9, "B": 0.5})
    elig = all_true_mask(p)
    s, e = p.close.index[0].date(), p.close.index[-1].date()
    ours = run_portfolio(simulate(p, rule.signals(p), elig, s, e), p, CM, one_slot,
                         5_000_000, 1.0, False, **timing(rule, p)).fills
    naut = run_nautilus(rule, p, elig, s, e, CM, one_slot, 5_000_000, 1.0, False)
    d = p.close.index
    # B@2 skipped (slot held by A) -> its would-be trade 2..12 blocks the B@11 signal
    assert list(zip(naut["symbol"], naut["entry_date"], naut["exit_date"])) == [
        ("A", d[2], d[10]), ("B", d[14], d[18])]
    assert parity_report(ours, naut)["matched_share"] == 1.0


def test_parity_report_lists_mismatches():
    p = make_panel(300, ["A", "B", "069500"], seed=7, products={"069500": "etf"})
    rule = SWING_FAMILIES["breakout"](SWING_FAMILIES["breakout"].grid()[0])
    ours = run_portfolio(simulate(p, rule.signals(p), all_true_mask(p), START, END), p, CM,
                         SWING, 5_000_000, 1.0, False, **timing(rule, p)).fills
    same = parity_report(ours, ours)
    assert same["n_ours"] == same["n_naut"] == len(ours)
    assert same["matched_share"] == 1.0 and same["exit_reason_agreement"] == 1.0
    assert same["net_pnl_diff"] == 0.0 and same["mismatches"] == []

    naut = ours.copy()
    naut.loc[0, "exit_reason"] = "max_hold"
    naut = naut.drop(index=len(naut) - 1)
    rep = parity_report(ours, naut)
    assert rep["n_naut"] == len(ours) - 1
    assert rep["matched_share"] == pytest.approx((len(ours) - 1) / len(ours))
    fields = {(m["symbol"], m["field"]) for m in rep["mismatches"]}
    assert (ours.loc[0, "symbol"], "exit_reason") in fields
    assert (ours.iloc[-1]["symbol"], "trade") in fields
    assert rep["net_pnl_diff"] == pytest.approx(-ours.iloc[-1]["net_pnl"])


def test_kr_integer_prices_trade_on_the_won_grid():
    p0 = make_panel(300, ["A", "B", "069500"], seed=7, products={"069500": "etf"})
    won = lambda df: (df * 300).round()                       # realistic KRW: whole won
    p = Panel(won(p0.open), won(p0.high), won(p0.low), won(p0.close), p0.volume, p0.meta)
    rule = SWING_FAMILIES["breakout"](SWING_FAMILIES["breakout"].grid()[0])
    elig = all_true_mask(p)
    ours = run_portfolio(simulate(p, rule.signals(p), elig, START, END), p, CM, SWING,
                         5_000_000, 1.0, False, **timing(rule, p)).fills
    naut = run_nautilus(rule, p, elig, START, END, CM, SWING, 5_000_000, 1.0, False)
    rep = parity_report(ours, naut)
    assert rep["matched_share"] == 1.0 and rep["exit_reason_agreement"] == 1.0
    assert (naut["qty"].values == ours["qty"].values).all()
    assert (naut["exit_price"] % 1 == 0).all()                # fills on the 0-decimal grid
    # only stop fills differ: the stop is floored onto the grid (< 1 won lower)
    px = {(m["symbol"], m["entry_date"]) for m in rep["mismatches"] if m["field"] == "exit_price"}
    stops = ours[ours["exit_reason"] == "stop"]
    assert px and px <= set(zip(stops["symbol"], stops["entry_date"]))
    diff = ours["exit_price"] - naut["exit_price"]
    assert ((diff >= 0) & (diff < 1)).all()
    assert {m["field"] for m in rep["mismatches"]} <= {"exit_price", "net_pnl"}


def test_data_end_closes_at_last_valid_close():
    p = flat_panel(["A", "B"])
    for f in (p.open, p.high, p.low, p.close):
        f.iloc[8:, 0] = np.nan                                 # A: no prices from row 8 on
    p.close.iloc[7, 0] = 10_100.0
    p.high.iloc[7, 0] = 10_100.0
    rule = FixedRule(entries={"A": [3], "B": [3]}, exits={"A": [20], "B": [6]},
                     conf={"A": 0.9, "B": 0.5})
    elig = all_true_mask(p)
    s, e = p.close.index[0].date(), p.close.index[-1].date()
    ours = run_portfolio(simulate(p, rule.signals(p), elig, s, e), p, CM, SWING,
                         5_000_000, 1.0, False, **timing(rule, p)).fills
    naut = run_nautilus(rule, p, elig, s, e, CM, SWING, 5_000_000, 1.0, False)
    d = p.close.index
    for df in (ours, naut):
        a = df[df["symbol"] == "A"]
        assert list(zip(a["entry_date"], a["exit_date"], a["exit_reason"])) == [
            (d[3], d[7], "data_end")]
        assert a["exit_price"].iloc[0] == 10_100.0
    rep = parity_report(ours, naut)
    assert rep["matched_share"] == 1.0 and rep["mismatches"] == []
    # cash comes back on the last valid day in both engines
    assert naut.attrs["final_balance"] == pytest.approx(5_000_000 + naut["net_pnl"].sum(),
                                                       abs=len(naut))


@pytest.mark.parametrize("entry_at,exit_at", [("close", "close"), ("next_open", "next_open")])
def test_suspension_then_crash_is_held_and_gap_stopped(entry_at, exit_at):
    """A is suspended rows 8..17 and reopens 50% lower: both engines hold through the gap
    and stop out at the reopening open (not at the last pre-suspension close)."""
    p = flat_panel(["A", "B"])
    for f in (p.open, p.high, p.low, p.close):
        f.iloc[8:18, 0] = np.nan
        f.iloc[18:, 0] = 5_000.0
    rule = FixedRule(entries={"A": [3], "B": [3]}, exits={"A": [25], "B": [6]},
                     conf={"A": 0.9, "B": 0.5}, entry_at=entry_at, exit_at=exit_at)
    elig = all_true_mask(p)
    s, e = p.close.index[0].date(), p.close.index[-1].date()
    ours = run_portfolio(simulate(p, rule.signals(p), elig, s, e), p, CM, SWING,
                         5_000_000, 1.0, False, **timing(rule, p)).fills
    naut = run_nautilus(rule, p, elig, s, e, CM, SWING, 5_000_000, 1.0, False)
    d = p.close.index
    for df in (ours, naut):
        a = df[df["symbol"] == "A"].iloc[0]
        assert (a["exit_date"], a["exit_reason"], a["exit_price"]) == (d[18], "gap_stop", 5_000.0)
        assert a["net_pnl"] < 0
    rep = parity_report(ours, naut)
    assert rep["matched_share"] == 1.0 and rep["exit_reason_agreement"] == 1.0
    assert rep["mismatches"] == []


def test_max_hold_counts_suspended_rows_in_both_engines():
    p = flat_panel(["A", "B"])
    for f in (p.open, p.high, p.low, p.close):
        f.iloc[6:9, 0] = np.nan                                # A suspended rows 6..8
    rule = FixedRule(entries={"A": [3]}, exits={}, conf={"A": 0.9, "B": 0.5}, max_hold=4)
    elig = all_true_mask(p)
    s, e = p.close.index[0].date(), p.close.index[-1].date()
    ours = run_portfolio(simulate(p, rule.signals(p), elig, s, e), p, CM, SWING,
                         5_000_000, 1.0, False, **timing(rule, p)).fills
    naut = run_nautilus(rule, p, elig, s, e, CM, SWING, 5_000_000, 1.0, False)
    d = p.close.index
    for df in (ours, naut):          # close entry row 3; row 7 (= 4 rows) is NaN -> row 9
        assert list(zip(df["entry_date"], df["exit_date"], df["exit_reason"])) == [
            (d[3], d[9], "max_hold")]
    assert parity_report(ours, naut)["mismatches"] == []


def test_parity_breakout_with_binding_max_positions():
    """40 symbols: max_positions binds, so same-day exit/entry ordering matters."""
    p = make_panel(1000, [f"S{i:02d}" for i in range(40)], seed=3)
    cm = CostModel(KR, lambda s: "stock")
    rule = SWING_FAMILIES["breakout"](SWING_FAMILIES["breakout"].grid()[0])
    elig = all_true_mask(p)
    s, e = date(2016, 6, 1), p.close.index[-1].date()
    res = run_portfolio(simulate(p, rule.signals(p), elig, s, e), p, cm, SWING, 5_000_000,
                        1.0, False, **timing(rule, p))
    assert (res.skipped["reason"] == "max_positions").any()
    ours = res.fills
    naut = run_nautilus(rule, p, elig, s, e, cm, SWING, 5_000_000, 1.0, False)
    assert len(naut) == len(ours)
    assert np.allclose(naut["net_pnl"].sum(), ours["net_pnl"].sum(), rtol=0.01)
    assert (naut["exit_reason"].values == ours["exit_reason"].values).mean() >= 0.95
    assert parity_report(ours, naut)["matched_share"] == 1.0


def test_close_entry_followed_by_missing_bar():
    p = flat_panel(["A", "B"])
    for f in (p.open, p.high, p.low, p.close):
        f.iloc[4:, 0] = np.nan                                 # A: no prices from row 4 on
    one_slot = replace(SWING, max_positions=1)
    rule = FixedRule(entries={"A": [3], "B": [10]}, exits={"A": [20], "B": [15]},
                     conf={"A": 0.9, "B": 0.5})
    elig = all_true_mask(p)
    s, e = p.close.index[0].date(), p.close.index[-1].date()
    ours = run_portfolio(simulate(p, rule.signals(p), elig, s, e), p, CM, one_slot,
                         5_000_000, 1.0, False, **timing(rule, p)).fills
    naut = run_nautilus(rule, p, elig, s, e, CM, one_slot, 5_000_000, 1.0, False)
    d = p.close.index
    expected = [("A", d[3], d[3], "data_end"), ("B", d[10], d[15], "signal")]
    for df in (ours, naut):
        assert list(zip(df["symbol"], df["entry_date"], df["exit_date"],
                        df["exit_reason"])) == expected
    rep = parity_report(ours, naut)
    assert rep["matched_share"] == 1.0 and rep["mismatches"] == []
