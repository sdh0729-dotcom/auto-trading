from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from autotrade.config import PocketConfig, load_market_raw
from autotrade.costs import CostModel
from autotrade.sizing import gap_allowance, run_portfolio
from autotrade.types import TRADE_COLUMNS, Panel

DAYS = pd.bdate_range("2024-01-01", periods=30, name="date")
US0 = replace(load_market_raw("us"), commission_rate=0.0, sec_fee_rate=0.0,
              slippage_min_pct=0.0, slippage_min_abs=0.0)
KR = replace(load_market_raw("kr"), commission_rate=0.00015)
POCKET = PocketConfig(risk_per_trade=0.01, max_position_weight=0.25, daily_loss_stop=0.03,
                      max_positions=4, drawdown_stop=0.15)
FREE = CostModel(US0, lambda s: "stock")


def panel(closes: dict, opens: dict | None = None) -> Panel:
    """closes/opens: symbol -> scalar or per-day list (len 30)."""
    def frame(d):
        return pd.DataFrame({s: (v if np.ndim(v) else [v] * len(DAYS)) for s, v in d.items()},
                            index=DAYS, dtype=float)
    c = frame(closes)
    o = frame(opens) if opens else c.copy()
    meta = pd.DataFrame({"name": "n", "security_type": "STOCK", "product": "stock",
                         "is_common_share": True, "delisted": False, "source": "t"},
                        index=pd.Index(list(closes), name="symbol"))
    return Panel(o, c, c, c, c * 0 + 1e6, meta)


def cand(symbol, entry, exit_, entry_price=100.0, exit_price=100.0, stop=0.05, conf=1.0,
         reason="signal"):
    return {"symbol": symbol, "entry_date": DAYS[entry], "entry_price": entry_price,
            "exit_date": DAYS[exit_], "exit_price": exit_price, "exit_reason": reason,
            "stop_pct": stop, "confidence": conf}


def run(rows, p, cm=FREE, pocket=POCKET, capital=1_000_000.0, fx=1.0, fractional=True,
        entry_at="next_open", exit_at="next_open"):
    df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    return run_portfolio(df, p, cm, pocket, capital, fx, fractional, entry_at=entry_at,
                         exit_at=exit_at)


def reasons(res):
    return dict(zip(res.skipped["symbol"], res.skipped["reason"]))


def test_gap_allowance_needs_20_obs_and_uses_p95():
    closes = [100.0] * 30
    opens = [98.0] * 30
    p = panel({"A": closes}, {"A": opens})
    assert gap_allowance(p, "A", DAYS[10]) == 0.0           # 9 observations (rows 1..9)
    assert gap_allowance(p, "A", DAYS[25]) == pytest.approx(100 / 98 - 1)


def test_gap_allowance_is_strictly_before_asof_and_ignores_favourable_gaps():
    opens = [100.0] * 30
    opens[22:] = [50.0] * 8                                  # gaps from row 22 on
    opens[5] = 120.0                                         # favourable gap: counts as 0
    p = panel({"A": [100.0] * 30}, {"A": opens})
    assert gap_allowance(p, "A", DAYS[22]) == 0.0            # row 22 itself must be ignored
    assert gap_allowance(p, "A", DAYS[29]) == pytest.approx(100 / 50 - 1)  # visible once earlier
    assert gap_allowance(p, "ZZZ", DAYS[25]) == 0.0


def test_gap_allowance_lookback_window():
    opens = [100.0] * 30
    opens[2:6] = [80.0] * 4                                  # old gaps, outside lookback=22
    p = panel({"A": [100.0] * 30}, {"A": opens})
    assert gap_allowance(p, "A", DAYS[28], lookback=100) == pytest.approx(100 / 80 - 1)
    assert gap_allowance(p, "A", DAYS[28], lookback=22) == 0.0


def test_risk_based_qty_with_confidence_and_gap():
    # equity 1,000,000; risk 1% * conf 0.5 = 5,000; gap = 100/98 - 1 (2.0408%);
    # qty = 5,000 / (100 * (0.05 + 0.020408163)) = 710.14 -> floor 710 (notional 71k < 250k cap)
    p = panel({"A": 100.0}, {"A": 98.0})
    rows = [cand("A", 25, 27, conf=0.5)]
    res = run(rows, p, fractional=False)
    assert res.fills["qty"].iloc[0] == 710
    res = run(rows, p, fractional=True)
    assert res.fills["qty"].iloc[0] == pytest.approx(5000 / (100 * (0.05 + 100 / 98 - 1)))
    assert res.fills["qty"].iloc[0] == pytest.approx(710.1449, rel=1e-6)


def test_weight_cap_25pct():
    # uncapped qty = 10,000 / (100 * 0.01) = 10,000 shares (1,000,000 notional); cap 250,000 -> 2,500
    res = run([cand("A", 1, 3, stop=0.01)], panel({"A": 100.0}))
    assert res.fills["qty"].iloc[0] == pytest.approx(2500)


def test_integer_shares_real_kr_skips_too_small():
    # capital 50,000, price 30,000: cap 12,500 < 1 share -> too_small
    cm = CostModel(KR, lambda s: "stock")
    res = run([cand("A", 1, 3, entry_price=30_000.0, exit_price=30_000.0)],
              panel({"A": 30_000.0}), cm=cm, capital=50_000.0, fractional=False)
    assert res.fills.empty
    assert reasons(res) == {"A": "too_small"}


def test_fractional_us_allows_small():
    # capital 50,000 KRW at 1/1400 -> 35.714 USD; risk qty = 0.35714 / (100*0.05) = 0.0714 < cap 0.0893
    res = run([cand("A", 1, 3)], panel({"A": 100.0}), capital=50_000.0, fx=1 / 1400,
              fractional=True)
    assert res.fills["qty"].iloc[0] == pytest.approx(0.0714286, rel=1e-5)
    assert res.skipped.empty


def test_max_positions_4_and_confidence_priority():
    syms = list("ABCDE")
    p = panel({s: 100.0 for s in syms})
    rows = [cand(s, 1, 8, conf=c) for s, c in zip(syms, (0.5, 0.9, 0.7, 0.6, 0.8))]
    res = run(rows, p)
    assert set(res.fills["symbol"]) == {"B", "E", "C", "D"}
    assert reasons(res) == {"A": "max_positions"}


def test_exits_free_slots_before_same_day_entries():
    pocket = replace(POCKET, max_positions=1)
    p = panel({"A": 100.0, "B": 100.0})
    res = run([cand("A", 1, 4), cand("B", 4, 6)], p, pocket=pocket)
    assert set(res.fills["symbol"]) == {"A", "B"}
    assert res.skipped.empty


@pytest.mark.parametrize("reason,exit_at,freed", [
    ("gap_stop", "next_open", True),     # fills at D's open, before D's open entries
    ("signal", "next_open", True),       # pending next_open exit fills at D's open
    ("stop", "next_open", False),        # intraday on D: after D's open entries
    ("signal", "close", False),          # at D's close
    ("max_hold", "next_open", False),    # at D's close
])
def test_open_entry_sees_only_exits_filled_by_the_open(reason, exit_at, freed):
    pocket = replace(POCKET, max_positions=1)
    p = panel({"A": 100.0, "B": 100.0})
    res = run([cand("A", 1, 4, reason=reason), cand("B", 4, 6)], p, pocket=pocket,
              exit_at=exit_at)
    if freed:
        assert set(res.fills["symbol"]) == {"A", "B"} and res.skipped.empty
    else:
        assert set(res.fills["symbol"]) == {"A"}
        assert reasons(res) == {"B": "max_positions"}


def test_open_entry_cash_excludes_same_day_intraday_and_close_proceeds():
    pocket = replace(POCKET, max_position_weight=0.6, risk_per_trade=1.0)
    p = panel({"A": 100.0, "B": 100.0})
    # A holds 60% of equity; it is stopped intraday on day 4 -> B's open entry lacks cash
    res = run([cand("A", 1, 4, reason="stop"), cand("B", 4, 6)], p, pocket=pocket)
    assert reasons(res) == {"B": "cash"}
    res = run([cand("A", 1, 4, reason="gap_stop"), cand("B", 4, 6)], p, pocket=pocket)
    assert res.skipped.empty


def test_close_entry_sees_all_exits_of_the_day():
    pocket = replace(POCKET, max_positions=1)
    p = panel({"A": 100.0, "B": 100.0})
    for reason in ("stop", "max_hold", "signal"):
        res = run([cand("A", 1, 4, reason=reason), cand("B", 4, 6)], p, pocket=pocket,
                  entry_at="close", exit_at="close")
        assert set(res.fills["symbol"]) == {"A", "B"} and res.skipped.empty


def test_open_entry_stopped_out_same_day_is_closed():
    p = panel({"A": 100.0})
    res = run([cand("A", 4, 4, exit_price=95.0, reason="stop")], p)
    assert len(res.fills) == 1
    assert res.equity.iloc[-1] == pytest.approx(1_000_000 + res.fills["net_pnl"].iloc[0])


def test_close_entry_with_same_day_period_end_is_closed():
    # simulate emits exit_date == entry_date when the bar after a close entry is missing
    pocket = replace(POCKET, max_positions=1)
    p = panel({"A": 100.0, "B": 100.0})
    res = run([cand("A", 3, 3, reason="period_end"), cand("B", 5, 8)], p, pocket=pocket,
              entry_at="close", exit_at="close")
    assert set(res.fills["symbol"]) == {"A", "B"} and res.skipped.empty
    assert res.equity.iloc[-1] == pytest.approx(1_000_000 + res.fills["net_pnl"].sum())


def test_position_never_closed_raises():
    p = panel({"A": 100.0, "B": 100.0})
    with pytest.raises(ValueError, match="never closed"):
        run([cand("B", 0, 10), cand("A", 5, 3)], p)           # exit before entry


def test_cash_shortage_skips():
    pocket = replace(POCKET, max_position_weight=0.4)       # 3 x 40% > 100%
    syms = list("ABC")
    p = panel({s: 100.0 for s in syms})
    rows = [cand(s, 1, 8, stop=0.01, conf=c) for s, c in zip(syms, (0.9, 0.8, 0.7))]
    res = run(rows, p, pocket=pocket)
    assert set(res.fills["symbol"]) == {"A", "B"}
    assert reasons(res) == {"C": "cash"}


def test_drawdown_stop_blocks_new_entries():
    # A: 25% weight, close falls to 30 on day 1 -> equity -17.5% (<= -15%) -> latch
    a = [100.0] + [30.0] * 29
    p = panel({"A": a, "B": 100.0, "C": 100.0})
    rows = [cand("A", 0, 15, stop=0.01, exit_price=30.0),
            cand("B", 3, 6), cand("C", 20, 22)]
    res = run(rows, p)
    assert set(res.fills["symbol"]) == {"A"}
    assert reasons(res) == {"B": "drawdown_stop", "C": "drawdown_stop"}


def test_daily_loss_stop_next_day():
    # A: 25% weight; close 84 on day 1 -> equity -4% that day -> day 2 entries blocked, day 3 ok
    a = [100.0] + [84.0] * 29
    p = panel({"A": a, "B": 100.0, "C": 100.0})
    rows = [cand("A", 0, 15, stop=0.01, exit_price=84.0), cand("B", 2, 6), cand("C", 3, 6)]
    res = run(rows, p)
    assert set(res.fills["symbol"]) == {"A", "C"}
    assert reasons(res) == {"B": "daily_loss_stop"}


def test_equity_marked_to_close_daily():
    # 25% weight -> 2,500 shares at 100; closes 100,110,120 then exit at 130 on day 3
    a = [100.0, 110.0, 120.0, 130.0] + [130.0] * 26
    res = run([cand("A", 0, 3, stop=0.01, exit_price=130.0)], panel({"A": a}))
    eq = res.equity
    assert list(eq.index) == list(DAYS[:4])
    assert eq.iloc[0] == pytest.approx(1_000_000)
    assert eq.iloc[1] == pytest.approx(1_000_000 + 2500 * 10)
    assert eq.iloc[2] == pytest.approx(1_000_000 + 2500 * 20)
    assert eq.iloc[3] == pytest.approx(1_000_000 + 2500 * 30)


def test_nan_close_carries_last_valid_close():
    a = [100.0, 110.0, np.nan, 120.0] + [120.0] * 26
    res = run([cand("A", 0, 3, stop=0.01, exit_price=120.0)], panel({"A": a}))
    assert res.equity.iloc[2] == pytest.approx(res.equity.iloc[1])


def test_sizing_uses_previous_close_equity_compounding():
    # A (25%) gains 40% by day 1 -> equity 1,100,000; B (day 2) is sized off that close
    a = [100.0, 140.0] + [140.0] * 28
    res = run([cand("A", 0, 1, stop=0.01, exit_price=140.0),
               cand("B", 2, 5, stop=0.01)], panel({"A": a, "B": 100.0}))
    eq1 = res.equity.iloc[1]
    assert eq1 == pytest.approx(1_000_000 + 2500 * 40)
    b = res.fills[res.fills["symbol"] == "B"].iloc[0]
    assert b["qty"] == pytest.approx(eq1 * 0.25 / 100)


def test_net_pnl_and_net_r_include_costs():
    cm = CostModel(KR, lambda s: "stock")
    buy = cm.buy_fill("A", 10_000.0)                         # +10 (1 tick)
    sell = cm.sell_fill("A", 11_000.0)                       # -10 (tick 10 below 20,000)
    res = run([cand("A", 1, 5, entry_price=10_000.0, exit_price=11_000.0, stop=0.05)],
              panel({"A": 10_000.0}), cm=cm, capital=10_000_000.0, fractional=False)
    f = res.fills.iloc[0]
    qty = f["qty"]
    # equity 10,000,000: risk qty = 100,000 / (10,000 * 0.05) = 200 ; cap = 250 -> 200
    assert qty == 200
    cost = qty * buy * 1.00015
    proceeds = qty * sell * (1 - 0.00015 - 0.0020)
    assert f["net_pnl"] == pytest.approx(proceeds - cost)
    assert f["net_r"] == pytest.approx((proceeds - cost) / (qty * 10_000.0 * 0.05))


def test_fill_cash_matches_net_returns():
    cm = CostModel(KR, lambda s: "stock")
    assert cm.fill_cash("A", "buy", 10, 10_000.0) == pytest.approx(10 * cm.buy_fill("A", 10_000.0) * 1.00015)
    assert cm.fill_cash("A", "sell", 10, 11_000.0) == pytest.approx(
        10 * cm.sell_fill("A", 11_000.0) * (1 - 0.00015 - 0.0020))
    with pytest.raises(ValueError):
        cm.fill_cash("A", "hold", 1, 1.0)


def test_empty_candidates():
    res = run([], panel({"A": 100.0}))
    assert res.fills.empty and res.skipped.empty and res.equity.empty
    assert list(res.fills.columns) == TRADE_COLUMNS + ["qty", "net_pnl", "net_r"]
    assert list(res.skipped.columns) == TRADE_COLUMNS + ["reason"]
