from dataclasses import replace

import pandas as pd
import pytest

from autotrade.config import load_market_raw
from autotrade.costs import CostModel, cost_in_r, tick_size
from autotrade.types import TRADE_COLUMNS

KR_T = replace(load_market_raw("kr"), commission_rate=0.00015)
US_T = replace(load_market_raw("us"), commission_rate=0.001)


def one_trade(entry, exit, stop_pct, symbol="X"):
    return pd.DataFrame([{
        "symbol": symbol, "entry_date": pd.Timestamp("2024-01-02"), "entry_price": entry,
        "exit_date": pd.Timestamp("2024-01-10"), "exit_price": exit,
        "exit_reason": "signal", "stop_pct": stop_pct, "confidence": 1.0}])[TRADE_COLUMNS]


def test_kr_tick_sizes():
    assert tick_size(1_990, KR_T, "stock") == 1
    assert tick_size(19_990, KR_T, "stock") == 10
    assert tick_size(20_000, KR_T, "stock") == 50
    assert tick_size(45_000, KR_T, "etf") == 5


def test_kr_stock_round_trip():
    cm = CostModel(KR_T, lambda s: "stock")
    assert cm.round_trip_cost_pct("A", 19_990) == pytest.approx(0.0003 + 0.0020 + 2 * 10 / 19_990, rel=1e-3)


def test_kr_etf_has_no_tax():
    cm = CostModel(KR_T, lambda s: "etf")
    assert cm.round_trip_cost_pct("069500", 45_000) == pytest.approx(0.0003 + 2 * 5 / 45_000, rel=1e-3)


def test_us_slippage_max_rule_and_sec_fee():
    cm = CostModel(US_T, lambda s: "stock")
    assert cm.buy_fill("X", 10.0) == pytest.approx(10.02)
    assert cm.buy_fill("Y", 100.0) == pytest.approx(100.05)
    t = one_trade(entry=100.0, exit=110.0, stop_pct=0.05)
    out = cm.net_returns(t)
    expected = (110 - 0.055) * (1 - 0.001 - 0.0000206) / ((100 + 0.05) * (1 + 0.001)) - 1
    assert out["net_return"].iloc[0] == pytest.approx(expected, rel=1e-6)
    assert out["net_r"].iloc[0] == pytest.approx(expected / 0.05, rel=1e-6)


def test_cost_in_r():
    assert cost_in_r(0.003, 0.03) == pytest.approx(0.1)


def test_net_returns_columns_gross_and_no_mutation():
    cm = CostModel(US_T, lambda s: "stock")
    t = one_trade(100.0, 110.0, 0.05)
    before = t.copy()
    out = cm.net_returns(t)
    assert list(out.columns) == TRADE_COLUMNS + ["gross_return", "net_return", "net_r"]
    assert out["gross_return"].iloc[0] == pytest.approx(0.10)
    pd.testing.assert_frame_equal(t, before)


def test_kr_sell_fill_and_etf_tax_in_net_returns():
    cm = CostModel(KR_T, lambda s: "etf")
    assert cm.buy_fill("E", 45_000) == 45_005
    assert cm.sell_fill("E", 45_000) == 44_995
    out = cm.net_returns(one_trade(45_000, 45_000, 0.02, "E"))
    expected = 44_995 * (1 - 0.00015) / (45_005 * (1 + 0.00015)) - 1
    assert out["net_return"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_requires_synced_commission():
    with pytest.raises(ValueError):
        CostModel(load_market_raw("kr"), lambda s: "stock")


def test_empty_trades():
    cm = CostModel(US_T, lambda s: "stock")
    out = cm.net_returns(one_trade(1, 1, 0.1).iloc[0:0])
    assert list(out.columns) == TRADE_COLUMNS + ["gross_return", "net_return", "net_r"]
    assert out.empty
