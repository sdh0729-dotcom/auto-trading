from dataclasses import replace

import pandas as pd

from autotrade.config import PocketConfig, load_market_raw
from autotrade.costs import CostModel
from autotrade.sizing.intraday import run_intraday_portfolio

US0 = replace(load_market_raw("us"), commission_rate=0.0, sec_fee_rate=0.0,
              slippage_min_pct=0.0, slippage_min_abs=0.0)
FREE = CostModel(US0, lambda s: "stock")
POCKET = PocketConfig(risk_per_trade=0.01, max_position_weight=0.25, daily_loss_stop=0.03)


def tr(sym, day, t_in, t_out, entry=100.0, exit_=100.0, stop=0.01, conf=1.0):
    return {"symbol": sym, "entry_date": pd.Timestamp(f"{day} {t_in}"), "entry_price": entry,
            "exit_date": pd.Timestamp(f"{day} {t_out}"), "exit_price": exit_,
            "exit_reason": "session_end", "stop_pct": stop, "confidence": conf}


def run(rows, capital=1_000_000.0):
    return run_intraday_portfolio(pd.DataFrame(rows), FREE, POCKET, capital, 1.0, False)


def test_daily_loss_stop_blocks_later_entries_same_day_only():
    # qty = min(1e6*.01/(100*.01)=10000, 1e6*.25/100=2500) = 2500; a -2% loss = -5000 each
    rows = [tr("A", "2024-01-02", "10:00", "10:30", exit_=96.0),     # -10000 = -1%
            tr("B", "2024-01-02", "10:31", "11:00", exit_=88.0),     # equity-based qty; big loss
            tr("C", "2024-01-02", "11:30", "12:00"),                 # blocked after the loss
            tr("D", "2024-01-03", "10:00", "10:30")]                 # next day: allowed
    res = run(rows)
    assert set(res.skipped["symbol"]) == {"C"}
    assert res.skipped.iloc[0]["reason"] == "daily_loss_stop"
    assert set(res.fills["symbol"]) == {"A", "B", "D"}


def test_sizing_uses_previous_day_equity():
    rows = [tr("A", "2024-01-02", "10:00", "10:30", exit_=104.0),    # +4 x 2500 = +10000
            tr("B", "2024-01-03", "10:00", "10:30")]
    res = run(rows)
    assert res.fills.set_index("symbol").loc["A", "qty"] == 2500
    assert res.equity.iloc[0] == 1_010_000
    assert res.fills.set_index("symbol").loc["B", "qty"] == 2525   # 1.01e6 * .25 / 100


def test_exit_before_entry_at_same_ts():
    rows = [tr("A", "2024-01-02", "10:00", "10:30", entry=100.0, exit_=100.0),
            tr("B", "2024-01-02", "10:30", "11:00")]
    # 4 positions of 25% would not fit twice; A's cash must be back before B enters
    big = [dict(r, entry_price=100.0) for r in rows]
    res = run_intraday_portfolio(pd.DataFrame(big), FREE,
                                 replace(POCKET, max_position_weight=1.0, risk_per_trade=1.0),
                                 1_000_000.0, 1.0, False)
    assert res.skipped.empty and len(res.fills) == 2
    res2 = run_intraday_portfolio(pd.DataFrame(big), FREE,
                                  replace(POCKET, max_position_weight=1.0, risk_per_trade=1.0,
                                          max_positions=1), 1_000_000.0, 1.0, False)
    assert res2.skipped.empty        # max_positions sees A already closed
