from datetime import date

import numpy as np
import pandas as pd
import pytest

from autotrade.rules.swing.breakout import Breakout
from autotrade.screen import simulate
from autotrade.types import TRADE_COLUMNS, Panel, Signals
from tests.fixtures import make_panel

DAYS = pd.bdate_range("2024-01-01", periods=8, name="date")  # d0..d7 (Mon 1 Jan .. Wed 10 Jan)
D = list(DAYS)
START, END = date(2024, 1, 1), date(2024, 1, 10)


def bars(rows: list[tuple[float, float, float, float]], sym: str = "A") -> Panel:
    """rows: (open, high, low, close) per day d0.. ; remaining days repeat a flat 100 bar."""
    rows = list(rows) + [(100.0, 100.0, 100.0, 100.0)] * (len(DAYS) - len(rows))
    o, h, l, c = (pd.DataFrame({sym: [r[k] for r in rows]}, index=DAYS) for k in range(4))
    vol = pd.DataFrame({sym: [1e6] * len(DAYS)}, index=DAYS)
    meta = pd.DataFrame({"name": ["n"], "security_type": ["STOCK"], "product": ["stock"],
                         "is_common_share": [True], "delisted": [False], "source": ["t"]},
                        index=pd.Index([sym], name="symbol"))
    return Panel(o, h, l, c, vol, meta)


def sigs(p: Panel, entries=(), exits=(), stop=0.05, conf=0.7, entry_at="next_open",
         exit_at="next_open", max_hold=20, stop_by_day=None, conf_by_day=None) -> Signals:
    """entries/exits: day indices where the single symbol's signal is True."""
    idx, cols = p.close.index, p.close.columns
    e = pd.DataFrame(False, index=idx, columns=cols)
    x = pd.DataFrame(False, index=idx, columns=cols)
    e.iloc[list(entries), 0] = True
    x.iloc[list(exits), 0] = True
    sp = pd.DataFrame(stop, index=idx, columns=cols)
    cf = pd.DataFrame(conf, index=idx, columns=cols)
    for k, v in (stop_by_day or {}).items():
        sp.iloc[k, 0] = v
    for k, v in (conf_by_day or {}).items():
        cf.iloc[k, 0] = v
    return Signals(e, x, sp, cf, entry_at, exit_at, max_hold)


def all_true(p: Panel) -> pd.DataFrame:
    return pd.DataFrame(True, index=p.close.index, columns=p.close.columns)


def only(t: pd.DataFrame) -> dict:
    assert len(t) == 1, t
    return t.iloc[0].to_dict()


FLAT = (100.0, 101.0, 99.0, 100.0)


def test_next_open_entry_and_signal_exit():
    # signal d1 -> enter open d2 @100; stop 5% (signal-row value) -> 95; exit signal d3 -> open d4 @108
    p = bars([FLAT, FLAT, (100, 103, 98, 102), (104, 107, 103, 106), (108, 109, 105, 107)])
    s = sigs(p, entries=[1], exits=[3], stop_by_day={1: 0.05, 2: 0.30}, conf_by_day={1: 0.7, 2: 0.1})
    t = simulate(p, s, all_true(p), START, END)
    assert list(t.columns) == TRADE_COLUMNS
    assert only(t) == {"symbol": "A", "entry_date": D[2], "entry_price": 100.0,
                       "exit_date": D[4], "exit_price": 108.0, "exit_reason": "signal",
                       "stop_pct": 0.05, "confidence": 0.7}


def test_gap_stop_fills_at_open():
    # 진입 100, stop 5% → 손절가 95, 다음 날 시가 90 → exit 90, reason gap_stop
    p = bars([FLAT, FLAT, FLAT, (90, 92, 88, 91)])
    t = only(simulate(p, sigs(p, entries=[1]), all_true(p), START, END))
    assert (t["entry_price"], t["exit_date"], t["exit_price"], t["exit_reason"]) == \
        (100.0, D[3], 90.0, "gap_stop")


def test_intraday_stop_fills_at_stop_price():
    # 시가 97, 저가 94 → exit 95, stop
    p = bars([FLAT, FLAT, FLAT, (97, 98, 94, 96)])
    t = only(simulate(p, sigs(p, entries=[1]), all_true(p), START, END))
    assert (t["exit_date"], t["exit_reason"]) == (D[3], "stop")
    assert t["exit_price"] == pytest.approx(95.0)


def test_next_open_entry_bar_checks_intraday_stop():
    # entry bar itself: open 100 (entry), low 94 -> stopped the same day at 95
    p = bars([FLAT, FLAT, (100, 101, 94, 99)])
    t = only(simulate(p, sigs(p, entries=[1]), all_true(p), START, END))
    assert (t["entry_date"], t["exit_date"], t["exit_reason"]) == (D[2], D[2], "stop")
    assert t["exit_price"] == pytest.approx(95.0)


def test_next_open_entry_bar_signal_exit_at_close():
    p = bars([FLAT, FLAT, (100, 104, 99, 103)])
    s = sigs(p, entries=[1], exits=[2], exit_at="close")
    t = only(simulate(p, s, all_true(p), START, END))
    assert (t["entry_date"], t["exit_date"], t["exit_price"], t["exit_reason"]) == \
        (D[2], D[2], 103.0, "signal")


def test_signal_exit_at_close_on_signal_row():
    p = bars([FLAT, FLAT, FLAT, (101, 106, 100, 105)])
    s = sigs(p, entries=[1], exits=[3], exit_at="close")
    t = only(simulate(p, s, all_true(p), START, END))
    assert (t["exit_date"], t["exit_price"], t["exit_reason"]) == (D[3], 105.0, "signal")


def test_close_entry_next_open_exit_overnight():
    # 종가 100 진입, 다음 날 시가 101 청산 (max_hold=1 must not pre-empt the open exit)
    p = bars([FLAT, FLAT, (101, 103, 100, 102)])
    s = sigs(p, entries=[1], exits=range(8), entry_at="close", exit_at="next_open", max_hold=1)
    t = only(simulate(p, s, all_true(p), START, END))
    assert (t["entry_date"], t["entry_price"], t["exit_date"], t["exit_price"], t["exit_reason"]) == \
        (D[1], 100.0, D[2], 101.0, "signal")


def test_close_entry_bar_is_not_stop_checked():
    # close entry at d1 @100 (stop 95); d1 low 90 happened before the entry -> ignored
    p = bars([FLAT, (100, 101, 90, 100), (101, 103, 100, 102)])
    s = sigs(p, entries=[1], exits=[1], entry_at="close", exit_at="next_open", max_hold=1)
    t = only(simulate(p, s, all_true(p), START, END))
    assert (t["exit_date"], t["exit_price"], t["exit_reason"]) == (D[2], 101.0, "signal")


def test_close_entry_without_exit_signal_hits_max_hold():
    p = bars([FLAT, FLAT, (101, 103, 100, 102)])
    s = sigs(p, entries=[1], entry_at="close", exit_at="next_open", max_hold=1)
    t = only(simulate(p, s, all_true(p), START, END))
    assert (t["exit_date"], t["exit_price"], t["exit_reason"]) == (D[2], 102.0, "max_hold")


def test_pending_open_exit_fills_at_open_in_physical_order():
    # exit signal d3 -> would sell at open d4; but d4 gaps below the stop -> gap_stop
    p = bars([FLAT, FLAT, FLAT, FLAT, (93, 96, 92, 95)])
    t = only(simulate(p, sigs(p, entries=[1], exits=[3]), all_true(p), START, END))
    assert (t["exit_date"], t["exit_price"], t["exit_reason"]) == (D[4], 93.0, "gap_stop")
    # physical order: open above the stop -> the pending exit sells at the open (99) before the
    # bar's low (94) could touch the stop; no intrabar stop on that bar
    p = bars([FLAT, FLAT, FLAT, FLAT, (99, 100, 94, 96)])
    t = only(simulate(p, sigs(p, entries=[1], exits=[3]), all_true(p), START, END))
    assert (t["exit_date"], t["exit_price"], t["exit_reason"]) == (D[4], 99.0, "signal")
    # overnight variant: close entry 100, next open 97 (> stop 95), low 90 -> signal at 97
    p = bars([FLAT, FLAT, (97, 98, 90, 92)])
    s = sigs(p, entries=[1], exits=range(8), entry_at="close", exit_at="next_open", max_hold=1)
    t = only(simulate(p, s, all_true(p), START, END))
    assert (t["exit_date"], t["exit_price"], t["exit_reason"]) == (D[2], 97.0, "signal")


def test_max_hold_exit():
    # signal d0 -> entry d1 (day 0); max_hold 3 -> exit close of d4 (day 3)
    p = bars([FLAT, FLAT, FLAT, FLAT, (100, 103, 99, 102)])
    t = only(simulate(p, sigs(p, entries=[0], max_hold=3), all_true(p), START, END))
    assert (t["entry_date"], t["exit_date"], t["exit_price"], t["exit_reason"]) == \
        (D[1], D[4], 102.0, "max_hold")


def test_ineligible_day_no_entry():
    p = bars([FLAT] * 8)
    elig = all_true(p)
    elig.iloc[1, 0] = False
    assert simulate(p, sigs(p, entries=[1], max_hold=2), elig, START, END).empty
    # eligibility is judged on the signal row only, not on the fill row
    elig = all_true(p)
    elig.iloc[2, 0] = False
    t = only(simulate(p, sigs(p, entries=[1], max_hold=2), elig, START, END))
    assert t["entry_date"] == D[2]


def test_one_position_per_symbol():
    p = bars([FLAT] * 8)
    t = only(simulate(p, sigs(p, entries=[0, 1, 2, 3], max_hold=10), all_true(p), START, END))
    assert t["entry_date"] == D[1]


def test_no_same_row_exit_and_reenter():
    # overnight with entries every day: enter d1 close, exit d2 open, cannot re-enter at d2 close,
    # re-enter d3 close, exit d4 open, ...
    p = bars([FLAT] * 8)
    s = sigs(p, entries=range(1, 8), exits=range(8), entry_at="close", exit_at="next_open",
             max_hold=1)
    t = simulate(p, s, all_true(p), START, END)
    assert list(t["entry_date"]) == [D[1], D[3], D[5]]
    assert list(t["exit_date"]) == [D[2], D[4], D[6]]
    assert list(t["exit_reason"]) == ["signal", "signal", "signal"]


def test_close_entry_on_last_row_is_skipped():
    # a close entry on the window's last row could only be closed at the same close
    # (zero-length trade that is pure cost) -> not taken
    p = bars([FLAT] * 8)
    s = sigs(p, entries=[4], entry_at="close", max_hold=5)
    assert simulate(p, s, all_true(p), START, date(2024, 1, 5)).empty
    t = only(simulate(p, s, all_true(p), START, END))
    assert (t["entry_date"], t["exit_date"], t["exit_reason"]) == (D[4], D[7], "period_end")


def test_reentry_allowed_on_row_after_exit():
    # exit signal d2 -> sell open d3; entry signal d2 would fill at open d3 (same row) -> refused;
    # entry signal d3 -> fill open d4 -> allowed
    p = bars([FLAT] * 8)
    s = sigs(p, entries=[0, 2, 3], exits=[2], max_hold=20)
    t = simulate(p, s, all_true(p), START, END)
    assert list(t["entry_date"]) == [D[1], D[4]]
    assert list(t["exit_date"]) == [D[3], D[7]]


def test_period_end_exit_at_end_close():
    p = bars([FLAT, FLAT, FLAT, FLAT, (100, 101, 99, 104), FLAT, FLAT, FLAT])
    t = only(simulate(p, sigs(p, entries=[1]), all_true(p), START, date(2024, 1, 5)))
    assert (t["exit_date"], t["exit_price"], t["exit_reason"]) == (D[4], 104.0, "period_end")


def test_entry_signal_on_last_row_needs_next_bar_in_window():
    p = bars([FLAT] * 8)
    assert simulate(p, sigs(p, entries=[4]), all_true(p), START, date(2024, 1, 5)).empty


def test_signal_before_start_is_ignored():
    p = bars([FLAT] * 8)
    assert simulate(p, sigs(p, entries=[0]), all_true(p), date(2024, 1, 2), END).empty


def test_nan_price_closes_at_last_valid_close():
    nan = (np.nan,) * 4
    p = bars([FLAT, FLAT, FLAT, (100, 102, 99, 101.5), nan, nan, nan, nan])
    t = only(simulate(p, sigs(p, entries=[1]), all_true(p), START, END))
    assert (t["exit_date"], t["exit_price"], t["exit_reason"]) == (D[3], 101.5, "period_end")


def test_nan_entry_price_means_no_trade():
    nan = (np.nan,) * 4
    p = bars([FLAT, FLAT, nan, nan, nan, nan, nan, nan])
    assert simulate(p, sigs(p, entries=[1]), all_true(p), START, END).empty


def test_invalid_stop_means_no_entry():
    p = bars([FLAT] * 8)
    for bad in (np.nan, np.inf, 0.0, -0.1):
        assert simulate(p, sigs(p, entries=[1], stop=bad), all_true(p), START, END).empty


def test_multiple_symbols_and_empty_frame_dtypes():
    p = make_panel(300, ["A", "B", "C"], seed=5)
    s = Breakout({"n": 20, "m": 10, "k": 2.0}).signals(p)
    t = simulate(p, s, all_true(p), date(2016, 1, 1), date(2017, 12, 31))
    assert set(t["symbol"]) <= {"A", "B", "C"} and len(t) > 0
    assert set(t["exit_reason"]) <= {"signal", "stop", "gap_stop", "max_hold", "period_end"}
    assert (t["exit_date"] >= t["entry_date"]).all()
    assert t["entry_date"].is_monotonic_increasing
    none = simulate(p, s, all_true(p) & False, date(2016, 1, 1), date(2017, 12, 31))
    assert list(none.columns) == TRADE_COLUMNS and none.empty


def test_no_trades_outside_period():
    p = make_panel(1500, ["A", "B", "C", "D"], seed=11)
    s = Breakout({"n": 20, "m": 5, "k": 2.0}).signals(p)
    t = simulate(p, s, all_true(p), date(2020, 1, 1), date(2020, 12, 31))
    assert len(t) > 0
    assert t["entry_date"].min() >= pd.Timestamp("2020-01-01") and t["exit_date"].max() <= pd.Timestamp("2020-12-31")


def test_future_bars_do_not_change_past_trades():
    p = make_panel(800, ["A", "B"], seed=12)
    s = Breakout({"n": 20, "m": 5, "k": 2.0}).signals(p)
    end = date(2017, 12, 29)
    t1 = simulate(p, s, all_true(p), date(2016, 1, 1), end)
    cut = p.close.index > pd.Timestamp(end)
    for f in (p.open, p.high, p.low, p.close):
        f.loc[cut] = f.loc[cut] * 5
    s.entries.loc[cut] = True
    s.exits.loc[cut] = True
    t2 = simulate(p, s, all_true(p), date(2016, 1, 1), end)
    pd.testing.assert_frame_equal(t1, t2)


@pytest.mark.slow
def test_speed_200_symbols_2500_days():
    import time
    syms = [f"S{i:03d}" for i in range(200)]
    p = make_panel(2500, syms, seed=21)
    elig = all_true(p)
    t0 = time.perf_counter()
    s = Breakout(Breakout.grid()[0]).signals(p)
    t = simulate(p, s, elig, date(2016, 1, 1), date(2026, 12, 31))
    elapsed = time.perf_counter() - t0
    assert len(t) > 0
    assert elapsed < 3.0, f"simulate took {elapsed:.2f}s"
