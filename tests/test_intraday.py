import numpy as np
import pandas as pd
import pytest

from autotrade.screen.intraday import (CANDIDATE_COLUMNS, SessionRules, VIConfig, execute,
                                       vi_halts)
from autotrade.types import EXIT_REASONS, TRADE_COLUMNS

KR = "Asia/Seoul"
US = "America/New_York"


def bars(day, rows, tz=KR):
    """rows: (HH:MM end, open, high, low, close)."""
    idx = pd.DatetimeIndex([pd.Timestamp(f"{day} {r[0]}", tz=tz) for r in rows], name="ts")
    return pd.DataFrame({"open": [r[1] for r in rows], "high": [r[2] for r in rows],
                         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
                         "volume": 1.0}, index=idx)


def flat(t, p=100.0):
    return (t, p, p, p, p)


def cand(day, sig, stop=0.02, target=np.nan, exit_ts=pd.NaT, tz=KR, conf=1.0):
    return {"day": pd.Timestamp(day), "signal_ts": pd.Timestamp(f"{day} {sig}", tz=tz),
            "stop_pct": stop, "target_pct": target, "exit_ts": exit_ts, "confidence": conf}


def cands(*rows):
    return pd.DataFrame(list(rows), columns=CANDIDATE_COLUMNS)


# KR: no entry bar ending after the closing-auction start (close - 10 min: 15:20 / 16:20)
KR_RULES = SessionRules(last_entry_lead=10, exit_lead_minutes=0, vi=None)
D = "2024-01-03"           # a regular 09:00-15:30 day (2024-01-02 opened at 10:00)


def test_entry_is_next_bar_open_never_the_signal_bar():
    b = bars(D, [("09:01", 90, 90, 90, 90), ("09:02", 100, 101, 99, 100.5),
                 ("09:03", 101, 102, 100, 101), ("15:30", 102, 102, 102, 102)])
    t = execute("A", b, cands(cand(D, "09:01")), KR_RULES)
    assert list(t.columns) == TRADE_COLUMNS and len(t) == 1
    r = t.iloc[0]
    assert r.entry_price == 100 and r.entry_date == pd.Timestamp(f"{D} 09:02", tz=KR)
    assert r.symbol == "A" and r.stop_pct == 0.02 and r.confidence == 1.0
    assert r.exit_reason == "session_end" and r.exit_price == 102
    assert r.exit_reason in EXIT_REASONS


def test_fills_skip_missing_minutes():
    b = bars(D, [flat("09:01"), ("09:05", 105, 105, 105, 105), flat("15:30", 106)])
    t = execute("A", b, cands(cand(D, "09:01")), KR_RULES)
    assert t.iloc[0].entry_price == 105
    assert t.iloc[0].entry_date == pd.Timestamp(f"{D} 09:05", tz=KR)


def test_stop_gap_fills_at_open():
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 95, 96, 94, 95), flat("15:30")])
    t = execute("A", b, cands(cand(D, "09:01", stop=0.02)), KR_RULES)
    r = t.iloc[0]
    assert r.exit_reason == "stop" and r.exit_price == 95


def test_stop_intrabar_fills_at_stop_price():
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 100, 97, 98), flat("15:30")])
    r = execute("A", b, cands(cand(D, "09:01", stop=0.02)), KR_RULES).iloc[0]
    assert r.exit_reason == "stop" and r.exit_price == pytest.approx(98.0)


def test_target_fills_and_gap_target_at_open():
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 104, 100, 103), flat("15:30")])
    r = execute("A", b, cands(cand(D, "09:01", target=0.03)), KR_RULES).iloc[0]
    assert r.exit_reason == "target" and r.exit_price == pytest.approx(103.0)
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 105, 106, 105, 105), flat("15:30")])
    r = execute("A", b, cands(cand(D, "09:01", target=0.03)), KR_RULES).iloc[0]
    assert r.exit_reason == "target" and r.exit_price == 105


def test_stop_checked_before_target_in_one_bar():
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 110, 90, 100), flat("15:30")])
    r = execute("A", b, cands(cand(D, "09:01", stop=0.02, target=0.03)), KR_RULES).iloc[0]
    assert r.exit_reason == "stop" and r.exit_price == pytest.approx(98.0)


def test_exit_ts_exits_at_that_bars_close():
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 101, 100, 100.7),
                 flat("09:04", 120), flat("15:30")])
    c = cand(D, "09:01", exit_ts=pd.Timestamp(f"{D} 09:03", tz=KR))
    r = execute("A", b, cands(c), KR_RULES).iloc[0]
    assert r.exit_reason == "signal" and r.exit_price == 100.7


def test_session_exit_uses_last_existing_bar():
    b = bars(D, [flat("09:01"), flat("09:02"), ("15:27", 100, 100, 100, 107)])
    r = execute("A", b, cands(cand(D, "09:01")), KR_RULES).iloc[0]
    assert r.exit_reason == "session_end" and r.exit_price == 107
    assert r.exit_date == pd.Timestamp(f"{D} 15:27", tz=KR)


def test_half_day_exits_at_last_bar():
    d = "2024-11-29"
    b = bars(d, [flat("09:31"), flat("09:32"), flat("12:59"), ("13:00", 100, 100, 100, 111)], US)
    r = execute("A", b, cands(cand(d, "09:31", tz=US)),
                SessionRules(None, 0, None)).iloc[0]
    assert r.exit_price == 111 and r.exit_reason == "session_end"


def test_us_half_day_post_market_bars_never_used():
    d = "2024-11-29"                       # NYSE closes 13:00; Toss still has 13:01-16:00
    b = bars(d, [flat("09:31"), flat("09:32"), flat("12:00", 99), flat("12:30"),
                 ("13:00", 100, 100, 100, 111),
                 ("13:01", 50, 50, 50, 50), flat("15:30", 130), flat("16:00", 140)], US)
    us = SessionRules(None, 0, None)
    r = execute("A", b, cands(cand(d, "09:31", tz=US)), us).iloc[0]
    assert r.exit_reason == "session_end" and r.exit_price == 111     # no post-market stop
    assert r.exit_date == pd.Timestamp(f"{d} 13:00", tz=US)
    for sig in ("13:00", "15:30"):         # the next bar would be post-market: no fill
        assert execute("A", b, cands(cand(d, sig, tz=US)), us).empty
    lead = execute("A", b, cands(cand(d, "09:31", tz=US)), SessionRules(None, 60, None))
    assert lead.iloc[0].exit_date == pd.Timestamp(f"{d} 12:00", tz=US)   # close - 1 h


def test_kr_exam_day_auction_exit_and_last_entry():
    d = "2024-11-14"                       # CSAT day: 10:00-16:30, auction from 16:20
    b = bars(d, [flat("10:01"), flat("15:30", 101), flat("16:19"), flat("16:20"),
                 flat("16:21"), ("16:30", 100, 100, 100, 105)])
    r = execute("A", b, cands(cand(d, "10:01")), KR_RULES).iloc[0]
    assert r.exit_reason == "session_end" and r.exit_price == 105
    assert r.exit_date == pd.Timestamp(f"{d} 16:30", tz=KR)
    ok = execute("A", b, cands(cand(d, "16:19")), KR_RULES)
    assert len(ok) == 1 and ok.iloc[0].entry_date == pd.Timestamp(f"{d} 16:20", tz=KR)
    assert execute("A", b, cands(cand(d, "16:20")), KR_RULES).empty


def test_kr_late_open_bars_before_the_open_never_fill():
    d = "2025-01-02"                       # opens 10:00; padded bars before it are no trades
    b = bars(d, [flat("09:01"), flat("09:02", 90), flat("10:01", 101), flat("15:30", 102)])
    r = execute("A", b, cands(cand(d, "09:01")), KR_RULES).iloc[0]
    assert r.entry_date == pd.Timestamp(f"{d} 10:01", tz=KR) and r.entry_price == 101


def test_no_trade_on_a_non_session_day():
    d = "2024-01-01"                       # KRX holiday
    b = bars(d, [flat("09:01"), flat("09:02"), flat("15:30")])
    assert execute("A", b, cands(cand(d, "09:01")), KR_RULES).empty


def test_no_entry_after_last_entry_kr():
    b = bars(D, [flat("15:19"), flat("15:20"), flat("15:21"), flat("15:30", 101)])
    ok = execute("A", b, cands(cand(D, "15:19")), KR_RULES)
    assert len(ok) == 1 and ok.iloc[0].entry_date == pd.Timestamp(f"{D} 15:20", tz=KR)
    assert execute("A", b, cands(cand(D, "15:20")), KR_RULES).empty


def test_us_real_exits_one_hour_early():
    d = "2024-03-11"
    b = bars(d, [flat("09:31"), flat("09:32"), ("14:59", 100, 100, 100, 104),
                 ("15:00", 100, 100, 100, 105), ("15:30", 100, 100, 100, 106),
                 ("16:00", 100, 100, 100, 107)], US)
    c = cands(cand(d, "09:31", tz=US))
    assert execute("A", b, c, SessionRules(None, 0, None)).iloc[0].exit_price == 107
    r = execute("A", b, c, SessionRules(None, 60, None)).iloc[0]
    assert r.exit_price == 105 and r.exit_reason == "session_end"


def test_overlapping_candidates_dropped():
    b = bars(D, [flat("09:01"), flat("09:02"), flat("09:03"), ("09:04", 100, 100, 97, 98),
                 flat("09:05"), flat("09:06"), flat("15:30")])
    c = cands(cand(D, "09:01"), cand(D, "09:03"), cand(D, "09:04"), cand(D, "09:05"))
    t = execute("A", b, c, KR_RULES)
    # first trade 09:02 -> stop in 09:04 bar; 09:03 dropped; 09:04 signal is not before
    # the exit bar end, so it may re-enter at 09:05
    assert t.entry_date.tolist() == [pd.Timestamp(f"{D} {x}", tz=KR) for x in ("09:02", "09:05")]


def test_multiple_days_and_edge_cases():
    d2 = "2024-01-04"
    b = pd.concat([bars(D, [flat("09:01"), flat("09:02"), flat("15:30")]),
                   bars(d2, [flat("09:01"), flat("09:02"), flat("15:30")])])
    c = cands(cand(D, "09:01"), cand(d2, "09:01"), cand("2024-01-05", "09:01"),
              cand(D, "15:30"))
    t = execute("A", b, c, KR_RULES)
    assert len(t) == 2 and t.exit_date.dt.day.tolist() == [3, 4]
    assert execute("A", b, cands(), KR_RULES).empty
    bad = cands(cand(D, "09:01"))
    bad["day"] = pd.Timestamp(d2)
    with pytest.raises(ValueError):
        execute("A", b, bad, KR_RULES)


VI = VIConfig(static_pct=0.10, dynamic_pct_kospi200=0.03, dynamic_pct_other=0.06,
              halt_minutes=2, etf_exempt=True)
VI_RULES = SessionRules(last_entry_lead=10, exit_lead_minutes=0, vi=VI)


def test_exit_ts_at_session_exit_bar_is_session_end():
    b = bars(D, [flat("09:01"), flat("09:02"), ("15:30", 100, 100, 100, 103)])
    c = cand(D, "09:01", exit_ts=pd.Timestamp(f"{D} 15:30", tz=KR))
    r = execute("A", b, cands(c), KR_RULES).iloc[0]
    assert r.exit_reason == "session_end" and r.exit_price == 103


def test_execute_many_days_matches_per_day():
    days = ["2024-01-03", "2024-01-04", "2024-01-05"]
    b = pd.concat([bars(d, [flat("09:01"), flat("09:02"), ("15:30", 100, 100, 100, 100 + i)])
                   for i, d in enumerate(days)])
    t = execute("A", b, cands(*[cand(d, "09:01") for d in days]), KR_RULES)
    assert t.exit_price.tolist() == [100, 101, 102]


def test_static_vi_halts_two_minutes():
    b = bars(D, [flat("09:01"), ("09:02", 100, 111, 100, 110), flat("09:03", 110),
                 flat("09:04", 110), flat("09:05", 110), flat("09:06", 110)])
    h = vi_halts(b, 100.0, 0.50, VI)
    assert h.index.equals(b.index)
    # trigger bar 09:02 is not halted; (09:02, 09:04] is
    assert h.tolist() == [False, False, True, True, False, False]


def test_static_reference_resets_to_auction_end_price():
    # after the auction the reference is the 09:05 open (110); +9% from it is no new trigger
    b = bars(D, [flat("09:01"), ("09:02", 100, 111, 100, 110), flat("09:03", 110),
                 flat("09:04", 110), ("09:05", 110, 119.9, 110, 119), flat("09:06", 119)])
    assert vi_halts(b, 100.0, 0.50, VI).tolist() == [False, False, True, True, False, False]


def test_dynamic_vi_uses_previous_close():
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 106, 100, 105),
                 flat("09:04", 105), flat("09:05", 105), flat("09:06", 105)])
    h = vi_halts(b, 100.0, 0.06, VI)
    assert h.tolist() == [False, False, False, True, True, False]
    # 5.9% from previous close is not enough
    b2 = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 105.9, 100, 105),
                  flat("09:04", 105)])
    assert not vi_halts(b2, 100.0, 0.06, VI).any()


def test_stop_inside_vi_fills_after_auction():
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 107, 100, 106),
                 ("09:04", 100, 100, 90, 92),          # halted: stop 98 breached
                 flat("09:05", 93), flat("09:06", 94), flat("15:30", 94)])
    t = execute("A", b, cands(cand(D, "09:01", stop=0.02)), VI_RULES, prev_close=100.0)
    r = t.iloc[0]
    assert r.exit_reason == "stop" and r.exit_price == 94
    assert r.exit_date == pd.Timestamp(f"{D} 09:06", tz=KR)


def test_target_inside_vi_needs_open_at_target():
    def mk(reopen):
        return bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 102.5, 100, 101),
                        ("09:04", 100, 104, 100, 104),      # halted: target 103 touched
                        flat("09:05", 101), ("09:06", reopen, reopen, reopen, reopen),
                        flat("15:30", 102)])
    c = cands(cand(D, "09:01", target=0.03))
    kw = dict(prev_close=100.0, dynamic_pct=0.02)
    t = execute("A", mk(101), c, VI_RULES, **kw)
    assert t.iloc[0].exit_reason == "session_end"   # open 101 < target: keep walking
    r = execute("A", mk(104), c, VI_RULES, **kw).iloc[0]
    assert r.exit_reason == "target" and r.exit_price == 104
    assert r.exit_date == pd.Timestamp(f"{D} 09:06", tz=KR)


def test_entry_inside_vi_deferred_and_late_entry_dropped():
    b = bars(D, [flat("09:01"), ("09:02", 100, 107, 100, 106), flat("09:03", 106),
                 flat("09:04", 106), flat("09:05", 107), flat("15:30", 107)])
    t = execute("A", b, cands(cand(D, "09:02")), VI_RULES, prev_close=100.0)
    assert t.iloc[0].entry_date == pd.Timestamp(f"{D} 09:05", tz=KR)
    assert t.iloc[0].entry_price == 107
    late = bars(D, [flat("15:17"), ("15:18", 100, 107, 100, 106), flat("15:19", 106),
                    flat("15:20", 106), flat("15:21", 106), flat("15:30", 106)])
    assert execute("A", late, cands(cand(D, "15:18")), VI_RULES, prev_close=100.0).empty


def test_etf_never_halts_and_no_vi_without_inputs():
    b = bars(D, [flat("09:01"), flat("09:02"), ("09:03", 100, 107, 100, 106),
                 ("09:04", 100, 100, 90, 92), flat("09:05", 93), flat("15:30", 94)])
    c = cands(cand(D, "09:01", stop=0.02))
    t4 = pd.Timestamp(f"{D} 09:04", tz=KR)
    etf = execute("ETF", b, c, VI_RULES, prev_close=100.0, product="etf")
    assert etf.iloc[0].exit_date == t4
    for kw in ({}, {"prev_close": None}):
        assert execute("A", b, c, VI_RULES, **kw).iloc[0].exit_date == t4
    assert execute("A", b, c, KR_RULES, prev_close=100.0).iloc[0].exit_date == t4
    # per-day series: a day missing from it has no VI
    ser = pd.Series({pd.Timestamp("2024-01-04"): 100.0})
    assert execute("A", b, c, VI_RULES, prev_close=ser).iloc[0].exit_date == t4
    ser = pd.Series({pd.Timestamp(D): 100.0})
    r = execute("A", b, c, VI_RULES, prev_close=ser).iloc[0]
    assert r.exit_date == pd.Timestamp(f"{D} 15:30", tz=KR) and r.exit_price == 94
