import numpy as np
import pandas as pd
import pytest

from autotrade.screen.intraday import CANDIDATE_COLUMNS, SessionRules, execute
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


KR_RULES = SessionRules(last_entry="15:20", exit_lead_minutes=0, vi=None)
D = "2024-01-02"


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
    d2 = "2024-01-03"
    b = pd.concat([bars(D, [flat("09:01"), flat("09:02"), flat("15:30")]),
                   bars(d2, [flat("09:01"), flat("09:02"), flat("15:30")])])
    c = cands(cand(D, "09:01"), cand(d2, "09:01"), cand("2024-01-04", "09:01"),
              cand(D, "15:30"))
    t = execute("A", b, c, KR_RULES)
    assert len(t) == 2 and t.exit_date.dt.day.tolist() == [2, 3]
    assert execute("A", b, cands(), KR_RULES).empty
    bad = cands(cand(D, "09:01"))
    bad["day"] = pd.Timestamp(d2)
    with pytest.raises(ValueError):
        execute("A", b, bad, KR_RULES)
