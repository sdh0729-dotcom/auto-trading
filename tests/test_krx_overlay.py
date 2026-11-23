from datetime import date

import pandas as pd
import pytest

from autotrade.data import delisted_kr, krx_overlay
from autotrade.data.krx_overlay import (NXT_START, apply_overlay, load_overlay,
                                        load_overlay_record, overlay_path, refresh_overlay,
                                        refresh_overlays, save_overlay_record)
from autotrade.data.store import load_bars, load_daily, load_panel, save_daily

TODAY = pd.Timestamp("2025-04-01")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(delisted_kr.time, "sleep", lambda s: None)


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path))
    return tmp_path


def toss_df(first="2025-01-02", last="2025-03-31", close=100.0):
    idx = pd.DatetimeIndex(pd.bdate_range(first, last), name="date")
    df = pd.DataFrame({"open": close, "high": close + 2, "low": close - 2, "close": close,
                       "volume": 1500.0, "source": "toss"}, index=idx)
    df.loc[df.index >= NXT_START, "open"] = close + 3        # NXT 08:00 pre-market open
    return df


def pykrx_raw(first="2025-01-17", last="2025-04-01", close=100.0):
    idx = pd.DatetimeIndex(pd.bdate_range(first, last), name="날짜")
    return pd.DataFrame({"시가": close, "고가": close + 1, "저가": close - 1, "종가": close,
                         "거래량": 1000, "등락률": 0.0}, index=idx)


def test_overlay_replaces_rows_from_nxt_start_only(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: pykrx_raw())
    assert refresh_overlay("A", TODAY) == "applied"
    bars = load_bars("kr", "A")
    pre, post = bars[bars.index < NXT_START], bars[bars.index >= NXT_START]
    assert (pre["source"] == "toss").all() and (pre["open"] == 100.0).all()
    assert (post["source"] == "krx").all() and (post["open"] == 100.0).all()
    assert (post["volume"] == 1000).all()
    assert len(bars) == len(toss_df())                      # same dates, no duplicates
    assert load_daily("kr", "A")["source"].eq("toss").all()  # Toss file untouched


def test_bars_of_today_are_not_stored(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: pykrx_raw())
    refresh_overlay("A", TODAY)
    assert load_overlay("A").index.max() < TODAY


def test_requests_adjusted_prices_from_before_nxt_start(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    calls = []

    def fake(start, end, sym, adjusted):
        calls.append((start, end, adjusted))
        return pykrx_raw()
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", fake)
    refresh_overlay("A", TODAY)
    start, end, adjusted = calls[0]
    assert adjusted is True and start < "20250304" and end == "20250331"


def test_scale_mismatch_keeps_toss_and_removes_stale_overlay(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: pykrx_raw())
    assert refresh_overlay("A", TODAY) == "applied"
    # a later adjustment only one source applied: closes differ by 2% on every day
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: pykrx_raw(close=98.0))
    assert refresh_overlay("A", TODAY) == "mismatch"
    assert not overlay_path("A").exists()
    assert load_bars("kr", "A")["source"].eq("toss").all()


def test_after_hours_outliers_do_not_fail_the_scale_check(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    raw = pykrx_raw()
    raw.loc[raw.index[-3:], "종가"] = 92.0         # NXT after-hours moved Toss' close 8%
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: raw)
    assert refresh_overlay("A", TODAY) == "applied"


def test_zero_volume_krx_days_become_holes(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    raw = pykrx_raw()
    halted = pd.Timestamp("2025-03-12")
    raw.loc[halted, ["시가", "고가", "저가", "거래량"]] = 0
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: raw)
    refresh_overlay("A", TODAY)
    assert halted not in load_bars("kr", "A").index


def test_no_post_nxt_toss_rows_means_no_overlay(tmp_data, monkeypatch):
    save_daily("kr", "OLD", toss_df(first="2024-01-02", last="2025-02-28"))
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: pytest.fail("no call"))
    assert refresh_overlay("OLD", TODAY) == "none"


def test_empty_pykrx_answer_reported(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: pd.DataFrame())
    rep = refresh_overlays(["A"], TODAY)
    assert rep.empty == ["A"] and rep.applied == 0
    assert load_bars("kr", "A")["source"].eq("toss").all()


def test_failure_is_counted_and_run_continues(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    save_daily("kr", "B", toss_df())

    def fake(start, end, sym, adjusted):
        if sym == "A":
            raise ConnectionError("krx down")
        return pykrx_raw()
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", fake)
    rep = refresh_overlays(["A", "B"], TODAY)
    assert rep.failed == ["A"] and rep.applied == 1


def test_us_bars_never_use_an_overlay(tmp_data):
    save_daily("us", "A", toss_df())
    assert load_bars("us", "A")["source"].eq("toss").all()


def test_panel_uses_the_overlay(tmp_data, monkeypatch):
    save_daily("kr", "A", toss_df())
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: pykrx_raw())
    refresh_overlay("A", TODAY)
    p = load_panel("kr", None, date(2025, 1, 2), date(2025, 3, 31))
    assert p.open.loc[NXT_START, "A"] == 100.0
    assert p.volume.loc[NXT_START, "A"] == 1000


def test_apply_overlay_without_one_returns_toss():
    t = toss_df()
    assert apply_overlay(t, None) is t


def test_record_round_trip(tmp_data):
    rep = krx_overlay.OverlayReport(applied=3, mismatch=["X"], empty=[], failed=["Y", "Z"])
    save_overlay_record(rep, 6, day=date(2026, 11, 22))
    rec = load_overlay_record()
    assert rec == {"date": "2026-11-22", "from": "2025-03-04", "requested": 6, "applied": 3,
                   "mismatch": 1, "empty": 0, "failed": 2}


def test_rounding_overshoot_is_absorbed_but_real_errors_stay(tmp_data, monkeypatch):
    from autotrade.data.quality import hard_issues
    raw = pykrx_raw()
    d1, d2 = pd.Timestamp("2025-03-05"), pd.Timestamp("2025-03-06")
    raw.loc[d1, ["시가", "고가", "저가", "종가"]] = [819275, 876387, 814351, 876388]  # 1 won
    d3 = pd.Timestamp("2025-03-07")
    raw.loc[d3, ["시가", "고가", "저가", "종가"]] = [3479, 3998, 3428, 4004]          # 0.15%
    raw.loc[d2, ["시가", "고가", "저가", "종가"]] = [100, 101, 99, 110]               # 9%
    monkeypatch.setattr(krx_overlay, "_pykrx_ohlcv", lambda *a, **k: raw)
    df = krx_overlay.fetch_krx_daily("A", date(2025, 1, 17), date(2025, 4, 1))
    assert df.loc[d1, "high"] == 876388 and df.loc[d1, "low"] == 814351
    assert df.loc[d3, "high"] == 4004
    assert df.loc[d2, "high"] == 101                       # not hidden
    assert any("ohlc" in i for i in hard_issues(df))
    assert hard_issues(df.drop(d2)) == []
