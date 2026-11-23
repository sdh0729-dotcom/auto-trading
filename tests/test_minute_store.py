import pandas as pd
import pytest

from autotrade.config import load_market_raw
from autotrade.data import minute_store as ms


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path))


def _frame(stamps, tz, start=1.0):
    idx = pd.DatetimeIndex(pd.to_datetime(stamps)).tz_localize(tz) if tz else None
    n = len(stamps)
    return pd.DataFrame({"open": start, "high": start, "low": start, "close": start,
                         "volume": [float(i) for i in range(n)]},
                        index=pd.DatetimeIndex(idx, name="ts"))


def test_month_partition_round_trip(tmp_path):
    df = _frame(["2024-02-01 09:01", "2024-01-31 15:30", "2024-01-31 10:00"], "Asia/Seoul")
    ms.save_minutes("kr", "005930", df)
    d = ms.minute_dir("kr", "005930")
    assert sorted(p.name for p in d.glob("*.parquet")) == ["2024-01.parquet", "2024-02.parquet"]
    out = ms.load_minutes("kr", "005930", pd.Timestamp("2024-01-01").date(),
                          pd.Timestamp("2024-02-29").date())
    assert len(out) == 3 and out.index.is_monotonic_increasing
    assert str(out.index.tz) == "Asia/Seoul" and out.index.name == "ts"


def test_save_merges_and_dedupes():
    a = _frame(["2024-01-03 09:01", "2024-01-03 09:02"], "Asia/Seoul", start=1.0)
    b = _frame(["2024-01-03 09:02", "2024-01-03 09:03"], "Asia/Seoul", start=2.0)
    ms.save_minutes("kr", "X", a)
    ms.save_minutes("kr", "X", b)
    d = pd.Timestamp("2024-01-03").date()
    out = ms.load_minutes("kr", "X", d, d)
    assert len(out) == 3
    assert out["close"].tolist() == [1.0, 2.0, 2.0]


def test_kr_trim_drops_nxt_hours():
    times = ["08:01", "08:59", "09:00", "09:01", "15:20", "15:30", "15:31", "20:00"]
    df = _frame([f"2024-01-03 {t}" for t in times], "Asia/Seoul")
    out = ms.regular_session(df, load_market_raw("kr"))
    assert [t.strftime("%H:%M") for t in out.index] == ["09:01", "15:20", "15:30"]


def _kept(day, times, market, tz):
    df = _frame([f"{day} {t}" for t in times], tz)
    return [t.strftime("%H:%M") for t in ms.regular_session(df, load_market_raw(market)).index]


def test_us_half_day_drops_pre_and_post_market():
    # NYSE closes 13:00 on 2024-11-29; Toss US pages carry pre/post-market bars
    times = ["04:01", "09:30", "09:31", "12:59", "13:00", "13:01", "15:30", "16:00", "20:00"]
    assert _kept("2024-11-29", times, "us", "America/New_York") == ["09:31", "12:59", "13:00"]


def test_kr_exam_day_keeps_the_1630_auction():
    # 2024-11-14 (CSAT): 10:00-16:30 KST, closing auction 16:20-16:30
    times = ["08:01", "09:01", "10:00", "10:01", "15:30", "15:31", "16:20", "16:30", "16:31"]
    assert _kept("2024-11-14", times, "kr", "Asia/Seoul") == \
        ["10:01", "15:30", "15:31", "16:20", "16:30"]


def test_kr_new_year_opens_at_ten():
    for day in ("2023-01-02", "2025-01-02"):
        assert _kept(day, ["09:01", "09:30", "10:00", "10:01", "15:30", "15:31"], "kr",
                     "Asia/Seoul") == ["10:01", "15:30"]


def test_non_session_days_have_no_regular_bars():
    assert _kept("2024-01-01", ["09:01", "15:30"], "kr", "Asia/Seoul") == []    # holiday
    assert _kept("2024-01-06", ["09:01", "15:30"], "kr", "Asia/Seoul") == []    # Saturday
    assert _kept("2024-11-28", ["09:31", "16:00"], "us", "America/New_York") == []  # Thanksgiving


def test_session_override_from_config():
    # exchange_calendars has 09:00-15:30 for the 2026-11-19 exam day; kr.yaml corrects it
    assert _kept("2026-11-19", ["09:01", "10:01", "15:31", "16:30", "16:31"], "kr",
                 "Asia/Seoul") == ["10:01", "15:31", "16:30"]


def test_exchange_calendars_verified_sessions():
    """Pins the package against dates verified by hand (ruling R9) and in the stored bars."""
    from autotrade.data.sessions import sessions

    def hm(t):
        return t.strftime("%H:%M")
    us = sessions("us", "2024-11-29", "2024-11-29").iloc[0]
    assert (hm(us.open), hm(us.close)) == ("09:30", "13:00") and pd.isna(us.auction)
    kr = sessions("kr", "2022-11-17", "2026-11-19")
    want = {"2024-11-14": ("10:00", "16:30", "16:20"), "2022-11-17": ("10:00", "16:30", "16:20"),
            "2025-01-02": ("10:00", "15:30", "15:20"), "2023-01-02": ("10:00", "15:30", "15:20"),
            "2024-11-13": ("09:00", "15:30", "15:20"), "2026-11-19": ("10:00", "16:30", "16:20")}
    for day, w in want.items():
        r = kr.loc[pd.Timestamp(day)]
        assert (hm(r.open), hm(r.close), hm(r.auction)) == w, day


def test_us_session_follows_dst():
    cfg = load_market_raw("us")
    for day, off in (("2024-03-08", 5), ("2024-03-11", 4)):
        locals_ = ["09:30", "09:31", "16:00", "16:01"]
        stamps = [pd.Timestamp(f"{day} {t}") + pd.Timedelta(hours=off) for t in locals_]
        df = _frame(stamps, "UTC").tz_convert("UTC")
        out = ms.regular_session(df, cfg)
        assert [t.tz_convert("America/New_York").strftime("%H:%M") for t in out.index] == \
            ["09:31", "16:00"]


def test_bounds_and_empty():
    d = pd.Timestamp("2024-01-02").date()
    assert ms.minute_bounds("kr", "NOPE") is None
    out = ms.load_minutes("kr", "NOPE", d, d)
    assert out.empty and list(out.columns) == ms.MINUTE_COLUMNS
    ms.save_minutes("kr", "A", _frame(["2024-01-02 10:00", "2024-03-05 11:00"], "Asia/Seoul"))
    lo, hi = ms.minute_bounds("kr", "A")
    assert lo.strftime("%Y-%m-%d %H:%M") == "2024-01-02 10:00"
    assert hi.strftime("%Y-%m-%d %H:%M") == "2024-03-05 11:00"
