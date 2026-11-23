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
    a = _frame(["2024-01-02 09:01", "2024-01-02 09:02"], "Asia/Seoul", start=1.0)
    b = _frame(["2024-01-02 09:02", "2024-01-02 09:03"], "Asia/Seoul", start=2.0)
    ms.save_minutes("kr", "X", a)
    ms.save_minutes("kr", "X", b)
    d = pd.Timestamp("2024-01-02").date()
    out = ms.load_minutes("kr", "X", d, d)
    assert len(out) == 3
    assert out["close"].tolist() == [1.0, 2.0, 2.0]


def test_kr_trim_drops_nxt_hours():
    times = ["08:01", "08:59", "09:00", "09:01", "15:20", "15:30", "15:31", "20:00"]
    df = _frame([f"2024-01-02 {t}" for t in times], "Asia/Seoul")
    out = ms.regular_session(df, load_market_raw("kr"))
    assert [t.strftime("%H:%M") for t in out.index] == ["09:01", "15:20", "15:30"]


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
