from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrade.data.collect import collect_daily, collect_meta
from autotrade.data.quality import check_daily, clean_daily, hard_issues
from autotrade.data.store import (calendar, load_daily, load_meta, load_panel, save_daily,
                                  save_meta, trading_dates)
from autotrade.toss.client import CandlePage, TossApiError
from tests.fixtures import make_panel


class FakeClient:
    """Serves 1d candles newest-first pages; `before` is inclusive like the real API."""

    def __init__(self, first="2015-06-01", last="2026-11-19", tz="+09:00", hour=0):
        days = pd.bdate_range(first, last)
        self.tz, self.hour = tz, hour
        self.days = [d.strftime("%Y-%m-%d") for d in days]
        self.pages_requested_last_run = 0
        self.not_found = False
        self.stocks = {}
        self.split = None            # ratio of a split: the whole adjusted series rescales

    def _bar(self, d, i):
        ts = pd.Timestamp(f"{d}T{self.hour:02d}:00:00{self.tz}")
        p = 100.0 + i % 50
        if self.split is not None:
            p /= self.split
        return {"timestamp": ts, "open": p, "high": p + 1, "low": p - 1, "close": p + 0.5,
                "volume": 1000.0 + i}

    def add_new_day(self, d):
        self.days.append(d)

    def get_candles(self, symbol, interval, before=None, count=200, adjusted=True):
        if self.not_found:
            raise TossApiError(404, "stock-not-found", "r1")
        if before is None:
            self.pages_requested_last_run = 0
        self.pages_requested_last_run += 1
        bars = [self._bar(d, i) for i, d in enumerate(self.days)]
        if before is not None:
            bars = [b for b in bars if b["timestamp"] <= before]
        page = bars[-count:]
        df = pd.DataFrame(page, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        more = len(bars) > len(page)
        nb = page[0]["timestamp"].to_pydatetime() if more else None  # inclusive boundary
        return CandlePage(df, nb)

    def list_stocks(self, market, status="ACTIVE", security_type=None):
        rows = [
            ("005930", "삼성전자", "STOCK", True), ("005935", "삼성전자우", "STOCK", False),
            ("069500", "KODEX 200", "ETF", True), ("500001", "SomeETN", "ETN", True),
            ("600001", "Warrant", "WARRANT", True),
        ]
        if status == "DELISTED":
            rows = [("999999", "폐지", "STOCK", True)]
        if market == "KOSDAQ":
            rows = [("A" + r[0], *r[1:]) for r in rows]
        return pd.DataFrame([{"symbol": s, "name": n, "security_type": t,
                              "is_common_share": c, "isin": "X"} for s, n, t, c in rows])


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path))
    return tmp_path


@pytest.fixture
def fake_client():
    return FakeClient()


def good_df(n=10):
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0,
                         "volume": 100.0}, index=idx)


def test_split_between_collects_refetches_full_history(tmp_data, caplog):
    """Reviewer probe: a 2:1 split rescales Toss' adjusted history between two collects;
    splicing the new bars onto the old scale would fake a +100% day."""
    c = FakeClient(first="2024-01-02", last="2025-01-31")
    collect_daily(c, "kr", "X", start=date(2024, 1, 1))
    c.add_new_day("2025-02-03")
    c.add_new_day("2025-02-04")
    c.split = 2.0                                  # 2:1 split: adjusted history halved
    collect_daily(c, "kr", "X", start=date(2024, 1, 1))
    df = load_daily("kr", "X")
    assert df["close"].pct_change().abs().max() < 0.4        # no fake -50% splice bar
    expected = pd.Series([c._bar(d, i)["close"] for i, d in enumerate(c.days)],
                         index=pd.to_datetime(c.days))
    assert np.allclose(df["close"].to_numpy(), expected.to_numpy())
    assert df.index.min() == pd.Timestamp("2024-01-02")
    assert "refetch" in caplog.text and "X" in caplog.text


def test_unchanged_overlap_keeps_incremental_fetch(tmp_data):
    c = FakeClient(first="2024-01-02", last="2025-01-31")
    collect_daily(c, "kr", "X", start=date(2024, 1, 1))
    c.add_new_day("2025-02-03")
    assert collect_daily(c, "kr", "X", start=date(2024, 1, 1)) == 1
    assert c.pages_requested_last_run == 1


def test_backfill_when_start_moves_earlier(tmp_data):
    c = FakeClient(first="2015-06-01", last="2025-01-31")
    collect_daily(c, "kr", "X", start=date(2024, 1, 1))
    assert load_daily("kr", "X").index.min() == pd.Timestamp("2024-01-01")
    added = collect_daily(c, "kr", "X", start=date(2016, 1, 1))
    df = load_daily("kr", "X")
    assert df.index.min() == pd.Timestamp("2016-01-01")
    assert added == len(pd.bdate_range("2016-01-01", "2023-12-29"))
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()


def test_history_starting_after_start_is_not_backfilled_again(tmp_data):
    c = FakeClient(first="2020-03-02", last="2025-01-31")      # listed after start
    collect_daily(c, "kr", "X", start=date(2016, 1, 1))
    c.add_new_day("2025-02-03")
    assert collect_daily(c, "kr", "X", start=date(2016, 1, 1)) == 1
    assert c.pages_requested_last_run == 1                    # no backfill pages


def test_save_daily_is_atomic(tmp_data, monkeypatch):
    save_daily("kr", "A", good_df())
    path = tmp_data / "kr" / "1d" / "A.parquet"

    def boom(self, target, *a, **k):
        Path(target).write_bytes(b"partial")                   # dies mid-write
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(pd.DataFrame, "to_parquet", boom)
        with pytest.raises(OSError):
            save_daily("kr", "A", good_df(3))
    assert len(load_daily("kr", "A")) == 10                   # old file intact
    assert [f.name for f in path.parent.iterdir()] == ["A.parquet"]   # no temp leftovers


def test_collect_paginates_until_start(fake_client, tmp_data):
    n = collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1))
    df = load_daily("kr", "005930")
    assert n == len(df) > 400  # more than two pages
    assert fake_client.pages_requested_last_run >= 3
    assert df.index.min() >= pd.Timestamp("2016-01-01")
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert set(df["source"]) == {"toss"}


def test_inclusive_before_boundary_not_duplicated(fake_client, tmp_data):
    collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1))
    assert not load_daily("kr", "005930").index.duplicated().any()


def test_incremental_fetches_only_newer(fake_client, tmp_data):
    collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1))
    before_len = len(load_daily("kr", "005930"))
    fake_client.add_new_day("2026-11-20")
    assert collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1)) == 1
    assert fake_client.pages_requested_last_run == 1
    assert len(load_daily("kr", "005930")) == before_len + 1
    assert collect_daily(fake_client, "kr", "005930", start=date(2016, 1, 1)) == 0


def test_start_exactly_on_bar_stops_without_extra_page(tmp_data):
    c = FakeClient(first="2015-01-01")
    collect_daily(c, "kr", "X", start=date(2016, 1, 4))  # 2016-01-04 is a trading day
    assert load_daily("kr", "X").index.min() == pd.Timestamp("2016-01-04")


def test_stock_not_found_returns_zero(fake_client, tmp_data, caplog):
    fake_client.not_found = True
    assert collect_daily(fake_client, "kr", "NOPE", start=date(2016, 1, 1)) == 0
    assert load_daily("kr", "NOPE") is None
    assert "stock-not-found" in caplog.text


def test_us_timestamps_become_new_york_dates(tmp_data):
    # 13:00+09:00 (EDT, summer) and 14:00+09:00 (EST, winter) are both NY midnight
    summer = FakeClient(first="2026-03-23", last="2026-03-25", hour=13)
    collect_daily(summer, "us", "SPY", start=date(2026, 1, 1))
    assert list(load_daily("us", "SPY").index) == list(pd.to_datetime(
        ["2026-03-23", "2026-03-24", "2026-03-25"]))
    winter = FakeClient(first="2026-01-06", last="2026-01-08", hour=14)
    collect_daily(winter, "us", "QQQ", start=date(2026, 1, 1))
    assert list(load_daily("us", "QQQ").index) == list(pd.to_datetime(
        ["2026-01-06", "2026-01-07", "2026-01-08"]))


def test_kr_timestamps_are_seoul_dates(tmp_data):
    c = FakeClient(first="2026-03-24", last="2026-03-25")
    collect_daily(c, "kr", "A", start=date(2026, 1, 1))
    assert list(load_daily("kr", "A").index) == list(pd.to_datetime(["2026-03-24", "2026-03-25"]))


def test_quality_flags():
    bad = good_df().copy()
    bad.loc[bad.index[3], "high"] = bad["low"].iloc[3] - 1
    assert any("ohlc" in s for s in check_daily(bad))
    assert check_daily(good_df()) == []


def test_clean_daily_drops_suspension_rows_only():
    d = good_df(5)
    d.loc[d.index[1], ["open", "high", "low", "volume"]] = 0.0      # suspension: O/H/L 0, vol 0
    d.loc[d.index[3], "volume"] = 0.0                               # no trades, real prices
    out = clean_daily(d)
    assert list(out.index) == [d.index[i] for i in (0, 2, 3, 4)]
    assert hard_issues(out) == []


def test_nan_cells_are_a_hard_issue():
    d = good_df(5)
    d.loc[d.index[2], "close"] = np.nan
    issues = hard_issues(d)
    assert any("NaN" in i for i in issues)


def test_toss_suspension_rows_are_not_stored(tmp_data):
    c = FakeClient(first="2026-03-23", last="2026-03-27")
    real_bar = c._bar

    def bar(d, i):
        b = real_bar(d, i)
        if d == "2026-03-25":
            b.update(open=0.0, high=0.0, low=0.0, volume=0.0)
        return b
    c._bar = bar
    collect_daily(c, "kr", "X", start=date(2026, 1, 1))
    df = load_daily("kr", "X")
    assert pd.Timestamp("2026-03-25") not in df.index and len(df) == 4
    assert hard_issues(df) == []


def test_quality_other_problems():
    d = good_df()
    dup = pd.concat([d, d.iloc[[2]]])
    assert any("duplicate" in s for s in check_daily(dup))
    d2 = good_df()
    d2.loc[d2.index[1], "low"] = 0.0
    assert any("non-positive" in s for s in check_daily(d2))
    d3 = good_df()
    d3.loc[d3.index[1], "volume"] = -1.0
    assert any("negative volume" in s for s in check_daily(d3))
    gap = pd.concat([good_df(3), good_df(3).set_axis(pd.bdate_range("2024-03-01", periods=3))])
    assert any("gap" in s for s in check_daily(gap))


def test_meta_excludes_preferred_etn_and_maps_product(fake_client, tmp_data):
    meta = collect_meta(fake_client, "kr")
    assert set(meta["product"]) <= {"stock", "etf"}
    assert "ETN" not in set(meta["security_type"])
    assert "WARRANT" not in set(meta["security_type"])
    assert meta.loc["069500", "product"] == "etf"
    assert not meta.loc["005935", "is_common_share"]  # preferred kept, flagged
    assert meta.loc["999999", "delisted"] and not meta.loc["005930", "delisted"]
    assert set(meta["market_board"]) == {"KOSPI", "KOSDAQ"}
    assert load_meta("kr").equals(meta)


def test_load_panel_nan_for_missing_days(tmp_data):
    a, b = good_df(5), good_df(5).iloc[1:]
    save_daily("kr", "A", a)
    save_daily("kr", "B", b)
    p = load_panel("kr", ["A", "B"], date(2024, 1, 1), date(2024, 12, 31))
    assert p.symbols == ["A", "B"] and len(p.close) == 5
    assert np.isnan(p.close.loc[a.index[0], "B"]) and p.close.loc[a.index[1], "B"] == 11.0
    assert list(p.meta.index) == ["A", "B"]
    p2 = load_panel("kr", None, date(2024, 1, 3), date(2024, 1, 4))
    assert len(p2.close) == 2


def test_make_panel_properties_and_overrides():
    p = make_panel(50, ["A", "B"], seed=1, products={"B": "etf"},
                   name={"A": "스팩1호"}, is_common_share={"A": False})
    assert p.close.shape == (50, 2) and p.close.index.tz is None
    assert (p.high >= np.maximum(p.open, p.close)).all().all()
    assert (p.low <= np.minimum(p.open, p.close)).all().all()
    assert (p.low > 0).all().all()
    assert p.meta.loc["A", "name"] == "스팩1호" and not p.meta.loc["A", "is_common_share"]
    assert p.meta.loc["B", "product"] == "etf" and p.meta.loc["B", "security_type"] == "ETF"
    assert p.meta.loc["A", "source"] == "synthetic" and not p.meta.loc["A", "delisted"]
    q = make_panel(50, ["A", "B"], seed=1)
    assert p.close.equals(q.close)


def _store_panel(p):
    for sym in p.close.columns:
        df = pd.DataFrame({k: getattr(p, k)[sym] for k in ("open", "high", "low", "close",
                                                             "volume")}).dropna()
        save_daily("kr", sym, df)
    save_meta("kr", p.meta)


def test_stray_date_is_not_a_trading_day(tmp_data):
    """Reviewer probe: one symbol's bar on a Saturday made the union calendar NaN for all
    others, emptying the universe for 20 days."""
    from dataclasses import replace
    from autotrade.config import load_lab, load_market_raw, load_pocket
    from autotrade.universe.build import eligible_mask
    q = make_panel(300, ["A", "B", "C"], seed=2)
    extra = pd.Timestamp("2016-06-04")                        # a Saturday
    for name in ("open", "high", "low", "close", "volume"):
        df = getattr(q, name).copy()
        df.loc[extra] = np.nan
        df.loc[extra, "C"] = 1e6 if name == "volume" else 100.0
        setattr(q, name, df.sort_index())
    _store_panel(q)
    kept, dropped = calendar("kr")
    assert list(dropped) == [extra] and extra not in kept
    s, e = date(2016, 1, 1), date(2017, 12, 31)
    assert extra not in trading_dates("kr", s, e)
    panel = load_panel("kr", None, s, e)
    assert extra not in panel.close.index
    assert panel.close.index.equals(trading_dates("kr", s, e))
    mc = replace(load_market_raw("us"), always_include=[])
    lab = replace(load_lab(), universe_top_n=3)
    m = eligible_mask(panel, mc, load_pocket("swing"), lab, "paper", 1300.0)
    pos = panel.close.index.searchsorted(extra)
    assert m[["A", "B"]].iloc[pos - 1:pos + 25].all().all()   # no 20-day hole


def test_calendar_keeps_dates_with_half_of_active_symbols(tmp_data):
    a, b = good_df(6), good_df(6).drop(good_df(6).index[2])
    save_daily("kr", "A", a)
    save_daily("kr", "B", b)                                   # B misses day 2: 1 of 2 = 50%
    kept, dropped = calendar("kr")
    assert list(kept) == list(a.index) and dropped.empty
