from datetime import date

import pandas as pd
import pytest

from autotrade.data import collect_minute as cm
from autotrade.data.minute_store import load_minutes, minute_dir
from autotrade.toss.client import CandlePage

BARS_PER_DAY = 390      # kr regular session 09:01 .. 15:30


class FakeMinuteClient:
    """Serves 1m candles newest-last within a page, newest page first; `before` inclusive."""

    def __init__(self, first="2024-01-02", n_days=3, fail_after=None):
        self.days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(first, periods=n_days)]
        self.fail_after, self.pages, self.calls = fail_after, 0, 0
        self.split = None

    def add_day(self):
        nxt = pd.Timestamp(self.days[-1]) + pd.offsets.BDay(1)
        self.days.append(nxt.strftime("%Y-%m-%d"))

    def series(self):
        rows = []
        for di, d in enumerate(self.days):
            for k in range(BARS_PER_DAY):
                ts = pd.Timestamp(f"{d}T09:01:00+09:00") + pd.Timedelta(minutes=k)
                p = 100.0 + (di * BARS_PER_DAY + k) % 50
                if self.split:
                    p /= self.split
                rows.append({"timestamp": ts, "open": p, "high": p + 1, "low": p - 1,
                             "close": p + 0.5, "volume": 10.0 + k})
        return rows

    def get_candles(self, symbol, interval, before=None, count=200, adjusted=True):
        assert interval == "1m"
        if before is None:
            self.calls += 1
        self.pages += 1
        if self.fail_after is not None and self.pages > self.fail_after:
            raise RuntimeError("boom")
        bars = self.series()
        if before is not None:
            bars = [b for b in bars if b["timestamp"] <= before]
        page = bars[-count:]
        df = pd.DataFrame(page, columns=["timestamp", "open", "high", "low", "close", "volume"])
        more = len(bars) > len(page)
        return CandlePage(df, page[0]["timestamp"].to_pydatetime() if more else None)


def _stored(client_days, since=None):
    df = load_minutes("kr", "A", date(2000, 1, 1), date(2100, 1, 1), regular_only=False)
    return df


def _full_ts(client, first_day=None):
    ts = [pd.Timestamp(b["timestamp"]).tz_convert("Asia/Seoul") for b in client.series()]
    if first_day:
        ts = [t for t in ts if t.strftime("%Y-%m-%d") >= first_day]
    return ts


TODAY = pd.Timestamp("2024-02-01")


def test_backward_paging_stops_at_since():
    c = FakeMinuteClient(n_days=3)
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 3), TODAY)
    df = _stored(c)
    assert rep.complete and (minute_dir("kr", "A") / "_complete").exists()
    assert sorted({t.strftime("%Y-%m-%d") for t in df.index}) == ["2024-01-03", "2024-01-04"]
    assert rep.added == len(df) == 2 * BARS_PER_DAY


def test_incremental_fetch_stops_at_stored_last_bar():
    c = FakeMinuteClient(n_days=3)
    cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    c.add_day()
    c.pages = 0
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    assert rep.complete and rep.added == BARS_PER_DAY
    assert c.pages == 2          # 390 new bars = 2 pages (overlap on the 2nd); full history is 8
    assert list(_stored(c).index) == _full_ts(c)


def test_resume_after_abort_has_no_gap():
    c = FakeMinuteClient(n_days=3, fail_after=2)
    with pytest.raises(RuntimeError):
        cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    assert not (minute_dir("kr", "A") / "_complete").exists()
    c.fail_after = None
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    assert rep.complete
    assert list(_stored(c).index) == _full_ts(c)


def test_split_refetches_minutes():
    c = FakeMinuteClient(n_days=3)
    cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    c.split = 2.0
    cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    df = _stored(c)
    assert list(df.index) == _full_ts(c)
    assert df["open"].iloc[0] == pytest.approx(c.series()[0]["open"])
    assert df["open"].max() < 100.0


def test_today_session_not_stored():
    c = FakeMinuteClient(n_days=3)       # 2024-01-02 .. 01-04
    cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), pd.Timestamp("2024-01-04"))
    days = {t.strftime("%Y-%m-%d") for t in _stored(c).index}
    assert days == {"2024-01-02", "2024-01-03"}


def _patch_universe(monkeypatch, days):
    import autotrade.data.store as store
    import autotrade.universe.build as build

    idx = pd.bdate_range("2024-01-01", periods=12)
    cols = ["A", "B", "C", "069500"]
    monkeypatch.setattr(store, "load_panel", lambda *a, **k: object())
    mask = pd.DataFrame(False, index=idx, columns=cols)
    mask.loc[idx[:days["A"]], "A"] = True
    mask.loc[idx[:days["B"]], "B"] = True
    monkeypatch.setattr(build, "eligible_mask", lambda *a, **k: mask)


def test_minute_symbols_union_ordered_by_days(monkeypatch):
    _patch_universe(monkeypatch, {"A": 10, "B": 3})
    out = cm.minute_symbols("kr", date(2024, 1, 1), date(2024, 12, 31), 100)
    assert out[:2] == ["A", "B"]
    assert {"069500", "102110"} <= set(out) and "C" not in out


def test_pauses_while_daily_job_runs(monkeypatch):
    from autotrade import pipeline

    lock = pipeline._daily_job_lock()
    lock.mkdir(parents=True)
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 2:
            lock.rmdir()

    seen = []
    monkeypatch.setattr(pipeline.time, "sleep", fake_sleep)
    monkeypatch.setattr(pipeline, "_client", lambda: object())
    monkeypatch.setattr(pipeline, "minute_symbols", lambda *a, **k: ["A", "B"])
    monkeypatch.setattr(pipeline, "collect_minutes",
                        lambda c, m, s, since, today: seen.append((s, lock.exists()))
                        or cm.MinuteReport(5, True))
    out = pipeline.collect_minute("kr", None)
    assert sleeps == [30, 30]
    assert seen == [("A", False), ("B", False)]
    assert out == {"symbols": 2, "added": 10, "complete": 2, "not_found": 0, "failed": 0}
