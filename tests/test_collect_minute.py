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


class PartialSplitClient(FakeMinuteClient):
    """Bars before `cutoff` are rescaled by 1/2 once `split_on` (older adjusted history)."""

    cutoff = pd.Timestamp("2024-01-04T00:00:00+09:00")
    split_on = False

    def series(self):
        rows = super().series()
        if self.split_on:
            for r in rows:
                if r["timestamp"] < self.cutoff:
                    for k in ("open", "high", "low", "close"):
                        r[k] /= 2
        return rows


def test_backward_mismatch_after_abort_refetches_at_new_scale():
    c = PartialSplitClient(n_days=3, fail_after=2)
    with pytest.raises(RuntimeError):
        cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    partial = len(_stored(c))
    assert 0 < partial < 3 * BARS_PER_DAY
    c.fail_after, c.split_on = None, True
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    df = _stored(c)
    assert rep.complete and list(df.index) == _full_ts(c)
    assert rep.added == len(df) == 3 * BARS_PER_DAY      # nothing from before the wipe
    expected = pd.DataFrame(c.series()).set_index("timestamp")["open"]
    assert df["open"].to_numpy() == pytest.approx(expected.to_numpy())


def test_repeated_mismatch_raises_instead_of_looping(monkeypatch):
    monkeypatch.setattr(cm, "_same_scale", lambda *a, **k: False)
    c = FakeMinuteClient(n_days=3)
    with pytest.raises(cm.PriceScaleError):
        cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)


def _spy_saves(monkeypatch):
    calls = []
    real = cm.save_minutes

    def spy(market, symbol, df, **kw):
        calls.append(len(df))
        return real(market, symbol, df, **kw)

    monkeypatch.setattr(cm, "save_minutes", spy)
    return calls


def test_backward_saves_in_batches(monkeypatch):
    c = FakeMinuteClient(n_days=31)          # 12,090 bars = 61 pages of 200
    calls = _spy_saves(monkeypatch)
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY + pd.Timedelta(days=60))
    assert rep.complete
    assert len(calls) <= -(-c.pages // cm.FLUSH_PAGES) + 1 and len(calls) < 10
    assert list(_stored(c).index) == _full_ts(c)
    assert rep.added == sum(calls) == len(_full_ts(c))


def test_abort_flushes_buffered_pages():
    c = FakeMinuteClient(n_days=31, fail_after=30)
    today = TODAY + pd.Timedelta(days=60)
    with pytest.raises(RuntimeError):
        cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), today)
    assert not (minute_dir("kr", "A") / "_complete").exists()
    got = _stored(c)
    assert len(got) == 30 * 200 - 29      # pages overlap by one bar
    assert list(got.index) == _full_ts(c)[-(30 * 200 - 29):]          # contiguous, no hole
    c.fail_after = None
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), today)
    assert rep.complete and (minute_dir("kr", "A") / "_complete").exists()
    assert list(_stored(c).index) == _full_ts(c)


class SplitAfterClient(FakeMinuteClient):
    """Prices halve (a split) once `split_after` pages have been served."""

    split_after = 3

    def get_candles(self, *a, **k):
        self.split = 2.0 if self.pages >= self.split_after else None
        return super().get_candles(*a, **k)


def test_mismatch_detected_inside_a_buffer():
    c = SplitAfterClient(n_days=3)           # 1,170 bars = 6 pages; split shows on page 4
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), TODAY)
    df = _stored(c)
    assert rep.complete and list(df.index) == _full_ts(c)
    assert rep.added == len(df) == 3 * BARS_PER_DAY
    assert df["open"].max() < 100.0                           # a single (post-split) scale


def _fail_second_write(monkeypatch):
    import autotrade.data.minute_store as ms

    real, n = ms.write_parquet_atomic, [0]

    def flaky(df, path):
        n[0] += 1
        if n[0] == 2:
            raise OSError("interrupted between month files")
        return real(df, path)

    monkeypatch.setattr(ms, "write_parquet_atomic", flaky)
    return lambda: monkeypatch.setattr(ms, "write_parquet_atomic", real)


def _assert_contiguous(c, got):
    full = _full_ts(c)
    assert len(got) and list(got.index) == full[full.index(got.index[0]):full.index(got.index[-1]) + 1]


def test_backward_partial_flush_stays_contiguous(monkeypatch):
    c = FakeMinuteClient(n_days=31)
    today = TODAY + pd.Timedelta(days=60)
    restore = _fail_second_write(monkeypatch)
    with pytest.raises(OSError):
        cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), today)
    got = _stored(c)
    assert got.index[-1] == _full_ts(c)[-1]            # the newest month landed first
    _assert_contiguous(c, got)
    restore()
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), today)
    assert rep.complete and list(_stored(c).index) == _full_ts(c)


def test_forward_partial_write_stays_contiguous(monkeypatch):
    c = FakeMinuteClient(n_days=21)                    # 2024-01-02 .. 01-30
    today = TODAY + pd.Timedelta(days=60)
    cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), today)
    for _ in range(5):
        c.add_day()                                    # 01-31 .. 02-06: spans the month edge
    restore = _fail_second_write(monkeypatch)
    with pytest.raises(OSError):
        cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), today)
    got = _stored(c)
    assert got.index[0] == _full_ts(c)[0]
    _assert_contiguous(c, got)
    restore()
    rep = cm.collect_minutes(c, "kr", "A", date(2024, 1, 2), today)
    assert rep.complete and list(_stored(c).index) == _full_ts(c)


def _fake_run(monkeypatch, seen):
    from autotrade import pipeline
    monkeypatch.setattr(pipeline, "_client", lambda: object())
    monkeypatch.setattr(pipeline, "minute_symbols", lambda *a, **k: ["A", "B", "C"])
    monkeypatch.setattr(pipeline, "collect_minutes",
                        lambda c, m, s, since, today: seen.append(s) or cm.MinuteReport(1, True))
    return pipeline


def test_second_collector_refuses_to_start(monkeypatch):
    import fcntl
    seen = []
    pipeline = _fake_run(monkeypatch, seen)
    path = pipeline.collect_lock_path("kr")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as other:                     # another process's lock
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(pipeline.LabError, match="already running"):
            pipeline.collect_minute("kr", None)
    assert seen == []                                  # refused before touching any symbol
    pipeline.collect_minute("kr", None)                # released: runs
    assert seen == ["A", "B", "C"]


def test_collector_holds_the_lock_for_the_whole_run(monkeypatch):
    seen = []
    pipeline = _fake_run(monkeypatch, seen)
    held = []
    monkeypatch.setattr(pipeline, "collect_minutes",
                        lambda c, m, s, since, today: held.append(pipeline.collect_running(m))
                        or cm.MinuteReport(1, True))
    pipeline.collect_minute("kr", None)
    assert held == [True, True, True]
    assert not pipeline.collect_running("kr")          # released afterwards


def test_lock_released_after_a_failed_run(monkeypatch):
    seen = []
    pipeline = _fake_run(monkeypatch, seen)

    def boom(c, m, s, since, today):
        raise KeyboardInterrupt
    monkeypatch.setattr(pipeline, "collect_minutes", boom)
    with pytest.raises(KeyboardInterrupt):
        pipeline.collect_minute("kr", None)
    assert not pipeline.collect_running("kr")


def test_limited_run_marks_and_full_run_clears(monkeypatch):
    seen = []
    pipeline = _fake_run(monkeypatch, seen)
    marker = pipeline.minute_limited_marker("kr")
    pipeline.collect_minute("kr", 2)
    assert seen == ["A", "B"] and marker.exists()
    assert "--limit 2" in marker.read_text()
    pipeline.collect_minute("kr", None)
    assert not marker.exists()


def test_aborted_full_run_keeps_the_limited_marker(monkeypatch):
    seen = []
    pipeline = _fake_run(monkeypatch, seen)
    pipeline.collect_minute("kr", 1)

    def boom(c, m, s, since, today):
        raise KeyboardInterrupt
    monkeypatch.setattr(pipeline, "collect_minutes", boom)
    with pytest.raises(KeyboardInterrupt):
        pipeline.collect_minute("kr", None)
    assert pipeline.minute_limited_marker("kr").exists()


def _fake_shard_run(monkeypatch, seen, syms):
    from autotrade import pipeline
    monkeypatch.setattr(pipeline, "_client", lambda: object())
    monkeypatch.setattr(pipeline, "minute_symbols", lambda *a, **k: list(syms))
    monkeypatch.setattr(pipeline, "collect_minutes",
                        lambda c, m, s, since, today: seen.append(s) or cm.MinuteReport(1, True))
    return pipeline


SYMS = [f"S{i:03d}" for i in range(60)]


def test_shards_partition_the_symbols_by_name(monkeypatch):
    import zlib
    parts = []
    for i in (1, 2, 3):
        seen = []
        pipeline = _fake_shard_run(monkeypatch, seen, SYMS)
        out = pipeline.collect_minute("kr", None, shard=(i, 3))
        assert out["symbols"] == len(seen)
        assert all(zlib.crc32(s.encode()) % 3 == i - 1 for s in seen)
        parts.append(seen)
    assert sorted(sum(parts, [])) == SYMS and all(parts)       # disjoint, complete, non-empty
    # a symbol's shard does not depend on the rest of the list (a later day's universe)
    seen = []
    pipeline = _fake_shard_run(monkeypatch, seen, list(reversed(SYMS))[:25])
    pipeline.collect_minute("kr", None, shard=(1, 3))
    assert set(seen) <= set(parts[0])


def test_shards_run_together_but_a_full_run_or_the_same_shard_refuses(monkeypatch):
    from autotrade import pipeline as pl
    inner = {}

    def nested(c, m, s, since, today):
        if not inner:
            inner["running"] = pl.collect_running(m)
            for key, kw in (("full", {}), ("same", {"shard": (1, 2)}), ("other", {"shard": (2, 2)})):
                try:
                    monkeypatch.setattr(pl, "collect_minutes",
                                        lambda *a: cm.MinuteReport(0, True))
                    pl.collect_minute(m, None, **kw)
                    inner[key] = "ran"
                except pl.LabError as e:
                    inner[key] = str(e)
                finally:
                    monkeypatch.setattr(pl, "collect_minutes", nested)
        return cm.MinuteReport(1, True)

    _fake_shard_run(monkeypatch, [], SYMS)
    monkeypatch.setattr(pl, "collect_minutes", nested)
    pl.collect_minute("kr", None, shard=(1, 2))
    assert inner["running"] is True
    assert "already running" in inner["full"] and "already running" in inner["same"]
    assert inner["other"] == "ran"
    assert not pl.collect_running("kr")


def test_a_full_run_blocks_every_shard(monkeypatch):
    import fcntl
    seen = []
    pipeline = _fake_shard_run(monkeypatch, seen, SYMS)
    path = pipeline.collect_lock_path("kr")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as other:                     # a full run in another process
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(pipeline.LabError, match="already running"):
            pipeline.collect_minute("kr", None, shard=(1, 2))
    assert seen == []


def test_shard_run_marks_partial_and_only_a_full_run_clears(monkeypatch):
    seen = []
    pipeline = _fake_shard_run(monkeypatch, seen, SYMS)
    marker = pipeline.minute_limited_marker("kr")
    pipeline.collect_minute("kr", None, shard=(1, 2))
    pipeline.collect_minute("kr", None, shard=(2, 2))
    assert marker.exists() and "--shard" in marker.read_text()
    pipeline.collect_minute("kr", None)
    assert not marker.exists()


@pytest.mark.parametrize("shard", [(0, 2), (3, 2), (1, 1), (1, 0)])
def test_bad_shard_refused(monkeypatch, shard):
    seen = []
    pipeline = _fake_shard_run(monkeypatch, seen, SYMS)
    with pytest.raises(pipeline.LabError, match="shard"):
        pipeline.collect_minute("kr", None, shard=shard)
    assert seen == []


def test_shard_and_limit_together_refused(monkeypatch):
    pipeline = _fake_shard_run(monkeypatch, [], SYMS)
    with pytest.raises(pipeline.LabError, match="shard"):
        pipeline.collect_minute("kr", 5, shard=(1, 2))
