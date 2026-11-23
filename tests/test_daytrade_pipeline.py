from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from autotrade import pipeline
from autotrade.data import store
from autotrade.data.collect_minute import COMPLETE_MARKER
from autotrade.data.minute_store import load_minutes, minute_dir, save_minutes
from autotrade.rules.daytrade.base import IntradayRule
from autotrade.rules.registry import DAYTRADE_FAMILIES
from autotrade.screen import run as screen_run
from autotrade.screen.intraday import CANDIDATE_COLUMNS
from autotrade.validate.experiments import ExperimentLog
from tests.fixtures import make_panel

SYMBOLS = ["069500", "A01", "A02", "A03"]
MINUTE_DAYS = pd.bdate_range("2023-01-02", periods=60)
SCALE = 80.0


class Toy(IntradayRule):
    """One trade per selected day: signal on the 10:00 bar, stop `stop`, session exit."""
    family = "toy"
    PARAM_KEYS = ("stop",)

    @classmethod
    def grid(cls):
        return [{"stop": 0.02}, {"stop": 0.0001}]

    def candidates(self, symbol, bars, days, daily):
        rows = []
        for d in days:
            ts = pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz="Asia/Seoul")
            if ts in bars.index:
                rows.append({"day": d, "signal_ts": ts, "stop_pct": self.params["stop"],
                             "target_pct": np.nan, "exit_ts": pd.NaT, "confidence": 1.0})
        return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _config_with_minute_start(tmp_path, monkeypatch, day) -> None:
    """A copy of the repo config whose kr minute_start is the synthetic store's first day."""
    import shutil
    from autotrade.config import repo_root
    cfg = tmp_path / "config"
    shutil.copytree(repo_root() / "config", cfg)
    kr = cfg / "markets" / "kr.yaml"
    text = kr.read_text(encoding="utf-8")
    assert "minute_start: 2022-11-23" in text
    kr.write_text(text.replace("minute_start: 2022-11-23", f"minute_start: {day:%Y-%m-%d}"),
                  encoding="utf-8")
    monkeypatch.setenv("AUTOTRADE_CONFIG", str(cfg))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "data"))
    _config_with_minute_start(tmp_path, monkeypatch, MINUTE_DAYS[0])
    monkeypatch.setenv("AUTOTRADE_RESEARCH", str(tmp_path / "research"))
    synced = tmp_path / "synced.yaml"
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(synced))
    synced.write_text(yaml.safe_dump({
        "commission_rate": {"kr": 0.00015, "us": 0.001}, "usd_krw": 1380.0,
        "synced_at": "2026-11-20T00:00:00+00:00",
        "source": {"commission_rate": {"kr": "manual", "us": "manual"}, "usd_krw": "api"}}))
    monkeypatch.setitem(DAYTRADE_FAMILIES, "toy", Toy)
    p = make_panel(520, SYMBOLS, seed=3, products={s: "etf" for s in SYMBOLS},
                   start="2022-01-03")
    for sym in SYMBOLS:
        df = pd.DataFrame({"open": p.open[sym] * SCALE, "high": p.high[sym] * SCALE,
                           "low": p.low[sym] * SCALE, "close": p.close[sym] * SCALE,
                           "volume": p.volume[sym]})
        df["source"] = "synthetic"
        store.save_daily("kr", sym, df)
    store.save_meta("kr", p.meta)
    rng = np.random.default_rng(1)
    for sym in SYMBOLS:
        frames = []
        for d in MINUTE_DAYS:
            ends = pd.date_range(f"{d:%Y-%m-%d} 09:01", f"{d:%Y-%m-%d} 15:30", freq="min",
                                 tz="Asia/Seoul")
            c = 8000 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(ends))))
            frames.append(pd.DataFrame({"open": c, "high": c * 1.0005, "low": c * 0.9995,
                                        "close": c, "volume": 100.0},
                                       index=pd.DatetimeIndex(ends, name="ts")))
        if sym != "A03":                      # A03 has no minute data on the last days
            save_minutes("kr", sym, pd.concat(frames))
        else:
            save_minutes("kr", sym, pd.concat(frames[:30]))
        (minute_dir("kr", sym) / COMPLETE_MARKER).touch()     # a finished collection
    return tmp_path


def test_daytrade_periods_use_the_minute_calendar(env):
    cal = pipeline.minute_calendar("kr")
    assert list(cal) == list(MINUTE_DAYS)
    pipeline.screen("daytrade", "kr", "toy")
    rec = pipeline.period_store.load_record(pipeline.periods_path("daytrade", "kr"))
    assert rec.periods.train[0] == MINUTE_DAYS[0].date()
    assert rec.periods.sealed[1] == MINUTE_DAYS[-1].date() == rec.data_cutoff
    assert [len(MINUTE_DAYS[(MINUTE_DAYS >= pd.Timestamp(a)) & (MINUTE_DAYS <= pd.Timestamp(b))])
            for a, b in (rec.periods.train, rec.periods.valid, rec.periods.sealed)] == [36, 12, 12]
    assert pipeline.periods_path("daytrade", "kr").name == "periods-daytrade-kr.json"


def test_screen_never_loads_bars_after_train_end(env, monkeypatch):
    seen = []
    real = screen_run.load_minutes

    def spy(market, symbol, start, end, *a, **k):
        seen.append(end)
        return real(market, symbol, start, end, *a, **k)
    monkeypatch.setattr(screen_run, "load_minutes", spy)
    pipeline.screen("daytrade", "kr", "toy")
    rec = pipeline.period_store.load_record(pipeline.periods_path("daytrade", "kr"))
    assert seen and max(seen) <= rec.periods.train[1] < rec.periods.sealed[0]


def test_cost_r_precheck_logs_and_skips(env):
    pipeline.screen("daytrade", "kr", "toy")
    rows = ExperimentLog(pipeline.experiments_path()).rows("daytrade", "kr")
    pre = rows[rows["stage"] == "precheck"]
    assert len(pre) == 1 and pre.iloc[0]["params"] == {"stop": 0.0001}
    assert pre.iloc[0]["metrics"]["cost_r"] > 0.15
    scr = rows[rows["stage"] == "screen"]
    assert scr["params"].tolist() == [{"stop": 0.02}]


def test_screen_logs_one_trial_per_grid_point(env, monkeypatch):
    monkeypatch.setattr(Toy, "grid", classmethod(lambda cls: [{"stop": 0.02}, {"stop": 0.03},
                                                                {"stop": 0.04}]))
    df = pipeline.screen("daytrade", "kr", "toy")
    rows = ExperimentLog(pipeline.experiments_path()).rows("daytrade", "kr")
    scr = rows[rows["stage"] == "screen"]
    assert len(scr) == 3 and scr["rule_id"].nunique() == 3 and len(df) == 3
    assert (df["n_trades"] > 0).all()
    assert set(df["rule_id"]) == set(scr["rule_id"])


def test_swing_helpers_still_refuse_unknown_pocket(env):
    with pytest.raises(pipeline.LabError):
        pipeline.screen("nope", "kr")


def test_screen_intraday_refuses_a_period_reaching_the_seal(env):
    from autotrade.validate.seal import SealViolation
    with pytest.raises(SealViolation):
        screen_run.screen_intraday("toy", "kr", [], date(2023, 1, 2), date(2023, 2, 1), None,
                                   None, None, None, None, "daytrade", date(2023, 2, 1), cost_r_max=0.15)


def test_validate_daytrade_reports_daily_metrics(env):
    pipeline.screen("daytrade", "kr", "toy")
    rec = pipeline.period_store.load_record(pipeline.periods_path("daytrade", "kr"))
    res = pipeline.validate("daytrade", "kr", 3)
    assert not res.table.empty
    assert {"valid_mean_daily_return", "valid_positive_day_share", "valid_worst_day"} \
        <= set(res.table.columns)
    rows = ExperimentLog(pipeline.experiments_path()).rows("daytrade", "kr")
    val = rows[rows["stage"] == "validate"]
    assert len(val) == 1 and val.iloc[0]["period"] == [
        rec.periods.valid[0].isoformat(), rec.periods.valid[1].isoformat()]
    m = val.iloc[0]["metrics"]
    assert m["valid_n_trades"] > 0 and m["validate_run"] == 1
    assert {"valid_mean_daily_return", "valid_positive_day_share", "valid_worst_day"} <= set(m)
    assert 0.0 <= m["valid_positive_day_share"] <= 1.0
    assert m["valid_worst_day"] <= m["valid_mean_daily_return"]
    assert len(m["valid_segment_net_sums"]) == pipeline.VALID_SEGMENTS
    rec2 = pipeline.period_store.load_record(pipeline.periods_path("daytrade", "kr"))
    assert rec2.validate_runs == 1


def test_validate_never_loads_bars_after_valid_end(env, monkeypatch):
    seen = []
    real = screen_run.load_minutes

    def spy(market, symbol, start, end, *a, **k):
        seen.append(end)
        return real(market, symbol, start, end, *a, **k)
    monkeypatch.setattr(screen_run, "load_minutes", spy)
    pipeline.screen("daytrade", "kr", "toy")
    seen.clear()
    pipeline.validate("daytrade", "kr", 3)
    rec = pipeline.period_store.load_record(pipeline.periods_path("daytrade", "kr"))
    assert seen and max(seen) <= rec.periods.valid[1] < rec.periods.sealed[0]


def test_daytrade_has_no_benchmark_gate(env, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("daytrade must not consult the benchmark")
    monkeypatch.setattr(pipeline, "benchmark_return", boom)
    pipeline.screen("daytrade", "kr", "toy")
    res = pipeline.validate("daytrade", "kr", 3)
    assert res.benchmark is None and res.benchmark_return is None
    pos = res.table[res.table["valid_net_pnl"] > 0]["rule_id"]
    assert set(res.recommended) == set(pos)


def test_daytrade_verify_final_still_refused(env):
    with pytest.raises(pipeline.LabError, match="daytrade"):
        pipeline.verify_final("daytrade", "kr", "x")


def test_symbol_choice_reads_no_data_after_train_end(env, monkeypatch):
    daily_ends = []
    real = pipeline.load_panel

    def spy(market, symbols, start, end):
        daily_ends.append(end)
        return real(market, symbols, start, end)
    monkeypatch.setattr(pipeline, "load_panel", spy)

    def boom(*a, **k):
        raise AssertionError("symbol choice must not use the full-history daily universe")
    monkeypatch.setattr(pipeline, "minute_symbols", boom)
    pipeline.screen("daytrade", "kr", "toy")
    rec = pipeline.period_store.load_record(pipeline.periods_path("daytrade", "kr"))
    assert daily_ends and max(daily_ends) <= rec.periods.train[1]


def test_cost_r_max_comes_from_the_lab_criteria(env, monkeypatch):
    pipeline.screen("daytrade", "kr", "toy")
    log = ExperimentLog(pipeline.experiments_path())
    assert log.n_trials("daytrade", "kr") == 2      # the skipped precheck point counts too
    assert not hasattr(screen_run, "PRECHECK_COST_R")


class Roll(Toy):
    """Toy trade on days whose 14-day rolling feature exists (like orb_sip / kr_open_rev)."""
    family = "roll"

    @classmethod
    def grid(cls):
        return [{"stop": 0.02}, {"stop": 0.03}]

    def day_features(self, symbol, bars):
        v = bars["volume"].groupby(bars.index.tz_localize(None).normalize()).sum()
        v.index.name = "day"
        return pd.DataFrame({"m": v.rolling(14, min_periods=14).mean().shift(1)})

    def select(self, day_features, daily, eligible):
        stacked = eligible.stack()
        elig = stacked[stacked.astype(bool)].index.set_names(["day", "symbol"])
        f = day_features.reindex(elig)
        return f.index[f["m"].notna().to_numpy()]


def _valid_setup(env):
    pipeline.screen("daytrade", "kr", "toy")
    s = pipeline._setup("daytrade", "kr")
    v0, v1 = s.periods.valid
    panel = pipeline._panel(s, v1)
    return (s, v0, v1, panel, pipeline._eligible(s, panel, "paper"),
            pipeline._cost_model(s.market, panel.meta), pipeline._session_rules(s.market))


def test_validate_features_warm_up_before_the_valid_start(env, monkeypatch):
    monkeypatch.setitem(DAYTRADE_FAMILIES, "roll", Roll)
    s, v0, v1, panel, elig, cm, sess = _valid_setup(env)
    grid = [Roll(p) for p in Roll.grid()]
    syms = pipeline._minute_universe("kr")
    cold = screen_run.intraday_trades(grid, "kr", syms, v0, v1, panel, elig, cm, sess)[0]
    assert all(not parts for parts in cold.values())      # no feature in the first 14 days
    warm = screen_run.intraday_trades(grid, "kr", syms, v0, v1, panel, elig, cm, sess,
                                      features_start=(pd.Timestamp(v0) - pd.Timedelta(days=30)).date())[0]
    first = pd.concat(warm[grid[0].rule_id])
    assert pd.Timestamp(first["entry_date"].min()).tz_localize(None).normalize() == pd.Timestamp(v0)
    assert pd.Timestamp(first["entry_date"].min()).tz_localize(None).normalize() >= pd.Timestamp(v0)


def test_validate_loads_minutes_once_per_family_pass_and_matches_per_rule(env, monkeypatch):
    s, v0, v1, panel, elig, cm, sess = _valid_setup(env)
    syms = pipeline._minute_universe("kr")
    grid = [Toy({"stop": 0.02}), Toy({"stop": 0.03})]
    one = [screen_run.intraday_trades([r], "kr", syms, v0, v1, panel, elig, cm, sess)[0]
           for r in grid]
    calls = []
    real = screen_run.load_minutes
    monkeypatch.setattr(screen_run, "load_minutes",
                        lambda m, sym, a, b, *x, **k: (calls.append(sym), real(m, sym, a, b, *x, **k))[1])
    both = screen_run.intraday_trades(grid, "kr", syms, v0, v1, panel, elig, cm, sess)[0]
    assert len(calls) == len(set(calls))                  # once per symbol, not per rule
    for r, solo in zip(grid, one):
        a, b = pd.concat(solo[r.rule_id]), pd.concat(both[r.rule_id])
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_validate_runs_one_minute_pass_per_family(env, monkeypatch):
    pipeline.screen("daytrade", "kr", "toy")
    calls = []
    real = screen_run.load_minutes
    monkeypatch.setattr(screen_run, "load_minutes",
                        lambda m, sym, a, b, *x, **k: (calls.append(sym), real(m, sym, a, b, *x, **k))[1])
    pipeline.validate("daytrade", "kr", 3)
    assert max(calls.count(sym) for sym in set(calls)) == 1   # toy has no features: 1 load


# ---------------------------------------------------------------- incomplete minute data
def _screen_refused(match):
    with pytest.raises(pipeline.LabError, match=match):
        pipeline.screen("daytrade", "kr", "toy")
    assert not pipeline.periods_path("daytrade", "kr").exists()     # nothing irreversible
    assert not pipeline.experiments_path().exists()


def test_screen_refuses_a_symbol_without_complete_marker(env):
    (minute_dir("kr", "A02") / COMPLETE_MARKER).unlink()
    _screen_refused("A02")


def test_validate_refuses_a_symbol_without_complete_marker(env):
    pipeline.screen("daytrade", "kr", "toy")
    (minute_dir("kr", "A01") / COMPLETE_MARKER).unlink()
    with pytest.raises(pipeline.LabError, match="A01"):
        pipeline.validate("daytrade", "kr", 3)


def test_screen_refuses_while_collect_minute_limited_marker_exists(env):
    pipeline.minute_limited_marker("kr").write_text("lab collect-minute --limit 3\n")
    _screen_refused("--limit")


def test_validate_refuses_while_collect_minute_limited_marker_exists(env):
    pipeline.screen("daytrade", "kr", "toy")
    pipeline.minute_limited_marker("kr").write_text("lab collect-minute --limit 3\n")
    with pytest.raises(pipeline.LabError, match="--limit"):
        pipeline.validate("daytrade", "kr", 3)


def test_screen_refuses_a_benchmark_history_short_of_minute_start(env, tmp_path, monkeypatch):
    # benchmark 069500 starts on MINUTE_DAYS[0]; minute_start 11 calendar days earlier
    _config_with_minute_start(tmp_path / "late", monkeypatch,
                              MINUTE_DAYS[0] - pd.Timedelta(days=pipeline.MINUTE_START_TOLERANCE_DAYS + 1))
    _screen_refused("069500")


def test_screen_tolerates_a_few_missing_first_days(env, tmp_path, monkeypatch):
    _config_with_minute_start(tmp_path / "ok", monkeypatch,
                              MINUTE_DAYS[0] - pd.Timedelta(days=pipeline.MINUTE_START_TOLERANCE_DAYS))
    pipeline.screen("daytrade", "kr", "toy")
    assert pipeline.periods_path("daytrade", "kr").exists()


def test_screen_refuses_a_missing_benchmark(env):
    import shutil
    shutil.rmtree(minute_dir("kr", "069500"))
    with pytest.raises(pipeline.LabError, match="069500"):
        pipeline.screen("daytrade", "kr", "toy")


def test_screen_and_validate_refuse_while_the_collector_runs(env):
    pipeline.screen("daytrade", "kr", "toy")
    with pipeline._collect_lock("kr"):
        with pytest.raises(pipeline.LabError, match="collect-minute"):
            pipeline.screen("daytrade", "kr", "toy")
        with pytest.raises(pipeline.LabError, match="collect-minute"):
            pipeline.validate("daytrade", "kr", 3)
    pipeline.validate("daytrade", "kr", 3)                 # lock released: allowed again

