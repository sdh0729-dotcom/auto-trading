"""Task 5: the `disclosure` swing family and the event-research guards (completeness gate,
event-table hash, limitations, trial count)."""
import json
import shutil
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from autotrade import forward, pipeline
from autotrade.cli import app
from autotrade.config import load_market_raw, repo_root
from autotrade.disclosure import events as E
from autotrade.disclosure.sizes_kr import size_targets
from autotrade.disclosure.store import (append_sizes, filings_path, load_filings, sizes_path,
                                        state_path, targets_path, write_json)
from autotrade.report import limitations
from autotrade.rules.base import atr_stop
from autotrade.rules.registry import EVENT_FAMILIES, SWING_FAMILIES
from autotrade.rules.swing.disclosure import Disclosure
from autotrade.validate.experiments import ExperimentLog
from tests.fixtures import make_panel
from tests.test_cli import SYMBOLS, cli_env, periods_of, run  # noqa: F401 (fixture)
from tests.test_events import BUYBACK, kr_filings, nth

STOCKS = [s for s in SYMBOLS if s != "069500"]


def ev_rows(p, rows):
    """rows: (symbol, row index, bundle_a, size_ok) -> Panel.events (direction +1)."""
    return pd.DataFrame([{"symbol": s, "signal_date": p.close.index[i], "event": "e",
                          "direction": 1, "size": np.nan, "size_ok": ok, "bundle_a": a,
                          "bundle_b": False, "positive": True, "filing_id": f"x{n}"}
                         for n, (s, i, a, ok) in enumerate(rows)], columns=E.EVENT_COLUMNS)


def rule(**kw):
    params = {"bundle": "bundle_a", "hold": 5, "k": 2.0, "confirm": "none", "sized": False}
    params.update(kw)
    return Disclosure(params)


# ---------------------------------------------------------------- the rule
def test_grid_has_108_points():
    g = Disclosure.grid()
    assert len(g) == 108
    assert len({Disclosure(p).rule_id for p in g}) == 108
    assert {p["bundle"] for p in g} == {"bundle_a", "bundle_b", "positive"}
    assert {p["hold"] for p in g} == {5, 10, 20}
    assert {p["k"] for p in g} == {2.0, 3.0}
    assert {p["confirm"] for p in g} == {"none", "ma20", "high20"}
    assert {p["sized"] for p in g} == {True, False}
    assert set(Disclosure.PARAM_KEYS) == {"bundle", "hold", "k", "confirm", "sized"}
    assert SWING_FAMILIES["disclosure"] is Disclosure and "disclosure" in EVENT_FAMILIES


def test_entry_only_on_signal_date_next_open():
    p = make_panel(100, ["A", "B"], seed=3)
    p = replace(p, events=ev_rows(p, [("A", 50, True, True)]))
    s = rule(hold=10, k=3.0).signals(p)
    assert s.entries.sum().sum() == 1 and s.entries.loc[p.close.index[50], "A"]
    assert not s.exits.any().any()
    assert s.entry_at == "next_open" and s.exit_at == "next_open" and s.max_hold_days == 10
    assert s.confidence.loc[p.close.index[50], "A"] == 1.0
    pd.testing.assert_frame_equal(s.stop_pct, atr_stop(p, 3.0))


def test_sized_true_skips_unknown_size():
    p = make_panel(100, ["A", "B"], seed=3)
    p = replace(p, events=ev_rows(p, [("A", 50, True, False), ("B", 60, True, True)]))
    sized = rule(sized=True).signals(p)
    assert sized.entries.sum().sum() == 1 and sized.entries.loc[p.close.index[60], "B"]
    loose = rule(sized=False).signals(p)
    assert loose.entries.sum().sum() == 2
    assert loose.confidence.loc[p.close.index[50], "A"] == 0.5     # size not passing
    assert loose.confidence.loc[p.close.index[60], "B"] == 1.0


def test_bundle_selects_the_event_column():
    p = make_panel(100, ["A"], seed=3)
    p = replace(p, events=ev_rows(p, [("A", 40, False, True), ("A", 70, True, True)]))
    assert rule(bundle="bundle_a").signals(p).entries.sum().sum() == 1
    assert not rule(bundle="bundle_b").signals(p).entries.any().any()
    assert rule(bundle="positive").signals(p).entries.sum().sum() == 2


def test_confirm_high20_requires_breakout():
    p = make_panel(100, ["A", "B"], seed=5)
    idx = p.close.index
    t = idx[50]
    for sym, factor in (("A", 1.05), ("B", 0.97)):
        prior = p.high[sym].iloc[30:50].max()
        p.close.loc[t, sym] = prior * factor
        p.open.loc[t, sym] = prior * factor
        p.low.loc[t, sym] = prior * factor * 0.99
        p.high.loc[t, sym] = prior * 1.5      # the day's own high never counts (prior 20 rows)
    p = replace(p, events=ev_rows(p, [("A", 50, True, True), ("B", 50, True, True)]))
    s = rule(confirm="high20").signals(p)
    assert s.entries.loc[t, "A"] and not s.entries.loc[t, "B"]
    assert rule(confirm="none").signals(p).entries.loc[t].all()


def test_confirm_ma20_is_close_above_sma20():
    p = make_panel(100, ["A", "B"], seed=6)
    t = p.close.index[50]
    sma = p.close.rolling(20).mean()
    p = replace(p, events=ev_rows(p, [("A", 50, True, True), ("B", 50, True, True)]))
    s = rule(confirm="ma20").signals(p)
    for sym in ("A", "B"):
        assert s.entries.loc[t, sym] == bool(p.close.loc[t, sym] > sma.loc[t, sym])


def test_entries_need_the_universe():
    p = make_panel(100, ["A", "B"], seed=3)
    p = replace(p, events=ev_rows(p, [("A", 50, True, True), ("B", 50, True, True)]))
    elig = pd.DataFrame(True, index=p.close.index, columns=p.close.columns)
    elig.loc[p.close.index[50], "B"] = False
    s = rule().signals(p, eligible=elig)
    assert s.entries.loc[p.close.index[50], "A"] and not s.entries.loc[p.close.index[50], "B"]


def test_no_events_raises():
    with pytest.raises(ValueError, match="disclosure rules need Panel.events"):
        rule().signals(make_panel(50, ["A"], seed=1))


# ---------------------------------------------------------------- lab fixtures
@pytest.fixture
def event_env(cli_env, tmp_path, monkeypatch):
    """cli_env + a private config copy (the event table may be edited) + buyback filings
    every 25 trading days on every non-benchmark symbol + a complete collection."""
    cfg = tmp_path / "config"
    shutil.copytree(repo_root() / "config", cfg)
    monkeypatch.setenv("AUTOTRADE_CONFIG", str(cfg))
    cal = pipeline.calendar("kr")[0]
    rows = []
    for j, sym in enumerate(STOCKS):
        for i in range(30 + j, len(cal) - 1, 25):
            rows.append((f"{sym}-{i}", sym, nth(cal, i), BUYBACK))
    kr_filings(*rows)
    fill_sizes()
    complete(STOCKS)
    cli_env.config = cfg
    return cli_env


def fill_sizes(skip=0):
    """A size row for every size target (the gate wants the size pass finished); the last
    `skip` targets get none."""
    t = size_targets(load_filings("kr"))
    t = t.iloc[:len(t) - skip]
    append_sizes("kr", pd.DataFrame({"filing_id": t["filing_id"], "measure": t["measure"],
                                     "value": 1e9}))


def complete(symbols, delisted_targets=0, delisted_mapped=0, done=None):
    write_json(targets_path("kr"), {"symbols": symbols, "targets": len(symbols),
                                    "mapped": len(symbols), "delisted_targets": delisted_targets,
                                    "delisted_mapped": delisted_mapped, "unmapped": []})
    done = symbols if done is None else done
    write_json(state_path("kr"), {s: {"done_through": "2026-11-20", "complete": True}
                                  for s in done})


def fails(args, *needles):
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 1, (r.output, r.exception)
    for n in needles:
        assert n in r.output, r.output
    return r


def edit_table(env):
    p = env.config / "events" / "kr.yaml"
    p.write_text(p.read_text(encoding="utf-8") + "# edited after the screen\n",
                 encoding="utf-8")


def log_rows(env):
    return ExperimentLog(env.research / "experiments.jsonl").rows("swing", "kr")


SCREEN = ["screen", "--pocket", "swing", "--market", "kr"]
VALIDATE = ["validate", "--pocket", "swing", "--market", "kr", "--top", "5"]


# ---------------------------------------------------------------- completeness gate
def test_screen_refuses_incomplete_events(event_env):
    complete(STOCKS, delisted_targets=10, delisted_mapped=5)
    fails(SCREEN + ["--family", "disclosure"], "coverage", "50%")
    assert not (event_env.research / "experiments.jsonl").exists()
    assert not (event_env.research / "periods-swing-kr.json").exists()


def test_screen_refuses_unfinished_size_pass(event_env):
    sizes_path("kr").unlink()
    fill_sizes(skip=3)
    fails(SCREEN + ["--family", "disclosure"], "3 kr filings still need size details",
          "collect-events")
    assert not (event_env.research / "experiments.jsonl").exists()
    fill_sizes()                                  # the missing rows arrive -> passes
    pipeline._refuse_incomplete_events("kr")


def test_us_coverage_message_names_the_cik_mapping(event_env):
    write_json(targets_path("us"), {"symbols": ["X"], "targets": 1, "mapped": 1,
                                    "delisted_targets": 10, "delisted_mapped": 5})
    write_json(state_path("us"), {"X": {"done_through": "2026-11-20", "complete": True}})
    with pytest.raises(pipeline.LabError, match="CIK mapping coverage 50%"):
        pipeline._refuse_incomplete_events("us")


def test_price_only_runs_do_not_touch_the_filings_store(event_env, monkeypatch):
    filings_path("kr").write_bytes(b"corrupt")
    (event_env.config / "events" / "kr.yaml").write_text("{{ not yaml", encoding="utf-8")
    run(SCREEN + ["--family", "breakout"])
    run(VALIDATE)
    start = periods_of(event_env).sealed[1] + timedelta(days=1)
    monkeypatch.setattr(forward, "_today", lambda market: start)
    run(["forward-register", "--market", "kr"])
    forward.run("swing", "kr")                    # price-only forward needs no DART data
    r = CliRunner().invoke(app, SCREEN + ["--family", "disclosure"])
    assert r.exit_code == 1 and "unreadable" in r.output, r.output


def test_screen_refuses_unfinished_targets(event_env):
    complete(STOCKS, done=STOCKS[:4])
    fails(SCREEN + ["--family", "disclosure"], "3 kr event target")
    assert not (event_env.research / "experiments.jsonl").exists()


@pytest.mark.parametrize("targets", [None, {"targets": 0}, {"symbols": []}])
def test_screen_refuses_missing_targets(event_env, targets):
    if targets is None:
        targets_path("kr").unlink()
    else:
        write_json(targets_path("kr"), targets)
    fails(SCREEN + ["--family", "events"], "collect-events")
    assert not (event_env.research / "experiments.jsonl").exists()


def test_screen_refuses_while_collector_runs(event_env, monkeypatch):
    monkeypatch.setattr(pipeline, "events_running", lambda market: True)
    fails(SCREEN + ["--family", "disclosure"], "running")


def test_coverage_at_80_percent_passes(event_env):
    complete(STOCKS, delisted_targets=10, delisted_mapped=8)
    pipeline._refuse_incomplete_events("kr")


def test_screen_without_filings_refuses(cli_env, tmp_path):
    complete(STOCKS)
    fails(SCREEN + ["--family", "disclosure"], "no filings")
    assert not (cli_env.research / "periods-swing-kr.json").exists()


# ---------------------------------------------------------------- family selection
def test_non_event_family_screen_unaffected_without_events(cli_env):
    """No events store, no targets.json: price families screen as before."""
    run(SCREEN + ["--family", "breakout"])
    run(SCREEN)                                       # `all` = the non-event families
    rows = log_rows(cli_env)
    assert not set(rows["family"]) & EVENT_FAMILIES
    assert not any("events_table" in m for m in rows["metrics"])
    assert set(rows["family"]) == set(SWING_FAMILIES) - EVENT_FAMILIES


def test_unknown_family_lists_events(cli_env):
    fails(SCREEN + ["--family", "nope"], "all, events")


def test_screen_events_logs_the_table_hash(event_env):
    run(SCREEN + ["--family", "events"])
    rows = log_rows(event_env)
    assert set(rows["family"]) == EVENT_FAMILIES
    assert len(rows) == sum(len(SWING_FAMILIES[f].grid()) for f in EVENT_FAMILIES)
    assert (rows["family"] == "disclosure").sum() == 108
    h = E.events_table_hash("kr")
    assert all(m["events_table"] == h for m in rows["metrics"])
    assert max(m["n_trades"] for m in rows["metrics"]) >= 30


# ---------------------------------------------------------------- event-table hash
def test_validate_refuses_changed_table(event_env):
    run(SCREEN + ["--family", "disclosure"])
    edit_table(event_env)
    fails(VALIDATE, "events table changed since screen of disclosure:", "re-screen")
    assert not log_rows(event_env)["stage"].eq("validate").any()


def test_validate_logs_the_hash_and_rescreen_clears_the_guard(event_env):
    run(SCREEN + ["--family", "disclosure"])
    edit_table(event_env)
    run(SCREEN + ["--family", "disclosure"])          # new trials on the new table
    run(VALIDATE)
    rows = log_rows(event_env)
    val = rows[rows["stage"] == "validate"]
    assert len(val) and all(m["events_table"] == E.events_table_hash("kr")
                            for m in val["metrics"])
    # the re-screen after the table change counts as new trials (DSR)
    assert ExperimentLog(event_env.research / "experiments.jsonl").n_trials("swing", "kr") \
        == 216


def test_validate_of_price_families_ignores_events(cli_env):
    """A pool without event families needs no targets.json / events at all."""
    run(SCREEN + ["--family", "breakout"])
    run(VALIDATE)


def test_validate_with_event_pool_refuses_incomplete(event_env):
    run(SCREEN + ["--family", "disclosure"])
    complete(STOCKS, done=STOCKS[:1])
    fails(VALIDATE, "event target")


def _event_validated(env) -> str:
    run(SCREEN + ["--family", "disclosure"])
    run(VALIDATE)
    rows = log_rows(env)
    val = rows[(rows["stage"] == "validate") & (rows["family"] == "disclosure")]
    assert len(val)
    return str(val.iloc[-1]["rule_id"])


def test_verify_refuses_changed_table_before_the_seal(event_env):
    rid = _event_validated(event_env)
    edit_table(event_env)
    fails(["verify", "--final", "--pocket", "swing", "--market", "kr", "--rule-id", rid],
          f"events table changed since screen of {rid}")
    assert not (event_env.research / "seal-registry.json").exists()


def test_forward_stores_and_checks_the_table(event_env, monkeypatch):
    rid = _event_validated(event_env)
    monkeypatch.setattr(pipeline, "_latest_recommended", lambda rows, valid: [rid])
    start = periods_of(event_env).sealed[1] + timedelta(days=1)
    monkeypatch.setattr(forward, "_today", lambda market: start)
    run(["forward-register", "--market", "kr"])
    rec = json.loads((event_env.research / "forward-swing-kr.json").read_text("utf-8"))
    assert rec["rules"][0]["events_table"] == E.events_table_hash("kr")
    seen = []
    real = forward.attach_events
    monkeypatch.setattr(forward, "attach_events",
                        lambda *a: seen.append(1) or real(*a))
    forward.run("swing", "kr")
    assert seen == [1]                            # an event rule: events attached
    edit_table(event_env)
    fails(["forward", "--market", "kr"], f"events table changed since screen of {rid}")


def test_forward_register_refuses_changed_table(event_env, monkeypatch):
    rid = _event_validated(event_env)
    monkeypatch.setattr(pipeline, "_latest_recommended", lambda rows, valid: [rid])
    start = periods_of(event_env).sealed[1] + timedelta(days=1)
    monkeypatch.setattr(forward, "_today", lambda market: start)
    edit_table(event_env)
    fails(["forward-register", "--market", "kr"], "events table changed")
    assert not (event_env.research / "forward-swing-kr.json").exists()


def test_missing_logged_hash_is_refused():
    with pytest.raises(pipeline.LabError, match="events table changed since screen of d:1"):
        pipeline._refuse_changed_table("kr", "d:1", {"n_trades": 3})


# ---------------------------------------------------------------- limitations
def test_limitations_event_lines(event_env):
    complete(STOCKS, delisted_targets=40, delisted_mapped=35)
    mc = load_market_raw("kr")
    lines = limitations(mc, family="disclosure")
    assert "국내 공시 시각 없음 → 공시일 다음 거래일 시가 가정" in lines
    assert "공시 제목·양식 기반 분류(규칙 표)" in lines
    assert "상폐 종목 공시 매핑률 88%" in lines
    plain = limitations(mc, family="breakout")
    assert not any("공시" in x for x in plain)
