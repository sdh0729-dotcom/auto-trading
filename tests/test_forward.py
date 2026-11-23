"""Forward test (전진 시험): the latest validate's recommended rules are frozen on registration
and scored only on trading days from the registration date on."""
import json
from datetime import timedelta

import pandas as pd
from typer.testing import CliRunner

from autotrade import forward, pipeline
from autotrade.cli import app
from tests.test_cli import cli_env, extend_store, periods_of, run  # noqa: F401 (fixture)


def screened_and_validated():
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "5"])


def recommended(env) -> list[str]:
    rows = pipeline.ExperimentLog(env.research / "experiments.jsonl").rows("swing", "kr")
    val = rows[rows["stage"] == "validate"]
    return list(val.iloc[-1]["metrics"]["recommended_ids"])


def register_after_cutoff(env, monkeypatch, days=1):
    start = periods_of(env).sealed[1] + timedelta(days=days)
    monkeypatch.setattr(forward, "_today", lambda market: start)
    run(["forward-register", "--market", "kr"])
    return start


def test_register_freezes_the_latest_recommendations(cli_env, monkeypatch):
    screened_and_validated()
    start = register_after_cutoff(cli_env, monkeypatch)
    rec = json.loads((cli_env.research / "forward-swing-kr.json").read_text(encoding="utf-8"))
    assert [r["rule_id"] for r in rec["rules"]] == recommended(cli_env)
    assert rec["start"] == start.isoformat()
    assert rec["capital_krw"] == 5_000_000 and rec["benchmark"] == "069500"
    assert all(r["family"] and isinstance(r["params"], dict) for r in rec["rules"])


def test_register_is_once_only(cli_env, monkeypatch):
    screened_and_validated()
    register_after_cutoff(cli_env, monkeypatch)
    r = CliRunner().invoke(app, ["forward-register", "--market", "kr"])
    assert r.exit_code == 1 and "already registered" in r.output


def test_register_refuses_a_start_inside_the_research_data(cli_env, monkeypatch):
    screened_and_validated()
    monkeypatch.setattr(forward, "_today", lambda market: periods_of(cli_env).sealed[1])
    r = CliRunner().invoke(app, ["forward-register", "--market", "kr"])
    assert r.exit_code == 1 and "after the data cutoff" in r.output
    assert not (cli_env.research / "forward-swing-kr.json").exists()


def test_register_needs_a_recommendation(cli_env, monkeypatch):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    monkeypatch.setattr(pipeline, "benchmark_return", lambda *a, **k: 10.0)  # nothing beats it
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "5"])
    monkeypatch.setattr(forward, "_today",
                        lambda market: periods_of(cli_env).sealed[1] + timedelta(days=1))
    r = CliRunner().invoke(app, ["forward-register", "--market", "kr"])
    assert r.exit_code == 1 and "recommended no rule" in r.output


def test_forward_before_any_new_trading_day(cli_env, monkeypatch):
    screened_and_validated()
    register_after_cutoff(cli_env, monkeypatch)
    r = run(["forward", "--market", "kr"])
    assert "no trading day since" in r.output


def test_forward_scores_only_days_from_the_start(cli_env, monkeypatch):
    screened_and_validated()
    start = register_after_cutoff(cli_env, monkeypatch)
    new_days = extend_store(60)
    res = forward.run("swing", "kr")
    assert res.last_date == new_days[-1]
    for r in res.rules:
        assert r["equity"].index.min() >= pd.Timestamp(start)
        assert r["equity"].index.max() == new_days[-1]
        assert r["return"] == r["equity"].iloc[-1] / 5_000_000 - 1
        assert (r["fills"]["entry_date"] >= pd.Timestamp(start)).all()
    out = run(["forward", "--market", "kr"]).output
    assert all(rid in out for rid in recommended(cli_env))
    assert "069500" in out


def test_forward_uses_the_frozen_params_not_the_experiment_log(cli_env, monkeypatch):
    screened_and_validated()
    register_after_cutoff(cli_env, monkeypatch)
    ids = recommended(cli_env)
    extend_store(30)
    (cli_env.research / "experiments.jsonl").unlink()      # later research never changes it
    res = forward.run("swing", "kr")
    assert [r["rule_id"] for r in res.rules] == ids


def test_forward_without_registration(cli_env):
    r = CliRunner().invoke(app, ["forward", "--market", "kr"])
    assert r.exit_code == 1 and "forward-register" in r.output
