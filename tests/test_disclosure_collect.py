import json
from datetime import date

import pandas as pd
import pytest
from typer.testing import CliRunner

from autotrade import pipeline
from autotrade.cli import app
from autotrade.data import store
from autotrade.disclosure import collect_kr
from autotrade.disclosure.collect_kr import collect_events_kr, event_targets
from autotrade.disclosure.dart import DartRateLimited
from autotrade.disclosure.store import (events_root, incomplete_targets, load_filings, read_json, state_path,
                                        targets_path)
from tests.fixtures import make_panel
from tests.test_cli import cli_env  # noqa: F401

TODAY = date(2026, 11, 20)


def filing(n, corp, dt="20240102", title=" 주요사항보고서(자기주식취득결정) "):
    return {"corp_code": corp, "corp_name": "x", "stock_code": "", "corp_cls": "Y",
            "report_nm": title, "rcept_no": f"2024{n:010d}", "rcept_dt": dt, "flr_nm": "회사",
            "rm": "유"}


class FakeClient:
    def __init__(self, data=None, hook=None):
        self.data = data or {}      # (corp_code, kind) -> rows
        self.calls = []
        self.corp_calls = 0
        self.hook = hook

    def corp_codes(self):
        self.corp_calls += 1
        return pd.DataFrame({"corp_code": ["C1", "C2", "C3"],
                             "stock_code": ["000001", "000002", ""],
                             "corp_name": ["a", "b", "c"]})

    def list_filings(self, corp_code, start, end, kind):
        self.calls.append((corp_code, start, end, kind))
        if self.hook:
            self.hook(corp_code, kind)
        return self.data.get((corp_code, kind), [])


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "data"))
    return tmp_path


def test_collect_writes_filings_and_marks_complete(env):
    c = FakeClient({("C1", "B"): [filing(1, "C1")], ("C1", "I"): [filing(2, "C1", "20240301")],
                    ("C2", "B"): [filing(3, "C2")]})
    out = collect_events_kr(c, ["000001", "000002", "999999"], today=TODAY)
    assert out == {"symbols": 3, "filings": 3, "complete": 2, "unmapped": 1,
                   "rate_limited": False}
    df = load_filings("kr")
    assert sorted(df["filing_id"]) == sorted(["2024" + f"{n:010d}" for n in (1, 2, 3)])
    row = df[df["filing_id"] == "2024" + "1".zfill(10)].iloc[0]
    assert (row["symbol"], row["corp_code"], row["kind"], row["filer"], row["remark"]) == \
        ("000001", "C1", "B", "회사", "유")
    assert row["title"] == "주요사항보고서(자기주식취득결정)"
    assert row["filed_date"] == date(2024, 1, 2)
    assert read_json(state_path("kr"), {})["000001"] == {"done_through": "2026-11-20",
                                                         "complete": True}
    assert c.corp_calls == 1
    assert read_json(targets_path("kr"), {})["unmapped"] == ["999999"]


def test_resume_skips_complete_and_continues_from_done_through(env):
    c = FakeClient()
    collect_events_kr(c, ["000001"], start=date(2015, 1, 1), today=TODAY)
    assert {x[1] for x in c.calls} == {date(2015, 1, 1)}
    c2 = FakeClient()
    collect_events_kr(c2, ["000001"], today=TODAY)
    assert c2.calls == []
    c3 = FakeClient()
    collect_events_kr(c3, ["000001"], today=date(2026, 11, 25))
    assert [(x[1], x[3]) for x in c3.calls] == [(TODAY, "B"), (TODAY, "I")]


def test_rate_limit_stops_cleanly_keeps_state(env):
    def hook(corp, kind):
        if corp == "C2":
            raise DartRateLimited("020", "limit")
    out = collect_events_kr(FakeClient(hook=hook), ["000001", "000002"], today=TODAY)
    assert out["rate_limited"] is True and out["complete"] == 1
    assert list(read_json(state_path("kr"), {})) == ["000001"]


def test_interrupt_mid_symbol_no_duplicates(env):
    data = {("C1", "B"): [filing(1, "C1")], ("C1", "I"): [filing(2, "C1")]}

    def hook(corp, kind):
        if kind == "I":
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        collect_events_kr(FakeClient(data, hook), ["000001"], today=TODAY)
    assert len(load_filings("kr")) == 1
    assert read_json(state_path("kr"), {}) == {}
    collect_events_kr(FakeClient(data), ["000001"], today=TODAY)
    df = load_filings("kr")
    assert len(df) == 2 and df["filing_id"].is_unique
    assert read_json(state_path("kr"), {})["000001"]["complete"] is True


def test_targets_exclude_etfs_and_record_coverage(env, monkeypatch):
    syms = ["A01", "A02", "ETF1"]
    p = make_panel(400, syms, seed=3, products={"ETF1": "etf"}, start="2024-01-02")
    for s in syms:
        df = pd.DataFrame({"open": p.open[s] * 80, "high": p.high[s] * 80,
                           "low": p.low[s] * 80, "close": p.close[s] * 80,
                           "volume": p.volume[s]})
        df["source"] = "synthetic"
        store.save_daily("kr", s, df)
    store.save_meta("kr", p.meta)
    monkeypatch.setattr("autotrade.data.collect.market_today",
                        lambda m: pd.Timestamp("2025-08-01"))
    assert event_targets("kr") == ["A01", "A02"]
    collect_events_kr(FakeClient(), ["000001", "000002", "777777"], today=TODAY,
                      delisted={"000002", "777777"})
    assert read_json(targets_path("kr"), {}) == {
        "symbols": ["000001", "000002", "777777"], "targets": 3, "mapped": 2, "delisted_targets": 2, "delisted_mapped": 1,
        "unmapped": ["777777"]}


def test_second_collector_refused(env):
    with pipeline._exclusive_lock(events_root("kr") / ".collect.lock", "x"):
        assert pipeline.events_running("kr") is True
        with pytest.raises(pipeline.LabError, match="lab collect-events --market kr"):
            pipeline.collect_events("kr")
    assert pipeline.events_running("kr") is False


def test_us_not_yet_available(env):
    with pytest.raises(pipeline.LabError, match="EDGAR"):
        pipeline.collect_events("us")


def test_cli_collect_events_runs(cli_env, monkeypatch):  # noqa: F811
    meta = store.load_meta("kr")
    meta["product"] = "stock"
    meta["security_type"] = "STOCK"
    meta["leverage_factor"] = float("nan")
    store.save_meta("kr", meta)
    fake = FakeClient({("C1", "B"): [filing(1, "C1")]})
    monkeypatch.setattr(pipeline, "make_dart_client_from_env", lambda: fake)
    monkeypatch.setattr(collect_kr, "event_targets", lambda m: ["000001", "000002"])
    monkeypatch.setattr(pipeline, "event_targets", lambda m: ["000001", "000002"])
    r = CliRunner().invoke(app, ["collect-events", "--market", "kr", "--limit", "1"])
    assert r.exit_code == 0, (r.output, r.exception)
    assert "symbols 2" in r.output and "filings 1" in r.output
    assert len(load_filings("kr")) == 1
    assert json.loads(targets_path("kr").read_text())["targets"] == 2
    assert incomplete_targets("kr") == ["000002"]


def test_limit_fetches_subset_but_targets_describe_all(env):
    out = collect_events_kr(FakeClient(), ["000001", "000002", "999999"], today=TODAY,
                            limit=1)
    t = read_json(targets_path("kr"), {})
    assert (t["targets"], t["mapped"], t["unmapped"]) == (3, 2, ["999999"])
    assert list(read_json(state_path("kr"), {})) == ["000001"] and out["complete"] == 1
    assert incomplete_targets("kr") == ["000002"]
    collect_events_kr(FakeClient(), ["000001", "000002", "999999"], today=TODAY)
    assert incomplete_targets("kr") == []
