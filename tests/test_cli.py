import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from autotrade import pipeline
from autotrade.cli import app
from autotrade.data import store
from autotrade.validate.experiments import ExperimentLog
from autotrade.validate.split import split_periods
from tests.fixtures import make_panel

SYMBOLS = ["069500", "A01", "A02", "A03", "A04", "A05", "A06", "A07"]
N_DAYS = 1000
SCALE = 80.0          # ~8,000 KRW ETFs: pass the KR tick-cost filter and the real price cap


@dataclass
class CliEnv:
    root: Path
    data: Path
    research: Path
    synced: Path


def write_synthetic(seed: int = 2) -> pd.DatetimeIndex:
    p = make_panel(N_DAYS, SYMBOLS, seed=seed, products={s: "etf" for s in SYMBOLS})
    for sym in SYMBOLS:
        df = pd.DataFrame({"open": p.open[sym] * SCALE, "high": p.high[sym] * SCALE,
                           "low": p.low[sym] * SCALE, "close": p.close[sym] * SCALE,
                           "volume": p.volume[sym]})
        df["source"] = "synthetic"
        store.save_daily("kr", sym, df)
    store.save_meta("kr", p.meta)
    return p.close.index


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    env = CliEnv(tmp_path, tmp_path / "data", tmp_path / "research",
                 tmp_path / "synced.yaml")
    monkeypatch.setenv("AUTOTRADE_DATA", str(env.data))
    monkeypatch.setenv("AUTOTRADE_RESEARCH", str(env.research))
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(env.synced))
    env.synced.write_text(yaml.safe_dump({
        "commission_rate": {"kr": 0.00015, "us": 0.001}, "usd_krw": 1380.0,
        "synced_at": "2026-11-20T00:00:00+00:00",
        "source": {"commission_rate": {"kr": "manual", "us": "manual"}, "usd_krw": "api"}}))

    def no_network(*a, **k):
        raise AssertionError("network client must not be created in this test")
    monkeypatch.setattr(pipeline, "make_client_from_env", no_network)
    write_synthetic()
    return env


def run(args):
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, (r.output, r.exception)
    return r


def first_recommended_rule_id(env: CliEnv) -> str:
    rows = ExperimentLog(env.research / "experiments.jsonl").rows("swing", "kr")
    val = rows[rows["stage"] == "validate"]
    rec = [r for r, m in zip(val["rule_id"], val["metrics"]) if m["recommended"]]
    assert rec, "validate recommended no rule on the synthetic panel"
    return rec[0]


def periods_of(env: CliEnv):
    idx = pd.bdate_range("2016-01-04", periods=N_DAYS)
    return split_periods(idx, (0.6, 0.2, 0.2))


def test_cli_verify_requires_final(cli_env):
    r = CliRunner().invoke(app, ["verify", "--pocket", "swing", "--market", "kr",
                                 "--rule-id", "x"])
    assert r.exit_code != 0 and "--final" in r.output
    assert not (cli_env.research / "seal-registry.json").exists()


def test_cli_end_to_end_on_synthetic(cli_env):
    r = run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    assert "breakout:" in r.output
    r = run(["validate", "--pocket", "swing", "--market", "kr", "--top", "3"])
    assert "PBO" in r.output
    rid = first_recommended_rule_id(cli_env)
    run(["verify", "--final", "--pocket", "swing", "--market", "kr", "--rule-id", rid])
    cards = list((cli_env.research / "rule-cards").glob("swing-kr-*.json"))
    assert cards
    card = json.loads(cards[0].read_text(encoding="utf-8"))
    assert card["rule_id"] == rid
    assert card["scenarios"].keys() == {"paper", "real"}
    assert card["scenarios"]["paper"]["capital"] == 5_000_000
    assert card["scenarios"]["real"]["capital"] == 50_000
    assert set(card["verdict"]) >= {"② 모의 자격", "실돈 자격"}
    assert card["parity"]["paper"]["matched_share"] == 1.0
    per = periods_of(cli_env)
    assert card["periods"]["sealed"] == [per.sealed[0].isoformat(), per.sealed[1].isoformat()]

    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    final = rows[rows["stage"] == "final"]
    assert list(final["rule_id"]) == [rid]
    assert final.iloc[0]["metrics"]["paper"]["n_trials"] == 27     # breakout grid size

    r = CliRunner().invoke(app, ["verify", "--final", "--pocket", "swing", "--market", "kr",
                                 "--rule-id", rid])
    assert r.exit_code != 0   # 재오픈 거부
    assert "already opened" in r.output and "Traceback" not in r.output

    r = run(["report", "--pocket", "swing", "--market", "kr"])
    board = cli_env.research / "leaderboard-swing-kr.md"
    assert board.exists() and rid in board.read_text(encoding="utf-8")


def test_screen_and_validate_never_load_sealed_prices(cli_env, monkeypatch):
    sealed_start = pd.Timestamp(periods_of(cli_env).sealed[0])
    ends = []
    real_load = pipeline.load_panel

    def spy(market, symbols, start, end):
        ends.append(pd.Timestamp(end))
        p = real_load(market, symbols, start, end)
        assert p.close.index.max() < sealed_start
        return p
    monkeypatch.setattr(pipeline, "load_panel", spy)
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "2"])
    assert ends and all(e < sealed_start for e in ends)
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    for period in rows["period"]:
        assert pd.Timestamp(period[1]) < sealed_start


def test_validate_records_walk_forward_portfolio_and_pbo(cli_env):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "3"])
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    val = rows[rows["stage"] == "validate"]
    assert len(val) == 3
    per = periods_of(cli_env)
    for period, m in zip(val["period"], val["metrics"]):
        assert period == [per.valid[0].isoformat(), per.valid[1].isoformat()]
        assert len(m["wf_test_net_sums"]) == 4
        assert {"valid_net_pnl", "valid_daily_sharpe", "valid_n_trades", "pbo",
                "recommended"} <= m.keys()
        assert 0.0 <= m["pbo"] <= 1.0
    assert len({m["pbo"] for m in val["metrics"]}) == 1         # one PBO for the run
    rec = [m for m in val["metrics"] if m["recommended"]]
    assert len(rec) <= 3 and all(m["valid_net_pnl"] > 0 for m in rec)
    # candidates were the top screened by mean_net_r with >= 30 train trades
    scr = rows[rows["stage"] == "screen"]
    ok = [(m["mean_net_r"], r) for r, m in zip(scr["rule_id"], scr["metrics"])
          if m["n_trades"] >= 30]
    top3 = [r for _, r in sorted(ok, reverse=True)[:3]]
    assert set(val["rule_id"]) == set(top3)


def test_verify_opens_seal_before_loading_sealed_data(cli_env, monkeypatch):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "3"])
    rid = first_recommended_rule_id(cli_env)
    registry = cli_env.research / "seal-registry.json"
    real_load = pipeline.load_panel
    seen = []

    def spy(market, symbols, start, end):
        seen.append(registry.exists() and rid in registry.read_text(encoding="utf-8"))
        return real_load(market, symbols, start, end)
    monkeypatch.setattr(pipeline, "load_panel", spy)
    # skip the slow Nautilus part: only the access order matters here
    monkeypatch.setattr(pipeline, "_final_scenarios",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.verify_final("swing", "kr", rid)
    assert seen and all(seen)


def test_verify_unknown_rule_does_not_open_seal(cli_env):
    r = CliRunner().invoke(app, ["verify", "--final", "--pocket", "swing", "--market", "kr",
                                 "--rule-id", "breakout:deadbeef"])
    assert r.exit_code != 0 and "Traceback" not in r.output
    assert not (cli_env.research / "seal-registry.json").exists()


def test_missing_sync_is_one_line_error(cli_env, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(cli_env.root / "nope.yaml"))
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "breakout"])
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit)
    assert "lab sync" in r.output and len(r.output.strip().splitlines()) == 1


def test_missing_data_is_one_line_error(cli_env, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(cli_env.root / "empty"))
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "all"])
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit)
    assert "lab collect" in r.output


def test_universe_lists_eligible_symbols(cli_env):
    r = run(["universe", "--market", "kr", "--pocket", "swing", "--scenario", "paper",
             "--date", "2017-03-01"])
    listed = r.output.split()
    assert "069500" in listed                       # always_include
    assert set(listed) & set(SYMBOLS[1:])
    r = CliRunner().invoke(app, ["universe", "--market", "kr", "--pocket", "swing",
                                 "--scenario", "paper", "--date", "2017-03-04"])  # Saturday
    assert r.exit_code == 1 and "trading day" in r.output


def test_journal_appends_with_time(cli_env):
    run(["journal", "첫 메모"])
    run(["journal", "두 번째"])
    files = list((cli_env.research / "journal").glob("*.md"))
    assert len(files) == 1 and files[0].name == f"{date.today().isoformat()}.md"
    text = files[0].read_text(encoding="utf-8")
    assert text.index("첫 메모") < text.index("두 번째")
    assert text.count(":") >= 4                     # HH:MM:SS per entry


class FakeClient:
    def __init__(self, commissions=None):
        self._comm = commissions

    def get_exchange_rate(self):
        return {"baseCurrency": "USD", "quoteCurrency": "KRW", "rate": "1380.5",
                "midRate": "1375"}

    def get_commissions(self):
        if self._comm is None:
            raise ValueError("commissions endpoint needs an account header: set "
                             "TOSS_ACCOUNT_SEQ")
        return self._comm


def test_sync_manual_commissions_without_account(cli_env, monkeypatch):
    out = cli_env.root / "new-synced.yaml"
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(out))
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: FakeClient(None))
    r = CliRunner().invoke(app, ["sync"])
    assert r.exit_code == 1 and "--kr-commission" in r.output and not out.exists()
    run(["sync", "--kr-commission", "0.00015", "--us-commission", "0.001"])
    d = yaml.safe_load(out.read_text())
    assert d["commission_rate"] == {"kr": 0.00015, "us": 0.001}
    assert d["usd_krw"] == 1380.5 and d["synced_at"]
    assert d["source"] == {"commission_rate": {"kr": "manual", "us": "manual"},
                           "usd_krw": "api"}
    r = CliRunner().invoke(app, ["sync", "--kr-commission", "0.015", "--us-commission",
                                 "0.001"])                    # 1.5%: not a decimal ratio
    assert r.exit_code == 1


def test_sync_commissions_from_api(cli_env, monkeypatch):
    out = cli_env.root / "api-synced.yaml"
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(out))
    comm = [{"marketCountry": "KR", "commissionRate": "0.00015", "startDate": "2020-01-01",
             "endDate": None},
            {"marketCountry": "US", "commissionRate": "0.009", "startDate": None,
             "endDate": "2020-12-31"},                       # expired
            {"marketCountry": "US", "commissionRate": "0.001", "startDate": "2021-01-01",
             "endDate": None}]
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: FakeClient(comm))
    run(["sync"])
    d = yaml.safe_load(out.read_text())
    assert d["commission_rate"] == {"kr": 0.00015, "us": 0.001}
    assert d["source"]["commission_rate"] == {"kr": "api", "us": "api"}


def test_collect_wiring_limit_and_quality_summary(cli_env, monkeypatch):
    meta = pd.DataFrame({"name": ["X", "Y", "Z", "KODEX 200", "D"],
                         "delisted": [False, False, False, False, True]},
                        index=pd.Index(["000010", "000020", "000030", "069500", "000040"],
                                       name="symbol"))
    calls = {"daily": [], "supplement": None}
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: object())
    monkeypatch.setattr(pipeline, "collect_meta", lambda client, market: meta)

    def fake_daily(client, market, symbol, start):
        calls["daily"].append(symbol)
        df = pd.DataFrame({"open": [10.0, 10.0], "high": [11.0, 9.0], "low": [9.0, 9.5],
                           "close": [10.0, 10.0], "volume": [1.0, 1.0]},
                          index=pd.to_datetime(["2016-01-04", "2016-01-05"]))
        if symbol == "000020":
            return 0                                  # nothing stored
        store.save_daily(market, symbol, df.iloc[:1] if symbol == "069500" else df)
        return 2

    def fake_supplement(start, symbols=None):
        calls["supplement"] = symbols
        from autotrade.data.delisted_kr import SupplementReport
        return SupplementReport(fetched=0, empty=list(symbols or []))
    monkeypatch.setattr(pipeline, "collect_daily", fake_daily)
    monkeypatch.setattr(pipeline, "supplement_delisted_kr", fake_supplement)
    r = run(["collect", "--market", "kr", "--limit", "3", "--since", "2016-01-01"])
    assert calls["daily"] == ["069500", "000010", "000020"]   # always_include first
    assert calls["supplement"] in (None, [])                  # delisted not within limit
    assert "clean 1" in r.output and "issues 1" in r.output and "missing 1" in r.output


def test_verify_scenarios_wiring_us(cli_env, monkeypatch):
    """US: paper 5,000,000 KRW non-fractional, real 50,000 KRW fractional, both converted
    with 1/usd_krw; Nautilus and run_portfolio get the same inputs and the rule's timing."""
    us_syms = ["SPY", "QQQ", "U1", "U2", "U3", "U4"]
    p = make_panel(N_DAYS, us_syms, seed=5, products={"SPY": "etf", "QQQ": "etf"})
    for sym in us_syms:
        df = pd.DataFrame({"open": p.open[sym], "high": p.high[sym], "low": p.low[sym],
                           "close": p.close[sym], "volume": p.volume[sym]})
        store.save_daily("us", sym, df)
    store.save_meta("us", p.meta)
    run(["screen", "--pocket", "swing", "--market", "us", "--family", "breakout"])
    run(["validate", "--pocket", "swing", "--market", "us", "--top", "1"])
    rid = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "us").query(
        "stage == 'validate'")["rule_id"].iloc[0]
    naut_calls, port_calls = [], []
    real_portfolio = pipeline.run_portfolio

    def fake_nautilus(rule, panel, elig, start, end, cm, pocket, capital, cpk, fractional):
        naut_calls.append((start, end, capital, cpk, fractional, panel.close.index.max()))
        return pd.DataFrame(columns=["symbol", "entry_date", "entry_price", "exit_date",
                                     "exit_price", "exit_reason", "stop_pct", "confidence",
                                     "qty", "net_pnl"])

    def spy_portfolio(cands, panel, cm, pocket, capital, cpk, fractional, **kw):
        port_calls.append((capital, cpk, fractional, kw))
        return real_portfolio(cands, panel, cm, pocket, capital, cpk, fractional, **kw)
    monkeypatch.setattr(pipeline, "run_nautilus", fake_nautilus)
    monkeypatch.setattr(pipeline, "run_portfolio", spy_portfolio)
    card, _, _ = pipeline.verify_final("swing", "us", rid)

    per = periods_of(cli_env)
    cpk = 1 / 1380.0
    assert [c[:5] for c in naut_calls] == [
        (per.sealed[0], per.sealed[1], 5_000_000, cpk, False),
        (per.sealed[0], per.sealed[1], 50_000, cpk, True)]
    assert all(c[5] == pd.Timestamp(per.sealed[1]) for c in naut_calls)  # warm-up history
    timing = {"entry_at": "next_open", "exit_at": "next_open"}           # breakout
    assert [c[:3] for c in port_calls] == [(5_000_000, cpk, False)] * 2 + \
        [(50_000, cpk, True)] * 2                     # sealed (parity) + valid per scenario
    assert all(c[3] == timing for c in port_calls)
    assert card.scenarios["real"]["fractional"] is True
    # only one validate row -> sr_var_trials falls back to the squared sealed Sharpe
    assert card.scenarios["paper"]["sr_var_trials_source"] == "sealed_sr_squared"


def test_nautilus_result_equity_matches_portfolio():
    from autotrade.config import load_market_raw, load_pocket
    from autotrade.costs import CostModel
    from autotrade.rules.registry import SWING_FAMILIES
    from autotrade.screen.simulate import simulate
    from autotrade.sizing import run_portfolio
    from autotrade.verify import run_nautilus
    from dataclasses import replace
    import numpy as np

    p = make_panel(300, ["A", "B", "C"], seed=7)
    cm = CostModel(replace(load_market_raw("kr"), commission_rate=0.00015), lambda s: "stock")
    swing = load_pocket("swing")
    rule = SWING_FAMILIES["breakout"](SWING_FAMILIES["breakout"].grid()[0])
    elig = pd.DataFrame(True, index=p.close.index, columns=p.close.columns)
    start, end = date(2016, 3, 1), date(2017, 1, 31)
    hist = p.slice(p.close.index[0], end)
    naut = run_nautilus(rule, p, elig, start, end, cm, swing, 5_000_000, 1.0, False)
    ours = run_portfolio(simulate(hist, rule.signals(hist), elig, start, end), hist, cm, swing,
                         5_000_000, 1.0, False, entry_at="next_open", exit_at="next_open")
    res = pipeline.nautilus_result(naut, hist, cm, 5_000_000, start, end)
    assert len(res.fills) == len(naut) > 0
    np.testing.assert_allclose(res.fills["net_r"], ours.fills["net_r"], rtol=1e-3)
    assert res.equity.iloc[-1] == pytest.approx(5_000_000 + naut["net_pnl"].sum())
    common = ours.equity.index
    diff = (res.equity.reindex(common) - ours.equity).abs()
    assert diff.max() <= len(naut)          # Money rounding: < 1 won per fill


def test_validate_ignores_screens_of_another_train_period(cli_env):
    log = ExperimentLog(cli_env.research / "experiments.jsonl")
    log.record("swing", "kr", "breakout", "breakout:00000000", {"n": 20, "m": 5, "k": 1.5},
               "screen", ("2016-01-04", "2017-01-02"), {"n_trades": 99, "mean_net_r": 1.0})
    r = CliRunner().invoke(app, ["validate", "--pocket", "swing", "--market", "kr"])
    assert r.exit_code == 1 and "lab screen" in r.output


def test_sync_missing_credentials_is_one_line_error(cli_env, monkeypatch):
    def missing(path):
        raise RuntimeError("missing environment variable TOSS_CLIENT_ID")
    monkeypatch.setattr(pipeline, "make_client_from_env", missing)
    r = CliRunner().invoke(app, ["sync", "--kr-commission", "0.00015", "--us-commission",
                                 "0.001"])
    assert r.exit_code == 1 and "TOSS_CLIENT_ID" in r.output
    assert len(r.output.strip().splitlines()) == 1
