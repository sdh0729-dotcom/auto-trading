import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
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
    rec = val.iloc[-1]["metrics"]["recommended_ids"]          # the latest validate run (P3)
    assert rec, "validate recommended no rule on the synthetic panel"
    return rec[0]


def extend_store(n: int, market: str = "kr") -> pd.DatetimeIndex:
    """A later `lab collect`: append n new business days (last bar repeated) to every symbol."""
    for f in sorted((store.data_root() / market / "1d").glob("*.parquet")):
        df = store.load_daily(market, f.stem)
        new_idx = pd.bdate_range(df.index[-1], periods=n + 1)[1:]
        more = pd.DataFrame([df.iloc[-1].to_dict()] * n, index=new_idx)
        store.save_daily(market, f.stem, pd.concat([df, more]))
    return new_idx


def periods_file(env: CliEnv, pocket="swing", market="kr") -> Path:
    return env.research / f"periods-{pocket}-{market}.json"


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
    assert card["scenarios"]["paper"]["capital_krw"] == 5_000_000
    assert card["scenarios"]["real"]["capital_krw"] == 50_000
    assert card["scenarios"]["real"]["capital_ccy"] == 50_000
    assert card["scenarios"]["real"]["currency"] == "KRW" and card["currency"] == "KRW"
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
    r = run(["validate", "--pocket", "swing", "--market", "kr", "--top", "3"])
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    val = rows[rows["stage"] == "validate"]
    scr = rows[rows["stage"] == "screen"]
    pool = {rid for rid, m in zip(scr["rule_id"], scr["metrics"]) if m["n_trades"] >= 30}
    assert set(val["rule_id"]) == pool and len(pool) > 3      # whole pool is logged
    assert all(m["pbo_candidates"] == len(pool) for m in val["metrics"])
    assert f"over {len(pool)} candidates" in r.output
    assert "검증 기간 구간 안정성" in r.output
    per = periods_of(cli_env)
    for period, m in zip(val["period"], val["metrics"]):
        assert period == [per.valid[0].isoformat(), per.valid[1].isoformat()]
        segs = m["valid_segments"]                  # 4 contiguous segments inside valid
        assert len(segs) == len(m["valid_segment_net_sums"]) == 4
        assert segs[0][0] == per.valid[0].isoformat() and segs[-1][1] == per.valid[1].isoformat()
        assert all(date.fromisoformat(b) < date.fromisoformat(c)
                   for (_, b), (c, _) in zip(segs, segs[1:]))
        assert m["valid_segments_positive"] == sum(x > 0 for x in m["valid_segment_net_sums"])
        assert not any(k.startswith("wf_") for k in m)
        assert {"valid_net_pnl", "valid_daily_sharpe", "valid_n_trades", "pbo",
                "recommended"} <= m.keys()
        assert 0.0 <= m["pbo"] <= 1.0
    assert len({m["pbo"] for m in val["metrics"]}) == 1         # one PBO for the run
    rec = [m for m in val["metrics"] if m["recommended"]]
    assert len(rec) <= 3 and all(m["valid_net_pnl"] > 0 for m in rec)
    # --top picks the candidates printed / recommended: top screened by mean_net_r with
    # >= 30 train trades
    ok = [(m["mean_net_r"], r) for r, m in zip(scr["rule_id"], scr["metrics"])
          if m["n_trades"] >= 30]
    top3 = [r for _, r in sorted(ok, reverse=True)[:3]]
    assert {rid for rid, m in zip(val["rule_id"], val["metrics"]) if m["in_top"]} == set(top3)
    assert {rid for rid, m in zip(val["rule_id"], val["metrics"]) if m["recommended"]} \
        <= set(top3)
    shown = [line.split()[0] for line in r.output.splitlines()[1:4]]
    assert set(shown) == set(top3)


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


def test_collect_isolates_failing_symbols(cli_env, monkeypatch):
    import httpx
    from autotrade.toss.client import TossApiError
    meta = pd.DataFrame({"name": ["X", "Y", "Z"], "delisted": [False] * 3},
                        index=pd.Index(["000010", "000020", "000030"], name="symbol"))
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: object())
    monkeypatch.setattr(pipeline, "collect_meta", lambda client, market: meta)
    done = []

    def fake_daily(client, market, symbol, start):
        if symbol == "000010":
            raise TossApiError(500, "internal", "r9")
        if symbol == "000020":
            raise httpx.ReadTimeout("slow")
        df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
                           "volume": 1.0}, index=pd.to_datetime(["2016-01-04"]))
        store.save_daily(market, symbol, df)
        done.append(symbol)
        return 1
    monkeypatch.setattr(pipeline, "collect_daily", fake_daily)
    monkeypatch.setattr(pipeline, "supplement_delisted_kr", lambda start, symbols=None: None)
    s = pipeline.collect("kr", date(2016, 1, 1))
    assert done == ["000030"]
    assert [f.split(":")[0] for f in s["failed"]] == ["000010", "000020"]
    assert "TossApiError" in s["failed"][0] and "ReadTimeout" in s["failed"][1]
    r = run(["collect", "--market", "kr", "--since", "2016-01-01"])
    assert "failed 2 (000010, 000020)" in r.output


def test_collect_reports_dropped_calendar_dates(cli_env, monkeypatch):
    meta = pd.DataFrame({"name": ["X"], "delisted": [False]},
                        index=pd.Index(["000010"], name="symbol"))
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: object())
    monkeypatch.setattr(pipeline, "collect_meta", lambda client, market: meta)

    def fake_daily(client, market, symbol, start):
        df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
                           "volume": 1.0}, index=pd.to_datetime(["2016-03-05"]))  # Saturday
        store.save_daily(market, symbol, df)
        return 1
    monkeypatch.setattr(pipeline, "collect_daily", fake_daily)
    monkeypatch.setattr(pipeline, "supplement_delisted_kr", lambda start, symbols=None: None)
    r = run(["collect", "--market", "kr", "--since", "2016-01-01"])
    assert "calendar dropped 1 (2016-03-05)" in r.output


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
    rid = pipeline.validate("swing", "us", 3).recommended[0]   # P3: only recommended rules
    naut_calls, port_calls = [], []
    real_portfolio = pipeline.run_portfolio

    def fake_nautilus(rule, panel, elig, start, end, cm, pocket, capital, cpk, fractional):
        naut_calls.append((start, end, capital, cpk, fractional, panel.close.index.max()))
        d = panel.close.index[panel.close.index >= pd.Timestamp(start)]
        px = float(panel.close.at[d[1], "U1"])
        return pd.DataFrame([{"symbol": "U1", "entry_date": d[1], "entry_price": px,
                              "exit_date": d[4], "exit_price": px, "exit_reason": "data_end",
                              "stop_pct": 0.05, "confidence": 1.0, "qty": 1.0,
                              "net_pnl": -0.01}])

    def spy_portfolio(cands, panel, cm, pocket, capital, cpk, fractional, **kw):
        port_calls.append((capital, cpk, fractional, kw))
        return real_portfolio(cands, panel, cm, pocket, capital, cpk, fractional, **kw)
    monkeypatch.setattr(pipeline, "run_nautilus", fake_nautilus)
    monkeypatch.setattr(pipeline, "run_portfolio", spy_portfolio)
    card, _md, _ = pipeline.verify_final("swing", "us", rid)

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
    assert card.scenarios["paper"]["n_data_end"] == 1           # data_end exits counted
    # I8: KRW capital and its USD conversion are labelled; P&L is in USD
    assert card.scenarios["real"]["capital_krw"] == 50_000
    assert card.scenarios["real"]["capital_ccy"] == pytest.approx(50_000 / 1380.0)
    assert card.scenarios["real"]["currency"] == "USD" and card.currency == "USD"
    assert "capital" not in card.scenarios["real"]
    # I7: the US real scenario is reference only until plan 2
    assert set(card.verdict) == {"② 모의 자격", "참고용(계획 2에서 확정)"}
    text = Path(_md).read_text(encoding="utf-8")
    assert "실돈 자격" not in text and "sealed_net_pnl (USD)" in text
    # the whole screened pool was validated -> trial variance from its validate Sharpes
    assert card.scenarios["paper"]["sr_var_trials_source"] == "validate_daily_sharpes"


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


def test_periods_fixed_on_first_screen_and_stable_across_collect(cli_env):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    rec = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    per = periods_of(cli_env)
    assert rec["sealed"] == [per.sealed[0].isoformat(), per.sealed[1].isoformat()]
    assert rec["data_cutoff"] == per.sealed[1].isoformat()
    extend_store(100)                                   # new data after the first screen
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "2"])
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    val = rows[rows["stage"] == "validate"]
    assert all(p == [per.valid[0].isoformat(), per.valid[1].isoformat()] for p in val["period"])
    after = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    assert {k: after[k] for k in ("train", "valid", "sealed", "data_cutoff")} == \
        {k: rec[k] for k in ("train", "valid", "sealed", "data_cutoff")}
    r = run(["periods", "--pocket", "swing", "--market", "kr"])
    assert per.sealed[0].isoformat() in r.output and "data cutoff" in r.output


def test_validate_and_verify_need_periods_from_screen(cli_env):
    r = CliRunner().invoke(app, ["validate", "--pocket", "swing", "--market", "kr"])
    assert r.exit_code == 1 and "lab screen" in r.output
    assert not periods_file(cli_env).exists()


def test_screen_and_validate_refuse_ranges_overlapping_an_opened_seal(cli_env):
    from autotrade.validate.seal import SealRegistry
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    per = periods_of(cli_env)
    SealRegistry(cli_env.research / "seal-registry.json").open(
        "swing", "kr", "breakout:x", sealed=(per.sealed[0].isoformat(),
                                             per.sealed[1].isoformat()))
    # a lost periods file is rebuilt from the tickets (P1); a file re-split on grown data
    # (e.g. replaced by hand) is what can still overlap the opened seal
    def resplit():
        from autotrade.validate import periods as period_store
        dates = store.trading_dates("kr", date(2016, 1, 1), date(2100, 1, 1))
        period_store.save_record(periods_file(cli_env),
                                 period_store.new_record("swing", "kr", dates, (0.6, 0.2, 0.2)))
    extend_store(100)
    resplit()
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    r = CliRunner().invoke(app, ["validate", "--pocket", "swing", "--market", "kr"])
    assert r.exit_code == 1 and "opened" in r.output and "Traceback" not in r.output
    extend_store(300)                                    # now even the train range overlaps
    resplit()
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "breakout"])
    assert r.exit_code == 1 and "opened" in r.output


def test_periods_new_cycle_preconditions(cli_env):
    from autotrade.validate.seal import SealRegistry
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    per = periods_of(cli_env)
    r = CliRunner().invoke(app, ["periods", "--pocket", "swing", "--market", "kr",
                                 "--new-cycle"])
    assert r.exit_code == 1 and "no final candidate" in r.output
    SealRegistry(cli_env.research / "seal-registry.json").open(
        "swing", "kr", "breakout:x", sealed=(per.sealed[0].isoformat(),
                                             per.sealed[1].isoformat()), cycle=1)
    extend_store(62)
    r = CliRunner().invoke(app, ["periods", "--pocket", "swing", "--market", "kr",
                                 "--new-cycle"])
    assert r.exit_code == 1 and "63" in r.output
    new_days = extend_store(1)
    run(["periods", "--pocket", "swing", "--market", "kr", "--new-cycle"])
    rec = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    assert rec["cycle"] == 2 and rec["data_cutoff"] == new_days[-1].date().isoformat()
    assert rec["train"] == [per.train[0].isoformat(), per.train[1].isoformat()]
    assert rec["valid"] == [per.valid[0].isoformat(), per.valid[1].isoformat()]
    first_new = pd.bdate_range(per.sealed[1], periods=2)[1]
    assert rec["sealed"] == [first_new.date().isoformat(), new_days[-1].date().isoformat()]


def test_limited_collect_blocks_research_until_a_full_collect(cli_env, monkeypatch):
    meta = pd.DataFrame({"name": ["X", "Y"], "delisted": [False, False]},
                        index=pd.Index(["A01", "A02"], name="symbol"))
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: object())
    monkeypatch.setattr(pipeline, "collect_meta", lambda client, market: meta)
    monkeypatch.setattr(pipeline, "collect_daily", lambda client, market, symbol, start: 0)
    monkeypatch.setattr(pipeline, "supplement_delisted_kr", lambda start, symbols=None: None)
    marker = cli_env.data / "kr" / ".limited"
    run(["collect", "--market", "kr", "--limit", "1"])
    assert marker.exists()
    for args in (["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"],
                 ["validate", "--pocket", "swing", "--market", "kr"],
                 ["verify", "--final", "--pocket", "swing", "--market", "kr", "--rule-id", "x"]):
        r = CliRunner().invoke(app, args)
        assert r.exit_code == 1 and "--limit" in r.output and "Traceback" not in r.output
    assert not (cli_env.research / "seal-registry.json").exists()
    assert not periods_file(cli_env).exists()
    run(["collect", "--market", "kr"])                   # full collect clears the marker
    assert not marker.exists()
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])


def test_pbo_and_trial_variance_pool_does_not_depend_on_top(cli_env):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    log = ExperimentLog(cli_env.research / "experiments.jsonl")
    valid = list(pipeline._iso(periods_of(cli_env).valid))
    a = pipeline.validate("swing", "kr", 1)
    var_a = pipeline._sr_var_trials(log.rows("swing", "kr"), valid)
    b = pipeline.validate("swing", "kr", 6)
    var_b = pipeline._sr_var_trials(log.rows("swing", "kr"), valid)
    assert a.pbo == b.pbo and len(a.table) == 1 and len(b.table) == 6
    assert var_a is not None and var_a == pytest.approx(var_b)
    rec = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    assert rec["validate_runs"] == 2


def test_sr_var_trials_uses_current_valid_period_only(cli_env):
    log = ExperimentLog(cli_env.research / "experiments.jsonl")
    for rid, sr, period in (("a", 0.1, ("2020-01-01", "2020-12-31")),
                            ("b", 0.3, ("2020-01-01", "2020-12-31")),
                            ("c", 9.0, ("2019-01-01", "2019-12-31"))):   # older periods
        log.record("swing", "kr", "breakout", rid, {}, "validate", period,
                   {"valid_daily_sharpe": sr})
    rows = log.rows("swing", "kr")
    assert pipeline._sr_var_trials(rows, ["2020-01-01", "2020-12-31"]) == \
        pytest.approx(np.var([0.1, 0.3], ddof=1))


def test_card_records_validate_runs_and_flags_reruns(cli_env, monkeypatch):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "3"])
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "5"])
    rid = first_recommended_rule_id(cli_env)
    monkeypatch.setattr(pipeline, "run_nautilus", lambda *a, **k: pd.DataFrame(columns=[
        "symbol", "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason",
        "stop_pct", "confidence", "qty", "net_pnl"]))
    card, md, js = pipeline.verify_final("swing", "kr", rid)
    assert card.validate_runs == 2
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["validate_runs"] == 2
    text = md.read_text(encoding="utf-8")
    assert "검증 실행 2회" in text
    assert "검증 기간 구간 안정성" in text                      # I6 label on the card


def test_pipeline_passes_the_scenario_mask_to_signals(cli_env, monkeypatch):
    from autotrade.rules.swing.breakout import Breakout
    calls = []
    real = Breakout.signals

    def spy(self, panel, eligible=None):
        calls.append(None if eligible is None else int(eligible.to_numpy().sum()))
        return real(self, panel, eligible=eligible)
    monkeypatch.setattr(Breakout, "signals", spy)
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    run(["validate", "--pocket", "swing", "--market", "kr", "--top", "1"])
    rid = first_recommended_rule_id(cli_env)
    monkeypatch.setattr(pipeline, "run_nautilus", lambda *a, **k: pd.DataFrame(columns=[
        "symbol", "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason",
        "stop_pct", "confidence", "qty", "net_pnl"]))
    n_before = len(calls)
    pipeline.verify_final("swing", "kr", rid)
    assert calls and None not in calls
    final = calls[n_before:]                     # paper and real masks differ in size
    assert len(final) == 4 and final[0] != final[2]


def test_universe_refuses_sealed_dates_once_periods_exist(cli_env):
    per = periods_of(cli_env)
    day = per.sealed[0].isoformat()
    run(["universe", "--market", "kr", "--date", day])          # no periods yet: allowed
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    r = CliRunner().invoke(app, ["universe", "--market", "kr", "--date", day])
    assert r.exit_code == 1 and "sealed" in r.output and "Traceback" not in r.output
    run(["universe", "--market", "kr", "--date", per.valid[1].isoformat()])


def test_sync_rejects_percent_like_api_commission(cli_env, monkeypatch):
    out = cli_env.root / "api-synced.yaml"
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(out))
    comm = [{"marketCountry": "KR", "commissionRate": "0.015", "startDate": None,
             "endDate": None},                                  # 1.5%: a percent, not a ratio
            {"marketCountry": "US", "commissionRate": "0.001", "startDate": None,
             "endDate": None}]
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: FakeClient(comm))
    r = CliRunner().invoke(app, ["sync"])
    assert r.exit_code == 1 and "0.015" in r.output and not out.exists()


def test_sync_stores_promo_end_and_warns_when_past(cli_env, monkeypatch):
    out = cli_env.root / "api-synced.yaml"
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(out))
    comm = [{"marketCountry": "KR", "commissionRate": "0.00015", "startDate": "2020-01-01",
             "endDate": "2026-12-31"},
            {"marketCountry": "US", "commissionRate": "0.001", "startDate": None,
             "endDate": None}]
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: FakeClient(comm))
    pipeline.sync(today=date(2026, 11, 20))
    d = yaml.safe_load(out.read_text())
    assert d["commission_end_date"] == {"kr": "2026-12-31", "us": None}
    assert pipeline.commission_warnings(date(2026, 12, 31)) == []
    w = pipeline.commission_warnings(date(2027, 1, 1))
    assert len(w) == 1 and "kr" in w[0] and "2026-12-31" in w[0] and "lab sync" in w[0]
    monkeypatch.setattr(pipeline, "commission_warnings", lambda today=None: w)
    r = run(["journal", "x"])
    assert "2026-12-31" in r.output                       # printed before any command runs


# ---------------------------------------------------------------- P1: periods file safety
def test_refused_screen_writes_no_periods_file(cli_env, monkeypatch):
    from autotrade.validate.seal import SealViolation

    def refuse(*a, **k):
        raise SealViolation("screen range overlaps the sealed period already opened")
    monkeypatch.setattr(pipeline, "_refuse_opened_overlap", refuse)
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "breakout"])
    assert r.exit_code == 1 and "opened" in r.output
    assert not periods_file(cli_env).exists()
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "nope"])
    assert r.exit_code == 1 and not periods_file(cli_env).exists()


def _open_ticket(env, rule_id, sealed, cutoff, cycle=1):
    from autotrade.validate.seal import SealRegistry
    return SealRegistry(env.research / "seal-registry.json").open(
        "swing", "kr", rule_id, sealed=(sealed[0].isoformat(), sealed[1].isoformat()),
        data_cutoff=cutoff.isoformat(), n_symbols=len(SYMBOLS), cycle=cycle)


def _split_keys(rec: dict) -> dict:
    return {k: rec[k] for k in ("train", "valid", "sealed", "data_cutoff", "cycle", "history")}


def test_lost_periods_file_is_rebuilt_from_seal_tickets(cli_env):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    before = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    per = periods_of(cli_env)
    _open_ticket(cli_env, "breakout:x", per.sealed, per.sealed[1])
    log = ExperimentLog(cli_env.research / "experiments.jsonl")
    log.record("swing", "kr", "breakout", "breakout:x", {}, "validate", pipeline._iso(per.valid),
               {"valid_daily_sharpe": 0.1, "validate_run": 2})
    periods_file(cli_env).unlink()
    extend_store(300)                 # a fresh split would now put train over the opened seal
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    after = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    assert _split_keys(after) == _split_keys(before)
    assert after["validate_runs"] == 2                # recovered from the experiment log


def test_lost_periods_file_rebuild_refuses_inconsistent_tickets(cli_env):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    per = periods_of(cli_env)
    later = pd.bdate_range(per.sealed[0], periods=5)[-1].date()
    _open_ticket(cli_env, "breakout:x", (later, per.sealed[1]), per.sealed[1])  # not the split
    periods_file(cli_env).unlink()
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "breakout"])
    assert r.exit_code == 1 and "Traceback" not in r.output
    assert "seal" in r.output and "periods" in r.output
    assert not periods_file(cli_env).exists()


def test_lost_periods_file_without_cutoff_in_tickets_is_refused(cli_env):
    from autotrade.validate.seal import SealRegistry
    per = periods_of(cli_env)
    SealRegistry(cli_env.research / "seal-registry.json").open(
        "swing", "kr", "breakout:x", sealed=pipeline._iso(per.sealed))   # no data_cutoff
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "breakout"])
    assert r.exit_code == 1 and "data cutoff" in r.output
    assert not periods_file(cli_env).exists()


def test_lost_periods_file_is_rebuilt_in_a_later_cycle(cli_env):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    per = periods_of(cli_env)
    _open_ticket(cli_env, "breakout:x", per.sealed, per.sealed[1])
    extend_store(70)
    run(["periods", "--pocket", "swing", "--market", "kr", "--new-cycle"])
    ExperimentLog(cli_env.research / "experiments.jsonl").record(     # rebuild checks it
        "swing", "kr", "breakout", "breakout:x", {}, "validate", pipeline._iso(per.valid),
        {"valid_daily_sharpe": 0.1, "validate_run": 1})
    before = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    sealed2 = tuple(date.fromisoformat(x) for x in before["sealed"])
    _open_ticket(cli_env, "breakout:x", sealed2, sealed2[1], cycle=2)
    periods_file(cli_env).unlink()
    extend_store(10)
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    after = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    assert after["cycle"] == 2 and _split_keys(after) == _split_keys(before)


# ---------------------------------------------------------------- P2: systemic collect failures
def _collect_env(monkeypatch, n: int, fail):
    """`n` symbols; fake_daily raises fail(symbol) when it returns an exception."""
    syms = [f"{i:06d}" for i in range(1, n + 1)]
    meta = pd.DataFrame({"name": syms, "delisted": [False] * n},
                        index=pd.Index(syms, name="symbol"))
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: object())
    monkeypatch.setattr(pipeline, "collect_meta", lambda client, market: meta)
    monkeypatch.setattr(pipeline, "supplement_delisted_kr", lambda start, symbols=None: None)
    calls = []

    def fake_daily(client, market, symbol, start):
        calls.append(symbol)
        e = fail(symbol)
        if e is not None:
            raise e
        return 0
    monkeypatch.setattr(pipeline, "collect_daily", fake_daily)
    return calls


@pytest.mark.parametrize("status,code,word", [(401, "invalid-token", "credentials"),
                                              (403, "ip-not-allowed", "ip-not-allowed")])
def test_collect_aborts_on_auth_or_forbidden(cli_env, monkeypatch, status, code, word):
    from autotrade.toss.client import TossApiError
    calls = _collect_env(monkeypatch, 5, lambda s: TossApiError(status, code, "rid1")
                         if s == "000002" else None)
    r = CliRunner().invoke(app, ["collect", "--market", "kr", "--since", "2016-01-01"])
    assert r.exit_code == 1 and "Traceback" not in r.output
    assert len(r.output.strip().splitlines()) == 1
    assert "aborted" in r.output and str(status) in r.output and word in r.output
    assert calls == ["000001", "000002"]                  # nothing after the systemic failure


def test_collect_aborts_after_consecutive_symbol_failures(cli_env, monkeypatch):
    import httpx
    calls = _collect_env(monkeypatch, 30, lambda s: httpx.ConnectError("offline"))
    with pytest.raises(pipeline.LabError, match="20 consecutive"):
        pipeline.collect("kr", date(2016, 1, 1))
    assert len(calls) == pipeline.MAX_CONSECUTIVE_FAILURES == 20


def test_collect_keeps_isolating_non_consecutive_failures(cli_env, monkeypatch):
    from autotrade.toss.client import TossApiError
    calls = _collect_env(monkeypatch, 60, lambda s: TossApiError(500, "internal", "r")
                         if int(s) % 10 else None)          # 9 failures, then one success
    s = pipeline.collect("kr", date(2016, 1, 1))
    assert len(calls) == 60 and len(s["failed"]) == 54


# ---------------------------------------------------------------- P3: verify only recommended
def _stop_final(monkeypatch):
    monkeypatch.setattr(pipeline, "_final_scenarios",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))


def test_validate_logs_the_run_recommendations(cli_env):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    res = pipeline.validate("swing", "kr", 3)
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    val = rows[rows["stage"] == "validate"]
    assert all(m["recommended_ids"] == res.recommended for m in val["metrics"])


def test_verify_refuses_a_rule_not_recommended_by_the_latest_validate(cli_env, monkeypatch):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    res = pipeline.validate("swing", "kr", 3)
    assert res.recommended
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    pool = list(rows[rows["stage"] == "validate"]["rule_id"])
    other = next(r for r in pool if r not in res.recommended)
    registry = cli_env.research / "seal-registry.json"
    _stop_final(monkeypatch)
    with pytest.raises(pipeline.LabError, match="recommended"):
        pipeline.verify_final("swing", "kr", other)          # validated, not recommended
    assert not registry.exists()
    # a later validate run recommends nothing: the earlier recommendation no longer counts
    log = ExperimentLog(cli_env.research / "experiments.jsonl")
    log.record("swing", "kr", "breakout", other, {}, "validate",
               pipeline._iso(periods_of(cli_env).valid),
               {"recommended": False, "recommended_ids": [], "validate_run": 2})
    with pytest.raises(pipeline.LabError, match="recommended"):
        pipeline.verify_final("swing", "kr", res.recommended[0])
    assert not registry.exists()


def test_verify_cycle_two_keeps_the_previous_finalist_rule(cli_env, monkeypatch):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    rid = pipeline.validate("swing", "kr", 3).recommended[0]
    monkeypatch.setattr(pipeline, "run_nautilus", lambda *a, **k: pd.DataFrame(columns=[
        "symbol", "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason",
        "stop_pct", "confidence", "qty", "net_pnl"]))
    pipeline.verify_final("swing", "kr", rid)
    extend_store(63)
    run(["periods", "--pocket", "swing", "--market", "kr", "--new-cycle"])
    log = ExperimentLog(cli_env.research / "experiments.jsonl")
    log.record("swing", "kr", "breakout", rid, {}, "validate",
               pipeline._iso(periods_of(cli_env).valid),
               {"recommended": False, "recommended_ids": [], "validate_run": 2})
    _stop_final(monkeypatch)
    with pytest.raises(RuntimeError, match="stop"):      # got past the checks: seal opened
        pipeline.verify_final("swing", "kr", rid)
    from autotrade.validate.seal import SealRegistry
    assert [t.cycle for t in SealRegistry(cli_env.research / "seal-registry.json")
            .tickets("swing", "kr")] == [1, 2]


# ---------------------------------------------------------------- P4: cards per cycle
def test_cycle_two_verify_keeps_the_cycle_one_card(cli_env, monkeypatch):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    rid = pipeline.validate("swing", "kr", 3).recommended[0]
    monkeypatch.setattr(pipeline, "run_nautilus", lambda *a, **k: pd.DataFrame(columns=[
        "symbol", "entry_date", "entry_price", "exit_date", "exit_price", "exit_reason",
        "stop_pct", "confidence", "qty", "net_pnl"]))
    _, md1, js1 = pipeline.verify_final("swing", "kr", rid)
    first = js1.read_text(encoding="utf-8")
    extend_store(63)
    run(["periods", "--pocket", "swing", "--market", "kr", "--new-cycle"])
    _, md2, js2 = pipeline.verify_final("swing", "kr", rid)
    stem = rid.replace(":", "-")
    assert (md1.name, md2.name) == (f"swing-kr-c1-{stem}.md", f"swing-kr-c2-{stem}.md")
    assert js1.read_text(encoding="utf-8") == first
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    assert [m["cycle"] for m in rows[rows["stage"] == "final"]["metrics"]] == [1, 2]
    board = pipeline.report("swing", "kr").read_text(encoding="utf-8")
    assert f"rule-cards/swing-kr-c1-{stem}.md" in board
    assert f"rule-cards/swing-kr-c2-{stem}.md" in board


# ---------------------------------------------------------------- P5: signal warm-up history
SWING_START = pd.Timestamp("2016-01-01")


def write_with_warmup() -> None:
    """1300 business days from 2014-06-02 (before swing_start - 400 days); 069500 (always
    included) trends up with a 10-day wave, so a 10/200 MA pullback rule enters about every
    10 days once its SMA200 exists."""
    p = make_panel(1300, SYMBOLS, seed=2, products={s: "etf" for s in SYMBOLS},
                   start="2014-06-02")
    t = np.arange(1300)
    wave = 8000 * np.exp(0.0015 * t) * (1 + 0.03 * np.sin(2 * np.pi * t / 10))
    for sym in SYMBOLS:
        c = wave if sym == "069500" else p.close[sym].to_numpy() * SCALE
        o = c if sym == "069500" else p.open[sym].to_numpy() * SCALE
        df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005,
                           "low": np.minimum(o, c) * 0.995, "close": c,
                           "volume": p.volume[sym].to_numpy()}, index=p.close.index)
        df["source"] = "synthetic"
        store.save_daily("kr", sym, df)
    store.save_meta("kr", p.meta)


def test_lab_config_has_warmup_days():
    from autotrade.config import load_lab
    lab = load_lab()
    assert lab.warmup_days == 400
    assert lab.history_start == date(2014, 11, 27)        # 2016-01-01 - 400 calendar days


def test_collect_fetches_the_warmup_history(cli_env, monkeypatch):
    meta = pd.DataFrame({"name": ["X", "D"], "delisted": [False, True]},
                        index=pd.Index(["000010", "000040"], name="symbol"))
    starts = {}
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: object())
    monkeypatch.setattr(pipeline, "collect_meta", lambda client, market: meta)

    def fake_daily(client, market, symbol, start):
        starts["toss"] = start
        return 0

    def fake_supplement(start, symbols=None):
        from autotrade.data.delisted_kr import SupplementReport
        starts["pykrx"] = start
        return SupplementReport()
    monkeypatch.setattr(pipeline, "collect_daily", fake_daily)
    monkeypatch.setattr(pipeline, "supplement_delisted_kr", fake_supplement)
    pipeline.collect("kr")
    assert starts == {"toss": date(2014, 11, 27), "pykrx": date(2014, 11, 27)}
    pipeline.collect("kr", since=date(2020, 1, 1))         # an explicit --since is kept
    assert starts == {"toss": date(2020, 1, 1), "pykrx": date(2020, 1, 1)}


def test_sma200_rule_enters_right_after_swing_start_with_no_trade_or_stat_before(cli_env):
    from autotrade.rules.swing.ma_trend import MaTrend
    write_with_warmup()
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "ma_trend"])
    s = pipeline._setup("swing", "kr")
    t0, t1 = s.periods.train
    cal = s.calendar
    assert pd.Timestamp(t0) == cal[0] and cal[0] >= SWING_START    # split from swing_start
    panel = pipeline._panel(s, t1)
    assert panel.close.index[0] < SWING_START                    # warm-up history loaded ...
    assert panel.close.index[0] >= pd.Timestamp("2014-11-27")    # ... from history_start
    elig = pipeline._eligible(s, panel, "paper")
    assert not elig[elig.index < SWING_START].to_numpy().any()   # no universe before start
    # universe statistics start at swing_start: a ranked (not always-included) symbol is
    # eligible only once adv_window days from swing_start exist
    first_ranked = elig["A01"][elig["A01"]].index.min()
    assert first_ranked >= cal[s.lab.adv_window]
    sig = MaTrend({"s": 10, "l": 200, "k": 2.0}).signals(panel, eligible=elig)
    trades = pipeline.simulate(panel, sig, elig, t0, t1)
    assert len(trades) and trades["entry_date"].min() >= SWING_START
    assert trades["entry_date"].min() <= cal[30]                 # not ~200 days later
    assert (trades["exit_date"] >= SWING_START).all()
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    assert all(pd.Timestamp(p[0]) >= SWING_START for p in rows["period"])


# ---------------------------------------------------------------- P6: actual supplement results
def test_collect_writes_the_supplement_record_and_reports_show_it(cli_env, monkeypatch):
    from autotrade.data.delisted_kr import SupplementReport
    meta = pd.DataFrame({"name": ["X", "D1", "D2", "D3", "D4"],
                         "delisted": [False, True, True, True, True]},
                        index=pd.Index(["000010", "000041", "000042", "000043", "000044"],
                                       name="symbol"))
    monkeypatch.setattr(pipeline, "make_client_from_env", lambda path: object())
    monkeypatch.setattr(pipeline, "collect_meta", lambda client, market: meta)
    monkeypatch.setattr(pipeline, "collect_daily", lambda client, market, symbol, start: 0)
    monkeypatch.setattr(pipeline, "supplement_delisted_kr", lambda start, symbols=None:
                        SupplementReport(fetched=1, empty=["000042"], failed=["000043"],
                                         skipped=["000044"]))
    path = cli_env.data / "kr" / "supplement.json"
    board = pipeline.report("swing", "kr").read_text(encoding="utf-8")
    assert "보충 기록 없음" in board and not path.exists()
    pipeline.collect("kr")
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert {k: rec[k] for k in ("requested", "fetched", "empty", "failed", "skipped")} == \
        {"requested": 4, "fetched": 1, "empty": 1, "failed": 1, "skipped": 1}
    assert rec["date"] == date.today().isoformat() and rec["since"] == "2014-11-27"
    board = pipeline.report("swing", "kr").read_text(encoding="utf-8")
    assert "보충 기록 없음" not in board
    assert "받음 1" in board and "실패 1" in board and "건너뜀 1" in board


# ---------------------------------------------------------------- review fixes (round 1)
def _insert_date_everywhere(day: str, market: str = "kr") -> None:
    """A stray bar on `day` for every symbol: the calendar keeps it as a trading date."""
    for f in sorted((store.data_root() / market / "1d").glob("*.parquet")):
        df = store.load_daily(market, f.stem)
        extra = pd.DataFrame([df.iloc[0].to_dict()], index=pd.DatetimeIndex([day]))
        store.save_daily(market, f.stem, pd.concat([df, extra]).sort_index())


def test_rebuild_refuses_train_valid_that_differ_from_the_logged_periods(cli_env):
    from autotrade.validate.split import split_periods
    extend_store(1)                                  # 1001 business days
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    per = split_periods(store.trading_dates("kr", date(2016, 1, 1), date(2100, 1, 1)))
    _open_ticket(cli_env, "breakout:x", per.sealed, per.sealed[1])
    ExperimentLog(cli_env.research / "experiments.jsonl").record(
        "swing", "kr", "breakout", "breakout:x", {}, "validate", pipeline._iso(per.valid),
        {"valid_daily_sharpe": 0.1, "validate_run": 3})
    periods_file(cli_env).unlink()
    _insert_date_everywhere("2016-03-05")       # a Saturday inside train: sealed unchanged,
    moved = split_periods(store.trading_dates("kr", date(2016, 1, 1), per.sealed[1]))
    assert moved.sealed == per.sealed and moved.train != per.train
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "breakout"])
    assert r.exit_code == 1 and "Traceback" not in r.output
    assert len(r.output.strip().splitlines()) == 1
    assert "train" in r.output and per.train[1].isoformat() in r.output
    assert not periods_file(cli_env).exists()


def test_rebuild_needs_logged_screen_and_validate_periods(cli_env):
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    per = periods_of(cli_env)
    _open_ticket(cli_env, "breakout:x", per.sealed, per.sealed[1])
    periods_file(cli_env).unlink()                  # screen logged, no validate row
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "breakout"])
    assert r.exit_code == 1 and "validate" in r.output
    assert not periods_file(cli_env).exists()


def _cycle_two_without_ticket(env):
    """Cycle 1 opened and validated, then `lab periods --new-cycle`: no cycle-2 ticket yet."""
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    per = periods_of(env)
    _open_ticket(env, "breakout:x", per.sealed, per.sealed[1])
    ExperimentLog(env.research / "experiments.jsonl").record(
        "swing", "kr", "breakout", "breakout:x", {}, "validate", pipeline._iso(per.valid),
        {"valid_daily_sharpe": 0.1, "validate_run": 1})
    extend_store(70)
    run(["periods", "--pocket", "swing", "--market", "kr", "--new-cycle"])
    return json.loads(periods_file(env).read_text(encoding="utf-8"))


def test_new_cycle_is_logged_and_a_lost_file_is_rebuilt_in_it(cli_env):
    before = _cycle_two_without_ticket(cli_env)
    rows = ExperimentLog(cli_env.research / "experiments.jsonl").rows("swing", "kr")
    started = rows[rows["stage"] == "cycle_started"]
    assert len(started) == 1
    m = started.iloc[0]["metrics"]
    assert m["cycle"] == 2 and m["sealed"] == before["sealed"]
    assert m["data_cutoff"] == before["data_cutoff"] and m["train"] == before["train"]
    assert m["valid"] == before["valid"] and m["started_at"]
    assert list(started.iloc[0]["period"]) == before["sealed"]
    periods_file(cli_env).unlink()
    extend_store(5)
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "breakout"])
    after = json.loads(periods_file(cli_env).read_text(encoding="utf-8"))
    assert after["cycle"] == 2 and _split_keys(after) == _split_keys(before)


def test_rebuild_refuses_a_logged_cycle_it_cannot_reproduce(cli_env):
    before = _cycle_two_without_ticket(cli_env)
    log = ExperimentLog(cli_env.research / "experiments.jsonl")
    bad = pd.bdate_range(before["sealed"][0], periods=3)[-1].date().isoformat()
    log.record("swing", "kr", "-", "-", {}, "cycle_started", (bad, before["sealed"][1]),
               {"cycle": 2, "sealed": [bad, before["sealed"][1]],
                "data_cutoff": before["data_cutoff"]})            # e.g. a hand-edited log
    periods_file(cli_env).unlink()
    r = CliRunner().invoke(app, ["screen", "--pocket", "swing", "--market", "kr",
                                 "--family", "breakout"])
    assert r.exit_code == 1 and "cycle 2" in r.output and "Traceback" not in r.output
    assert not periods_file(cli_env).exists()


def test_collecting_past_the_cutoff_does_not_change_sealed_run_signals(cli_env):
    from autotrade.rules.swing.tom import Tom
    n = len(pd.bdate_range("2016-01-04", "2019-08-30"))     # cutoff Fri 2019-08-30: August's
    p = make_panel(n, SYMBOLS, seed=2, products={s: "etf" for s in SYMBOLS})  # last trading
    for sym in SYMBOLS:                                       # day, not its last calendar day
        df = pd.DataFrame({c: getattr(p, c)[sym] * (SCALE if c != "volume" else 1)
                           for c in ("open", "high", "low", "close", "volume")})
        store.save_daily("kr", sym, df)
    run(["screen", "--pocket", "swing", "--market", "kr", "--family", "tom"])

    def sealed_run():
        s = pipeline._setup("swing", "kr")
        assert s.record.data_cutoff == date(2019, 8, 30)
        panel = pipeline._panel(s, s.periods.sealed[1])
        assert panel.calendar.max() <= pd.Timestamp(s.record.data_cutoff)
        rule = Tom({"n": 2, "m": 3, "k": 2.0})
        return rule.signals(panel, eligible=pipeline._eligible(s, panel, "paper")).entries
    first = sealed_run()
    extend_store(10)                                   # a later collect: September bars
    pd.testing.assert_frame_equal(sealed_run(), first)
