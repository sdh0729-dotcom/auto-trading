import json
import math
from dataclasses import replace
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from autotrade.config import load_market_raw
from autotrade.costs import CostModel
from autotrade.rules.swing.breakout import Breakout
from autotrade.screen import screen_family, simulate
from autotrade.validate.experiments import ExperimentLog
from autotrade.validate.seal import SealViolation
from tests.fixtures import make_panel

KR_T = replace(load_market_raw("kr"), commission_rate=0.00015)
SCREEN_COLS = ["rule_id", "params", "n_trades", "mean_net_r", "t_stat", "profit_factor",
               "win_rate", "sum_net_return"]


@pytest.fixture
def setup(tmp_path):
    # prices ~10,000 KRW so KR tick costs are realistic
    p = make_panel(1100, ["A", "B", "C", "D", "E"], seed=9)
    for f in (p.open, p.high, p.low, p.close):
        f *= 100
    elig = pd.DataFrame(True, index=p.close.index, columns=p.close.columns)
    cm = CostModel(KR_T, lambda s: "stock")
    log = ExperimentLog(tmp_path / "experiments.jsonl")
    return p, elig, cm, log


def test_screen_refuses_sealed_period(setup):
    p, elig, cm, log = setup
    with pytest.raises(SealViolation):
        screen_family("breakout", p, elig, date(2016, 1, 1), date(2024, 6, 1), cm, log, "swing", "kr",
                      sealed_start=date(2024, 1, 1))
    with pytest.raises(SealViolation):  # end == sealed_start is already inside the seal
        screen_family("breakout", p, elig, date(2016, 1, 1), date(2024, 1, 1), cm, log, "swing", "kr",
                      sealed_start=date(2024, 1, 1))
    assert log.n_trials("swing", "kr") == 0


def test_screen_logs_every_param(setup):
    p, elig, cm, log = setup
    df = screen_family("breakout", p, elig, date(2016, 1, 1), date(2019, 12, 31), cm, log, "swing", "kr",
                       sealed_start=date(2024, 1, 1))
    assert log.n_trials("swing", "kr") == len(Breakout.grid()) == len(df)
    assert list(df.columns) == SCREEN_COLS
    assert list(df["params"]) == Breakout.grid()
    assert list(df["rule_id"]) == [Breakout(g).rule_id for g in Breakout.grid()]
    rows = log.rows("swing", "kr")
    assert set(rows["stage"]) == {"screen"} and set(rows["family"]) == {"breakout"}
    assert rows["period"].iloc[0] == ["2016-01-01", "2019-12-31"]


def test_screen_metrics_match_simulation(setup):
    p, elig, cm, log = setup
    df = screen_family("breakout", p, elig, date(2016, 1, 1), date(2019, 12, 31), cm, log, "swing", "kr",
                       sealed_start=date(2024, 1, 1))
    g = Breakout.grid()[4]
    net = cm.net_returns(simulate(p, Breakout(g).signals(p), elig, date(2016, 1, 1), date(2019, 12, 31)))
    row = df.iloc[4]
    r, ret = net["net_r"], net["net_return"]
    assert row["n_trades"] == len(net) > 2
    assert row["mean_net_r"] == pytest.approx(r.mean())
    assert row["t_stat"] == pytest.approx(r.mean() / (r.std(ddof=1) / math.sqrt(len(r))))
    assert row["profit_factor"] == pytest.approx(ret[ret > 0].sum() / abs(ret[ret < 0].sum()))
    assert row["win_rate"] == pytest.approx((ret > 0).mean())
    assert row["sum_net_return"] == pytest.approx(ret.sum())
    logged = log.rows("swing", "kr").iloc[4]
    assert logged["rule_id"] == row["rule_id"] and logged["params"] == g
    assert logged["metrics"]["n_trades"] == row["n_trades"]
    assert logged["metrics"]["t_stat"] == pytest.approx(row["t_stat"])


def test_screen_never_reads_data_after_end(setup):
    """Sealed-period bars must not influence screening at all (Review Focus 4)."""
    p, elig, cm, log = setup
    end = date(2018, 12, 31)
    df1 = screen_family("breakout", p, elig, date(2016, 1, 1), end, cm, log, "swing", "kr",
                        sealed_start=date(2019, 1, 1))
    cut = p.close.index > pd.Timestamp(end)
    for f in (p.open, p.high, p.low, p.close):
        f.loc[cut] = np.nan
    elig.loc[cut] = False
    df2 = screen_family("breakout", p, elig, date(2016, 1, 1), end, cm, log, "swing", "kr",
                        sealed_start=date(2019, 1, 1))
    pd.testing.assert_frame_equal(df1, df2)


def test_screen_no_trades_metrics(setup):
    p, elig, cm, log = setup
    df = screen_family("breakout", p, elig & False, date(2016, 1, 1), date(2019, 12, 31), cm, log,
                       "swing", "kr", sealed_start=date(2024, 1, 1))
    assert (df["n_trades"] == 0).all()
    assert df["mean_net_r"].isna().all() and df["t_stat"].isna().all()
    assert df["profit_factor"].isna().all() and df["win_rate"].isna().all()
    assert (df["sum_net_return"] == 0).all()


def test_screen_unknown_family(setup):
    p, elig, cm, log = setup
    with pytest.raises(KeyError):
        screen_family("nope", p, elig, date(2016, 1, 1), date(2019, 12, 31), cm, log, "swing", "kr",
                      sealed_start=date(2024, 1, 1))


# --- ExperimentLog ---------------------------------------------------------------------------

def test_log_is_append_only_jsonl_with_timestamp(tmp_path):
    path = tmp_path / "sub" / "exp.jsonl"
    ExperimentLog(path).record("swing", "kr", "breakout", "breakout:aaaa", {"n": 20}, "screen",
                               ("2016-01-01", "2019-12-31"), {"n_trades": 3, "pf": float("inf")})
    ExperimentLog(path).record("swing", "kr", "breakout", "breakout:aaaa", {"n": 20}, "validate",
                               ("2020-01-01", "2021-12-31"), {"n_trades": np.int64(4)})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["stage"] == "screen" and first["params"] == {"n": 20}
    assert first["period"] == ["2016-01-01", "2019-12-31"]
    assert first["metrics"]["pf"] == float("inf")
    datetime.fromisoformat(first["ts"])
    assert json.loads(lines[1])["metrics"]["n_trades"] == 4


def test_n_trials_counts_distinct_screen_rule_ids(tmp_path):
    log = ExperimentLog(tmp_path / "exp.jsonl")
    period = ("2016-01-01", "2019-12-31")
    log.record("swing", "kr", "breakout", "breakout:a", {}, "screen", period, {})
    log.record("swing", "kr", "breakout", "breakout:a", {}, "screen", period, {})  # repeat run
    log.record("swing", "kr", "breakout", "breakout:b", {}, "screen", period, {})
    log.record("swing", "kr", "breakout", "breakout:c", {}, "validate", period, {})
    log.record("swing", "us", "breakout", "breakout:d", {}, "screen", period, {})
    log.record("daytrade", "kr", "orb", "orb:e", {}, "screen", period, {})
    assert log.n_trials("swing", "kr") == 2
    assert log.n_trials("swing", "us") == 1
    assert log.n_trials("daytrade", "us") == 0
    assert len(log.rows("swing", "kr")) == 4
    assert log.rows("daytrade", "us").empty


def test_log_rejects_unknown_stage(tmp_path):
    log = ExperimentLog(tmp_path / "exp.jsonl")
    with pytest.raises(ValueError):
        log.record("swing", "kr", "breakout", "breakout:a", {}, "sealed", ("a", "b"), {})
    assert not (tmp_path / "exp.jsonl").exists()


def test_screen_rule_never_sees_bars_after_end(setup, monkeypatch):
    p, elig, cm, log = setup
    seen = []

    class Spy(Breakout):
        family = "spy"

        @classmethod
        def grid(cls):
            return [{"n": 20, "m": 5, "k": 2.0}]

        def signals(self, panel):
            seen.append(panel.close.index.max())
            return super().signals(panel)

    monkeypatch.setitem(__import__("autotrade.screen.run", fromlist=["x"]).SWING_FAMILIES, "spy", Spy)
    screen_family("spy", p, elig, date(2016, 1, 1), date(2018, 6, 29), cm, log, "swing", "kr",
                  sealed_start=date(2019, 1, 1))
    assert seen == [pd.Timestamp("2018-06-29")] and p.close.index.max() > seen[0]
