from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from autotrade.config import (ConfigError, load_lab, load_market, load_market_raw,
                               load_pocket, load_synced)
from autotrade.types import TRADE_COLUMNS, Panel


@pytest.fixture
def synced(tmp_path, monkeypatch):
    p = tmp_path / "synced.yaml"
    p.write_text("commission_rate: {kr: 0.00015, us: 0.0025}\n"
                 "usd_krw: 1400.0\nsynced_at: '2026-11-19'\n")
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(p))
    return p


def test_kr_market_values(synced):
    m = load_market("kr")
    assert m.sell_tax_rate == {"stock": 0.0020, "etf": 0.0}
    assert m.tick_table["stock"][:3] == [(2000, 1), (5000, 5), (20000, 10)]
    assert m.tick_table["stock"][-1] == (float("inf"), 1000)
    assert m.tick_table["etf"] == [(2000, 1), (float("inf"), 5)]
    assert m.max_tick_cost_pct == 0.001
    assert m.always_include == ["069500", "102110"]
    assert m.commission_rate == 0.00015
    assert m.fractional_orders is False


def test_us_market_values(synced):
    m = load_market("us")
    assert m.sec_fee_rate == 0.0000206
    assert m.slippage_min_pct == 0.0005 and m.slippage_min_abs == 0.02
    assert m.min_price == 5.0 and m.min_avg_volume_14d == 1_000_000
    assert m.always_include == ["SPY", "QQQ"]
    assert m.sell_tax_rate == {"stock": 0.0, "etf": 0.0}
    assert m.fractional_orders is True


def test_market_raw_has_no_commission_and_replace_works():
    m = load_market_raw("kr")
    assert m.commission_rate is None
    assert replace(m, commission_rate=0.001).commission_rate == 0.001


def test_pocket_and_lab_values():
    s = load_pocket("swing")
    assert (s.risk_per_trade, s.max_position_weight, s.max_positions) == (0.01, 0.25, 4)
    assert (s.drawdown_stop, s.daily_loss_stop, s.hold_days) == (0.15, 0.03, (2, 20))
    lab = load_lab()
    assert lab.split == (0.6, 0.2, 0.2) and lab.swing_start == date(2016, 1, 1)
    assert lab.capital == {"real": 50_000, "paper": 5_000_000}
    assert lab.criteria.t_min == 2.0 and lab.criteria.dsr_min == 0.95
    assert lab.criteria.max_final_candidates == 3


def test_commission_requires_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_SYNCED", str(tmp_path / "missing.yaml"))
    with pytest.raises(ConfigError):
        load_synced()
    with pytest.raises(ConfigError):
        load_market("kr")


def test_panel_slice_inclusive_and_symbols():
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"A": range(5), "B": range(5)}, index=idx)
    meta = pd.DataFrame({"name": ["a", "b"]}, index=["A", "B"])
    p = Panel(df, df, df, df, df, meta)
    s = p.slice(date(2024, 1, 2), date(2024, 1, 4))
    assert list(s.close.index) == list(idx[1:4])
    assert s.meta is meta and s.symbols == ["A", "B"]


def test_trade_columns():
    assert TRADE_COLUMNS[0] == "symbol" and len(TRADE_COLUMNS) == 8


def test_relative_env_paths_resolve_against_repo_root(tmp_path, monkeypatch):
    from autotrade import pipeline
    from autotrade.config import _config_root, _synced_path, repo_root
    from autotrade.data.store import data_root
    root = repo_root()
    assert (root / "pyproject.toml").exists()
    monkeypatch.chdir(tmp_path)                       # cwd elsewhere
    monkeypatch.setenv("AUTOTRADE_DATA", "./data-rel")
    monkeypatch.setenv("AUTOTRADE_RESEARCH", "research-rel")
    monkeypatch.setenv("AUTOTRADE_CONFIG", "config")
    monkeypatch.setenv("AUTOTRADE_SYNCED", "config/synced-rel.yaml")
    assert data_root() == root / "data-rel"
    assert pipeline.research_root() == root / "research-rel"
    assert _config_root() == root / "config"
    assert _synced_path() == root / "config" / "synced-rel.yaml"
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "abs"))
    assert data_root() == tmp_path / "abs"           # absolute paths unchanged


def test_cli_loads_dotenv_before_computing_paths(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from autotrade.cli import app
    env = tmp_path / ".env"
    env.write_text(f"AUTOTRADE_RESEARCH={tmp_path / 'from-dotenv'}\n")
    monkeypatch.setenv("AUTOTRADE_DOTENV", str(env))
    monkeypatch.delenv("AUTOTRADE_RESEARCH")            # unset now, restored at teardown
    from autotrade import pipeline                      # without .env: tmp, never the repo
    monkeypatch.setattr(pipeline, "repo_root", lambda: tmp_path / "fallback")
    r = CliRunner().invoke(app, ["journal", "메모"])
    assert r.exit_code == 0, r.output
    assert list((tmp_path / "from-dotenv" / "journal").glob("*.md"))
