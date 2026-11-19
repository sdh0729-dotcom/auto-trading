from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from autotrade.types import Market, Pocket, Product


class ConfigError(Exception):
    pass


def _config_root() -> Path:
    env = os.environ.get("AUTOTRADE_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config"


def _synced_path() -> Path:
    env = os.environ.get("AUTOTRADE_SYNCED")
    return Path(env) if env else _config_root() / "synced.yaml"


def _read(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"invalid config file: {path}")
    return data


@dataclass(frozen=True)
class MarketConfig:
    market: Market
    currency: str
    sell_tax_rate: dict[Product, float]
    always_include: list[str]
    exclude_name_keywords: list[str]
    delisted_supplement: bool
    tick_table: dict[Product, list[tuple[float, float]]] | None = None
    slippage_ticks: int | None = None
    max_tick_cost_pct: float | None = None
    session_open: str | None = None
    session_close: str | None = None
    closing_auction_start: str | None = None
    sec_fee_rate: float | None = None
    slippage_min_pct: float | None = None
    slippage_min_abs: float | None = None
    min_price: float | None = None
    min_avg_volume_14d: int | None = None
    commission_rate: float | None = None
    fractional_orders: bool = False


@dataclass(frozen=True)
class PocketConfig:
    risk_per_trade: float
    max_position_weight: float
    daily_loss_stop: float
    max_positions: int | None = None
    drawdown_stop: float | None = None
    hold_days: tuple[int, int] | None = None


@dataclass(frozen=True)
class Criteria:
    t_min: float
    dsr_min: float
    pbo_max: float
    top_removed_n: int
    pf_min: float
    mdd_max: float
    min_trades: int
    max_final_candidates: int
    cost_r_max: float


@dataclass(frozen=True)
class LabConfig:
    split: tuple[float, float, float]
    swing_start: date
    capital: dict[str, float]
    universe_top_n: int
    adv_window: int
    criteria: Criteria


@dataclass(frozen=True)
class SyncedValues:
    commission_rate: dict[Market, float]
    usd_krw: float
    synced_at: str


def load_market_raw(m: Market) -> MarketConfig:
    d = _read(_config_root() / "markets" / f"{m}.yaml")
    d.pop("commission_rate", None)
    if "tick_table" in d:
        d["tick_table"] = {p: [(float(u), float(t)) for u, t in rows]
                           for p, rows in d["tick_table"].items()}
    try:
        return MarketConfig(**d)
    except TypeError as e:
        raise ConfigError(f"invalid market config {m}: {e}") from e


def load_market(m: Market) -> MarketConfig:
    from dataclasses import replace
    raw = load_market_raw(m)
    synced = load_synced()
    if m not in synced.commission_rate:
        raise ConfigError("run `lab sync` first")
    return replace(raw, commission_rate=synced.commission_rate[m])


def load_pocket(p: Pocket) -> PocketConfig:
    d = _read(_config_root() / "pockets" / f"{p}.yaml")
    if "hold_days" in d:
        d["hold_days"] = tuple(d["hold_days"])
    try:
        return PocketConfig(**d)
    except TypeError as e:
        raise ConfigError(f"invalid pocket config {p}: {e}") from e


def load_lab() -> LabConfig:
    d = _read(_config_root() / "lab.yaml")
    try:
        return LabConfig(
            split=tuple(d["split"]),
            swing_start=d["swing_start"],
            capital=dict(d["capital"]),
            universe_top_n=d["universe_top_n"],
            adv_window=d["adv_window"],
            criteria=Criteria(**d["criteria"]),
        )
    except (KeyError, TypeError) as e:
        raise ConfigError(f"invalid lab config: {e}") from e


def load_synced() -> SyncedValues:
    path = _synced_path()
    if not path.exists():
        raise ConfigError("run `lab sync` first")
    d = _read(path)
    try:
        return SyncedValues(commission_rate=dict(d["commission_rate"]),
                            usd_krw=float(d["usd_krw"]),
                            synced_at=str(d["synced_at"]))
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError("run `lab sync` first") from e
