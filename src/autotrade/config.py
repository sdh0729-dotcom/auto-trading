from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import yaml

from autotrade.types import Market, Pocket, Product, VIConfig


class ConfigError(Exception):
    pass


def repo_root() -> Path:
    """The directory containing pyproject.toml above this package (falls back to cwd)."""
    for d in Path(__file__).resolve().parents:
        if (d / "pyproject.toml").exists():
            return d
    return Path.cwd()


def env_path(name: str, default: Path) -> Path:
    """Path from environment variable `name`; relative values resolve against the repo root
    (not the current directory), unset/empty gives `default`."""
    value = os.environ.get(name)
    if not value:
        return default
    p = Path(value).expanduser()
    return p if p.is_absolute() else repo_root() / p


_LOADED_DOTENV: set[Path] = set()


def load_env() -> None:
    """Load the repo's .env (or $AUTOTRADE_DOTENV) into os.environ once per process; values
    already in the environment win."""
    from dotenv import load_dotenv

    path = env_path("AUTOTRADE_DOTENV", repo_root() / ".env")
    if path in _LOADED_DOTENV:
        return
    _LOADED_DOTENV.add(path)
    if path.exists():
        load_dotenv(path, override=False)


def _config_root() -> Path:
    return env_path("AUTOTRADE_CONFIG", repo_root() / "config")


def _synced_path() -> Path:
    return env_path("AUTOTRADE_SYNCED", _config_root() / "synced.yaml")


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
    benchmark: str | None = None      # buy-and-hold symbol validate must beat
    # exclude ETFs whose leverage factor is unknown (US: names are bare tickers, so the
    # name keywords cannot catch leveraged/inverse ETFs without it)
    etf_leverage_required: bool = False
    minute_start: date | None = None  # first day of stored 1-minute history
    # ORB stocks-in-play ATR14 floor of the prior days: absolute price (US) or fraction of the
    # prior close (KR); None = no floor
    orb_min_atr: float | None = None
    orb_min_atr_pct: float | None = None
    vi: VIConfig | None = None        # KR volatility-interruption auctions
    # per-day regular sessions (data/sessions.py): exchange_calendars code (XKRX / XNYS) and
    # {date: (open, close)} local "HH:MM" corrections for days the package gets wrong
    exchange_calendar: str | None = None
    session_overrides: dict[date, tuple[str, str]] | None = None


@dataclass(frozen=True)
class PocketConfig:
    risk_per_trade: float
    max_position_weight: float
    daily_loss_stop: float
    max_positions: int | None = None
    drawdown_stop: float | None = None
    hold_days: tuple[int, int] | None = None
    benchmark_gate: bool = True       # validate recommends only rules beating the benchmark


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
    warmup_days: int = 0     # calendar days of signal warm-up history before swing_start

    @property
    def history_start(self) -> date:
        """First date collected / loaded for signals: swing_start - warmup_days."""
        return self.swing_start - timedelta(days=self.warmup_days)


@dataclass(frozen=True)
class SyncedValues:
    commission_rate: dict[Market, float]
    usd_krw: float
    synced_at: str
    # promotional rate end date from the commissions API (ISO) or None
    commission_end_date: dict = field(default_factory=dict)


def load_market_raw(m: Market) -> MarketConfig:
    d = _read(_config_root() / "markets" / f"{m}.yaml")
    d.pop("commission_rate", None)
    if "tick_table" in d:
        d["tick_table"] = {p: [(float(u), float(t)) for u, t in rows]
                           for p, rows in d["tick_table"].items()}
    if "vi" in d:
        try:
            d["vi"] = VIConfig(**d["vi"])
        except TypeError as e:
            raise ConfigError(f"invalid vi config {m}: {e}") from e
    if d.get("session_overrides"):
        try:
            d["session_overrides"] = {
                (k if isinstance(k, date) else date.fromisoformat(str(k))): (str(v[0]), str(v[1]))
                for k, v in d["session_overrides"].items()}
        except (TypeError, ValueError, IndexError) as e:
            raise ConfigError(f"invalid session_overrides {m}: {e}") from e
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
        lab = LabConfig(
            split=tuple(d["split"]),
            swing_start=d["swing_start"],
            capital=dict(d["capital"]),
            universe_top_n=d["universe_top_n"],
            adv_window=d["adv_window"],
            criteria=Criteria(**d["criteria"]),
            warmup_days=int(d.get("warmup_days", 0)),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError(f"invalid lab config: {e}") from e
    if lab.warmup_days < 0:
        raise ConfigError(f"invalid lab config: warmup_days must be >= 0, got {lab.warmup_days}")
    return lab


def load_synced() -> SyncedValues:
    path = _synced_path()
    if not path.exists():
        raise ConfigError("run `lab sync` first")
    d = _read(path)
    try:
        return SyncedValues(commission_rate=dict(d["commission_rate"]),
                            usd_krw=float(d["usd_krw"]),
                            synced_at=str(d["synced_at"]),
                            commission_end_date=dict(d.get("commission_end_date") or {}))
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError("run `lab sync` first") from e
