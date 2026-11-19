"""Capital-constrained portfolio on top of the simulator's fixed-amount candidates.

Per date: exits dated that day first (free cash and slots), then entries dated that day in
confidence-descending order, then equity is marked to the close (NaN close carries the last
valid one). Sizing equity is the equity marked at the previous trading day's close (initial
capital on the first day). Skip-reason precedence: drawdown_stop, daily_loss_stop,
max_positions, too_small, cash.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from autotrade.config import PocketConfig
from autotrade.costs import CostModel
from autotrade.types import TRADE_COLUMNS, Panel

FILL_COLUMNS = TRADE_COLUMNS + ["qty", "net_pnl", "net_r"]
SKIP_COLUMNS = TRADE_COLUMNS + ["reason"]
MIN_GAP_OBSERVATIONS = 20


@dataclass
class PortfolioResult:
    equity: pd.Series
    fills: pd.DataFrame
    skipped: pd.DataFrame


def gap_allowance(panel: Panel, symbol: str, asof: pd.Timestamp, lookback: int = 250) -> float:
    """95th percentile of adverse open gaps max(0, prev_close/open - 1) over the last
    `lookback` rows strictly before `asof`; 0.0 with fewer than 20 observations."""
    if symbol not in panel.close.columns or symbol not in panel.open.columns:
        return 0.0
    prev_close = panel.close[symbol].shift(1)
    gap = (prev_close / panel.open[symbol] - 1).replace([np.inf, -np.inf], np.nan)
    gap = gap[gap.index < pd.Timestamp(asof)].iloc[-lookback:].dropna()
    if len(gap) < MIN_GAP_OBSERVATIONS:
        return 0.0
    return float(np.quantile(gap.clip(lower=0.0).to_numpy(), 0.95))


class RiskState:
    """Pocket entry gates on prior-close equity, carried across trading days.

    `begin_day()` before the day's entries sets `latched` (sticky drawdown stop: prior-close
    equity <= peak * (1 - drawdown_stop)) and `day_loss` (prior day's return <=
    -daily_loss_stop); `end_day(equity)` records that day's close-marked equity."""

    def __init__(self, pocket: PocketConfig, equity: float):
        self.pocket = pocket
        self.prev_eq = self.prev_prev_eq = self.peak = equity
        self.latched = False
        self.day_loss = False

    def begin_day(self) -> None:
        if self.pocket.drawdown_stop and \
                self.prev_eq <= self.peak * (1 - self.pocket.drawdown_stop):
            self.latched = True
        self.day_loss = self.prev_eq / self.prev_prev_eq - 1 <= -self.pocket.daily_loss_stop

    def end_day(self, equity: float) -> None:
        self.prev_prev_eq, self.prev_eq = self.prev_eq, equity
        self.peak = max(self.peak, equity)


def decide_entry(risk: RiskState, n_open: int, cash: float, symbol: str, entry_price: float,
                 stop_pct: float, confidence: float, gap: float, cost_model: CostModel,
                 fractional: bool) -> tuple[float, float, str | None]:
    """Size one entry candidate -> (qty, buy_cash, skip_reason or None).

    qty = min(equity * risk_per_trade * confidence / (price * (stop_pct + gap)),
              equity * max_position_weight / price), floored unless fractional; equity is
    the prior-close mark. Skip precedence: drawdown_stop, daily_loss_stop, max_positions,
    too_small, cash."""
    pocket = risk.pocket
    if risk.latched:
        return 0.0, 0.0, "drawdown_stop"
    if risk.day_loss:
        return 0.0, 0.0, "daily_loss_stop"
    if pocket.max_positions is not None and n_open >= pocket.max_positions:
        return 0.0, 0.0, "max_positions"
    eq = risk.prev_eq
    risk_qty = eq * pocket.risk_per_trade * confidence / (entry_price * (stop_pct + gap))
    qty = min(risk_qty, eq * pocket.max_position_weight / entry_price)
    if not fractional:
        qty = float(np.floor(qty))
    if not qty > 0:
        return qty, 0.0, "too_small"
    buy_cash = cost_model.fill_cash(symbol, "buy", qty, entry_price)
    if buy_cash > cash:
        return qty, buy_cash, "cash"
    return qty, buy_cash, None


def run_portfolio(candidates: pd.DataFrame, panel: Panel, cost_model: CostModel,
                  pocket: PocketConfig, capital: float, currency_per_krw: float,
                  fractional: bool) -> PortfolioResult:
    cap0 = capital * currency_per_krw
    if candidates.empty:
        return PortfolioResult(pd.Series(dtype=float, name="equity",
                                         index=pd.DatetimeIndex([], name="date")),
                               pd.DataFrame(columns=FILL_COLUMNS),
                               pd.DataFrame(columns=SKIP_COLUMNS))
    cands = candidates[TRADE_COLUMNS].reset_index(drop=True)
    idx = panel.close.index
    lo, hi = cands["entry_date"].min(), cands["exit_date"].max()
    dates = idx[(idx >= lo) & (idx <= hi)]
    known = set(dates)
    if not (set(cands["entry_date"]) <= known and set(cands["exit_date"]) <= known):
        raise ValueError("candidate dates must be trading days of the panel")
    close = panel.close.ffill()

    by_entry = {d: g for d, g in cands.groupby("entry_date")}
    by_exit: dict[pd.Timestamp, list[int]] = {}
    for i, d in cands["exit_date"].items():
        by_exit.setdefault(d, []).append(i)

    cash = cap0
    risk = RiskState(pocket, cap0)
    open_pos: dict[int, dict] = {}
    fills: list[dict] = []
    skipped: list[dict] = []
    eq_values: list[float] = []

    for d in dates:
        risk.begin_day()

        for i in by_exit.get(d, []):
            pos = open_pos.pop(i, None)
            if pos is None:
                continue
            t = cands.loc[i]
            proceeds = cost_model.fill_cash(t.symbol, "sell", pos["qty"], t.exit_price)
            cash += proceeds
            pnl = proceeds - pos["buy_cash"]
            fills.append({**t.to_dict(), "qty": pos["qty"], "net_pnl": pnl,
                          "net_r": pnl / (pos["qty"] * t.entry_price * t.stop_pct)})

        if d in by_entry:
            day = by_entry[d].sort_values("confidence", ascending=False, kind="stable")
            for i, t in day.iterrows():
                qty, buy_cash, reason = decide_entry(
                    risk, len(open_pos), cash, t.symbol, t.entry_price, t.stop_pct,
                    t.confidence, gap_allowance(panel, t.symbol, d), cost_model, fractional)
                if reason:
                    skipped.append({**t.to_dict(), "reason": reason})
                else:
                    cash -= buy_cash
                    open_pos[i] = {"qty": qty, "buy_cash": buy_cash}

        value = cash
        for i, pos in open_pos.items():
            px = close.at[d, cands.at[i, "symbol"]]
            if not np.isfinite(px):
                px = cands.at[i, "entry_price"]
            value += pos["qty"] * px
        eq_values.append(value)
        risk.end_day(value)

    equity = pd.Series(eq_values, index=pd.DatetimeIndex(dates, name="date"), name="equity")
    fills_df = pd.DataFrame(fills, columns=FILL_COLUMNS).sort_values(
        ["entry_date", "symbol"], kind="stable").reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped, columns=SKIP_COLUMNS).reset_index(drop=True)
    return PortfolioResult(equity, fills_df, skipped_df)
