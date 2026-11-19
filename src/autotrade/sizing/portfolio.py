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
    peak = cap0
    prev_eq, prev_prev_eq = cap0, cap0
    latched = False
    open_pos: dict[int, dict] = {}
    fills: list[dict] = []
    skipped: list[dict] = []
    eq_values: list[float] = []

    for d in dates:
        if pocket.drawdown_stop and prev_eq <= peak * (1 - pocket.drawdown_stop):
            latched = True
        day_loss = prev_eq / prev_prev_eq - 1 <= -pocket.daily_loss_stop

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
                reason = None
                qty = 0.0
                if latched:
                    reason = "drawdown_stop"
                elif day_loss:
                    reason = "daily_loss_stop"
                elif pocket.max_positions is not None and len(open_pos) >= pocket.max_positions:
                    reason = "max_positions"
                else:
                    gap = gap_allowance(panel, t.symbol, d)
                    risk_qty = (prev_eq * pocket.risk_per_trade * t.confidence
                                / (t.entry_price * (t.stop_pct + gap)))
                    qty = min(risk_qty, prev_eq * pocket.max_position_weight / t.entry_price)
                    if not fractional:
                        qty = float(np.floor(qty))
                    if not qty > 0:
                        reason = "too_small"
                    else:
                        buy_cash = cost_model.fill_cash(t.symbol, "buy", qty, t.entry_price)
                        if buy_cash > cash:
                            reason = "cash"
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
        prev_prev_eq, prev_eq = prev_eq, value
        peak = max(peak, value)

    equity = pd.Series(eq_values, index=pd.DatetimeIndex(dates, name="date"), name="equity")
    fills_df = pd.DataFrame(fills, columns=FILL_COLUMNS).sort_values(
        ["entry_date", "symbol"], kind="stable").reset_index(drop=True)
    skipped_df = pd.DataFrame(skipped, columns=SKIP_COLUMNS).reset_index(drop=True)
    return PortfolioResult(equity, fills_df, skipped_df)
