"""Capital-constrained portfolio for same-day (intraday) trades.

Trades carry bar-END timestamps in entry_date / exit_date. Per local day, events run in time
order (at the same timestamp exits before entries; entries by confidence, descending). Sizing
equity is the previous day's close equity (initial capital on the first day); an entry pays
from the day's running cash and every position held at the same time counts against it. The
book is flat after the last exit, so equity is marked once per day (= the cash). Skip reasons,
in precedence order: daily_loss_stop (the day's realized, cost-net P&L <= -daily_loss_stop x
day-start equity blocks the entries after that moment, that day only), max_positions (only
when the pocket sets one), too_small, cash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.config import PocketConfig
from autotrade.costs import CostModel
from autotrade.sizing.portfolio import FILL_COLUMNS, SKIP_COLUMNS, PortfolioResult
from autotrade.types import TRADE_COLUMNS

_EXIT, _ENTRY, _LATE_EXIT = 0, 1, 2


def local_day(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    return ts.dt.normalize()


def run_intraday_portfolio(trades: pd.DataFrame, cost_model: CostModel, pocket: PocketConfig,
                           capital: float, currency_per_krw: float,
                           fractional: bool) -> PortfolioResult:
    cap0 = capital * currency_per_krw
    if trades.empty:
        return PortfolioResult(pd.Series(dtype=float, name="equity",
                                         index=pd.DatetimeIndex([], name="date")),
                               pd.DataFrame(columns=FILL_COLUMNS),
                               pd.DataFrame(columns=SKIP_COLUMNS))
    t = trades[TRADE_COLUMNS].reset_index(drop=True)
    day = local_day(t["entry_date"])
    if not (day == local_day(t["exit_date"])).all():
        raise ValueError("intraday trades must exit on their entry day")
    cash = cap0
    fills: list[dict] = []
    skipped: list[dict] = []
    days, eq_values = [], []
    for d, idx in day.groupby(day, sort=True):
        day_start = cash                       # flat book: previous close equity
        events = []
        for i in idx.index:
            same = t.at[i, "exit_date"] <= t.at[i, "entry_date"]
            events.append((t.at[i, "entry_date"], _ENTRY, -float(t.at[i, "confidence"]), i))
            events.append((t.at[i, "exit_date"], _LATE_EXIT if same else _EXIT, 0.0, i))
        events.sort(key=lambda e: e[:3])
        open_pos: dict[int, dict] = {}
        realized = 0.0
        for _, kind, _, i in events:
            row = t.loc[i]
            if kind == _ENTRY:
                qty, buy_cash, reason = _decide(
                    realized, day_start, len(open_pos), cash, row, cost_model, pocket,
                    fractional)
                if reason:
                    skipped.append({**row.to_dict(), "reason": reason})
                else:
                    cash -= buy_cash
                    open_pos[i] = {"qty": qty, "buy_cash": buy_cash}
                continue
            pos = open_pos.pop(i, None)
            if pos is None:
                continue
            proceeds = cost_model.fill_cash(row.symbol, "sell", pos["qty"], row.exit_price)
            cash += proceeds
            pnl = proceeds - pos["buy_cash"]
            realized += pnl
            fills.append({**row.to_dict(), "qty": pos["qty"], "net_pnl": pnl,
                          "net_r": pnl / (pos["qty"] * row.entry_price * row.stop_pct)})
        if open_pos:
            raise ValueError("positions never closed within the day")
        days.append(d)
        eq_values.append(cash)
    equity = pd.Series(eq_values, index=pd.DatetimeIndex(days, name="date"), name="equity")
    fills_df = pd.DataFrame(fills, columns=FILL_COLUMNS).sort_values(
        ["entry_date", "symbol"], kind="stable").reset_index(drop=True)
    return PortfolioResult(equity, fills_df,
                           pd.DataFrame(skipped, columns=SKIP_COLUMNS).reset_index(drop=True))


def _decide(realized: float, day_start: float, n_open: int, cash: float, row,
            cost_model: CostModel, pocket: PocketConfig, fractional: bool
            ) -> tuple[float, float, str | None]:
    if realized <= -pocket.daily_loss_stop * day_start:
        return 0.0, 0.0, "daily_loss_stop"
    if pocket.max_positions is not None and n_open >= pocket.max_positions:
        return 0.0, 0.0, "max_positions"
    risk_qty = day_start * pocket.risk_per_trade * row.confidence / (row.entry_price * row.stop_pct)
    qty = min(risk_qty, day_start * pocket.max_position_weight / row.entry_price)
    if not fractional:
        qty = float(np.floor(qty))
    if not qty > 0:
        return qty, 0.0, "too_small"
    buy_cash = cost_model.fill_cash(row.symbol, "buy", qty, row.entry_price)
    if buy_cash > cash:
        return qty, buy_cash, "cash"
    return qty, buy_cash, None
