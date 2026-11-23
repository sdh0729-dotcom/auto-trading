"""NautilusTrader strategy that executes precomputed rule signals on daily data.

The strategy never evaluates rule logic: it reads a long signal frame (one row per window
trading day x symbol: entry, exit, stop_pct, confidence, gap, last_bar) and turns it into
orders. The same strategy is meant to execute live/paper signals later.

Daily event layout (all timestamps are the trading date + local session time, as UTC ns):
- `open_ns`: one TradeTick per symbol at the day's open. The matching engine fills resting
  stop orders on it first -> a stop filled at `open_ns` is a `gap_stop` (fills at the open).
- `open_ns + 1` alert: pending exit_at="next_open" signals sell at market (= the open).
- `open_ns + 2` alert: entry_at="next_open" entries buy at market (= the open), largest
  confidence first, sized by `decide_entry` (shared with `run_portfolio`). After the fill a
  STOP_MARKET sell is placed, so the entry bar is stop-checked from its high/low on.
- `close_ns`: the daily Bar (1-DAY-LAST-EXTERNAL). Bar execution walks O->H->L->C; a stop
  crossed intrabar fills at its trigger price -> `stop`.
- `close_ns + 1` alert: exit_at="close" signal, max_hold, period_end (window's last day) and
  data_end (`last_bar`: the symbol's last price inside the window) sell at market (= the
  close); otherwise a next_open exit signal is registered as pending.
- `close_ns + 2` alert: entry_at="close" entries buy at the close (not on the last day).
- `close_ns + 3` alert: equity is marked (account cash + positions x close) for the next
  day's risk gates.

Portfolio semantics mirror `run_portfolio` on `simulate` candidates: a candidate that sizing
skips still blocks its symbol until the exit it would have had (tracked as a "phantom" with
the same exit rules, without orders).

Days without a price for a held symbol (suspension) are held through like in `simulate`: no
tick/bar arrives, so the resting stop cannot fill, no exit is checked and a pending
next_open exit stays pending; max_hold keeps counting window days. The resting stop then
meets the reopening open first (`gap_stop`). The one case the close alerts cannot reach is
a close entry on the symbol's last bar: it is sold `data_end` (labelled with that day, at
the book's last close) at the next day's close alert.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, TradeTick
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

from autotrade.config import PocketConfig
from autotrade.costs import CostModel
from autotrade.sizing import RiskState, decide_entry
from autotrade.types import EqualWeight
from autotrade.types import TRADE_COLUMNS

SIGNAL_COLUMNS = ["date", "symbol", "entry", "exit", "stop_pct", "confidence", "gap",
                  "last_bar"]
OUTPUT_COLUMNS = TRADE_COLUMNS + ["qty", "net_pnl"]
DAY_NS = 86_400 * 10**9


def session_ns(day: pd.Timestamp, hhmm: str) -> int:
    """`day` (midnight) + local session time "HH:MM", as integer ns."""
    h, m = (int(x) for x in hhmm.split(":"))
    return pd.Timestamp(day).normalize().value + (h * 3600 + m * 60) * 10**9


class CostModelFeeModel(FeeModel):
    """Charges per fill exactly what `CostModel.fill_cash` adds/removes beyond the notional.

    buy  fee = fill_cash(buy)  - qty*px = qty*px*commission + qty*slip*(1 + commission)
    sell fee = qty*px - fill_cash(sell) = qty*px*(commission + sell_tax + sec_fee)
                                          + qty*slip*(1 - commission - sell_tax - sec_fee)
    Slippage is charged as part of the fee (fills stay at bar prices) so cash matches
    `fill_cash` up to the currency's Money precision (KRW: 1 won)."""

    def __init__(self, cost_model: CostModel):
        super().__init__()
        self._cm = cost_model

    def get_commission(self, order, fill_qty, fill_px, instrument):
        qty, px = fill_qty.as_double(), fill_px.as_double()
        symbol = instrument.id.symbol.value
        if order.side == OrderSide.BUY:
            fee = self._cm.fill_cash(symbol, "buy", qty, px) - qty * px
        else:
            fee = qty * px - self._cm.fill_cash(symbol, "sell", qty, px)
        return Money(fee, instrument.quote_currency)


class SignalStrategyConfig(StrategyConfig, frozen=True):
    signals_path: str                    # parquet with SIGNAL_COLUMNS
    venue: str
    symbols: list[str]
    entry_at: str                        # "next_open" | "close"
    exit_at: str                         # "next_open" | "close"
    max_hold_days: int
    risk_per_trade: float
    max_position_weight: float
    daily_loss_stop: float
    max_positions: int | None = None
    drawdown_stop: float | None = None
    fractional: bool = False
    equal_slots: int | None = None       # Signals.equal_weight (rotation sizing), else risk
    equal_max_weight: float | None = None
    open_time: str = "09:00"
    close_time: str = "15:30"
    bar_spec: str = "1-DAY-LAST-EXTERNAL"


@dataclass
class _Trade:
    symbol: str
    entry_row: int
    stop_pct: float
    confidence: float
    live: bool
    entry_price: float = math.nan       # phantom: signal price; live: average fill price
    stop_price: float = math.nan
    target_qty: float = 0.0
    qty: float = 0.0
    entry_value: float = 0.0
    entry_fee: float = 0.0
    exit_qty: float = 0.0
    exit_value: float = 0.0
    exit_fee: float = 0.0
    pending: bool = False               # exit_at="next_open" signal waiting for next open
    exiting: str | None = None          # reason of a submitted exit order
    exit_row: int | None = None
    stop_order: object = None


class SignalStrategy(Strategy):
    def __init__(self, config: SignalStrategyConfig, cost_model: CostModel):
        super().__init__(config)
        if config.entry_at not in ("next_open", "close") or \
                config.exit_at not in ("next_open", "close"):
            raise ValueError("entry_at/exit_at must be 'next_open' or 'close'")
        self.cm = cost_model
        self.venue = Venue(config.venue)
        self.pocket = PocketConfig(config.risk_per_trade, config.max_position_weight,
                                   config.daily_loss_stop, config.max_positions,
                                   config.drawdown_stop)
        self._equal_weight = None if config.equal_slots is None else \
            EqualWeight(config.equal_slots, config.equal_max_weight)
        frame = pd.read_parquet(config.signals_path)
        frame = frame[frame["symbol"].isin(config.symbols)]
        self.dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        self.symbols = sorted(config.symbols)
        self._row_of_day = {d.value: i for i, d in enumerate(self.dates)}
        self._open_ns = [session_ns(d, config.open_time) for d in self.dates]
        self._sig = {}
        for sym in self.symbols:
            g = frame[frame["symbol"] == sym].set_index("date").reindex(self.dates)
            self._sig[sym] = {
                "entry": g["entry"].eq(True).to_numpy(),
                "exit": g["exit"].eq(True).to_numpy(),
                "stop_pct": g["stop_pct"].to_numpy(dtype=float),
                "confidence": g["confidence"].to_numpy(dtype=float),
                "gap": g["gap"].to_numpy(dtype=float),
                "last_bar": g["last_bar"].eq(True).to_numpy(),
            }
        self._iid = {s: InstrumentId(Symbol(s), self.venue) for s in self.symbols}
        self._pos: dict[str, _Trade] = {}
        self._last_exit_row: dict[str, int] = {}
        self._open_px: dict[str, tuple[int, float]] = {}
        self._bar: dict[str, tuple[int, float, float]] = {}       # row, low, close
        self._orders: dict = {}                                   # client_order_id -> (kind, trade)
        self._risk: RiskState | None = None
        self.trades: list[dict] = []
        self.errors: list[str] = []

    # ------------------------------------------------------------------ lifecycle / data
    def on_start(self) -> None:
        for sym, iid in self._iid.items():
            self.subscribe_trade_ticks(iid)
            self.subscribe_bars(BarType.from_str(f"{iid}-{self.config.bar_spec}"))
        steps = [(self.config.open_time, 1, self._open_exits),
                 (self.config.open_time, 2, self._open_entries),
                 (self.config.close_time, 1, self._close_exits),
                 (self.config.close_time, 2, self._close_entries),
                 (self.config.close_time, 3, self._mark)]
        for i, d in enumerate(self.dates):
            for hhmm, k, cb in steps:
                self.clock.set_time_alert_ns(f"{cb.__name__}-{i}", session_ns(d, hhmm) + k, cb)

    def on_trade_tick(self, tick: TradeTick) -> None:
        self._open_px[tick.instrument_id.symbol.value] = (self._row(tick.ts_init),
                                                          tick.price.as_double())

    def on_bar(self, bar: Bar) -> None:
        self._bar[bar.bar_type.instrument_id.symbol.value] = (
            self._row(bar.ts_init), bar.low.as_double(), bar.close.as_double())

    def _row(self, ts_ns: int) -> int:
        return self._row_of_day[ts_ns - ts_ns % DAY_NS]

    @staticmethod
    def _event_row(event) -> int:
        return int(event.name.rsplit("-", 1)[1])

    # ------------------------------------------------------------------ daily steps
    def _open_exits(self, event) -> None:
        i = self._event_row(event)
        if self._risk is None:
            self._risk = RiskState(self.pocket, self._cash())
        self._risk.begin_day()
        for sym, t in list(self._pos.items()):
            if t.exiting or self._open_px.get(sym, (-1,))[0] != i:
                continue  # exit already working / no bar today (handled at the close)
            if t.live:
                if t.pending:
                    self._exit(t, "signal", i)
            else:
                o = self._open_px[sym][1]
                if o <= t.stop_price or t.pending:
                    self._close_phantom(t, i)

    def _open_entries(self, event) -> None:
        i = self._event_row(event)
        if self.config.entry_at != "next_open" or i == 0:
            return
        cands = [s for s in self.symbols if self._sig[s]["entry"][i - 1]
                 and self._can_enter(s, i) and self._open_px.get(s, (-1,))[0] == i]
        self._enter(cands, i - 1, i, {s: self._open_px[s][1] for s in cands})

    def _close_exits(self, event) -> None:
        i = self._event_row(event)
        last = i == len(self.dates) - 1
        for sym, t in list(self._pos.items()):
            if t.exiting:
                continue
            bar = self._bar.get(sym)
            if bar is None or bar[0] != i:          # no price today: held through ...
                last_row = bar[0] if bar is not None else t.entry_row
                if self._sig[sym]["last_bar"][last_row]:   # ... unless prices ended there
                    if t.live:
                        self._exit(t, "data_end", last_row)
                    else:
                        self._close_phantom(t, last_row)
                continue
            if not t.live and bar[1] <= t.stop_price:
                self._close_phantom(t, i)
                continue
            sig = self._sig[sym]
            reason = None
            if self.config.exit_at == "close" and sig["exit"][i]:
                reason = "signal"
            elif i - t.entry_row >= self.config.max_hold_days:
                reason = "max_hold"
            elif last:
                reason = "period_end"
            elif sig["last_bar"][i]:
                reason = "data_end"
            if reason is None:
                t.pending = self.config.exit_at == "next_open" and bool(sig["exit"][i])
            elif t.live:
                self._exit(t, reason, i)
            else:
                self._close_phantom(t, i)

    def _close_entries(self, event) -> None:
        i = self._event_row(event)
        if self.config.entry_at != "close" or i == len(self.dates) - 1:
            return
        cands = [s for s in self.symbols if self._sig[s]["entry"][i]
                 and self._can_enter(s, i) and self._bar.get(s, (-1,))[0] == i]
        self._enter(cands, i, i, {s: self._bar[s][2] for s in cands})

    def _mark(self, event) -> None:
        equity = self._cash()
        for sym, iid in self._iid.items():
            qty = float(self.portfolio.net_position(iid))
            if qty:
                equity += qty * self._bar[sym][2]
        self._risk.end_day(equity)

    # ------------------------------------------------------------------ helpers
    def _cash(self) -> float:
        ccy = self.cache.instrument(self._iid[self.symbols[0]]).quote_currency
        return self.portfolio.account(self.venue).balance_total(ccy).as_double()

    def _can_enter(self, sym: str, fill_row: int) -> bool:
        return sym not in self._pos and self._last_exit_row.get(sym, -1) < fill_row

    def _enter(self, cands: list[str], sig_row: int, fill_row: int, price: dict) -> None:
        def key(s):
            c = self._sig[s]["confidence"][sig_row]
            return (0, -c, s) if np.isfinite(c) else (1, 0.0, s)
        cash = self._cash()
        n_open = sum(t.live for t in self._pos.values())
        for sym in sorted(cands, key=key):
            sig = self._sig[sym]
            sp, conf, gap = (float(sig[k][sig_row]) for k in ("stop_pct", "confidence", "gap"))
            px = price[sym]
            qty, buy_cash, reason = decide_entry(self._risk, n_open, cash, sym, px, sp, conf,
                                                 gap, self.cm, self.config.fractional,
                                                 self._equal_weight)
            pending = self.config.entry_at == "close" and self.config.exit_at == "next_open" \
                and bool(sig["exit"][sig_row])
            inst = self.cache.instrument(self._iid[sym])
            q = Quantity(_floor(qty, inst.size_precision), inst.size_precision)
            if reason is None and not q.as_double() > 0:
                reason = "too_small"              # fractional qty below the size grid
            t = _Trade(sym, fill_row, sp, conf, live=reason is None, pending=pending)
            self._pos[sym] = t
            if reason:                            # skipped: phantom blocks the symbol
                t.entry_price, t.stop_price = px, px * (1 - sp)
                continue
            t.target_qty = q.as_double()
            order = self.order_factory.market(self._iid[sym], OrderSide.BUY, q)
            self._orders[order.client_order_id] = ("entry", t)
            self.submit_order(order)
            cash -= buy_cash
            n_open += 1

    def _exit(self, t: _Trade, reason: str, exit_row: int) -> None:
        t.exiting, t.exit_row = reason, exit_row
        if t.stop_order is not None and self.cache.order(t.stop_order).is_open:
            self.cancel_order(self.cache.order(t.stop_order))
        inst = self.cache.instrument(self._iid[t.symbol])
        order = self.order_factory.market(self._iid[t.symbol], OrderSide.SELL,
                                          Quantity(t.qty, inst.size_precision),
                                          reduce_only=True)
        self._orders[order.client_order_id] = ("exit", t)
        self.submit_order(order)

    def _close_phantom(self, t: _Trade, exit_row: int) -> None:
        del self._pos[t.symbol]
        self._last_exit_row[t.symbol] = exit_row

    # ------------------------------------------------------------------ order events
    def on_order_filled(self, event) -> None:
        kind, t = self._orders.get(event.client_order_id, (None, None))
        if t is None:
            return
        qty, px = event.last_qty.as_double(), event.last_px.as_double()
        fee = event.commission.as_double()
        if kind == "entry":
            t.qty += qty
            t.entry_value += qty * px
            t.entry_fee += fee
            if t.qty >= t.target_qty - 1e-12:
                self._place_stop(t)
            return
        if kind == "stop" and t.exiting is None:
            row = self._row(event.ts_event)
            t.exiting = "gap_stop" if event.ts_event == self._open_ns[row] else "stop"
            t.exit_row = row
        t.exit_qty += qty
        t.exit_value += qty * px
        t.exit_fee += fee
        if t.exit_qty >= t.qty - 1e-12:
            self._record(t)

    def _place_stop(self, t: _Trade) -> None:
        inst = self.cache.instrument(self._iid[t.symbol])
        t.entry_price = t.entry_value / t.qty
        t.stop_price = t.entry_price * (1 - t.stop_pct)
        trigger = Price(_floor(t.stop_price, inst.price_precision), inst.price_precision)
        order = self.order_factory.stop_market(
            self._iid[t.symbol], OrderSide.SELL, Quantity(t.qty, inst.size_precision), trigger,
            time_in_force=TimeInForce.GTC, reduce_only=True)
        t.stop_order = order.client_order_id
        self._orders[order.client_order_id] = ("stop", t)
        self.submit_order(order)

    def _record(self, t: _Trade) -> None:
        if t.stop_order is not None and self.cache.order(t.stop_order).is_open:
            self.cancel_order(self.cache.order(t.stop_order))
        self.trades.append({
            "symbol": t.symbol, "entry_date": self.dates[t.entry_row],
            "entry_price": t.entry_price, "exit_date": self.dates[t.exit_row],
            "exit_price": t.exit_value / t.exit_qty, "exit_reason": t.exiting,
            "stop_pct": t.stop_pct, "confidence": t.confidence, "qty": t.qty,
            "net_pnl": (t.exit_value - t.exit_fee) - (t.entry_value + t.entry_fee),
        })
        del self._pos[t.symbol]
        self._last_exit_row[t.symbol] = t.exit_row

    def on_order_rejected(self, event) -> None:
        self.errors.append(f"rejected: {event}")

    def on_order_denied(self, event) -> None:
        self.errors.append(f"denied: {event}")

    @property
    def open_symbols(self) -> list[str]:
        """Symbols still holding a live or phantom position."""
        return sorted(self._pos)

    def trades_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.trades, columns=OUTPUT_COLUMNS)
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        for k in ("entry_price", "exit_price", "stop_pct", "confidence", "qty", "net_pnl"):
            df[k] = df[k].astype(float)
        df["symbol"] = df["symbol"].astype(object)
        df["exit_reason"] = df["exit_reason"].astype(object)
        return df.sort_values(["entry_date", "symbol"], kind="stable").reset_index(drop=True)


def _floor(x: float, precision: int) -> float:
    """Floor to `precision` decimals (a sell stop floored onto the price grid triggers on
    exactly the grid prices <= the unrounded stop)."""
    scale = 10 ** precision
    return math.floor(round(x * scale, 6)) / scale
