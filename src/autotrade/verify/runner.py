"""Independent re-execution of a rule's signals in NautilusTrader + parity report.

`run_nautilus` computes `rule.signals` on the panel history through `end` (warm-up
included), keeps entries whose signal row lies in [start, end] and is eligible, and lets
`SignalStrategy` execute them on a BacktestEngine: one cash-account venue per market (KRW /
USD), one equity instrument per symbol, per day an open-price TradeTick followed by the
daily bar (see `signal_strategy` for the event layout and why). The venue runs with
`use_message_queue=False`, so an order submitted from a strategy alert is matched at once
against the book, which holds the open after the open tick and the close after the bar
(with the queue, orders from the last day's alerts would never be processed).

Known, documented differences to the simulator (`parity_report` lists any that occur):
- Prices are put on the instrument's price grid. The grid is the market's (KR 0 decimals,
  US 2) unless the panel's prices need more digits (synthetic/adjusted data), then the
  fewest decimals (<= 9) that represent them. Sell stops are floored onto the grid: they
  trigger on exactly the same grid prices but fill up to one grid step below the
  simulator's unrounded stop price.
- Fees are Money in the account currency (KRW: whole won), so cash can differ from
  `CostModel.fill_cash` by < 1 currency unit per fill.
- Fractional US quantities are floored to 6 decimals. Nautilus' `Equity` fixes size
  precision 0, so fractional instruments are plain `Instrument`s with AssetClass.EQUITY.
- Bars carry a very large volume: like the simulator, fills are never liquidity-capped.
- Suspended days (no price) are held through in both engines; prices that end inside the
  window close `data_end` at the last valid close on that day. Only a close entry on the
  symbol's very last bar (a zero-length `data_end` trade) is sold by Nautilus on the next
  window day (at that last close), so its cash arrives one day later than in
  `run_portfolio`.
"""
from __future__ import annotations

import tempfile
import warnings
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import KRW, USD
from nautilus_trader.model.data import Bar, BarType, TradeTick
from nautilus_trader.model.enums import (AccountType, AggressorSide, AssetClass,
                                         InstrumentClass, OmsType)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TradeId, Venue
from nautilus_trader.model.instruments import Equity, Instrument
from nautilus_trader.model.objects import Money, Price, Quantity

from autotrade.config import PocketConfig
from autotrade.costs import CostModel
from autotrade.rules.base import Rule
from autotrade.sizing import gap_allowance
from autotrade.types import Panel
from autotrade.verify.signal_strategy import (OUTPUT_COLUMNS, SIGNAL_COLUMNS,
                                              CostModelFeeModel, SignalStrategy,
                                              SignalStrategyConfig, session_ns)

VENUES = {"kr": "KRX", "us": "US"}
CURRENCIES = {"KRW": KRW, "USD": USD}
PRICE_PRECISION = {"kr": 0, "us": 2}
MAX_PRICE_PRECISION = 9
FRACTIONAL_SIZE_PRECISION = 6
SESSIONS = {"kr": ("09:00", "15:30"), "us": ("09:30", "16:00")}
BAR_SPEC = "1-DAY-LAST-EXTERNAL"
LIQUIDITY = 10**10  # bar/tick size: fills are never capped by volume (as in the simulator)


def signal_frame(panel: Panel, signals, eligible: pd.DataFrame, start: date,
                 end: date) -> pd.DataFrame:
    """Long frame (SIGNAL_COLUMNS), one row per window day x symbol. `entry` already
    requires eligibility and a valid stop on the signal row; `gap` is the sizing gap
    allowance as of the entry's fill date (NaN where there is no fillable entry);
    `last_bar` marks the symbol's last row with a full OHLC price inside the window."""
    idx, cols = panel.close.index, panel.close.columns
    win = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]

    def num(df):
        return df.reindex(index=win, columns=cols).astype(float)

    def flag(df):
        return df.reindex(index=win, columns=cols).eq(True)

    stop, conf = num(signals.stop_pct), num(signals.confidence)
    with np.errstate(invalid="ignore"):
        entry = flag(signals.entries) & flag(eligible) & np.isfinite(stop) & (stop > 0)
    ohlc = [num(f).to_numpy() for f in (panel.open, panel.high, panel.low, panel.close)]
    valid = np.logical_and.reduce([np.isfinite(x) for x in ohlc])
    later = np.flip(np.cumsum(np.flip(valid, axis=0), axis=0), axis=0) - valid
    last_bar = pd.DataFrame(valid & (later == 0), index=win, columns=cols)
    gap = pd.DataFrame(np.nan, index=win, columns=cols)
    shift = 1 if signals.entry_at == "next_open" else 0
    for r, c in zip(*np.nonzero(entry.to_numpy())):
        if r + shift < len(win):
            gap.iat[r, c] = gap_allowance(panel, cols[c], win[r + shift])
    parts = {"entry": entry, "exit": flag(signals.exits), "stop_pct": stop,
             "confidence": conf, "gap": gap, "last_bar": last_bar}
    long = pd.concat({k: v.stack(future_stack=True) for k, v in parts.items()}, axis=1)
    long.index.names = ["date", "symbol"]
    return long.reset_index()[SIGNAL_COLUMNS]


def _price_precision(panel: Panel, market: str) -> int:
    v = np.concatenate([f.to_numpy(dtype=float).ravel()
                        for f in (panel.open, panel.high, panel.low, panel.close)])
    v = v[np.isfinite(v)]
    for p in range(PRICE_PRECISION[market], MAX_PRICE_PRECISION):
        if np.all(np.abs(np.round(v, p) - v) <= 1e-9 * np.maximum(1.0, np.abs(v))):
            return p
    return MAX_PRICE_PRECISION


def _instrument(iid: InstrumentId, currency, price_prec: int, size_prec: int) -> Instrument:
    tick = Price(10.0**-price_prec, price_prec)
    if size_prec == 0:
        return Equity(instrument_id=iid, raw_symbol=iid.symbol, currency=currency,
                      price_precision=price_prec, price_increment=tick,
                      lot_size=Quantity.from_int(1), ts_event=0, ts_init=0)
    return Instrument(instrument_id=iid, raw_symbol=iid.symbol, asset_class=AssetClass.EQUITY,
                      instrument_class=InstrumentClass.SPOT, quote_currency=currency,
                      is_inverse=False, price_precision=price_prec, size_precision=size_prec,
                      size_increment=Quantity(10.0**-size_prec, size_prec),
                      multiplier=Quantity.from_int(1), margin_init=Decimal(0),
                      margin_maint=Decimal(0), maker_fee=Decimal(0), taker_fee=Decimal(0),
                      ts_event=0, ts_init=0, price_increment=tick)


def _market_data(panel: Panel, win: pd.DatetimeIndex, venue: Venue, price_prec: int,
                 size_prec: int, open_time: str, close_time: str) -> list:
    data = []
    size = Quantity(LIQUIDITY, size_prec)
    for sym in panel.close.columns:
        iid = InstrumentId(Symbol(sym), venue)
        bar_type = BarType.from_str(f"{iid}-{BAR_SPEC}")
        ohlc = [f[sym].reindex(win).to_numpy(dtype=float)
                for f in (panel.open, panel.high, panel.low, panel.close)]
        for i, d in enumerate(win):
            o, h, l, c = (x[i] for x in ohlc)
            if not np.isfinite([o, h, l, c]).all():
                continue
            t0, t1 = session_ns(d, open_time), session_ns(d, close_time)
            px = [Price(x, price_prec) for x in (o, h, l, c)]
            data.append(TradeTick(iid, px[0], size, AggressorSide.NO_AGGRESSOR,
                                  TradeId(f"{sym}-{i}"), t0, t0))
            data.append(Bar(bar_type, *px, size, t1, t1))
    return data


def run_nautilus(rule: Rule, panel: Panel, eligible: pd.DataFrame, start: date, end: date,
                 cost_model: CostModel, pocket: PocketConfig, capital: float,
                 currency_per_krw: float, fractional: bool) -> pd.DataFrame:
    """Trades (TRADE_COLUMNS + qty, net_pnl) of `rule` executed by SignalStrategy in
    NautilusTrader; `attrs["final_balance"]` is the venue account's final cash."""
    hist = panel.slice(panel.close.index[0], end)
    signals = rule.signals(hist)
    frame = signal_frame(hist, signals, eligible, start, end)
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    market = cost_model.market
    venue = Venue(VENUES[market.market])
    currency = CURRENCIES[market.currency]
    open_time, close_time = (market.session_open or SESSIONS[market.market][0],
                             market.session_close or SESSIONS[market.market][1])
    win = pd.DatetimeIndex(frame["date"].unique())
    price_prec = _price_precision(hist.slice(win[0], win[-1]), market.market)
    size_prec = FRACTIONAL_SIZE_PRECISION if fractional else 0

    engine = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(bypass_logging=True)))
    try:
        engine.add_venue(venue, OmsType.NETTING, AccountType.CASH,
                         [Money(capital * currency_per_krw, currency)], base_currency=currency,
                         fee_model=CostModelFeeModel(cost_model), bar_execution=True,
                         trade_execution=True, bar_adaptive_high_low_ordering=False,
                         use_message_queue=False)  # orders hit the book when submitted
        for sym in hist.close.columns:
            engine.add_instrument(_instrument(InstrumentId(Symbol(sym), venue), currency,
                                              price_prec, size_prec))
        engine.add_data(_market_data(hist, win, venue, price_prec, size_prec,
                                     open_time, close_time))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signals.parquet"
            frame.to_parquet(path)
            config = SignalStrategyConfig(
                signals_path=str(path), venue=venue.value,
                symbols=list(hist.close.columns), entry_at=signals.entry_at,
                exit_at=signals.exit_at, max_hold_days=signals.max_hold_days,
                risk_per_trade=pocket.risk_per_trade,
                max_position_weight=pocket.max_position_weight,
                daily_loss_stop=pocket.daily_loss_stop, max_positions=pocket.max_positions,
                drawdown_stop=pocket.drawdown_stop, fractional=fractional,
                open_time=open_time, close_time=close_time, bar_spec=BAR_SPEC)
            strategy = SignalStrategy(config, cost_model)
        engine.add_strategy(strategy)
        with warnings.catch_warnings():  # third-party: engine calls pd.Timestamp.utcnow()
            warnings.filterwarnings("ignore", message="Timestamp.utcnow is deprecated")
            # run past the last close so that day's exit/mark alerts (close_ns + k) fire
            engine.run(end=pd.Timestamp(session_ns(win[-1], close_time) + 10, tz="UTC"))
        if strategy.errors:
            raise RuntimeError("Nautilus orders failed: " + "; ".join(strategy.errors[:5]))
        if strategy.open_symbols:
            raise RuntimeError(f"positions left open: {strategy.open_symbols}")
        trades = strategy.trades_frame()
        trades.attrs["final_balance"] = \
            engine.portfolio.account(venue).balance_total(currency).as_double()
        return trades
    finally:
        engine.dispose()


def parity_report(ours: pd.DataFrame, naut: pd.DataFrame, price_rtol: float = 1e-6,
                  qty_rtol: float = 1e-4, pnl_atol: float = 1.0,
                  pnl_rtol: float = 1e-3) -> dict:
    """Compare simulator+portfolio fills with Nautilus trades, matched on (symbol,
    entry_date). Mismatches beyond tolerance are listed per field; unmatched trades as
    field "trade". net_pnl tolerance is max(pnl_atol, pnl_rtol * |ours|)."""
    key = ["symbol", "entry_date"]
    o = ours.set_index(key)
    n = naut.set_index(key)
    common = o.index.intersection(n.index)
    mismatches: list[dict] = []

    def add(k, field, a, b, exit_date):
        mismatches.append({"symbol": k[0], "entry_date": k[1], "exit_date": exit_date,
                           "field": field, "ours": a, "naut": b})

    for k in o.index.difference(n.index):
        add(k, "trade", "present", None, o.at[k, "exit_date"])
    for k in n.index.difference(o.index):
        add(k, "trade", None, "present", n.at[k, "exit_date"])
    agree = 0
    for k in common:
        a, b = o.loc[k], n.loc[k]
        agree += a["exit_reason"] == b["exit_reason"]
        for f in ("exit_date", "exit_reason"):
            if a[f] != b[f]:
                add(k, f, a[f], b[f], a["exit_date"])
        for f in ("entry_price", "exit_price"):
            if not np.isclose(a[f], b[f], rtol=price_rtol, atol=0.0):
                add(k, f, a[f], b[f], a["exit_date"])
        if not np.isclose(a["qty"], b["qty"], rtol=qty_rtol, atol=1e-6):
            add(k, "qty", a["qty"], b["qty"], a["exit_date"])
        if abs(a["net_pnl"] - b["net_pnl"]) > max(pnl_atol, pnl_rtol * abs(a["net_pnl"])):
            add(k, "net_pnl", a["net_pnl"], b["net_pnl"], a["exit_date"])
    pnl_o, pnl_n = float(ours["net_pnl"].sum()), float(naut["net_pnl"].sum())
    denom = max(len(o), len(n))
    return {
        "n_ours": len(o), "n_naut": len(n), "matched": len(common),
        "matched_share": len(common) / denom if denom else 1.0,
        "exit_reason_agreement": agree / len(common) if len(common) else float("nan"),
        "net_pnl_ours": pnl_o, "net_pnl_naut": pnl_n, "net_pnl_diff": pnl_n - pnl_o,
        "net_pnl_rel_diff": (pnl_n - pnl_o) / abs(pnl_o) if pnl_o else float("nan"),
        "mismatches": mismatches,
    }
