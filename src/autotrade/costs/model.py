from __future__ import annotations

from typing import Callable

import pandas as pd

from autotrade.config import MarketConfig
from autotrade.types import TRADE_COLUMNS, Product


def tick_size(price: float, market: MarketConfig, product: Product) -> float:
    """Tick for `price` from the market's tick table (upper bound exclusive)."""
    if not market.tick_table:
        raise ValueError(f"market {market.market} has no tick table")
    for upper, tick in market.tick_table[product]:
        if price < upper:
            return tick
    raise ValueError("tick table must end with an infinite bound")


def cost_in_r(round_trip_pct: float, stop_pct: float) -> float:
    return round_trip_pct / stop_pct


class CostModel:
    """Per-market costs. Slippage model is chosen by which config fields are set:
    slippage_ticks -> ticks x tick size; slippage_min_pct/abs -> max(pct, abs)."""

    def __init__(self, market: MarketConfig, product_of: Callable[[str], Product]):
        if market.commission_rate is None:
            raise ValueError("commission_rate is not synced; run `lab sync` first")
        if market.slippage_ticks is None and market.slippage_min_pct is None \
                and market.slippage_min_abs is None:
            raise ValueError(f"market {market.market} defines no slippage model")
        self.market = market
        self.product_of = product_of
        self.commission = market.commission_rate
        self.sec_fee = market.sec_fee_rate or 0.0

    def _slippage(self, symbol: str, price: float) -> float:
        m = self.market
        if m.slippage_ticks is not None:
            return m.slippage_ticks * tick_size(price, m, self.product_of(symbol))
        return max(price * (m.slippage_min_pct or 0.0), m.slippage_min_abs or 0.0)

    def _sell_tax(self, symbol: str) -> float:
        return self.market.sell_tax_rate[self.product_of(symbol)]

    def buy_fill(self, symbol: str, price: float) -> float:
        return price + self._slippage(symbol, price)

    def sell_fill(self, symbol: str, price: float) -> float:
        return price - self._slippage(symbol, price)

    def round_trip_cost_pct(self, symbol: str, price: float) -> float:
        slip = self._slippage(symbol, price) * 2
        return 2 * self.commission + self._sell_tax(symbol) + self.sec_fee + slip / price

    def net_returns(self, trades: pd.DataFrame) -> pd.DataFrame:
        out = trades.copy()
        gross, net = [], []
        for t in trades.itertuples(index=False):
            buy = self.buy_fill(t.symbol, t.entry_price)
            sell = self.sell_fill(t.symbol, t.exit_price)
            proceeds = sell * (1 - self.commission - self._sell_tax(t.symbol) - self.sec_fee)
            net.append(proceeds / (buy * (1 + self.commission)) - 1)
            gross.append(t.exit_price / t.entry_price - 1)
        out["gross_return"] = pd.Series(gross, index=out.index, dtype=float)
        out["net_return"] = pd.Series(net, index=out.index, dtype=float)
        out["net_r"] = out["net_return"] / out["stop_pct"]
        return out
