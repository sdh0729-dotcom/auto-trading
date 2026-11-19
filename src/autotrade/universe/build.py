from __future__ import annotations

import re

import numpy as np
import pandas as pd

from autotrade.config import LabConfig, MarketConfig, PocketConfig
from autotrade.types import Panel, Scenario

_ALLOWED_PRODUCTS = {"stock", "etf"}
_MIN_VOLUME_WINDOW = 14


def _tick_frame(close_prev: pd.DataFrame, market: MarketConfig,
                product: pd.Series) -> pd.DataFrame:
    """Vectorized tick size for each (date, symbol), same table as costs.model.tick_size
    (upper bound exclusive). NaN where price or product is unknown."""
    out = pd.DataFrame(np.nan, index=close_prev.index, columns=close_prev.columns)
    for prod, rows in market.tick_table.items():
        cols = product.index[product == prod]
        if len(cols) == 0:
            continue
        uppers = np.array([u for u, _ in rows], dtype=float)
        ticks = np.array([t for _, t in rows], dtype=float)
        prices = close_prev[cols]
        idx = np.searchsorted(uppers, prices.to_numpy(dtype=float), side="right")
        vals = ticks[np.minimum(idx, len(ticks) - 1)]
        out[cols] = np.where(prices.isna().to_numpy(), np.nan, vals)
    return out


def eligible_mask(panel: Panel, market: MarketConfig, pocket: PocketConfig,
                  lab: LabConfig, scenario: Scenario, usd_krw: float) -> pd.DataFrame:
    """Point-in-time eligibility (date x symbol, bool).

    Every value at date t is derived from data through t-1 only: all rolling
    statistics are computed on the full frame and then shifted by one row.

    usd_krw is reserved for scenario conversions (e.g. USD price caps against KRW
    capital); it is currently unused.

    Rules:
    - False if close at t-1 is NaN (not listed / delisted / suspended).
    - Always False: is_common_share is False, name contains an exclude keyword
      (case-insensitive), or product not in {stock, etf}.
    - Rank: top `universe_top_n` by `adv_window`-day mean trading value (close*volume)
      among non-excluded symbols with data, within this market.
    - Filters run only when the market config defines them: tick cost
      (max_tick_cost_pct), min_price on prior close, 14-day mean volume
      (min_avg_volume_14d). Real scenario without fractional orders also caps the
      prior close at capital["real"] * max_position_weight.
    - market.always_include symbols skip rank and filters but not data/exclusions.
    """
    close, volume = panel.close, panel.volume
    cols = close.columns
    meta = panel.meta.reindex(cols)

    close_prev = close.shift(1)
    has_data = close_prev.notna()

    # --- static exclusions (per symbol) ---
    common = meta["is_common_share"].fillna(True).astype(bool)
    allowed = meta["product"].isin(_ALLOWED_PRODUCTS)
    excluded = ~common | ~allowed
    if market.exclude_name_keywords:
        pat = "|".join(re.escape(k) for k in market.exclude_name_keywords)
        excluded |= meta["name"].fillna("").astype(str).str.contains(pat, case=False, regex=True)
    ok_symbol = pd.Series(~excluded.to_numpy(), index=cols)

    # --- rank by trailing trading value, shifted one day ---
    adv = (close * volume).rolling(lab.adv_window).mean().shift(1)
    candidates = adv.where(has_data & ok_symbol.to_numpy()[None, :])
    in_top = candidates.rank(axis=1, ascending=False, method="first") <= lab.universe_top_n

    passes = in_top & has_data

    # --- data-driven filters ---
    if market.max_tick_cost_pct is not None:
        if not market.tick_table:
            raise ValueError(f"market {market.market} sets max_tick_cost_pct without tick_table")
        tick = _tick_frame(close_prev, market, meta["product"])
        passes &= (tick / close_prev) <= market.max_tick_cost_pct
    if market.min_price is not None:
        passes &= close_prev > market.min_price
    if market.min_avg_volume_14d is not None:
        avg_vol = volume.rolling(_MIN_VOLUME_WINDOW).mean().shift(1)
        passes &= avg_vol >= market.min_avg_volume_14d
    if scenario == "real" and not market.fractional_orders:
        passes &= close_prev <= lab.capital["real"] * pocket.max_position_weight

    # --- always-include bypass (still needs data and no exclusion) ---
    always = pd.Series(cols.isin(market.always_include), index=cols)
    passes |= has_data & always.to_numpy()[None, :]

    mask = passes & has_data & ok_symbol.to_numpy()[None, :]
    return mask.fillna(False).astype(bool)
