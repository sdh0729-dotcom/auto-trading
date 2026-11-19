from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.types import META_COLUMNS, Panel, Product


def make_panel(n_days: int, symbols: list[str], seed: int,
               products: dict[str, Product] | None = None, start: str = "2016-01-04",
               **overrides) -> Panel:
    """Synthetic random-walk OHLCV panel on a tz-naive business-day index.

    overrides: meta column -> {symbol: value}, e.g. name={"A": "스팩1호"}.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days, name="date")
    shape = (n_days, len(symbols))
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, shape), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, shape))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, shape)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, shape)))
    vol = rng.integers(100_000, 5_000_000, shape).astype(float)

    def frame(a):
        return pd.DataFrame(a, index=idx, columns=symbols)

    products = products or {}
    prod = {s: products.get(s, "stock") for s in symbols}
    meta = pd.DataFrame({
        "name": {s: f"Name {s}" for s in symbols},
        "security_type": {s: "ETF" if prod[s] == "etf" else "STOCK" for s in symbols},
        "product": prod,
        "is_common_share": {s: True for s in symbols},
        "delisted": {s: False for s in symbols},
        "source": {s: "synthetic" for s in symbols},
    }, index=pd.Index(symbols, name="symbol"))
    for col, mapping in overrides.items():
        if col not in META_COLUMNS:
            raise ValueError(f"unknown meta column {col}")
        for s, v in mapping.items():
            meta.loc[s, col] = v
    return Panel(frame(open_), frame(hi), frame(lo), frame(close), frame(vol), meta)
