"""Technical indicators. Every value at date t uses data through t only (current bar included).

Conventions:
- ATR(n): simple rolling mean of True Range over n days; TR = max(H-L, |H-prevC|, |L-prevC|).
  The first row's TR falls back to H-L (no prior close).
- RSI(n): Wilder smoothing (EWM alpha=1/n) of gains/losses; monotonic rise -> 100, fall -> 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.types import Panel


def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).mean()


def rolling_high(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).max()


def rolling_low(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).min()


def atr(panel: Panel, n: int) -> pd.DataFrame:
    prev = panel.close.shift(1)
    tr = np.fmax(np.fmax(panel.high - panel.low, (panel.high - prev).abs()),
                 (panel.low - prev).abs())
    return tr.rolling(n).mean()


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return out
