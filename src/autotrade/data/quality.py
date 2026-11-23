from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP_BDAYS = 10
ROUNDING_TOL = 0.005    # pykrx adjusted-price rounding; measured up to 0.35% on sub-1,000-won stocks


PRICE = ["open", "high", "low", "close"]


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every row with volume 0: no trade happened that day, so nothing could have been
    filled. This covers suspensions as pykrx returns them (O/H/L=0 with the previous close)
    and as Toss returns them (all prices frozen at the previous close); left in, a frozen
    row would let a backtest buy or sell at a pre-suspension price. Dropped rows become NaN
    in the panel, where positions are held through. Shared by every daily source."""
    return df[df["volume"] != 0]


def fix_rounding(df: pd.DataFrame) -> pd.DataFrame:
    """pykrx adjusted bars are rounded per field, so the high can come out a few won below
    the close (high 876,387 / close 876,388; high 381 / close 382). Widen high/low to cover
    open and close when the overshoot is <= ROUNDING_TOL of the price; larger overshoots are
    left for the quality check to reject. For pykrx sources only, never for Toss bars."""
    out = df.copy()
    top = out[["open", "close"]].max(axis=1)
    bottom = out[["open", "close"]].min(axis=1)
    up = (top > out["high"]) & ((top - out["high"]) <= ROUNDING_TOL * top)
    down = (bottom < out["low"]) & ((out["low"] - bottom) <= ROUNDING_TOL * bottom)
    out.loc[up, "high"] = top[up]
    out.loc[down, "low"] = bottom[down]
    return out


def check_daily(df: pd.DataFrame) -> list[str]:
    """Return problem descriptions for a daily frame (index=date). Empty list means clean."""
    issues: list[str] = []
    nan = df[PRICE + ["volume"]].isna()
    if nan.to_numpy().any():
        first = df.index[nan.any(axis=1).to_numpy()][0]
        issues.append(f"NaN cells: {int(nan.to_numpy().sum())}, first {pd.Timestamp(first).date()}")
    if df.index.duplicated().any():
        issues.append(f"duplicate index: {int(df.index.duplicated().sum())} rows")
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c))
    if bad.any():
        issues.append(f"ohlc inconsistent: {int(bad.sum())} rows, first {df.index[bad.to_numpy()][0].date()}")
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if nonpos.any():
        issues.append(f"non-positive price: {int(nonpos.sum())} rows")
    if (df["volume"] < 0).any():
        issues.append(f"negative volume: {int((df['volume'] < 0).sum())} rows")
    days = df.index.sort_values().unique().values.astype("datetime64[D]")
    if len(days) > 1:
        gaps = np.busday_count(days[:-1], days[1:])
        # consecutive trading days are 1 busday apart; holidays cannot be excluded, so warn only
        if (gaps > MAX_GAP_BDAYS).any():
            i = int(np.argmax(gaps > MAX_GAP_BDAYS))
            issues.append(f"gap > {MAX_GAP_BDAYS} business days: {days[i]} -> {days[i + 1]} (warning)")
    return issues


def hard_issues(df: pd.DataFrame) -> list[str]:
    """Issues from check_daily excluding warnings (entries marked "(warning)")."""
    return [i for i in check_daily(df) if not i.endswith("(warning)")]
