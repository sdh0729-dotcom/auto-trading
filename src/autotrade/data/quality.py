from __future__ import annotations

import numpy as np
import pandas as pd

MAX_GAP_BDAYS = 10


def check_daily(df: pd.DataFrame) -> list[str]:
    """Return problem descriptions for a daily frame (index=date). Empty list means clean."""
    issues: list[str] = []
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
