"""KRX regular-session daily bars for KR symbols from NXT_START on (pykrx).

Since NXT (the alternative exchange) opened on 2025-03-04, Toss KR daily bars combine KRX
and NXT trades: the open is NXT's 08:00 pre-market price, the close NXT's 20:00 after-market
price, and volume ~1.5x KRX. The strategies trade the KRX regular session only, so from
NXT_START the stored Toss rows are replaced, at load time, by KRX-only rows kept in a
separate overlay file (data/kr/krx/<symbol>.parquet). The Toss file itself is never changed,
so incremental Toss collects keep working.

Scale check: pykrx and Toss are both adjusted series, but each adjusts on its own. The
overlay is accepted only when the median |krx_close / toss_close - 1| over the shared dates
is <= SCALE_TOL (measured: 0-0.02% on matched symbols; NXT after-hours moves leave the
median near 0 while single days differ by several %). Otherwise no overlay is stored and the
symbol keeps its Toss bars (counted as a mismatch and stated in the report limitations).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from autotrade.data.delisted_kr import CALL_INTERVAL_S, PYKRX_COLUMNS, _pykrx_ohlcv
from autotrade.data.quality import clean_daily
from autotrade.data.store import DAILY_COLUMNS, data_root, load_daily, write_parquet_atomic

NXT_START = pd.Timestamp("2025-03-04")
FETCH_LEAD_DAYS = 45          # also fetch this much before NXT_START: shared pre-NXT dates
SCALE_TOL = 0.005
MIN_SHARED_DAYS = 5
SOURCE = "krx"
log = logging.getLogger(__name__)


@dataclass
class OverlayReport:
    applied: int = 0
    mismatch: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def overlay_path(symbol: str) -> Path:
    return data_root() / "kr" / "krx" / f"{symbol}.parquet"


def load_overlay(symbol: str) -> pd.DataFrame | None:
    path = overlay_path(symbol)
    return pd.read_parquet(path) if path.exists() else None


def fetch_krx_daily(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Adjusted KRX daily bars in [start, end], zero-volume rows dropped, source 'krx'."""
    raw = _pykrx_ohlcv(start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), symbol, adjusted=True)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS, index=pd.DatetimeIndex([], name="date"))
    df = raw.rename(columns=PYKRX_COLUMNS)[list(PYKRX_COLUMNS.values())].astype(float)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    df.index.name = "date"
    df = clean_daily(df)
    df["source"] = SOURCE
    return df[DAILY_COLUMNS]


def same_scale(toss: pd.DataFrame, krx: pd.DataFrame) -> bool:
    """True when the two adjusted series share >= MIN_SHARED_DAYS dates and their median
    relative close difference is <= SCALE_TOL."""
    common = toss.index.intersection(krx.index)
    if len(common) < MIN_SHARED_DAYS:
        return False
    a = toss.loc[common, "close"].to_numpy(dtype=float)
    b = krx.loc[common, "close"].to_numpy(dtype=float)
    return bool(np.median(np.abs(a / b - 1.0)) <= SCALE_TOL)


def refresh_overlay(symbol: str, today: pd.Timestamp) -> str:
    """Refetch one symbol's overlay (always the whole window, so a new adjustment never mixes
    scales). Returns 'applied' | 'mismatch' | 'empty' | 'none' (no Toss bars on or after
    NXT_START, nothing to replace). A stale overlay is removed unless 'applied'."""
    path = overlay_path(symbol)
    toss = load_daily("kr", symbol)
    if toss is None or toss.empty or toss.index.max() < NXT_START:
        path.unlink(missing_ok=True)
        return "none"
    start = (NXT_START - timedelta(days=FETCH_LEAD_DAYS)).date()
    krx = fetch_krx_daily(symbol, start, (today - timedelta(days=1)).date())
    krx = krx[krx.index < today]
    if krx[krx.index >= NXT_START].empty:
        path.unlink(missing_ok=True)
        return "empty"
    if not same_scale(toss, krx):
        path.unlink(missing_ok=True)
        return "mismatch"
    write_parquet_atomic(krx[krx.index >= NXT_START], path)
    return "applied"


def refresh_overlays(symbols: list[str], today: pd.Timestamp) -> OverlayReport:
    report = OverlayReport()
    for i, sym in enumerate(symbols):
        if i:
            time.sleep(CALL_INTERVAL_S)
        try:
            status = refresh_overlay(sym, today)
        except Exception as e:      # noqa: BLE001 - pykrx/KRX errors carry no secrets
            log.warning("krx overlay %s failed: %s: %s", sym, type(e).__name__, e)
            overlay_path(sym).unlink(missing_ok=True)
            report.failed.append(sym)
            continue
        if status == "applied":
            report.applied += 1
        elif status == "mismatch":
            report.mismatch.append(sym)
        elif status == "empty":
            report.empty.append(sym)
    return report


def apply_overlay(toss: pd.DataFrame, overlay: pd.DataFrame | None) -> pd.DataFrame:
    """Toss rows before NXT_START + overlay rows from NXT_START; Toss unchanged without one."""
    if overlay is None or overlay.empty:
        return toss
    out = pd.concat([toss[toss.index < NXT_START], overlay[overlay.index >= NXT_START]])
    out = out[DAILY_COLUMNS].sort_index()
    out.attrs = dict(toss.attrs)
    return out


def overlay_record_path() -> Path:
    return data_root() / "kr" / "krx_overlay.json"


def save_overlay_record(report: OverlayReport, requested: int, day: date | None = None) -> Path:
    """Counts of the last overlay refresh (never prices). Atomic."""
    path = overlay_record_path()
    rec = {"date": (day or date.today()).isoformat(), "from": NXT_START.date().isoformat(),
           "requested": int(requested), "applied": int(report.applied),
           "mismatch": len(report.mismatch), "empty": len(report.empty),
           "failed": len(report.failed)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_overlay_record() -> dict | None:
    try:
        rec = json.loads(overlay_record_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None
