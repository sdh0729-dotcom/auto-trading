from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from autotrade.data.store import PRICE_COLUMNS, load_daily, save_daily, save_meta
from autotrade.toss.client import TossApiError, TossClient
from autotrade.types import Market

log = logging.getLogger(__name__)

TZ = {"kr": "Asia/Seoul", "us": "America/New_York"}
BOARDS = {"kr": ["KOSPI", "KOSDAQ"], "us": ["NYSE", "NASDAQ", "AMEX"]}
PRODUCT = {"STOCK": "stock", "FOREIGN_STOCK": "stock", "ETF": "etf", "FOREIGN_ETF": "etf"}


def _to_local_dates(ts: pd.Series, market: Market) -> pd.DatetimeIndex:
    """Trading date in the market's local calendar (tz-naive, midnight)."""
    local = pd.DatetimeIndex(ts).tz_convert(TZ[market])
    return pd.DatetimeIndex(local.tz_localize(None).normalize(), name="date")


def collect_daily(client: TossClient, market: Market, symbol: str, start: date) -> int:
    """Fetch daily bars newer than what is stored (back to `start`); return rows added."""
    existing = load_daily(market, symbol)
    last = existing.index.max() if existing is not None and len(existing) else None
    start_ts = pd.Timestamp(start)
    pages: list[pd.DataFrame] = []
    before = None
    try:
        while True:
            page = client.get_candles(symbol, "1d", before=before, count=200, adjusted=True)
            if len(page.df):
                bars = page.df.copy()
                bars.index = _to_local_dates(bars["timestamp"], market)
                bars = bars[PRICE_COLUMNS]
                pages.append(bars)
                oldest = bars.index.min()
                if (last is not None and oldest <= last) or oldest <= start_ts:
                    break
            if page.next_before is None:
                break
            before = page.next_before
    except TossApiError as e:
        if e.status == 404 and e.code == "stock-not-found":
            log.warning("stock-not-found: %s %s", market, symbol)
            return 0
        raise
    if not pages:
        return 0
    new = pd.concat(pages)
    new = new[new.index >= start_ts]
    new["source"] = "toss"
    merged = pd.concat([existing, new]) if existing is not None else new
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    save_daily(market, symbol, merged)
    return len(merged) - (len(existing) if existing is not None else 0)


def collect_meta(client: TossClient, market: Market) -> pd.DataFrame:
    """Instrument master for a market (ACTIVE + DELISTED), indexed by symbol; saved to store."""
    parts = []
    for board in BOARDS[market]:
        for status in ("ACTIVE", "DELISTED"):
            df = client.list_stocks(board, status=status).copy()
            df["delisted"] = status == "DELISTED"
            df["market_board"] = board
            parts.append(df)
    allm = pd.concat(parts, ignore_index=True)
    allm = allm[allm["security_type"].isin(PRODUCT)].copy()
    allm["product"] = allm["security_type"].map(PRODUCT)
    allm["source"] = "toss"
    allm = allm.drop_duplicates("symbol", keep="first").set_index("symbol").sort_index()
    cols = ["name", "security_type", "product", "is_common_share", "delisted", "source",
            "market_board", "isin"]
    allm = allm[cols]
    save_meta(market, allm)
    return allm
