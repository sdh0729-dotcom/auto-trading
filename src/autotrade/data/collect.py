from __future__ import annotations

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd

from autotrade.data.quality import clean_daily
from autotrade.data.store import PRICE_COLUMNS, load_daily, save_daily, save_meta
from autotrade.toss.client import TossApiError, TossClient
from autotrade.types import TZ, Market

log = logging.getLogger(__name__)

BOARDS = {"kr": ["KOSPI", "KOSDAQ"], "us": ["NYSE", "NASDAQ", "AMEX"]}
PRODUCT = {"STOCK": "stock", "FOREIGN_STOCK": "stock", "ETF": "etf", "FOREIGN_ETF": "etf"}
OHLC = ["open", "high", "low", "close"]
US_TICK = 0.01
REL_TOL = 1e-6
# attrs key on a stored daily file: history is complete from this date (a fetch reached it
# or the API had no older bars), so `existing.index.min() > start` needs no backfill
HISTORY_FROM = "history_from"


class PaginationError(Exception):
    """The candles API returned a `nextBefore` that does not move strictly back in time."""


def _to_local_dates(ts: pd.Series, market: Market) -> pd.DatetimeIndex:
    """Trading date in the market's local calendar (tz-naive, midnight)."""
    local = pd.DatetimeIndex(ts).tz_convert(TZ[market])
    return pd.DatetimeIndex(local.tz_localize(None).normalize(), name="date")


def _tick(market: Market, prices: np.ndarray) -> np.ndarray:
    """Smallest tick over the market's products at each price (US: one cent)."""
    if market != "kr":
        return np.full(prices.shape, US_TICK)
    from autotrade.config import load_market_raw

    out = np.full(prices.shape, np.inf)
    for rows in load_market_raw(market).tick_table.values():
        uppers = np.array([u for u, _ in rows])
        ticks = np.array([t for _, t in rows])
        idx = np.minimum(np.searchsorted(uppers, prices, side="right"), len(ticks) - 1)
        out = np.minimum(out, ticks[idx])
    return out


def same_prices(old: pd.DataFrame, new: pd.DataFrame, market: Market) -> bool:
    """True when both frames share at least one date and every OHLC value on the shared
    dates agrees within max(0.5 tick, 1e-6 relative)."""
    common = old.index.intersection(new.index)
    if len(common) == 0:
        return False
    a = old.loc[common, OHLC].to_numpy(dtype=float)
    b = new.loc[common, OHLC].to_numpy(dtype=float)
    tol = np.maximum(0.5 * _tick(market, np.abs(b)), REL_TOL * np.abs(b))
    return bool((np.abs(a - b) <= tol).all())


def _fetch(client: TossClient, market: Market, symbol: str, stop_at: pd.Timestamp,
           today: pd.Timestamp, before: datetime | None = None
           ) -> tuple[pd.DataFrame | None, bool]:
    """Newest-first pages (from `before`, inclusive) until a page's oldest bar is <= stop_at
    or the history ends. Returns (bars or None, complete) where complete means no bar on or
    after stop_at is missing. Bars dated `today` or later (a session that may still be
    open) are dropped."""
    pages: list[pd.DataFrame] = []
    complete = False
    while True:
        page = client.get_candles(symbol, "1d", before=before, count=200, adjusted=True)
        if len(page.df):
            bars = page.df.copy()
            bars.index = _to_local_dates(bars["timestamp"], market)
            pages.append(bars[PRICE_COLUMNS])
            if bars.index.min() <= stop_at:
                complete = True
                break
        if page.next_before is None:
            complete = True
            break
        if before is not None and page.next_before >= before:
            raise PaginationError(f"{market} {symbol}: nextBefore {page.next_before} is not "
                                  f"before the previous cursor {before}")
        before = page.next_before
    if not pages:
        return None, complete
    new = clean_daily(pd.concat(pages))
    new = new[~new.index.duplicated(keep="first") & (new.index < today)].sort_index()
    return (new if len(new) else None), complete


def market_today(market: Market, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The market's current local session date (tz-naive midnight)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    return now.tz_convert(TZ[market]).tz_localize(None).normalize()


def _local_midnight(day: pd.Timestamp, market: Market) -> datetime:
    return pd.Timestamp(day).tz_localize(TZ[market]).to_pydatetime()


def collect_daily(client: TossClient, market: Market, symbol: str, start: date,
                  today: date | None = None) -> int:
    """Fetch daily bars newer than what is stored (back to `start`); return rows added.

    Bars dated on/after `today` (default: the market's local date) are never stored: that
    session may still be open, so its bar is not final.

    `start` only bounds what is fetched; stored rows older than `start` are never removed.

    The stored bar(s) on the overlap date(s) are compared with the fetched ones: a different
    price (adjusted history rescaled, e.g. by a split) replaces the file by a refetch from
    the earliest of `start` and the stored history start, so one price scale remains. A
    stored history that starts after `start` is backfilled (older pages until `start`)
    unless it is already known complete from `start`."""
    start_ts = pd.Timestamp(start)
    today_ts = market_today(market) if today is None else pd.Timestamp(today)
    existing = load_daily(market, symbol)
    if existing is not None and existing.empty:
        existing = None
    try:
        if existing is None:
            merged = _full(client, market, symbol, start_ts, today_ts)
        else:
            merged = _incremental(client, market, symbol, start_ts, today_ts, existing)
    except TossApiError as e:
        if e.status == 404 and e.code == "stock-not-found":
            log.warning("stock-not-found: %s %s", market, symbol)
            return 0
        raise
    if merged is None:
        return 0
    save_daily(market, symbol, merged)
    return max(0, len(merged) - (len(existing) if existing is not None else 0))


def _finish(bars: pd.DataFrame, history_from) -> pd.DataFrame:
    out = bars.copy()
    out["source"] = out["source"].fillna("toss") if "source" in out.columns else "toss"
    out.attrs = {HISTORY_FROM: history_from} if history_from else {}
    return out


def _full(client, market, symbol, start_ts, today_ts) -> pd.DataFrame | None:
    new, complete = _fetch(client, market, symbol, start_ts, today_ts)
    if new is None:
        return None
    new = new[new.index >= start_ts]
    return _finish(new, start_ts.date().isoformat() if complete else None)


def _refetch_from(existing: pd.DataFrame, start_ts: pd.Timestamp) -> pd.Timestamp:
    """Start of a full refetch that replaces `existing`: nothing already stored is dropped."""
    points = [start_ts, existing.index.min()]
    if existing.attrs.get(HISTORY_FROM):
        points.append(pd.Timestamp(existing.attrs[HISTORY_FROM]))
    return min(points)


def _incremental(client, market, symbol, start_ts, today_ts, existing
                 ) -> pd.DataFrame | None:
    last = existing.index.max()
    new, _ = _fetch(client, market, symbol, last, today_ts)
    if new is None:
        return None
    if not same_prices(existing, new, market):
        log.warning("%s %s: stored prices differ from the API on the overlap (adjustment "
                    "changed, e.g. a split); refetching the full history from %s",
                    market, symbol, _refetch_from(existing, start_ts).date())
        return _full(client, market, symbol, _refetch_from(existing, start_ts), today_ts)
    merged = pd.concat([existing, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    history_from = existing.attrs.get(HISTORY_FROM)
    known = history_from is not None and pd.Timestamp(history_from) <= start_ts
    if merged.index.min() > start_ts and not known:
        older, complete = _fetch(client, market, symbol, start_ts, today_ts,
                                 before=_local_midnight(merged.index.min(), market))
        if older is not None:
            if not same_prices(merged, older, market):
                log.warning("%s %s: older bars do not match the stored scale; refetching "
                            "the full history from %s", market, symbol,
                            _refetch_from(existing, start_ts).date())
                return _full(client, market, symbol, _refetch_from(existing, start_ts),
                             today_ts)
            older = older[older.index >= start_ts]
            merged = pd.concat([older, merged])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if complete:
            history_from = min(start_ts, pd.Timestamp(history_from or start_ts)).date().isoformat()
    return _finish(merged, history_from)


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
    allm = add_etf_info(client, allm)
    save_meta(market, allm)
    return allm


def add_etf_info(client: TossClient, meta: pd.DataFrame) -> pd.DataFrame:
    """leverage_factor (float; 1.0 plain, 2/3 leveraged, negative inverse; NaN unknown) and
    english_name for every ETF, from /api/v1/stocks. The universe excludes ETFs whose
    factor is not 1 (US names from /stocks/all are bare tickers, so name keywords miss
    TQQQ/SPXU). Stocks are not looked up."""
    out = meta.copy()
    out["leverage_factor"] = np.nan
    out["english_name"] = pd.Series(pd.NA, index=out.index, dtype="string")
    etfs = [str(s) for s in out.index[out["product"] == "etf"]]
    for row in client.get_stocks(etfs) if etfs else []:
        sym = str(row.get("symbol"))
        if sym not in out.index:
            continue
        lf = row.get("leverageFactor")
        try:
            out.loc[sym, "leverage_factor"] = float(lf) if lf is not None else np.nan
        except (TypeError, ValueError):
            pass
        out.loc[sym, "english_name"] = row.get("englishName")
    out["english_name"] = out["english_name"].astype("string")
    unknown = int(out.loc[out["product"] == "etf", "leverage_factor"].isna().sum())
    if unknown:
        log.warning("%d ETFs without a leverage factor from /api/v1/stocks", unknown)
    return out
