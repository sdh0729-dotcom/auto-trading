"""Resumable 1-minute bar collector (Toss candles, newest-first pages) into the minute store.

Timestamps are bar END times in the market tz. Bars of the market's current session date are
never stored. A symbol is collected in two parts: a forward part (now back to the latest
stored bar; saved once complete, so the stored range never has a gap) and a backward part
(earliest stored bar back to `since`; saved page by page, so an abort resumes without a gap).
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, replace
from datetime import date

import pandas as pd

from autotrade.data.collect import PaginationError, same_prices
from autotrade.data.minute_store import (MINUTE_COLUMNS, load_minutes, minute_bounds,
                                         minute_dir, save_minutes)
from autotrade.toss.client import TossClient
from autotrade.types import TZ, Market

log = logging.getLogger(__name__)

COMPLETE_MARKER = "_complete"
PAGE = 200


class PriceScaleError(Exception):
    """Prices keep changing scale between calls: refetching once did not settle them."""


@dataclass
class MinuteReport:
    added: int
    complete: bool


def _page_frame(page, market: Market) -> pd.DataFrame:
    df = page.df
    if df.empty:
        return pd.DataFrame(columns=MINUTE_COLUMNS,
                            index=pd.DatetimeIndex([], name="ts", tz=TZ[market]))
    idx = pd.DatetimeIndex(df["timestamp"]).tz_convert(TZ[market])
    out = df[MINUTE_COLUMNS].copy()
    out.index = pd.DatetimeIndex(idx, name="ts")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.astype(float)


def _local_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(idx.tz_localize(None).normalize())


def _same_scale(market: Market, symbol: str, new: pd.DataFrame, hi: pd.Timestamp) -> bool:
    """True unless the stored bars on timestamps shared with `new` carry other prices."""
    if new.empty:
        return True
    stored = load_minutes(market, symbol, new.index.min().date(), hi.date(),
                          regular_only=False)
    common = stored.index.intersection(new.index)
    if len(common) == 0:
        return True
    return same_prices(stored.loc[common], new.loc[common], market)


def _wipe(market: Market, symbol: str) -> None:
    shutil.rmtree(minute_dir(market, symbol), ignore_errors=True)


def _pages(client: TossClient, market: Market, symbol: str, before):
    """Newest-first pages (frames) from `before` (inclusive); ends with an empty/last page."""
    while True:
        page = client.get_candles(symbol, "1m", before=before, count=PAGE, adjusted=True)
        yield _page_frame(page, market), page.next_before
        if page.next_before is None or not len(page.df):
            return
        if before is not None and page.next_before >= before:
            raise PaginationError(f"{market} {symbol}: nextBefore {page.next_before} is not "
                                  f"before the previous cursor {before}")
        before = page.next_before


def _forward(client, market, symbol, today_ts, hi) -> tuple[pd.DataFrame, bool]:
    """Bars newer than `hi`; second value False when the stored scale no longer matches."""
    parts: list[pd.DataFrame] = []
    for bars, _ in _pages(client, market, symbol, None):
        if bars.empty:
            break
        parts.append(bars)
        if bars.index.min() <= hi:
            break
    if not parts:
        return pd.DataFrame(columns=MINUTE_COLUMNS), True
    allb = pd.concat(parts)
    allb = allb[~allb.index.duplicated(keep="first")].sort_index()
    if not _same_scale(market, symbol, allb[allb.index <= hi], hi):
        return allb, False
    new = allb[allb.index > hi]
    return new[_local_dates(new.index) < today_ts], True


def collect_minutes(client: TossClient, market: Market, symbol: str, since: date,
                    today: pd.Timestamp, _refetched: bool = False) -> MinuteReport:
    """Collect 1-minute bars back to `since` (local date); return rows added.

    `today` is the market's current session date: its bars are never stored. A changed price
    scale on the overlap (split / adjustment) deletes the symbol's minute files and refetches
    them from scratch. `_complete` marks that the backward part reached `since` (or the
    end of the history), so later runs only fetch forward. A second scale mismatch right
    after such a refetch raises PriceScaleError instead of looping."""
    today_ts, since_ts = pd.Timestamp(today).normalize(), pd.Timestamp(since)
    d = minute_dir(market, symbol)
    added = 0
    bounds = minute_bounds(market, symbol)
    if bounds is not None:
        new, ok = _forward(client, market, symbol, today_ts, bounds[1])
        if not ok:
            log.warning("%s %s: stored minute prices differ from the API (adjustment "
                        "changed); refetching from scratch", market, symbol)
            _wipe(market, symbol)
            bounds = None
        elif len(new):
            save_minutes(market, symbol, new)
            added += len(new)
    marker = d / COMPLETE_MARKER
    if bounds is not None and marker.exists():
        return MinuteReport(added, True)
    before = bounds[0].to_pydatetime() if bounds is not None else None
    lo = bounds[0] if bounds is not None else None
    complete = False
    for bars, next_before in _pages(client, market, symbol, before):
        if bars.empty:
            complete = True
            break
        if lo is not None and not _same_scale(market, symbol, bars[bars.index >= lo], lo):
            log.warning("%s %s: older minute prices do not match the stored scale; "
                        "refetching from scratch", market, symbol)
            if _refetched:
                raise PriceScaleError(f"{market} {symbol}: minute prices changed scale "
                                      "again right after a refetch")
            _wipe(market, symbol)       # nothing stored before the wipe counts as added
            return collect_minutes(client, market, symbol, since, today, _refetched=True)
        keep = bars[(_local_dates(bars.index) < today_ts)
                    & (_local_dates(bars.index) >= since_ts)]
        if lo is not None:
            keep = keep[keep.index < lo]
        if len(keep):
            save_minutes(market, symbol, keep)
            added += len(keep)
            lo = keep.index.min() if lo is None else min(lo, keep.index.min())
        if _local_dates(bars.index).min() < since_ts or next_before is None:
            complete = True
            break
    if complete:
        d.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
    return MinuteReport(added, complete)


def minute_symbols(market: Market, start: date, end: date, top_n: int) -> list[str]:
    """Symbols in the daily paper universe (top `top_n`) on any day of [start, end] plus the
    market's always_include, most universe days first (ties: always_include, then symbol)."""
    from autotrade.config import load_lab, load_market_raw, load_pocket
    from autotrade.data.store import load_panel
    from autotrade.universe.build import eligible_mask

    mc = load_market_raw(market)
    lab = replace(load_lab(), universe_top_n=top_n)
    panel = load_panel(market, None, start, end)
    mask = eligible_mask(panel, mc, load_pocket("daytrade"), lab, "paper", 1.0)
    days = mask.loc[(mask.index >= pd.Timestamp(start)) & (mask.index <= pd.Timestamp(end))]
    counts = days.sum(axis=0)
    counts = counts[counts > 0]
    for s in mc.always_include:
        counts[s] = counts.get(s, 0)
    first = {s: i for i, s in enumerate(mc.always_include)}
    return sorted((str(s) for s in counts.index),
                  key=lambda s: (-int(counts[s]), first.get(s, len(first)), s))
