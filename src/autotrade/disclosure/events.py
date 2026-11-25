"""Event table, classification, signal dates and `Panel.events` (spec §5, §6, §10-1).

Raw filings (data/<market>/events/filings.parquet) become dated events: the SIGNAL DATE of an
event is the trading day t whose next open is the first open at which the filing was known
(rules enter at t+1's open, `entry_at="next_open"` only).

 - KR (filing date D, no time): signal = last trading day <= D. A filing made during or before
   the session of D is therefore used one day late (conservative).
 - US (`accepted_at`, converted to America/New_York; D = the New York date): if D is a trading
   day and the time is before 09:30 -> signal = the trading day before D (entry at D's open);
   otherwise signal = last trading day <= D. An early-close day changes nothing: anything
   accepted after that day's open maps to D, entry at the next open.
 - "Trading day" = a date of the calendar passed in. A D before the first calendar date has
   no trading day <= D; a D after the LAST calendar date has an unknown trading-day status
   (mapping it to the last stored day would put a later filing on an earlier signal date,
   i.e. look-ahead): both are dropped.

Classification (config/events/<market>.yaml, first match wins): KR on the correction-stripped
title (also tried with all whitespace removed), US on "<form>|<items>". Corrections are not
events: KR titles with a leading `[...]` tag (기재정정, 첨부정정, 첨부추가, 변경등록, ...), US
forms ending in "/A". DART's `remark` column is NOT used: its 정/철 flags describe LATER
corrections / withdrawals (look-ahead). Withdrawals (KR titles containing 철회) are events
with direction 0 and never remove the original event.

Repeats: rows are ordered by (symbol, event, signal_date, filing_id); the first is kept, and a
later row of the same symbol + event is dropped when its signal date lies fewer than
`repeat_days` calendar rows after the signal date of the last KEPT row (distance counted in
rows of the trading calendar; distance 0 = same day). Dropped rows never restart the window:
with repeat_days 20, rows 10 / 15 / 29 / 30 keep 10 and 30.

Sizes: `size_ok` = size >= size_min; False when the size is NaN; True when the event has no
size criterion (its size is NaN). KR buyback size = buyback_krw / adv at (signal date,
symbol), adv = 20-row mean of close x volume over the panel rows ending at and including the
signal date (`attach_events` computes it); adv missing -> NaN -> not ok. Other KR sizes are
ratios from sizes.parquet; the US insider size is the filing column `insider_usd` (USD).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from autotrade.config import ConfigError, _config_root
from autotrade.disclosure.store import load_filings, load_sizes, read_json, targets_path
from autotrade.types import Panel

EVENT_COLUMNS = ["symbol", "signal_date", "event", "direction", "size", "size_ok",
                 "bundle_a", "bundle_b", "positive", "filing_id"]
BUNDLES = ("bundle_a", "bundle_b")
FRAME_COLUMNS = (*BUNDLES, "positive")
ADV_DAYS = 20
ADV_MEASURES = frozenset({"buyback_krw"})       # amount in KRW -> divided by adv
NY = ZoneInfo("America/New_York")
US_OPEN_MINUTES = 9 * 60 + 30
_TAGS = re.compile(r"^\s*(\[[^\]]*\]\s*)+")


def strip_corrections(title: str) -> str:
    """The title without its leading `[...]` tags (기재정정, 첨부추가, ...)."""
    return _TAGS.sub("", str(title)).strip()


# ---------------------------------------------------------------- table
@dataclass(frozen=True)
class EventDef:
    name: str
    direction: int
    pattern: re.Pattern
    size: str | None = None
    size_min: float | None = None
    require_positive: str | None = None


@dataclass(frozen=True)
class EventTable:
    repeat_days: int
    negative_window: int
    events: tuple[EventDef, ...]
    bundles: dict[str, tuple[str, ...]]

    def by_name(self) -> dict[str, EventDef]:
        return {d.name: d for d in self.events}


def table_path(market: str) -> Path:
    return Path(_config_root()) / "events" / f"{market}.yaml"


def read_table(market: str) -> dict:
    """The raw yaml of config/events/<market>.yaml (the one reader of that file)."""
    p = table_path(market)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConfigError(f"event table {p} is missing") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"event table {p} is not a mapping")
    return raw


def load_table(market: str) -> EventTable:
    raw = read_table(market)
    where = table_path(market)
    try:
        repeat, neg = int(raw["repeat_days"]), int(raw["negative_window"])
        defs = []
        for e in raw["events"]:
            unknown = set(e) - {"name", "direction", "match", "size", "size_min",
                                "require_positive"}
            if unknown:
                raise ConfigError(f"{where}: unknown keys {sorted(unknown)} in {e}")
            if e["direction"] not in (-1, 0, 1):
                raise ConfigError(f"{where}: {e['name']}: direction must be -1, 0 or 1")
            if ("size" in e) != ("size_min" in e):
                raise ConfigError(f"{where}: {e['name']}: size and size_min go together")
            defs.append(EventDef(str(e["name"]), int(e["direction"]), re.compile(e["match"]),
                                 e.get("size"),
                                 float(e["size_min"]) if "size_min" in e else None,
                                 e.get("require_positive")))
        bundles = {b: tuple(raw["bundles"][b]) for b in BUNDLES}
    except (KeyError, TypeError, ValueError, re.error) as e:
        raise ConfigError(f"{where}: malformed event table ({type(e).__name__}: {e})") from e
    names = [d.name for d in defs]
    if len(set(names)) != len(names):
        raise ConfigError(f"{where}: duplicate event names")
    for b, members in bundles.items():
        if not set(members) <= set(names):
            raise ConfigError(f"{where}: bundle {b} names unknown events "
                              f"{sorted(set(members) - set(names))}")
    if repeat < 1 or neg < 1:
        raise ConfigError(f"{where}: repeat_days and negative_window must be >= 1")
    return EventTable(repeat, neg, tuple(defs), bundles)


def events_table_hash(market: str) -> str:
    """sha1 of the event table file's bytes, 12 hex chars (logged with event-family rows)."""
    return hashlib.sha1(table_path(market).read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- classification
def _us_key(form, items) -> str:
    items = "" if items is None or (isinstance(items, float) and np.isnan(items)) else items
    return str(form).strip() + "|" + re.sub(r"\s+", "", str(items))


def classify(filings: pd.DataFrame, market: str, table: EventTable | None = None
             ) -> pd.DataFrame:
    """filings + `key` (what the table matches), `correction` (bool) and `event` (name, or
    None when unmatched / a correction)."""
    table = table or load_table(market)
    out = filings.copy()
    if market == "kr":
        titles = out["title"].astype(str)
        out["key"] = titles.map(strip_corrections)
        out["correction"] = out["key"] != titles.str.strip()
        names = {}
        for t in out["key"].unique():
            compact = re.sub(r"\s+", "", t)
            names[t] = next((d.name for d in table.events
                             if d.pattern.search(t) or d.pattern.search(compact)), None)
        out["event"] = out["key"].map(names)
    else:
        out["key"] = [_us_key(f, i) for f, i in zip(out["form"], out["items"])]
        out["correction"] = out["form"].astype(str).str.strip().str.endswith("/A")
        ev = []
        for rec in out.to_dict("records"):
            name = None
            for d in table.events:
                if not d.pattern.search(rec["key"]):
                    continue
                if d.require_positive is not None:
                    v = pd.to_numeric(rec.get(d.require_positive), errors="coerce")
                    if not (v > 0):
                        continue
                name = d.name
                break
            ev.append(name)
        out["event"] = ev
    out["event"] = out["event"].astype(object).where(~out["correction"], None)
    return out


# ---------------------------------------------------------------- signal dates
def _calendar(calendar) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(calendar).normalize().unique().sort_values()


def _last_on_or_before(days: pd.Series, cal: pd.DatetimeIndex) -> np.ndarray:
    """Calendar row of the last trading day <= each date; -1 when none or unknown (NaT, or
    after the last calendar date)."""
    vals = days.to_numpy(dtype="datetime64[ns]")
    pos = cal.searchsorted(vals, side="right") - 1
    bad = pd.isna(vals) | (len(cal) == 0)
    if len(cal):
        bad |= vals > cal[-1].to_datetime64()
    return np.where(bad, -1, pos)


def signal_rows_kr(filed_date: pd.Series, cal: pd.DatetimeIndex) -> np.ndarray:
    return _last_on_or_before(pd.to_datetime(filed_date).dt.normalize(), cal)


def signal_rows_us(accepted_at: pd.Series, cal: pd.DatetimeIndex) -> np.ndarray:
    ny = pd.to_datetime(accepted_at, utc=True).dt.tz_convert(NY)
    d = ny.dt.tz_localize(None).dt.normalize()
    pos = _last_on_or_before(d, cal)
    pre_open = (ny.dt.hour * 60 + ny.dt.minute < US_OPEN_MINUTES).to_numpy()
    on_day = d.isin(cal).to_numpy()
    pos = np.where((pos >= 0) & on_day & pre_open, pos - 1, pos)
    return np.where(pos >= 0, pos, -1)


# ---------------------------------------------------------------- events
def _empty() -> pd.DataFrame:
    return pd.DataFrame({"symbol": pd.Series(dtype=object),
                         "signal_date": pd.Series(dtype="datetime64[ns]"),
                         "event": pd.Series(dtype=object), "direction": pd.Series(dtype="int64"),
                         "size": pd.Series(dtype=float), "size_ok": pd.Series(dtype=bool),
                         "bundle_a": pd.Series(dtype=bool), "bundle_b": pd.Series(dtype=bool),
                         "positive": pd.Series(dtype=bool),
                         "filing_id": pd.Series(dtype=object)})


def _drop_repeats(df: pd.DataFrame, repeat_days: int) -> pd.DataFrame:
    """See the module docstring: distance in calendar rows to the last KEPT row."""
    df = df.sort_values(["symbol", "event", "signal_date", "filing_id"], kind="mergesort")
    keep = np.zeros(len(df), dtype=bool)
    last_key, last_row = None, 0
    for i, (key, row) in enumerate(zip(zip(df["symbol"], df["event"]), df["_row"])):
        if key != last_key or row - last_row >= repeat_days:
            keep[i] = True
            last_key, last_row = key, row
    return df[keep]


def _lookup(frame: pd.DataFrame, dates: pd.Series, symbols: pd.Series) -> np.ndarray:
    r = frame.index.get_indexer(pd.DatetimeIndex(dates))
    c = frame.columns.get_indexer(pd.Index(symbols))
    ok = (r >= 0) & (c >= 0)
    out = np.full(len(r), np.nan)
    out[ok] = frame.to_numpy(dtype=float)[r[ok], c[ok]]
    return out


def load_events(market: str, calendar: pd.DatetimeIndex, adv: pd.DataFrame | None = None
                ) -> pd.DataFrame | None:
    """Events of the stored filings on `calendar` (columns EVENT_COLUMNS, sorted by
    signal_date, symbol, filing_id); None when no filings are stored. `adv`: date x symbol
    20-row mean of close x volume (for the KR buyback size; None -> those sizes are NaN)."""
    filings = load_filings(market)
    if filings.empty:
        return None
    table = load_table(market)
    cal = _calendar(calendar)
    df = classify(filings, market, table)
    df = df[df["event"].notna() & df["symbol"].notna()].copy()
    if df.empty:
        return _empty()
    df["symbol"] = df["symbol"].astype(str)
    df["filing_id"] = df["filing_id"].astype(str)
    rows = (signal_rows_kr(df["filed_date"], cal) if market == "kr"
            else signal_rows_us(df["accepted_at"], cal))
    df["_row"] = rows
    df = df[df["_row"] >= 0].copy()
    if df.empty:
        return _empty()
    df["signal_date"] = cal[df["_row"].to_numpy()]
    df = _drop_repeats(df, table.repeat_days)

    defs = table.by_name()
    sizes = load_sizes(market)
    stored = {(f, m): v for f, m, v in zip(sizes["filing_id"].astype(str), sizes["measure"],
                                           sizes["value"])}
    size = np.full(len(df), np.nan)
    ok = np.ones(len(df), dtype=bool)
    adv_at = (_lookup(adv, df["signal_date"], df["symbol"]) if adv is not None
              else np.full(len(df), np.nan))
    for i, rec in enumerate(df.to_dict("records")):
        d = defs[rec["event"]]
        if d.size is None:
            continue
        if d.size in df.columns:
            v = pd.to_numeric(rec[d.size], errors="coerce")
        else:
            v = stored.get((rec["filing_id"], d.size), np.nan)
        v = float(v) if v is not None and not pd.isna(v) else np.nan
        if d.size in ADV_MEASURES:
            a = adv_at[i]
            v = v / a if a > 0 else np.nan          # adv NaN / 0 / missing -> NaN
        size[i] = v
        ok[i] = bool(v >= d.size_min) if not np.isnan(v) else False

    out = pd.DataFrame({
        "symbol": df["symbol"].to_numpy(dtype=object),
        "signal_date": pd.DatetimeIndex(df["signal_date"]).astype("datetime64[ns]"),
        "event": df["event"].to_numpy(dtype=object),
        "direction": df["event"].map(lambda n: defs[n].direction).to_numpy(dtype="int64"),
        "size": size, "size_ok": ok,
        "bundle_a": df["event"].isin(table.bundles["bundle_a"]).to_numpy(dtype=bool),
        "bundle_b": df["event"].isin(table.bundles["bundle_b"]).to_numpy(dtype=bool),
        "positive": (df["event"].map(lambda n: defs[n].direction) == 1).to_numpy(dtype=bool),
        "filing_id": df["filing_id"].to_numpy(dtype=object),
    })
    return out.sort_values(["signal_date", "symbol", "filing_id"],
                           kind="mergesort").reset_index(drop=True)[EVENT_COLUMNS]


def adv_frame(panel: Panel) -> pd.DataFrame:
    """20-row mean of close x volume over the panel rows, ending at and including t."""
    return (panel.close * panel.volume).rolling(ADV_DAYS, min_periods=ADV_DAYS).mean()


def attach_events(panel: Panel, market: str, end: date) -> Panel:
    """`panel` with `events` = the market's events with signal_date <= end (None when no
    filings are stored). Signal dates use `panel.calendar` (the stored trading dates when the
    panel has none); buyback sizes use the adv of this panel."""
    if panel.calendar is not None:
        cal = panel.calendar
    else:
        from autotrade.data.store import calendar
        cal = calendar(market)[0]
    ev = load_events(market, cal, adv_frame(panel))
    if ev is not None:
        ev = ev[ev["signal_date"] <= pd.Timestamp(end)].reset_index(drop=True)
    return replace(panel, events=ev)


# ---------------------------------------------------------------- rule helpers
def _events(panel: Panel) -> pd.DataFrame:
    if panel.events is None:
        raise ValueError("panel has no events: event rules need attach_events (filings "
                         "collected)")
    return panel.events


def _mark(panel: Panel, ev: pd.DataFrame) -> pd.DataFrame:
    idx, cols = panel.close.index, panel.close.columns
    a = np.zeros((len(idx), len(cols)), dtype=bool)
    if len(ev):
        r = idx.get_indexer(pd.DatetimeIndex(ev["signal_date"]))
        c = cols.get_indexer(pd.Index(ev["symbol"]))
        ok = (r >= 0) & (c >= 0)
        a[r[ok], c[ok]] = True
    return pd.DataFrame(a, index=idx, columns=cols)


def event_frame(panel: Panel, column: str, min_size: bool) -> pd.DataFrame:
    """Bool date x symbol on panel.close: True where an event of `column` (bundle_a,
    bundle_b or positive) has signal_date = t; with `min_size` only events with size_ok."""
    if column not in FRAME_COLUMNS:
        raise ValueError(f"column must be one of {FRAME_COLUMNS}, got {column!r}")
    ev = _events(panel)
    sel = ev[column].astype(bool)
    if min_size:
        sel &= ev["size_ok"].astype(bool)
    return _mark(panel, ev[sel])


def recent_negative(panel: Panel, window: int) -> pd.DataFrame:
    """Bool date x symbol on panel.close: True where a direction -1 event's signal date lies
    within the last `window` panel rows including t."""
    if window < 1:
        raise ValueError("window must be >= 1")
    ev = _events(panel)
    neg = _mark(panel, ev[ev["direction"] == -1])
    return neg.astype(np.int64).rolling(window, min_periods=1).sum().gt(0)


# ---------------------------------------------------------------- report (lab events)
def report(market: str, calendar: pd.DatetimeIndex, unmatched: int = 20) -> dict:
    """Counts by event x signal year (repeats dropped), target coverage and the most
    frequent unmatched keys (correction-stripped titles / form|items), for finalizing the
    table from title frequencies only (no prices, no returns)."""
    filings = load_filings(market)
    cl = classify(filings, market) if not filings.empty else filings.assign(
        key=pd.Series(dtype=object), correction=pd.Series(dtype=bool),
        event=pd.Series(dtype=object))
    ev = load_events(market, calendar)
    ev = _empty() if ev is None else ev
    counts = (pd.crosstab(ev["event"], ev["signal_date"].dt.year) if len(ev)
              else pd.DataFrame())
    miss = cl[cl["event"].isna() & ~cl["correction"].astype(bool)]
    top = miss["key"].value_counts().head(unmatched) if unmatched > 0 else miss["key"].iloc[:0]
    return {"filings": len(cl), "corrections": int(cl["correction"].sum()),
            "unmatched_total": len(miss), "events": len(ev), "counts": counts,
            "targets": read_json(targets_path(market), {}),
            "unmatched": [(str(k), int(v)) for k, v in top.items()]}
