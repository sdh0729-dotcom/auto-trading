"""Resumable KR filing collector (DART list.json, kinds B and I) into the event store.

Completeness contract: targets.json always describes the FULL target list (`--limit` only
limits which symbols are fetched in a run). The completeness gate is "every MAPPED target is
complete in state.json" (`store.incomplete_targets`). Unmapped targets cannot be fetched, so
they are excluded from completeness and gated only through coverage
(delisted_mapped / delisted_targets >= 0.80).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from autotrade.disclosure.dart import DartRateLimited
from autotrade.disclosure.store import (FILING_COLUMNS, append_filings, read_json,
                                        state_path, targets_path, write_json)

KINDS = ("B", "I")
DEFAULT_START = date(2015, 1, 1)


def event_targets(market: str) -> list[str]:
    """Stock symbols (no ETFs: they file nothing) in the swing paper universe on any day of
    [swing_start, today]."""
    from autotrade.config import load_lab, load_market_raw, load_pocket
    from autotrade.data.collect import market_today
    from autotrade.data.store import load_panel
    from autotrade.universe.build import eligible_mask

    lab = load_lab()
    start, end = lab.swing_start, market_today(market).date()
    panel = load_panel(market, None, start, end)
    mask = eligible_mask(panel, load_market_raw(market), load_pocket("swing"), lab, "paper", 1.0)
    days = mask.loc[(mask.index >= pd.Timestamp(start)) & (mask.index <= pd.Timestamp(end))]
    stocks = set(panel.meta.index[panel.meta["product"] == "stock"].astype(str))
    return sorted(str(s) for s in days.columns[days.any(axis=0)] if str(s) in stocks)


def _rows(raw: list[dict], symbol: str, kind: str) -> pd.DataFrame:
    rows = [{"filing_id": str(r["rcept_no"]), "symbol": symbol,
             "corp_code": str(r.get("corp_code", "")),
             "filed_date": date(int(str(r["rcept_dt"])[:4]), int(str(r["rcept_dt"])[4:6]),
                                int(str(r["rcept_dt"])[6:8])),
             "title": str(r.get("report_nm", "")).strip(), "kind": kind,
             "filer": str(r.get("flr_nm", "")), "remark": str(r.get("rm", ""))} for r in raw]
    return pd.DataFrame(rows, columns=FILING_COLUMNS)


def collect_events_kr(client, symbols: list[str], start: date = DEFAULT_START,
                      today: date | None = None, delisted: set[str] | frozenset = frozenset(),
                      limit: int | None = None) -> dict:
    """Filings of kinds B and I for each symbol from `start` (or its `done_through`) to `today`.
    Rows are appended per kind before the state write; a symbol is `complete` only once both
    kinds went through `today`. `limit` fetches only the first N mapped targets. DartRateLimited stops the run cleanly (state kept)."""
    if today is None:
        from autotrade.data.collect import market_today
        today = market_today("kr").date()
    corp = client.corp_codes()                      # one download per run
    code_of = {s: c for c, s in zip(corp["corp_code"], corp["stock_code"]) if s}
    mapped = [s for s in symbols if s in code_of]
    unmapped = [s for s in symbols if s not in code_of]
    dl = [s for s in symbols if s in delisted]
    # always the FULL target list, even with `limit` (see module docstring)
    write_json(targets_path("kr"), {"symbols": list(symbols),
        "targets": len(symbols), "mapped": len(mapped), "delisted_targets": len(dl),
        "delisted_mapped": sum(s in code_of for s in dl), "unmapped": unmapped})
    state = read_json(state_path("kr"), {})
    out = {"symbols": len(symbols), "filings": 0, "complete": 0, "unmapped": len(unmapped),
           "rate_limited": False}
    try:
        for sym in (mapped if limit is None else mapped[:limit]):
            st = state.get(sym, {})
            done = date.fromisoformat(st["done_through"]) if st.get("done_through") else None
            if st.get("complete") and done is not None and done >= today:
                out["complete"] += 1
                continue
            begin = done or start
            for kind in KINDS:
                rows = _rows(client.list_filings(code_of[sym], begin, today, kind), sym, kind)
                append_filings("kr", rows)
                out["filings"] += len(rows)
            state[sym] = {"done_through": today.isoformat(), "complete": True}
            write_json(state_path("kr"), state)
            out["complete"] += 1
    except DartRateLimited:
        out["rate_limited"] = True
    return out
