"""Resumable US filing collector (SEC EDGAR submissions) into the event store.

Completeness contract: as collect_kr: targets.json always describes the FULL target list
(`limit` only limits what is fetched); the gate is "every MAPPED target is complete"
(`store.incomplete_targets`). US coverage = mapped share of ALL targets, so targets.json sets
`delisted_targets = targets` and `delisted_mapped = mapped` (the coverage gate then reads
mapped/targets >= 0.80 instead of silently passing).

Kept forms: 8-K, 8-K/A, 4, 424B1-424B8, filed on/after the start (2015-01-01).
Raw columns: filing_id (accession), symbol, cik, filed_date, accepted_at (UTC, see
edgar.parse_acceptance, assumption A1), form, items ("1.01,9.01" or ""), insider_usd.

Insider amounts: only a Form 4 row carries `insider_usd` = sum(shares x price) over code-P
rows of its accession (NaN otherwise). The amount comes from SEC's quarterly insider data
sets, published only after a quarter ends; the insider HORIZON is the last day of the newest
quarter whose set is published (404 = not published, probed back up to 8 quarters, remembered
per run). Per-symbol state is split (ruling R15):
 - `done_through`: 8-K / 8-K/A / 424B* rows, collected through today (like KR);
 - `insider_through`: Form 4 rows, collected through the insider horizon. Form 4 rows filed
   after it are NOT stored yet (a stored row is never updated); a later run stores them, with
   insider_usd, once their quarter is published.
`complete` = done_through reached today AND insider_through reached the current horizon.
A Form 4 whose quarter data cannot be read (any error but 404, or an unpublished quarter
before the horizon) raises EdgarError: that symbol is not marked complete and no NaN row is
stored (ruling R14). Per-quarter aggregates (accession -> insider_usd) are cached under
events_root("us")/insider/. Tickers sharing a CIK (share classes) each get their own rows
(store dedupe is (filing_id, symbol); one submissions fetch per CIK per run). Ticker matching
upper-cases and treats "." as "-" ("BRK.B" = "BRK-B").
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from autotrade.data.store import write_parquet_atomic
from autotrade.disclosure.edgar import (EdgarError, EdgarNotFound, EdgarRateLimited, insider_buy_usd,
                                        parse_acceptance)
from autotrade.disclosure.store import (FILING_COLUMNS_US, append_filings, events_root,
                                        read_json, state_path, targets_path, write_json)

KEEP_FORMS = frozenset({"8-K", "8-K/A", "4"} | {f"424B{i}" for i in range(1, 9)})
DEFAULT_START = date(2015, 1, 1)
MAX_QUARTER_PROBES = 8


def _norm(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def _quarter(d: date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def _quarter_end(y: int, q: int) -> date:
    return date(y + (q == 4), q * 3 % 12 + 1, 1) - timedelta(days=1)


def _prev(y: int, q: int) -> tuple[int, int]:
    return (y, q - 1) if q > 1 else (y - 1, 4)


def _insider_cache(y: int, q: int):
    return events_root("us") / "insider" / f"{y}q{q}.parquet"


class _Insider:
    """accession -> insider_usd per quarter; disk cache, then the client."""

    def __init__(self, client):
        self.client = client
        self.mem: dict[tuple[int, int], pd.Series | None] = {}   # None = not published (404)

    def get(self, y: int, q: int) -> pd.Series | None:
        """None when that quarter's data set is not published."""
        if (y, q) in self.mem:
            return self.mem[(y, q)]
        p = _insider_cache(y, q)
        if p.exists():
            df = pd.read_parquet(p)
            s = pd.Series(df["insider_usd"].to_numpy(dtype=float), index=df["accession"])
        else:
            try:
                s = insider_buy_usd(self.client.insider_quarter(y, q))
            except EdgarNotFound:
                self.mem[(y, q)] = None
                return None
            write_parquet_atomic(pd.DataFrame({"accession": s.index.astype(str),
                                               "insider_usd": s.to_numpy(dtype=float)}), p)
        self.mem[(y, q)] = s
        return s

    def horizon(self, today: date) -> date | None:
        y, q = _quarter(today)
        for _ in range(MAX_QUARTER_PROBES):
            if self.get(y, q) is not None:
                return min(_quarter_end(y, q), today)
            y, q = _prev(y, q)
        return None


def _rows(raw: list[dict], symbol: str, cik: str, begin: date, begin_ins: date, today: date,
          horizon: date, insider: _Insider) -> pd.DataFrame:
    out = []
    for r in raw:
        form = str(r.get("form") or "").strip()
        if form not in KEEP_FORMS:
            continue
        try:
            filed = date.fromisoformat(str(r["filingDate"])[:10])
        except (KeyError, ValueError):
            continue
        usd = float("nan")
        if form == "4":
            if filed < max(begin_ins, DEFAULT_START) or filed > horizon:
                continue
            s = insider.get(*_quarter(filed))
            if s is None:
                raise EdgarError("insider", "insider data set missing before the horizon")
            usd = float(s.get(str(r["accessionNumber"]), float("nan")))
        elif filed < max(begin, DEFAULT_START) or filed > today:
            continue
        out.append({"filing_id": str(r["accessionNumber"]), "symbol": symbol, "cik": cik,
                    "filed_date": filed, "accepted_at": parse_acceptance(r.get("acceptanceDateTime")),
                    "form": form,
                    "items": ",".join(x.strip() for x in str(r.get("items") or "").split(",")
                                      if x.strip()),
                    "insider_usd": usd})
    df = pd.DataFrame(out, columns=FILING_COLUMNS_US)
    df["accepted_at"] = pd.to_datetime(df["accepted_at"], utc=True)
    df["insider_usd"] = df["insider_usd"].astype(float)
    return df


def collect_events_us(client, symbols: list[str], start: date = DEFAULT_START,
                      today: date | None = None, limit: int | None = None) -> dict:
    """8-K / 4 / 424B filings of each symbol from `start` (or its `done_through`) to the
    insider-data horizon. Rows are appended before the state write. `limit` fetches only the
    first N mapped targets. EdgarRateLimited stops the run cleanly (state kept)."""
    if today is None:
        from autotrade.data.collect import market_today
        today = market_today("us").date()
    tk = client.tickers()
    cik_of: dict[str, str] = {}
    for t, c in zip(tk["ticker"], tk["cik"]):
        cik_of.setdefault(_norm(t), str(c))
    mapped = [s for s in symbols if _norm(s) in cik_of]
    unmapped = [s for s in symbols if _norm(s) not in cik_of]
    write_json(targets_path("us"), {"symbols": list(symbols), "targets": len(symbols),
        "mapped": len(mapped), "delisted_targets": len(symbols), "delisted_mapped": len(mapped),
        "unmapped": unmapped})
    state = read_json(state_path("us"), {})
    out = {"symbols": len(symbols), "filings": 0, "complete": 0, "unmapped": len(unmapped),
           "rate_limited": False, "horizon": None}
    insider = _Insider(client)
    try:
        horizon = insider.horizon(today)
        if horizon is None:
            raise EdgarError("insider", "no insider data set available")
        out["horizon"] = horizon.isoformat()
        subs: dict[str, list[dict]] = {}        # one submissions fetch per CIK per run
        for sym in (mapped if limit is None else mapped[:limit]):
            cik = cik_of[_norm(sym)]
            st = state.get(sym, {})
            done = date.fromisoformat(st["done_through"]) if st.get("done_through") else None
            ins = date.fromisoformat(st["insider_through"]) if st.get("insider_through") else None
            if (st.get("complete") and done is not None and done >= today
                    and ins is not None and ins >= horizon):
                out["complete"] += 1
                continue
            begin, begin_ins = done or start, ins or start
            if cik not in subs:
                subs[cik] = client.submissions(cik, since=min(begin, begin_ins))
            rows = _rows(subs[cik], sym, cik, begin, begin_ins, today, horizon, insider)
            append_filings("us", rows)
            out["filings"] += len(rows)
            state[sym] = {"done_through": today.isoformat(),
                          "insider_through": horizon.isoformat(), "complete": True}
            write_json(state_path("us"), state)
            out["complete"] += 1
    except EdgarRateLimited:
        out["rate_limited"] = True
    return out
