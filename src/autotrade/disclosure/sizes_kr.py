"""KR event sizes: buyback amount, supply-contract revenue ratio, rights-offering dilution.

Output: events/sizes.parquet (filing_id, measure, value). Fetch order (spec §10-1 #8):
buyback -> supply contract -> rights offering.

NaN vs. retry. A NaN row is written only for DETERMINISTIC outcomes (and is then never
refetched): parse_supply_ratio returns None, the filing's row is absent from a successful
detail reply, a value is unparseable, or DART answers a non-transient status other than rate
limit (e.g. 100/101 bad parameters). TRANSIENT failures (DartError status "http", "800"
maintenance, "900" unknown) and DartRateLimited write NO row for the affected filing(s) and
stop the pass cleanly (rows done so far are flushed): `rate_limited=True` for the limit,
`stopped="transient"` otherwise; the next run retries.

Endpoints / fields (DART OpenAPI), confirmed against real DART on 2026-11-25 by the controller:
 - buyback   `tsstkAqDecsn.json` (자기주식 취득 결정): `aqpln_prc_ostk` (common) +
   `aqpln_prc_estk` (other) planned acquisition amount, KRW. Value = sum of the present ones.
   (Confirmed on a Samsung filing, 2024-11.)
 - rights    `piicDecsn.json` (유상증자 결정): new shares `nstk_ostk_cnt` + `nstk_estk_cnt`,
   shares before `bfic_tisstk_ostk` + `bfic_tisstk_estk` (comma strings, "-" when blank).
   dilution = new / existing, common and other summed on both sides (a missing side counts 0;
   existing <= 0 -> NaN). Rows also carry `ic_mthn` (e.g. "제3자배정증자"), not used here.
 - supply    no structured API: `document(rcept_no)` zip -> text; the figure after the
   "매출액 대비(%)" label (spaces / line breaks / tags may sit between its characters).
   Worked on two real 단일판매ㆍ공급계약체결(자율공시) documents (HD현대중공업): 0.0241, 0.1657.
Detail rows are matched to filings by `rcept_no`. One detail call per corp over the span
of that corp's target filing dates.
Targets: titles matched on the title with whitespace also removed; correction-tagged filings
(leading `[...]`) and titles containing 철회 get no size (Task 4 drops corrections).
"""
from __future__ import annotations

import io
import re
import zipfile
import pandas as pd

from autotrade.disclosure.dart import DartError, DartRateLimited
from autotrade.disclosure.events import read_table, strip_corrections  # noqa: F401 (re-export)
from autotrade.disclosure.store import append_sizes, load_filings, load_sizes

ORDER = ("buyback_krw", "supply_revenue_ratio", "rights_dilution")
DETAIL = {"buyback_krw": ("tsstkAqDecsn.json", ("aqpln_prc_ostk", "aqpln_prc_estk")),
          "rights_dilution": ("piicDecsn.json", None)}
BATCH = 25


def size_patterns(market: str) -> dict[str, str]:
    """measure -> title regex, from the `size` entries of config/events/<market>.yaml (read
    by events.read_table, the one reader of that file)."""
    return {e["size"]: e["match"] for e in read_table(market)["events"] if e.get("size")}


def size_targets(filings: pd.DataFrame, market: str = "kr") -> pd.DataFrame:
    """Filings whose stripped title matches a size event; rows ordered buyback -> supply ->
    rights (then filed_date). Columns: the filing columns plus `measure`."""
    pats = {m: re.compile(p) for m, p in size_patterns(market).items()}
    rows = []
    for rec in filings.to_dict("records"):
        t = strip_corrections(rec["title"])
        if t != str(rec["title"]).strip() or "철회" in str(rec["title"]):
            continue                    # corrections (leading [..] tag) and withdrawals
        compact = re.sub(r"\s+", "", t)
        for m in ORDER:
            if m in pats and (pats[m].search(t) or pats[m].search(compact)):
                rows.append({**rec, "measure": m})
                break
    out = pd.DataFrame(rows, columns=[*filings.columns, "measure"])
    out["_o"] = out["measure"].map({m: i for i, m in enumerate(ORDER)})
    return (out.sort_values(["_o", "filed_date", "filing_id"]).drop(columns="_o")
            .reset_index(drop=True))


def _num(v) -> float | None:
    s = str(v).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _sum(row: dict, keys) -> float | None:
    vals = [_num(row.get(k)) for k in keys]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def buyback_value(row: dict) -> float:
    v = _sum(row, ("aqpln_prc_ostk", "aqpln_prc_estk"))
    return float("nan") if v is None else v


def dilution_value(row: dict) -> float:
    new = _sum(row, ("nstk_ostk_cnt", "nstk_estk_cnt"))
    old = _sum(row, ("bfic_tisstk_ostk", "bfic_tisstk_estk"))
    if new is None or not old or old <= 0:
        return float("nan")
    return new / old


_VALUE = {"buyback_krw": buyback_value, "rights_dilution": dilution_value}
_SPACED = lambda w: r"\s*".join(map(re.escape, w))  # noqa: E731
_LABEL = re.compile(_SPACED("매출액") + r"\s*" + _SPACED("대비") + r"\s*\(\s*%\s*\)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_supply_ratio(doc: bytes) -> float | None:
    """Ratio from the '매출액 대비(%)' figure (12.5 -> 0.125); None if absent/unreadable."""
    try:
        with zipfile.ZipFile(io.BytesIO(doc)) as z:
            raw = b" ".join(z.read(n) for n in z.namelist())
    except zipfile.BadZipFile:
        return None
    for enc in ("utf-8", "euc-kr"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    m = _LABEL.search(text)
    if not m:
        return None
    tail = text[m.end():m.end() + 60]
    n = _NUMBER.search(tail)
    if not n or re.search(r"[가-힣A-Za-z]", tail[:n.start()]):
        return None
    try:
        return float(n.group().replace(",", "")) / 100.0
    except ValueError:
        return None


TRANSIENT = ("http", "800", "900")


def collect_sizes_kr(client, limit: int | None = None) -> dict:
    """Fetch sizes for target filings not yet in sizes.parquet. `limit` caps API calls.
    See the module docstring for NaN vs. transient-stop semantics."""
    targets = size_targets(load_filings("kr"))
    have = load_sizes("kr")
    todo = targets[~targets["filing_id"].isin(set(have["filing_id"]))]
    out = {"targets": len(targets), "added": 0, "rate_limited": False, "stopped": None}
    calls = 0
    batch: list[dict] = []

    def flush():
        if batch:
            append_sizes("kr", pd.DataFrame(batch))
            out["added"] += len(batch)
            batch.clear()

    nan = float("nan")
    try:
        for measure in ORDER:
            part = todo[todo["measure"] == measure]
            if measure == "supply_revenue_ratio":
                for rec in part.to_dict("records"):
                    if limit is not None and calls >= limit:
                        return out
                    calls += 1
                    try:
                        v = parse_supply_ratio(client.document(rec["filing_id"]))
                    except DartRateLimited:
                        raise
                    except DartError as e:
                        if e.status in TRANSIENT:
                            raise
                        v = None
                    batch.append({"filing_id": rec["filing_id"], "measure": measure,
                                  "value": nan if v is None else v})
                    if len(batch) >= BATCH:
                        flush()
                continue
            endpoint = DETAIL[measure][0]
            for corp, g in part.groupby("corp_code", sort=True):
                if limit is not None and calls >= limit:
                    return out
                calls += 1
                try:
                    rows = client.detail(endpoint, corp, min(g["filed_date"]), max(g["filed_date"]))
                except DartRateLimited:
                    raise
                except DartError as e:
                    if e.status in TRANSIENT:
                        raise
                    rows = []
                by = {str(r.get("rcept_no")): r for r in rows}
                for rec in g.to_dict("records"):
                    r = by.get(rec["filing_id"])
                    batch.append({"filing_id": rec["filing_id"], "measure": measure,
                                  "value": nan if r is None else _VALUE[measure](r)})
                flush()
    except DartRateLimited:
        out["rate_limited"] = True
    except DartError:
        out["stopped"] = "transient"
    finally:
        flush()
    return out
