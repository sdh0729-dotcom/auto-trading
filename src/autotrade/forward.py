"""Forward test (전진 시험): paper-trade the latest validate's recommended rules on days that
did not exist when they were chosen.

`register` freezes the rules (family + params), the paper capital and the benchmark into
research/forward-<pocket>-<market>.json, once; its start is the registration date, which must
lie after the period record's data cutoff (nothing on or after it fed screen / validate).
`run` scores the frozen rules on stored daily bars from the start on, with the same signals,
universe, costs and sizing as validate. History before the start (including the sealed
period) is read only as indicator warm-up: no metric covers a day before the start, so the
seal stays meaningful. No order API is involved.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from autotrade import pipeline as P
from autotrade.data.store import load_bars, load_panel
from autotrade.disclosure.events import attach_events
from autotrade.screen.simulate import simulate
from autotrade.sizing import run_portfolio
from autotrade.validate.criteria import period_equity
from autotrade.validate.experiments import ExperimentLog

OPEN_REASONS = ("period_end", "data_end")      # simulate closes still-open trades this way
LOAD_UNTIL = date(2100, 1, 1)                  # collect never stores an unfinished session


def _today(market: str) -> date:
    return P.market_today(market).date()


def forward_path(pocket: str, market: str) -> Path:
    return P.research_root() / f"forward-{pocket}-{market}.json"


def load_record(pocket: str, market: str) -> dict:
    path = forward_path(pocket, market)
    if not path.exists():
        raise P.LabError(f"no forward test for {pocket}/{market}: run `lab forward-register "
                         f"--market {market}` first")
    return json.loads(path.read_text(encoding="utf-8"))


def register(pocket: str, market: str) -> dict:
    """Freeze the latest validate run's recommended rules; refuses a second registration."""
    s = P._setup(pocket, market)
    path = forward_path(pocket, market)
    if path.exists():
        raise P.LabError(f"forward test {pocket}/{market} already registered ({path.name}); "
                         "its rules are frozen")
    start = _today(market)
    if start <= s.record.data_cutoff:
        raise P.LabError(f"forward start {start} must be after the data cutoff "
                         f"{s.record.data_cutoff} of {pocket}/{market}")
    rows = ExperimentLog(P.experiments_path()).rows(pocket, market)
    valid = list(P._iso(s.periods.valid))
    ids = P._latest_recommended(rows, valid)
    if not ids:
        raise P.LabError(f"the latest validate of {pocket}/{market} recommended no rule")
    val = rows[(rows["stage"] == "validate") & rows["period"].map(lambda p: list(p) == valid)]
    rules = []
    for rid in ids:
        row = val[val["rule_id"] == rid].iloc[-1]
        rule = P._rule(P._families(pocket), row["family"], row["params"], rid)   # rebuildable
        rules.append({"rule_id": rule.rule_id, "family": row["family"],
                      "params": dict(row["params"]),
                      "valid_return": row["metrics"].get("valid_return")})
    rec = {"version": 1, "pocket": pocket, "market": market, "start": start.isoformat(),
           "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "data_cutoff": s.record.data_cutoff.isoformat(),
           "valid": valid, "validate_run": int(val.iloc[-1]["metrics"].get("validate_run", 0)),
           "capital_krw": s.lab.capital["paper"], "scenario": "paper",
           "benchmark": s.market.benchmark, "rules": rules}
    _write_once(path, rec)
    return rec


def _write_once(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        os.link(tmp, path)                       # fails if the file appeared meanwhile
    finally:
        Path(tmp).unlink(missing_ok=True)


@dataclass
class ForwardResult:
    record: dict
    last_date: pd.Timestamp | None
    benchmark_return: float = math.nan
    benchmark_monthly: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rules: list[dict] = field(default_factory=list)


def _monthly(equity: pd.Series, base: float) -> pd.Series:
    """Return of each calendar month (the first month from `base`)."""
    ends = equity.groupby(equity.index.to_period("M")).last()
    return ends / ends.shift(1).fillna(base) - 1.0


def run(pocket: str, market: str) -> ForwardResult:
    rec = load_record(pocket, market)
    s = P._setup(pocket, market)
    start = pd.Timestamp(rec["start"])
    panel = load_panel(market, None, s.lab.history_start, LOAD_UNTIL)   # every stored day
    panel = attach_events(panel, market, LOAD_UNTIL)     # ETFs file nothing: KR ETF unchanged
    cal = panel.close.index
    days = cal[cal >= start]
    if len(days) == 0:
        return ForwardResult(rec, None)
    last = days[-1]
    elig = P._eligible(s, panel, rec["scenario"])
    cm = P._cost_model(s.market, panel.meta)
    cpk = P._currency_per_krw(s.market, s.synced)
    cap0 = rec["capital_krw"] * cpk
    out = ForwardResult(rec, last)
    bench = rec.get("benchmark")
    if bench:
        c = load_bars(market, bench)
        c = c["close"][(c.index >= start) & (c.index <= last)].dropna() if c is not None \
            else pd.Series(dtype=float)
        if len(c) >= 1:
            base = float(c.iloc[0])
            out.benchmark_return = float(c.iloc[-1] / base - 1.0)
            out.benchmark_monthly = _monthly(c / base * cap0, cap0)
    for r in rec["rules"]:
        rule = P._rule(P._families(pocket), r["family"], r["params"], r["rule_id"])
        sig = rule.signals(panel, eligible=elig)
        res = run_portfolio(simulate(panel, sig, elig, start.date(), last.date()), panel, cm,
                            s.pocket, rec["capital_krw"], cpk, False, entry_at=sig.entry_at,
                            exit_at=sig.exit_at, equal_weight=sig.equal_weight)
        eq = period_equity(res, cal, start.date(), last.date(), cap0)
        fills = res.fills
        held = fills[fills["exit_reason"].isin(OPEN_REASONS) & (fills["exit_date"] == last)]
        out.rules.append({
            "rule_id": r["rule_id"], "family": r["family"], "params": r["params"],
            "equity": eq, "return": float(eq.iloc[-1] / cap0 - 1.0),
            "monthly": _monthly(eq, cap0), "fills": fills,
            "n_closed": int(len(fills) - len(held)), "holdings": held,
        })
    return out
