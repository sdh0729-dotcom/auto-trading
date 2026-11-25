"""Research workflow behind the `lab` CLI: sync -> collect -> universe -> screen -> validate
-> verify --final -> report -> journal.

Periods: on the first `screen` of a pocket x market the stored trading dates from
`lab.swing_start` on are split by `lab.split` into train / valid / sealed and persisted in
research/periods-{pocket}-{market}.json (with the data cutoff) once the screen's checks
passed; a lost file is rebuilt from the seal tickets, not re-split. Every later command reads that
file, so later collects never move a boundary (`lab periods --new-cycle` starts a
re-validation cycle, spec 7.9). `screen` loads prices only through the train end, `validate`
only through the valid end, and both refuse ranges overlapping an already-opened sealed
period; the sealed prices are loaded by `verify_final` after it has opened the seal ticket.
Signals are always computed on the panel history from `lab.history_start` (swing_start -
warmup_days) through the end of the period being evaluated (warm-up), while the universe
statistics start at swing_start (no symbol is eligible before it) and simulate /
run_portfolio / run_nautilus are limited to the period itself.
"""
from __future__ import annotations

import fcntl
import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yaml

from autotrade.config import (ConfigError, LabConfig, MarketConfig, PocketConfig, SyncedValues,
                              _synced_path, env_path, load_lab, load_market, load_market_raw,
                              load_pocket, load_synced, repo_root)
from autotrade.costs import CostModel
from autotrade.data.collect import collect_daily, collect_meta, market_today
from autotrade.data.collect_minute import COMPLETE_MARKER, collect_minutes, minute_symbols
from autotrade.data.delisted_kr import (KindError, add_delisted_meta, kind_delisted,
                                        save_supplement_record, supplement_delisted_kr)
from autotrade.data.minute_store import minute_bounds, minute_calendar, minute_dir
from autotrade.data.krx_overlay import refresh_overlays, save_overlay_record
from autotrade.data.quality import check_daily, hard_issues
from autotrade.data.sessions import auction_lead_minutes
from autotrade.data.store import (calendar, data_root, load_meta, load_bars, load_daily, load_panel,
                                  save_meta, stored_symbols, trading_dates)
from autotrade.disclosure.collect_kr import collect_events_kr, event_targets
from autotrade.disclosure.sizes_kr import collect_sizes_kr
from autotrade.disclosure.dart import make_dart_client_from_env
from autotrade.disclosure.events import attach_events
from autotrade.disclosure.events import report as events_report_of
from autotrade.disclosure.store import incomplete_targets, load_filings
from autotrade.disclosure.store import lock_path as events_lock_path
from autotrade.report import (CONFIDENCE_FORMULAS, RuleCard, limitations, write_leaderboard,
                              write_rule_card)
from autotrade.rules.registry import DAYTRADE_FAMILIES, SWING_FAMILIES
from autotrade.screen.intraday import SessionRules
from autotrade.screen.run import (MIN_TRAIN_TRADES, intraday_trades, screen_family,
                                  screen_intraday)
from autotrade.screen.simulate import simulate
from autotrade.sizing import PortfolioResult, run_portfolio
from autotrade.sizing.intraday import local_day, run_intraday_portfolio
from autotrade.toss.client import TossApiError, make_client_from_env
from autotrade.types import TRADE_COLUMNS, TZ, Panel
from autotrade.universe.build import eligible_mask
from autotrade.validate.criteria import evaluate, period_equity
from autotrade.validate.experiments import ExperimentLog
from autotrade.validate import periods as period_store
from autotrade.validate.pbo import pbo_cscv
from autotrade.validate.periods import PeriodError, PeriodRecord
from autotrade.validate.seal import SealRegistry, SealViolation
from autotrade.validate.split import Periods, period_segments
from autotrade.validate.stats import max_drawdown, sharpe
from autotrade.verify import parity_report, run_nautilus

MARKETS = ("kr", "us")
SCENARIOS = ("paper", "real")
VALID_SEGMENTS = 4               # "검증 기간 구간 안정성": valid cut into 4 test segments
PBO_SPLITS = (16, 8, 4, 2)
FEATURE_WARMUP_DAYS = 30         # calendar days of minute bars before valid for rolling features
# the benchmark's first stored minute bar may lie up to this many calendar days after the
# market's minute_start (Toss's history can begin a few days late) before screen / validate
# call its minute history short
MINUTE_START_TOLERANCE_DAYS = 10
MAX_COMMISSION = 0.01            # a commission ratio >= 1% is almost surely a percent typo
PARITY_MISMATCHES_KEPT = 20
FAMILIES = {"swing": SWING_FAMILIES, "daytrade": DAYTRADE_FAMILIES}
VERDICT_PAPER, VERDICT_REAL = "② 모의 자격", "실돈 자격"
# US real money: fractional-order limits (amount orders, exit 1 h before the close, no
# server-side stop) are not modelled yet -> reference only until plan 2 (spec 5)
VERDICT_REAL_US = "참고용(계획 2에서 확정)"


log = logging.getLogger(__name__)


class LabError(Exception):
    """Expected, user-fixable failure: the CLI prints the message on one line."""


# ---------------------------------------------------------------- paths
def research_root() -> Path:
    return env_path("AUTOTRADE_RESEARCH", repo_root() / "research")


def experiments_path() -> Path:
    return research_root() / "experiments.jsonl"


def seal_path() -> Path:
    return research_root() / "seal-registry.json"


def periods_path(pocket: str, market: str) -> Path:
    return research_root() / f"periods-{pocket}-{market}.json"


def limited_marker(market: str) -> Path:
    """Exists while the stored data comes from `lab collect --limit` (a partial universe)."""
    return data_root() / market / ".limited"


def _refuse_limited(market: str) -> None:
    if limited_marker(market).exists():
        raise LabError(f"{market} data comes from `lab collect --limit` (partial universe): "
                       f"run a full `lab collect --market {market}` first")


def minute_root(market: str) -> Path:
    return data_root() / market / "1m"


def minute_limited_marker(market: str) -> Path:
    """Exists while the stored minute data comes from `lab collect-minute --limit`."""
    return minute_root(market) / ".limited"


def collect_lock_path(market: str) -> Path:
    """flock()ed by a running `lab collect-minute` for its whole run (one writer at a time)."""
    return minute_root(market) / ".collect.lock"


@contextmanager
def _exclusive_lock(path: Path, message: str):
    """Hold an flock on `path` (non-blocking): LabError(message) if another process holds it.
    The lock dies with its process, so a killed run never leaves a stale lock behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LabError(message) from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()


def _collect_lock(market: str):
    """The market's minute-collector lock: LabError if another run holds it."""
    path = collect_lock_path(market)
    return _exclusive_lock(path, f"`lab collect-minute --market {market}` is already running "
                                 f"(lock {path}): wait for it to finish")


def _lock_held(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def collect_running(market: str) -> bool:
    """True while some process holds the market's collector lock."""
    return _lock_held(collect_lock_path(market))


def events_running(market: str) -> bool:
    """True while some process holds the market's `lab collect-events` lock."""
    return _lock_held(events_lock_path(market))


def _refuse_incomplete_minutes(market: str) -> None:
    """Daytrade screen / validate run only on a finished minute collection: no collector
    running, no `--limit` partial set, every stored symbol's history marked `_complete`, and
    the benchmark's history (the minute calendar) reaching back to minute_start (first bar
    within MINUTE_START_TOLERANCE_DAYS calendar days of it). Checked before anything is
    written, since a premature screen fixes the periods file and logs trials for good."""
    fix = f"finish `lab collect-minute --market {market}` first"
    if collect_running(market):
        raise LabError(f"`lab collect-minute --market {market}` is running: wait for it to "
                       "finish (its minute data is still changing)")
    if minute_limited_marker(market).exists():
        raise LabError(f"{market} minute data comes from `lab collect-minute --limit` "
                       f"(partial symbol set): run a full `lab collect-minute --market "
                       f"{market}` first")
    root = minute_root(market)
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    partial = [p.name for p in dirs if not (p / COMPLETE_MARKER).exists()]
    if partial:
        raise LabError(f"{len(partial)} {market} symbol(s) have incomplete minute history (no "
                       f"{COMPLETE_MARKER} marker: {', '.join(partial[:5])}"
                       f"{', ...' if len(partial) > 5 else ''}): {fix}")
    mc = load_market_raw(market)
    if mc.benchmark is None or mc.minute_start is None:
        raise LabError(f"{market} config needs a benchmark and a minute_start for daytrade")
    bounds = minute_bounds(market, mc.benchmark)
    if bounds is None:
        raise LabError(f"benchmark {mc.benchmark} has no minute data: {fix}")
    first = bounds[0].tz_convert(TZ[market]).tz_localize(None).normalize()
    limit = pd.Timestamp(mc.minute_start) + pd.Timedelta(days=MINUTE_START_TOLERANCE_DAYS)
    if first > limit:
        raise LabError(f"benchmark {mc.benchmark} minute history starts {first.date()}, more "
                       f"than {MINUTE_START_TOLERANCE_DAYS} days after minute_start "
                       f"{mc.minute_start}: {fix}")


def cards_dir() -> Path:
    return research_root() / "rule-cards"


def leaderboard_path(pocket: str, market: str) -> Path:
    return research_root() / f"leaderboard-{pocket}-{market}.md"


def journal_path(day: date) -> Path:
    return research_root() / "journal" / f"{day.isoformat()}.md"


def token_path() -> Path:
    return data_root() / "toss.token.json"


# ---------------------------------------------------------------- shared setup
@dataclass
class Setup:
    market: MarketConfig
    pocket: PocketConfig
    lab: LabConfig
    synced: SyncedValues
    periods: Periods
    calendar: pd.DatetimeIndex   # stored trading dates from lab.swing_start to the data cutoff
    record: PeriodRecord         # (daytrade: days with minute bars of the benchmark)
    name: str = "swing"          # the pocket's name


def _check_market(market: str) -> None:
    if market not in MARKETS:
        raise LabError(f"unknown market {market!r} (choose from {', '.join(MARKETS)})")


def _families(pocket: str) -> dict:
    if pocket not in FAMILIES:
        raise LabError(f"no rule families for pocket {pocket!r} yet "
                       f"(available: {', '.join(FAMILIES)})")
    return FAMILIES[pocket]


def _swing_only(pocket: str, what: str) -> None:
    if pocket != "swing":
        raise LabError(f"`{what}` does not support the {pocket} pocket yet")


def _pocket_dates(pocket: str, market: str, lab: LabConfig, end: date) -> pd.DatetimeIndex:
    """Trading dates the period split runs on, through `end`: stored daily dates from
    lab.swing_start (swing) or the minute calendar (daytrade; days with benchmark minute bars
    from the market's minute_start)."""
    if pocket == "daytrade":
        cal = minute_calendar(market)
        return cal[cal <= pd.Timestamp(end)]
    return trading_dates(market, lab.swing_start, end)


def _no_data(pocket: str, market: str) -> LabError:
    if pocket == "daytrade":
        return LabError(f"no minute data for {market}: run `lab collect-minute --market "
                        f"{market}` first")
    return LabError(f"no daily data for {market}: run `lab collect --market {market}` first")


def _load_periods(pocket: str, market: str) -> PeriodRecord | None:
    try:
        return period_store.load_record(periods_path(pocket, market))
    except PeriodError as e:
        raise LabError(str(e)) from e


def _validate_runs_logged(pocket: str, market: str, valid: tuple[date, date]) -> int:
    """Highest `validate_run` the experiment log holds for the valid period (0 if none)."""
    rows = ExperimentLog(experiments_path()).rows(pocket, market)
    want = list(_iso(valid))
    runs = [int(m.get("validate_run", 0) or 0) for st, p, m in
            zip(rows["stage"], rows["period"], rows["metrics"])
            if st == "validate" and list(p) == want and isinstance(m, dict)]
    return max(runs, default=0)


def _logged_period(rows: pd.DataFrame, stage: str) -> list[str] | None:
    """Period [start, end] of the latest `stage` row in the experiment log, or None."""
    rows = rows[rows["stage"] == stage]
    return [str(x) for x in rows.iloc[-1]["period"]] if len(rows) else None


def _logged_cycles(rows: pd.DataFrame) -> dict[int, tuple[str, str, str]]:
    """{cycle: (sealed start, sealed end, data cutoff)} of the latest "cycle_started" row
    per cycle (written by `lab periods --new-cycle`)."""
    out = {}
    started = rows[rows["stage"] == "cycle_started"]
    for p, m in zip(started["period"], started["metrics"]):
        if isinstance(m, dict) and "cycle" in m:
            out[int(m["cycle"])] = (str(p[0]), str(p[1]), str(m.get("data_cutoff", "")))
    return out


def _new_periods(pocket: str, market: str, lab: LabConfig) -> PeriodRecord:
    """The period record for a missing file: rebuilt from the seal tickets when some exist
    (a fresh split could move over an opened seal), else split from the stored dates. A
    rebuilt train / valid must equal the periods of the latest logged screen / validate."""
    tickets = SealRegistry(seal_path()).tickets(pocket, market)
    cal = _pocket_dates(pocket, market, lab, date.today())
    if len(cal) == 0:
        raise _no_data(pocket, market)
    if tickets:
        rows = ExperimentLog(experiments_path()).rows(pocket, market)
        logged = {st: _logged_period(rows, st) for st in ("screen", "validate")}
        try:
            rec = period_store.rebuild_record(pocket, market, tickets, cal, lab.split,
                                              train=logged["screen"],
                                              valid=logged["validate"],
                                              started=_logged_cycles(rows))
            missing = [st for st, p in logged.items() if p is None]
            if missing:
                raise PeriodError(f"no logged {' / '.join(missing)} period to check the "
                                  "rebuilt train / valid against")
        except PeriodError as e:
            raise LabError(f"{periods_path(pocket, market)} is missing and the seal tickets "
                           f"of {pocket}/{market} cannot rebuild it ({e}): restore the periods "
                           "file") from e
        return replace(rec, validate_runs=_validate_runs_logged(pocket, market,
                                                                rec.periods.valid))
    try:
        return period_store.new_record(pocket, market, cal, lab.split)
    except ValueError as e:
        raise LabError(f"not enough {'minute' if pocket == 'daytrade' else 'daily'} data for "
                       f"{market} to split periods: {e}") from e


def _setup(pocket: str, market: str, create: bool = False) -> Setup:
    """Configs + the persisted period split (no prices are read). Only `create` (the first
    `screen`) may make the period record (see `_new_periods`); it is written only after the
    screen's checks passed on it."""
    _check_market(market)
    _families(pocket)
    _refuse_limited(market)
    mc, pc, lab, synced = load_market(market), load_pocket(pocket), load_lab(), load_synced()
    rec = _load_periods(pocket, market)
    if rec is None:
        if not create:
            raise LabError(f"no periods for {pocket}/{market}: run `lab screen` first")
        rec = _new_periods(pocket, market, lab)
        _refuse_opened_overlap(pocket, market, *rec.periods.train, "screen")
        period_store.save_record(periods_path(pocket, market), rec)
    cal = _pocket_dates(pocket, market, lab, rec.data_cutoff)
    if len(cal) == 0:
        raise _no_data(pocket, market)
    return Setup(mc, pc, lab, synced, rec.periods, cal, rec, pocket)


def _refuse_opened_overlap(pocket: str, market: str, start: date, end: date,
                           what: str) -> None:
    """screen / validate never touch a sealed period that some rule already opened."""
    for t in SealRegistry(seal_path()).tickets(pocket, market):
        if t.sealed is None:
            continue
        s0, s1 = (date.fromisoformat(x) for x in t.sealed)
        if start <= s1 and s0 <= end:
            raise SealViolation(f"{what} range {start}~{end} overlaps the sealed period "
                                f"{s0}~{s1} already opened (rule {t.rule_id})")


def _panel(s: Setup, end: date) -> Panel:
    """Prices from lab.history_start (signal warm-up before swing_start) through `end`; its
    `calendar` (trading dates only) is cut at the period record's data cutoff, so dates
    collected later never change what a screen / validate / verify run computes."""
    p = load_panel(s.market.market, None, s.lab.history_start, end)
    idx = p.close.index
    # the daytrade calendar is the minute one: only the swing calendar mirrors the daily bars
    if s.name == "swing" and not idx[idx >= pd.Timestamp(s.lab.swing_start)].equals(
            s.calendar[s.calendar <= pd.Timestamp(end)]):
        raise LabError("stored data changed while running; re-run the command")
    if p.calendar is not None:
        p = replace(p, calendar=p.calendar[p.calendar <= pd.Timestamp(s.record.data_cutoff)])
    # events with signal_date <= end only: no event dated after `end` (e.g. in the sealed
    # period) reaches screen / validate. An event dated `end` itself is kept although it is
    # tradable only at the next period's first open; harmless, simulate never fills an entry
    # signalled on the panel's last row (e >= n)
    return attach_events(p, s.market.market, end)


def _eligible(s: Setup, panel: Panel, scenario: str) -> pd.DataFrame:
    """Universe mask on the panel's index: statistics over [swing_start, ...] only, so the
    warm-up rows (before swing_start) are never eligible and feed no universe window."""
    idx = panel.close.index
    lab_panel = panel.slice(s.lab.swing_start, idx.max()) if len(idx) else panel
    mask = eligible_mask(lab_panel, s.market, s.pocket, s.lab, scenario, s.synced.usd_krw)
    return mask.reindex(index=idx, columns=panel.close.columns, fill_value=False).astype(bool)


def _cost_model(market: MarketConfig, meta: pd.DataFrame) -> CostModel:
    prod = meta["product"].to_dict() if "product" in meta.columns else {}
    return CostModel(market, lambda s: prod.get(s) if prod.get(s) in ("stock", "etf")
                     else "stock")


def _currency_per_krw(market: MarketConfig, synced: SyncedValues) -> float:
    return 1.0 if market.currency == "KRW" else 1.0 / synced.usd_krw


def _iso(p: tuple[date, date]) -> tuple[str, str]:
    return p[0].isoformat(), p[1].isoformat()


def _rule(families: dict, family: str, params: dict, rule_id: str | None = None):
    if family not in families:
        raise LabError(f"unknown family {family!r}")
    rule = families[family](params)
    if rule_id is not None and rule.rule_id != rule_id:
        raise LabError(f"logged params of {rule_id} rebuild a different rule {rule.rule_id}")
    return rule


# ---------------------------------------------------------------- sync
def _pick_commission(rows: list[dict], country: str, today: date
                     ) -> tuple[float, str | None] | None:
    """(rate, endDate or None) of the first row of `country` in effect on `today`."""
    for r in rows:
        if str(r.get("marketCountry")).upper() != country:
            continue
        start, end = r.get("startDate"), r.get("endDate")
        if start and date.fromisoformat(start) > today:
            continue
        if end and date.fromisoformat(end) < today:
            continue
        return float(r["commissionRate"]), (str(end) if end else None)
    return None


def commission_warnings(today: date | None = None) -> list[str]:
    """One warning per market whose synced (promotional) commission ended before `today`;
    nothing when synced.yaml is missing or unreadable."""
    today = today or date.today()
    try:
        synced = load_synced()
    except ConfigError:
        return []
    out = []
    for m, end in sorted(synced.commission_end_date.items()):
        if end and date.fromisoformat(str(end)) < today:
            out.append(f"warning: the synced {m} commission rate ended {end} (synced "
                       f"{synced.synced_at}): run `lab sync` again")
    return out


def _client():
    """Read-only Toss client from the environment; setup problems become LabError (the
    messages name variables, never their values)."""
    try:
        return make_client_from_env(token_path())
    except RuntimeError as e:
        raise LabError(str(e)) from e
    except ValueError as e:
        raise LabError("TOSS_ACCOUNT_SEQ must be an integer") from e


def sync(kr_commission: float | None = None, us_commission: float | None = None,
         today: date | None = None) -> dict:
    """Fetch USD/KRW and commission rates; manual rates are used where given (and where the
    commissions endpoint is unavailable without TOSS_ACCOUNT_SEQ). Writes synced.yaml only
    when both markets have a commission."""
    today = today or date.today()
    manual = {"kr": kr_commission, "us": us_commission}
    for m, v in manual.items():
        if v is not None and not 0 <= v < MAX_COMMISSION:
            raise LabError(f"--{m}-commission must be a decimal ratio below {MAX_COMMISSION} "
                           f"(e.g. 0.00015 for 0.015%), got {v}")
    client = _client()
    try:
        usd_krw = float(client.get_exchange_rate()["rate"])
        api_rows = None
        if any(v is None for v in manual.values()):
            try:
                api_rows = client.get_commissions()
            except ValueError:          # no TOSS_ACCOUNT_SEQ: endpoint needs the account
                api_rows = None
    except TossApiError as e:
        raise LabError(f"sync failed: {e}") from e
    except httpx.HTTPError as e:
        raise LabError(f"sync failed: network error ({type(e).__name__})") from e
    rates, source, ends = {}, {}, {}
    for m, country in (("kr", "KR"), ("us", "US")):
        if manual[m] is not None:
            rates[m], source[m], ends[m] = float(manual[m]), "manual", None
            continue
        v = _pick_commission(api_rows, country, today) if api_rows is not None else None
        if v is None:
            raise LabError(f"no {m} commission: set TOSS_ACCOUNT_SEQ for the commissions "
                           f"API or pass --{m}-commission (decimal ratio)")
        if not 0 <= v[0] < MAX_COMMISSION:
            raise LabError(f"API {m} commission {v[0]} is not a decimal ratio below "
                           f"{MAX_COMMISSION}; check the API response or pass "
                           f"--{m}-commission")
        (rates[m], ends[m]), source[m] = v, "api"
    out = {"commission_rate": rates, "usd_krw": usd_krw,
           "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": {"commission_rate": source, "usd_krw": "api"},
           "commission_end_date": ends}
    path = _synced_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"path": str(path), **out}


# ---------------------------------------------------------------- collect
# failures that hit every symbol alike: abort the whole collect instead of retrying ~2.5k
SYSTEMIC_STATUS = {401: "credentials rejected (401 after one re-auth): check TOSS_CLIENT_ID "
                        "/ TOSS_CLIENT_SECRET",
                   403: "access forbidden (403): check the API key's permissions / IP "
                        "allowlist"}
MAX_CONSECUTIVE_FAILURES = 20


def _collect_failed(market: str, sym: str, e: Exception, failed: list[str],
                    streak: int) -> int:
    """Record one symbol's failure; abort after MAX_CONSECUTIVE_FAILURES in a row (offline,
    API down). Returns the new streak."""
    log.warning("collect %s %s failed: %s: %s", market, sym, type(e).__name__, e)
    failed.append(f"{sym}: {type(e).__name__}: {e}")
    streak += 1
    if streak >= MAX_CONSECUTIVE_FAILURES:    # error type (+ API code) only: one line, and
        code = f" code={e.code}" if isinstance(e, TossApiError) else ""   # no message text
        raise LabError(f"collect aborted after {streak} consecutive symbol failures (last "
                       f"{sym}: {type(e).__name__}{code})") from e
    return streak


def _daily_job_lock() -> Path:
    return data_root() / "logs" / ".lock"


def _wait_for_daily_job() -> None:
    """The daily job holds data/logs/.lock while it runs: stay off the API until it is done."""
    while _daily_job_lock().exists():
        time.sleep(30)


def events_report(market: str, unmatched: int = 20) -> dict:
    """`lab events`: counts by event x year, target coverage, frequent unmatched titles."""
    _check_market(market)
    if load_filings(market).empty:
        raise LabError(f"no filings stored for {market}: run `lab collect-events --market "
                       f"{market}` first")
    out = events_report_of(market, calendar(market)[0], unmatched)
    out["incomplete"] = len(incomplete_targets(market))
    return out


def collect_events(market: str, limit: int | None = None) -> dict:
    """Company filings (DART, KR) of the swing paper-universe stocks into the event store.
    Resumable per symbol; `limit` keeps the first N targets. One run per market at a time
    (flock on events_root/.collect.lock). A DART rate limit stops the run cleanly."""
    _check_market(market)
    if market == "us":
        raise LabError("US filings collector arrives with the EDGAR task")
    path = events_lock_path(market)
    with _exclusive_lock(path, f"`lab collect-events --market {market}` is already running "
                               f"(lock {path}): wait for it to finish"):
        syms = event_targets(market)
        meta = load_meta(market)
        delisted = {str(s) for s in meta.index[meta["delisted"].astype(bool)]}
        client = make_dart_client_from_env()
        try:
            out = collect_events_kr(client, syms, today=market_today(market).date(),
                                    delisted=delisted, limit=limit)
            if not out["rate_limited"]:             # sizes after listing, same run
                sz = collect_sizes_kr(client)
                out["sizes_added"] = sz["added"]
                out["rate_limited"] = sz["rate_limited"]
                out["sizes_stopped"] = sz["stopped"]
            return out
        except httpx.HTTPError as e:
            raise LabError(f"collect-events failed: network error ({type(e).__name__})") from e


def collect_minute(market: str, limit: int | None = None) -> dict:
    """1-minute bars back to the market's minute_start for the daily universe symbols
    (top lab.universe_top_n on any day since then, plus always_include), most universe days
    first. Resumable; `limit` keeps the first N symbols and marks the minute data as partial
    (`minute_limited_marker`; daytrade screen / validate refuse it) until a full run reaches
    the end. Skips stock-not-found symbols; MAX_CONSECUTIVE_FAILURES in a row (or a Toss
    401/403) abort. One run per market at a time: the whole run holds `collect_lock_path`
    (flock), a second run refuses to start (two writers would merge and rewrite the same
    month files and drop each other's bars)."""
    _check_market(market)
    mc = load_market_raw(market)
    if mc.minute_start is None:
        raise LabError(f"{market} has no minute_start in its config")
    with _collect_lock(market):
        return _collect_minute_locked(market, mc, limit)


def _collect_minute_locked(market: str, mc: MarketConfig, limit: int | None) -> dict:
    today = market_today(market)
    syms = minute_symbols(market, mc.minute_start, today.date(), load_lab().universe_top_n)
    marker = minute_limited_marker(market)
    if limit is not None:
        syms = syms[:limit]
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"lab collect-minute --limit {limit}\n", encoding="utf-8")
    client = _client()
    out = {"symbols": len(syms), "added": 0, "complete": 0, "not_found": 0, "failed": 0}
    failed: list[str] = []
    streak = 0
    try:
        for sym in syms:
            _wait_for_daily_job()
            try:
                rep = collect_minutes(client, market, sym, mc.minute_start, today)
                out["added"] += rep.added
                out["complete"] += int(rep.complete)
                streak = 0
            except TossApiError as e:
                if e.status == 404 and e.code == "stock-not-found":
                    out["not_found"] += 1
                    streak = 0
                elif e.status in SYSTEMIC_STATUS:
                    raise LabError(f"collect-minute aborted at {sym}: "
                                   f"{SYSTEMIC_STATUS[e.status]} (Toss API status={e.status} "
                                   f"code={e.code})") from e
                else:
                    streak = _collect_failed(market, sym, e, failed, streak)
            except (LabError, KeyboardInterrupt):
                raise
            except Exception as e:      # noqa: BLE001 - messages carry no secrets
                streak = _collect_failed(market, sym, e, failed, streak)
    except httpx.HTTPError as e:
        raise LabError(f"collect-minute failed: network error ({type(e).__name__})") from e
    out["failed"] = len(failed)
    if limit is None:               # a full run went over every symbol
        marker.unlink(missing_ok=True)
    return out


def collect(market: str, since: date | None = None, limit: int | None = None) -> dict:
    """Meta -> incremental daily bars (Toss) from `since` (default lab.history_start) ->
    (KR) pykrx delisted supplement (its counts saved to data/<market>/supplement.json for the
    reports) -> quality summary. `limit` keeps the first N symbols, always_include first, and
    marks the data as partial (screen / validate / verify --final refuse it) until a full
    collect. A Toss 401/403 or MAX_CONSECUTIVE_FAILURES failing symbols in a row abort it."""
    _check_market(market)
    mc = load_market_raw(market)
    start = since or load_lab().history_start      # swing_start - warmup_days
    client = _client()
    try:
        meta = collect_meta(client, market)
        if mc.delisted_supplement:      # Toss lists no delisted KR symbols: KIND does
            try:
                meta = add_delisted_meta(meta, kind_delisted(start))
            except (KindError, httpx.HTTPError) as e:
                raise LabError(f"collect failed: KIND delisting list unavailable "
                               f"({type(e).__name__}); without it the delisted supplement "
                               "would be empty (survivorship bias)") from e
            save_meta(market, meta)
        syms = [str(s) for s in meta.index]
        first = [s for s in mc.always_include if s in set(syms)]
        ordered = first + sorted(s for s in syms if s not in set(first))
        if limit is not None:
            ordered = ordered[:limit]
        delisted = set(str(s) for s in meta.index[meta["delisted"].astype(bool)])
        via_pykrx = [s for s in ordered if s in delisted] if mc.delisted_supplement else []
        via_toss = [s for s in ordered if s not in set(via_pykrx)]
        added, failed, streak = 0, [], 0
        for sym in via_toss:            # one failing symbol is reported, the run continues
            try:
                added += collect_daily(client, market, sym, start)
                streak = 0
            except TossApiError as e:
                if e.status in SYSTEMIC_STATUS:     # the client already re-authed once on 401
                    raise LabError(f"collect aborted at {sym}: {SYSTEMIC_STATUS[e.status]} "
                                   f"(Toss API status={e.status} code={e.code})") from e
                streak = _collect_failed(market, sym, e, failed, streak)
            except Exception as e:      # noqa: BLE001 - messages carry no secrets
                streak = _collect_failed(market, sym, e, failed, streak)
    except TossApiError as e:
        raise LabError(f"collect failed: {e}") from e
    except httpx.HTTPError as e:
        raise LabError(f"collect failed: network error ({type(e).__name__})") from e
    overlay = None
    if market == "kr":              # KRX regular-session bars in place of KRX+NXT ones
        have = [s for s in via_toss if load_daily(market, s) is not None]
        overlay = refresh_overlays(have, market_today(market))
        save_overlay_record(overlay, len(have))
    supplement = None
    if via_pykrx:
        supplement = supplement_delisted_kr(start, symbols=via_pykrx)
        if supplement is not None:      # what reports state, instead of the config flag
            save_supplement_record(supplement, start, len(via_pykrx), market)
    marker = limited_marker(market)
    if limit is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"lab collect --limit {limit}\n", encoding="utf-8")
    else:
        marker.unlink(missing_ok=True)
    summary = {"symbols": len(ordered), "rows_added": added, "clean": 0, "issues": 0,
               "warnings": 0, "missing": 0, "failed": failed}
    for sym in ordered:
        df = load_bars(market, sym)
        if df is None or df.empty:
            summary["missing"] += 1
        elif hard_issues(df):
            summary["issues"] += 1
        elif check_daily(df):
            summary["warnings"] += 1
        else:
            summary["clean"] += 1
    _, dropped = calendar(market)
    summary["calendar_dropped"] = [d.date().isoformat() for d in dropped]
    if overlay is not None:
        summary["krx_applied"] = overlay.applied
        summary["krx_mismatch"] = len(overlay.mismatch)
        summary["krx_empty"] = len(overlay.empty)
        summary["krx_failed"] = len(overlay.failed)
    if supplement is not None:
        summary["delisted_fetched"] = supplement.fetched
        summary["delisted_empty"] = len(supplement.empty)
        summary["delisted_failed"] = len(supplement.failed)
        summary["delisted_skipped"] = len(supplement.skipped)
    return summary


# ---------------------------------------------------------------- universe
def universe(market: str, pocket: str, scenario: str, day: date) -> list[str]:
    _check_market(market)
    if scenario not in SCENARIOS:
        raise LabError(f"unknown scenario {scenario!r} (choose from {', '.join(SCENARIOS)})")
    mc, pc, lab, synced = load_market(market), load_pocket(pocket), load_lab(), load_synced()
    rec = _load_periods(pocket, market)
    if rec is not None and day >= rec.periods.sealed[0]:
        raise LabError(f"{day} is not before the sealed period of {pocket}/{market} (from "
                       f"{rec.periods.sealed[0]}): lab universe shows train/valid dates only")
    p = load_panel(market, None, lab.swing_start, day)
    ts = pd.Timestamp(day)
    if len(p.close.index) == 0:
        raise LabError(f"no daily data for {market}: run `lab collect --market {market}` first")
    if ts not in p.close.index:
        raise LabError(f"{day} is not a trading day in the stored {market} data")
    mask = eligible_mask(p, mc, pc, lab, scenario, synced.usd_krw)
    row = mask.loc[ts]
    return sorted(str(s) for s in row.index[row.to_numpy()])


# ---------------------------------------------------------------- screen
def screen(pocket: str, market: str, family: str = "all") -> pd.DataFrame:
    """Every grid point of the chosen families on the train period (fixed amount, paper
    universe), logged as screen trials; sorted by mean_net_r."""
    _check_market(market)
    fams = _families(pocket)
    names = list(fams) if family == "all" else [family]
    for n in names:
        if n not in fams:
            raise LabError(f"unknown family {n!r} (choose from all, {', '.join(fams)})")
    if pocket == "daytrade":
        _check_market(market)
        _refuse_incomplete_minutes(market)
    s = _setup(pocket, market, create=True)
    t0, t1 = s.periods.train
    _refuse_opened_overlap(pocket, market, t0, t1, "screen")
    panel = _panel(s, t1)
    elig = _eligible(s, panel, "paper")
    cm = _cost_model(s.market, panel.meta)
    log = ExperimentLog(experiments_path())
    frames = []
    if pocket == "daytrade":
        rules = _session_rules(s.market)
        syms = _minute_universe(market)
        for n in names:
            df = screen_intraday(n, market, syms, t0, t1, panel, elig, cm, rules, log, pocket,
                                 sealed_start=s.periods.sealed[0],
                                 cost_r_max=s.lab.criteria.cost_r_max)
            frames.append(df.assign(family=n))
        return _screen_table(frames)
    for n in names:
        df = screen_family(n, panel, elig, t0, t1, cm, log, pocket, market,
                           sealed_start=s.periods.sealed[0])
        frames.append(df.assign(family=n))
    return _screen_table(frames)


def _screen_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("mean_net_r", ascending=False, na_position="last",
                           kind="stable").reset_index(drop=True)


def _session_rules(mc: MarketConfig) -> SessionRules:
    """Paper-mode session rules: KR stops entering at each day's closing-auction start (close
    - 10 min) and has VI auctions; US trades to the close. Exits run to the day's last
    regular bar (no early exit lead)."""
    return SessionRules(last_entry_lead=auction_lead_minutes(mc), exit_lead_minutes=0,
                        vi=mc.vi)


def _minute_universe(market: str) -> list[str]:
    """Symbols that have stored minute data, sorted. Deliberately independent of any daily
    data: the screen's symbol set must not look past the train end."""
    root = minute_dir(market, "x").parent
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


# ---------------------------------------------------------------- validate
@dataclass
class ValidateResult:
    table: pd.DataFrame          # the `top` candidates
    pbo: float                   # over the whole pool
    recommended: list[str]
    n_pool: int = 0
    benchmark: str | None = None
    benchmark_return: float | None = None


def benchmark_return(market: str, symbol: str, start: date, end: date) -> float:
    """Buy-and-hold price return of `symbol` from its first to its last close in [start, end]
    (no costs, so the bar a rule must clear is, if anything, slightly high)."""
    df = load_bars(market, symbol)
    if df is None or df.empty:
        raise LabError(f"benchmark {symbol} has no stored prices: run `lab collect --market {market}`")
    c = df["close"][(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].dropna()
    if len(c) < 2:
        raise LabError(f"benchmark {symbol} has no prices in {start}~{end}")
    return float(c.iloc[-1] / c.iloc[0] - 1.0)


def _pbo(perf: pd.DataFrame) -> float:
    if perf.shape[1] < 2:
        return math.nan
    for n in PBO_SPLITS:
        try:
            return pbo_cscv(perf, n_splits=n)
        except ValueError:
            continue
    return math.nan


def _daily_returns(res: PortfolioResult, calendar, period, capital0: float) -> pd.Series:
    eq = period_equity(res, calendar, period[0], period[1], capital0)
    return eq.pct_change().dropna()


def validate(pocket: str, market: str, top: int) -> ValidateResult:
    """The pool = every rule screened on the current train period with >= MIN_TRAIN_TRADES
    train trades (fixed; `top` does not change it): fixed-amount net sums on VALID_SEGMENTS
    consecutive segments of the valid period ("검증 기간 구간 안정성", params fixed),
    paper portfolio on the valid period, PBO over the whole pool's valid daily returns, each
    pool rule logged as stage "validate" (the trial-variance pool of verify). `top` only
    picks the candidates printed and recommended: the first `top` by train mean_net_r;
    recommend up to max_final_candidates of them with valid net_pnl > 0 and a valid return
    (net_pnl / capital) above the market benchmark's buy-and-hold return, by valid daily
    Sharpe. Every run increments the period file's validate_runs counter.

    daytrade: the same pool / segments / PBO / logging, but the valid trades are the executed
    same-day trades of the rule on the valid period (minute bars loaded only through the valid
    end; the segments cut those trades by entry day), the portfolio is
    `run_intraday_portfolio`, and the benchmark gate applies only when the pocket's
    `benchmark_gate` is true (daytrade: false). Daily returns include the first day's return
    from the initial capital and zero days; metrics add valid_mean_daily_return,
    valid_positive_day_share (share of days with return > 0 among the days with at least one
    trade) and valid_worst_day (min daily return)."""
    if top < 1:
        raise LabError("--top must be >= 1")
    if pocket == "daytrade":
        _check_market(market)
        _refuse_incomplete_minutes(market)
    s = _setup(pocket, market)
    fams = _families(pocket)
    log = ExperimentLog(experiments_path())
    rows = log.rows(pocket, market)
    train = list(_iso(s.periods.train))
    scr = rows[(rows["stage"] == "screen") & rows["period"].map(lambda p: list(p) == train)]
    scr = scr.drop_duplicates("rule_id", keep="last")
    if scr.empty:
        raise LabError(f"no screen results for {pocket}/{market} on the current train period "
                       f"{train[0]}~{train[1]}: run `lab screen` first")
    scr = scr.assign(n_trades=[m.get("n_trades", 0) for m in scr["metrics"]],
                     mean_net_r=[m.get("mean_net_r", math.nan) for m in scr["metrics"]])
    cands = scr[scr["n_trades"] >= MIN_TRAIN_TRADES].sort_values(
        "mean_net_r", ascending=False, na_position="last", kind="stable")
    if cands.empty:
        raise LabError(f"no screened rule has >= {MIN_TRAIN_TRADES} train trades")
    top_ids = list(cands["rule_id"].head(top))

    (t0, _), (v0, v1) = s.periods.train, s.periods.valid
    _refuse_opened_overlap(pocket, market, t0, v1, "validate")
    panel = _panel(s, v1)
    daytrade = pocket == "daytrade"
    cal = s.calendar if daytrade else panel.close.index
    elig = _eligible(s, panel, "paper")
    cm = _cost_model(s.market, panel.meta)
    cpk = _currency_per_krw(s.market, s.synced)
    syms = _minute_universe(market) if daytrade else []
    session = _session_rules(s.market) if daytrade else None
    capital = s.lab.capital["paper"]
    segments = period_segments(v0, v1, n=VALID_SEGMENTS)
    bench = s.market.benchmark if s.pocket.benchmark_gate else None
    bench_ret = benchmark_return(market, bench, v0, v1) if bench else None

    fam_trades: dict[str, list] = {}       # rule_id -> trade frames (daytrade: one pass per family)
    if daytrade:
        fstart = max((pd.Timestamp(v0) - pd.Timedelta(days=FEATURE_WARMUP_DAYS)),
                     s.calendar[0]).date()   # train-period bars only, never the sealed ones
        for fam in dict.fromkeys(cands["family"]):
            grid = [_rule(fams, f, p, r) for f, p, r in zip(cands["family"], cands["params"],
                                                            cands["rule_id"]) if f == fam]
            fam_trades.update(intraday_trades(grid, market, syms, v0, v1, panel, elig, cm,
                                              session, features_start=fstart)[0])
    metrics, returns = {}, {}
    for fam, params, rid, scr_r in zip(cands["family"], cands["params"], cands["rule_id"],
                                       cands["mean_net_r"]):
        rule = _rule(fams, fam, params, rid)
        extra = {}
        if daytrade:
            parts = fam_trades[rid]
            trades = pd.concat(parts, ignore_index=True) if parts \
                else pd.DataFrame(columns=TRADE_COLUMNS)
            net = cm.net_returns(trades)
            day = local_day(net["entry_date"])
            seg_sums = [float(net["net_return"][(day >= pd.Timestamp(a))
                                                & (day <= pd.Timestamp(b))].sum())
                        for a, b in segments]
            res = run_intraday_portfolio(trades, cm, s.pocket, capital, cpk, False)
            eq = period_equity(res, cal, v0, v1, capital * cpk)
            rets = pd.concat([pd.Series([capital * cpk], index=[eq.index[0] - pd.Timedelta(days=1)]),
                              eq]).pct_change().dropna()
            traded = rets[rets.index.isin(local_day(res.fills["entry_date"]))]
            extra = {"valid_mean_daily_return": float(rets.mean()),
                     "valid_positive_day_share":
                         float((traded > 0).mean()) if len(traded) else math.nan,
                     "valid_worst_day": float(rets.min())}
        else:
            sig = rule.signals(panel, eligible=elig)  # history through the valid end
            seg_sums = []
            for a, b in segments:                     # params fixed; valid period only
                trades = simulate(panel, sig, elig, a, b)
                seg_sums.append(float(cm.net_returns(trades)["net_return"].sum()))
            res = run_portfolio(simulate(panel, sig, elig, v0, v1), panel, cm, s.pocket,
                                capital, cpk, False, entry_at=sig.entry_at,
                                exit_at=sig.exit_at, equal_weight=sig.equal_weight)
            eq = period_equity(res, cal, v0, v1, capital * cpk)
            rets = eq.pct_change().dropna()
        returns[rid] = rets
        metrics[rid] = {
            "family": fam, "params": params, "screen_mean_net_r": scr_r,
            "valid_segments": [[a.isoformat(), b.isoformat()] for a, b in segments],
            "valid_segment_net_sums": seg_sums,
            "valid_segments_positive": int(sum(x > 0 for x in seg_sums)),
            "valid_n_trades": int(len(res.fills)),
            "valid_net_pnl": float(res.fills["net_pnl"].astype(float).sum()),
            "valid_return": float(res.fills["net_pnl"].astype(float).sum()) / (capital * cpk),
            "benchmark": bench, "benchmark_return": bench_ret,
            "valid_daily_sharpe": sharpe(rets.to_numpy(), periods_per_year=1),
            "valid_sharpe_annual": sharpe(rets.to_numpy()),
            "valid_mdd": max_drawdown(eq), "scenario": "paper", "capital_krw": capital,
            "capital_ccy": capital * cpk, "currency": s.market.currency, **extra,
        }
    pbo = _pbo(pd.DataFrame(returns).fillna(0.0))
    ranked = sorted((rid for rid in top_ids if metrics[rid]["valid_net_pnl"] > 0
                     and (bench_ret is None or metrics[rid]["valid_return"] > bench_ret)),
                    key=lambda r: (-metrics[r]["valid_daily_sharpe"]
                                   if np.isfinite(metrics[r]["valid_daily_sharpe"])
                                   else math.inf))
    recommended = ranked[: s.lab.criteria.max_final_candidates]
    rec = replace(s.record, validate_runs=s.record.validate_runs + 1)
    period_store.save_record(periods_path(pocket, market), rec)
    for rid, m in metrics.items():
        m.update(pbo=pbo, pbo_candidates=len(metrics), in_top=rid in top_ids,
                 recommended=rid in recommended, recommended_ids=list(recommended),
                 validate_run=rec.validate_runs)
        log.record(pocket, market, m["family"], rid, m["params"], "validate", _iso((v0, v1)),
                   {k: v for k, v in m.items() if k not in ("family", "params")})
    cols = ["valid_net_pnl", "valid_return", "valid_daily_sharpe", "valid_n_trades",
            "valid_segments_positive", "recommended"]
    if daytrade:
        cols += ["valid_mean_daily_return", "valid_positive_day_share", "valid_worst_day"]
    table = pd.DataFrame([{"rule_id": rid, **{k: metrics[rid][k] for k in cols}}
                          for rid in top_ids])
    table = table.sort_values("valid_daily_sharpe", ascending=False, na_position="last",
                              kind="stable").reset_index(drop=True)
    return ValidateResult(table, pbo, recommended, len(metrics), bench, bench_ret)


# ---------------------------------------------------------------- verify --final
def nautilus_result(trades: pd.DataFrame, panel: Panel, cost_model: CostModel,
                    capital0: float, start: date, end: date) -> PortfolioResult:
    """PortfolioResult from run_nautilus trades: fills get net_r like run_portfolio; equity is
    cash + positions marked at the (forward-filled) close on every panel day in [start, end].
    Entry cash is `cost_model.fill_cash(buy)` (what the Nautilus fee model charges), the exit
    returns that plus the trade's net_pnl, so the final equity is capital0 + sum(net_pnl)."""
    fills = trades.copy()
    if len(fills):
        fills["net_r"] = fills["net_pnl"] / (fills["qty"] * fills["entry_price"]
                                             * fills["stop_pct"])
    else:
        fills["net_r"] = pd.Series(dtype=float)
    idx = panel.close.index
    days = idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]
    close = panel.close.ffill()
    cash, values = capital0, []
    open_pos: dict[int, tuple[str, float]] = {}
    buy_cash: dict[int, float] = {}
    by_entry = {d: g.index.tolist() for d, g in fills.groupby("entry_date")} if len(fills) else {}
    by_exit = {d: g.index.tolist() for d, g in fills.groupby("exit_date")} if len(fills) else {}
    for d in days:
        for i in by_entry.get(d, []):
            sym, qty = fills.at[i, "symbol"], fills.at[i, "qty"]
            buy_cash[i] = cost_model.fill_cash(sym, "buy", qty, fills.at[i, "entry_price"])
            cash -= buy_cash[i]
            open_pos[i] = (sym, qty)
        for i in by_exit.get(d, []):
            cash += buy_cash.pop(i) + fills.at[i, "net_pnl"]
            open_pos.pop(i, None)
        v = cash
        for sym, qty in open_pos.values():
            px = close.at[d, sym]
            v += qty * (px if np.isfinite(px) else 0.0)
        values.append(v)
    equity = pd.Series(values, index=pd.DatetimeIndex(days, name="date"), name="equity")
    return PortfolioResult(equity, fills, pd.DataFrame())


def _sr_var_trials(log_rows: pd.DataFrame, valid_period: list[str]) -> float | None:
    """Variance (ddof=1) of the validate-stage daily Sharpes on `valid_period` (latest per
    rule; validate logs the whole screened pool); None if < 2."""
    val = log_rows[(log_rows["stage"] == "validate")
                   & log_rows["period"].map(lambda p: list(p) == list(valid_period))]
    val = val.drop_duplicates("rule_id", keep="last")
    srs = np.array([float(m.get("valid_daily_sharpe", math.nan)) for m in val["metrics"]])
    srs = srs[np.isfinite(srs)]
    return float(np.var(srs, ddof=1)) if len(srs) >= 2 else None


def _latest_recommended(log_rows: pd.DataFrame, valid_period: list[str]) -> list[str]:
    """`recommended_ids` of the latest `lab validate` run on `valid_period` (its last logged
    row; every row of a run carries the run's list); [] without one."""
    val = log_rows[(log_rows["stage"] == "validate")
                   & log_rows["period"].map(lambda p: list(p) == list(valid_period))]
    if val.empty or not isinstance(val.iloc[-1]["metrics"], dict):
        return []
    return [str(r) for r in val.iloc[-1]["metrics"].get("recommended_ids") or []]


def _parity_summary(rep: dict) -> dict:
    out = {k: v for k, v in rep.items() if k != "mismatches"}
    out["n_mismatches"] = len(rep["mismatches"])
    out["mismatches"] = rep["mismatches"][:PARITY_MISMATCHES_KEPT]
    return out


def _final_scenarios(rule, panel: Panel, s: Setup, pbo: float, n_trials: int,
                     sr_var: float | None) -> tuple[dict, dict]:
    """Per scenario: Nautilus on the sealed period (verdict), run_portfolio on the same
    inputs (parity), run_portfolio on the valid period (valid_positive)."""
    (v0, v1), (s0, s1) = s.periods.valid, s.periods.sealed
    cm = _cost_model(s.market, panel.meta)
    cpk = _currency_per_krw(s.market, s.synced)
    cal = panel.close.index
    first = cal[0]
    vpanel = panel.slice(first, v1)
    scenarios, parity = {}, {}
    for scen in SCENARIOS:
        capital = s.lab.capital[scen]
        cap0 = capital * cpk
        frac = s.market.fractional_orders if scen == "real" else False
        elig = _eligible(s, panel, scen)
        velig = elig.loc[elig.index <= pd.Timestamp(v1)]
        sig = rule.signals(panel, eligible=elig)             # history through sealed end
        vsig = rule.signals(vpanel, eligible=velig)          # history through valid end
        naut = run_nautilus(rule, panel, elig, s0, s1, cm, s.pocket, capital, cpk, frac)
        ours = run_portfolio(simulate(panel, sig, elig, s0, s1), panel, cm, s.pocket, capital,
                             cpk, frac, entry_at=sig.entry_at, exit_at=sig.exit_at,
                             equal_weight=sig.equal_weight)
        parity[scen] = _parity_summary(parity_report(ours.fills, naut))
        sealed_res = nautilus_result(naut, panel, cm, cap0, s0, s1)
        valid_res = run_portfolio(simulate(vpanel, vsig, velig, v0, v1), vpanel, cm, s.pocket,
                                  capital, cpk, frac, entry_at=vsig.entry_at,
                                  exit_at=vsig.exit_at, equal_weight=vsig.equal_weight)
        if sr_var is None:      # conservative fallback: the sealed Sharpe itself, squared
            sr = sharpe(_daily_returns(sealed_res, cal, (s0, s1), cap0).to_numpy(),
                        periods_per_year=1)
            var, var_src = (sr ** 2 if np.isfinite(sr) else math.nan), "sealed_sr_squared"
        else:
            var, var_src = sr_var, "validate_daily_sharpes"
        v = evaluate(sealed_res, valid_res, pbo, n_trials, var, s.lab.criteria, calendar=cal,
                     period=(s0, s1), initial_capital=cap0)
        cost_r = [cm.round_trip_cost_pct(sym, px) / sp for sym, px, sp in
                  zip(naut["symbol"], naut["entry_price"], naut["stop_pct"])]
        scenarios[scen] = {
            "capital_krw": capital, "capital_ccy": cap0, "currency": s.market.currency,
            "currency_per_krw": cpk,
            "fractional": frac, "passed": v.passed, "failed": list(v.failed),
            "sr_var_trials_source": var_src,
            "mean_cost_r": float(np.mean(cost_r)) if cost_r else math.nan,
            "n_data_end": int((naut["exit_reason"] == "data_end").sum()),
            "valid_n_trades": int(len(valid_res.fills)),
            "metrics": v.metrics,
        }
    return scenarios, parity


def verify_final(pocket: str, market: str, rule_id: str) -> tuple[RuleCard, Path, Path]:
    """Open the seal for `rule_id` (once per pocket x market), then evaluate it on the sealed
    period in NautilusTrader for both capital scenarios and write the rule card. Cycle 1 opens
    only a rule recommended by the latest `lab validate` run on the current valid period;
    later cycles only the previous cycle's final candidates."""
    _swing_only(pocket, "verify --final")
    s = _setup(pocket, market)
    fams = _families(pocket)
    log = ExperimentLog(experiments_path())
    rows = log.rows(pocket, market)
    mine = rows[rows["rule_id"] == rule_id]
    scr, val = mine[mine["stage"] == "screen"], mine[mine["stage"] == "validate"]
    if scr.empty:
        raise LabError(f"rule {rule_id} has no screen record for {pocket}/{market}")
    if val.empty:
        raise LabError(f"rule {rule_id} has no validate record: run `lab validate` first")
    vrow = val.iloc[-1]
    if list(vrow["period"]) != list(_iso(s.periods.valid)):
        raise LabError(f"stored data changed since validate (valid period "
                       f"{vrow['period'][0]}~{vrow['period'][1]} vs now "
                       f"{s.periods.valid[0]}~{s.periods.valid[1]}): re-run `lab screen` "
                       "and `lab validate`")
    cycle = s.record.cycle
    if cycle == 1:           # cycle >= 2: the previous cycle's finalists (SealRegistry.open)
        latest = _latest_recommended(rows, list(_iso(s.periods.valid)))
        if rule_id not in latest:
            raise LabError(f"rule {rule_id} is not recommended by the latest `lab validate` "
                           f"run on the valid period {s.periods.valid[0]}~"
                           f"{s.periods.valid[1]} (recommended: "
                           f"{', '.join(latest) or 'none'}): verify --final opens only those")
    family, params = scr.iloc[-1]["family"], scr.iloc[-1]["params"]
    rule = _rule(fams, family, params, rule_id)
    n_trials = log.n_trials(pocket, market)
    pbo = float(vrow["metrics"].get("pbo", math.nan))
    sr_var = _sr_var_trials(rows, list(_iso(s.periods.valid)))

    registry = SealRegistry(seal_path(), s.lab.criteria.max_final_candidates)
    allowed = None
    if cycle > 1:            # spec 7.9: re-validate only the previous cycle's final candidates
        allowed = {t.rule_id for t in registry.tickets(pocket, market) if t.cycle == cycle - 1}
    ticket = registry.open(pocket, market, rule_id, sealed=_iso(s.periods.sealed),
                           data_cutoff=s.record.data_cutoff.isoformat(),
                           n_symbols=len(stored_symbols(market)), cycle=cycle,
                           allowed_rules=allowed)    # before any sealed price is read
    panel = _panel(s, s.periods.sealed[1])
    sealed = registry.load_sealed(panel, s.periods, ticket)  # validates the ticket
    if not sealed.close.index.equals(s.calendar[(s.calendar >= pd.Timestamp(
            s.periods.sealed[0]))]):
        raise LabError("sealed slice does not match the period split")

    scenarios, parity = _final_scenarios(rule, panel, s, pbo, n_trials, sr_var)
    real_label = VERDICT_REAL_US if market == "us" else VERDICT_REAL
    verdict = {VERDICT_PAPER: scenarios["paper"]["passed"],
               real_label: scenarios["paper"]["passed"] and scenarios["real"]["passed"]}
    final_metrics = {scen: {"passed": d["passed"], "failed": d["failed"], **d["metrics"],
                            "sr_var_trials_source": d["sr_var_trials_source"]}
                     for scen, d in scenarios.items()}
    final_metrics["verdict"] = verdict
    final_metrics["validate_runs"] = s.record.validate_runs
    final_metrics["cycle"] = cycle
    final_metrics["parity_matched_share"] = {k: p["matched_share"] for k, p in parity.items()}
    log.record(pocket, market, family, rule_id, params, "final", _iso(s.periods.sealed),
               final_metrics)
    card = RuleCard(
        pocket=pocket, market=market, rule_id=rule_id, family=family, params=params,
        confidence_formula=CONFIDENCE_FORMULAS.get(family, "(미기재)"),
        periods={k: list(_iso(getattr(s.periods, k))) for k in ("train", "valid", "sealed")},
        metrics_by_stage={"screen": scr.iloc[-1]["metrics"], "validate": vrow["metrics"]},
        scenarios=scenarios, verdict=verdict, parity=parity,
        limitations=limitations(s.market, family=family, pocket=pocket),
        validate_runs=s.record.validate_runs, currency=s.market.currency, cycle=cycle)
    md, js = write_rule_card(card, cards_dir())
    return card, md, js


# ---------------------------------------------------------------- periods
def periods(pocket: str, market: str, new_cycle: bool = False) -> dict:
    """The persisted periods of pocket x market and the seal tickets opened on them;
    `new_cycle` first starts a re-validation cycle (spec 7.9): needs a final candidate opened
    in the current cycle and >= 63 trading days stored after the current sealed end."""
    _check_market(market)
    _families(pocket)
    rec = _load_periods(pocket, market)
    if rec is None:
        raise LabError(f"no periods for {pocket}/{market}: run `lab screen` first")
    tickets = SealRegistry(seal_path()).tickets(pocket, market)
    if new_cycle:
        _swing_only(pocket, "periods --new-cycle")
        if not any(t.cycle == rec.cycle for t in tickets):
            raise LabError(f"no final candidate opened the seal in cycle {rec.cycle}: "
                           "nothing to re-validate")
        dates = trading_dates(market, load_lab().swing_start, date.today())
        try:
            rec = period_store.new_cycle(rec, dates)
        except PeriodError as e:
            raise LabError(str(e)) from e
        period_store.save_record(periods_path(pocket, market), rec)
        ExperimentLog(experiments_path()).record(     # durable: a lost file is rebuilt in it
            pocket, market, "-", "-", {}, "cycle_started", _iso(rec.periods.sealed),
            {"cycle": rec.cycle, **{k: list(_iso(getattr(rec.periods, k)))
                                    for k in ("train", "valid", "sealed")},
             "data_cutoff": rec.data_cutoff.isoformat(),
             "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return {"path": str(periods_path(pocket, market)), "record": rec, "tickets": tickets}


# ---------------------------------------------------------------- report / journal
def report(pocket: str, market: str) -> Path:
    _check_market(market)
    out = leaderboard_path(pocket, market)
    return write_leaderboard(ExperimentLog(experiments_path()), pocket, market, out,
                             limitations=limitations(load_market_raw(market), pocket=pocket),
                             cards_rel=Path(os.path.relpath(cards_dir(), out.parent)).as_posix())


def journal(memo: str, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    path = journal_path(now.date())
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if new:
            f.write(f"# {now.date().isoformat()} 실험 일지\n\n")
        f.write(f"- {now.strftime('%H:%M:%S')} {memo}\n")
    return path


__all__ = ["ConfigError", "LabError", "collect", "commission_warnings", "journal",
           "nautilus_result", "periods",
           "report", "screen", "sync", "universe", "validate", "verify_final"]
