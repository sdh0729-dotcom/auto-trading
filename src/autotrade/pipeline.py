"""Research workflow behind the `lab` CLI: sync -> collect -> universe -> screen -> validate
-> verify --final -> report -> journal.

Periods: on the first `screen` of a pocket x market the stored trading dates from
`lab.swing_start` on are split by `lab.split` into train / valid / sealed and persisted in
research/periods-{pocket}-{market}.json (with the data cutoff); every later command reads that
file, so later collects never move a boundary (`lab periods --new-cycle` starts a
re-validation cycle, spec 7.9). `screen` loads prices only through the train end, `validate`
only through the valid end, and both refuse ranges overlapping an already-opened sealed
period; the sealed prices are loaded by `verify_final` after it has opened the seal ticket. Signals are always computed on the panel history through the end of the
period being evaluated (warm-up), while simulate / run_portfolio / run_nautilus are limited
to the period itself.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
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
from autotrade.data.collect import collect_daily, collect_meta
from autotrade.data.delisted_kr import supplement_delisted_kr
from autotrade.data.quality import check_daily, hard_issues
from autotrade.data.store import (calendar, data_root, load_daily, load_panel, stored_symbols,
                                  trading_dates)
from autotrade.report import (CONFIDENCE_FORMULAS, RuleCard, limitations, write_leaderboard,
                              write_rule_card)
from autotrade.rules.registry import SWING_FAMILIES
from autotrade.screen.run import MIN_TRAIN_TRADES, screen_family
from autotrade.screen.simulate import simulate
from autotrade.sizing import PortfolioResult, run_portfolio
from autotrade.toss.client import TossApiError, make_client_from_env
from autotrade.types import Panel
from autotrade.universe.build import eligible_mask
from autotrade.validate.criteria import evaluate, period_equity
from autotrade.validate.experiments import ExperimentLog
from autotrade.validate import periods as period_store
from autotrade.validate.pbo import pbo_cscv
from autotrade.validate.periods import PeriodError, PeriodRecord
from autotrade.validate.seal import SealRegistry, SealViolation
from autotrade.validate.split import Periods, walk_forward_windows
from autotrade.validate.stats import max_drawdown, sharpe
from autotrade.verify import parity_report, run_nautilus

MARKETS = ("kr", "us")
SCENARIOS = ("paper", "real")
WALK_FORWARD_WINDOWS = 4
PBO_SPLITS = (16, 8, 4, 2)
MAX_COMMISSION = 0.01            # a commission ratio >= 1% is almost surely a percent typo
PARITY_MISMATCHES_KEPT = 20
FAMILIES = {"swing": SWING_FAMILIES}
VERDICT_PAPER, VERDICT_REAL = "② 모의 자격", "실돈 자격"


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
    record: PeriodRecord


def _check_market(market: str) -> None:
    if market not in MARKETS:
        raise LabError(f"unknown market {market!r} (choose from {', '.join(MARKETS)})")


def _families(pocket: str) -> dict:
    if pocket not in FAMILIES:
        raise LabError(f"no rule families for pocket {pocket!r} yet "
                       f"(available: {', '.join(FAMILIES)})")
    return FAMILIES[pocket]


def _load_periods(pocket: str, market: str) -> PeriodRecord | None:
    try:
        return period_store.load_record(periods_path(pocket, market))
    except PeriodError as e:
        raise LabError(str(e)) from e


def _setup(pocket: str, market: str, create: bool = False) -> Setup:
    """Configs + the persisted period split (no prices are read). Only `create` (the first
    `screen`) may split the stored trading dates and write the period file."""
    _check_market(market)
    _families(pocket)
    _refuse_limited(market)
    mc, pc, lab, synced = load_market(market), load_pocket(pocket), load_lab(), load_synced()
    rec = _load_periods(pocket, market)
    if rec is None:
        if not create:
            raise LabError(f"no periods for {pocket}/{market}: run `lab screen` first")
        cal = trading_dates(market, lab.swing_start, date.today())
        if len(cal) == 0:
            raise LabError(f"no daily data for {market}: run `lab collect --market {market}` "
                           "first")
        try:
            rec = period_store.new_record(pocket, market, cal, lab.split)
        except ValueError as e:
            raise LabError(f"not enough daily data for {market} to split periods: {e}") from e
        period_store.save_record(periods_path(pocket, market), rec)
    cal = trading_dates(market, lab.swing_start, rec.data_cutoff)
    if len(cal) == 0:
        raise LabError(f"no daily data for {market}: run `lab collect --market {market}` first")
    return Setup(mc, pc, lab, synced, rec.periods, cal, rec)


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
    p = load_panel(s.market.market, None, s.lab.swing_start, end)
    if not p.close.index.equals(s.calendar[s.calendar <= pd.Timestamp(end)]):
        raise LabError("stored data changed while running; re-run the command")
    return p


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
def _pick_commission(rows: list[dict], country: str, today: date) -> float | None:
    for r in rows:
        if str(r.get("marketCountry")).upper() != country:
            continue
        start, end = r.get("startDate"), r.get("endDate")
        if start and date.fromisoformat(start) > today:
            continue
        if end and date.fromisoformat(end) < today:
            continue
        return float(r["commissionRate"])
    return None


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
    rates, source = {}, {}
    for m, country in (("kr", "KR"), ("us", "US")):
        if manual[m] is not None:
            rates[m], source[m] = float(manual[m]), "manual"
            continue
        v = _pick_commission(api_rows, country, today) if api_rows is not None else None
        if v is None:
            raise LabError(f"no {m} commission: set TOSS_ACCOUNT_SEQ for the commissions "
                           f"API or pass --{m}-commission (decimal ratio)")
        rates[m], source[m] = v, "api"
    out = {"commission_rate": rates, "usd_krw": usd_krw,
           "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": {"commission_rate": source, "usd_krw": "api"}}
    path = _synced_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"path": str(path), **out}


# ---------------------------------------------------------------- collect
def collect(market: str, since: date | None = None, limit: int | None = None) -> dict:
    """Meta -> incremental daily bars (Toss) -> (KR) pykrx delisted supplement -> quality
    summary. `limit` keeps the first N symbols, always_include first, and marks the data
    as partial (screen / validate / verify --final refuse it) until a full collect."""
    _check_market(market)
    mc = load_market_raw(market)
    start = since or load_lab().swing_start
    client = _client()
    try:
        meta = collect_meta(client, market)
        syms = [str(s) for s in meta.index]
        first = [s for s in mc.always_include if s in set(syms)]
        ordered = first + sorted(s for s in syms if s not in set(first))
        if limit is not None:
            ordered = ordered[:limit]
        delisted = set(str(s) for s in meta.index[meta["delisted"].astype(bool)])
        via_pykrx = [s for s in ordered if s in delisted] if mc.delisted_supplement else []
        via_toss = [s for s in ordered if s not in set(via_pykrx)]
        added, failed = 0, []
        for sym in via_toss:            # one failing symbol is reported, the run continues
            try:
                added += collect_daily(client, market, sym, start)
            except Exception as e:      # noqa: BLE001 - messages carry no secrets
                log.warning("collect %s %s failed: %s: %s", market, sym, type(e).__name__, e)
                failed.append(f"{sym}: {type(e).__name__}: {e}")
    except TossApiError as e:
        raise LabError(f"collect failed: {e}") from e
    except httpx.HTTPError as e:
        raise LabError(f"collect failed: network error ({type(e).__name__})") from e
    supplement = None
    if via_pykrx:
        supplement = supplement_delisted_kr(start, symbols=via_pykrx)
    marker = limited_marker(market)
    if limit is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"lab collect --limit {limit}\n", encoding="utf-8")
    else:
        marker.unlink(missing_ok=True)
    summary = {"symbols": len(ordered), "rows_added": added, "clean": 0, "issues": 0,
               "warnings": 0, "missing": 0, "failed": failed}
    for sym in ordered:
        df = load_daily(market, sym)
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
    s = _setup(pocket, market, create=True)
    fams = _families(pocket)
    names = list(fams) if family == "all" else [family]
    for n in names:
        if n not in fams:
            raise LabError(f"unknown family {n!r} (choose from all, {', '.join(fams)})")
    t0, t1 = s.periods.train
    _refuse_opened_overlap(pocket, market, t0, t1, "screen")
    panel = _panel(s, t1)
    elig = eligible_mask(panel, s.market, s.pocket, s.lab, "paper", s.synced.usd_krw)
    cm = _cost_model(s.market, panel.meta)
    log = ExperimentLog(experiments_path())
    frames = []
    for n in names:
        df = screen_family(n, panel, elig, t0, t1, cm, log, pocket, market,
                           sealed_start=s.periods.sealed[0])
        frames.append(df.assign(family=n))
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("mean_net_r", ascending=False, na_position="last",
                           kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- validate
@dataclass
class ValidateResult:
    table: pd.DataFrame
    pbo: float
    recommended: list[str]


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
    """Top-`top` screened rules (mean_net_r, >= MIN_TRAIN_TRADES train trades): walk-forward
    test-segment net sums over train+valid, paper portfolio on the valid period, PBO over the
    candidates' valid daily returns; recommend up to max_final_candidates with valid net_pnl
    > 0 by valid daily Sharpe. Each candidate is logged as stage "validate"."""
    if top < 1:
        raise LabError("--top must be >= 1")
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
        "mean_net_r", ascending=False, na_position="last", kind="stable").head(top)
    if cands.empty:
        raise LabError(f"no screened rule has >= {MIN_TRAIN_TRADES} train trades")

    (t0, _), (v0, v1) = s.periods.train, s.periods.valid
    _refuse_opened_overlap(pocket, market, t0, v1, "validate")
    panel = _panel(s, v1)
    cal = panel.close.index
    elig = eligible_mask(panel, s.market, s.pocket, s.lab, "paper", s.synced.usd_krw)
    cm = _cost_model(s.market, panel.meta)
    cpk = _currency_per_krw(s.market, s.synced)
    capital = s.lab.capital["paper"]
    windows = walk_forward_windows(t0, v1, n=WALK_FORWARD_WINDOWS)

    metrics, returns = {}, {}
    for fam, params, rid, scr_r in zip(cands["family"], cands["params"], cands["rule_id"],
                                       cands["mean_net_r"]):
        rule = _rule(fams, fam, params, rid)
        sig = rule.signals(panel)                 # history through the valid end
        wf = []
        for _, _, a, b in windows:                # params fixed: test segments only
            trades = simulate(panel, sig, elig, a, b)
            wf.append(float(cm.net_returns(trades)["net_return"].sum()))
        res = run_portfolio(simulate(panel, sig, elig, v0, v1), panel, cm, s.pocket, capital,
                            cpk, False, entry_at=sig.entry_at, exit_at=sig.exit_at)
        eq = period_equity(res, cal, v0, v1, capital * cpk)
        rets = eq.pct_change().dropna()
        returns[rid] = rets
        metrics[rid] = {
            "family": fam, "params": params, "screen_mean_net_r": scr_r,
            "wf_windows": [[a.isoformat(), b.isoformat()] for _, _, a, b in windows],
            "wf_test_net_sums": wf, "wf_positive_windows": int(sum(x > 0 for x in wf)),
            "valid_n_trades": int(len(res.fills)),
            "valid_net_pnl": float(res.fills["net_pnl"].astype(float).sum()),
            "valid_daily_sharpe": sharpe(rets.to_numpy(), periods_per_year=1),
            "valid_sharpe_annual": sharpe(rets.to_numpy()),
            "valid_mdd": max_drawdown(eq), "scenario": "paper", "capital": capital,
        }
    pbo = _pbo(pd.DataFrame(returns).fillna(0.0))
    ranked = sorted((rid for rid, m in metrics.items() if m["valid_net_pnl"] > 0),
                    key=lambda r: (-metrics[r]["valid_daily_sharpe"]
                                   if np.isfinite(metrics[r]["valid_daily_sharpe"])
                                   else math.inf))
    recommended = ranked[: s.lab.criteria.max_final_candidates]
    for rid, m in metrics.items():
        m.update(pbo=pbo, pbo_candidates=len(metrics), recommended=rid in recommended)
        log.record(pocket, market, m["family"], rid, m["params"], "validate", _iso((v0, v1)),
                   {k: v for k, v in m.items() if k not in ("family", "params")})
    table = pd.DataFrame([{"rule_id": rid, **{k: m[k] for k in (
        "valid_net_pnl", "valid_daily_sharpe", "valid_n_trades", "wf_positive_windows",
        "recommended")}} for rid, m in metrics.items()])
    table = table.sort_values("valid_daily_sharpe", ascending=False, na_position="last",
                              kind="stable").reset_index(drop=True)
    return ValidateResult(table, pbo, recommended)


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


def _sr_var_trials(log_rows: pd.DataFrame) -> float | None:
    """Variance (ddof=1) of the validate-stage daily Sharpes (latest per rule); None if < 2."""
    val = log_rows[log_rows["stage"] == "validate"].drop_duplicates("rule_id", keep="last")
    srs = np.array([float(m.get("valid_daily_sharpe", math.nan)) for m in val["metrics"]])
    srs = srs[np.isfinite(srs)]
    return float(np.var(srs, ddof=1)) if len(srs) >= 2 else None


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
    sig = rule.signals(panel)                                # history through sealed end
    vpanel = panel.slice(first, v1)
    vsig = rule.signals(vpanel)                              # history through valid end
    scenarios, parity = {}, {}
    for scen in SCENARIOS:
        capital = s.lab.capital[scen]
        cap0 = capital * cpk
        frac = s.market.fractional_orders if scen == "real" else False
        elig = eligible_mask(panel, s.market, s.pocket, s.lab, scen, s.synced.usd_krw)
        naut = run_nautilus(rule, panel, elig, s0, s1, cm, s.pocket, capital, cpk, frac)
        ours = run_portfolio(simulate(panel, sig, elig, s0, s1), panel, cm, s.pocket, capital,
                             cpk, frac, entry_at=sig.entry_at, exit_at=sig.exit_at)
        parity[scen] = _parity_summary(parity_report(ours.fills, naut))
        sealed_res = nautilus_result(naut, panel, cm, cap0, s0, s1)
        velig = elig.loc[elig.index <= pd.Timestamp(v1)]
        valid_res = run_portfolio(simulate(vpanel, vsig, velig, v0, v1), vpanel, cm, s.pocket,
                                  capital, cpk, frac, entry_at=vsig.entry_at,
                                  exit_at=vsig.exit_at)
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
            "capital": capital, "currency": s.market.currency, "currency_per_krw": cpk,
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
    period in NautilusTrader for both capital scenarios and write the rule card."""
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
    family, params = scr.iloc[-1]["family"], scr.iloc[-1]["params"]
    rule = _rule(fams, family, params, rule_id)
    n_trials = log.n_trials(pocket, market)
    pbo = float(vrow["metrics"].get("pbo", math.nan))
    sr_var = _sr_var_trials(rows)

    registry = SealRegistry(seal_path(), s.lab.criteria.max_final_candidates)
    cycle = s.record.cycle
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
    verdict = {VERDICT_PAPER: scenarios["paper"]["passed"],
               VERDICT_REAL: scenarios["paper"]["passed"] and scenarios["real"]["passed"]}
    final_metrics = {scen: {"passed": d["passed"], "failed": d["failed"], **d["metrics"],
                            "sr_var_trials_source": d["sr_var_trials_source"]}
                     for scen, d in scenarios.items()}
    final_metrics["verdict"] = verdict
    final_metrics["parity_matched_share"] = {k: p["matched_share"] for k, p in parity.items()}
    log.record(pocket, market, family, rule_id, params, "final", _iso(s.periods.sealed),
               final_metrics)
    card = RuleCard(
        pocket=pocket, market=market, rule_id=rule_id, family=family, params=params,
        confidence_formula=CONFIDENCE_FORMULAS.get(family, "(미기재)"),
        periods={k: list(_iso(getattr(s.periods, k))) for k in ("train", "valid", "sealed")},
        metrics_by_stage={"screen": scr.iloc[-1]["metrics"], "validate": vrow["metrics"]},
        scenarios=scenarios, verdict=verdict, parity=parity,
        limitations=limitations(s.market, family=family))
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
        if not any(t.cycle == rec.cycle for t in tickets):
            raise LabError(f"no final candidate opened the seal in cycle {rec.cycle}: "
                           "nothing to re-validate")
        dates = trading_dates(market, load_lab().swing_start, date.today())
        try:
            rec = period_store.new_cycle(rec, dates)
        except PeriodError as e:
            raise LabError(str(e)) from e
        period_store.save_record(periods_path(pocket, market), rec)
    return {"path": str(periods_path(pocket, market)), "record": rec, "tickets": tickets}


# ---------------------------------------------------------------- report / journal
def report(pocket: str, market: str) -> Path:
    _check_market(market)
    return write_leaderboard(ExperimentLog(experiments_path()), pocket, market,
                             leaderboard_path(pocket, market),
                             limitations=limitations(load_market_raw(market)))


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


__all__ = ["ConfigError", "LabError", "collect", "journal", "nautilus_result", "periods",
           "report", "screen", "sync", "universe", "validate", "verify_final"]
