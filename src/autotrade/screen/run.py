"""Family-level screening on the learning period: every grid point is simulated, costed,
summarized and logged as one trial."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from autotrade.costs.model import CostModel
from autotrade.data.minute_store import load_minutes
from autotrade.rules.registry import DAYTRADE_FAMILIES, SWING_FAMILIES
from autotrade.screen.intraday import SessionRules, execute
from autotrade.screen.simulate import simulate
from autotrade.types import TRADE_COLUMNS, Panel
from autotrade.validate.experiments import ExperimentLog
from autotrade.validate.seal import SealViolation

# `lab validate` takes only screened rules with at least this many train-period trades.
MIN_TRAIN_TRADES = 30

SCREEN_COLUMNS = ["rule_id", "params", "n_trades", "mean_net_r", "t_stat", "profit_factor",
                  "win_rate", "sum_net_return", "n_data_end"]


def trade_metrics(net: pd.DataFrame) -> dict:
    """Summary of CostModel.net_returns output (net_return, net_r per trade). `n_data_end`
    counts trades closed because the symbol's prices ended inside the window."""
    n = len(net)
    r = net["net_r"].astype(float)
    ret = net["net_return"].astype(float)
    mean_r = float(r.mean()) if n else math.nan
    std = float(r.std(ddof=1)) if n >= 2 else math.nan
    t_stat = mean_r / (std / math.sqrt(n)) if n >= 2 and std > 0 else math.nan
    gain, loss = float(ret[ret > 0].sum()), float(-ret[ret < 0].sum())
    if loss > 0:
        pf = gain / loss
    else:
        pf = math.inf if gain > 0 else math.nan
    return {"n_trades": n, "mean_net_r": mean_r, "t_stat": t_stat, "profit_factor": pf,
            "win_rate": float((ret > 0).mean()) if n else math.nan,
            "sum_net_return": float(ret.sum()),
            "n_data_end": int((net["exit_reason"] == "data_end").sum()) if n else 0}


def screen_family(family: str, panel: Panel, eligible: pd.DataFrame, start: date, end: date,
                  cost_model: CostModel, log: ExperimentLog, pocket: str, market: str,
                  sealed_start: date) -> pd.DataFrame:
    """One row per grid params. Refuses any period reaching the seal; bars after `end` are
    dropped before signals are computed, so sealed data cannot leak in."""
    if pd.Timestamp(end) >= pd.Timestamp(sealed_start):
        raise SealViolation(f"screen end {end} reaches sealed period starting {sealed_start}")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"start {start} is after end {end}")
    cls = SWING_FAMILIES[family]
    first = panel.close.index.min() if len(panel.close.index) else pd.Timestamp(start)
    visible = panel.slice(first, end)
    elig = eligible.loc[eligible.index <= pd.Timestamp(end)]
    period = (pd.Timestamp(start).date().isoformat(), pd.Timestamp(end).date().isoformat())
    out = []
    for params in cls.grid():
        rule = cls(params)
        trades = simulate(visible, rule.signals(visible, eligible=elig), elig, start, end)
        metrics = trade_metrics(cost_model.net_returns(trades))
        log.record(pocket, market, family, rule.rule_id, params, "screen", period, metrics)
        out.append({"rule_id": rule.rule_id, "params": params, **metrics})
    return pd.DataFrame(out, columns=SCREEN_COLUMNS)


def _stack_features(per_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [f.assign(symbol=s).set_index("symbol", append=True)
              for s, f in per_symbol.items() if len(f)]
    if not frames:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["day", "symbol"]))
    out = pd.concat(frames)
    out.index = out.index.set_names(["day", "symbol"])
    return out


def intraday_trades(grid: list, market: str, symbols: list[str], start: date, end: date,
                    daily: Panel, eligible: pd.DataFrame, cost_model: CostModel,
                    rules: SessionRules, features_start: date | None = None
                    ) -> tuple[dict, dict]:
    """Executed same-day trades of every rule in `grid` (one family) over [start, end]:
    -> ({rule_id: [trade frames]}, {rule_id: ([round-trip cost pct per candidate],
    [stop_pct per candidate])}). Minute bars are loaded per symbol only through `end`; the
    day features read them from `features_start` (default `start`; an earlier date is the
    warm-up of rolling features) while selection and candidates stay within [start, end]; the
    shared core of the screen (train) and validate (valid) trade generation."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    elig = eligible.loc[(eligible.index >= t0) & (eligible.index <= t1)]
    feats = pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["day", "symbol"]))
    if type(grid[0]).needs_features():
        fstart = start if features_start is None else min(features_start, start)
        per = {s: grid[0].day_features(s, load_minutes(market, s, fstart, end))
               for s in symbols}
        feats = _stack_features(per)
    picked = {}
    for rule in grid:
        sel = rule.select(feats, daily, elig)
        picked[rule.rule_id] = pd.DataFrame(index=sel).reset_index().groupby("symbol")["day"] \
            .apply(lambda d: pd.DatetimeIndex(sorted(set(d)))).to_dict() if len(sel) else {}
    product = daily.meta["product"].to_dict() if "product" in daily.meta.columns else {}
    trades = {r.rule_id: [] for r in grid}
    cost = {r.rule_id: ([], []) for r in grid}        # (cost pct per candidate, stop_pct)
    for sym in symbols:
        if not any(sym in p for p in picked.values()):
            continue
        bars = load_minutes(market, sym, start, end)
        if bars.empty:
            continue
        have = pd.DatetimeIndex(bars.index.tz_localize(None)
                                .normalize().unique())
        prev = (daily.close[sym].shift(1).dropna() if sym in daily.close.columns else None)
        for rule in grid:
            days = picked[rule.rule_id].get(sym)
            if days is None:
                continue
            days = days[days.isin(have) & (days >= t0) & (days <= t1)]
            if not len(days):
                continue
            cands = rule.candidates(sym, bars, days, daily)
            if not len(cands):
                continue
            px = bars["close"].reindex(cands["signal_ts"]).to_numpy(float)
            cost[rule.rule_id][0].extend(cost_model.round_trip_cost_pct(sym, p)
                                         for p in px if np.isfinite(p) and p > 0)
            cost[rule.rule_id][1].extend(cands["stop_pct"].astype(float))
            tr = execute(sym, bars, cands, rules, prev_close=prev,
                         product=product.get(sym, "stock"))
            if len(tr):
                trades[rule.rule_id].append(tr)
    return trades, cost


def screen_intraday(family: str, market: str, symbols: list[str], start: date, end: date,
                    daily: Panel, eligible: pd.DataFrame, cost_model: CostModel,
                    rules: SessionRules, log: ExperimentLog, pocket: str,
                    sealed_start: date, cost_r_max: float) -> pd.DataFrame:
    """One row per grid params (as `screen_family`). Minute bars are loaded per symbol only
    through `end`. Pass 1 builds the family's day features, then `select` picks the
    symbol-days per grid point and every symbol's bars are loaded once more to build and
    execute the candidates of all grid points. A grid point whose median round-trip cost
    exceeds `cost_r_max` (lab.criteria) of its median stop is logged as stage "precheck" and skipped."""
    if pd.Timestamp(end) >= pd.Timestamp(sealed_start):
        raise SealViolation(f"screen end {end} reaches sealed period starting {sealed_start}")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"start {start} is after end {end}")
    cls = DAYTRADE_FAMILIES[family]
    grid = [cls(p) for p in cls.grid()]
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    period = (t0.date().isoformat(), t1.date().isoformat())
    trades, cost = intraday_trades(grid, market, symbols, start, end, daily, eligible,
                                   cost_model, rules)
    out = []
    for rule in grid:
        pcts, stops = cost[rule.rule_id]
        cost_r = float(np.median(pcts) / np.median(stops)) if pcts and stops else math.nan
        if cost_r > cost_r_max:
            log.record(pocket, market, family, rule.rule_id, rule.params, "precheck", period,
                       {"cost_r": cost_r})
            continue
        parts = trades[rule.rule_id]
        tr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=TRADE_COLUMNS)
        metrics = trade_metrics(cost_model.net_returns(tr))
        log.record(pocket, market, family, rule.rule_id, rule.params, "screen", period, metrics)
        out.append({"rule_id": rule.rule_id, "params": rule.params, **metrics})
    return pd.DataFrame(out, columns=SCREEN_COLUMNS)
