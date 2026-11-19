"""Family-level screening on the learning period: every grid point is simulated, costed,
summarized and logged as one trial."""
from __future__ import annotations

import math
from datetime import date

import pandas as pd

from autotrade.costs.model import CostModel
from autotrade.rules.registry import SWING_FAMILIES
from autotrade.screen.simulate import simulate
from autotrade.types import Panel
from autotrade.validate.experiments import ExperimentLog
from autotrade.validate.seal import SealViolation

SCREEN_COLUMNS = ["rule_id", "params", "n_trades", "mean_net_r", "t_stat", "profit_factor",
                  "win_rate", "sum_net_return"]


def trade_metrics(net: pd.DataFrame) -> dict:
    """Summary of CostModel.net_returns output (net_return, net_r per trade)."""
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
            "sum_net_return": float(ret.sum())}


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
        trades = simulate(visible, rule.signals(visible), elig, start, end)
        metrics = trade_metrics(cost_model.net_returns(trades))
        log.record(pocket, market, family, rule.rule_id, params, "screen", period, metrics)
        out.append({"rule_id": rule.rule_id, "params": params, **metrics})
    return pd.DataFrame(out, columns=SCREEN_COLUMNS)
