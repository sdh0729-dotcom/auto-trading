"""Append-only experiment log (jsonl). Every screened parameter set is one trial for DSR.
Besides the research stages it holds one "cycle_started" row per `lab periods --new-cycle`
(the new cycle's periods), so a lost periods file can be rebuilt in that cycle."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Stage = Literal["screen", "precheck", "validate", "final", "cycle_started"]
STAGES = ("screen", "precheck", "validate", "final", "cycle_started")
_FIELDS = ["ts", "pocket", "market", "family", "rule_id", "params", "stage", "period", "metrics"]


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


class ExperimentLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def record(self, pocket: str, market: str, family: str, rule_id: str, params: dict,
               stage: Stage, period: tuple[str, str], metrics: dict) -> None:
        """Append one line. Non-finite floats are written as NaN/Infinity (Python json)."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
        row = {"ts": datetime.now(timezone.utc).isoformat(), "pocket": pocket, "market": market,
               "family": family, "rule_id": rule_id, "params": params, "stage": stage,
               "period": [str(period[0]), str(period[1])], "metrics": metrics}
        line = json.dumps(row, default=_default, ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def rows(self, pocket: str, market: str) -> pd.DataFrame:
        rows = [r for r in self._all() if r["pocket"] == pocket and r["market"] == market]
        return pd.DataFrame(rows, columns=_FIELDS)

    def n_trials(self, pocket: str, market: str) -> int:
        """Distinct rule_ids tried for this pocket x market (DSR trial count): every grid
        point generated counts, including those the cost precheck skipped."""
        rows = self.rows(pocket, market)
        return int(rows.loc[rows["stage"].isin(("screen", "precheck")), "rule_id"].nunique())
