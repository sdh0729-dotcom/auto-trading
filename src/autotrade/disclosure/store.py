"""Event store: raw filings per market under data/<market>/events/ (parquet + small JSON files)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from autotrade.data.store import data_root, write_parquet_atomic

FILING_COLUMNS = ["filing_id", "symbol", "corp_code", "filed_date", "title", "kind", "filer",
                  "remark"]


def events_root(market: str) -> Path:
    return data_root() / market / "events"


def filings_path(market: str) -> Path:
    return events_root(market) / "filings.parquet"


def targets_path(market: str) -> Path:
    return events_root(market) / "targets.json"


def state_path(market: str) -> Path:
    return events_root(market) / "state.json"


def lock_path(market: str) -> Path:
    return events_root(market) / ".collect.lock"


def load_filings(market: str) -> pd.DataFrame:
    p = filings_path(market)
    if not p.exists():
        return pd.DataFrame(columns=FILING_COLUMNS)
    return pd.read_parquet(p)


def append_filings(market: str, df: pd.DataFrame) -> None:
    """Add rows (dedupe on filing_id, stored row wins); atomic temp + rename."""
    if df.empty:
        return
    merged = pd.concat([load_filings(market), df[FILING_COLUMNS]], ignore_index=True)
    merged = merged.drop_duplicates("filing_id", keep="first").reset_index(drop=True)
    write_parquet_atomic(merged, filings_path(market))


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
