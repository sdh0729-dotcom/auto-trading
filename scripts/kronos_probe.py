"""Small CPU-only Kronos directional-hit-rate probe. Not a full-universe run.

    uv run python scripts/kronos_probe.py --market kr --symbols 10 --days 250

Uses the local store; if it is empty falls back to a synthetic panel (pipeline check only,
journaled as "synthetic — not a measurement").
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from autotrade.rules.swing import kronos as kr  # noqa: E402

HORIZON = 5
CONTEXT = 120


def get_panel(market: str, n_symbols: int, days: int):
    from autotrade.data.store import load_panel
    end = date.today()
    start = end - timedelta(days=int((days + CONTEXT) * 1.6))
    try:
        panel = load_panel(market, None, start, end)
    except Exception:
        panel = None
    if panel is not None and panel.close.shape[1] > 0 and len(panel.close) > days:
        cols = list(panel.close.mean().sort_values(ascending=False).index[:n_symbols])
        return panel, cols, False
    from tests.fixtures import make_panel
    syms = [f"SYN{i}" for i in range(n_symbols)]
    return make_panel(days + CONTEXT + HORIZON, syms, seed=0), syms, True


def run(panel, cols, days, step, n_paths):
    hits = []
    n = len(panel.close)
    for sym in cols:
        ohlcv = panel.close[[sym]].rename(columns={sym: "close"})
        for c in ("open", "high", "low", "volume"):
            ohlcv[c] = getattr(panel, c)[sym]
        ohlcv = ohlcv[["open", "high", "low", "close", "volume"]]
        for i in range(max(CONTEXT, n - days - HORIZON), n - HORIZON, step):
            hist = ohlcv.iloc[i - CONTEXT + 1:i + 1].copy()
            if hist.isna().any().any():
                continue
            hist.attrs["horizon"] = HORIZON
            paths = np.asarray(kr._predict_paths(hist, n_paths))
            pred = paths[:, HORIZON - 1].mean() / hist["close"].iloc[-1] - 1
            actual = ohlcv["close"].iloc[i + HORIZON] / hist["close"].iloc[-1] - 1
            hits.append(np.sign(pred) == np.sign(actual))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="kr")
    ap.add_argument("--symbols", type=int, default=10)
    ap.add_argument("--days", type=int, default=250)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--paths", type=int, default=4)
    a = ap.parse_args()
    panel, cols, synthetic = get_panel(a.market, a.symbols, a.days)
    t0 = time.time()
    hits = run(panel, cols, a.days, a.step, a.paths)
    wall = time.time() - t0
    k, n = int(sum(hits)), len(hits)
    p = binomtest(k, n, 0.5).pvalue if n else float("nan")
    label = "synthetic — not a measurement" if synthetic else "real data"
    line = (f"Kronos probe ({label}): market={a.market} symbols={len(cols)} days={a.days} "
            f"step={a.step} paths={a.paths} n={n} hit_rate={k / n if n else float('nan'):.3f} "
            f"vs 0.5 binomial_p={p:.4f} wall={wall:.1f}s")
    print(line)
    j = ROOT / "research" / "journal" / f"{date.today().isoformat()}.md"
    with j.open("a") as f:
        f.write(f"\n## Kronos probe\n- {line}\n")


if __name__ == "__main__":
    main()
