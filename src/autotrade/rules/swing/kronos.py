"""Kronos daily-candle foundation model as an optional swing family.

The model call is isolated in `_predict_paths` and imports torch/Kronos lazily, so this module
(and its tests) never need the optional `kronos` extra. Not registered in SWING_FAMILIES until a
real-data probe (scripts/kronos_probe.py) shows a hit rate distinguishable from 50%.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

from autotrade.rules.base import BaseRule, atr_stop, finalize
from autotrade.types import Panel, Signals

HORIZON = [5, 10]
THRESHOLD = [0.01, 0.02, 0.03]
K = [2.0, 3.0]
N_PATHS = 8
MIN_HISTORY = 30   # ATR14 warm-up + a little context
MAX_CONTEXT = 400  # bars passed to the model (kept cheap on CPU)
SEED = 0           # fixed sampling seed: reruns give the same signals


def _predict_paths(history: pd.DataFrame, n_paths: int, seed: int = SEED) -> np.ndarray:
    """Sample `n_paths` future close paths (n_paths x horizon) from Kronos on CPU, seeded.

    `history.attrs['horizon']` is the path length; `history` has columns open/high/low/close/volume on a DatetimeIndex. Requires the `kronos`
    extra plus Kronos source on the path (vendor/kronos, see scripts/kronos_probe.py).
    """
    from autotrade.rules.swing._kronos_model import sample_paths  # lazy: pulls in torch

    return sample_paths(history, n_paths, int(history.attrs["horizon"]),
                        model_name=os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-mini"),
                        seed=seed)


class KronosDaily(BaseRule):
    family = "kronos"
    PARAM_KEYS = ("horizon", "threshold", "k")

    @classmethod
    def grid(cls):
        return [{"horizon": h, "threshold": t, "k": k}
                for h, t, k in itertools.product(HORIZON, THRESHOLD, K)]

    def signals(self, panel: Panel) -> Signals:
        horizon, threshold, k = (self.params[x] for x in self.PARAM_KEYS)
        entries = pd.DataFrame(False, index=panel.close.index, columns=panel.close.columns)
        conf = pd.DataFrame(np.nan, index=panel.close.index, columns=panel.close.columns)
        daily_vol = np.log(panel.close).diff().rolling(20).std()
        for sym in panel.close.columns:
            ohlcv = pd.DataFrame({"open": panel.open[sym], "high": panel.high[sym],
                                  "low": panel.low[sym], "close": panel.close[sym],
                                  "volume": panel.volume[sym]})
            for i in range(MIN_HISTORY, len(ohlcv)):
                hist = ohlcv.iloc[max(0, i + 1 - MAX_CONTEXT):i + 1]   # rows <= t only
                if hist["close"].isna().any():
                    continue
                hist.attrs["symbol"] = sym
                hist.attrs["horizon"] = horizon
                paths = np.asarray(_predict_paths(hist, N_PATHS))
                rets = paths[:, horizon - 1] / hist["close"].iloc[-1] - 1
                dv = daily_vol[sym].iloc[i]
                scale = dv * np.sqrt(horizon) if dv and dv > 0 else np.nan
                ts = panel.close.index[i]
                entries.loc[ts, sym] = bool(rets.mean() > threshold)
                if np.isfinite(scale):
                    conf.loc[ts, sym] = 1.0 / (1.0 + rets.std() / scale)
        return finalize(panel, entries, pd.DataFrame(False, index=entries.index,
                                                     columns=entries.columns),
                        atr_stop(panel, k), conf, "next_open", "next_open", horizon)
