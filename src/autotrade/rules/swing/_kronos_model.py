"""Kronos CPU sampling backend (optional extra `kronos`; Kronos source vendored in vendor/kronos)."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parents[4] / "vendor" / "kronos"


@lru_cache(maxsize=2)
def _predictor(model_name: str):
    if VENDOR.exists() and str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    from model import Kronos, KronosPredictor, KronosTokenizer  # vendored Kronos repo

    tok_name = ("NeoQuasar/Kronos-Tokenizer-2k" if "mini" in model_name
                else "NeoQuasar/Kronos-Tokenizer-base")
    tok = KronosTokenizer.from_pretrained(tok_name)
    model = Kronos.from_pretrained(model_name)
    return KronosPredictor(model, tok, device="cpu", max_context=512)


def _seed(seed: int) -> None:
    """Seed every RNG the sampler may use (Kronos samples with torch)."""
    np.random.seed(seed)
    try:
        import torch
    except ImportError:          # the optional extra is missing: nothing else to seed
        return
    torch.manual_seed(seed)


def sample_paths(history: pd.DataFrame, n_paths: int, horizon: int, model_name: str,
                 seed: int = 0) -> np.ndarray:
    """`n_paths` sampled close paths; the same history and seed give the same paths."""
    pred = _predictor(model_name)
    _seed(seed)
    x = history[["open", "high", "low", "close", "volume"]].copy()
    x["amount"] = x["close"] * x["volume"]
    x_ts = pd.Series(history.index)
    last = history.index[-1]
    y_ts = pd.Series(pd.bdate_range(last, periods=horizon + 1)[1:])
    out = []
    for _ in range(n_paths):   # each call samples one stochastic path
        df = pred.predict(df=x.reset_index(drop=True), x_timestamp=x_ts, y_timestamp=y_ts,
                          pred_len=horizon, T=1.0, top_p=0.9, sample_count=1, verbose=False)
        out.append(df["close"].to_numpy())
    return np.vstack(out)
