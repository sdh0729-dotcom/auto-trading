"""Daily-bar trade simulator: fixed notional, at most one open position per symbol.

Fill rules (brief, Task 9):
- Entry decision on signal row t needs entries[t], eligible[t], a valid stop_pct[t] and no
  open position. `entry_at="next_open"` fills at open[t+1]; `"close"` fills at close[t].
  Either way the fill row must lie inside [start, end] and be later than the symbol's
  previous exit row (no same-row exit-and-reenter). A `close` entry on the window's last
  row is not taken (it could only be closed at the same price).
- stop = entry_price * (1 - stop_pct[t]); stop_pct / confidence are the signal-row values.
- On each later bar, in physical order: (1) open <= stop -> `gap_stop` at open; then, if an
  exit_at="next_open" signal from the previous bar is pending, it fills at this open ->
  `signal` (no intrabar stop check on such a bar, the position is already gone);
  (2) low <= stop -> `stop` at the stop price; (3) exit_at="close" signal -> `signal` at the
  signal row's close; (4) held rows since entry (entry row = 0) >= max_hold_days -> `max_hold`
  at close.
- A `next_open` entry bar is checked from (2) on. A `close` entry bar is not price-checked;
  only an exit_at="next_open" signal on it is registered (fills at the next open).
- Still open on the last row of the window -> `period_end` at that close. A bar with a
  missing price while open -> `period_end` at the last valid close (and its date).
"""
from __future__ import annotations

from bisect import bisect_left
from datetime import date

import numpy as np
import pandas as pd

from autotrade.types import TRADE_COLUMNS, Panel, Signals


def simulate(panel: Panel, signals: Signals, eligible: pd.DataFrame,
             start: date, end: date) -> pd.DataFrame:
    """Trades (TRADE_COLUMNS) whose entry and exit rows both lie in [start, end]."""
    idx = panel.close.index
    in_window = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    dates = idx[in_window]
    if len(dates) == 0:
        return _frame([])
    cols = panel.close.columns

    def num(df: pd.DataFrame) -> np.ndarray:
        return df.reindex(index=idx, columns=cols).loc[in_window].to_numpy(dtype=float)

    def flag(df: pd.DataFrame) -> np.ndarray:  # missing / NaN -> False
        return df.reindex(index=idx, columns=cols).loc[in_window].eq(True).to_numpy()

    o, h, l, c = (num(f) for f in (panel.open, panel.high, panel.low, panel.close))
    stop = num(signals.stop_pct)
    conf = num(signals.confidence)
    with np.errstate(invalid="ignore"):
        candidates = flag(signals.entries) & flag(eligible) & np.isfinite(stop) & (stop > 0)
    exits = flag(signals.exits)
    valid = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)

    next_open_entry = signals.entry_at == "next_open"
    next_open_exit = signals.exit_at == "next_open"
    max_hold = signals.max_hold_days

    n = len(dates)
    trades = []
    for j, sym in enumerate(cols):
        cand_rows = np.flatnonzero(candidates[:, j]).tolist()
        if not cand_rows:
            continue
        col = (o[:, j].tolist(), l[:, j].tolist(), c[:, j].tolist(), valid[:, j].tolist(),
               exits[:, j].tolist())
        k = 0
        while k < len(cand_rows):
            t = cand_rows[k]
            e = t + 1 if next_open_entry else t
            if e >= n or (not next_open_entry and e == n - 1) or not col[3][e]:
                k += 1
                continue
            price = col[0][e] if next_open_entry else col[2][e]
            sp = float(stop[t, j])
            x_row, x_price, reason = _hold(col, e, price, price * (1 - sp), next_open_entry,
                                           next_open_exit, max_hold, n)
            trades.append((sym, dates[e], price, dates[x_row], x_price, reason, sp,
                           float(conf[t, j])))
            # next fill row must be > x_row: next_open needs t >= x_row, close needs t > x_row
            k = bisect_left(cand_rows, x_row if next_open_entry else x_row + 1, k + 1)
    return _frame(trades)


def _hold(col, e: int, price: float, stop: float, next_open_entry: bool,
          next_open_exit: bool, max_hold: int, n: int) -> tuple[int, float, str]:
    """Walk bars from entry row e; return (exit_row, exit_price, reason)."""
    o, l, c, valid, exits = col
    last_row, last_close = e, price
    pending = False  # exit_at="next_open" signal waiting for this bar's open
    if not next_open_entry:  # close entry: entry bar is not price-checked
        last_close = c[e]
        pending = next_open_exit and exits[e]
        r = e + 1
    else:
        r = e
    while r < n:
        if not valid[r]:
            return last_row, last_close, "period_end"
        if r > e and o[r] <= stop:                               # (1) not on entry bar
            return r, o[r], "gap_stop"
        # Physical order: a pending next_open exit sells at this bar's open (after the gap
        # check, same price) before any intrabar stop on this bar can happen.
        if pending:                                              # (3) next_open exit
            return r, o[r], "signal"
        if l[r] <= stop:                                         # (2)
            return r, stop, "stop"
        if exits[r] and not next_open_exit:                      # (3) close exit
            return r, c[r], "signal"
        if r - e >= max_hold:                                    # (4)
            return r, c[r], "max_hold"
        pending = next_open_exit and exits[r]
        last_row, last_close = r, c[r]
        r += 1
    return last_row, last_close, "period_end"


def _frame(trades: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    for k in ("entry_price", "exit_price", "stop_pct", "confidence"):
        df[k] = df[k].astype(float)
    df["exit_reason"] = df["exit_reason"].astype(object)
    df["symbol"] = df["symbol"].astype(object)
    return df.sort_values(["entry_date", "symbol"], kind="stable").reset_index(drop=True)
