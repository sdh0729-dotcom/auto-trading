from __future__ import annotations

from autotrade.rules.base import Rule
from autotrade.rules.swing.breakout import Breakout
from autotrade.rules.swing.ma_trend import MaTrend
from autotrade.rules.swing.momentum import Momentum
from autotrade.rules.swing.overnight import Overnight
from autotrade.rules.swing.oversold import Oversold
from autotrade.rules.swing.reversal import Reversal
from autotrade.rules.swing.tom import Tom

SWING_FAMILIES: dict[str, type[Rule]] = {
    "breakout": Breakout, "ma_trend": MaTrend, "oversold": Oversold, "momentum": Momentum,
    "reversal": Reversal, "overnight": Overnight, "tom": Tom,
}
