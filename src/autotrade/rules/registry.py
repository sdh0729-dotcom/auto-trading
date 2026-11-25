from __future__ import annotations

from autotrade.rules.base import Rule
from autotrade.rules.daytrade.base import IntradayRule
from autotrade.rules.daytrade.gap import GapFade, GapGo, OpenReversal
from autotrade.rules.daytrade.momentum import CloseMomentum, IntradayMomentum
from autotrade.rules.daytrade.orb import Orb, OrbStocksInPlay
from autotrade.rules.swing.breakout import Breakout
from autotrade.rules.swing.disclosure import Disclosure
from autotrade.rules.swing.etf_rotation import EtfRotation
from autotrade.rules.swing.ma_trend import MaTrend
from autotrade.rules.swing.momentum import Momentum
from autotrade.rules.swing.negf import with_negative_filter
from autotrade.rules.swing.overnight import Overnight
from autotrade.rules.swing.oversold import Oversold
from autotrade.rules.swing.reversal import Reversal
from autotrade.rules.swing.tom import Tom

SWING_FAMILIES: dict[str, type[Rule]] = {
    "breakout": Breakout, "ma_trend": MaTrend, "oversold": Oversold, "momentum": Momentum,
    "reversal": Reversal, "overnight": Overnight, "tom": Tom, "etf_rotation": EtfRotation,
    "disclosure": Disclosure,
}
for _base in (Breakout, MaTrend, Momentum, Reversal):
    _neg = with_negative_filter(_base)
    SWING_FAMILIES[_neg.family] = _neg

# swing families that read Panel.events (company filings): screened by name or with
# `lab screen --family events` (never by `all`), and guarded by the event-completeness gate
# and the event-table hash in screen / validate / verify --final / forward
EVENT_FAMILIES: set[str] = {"disclosure", "breakout_negf", "ma_trend_negf", "momentum_negf",
                            "reversal_negf"}

# intraday families (Tasks 6-8)
DAYTRADE_FAMILIES: dict[str, type[IntradayRule]] = {"orb_sip": OrbStocksInPlay, "orb": Orb,
                                                   "intraday_mom": IntradayMomentum,
                                                   "close_mom": CloseMomentum, "gap_go": GapGo,
                                                   "gap_fade": GapFade, "kr_open_rev": OpenReversal}
