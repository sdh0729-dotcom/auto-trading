import numpy as np
import pandas as pd
import pytest

from autotrade.rules.daytrade.orb import Orb, OrbStocksInPlay
from autotrade.rules.registry import DAYTRADE_FAMILIES
from autotrade.types import Panel
from tests.fixtures import make_panel

TZ = "America/New_York"


def _regular_days(n, start="2024-01-02"):
    """The first `n` days from `start` that are regular sessions on BOTH exchanges (KR
    09:00-15:30, US 09:30-16:00): synthetic bars use the usual clock hours, and holidays /
    late opens / half-days would drop or move them (data/sessions.py)."""
    from autotrade.data.sessions import sessions
    end = pd.Timestamp(start) + pd.Timedelta(days=3 * n + 30)
    kr, us = sessions("kr", start, end), sessions("us", start, end)

    def hours(t):
        return t.dt.strftime("%H:%M")
    kr = kr.index[(hours(kr["open"]) == "09:00") & (hours(kr["close"]) == "15:30")]
    us = us.index[(hours(us["open"]) == "09:30") & (hours(us["close"]) == "16:00")]
    return pd.DatetimeIndex(kr.intersection(us)[:n])


DAYS = _regular_days(30)


def day_bars(day, base=100.0, first_up=True, v5=1000.0, breakout_at="09:50", n=60):
    """US session minutes 09:31.. (END-stamped); OR(5) high = base+1, low = base-1."""
    ends = pd.date_range(f"{day:%Y-%m-%d} 09:31", periods=n, freq="min", tz=TZ)
    o = np.full(n, base)
    c = np.full(n, base)
    h = np.full(n, base + 0.5)
    lo = np.full(n, base - 0.5)
    v = np.full(n, v5 / 5)
    if first_up:
        c[:5] = base + 0.2
    else:
        c[:5] = base - 0.2
    h[2], lo[3] = base + 1.0, base - 1.0
    ts = pd.Timestamp(f"{day:%Y-%m-%d} {breakout_at}", tz=TZ)
    h[list(ends).index(ts)] = base + 3.0
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v},
                        index=pd.DatetimeIndex(ends, name="ts"))


def daily_panel(symbols, **kw):
    p = make_panel(120, symbols, seed=2, start="2023-11-01")
    for s in symbols:
        p.close[s] = 50.0
        p.open[s] = 50.0
        p.high[s] = 52.0
        p.low[s] = 48.0
        p.volume[s] = 2_000_000.0
    return p


def test_orb_entry_only_after_the_range_ends():
    d = DAYS[10]
    bars = day_bars(d, breakout_at="09:50")
    # a spike INSIDE the range (09:33 bar, high base+1) must never be the signal
    cands = Orb({"or_minutes": 15, "stop": "or_low", "target_r": np.nan}).candidates(
        "X", bars, pd.DatetimeIndex([d]), daily_panel(["X"]))
    assert len(cands) == 1
    assert cands.iloc[0]["signal_ts"] == pd.Timestamp(f"{d:%Y-%m-%d} 09:50", tz=TZ)
    assert cands.iloc[0]["signal_ts"] > pd.Timestamp(f"{d:%Y-%m-%d} 09:45", tz=TZ)
    # the 5 minute range ends 09:35; a breakout bar at 09:36 is accepted, at 09:35 is not
    c5 = Orb({"or_minutes": 5, "stop": "or_low", "target_r": 2.0}).candidates(
        "X", day_bars(d, breakout_at="09:36"), pd.DatetimeIndex([d]), daily_panel(["X"]))
    assert c5.iloc[0]["signal_ts"] == pd.Timestamp(f"{d:%Y-%m-%d} 09:36", tz=TZ)
    assert c5.iloc[0]["stop_pct"] == pytest.approx(2.0 / 101.0)
    assert c5.iloc[0]["target_pct"] == pytest.approx(2 * 2.0 / 101.0)


def _sip_env():
    syms = [f"S{i:02d}" for i in range(25)]
    panel = daily_panel(syms)
    feats = []
    for i, s in enumerate(syms):
        frames = []
        for d in DAYS:
            frames.append(day_bars(d, v5=1000.0 * (1 + (i * 0.1 if d == DAYS[-1] else 0))))
        f = OrbStocksInPlay({"stop_atr": 0.5, "or_minutes": 5}).day_features(
            s, pd.concat(frames))
        feats.append(f.assign(symbol=s).set_index("symbol", append=True))
    stacked = pd.concat(feats)
    stacked.index = stacked.index.set_names(["day", "symbol"])
    elig = pd.DataFrame(True, index=panel.close.index, columns=syms)
    return panel, stacked, elig, syms


def test_orb_sip_picks_top_relative_volume():
    panel, stacked, elig, syms = _sip_env()
    # panel index must contain the feature days
    panel = make_panel(120, syms, seed=2, start="2023-11-01")
    for s in syms:
        panel.close[s], panel.open[s] = 50.0, 50.0
        panel.high[s], panel.low[s], panel.volume[s] = 52.0, 48.0, 2_000_000.0
    elig = pd.DataFrame(True, index=panel.close.index, columns=syms)
    sel = OrbStocksInPlay({"stop_atr": 0.5, "or_minutes": 5}).select(stacked, panel, elig)
    last = [s for d, s in sel if d == DAYS[-1]]
    assert len(last) == 20
    assert set(last) == set(syms[5:])          # rel volume rises with i; S05..S24 are top 20
    per_day = pd.Series(1, index=sel).groupby(level="day").sum()
    assert per_day.max() == 20                 # never more than 20 per day
    assert DAYS[5] not in set(d for d, _ in sel)   # first 14 days have no 14-day mean yet


def test_orb_paper_stop_is_ten_percent_atr():
    d = DAYS[20]
    panel = daily_panel(["X"])
    a = ((panel.high - panel.low).rolling(14).mean().shift(1))["X"]
    atr_d = float(a.loc[d])
    assert atr_d == pytest.approx(4.0)
    cands = OrbStocksInPlay({"stop_atr": 0.10, "or_minutes": 5}).candidates(
        "X", day_bars(d), pd.DatetimeIndex([d]), panel)
    assert cands.iloc[0]["stop_pct"] == pytest.approx(0.10 * atr_d / 101.0)
    # a down first bar gives no trade
    none = OrbStocksInPlay({"stop_atr": 0.10, "or_minutes": 5}).candidates(
        "X", day_bars(d, first_up=False), pd.DatetimeIndex([d]), panel)
    assert none.empty


def test_orb_signal_uses_completed_bars_only():
    d = DAYS[12]
    panel = daily_panel(["X"])
    bars = day_bars(d, breakout_at="09:50")
    cut = pd.Timestamp(f"{d:%Y-%m-%d} 09:50", tz=TZ)
    changed = bars.copy()
    changed.loc[changed.index > cut, ["open", "high", "low", "close"]] = 500.0
    for rule in (Orb({"or_minutes": 5, "stop": 0.5, "target_r": 2.0}),
                 OrbStocksInPlay({"stop_atr": 0.5, "or_minutes": 5})):
        a = rule.candidates("X", bars, pd.DatetimeIndex([d]), panel)
        b = rule.candidates("X", changed, pd.DatetimeIndex([d]), panel)
        pd.testing.assert_frame_equal(a, b)
    # the feature frame of earlier days is unchanged by later bars as well
    f = OrbStocksInPlay({"stop_atr": 0.5, "or_minutes": 5})
    full = pd.concat([day_bars(x) for x in DAYS])
    part = pd.concat([day_bars(x) for x in DAYS[:20]])
    pd.testing.assert_frame_equal(f.day_features("X", full).iloc[:20],
                                  f.day_features("X", part))


def test_families_registered_with_documented_grids():
    assert DAYTRADE_FAMILIES["orb_sip"] is OrbStocksInPlay and DAYTRADE_FAMILIES["orb"] is Orb
    assert len(Orb.grid()) == 18 and len(OrbStocksInPlay.grid()) == 4
    assert len({Orb(p).rule_id for p in Orb.grid()}) == 18


def test_atr_floors_come_from_market_config():
    from autotrade.config import load_market_raw
    assert load_market_raw("us").orb_min_atr == 0.50 and load_market_raw("us").orb_min_atr_pct is None
    assert load_market_raw("kr").orb_min_atr_pct == 0.01 and load_market_raw("kr").orb_min_atr is None


def test_orb_sip_select_applies_the_us_atr_floor_and_trusts_eligible():
    panel, stacked, elig, syms = _sip_env()
    panel = make_panel(120, syms, seed=2, start="2023-11-01")
    for s in syms:
        panel.close[s], panel.open[s] = 50.0, 50.0
        panel.high[s], panel.low[s], panel.volume[s] = 52.0, 48.0, 2_000_000.0
    panel.high["S24"], panel.low["S24"] = 50.1, 49.9          # ATR 0.2 < $0.50
    elig = pd.DataFrame(True, index=panel.close.index, columns=syms)
    elig["S23"] = False                                       # universe says no
    sel = OrbStocksInPlay({"stop_atr": 0.5, "or_minutes": 5}).select(stacked, panel, elig)
    last = {s for d, s in sel if d == DAYS[-1]}
    assert "S24" not in last and "S23" not in last and "S22" in last


# ---- momentum families (Task 7) ----
KR_TZ = "Asia/Seoul"


def _mom_bars(day, tz, start, n, closes):
    ends = pd.date_range(f"{day:%Y-%m-%d} {start}", periods=n, freq="min", tz=tz)
    c = np.asarray(closes(ends), dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1000.0},
                        index=pd.DatetimeIndex(ends, name="ts"))


def _mom_panel(symbols, prev=100.0, products=None):
    p = make_panel(120, symbols, seed=3, start="2023-11-01", products=products)
    for s in symbols:
        p.close[s] = prev
    return p


def test_intraday_mom_signal_bar_and_exit_kr():
    from autotrade.rules.daytrade.momentum import IntradayMomentum
    d = DAYS[20]
    # prev close 100; closes 101 through the bar ending 09:30, 100 afterwards
    bars = _mom_bars(d, KR_TZ, "09:01", 380,
                     lambda e: np.where(e <= pd.Timestamp(f"{d:%Y-%m-%d} 09:30", tz=KR_TZ), 101.0, 100.0))
    panel = _mom_panel(["069500"], products={"069500": "etf"})
    days = pd.DatetimeIndex([pd.Timestamp(d)])
    c = IntradayMomentum({"threshold": 0.005}).candidates("069500", bars, days, panel)
    assert len(c) == 1
    assert c.iloc[0]["signal_ts"] == pd.Timestamp(f"{d:%Y-%m-%d} 14:50", tz=KR_TZ)
    assert c.iloc[0]["exit_ts"] == pd.Timestamp(f"{d:%Y-%m-%d} 15:20", tz=KR_TZ)
    assert c.iloc[0]["stop_pct"] == 0.02 and c.iloc[0]["confidence"] == pytest.approx(1.0)
    assert IntradayMomentum({"threshold": 0.02}).candidates("069500", bars, days, panel).empty
    # missing 14:50 bar -> last bar at or before it, never a later one
    gap = bars.drop(bars.index[(bars.index > pd.Timestamp(f"{d:%Y-%m-%d} 14:45", tz=KR_TZ))
                               & (bars.index <= pd.Timestamp(f"{d:%Y-%m-%d} 14:50", tz=KR_TZ))])
    g = IntradayMomentum({"threshold": 0.0}).candidates("069500", gap, days, panel)
    assert g.iloc[0]["signal_ts"] == pd.Timestamp(f"{d:%Y-%m-%d} 14:45", tz=KR_TZ)
    # US: signal 15:30, exit at session end
    ub = _mom_bars(d, TZ, "09:31", 390,
                   lambda e: np.where(e <= pd.Timestamp(f"{d:%Y-%m-%d} 10:00", tz=TZ), 101.0, 100.0))
    u = IntradayMomentum({"threshold": 0.0}).candidates(
        "SPY", ub, days, _mom_panel(["SPY"], products={"SPY": "etf"}))
    assert u.iloc[0]["signal_ts"] == pd.Timestamp(f"{d:%Y-%m-%d} 15:30", tz=TZ)
    assert pd.isna(u.iloc[0]["exit_ts"])


def test_close_mom_reversal_goes_long_after_a_down_day():
    from autotrade.rules.daytrade.momentum import CloseMomentum
    d = DAYS[20]
    days = pd.DatetimeIndex([pd.Timestamp(d)])
    panel = _mom_panel(["SPY"], products={"SPY": "etf"})
    down = _mom_bars(d, TZ, "09:31", 390, lambda e: np.full(len(e), 99.0))
    up = _mom_bars(d, TZ, "09:31", 390, lambda e: np.full(len(e), 101.0))
    rev = CloseMomentum({"direction": "reversal", "min_move": 0.0})
    mom = CloseMomentum({"direction": "momentum", "min_move": 0.0})
    assert len(rev.candidates("SPY", down, days, panel)) == 1
    assert rev.candidates("SPY", up, days, panel).empty
    assert len(mom.candidates("SPY", up, days, panel)) == 1
    assert mom.candidates("SPY", down, days, panel).empty
    # min_move 0.005 filters a 0.1% move
    flat = _mom_bars(d, TZ, "09:31", 390, lambda e: np.full(len(e), 99.9))
    assert CloseMomentum({"direction": "reversal", "min_move": 0.005}).candidates(
        "SPY", flat, days, panel).empty
    assert len(CloseMomentum.grid()) == 4


def test_momentum_families_trade_index_etfs_only():
    from autotrade.rules.daytrade.momentum import CloseMomentum, IntradayMomentum
    syms = ["SPY", "QQQ", "XLK", "AAPL"]
    panel = _mom_panel(syms, products={"SPY": "etf", "QQQ": "etf", "XLK": "etf"})
    elig = pd.DataFrame(True, index=panel.close.index, columns=syms)
    elig["QQQ"] = False                                  # universe says no
    feats = pd.DataFrame({"market": "us"},
                         index=pd.MultiIndex.from_product([[DAYS[20]], syms], names=["day", "symbol"]))
    cm = {s for _, s in CloseMomentum({"direction": "momentum", "min_move": 0.0}).select(feats, panel, elig)}
    im = {s for _, s in IntradayMomentum({"threshold": 0.0}).select(feats, panel, elig)}
    assert cm == {"SPY"}
    assert im == {"SPY", "XLK"}
    assert DAYTRADE_FAMILIES["intraday_mom"] is IntradayMomentum
    assert DAYTRADE_FAMILIES["close_mom"] is CloseMomentum and len(IntradayMomentum.grid()) == 3


# ---- gap and KR opening-overreaction families (Task 8) ----

def _ohlc_bars(day, tz, start, rows):
    """rows: (open, high, low, close, volume) per minute from `start` (END-stamped)."""
    ends = pd.date_range(f"{day:%Y-%m-%d} {start}", periods=len(rows), freq="min", tz=tz)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=ends)
    df.index.name = "ts"
    return df


def _gap_day(day, open_px, closes_after, low=None, n=40):
    """US day: first bar opens at open_px; bars close at `closes_after(i)`."""
    rows = []
    for i in range(n):
        c = closes_after(i)
        rows.append((open_px if i == 0 else c, max(c, open_px), c if low is None else low(i), c, 100.0))
    return _ohlc_bars(day, TZ, "09:31", rows)


def test_gap_go_needs_confirmation():
    from autotrade.rules.daytrade.gap import GapGo
    d = DAYS[20]
    days = pd.DatetimeIndex([d])
    panel = _mom_panel(["X"], prev=100.0)
    # +5% gap; confirm bar (09:35) closes above the open -> signal at 09:35
    up = _gap_day(d, 105.0, lambda i: 106.0, low=lambda i: 104.0 if i == 2 else 105.5)
    c = GapGo({"gap": 0.04, "confirm": 5}).candidates("X", up, days, panel)
    assert len(c) == 1
    assert c.iloc[0]["signal_ts"] == pd.Timestamp(f"{d:%Y-%m-%d} 09:35", tz=TZ)
    assert c.iloc[0]["stop_pct"] == pytest.approx((106.0 - 104.0) / 106.0)
    assert pd.isna(c.iloc[0]["exit_ts"]) and pd.isna(c.iloc[0]["target_pct"])
    assert c.iloc[0]["confidence"] == pytest.approx(min(1.0, 0.05 / 0.08))
    # confirmation bar closes below the open -> no trade
    down = _gap_day(d, 105.0, lambda i: 104.0)
    assert GapGo({"gap": 0.04, "confirm": 5}).candidates("X", down, days, panel).empty
    # gap smaller than the threshold -> no trade
    assert GapGo({"gap": 0.04, "confirm": 5}).candidates(
        "X", _gap_day(d, 102.0, lambda i: 103.0), days, panel).empty
    # a bar after the confirm window must not be used: close above open only from 09:36
    late = _gap_day(d, 105.0, lambda i: 106.0 if i >= 5 else 104.0)
    assert GapGo({"gap": 0.04, "confirm": 5}).candidates("X", late, days, panel).empty
    # day-0 prev close uses strictly earlier rows (panel includes d: d's own close is 999)
    p2 = _mom_panel(["X"], prev=100.0)
    p2.close.loc[d, "X"] = 999.0
    assert len(GapGo({"gap": 0.04, "confirm": 5}).candidates("X", up, days, p2)) == 1


def test_gap_fade_target_is_half_gap():
    from autotrade.rules.daytrade.gap import GapFade
    d = DAYS[20]
    days = pd.DatetimeIndex([d])
    panel = _mom_panel(["X"], prev=100.0)
    # -5% gap: opens 95; first bar low 94; signal bar (09:35) closes 95 > 94
    rows = [(95.0, 95.5, 94.0, 95.0, 100.0)] + [(95.0, 95.5, 94.5, 95.0, 100.0)] * 39
    bars = _ohlc_bars(d, TZ, "09:31", rows)
    c = GapFade({"gap": 0.04, "confirm": 5, "stop_pct": 0.02}).candidates("X", bars, days, panel)
    assert len(c) == 1
    assert c.iloc[0]["signal_ts"] == pd.Timestamp(f"{d:%Y-%m-%d} 09:35", tz=TZ)
    assert c.iloc[0]["target_pct"] == pytest.approx(0.05 / 2)
    assert c.iloc[0]["stop_pct"] == 0.02 and pd.isna(c.iloc[0]["exit_ts"])
    assert c.iloc[0]["confidence"] == pytest.approx(min(1.0, 0.05 / 0.08))
    # signal close at/below the first bar's low -> no trade
    rows2 = [(95.0, 95.5, 94.0, 95.0, 100.0)] + [(94.0, 94.0, 93.0, 93.5, 100.0)] * 39
    assert GapFade({"gap": 0.04, "confirm": 5, "stop_pct": 0.02}).candidates(
        "X", _ohlc_bars(d, TZ, "09:31", rows2), days, panel).empty
    # a gap UP is not faded
    assert GapFade({"gap": 0.04, "confirm": 5, "stop_pct": 0.02}).candidates(
        "X", _gap_day(d, 105.0, lambda i: 106.0), days, panel).empty


def _rev_bars(day, tz, first30_close, vol30):
    """KR day: 09:01.. 120 bars; the bar ending 09:30 closes at first30_close."""
    ends = pd.date_range(f"{day:%Y-%m-%d} 09:01", periods=120, freq="min", tz=tz)
    c = np.where(ends <= pd.Timestamp(f"{day:%Y-%m-%d} 09:30", tz=tz), first30_close, 90.0)
    v = np.where(ends <= pd.Timestamp(f"{day:%Y-%m-%d} 09:30", tz=tz), vol30 / 30, 10.0)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": v},
                        index=pd.DatetimeIndex(ends, name="ts"))


def test_kr_open_rev_skips_us():
    from autotrade.rules.daytrade.gap import OpenReversal
    rule = OpenReversal({"drop": 0.03, "hold": 60, "stop_pct": 0.02})
    days_idx = DAYS[:16]
    kr = pd.concat([_rev_bars(x, KR_TZ, 100.0, 3000.0) for x in days_idx[:15]]
                   + [_rev_bars(days_idx[15], KR_TZ, 95.0, 9000.0)])
    panel = _mom_panel(["005930"], prev=100.0)
    feats = rule.day_features("005930", kr)
    # 14 prior days exist only for the 16th day (days 1..14 -> day 15 is the first with 14 priors)
    assert feats["avg14"].iloc[:14].isna().all() and feats["avg14"].iloc[14] == pytest.approx(3000.0)
    stacked = pd.concat({"005930": feats}, names=["symbol"]).swaplevel().sort_index()
    elig = pd.DataFrame(True, index=panel.close.index, columns=panel.close.columns)
    sel = rule.select(stacked, panel, elig)
    assert list(sel) == [(days_idx[15], "005930")]  # only the day with >= 2x the average
    last = days_idx[15]
    c = rule.candidates("005930", kr, pd.DatetimeIndex([last]), panel)
    assert len(c) == 1
    assert c.iloc[0]["signal_ts"] == pd.Timestamp(f"{last:%Y-%m-%d} 09:30", tz=KR_TZ)
    assert c.iloc[0]["exit_ts"] == pd.Timestamp(f"{last:%Y-%m-%d} 10:30", tz=KR_TZ)
    assert c.iloc[0]["stop_pct"] == 0.02 and c.iloc[0]["confidence"] == pytest.approx(0.5)
    # drop not deep enough -> no candidate
    assert OpenReversal({"drop": 0.05, "hold": 60, "stop_pct": 0.02}).candidates(
        "005930", kr, pd.DatetimeIndex([last]), _mom_panel(["005930"], prev=97.0)).empty
    # US bars: nothing selected, nothing generated
    us = pd.concat([_rev_bars(x, TZ, 95.0, 9000.0) for x in days_idx])
    assert rule.day_features("SPY", us).empty or (rule.day_features("SPY", us)["market"] == "us").all()
    ufeat = pd.concat({"SPY": rule.day_features("SPY", us)}, names=["symbol"]).swaplevel().sort_index()
    uelig = pd.DataFrame(True, index=panel.close.index, columns=["SPY"])
    assert len(rule.select(ufeat, panel, uelig)) == 0
    assert rule.candidates("SPY", us, pd.DatetimeIndex(days_idx), _mom_panel(["SPY"])).empty


def test_gap_families_registered_with_documented_grids():
    from autotrade.rules.daytrade.gap import GapFade, GapGo, OpenReversal
    assert DAYTRADE_FAMILIES["gap_go"] is GapGo and DAYTRADE_FAMILIES["gap_fade"] is GapFade
    assert DAYTRADE_FAMILIES["kr_open_rev"] is OpenReversal
    assert len(GapGo.grid()) == 4 and len(GapFade.grid()) == 8 and len(OpenReversal.grid()) == 8


# ---- per-day sessions: late opens, exam days, half-days (data/sessions.py) ----

def _special_panel(symbols, day, prev=100.0, products=None):
    p = make_panel(60, symbols, seed=3, start=f"{pd.Timestamp(day) - pd.Timedelta(days=50):%Y-%m-%d}",
                   products=products)
    assert pd.Timestamp(day) in p.close.index
    for s in symbols:
        p.close[s] = prev
    return p


def test_close_mom_us_half_day_signals_before_the_1300_close():
    from autotrade.data.minute_store import regular_session
    from autotrade.config import load_market_raw
    from autotrade.rules.daytrade.momentum import CloseMomentum, IntradayMomentum
    d = pd.Timestamp("2024-11-29")                  # NYSE closes 13:00; Toss has bars to 20:00
    days = pd.DatetimeIndex([d])
    panel = _special_panel(["SPY"], d, products={"SPY": "etf"})
    raw = _mom_bars(d, TZ, "04:01", 959, lambda e: np.full(len(e), 101.0))    # 04:01..19:59
    mom = CloseMomentum({"direction": "momentum", "min_move": 0.0})
    for bars in (raw, regular_session(raw, load_market_raw("us"))):
        c = mom.candidates("SPY", bars, days, panel)
        assert c.iloc[0]["signal_ts"] == pd.Timestamp("2024-11-29 12:30", tz=TZ)   # close - 30
        assert pd.isna(c.iloc[0]["exit_ts"])
        i = IntradayMomentum({"threshold": 0.0}).candidates("SPY", bars, days, panel)
        assert i.iloc[0]["signal_ts"] == pd.Timestamp("2024-11-29 12:30", tz=TZ)
    # no post-market bar is ever a signal: a post-market-only rally is not seen
    post = _mom_bars(d, TZ, "04:01", 959,
                     lambda e: np.where(e > pd.Timestamp("2024-11-29 13:00", tz=TZ), 101.0, 99.0))
    assert mom.candidates("SPY", post, days, panel).empty


def test_intraday_mom_kr_exam_day_times_follow_the_late_session():
    from autotrade.rules.daytrade.momentum import IntradayMomentum
    d = pd.Timestamp("2024-11-14")                  # CSAT: 10:00-16:30, auction from 16:20
    days = pd.DatetimeIndex([d])
    panel = _special_panel(["069500"], d, products={"069500": "etf"})
    # 101 through the bar ending 10:30 (open + 30), 99 afterwards; 09:01-09:59 padded at 90
    bars = _mom_bars(d, KR_TZ, "09:01", 450,
                     lambda e: np.where(e <= pd.Timestamp("2024-11-14 10:00", tz=KR_TZ), 90.0,
                                        np.where(e <= pd.Timestamp("2024-11-14 10:30", tz=KR_TZ),
                                                 101.0, 99.0)))
    c = IntradayMomentum({"threshold": 0.005}).candidates("069500", bars, days, panel)
    assert len(c) == 1
    assert c.iloc[0]["signal_ts"] == pd.Timestamp("2024-11-14 15:50", tz=KR_TZ)
    assert c.iloc[0]["exit_ts"] == pd.Timestamp("2024-11-14 16:20", tz=KR_TZ)


def test_orb_and_gap_windows_start_at_the_kr_new_year_open():
    from autotrade.rules.daytrade.gap import GapGo
    d = pd.Timestamp("2025-01-02")                  # KRX opens 10:00
    days = pd.DatetimeIndex([d])
    panel = _special_panel(["005930"], d)
    rows = [(105.0, 106.0, 104.0, 105.5, 100.0)] * 5 + [(105.5, 108.0, 105.5, 107.0, 100.0)] \
        + [(107.0, 107.0, 107.0, 107.0, 100.0)] * 60
    bars = _ohlc_bars(d, KR_TZ, "10:01", rows)
    c = Orb({"or_minutes": 5, "stop": "or_low", "target_r": np.nan}).candidates(
        "005930", bars, days, panel)
    assert len(c) == 1 and c.iloc[0]["signal_ts"] == pd.Timestamp("2025-01-02 10:06", tz=KR_TZ)
    g = GapGo({"gap": 0.04, "confirm": 5}).candidates("005930", bars, days, panel)
    assert len(g) == 1 and g.iloc[0]["signal_ts"] == pd.Timestamp("2025-01-02 10:05", tz=KR_TZ)
