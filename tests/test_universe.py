from dataclasses import replace

import numpy as np
import pandas as pd

from autotrade.config import load_lab, load_market_raw, load_pocket
from autotrade.universe import eligible_mask
from tests.fixtures import make_panel

KR = load_market_raw("kr")
US = load_market_raw("us")
SWING = load_pocket("swing")
LAB = load_lab()


def _flat(p, sym, price, volume=1_000_000.0):
    p.close[sym] = price
    p.volume[sym] = volume


def test_point_in_time_no_lookahead():
    p = make_panel(60, ["A", "B", "C"], seed=1)
    m1 = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    p.close.iloc[40:, 0] *= 100
    p.volume.iloc[40:, 0] *= 100
    m2 = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    pd.testing.assert_frame_equal(m1.iloc[:41], m2.iloc[:41])


def test_shape_and_dtype():
    p = make_panel(30, ["A", "B"], seed=1)
    m = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert m.shape == p.close.shape and (m.dtypes == bool).all()
    assert m.index.equals(p.close.index) and list(m.columns) == ["A", "B"]


def test_first_day_is_false():
    p = make_panel(30, ["A"], seed=1)
    _flat(p, "A", 10_000.0)
    assert not eligible_mask(p, KR, SWING, LAB, "paper", 1400).iloc[0, 0]


def test_top_n_by_20d_trading_value():
    syms = ["A", "B", "C", "D"]
    p = make_panel(40, syms, seed=2)
    for s, v in zip(syms, [4e6, 3e6, 2e6, 1e6]):
        _flat(p, s, 10_000.0, v)
    lab = replace(LAB, universe_top_n=2)
    m = eligible_mask(p, KR, SWING, lab, "paper", 1400)
    assert m.iloc[-1].to_dict() == {"A": True, "B": True, "C": False, "D": False}


def test_top_n_ranking_uses_trading_value_not_volume():
    p = make_panel(40, ["A", "B"], seed=2)
    _flat(p, "A", 10_000.0, 1e6)   # value 1e10
    _flat(p, "B", 1_000.0, 5e6)    # value 5e9, more shares
    lab = replace(LAB, universe_top_n=1)
    m = eligible_mask(p, KR, SWING, lab, "paper", 1400)
    assert bool(m["A"].iloc[-1]) and not m["B"].iloc[-1]


def test_kr_tick_cost_filter():
    p = make_panel(40, ["LO", "HI"], seed=3)
    _flat(p, "LO", 1990.0)   # tick 1 -> 0.05%
    _flat(p, "HI", 2010.0)   # tick 5 -> 0.25%
    m = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert bool(m["LO"].iloc[-1]) and not m["HI"].iloc[-1]


def test_kr_tick_cost_uses_product_table():
    p = make_panel(40, ["E", "S"], seed=3, products={"E": "etf"})
    _flat(p, "E", 10_000.0)  # etf tick 5 -> 0.05% ok
    _flat(p, "S", 10_000.0)  # stock tick 10 -> 0.1% (boundary, <= passes)
    m = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert m["E"].iloc[-1] and m["S"].iloc[-1]
    _flat(p, "S", 4_000.0)   # stock tick 5 -> 0.125% fails
    m = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert not m["S"].iloc[-1]


def test_real_scenario_price_cap_kr():
    p = make_panel(40, ["CHEAP", "DEAR"], seed=4)
    _flat(p, "CHEAP", 12_000.0)
    _flat(p, "DEAR", 13_000.0)
    real = eligible_mask(p, KR, SWING, LAB, "real", 1400)
    paper = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert bool(real["CHEAP"].iloc[-1]) and not real["DEAR"].iloc[-1]
    assert paper["CHEAP"].iloc[-1] and paper["DEAR"].iloc[-1]


def test_real_scenario_no_price_cap_when_fractional_us():
    p = make_panel(40, ["X"], seed=4)
    _flat(p, "X", 500.0, 2_000_000.0)
    assert eligible_mask(p, US, SWING, LAB, "real", 1400)["X"].iloc[-1]


def test_us_filters_price_and_volume():
    p = make_panel(40, ["OK", "CHEAP", "THIN"], seed=5)
    _flat(p, "OK", 50.0, 2_000_000.0)
    _flat(p, "CHEAP", 4.9, 2_000_000.0)
    _flat(p, "THIN", 50.0, 900_000.0)
    m = eligible_mask(p, US, SWING, LAB, "paper", 1400)
    assert m.iloc[-1].to_dict() == {"OK": True, "CHEAP": False, "THIN": False}


def test_us_price_uses_prior_close_not_current():
    p = make_panel(40, ["X"], seed=5)
    _flat(p, "X", 50.0, 2_000_000.0)
    p.close.iloc[-1, 0] = 1.0
    assert eligible_mask(p, US, SWING, LAB, "paper", 1400)["X"].iloc[-1]


def test_excludes_preferred_spac_leveraged():
    syms = ["PREF", "SPAC", "LEV", "INV", "OK"]
    p = make_panel(40, syms, seed=6,
                   is_common_share={"PREF": False},
                   name={"SPAC": "OO스팩1호", "LEV": "KODEX 레버리지", "INV": "KODEX 인버스"})
    for s in syms:
        _flat(p, s, 10_000.0)
    m = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert m.iloc[-1].to_dict() == {"PREF": False, "SPAC": False, "LEV": False,
                                    "INV": False, "OK": True}


def test_exclusion_keywords_ascii_case_insensitive():
    p = make_panel(40, ["U", "OK"], seed=6, name={"U": "ProShares ultra QQQ"})
    _flat(p, "U", 50.0, 2e6)
    _flat(p, "OK", 50.0, 2e6)
    m = eligible_mask(p, US, SWING, LAB, "paper", 1400)
    assert not m["U"].iloc[-1] and m["OK"].iloc[-1]


def test_excludes_non_stock_etf_products():
    p = make_panel(40, ["A"], seed=6)
    p.meta["product"] = p.meta["product"].astype(object)
    p.meta.loc["A", "product"] = "etn"
    _flat(p, "A", 10_000.0)
    assert not eligible_mask(p, KR, SWING, LAB, "paper", 1400)["A"].iloc[-1]


def test_always_include_index_etfs():
    syms = ["069500", "X1", "X2"]
    p = make_panel(40, syms, seed=7, products={"069500": "etf"})
    _flat(p, "069500", 30_000.0, 1_000.0)   # tiny trading value, would fail price caps too
    _flat(p, "X1", 10_000.0, 3e6)
    _flat(p, "X2", 10_000.0, 2e6)
    lab = replace(LAB, universe_top_n=1)
    m = eligible_mask(p, KR, SWING, lab, "real", 1400)
    assert m["X1"].iloc[-1]
    assert m["069500"].iloc[-1]
    assert not m["X2"].iloc[-1]


def test_always_include_still_needs_data_and_exclusions():
    syms = ["069500", "102110"]
    p = make_panel(40, syms, seed=7, products={"069500": "etf", "102110": "etf"},
                   name={"102110": "TIGER 레버리지"})
    p.close.loc[p.close.index[:10], "069500"] = np.nan
    p.volume.loc[p.volume.index[:10], "069500"] = np.nan
    m = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert not m["069500"].iloc[:11].any()
    assert m["069500"].iloc[-1]
    assert not m["102110"].any()


def test_delisted_symbol_eligible_only_while_trading():
    p = make_panel(60, ["A", "B"], seed=8)
    _flat(p, "A", 10_000.0)
    _flat(p, "B", 10_000.0)
    p.close.iloc[40:, 0] = np.nan
    p.volume.iloc[40:, 0] = np.nan
    m = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert m["A"].iloc[30] and m["A"].iloc[40]       # t=40 uses t-1=39 data
    assert not m["A"].iloc[41:].any()                # t-1 NaN
    assert m["B"].iloc[-1]


def test_not_yet_listed_is_false_and_nan_does_not_crash_rank():
    p = make_panel(60, ["A", "B"], seed=8)
    _flat(p, "A", 10_000.0)
    _flat(p, "B", 10_000.0)
    p.close.iloc[:25, 1] = np.nan
    p.volume.iloc[:25, 1] = np.nan
    m = eligible_mask(p, KR, SWING, LAB, "paper", 1400)
    assert not m["B"].iloc[:26].any()
    assert m["B"].iloc[-1]
