"""Task 4: event classification, signal dates (no look-ahead), repeats, sizes, Panel.events."""
import shutil
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from autotrade import forward, pipeline
from autotrade.cli import app
from autotrade.data.store import write_parquet_atomic
from autotrade.disclosure import events as E
from autotrade.disclosure.store import (FILING_COLUMNS, append_filings, append_sizes,
                                        filings_path, targets_path, write_json)
from autotrade.rules.registry import SWING_FAMILIES
from tests.fixtures import make_panel
from tests.test_cli import cli_env, extend_store, periods_of, run  # noqa: F401 (fixture)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"

# KR: 2024 business days without 2024-03-01 (삼일절, a Friday)
KR_CAL = pd.bdate_range("2024-01-02", "2024-12-31").drop(pd.Timestamp("2024-03-01"))
# US: 2024 business days without 2024-07-04 and 2024-11-28 (11-29 is an early close)
US_CAL = pd.bdate_range("2024-01-02", "2024-12-31").drop(
    [pd.Timestamp("2024-07-04"), pd.Timestamp("2024-11-28")])


def kr_filings(*rows):
    """rows: (filing_id, symbol, filed 'YYYY-MM-DD', title)."""
    df = pd.DataFrame([{"filing_id": f, "symbol": s, "corp_code": "0000" + s,
                        "filed_date": date.fromisoformat(d), "title": t, "kind": "B",
                        "filer": "x", "remark": ""} for f, s, d, t in rows],
                      columns=FILING_COLUMNS)
    append_filings("kr", df)


def us_filings(*rows):
    """rows: (filing_id, symbol, accepted_at NY 'YYYY-MM-DD HH:MM', form, items, insider_usd)."""
    recs = []
    for f, s, ny, form, items, usd in rows:
        ts = pd.Timestamp(ny, tz="America/New_York").tz_convert("UTC")
        recs.append({"filing_id": f, "symbol": s, "cik": "1", "filed_date": ts.date(),
                     "accepted_at": ts, "form": form, "items": items, "insider_usd": usd})
    write_parquet_atomic(pd.DataFrame(recs), filings_path("us"))


def nth(cal, i) -> str:
    return cal[i].date().isoformat()


BUYBACK = "주요사항보고서(자기주식취득결정)"
SUPPLY = "단일판매ㆍ공급계약체결"


# ---------------------------------------------------------------- signal dates
def test_kr_weekend_filing_maps_to_friday():
    kr_filings(("f1", "A", "2024-03-09", BUYBACK))
    ev = E.load_events("kr", KR_CAL)
    assert list(ev["signal_date"]) == [pd.Timestamp("2024-03-08")]


def test_kr_weekday_filing_maps_to_same_day():
    kr_filings(("f1", "A", "2024-03-12", BUYBACK))
    assert list(E.load_events("kr", KR_CAL)["signal_date"]) == [pd.Timestamp("2024-03-12")]


def test_kr_holiday_filing_maps_to_the_trading_day_before():
    kr_filings(("f1", "A", "2024-03-01", BUYBACK))
    assert list(E.load_events("kr", KR_CAL)["signal_date"]) == [pd.Timestamp("2024-02-29")]


def test_us_acceptance_times():
    us_filings(("a1", "X1", "2024-07-09 08:00", "8-K", "1.01", np.nan),   # before the open
               ("a2", "X2", "2024-07-10 12:00", "8-K", "8.01,1.01", np.nan),  # in session
               ("a3", "X3", "2024-07-11 17:00", "8-K", "1.01,9.01", np.nan),  # after close
               ("a4", "Y", "2024-07-04 08:00", "8-K", "1.01", np.nan),   # holiday
               ("a5", "Z", "2024-07-08 08:00", "8-K", "1.01", np.nan),   # Monday pre-open
               ("a6", "W", "2024-07-09 22:00", "8-K", "1.01", np.nan),   # UTC date is 07-10
               ("a7", "V", "2024-07-09 09:30", "8-K", "1.01", np.nan))   # exactly at the open
    ev = E.load_events("us", US_CAL).set_index("filing_id")["signal_date"]
    assert ev["a1"] == pd.Timestamp("2024-07-08")
    assert ev["a2"] == pd.Timestamp("2024-07-10")
    assert ev["a3"] == pd.Timestamp("2024-07-11")
    assert ev["a4"] == pd.Timestamp("2024-07-03")
    assert ev["a5"] == pd.Timestamp("2024-07-05")
    assert ev["a6"] == pd.Timestamp("2024-07-09")
    assert ev["a7"] == pd.Timestamp("2024-07-09")


def test_us_early_close_day_changes_nothing():
    """2024-11-29 closes at 13:00: a 14:00 acceptance is after the close but still maps to
    D (entry at the next open, 12-02); a pre-open one maps to the day before (11-27)."""
    us_filings(("a1", "X", "2024-11-29 14:00", "8-K", "1.01", np.nan),
               ("a2", "Y", "2024-11-29 08:00", "8-K", "1.01", np.nan))
    ev = E.load_events("us", US_CAL).set_index("filing_id")["signal_date"]
    assert ev["a1"] == pd.Timestamp("2024-11-29")
    assert ev["a2"] == pd.Timestamp("2024-11-27")


def test_events_outside_the_calendar_are_dropped():
    """Before the first calendar date: no trading day <= D. After the last: D's trading-day
    status is unknown, so mapping it to the last stored day would be look-ahead."""
    kr_filings(("f1", "A", "2023-12-29", BUYBACK), ("f2", "A", "2025-01-02", BUYBACK),
               ("f3", "B", "2024-12-31", BUYBACK))
    assert list(E.load_events("kr", KR_CAL)["filing_id"]) == ["f3"]
    us_filings(("a1", "X", "2024-01-02 08:00", "8-K", "1.01", np.nan),   # day before: none
               ("a2", "X", "2025-01-01 08:00", "8-K", "1.01", np.nan))
    assert E.load_events("us", US_CAL).empty


# ---------------------------------------------------------------- classification
def test_corrections_are_not_events():
    kr_filings(("f1", "A", "2024-03-12", BUYBACK),
               ("f2", "A", "2024-05-13", "[기재정정]" + BUYBACK),
               ("f3", "B", "2024-05-13", "[첨부정정] [첨부추가]" + SUPPLY),
               ("f4", "C", "2024-05-13", "[변경등록]무상증자결정"))
    assert list(E.load_events("kr", KR_CAL)["filing_id"]) == ["f1"]
    us_filings(("a1", "X", "2024-07-09 12:00", "8-K/A", "1.01", np.nan),
               ("a2", "X", "2024-07-09 12:00", "4/A", "", 500_000.0))
    assert E.load_events("us", US_CAL).empty


def test_withdrawal_kept_with_direction_zero():
    kr_filings(("f1", "A", "2024-03-12", "유상증자결정"),
               ("f2", "A", "2024-03-14", "유상증자결정(철회)"))
    ev = E.load_events("kr", KR_CAL).set_index("filing_id")
    assert ev.loc["f1", "event"] == "rights_offering" and ev.loc["f1", "direction"] == -1
    assert ev.loc["f2", "event"] == "withdrawal" and ev.loc["f2", "direction"] == 0
    assert not ev.loc["f2", "positive"]


def test_unmatched_titles_are_not_events():
    kr_filings(("f1", "A", "2024-03-12", "임원ㆍ주요주주특정증권등소유상황보고서"))
    ev = E.load_events("kr", KR_CAL)
    assert ev is not None and ev.empty and list(ev.columns) == E.EVENT_COLUMNS


def test_title_whitespace_does_not_hide_an_event():
    kr_filings(("f1", "A", "2024-03-12", "주요사항보고서 (자기주식 취득 결정)"))
    assert list(E.load_events("kr", KR_CAL)["event"]) == ["buyback"]


def test_us_table():
    us_filings(("a1", "X", "2024-07-09 12:00", "8-K", "2.02,1.01", np.nan),
               ("a2", "X", "2024-07-09 12:00", "8-K", "3.01", np.nan),
               ("a3", "X", "2024-07-09 12:00", "8-K", "4.02", np.nan),
               ("a4", "X", "2024-07-09 12:00", "424B5", "", np.nan),
               ("a5", "X", "2024-07-09 12:00", "4", "", 150_000.0),
               ("a6", "Y", "2024-07-09 12:00", "4", "", 50_000.0),
               ("a7", "Z", "2024-07-09 12:00", "4", "", np.nan),        # no code-P buy
               ("a8", "Z", "2024-07-09 12:00", "8-K", "11.01", np.nan),  # not item 1.01
               ("a9", "Z", "2024-07-09 12:00", "8-K", "2.02", np.nan))
    ev = E.load_events("us", US_CAL).set_index("filing_id")
    assert dict(ev["event"]) == {"a1": "8k_101", "a2": "8k_301", "a3": "8k_402",
                                 "a4": "offering_424b", "a5": "insider_buy",
                                 "a6": "insider_buy"}
    assert dict(ev["direction"]) == {"a1": 1, "a2": -1, "a3": -1, "a4": 0, "a5": 1, "a6": 1}
    assert ev.loc["a5", "size"] == 150_000.0 and ev.loc["a5", "size_ok"]
    assert ev.loc["a6", "size"] == 50_000.0 and not ev.loc["a6", "size_ok"]
    assert ev.loc["a1", "bundle_a"] and ev.loc["a5", "bundle_b"] and ev.loc["a6", "bundle_b"]
    assert list(ev.index[ev["positive"]]) == ["a1", "a5", "a6"]


def test_kr_bundles_and_positive():
    kr_filings(("f1", "A", "2024-03-12", BUYBACK), ("f2", "B", "2024-03-12", SUPPLY),
               ("f3", "C", "2024-03-12", "무상증자결정"), ("f4", "D", "2024-03-12", "감자결정"))
    ev = E.load_events("kr", KR_CAL).set_index("filing_id")
    assert list(ev.index[ev["bundle_a"]]) == ["f1"]
    assert list(ev.index[ev["bundle_b"]]) == ["f2"]
    assert list(ev.index[ev["positive"]]) == ["f1", "f2", "f3"]
    assert ev.loc["f4", "direction"] == -1


# ---------------------------------------------------------------- repeats
def test_repeat_within_20_days_dropped():
    """Repeat = same symbol + event with a signal date fewer than 20 calendar rows after the
    last KEPT one: rows 10 (kept), 15 (dropped), 29 (dropped: 19 rows), 30 (kept: 20 rows
    after row 10 -- the dropped row 15/29 does not restart the window)."""
    kr_filings(("f10", "A", nth(KR_CAL, 10), BUYBACK), ("f15", "A", nth(KR_CAL, 15), BUYBACK),
               ("f29", "A", nth(KR_CAL, 29), BUYBACK), ("f30", "A", nth(KR_CAL, 30), BUYBACK),
               ("g15", "B", nth(KR_CAL, 15), BUYBACK),     # another symbol: own window
               ("h15", "A", nth(KR_CAL, 15), SUPPLY))      # another event: own window
    ev = E.load_events("kr", KR_CAL)
    assert sorted(ev["filing_id"]) == ["f10", "f30", "g15", "h15"]


def test_same_day_duplicates_keep_the_first_filing_id():
    kr_filings(("f2", "A", "2024-03-12", BUYBACK), ("f1", "A", "2024-03-12", BUYBACK))
    assert list(E.load_events("kr", KR_CAL)["filing_id"]) == ["f1"]


# ---------------------------------------------------------------- sizes
def test_size_unknown_is_not_ok():
    kr_filings(("f1", "A", "2024-03-12", SUPPLY), ("f2", "B", "2024-03-12", SUPPLY),
               ("f3", "C", "2024-03-12", SUPPLY), ("f4", "D", "2024-03-12", SUPPLY),
               ("f5", "E", "2024-03-12", "유상증자결정"))
    append_sizes("kr", pd.DataFrame({
        "filing_id": ["f1", "f2", "f3", "f5"],
        "measure": ["supply_revenue_ratio"] * 3 + ["rights_dilution"],
        "value": [0.15, 0.05, np.nan, 0.2]}))
    ev = E.load_events("kr", KR_CAL).set_index("filing_id")
    assert ev.loc["f1", "size"] == 0.15 and ev.loc["f1", "size_ok"]
    assert not ev.loc["f2", "size_ok"]
    assert np.isnan(ev.loc["f3", "size"]) and not ev.loc["f3", "size_ok"]   # unreadable
    assert np.isnan(ev.loc["f4", "size"]) and not ev.loc["f4", "size_ok"]   # never fetched
    assert ev.loc["f5", "size_ok"]


def test_event_without_size_rule_is_ok():
    kr_filings(("f1", "A", "2024-03-12", "무상증자결정"), ("f2", "B", "2024-03-12", "감자결정"))
    ev = E.load_events("kr", KR_CAL)
    assert ev["size_ok"].all() and ev["size"].isna().all()


def test_buyback_size_is_amount_over_adv():
    kr_filings(("f1", "A", "2024-03-12", BUYBACK), ("f2", "B", "2024-03-12", BUYBACK),
               ("f3", "C", "2024-03-12", BUYBACK))
    append_sizes("kr", pd.DataFrame({"filing_id": ["f1", "f2", "f3"],
                                     "measure": ["buyback_krw"] * 3,
                                     "value": [3e9, 5e8, 3e9]}))
    adv = pd.DataFrame({"A": 1e9, "B": 1e9, "C": np.nan}, index=KR_CAL)
    ev = E.load_events("kr", KR_CAL, adv).set_index("filing_id")
    assert ev.loc["f1", "size"] == pytest.approx(3.0) and ev.loc["f1", "size_ok"]
    assert ev.loc["f2", "size"] == pytest.approx(0.5) and not ev.loc["f2", "size_ok"]
    assert np.isnan(ev.loc["f3", "size"]) and not ev.loc["f3", "size_ok"]
    no_adv = E.load_events("kr", KR_CAL).set_index("filing_id")
    assert no_adv["size"].isna().all() and not no_adv["size_ok"].any()


# ---------------------------------------------------------------- frame contract
def test_no_filings_means_none():
    assert E.load_events("kr", KR_CAL) is None
    assert E.load_events("us", US_CAL) is None


def test_events_are_deterministic():
    rows = [(f"f{i:02d}", "ABC"[i % 3], nth(KR_CAL, 3 * i), [BUYBACK, SUPPLY, "감자결정"][i % 3])
            for i in range(30)]
    kr_filings(*rows)
    a = E.load_events("kr", KR_CAL)
    filings_path("kr").unlink()
    kr_filings(*reversed(rows))
    b = E.load_events("kr", KR_CAL)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == E.EVENT_COLUMNS
    assert a.equals(a.sort_values(["signal_date", "symbol", "filing_id"])
                    .reset_index(drop=True))
    assert a["signal_date"].dtype == "datetime64[ns]" and a["direction"].dtype == "int64"
    assert all(a[c].dtype == bool for c in ("size_ok", "bundle_a", "bundle_b", "positive"))


def test_events_table_hash_changes_with_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    shutil.copytree(REPO_CONFIG / "events", cfg / "events")
    monkeypatch.setenv("AUTOTRADE_CONFIG", str(cfg))
    h = E.events_table_hash("kr")
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)
    assert h != E.events_table_hash("us")
    p = cfg / "events" / "kr.yaml"
    p.write_text(p.read_text(encoding="utf-8").replace("size_min: 0.10", "size_min: 0.20", 1),
                 encoding="utf-8")
    assert E.events_table_hash("kr") != h


def test_sizes_kr_reads_the_same_table():
    from autotrade.disclosure import sizes_kr
    t = E.load_table("kr")
    assert sizes_kr.size_patterns("kr") == {d.size: d.pattern.pattern for d in t.events
                                            if d.size}


def test_bad_table_is_refused(tmp_path, monkeypatch):
    (tmp_path / "events").mkdir()
    (tmp_path / "events" / "kr.yaml").write_text(
        "repeat_days: 20\nnegative_window: 20\nevents:\n"
        "  - {name: a, direction: 2, match: x}\nbundles: {bundle_a: [a], bundle_b: [a]}\n",
        encoding="utf-8")
    monkeypatch.setenv("AUTOTRADE_CONFIG", str(tmp_path))
    with pytest.raises(E.ConfigError, match="direction"):
        E.load_table("kr")


# ---------------------------------------------------------------- Panel.events
def panel_events(p, rows):
    """rows: (symbol, row index, direction, bundle_a, size_ok)."""
    return pd.DataFrame([{"symbol": s, "signal_date": p.close.index[i], "event": "e",
                          "direction": d, "size": np.nan, "size_ok": ok, "bundle_a": a,
                          "bundle_b": False, "positive": d == 1, "filing_id": f"x{n}"}
                         for n, (s, i, d, a, ok) in enumerate(rows)], columns=E.EVENT_COLUMNS)


def test_slice_drops_events_outside_window():
    p = make_panel(100, ["A", "B"], seed=1)
    p = replace(p, events=panel_events(p, [("A", 5, 1, True, True), ("A", 50, 1, True, True),
                                           ("B", 90, -1, False, True)]))
    s = p.slice(p.close.index[10].date(), p.close.index[90].date())
    assert list(s.events["filing_id"]) == ["x1", "x2"]
    assert make_panel(20, ["A"], seed=1).slice(date(2016, 1, 1), date(2016, 2, 1)).events is None


def test_attach_events_up_to_end_with_adv_sizes():
    p = make_panel(300, ["A", "B"], seed=4)
    p = replace(p, calendar=p.close.index)
    idx = p.close.index
    kr_filings(("f1", "A", nth(idx, 100), BUYBACK), ("f2", "B", nth(idx, 200), BUYBACK),
               ("f3", "Z", nth(idx, 100), "감자결정"))
    append_sizes("kr", pd.DataFrame({"filing_id": ["f1"], "measure": ["buyback_krw"],
                                     "value": [1e9]}))
    a = E.attach_events(p, "kr", idx[150].date())
    assert list(a.events["filing_id"]) == ["f1", "f3"]
    adv = float((p.close["A"] * p.volume["A"]).iloc[81:101].mean())
    assert a.events.set_index("filing_id").loc["f1", "size"] == pytest.approx(1e9 / adv)
    assert len(E.attach_events(p, "kr", idx[-1].date()).events) == 3
    pd.testing.assert_frame_equal(a.close, p.close)


def test_panel_attaches_only_events_up_to_end(cli_env):
    """The screen panel (prices through the train end) carries no event dated after it."""
    s = pipeline._setup("swing", "kr", create=True)
    t1 = s.periods.train[1]
    cal = s.calendar
    i = cal.get_loc(pd.Timestamp(t1))
    kr_filings(("in", "A01", nth(cal, i - 3), BUYBACK), ("out", "A02", nth(cal, i + 1), BUYBACK))
    screen_panel = pipeline._panel(s, t1)
    assert list(screen_panel.events["filing_id"]) == ["in"]
    valid_panel = pipeline._panel(s, s.periods.valid[1])
    assert list(valid_panel.events["filing_id"]) == ["in", "out"]


def test_forward_panel_carries_events(cli_env, monkeypatch):
    from tests.test_forward import register_after_cutoff, screened_and_validated
    screened_and_validated()
    register_after_cutoff(cli_env, monkeypatch)
    new_days = extend_store(5)
    kr_filings(("fw", "A01", new_days[2].date().isoformat(), BUYBACK))
    seen = []
    real = forward.attach_events

    def spy(panel, market, end):
        out = real(panel, market, end)
        seen.append(out.events)
        return out
    monkeypatch.setattr(forward, "attach_events", spy)
    forward.run("swing", "kr")
    assert len(seen) == 1 and list(seen[0]["filing_id"]) == ["fw"]


@pytest.mark.parametrize("family", SWING_FAMILIES)
def test_existing_families_unchanged_with_events(family):
    p = make_panel(400, ["A", "B", "C", "E1"], seed=7, products={"E1": "etf"})
    rng = np.random.default_rng(0)
    rows = [(str(rng.choice(["A", "B", "C", "E1"])), int(rng.integers(0, 400)),
             int(rng.choice([-1, 0, 1])), bool(rng.integers(0, 2)), bool(rng.integers(0, 2)))
            for _ in range(60)]
    pe = replace(p, events=panel_events(p, rows))
    for params in SWING_FAMILIES[family].grid()[:4]:
        a = SWING_FAMILIES[family](params).signals(p)
        b = SWING_FAMILIES[family](params).signals(pe)
        for f in ("entries", "exits", "stop_pct", "confidence"):
            pd.testing.assert_frame_equal(getattr(a, f), getattr(b, f))
        assert (a.entry_at, a.exit_at, a.max_hold_days, a.equal_weight) == \
               (b.entry_at, b.exit_at, b.max_hold_days, b.equal_weight)


def test_event_frame_and_recent_negative():
    p = make_panel(40, ["A", "B"], seed=2)
    idx = p.close.index
    p = replace(p, events=panel_events(p, [("A", 5, 1, True, True), ("B", 7, 1, False, False),
                                           ("B", 10, -1, False, True), ("Q", 3, 1, True, True)]))
    fa = E.event_frame(p, "bundle_a", min_size=False)
    assert fa.dtypes.eq(bool).all() and fa.index.equals(idx) and list(fa.columns) == ["A", "B"]
    assert fa.sum().sum() == 1 and fa.loc[idx[5], "A"]
    pos = E.event_frame(p, "positive", min_size=False)
    assert pos.sum().sum() == 2 and pos.loc[idx[7], "B"]
    assert E.event_frame(p, "positive", min_size=True).sum().sum() == 1
    neg = E.recent_negative(p, 3)
    assert list(neg.index[neg["B"]]) == list(idx[10:13]) and not neg["A"].any()
    with pytest.raises(ValueError):
        E.event_frame(p, "direction", min_size=False)
    with pytest.raises(ValueError):
        E.event_frame(make_panel(5, ["A"], seed=1), "positive", min_size=False)
    with pytest.raises(ValueError):
        E.recent_negative(make_panel(5, ["A"], seed=1), 20)


# ---------------------------------------------------------------- CLI
def test_cli_events_report(cli_env):
    cal = pipeline.calendar("kr")[0]
    kr_filings(("f1", "A01", nth(cal, 10), BUYBACK), ("f2", "A02", nth(cal, 300), SUPPLY),
               ("f3", "A02", nth(cal, 301), "기업설명회(IR)개최"),
               ("f4", "A03", nth(cal, 302), "기업설명회(IR)개최"),
               ("f5", "A03", nth(cal, 303), "[기재정정]기업설명회(IR)개최"),
               ("f6", "A03", nth(cal, 304), "임원ㆍ주요주주특정증권등소유상황보고서"))
    write_json(targets_path("kr"), {"symbols": ["A01", "A02", "A03"], "targets": 3,
                                    "mapped": 3, "delisted_targets": 0, "delisted_mapped": 0,
                                    "unmapped": []})
    r = run(["events", "--market", "kr", "--unmatched", "5"])
    out = r.output
    assert "buyback" in out and "supply_contract" in out and str(cal[10].year) in out
    assert "기업설명회(IR)개최" in out and "2" in out
    assert out.index("기업설명회(IR)개최") < out.index("임원ㆍ주요주주특정증권등소유상황보고서")
    assert "targets 3" in out


def test_cli_events_without_filings(cli_env):
    r = CliRunner().invoke(app, ["events", "--market", "kr"])
    assert r.exit_code == 1 and "no filings" in r.output
