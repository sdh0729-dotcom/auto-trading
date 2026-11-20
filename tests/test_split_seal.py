import json
from datetime import date, timedelta

import pandas as pd
import pytest

from autotrade.validate.seal import (SealLimitError, SealRegistry, SealReuseError, SealTicket,
                                     SealViolation)
from autotrade.validate.split import Periods, split_periods, walk_forward_windows
from tests.fixtures import make_panel

DATES = pd.bdate_range("2016-01-01", periods=1000, name="date")


def _count(dates, period):
    s, e = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    return int(((dates >= s) & (dates <= e)).sum())


def test_split_fractions_and_order():
    p = split_periods(DATES, (0.6, 0.2, 0.2))
    assert isinstance(p, Periods)
    for part in (p.train, p.valid, p.sealed):
        assert all(type(d) is date for d in part)
        assert part[0] <= part[1]
    assert p.train[0] == DATES[0].date() and p.sealed[1] == DATES[-1].date()
    assert p.train[1] < p.valid[0] and p.valid[1] < p.sealed[0]
    # contiguous: each part starts on the trading day right after the previous one ends
    pos = {d.date(): i for i, d in enumerate(DATES)}
    assert pos[p.valid[0]] == pos[p.train[1]] + 1
    assert pos[p.sealed[0]] == pos[p.valid[1]] + 1
    counts = [_count(DATES, x) for x in (p.train, p.valid, p.sealed)]
    assert sum(counts) == len(DATES)
    for n, f in zip(counts, (0.6, 0.2, 0.2)):
        assert abs(n - f * len(DATES)) <= 1


def test_split_odd_length_within_one_day():
    dates = DATES[:997]
    p = split_periods(dates)
    counts = [_count(dates, x) for x in (p.train, p.valid, p.sealed)]
    assert sum(counts) == 997
    for n, f in zip(counts, (0.6, 0.2, 0.2)):
        assert abs(n - f * 997) <= 1


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        split_periods(DATES, (0.6, 0.2, 0.1))
    with pytest.raises(ValueError):
        split_periods(DATES, (0.8, 0.2, 0.0))
    with pytest.raises(ValueError):
        split_periods(DATES[::-1])
    with pytest.raises(ValueError):
        split_periods(DATES[:2])


def test_walk_forward_windows_no_overlap():
    start, end = date(2016, 1, 1), date(2021, 12, 31)
    w = walk_forward_windows(start, end, n=4)
    assert len(w) == 4
    one = timedelta(days=1)
    for i, (fs, fe, ts, te) in enumerate(w):
        assert all(type(d) is date for d in (fs, fe, ts, te))
        assert fs == start                      # expanding: fit start fixed
        assert fe < ts and ts == fe + one       # test follows fit immediately
        assert ts <= te
        if i:
            assert ts == w[i - 1][3] + one      # test segments contiguous, no overlap
            assert fe == w[i - 1][3]            # fit expands through the previous test
    assert w[-1][3] == end
    total = (end - start).days + 1
    seg = [(w[0][1] - start).days + 1] + [(te - ts).days + 1 for _, _, ts, te in w]
    assert sum(seg) == total and max(seg) - min(seg) <= 1


def test_walk_forward_rejects_bad_input():
    with pytest.raises(ValueError):
        walk_forward_windows(date(2020, 1, 1), date(2020, 1, 3), n=4)
    with pytest.raises(ValueError):
        walk_forward_windows(date(2020, 1, 1), date(2021, 1, 1), n=0)


def test_seal_reuse_and_limit(tmp_path):
    reg = SealRegistry(tmp_path / "s.json")
    for r in ["a", "b", "c"]:
        reg.open("swing", "kr", r)
    with pytest.raises(SealReuseError):
        reg.open("swing", "kr", "a")
    with pytest.raises(SealLimitError):
        reg.open("swing", "kr", "d")
    reg.open("swing", "us", "d")   # 다른 시장은 별도 한도
    assert issubclass(SealReuseError, SealViolation) and issubclass(SealLimitError, SealViolation)


def test_seal_persisted_across_instances(tmp_path):
    path = tmp_path / "s.json"
    t = SealRegistry(path).open("swing", "kr", "a")
    assert isinstance(t, SealTicket)
    assert (t.pocket, t.market, t.rule_id) == ("swing", "kr", "a") and t.opened_at
    with pytest.raises(Exception):
        t.rule_id = "b"  # frozen
    SealRegistry(path).open("swing", "kr", "b")
    reg = SealRegistry(path)
    with pytest.raises(SealReuseError):
        reg.open("swing", "kr", "a")
    reg.open("swing", "kr", "c")
    with pytest.raises(SealLimitError):
        reg.open("swing", "kr", "d")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["tickets"]
    assert [r["rule_id"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["opened_at"] == t.opened_at
    assert [p.name for p in tmp_path.iterdir()] == ["s.json"]  # no temp files left behind


def test_load_sealed_requires_issued_ticket(tmp_path):
    panel = make_panel(300, ["A", "B"], seed=3)
    periods = split_periods(panel.close.index)
    reg = SealRegistry(tmp_path / "s.json")
    ticket = reg.open("swing", "kr", "a")

    sealed = SealRegistry(tmp_path / "s.json").load_sealed(panel, periods, ticket)
    expected = panel.slice(*periods.sealed)
    pd.testing.assert_frame_equal(sealed.close, expected.close)
    assert sealed.close.index[0].date() == periods.sealed[0]
    assert sealed.close.index[-1].date() == periods.sealed[1]

    forged = SealTicket(ticket.pocket, ticket.market, ticket.rule_id, ticket.opened_at)
    with pytest.raises(SealViolation):
        reg.load_sealed(panel, periods, forged)
    with pytest.raises(SealViolation):
        SealRegistry(tmp_path / "other.json").load_sealed(panel, periods, ticket)
    with pytest.raises(SealViolation):
        reg.load_sealed(panel, periods, None)


# --- persisted periods (C1) ------------------------------------------------------------------
from autotrade.validate.periods import (MIN_NEW_CYCLE_DAYS, PeriodError, PeriodRecord,  # noqa: E402
                                        load_record, new_cycle, new_record, save_record)


def test_period_record_round_trip(tmp_path):
    rec = new_record("swing", "kr", DATES, (0.6, 0.2, 0.2))
    assert rec.periods == split_periods(DATES, (0.6, 0.2, 0.2))
    assert rec.data_cutoff == DATES[-1].date() and rec.cycle == 1 and rec.validate_runs == 0
    path = tmp_path / "periods-swing-kr.json"
    save_record(path, rec)
    assert load_record(path) == rec
    assert load_record(tmp_path / "missing.json") is None
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["sealed"] == [rec.periods.sealed[0].isoformat(), rec.periods.sealed[1].isoformat()]
    assert d["data_cutoff"] == DATES[-1].date().isoformat() and d["created_at"]
    assert [f.name for f in tmp_path.iterdir()] == ["periods-swing-kr.json"]


def test_new_cycle_needs_63_new_trading_days():
    rec = new_record("swing", "kr", DATES, (0.6, 0.2, 0.2))
    more = pd.bdate_range(DATES[0], periods=len(DATES) + MIN_NEW_CYCLE_DAYS - 1, name="date")
    with pytest.raises(PeriodError, match="63"):
        new_cycle(rec, more)
    more = pd.bdate_range(DATES[0], periods=len(DATES) + MIN_NEW_CYCLE_DAYS, name="date")
    nxt = new_cycle(rec, more)
    assert nxt.periods.train == rec.periods.train and nxt.periods.valid == rec.periods.valid
    assert nxt.periods.sealed == (more[len(DATES)].date(), more[-1].date())
    assert nxt.data_cutoff == more[-1].date() and nxt.cycle == 2
    assert nxt.history == ((rec.periods.sealed[0].isoformat(), rec.periods.sealed[1].isoformat(),
                            rec.data_cutoff.isoformat()),)


def test_seal_ticket_records_period_cutoff_and_symbols(tmp_path):
    reg = SealRegistry(tmp_path / "s.json")
    t = reg.open("swing", "kr", "a", sealed=("2024-01-02", "2025-12-30"),
                 data_cutoff="2025-12-30", n_symbols=812, cycle=1)
    row = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["tickets"][0]
    assert (row["sealed_start"], row["sealed_end"], row["data_cutoff"], row["n_symbols"],
            row["cycle"]) == ("2024-01-02", "2025-12-30", "2025-12-30", 812, 1)
    assert reg.tickets("swing", "kr") == [t]


def test_new_cycle_reopens_only_previous_final_candidates(tmp_path):
    reg = SealRegistry(tmp_path / "s.json")
    for r in ["a", "b"]:
        reg.open("swing", "kr", r, sealed=("2024-01-02", "2025-12-30"), cycle=1)
    with pytest.raises(SealReuseError):
        reg.open("swing", "kr", "a", sealed=("2024-01-02", "2025-12-30"), cycle=1)
    prev = {t.rule_id for t in reg.tickets("swing", "kr") if t.cycle == 1}
    reg.open("swing", "kr", "a", sealed=("2026-01-02", "2026-04-30"), cycle=2,
             allowed_rules=prev)                                 # re-validation of a
    with pytest.raises(SealViolation, match="previous final candidates"):
        reg.open("swing", "kr", "z", sealed=("2026-01-02", "2026-04-30"), cycle=2,
                 allowed_rules=prev)
    with pytest.raises(SealReuseError):
        reg.open("swing", "kr", "a", sealed=("2026-01-02", "2026-04-30"), cycle=2,
                 allowed_rules=prev)


def test_load_sealed_rejects_ticket_of_another_sealed_period(tmp_path):
    panel = make_panel(300, ["A", "B"], seed=3)
    periods = split_periods(panel.close.index)
    reg = SealRegistry(tmp_path / "s.json")
    ticket = reg.open("swing", "kr", "a", sealed=("2016-01-04", "2016-02-01"))
    with pytest.raises(SealViolation, match="sealed period"):
        reg.load_sealed(panel, periods, ticket)
