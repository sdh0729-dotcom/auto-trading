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
