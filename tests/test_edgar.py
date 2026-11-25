"""Task 7: EDGAR client + US collector (no network: MockTransport / fake client)."""
import io
import zipfile
from datetime import date

import httpx
import pandas as pd
import pytest

from autotrade import pipeline
from autotrade.disclosure import collect_us
from autotrade.disclosure.collect_us import collect_events_us
from autotrade.disclosure.edgar import (EdgarClient, EdgarError, EdgarNotFound,
                                        EdgarRateLimited, insider_buy_usd,
                                        make_edgar_client_from_env, parse_acceptance)
from autotrade.disclosure.events import load_events
from autotrade.disclosure.store import (events_root, incomplete_targets, load_filings,
                                        read_json, state_path, targets_path)

UA = "Lab Research secret.person@example.com"
CAL = pd.bdate_range("2024-01-02", "2024-12-31")


def client(handler) -> EdgarClient:
    return EdgarClient(UA, httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)


def cols(rows):
    keys = ["accessionNumber", "filingDate", "acceptanceDateTime", "form", "items"]
    return {k: [r.get(k, "") for r in rows] for k in keys}


def test_parse_acceptance_is_new_york_wall_time():
    # Z notwithstanding: 16:05:22 is Eastern (EST, UTC-5 in March before the DST start)
    assert parse_acceptance("2024-03-01T16:05:22.000Z") == pd.Timestamp("2024-03-01 21:05:22",
                                                                       tz="UTC")
    # EDT in July: UTC-4
    assert parse_acceptance("2024-07-02T09:29:00.000Z") == pd.Timestamp("2024-07-02 13:29",
                                                                       tz="UTC")
    assert str(parse_acceptance("2024-03-01T16:05:22.000Z").tz) == "UTC"
    assert pd.isna(parse_acceptance("")) and pd.isna(parse_acceptance(None))


def test_submissions_merges_older_pages():
    seen = []

    def h(req: httpx.Request):
        seen.append(str(req.url))
        assert req.headers["user-agent"] == UA
        if req.url.path.endswith("CIK0000320193.json"):
            return httpx.Response(200, json={"filings": {
                "recent": cols([{"accessionNumber": "R1", "form": "8-K"}]),
                "files": [{"name": "CIK0000320193-submissions-001.json",
                           "filingFrom": "2010-01-01", "filingTo": "2016-12-31"}]}})
        assert req.url.path.endswith("CIK0000320193-submissions-001.json")
        return httpx.Response(200, json=cols([{"accessionNumber": "O1", "form": "4"},
                                              {"accessionNumber": "O2", "form": "8-K"}]))

    rows = client(h).submissions("320193")
    assert [r["accessionNumber"] for r in rows] == ["R1", "O1", "O2"]
    assert set(rows[0]) == {"accessionNumber", "filingDate", "acceptanceDateTime", "form",
                            "items"}
    assert "data.sec.gov/submissions/CIK0000320193.json" in seen[0]
    # `since` skips pages that end before it
    seen.clear()
    assert len(client(h).submissions("320193", since=date(2020, 1, 1))) == 1
    assert len(seen) == 1


def _timeout(req):
    raise httpx.ConnectTimeout("boom " + str(req.url))


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(500, text="x"),
    lambda r: httpx.Response(200, text="not json"),
    _timeout,
    lambda r: httpx.Response(429),
])
def test_user_agent_header_sent_and_never_in_errors(handler):
    sent = []

    def h(req):
        sent.append(req.headers["user-agent"])
        return handler(req)

    with pytest.raises(EdgarError) as ei:
        client(h).tickers()
    assert sent == [UA]
    for text in (str(ei.value), repr(ei.value)):
        assert "secret.person" not in text and "example.com" not in text
        assert "sec.gov" not in text
    e = ei.value
    assert e.__cause__ is None and (e.__context__ is None or e.__suppress_context__)


def test_rate_limit_and_missing_insider_set():
    with pytest.raises(EdgarRateLimited):
        client(lambda r: httpx.Response(429)).tickers()
    with pytest.raises(EdgarNotFound):
        client(lambda r: httpx.Response(404)).insider_quarter(2024, 1)


def test_tickers_pads_cik():
    c = client(lambda r: httpx.Response(200, json={
        "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple"}}))
    t = c.tickers()
    assert list(t["ticker"]) == ["AAPL"] and list(t["cik"]) == ["0000320193"]


def test_missing_user_agent_refuses(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setenv("AUTOTRADE_DOTENV", str(tmp_path / "none.env"))
    with pytest.raises(pipeline.LabError, match="SEC_USER_AGENT missing in .env"):
        make_edgar_client_from_env()


def insider_zip(rows) -> bytes:
    head = ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "TRANS_CODE", "TRANS_SHARES",
            "TRANS_PRICEPERSHARE"]
    body = "\n".join("\t".join([a, "1", c, s, p]) for a, c, s, p in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("NONDERIV_TRANS.tsv", "\t".join(head) + "\n" + body + "\n")
        z.writestr("SUBMISSION.tsv", "x\n")
    return buf.getvalue()


def test_insider_buy_amount_sums_code_p_only():
    rows = [("A1", "P", "100", "10.5"), ("A1", "P", "50", "20"), ("A1", "S", "1000", "99"),
            ("A1", "A", "10", "1"), ("A2", "S", "5", "5"), ("A3", "P", "10", "")]
    c = client(lambda r: httpx.Response(200, content=insider_zip(rows)))
    s = insider_buy_usd(c.insider_quarter(2024, 1))
    assert s.to_dict() == {"A1": 100 * 10.5 + 50 * 20.0}


def test_insider_layout_error_is_edgar_error():
    c = client(lambda r: httpx.Response(200, content=b"not a zip"))
    with pytest.raises(EdgarError, match="insider layout"):
        c.insider_quarter(2024, 1)


# ---------------------------------------------------------------- collector
TODAY = date(2024, 8, 20)    # 2024q2 data set is the newest available -> horizon 2024-06-30


class FakeEdgar:
    def __init__(self, subs=None, avail=((2024, 1), (2024, 2)), hook=None):
        self.subs = subs or {}
        self.avail = set(avail)
        self.calls = []
        self.insider_calls = []
        self.ticker_calls = 0
        self.hook = hook

    def tickers(self):
        self.ticker_calls += 1
        return pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOG", "GOOGL"],
                             "cik": ["0000000001", "0000000002", "0000000003", "0000000003"]})

    def submissions(self, cik, since=None):
        self.calls.append(cik)
        if self.hook:
            self.hook(cik)
        return self.subs.get(cik, [])

    def insider_quarter(self, y, q):
        self.insider_calls.append((y, q))
        if (y, q) not in self.avail:
            raise EdgarNotFound("404", "not available")
        return pd.DataFrame({"accession": ["A-4"], "code": ["P"], "shares": [200.0],
                             "price": [10.0]})


def sub(acc, filed, accepted, form, items=""):
    return {"accessionNumber": acc, "filingDate": filed, "acceptanceDateTime": accepted,
            "form": form, "items": items}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "data"))
    return tmp_path


def fake():
    return FakeEdgar({"0000000001": [
        sub("A-8k", "2024-02-01", "2024-02-01T08:30:00.000Z", "8-K", "1.01, 9.01"),
        sub("A-8ka", "2024-02-02", "2024-02-02T08:30:00.000Z", "8-K/A", "1.01"),
        sub("A-4", "2024-03-04", "2024-03-04T18:00:00.000Z", "4"),
        sub("A-4b", "2024-03-05", "2024-03-05T18:00:00.000Z", "4"),
        sub("A-10q", "2024-05-01", "2024-05-01T08:30:00.000Z", "10-Q"),
        sub("A-old", "2014-12-31", "2014-12-31T08:30:00.000Z", "8-K", "1.01"),
        sub("A-424", "2024-04-01", "2024-04-01T10:00:00.000Z", "424B5"),
        sub("A-late", "2024-07-15", "2024-07-15T10:00:00.000Z", "8-K", "1.01"),   # 8-K: no horizon
        sub("A-4late", "2024-07-16", "2024-07-16T18:00:00.000Z", "4"),            # beyond horizon
    ]})


def test_us_collect_rows_state_and_targets(env):
    c = fake()
    out = collect_events_us(c, ["AAPL", "MSFT", "GOOG", "GOOGL", "ZZZZ"], today=TODAY)
    allrows = load_filings("us")
    df = allrows[allrows["symbol"] == "AAPL"].set_index("filing_id")
    assert set(df.loc[["A-8k"]].index) == {"A-8k"}
    assert set(df.index) == {"A-8k", "A-8ka", "A-4", "A-4b", "A-424", "A-late"}
    assert df.loc["A-8k", "items"] == "1.01,9.01"
    assert df.loc["A-424", "items"] == ""
    assert df.loc["A-8k", "accepted_at"] == pd.Timestamp("2024-02-01 13:30", tz="UTC")
    assert df.loc["A-8k", "filed_date"] == date(2024, 2, 1)
    assert df.loc["A-4", "insider_usd"] == 2000.0
    assert pd.isna(df.loc["A-4b", "insider_usd"]) and pd.isna(df.loc["A-8k", "insider_usd"])
    assert str(df["accepted_at"].dt.tz) == "UTC"
    t = read_json(targets_path("us"), {})
    assert t["targets"] == 5 and t["mapped"] == 4 and t["unmapped"] == ["ZZZZ"]
    assert t["delisted_targets"] == 5 and t["delisted_mapped"] == 4     # coverage = mapped/all
    st = read_json(state_path("us"), {})
    assert st["AAPL"] == {"done_through": "2024-08-20", "insider_through": "2024-06-30",
                          "complete": True}
    assert out["complete"] == 4 and out["horizon"] == "2024-06-30" and not out["rate_limited"]
    assert incomplete_targets("us") == []
    # GOOG/GOOGL share a CIK: fetched once per run, rows for each ticker (ruling R16)
    assert c.calls.count("0000000003") == 1


def test_us_collect_resume_and_lock(env):
    c = fake()
    collect_events_us(c, ["AAPL", "MSFT"], today=TODAY)
    n = len(load_filings("us"))
    c2 = fake()
    collect_events_us(c2, ["AAPL", "MSFT"], today=TODAY)          # complete: nothing refetched
    assert c2.calls == [] and len(load_filings("us")) == n
    # a later quarter set appears: symbols continue from done_through, no duplicates
    c3 = fake()
    c3.avail.add((2024, 3))
    collect_events_us(c3, ["AAPL"], today=date(2024, 10, 20))
    assert c3.calls == ["0000000001"]
    ids = list(load_filings("us")["filing_id"])
    assert "A-4late" in ids and len(ids) == len(set(ids))
    # second collector refused
    with pipeline._exclusive_lock(events_root("us") / ".collect.lock", "x"):
        assert pipeline.events_running("us") is True
        with pytest.raises(pipeline.LabError, match="lab collect-events --market us"):
            pipeline.collect_events("us")


def test_us_rate_limit_stops_cleanly_and_limit_describes_all(env):
    def hook(cik):
        if cik == "0000000002":
            raise EdgarRateLimited("429", "rate limited")

    c = fake()
    c.hook = hook
    out = collect_events_us(c, ["AAPL", "MSFT"], today=TODAY)
    assert out["rate_limited"] is True and out["complete"] == 1
    assert incomplete_targets("us") == ["MSFT"]
    assert len(load_filings("us")) == 6
    out = collect_events_us(fake(), ["AAPL", "MSFT"], today=TODAY, limit=1)
    assert read_json(targets_path("us"), {})["targets"] == 2


def test_no_insider_set_available_errors(env):
    with pytest.raises(EdgarError, match="insider"):
        collect_events_us(FakeEdgar(avail=()), ["AAPL"], today=TODAY)


def test_insider_quarter_cached_on_disk(env):
    c = fake()
    collect_events_us(c, ["AAPL"], today=TODAY)
    assert (events_root("us") / "insider" / "2024q2.parquet").exists()
    c2 = fake()
    collect_events_us(c2, ["AAPL"], today=date(2024, 8, 21))   # complete, but horizon probe
    assert c2.insider_calls == [(2024, 3)]        # only the unpublished quarter is probed


def test_collector_output_feeds_load_events(env):
    """Shapes cannot drift: collector rows -> classify -> signal dates -> sizes."""
    c = FakeEdgar({"0000000001": [
        sub("A-8k", "2024-02-01", "2024-02-01T08:30:00.000Z", "8-K", "1.01,9.01"),    # pre-open
        sub("A-301", "2024-02-06", "2024-02-06T10:00:00.000Z", "8-K", "3.01"),
        sub("A-4", "2024-03-04", "2024-03-04T18:00:00.000Z", "4"),
    ]})
    c.insider_quarter = lambda y, q: pd.DataFrame(
        {"accession": ["A-4"], "code": ["P"], "shares": [20000.0], "price": [10.0]})
    collect_events_us(c, ["AAPL"], today=TODAY)
    ev = load_events("us", CAL).set_index("filing_id")
    assert ev.loc["A-8k", "event"] == "8k_101" and ev.loc["A-8k", "direction"] == 1
    assert ev.loc["A-8k", "signal_date"] == pd.Timestamp("2024-01-31")   # before 09:30 NY
    assert ev.loc["A-301", "event"] == "8k_301" and ev.loc["A-301", "direction"] == -1
    assert ev.loc["A-301", "signal_date"] == pd.Timestamp("2024-02-06")
    assert ev.loc["A-4", "event"] == "insider_buy" and ev.loc["A-4", "size"] == 200000.0
    assert bool(ev.loc["A-4", "size_ok"]) is True


def shared_fake(**kw):
    c = FakeEdgar({"0000000003": [
        sub("G-8k", "2024-02-01", "2024-02-01T08:30:00.000Z", "8-K", "1.01")]}, **kw)
    return c


def test_shared_cik_tickers_each_get_rows_and_events(env):
    c = shared_fake()
    collect_events_us(c, ["GOOG", "GOOGL"], today=TODAY)
    assert c.calls == ["0000000003"]
    df = load_filings("us")
    assert sorted(df["symbol"]) == ["GOOG", "GOOGL"]
    ev = load_events("us", CAL)
    assert sorted(ev["symbol"]) == ["GOOG", "GOOGL"]
    assert list(ev["event"]) == ["8k_101", "8k_101"]


def test_form4_after_horizon_stored_only_when_quarter_published(env):
    c = fake()
    collect_events_us(c, ["AAPL"], today=TODAY)
    ids = set(load_filings("us")["filing_id"])
    assert "A-late" in ids and "A-4late" not in ids        # 8-K stored now, Form 4 later
    c3 = fake()
    c3.avail.add((2024, 3))
    orig = c3.insider_quarter

    def iq(y, q):
        orig(y, q)                      # raises NotFound for unpublished quarters
        return pd.DataFrame({"accession": ["A-4late"], "code": ["P"], "shares": [10.0],
                             "price": [3.0]})
    c3.insider_quarter = iq
    collect_events_us(c3, ["AAPL"], today=date(2024, 10, 20))
    df = load_filings("us").set_index("filing_id")
    assert df.loc["A-4late", "insider_usd"] == 30.0
    assert read_json(state_path("us"), {})["AAPL"]["insider_through"] == "2024-09-30"


def test_insider_403_not_complete_and_no_nan_row(env):
    class Forbidden(FakeEdgar):
        def insider_quarter(self, y, q):
            if (y, q) == (2024, 1):
                raise EdgarError("http", "HTTP 403")
            return super().insider_quarter(y, q)

    c = Forbidden({"0000000001": [
        sub("A-4", "2024-03-04", "2024-03-04T18:00:00.000Z", "4"),
        sub("A-8k", "2024-02-01", "2024-02-01T08:30:00.000Z", "8-K", "1.01")]})
    with pytest.raises(EdgarError, match="403"):
        collect_events_us(c, ["AAPL"], today=TODAY)
    assert "AAPL" not in read_json(state_path("us"), {})
    assert incomplete_targets("us") == ["AAPL"]
    df = load_filings("us")
    assert df.empty or "A-4" not in set(df["filing_id"])


def test_unpublished_newest_quarter_backs_off_and_403_is_not_not_found():
    c = client(lambda r: httpx.Response(403))
    with pytest.raises(EdgarError) as ei:
        c.insider_quarter(2024, 1)
    assert not isinstance(ei.value, EdgarNotFound)
    out = FakeEdgar(avail=((2024, 1),))
    assert collect_us._Insider(out).horizon(date(2024, 8, 1)) == date(2024, 3, 31)
    assert out.insider_calls == [(2024, 3), (2024, 2), (2024, 1)]
    ins = collect_us._Insider(out)
    ins.get(2024, 3); ins.get(2024, 3)
    assert out.insider_calls.count((2024, 3)) == 2         # once per _Insider (per run)
