import io
import math
import zipfile
from datetime import date

import pandas as pd
import pytest

from autotrade.disclosure import sizes_kr
from autotrade.disclosure.dart import DartError, DartRateLimited
from autotrade.disclosure.sizes_kr import (collect_sizes_kr, parse_supply_ratio, size_patterns,
                                           size_targets)
from autotrade.disclosure.store import FILING_COLUMNS, append_filings, load_sizes


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOTRADE_DATA", str(tmp_path / "data"))
    return tmp_path


def zipped(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("doc.xml", text.encode("utf-8"))
    return buf.getvalue()


def filings(*rows):
    out = []
    for fid, sym, corp, title, d in rows:
        out.append({"filing_id": fid, "symbol": sym, "corp_code": corp,
                    "filed_date": d, "title": title, "kind": "B", "filer": "x", "remark": ""})
    return pd.DataFrame(out, columns=FILING_COLUMNS)


BUY = "주요사항보고서(자기주식취득결정)"
SUP = "단일판매ㆍ공급계약체결"
RIGHTS = "주요사항보고서(유상증자결정)"


class Fake:
    def __init__(self, detail=None, docs=None, fail=None):
        self.det, self.docs, self.fail = detail or {}, docs or {}, fail or {}
        self.calls = []

    def detail(self, endpoint, corp_code, start, end):
        self.calls.append(("detail", endpoint, corp_code, start, end))
        if "detail" in self.fail:
            raise self.fail["detail"]
        return self.det.get((endpoint, corp_code), [])

    def document(self, rcept_no):
        self.calls.append(("document", rcept_no))
        if rcept_no in self.fail:
            raise self.fail[rcept_no]
        return self.docs[rcept_no]


def test_parse_supply_ratio_reads_percent():
    html = "<tr><td>최근 매출액(원)</td><td>1,000</td></tr><tr><td>매 출 액\n대비(%)</td><td>12.50</td></tr>"
    assert parse_supply_ratio(zipped(html)) == pytest.approx(0.125)
    assert parse_supply_ratio(zipped("<p>매출액대비(%)</p><p> 12.5 </p>")) == pytest.approx(0.125)
    assert parse_supply_ratio(zipped("<td>매출액<br/>대비(%)</td><td>-3</td>")) == pytest.approx(-0.03)


def test_parse_supply_ratio_missing_returns_none():
    assert parse_supply_ratio(zipped("<p>계약금액 100</p>")) is None
    assert parse_supply_ratio(zipped("<p>매출액 대비(%)</p><p>해당없음</p>")) is None
    assert parse_supply_ratio(b"not a zip") is None


def test_size_patterns_and_targets_strip_corrections():
    assert set(size_patterns("kr")) == {"buyback_krw", "supply_revenue_ratio", "rights_dilution"}
    f = filings(("1", "A", "C1", "[기재정정]" + SUP, date(2024, 1, 2)),
                ("2", "A", "C1", "[첨부정정][기재정정] " + BUY, date(2024, 1, 3)),
                ("3", "A", "C1", RIGHTS, date(2024, 1, 4)),
                ("4", "A", "C1", "주요사항보고서(무상증자결정)", date(2024, 1, 5)),
                ("5", "A", "C1", "[철회]" + BUY, date(2024, 1, 6)))
    t = size_targets(f)
    assert dict(zip(t["filing_id"], t["measure"])) == {
        "2": "buyback_krw", "1": "supply_revenue_ratio", "3": "rights_dilution"}
    assert list(t["measure"]) == ["buyback_krw", "supply_revenue_ratio", "rights_dilution"]


def test_rights_dilution_is_new_over_existing_shares(env):
    append_filings("kr", filings(("R1", "A", "C1", RIGHTS, date(2024, 1, 4))))
    c = Fake(detail={("piicDecsn.json", "C1"): [
        {"rcept_no": "R1", "nstk_ostk_cnt": "1,000", "nstk_estk_cnt": "-",
         "bfic_tisstk_ostk": "8,000", "bfic_tisstk_estk": "2,000"}]})
    collect_sizes_kr(c)
    s = load_sizes("kr")
    assert s.iloc[0]["measure"] == "rights_dilution"
    assert s.iloc[0]["value"] == pytest.approx(0.1)


def test_buyback_sums_common_and_other(env):
    append_filings("kr", filings(("B1", "A", "C1", BUY, date(2024, 1, 4)),
                                 ("B2", "A", "C1", BUY, date(2024, 3, 4))))
    c = Fake(detail={("tsstkAqDecsn.json", "C1"): [
        {"rcept_no": "B1", "aqpln_prc_ostk": "1,000,000", "aqpln_prc_estk": "-"},
        {"rcept_no": "B2", "aqpln_prc_ostk": "100", "aqpln_prc_estk": "50"}]})
    collect_sizes_kr(c)
    s = load_sizes("kr").set_index("filing_id")["value"]
    assert s["B1"] == 1_000_000 and s["B2"] == 150
    # one span call per corp
    assert c.calls == [("detail", "tsstkAqDecsn.json", "C1", date(2024, 1, 4), date(2024, 3, 4))]


def test_size_pass_records_nan_on_failure_and_never_refetches(env):
    append_filings("kr", filings(("B1", "A", "C1", BUY, date(2024, 1, 4)),
                                 ("S1", "A", "C1", SUP, date(2024, 1, 5)),
                                 ("S2", "A", "C1", SUP, date(2024, 1, 6)),
                                 ("R1", "A", "C1", RIGHTS, date(2024, 1, 7))))
    c = Fake(docs={"S1": zipped("<p>nothing</p>")}, fail={"S2": DartError("800", "x")})
    out = collect_sizes_kr(c)
    s = load_sizes("kr").set_index("filing_id")["value"]
    assert len(s) == 4 and s.isna().all() and out["rate_limited"] is False
    c2 = Fake()
    collect_sizes_kr(c2)
    assert c2.calls == []


def test_size_pass_resumes_after_rate_limit(env):
    append_filings("kr", filings(("S1", "A", "C1", SUP, date(2024, 1, 5)),
                                 ("S2", "A", "C1", SUP, date(2024, 1, 6)),
                                 ("R1", "A", "C1", RIGHTS, date(2024, 1, 7))))
    docs = {"S1": zipped("매출액대비(%) 20"), "S2": zipped("매출액대비(%) 5")}
    out = collect_sizes_kr(Fake(docs=docs, fail={"S2": DartRateLimited("020", "x")}))
    assert out["rate_limited"] is True
    assert dict(zip(load_sizes("kr")["filing_id"], load_sizes("kr")["value"])) == {"S1": 0.2}
    c = Fake(docs=docs, detail={("piicDecsn.json", "C1"): []})
    out = collect_sizes_kr(c)
    assert out["rate_limited"] is False
    assert [x for x in c.calls if x[0] == "document"] == [("document", "S2")]
    s = load_sizes("kr").set_index("filing_id")["value"]
    assert s["S1"] == pytest.approx(0.2) and s["S2"] == pytest.approx(0.05)
    assert math.isnan(s["R1"])
