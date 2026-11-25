from __future__ import annotations

import io
import zipfile
from datetime import date

import httpx
import pytest

from autotrade.disclosure.dart import (DartClient, DartError, DartRateLimited,
                                       make_dart_client_from_env)

KEY = "SECRETKEY123"
D0, D1 = date(2024, 1, 1), date(2024, 12, 31)


def client(handler) -> DartClient:
    return DartClient(KEY, httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)


def test_list_filings_pages_until_total():
    seen = []

    def h(req: httpx.Request):
        q = dict(req.url.params)
        seen.append(q)
        page = int(q["page_no"])
        n = 100 if page == 1 else 50
        rows = [{"rcept_no": f"{page}-{i}"} for i in range(n)]
        return httpx.Response(200, json={"status": "000", "message": "ok", "page_no": page,
                                         "page_count": 100, "total_count": 150, "total_page": 2,
                                         "list": rows})

    rows = client(h).list_filings("00126380", D0, D1, "B")
    assert len(rows) == 150
    assert [s["page_no"] for s in seen] == ["1", "2"]
    assert seen[0]["page_count"] == "100" and seen[0]["pblntf_ty"] == "B"
    assert seen[0]["bgn_de"] == "20240101" and seen[0]["crtfc_key"] == KEY


def test_no_data_status_is_empty():
    c = client(lambda r: httpx.Response(200, json={"status": "013", "message": "no data"}))
    assert c.list_filings("1", D0, D1, "B") == []
    assert c.detail("bsnTpAtsList.json", "1", D0, D1) == []


def test_rate_limit_raises_rate_limited():
    c = client(lambda r: httpx.Response(200, json={"status": "020", "message": "limit"}))
    with pytest.raises(DartRateLimited) as ei:
        c.list_filings("1", D0, D1, "B")
    assert isinstance(ei.value, DartError) and ei.value.status == "020"


def _timeout(req):
    raise httpx.ConnectTimeout("boom " + str(req.url))


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(500, text="x"),
    _timeout,
    lambda r: httpx.Response(200, json={"status": "900", "message": "bad"}),
])
def test_errors_never_contain_key(handler):
    with pytest.raises(DartError) as ei:
        client(handler).list_filings("1", D0, D1, "B")
    for text in (str(ei.value), repr(ei.value)):
        assert KEY not in text and "crtfc_key" not in text and "opendart" not in text
    e = ei.value
    assert e.__cause__ is None and (e.__context__ is None or e.__suppress_context__)


def test_corp_codes_parses_zip():
    xml = ("<?xml version='1.0' encoding='UTF-8'?><result>"
           "<list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name>"
           "<stock_code>005930</stock_code><modify_date>20240101</modify_date></list>"
           "<list><corp_code>00000001</corp_code><corp_name>비상장</corp_name>"
           "<stock_code> </stock_code><modify_date>20240101</modify_date></list>"
           "<list><corp_code>00000002</corp_code><corp_name>무</corp_name>"
           "<stock_code/><modify_date>20240101</modify_date></list></result>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("CORPCODE.xml", xml)
    df = client(lambda r: httpx.Response(200, content=buf.getvalue())).corp_codes()
    assert list(df.columns) == ["corp_code", "stock_code", "corp_name"]
    assert len(df) == 3
    assert df.loc[0, "stock_code"] == "005930" and df.loc[0, "corp_name"] == "삼성전자"
    assert list(df["stock_code"])[1:] == ["", ""]


def test_document_returns_raw_bytes():
    assert client(lambda r: httpx.Response(200, content=b"PK\x03")).document("2024") == b"PK\x03"


def test_missing_key_message(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("")
    monkeypatch.setenv("AUTOTRADE_DOTENV", str(tmp_path / ".env"))
    monkeypatch.delenv("DART_API_KEY", raising=False)
    with pytest.raises(DartError) as ei:
        make_dart_client_from_env()
    assert ei.value.status == "config"
    assert ei.value.message == "DART_API_KEY missing in .env"
