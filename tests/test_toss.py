from __future__ import annotations

import json
import os
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from autotrade.toss.auth import TossAuth
from autotrade.toss.client import BASE, TossApiError, TossClient, make_client_from_env
from autotrade.toss.ratelimit import RateLimiter

OK_EMPTY = {"result": {"candles": [], "nextBefore": None}}


@pytest.fixture
def client(tmp_path, respx_mock):
    respx_mock.post(f"{BASE}/oauth2/token").respond(
        json={"access_token": "tok", "token_type": "Bearer", "expires_in": 86400})
    http = httpx.Client()
    auth = TossAuth("id", "sec", tmp_path / "t.json", http)
    fast = RateLimiter({g: 1e6 for g in
                        ["MARKET_DATA_CHART", "STOCK_ALL", "ORDER_INFO", "MARKET_INFO", "AUTH"]})
    return TossClient(auth, http, fast, account_seq=7)


def test_token_cached_across_instances(tmp_path, respx_mock):
    route = respx_mock.post(f"{BASE}/oauth2/token").respond(
        json={"access_token": "t1", "token_type": "Bearer", "expires_in": 86400})
    p = tmp_path / "toss.token.json"
    assert TossAuth("id", "sec", p, httpx.Client()).token() == "t1"
    assert TossAuth("id", "sec", p, httpx.Client()).token() == "t1"
    assert route.call_count == 1
    assert oct(p.stat().st_mode & 0o777) == "0o600"
    body = route.calls.last.request.content.decode()
    assert parse_qs(body)["grant_type"] == ["client_credentials"]
    assert set(json.loads(p.read_text())) == {"access_token", "expires_at"}
    assert [f.name for f in tmp_path.iterdir()] == ["toss.token.json"]  # no temp leftovers


def test_token_refreshed_near_expiry(tmp_path, respx_mock):
    p = tmp_path / "toss.token.json"
    p.write_text(json.dumps({"access_token": "old", "expires_at": time.time() + 200}))
    route = respx_mock.post(f"{BASE}/oauth2/token").respond(
        json={"access_token": "new", "token_type": "Bearer", "expires_in": 86400})
    assert TossAuth("id", "sec", p, httpx.Client()).token() == "new"
    assert route.call_count == 1
    assert json.loads(p.read_text())["access_token"] == "new"


def test_token_reused_when_far_from_expiry(tmp_path, respx_mock):
    p = tmp_path / "toss.token.json"
    p.write_text(json.dumps({"access_token": "old", "expires_at": time.time() + 3600}))
    route = respx_mock.post(f"{BASE}/oauth2/token").respond(json={})
    assert TossAuth("id", "sec", p, httpx.Client()).token() == "old"
    assert route.call_count == 0


def test_token_error_oauth2_format(tmp_path, respx_mock):
    respx_mock.post(f"{BASE}/oauth2/token").respond(
        401, json={"error": "invalid_client", "error_description": "bad"})
    with pytest.raises(TossApiError) as e:
        TossAuth("id", "sec-value", tmp_path / "t.json", httpx.Client()).token()
    assert e.value.status == 401 and e.value.code == "invalid_client"
    assert "sec-value" not in str(e.value)


def test_candles_parsed_and_sorted(respx_mock, client):
    route = respx_mock.get(f"{BASE}/api/v1/candles").respond(json={"result": {"candles": [
        {"timestamp": "2026-03-25T00:00:00+09:00", "openPrice": "71600", "highPrice": "72300", "lowPrice": "71500", "closePrice": "72000", "volume": "3521000", "currency": "KRW"},
        {"timestamp": "2026-03-24T00:00:00+09:00", "openPrice": "71200", "highPrice": "71800", "lowPrice": "71000", "closePrice": "71600", "volume": "2984000", "currency": "KRW"}],
        "nextBefore": "2026-03-24T00:00:00+09:00"}})
    page = client.get_candles("005930", "1d")
    assert list(page.df["close"]) == [71600.0, 72000.0]
    assert list(page.df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert page.df["timestamp"].is_monotonic_increasing
    assert page.df["timestamp"].dt.tz is not None
    assert page.df["volume"].dtype == float
    assert page.next_before.isoformat() == "2026-03-24T00:00:00+09:00"
    req = route.calls.last.request
    q = parse_qs(urlparse(str(req.url)).query)
    assert q == {"symbol": ["005930"], "interval": ["1d"], "count": ["200"], "adjusted": ["true"]}
    assert req.headers["Authorization"] == "Bearer tok"


def test_candles_empty(respx_mock, client):
    respx_mock.get(f"{BASE}/api/v1/candles").respond(json=OK_EMPTY)
    page = client.get_candles("005930", "1d")
    assert page.df.empty and page.next_before is None
    assert list(page.df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]


def test_before_plus_sign_encoded(respx_mock, client):
    route = respx_mock.get(f"{BASE}/api/v1/candles").respond(json=OK_EMPTY)
    client.get_candles("005930", "1d", before=datetime.fromisoformat("2026-03-25T09:00:00+09:00"))
    url = str(route.calls.last.request.url)
    assert "before=2026-03-25T09%3A00%3A00%2B09%3A00" in url
    assert "+09" not in url


def test_adjusted_false(respx_mock, client):
    route = respx_mock.get(f"{BASE}/api/v1/candles").respond(json=OK_EMPTY)
    client.get_candles("005930", "1d", adjusted=False)
    assert "adjusted=false" in str(route.calls.last.request.url)


def test_429_retry_after(respx_mock, client, monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    respx_mock.get(f"{BASE}/api/v1/candles").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "2"}, json={"error": {"requestId": "r", "code": "rate-limited", "message": ""}}),
        httpx.Response(200, json=OK_EMPTY)])
    client.get_candles("005930", "1d")
    assert 2 in sleeps


def test_429_gives_up_after_5(respx_mock, client, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    route = respx_mock.get(f"{BASE}/api/v1/candles").respond(
        429, headers={"Retry-After": "1"}, json={"error": {"requestId": "r", "code": "rate-limited", "message": ""}})
    with pytest.raises(TossApiError) as e:
        client.get_candles("005930", "1d")
    assert e.value.status == 429 and route.call_count == 5


def test_error_has_no_secret(respx_mock, client):
    respx_mock.get(f"{BASE}/api/v1/candles").respond(404, json={"error": {"requestId": "r1", "code": "stock-not-found", "message": ""}})
    with pytest.raises(TossApiError) as e:
        client.get_candles("XXXX", "1d")
    assert e.value.code == "stock-not-found" and e.value.request_id == "r1" and e.value.status == 404
    assert "sec" not in str(e.value) and "tok" not in str(e.value)


def test_list_stocks(respx_mock, client):
    route = respx_mock.get(f"{BASE}/api/v1/stocks/all").respond(json={"result": [
        {"symbol": "005930", "name": "삼성전자", "securityType": "STOCK", "isCommonShare": True, "isinCode": "KR7005930003"},
        {"symbol": "069500", "name": "KODEX 200", "securityType": "ETF", "isCommonShare": True, "isinCode": "KR7069500007"}]})
    df = client.list_stocks("KOSPI", security_type="ETF")
    assert list(df.columns) == ["symbol", "name", "security_type", "is_common_share", "isin"]
    assert list(df["symbol"]) == ["005930", "069500"]
    assert list(df["isin"])[1] == "KR7069500007"
    q = parse_qs(urlparse(str(route.calls.last.request.url)).query)
    assert q == {"market": ["KOSPI"], "status": ["ACTIVE"], "securityType": ["ETF"]}


def test_commissions_and_exchange_rate(respx_mock, client):
    c = respx_mock.get(f"{BASE}/api/v1/commissions").respond(json={"result": [
        {"marketCountry": "KR", "commissionRate": "0.00015", "startDate": "2026-01-01", "endDate": None}]})
    r = respx_mock.get(f"{BASE}/api/v1/exchange-rate").respond(json={"result": {
        "baseCurrency": "USD", "quoteCurrency": "KRW", "rate": "1380.5"}})
    assert client.get_commissions()[0]["commissionRate"] == "0.00015"
    assert c.calls.last.request.headers["X-Tossinvest-Account"] == "7"
    assert client.get_exchange_rate()["rate"] == "1380.5"
    q = parse_qs(urlparse(str(r.calls.last.request.url)).query)
    assert q == {"baseCurrency": ["USD"], "quoteCurrency": ["KRW"]}


def test_commissions_requires_account_seq(tmp_path, respx_mock):
    http = httpx.Client()
    c = TossClient(TossAuth("id", "sec", tmp_path / "t.json", http), http, RateLimiter({"ORDER_INFO": 6}))
    with pytest.raises(ValueError, match="TOSS_ACCOUNT_SEQ"):
        c.get_commissions()


def test_ratelimiter_spaces_calls(monkeypatch):
    now = [100.0]
    sleeps = []
    monkeypatch.setattr("time.monotonic", lambda: now[0])
    monkeypatch.setattr("time.sleep", lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)))
    rl = RateLimiter({"STOCK_ALL": 1, "MARKET_DATA_CHART": 20})
    rl.acquire("STOCK_ALL")
    rl.acquire("STOCK_ALL")
    assert sleeps == [pytest.approx(1.0)]
    rl.acquire("MARKET_DATA_CHART")  # other group unaffected
    assert len(sleeps) == 1
    with pytest.raises(KeyError):
        rl.acquire("NOPE")


def test_make_client_from_env_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("autotrade.toss.client.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("TOSS_CLIENT_ID", raising=False)
    monkeypatch.setenv("TOSS_CLIENT_SECRET", "super-secret")
    with pytest.raises(RuntimeError, match="TOSS_CLIENT_ID") as e:
        make_client_from_env(tmp_path / "t.json")
    assert "super-secret" not in str(e.value)
    monkeypatch.setenv("TOSS_CLIENT_ID", "cid")
    monkeypatch.delenv("TOSS_CLIENT_SECRET")
    with pytest.raises(RuntimeError, match="TOSS_CLIENT_SECRET"):
        make_client_from_env(tmp_path / "t.json")


def test_make_client_from_env_ok(monkeypatch, tmp_path):
    monkeypatch.setattr("autotrade.toss.client.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("TOSS_CLIENT_ID", "cid")
    monkeypatch.setenv("TOSS_CLIENT_SECRET", "csec")
    assert isinstance(make_client_from_env(tmp_path / "t.json"), TossClient)
