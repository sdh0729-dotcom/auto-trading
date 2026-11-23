from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd
from dotenv import load_dotenv

from autotrade.toss.auth import BASE, TossApiError, TossAuth, parse_error
from autotrade.toss.ratelimit import RateLimiter

__all__ = ["BASE", "TossApiError", "TossClient", "CandlePage", "make_client_from_env", "RATES"]

# Per-second limits (Rate Limits Group per endpoint, from docs/toss/openapi.json).
# MARKET_INFO 3/s per Toss overview Rate Limits table.
STOCKS_PER_CALL = 200
RATES = {"MARKET_DATA_CHART": 20, "MARKET_DATA": 15, "STOCK_ALL": 1, "STOCK": 5,
         "ORDER_INFO": 6, "MARKET_INFO": 3, "AUTH": 5}
MAX_ATTEMPTS = 5
CANDLE_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _backoff(attempt: int) -> int:
    return 2 ** (attempt - 1)


@dataclass
class CandlePage:
    df: pd.DataFrame
    next_before: datetime | None


class TossClient:
    """Read-only Toss Open API client (market data / stock list / fees / FX only)."""

    def __init__(self, auth: TossAuth, http: httpx.Client, limiter: RateLimiter,
                 account_seq: int | None = None):
        self._auth, self._http, self._limiter = auth, http, limiter
        self._account_seq = account_seq

    def _get(self, path: str, group: str, params: dict | None = None,
             headers: dict | None = None):
        """GET with retries: 429 waits Retry-After, 5xx and timeouts back off 1, 2, 4, 8 s
        (at most MAX_ATTEMPTS calls); a 401 drops the cached token and retries once."""
        reauthed = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            last = attempt == MAX_ATTEMPTS
            self._limiter.acquire(group)
            token = self._auth.token()
            try:
                resp = self._http.get(
                    f"{BASE}{path}", params=params,
                    headers={"Authorization": f"Bearer {token}", **(headers or {})})
            except httpx.TimeoutException:
                if last:
                    raise
                time.sleep(_backoff(attempt))
                continue
            if resp.status_code == 429 and not last:
                try:
                    wait = float(resp.headers.get("Retry-After", 1))
                except ValueError:
                    wait = 1.0
                time.sleep(int(wait) if wait == int(wait) else wait)
                continue
            if resp.status_code >= 500 and not last:
                time.sleep(_backoff(attempt))
                continue
            if resp.status_code == 401 and not reauthed and not last:
                reauthed = True
                self._auth.invalidate(token)
                continue
            if resp.status_code != 200:
                raise parse_error(resp)
            return resp.json()["result"]
        raise AssertionError("unreachable")

    def get_candles(self, symbol: str, interval: Literal["1m", "1d"],
                    before: datetime | None = None, count: int = 200,
                    adjusted: bool = True) -> CandlePage:
        params: dict = {"symbol": symbol, "interval": interval, "count": count,
                        "adjusted": "true" if adjusted else "false"}
        if before is not None:
            if before.tzinfo is None:
                raise ValueError("before must be timezone-aware")
            params["before"] = before.isoformat()  # httpx percent-encodes '+' as %2B
        res = self._get("/api/v1/candles", "MARKET_DATA_CHART", params)
        rows = [{"timestamp": c["timestamp"], "open": c["openPrice"], "high": c["highPrice"],
                 "low": c["lowPrice"], "close": c["closePrice"], "volume": c["volume"]}
                for c in res["candles"]]
        df = pd.DataFrame(rows, columns=CANDLE_COLS)
        if rows:
            df["timestamp"] = pd.to_datetime(df["timestamp"].map(datetime.fromisoformat))
            df[CANDLE_COLS[1:]] = df[CANDLE_COLS[1:]].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
        nb = res.get("nextBefore")
        return CandlePage(df, datetime.fromisoformat(nb) if nb else None)

    def list_stocks(self, market: str, status: str = "ACTIVE",
                    security_type: str | None = None) -> pd.DataFrame:
        params = {"market": market, "status": status}
        if security_type:
            params["securityType"] = security_type
        res = self._get("/api/v1/stocks/all", "STOCK_ALL", params)
        return pd.DataFrame(
            [{"symbol": s["symbol"], "name": s["name"], "security_type": s["securityType"],
              "is_common_share": s["isCommonShare"], "isin": s.get("isinCode")} for s in res],
            columns=["symbol", "name", "security_type", "is_common_share", "isin"])

    def get_stocks(self, symbols: list[str]) -> list[dict]:
        """Stock info (/api/v1/stocks) for `symbols`, 200 per call (the API maximum)."""
        out: list[dict] = []
        for i in range(0, len(symbols), STOCKS_PER_CALL):
            batch = symbols[i:i + STOCKS_PER_CALL]
            out.extend(self._get("/api/v1/stocks", "STOCK", {"symbols": ",".join(batch)}) or [])
        return out

    def get_commissions(self) -> list[dict]:
        if self._account_seq is None:
            raise ValueError("commissions endpoint needs an account header: set TOSS_ACCOUNT_SEQ")
        return self._get("/api/v1/commissions", "ORDER_INFO",
                         headers={"X-Tossinvest-Account": str(self._account_seq)})

    def get_exchange_rate(self) -> dict:
        return self._get("/api/v1/exchange-rate", "MARKET_INFO",
                         {"baseCurrency": "USD", "quoteCurrency": "KRW"})


def make_client_from_env(token_path: Path) -> TossClient:
    from autotrade.config import env_path, repo_root

    load_dotenv(env_path("AUTOTRADE_DOTENV", repo_root() / ".env"), override=False)
    vals = {}
    for name in ("TOSS_CLIENT_ID", "TOSS_CLIENT_SECRET"):
        v = os.environ.get(name)
        if not v:
            raise RuntimeError(f"missing environment variable {name}")
        vals[name] = v
    seq = os.environ.get("TOSS_ACCOUNT_SEQ")
    http = httpx.Client(timeout=30.0)
    limiter = RateLimiter(RATES)
    auth = TossAuth(vals["TOSS_CLIENT_ID"], vals["TOSS_CLIENT_SECRET"], Path(token_path),
                    http, limiter)
    return TossClient(auth, http, limiter, int(seq) if seq else None)
