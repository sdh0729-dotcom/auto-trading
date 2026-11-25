from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import date
from xml.etree import ElementTree

import httpx
import pandas as pd
from dotenv import load_dotenv

__all__ = ["BASE", "DartError", "DartRateLimited", "DartClient", "make_dart_client_from_env"]

BASE = "https://opendart.fss.or.kr/api/"
PAGE_COUNT = 100
CORP_COLS = ["corp_code", "stock_code", "corp_name"]


class DartError(Exception):
    """DART failure. Carries only a status and a message; never a URL or the API key."""

    def __init__(self, status: str, message: str):
        super().__init__(status, message)
        self.status = status
        self.message = message

    def __str__(self) -> str:
        return f"DART {self.status}: {self.message}"

    def __repr__(self) -> str:
        return f"DartError(status={self.status!r}, message={self.message!r})"


class DartRateLimited(DartError):
    pass


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


class DartClient:
    """Read-only OpenDART client. The key travels in the query string, so every error
    path is scrubbed: only exception class names and status codes are ever surfaced."""

    def __init__(self, key: str, http: httpx.Client | None = None, min_interval: float = 0.1):
        self._key = key
        self._http = http if http is not None else httpx.Client(timeout=30.0)
        self._min_interval = min_interval
        self._last = 0.0

    def _get(self, path: str, params: dict) -> httpx.Response:
        wait = self._last + self._min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self._http.get(f"{BASE}{path}", params={"crtfc_key": self._key, **params})
        except httpx.HTTPError as e:
            self._last = time.monotonic()
            raise DartError("http", type(e).__name__) from None
        self._last = time.monotonic()
        if resp.status_code != 200:
            raise DartError("http", f"HTTP {resp.status_code}")
        return resp

    @staticmethod
    def _raise_status(status: str, message: str) -> None:
        if status == "020":
            raise DartRateLimited(status, message)
        raise DartError(status, message)

    def _json(self, path: str, params: dict) -> dict:
        resp = self._get(path, params)
        try:
            body = resp.json()
        except ValueError:
            raise DartError("http", "invalid JSON") from None
        if not isinstance(body, dict):
            raise DartError("http", "unexpected JSON body")
        status = str(body.get("status", ""))
        if status not in ("000", "013"):
            self._raise_status(status, str(body.get("message", "")))
        return body

    def _paged(self, path: str, params: dict) -> list[dict]:
        rows: list[dict] = []
        page = 1
        while True:
            body = self._json(path, {**params, "page_no": page, "page_count": PAGE_COUNT})
            if str(body.get("status")) == "013":
                return rows
            rows.extend(body.get("list") or [])
            try:
                total_page = int(body.get("total_page") or 1)
            except (TypeError, ValueError):
                raise DartError("http", "bad total_page") from None
            if page >= total_page:
                return rows
            page += 1

    def _zip_or_status(self, path: str, params: dict) -> bytes:
        """Raw zip bytes; DART answers errors with HTTP 200 and an XML status body."""
        content = self._get(path, params).content
        if content[:2] == b"PK":
            return content
        try:
            root = ElementTree.fromstring(content)
            status = (root.findtext("status") or "").strip()
            message = (root.findtext("message") or "").strip()
        except ElementTree.ParseError:
            raise DartError("http", "unreadable reply") from None
        self._raise_status(status or "http", message or "unreadable reply")
        raise AssertionError("unreachable")  # pragma: no cover

    def corp_codes(self) -> pd.DataFrame:
        content = self._zip_or_status("corpCode.xml", {})
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                root = ElementTree.fromstring(z.read("CORPCODE.xml"))
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError):
            raise DartError("corp_code", "unreadable corpCode.xml reply") from None
        rows = [{"corp_code": (e.findtext("corp_code") or "").strip(),
                 "stock_code": (e.findtext("stock_code") or "").strip(),
                 "corp_name": (e.findtext("corp_name") or "").strip()}
                for e in root.iter("list")]
        return pd.DataFrame(rows, columns=CORP_COLS)

    def list_filings(self, corp_code: str, start: date, end: date, kind: str) -> list[dict]:
        return self._paged("list.json", {"corp_code": corp_code, "bgn_de": _ymd(start),
                                         "end_de": _ymd(end), "pblntf_ty": kind})

    def detail(self, endpoint: str, corp_code: str, start: date, end: date) -> list[dict]:
        body = self._json(endpoint, {"corp_code": corp_code, "bgn_de": _ymd(start),
                                     "end_de": _ymd(end)})
        return list(body.get("list") or [])

    def document(self, rcept_no: str) -> bytes:
        return self._zip_or_status("document.xml", {"rcept_no": rcept_no})


def make_dart_client_from_env() -> DartClient:
    from autotrade.config import env_path, repo_root

    load_dotenv(env_path("AUTOTRADE_DOTENV", repo_root() / ".env"), override=False)
    key = os.environ.get("DART_API_KEY")
    if not key:
        raise DartError("config", "DART_API_KEY missing in .env")
    return DartClient(key)
