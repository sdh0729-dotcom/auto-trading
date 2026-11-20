from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import httpx

BASE = "https://openapi.tossinvest.com"
REFRESH_MARGIN_S = 300


class TossApiError(Exception):
    def __init__(self, status: int, code: str, request_id: str):
        self.status, self.code, self.request_id = status, code, request_id
        super().__init__(f"Toss API error: status={status} code={code} request_id={request_id}")


def parse_error(resp: httpx.Response) -> TossApiError:
    """Build an error from either the BFF envelope or the OAuth2 format. Never echoes secrets."""
    code, rid = "unknown", ""
    try:
        err = resp.json().get("error")
        if isinstance(err, dict):
            code, rid = str(err.get("code", code)), str(err.get("requestId", ""))
        elif isinstance(err, str):
            code = err
    except (ValueError, AttributeError):
        pass
    return TossApiError(resp.status_code, code, rid)


class TossAuth:
    def __init__(self, client_id: str, client_secret: str, token_path: Path,
                 http: httpx.Client, limiter=None):
        self._id, self._secret = client_id, client_secret
        self._path = Path(token_path)
        self._http = http
        self._limiter = limiter

    def _read_cache(self) -> str | None:
        try:
            d = json.loads(self._path.read_text())
            if d["expires_at"] - time.time() > REFRESH_MARGIN_S:
                return str(d["access_token"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def _write_cache(self, token: str, expires_at: float) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=self._path.name + ".")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump({"access_token": token, "expires_at": expires_at}, f)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def invalidate(self, token: str) -> None:
        """Forget `token` after the API rejected it (401). The cache is removed only when it
        still holds that token: a newer one written by another process is kept."""
        try:
            if json.loads(self._path.read_text()).get("access_token") != token:
                return
        except (OSError, ValueError, AttributeError):
            pass
        self._path.unlink(missing_ok=True)

    def token(self) -> str:
        cached = self._read_cache()
        if cached:
            return cached
        if self._limiter:
            self._limiter.acquire("AUTH")
        resp = self._http.post(f"{BASE}/oauth2/token", data={
            "grant_type": "client_credentials",
            "client_id": self._id, "client_secret": self._secret})
        if resp.status_code != 200:
            raise parse_error(resp)
        body = resp.json()
        token = body["access_token"]
        self._write_cache(token, time.time() + float(body["expires_in"]))
        return token
