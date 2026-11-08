"""Client for the KinetixAI API (ask, whoami, token self-revoke)."""
from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import httpx


class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def is_loopback_host(host: str | None) -> bool:
    """Exactly 'localhost' or a loopback IP (127.0.0.0/8, ::1). Not '127.evil.com'."""
    host = (host or "").lower().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _loopback(url: str) -> bool:
    try:
        return is_loopback_host(urlparse(url).hostname)
    except ValueError:
        return False


class KinetixApi:
    def __init__(self, endpoint: str, client: httpx.Client, token: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.client = client
        self.token = token
        try:
            scheme = urlparse(self.endpoint).scheme
        except ValueError:
            raise ApiError(f"{self.endpoint!r} is not a valid endpoint.")
        if token and scheme != "https" and not _loopback(self.endpoint):
            raise ApiError(
                f"refusing to send your API token over plain http to {self.endpoint}. "
                "Use an https endpoint (or localhost for development)."
            )

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _call(self, method: str, path: str, **kwargs) -> dict:
        try:
            r = self.client.request(method, f"{self.endpoint}{path}", headers=self._headers(), **kwargs)
        except httpx.TimeoutException:
            raise ApiError(f"{self.endpoint} timed out. Try again in a moment.")
        except (httpx.InvalidURL, httpx.UnsupportedProtocol, ValueError):
            raise ApiError(f"{self.endpoint!r} is not a valid endpoint. Fix it with `kx config set endpoint https://kinetixai.ai`.")
        except httpx.HTTPError as exc:
            raise ApiError(f"could not reach {self.endpoint} ({exc.__class__.__name__}).")
        if r.status_code == 404:
            raise ApiError(
                "the KinetixAI API is not available at this endpoint. "
                "Check it with `kx config get endpoint`.", 404)
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code == 401:
            raise ApiError(data.get("error") or "not signed in. Run `kx login`.", 401)
        if r.status_code == 429:
            raise ApiError("rate limited by KinetixAI. Wait a minute and try again.", 429)
        if r.status_code >= 400:
            raise ApiError(data.get("error") or f"API returned {r.status_code}", r.status_code)
        if isinstance(data, dict) and data.get("ok") is False and data.get("error"):
            raise ApiError(data["error"], r.status_code)
        return data if isinstance(data, dict) else {"data": data}

    def ask(self, question: str) -> dict:
        """POST /api/ask -> {answer, citations, grounded}."""
        return self._call("POST", "/api/ask", json={"question": question})

    def me(self) -> dict | None:
        """GET /api/auth/me -> the signed-in user, or None."""
        return self._call("GET", "/api/auth/me").get("user")

    def revoke_current(self) -> dict:
        """DELETE /api/auth/tokens?id=current: a token revoking itself."""
        return self._call("DELETE", "/api/auth/tokens", params={"id": "current"})
