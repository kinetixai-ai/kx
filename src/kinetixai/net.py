"""HTTP plumbing shared by every source: one client shape, polite retries,
and errors that say which service failed and what to do about it."""
from __future__ import annotations

import time
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from . import __version__

UA = f"kinetixai-cli/{__version__} (+https://kinetixai.ai/cli/)"
TIMEOUT = httpx.Timeout(20.0, connect=10.0)
MAX_WAIT = 8.0  # never sleep longer than this for a Retry-After
_sleep = time.sleep  # patched in tests


class SourceError(Exception):
    """A public service could not answer. The message is safe to show."""

    def __init__(self, source: str, message: str, status: int | None = None):
        super().__init__(f"{source}: {message}")
        self.source = source
        self.status = status


class RateLimited(SourceError):
    pass


def client(headers: dict | None = None, timeout: httpx.Timeout | float | None = None) -> httpx.Client:
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    return httpx.Client(timeout=timeout or TIMEOUT, follow_redirects=True, headers=h)


def _retry_after(r: httpx.Response) -> float | None:
    raw = r.headers.get("retry-after")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
            except (TypeError, ValueError):
                return None
    reset = r.headers.get("x-ratelimit-reset")
    if reset and r.headers.get("x-ratelimit-remaining") == "0":
        try:
            return max(0.0, float(reset) - time.time())
        except ValueError:
            return None
    return None


def _is_rate_limited(r: httpx.Response) -> bool:
    if r.status_code == 429:
        return True
    # GitHub signals an exhausted quota with 403 + remaining=0
    return r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0"


def request(
    c: httpx.Client,
    method: str,
    url: str,
    *,
    source: str,
    retries: int = 2,
    allow_404: bool = False,
    **kwargs: Any,
) -> httpx.Response | None:
    attempt = 0
    while True:
        try:
            r = c.request(method, url, **kwargs)
        except httpx.TimeoutException:
            if attempt < retries:
                attempt += 1
                _sleep(0.5 * attempt)
                continue
            raise SourceError(source, "timed out. Try again, or check your connection.")
        except httpx.HTTPError as exc:
            raise SourceError(source, f"unreachable ({exc.__class__.__name__}). Check your connection or proxy.")
        if _is_rate_limited(r) or r.status_code in (502, 503, 504):
            wait = _retry_after(r)
            wait = 1.0 * (attempt + 1) if wait is None else wait
            if attempt < retries and wait <= MAX_WAIT:
                attempt += 1
                _sleep(wait)
                continue
            if _is_rate_limited(r):
                hint = f" Try again in about {int(wait) + 1}s." if wait else ""
                raise RateLimited(source, "rate limit reached." + hint, r.status_code)
            raise SourceError(source, f"temporarily unavailable ({r.status_code}).", r.status_code)
        if r.status_code == 404 and allow_404:
            return None
        if r.status_code >= 400:
            raise SourceError(source, f"returned HTTP {r.status_code}.", r.status_code)
        return r


def get_json(c: httpx.Client, url: str, *, source: str, allow_404: bool = False, **kwargs: Any) -> Any:
    r = request(c, "GET", url, source=source, allow_404=allow_404, **kwargs)
    if r is None:
        return None
    try:
        return r.json()
    except ValueError:
        raise SourceError(source, "sent a response that is not JSON.")
