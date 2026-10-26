"""Thin client for the KinetixAI backend API (ask)."""
from __future__ import annotations

import httpx


class ApiError(Exception):
    pass


class KinetixApi:
    def __init__(self, endpoint: str, client: httpx.Client):
        self.endpoint = endpoint.rstrip("/")
        self.client = client

    def ask(self, question: str) -> dict:
        """POST /api/ask -> {answer, citations}. Raises ApiError on failure."""
        try:
            r = self.client.post(
                f"{self.endpoint}/api/ask",
                json={"question": question},
            )
        except httpx.HTTPError as exc:
            raise ApiError(f"could not reach {self.endpoint} ({exc.__class__.__name__})")
        if r.status_code == 404:
            raise ApiError(
                "the KinetixAI API is not available at this endpoint yet. "
                "Set a different one with `kx config set endpoint <url>`."
            )
        if r.status_code >= 400:
            raise ApiError(f"API returned {r.status_code}")
        data = r.json()
        if not data.get("ok", True) and data.get("error"):
            raise ApiError(data["error"])
        return data
