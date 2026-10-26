"""Semantic Scholar paper search via the Graph API."""
from __future__ import annotations

import httpx

from ..models import Paper

API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,authors,venue,externalIds,url,citationCount,openAccessPdf"


def search(query: str, limit: int, client: httpx.Client) -> list[Paper]:
    r = client.get(API, params={"query": query, "limit": limit, "fields": FIELDS})
    r.raise_for_status()
    data = r.json()
    return [_paper(p) for p in data.get("data", [])]


def _paper(p: dict) -> Paper:
    ext = p.get("externalIds") or {}
    oa = p.get("openAccessPdf") or {}
    return Paper(
        title=p.get("title") or "",
        authors=[a.get("name", "") for a in (p.get("authors") or [])],
        year=p.get("year"),
        venue=p.get("venue") or None,
        url=p.get("url") or None,
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        pdf_url=oa.get("url") or None,
        citations=p.get("citationCount"),
        source="Semantic Scholar",
    )
