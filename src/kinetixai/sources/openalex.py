"""OpenAlex works search."""
from __future__ import annotations

import httpx

from ..models import Paper

API = "https://api.openalex.org/works"


def search(query: str, limit: int, client: httpx.Client) -> list[Paper]:
    r = client.get(API, params={"search": query, "per-page": limit})
    r.raise_for_status()
    data = r.json()
    return [_paper(w) for w in data.get("results", [])]


def _paper(w: dict) -> Paper:
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (w.get("authorships") or [])
    ]
    loc = w.get("primary_location") or {}
    source = loc.get("source") or {}
    doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
    return Paper(
        title=w.get("display_name") or "",
        authors=authors,
        year=w.get("publication_year"),
        venue=source.get("display_name") or None,
        url=w.get("id") or None,
        doi=doi,
        pdf_url=(loc.get("pdf_url") or None),
        citations=w.get("cited_by_count"),
        source="OpenAlex",
    )
