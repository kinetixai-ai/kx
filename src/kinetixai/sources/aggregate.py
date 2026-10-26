"""Federated search across the public sources, with de-duplication."""
from __future__ import annotations

import httpx

from ..models import Paper
from . import arxiv, openalex, semantic_scholar

SOURCES = {
    "arxiv": arxiv.search,
    "semantic": semantic_scholar.search,
    "openalex": openalex.search,
}


def search(
    query: str,
    limit: int,
    client: httpx.Client,
    sources: list[str] | None = None,
) -> tuple[list[Paper], list[str]]:
    """Return (merged papers, warnings). One failing source never aborts the rest."""
    names = sources or list(SOURCES)
    merged: dict[str, Paper] = {}
    warnings: list[str] = []
    for name in names:
        fn = SOURCES.get(name)
        if fn is None:
            warnings.append(f"unknown source: {name}")
            continue
        try:
            for paper in fn(query, limit, client):
                existing = merged.get(paper.key())
                if existing is None:
                    merged[paper.key()] = paper
                else:
                    _enrich(existing, paper)
        except Exception as exc:  # one source down must not kill the search
            warnings.append(f"{name} unavailable ({exc.__class__.__name__})")
    papers = sorted(
        merged.values(),
        key=lambda p: (p.citations or -1, p.year or -1),
        reverse=True,
    )
    return papers, warnings


def _enrich(target: Paper, other: Paper) -> None:
    """Fill gaps on a kept paper from a duplicate found in another source."""
    for attr in ("doi", "arxiv_id", "pdf_url", "venue", "year", "url"):
        if getattr(target, attr) in (None, "") and getattr(other, attr) not in (None, ""):
            setattr(target, attr, getattr(other, attr))
    if (target.citations is None) and (other.citations is not None):
        target.citations = other.citations
    if other.source and other.source not in target.source:
        target.source = f"{target.source}, {other.source}"
