"""arXiv search via the public Atom API."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from ..models import Paper

API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"


def search(query: str, limit: int, client: httpx.Client) -> list[Paper]:
    r = client.get(
        API,
        params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
    )
    r.raise_for_status()
    return _parse(r.text)


def _parse(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        title = (entry.findtext(f"{ATOM}title") or "").strip().replace("\n", " ")
        published = entry.findtext(f"{ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        authors = [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in entry.findall(f"{ATOM}author")
        ]
        abs_url = (entry.findtext(f"{ATOM}id") or "").strip()
        arxiv_id = abs_url.rsplit("/abs/", 1)[-1] if "/abs/" in abs_url else None
        pdf_url = None
        for link in entry.findall(f"{ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
        summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split()) or None
        papers.append(
            Paper(
                title=title,
                authors=authors,
                year=year,
                venue="arXiv",
                url=abs_url or None,
                arxiv_id=arxiv_id,
                pdf_url=pdf_url,
                source="arXiv",
                abstract=summary,
            )
        )
    return papers
