"""Shared data models for CLI results."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    citations: int | None = None
    source: str = ""

    def key(self) -> str:
        """De-duplication key: prefer DOI, else a normalized title."""
        if self.doi:
            return "doi:" + self.doi.lower().strip()
        if self.arxiv_id:
            # strip the version suffix so 2205.14135 and 2205.14135v2 match
            base = re.sub(r"v\d+$", "", self.arxiv_id.lower().strip())
            return "arxiv:" + base
        title = re.sub(r"[^a-z0-9]", "", (self.title or "").lower())
        return "title:" + title

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "url": self.url,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "pdf_url": self.pdf_url,
            "citations": self.citations,
            "source": self.source,
        }


@dataclass
class Researcher:
    name: str
    orcid: str | None = None
    affiliation: str | None = None
    url: str | None = None
    works: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "orcid": self.orcid,
            "affiliation": self.affiliation,
            "url": self.url,
            "works": self.works,
            "source": self.source,
        }
