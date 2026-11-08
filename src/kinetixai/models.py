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
    abstract: str | None = None

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
            "abstract": self.abstract,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        names = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in d.items() if k in names}
        data.setdefault("title", "")
        authors = data.get("authors") or []
        if isinstance(authors, str):
            authors = [a.strip() for a in re.split(r",| and ", authors) if a.strip()]
        data["authors"] = list(authors)
        return cls(**data)


@dataclass
class Repo:
    """A code repository that implements a paper."""
    name: str
    url: str
    stars: int | None = None
    description: str | None = None
    language: str | None = None
    license: str | None = None
    updated: str | None = None
    source: str = ""
    official: bool = False  # linked by the authors on the paper's HF page

    def to_dict(self) -> dict:
        return {
            "name": self.name, "url": self.url, "stars": self.stars, "official": self.official,
            "description": self.description, "language": self.language,
            "license": self.license, "updated": self.updated, "source": self.source,
        }


@dataclass
class HubItem:
    """A Hugging Face model or dataset."""
    id: str
    kind: str  # "model" | "dataset"
    downloads: int | None = None
    likes: int | None = None
    license: str | None = None
    task: str | None = None
    library: str | None = None
    updated: str | None = None

    @property
    def url(self) -> str:
        base = "https://huggingface.co/"
        return base + ("datasets/" if self.kind == "dataset" else "") + self.id

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "url": self.url,
            "downloads": self.downloads, "likes": self.likes, "license": self.license,
            "task": self.task, "library": self.library, "updated": self.updated,
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
