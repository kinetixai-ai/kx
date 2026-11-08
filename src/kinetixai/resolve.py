"""Turn whatever the user typed (arXiv id or URL, DOI, HF paper link) into a Paper."""
from __future__ import annotations

import re

import httpx

from . import net
from .models import Paper
from .sources import arxiv, openalex, semantic_scholar

_NEW = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD = r"[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
ARXIV_RE = re.compile(rf"^(?:{_NEW}|{_OLD})$")
_IN_URL = re.compile(rf"(?:arxiv\.org/(?:abs|pdf|html)/|huggingface\.co/papers/|hf\.co/papers/)({_NEW}|{_OLD})")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def arxiv_id(text: str) -> str | None:
    """'arXiv:2205.14135', '2205.14135v2', an arxiv.org or HF paper URL -> '2205.14135v2'."""
    t = (text or "").strip()
    m = _IN_URL.search(t)
    if m:
        return m.group(1).removesuffix(".pdf")
    if t.lower().startswith("arxiv:"):
        t = t[6:].strip()
    return t if ARXIV_RE.match(t) else None


def base_id(aid: str) -> str:
    return re.sub(r"v\d+$", "", aid)


def doi(text: str) -> str | None:
    t = (text or "").strip()
    for pre in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if t.lower().startswith(pre):
            t = t[len(pre):]
    return t if DOI_RE.match(t) else None


def resolve(ident: str, client: httpx.Client) -> Paper | None:
    """Fetch one paper's metadata. arXiv ids go to arXiv (Semantic Scholar as a
    fallback); DOIs go to Semantic Scholar, then OpenAlex."""
    aid = arxiv_id(ident)
    if aid:
        try:
            r = net.request(client, "GET", arxiv.API, source="arXiv", params={"id_list": aid, "max_results": 1})
            papers = arxiv._parse(r.text) if r is not None else []
            papers = [p for p in papers if p.title and p.title.lower() != "error"]
            if papers:
                return papers[0]
        except (net.SourceError, ValueError):
            pass
        data = net.get_json(client, f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{base_id(aid)}",
                            source="Semantic Scholar", allow_404=True,
                            params={"fields": semantic_scholar.FIELDS})
        return semantic_scholar._paper(data) if data else None

    d = doi(ident)
    if d:
        try:
            data = net.get_json(client, f"https://api.semanticscholar.org/graph/v1/paper/DOI:{d}",
                                source="Semantic Scholar", allow_404=True,
                                params={"fields": semantic_scholar.FIELDS})
            if data:
                return semantic_scholar._paper(data)
        except net.SourceError:
            pass
        data = net.get_json(client, f"https://api.openalex.org/works/doi:{d}", source="OpenAlex", allow_404=True)
        return openalex._paper(data) if data else None
    return None
