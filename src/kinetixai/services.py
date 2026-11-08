"""The operations behind both the CLI and the MCP server.

Every function takes plain arguments, returns plain JSON-able dicts and
never prints, so the same code serves `kx ... --json` and MCP tool calls.
Failures of one public source become warnings rather than errors.
"""
from __future__ import annotations

import httpx

from . import library, net
from .models import Repo
from .rag import index as rag_index
from .resolve import arxiv_id, base_id, resolve
from .sources import aggregate, github, huggingface


def _client(c: httpx.Client | None) -> tuple[httpx.Client, bool]:
    return (c, False) if c is not None else (net.client(), True)


def search_papers(query: str, limit: int = 10, sources: list[str] | None = None,
                  client: httpx.Client | None = None) -> dict:
    c, own = _client(client)
    try:
        papers, warnings = aggregate.search(query, limit, c, sources)
    finally:
        if own:
            c.close()
    return {"query": query, "papers": [p.to_dict() for p in papers], "warnings": warnings}


def get_paper(ident: str, client: httpx.Client | None = None) -> dict | None:
    c, own = _client(client)
    try:
        p = resolve(ident, c)
    finally:
        if own:
            c.close()
    return p.to_dict() if p else None


def find_code(paper: str, limit: int = 10, client: httpx.Client | None = None) -> dict:
    """Implementations of a paper: the authors' repo from its Hugging Face
    paper page, then GitHub repos that mention the arXiv id, by stars."""
    c, own = _client(client)
    warnings: list[str] = []
    repos: dict[str, Repo] = {}
    info = None
    aid = arxiv_id(paper)
    try:
        if aid:
            bid = base_id(aid)
            hf = None
            try:
                hf = huggingface.paper(bid, c)
            except net.SourceError as exc:
                warnings.append(str(exc))
            if hf:
                info = {"arxiv_id": bid, "title": hf.get("title"), "upvotes": hf.get("upvotes"),
                        "hf_url": f"https://huggingface.co/papers/{bid}"}
                name = github.parse_repo_url(hf.get("githubRepo") or "")
                if name:
                    repos[name.lower()] = Repo(name=name, url=f"https://github.com/{name}",
                                               stars=hf.get("githubStars"), source="Hugging Face", official=True)
        found = []
        try:
            found = (github.search_for_arxiv(base_id(aid), limit, c) if aid
                     else github.search_repos(paper, limit, c))
        except net.SourceError as exc:
            warnings.append(str(exc))
        for r in found:
            key = r.name.lower()
            if key in repos:  # GitHub has fresher stars and details for the official repo
                r.official = repos[key].official
            repos[key] = r
    finally:
        if own:
            c.close()
    ordered = sorted(repos.values(), key=lambda r: (not r.official, -(r.stars or 0)))[:limit]
    return {"query": paper, "arxiv_id": base_id(aid) if aid else None, "paper": info,
            "repos": [r.to_dict() for r in ordered], "warnings": warnings}


def _hub(kind: str, query: str, limit: int, sort: str, task: str | None, client: httpx.Client | None) -> dict:
    c, own = _client(client)
    warnings: list[str] = []
    items = []
    aid = arxiv_id(query)
    try:
        if kind == "model":
            items = (huggingface.models_for_arxiv(base_id(aid), limit, c, sort) if aid
                     else huggingface.search_models(query, limit, c, sort, task))
        else:
            items = (huggingface.datasets_for_arxiv(base_id(aid), limit, c, sort) if aid
                     else huggingface.search_datasets(query, limit, c, sort))
    except net.SourceError as exc:
        warnings.append(str(exc))
    finally:
        if own:
            c.close()
    key = "models" if kind == "model" else "datasets"
    return {"query": query, "arxiv_id": base_id(aid) if aid else None,
            key: [i.to_dict() for i in items], "warnings": warnings}


def find_models(query: str, limit: int = 10, sort: str = "downloads", task: str | None = None,
                client: httpx.Client | None = None) -> dict:
    return _hub("model", query, limit, sort, task, client)


def find_datasets(query: str, limit: int = 10, sort: str = "downloads",
                  client: httpx.Client | None = None) -> dict:
    return _hub("dataset", query, limit, sort, None, client)


def library_search(query: str, limit: int = 10) -> dict:
    hits = library.search(query, limit)
    return {"query": query, "papers": hits, "total": len(library.load())}


def rag_ask(question: str, k: int = 5, llm: bool = True, model: str | None = None) -> dict:
    return rag_index.ask(question, k=k, llm=llm, model=model)
