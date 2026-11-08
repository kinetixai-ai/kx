"""GitHub repository search (public REST API, no key needed).

Unauthenticated search allows about 10 requests a minute. If GITHUB_TOKEN or
GH_TOKEN is set in the environment it is sent, which raises the limit; it is
never required and never stored.
"""
from __future__ import annotations

import os

import httpx

from .. import net
from ..models import Repo

API = "https://api.github.com"
SOURCE = "GitHub"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def search_repos(query: str, limit: int, client: httpx.Client) -> list[Repo]:
    data = net.get_json(
        client, f"{API}/search/repositories", source=SOURCE, headers=_headers(),
        params={"q": query, "sort": "stars", "order": "desc", "per_page": max(1, min(limit, 50))},
    )
    return [_repo(i) for i in (data or {}).get("items", [])][:limit]


def search_for_arxiv(arxiv_id: str, limit: int, client: httpx.Client) -> list[Repo]:
    """Repos whose name, description, topics or README mention the arXiv id."""
    return search_repos(f"{arxiv_id} in:name,description,readme,topics", limit, client)


def repo(full_name: str, client: httpx.Client) -> Repo | None:
    data = net.get_json(client, f"{API}/repos/{full_name}", source=SOURCE, headers=_headers(), allow_404=True)
    return _repo(data) if data else None


def _repo(i: dict) -> Repo:
    lic = i.get("license") or {}
    return Repo(
        name=i.get("full_name") or i.get("name") or "",
        url=i.get("html_url") or "",
        stars=i.get("stargazers_count"),
        description=i.get("description") or None,
        language=i.get("language") or None,
        license=(lic.get("spdx_id") if lic.get("spdx_id") not in (None, "NOASSERTION") else lic.get("name")) or None,
        updated=(i.get("pushed_at") or i.get("updated_at") or None),
        source=SOURCE,
    )


def parse_repo_url(url: str) -> str | None:
    """'https://github.com/owner/repo(.git)(/...)' -> 'owner/repo'."""
    if not url or "github.com/" not in url:
        return None
    tail = url.split("github.com/", 1)[1].strip("/")
    parts = tail.split("/")
    if len(parts) < 2:
        return None
    name = parts[1][:-4] if parts[1].endswith(".git") else parts[1]
    return f"{parts[0]}/{name}"
