"""Hugging Face Hub: papers, models and datasets (public API, no key needed).

HF_TOKEN, if set in the environment, is sent for higher rate limits; it is
never required and never stored.
"""
from __future__ import annotations

import os

import httpx

from .. import net
from ..models import HubItem

BASE = "https://huggingface.co"
SOURCE = "Hugging Face"
SORTS = {"downloads", "likes", "trending", "updated"}
_SORT_FIELD = {"downloads": "downloads", "likes": "likes", "trending": "trendingScore", "updated": "lastModified"}


def _headers() -> dict:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def paper(arxiv_id: str, client: httpx.Client) -> dict | None:
    """The HF paper page for an arXiv id, or None if HF has none."""
    return net.get_json(client, f"{BASE}/api/papers/{arxiv_id}", source=SOURCE,
                        headers=_headers(), allow_404=True)


def _list(kind: str, params: dict, limit: int, sort: str, client: httpx.Client) -> list[HubItem]:
    params = dict(params)
    params.update({"limit": max(1, min(limit, 100)), "sort": _SORT_FIELD.get(sort, "downloads"), "direction": -1})
    path = "models" if kind == "model" else "datasets"
    data = net.get_json(client, f"{BASE}/api/{path}", source=SOURCE, headers=_headers(), params=params)
    return [_item(d, kind) for d in (data or []) if isinstance(d, dict)][:limit]


def search_models(query: str, limit: int, client: httpx.Client, sort: str = "downloads",
                  task: str | None = None) -> list[HubItem]:
    params = {"search": query}
    if task:
        params["pipeline_tag"] = task
    return _list("model", params, limit, sort, client)


def models_for_arxiv(arxiv_id: str, limit: int, client: httpx.Client, sort: str = "downloads") -> list[HubItem]:
    """Models whose card cites the paper (HF tags them arxiv:<id>)."""
    return _list("model", {"filter": f"arxiv:{arxiv_id}"}, limit, sort, client)


def search_datasets(query: str, limit: int, client: httpx.Client, sort: str = "downloads") -> list[HubItem]:
    return _list("dataset", {"search": query}, limit, sort, client)


def datasets_for_arxiv(arxiv_id: str, limit: int, client: httpx.Client, sort: str = "downloads") -> list[HubItem]:
    return _list("dataset", {"filter": f"arxiv:{arxiv_id}"}, limit, sort, client)


def _item(d: dict, kind: str) -> HubItem:
    tags = d.get("tags") or []
    lic = next((t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("license:")), None)
    if not lic:
        lic = (d.get("cardData") or {}).get("license")
        if isinstance(lic, list):
            lic = ", ".join(map(str, lic))
    task = d.get("pipeline_tag")
    if not task and kind == "dataset":
        task = next((t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("task_categories:")), None)
    return HubItem(
        id=d.get("id") or d.get("modelId") or "",
        kind=kind,
        downloads=d.get("downloads"),
        likes=d.get("likes"),
        license=lic or None,
        task=task or None,
        library=d.get("library_name") or None,
        updated=d.get("lastModified") or d.get("createdAt") or None,
    )
