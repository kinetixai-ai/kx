"""Researcher lookup via the ORCID public API."""
from __future__ import annotations

import re

import httpx

from ..models import Researcher

BASE = "https://pub.orcid.org/v3.0"
HEADERS = {"Accept": "application/json"}
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


def is_orcid(q: str) -> bool:
    return bool(ORCID_RE.match(q.strip()))


def lookup(query: str, limit: int, client: httpx.Client) -> list[Researcher]:
    query = query.strip()
    if is_orcid(query):
        return [_record(query, client)]
    return _search_by_name(query, limit, client)


def _record(orcid: str, client: httpx.Client) -> Researcher:
    r = client.get(f"{BASE}/{orcid}/record", headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    person = data.get("person") or {}
    name = person.get("name") or {}
    given = (name.get("given-names") or {}).get("value", "")
    family = (name.get("family-name") or {}).get("value", "")
    full = (given + " " + family).strip() or orcid
    works = []
    groups = ((data.get("activities-summary") or {}).get("works") or {}).get("group", [])
    for g in groups[:10]:
        summaries = g.get("work-summary") or []
        if summaries:
            title = ((summaries[0].get("title") or {}).get("title") or {}).get("value")
            if title:
                works.append(title)
    return Researcher(
        name=full,
        orcid=orcid,
        url=f"https://orcid.org/{orcid}",
        works=works,
        source="ORCID",
    )


def _search_by_name(query: str, limit: int, client: httpx.Client) -> list[Researcher]:
    r = client.get(
        f"{BASE}/expanded-search/",
        params={"q": query, "rows": limit},
        headers=HEADERS,
    )
    r.raise_for_status()
    data = r.json()
    out: list[Researcher] = []
    for res in (data.get("expanded-result") or [])[:limit]:
        orcid = res.get("orcid-id")
        given = res.get("given-names") or ""
        family = res.get("family-names") or ""
        insts = res.get("institution-name") or []
        out.append(
            Researcher(
                name=(given + " " + family).strip() or (orcid or "Unknown"),
                orcid=orcid,
                affiliation=insts[0] if insts else None,
                url=f"https://orcid.org/{orcid}" if orcid else None,
                source="ORCID",
            )
        )
    return out
