"""Structured JSON bundle export — KinetixAI Paper Lab compatible.

Tabs: visualization, architecture, mnemonic, cheatsheet, flashcards,
plus bibtex (verified metadata only) and rag_chunks ready for
`kx library add --from-file` / `kx rag index`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ... import cite as cite_mod


def _flashcards(primer: dict) -> list[dict]:
    cards = []
    for page in primer.get("pages", []):
        for b in page.get("blocks", []):
            if b.get("type") == "kv":
                cards += [{"front": str(k), "back": str(v)} for k, v in b.get("pairs", [])]
            elif b.get("type") == "callout" and b.get("style") in ("trap", "misreading"):
                cards.append({"front": b.get("title", ""), "back": b.get("body", "")})
        if page.get("takeaway"):
            cards.append({"front": f"Key takeaway: {page.get('title')}", "back": page["takeaway"]})
    return cards


def _cheatsheet(primer: dict) -> list[dict]:
    rows = []
    for page in primer.get("pages", []):
        for b in page.get("blocks", []):
            if b.get("type") == "code":
                rows.append({"title": b.get("title") or b.get("lang") or "snippet",
                             "lang": b.get("lang"), "code": b.get("code")})
            elif b.get("type") == "table":
                rows.append({"title": b.get("title") or "table",
                             "headers": b.get("headers"), "rows": b.get("rows")})
    return rows


def _architecture(primer: dict) -> list[dict]:
    flows = []
    for page in primer.get("pages", []):
        for b in page.get("blocks", []):
            if b.get("type") == "flow":
                flows.append({"title": b.get("title") or page.get("title"),
                              "steps": b.get("steps", [])})
    return flows


def _mnemonics(primer: dict) -> list[dict]:
    out = []
    for page in primer.get("pages", []):
        for b in page.get("blocks", []):
            if b.get("type") == "callout" and b.get("style") == "mnemonic":
                out.append({"mnemonic": b.get("title"), "expansion": b.get("body")})
    return out


def _rag_chunks(primer: dict) -> list[dict]:
    chunks = []
    for i, page in enumerate(primer.get("pages", []), 1):
        text = " ".join(b.get("body", "") for b in page.get("blocks", [])
                        if b.get("type") == "section").strip()
        if text:
            chunks.append({"id": f"page-{i}", "title": page.get("title"),
                           "text": text, "takeaway": page.get("takeaway")})
    return chunks


def primer_to_json(primer_dict: dict, out_path: Path) -> Path:
    citations = primer_dict.get("citations") or []
    bundle = {
        "schema": "kinetixai.primer/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": primer_dict.get("title"),
        "subtitle": primer_dict.get("subtitle"),
        "topic": primer_dict.get("topic"),
        "category": primer_dict.get("category"),
        "personas": primer_dict.get("personas"),
        "author": primer_dict.get("author"),
        "watermark": primer_dict.get("watermark"),
        "pages": primer_dict.get("pages"),
        "tabs": {
            "visualization": _architecture(primer_dict),   # same flow data drives both
            "architecture": _architecture(primer_dict),
            "mnemonic": _mnemonics(primer_dict),
            "cheatsheet": _cheatsheet(primer_dict),
            "flashcards": _flashcards(primer_dict),
        },
        "bibtex": cite_mod.bibtex_many(citations) if citations else "",
        "citations": citations,
        "rag_chunks": _rag_chunks(primer_dict),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return out_path
