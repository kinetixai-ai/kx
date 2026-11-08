"""Structured page model for Primers, plus the word-budget validator.

A Primer is a list of pages. Each page is a list of typed blocks the
renderer knows how to lay out. The LLM (or the offline outliner) must
produce exactly this JSON shape; ``validate_primer`` rejects anything
else so a bad model response never reaches the renderer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Layout laws (TriFleetRCA / TriParseGPU spec)
WORDS_MIN = 380
WORDS_MAX = 460
VISUAL_SHARE_MIN = 0.35
VISUAL_SHARE_MAX = 0.45

BLOCK_TYPES = {
    "section",      # {"type":"section","num":"1","title":...,"body":...}
    "bento",        # {"type":"bento","columns":2|3,"cards":[{"title","body","icon"?}]}
    "table",        # {"type":"table","title"?,"headers":[...],"rows":[[...],...]}
    "flow",         # {"type":"flow","title"?,"steps":["a","b",...]}  (SVG arrows)
    "code",         # {"type":"code","title"?,"lang"?,"code":"..."}   (terminal box)
    "callout",      # {"type":"callout","style":"trap|tip|security|mnemonic","title","body"}
    "starburst",    # {"type":"starburst","text":"Story to Silicon!"}
    "kv",           # {"type":"kv","title"?,"pairs":[["k","v"],...]}
}
VISUAL_TYPES = {"bento", "table", "flow", "code", "kv", "starburst"}
CALLOUT_STYLES = {"trap", "tip", "security", "mnemonic", "benchmark", "misreading"}


class PrimerSchemaError(ValueError):
    pass


@dataclass
class Page:
    title: str
    subtitle: str = ""
    takeaway: str = ""
    blocks: list[dict] = field(default_factory=list)

    def word_count(self) -> int:
        return sum(_block_words(b) for b in self.blocks) + _words(self.title) + _words(self.subtitle) + _words(self.takeaway)

    def visual_share(self) -> float:
        total = len(self.blocks) or 1
        return sum(1 for b in self.blocks if b.get("type") in VISUAL_TYPES) / total


@dataclass
class Primer:
    title: str
    subtitle: str = ""
    author: str = "Rohit Patel"
    watermark: str = "Rohit Patel | Narmadya Innovations"
    badge: str = "Story to Silicon!"
    topic: str = ""
    category: str = ""
    personas: list[str] = field(default_factory=list)
    pages: list[Page] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)  # paper dicts (verified metadata only)

    def to_dict(self) -> dict:
        return {
            "title": self.title, "subtitle": self.subtitle, "author": self.author,
            "watermark": self.watermark, "badge": self.badge, "topic": self.topic,
            "category": self.category, "personas": self.personas,
            "pages": [{"title": p.title, "subtitle": p.subtitle, "takeaway": p.takeaway,
                       "blocks": p.blocks} for p in self.pages],
            "citations": self.citations,
        }


def _words(text: str | None) -> int:
    return len(re.findall(r"\S+", text or ""))


def _block_words(b: dict) -> int:
    t = b.get("type")
    if t == "section":
        return _words(b.get("title")) + _words(b.get("body"))
    if t == "bento":
        return sum(_words(c.get("title")) + _words(c.get("body")) for c in b.get("cards", []))
    if t == "table":
        return _words(b.get("title")) + sum(_words(str(c)) for r in b.get("rows", []) for c in r) \
            + sum(_words(str(h)) for h in b.get("headers", []))
    if t == "flow":
        return _words(b.get("title")) + sum(_words(s) for s in b.get("steps", []))
    if t == "code":
        return _words(b.get("title")) + min(_words(b.get("code")), 60)  # code counts capped
    if t in ("callout", "starburst"):
        return _words(b.get("title")) + _words(b.get("body")) + _words(b.get("text"))
    if t == "kv":
        return sum(_words(str(k)) + _words(str(v)) for k, v in b.get("pairs", []))
    return 0


def page_from_dict(d: dict) -> Page:
    if not isinstance(d, dict) or not d.get("title"):
        raise PrimerSchemaError("each page needs at least a 'title'.")
    blocks = d.get("blocks") or []
    if not isinstance(blocks, list):
        raise PrimerSchemaError(f"page '{d['title']}': 'blocks' must be a list.")
    for b in blocks:
        if not isinstance(b, dict) or b.get("type") not in BLOCK_TYPES:
            raise PrimerSchemaError(
                f"page '{d['title']}': unknown block {b!r}; allowed types: {sorted(BLOCK_TYPES)}")
        if b["type"] == "callout" and b.get("style") not in CALLOUT_STYLES:
            b["style"] = "tip"
    return Page(title=str(d["title"]), subtitle=str(d.get("subtitle") or ""),
                takeaway=str(d.get("takeaway") or ""), blocks=blocks)


def primer_from_dict(d: dict) -> Primer:
    if not isinstance(d, dict):
        raise PrimerSchemaError("primer must be a JSON object.")
    pages_raw = d.get("pages")
    if not pages_raw or not isinstance(pages_raw, list):
        raise PrimerSchemaError("primer needs a non-empty 'pages' list.")
    return Primer(
        title=str(d.get("title") or "Untitled Primer"),
        subtitle=str(d.get("subtitle") or ""),
        author=str(d.get("author") or "Rohit Patel"),
        watermark=str(d.get("watermark") or "Rohit Patel | Narmadya Innovations"),
        badge=str(d.get("badge") or "Story to Silicon!"),
        topic=str(d.get("topic") or ""),
        category=str(d.get("category") or ""),
        personas=list(d.get("personas") or []),
        pages=[page_from_dict(p) for p in pages_raw],
        citations=list(d.get("citations") or []),
    )


def validate_budget(primer: Primer) -> list[str]:
    """Return human-readable budget violations (empty list = all pages in spec)."""
    problems: list[str] = []
    for i, p in enumerate(primer.pages, 1):
        w = p.word_count()
        if w > WORDS_MAX:
            problems.append(f"page {i} '{p.title}': {w} words (max {WORDS_MAX}) — trim prose ~15%")
        elif w < WORDS_MIN:
            problems.append(f"page {i} '{p.title}': {w} words (min {WORDS_MIN}) — too thin")
        share = p.visual_share()
        if p.blocks and not (VISUAL_SHARE_MIN <= share <= VISUAL_SHARE_MAX + 0.20):
            problems.append(f"page {i} '{p.title}': visual share {share:.0%} "
                            f"(target {VISUAL_SHARE_MIN:.0%}–{VISUAL_SHARE_MAX:.0%})")
    return problems


def compact(primer: Primer) -> int:
    """Deterministic compaction pass: trim the longest prose bodies on
    over-budget pages by ~15%. Returns the number of pages touched."""
    touched = 0
    for p in primer.pages:
        if p.word_count() <= WORDS_MAX:
            continue
        touched += 1
        prose = [b for b in p.blocks if b.get("type") == "section" and b.get("body")]
        prose.sort(key=lambda b: _words(b["body"]), reverse=True)
        for b in prose:
            if p.word_count() <= WORDS_MAX:
                break
            words = b["body"].split()
            keep = max(20, int(len(words) * 0.85))
            sentinel = " ".join(words[:keep])
            # cut at the last sentence boundary so pages never end mid-sentence
            m = re.match(r"^(.+[.!?])[^.!?]*$", sentinel, re.S)
            b["body"] = (m.group(1) if m else sentinel)
    return touched
