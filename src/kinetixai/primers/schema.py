"""Structured page model for Primers, plus budget and density validators.

A Primer is a list of pages. Each page is a list of typed blocks the
renderer knows how to lay out. The LLM (or the offline outliner) must
produce exactly this JSON shape; ``validate_primer`` rejects anything
else so a bad model response never reaches the renderer.

Density (the TriFleetRCA standard): every page carries 5-7 numbered
sections and a mandatory mix of visual blocks, so pages come out packed
rather than one-paragraph sparse. ``density`` reports what a page is
missing; the builder uses that to drive an expansion pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Layout laws (TriFleetRCA / TriParseGPU spec).
#
# The word budget is what actually fits beside the mandatory visuals. The
# five required visual blocks cost ~119mm of the ~208mm printable area, so
# roughly 89mm is left for prose: about 200 words across 5-7 sections, plus
# the words inside tables and cards. A higher target cannot be printed, it
# can only be clipped.
WORDS_MIN = 225
WORDS_MAX = 420
WORDS_FLOOR = 200          # below this a page is sparse and gets expanded
SECTIONS_MIN = 5
SECTIONS_MAX = 7
VISUAL_SHARE_MIN = 0.35
VISUAL_SHARE_MAX = 0.45

BLOCK_TYPES = {
    "section",      # {"type":"section","num":"1","title":...,"body":...}
    "bento",        # {"type":"bento","columns":2|3,"cards":[{"title","body","icon"?}]}
    "table",        # {"type":"table","title"?,"headers":[...],"rows":[[...],...]}
    "flow",         # {"type":"flow","title"?,"steps":["a","b",...]}  (SVG arrows)
    "steps",        # {"type":"steps","title"?,"items":[...]}  (numbered vertical)
    "hierarchy",    # {"type":"hierarchy","title"?,"levels":[{"label","detail"}]}
    "chart",        # {"type":"chart","title"?,"unit"?,"note"?,"bars":[{"label","value"}]}
    "code",         # {"type":"code","title"?,"lang"?,"code":"..."}   (terminal box)
    "callout",      # {"type":"callout","style":"trap|tip|security|mnemonic","title","body"}
    "cloud",        # {"type":"cloud","text":"..."}  (pink scalloped comic cloud)
    "starburst",    # {"type":"starburst","text":"Story to Silicon!"}
    "kv",           # {"type":"kv","title"?,"pairs":[["k","v"],...]}
}

# The spec and the prompts use descriptive names; accept both so a model
# that echoes the blueprint's wording still produces valid blocks.
BLOCK_ALIASES = {
    "pipeline_flow": "flow",
    "hierarchy_stack": "hierarchy",
    "comparison_table": "table",
    "bento_cards": "bento",
    "code_runbook": "code",
    "callout_cloud": "cloud",
    "bar_chart": "chart",
    "numbered_steps": "steps",
    "mnemonic_box": "callout",
    "paragraph": "section",
    "text": "section",
}

VISUAL_TYPES = {"bento", "table", "flow", "steps", "hierarchy", "chart", "code", "kv",
                "starburst", "cloud"}
DIAGRAM_TYPES = {"flow", "hierarchy", "steps"}
CALLOUT_STYLES = {"trap", "tip", "security", "mnemonic", "benchmark", "misreading", "warning"}

# What a dense page must contain, by the spec's density gate.
REQUIRED_KINDS: dict[str, set[str]] = {
    "a diagram (flow, steps or hierarchy)": DIAGRAM_TYPES,
    "a table": {"table"},
    "a card grid (bento, 3+ cards)": {"bento"},
    "a code or runbook block": {"code"},
    "a comic cloud callout": {"cloud"},
}


class PrimerSchemaError(ValueError):
    pass


@dataclass
class Page:
    title: str
    subtitle: str = ""
    takeaway: str = ""
    blocks: list[dict] = field(default_factory=list)
    must_have: list[str] = field(default_factory=list)  # from the page archetype

    def word_count(self) -> int:
        return (sum(_block_words(b) for b in self.blocks)
                + _words(self.title) + _words(self.subtitle) + _words(self.takeaway))

    def sections(self) -> list[dict]:
        return [b for b in self.blocks if b.get("type") == "section"]

    def visual_share(self) -> float:
        total = len(self.blocks) or 1
        return sum(1 for b in self.blocks if b.get("type") in VISUAL_TYPES) / total

    def kinds(self) -> set[str]:
        return {b.get("type") for b in self.blocks}


@dataclass
class Primer:
    title: str
    subtitle: str = ""
    author: str = "Rohit Patel"
    watermark: str = "Narmadya Innovations"
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
        return (_words(b.get("title"))
                + sum(_words(str(c)) for r in b.get("rows", []) for c in r)
                + sum(_words(str(h)) for h in b.get("headers", [])))
    if t == "flow":
        return _words(b.get("title")) + sum(_words(s) for s in b.get("steps", []))
    if t == "steps":
        return _words(b.get("title")) + sum(_words(str(s)) for s in b.get("items", []))
    if t == "hierarchy":
        return _words(b.get("title")) + sum(
            _words(lv.get("label")) + _words(lv.get("detail")) for lv in b.get("levels", []))
    if t == "chart":
        return (_words(b.get("title")) + _words(b.get("note"))
                + sum(_words(str(bar.get("label"))) for bar in b.get("bars", [])))
    if t == "code":
        return _words(b.get("title")) + min(_words(b.get("code")), 60)  # code counts capped
    if t in ("callout", "starburst", "cloud"):
        return _words(b.get("title")) + _words(b.get("body")) + _words(b.get("text"))
    if t == "kv":
        return sum(_words(str(k)) + _words(str(v)) for k, v in b.get("pairs", []))
    return 0


def _normalize_block(b: Any, where: str) -> dict:
    if not isinstance(b, dict):
        raise PrimerSchemaError(f"{where}: block must be an object, got {type(b).__name__}")
    kind = b.get("type")
    kind = BLOCK_ALIASES.get(kind, kind)
    if kind not in BLOCK_TYPES:
        raise PrimerSchemaError(
            f"{where}: unknown block type {b.get('type')!r}; allowed: {sorted(BLOCK_TYPES)}")
    b["type"] = kind
    if kind == "callout" and b.get("style") not in CALLOUT_STYLES:
        b["style"] = "tip"
    if kind == "chart":
        bars = []
        for bar in b.get("bars") or []:
            try:
                bars.append({"label": str(bar.get("label", "")), "value": float(bar.get("value", 0))})
            except (TypeError, ValueError):
                continue  # a bar without a number is not a bar
        b["bars"] = bars
    return b


def page_from_dict(d: dict) -> Page:
    if not isinstance(d, dict) or not d.get("title"):
        raise PrimerSchemaError("each page needs at least a 'title'.")
    blocks = d.get("blocks") or []
    if not isinstance(blocks, list):
        raise PrimerSchemaError(f"page '{d['title']}': 'blocks' must be a list.")
    where = f"page '{d['title']}'"
    return Page(title=str(d["title"]), subtitle=str(d.get("subtitle") or ""),
                takeaway=str(d.get("takeaway") or ""),
                blocks=[_normalize_block(b, where) for b in blocks])


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
        watermark=str(d.get("watermark") or "Narmadya Innovations"),
        badge=str(d.get("badge") or "Story to Silicon!"),
        topic=str(d.get("topic") or ""),
        category=str(d.get("category") or ""),
        personas=list(d.get("personas") or []),
        pages=[page_from_dict(p) for p in pages_raw],
        citations=list(d.get("citations") or []),
    )


# ── density gate ──────────────────────────────────────────────────────────

@dataclass
class Density:
    """What one page has, and what it still needs to reach the standard."""
    sections: int
    words: int
    missing_kinds: list[str] = field(default_factory=list)
    thin_bento: bool = False

    @property
    def ok(self) -> bool:
        return (self.sections >= SECTIONS_MIN and self.words >= WORDS_FLOOR
                and not self.missing_kinds and not self.thin_bento)

    def reasons(self) -> list[str]:
        out: list[str] = []
        if self.sections < SECTIONS_MIN:
            out.append(f"{self.sections} numbered sections (need {SECTIONS_MIN}-{SECTIONS_MAX})")
        if self.words < WORDS_FLOOR:
            out.append(f"{self.words} words (need at least {WORDS_FLOOR}, target {WORDS_MIN}-{WORDS_MAX})")
        if self.thin_bento:
            out.append("the card grid has fewer than 3 cards")
        out += [f"missing {k}" for k in self.missing_kinds]
        return out


def required_kinds(must_have: list[str] | None = None) -> dict[str, set[str]]:
    """What this page must contain: its archetype's list, or the default mix."""
    if not must_have:
        return REQUIRED_KINDS
    out: dict[str, set[str]] = {}
    for name in must_have:
        if name in DIAGRAM_TYPES:
            out.setdefault("a diagram (flow, steps or hierarchy)", set()).update(DIAGRAM_TYPES)
        else:
            out[f"a {name} block"] = {name}
    return out


def density(page: Page, min_sections: int = SECTIONS_MIN,
            must_have: list[str] | None = None) -> Density:
    kinds = page.kinds()
    missing = [label for label, allowed in required_kinds(must_have).items()
               if not (kinds & allowed)]
    thin = any(b.get("type") == "bento" and len(b.get("cards") or []) < 3 for b in page.blocks)
    d = Density(sections=len(page.sections()), words=page.word_count(),
                missing_kinds=missing, thin_bento=thin)
    if d.sections < min_sections:  # honor a stricter --min-sections-per-page
        d.sections = len(page.sections())
    return d


def validate_density(primer: Primer, min_sections: int = SECTIONS_MIN) -> list[str]:
    problems: list[str] = []
    for i, p in enumerate(primer.pages, 1):
        d = density(p, min_sections)
        if d.sections < min_sections or not d.ok:
            problems.append(f"page {i} '{p.title}': " + "; ".join(d.reasons()))
    return problems


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
        if p.blocks and not (VISUAL_SHARE_MIN <= share <= VISUAL_SHARE_MAX + 0.30):
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


def renumber_sections(primer: Primer) -> None:
    """Section pills read 1..n down each page, whatever the model emitted."""
    for p in primer.pages:
        for i, b in enumerate(p.sections(), 1):
            b["num"] = str(i)


# ── printable-height model ────────────────────────────────────────────────
# A page is locked to 210x297mm with overflow hidden, so content that does
# not fit is CLIPPED rather than flowing: a chart the model wrote and the
# user paid for would vanish without a word. These estimates (mm, derived
# from the template's own metrics) let the builder see that coming and say
# so, instead of shipping a page with its last blocks missing.

PAGE_BUDGET_MM = 208.0      # main is 222mm; the estimator runs ~6% light,
                            # so the budget keeps a margin rather than clipping
_LINE_MM = 3.72             # 10.25pt at line-height 1.42
_CHARS_PER_LINE = 96        # at ~186mm of content width


def _prose_mm(text: str | None, chars_per_line: int = _CHARS_PER_LINE) -> float:
    n = len(text or "")
    if not n:
        return 0.0
    import math
    return math.ceil(n / chars_per_line) * _LINE_MM


def block_height_mm(b: dict) -> float:
    """Estimated printed height of one block, including its bottom margin."""
    import math
    t = b.get("type")
    gap = 2.1
    if t == "section":
        return 8.4 + _prose_mm(b.get("body")) + gap
    if t == "bento":
        cols = b.get("columns") if b.get("columns") in (2, 3, 4) else 3
        cards = b.get("cards") or []
        rows = math.ceil(len(cards) / cols) if cards else 0
        per_col = max(1, _CHARS_PER_LINE // cols - 6)
        tallest = 0.0
        for r in range(rows):
            row = cards[r * cols:(r + 1) * cols]
            tallest += 4.4 + max((_prose_mm(c.get("body"), per_col) for c in row), default=0) + 4.4
        return tallest + gap
    if t == "table":
        head = 6.0 if b.get("title") else 0
        return head + 6.0 + 6.2 * len(b.get("rows") or []) + gap
    if t == "flow":
        return (5.0 if b.get("title") else 0) + 16 + gap
    if t == "steps":
        items = b.get("items") or []
        return ((5.0 if b.get("title") else 0)
                + sum(max(5.2, _prose_mm(str(s), 86) + 1.4) for s in items) + gap)
    if t == "hierarchy":
        levels = b.get("levels") or []
        return (5.0 if b.get("title") else 0) + len(levels) * 9.0 + gap
    if t == "chart":
        if not (b.get("bars") or []):
            return 0.0
        return (5.0 if b.get("title") else 0) + 34 + (4.0 if b.get("note") else 0) + gap
    if t == "code":
        lines = (b.get("code") or "").count("\n") + 1
        return 4.6 + lines * 3.0 + 4.0 + gap
    if t == "callout":
        return 4.6 + _prose_mm(b.get("body"), 90) + 3.6 + gap
    if t == "kv":
        return ((5.0 if b.get("title") else 0)
                + 3.6 * len(b.get("pairs") or []) + gap)
    if t == "cloud":
        return 16 + gap
    if t == "starburst":
        return 10 + gap
    return 0.0


def page_height_mm(page: Page) -> float:
    return sum(block_height_mm(b) for b in page.blocks)


def overflow_mm(page: Page) -> float:
    """How far past the printable area this page runs (0.0 when it fits)."""
    return max(0.0, page_height_mm(page) - PAGE_BUDGET_MM)


def _droppable(page: Page, must_have: list[str] | None = None) -> list[int]:
    """Indexes of blocks that are safe to remove, least valuable first.

    A block that is the ONLY one of a required kind, or one of the first
    five sections, is load-bearing: the density gate asked for it. Extras
    and duplicates are what a page can afford to lose.
    """
    required_now: set[int] = set()
    for allowed in required_kinds(must_have).values():
        present = [i for i, b in enumerate(page.blocks) if b.get("type") in allowed]
        if len(present) == 1:
            required_now.add(id(page.blocks[present[0]]))
    seen_sections = 0
    keep: set[int] = set(required_now)
    for b in page.blocks:
        if b.get("type") == "section":
            seen_sections += 1
            if seen_sections <= SECTIONS_MIN:
                keep.add(id(b))
    idx = [i for i, b in enumerate(page.blocks) if id(b) not in keep]
    idx.sort(key=lambda i: block_height_mm(page.blocks[i]), reverse=True)
    return idx


def fit_page(page: Page, must_have: list[str] | None = None) -> list[str]:
    """Shrink a page until it fits, reporting what was done.

    Prose is trimmed first, because a shorter paragraph still says what it
    said. Then redundant blocks go — extras and duplicates, never the only
    copy of something the density gate required. Dropping anything is
    reported: losing a block silently is the bug this function prevents.
    """
    notes: list[str] = []
    if overflow_mm(page) <= 0:
        return notes

    # 1. trim the longest prose bodies, down to 60% of their length
    prose = [b for b in page.blocks if b.get("type") == "section" and b.get("body")]
    prose.sort(key=lambda b: len(b["body"]), reverse=True)
    original = {id(b): len(b["body"]) for b in prose}
    for _ in range(12):
        if overflow_mm(page) <= 0:
            break
        for b in prose:
            if overflow_mm(page) <= 0:
                break
            if len(b["body"]) <= original[id(b)] * 0.45:
                continue
            words = b["body"].split()
            trimmed = " ".join(words[:max(20, int(len(words) * 0.9))])
            m = re.match(r"^(.+[.!?])[^.!?]*$", trimmed, re.S)
            b["body"] = m.group(1) if m else trimmed
    if any(len(b["body"]) < original[id(b)] for b in prose):
        notes.append("trimmed prose to fit the printable area")

    # 2. drop redundant blocks, largest first
    while overflow_mm(page) > 0:
        idx = _droppable(page, must_have)
        if not idx:
            break
        dropped = page.blocks.pop(idx[0])
        label = dropped.get("title") or dropped.get("text") or dropped.get("type")
        notes.append(f"dropped a redundant {dropped.get('type')} block ({str(label)[:40]!r}) "
                     "to fit the printable area")

    # 3. last resort: the page is over even stripped to essentials
    if overflow_mm(page) > 0:
        notes.append(f"page still exceeds the printable area by "
                     f"{overflow_mm(page):.0f}mm with only required blocks left — "
                     "content may be clipped; try --pages with a higher count")
    return notes
