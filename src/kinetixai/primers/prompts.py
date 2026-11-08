"""System prompts and JSON-schema enforcement for `kx primers build`.

Two modes:

* ``iterative`` (default) — one call per page. The model sees only that
  page's archetype from ``blueprint.py`` and its density quota, which is
  what lets a 7B model fill eight dense pages instead of emitting one
  paragraph per page and stopping.
* ``single`` — the whole primer in one call. Cheaper, and fine for a
  large model, but prone to sparse pages.

Either way the model gets the verified sources and is told it may cite
nothing else; citations are injected from fetched metadata, never from
the model.
"""
from __future__ import annotations

import json

from .blueprint import PageSpec, page_prompt_block, page_skeleton
from .schema import BLOCK_TYPES, SECTIONS_MAX, SECTIONS_MIN, WORDS_MAX, WORDS_MIN

BLOCK_REFERENCE = """\
{"type": "section",   "num": "1", "title": "...", "body": "<plain prose, no markdown>"}
{"type": "bento",     "columns": 3, "cards": [{"title": "...", "body": "..."}]}      // 3+ cards
{"type": "table",     "title": "...", "headers": ["..."], "rows": [["...", "..."]]}
{"type": "flow",      "title": "...", "steps": ["Fetch", "Decode", "Execute"]}       // left-to-right arrows
{"type": "steps",     "title": "...", "items": ["Set up", "Lock clocks", "Run"]}     // numbered, top-to-bottom
{"type": "hierarchy", "title": "...", "levels": [{"label": "Grid", "detail": "..."}]} // wide -> narrow
{"type": "chart",     "title": "...", "unit": "x", "note": "higher is better",
                      "bars": [{"label": "Baseline", "value": 1.0}, {"label": "Fused", "value": 2.1}]}
{"type": "code",      "title": "...", "lang": "bash|python|cuda", "code": "..."}
{"type": "callout",   "style": "trap|tip|security|mnemonic|benchmark|misreading|warning",
                      "title": "...", "body": "..."}
{"type": "cloud",     "text": "<short punchy line, the comic cloud>"}
{"type": "kv",        "title": "...", "pairs": [["term", "definition"]]}
{"type": "starburst", "text": "Story to Silicon!"}"""

_RULES = f"""\
HARD RULES
1. Reply with ONE JSON object only — no markdown fences, no commentary, no reasoning.
2. Allowed block types: {sorted(BLOCK_TYPES)}. Nothing else.
3. Prose is plain text: no markdown, no LaTeX delimiters. Write maths in words or
   simple inline notation (O(n^2), w=16).
4. Every factual claim must come from the SOURCE MATERIAL or the VERIFIED SOURCES
   you are given. Never invent citations, DOIs, benchmark numbers, customers or
   quotes. If a number is not in the sources, describe the mechanism without it.
5. Benchmarks use the Three Honest Columns: (1) kernel/compute only, (2) + host-device
   or network transfer, (3) end-to-end wall time. Never a single flattering speedup.
6. Coin at most one mnemonic per primer (callout style "mnemonic"), derivable from
   the content."""

DENSITY_RULES = f"""\
DENSITY QUOTA — a page that misses this is rejected and regenerated
* {SECTIONS_MIN} to {SECTIONS_MAX} "section" blocks, numbered "1" upward, each with real prose.
* {WORDS_MIN}-{WORDS_MAX} words on the page in total, counting table and card text.
* At least one diagram (flow, steps or hierarchy), one table, one bento with 3+
  cards, one code block, and one cloud callout.
* Fill the page. A page with one paragraph and white space is a failure.

SPACE BUDGET — the page is a fixed A4 sheet and anything past it is cut
This is the constraint people get wrong. One printed page holds roughly
10 to 13 blocks in total, so write tight:
* Each "section" body: 30-45 words. Two sentences, not a lecture.
* 5 to 7 visual blocks in total, not more. Pick the ones that carry meaning.
* A chart is expensive (a quarter of the page). Use at most two per page, and
  only where numbers from the sources justify one.
* A table of 4-6 rows, a bento of 3 cards, a code block of 2-4 lines.
Going over does not make a richer page — it makes a page with its last
blocks missing."""

BLOCK_SPEC = f"""\
BLOCK REFERENCE (exact shapes)
{BLOCK_REFERENCE}"""

SYSTEM_PROMPT = f"""You are the KinetixAI Primer engine ("Story to Silicon"). You turn a topic
plus verified source material into a page-by-page layout model for an A4 print
renderer, in the TriFleetRCA house style: dense, structured, visual pages.

{_RULES}

{DENSITY_RULES}

{BLOCK_SPEC}
"""

PAGE_SYSTEM_PROMPT = f"""You are the KinetixAI Primer engine ("Story to Silicon"). You write ONE page
of an A4 print primer in the TriFleetRCA house style: dense, structured, visual.

Reply with ONE JSON object for this single page:
{{"title": "...", "subtitle": "...", "takeaway": "<one sentence for the footer bar>",
  "blocks": [ ... ]}}

{_RULES}

{DENSITY_RULES}

{BLOCK_SPEC}
"""

# Persona modules — appended per --persona flag.
PERSONAS: dict[str, str] = {
    "student": (
        "PERSONA student: start each concept with plain-language intuition before "
        "matrix calculus or CUDA/PyTorch code; include a callout with style 'trap' "
        "carrying DSA/systems interview traps (complexity, memory hierarchy, warp "
        "divergence); include flashcard-worthy kv blocks."
    ),
    "phd": (
        "PERSONA phd: include a table comparing related work (Work, Year, Mechanism, "
        "Hardware, Limitations) built ONLY from the verified sources; mark empirical vs "
        "theoretical claims explicitly; add a callout with style 'tip' naming plausible "
        "target venues (e.g. NeurIPS, ICSE, ASE, USENIX) based on the artifact profile."
    ),
    "researcher": (
        "PERSONA researcher: frame the research gap — what the verified sources show "
        "exists vs what is missing. Phrase gaps as 'not found in the surveyed sources', "
        "never as established fact."
    ),
    "professor": (
        "PERSONA professor: structure content so the pages map to a 45-minute lecture; "
        "include a callout with style 'misreading' listing the top 3 student "
        "misconceptions with the correct counter-statement."
    ),
    "professional": (
        "PERSONA professional: enforce Three Honest Columns on any speedup claim; add a "
        "callout with style 'benchmark' carrying an observability recipe (eBPF uprobes, "
        "always-on 1 Hz telemetry via DCGM/Prometheus, deep profiling on idle boxes)."
    ),
    "lab": (
        "PERSONA lab: include a reproducibility code block with lock-clock/deterministic "
        "environment commands and a CI gate note ('gate on accuracy before speed sweeps')."
    ),
    "ai-security": (
        "PERSONA ai-security: add a callout with style 'security' carrying a threat matrix "
        "touching RAG poisoning, prompt injection, unvalidated runbook execution and "
        "lineage tampering, plus the guarded-pipeline counterpart (ingest guards, "
        "retrieval sanitization, zero-trust process lineage, on-device masking)."
    ),
}
PERSONAS["security"] = PERSONAS["ai-security"]


def persona_text(personas: list[str]) -> str:
    seen, out = set(), []
    for p in personas:
        body = PERSONAS.get(p.strip().lower())
        if body and body not in seen:
            seen.add(body)
            out.append(body)
    return "\n".join(out) or "PERSONA: general technical reader."


def _sources_json(sources: list[dict], limit: int = 12) -> str:
    rows = [{k: p.get(k) for k in ("title", "authors", "year", "arxiv_id", "doi", "abstract")}
            for p in sources[:limit]]
    return json.dumps(rows, ensure_ascii=False, indent=1)[:12000]


def build_user_prompt(topic: str, pages: int, personas: list[str],
                      source_text: str, sources: list[dict]) -> str:
    """Whole-primer prompt (single-call mode)."""
    parts = [
        f"TOPIC: {topic}",
        f"PAGES: exactly {pages} pages, each meeting the density quota.",
        persona_text(personas),
        "VERIFIED SOURCES (the only citable works):",
        _sources_json(sources),
    ]
    if source_text:
        parts.append("SOURCE MATERIAL (from the user's file/repo/paper):\n" + source_text[:20000])
    parts.append("Return the JSON object now.")
    return "\n\n".join(parts)


def build_page_prompt(topic: str, spec: PageSpec, index: int, total: int,
                      personas: list[str], source_text: str, sources: list[dict],
                      previous_titles: list[str]) -> str:
    """Per-page prompt (iterative mode)."""
    parts = [
        f"TOPIC: {topic}",
        page_prompt_block(spec, index, total, topic),
        persona_text(personas),
    ]
    if previous_titles:
        parts.append("Pages already written (do not repeat their content):\n- "
                     + "\n- ".join(previous_titles))
    parts.append("VERIFIED SOURCES (the only citable works):\n" + _sources_json(sources, 8))
    if source_text:
        parts.append("SOURCE MATERIAL (from the user's file/repo/paper):\n"
                     + source_text[:14000])
    # last, because it is the instruction that matters most and recency wins
    parts.append(page_skeleton(spec))
    parts.append("Return the JSON object for this one page now.")
    return "\n\n".join(parts)


def expand_prompt(page_json: dict, missing: list[str], spec: PageSpec,
                  topic: str, index: int, total: int) -> str:
    """Ask the model to complete a sparse page instead of discarding it."""
    return "\n\n".join([
        f"TOPIC: {topic}",
        f"This is page {index} of {total} ({spec.role}). The page below is too sparse "
        "to print — it would leave most of an A4 page blank.",
        "WHAT IS MISSING:\n- " + "\n- ".join(missing),
        "KEEP every existing block exactly as it is, then ADD the blocks it lacks.",
        "THE PAGE SO FAR:\n" + json.dumps(page_json, ensure_ascii=False)[:9000],
        page_skeleton(spec),
        "Return the COMPLETE page as one JSON object (existing blocks plus the new ones).",
    ])


def repair_prompt(problems: list[str]) -> str:
    return ("Your previous JSON violated the layout budget. Fix ONLY these pages and "
            "return the FULL corrected JSON object again:\n- " + "\n- ".join(problems))
