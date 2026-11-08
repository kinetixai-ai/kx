"""System prompt and JSON-schema enforcement for `kx primers build`.

The model is asked for ONE thing: a JSON object matching the page model in
``schema.py``. Nothing free-form ever reaches the renderer, and citations
are injected from verified metadata (kx search / resolve) — the model is
told explicitly it may only reference the sources it was given.
"""
from __future__ import annotations

import json

from .schema import BLOCK_TYPES, WORDS_MAX, WORDS_MIN

PAGE_JSON_SPEC = """\
{
  "title": "<primer title>",
  "subtitle": "<one-line subtitle>",
  "pages": [
    {
      "title": "<page title, uppercase-friendly>",
      "subtitle": "<page strapline>",
      "takeaway": "<ONE sentence Key Takeaway for the footer bar>",
      "blocks": [
        {"type": "section", "num": "1", "title": "...", "body": "<plain prose, no markdown>"},
        {"type": "bento", "columns": 3, "cards": [{"title": "...", "body": "...", "icon": "gpu|ram|net|lock|math|code|flow|idea"}]},
        {"type": "table", "title": "...", "headers": ["..."], "rows": [["...", "..."]]},
        {"type": "flow", "title": "...", "steps": ["Fetch", "Decode", "Execute"]},
        {"type": "code", "title": "...", "lang": "bash|python|cuda", "code": "..."},
        {"type": "callout", "style": "trap|tip|security|mnemonic|benchmark|misreading", "title": "...", "body": "..."},
        {"type": "kv", "title": "...", "pairs": [["term", "definition"]]},
        {"type": "starburst", "text": "Story to Silicon!"}
      ]
    }
  ]
}"""

SYSTEM_PROMPT = f"""You are the KinetixAI Primer engine ("Story to Silicon"). You convert a topic
plus verified source material into a page-by-page layout model for an A4
print renderer.

OUTPUT CONTRACT (hard rules):
1. Reply with ONE JSON object only — no markdown fences, no commentary.
2. Follow this exact shape:
{PAGE_JSON_SPEC}
3. Allowed block types: {sorted(BLOCK_TYPES)}. Nothing else.
4. Each page must carry {WORDS_MIN}-{WORDS_MAX} words total and 35-45% of its
   blocks must be visual/structured (bento, table, flow, code, kv, starburst).
5. Page 1 is the executive overview: plain-language visual intuition BEFORE
   any mathematics or code ("Story to Silicon" scaffolding).
6. Every factual claim must come from the SOURCE MATERIAL or the VERIFIED
   SOURCES list you are given. Never invent citations, DOIs, benchmark
   numbers, customers or quotes. If a number is not in the sources, write
   the mechanism without the number.
7. Benchmarks, when present, use the Three Honest Columns: (1) kernel/compute
   only, (2) + host-device/network transfer, (3) end-to-end wall time.
8. Mnemonics: coin at most one catchy mnemonic per primer (callout style
   "mnemonic") and make it derivable from the content, e.g. first letters of
   pipeline stages.
9. Prose is plain text: no markdown, no LaTeX delimiters; write math in
   words or simple inline notation (O(n^2), w=16).
"""

# Persona modules — appended to the system prompt per --persona flag.
PERSONAS: dict[str, str] = {
    "student": (
        "PERSONA student: start each concept with plain-language intuition before "
        "matrix calculus or CUDA/PyTorch code; add one 'callout' style 'trap' per "
        "primer with DSA/Systems interview traps (complexity, memory hierarchy, warp "
        "divergence); include flashcard-worthy kv blocks."
    ),
    "phd": (
        "PERSONA phd: include a 'table' comparing related work (Work, Year, Mechanism, "
        "Hardware, Limitations) built ONLY from the verified sources; mark empirical vs "
        "theoretical claims explicitly; add a 'callout' style 'tip' naming plausible "
        "target venues (e.g. NeurIPS, ICSE, ASE, USENIX) based on the artifact profile."
    ),
    "researcher": (
        "PERSONA researcher: add a research-gap framing — what the verified sources "
        "show exists vs what is missing; never present the gap as established fact, "
        "phrase it as 'not found in the surveyed sources'."
    ),
    "professor": (
        "PERSONA professor: structure content so pages map to a 45-minute lecture; add "
        "one 'callout' style 'misreading' listing the top 3 student misconceptions on "
        "the topic with the correct counter-statement."
    ),
    "professional": (
        "PERSONA professional: enforce Three Honest Columns on any speedup claim; add "
        "a 'callout' style 'benchmark' with an observability recipe (eBPF uprobes, "
        "always-on 1 Hz telemetry via DCGM/Prometheus, deep profiling on idle boxes) "
        "with a Setup Traps & Fixes table where relevant."
    ),
    "lab": (
        "PERSONA lab: include a reproducibility block — a 'code' block with lock-clock/"
        "deterministic environment commands and a CI gate note ('gate on accuracy "
        "before speed sweeps')."
    ),
    "ai-security": (
        "PERSONA ai-security: add a 'callout' style 'security' threat matrix touching "
        "RAG poisoning, prompt injection, unvalidated runbook execution and lineage "
        "tampering, plus the guarded-pipeline counterpart (ingest guards, retrieval "
        "sanitization, zero-trust process lineage, on-device masking)."
    ),
}
# aliases
PERSONAS["security"] = PERSONAS["ai-security"]


def build_user_prompt(topic: str, pages: int, personas: list[str],
                      source_text: str, sources: list[dict]) -> str:
    src_lines = [
        {k: p.get(k) for k in ("title", "authors", "year", "arxiv_id", "doi", "url", "abstract")}
        for p in sources
    ]
    persona_text = "\n".join(PERSONAS[p] for p in personas if p in PERSONAS)
    parts = [
        f"TOPIC: {topic}",
        f"PAGES: exactly {pages} pages.",
        persona_text or "PERSONA: general technical reader.",
        "VERIFIED SOURCES (the only citable works):",
        json.dumps(src_lines, ensure_ascii=False, indent=1)[:12000],
    ]
    if source_text:
        parts.append("SOURCE MATERIAL (extracted from the user's file/repo/paper):\n"
                     + source_text[:24000])
    parts.append("Return the JSON object now.")
    return "\n\n".join(parts)


def repair_prompt(problems: list[str]) -> str:
    return ("Your previous JSON violated the layout budget. Fix ONLY these pages and "
            "return the FULL corrected JSON object again:\n- " + "\n- ".join(problems))
