"""LLM engine adapters for the Primer builder.

Engines: local-vllm (OpenAI-compatible), ollama, openai, anthropic, gemini —
plus ``outline``, a deterministic no-LLM fallback that structures the source
material into valid pages so the pipeline runs (and is testable) offline.

Keys come from the environment only: OPENAI_API_KEY, ANTHROPIC_API_KEY,
GEMINI_API_KEY. Endpoints: KX_VLLM_URL (default http://localhost:8000/v1),
config ollama_url for Ollama. Nothing is ever logged.
"""
from __future__ import annotations

import json
import os
import re
import textwrap

import httpx

from .. import config
from .schema import WORDS_MAX, Page, Primer, PrimerSchemaError, primer_from_dict

ENGINES = ("local-vllm", "ollama", "openai", "anthropic", "gemini", "outline")
TIMEOUT = httpx.Timeout(300.0, connect=15.0)


class EngineError(RuntimeError):
    pass


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _parse(text: str) -> dict:
    text = _strip_fences(text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise EngineError("the model returned no JSON object.")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise EngineError(f"the model returned invalid JSON ({exc}).")


def _openai_compatible(base: str, model: str, system: str, user: str,
                       key: str | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.3}
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(f"{base.rstrip('/')}/chat/completions", json=body, headers=headers)
        if r.status_code >= 400:
            raise EngineError(f"LLM endpoint {base}: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]["content"]


def _anthropic(model: str, system: str, user: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EngineError("set ANTHROPIC_API_KEY to use --engine anthropic.")
    body = {"model": model, "max_tokens": 16000, "system": system,
            "messages": [{"role": "user", "content": user}]}
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post("https://api.anthropic.com/v1/messages", json=body,
                        headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        if r.status_code >= 400:
            raise EngineError(f"Anthropic API: HTTP {r.status_code}: {r.text[:300]}")
        return "".join(b.get("text", "") for b in r.json().get("content", []))


def _gemini(model: str, system: str, user: str) -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EngineError("set GEMINI_API_KEY to use --engine gemini.")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}]}
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(url, json=body)
        if r.status_code >= 400:
            raise EngineError(f"Gemini API: HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def complete(engine: str, model: str | None, system: str, user: str) -> str:
    if engine == "local-vllm":
        base = os.environ.get("KX_VLLM_URL", "http://localhost:8000/v1")
        return _openai_compatible(base, model or "Qwen/Qwen2.5-14B-Instruct", system, user,
                                  key=os.environ.get("KX_VLLM_KEY"))
    if engine == "ollama":
        base = config.get("ollama_url", "http://localhost:11434").rstrip("/") + "/v1"
        return _openai_compatible(base, model or config.get("ollama_model", "llama3.1"),
                                  system, user)
    if engine == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EngineError("set OPENAI_API_KEY to use --engine openai.")
        return _openai_compatible("https://api.openai.com/v1", model or "gpt-4o-mini",
                                  system, user, key=key)
    if engine == "anthropic":
        return _anthropic(model or "claude-sonnet-4-5", system, user)
    if engine == "gemini":
        return _gemini(model or "gemini-2.0-flash", system, user)
    raise EngineError(f"unknown engine '{engine}'. Engines: {', '.join(ENGINES)}")


def synthesize(engine: str, model: str | None, system: str, user: str) -> Primer:
    """One LLM call -> parsed, schema-checked Primer."""
    raw = complete(engine, model, system, user)
    data = _parse(raw)
    try:
        return primer_from_dict(data)
    except PrimerSchemaError as exc:
        raise EngineError(f"the model's JSON failed schema validation: {exc}")


# ── offline outliner (no LLM) ─────────────────────────────────────────────

def outline(topic: str, pages: int, personas: list[str],
            source_text: str, sources: list[dict],
            watermark: str) -> Primer:
    """Deterministic structure-only primer: splits the source material into
    page-sized sections with tables/flows built from headings and sources.
    Clearly labelled as a draft — no claims are invented, only reorganized."""
    cleaned = re.sub(r"^#{1,6}\s*", "", source_text or "", flags=re.M)
    paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paras:
        paras = [f"No source material was provided for '{topic}'. This draft contains "
                 f"layout scaffolding only; re-run with --engine and a model, or pass "
                 f"--source, to fill it with verified content."]
    per_page = max(1, len(paras) // pages or 1)
    primer = Primer(title=topic, subtitle="Draft outline (no-LLM mode)",
                    watermark=watermark, topic=topic, personas=personas)
    src_rows = [[p.get("title", "")[:60], str(p.get("year") or ""),
                 p.get("arxiv_id") or p.get("doi") or ""] for p in sources[:10]]
    for i in range(pages):
        chunk = paras[i * per_page:(i + 1) * per_page] or ["(empty)"]
        body = textwrap.shorten(" ".join(chunk), width=WORDS_MAX * 6, placeholder=" …")
        blocks = [
            {"type": "section", "num": str(i + 1), "title": f"Part {i + 1}", "body": body},
            {"type": "flow", "title": "Reading path", "steps": ["Story", "Math", "Silicon"]},
        ]
        if i == 0 and src_rows:
            blocks.append({"type": "table", "title": "Verified sources",
                           "headers": ["Work", "Year", "ID"], "rows": src_rows})
        primer.pages.append(  # takeaway kept honest: it is a draft
            Page(title=f"{topic} — part {i + 1}", subtitle="Draft outline",
                 takeaway="Draft outline page — verify before publishing.", blocks=blocks))
    return primer
