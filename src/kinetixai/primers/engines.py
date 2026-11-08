"""LLM engine router for the Primer builder.

Engines: local-vllm / vllm (OpenAI-compatible), ollama, openai,
anthropic / claude, gemini, mcp (delegate to the host agent via MCP
sampling), and ``outline`` — a deterministic no-LLM fallback that
structures the source material into valid pages so the pipeline runs,
and is testable, offline.

Keys come from the environment only: OPENAI_API_KEY, ANTHROPIC_API_KEY,
GEMINI_API_KEY. Endpoints: --api-base, else KX_VLLM_URL / VLLM_BASE_URL
(default http://localhost:8000/v1) for vLLM and the configured
ollama_url for Ollama. Nothing is ever logged.
"""
from __future__ import annotations

import json
import os
import re
import textwrap

import httpx

from .. import config
from .schema import (WORDS_MAX, Page, Primer, PrimerSchemaError, page_from_dict,
                     primer_from_dict)

# Canonical names, plus the aliases people actually type.
ALIASES = {"vllm": "local-vllm", "claude": "anthropic", "google": "gemini",
           "gpt": "openai", "none": "outline", "offline": "outline"}
ENGINES = ("local-vllm", "ollama", "openai", "anthropic", "gemini", "mcp", "outline")
ALL_ENGINE_NAMES = tuple(sorted(set(ENGINES) | set(ALIASES)))
TIMEOUT = httpx.Timeout(300.0, connect=15.0)

# Reasoning models (qwen3, deepseek-r1, ...) narrate before answering.
_THINK = re.compile(r"<(think|thinking|reasoning|scratchpad)>.*?</\1>", re.S | re.I)
_THINK_OPEN = re.compile(r"<(?:think|thinking|reasoning|scratchpad)>.*$", re.S | re.I)
MAX_TOKENS = 16000


class EngineError(RuntimeError):
    pass


def canonical(engine: str) -> str:
    name = (engine or "").strip().lower()
    return ALIASES.get(name, name)


def _strip_fences(text: str) -> str:
    text = _THINK.sub("", text or "")
    text = _THINK_OPEN.sub("", text)  # truncated: opening tag, no close
    m = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def _balanced(text: str) -> str | None:
    """First complete {...} object, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse(text: str) -> dict:
    cleaned = _strip_fences(text)
    candidate = _balanced(cleaned)
    if candidate is None:
        snippet = (text or "").strip()[:200].replace("\n", " ")
        if not snippet:
            raise EngineError(
                "the model returned an empty response. Reasoning models often spend the "
                "whole budget on <think>; try a non-reasoning model (qwen2.5, llama3.1) "
                "or raise the model's context length.")
        raise EngineError(f"the model returned no JSON object. It said: {snippet!r}")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise EngineError(f"the model returned invalid JSON ({exc}).")


def _message_text(payload: dict, where: str) -> str:
    """Content from an OpenAI-shaped reply, tolerating reasoning fields."""
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError):
        raise EngineError(f"{where}: unexpected response shape: {str(payload)[:200]}")
    msg = choice.get("message") or {}
    text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not text.strip() and choice.get("finish_reason") == "length":
        raise EngineError(
            f"{where}: the response hit the token limit before producing any content. "
            "Raise the model's context window (Ollama: a Modelfile with a larger "
            "num_ctx) or use a non-reasoning model.")
    return text


def _openai_compatible(base: str, model: str, system: str, user: str,
                       key: str | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.3,
            "max_tokens": MAX_TOKENS,
            # JSON mode: supported by OpenAI, vLLM and Ollama's /v1 endpoint.
            "response_format": {"type": "json_object"}}
    url = f"{base.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(url, json=body, headers=headers)
        if r.status_code == 400 and "response_format" in r.text:
            body.pop("response_format")  # endpoint predates JSON mode
            r = client.post(url, json=body, headers=headers)
        if r.status_code >= 400:
            raise EngineError(f"LLM endpoint {base}: HTTP {r.status_code}: {r.text[:300]}")
        return _message_text(r.json(), f"LLM endpoint {base}")


def _anthropic(model: str, system: str, user: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EngineError("set ANTHROPIC_API_KEY to use --engine claude.")
    body = {"model": model, "max_tokens": MAX_TOKENS, "system": system,
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
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "maxOutputTokens": MAX_TOKENS}}
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(url, json=body)
        if r.status_code >= 400:
            raise EngineError(f"Gemini API: HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise EngineError(f"Gemini API: unexpected response: {str(data)[:200]}")


# ── MCP sampling bridge ───────────────────────────────────────────────────
# When kx runs as an MCP server, a tool can ask the HOST agent (Claude Code,
# Cursor, Claude Desktop) to run the completion on its own model. The host
# sets this hook for the duration of a tool call; --engine mcp then needs no
# API key of its own.

_SAMPLER = None


def set_sampler(fn) -> None:
    """Install the host's sampling callback: fn(system, user) -> str."""
    global _SAMPLER
    _SAMPLER = fn


def sampler_available() -> bool:
    return _SAMPLER is not None


def _mcp(system: str, user: str) -> str:
    if _SAMPLER is None:
        raise EngineError(
            "--engine mcp delegates generation to the host agent and only works while "
            "kx is running as an MCP server (kx mcp serve) with a client that supports "
            "sampling. From a normal shell, use --engine ollama, vllm, claude, gemini "
            "or openai.")
    try:
        return _SAMPLER(system, user)
    except EngineError:
        raise
    except Exception as exc:  # noqa: BLE001 — the host's failure, reported as ours
        raise EngineError(f"the host agent's sampling call failed: {exc}")


def complete(engine: str, model: str | None, system: str, user: str,
             api_base: str | None = None) -> str:
    name = canonical(engine)
    if name == "local-vllm":
        base = (api_base or os.environ.get("KX_VLLM_URL")
                or os.environ.get("VLLM_BASE_URL") or "http://localhost:8000/v1")
        return _openai_compatible(base, model or "Qwen/Qwen2.5-14B-Instruct", system, user,
                                  key=os.environ.get("KX_VLLM_KEY"))
    if name == "ollama":
        base = api_base or (os.environ.get("OLLAMA_HOST")
                            or config.get("ollama_url", "http://localhost:11434"))
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return _openai_compatible(base, model or config.get("ollama_model", "llama3.1"),
                                  system, user)
    if name == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EngineError("set OPENAI_API_KEY to use --engine openai.")
        return _openai_compatible(api_base or "https://api.openai.com/v1",
                                  model or "gpt-4o-mini", system, user, key=key)
    if name == "anthropic":
        return _anthropic(model or "claude-sonnet-4-5", system, user)
    if name == "gemini":
        return _gemini(model or "gemini-2.0-flash", system, user)
    if name == "mcp":
        return _mcp(system, user)
    raise EngineError(f"unknown engine '{engine}'. Engines: {', '.join(ALL_ENGINE_NAMES)}")


RETRY_NUDGE = ("\n\nYour last reply could not be parsed. Reply with the JSON object "
               "ALONE: no commentary, no reasoning, no markdown fences. Start with "
               "{ and end with }.")


def _attempt(engine, model, system, user, api_base, retries, parse):
    """Shared retry loop: chatty and reasoning models produce a malformed
    reply often enough that absorbing one retry here beats failing a build."""
    attempt, last = 0, None
    while attempt <= retries:
        prompt = user if attempt == 0 else user + RETRY_NUDGE
        raw = complete(engine, model, system, prompt, api_base=api_base)
        try:
            return parse(_parse(raw))
        except (EngineError, PrimerSchemaError) as exc:
            last = exc if isinstance(exc, EngineError) else EngineError(
                f"the model's JSON failed schema validation: {exc}")
            attempt += 1
    raise last


def synthesize(engine: str, model: str | None, system: str, user: str,
               retries: int = 1, api_base: str | None = None) -> Primer:
    """One LLM call -> parsed, schema-checked Primer (whole document)."""
    return _attempt(engine, model, system, user, api_base, retries, primer_from_dict)


def synthesize_page(engine: str, model: str | None, system: str, user: str,
                    retries: int = 1, api_base: str | None = None) -> Page:
    """One LLM call -> parsed, schema-checked single Page."""
    def parse(data: dict) -> Page:
        # tolerate a model that wraps the page in {"pages": [...]} anyway
        if "blocks" not in data and isinstance(data.get("pages"), list) and data["pages"]:
            data = data["pages"][0]
        return page_from_dict(data)
    return _attempt(engine, model, system, user, api_base, retries, parse)


# ── offline outliner (no LLM) ─────────────────────────────────────────────

def outline(topic: str, pages: int, personas: list[str],
            source_text: str, sources: list[dict],
            watermark: str) -> Primer:
    """Deterministic structure-only primer: splits the source material into
    page-sized sections with tables/flows built from headings and sources.
    Clearly labelled as a draft — no claims are invented, only reorganized."""
    from . import blueprint  # local import: blueprint imports nothing from here

    cleaned = re.sub(r"^#{1,6}\s*", "", source_text or "", flags=re.M)
    paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paras:
        paras = [f"No source material was provided for '{topic}'. This draft contains "
                 f"layout scaffolding only; re-run with --engine and a model, or pass "
                 f"--source, to fill it with verified content."]
    specs = blueprint.plan(pages)
    per_page = max(1, len(paras) // len(specs) or 1)
    primer = Primer(title=topic, subtitle="Draft outline (no-LLM mode)",
                    watermark=watermark, topic=topic, personas=personas)
    src_rows = [[p.get("title", "")[:60], str(p.get("year") or ""),
                 p.get("arxiv_id") or p.get("doi") or ""] for p in sources[:10]]
    for i, spec in enumerate(specs):
        chunk = paras[i * per_page:(i + 1) * per_page] or ["(no further source text)"]
        body = textwrap.shorten(" ".join(chunk), width=WORDS_MAX * 6, placeholder=" …")
        blocks: list[dict] = [
            {"type": "section", "num": "1", "title": spec.sections[0], "body": body},
            {"type": "flow", "title": "Reading path", "steps": ["Story", "Math", "Silicon"]},
            {"type": "bento", "columns": 3, "cards": [
                {"title": s, "body": "To be written from the source material."}
                for s in spec.sections[1:4]]},
        ]
        if i == 0 and src_rows:
            blocks.append({"type": "table", "title": "Verified sources",
                           "headers": ["Work", "Year", "ID"], "rows": src_rows})
        blocks.append({"type": "cloud", "text": "Draft — verify before publishing!"})
        primer.pages.append(
            Page(title=f"{topic} — {spec.role}", subtitle="Draft outline",
                 takeaway="Draft outline page — verify before publishing.", blocks=blocks))
    return primer
