"""The engines a Primer can be written with.

Almost every inference server and hosted API now speaks the OpenAI chat
format, so a provider here is mostly a name, a default endpoint and the
environment variable holding its key. That is what makes breadth cheap:
self-hosted llama.cpp, LM Studio, Jan, LocalAI, vLLM and TGI sit beside
Groq, Together, OpenRouter, DeepSeek and the big three without a special
case each.

Three providers are not OpenAI-shaped and are handled natively in
``engines.py``: ollama (its own /api/chat, so num_ctx can be set),
anthropic (the messages API) and gemini (generateContent).

Endpoints move. Every one of these can be overridden with ``--api-base``
or the provider's own ``*_BASE_URL`` environment variable, so a changed
URL is a flag away rather than a new release.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    base: str                      # default endpoint
    keys: tuple[str, ...] = ()     # API key env vars; empty means none needed
    default_model: str = ""
    local: bool = False            # runs on the user's own hardware
    protocol: str = "openai"       # openai | ollama | anthropic | gemini | mcp | none
    base_env: str = ""             # env var that overrides the endpoint
    note: str = ""
    aliases: tuple[str, ...] = ()


_P: list[Provider] = [
    # ── self-hosted / open source ────────────────────────────────────────
    Provider("ollama", "Ollama", "http://localhost:11434", (), "llama3.1",
             local=True, protocol="ollama", base_env="OLLAMA_HOST",
             note="Native API, so kx can raise num_ctx past Ollama's 4096 default."),
    Provider("vllm", "vLLM", "http://localhost:8000/v1", ("KX_VLLM_KEY",),
             "Qwen/Qwen2.5-14B-Instruct", local=True, base_env="KX_VLLM_URL",
             note="Best local quality on a big GPU.", aliases=("local-vllm",)),
    Provider("llamacpp", "llama.cpp server", "http://localhost:8080/v1", (), "",
             local=True, base_env="LLAMACPP_BASE_URL",
             note="llama-server --host 0.0.0.0 --port 8080.", aliases=("llama-cpp", "llama")),
    Provider("lmstudio", "LM Studio", "http://localhost:1234/v1", (), "",
             local=True, base_env="LMSTUDIO_BASE_URL",
             note="Turn on the local server in LM Studio's Developer tab.",
             aliases=("lm-studio",)),
    Provider("jan", "Jan", "http://localhost:1337/v1", (), "",
             local=True, base_env="JAN_BASE_URL", note="Jan's local API server."),
    Provider("localai", "LocalAI", "http://localhost:8080/v1", (), "",
             local=True, base_env="LOCALAI_BASE_URL",
             note="Self-hosted, OpenAI-compatible, runs GGUF and more."),
    Provider("tgi", "Hugging Face TGI", "http://localhost:3000/v1", ("HF_TOKEN",), "",
             local=True, base_env="TGI_BASE_URL",
             note="Text Generation Inference, self-hosted.",
             aliases=("text-generation-inference",)),
    Provider("textgen", "text-generation-webui", "http://localhost:5000/v1", (), "",
             local=True, base_env="TEXTGEN_BASE_URL",
             note="oobabooga, with --api enabled.", aliases=("oobabooga",)),
    Provider("llamafile", "llamafile", "http://localhost:8080/v1", (), "",
             local=True, base_env="LLAMAFILE_BASE_URL",
             note="A single-file local model server."),

    # ── hosted ───────────────────────────────────────────────────────────
    Provider("anthropic", "Anthropic (Claude)", "https://api.anthropic.com",
             ("ANTHROPIC_API_KEY",), "claude-sonnet-4-5", protocol="anthropic",
             note="Richest diagrams and tables in testing.", aliases=("claude",)),
    Provider("openai", "OpenAI (ChatGPT models)", "https://api.openai.com/v1",
             ("OPENAI_API_KEY",), "gpt-4o-mini", base_env="OPENAI_BASE_URL",
             aliases=("chatgpt", "gpt")),
    Provider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com",
             ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "gemini-2.0-flash",
             protocol="gemini", aliases=("google",)),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
             ("OPENROUTER_API_KEY",), "meta-llama/llama-3.3-70b-instruct",
             base_env="OPENROUTER_BASE_URL",
             note="One key, most open models."),
    # No pinned default: Groq rotates its catalogue and retires ids, so a
    # pinned name 404s on every page. The served list is the source of truth.
    Provider("groq", "Groq", "https://api.groq.com/openai/v1", ("GROQ_API_KEY",),
             "", base_env="GROQ_BASE_URL",
             note="Open models, very fast."),
    Provider("together", "Together AI", "https://api.together.xyz/v1",
             ("TOGETHER_API_KEY",), "meta-llama/Llama-3.3-70B-Instruct-Turbo",
             base_env="TOGETHER_BASE_URL", note="Open models, hosted."),
    Provider("deepseek", "DeepSeek", "https://api.deepseek.com/v1",
             ("DEEPSEEK_API_KEY",), "deepseek-chat", base_env="DEEPSEEK_BASE_URL"),
    Provider("mistral", "Mistral", "https://api.mistral.ai/v1",
             ("MISTRAL_API_KEY",), "mistral-large-latest", base_env="MISTRAL_BASE_URL"),
    Provider("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1",
             ("FIREWORKS_API_KEY",), "accounts/fireworks/models/llama-v3p3-70b-instruct",
             base_env="FIREWORKS_BASE_URL"),
    Provider("cerebras", "Cerebras", "https://api.cerebras.ai/v1",
             ("CEREBRAS_API_KEY",), "llama-3.3-70b", base_env="CEREBRAS_BASE_URL"),
    Provider("xai", "xAI (Grok)", "https://api.x.ai/v1", ("XAI_API_KEY",),
             "grok-2-latest", base_env="XAI_BASE_URL", aliases=("grok",)),

    # ── not a model of its own ───────────────────────────────────────────
    Provider("mcp", "The calling agent (MCP sampling)", "", (), "", protocol="mcp",
             note="Claude Code, Cursor or any MCP host writes the pages. No key needed."),
    Provider("outline", "No model (deterministic draft)", "", (), "", protocol="none",
             local=True, note="Structures your source material. Clearly labelled a draft.",
             aliases=("none", "offline")),
]

PROVIDERS: dict[str, Provider] = {p.name: p for p in _P}
ALIASES: dict[str, str] = {a: p.name for p in _P for a in p.aliases}
NAMES: tuple[str, ...] = tuple(PROVIDERS)
ALL_NAMES: tuple[str, ...] = tuple(sorted(set(NAMES) | set(ALIASES)))

# Tried in this order when --engine is left out: a local server costs
# nothing per page, so it is preferred over a metered API.
AUTO_ORDER = ("ollama", "vllm", "lmstudio", "llamacpp", "jan", "localai", "textgen",
              "anthropic", "openai", "gemini", "groq", "openrouter", "together",
              "deepseek", "mistral", "fireworks", "cerebras", "xai")


def resolve(name: str) -> str:
    """'claude' -> 'anthropic', 'local-vllm' -> 'vllm'."""
    key = (name or "").strip().lower()
    return ALIASES.get(key, key)


def get(name: str) -> Provider | None:
    return PROVIDERS.get(resolve(name))


def base_url(p: Provider, override: str | None = None) -> str:
    if override:
        return override
    if p.base_env and os.environ.get(p.base_env):
        return os.environ[p.base_env]
    if p.name == "vllm" and os.environ.get("VLLM_BASE_URL"):
        return os.environ["VLLM_BASE_URL"]
    return p.base


def api_key(p: Provider) -> str | None:
    for env in p.keys:
        value = os.environ.get(env)
        if value:
            return value
    return None


def configured(p: Provider) -> bool:
    """True when this engine could be used right now without more setup.

    For a hosted provider that means a key is present. For a local server
    it means nothing here — reachability is a separate, slower check.
    """
    if p.protocol in ("mcp", "none"):
        return True
    if not p.keys:
        return p.local
    return api_key(p) is not None or (p.local and p.name == "vllm")


def _probe_url(p: Provider) -> str:
    url = base_url(p).rstrip("/")
    if p.protocol == "ollama":
        return (url[:-3].rstrip("/") if url.endswith("/v1") else url) + "/api/tags"
    return url + "/models"


# 2s is right for localhost and far too short for an API over the internet:
# the probe timed out, the list came back empty, and a stale pinned default
# was used instead of what the account actually serves.
LOCAL_PROBE_TIMEOUT = 2.0
HOSTED_PROBE_TIMEOUT = 10.0


def probe_timeout(p: Provider) -> float:
    return LOCAL_PROBE_TIMEOUT if p.local else HOSTED_PROBE_TIMEOUT


def models(p: Provider, timeout: float | None = None) -> list[str]:
    """Model ids a local server is serving, newest API shape first.

    Doubles as the reachability test: a port answering with something that
    is not a model list is not an inference server. 8080 and 1234 are busy
    ports on a developer's machine, and picking the wrong listener and
    failing eight times is worse than not finding one.
    """
    import httpx
    timeout = probe_timeout(p) if timeout is None else timeout
    # A hosted endpoint answers /models only for a caller it recognises, so
    # without the key this returned [] and the stale default_model was used.
    headers = {}
    if not p.local:
        key = api_key(p)
        if key:
            headers["Authorization"] = f"Bearer {key}"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(_probe_url(p), headers=headers)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:  # noqa: BLE001 — not an inference server, or not up
        return []
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("data"), list):            # OpenAI shape
        return [str(m.get("id")) for m in data["data"] if isinstance(m, dict) and m.get("id")]
    if isinstance(data.get("models"), list):          # Ollama shape
        return [str(m.get("name") or m.get("model")) for m in data["models"]
                if isinstance(m, dict) and (m.get("name") or m.get("model"))]
    return []


# Models that cannot hold a chat: picking one of these would fail in a way
# that looks like the engine being broken.
_NOT_CHAT = ("embed", "bge-", "bge:", "e5-", "minilm", "gte-", "rerank",
             "nomic-embed", "all-minilm", "mxbai-embed", "snowflake-arctic-embed",
             # Hosted catalogues list speech, TTS and safety classifiers next
             # to the chat models. None of them can write a primer page.
             "whisper", "tts", "orpheus", "prompt-guard", "guard-", "safeguard",
             "-guard", "moderation", "stable-diffusion", "flux",
             # Legacy completion and non-text ids sit in hosted catalogues too,
             # and some sort ahead of the chat models alphabetically.
             "dall-e", "babbage", "davinci", "curie", "ada-", "sora",
             "realtime", "transcribe", "image-", "-image", "video")

# A primer page is a long structured JSON object, so size genuinely helps.
# Used only to order an unfamiliar catalogue; never to override --model.
_PREFER = ("120b", "70b", "72b", "mixtral", "405b", "32b", "27b", "20b")


def chat_models(p: Provider, timeout: float | None = None) -> list[str]:
    """Served models that can actually answer a chat request."""
    return [m for m in models(p, timeout)
            if not any(bad in m.lower() for bad in _NOT_CHAT)]


def rank_models(served: list[str]) -> list[str]:
    """Capable-looking models first, so an unattended pick is a sane one."""
    def key(m: str) -> tuple[int, str]:
        low = m.lower()
        for i, tag in enumerate(_PREFER):
            if tag in low:
                return (i, m)
        return (len(_PREFER), m)
    return sorted(served, key=key)


def choose_model(p: Provider, preferred: str | None = None,
                 timeout: float | None = None) -> str | None:
    """The model to use on a given endpoint.

    A local server serves whatever is loaded, and a hosted key serves
    whatever that account is entitled to, so a name from config or from a
    stale built-in default only wins when the endpoint actually has it.
    """
    served = chat_models(p, timeout)
    if preferred and (not served or preferred in served):
        return preferred
    if preferred and served:
        base = preferred.split(":")[0]
        for m in served:                      # llama3.1 matches llama3.1:8b
            if m.split(":")[0] == base:
                return m
    ranked = rank_models(served)
    return ranked[0] if ranked else preferred


def reachable(p: Provider, timeout: float = 2.0) -> bool | None:
    """Is a local inference server answering here? None when it does not apply."""
    if not p.local or p.protocol in ("mcp", "none"):
        return None
    return bool(models(p, timeout))


def pick(prefer: str | None = None, check_local: bool = True) -> tuple[str, str]:
    """Choose an engine. Returns (name, why) so the choice can be reported."""
    if prefer and resolve(prefer) not in ("auto", ""):
        name = resolve(prefer)
        return name, "requested"
    for name in AUTO_ORDER:
        p = PROVIDERS[name]
        if p.local:
            if not check_local:
                continue
            served = chat_models(p)
            if served:
                return name, (f"{p.label} at {base_url(p)} is serving "
                              f"{served[0]}" + (f" (+{len(served) - 1} more)"
                                                if len(served) > 1 else ""))
        elif api_key(p):
            return name, f"{p.keys[0]} is set"
    return "outline", ("no model engine found — writing a structural draft. "
                       "Run a local server (Ollama, LM Studio, llama.cpp) or set an "
                       "API key; `kx primers engines` lists every option.")


def status(check_local: bool = True) -> list[dict]:
    """One row per engine for `kx primers engines`."""
    rows = []
    for p in _P:
        live = reachable(p) if (check_local and p.local) else None
        rows.append({
            "engine": p.name,
            "label": p.label,
            "kind": "local" if p.local else ("agent" if p.protocol == "mcp" else "hosted"),
            "endpoint": base_url(p) or None,
            "keys": list(p.keys),
            "key_set": api_key(p) is not None if p.keys else None,
            "reachable": live,
            "default_model": p.default_model or None,
            "aliases": list(p.aliases),
            "ready": bool(live) if live is not None else configured(p),
            "note": p.note,
        })
    return rows
