"""Minimal Ollama client (http://localhost:11434): list models, embed, chat."""
from __future__ import annotations

from urllib.parse import urlparse

import httpx

from .. import net

SOURCE = "Ollama"


def is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"} or host.startswith("127.")


def models(url: str, client: httpx.Client | None = None) -> list[str] | None:
    """Installed model names, or None when Ollama is not running."""
    own = client is None
    c = client or httpx.Client(timeout=httpx.Timeout(2.0, connect=1.0))
    try:
        r = c.get(url.rstrip("/") + "/api/tags")
        if r.status_code != 200:
            return None
        return [m.get("name", "") for m in (r.json().get("models") or [])]
    except (httpx.HTTPError, ValueError):
        return None
    finally:
        if own:
            c.close()


def has_model(installed: list[str] | None, name: str) -> bool:
    if not installed:
        return False
    want = name if ":" in name else name + ":latest"
    return any(m == name or m == want or m.split(":")[0] == name for m in installed)


def embed(url: str, model: str, texts: list[str], client: httpx.Client) -> list[list[float]]:
    base = url.rstrip("/")
    r = net.request(client, "POST", base + "/api/embed", source=SOURCE, allow_404=True, retries=0,
                    json={"model": model, "input": texts})
    if r is not None:
        return r.json().get("embeddings") or []
    # Ollama < 0.3 only has the single-prompt endpoint
    out = []
    for t in texts:
        r = net.request(client, "POST", base + "/api/embeddings", source=SOURCE, retries=0,
                        json={"model": model, "prompt": t})
        out.append(r.json().get("embedding") or [])
    return out


def chat(url: str, model: str, messages: list[dict], client: httpx.Client) -> str:
    r = net.request(client, "POST", url.rstrip("/") + "/api/chat", source=SOURCE, retries=0,
                    json={"model": model, "messages": messages, "stream": False,
                          "options": {"temperature": 0.2}})
    return ((r.json().get("message") or {}).get("content") or "").strip()
