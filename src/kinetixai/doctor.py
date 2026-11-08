"""`kx doctor`: what kx can see on this machine, and what to fix."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path
import time

import httpx

from . import __version__, auth, config, library, net
from .rag import index as rag_index
from .rag import ollama

EXTRAS = {
    "pypdf": ("rag", "read PDFs for kx rag index"),
    "mcp": ("mcp", "kx mcp server for agents"),
    "jinja2": ("primers", "kx primers build and kx convert"),
    "keyring": ("keyring", "store your token in the OS keychain"),
}

PROBES = [
    ("KinetixAI", None),
    ("arXiv", "https://export.arxiv.org/api/query?id_list=1706.03762&max_results=1"),
    ("Semantic Scholar", "https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.03762?fields=title"),
    ("OpenAlex", "https://api.openalex.org/works?per-page=1"),
    ("Hugging Face", "https://huggingface.co/api/models?limit=1"),
    ("GitHub", "https://api.github.com/rate_limit"),
]


def _probe(name: str, url: str, c: httpx.Client) -> dict:
    t0 = time.perf_counter()
    try:
        r = c.get(url)
        ms = int((time.perf_counter() - t0) * 1000)
        ok = r.status_code < 500 and r.status_code not in (403, 407)
        return {"name": name, "url": url, "ok": ok, "status": r.status_code, "ms": ms}
    except httpx.HTTPError as exc:
        return {"name": name, "url": url, "ok": False, "status": None, "error": exc.__class__.__name__}


def _commands() -> set[str]:
    """Top-level commands this build actually exposes."""
    try:
        from .cli import app
        names: set[str] = set()
        for c in getattr(app, "registered_commands", []):
            # Typer leaves name None when it comes from the function name
            name = c.name or getattr(c.callback, "__name__", "")
            if name:
                names.add(name.rstrip("_").replace("_", "-"))
        for g in getattr(app, "registered_groups", []):
            if g.name:
                names.add(g.name)
        return names
    except Exception:  # noqa: BLE001 — doctor must never be the thing that breaks
        return set()


def run(offline: bool = False) -> dict:
    cfg = config.load()
    endpoint = cfg["endpoint"].rstrip("/")
    py_ok = sys.version_info >= (3, 10)
    checks: dict = {
        "kx": __version__,
        "python": {"version": platform.python_version(), "ok": py_ok, "executable": sys.executable},
        "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "encoding": {"stdout": getattr(sys.stdout, "encoding", None)},
        "paths": {
            "config": str(config.config_path()),
            "data": str(config.data_dir()),
            "library": str(library.library_path()),
            "rag_index": str(rag_index.index_path()),
            "kx": shutil.which("kx"),
            # Which kx is actually running, which is not always the one on
            # PATH: a venv, a pipx shim and a stale install look identical
            # until a command is missing.
            "running": sys.executable,
            "package": str(Path(__file__).resolve().parent),
        },
        "commands": sorted(_commands()),
        "endpoint": endpoint,
        "auth": {"signed_in_token": auth.source(endpoint) is not None, "token_source": auth.source(endpoint),
                 "store": auth.store_name()},
        "library": {"papers": None},
        "extras": {},
        "ollama": {},
        "network": [],
    }
    try:
        checks["library"]["papers"] = len(library.load())
    except library.LibraryError as exc:
        checks["library"]["error"] = str(exc)
    idx = rag_index.load_index()
    checks["library"]["rag_index"] = (
        {"chunks": len(idx.get("chunks", [])), "embed_model": idx.get("embed_model"), "built_at": idx.get("built_at")}
        if idx else None
    )
    for mod, (extra, why) in EXTRAS.items():
        checks["extras"][extra] = {"module": mod, "installed": importlib.util.find_spec(mod) is not None, "for": why}

    url = cfg["ollama_url"]
    installed = ollama.models(url)
    checks["ollama"] = {
        "url": url,
        "running": installed is not None,
        "local": ollama.is_local(url),
        "binary": shutil.which("ollama"),
        "models": installed or [],
        "answer_model": cfg["ollama_model"],
        "answer_model_ready": ollama.has_model(installed, cfg["ollama_model"]),
        "embed_model": cfg["embed_model"],
        "embed_model_ready": ollama.has_model(installed, cfg["embed_model"]),
    }
    if not offline:
        with net.client(timeout=httpx.Timeout(6.0, connect=4.0)) as c:
            for name, probe in PROBES:
                checks["network"].append(_probe(name, probe or f"{endpoint}/api/auth/me", c))
    checks["advice"] = advice(checks)
    checks["ok"] = py_ok and all(n["ok"] for n in checks["network"])
    return checks


def advice(ch: dict) -> list[str]:
    out = []
    if not ch["python"]["ok"]:
        out.append("kx needs Python 3.10 or newer.")
    if not ch["paths"]["kx"]:
        out.append("kx is not on PATH. With pipx run `pipx ensurepath`, then open a new terminal.")
    if not ch["auth"]["signed_in_token"]:
        out.append("Not logged in (only `kx ask` uses an account): create a token at "
                   f"{ch['endpoint']}/account/tokens/ and run `kx login`.")
    if not ch["extras"]["rag"]["installed"]:
        out.append("PDF text for local RAG: pipx inject kinetixai pypdf")
    if not ch["extras"]["primers"]["installed"]:
        tips.append("pip install 'kinetixai[primers]'  (kx primers build, kx convert; "
                    "with pipx: pipx inject kinetixai jinja2 python-pptx pypdfium2 "
                    "pillow weasyprint pyyaml)")
    if not ch["extras"]["mcp"]["installed"]:
        out.append("MCP server for agents: pipx inject kinetixai mcp")
    o = ch["ollama"]
    if not o["running"]:
        out.append("Ollama not running: kx rag ask will return passages, not written answers. "
                   "Install from https://ollama.com and run `ollama serve`.")
    else:
        if not o["answer_model_ready"]:
            out.append(f"ollama pull {o['answer_model']}  (written answers for kx rag ask)")
        if not o["embed_model_ready"]:
            out.append(f"ollama pull {o['embed_model']}  (semantic search for kx rag index)")
    for n in ch["network"]:
        if not n["ok"]:
            out.append(f"{n['name']} unreachable ({n.get('status') or n.get('error')}). "
                       "Check your network, proxy (HTTPS_PROXY) or firewall.")
    if os.name == "nt" and (ch["encoding"]["stdout"] or "").lower() not in ("utf-8", "utf8"):
        out.append("For best output on Windows use Windows Terminal, or set PYTHONUTF8=1.")
    return out
