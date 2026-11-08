"""`kx mcp`: an MCP server over stdio for open-source agents.

Needs the optional `mcp` package (pip install "kinetixai[mcp]"). Works with
both the 1.x SDK (FastMCP) and 2.x (MCPServer). Tools return JSON; nothing
is printed to stdout except the protocol itself.
"""
from __future__ import annotations

import functools
import inspect
import json
import shutil
import sys
from pathlib import Path

from . import __version__, services

INSTRUCTIONS = (
    "KinetixAI research and publishing tools. search_papers finds papers across arXiv, Semantic Scholar and "
    "OpenAlex. get_paper resolves an arXiv id or DOI. find_code, find_models and find_datasets link a paper "
    "(use its arXiv id) to GitHub implementations and Hugging Face models and datasets. library_search and "
    "rag_ask work over the user's local paper library; rag_ask answers privately on this machine. "
    "search_verified_papers returns only domain-relevant sources for a topic. build_primer compiles a "
    "'Story to Silicon' A4 primer (PDF, PNG, PPTX, JSON) from a topic; convert_document does the same from "
    "an existing deck, paper or repo. Both can borrow YOUR model for the writing by passing engine='mcp', "
    "which needs no API key on the user's machine."
)

TOOLS = ["search_papers", "get_paper", "find_code", "find_models", "find_datasets", "library_search",
         "rag_ask", "search_verified_papers", "build_primer", "convert_document"]


class MissingMcp(Exception):
    pass


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2
        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x
        return FastMCP
    except ImportError:
        raise MissingMcp(
            'the MCP server needs the mcp package: pipx inject kinetixai mcp  '
            '(or pip install "kinetixai[mcp]")'
        )


def _clip(n: int, lo: int = 1, hi: int = 50) -> int:
    return max(lo, min(int(n), hi))


def _safe(fn):
    """Tool results, never tracebacks: a source being down is an answer an
    agent can act on, and a traceback on stderr is noise in the client's log."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a structured result
            return {"ok": False, "error": str(exc) or exc.__class__.__name__, "warnings": [str(exc)]}
    return wrapper


def build_server():
    Server = _server_class()
    kwargs = {"instructions": INSTRUCTIONS}
    if "version" in inspect.signature(Server.__init__).parameters:  # 2.x reports it in serverInfo
        kwargs["version"] = __version__
    server = Server("kinetixai", **kwargs)

    @server.tool()
    @_safe
    def search_papers(query: str, limit: int = 10) -> dict:
        """Search research papers across arXiv, Semantic Scholar and OpenAlex (merged, de-duplicated, ranked by citations)."""
        return services.search_papers(query, _clip(limit))

    @server.tool()
    @_safe
    def get_paper(id: str) -> dict:
        """Full metadata and abstract for one paper, by arXiv id, arXiv/HF URL, or DOI."""
        p = services.get_paper(id)
        return {"paper": p, "found": p is not None, "warnings": []}

    @server.tool()
    @_safe
    def find_code(paper: str, limit: int = 10) -> dict:
        """Code implementations of a paper (arXiv id preferred): the authors' official repo plus GitHub repos by stars."""
        return services.find_code(paper, _clip(limit))

    @server.tool()
    @_safe
    def find_models(query: str, limit: int = 10, sort: str = "downloads") -> dict:
        """Hugging Face models for a search query, or that cite a paper when given an arXiv id. Sort: downloads, likes, trending, updated."""
        return services.find_models(query, _clip(limit), sort=sort)

    @server.tool()
    @_safe
    def find_datasets(query: str, limit: int = 10, sort: str = "downloads") -> dict:
        """Hugging Face datasets for a search query, or that cite a paper when given an arXiv id."""
        return services.find_datasets(query, _clip(limit), sort=sort)

    @server.tool()
    @_safe
    def library_search(query: str, limit: int = 10) -> dict:
        """Search the user's local KinetixAI paper library by keyword (title, authors, abstract, tags)."""
        return services.library_search(query, _clip(limit))

    @server.tool()
    @_safe
    def rag_ask(question: str, k: int = 5, use_llm: bool = True) -> dict:
        """Answer a question from the user's local library with cited passages. Runs on this machine
        (BM25, plus a local Ollama model when available). Run `kx rag index` first."""
        return services.rag_ask(question, k=_clip(k, 1, 20), llm=use_llm)

    # ── Primers ───────────────────────────────────────────────────────────

    @server.tool()
    @_safe
    def search_verified_papers(topic: str, limit: int = 12) -> dict:
        """Papers that are actually about a topic. Unlike search_papers this filters by
        domain, so 'GPU kernel fusion' does not return tokamak fusion or image fusion.
        Returns the kept sources and a note for every result dropped."""
        from .primers import search as primer_search
        raw = services.search_papers(topic, _clip(limit * 2, 1, 50))
        papers = raw.get("papers") or []
        kept, notes = primer_search.filter_sources(papers, topic, limit=_clip(limit))
        return {"ok": True, "topic": topic, "papers": kept, "dropped": notes,
                "warnings": raw.get("warnings") or []}

    @server.tool()
    @_safe
    def build_primer(topic: str, pages: int = 8, personas: str = "student",
                     out_dir: str = "./dist/primers", engine: str = "mcp",
                     model: str | None = None, pdfs: int = 1, images: int = 0,
                     ppts: int = 0, json_bundles: int = 1,
                     watermark: str = "Narmadya Innovations") -> dict:
        """Build a 'Story to Silicon' A4 primer: research, write, and compile a styled PDF
        (plus optional 300 DPI PNGs, an editable PPTX and a Paper Lab JSON bundle).

        engine='mcp' (the default here) asks YOU to write the pages, so no API key is
        needed on the user's machine; use 'ollama', 'vllm', 'claude', 'gemini' or
        'openai' to run a model directly, or 'outline' for a draft with no model at all.
        Writes files to out_dir and returns their paths."""
        return _run_primer(server, topic=topic, source=None, pages=pages, personas=personas,
                           out_dir=out_dir, engine=engine, model=model, pdfs=pdfs,
                           images=images, ppts=ppts, json_bundles=json_bundles,
                           watermark=watermark)

    @server.tool()
    @_safe
    def convert_document(source: str, topic: str | None = None, pages: int = 8,
                         personas: str = "professional", out_dir: str = "./dist/convert",
                         engine: str = "mcp", model: str | None = None, pdfs: int = 1,
                         images: int = 0, ppts: int = 0, json_bundles: int = 1,
                         watermark: str = "Narmadya Innovations") -> dict:
        """Turn an existing .pptx, .pdf, .md file, a GitHub repo URL, or an arXiv id/DOI
        into the same 8-page primer format. Same engine options as build_primer."""
        return _run_primer(server, topic=topic or source, source=source, pages=pages,
                           personas=personas, out_dir=out_dir, engine=engine, model=model,
                           pdfs=pdfs, images=images, ppts=ppts, json_bundles=json_bundles,
                           watermark=watermark)

    return server


def _host_sampler(server):
    """A callable that asks the MCP host to run one completion for us.

    Sampling is optional in the protocol and many clients do not implement it,
    so this returns None when it is unavailable rather than failing mid-build.
    """
    try:
        from mcp.server.fastmcp import Context  # noqa: F401
    except ImportError:
        return None

    def sample(system: str, user: str) -> str:
        ctx = None
        for getter in ("get_context", "request_context"):
            try:
                ctx = getattr(server, getter)
                ctx = ctx() if callable(ctx) else ctx
                break
            except Exception:  # noqa: BLE001
                continue
        if ctx is None:
            raise RuntimeError("no request context: this client does not expose sampling")
        import anyio

        async def _ask() -> str:
            result = await ctx.session.create_message(
                messages=[{"role": "user",
                           "content": {"type": "text", "text": user}}],
                system_prompt=system,
                max_tokens=16000,
            )
            content = getattr(result, "content", None)
            text = getattr(content, "text", None)
            if not text:
                raise RuntimeError("the host returned no text")
            return text

        return anyio.from_thread.run(_ask) if _in_worker_thread() else anyio.run(_ask)

    return sample


def _in_worker_thread() -> bool:
    try:
        import anyio.from_thread
        anyio.from_thread.threadlocals.current_async_backend  # noqa: B018
        return True
    except Exception:  # noqa: BLE001
        return False


def _run_primer(server, *, topic, source, pages, personas, out_dir, engine, model,
                pdfs, images, ppts, json_bundles, watermark) -> dict:
    """Shared body for build_primer and convert_document."""
    from .primers import builder as primer_builder, engines as primer_engines

    engine = primer_engines.canonical(engine)
    if engine == "mcp":
        sampler = _host_sampler(server)
        if sampler is None:
            return {"ok": False, "error": "this MCP client does not support sampling; "
                                          "pass engine='ollama', 'vllm', 'claude', "
                                          "'gemini', 'openai' or 'outline'.",
                    "warnings": []}
        primer_engines.set_sampler(sampler)

    req = primer_builder.BuildRequest(
        topic=topic, source=source, pages=int(pages),
        personas=[p.strip() for p in str(personas).split(",") if p.strip()],
        pdfs=int(pdfs), images=int(images), ppts=int(ppts), jsons=int(json_bundles),
        watermark=watermark, engine=engine, model=model, out_dir=Path(out_dir))
    try:
        result = primer_builder.build(req)
    finally:
        if engine == "mcp":
            primer_engines.set_sampler(None)
    return {"ok": True, **result.to_dict()}


def run() -> None:
    build_server().run()  # stdio by default in both SDK versions


# ── Ready-to-paste client configuration ──────────────────────────────────

CLIENTS = {
    "claude-code": "Claude Code (CLI)",
    "claude-desktop": "Claude Desktop",
    "goose": "Goose",
    "cline": "Cline (VS Code)",
    "continue": "Continue",
    "openhands": "OpenHands",
    "open-webui": "Open WebUI (via mcpo)",
    "cursor": "Cursor",
}


def command_path() -> str:
    """Absolute path to kx, so GUI apps with a short PATH still find it."""
    found = shutil.which("kx")
    if found:
        return str(Path(found).resolve())
    argv0 = Path(sys.argv[0])
    if argv0.name.lower().startswith("kx") and argv0.exists():
        return str(argv0.resolve())
    return "kx"


def _q(s: str) -> str:
    """A double-quoted string that is valid JSON, YAML and TOML (escapes Windows backslashes)."""
    return json.dumps(s)


def _shell_cmd(cmd: str) -> str:
    """The kx path as a shell word: double-quoted when it has spaces
    (C:\\Program Files\\...), which bash, zsh, PowerShell and cmd all accept."""
    return f'"{cmd}"' if any(c.isspace() for c in cmd) else cmd


def config_snippet(client: str, cmd: str | None = None) -> dict:
    cmd = cmd or command_path()
    std = json.dumps({"mcpServers": {"kinetixai": {"command": cmd, "args": ["mcp", "serve"]}}}, indent=2)
    home = "~"
    if client == "claude-code":
        return {"client": client, "file": "terminal", "format": "shell",
                "snippet": f"claude mcp add kx -- {_shell_cmd(cmd)} mcp serve",
                "note": "Then `claude mcp list` to verify. Ask Claude Code to build a "
                        "primer and it will call kx directly."}
    if client in ("claude-desktop",):
        path = {
            "darwin": f"{home}/Library/Application Support/Claude/claude_desktop_config.json",
            "win32": r"%APPDATA%\Claude\claude_desktop_config.json",
        }.get(sys.platform, f"{home}/.config/Claude/claude_desktop_config.json")
        return {"client": client, "file": path, "format": "json", "snippet": std,
                "note": "Merge into mcpServers, then restart Claude Desktop."}
    if client == "cline":
        return {"client": client, "file": "cline_mcp_settings.json (Cline > MCP Servers > Configure)",
                "format": "json", "snippet": std, "note": "Cline reloads the server list on save."}
    if client == "cursor":
        return {"client": client, "file": f"{home}/.cursor/mcp.json (or .cursor/mcp.json in a project)",
                "format": "json", "snippet": std, "note": ""}
    if client == "goose":
        snippet = (
            "extensions:\n"
            "  kinetixai:\n"
            "    name: kinetixai\n"
            "    type: stdio\n"
            f"    cmd: {_q(cmd)}\n"
            "    args: [\"mcp\", \"serve\"]\n"
            "    enabled: true\n"
            "    timeout: 300\n"
        )
        goose_file = (r"%APPDATA%\Block\goose\config\config.yaml" if sys.platform == "win32"
                      else f"{home}/.config/goose/config.yaml")
        return {"client": client, "file": goose_file, "format": "yaml", "snippet": snippet,
                "note": "Or for one session: goose session --with-extension '" + _shell_cmd(cmd) + " mcp serve'"}
    if client == "continue":
        snippet = (
            "mcpServers:\n"
            "  - name: KinetixAI\n"
            f"    command: {_q(cmd)}\n"
            "    args: [\"mcp\", \"serve\"]\n"
        )
        return {"client": client, "file": f"{home}/.continue/config.yaml", "format": "yaml",
                "snippet": snippet, "note": "MCP tools are available in Continue's agent mode."}
    if client == "openhands":
        snippet = (
            "[mcp]\n"
            f"stdio_servers = [{{ name = \"kinetixai\", command = {_q(cmd)}, args = [\"mcp\", \"serve\"] }}]\n"
        )
        return {"client": client, "file": "config.toml (OpenHands)", "format": "toml", "snippet": snippet,
                "note": "In the OpenHands UI: Settings > MCP > add a stdio server with the same command."}
    if client == "open-webui":
        snippet = f"uvx mcpo --port 8000 -- {_shell_cmd(cmd)} mcp serve"
        return {"client": client, "file": "terminal", "format": "shell", "snippet": snippet,
                "note": "mcpo exposes the tools as an OpenAPI server. In Open WebUI: Settings > Tools > "
                        "add http://localhost:8000."}
    raise ValueError(f"unknown client {client!r}; choose from: {', '.join(CLIENTS)}")


def info() -> dict:
    try:
        _server_class()
        available = True
    except MissingMcp:
        available = False
    return {"server": "kinetixai", "version": __version__, "transport": "stdio", "tools": TOOLS,
            "mcp_installed": available, "command": [command_path(), "mcp", "serve"], "clients": list(CLIENTS)}
