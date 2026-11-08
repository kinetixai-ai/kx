"""`kx mcp`: an MCP server over stdio for open-source agents.

Needs the optional `mcp` package (pip install "kinetixai[mcp]"). Works with
both the 1.x SDK (FastMCP) and 2.x (MCPServer). Tools return JSON; nothing
is printed to stdout except the protocol itself.
"""
from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path

from . import __version__, services

INSTRUCTIONS = (
    "KinetixAI research tools. search_papers finds papers across arXiv, Semantic Scholar and OpenAlex. "
    "get_paper resolves an arXiv id or DOI. find_code, find_models and find_datasets link a paper (use its "
    "arXiv id) to GitHub implementations and Hugging Face models and datasets. library_search and rag_ask "
    "work over the user's local paper library; rag_ask answers privately on this machine."
)

TOOLS = ["search_papers", "get_paper", "find_code", "find_models", "find_datasets", "library_search", "rag_ask"]


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


def build_server():
    Server = _server_class()
    kwargs = {"instructions": INSTRUCTIONS}
    if "version" in inspect.signature(Server.__init__).parameters:  # 2.x reports it in serverInfo
        kwargs["version"] = __version__
    server = Server("kinetixai", **kwargs)

    @server.tool()
    def search_papers(query: str, limit: int = 10) -> dict:
        """Search research papers across arXiv, Semantic Scholar and OpenAlex (merged, de-duplicated, ranked by citations)."""
        return services.search_papers(query, _clip(limit))

    @server.tool()
    def get_paper(id: str) -> dict:
        """Full metadata and abstract for one paper, by arXiv id, arXiv/HF URL, or DOI."""
        p = services.get_paper(id)
        return {"paper": p, "found": p is not None}

    @server.tool()
    def find_code(paper: str, limit: int = 10) -> dict:
        """Code implementations of a paper (arXiv id preferred): the authors' official repo plus GitHub repos by stars."""
        return services.find_code(paper, _clip(limit))

    @server.tool()
    def find_models(query: str, limit: int = 10, sort: str = "downloads") -> dict:
        """Hugging Face models for a search query, or that cite a paper when given an arXiv id. Sort: downloads, likes, trending, updated."""
        return services.find_models(query, _clip(limit), sort=sort)

    @server.tool()
    def find_datasets(query: str, limit: int = 10, sort: str = "downloads") -> dict:
        """Hugging Face datasets for a search query, or that cite a paper when given an arXiv id."""
        return services.find_datasets(query, _clip(limit), sort=sort)

    @server.tool()
    def library_search(query: str, limit: int = 10) -> dict:
        """Search the user's local KinetixAI paper library by keyword (title, authors, abstract, tags)."""
        return services.library_search(query, _clip(limit))

    @server.tool()
    def rag_ask(question: str, k: int = 5, use_llm: bool = True) -> dict:
        """Answer a question from the user's local library with cited passages. Runs on this machine
        (BM25, plus a local Ollama model when available). Run `kx rag index` first."""
        try:
            return services.rag_ask(question, k=_clip(k, 1, 20), llm=use_llm)
        except Exception as exc:  # an empty index is a normal answer for an agent
            return {"question": question, "answer": None, "passages": [], "error": str(exc)}

    return server


def run() -> None:
    build_server().run()  # stdio by default in both SDK versions


# ── Ready-to-paste client configuration ──────────────────────────────────

CLIENTS = {
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


def config_snippet(client: str, cmd: str | None = None) -> dict:
    cmd = cmd or command_path()
    std = json.dumps({"mcpServers": {"kinetixai": {"command": cmd, "args": ["mcp"]}}}, indent=2)
    home = "~"
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
            "    args: [\"mcp\"]\n"
            "    enabled: true\n"
            "    timeout: 300\n"
        )
        return {"client": client, "file": f"{home}/.config/goose/config.yaml", "format": "yaml",
                "snippet": snippet,
                "note": f"Or for one session: goose session --with-extension {_q(cmd + ' mcp')}"}
    if client == "continue":
        snippet = (
            "mcpServers:\n"
            "  - name: KinetixAI\n"
            f"    command: {_q(cmd)}\n"
            "    args: [\"mcp\"]\n"
        )
        return {"client": client, "file": f"{home}/.continue/config.yaml", "format": "yaml",
                "snippet": snippet, "note": "MCP tools are available in Continue's agent mode."}
    if client == "openhands":
        snippet = (
            "[mcp]\n"
            f"stdio_servers = [{{ name = \"kinetixai\", command = {_q(cmd)}, args = [\"mcp\"] }}]\n"
        )
        return {"client": client, "file": "config.toml (OpenHands)", "format": "toml", "snippet": snippet,
                "note": "In the OpenHands UI: Settings > MCP > add a stdio server with the same command."}
    if client == "open-webui":
        snippet = f"uvx mcpo --port 8000 -- {cmd} mcp"
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
            "mcp_installed": available, "command": [command_path(), "mcp"], "clients": list(CLIENTS)}
