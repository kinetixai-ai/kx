"""kx: the KinetixAI command line."""
from __future__ import annotations

import functools
import getpass
import json
import os
import re
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

import httpx
import typer

from . import __version__, auth, cite as cite_mod, config, doctor as doctor_mod, library, net, output, services
from .api import ApiError, KinetixApi
from .models import Paper
from .rag import index as rag_index
from .rag.text import MissingExtra
from .resolve import arxiv_id, doi, resolve
from .sources import aggregate, orcid

TYPER_SAFE = {"pretty_exceptions_enable": False, "pretty_exceptions_show_locals": False}
app = typer.Typer(
    **TYPER_SAFE,
    add_completion=True,
    no_args_is_help=True,
    help=(
        "KinetixAI command line: papers -> code -> models -> datasets -> your own private Q&A.\n\n"
        "Every command takes --json for scripting. Docs: https://kinetixai.ai/cli/"
    ),
)
config_app = typer.Typer(**TYPER_SAFE, no_args_is_help=True, help="View or change settings.")
library_app = typer.Typer(**TYPER_SAFE, no_args_is_help=True, help="Your local paper library (stored on this machine).")
rag_app = typer.Typer(**TYPER_SAFE, no_args_is_help=True, help="Private question answering over your library, on this machine.")
app.add_typer(config_app, name="config")
app.add_typer(library_app, name="library")
app.add_typer(rag_app, name="rag")

JSON = typer.Option(False, "--json", help="Machine-readable JSON on stdout.")
UA = net.UA  # kept for backwards compatibility


def _client() -> httpx.Client:
    return net.client()


def handled(fn):
    """Turn expected failures into one clear line on stderr and exit code 1."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (net.SourceError, ApiError, library.LibraryError, rag_index.RagError, MissingExtra) as exc:
            output.error(str(exc))
            raise typer.Exit(code=1)
    return wrapper


def _version_cb(value: bool) -> None:
    if value:
        output.console.print(f"kx {__version__}")
        raise typer.Exit()


@app.callback()
def main_options(
    version_: bool = typer.Option(False, "--version", "-V", callback=_version_cb, is_eager=True,
                                  help="Print the version and exit."),
) -> None:
    pass


@app.command()
def version(json_out: bool = JSON) -> None:
    """Print the kx version."""
    if json_out:
        output.emit_json({"version": __version__, "python": sys.version.split()[0], "platform": sys.platform})
    else:
        output.console.print(f"kx {__version__}")


@app.command()
def doctor(
    offline: bool = typer.Option(False, "--offline", help="Skip the network checks."),
    json_out: bool = JSON,
) -> None:
    """Check Python, paths, login, optional extras, Ollama and network reachability."""
    ch = doctor_mod.run(offline=offline)
    if json_out:
        output.emit_json(ch)
    else:
        output.doctor(ch)


# ── Papers ────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., metavar="QUERY", help="What to search for."),
    limit: int = typer.Option(None, "--limit", "-n", min=1, max=100, help="Max results per source."),
    source: str = typer.Option("all", "--source", "-s", help="all | arxiv | semantic | openalex"),
    json_out: bool = JSON,
    open_: bool = typer.Option(False, "--open", help="Open the top result in a browser."),
) -> None:
    """Search papers across arXiv, Semantic Scholar and OpenAlex."""
    if source != "all" and source not in aggregate.SOURCES:
        output.error(f"--source must be one of: all, {', '.join(aggregate.SOURCES)}")
        raise typer.Exit(code=2)
    limit = limit or config.get("limit", 10)
    sources = None if source == "all" else [source]
    with _client() as client:
        papers, warnings = aggregate.search(query, limit, client, sources)
    for w in warnings:
        output.warn(w)
    output.papers(papers, json_out)
    if not papers and warnings:
        raise typer.Exit(code=1)  # every source failed: not the same as "no results"
    if open_ and papers and papers[0].url:
        webbrowser.open(papers[0].url)


@app.command()
def scholar(
    query: str = typer.Argument(..., metavar="QUERY", help="Researcher name or ORCID iD."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=100),
    json_out: bool = JSON,
) -> None:
    """Look up a researcher (name or ORCID iD) and their recent work."""
    if "scholar.google" in query:
        output.err.print(
            "[yellow]note:[/yellow] Google Scholar has no public API. "
            "Search by name or ORCID iD instead."
        )
        raise typer.Exit(code=2)
    with _client() as client:
        try:
            people = orcid.lookup(query, limit, client)
        except httpx.HTTPError as exc:
            output.error(f"ORCID unavailable ({exc.__class__.__name__})")
            raise typer.Exit(code=1)
    output.researchers(people, json_out)


@app.command()
@handled
def get(
    ident: str = typer.Argument(..., metavar="IDENT", help="arXiv id, DOI, or arXiv URL."),
    pdf: bool = typer.Option(False, "--pdf", help="Also download the PDF if available."),
    out: Path = typer.Option(Path("."), "--out", "-o", help="Directory for the PDF."),
    bibtex: bool = typer.Option(False, "--bibtex", help="Print a BibTeX entry instead."),
    json_out: bool = JSON,
) -> None:
    """Fetch a single paper's metadata (and optionally its PDF)."""
    with _client() as client:
        paper = _resolve(ident, client)
        if paper is None:
            output.error(f"could not resolve '{ident}'.")
            raise typer.Exit(code=1)
        if bibtex:
            text = cite_mod.bibtex(paper.to_dict())
            if json_out:
                output.emit_json({"id": ident, "format": "bibtex", "citation": text})
            else:
                print(text)
        elif json_out and not pdf:
            output.emit_json(paper.to_dict())
        elif not json_out:
            output.paper_detail(paper.to_dict())
        if pdf:
            if not paper.pdf_url:
                output.err.print("[yellow]note:[/yellow] no open PDF available.")
                raise typer.Exit(code=2)
            dest = _download(paper, out, client)
            if json_out:
                output.emit_json({**paper.to_dict(), "saved": str(dest)})
            else:
                output.console.print(f"[green]saved[/green] {dest}")


@app.command()
@handled
def cite(
    ref: str = typer.Argument(..., metavar="REF", help="arXiv id, DOI, or a key / number from `kx library list`."),
    fmt: str = typer.Option("bibtex", "--format", "-f", help="bibtex | apa"),
    json_out: bool = JSON,
) -> None:
    """Print a citation (BibTeX or APA). Uses your library first, then looks it up."""
    if fmt not in cite_mod.FORMATS:
        output.error(f"--format must be one of: {', '.join(cite_mod.FORMATS)}")
        raise typer.Exit(code=2)
    entry = library.find(ref)
    if entry is None:
        with _client() as client:
            p = _resolve(ref, client)
        if p is None:
            output.error(f"'{ref}' is not in your library and could not be looked up.")
            raise typer.Exit(code=1)
        entry = p.to_dict()
    text = cite_mod.format_citation(entry, fmt)
    if json_out:
        output.emit_json({"id": ref, "format": fmt, "citation": text})
    else:
        print(text)


# ── Code, models, datasets ────────────────────────────────────────────────

def _warn_all(data: dict) -> None:
    for w in data.get("warnings") or []:
        output.warn(w)


@app.command()
@handled
def code(
    paper: str = typer.Argument(..., metavar="PAPER", help="arXiv id or URL (best), or free text."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=100),
    json_out: bool = JSON,
) -> None:
    """Find code for a paper: the authors' repo (via Hugging Face) and GitHub implementations by stars."""
    data = services.find_code(paper, limit)
    _warn_all(data)
    if json_out:
        output.emit_json(data)
    else:
        if not data["arxiv_id"]:
            output.err.print("[dim]Tip: pass an arXiv id (e.g. 2205.14135) for exact matches.[/dim]")
        output.repos(data)
    if not data["repos"] and data["warnings"]:
        raise typer.Exit(code=1)


@app.command()
@handled
def models(
    query: str = typer.Argument(..., metavar="QUERY", help="Search text, or an arXiv id for models that cite the paper."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=100),
    sort: str = typer.Option("downloads", "--sort", help="downloads | likes | trending | updated"),
    task: Optional[str] = typer.Option(None, "--task", help="Pipeline tag, e.g. text-generation."),
    json_out: bool = JSON,
) -> None:
    """Find Hugging Face models (downloads, likes, license)."""
    data = services.find_models(query, limit, sort=sort, task=task)
    _warn_all(data)
    if json_out:
        output.emit_json(data)
    else:
        output.hub(data["models"], "models")
    if not data["models"] and data["warnings"]:
        raise typer.Exit(code=1)


@app.command()
@handled
def datasets(
    query: str = typer.Argument(..., metavar="QUERY", help="Search text, or an arXiv id for datasets that cite the paper."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=100),
    sort: str = typer.Option("downloads", "--sort", help="downloads | likes | trending | updated"),
    json_out: bool = JSON,
) -> None:
    """Find Hugging Face datasets (downloads, likes, license)."""
    data = services.find_datasets(query, limit, sort=sort)
    _warn_all(data)
    if json_out:
        output.emit_json(data)
    else:
        output.hub(data["datasets"], "datasets")
    if not data["datasets"] and data["warnings"]:
        raise typer.Exit(code=1)


# ── Account ───────────────────────────────────────────────────────────────

def _endpoint(override: str | None = None) -> str:
    if override:
        try:
            return config.validate("endpoint", override)
        except ValueError as exc:
            output.error(str(exc))
            raise typer.Exit(code=2)
    return config.get("endpoint").rstrip("/")


@app.command()
@handled
def login(
    token_stdin: bool = typer.Option(False, "--token-stdin", help="Read the token from stdin (for scripts)."),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="Override the configured endpoint."),
    json_out: bool = JSON,
) -> None:
    """Sign in with a personal API token from /account/tokens/."""
    ep = _endpoint(endpoint)
    if token_stdin:
        token = sys.stdin.readline().strip()
    elif os.environ.get("KX_TOKEN"):
        token = os.environ["KX_TOKEN"].strip()
    else:
        output.err.print(f"Create a token at {ep}/account/tokens/ and paste it below (input is hidden).")
        token = getpass.getpass("Token: ").strip()
    if not auth.valid(token):
        output.error("that does not look like a KinetixAI token (they start with kx_).")
        raise typer.Exit(code=1)
    with _client() as client:
        user = KinetixApi(ep, client, token).me()
    if not user:
        output.error("the token was not accepted (revoked, expired, or for another endpoint).")
        raise typer.Exit(code=1)
    where = auth.save(ep, token)
    store = "keyring" if where == "keyring" else "file"
    if json_out:
        output.emit_json({"ok": True, "endpoint": ep, "email": user.get("email"), "store": store,
                          "path": None if store == "keyring" else where})
    else:
        output.console.print(f"[green]Logged in[/green] to {ep} as [bold]{user.get('email')}[/bold].")
        output.console.print(f"[dim]Token saved in {'your OS keyring' if store == 'keyring' else where}.[/dim]")


@app.command()
@handled
def logout(
    local: bool = typer.Option(False, "--local", help="Only forget the token here; do not revoke it."),
    endpoint: Optional[str] = typer.Option(None, "--endpoint"),
    json_out: bool = JSON,
) -> None:
    """Revoke this machine's token and forget it."""
    ep = _endpoint(endpoint)
    token = auth.load(ep)
    revoked = False
    if token and not local:
        try:
            with _client() as client:
                KinetixApi(ep, client, token).revoke_current()
            revoked = True
        except ApiError as exc:
            if exc.status != 401:
                output.warn(f"could not revoke the token on the server ({exc}). Revoke it at {ep}/account/tokens/.")
    removed = auth.delete(ep)
    if os.environ.get("KX_TOKEN"):
        output.warn("KX_TOKEN is set in your environment; unset it too.")
    if json_out:
        output.emit_json({"ok": True, "endpoint": ep, "revoked": revoked, "removed": removed})
    elif not token:
        output.console.print("Not logged in.")
    else:
        output.console.print("[green]Logged out.[/green]" + (" Token revoked." if revoked else ""))


@app.command()
@handled
def whoami(
    endpoint: Optional[str] = typer.Option(None, "--endpoint"),
    json_out: bool = JSON,
) -> None:
    """Show who you are signed in as."""
    ep = _endpoint(endpoint)
    token = auth.load(ep)
    if not token:
        if json_out:
            output.emit_json({"endpoint": ep, "user": None})
        else:
            output.error(f"not logged in. Create a token at {ep}/account/tokens/ and run `kx login`.")
        raise typer.Exit(code=1)
    with _client() as client:
        user = KinetixApi(ep, client, token).me()
    if json_out:
        output.emit_json({"endpoint": ep, "user": user, "token_source": auth.source(ep)})
        if not user:
            raise typer.Exit(code=1)
        return
    if not user:
        output.error("your token is no longer valid (revoked or expired). Run `kx login` again.")
        raise typer.Exit(code=1)
    output.console.print(f"[bold]{user.get('email')}[/bold]" + (f"  ({user['name']})" if user.get("name") else ""))
    access = user.get("access") or {}
    plan = access.get("label") or (user.get("plan") or "free").title()
    output.console.print(f"[dim]Plan:[/dim] {plan}   [dim]Endpoint:[/dim] {ep}")


@app.command()
@handled
def ask(
    question: str = typer.Argument(..., metavar="QUESTION", help="Your question for KinetixAI."),
    json_out: bool = JSON,
) -> None:
    """Ask KinetixAI and get a grounded, cited answer (uses your account when logged in)."""
    ep = _endpoint()
    token = auth.load(ep)
    with _client() as client:
        data = KinetixApi(ep, client, token).ask(question)
    output.answer(data, json_out)


# ── Library ───────────────────────────────────────────────────────────────

@library_app.command("add")
@handled
def library_add(
    refs: List[str] = typer.Argument(None, metavar="REFS", help="arXiv ids, DOIs, or paths to local PDFs / notes."),
    from_file: Optional[Path] = typer.Option(None, "--from-file", help="Import papers from a JSON file (offline)."),
    pdf: bool = typer.Option(False, "--pdf", help="Download the open PDF too (used by kx rag index)."),
    copy: bool = typer.Option(False, "--copy", help="Copy local files into the library folder."),
    tag: List[str] = typer.Option(None, "--tag", "-t", help="Tag to attach (repeatable)."),
    json_out: bool = JSON,
) -> None:
    """Add papers to your library."""
    refs = refs or []
    tags = tag or []
    if not refs and not from_file:
        output.error("give at least one arXiv id, DOI or file, or --from-file.")
        raise typer.Exit(code=2)
    added: list[dict] = []
    failed: list[dict] = []
    if from_file:
        for p, extras in library.read_import_file(from_file):
            entry, created = library.add(p, tags + extras["tags"], extras["pdf_path"])
            added.append({**entry, "created": created})
    with _client() as client:
        for ref in refs:
            path = Path(ref).expanduser()
            try:
                if path.is_file():
                    if path.suffix.lower() == ".pdf":
                        p, dest = library.from_local_pdf(path, copy=copy)
                    else:
                        p, dest = Paper(title=path.stem, source="local file"), path.resolve()
                    entry, created = library.add(p, tags, dest)
                elif arxiv_id(ref) or doi(ref):
                    p = resolve(ref, client)
                    if p is None:
                        raise library.LibraryError(f"could not find '{ref}'.")
                    dest = None
                    if pdf and p.pdf_url:
                        dest = _download(p, library.pdf_dir(), client)
                    elif pdf:
                        output.warn(f"no open PDF for {ref}; added metadata only.")
                    entry, created = library.add(p, tags, dest)
                else:
                    raise library.LibraryError(f"'{ref}' is not an arXiv id, DOI or existing file.")
                added.append({**entry, "created": created})
            except (net.SourceError, library.LibraryError) as exc:
                failed.append({"ref": ref, "error": str(exc)})
                output.error(f"{ref}: {exc}")
    if json_out:
        output.emit_json({"added": added, "failed": failed, "library": str(library.library_path())})
    else:
        for e in added:
            verb = "added" if e["created"] else "updated"
            output.console.print(f"[green]{verb}[/green] {e['key']}  {e.get('title') or ''}")
    if failed and not added:
        raise typer.Exit(code=1)


@library_app.command("list")
@handled
def library_list(
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Only papers with this tag."),
    json_out: bool = JSON,
) -> None:
    """List the papers in your library."""
    items = library.load()
    if tag:
        items = [i for i in items if tag in (i.get("tags") or [])]
    if json_out:
        output.emit_json(items)
    else:
        output.library_table(items)


@library_app.command("search")
@handled
def library_search(
    query: str = typer.Argument(..., metavar="QUERY", help="Words to look for."),
    limit: int = typer.Option(10, "--limit", "-n", min=1, max=100),
    json_out: bool = JSON,
) -> None:
    """Keyword search over your library (offline)."""
    hits = library.search(query, limit)
    if json_out:
        output.emit_json(hits)
    else:
        output.library_table(hits)


@library_app.command("remove")
@handled
def library_remove(ref: str = typer.Argument(..., help="Key, arXiv id, DOI or number from `kx library list`."),
                   json_out: bool = JSON) -> None:
    """Remove a paper from your library (files on disk are left alone)."""
    entry = library.remove(ref)
    if entry is None:
        output.error(f"'{ref}' is not in your library.")
        raise typer.Exit(code=1)
    if json_out:
        output.emit_json({"removed": entry["key"]})
    else:
        output.console.print(f"[green]removed[/green] {entry['key']}")


@library_app.command("export")
@handled
def library_export(
    fmt: str = typer.Option("bibtex", "--format", "-f", help="bibtex | apa | json"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write to a file instead of stdout."),
    json_out: bool = JSON,
) -> None:
    """Export your library as BibTeX, APA or JSON."""
    items = library.load()
    if fmt == "json":
        text = json.dumps({"version": library.VERSION, "papers": items}, indent=2, ensure_ascii=False)
    elif fmt == "bibtex":
        text = cite_mod.bibtex_many(items)  # unique keys even for same-author, same-year papers
    elif fmt == "apa":
        text = "\n".join(cite_mod.apa(i) for i in items)
    else:
        output.error("--format must be bibtex, apa or json.")
        raise typer.Exit(code=2)
    if out:
        config.write_text_atomic(Path(out), text + "\n")
    if json_out:
        output.emit_json({"format": fmt, "count": len(items), "path": str(out) if out else None,
                          "content": None if out else text})
    elif out:
        output.console.print(f"[green]wrote[/green] {len(items)} entries to {out}")
    else:
        print(text)


# ── Local RAG ─────────────────────────────────────────────────────────────

@rag_app.command("index")
@handled
def rag_index_cmd(
    path: List[Path] = typer.Option(None, "--path", "-p", help="Extra folder or file of PDFs/notes (remembered)."),
    forget: List[Path] = typer.Option(None, "--forget", help="Stop indexing a remembered path."),
    forget_all: bool = typer.Option(False, "--forget-all-paths", help="Stop indexing every remembered path."),
    embeddings: Optional[bool] = typer.Option(None, "--embeddings/--no-embeddings",
                                              help="Require / skip Ollama embeddings (default: use if running)."),
    json_out: bool = JSON,
) -> None:
    """Build the local index from your library (plus any --path)."""
    def progress(done, total):
        if not json_out:
            output.err.print(f"[dim]embedded {done}/{total} chunks[/dim]")
    summary = rag_index.build([str(p) for p in path or []], embeddings=embeddings, progress=progress,
                              forget=[str(p) for p in forget or []], forget_all=forget_all)
    for w in summary["warnings"]:
        output.warn(w)
    if json_out:
        output.emit_json(summary)
    else:
        mode = f"BM25 + {summary['embeddings']} embeddings" if summary["embeddings"] else "BM25 keyword search"
        output.console.print(f"[green]Indexed[/green] {summary['documents']} documents, "
                             f"{summary['chunks']} chunks ({mode}).")
        output.console.print(f"[dim]{summary['index']}. Stored on this machine only.[/dim]")


@rag_app.command("ask")
@handled
def rag_ask(
    question: str = typer.Argument(..., metavar="QUESTION", help="Your question."),
    k: int = typer.Option(5, "-k", min=1, max=50, help="Passages to retrieve."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Return passages only, even if Ollama is running."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model (default: config ollama_model)."),
    json_out: bool = JSON,
) -> None:
    """Ask your library. Private: runs on this machine (Ollama for answers if it is running)."""
    res = rag_index.ask(question, k=k, llm=not no_llm, model=model)
    if json_out:
        output.emit_json(res)
    else:
        output.rag_result(res)


@rag_app.command("status")
@handled
def rag_status(json_out: bool = JSON) -> None:
    """Show what is indexed."""
    idx = rag_index.load_index()
    data = ({"built_at": idx.get("built_at"), "documents": len(idx.get("docs", [])),
             "chunks": len(idx.get("chunks", [])), "embed_model": idx.get("embed_model"),
             "paths": idx.get("paths", []), "index": str(rag_index.index_path())} if idx else None)
    if json_out:
        output.emit_json({"index": data})
    elif not data:
        output.console.print("No index yet. Run: kx rag index")
    else:
        output.console.print(f"{data['documents']} documents, {data['chunks']} chunks, built {data['built_at']}"
                             + (f", embeddings {data['embed_model']}" if data["embed_model"] else ", BM25 only"))
        for p in data["paths"]:
            output.console.print(f"  [dim]{p}[/dim]")


# ── MCP ───────────────────────────────────────────────────────────────────

@app.command()
def mcp(
    config_for: Optional[str] = typer.Option(
        None, "--config", help="Print a config snippet for: claude-desktop, goose, cline, continue, openhands, open-webui, cursor."),
    info: bool = typer.Option(False, "--info", help="List the tools and whether the mcp package is installed."),
    json_out: bool = JSON,
) -> None:
    """Run an MCP server on stdio so agents (Goose, Cline, Continue, OpenHands, ...) can use kx."""
    from . import mcp_server
    if config_for:
        try:
            snip = mcp_server.config_snippet(config_for)
        except ValueError as exc:
            output.error(str(exc))
            raise typer.Exit(code=2)
        if json_out:
            output.emit_json(snip)
        else:
            output.err.print(f"# {mcp_server.CLIENTS[config_for]}: {snip['file']}")
            print(snip["snippet"].rstrip())
            if snip.get("note"):
                output.err.print(f"# {snip['note']}")
        return
    if info or json_out:
        data = mcp_server.info()
        if json_out:
            output.emit_json(data)
        else:
            output.console.print(f"Tools: {', '.join(data['tools'])}")
            output.console.print(f"mcp package installed: {data['mcp_installed']}")
        return
    try:
        mcp_server.run()
    except mcp_server.MissingMcp as exc:
        output.error(str(exc))
        raise typer.Exit(code=1)


# ── Config ────────────────────────────────────────────────────────────────

@config_app.command("show")
def config_show(json_out: bool = JSON) -> None:
    """Print the current settings and config file path."""
    cfg = config.load()
    if json_out:
        output.emit_json({"settings": cfg, "path": str(config.config_path()), "data_dir": str(config.data_dir())})
        return
    for k, v in cfg.items():
        output.console.print(f"{k} = {v}")
    output.console.print(f"[dim]{config.config_path()}[/dim]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., metavar="KEY"), json_out: bool = JSON) -> None:
    """Print one setting."""
    cfg = config.load()
    if key not in cfg:
        output.error(f"unknown setting {key!r}. Settings: {', '.join(sorted(config.KEYS))}")
        raise typer.Exit(code=1)
    if json_out:
        output.emit_json({key: cfg[key]})
    else:
        output.console.print(str(cfg[key]))


@config_app.command("set")
def config_set(key: str = typer.Argument(..., metavar="KEY"), value: str = typer.Argument(..., metavar="VALUE"),
               json_out: bool = JSON) -> None:
    """Change one setting: endpoint, limit, ollama_url, ollama_model, embed_model, token_store (auto|file)."""
    try:
        val = config.validate(key, value)
    except ValueError as exc:
        output.error(str(exc))
        raise typer.Exit(code=2)
    path = config.set_value(key, val)
    if json_out:
        output.emit_json({key: val, "path": str(path)})
    else:
        output.console.print(f"[green]set[/green] {key} = {val}  [dim]{path}[/dim]", soft_wrap=True)


# ── helpers ───────────────────────────────────────────────────────────────

def _resolve(ident: str, client: httpx.Client) -> Paper | None:
    return resolve(ident, client)


def _download(paper: Paper, out: Path, client: httpx.Client) -> Path:
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    raw = paper.arxiv_id or paper.doi or "paper"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", raw)  # safe on every filesystem, Windows included
    dest = out / f"{name}.pdf"
    tmp = dest.with_suffix(".part")
    try:
        with client.stream("GET", paper.pdf_url) as resp:
            if resp.status_code >= 400:
                raise net.SourceError("PDF", f"download failed (HTTP {resp.status_code}).")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPError as exc:
        raise net.SourceError("PDF", f"download failed ({exc.__class__.__name__}).")
    os.replace(tmp, dest)
    return dest


def main() -> None:
    """Console entry point. Keeps output from crashing on non-UTF-8 consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    app()


if __name__ == "__main__":
    main()
