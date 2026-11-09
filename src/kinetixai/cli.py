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
primers_app = typer.Typer(**TYPER_SAFE, no_args_is_help=True,
                          help='"Story to Silicon" Primers: generate styled PDF/PPTX/PNG/JSON bundles.')
app.add_typer(config_app, name="config")
app.add_typer(library_app, name="library")
app.add_typer(rag_app, name="rag")
app.add_typer(primers_app, name="primers")

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


# ── Primers ───────────────────────────────────────────────────────────────

def _primer_deps():
    """Import the primers package lazily with one clear install hint."""
    try:
        from .primers import builder as primers_builder  # noqa: F401
        from .primers import catalog as primers_catalog  # noqa: F401
        from .primers import engines as primers_engines  # noqa: F401
        from .primers import ingest as primers_ingest    # noqa: F401
        from .primers import renderer as primers_renderer  # noqa: F401
        return primers_builder, primers_catalog, primers_engines, primers_ingest, primers_renderer
    except ImportError as exc:
        output.error(f"primers needs extra packages ({exc.name}). "
                     "Install them with: pip install 'kinetixai[primers]'")
        raise typer.Exit(code=1)


def _primer_errors():
    from .primers import builder as b, catalog as c, engines as e, ingest as i, renderer as r
    from .primers.exporters.images import ImageExportError
    from .primers.exporters.pptx import PptxExportError
    return (e.EngineError, i.IngestError, r.RenderError, c.ManifestError,
            b.MissingDependency, ImageExportError, PptxExportError)


def _check_engine(engine: str, primers_engines) -> str:
    """Accept any alias (claude, lm-studio, ...) and return the canonical name."""
    name = primers_engines.canonical(engine)
    if name == "auto":
        return name
    if name not in primers_engines.ENGINES:
        output.error(f"--engine must be one of: {', '.join(primers_engines.ALL_ENGINE_NAMES)}")
        raise typer.Exit(code=2)
    return name


def _run_build(req, json_out: bool) -> dict:
    from .primers import builder as primers_builder
    log = (lambda s: None) if json_out else (lambda s: output.err.print(f"[dim]{s}[/dim]"))
    try:
        res = primers_builder.build(req, log=log)
    except _primer_errors() as exc:
        output.error(str(exc))
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        output.error("interrupted.")
        raise typer.Exit(code=130)
    except Exception as exc:  # noqa: BLE001 — a build must never end in a traceback
        output.error(f"{exc.__class__.__name__}: {exc}")
        output.err.print("[dim]If this looks like a bug, please report it with this line "
                         "at https://github.com/kinetixai-ai/kx/issues[/dim]")
        raise typer.Exit(code=1)
    for w in res.warnings:
        output.warn(w)
    d = res.to_dict()
    if json_out:
        output.emit_json(d)
        return d
    output.console.print(f"[green]built[/green] [bold]{d['title']}[/bold] ({d['pages']} pages, "
                         f"{d['citations']} verified citations)")
    for label, items in (("pdf", d["pdfs"]), ("png", d["images"]),
                         ("pptx", d["ppts"]), ("json", d["json"])):
        for p in items:
            output.console.print(f"  {label:>4}  {p}")
    if d["html"]:
        output.console.print(f"  html  {d['html']}  [dim](editable)[/dim]")
    if d.get("copied"):
        folder = Path(d["copied"][0]).parent
        folder = folder.parent if folder.name == "images" else folder
        output.console.print(f"  [green]copied[/green] {len(d['copied'])} files to {folder}")
    return d


@primers_app.command("build")
def primers_build(
    topic: str = typer.Option(..., "--topic", "-t", help="Topic, e.g. 'GPU & CUDA: Kernel Fusion'."),
    source: Optional[str] = typer.Option(None, "--source", "-s",
                                         help="Optional .pptx/.pdf/.md file, GitHub URL, or arXiv id/DOI."),
    category: str = typer.Option("", "--category", help="Catalog category slug, e.g. gpu, rag, security."),
    slug: str = typer.Option("", "--slug",
                             help="Output filename stem. Set it to match a published page, "
                                  "e.g. --slug gpu-cuda-kernel-fusion. Default: from the title."),
    persona: str = typer.Option("student", "--persona",
                                help="Comma list: student,phd,researcher,professor,professional,lab,ai-security."),
    pages: int = typer.Option(8, "--pages", min=1, max=42, help="Exact page count per PDF."),
    pdfs: int = typer.Option(1, "--pdfs", min=0, help="Styled PDF variants to generate."),
    ppts: int = typer.Option(0, "--ppts", min=0, help="Editable .pptx decks (with speaker notes)."),
    images: int = typer.Option(0, "--images", min=0, help="300 DPI PNG page cards to export."),
    json_n: int = typer.Option(0, "--json-bundles", min=0,
                               help="Paper Lab JSON bundles (tabs + bibtex + rag_chunks)."),
    watermark: Optional[str] = typer.Option(None, "--watermark",
                                            help="Running header/footer watermark. Default comes from "
                                                 "`kx config set primer_watermark`."),
    author: str = typer.Option("Rohit Patel", "--author"),
    engine: Optional[str] = typer.Option(None, "--engine",
                                         help="auto | ollama | vllm | lmstudio | llamacpp | claude | "
                                              "gemini | openai | groq | openrouter | mcp | outline ... "
                                              "(`kx primers engines` lists all)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name for the chosen engine."),
    api_base: Optional[str] = typer.Option(None, "--api-base",
                                           help="Override the engine's endpoint, e.g. http://localhost:8000/v1"),
    mode: str = typer.Option("iterative", "--mode",
                             help="iterative-per-page (dense, one call per page) | single (one call, large models)"),
    num_ctx: Optional[int] = typer.Option(None, "--num-ctx", min=2048,
                                          help="Ollama context window. Its 4096 default is too small "
                                               "for a full page; kx asks for 32768 unless set."),
    min_sections: int = typer.Option(5, "--min-sections-per-page", min=1, max=9,
                                     help="Density floor: numbered sections required per page."),
    relevance: float = typer.Option(0.18, "--relevance", min=0.0, max=1.0,
                                    help="Citation domain-match threshold; 0 disables filtering."),
    max_files: int = typer.Option(0, "--max-files", min=0, max=200,
                                  help="For a GitHub source: how many files to read "
                                       "(0 uses the default, 40)."),
    verify_citations: bool = typer.Option(False, "--verify-citations",
                                          help="Re-resolve every citation; drop any that fail."),
    out_dir: Path = typer.Option(Path("./dist/primers"), "--out-dir", "-o"),
    copy_to: Optional[Path] = typer.Option(None, "--copy-to",
                                           help="Also copy the finished files here, e.g. a notes "
                                                "folder or a shared drive. Set a permanent one with "
                                                "`kx config set primer_copy_to <folder>`."),
    json_out: bool = JSON,
) -> None:
    """Research, synthesize and compile a 'Story to Silicon' Primer bundle."""
    primers_builder, _, primers_engines, _, _ = _primer_deps()
    engine = _check_engine(engine or config.get("primer_engine", "auto"), primers_engines)
    model = model or (config.get("primer_model", "") or None)
    mode_v = "iterative" if mode.startswith("iterative") else mode
    if mode_v not in primers_builder.MODES:
        output.error(f"--mode must be one of: {', '.join(primers_builder.MODES)}")
        raise typer.Exit(code=2)
    req = primers_builder.BuildRequest(
        topic=topic, source=source, category=category, slug=slug,
        personas=[p.strip() for p in persona.split(",") if p.strip()],
        pages=pages, pdfs=pdfs, ppts=ppts, images=images, jsons=json_n,
        watermark=watermark or config.get("primer_watermark", "Narmadya Innovations"),
        author=author, engine=engine, model=model,
        api_base=api_base, num_ctx=num_ctx, mode=mode_v,
        min_sections=min_sections, relevance=relevance, max_files=max_files,
        verify_citations=verify_citations, out_dir=out_dir,
        copy_to=copy_to or (config.get("primer_copy_to", "") or None))
    _run_build(req, json_out)


@primers_app.command("compare")
def primers_compare(
    topic: str = typer.Option(..., "--topic", "-t", help="Topic to sample."),
    models: str = typer.Option("", "--models", "-m",
                               help="Comma list. Default: every chat model the engine serves."),
    engine: Optional[str] = typer.Option(None, "--engine",
                                         help="Which backend the models belong to."),
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    persona: str = typer.Option("student", "--persona"),
    pages: int = typer.Option(1, "--pages", min=1, max=4,
                              help="Pages per candidate. 1 is usually enough to judge."),
    api_base: Optional[str] = typer.Option(None, "--api-base"),
    num_ctx: Optional[int] = typer.Option(None, "--num-ctx", min=2048),
    out_dir: Path = typer.Option(Path("./dist/compare"), "--out-dir", "-o"),
    json_out: bool = JSON,
) -> None:
    """Build the same sample with several models and show time and density.

    Finding out after an eight-page run that a model is too slow or writes
    thin pages is expensive. This builds one page per candidate so you can
    read them side by side and then set the winner with
    `kx config set primer_model <name>`.
    """
    primers_builder, _, primers_engines, _, _ = _primer_deps()
    from .primers import providers as pv
    name = _check_engine(engine or config.get("primer_engine", "auto"), primers_engines)
    if name == "auto":
        name, why = pv.pick()
        if not json_out:
            output.err.print(f"[dim]engine: {name} ({why})[/dim]")
    wanted = [m.strip() for m in models.split(",") if m.strip()]
    if not wanted:
        provider = pv.get(name)
        wanted = pv.chat_models(provider) if provider and provider.local else []
        if not wanted:
            output.error("pass --models with the names to compare "
                         f"(no local models found for engine '{name}').")
            raise typer.Exit(code=2)
        if not json_out:
            output.err.print(f"[dim]comparing {len(wanted)} models served by {name}[/dim]")

    req = primers_builder.BuildRequest(
        topic=topic, source=source,
        personas=[p.strip() for p in persona.split(",") if p.strip()],
        pages=pages, pdfs=1, ppts=0, images=0, jsons=0,
        engine=name, api_base=api_base, num_ctx=num_ctx,
        watermark=config.get("primer_watermark", "Narmadya Innovations"),
        out_dir=out_dir)
    log = (lambda s: None) if json_out else (lambda s: output.err.print(f"[dim]{s}[/dim]"))
    rows = primers_builder.compare_models(req, wanted, log=log)
    if json_out:
        output.emit_json({"engine": name, "topic": topic, "results": rows})
        return
    output.compare_table(rows)
    good = [r for r in rows if r.get("ok")]
    if good:
        best = min(good, key=lambda r: (-r["met_quota"], r["seconds"]))
        output.console.print(f"\n[dim]fastest that met the quota:[/dim] [bold]{best['model']}[/bold]")
        output.console.print(f"[dim]keep it:[/dim] kx config set primer_model {best['model']}")


@primers_app.command("engines")
def primers_engines_cmd(
    check: bool = typer.Option(True, "--check/--no-check",
                               help="Probe local servers to see which are running."),
    json_out: bool = JSON,
) -> None:
    """List every engine a Primer can be written with, and which are ready."""
    _primer_deps()
    from .primers import providers as pv
    rows = pv.status(check_local=check)
    if json_out:
        chosen, why = pv.pick(check_local=check)
        output.emit_json({"engines": rows, "auto": {"engine": chosen, "why": why}})
        return
    output.engines_table(rows)
    chosen, why = pv.pick(check_local=check)
    output.console.print(f"\n[dim]auto would use:[/dim] [bold]{chosen}[/bold]  [dim]({why})[/dim]")
    output.console.print("[dim]Set a default:[/dim] kx config set primer_engine <name>")


@primers_app.command("models")
def primers_models_cmd(
    engine: str = typer.Argument(..., help="Engine name, e.g. groq, gemini, ollama."),
    verify: bool = typer.Option(False, "--verify",
                                help="Send a tiny request to each model to see which "
                                     "can actually be called. A listed model can still "
                                     "be blocked at the project level."),
    json_out: bool = JSON,
) -> None:
    """List the chat models an engine's endpoint actually serves.

    A built-in default can go stale and a hosted key is only entitled to
    part of the catalogue, so this is the list --model can be set from.
    Add --verify to find out which of them the project actually permits.
    """
    _primer_deps()
    from .primers import providers as pv
    p = pv.get(engine)
    if p is None:
        output.error(f"unknown engine {engine!r}. Try: kx primers engines")
        raise typer.Exit(2)
    if p.protocol in ("mcp", "none"):
        output.error(f"--engine {p.name} does not use a model.")
        raise typer.Exit(2)
    if p.keys and not pv.api_key(p) and not p.local:
        output.error(f"set {p.keys[0]} first, then re-run.")
        raise typer.Exit(2)
    served = pv.rank_models(pv.chat_models(p, timeout=10.0))
    if not served and not json_out:
        output.error(f"{p.label} returned no chat models. "
                     "Check the key, or the endpoint if it is self-hosted.")
        raise typer.Exit(1)

    checks: list[dict] = []
    usable = served
    if verify and served:
        from .primers import engines as primers_engines
        if not json_out:
            output.err.print(f"[dim]checking {len(served)} model(s)…[/dim]")
        # Check all of them, not just up to the first that works: the point
        # of --verify is to see the whole picture before a long batch run.
        for m in served:
            ok, why = primers_engines.verify_model(p.name, m)
            checks.append({"model": m, "callable": ok, "detail": why})
        usable = [c["model"] for c in checks if c["callable"]]

    if json_out:
        output.emit_json({"engine": p.name, "models": served,
                          "verified": verify, "checks": checks,
                          "callable": usable if verify else None,
                          "would_use": (usable[0] if usable else None)})
        return

    output.console.print(f"[bold]{p.label}[/bold] serves {len(served)} chat model(s):")
    by_model = {c["model"]: c for c in checks}
    for m in served:
        c = by_model.get(m)
        if c is None:
            output.console.print(f"  {m}")
        elif c["callable"]:
            output.console.print(f"  [green]✓[/green] {m}")
        else:
            output.console.print(f"  [red]✗[/red] {m}  [dim]{c['detail']}[/dim]")
    if verify and not usable:
        output.error("none of them can be called with this key. "
                     "If they show as blocked, enable them in the provider's "
                     "project settings, then re-run.")
        raise typer.Exit(1)
    output.console.print(
        f"\n[dim]without --model, kx would use:[/dim] [bold]{usable[0]}[/bold]")


@primers_app.command("autobuild")
def primers_autobuild(
    manifest: Path = typer.Option(Path("./primers_catalog.yaml"), "--manifest",
                                  help="YAML manifest of catalog tracks."),
    parallel: int = typer.Option(2, "--parallel", min=1, max=8, help="Concurrent builds."),
    json_out: bool = JSON,
) -> None:
    """Batch-build every Primer track in a YAML manifest."""
    _, primers_catalog, *_ = _primer_deps()
    if not manifest.is_file():
        output.error(f"manifest not found: {manifest}")
        raise typer.Exit(code=2)
    log = (lambda s: None) if json_out else (lambda s: output.err.print(s))
    try:
        results = primers_catalog.autobuild(manifest, parallel=parallel, log=log)
    except _primer_errors() as exc:
        output.error(str(exc))
        raise typer.Exit(code=1)
    ok = sum(1 for r in results if r["ok"])
    if json_out:
        output.emit_json({"built": ok, "failed": len(results) - ok, "tracks": results})
    else:
        output.console.print(f"[green]{ok}[/green]/{len(results)} tracks built.")
    if ok < len(results):
        raise typer.Exit(code=1)


@app.command()
def convert(
    source: str = typer.Argument(..., metavar="SOURCE",
                                 help="A .pptx/.pdf/.md file, GitHub URL, or arXiv id/DOI."),
    style: str = typer.Option("trifleet-8page", "--style", help="Output style preset."),
    topic: Optional[str] = typer.Option(None, "--topic", "-t",
                                        help="Title override (default: derived from the source)."),
    persona: str = typer.Option("professional", "--persona"),
    pages: int = typer.Option(8, "--pages", min=1, max=42),
    pdfs: int = typer.Option(1, "--pdfs", min=0),
    ppts: int = typer.Option(0, "--ppts", min=0),
    images: int = typer.Option(0, "--images", min=0),
    json_n: int = typer.Option(0, "--json-bundles", min=0),
    watermark: Optional[str] = typer.Option(None, "--watermark"),
    author: str = typer.Option("Rohit Patel", "--author"),
    engine: Optional[str] = typer.Option(None, "--engine",
                                         help="auto | ollama | vllm | claude | gemini | openai | mcp | "
                                              "outline ... (`kx primers engines` lists all)"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    api_base: Optional[str] = typer.Option(None, "--api-base"),
    num_ctx: Optional[int] = typer.Option(None, "--num-ctx", min=2048),
    mode: str = typer.Option("iterative", "--mode"),
    max_files: int = typer.Option(0, "--max-files", min=0, max=200,
                                  help="For a GitHub source: how many files to read."),
    out: Optional[Path] = typer.Option(None, "--out", "-o",
                                       help="Output PDF path; its folder receives the other artifacts."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir",
                                           help="Output folder, when you do not need an exact PDF name. "
                                                "Same flag as `kx primers build`."),
    copy_to: Optional[Path] = typer.Option(None, "--copy-to",
                                           help="Also copy the finished files here."),
    json_out: bool = JSON,
) -> None:
    """Convert an existing deck, paper or repo into the TriFleet 8-page format."""
    primers_builder, _, primers_engines, _, _ = _primer_deps()
    engine = _check_engine(engine or config.get("primer_engine", "auto"), primers_engines)
    model = model or (config.get("primer_model", "") or None)
    if out and out_dir:
        output.error("pass --out (an exact PDF path) or --out-dir (a folder), not both.")
        raise typer.Exit(code=2)
    name = topic or Path(source).stem.replace("_", " ").replace("-", " ").strip() or source
    target_dir = out.parent if out else (out_dir or Path("./dist/convert"))
    req = primers_builder.BuildRequest(
        topic=name, source=source, personas=[p.strip() for p in persona.split(",") if p.strip()],
        pages=pages, pdfs=pdfs, ppts=ppts, images=images, jsons=json_n,
        watermark=watermark or config.get("primer_watermark", "Narmadya Innovations"),
        author=author, engine=engine, model=model,
        api_base=api_base, num_ctx=num_ctx, max_files=max_files,
        mode="iterative" if mode.startswith("iterative") else mode,
        out_dir=target_dir, style=style,
        copy_to=copy_to or (config.get("primer_copy_to", "") or None))
    result = _run_build(req, json_out)
    if out:
        # Rename the file that was actually built. Guessing the name from the
        # requested title silently did nothing when an LLM retitled the primer.
        built = Path(result["pdfs"][0]) if result.get("pdfs") else None
        if built is None:
            output.warn("--out was given but no PDF was produced (--pdfs 0?); "
                        "nothing to rename.")
        elif built.resolve() != out.resolve():
            out.parent.mkdir(parents=True, exist_ok=True)
            os.replace(built, out)
            if not json_out:
                output.console.print(f"[green]saved[/green] {out}")


# ── MCP ───────────────────────────────────────────────────────────────────

mcp_app = typer.Typer(**TYPER_SAFE, invoke_without_command=True,
                      help="Run kx as an MCP server so agents (Claude Code, Cursor, Goose, ...) can use it.")
app.add_typer(mcp_app, name="mcp")


def _serve() -> None:
    from . import mcp_server
    try:
        mcp_server.run()
    except mcp_server.MissingMcp as exc:
        output.error(str(exc))
        raise typer.Exit(code=1)


def _print_snippet(client: str, json_out: bool) -> None:
    from . import mcp_server
    try:
        snip = mcp_server.config_snippet(client)
    except ValueError as exc:
        output.error(str(exc))
        raise typer.Exit(code=2)
    if json_out:
        output.emit_json(snip)
        return
    output.err.print(f"# {mcp_server.CLIENTS[client]}: {snip['file']}")
    print(snip["snippet"].rstrip())
    if snip.get("note"):
        output.err.print(f"# {snip['note']}")


def _print_info(json_out: bool) -> None:
    from . import mcp_server
    data = mcp_server.info()
    if json_out:
        output.emit_json(data)
    else:
        output.console.print(f"Tools: {', '.join(data['tools'])}")
        output.console.print(f"mcp package installed: {data['mcp_installed']}")


@mcp_app.callback()
def mcp_main(
    ctx: typer.Context,
    config_for: Optional[str] = typer.Option(
        None, "--config", help="Print a config snippet for: claude-code, claude-desktop, goose, "
                               "cline, continue, openhands, open-webui, cursor."),
    info: bool = typer.Option(False, "--info", help="List the tools and whether the mcp package is installed."),
    json_out: bool = JSON,
) -> None:
    """Run the server on stdio. `kx mcp` with no subcommand still works."""
    if ctx.invoked_subcommand is not None:
        return
    if config_for:
        _print_snippet(config_for, json_out)
        return
    if info or json_out:
        _print_info(json_out)
        return
    _serve()


@mcp_app.command("serve")
def mcp_serve() -> None:
    """Run the MCP server on stdio (what an agent client launches)."""
    _serve()


@mcp_app.command("install")
def mcp_install(
    client: str = typer.Argument("claude-code", metavar="CLIENT",
                                 help="claude-code | claude-desktop | cursor | goose | cline | "
                                      "continue | openhands | open-webui"),
    json_out: bool = JSON,
) -> None:
    """Print the configuration that registers kx with an agent client."""
    _print_snippet(client, json_out)


@mcp_app.command("info")
def mcp_info(json_out: bool = JSON) -> None:
    """List the tools this server exposes and whether the mcp package is installed."""
    _print_info(json_out)


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
    """Change one setting. `kx config show` lists them all with their values.

    endpoint, limit, ollama_url, ollama_model, embed_model, token_store
    (auto|file), primer_engine, primer_model, primer_watermark, primer_copy_to.
    """
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
