"""kx: the KinetixAI command line."""
from __future__ import annotations

import re
import webbrowser
from pathlib import Path

import httpx
import typer

from . import __version__, config, output
from .api import ApiError, KinetixApi
from .models import Paper
from .sources import aggregate, arxiv, orcid, semantic_scholar

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="KinetixAI command line: search papers, look up researchers, and ask KinetixAI.",
)
config_app = typer.Typer(no_args_is_help=True, help="View or change settings.")
app.add_typer(config_app, name="config")

ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
UA = f"kinetixai-cli/{__version__} (+https://kinetixai.ai)"


def _client() -> httpx.Client:
    return httpx.Client(timeout=25.0, follow_redirects=True, headers={"User-Agent": UA})


@app.command()
def version() -> None:
    """Print the kx version."""
    output.console.print(f"kx {__version__}")


@app.command()
def search(
    query: str = typer.Argument(..., help="What to search for."),
    limit: int = typer.Option(None, "--limit", "-n", help="Max results per source."),
    source: str = typer.Option(
        "all", "--source", "-s", help="all | arxiv | semantic | openalex"
    ),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON."),
    open_: bool = typer.Option(False, "--open", help="Open the top result in a browser."),
) -> None:
    """Search papers across arXiv, Semantic Scholar and OpenAlex."""
    limit = limit or config.get("limit", 10)
    sources = None if source == "all" else [source]
    with _client() as client:
        papers, warnings = aggregate.search(query, limit, client, sources)
    for w in warnings:
        output.err.print(f"[yellow]warning:[/yellow] {w}")
    output.papers(papers, json_out)
    if open_ and papers and papers[0].url:
        webbrowser.open(papers[0].url)


@app.command()
def scholar(
    query: str = typer.Argument(..., help="Researcher name or ORCID iD."),
    limit: int = typer.Option(10, "--limit", "-n"),
    json_out: bool = typer.Option(False, "--json"),
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
            output.err.print(f"[red]error:[/red] ORCID unavailable ({exc.__class__.__name__})")
            raise typer.Exit(code=1)
    output.researchers(people, json_out)


@app.command()
def get(
    ident: str = typer.Argument(..., help="arXiv id, DOI, or arXiv URL."),
    pdf: bool = typer.Option(False, "--pdf", help="Also download the PDF if available."),
    out: Path = typer.Option(Path("."), "--out", "-o", help="Directory for the PDF."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Fetch a single paper's metadata (and optionally its PDF)."""
    with _client() as client:
        paper = _resolve(ident, client)
        if paper is None:
            output.err.print(f"[red]error:[/red] could not resolve '{ident}'.")
            raise typer.Exit(code=1)
        output.papers([paper], json_out)
        if pdf:
            if not paper.pdf_url:
                output.err.print("[yellow]note:[/yellow] no open PDF available.")
                raise typer.Exit(code=2)
            dest = _download(paper, out, client)
            output.console.print(f"[green]saved[/green] {dest}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question for KinetixAI."),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Ask KinetixAI and get a grounded, cited answer."""
    endpoint = config.get("endpoint")
    with _client() as client:
        api = KinetixApi(endpoint, client)
        try:
            data = api.ask(question)
        except ApiError as exc:
            output.err.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1)
    output.answer(data, json_out)


@config_app.command("show")
def config_show() -> None:
    """Print the current settings and config file path."""
    cfg = config.load()
    for k, v in cfg.items():
        output.console.print(f"{k} = {v}")
    output.console.print(f"[dim]{config.config_path()}[/dim]")


@config_app.command("get")
def config_get(key: str) -> None:
    """Print one setting."""
    output.console.print(str(config.get(key, "")))


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Change one setting (e.g. endpoint, limit)."""
    val: object = value
    if key == "limit" and value.isdigit():
        val = int(value)
    path = config.set_value(key, val)
    output.console.print(f"[green]set[/green] {key} = {val}  [dim]{path}[/dim]")


def _resolve(ident: str, client: httpx.Client) -> Paper | None:
    ident = ident.strip()
    if "arxiv.org/abs/" in ident:
        ident = ident.rsplit("/abs/", 1)[-1]
    if ARXIV_ID.match(ident):
        r = client.get(arxiv.API, params={"id_list": ident, "max_results": 1})
        r.raise_for_status()
        papers = arxiv._parse(r.text)
        return papers[0] if papers else None
    doi = ident.replace("https://doi.org/", "")
    r = client.get(
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
        params={"fields": semantic_scholar.FIELDS},
    )
    if r.status_code == 200:
        return semantic_scholar._paper(r.json())
    return None


def _download(paper: Paper, out: Path, client: httpx.Client) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    name = (paper.arxiv_id or paper.doi or "paper").replace("/", "_")
    dest = out / f"{name}.pdf"
    with client.stream("GET", paper.pdf_url) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    return dest


if __name__ == "__main__":
    app()
