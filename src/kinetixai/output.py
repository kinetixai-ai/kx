"""Rendering: human tables (rich) or machine JSON.

JSON is written as ASCII (non-ASCII escaped), so piping works on any
console code page, including legacy Windows ones.
"""
from __future__ import annotations

import functools
import json
import sys

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .models import Paper, Researcher

console = Console()
err = Console(stderr=True)


def emit_json(data) -> None:
    sys.stdout.write(json.dumps(data, indent=2, ensure_ascii=True, default=str) + "\n")
    sys.stdout.flush()


def warn(msg: str) -> None:
    err.print(f"[yellow]warning:[/yellow] {escape(str(msg))}", soft_wrap=True)


def error(msg: str) -> None:
    err.print(f"[red]error:[/red] {escape(str(msg))}", soft_wrap=True)


def _short_authors(authors: list[str], n: int = 3) -> str:
    if not authors:
        return ""
    head = ", ".join(authors[:n])
    return head + (f" +{len(authors) - n}" if len(authors) > n else "")


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _num(n) -> str:
    if n is None:
        return ""
    n = int(n)
    for div, suf in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if n >= div:
            return f"{n / div:.1f}{suf}".replace(".0", "")
    return str(n)


def papers(items: list[Paper], as_json: bool) -> None:
    if as_json:
        emit_json([p.to_dict() for p in items])
        return
    if not items:
        console.print("[dim]No results.[/dim]")
        return
    table = Table(show_lines=False, expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", ratio=5, no_wrap=False)
    table.add_column("Authors", ratio=3, no_wrap=False)
    table.add_column("Year", width=5)
    table.add_column("Cites", justify="right", width=6)
    table.add_column("Source", width=16)
    for i, p in enumerate(items, 1):
        table.add_row(
            str(i),
            escape(p.title or ""),
            escape(_short_authors(p.authors)),
            str(p.year or ""),
            str(p.citations if p.citations is not None else ""),
            escape(p.source),
        )
    console.print(table)


def paper_detail(p: dict) -> None:
    console.print(f"[bold]{escape(p.get('title') or '')}[/bold]")
    meta = [x for x in [_short_authors(p.get("authors") or [], 6), str(p.get("year") or ""), p.get("venue") or ""] if x]
    if meta:
        console.print(escape("  ·  ".join(meta)))
    for label, key in (("arXiv", "arxiv_id"), ("DOI", "doi"), ("URL", "url"), ("PDF", "pdf_url")):
        if p.get(key):
            console.print(f"[dim]{label}:[/dim] {escape(str(p[key]))}")
    if p.get("citations") is not None:
        console.print(f"[dim]Citations:[/dim] {p['citations']}")
    if p.get("abstract"):
        console.print()
        console.print(escape(p["abstract"]))


def researchers(items: list[Researcher], as_json: bool) -> None:
    if as_json:
        emit_json([r.to_dict() for r in items])
        return
    if not items:
        console.print("[dim]No researchers found.[/dim]")
        return
    for r in items:
        console.print(f"[bold]{escape(r.name)}[/bold]" + (f"  [dim]{r.orcid}[/dim]" if r.orcid else ""))
        if r.affiliation:
            console.print(f"  {escape(r.affiliation)}")
        if r.url:
            console.print(f"  [blue]{r.url}[/blue]")
        for w in r.works[:5]:
            console.print(f"    - {escape(w)}")
        console.print()


def answer(data: dict, as_json: bool) -> None:
    if as_json:
        emit_json(data)
        return
    console.print(escape(data.get("answer", "").strip()))
    cites = data.get("citations") or []
    if cites:
        console.print()
        console.print("[dim]Sources[/dim]")
        for c in cites:
            console.print(f"  \\[{c.get('n', '')}] {escape(str(c.get('source', '')))}")


def repos(data: dict) -> None:
    info = data.get("paper")
    if info and info.get("title"):
        console.print(f"[bold]{escape(info['title'])}[/bold]  [dim]{info.get('hf_url', '')}[/dim]")
    items = data.get("repos") or []
    if not items:
        console.print("[dim]No implementations found.[/dim]")
        return
    table = Table(expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Repository", ratio=4, min_width=18, overflow="fold")
    table.add_column("Stars", justify="right", width=7)
    table.add_column("Lang", width=10, overflow="ellipsis", no_wrap=True)
    table.add_column("License", width=12, overflow="ellipsis", no_wrap=True)
    table.add_column("About", ratio=3)
    for i, r in enumerate(items, 1):
        name = escape(r["name"]) + (" [green](official)[/green]" if r.get("official") else "")
        table.add_row(str(i), name, _num(r.get("stars")), escape(r.get("language") or ""),
                      escape(r.get("license") or ""), escape(_clip(r.get("description") or "", 90)))
    console.print(table)
    console.print("[dim]Open: https://github.com/<repository>[/dim]")


def hub(items: list[dict], kind: str) -> None:
    if not items:
        console.print(f"[dim]No {kind} found.[/dim]")
        return
    table = Table(expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Model" if kind == "models" else "Dataset", ratio=4)
    table.add_column("Downloads", justify="right", width=10)
    table.add_column("Likes", justify="right", width=7)
    table.add_column("License", width=14)
    table.add_column("Task", ratio=2)
    for i, m in enumerate(items, 1):
        table.add_row(str(i), escape(m["id"]), _num(m.get("downloads")), _num(m.get("likes")),
                      escape(m.get("license") or ""), escape(m.get("task") or ""))
    console.print(table)


def library_table(items: list[dict]) -> None:
    if not items:
        console.print("[dim]Your library is empty. Add one with: kx library add 1706.03762[/dim]")
        return
    table = Table(expand=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Key", ratio=2, min_width=16, overflow="fold")
    table.add_column("Title", ratio=5)
    table.add_column("Year", width=5)
    table.add_column("PDF", width=4)
    table.add_column("Tags", ratio=1, min_width=8, overflow="fold")
    for i, e in enumerate(items, 1):
        table.add_row(str(i), escape(e["key"]), escape(e.get("title") or ""), str(e.get("year") or ""),
                      "yes" if e.get("pdf_path") else "", escape(", ".join(e.get("tags") or [])))
    console.print(table)


def rag_result(res: dict) -> None:
    if res.get("answer"):
        console.print(escape(res["answer"]))
        console.print()
        console.print(f"[dim]Answered by {escape(res.get('model') or '')} via Ollama "
                      f"({res['retrieval']} retrieval).[/dim]")
        console.print("[dim]Sources[/dim]")
        for p in res["passages"]:
            page = f", p. {p['page']}" if p.get("page") else ""
            console.print(f"  \\[{p['n']}] {escape(p['title'])}{page}  [dim]{escape(p['key'])}[/dim]")
    else:
        if res.get("note"):
            console.print(f"[dim]{escape(res['note'])}[/dim]")
            console.print()
        for p in res["passages"]:
            page = f", p. {p['page']}" if p.get("page") else ""
            console.print(f"[bold]\\[{p['n']}] {escape(p['title'])}{page}[/bold]  [dim]{escape(p['key'])}[/dim]")
            text = p["text"]
            console.print("    " + escape(text[:600] + ("..." if len(text) > 600 else "")))
            console.print()
    if res.get("local"):
        console.print("[dim]Private: retrieval and answer ran on this machine; nothing was sent anywhere.[/dim]")
    else:
        console.print(f"[yellow]Note:[/yellow] text was sent to the Ollama server at {escape(res['ollama_url'])}.")


def doctor(ch: dict) -> None:
    print_ = functools.partial(console.print, soft_wrap=True)

    def mark(ok):
        return "[green]ok[/green]" if ok else "[red]no[/red]"

    print_(f"[bold]kx {ch['kx']}[/bold]")
    py = ch["python"]
    print_(f"  Python      {py['version']}  {mark(py['ok'])}  [dim]{escape(py['executable'])}[/dim]")
    o = ch["os"]
    print_(f"  OS          {o['system']} {o['release']} ({o['machine']})")
    for k, v in ch["paths"].items():
        print_(f"  {k:<11} [dim]{escape(str(v))}[/dim]")
    print_(f"  Endpoint    {escape(ch['endpoint'])}")
    a = ch["auth"]
    print_(f"  Token       {mark(a['signed_in_token'])} {('(' + a['token_source'] + ')') if a['token_source'] else ''}"
                  f"  [dim]store: {a['store']}[/dim]")
    lib = ch["library"]
    idx = lib.get("rag_index")
    print_(f"  Library     {lib.get('papers')} papers; RAG index: "
                  + (f"{idx['chunks']} chunks" + (f", embeddings {idx['embed_model']}" if idx.get("embed_model") else "")
                     if idx else "not built"))
    for name, e in ch["extras"].items():
        label = f"\\[{name}] extra"
        print_(f"  {label:<13}{mark(e['installed'])}  [dim]{e['for']}[/dim]")
    ol = ch["ollama"]
    print_(f"  Ollama      {mark(ol['running'])}  [dim]{escape(ol['url'])}[/dim]"
                  + (f"  models: {escape(', '.join(ol['models'][:6]))}" if ol["models"] else ""))
    for n in ch["network"]:
        detail = f"{n.get('status')} in {n['ms']} ms" if n.get("ms") is not None else n.get("error", "")
        print_(f"  {n['name']:<17} {mark(n['ok'])}  [dim]{detail}[/dim]")
    if ch["advice"]:
        print_()
        print_("[bold]Suggestions[/bold]")
        for line in ch["advice"]:
            print_(f"  - {escape(line)}")
