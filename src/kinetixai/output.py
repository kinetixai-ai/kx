"""Rendering: human tables (rich) or machine JSON."""
from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from .models import Paper, Researcher

console = Console()
err = Console(stderr=True)


def _short_authors(authors: list[str], n: int = 3) -> str:
    if not authors:
        return ""
    head = ", ".join(authors[:n])
    return head + (f" +{len(authors) - n}" if len(authors) > n else "")


def papers(items: list[Paper], as_json: bool) -> None:
    if as_json:
        console.print_json(json.dumps([p.to_dict() for p in items]))
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
            p.title or "",
            _short_authors(p.authors),
            str(p.year or ""),
            str(p.citations if p.citations is not None else ""),
            p.source,
        )
    console.print(table)


def researchers(items: list[Researcher], as_json: bool) -> None:
    if as_json:
        console.print_json(json.dumps([r.to_dict() for r in items]))
        return
    if not items:
        console.print("[dim]No researchers found.[/dim]")
        return
    for r in items:
        console.print(f"[bold]{r.name}[/bold]" + (f"  [dim]{r.orcid}[/dim]" if r.orcid else ""))
        if r.affiliation:
            console.print(f"  {r.affiliation}")
        if r.url:
            console.print(f"  [blue]{r.url}[/blue]")
        for w in r.works[:5]:
            console.print(f"    - {w}")
        console.print()


def answer(data: dict, as_json: bool) -> None:
    if as_json:
        console.print_json(json.dumps(data))
        return
    console.print(data.get("answer", "").strip())
    cites = data.get("citations") or []
    if cites:
        console.print()
        console.print("[dim]Sources[/dim]")
        for c in cites:
            n = c.get("n", "")
            src = c.get("source", "")
            console.print(f"  [{n}] {src}")
