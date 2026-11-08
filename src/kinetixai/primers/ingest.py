"""Source ingestion for `kx primers build --source` and `kx convert`.

Accepts: a local .pptx / .pdf / .md / .txt file, a GitHub repo URL, or an
arXiv id / DOI / arXiv URL. Returns extracted text plus any verified paper
metadata found along the way (used for citations — never invented).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .. import net
from ..resolve import arxiv_id as _arxiv_id, resolve


class IngestError(RuntimeError):
    pass


@dataclass
class SourceMaterial:
    kind: str                      # topic | pptx | pdf | markdown | github | arxiv
    name: str
    text: str = ""
    papers: list[dict] = field(default_factory=list)   # verified metadata dicts
    warnings: list[str] = field(default_factory=list)


def _pptx_text(path: Path) -> str:
    try:
        from pptx import Presentation  # python-pptx, optional extra
    except ImportError:
        raise IngestError("reading .pptx needs python-pptx: pip install 'kinetixai[primers]'")
    prs = Presentation(str(path))
    out: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        lines = [f"## Slide {i}"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(run.text for run in para.runs).strip()
                    if t:
                        lines.append(t)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [c.text_frame.text.strip() for c in row.cells]
                    lines.append(" | ".join(cells))
        notes = getattr(slide, "notes_slide", None)
        if notes is not None and notes.notes_text_frame is not None:
            nt = notes.notes_text_frame.text.strip()
            if nt:
                lines.append(f"(speaker notes) {nt}")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def _pdf_text(path: Path, max_pages: int = 60) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise IngestError("reading PDFs needs pypdf: pip install 'kinetixai[rag]'")
    reader = PdfReader(str(path))
    pages = reader.pages[:max_pages]
    return "\n\n".join((p.extract_text() or "") for p in pages)


_GH = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)")


def _github_text(url: str, client: httpx.Client) -> tuple[str, list[str]]:
    m = _GH.search(url)
    if not m:
        raise IngestError(f"'{url}' is not a GitHub repository URL.")
    owner, repo = m.group(1), m.group(2).removesuffix(".git")
    warnings: list[str] = []
    text_parts: list[str] = []
    r = client.get(f"https://api.github.com/repos/{owner}/{repo}",
                   headers={"Accept": "application/vnd.github+json"})
    if r.status_code == 200:
        meta = r.json()
        text_parts.append(f"# {meta.get('full_name')}\n{meta.get('description') or ''}\n"
                          f"Language: {meta.get('language')}  Stars: {meta.get('stargazers_count')}")
    else:
        warnings.append(f"GitHub repo metadata unavailable (HTTP {r.status_code}).")
    for branch in ("HEAD",):
        rr = client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md")
        if rr.status_code == 200 and rr.text.strip():
            text_parts.append(rr.text)
            break
    else:
        warnings.append("no README.md found; using repo metadata only.")
    return "\n\n".join(text_parts), warnings


def load(source: str | None, topic: str, client: httpx.Client | None = None) -> SourceMaterial:
    """Build SourceMaterial from --source (or just the topic when None)."""
    if not source:
        return SourceMaterial(kind="topic", name=topic)

    p = Path(source).expanduser()
    if p.is_file():
        suffix = p.suffix.lower()
        if suffix == ".pptx":
            return SourceMaterial(kind="pptx", name=p.name, text=_pptx_text(p))
        if suffix == ".pdf":
            return SourceMaterial(kind="pdf", name=p.name, text=_pdf_text(p))
        if suffix in (".md", ".markdown", ".txt"):
            return SourceMaterial(kind="markdown", name=p.name,
                                  text=p.read_text(encoding="utf-8", errors="replace"))
        raise IngestError(f"unsupported source file type '{suffix}' "
                          "(supported: .pptx, .pdf, .md, .txt).")

    own_client = client is None
    client = client or net.client()
    try:
        if "github.com" in source:
            text, warnings = _github_text(source, client)
            return SourceMaterial(kind="github", name=source, text=text, warnings=warnings)
        if _arxiv_id(source) or source.startswith("10."):
            paper = resolve(source, client)
            if paper is None:
                raise IngestError(f"could not resolve '{source}' on arXiv/Crossref.")
            d = paper.to_dict()
            text = f"# {d.get('title')}\n\nAuthors: {', '.join(d.get('authors') or [])}\n\n" \
                   f"{d.get('abstract') or ''}"
            return SourceMaterial(kind="arxiv", name=source, text=text, papers=[d])
        raise IngestError(
            f"'{source}' is not an existing file, a GitHub URL, or an arXiv id/DOI.")
    finally:
        if own_client:
            client.close()
