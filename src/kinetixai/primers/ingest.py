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


# A README is written for GitHub, not for print: badges, link syntax and HTML
# comments carried straight through and appeared verbatim in the PDF.
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")           # ![alt](url)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")            # [text](url) -> text
_MD_REFLINK = re.compile(r"\[([^\]]*)\]\[[^\]]*\]")        # [text][ref] -> text
_MD_REFDEF = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.M)  # [ref]: url
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_HTML_TAG = re.compile(r"<(?:/?)(?:p|br|div|span|img|a|h[1-6]|b|i|em|strong|"
                       r"center|details|summary|table|tr|td|th|ul|ol|li)\b[^>]*>",
                       re.I)
_BADGE_LINE = re.compile(r"^[\s|]*$", re.M)


def clean_markdown(text: str) -> str:
    """Readable prose from a README: drop badges, keep the words."""
    text = _HTML_COMMENT.sub("", text or "")
    text = _MD_IMAGE.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_REFLINK.sub(r"\1", text)
    text = _MD_REFDEF.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.M)   # rules
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                 # **bold**
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)  # *italic*
    text = re.sub(r"`([^`\n]+)`", r"\1", text)                   # `code`
    # Underscores are left alone on purpose: stripping them would mangle
    # identifiers like primer_copy_to, which READMEs are full of.
    # Unpaired markers are left over from a README with unbalanced markup,
    # or from prose truncated mid-emphasis. They carry no meaning in print.
    text = text.replace("**", "").replace("``", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@dataclass
class SourceMaterial:
    kind: str                      # topic | pptx | pdf | markdown | github | arxiv
    name: str
    text: str = ""
    papers: list[dict] = field(default_factory=list)   # verified metadata dicts
    files: list[str] = field(default_factory=list)     # source files actually read
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

_RST_DIRECTIVE = re.compile(r"^\s*\.\.\s+\w[\w-]*::.*(?:\n(?:\s{3,}.*|\s*))*", re.M)
_RST_ROLE = re.compile(r":\w+:`([^`]*)`")
_RST_LINK = re.compile(r"`([^<`]+?)\s*<[^>]+>`_+")


def _strip_rst(text: str) -> str:
    """Readable prose from reStructuredText: drop directives and roles."""
    text = _RST_DIRECTIVE.sub("", text or "")
    text = _RST_LINK.sub(r"\1", text)
    text = _RST_ROLE.sub(r"\1", text)
    text = re.sub(r"^[=~^\-`\'\"#*+]{3,}\s*$", "", text, flags=re.M)   # underlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    # One URL was tried, README.md at HEAD, so an .rst or lower-case project
    # contributed nothing but its metadata.
    readme = None
    for name in ("README.md", "README.rst", "README.txt", "README",
                 "readme.md", "docs/README.md"):
        rr = client.get(f"https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{name}")
        if rr.status_code == 200 and rr.text.strip():
            readme = (name, rr.text)
            break
    if readme is None:
        warnings.append("no README found; using repo metadata only.")
    else:
        name, body = readme
        if name.endswith(".rst"):
            body = _strip_rst(body)
        text_parts.append(body)
    return "\n\n".join(text_parts), warnings


def load(source: str | None, topic: str, client: httpx.Client | None = None,
         max_files: int = 0) -> SourceMaterial:
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
            raw = p.read_text(encoding="utf-8", errors="replace")
            text = clean_markdown(raw) if suffix != ".txt" else raw
            return SourceMaterial(kind="markdown", name=p.name, text=text)
        raise IngestError(f"unsupported source file type '{suffix}' "
                          "(supported: .pptx, .pdf, .md, .txt).")

    own_client = client is None
    client = client or net.client()
    try:
        if "github.com" in source:
            from . import repo as repo_mod
            try:
                digest = repo_mod.fetch(source, topic, client,
                                        max_files=max_files or repo_mod.DEFAULT_MAX_FILES)
            except repo_mod.RepoError as exc:
                raise IngestError(str(exc))
            return SourceMaterial(kind="github", name=digest.ref.slug,
                                  text=digest.to_text(), warnings=digest.warnings,
                                  files=[p for p, _ in digest.files])
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
