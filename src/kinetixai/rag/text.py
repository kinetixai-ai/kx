"""Text extraction and chunking."""
from __future__ import annotations

import re
from pathlib import Path

STOP = frozenset(
    "a an and are as at be but by can for from has have how in into is it its of on or that the their "
    "there these this to was we were what when where which while who why will with within without you your "
    "our not also than then thus such via using use used based".split()
)
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".tex"}


class MissingExtra(Exception):
    pass


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOP and len(t) > 1]


def pdf_pages(path: Path) -> list[tuple[int, str]]:
    """[(page_number, text)] using pypdf (the [rag] extra)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise MissingExtra('reading PDFs needs pypdf: pipx inject kinetixai pypdf  (or pip install "kinetixai[rag]")')
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = " ".join(txt.split())
        if txt:
            out.append((i, txt))
    return out


def file_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        return list(pdf_pages(path))
    if path.suffix.lower() in TEXT_SUFFIXES:
        return [(None, path.read_text(encoding="utf-8", errors="replace"))]
    return []


def chunk(text: str, size: int = 160, overlap: int = 40) -> list[str]:
    """Split into overlapping windows of roughly `size` words."""
    words = (text or "").split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out
