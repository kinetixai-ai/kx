"""Your local paper library: one JSON file in the data directory.

Nothing here touches the network. Entries are Paper dicts plus a stable
key ("arxiv:2205.14135", "doi:10.x/y" or "local:<slug>"), tags, when it was
added, and an optional path to a PDF on disk (used by `kx rag index`).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .models import Paper

VERSION = 1


def library_path() -> Path:
    return config.data_dir() / "library.json"


def pdf_dir() -> Path:
    return config.data_dir() / "pdfs"


def _slug(text: str) -> str:
    """ASCII slug for readability only; may be empty (e.g. a CJK-only title)."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40].strip("-")


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _meta_digest(p: Paper) -> str:
    norm = lambda t: " ".join(unicodedata.normalize("NFKC", t or "").casefold().split())
    raw = "\x1f".join([norm(p.title), "|".join(norm(a) for a in p.authors or []), str(p.year or "")])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def key_for(p: Paper, file: Path | None = None) -> str:
    """arxiv:<id> or doi:<doi> when known. Otherwise local:<slug>-<hash>, where
    the hash is of the file's bytes (so two different PDFs never share a key,
    whatever their titles) or, with no file, of title + authors + year."""
    if p.arxiv_id:
        return "arxiv:" + re.sub(r"v\d+$", "", p.arxiv_id.strip())
    if p.doi:
        return "doi:" + p.doi.strip().lower()
    digest = file_digest(file) if file is not None and Path(file).is_file() else _meta_digest(p)
    slug = _slug(p.title)
    return "local:" + (f"{slug}-" if slug else "") + digest[:12]


def load() -> list[dict]:
    path = library_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LibraryError(f"could not read {path} ({exc.__class__.__name__}). Fix or move the file.")
    items = data.get("papers") if isinstance(data, dict) else data
    return [i for i in (items or []) if isinstance(i, dict) and i.get("key")]


def save(items: list[dict]) -> Path:
    path = library_path()
    config.write_text_atomic(path, json.dumps({"version": VERSION, "papers": items}, indent=2, ensure_ascii=False) + "\n")
    return path


class LibraryError(Exception):
    pass


def add(paper: Paper, tags: list[str] | None = None, pdf_path: Path | None = None) -> tuple[dict, bool]:
    """Add or update a paper. Returns (entry, created)."""
    items = load()
    key = key_for(paper, pdf_path)
    entry = next((i for i in items if i["key"] == key), None)
    created = entry is None
    fresh = paper.to_dict()
    if created:
        entry = {"key": key, **fresh, "tags": [], "pdf_path": None,
                 "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        items.append(entry)
    else:
        for k, v in fresh.items():  # fill gaps, never blank out what we had
            if v not in (None, "", []):
                entry[k] = v
    for t in tags or []:
        if t and t not in entry["tags"]:
            entry["tags"].append(t)
    if pdf_path is not None:
        entry["pdf_path"] = str(Path(pdf_path).expanduser().resolve())
    save(items)
    return entry, created


def find(ref: str, items: list[dict] | None = None) -> dict | None:
    """Look up by key, bare arXiv id / DOI, or 1-based position in `kx library list`."""
    items = load() if items is None else items
    ref = (ref or "").strip()
    if ref.isdigit() and 1 <= int(ref) <= len(items) and "." not in ref:
        return items[int(ref) - 1]
    low = ref.lower()
    if low.startswith("arxiv:"):
        low = low[6:]
    low = re.sub(r"v\d+$", "", low)
    for i in items:
        k = i["key"].lower()
        if k == ref.lower() or k.split(":", 1)[1] == low:
            return i
    return None


def remove(ref: str) -> dict | None:
    items = load()
    entry = find(ref, items)
    if entry is None:
        return None
    items = [i for i in items if i["key"] != entry["key"]]
    save(items)
    return entry


def search(query: str, limit: int = 10, items: list[dict] | None = None) -> list[dict]:
    """Simple keyword match over title, authors, abstract and tags."""
    items = load() if items is None else items
    terms = [t for t in re.findall(r"\w+", (query or "").lower()) if len(t) > 1]
    if not terms:
        return items[:limit]
    scored = []
    for i in items:
        hay_title = (i.get("title") or "").lower()
        hay = " ".join([hay_title, " ".join(i.get("authors") or []), i.get("abstract") or "",
                        " ".join(i.get("tags") or [])]).lower()
        score = sum(hay.count(t) + 3 * hay_title.count(t) for t in terms)
        if score:
            scored.append((score, i))
    scored.sort(key=lambda s: -s[0])
    return [i for _, i in scored[:limit]]


def from_local_pdf(path: Path, copy: bool = False) -> tuple[Paper, Path]:
    """A Paper for a PDF on disk. Uses the PDF's own title when pypdf is
    installed, otherwise the file name."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise LibraryError(f"no such file: {path}")
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    authors: list[str] = []
    try:
        from pypdf import PdfReader  # optional extra [rag]
        meta = PdfReader(str(path)).metadata or {}
        if getattr(meta, "title", None):
            title = str(meta.title).strip() or title
        if getattr(meta, "author", None):
            authors = [a.strip() for a in re.split(r";|,| and ", str(meta.author)) if a.strip()]
    except Exception:
        pass
    dest = path
    if copy:
        pdf_dir().mkdir(parents=True, exist_ok=True)
        # The content hash in the name means two different files called
        # paper.pdf never overwrite each other's copy.
        dest = pdf_dir() / f"{path.stem}-{file_digest(path)[:10]}{path.suffix}"
        if dest.resolve() != path and not dest.exists():
            shutil.copy2(path, dest)
    return Paper(title=title, authors=authors, source="local file"), dest


def read_import_file(path: Path) -> list[tuple[Paper, dict]]:
    """Papers from a JSON file: one object, a list, or a `kx library export --format json` dump."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LibraryError(f"could not read {path} ({exc.__class__.__name__}).")
    if isinstance(data, dict) and "papers" in data:
        data = data["papers"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise LibraryError(f"{path} must hold a paper object or a list of them.")
    out = []
    base = Path(path).expanduser().resolve().parent
    for d in data:
        if isinstance(d, dict) and (d.get("title") or d.get("arxiv_id") or d.get("doi")):
            raw_tags = d.get("tags") or []
            if isinstance(raw_tags, str):
                raw_tags = [t.strip() for t in raw_tags.split(",")]
            elif not isinstance(raw_tags, list):
                raw_tags = [raw_tags]
            extras = {"tags": [str(t) for t in raw_tags if str(t).strip()], "pdf_path": None}
            if d.get("pdf_path"):
                pdf = Path(str(d["pdf_path"])).expanduser()
                extras["pdf_path"] = pdf if pdf.is_absolute() else base / pdf
            out.append((Paper.from_dict(d), extras))
    return out
