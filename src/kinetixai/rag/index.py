"""Build and query the local RAG index (data_dir/rag/index.json)."""
from __future__ import annotations

import base64
import hashlib
import json
import math
from array import array
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .. import config, library, net
from . import ollama
from .bm25 import BM25
from .text import MissingExtra, TEXT_SUFFIXES, chunk, file_pages, tokens

VERSION = 1
EMBED_BATCH = 32
SYSTEM = (
    "You answer questions about research papers using only the numbered passages provided. "
    "Cite every claim with the passage number in square brackets, like [2]. "
    "If the passages do not contain the answer, say so plainly. Be concise."
)


class RagError(Exception):
    pass


def index_path() -> Path:
    return config.data_dir() / "rag" / "index.json"


def _pack(vec: list[float]) -> str:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return base64.b64encode(array("f", [x / n for x in vec]).tobytes()).decode("ascii")


def _unpack(s: str) -> array:
    a = array("f")
    a.frombytes(base64.b64decode(s))
    return a


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_index() -> dict | None:
    p = index_path()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _scan(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser()
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES | {".pdf"})
        elif p.is_file():
            files.append(p)
    return files


def build(extra_paths: list[str] | None = None, embeddings: bool | None = None,
          progress=None) -> dict:
    """(Re)build the index from the library (+ any remembered or new paths).

    embeddings: True = require Ollama, False = BM25 only, None = use Ollama if it is up.
    Returns a summary dict.
    """
    cfg = config.load()
    old = load_index() or {}
    paths = list(dict.fromkeys((old.get("paths") or []) + [str(Path(p).expanduser().resolve()) for p in extra_paths or []]))
    warnings: list[str] = []
    docs: list[dict] = []
    chunks: list[dict] = []

    def add_doc(doc: dict, pages: list[tuple[int | None, str]]) -> None:
        di = len(docs)
        docs.append(doc)
        for page, text in pages:
            for piece in chunk(text):
                chunks.append({"doc": di, "page": page, "text": piece})

    for e in library.load():
        head = ". ".join(x for x in [e.get("title") or "", e.get("abstract") or ""] if x)
        pages: list[tuple[int | None, str]] = [(None, head)] if head else []
        pdf = e.get("pdf_path")
        if pdf:
            try:
                pages += file_pages(Path(pdf))
            except MissingExtra as exc:
                warnings.append(str(exc))
            except Exception as exc:
                warnings.append(f"skipped PDF for {e['key']} ({exc.__class__.__name__})")
        add_doc({"key": e["key"], "title": e.get("title") or e["key"], "path": pdf, "kind": "library"}, pages)

    for f in _scan(paths):
        try:
            pages = file_pages(f)
        except MissingExtra as exc:
            warnings.append(str(exc))
            continue
        except Exception as exc:
            warnings.append(f"skipped {f.name} ({exc.__class__.__name__})")
            continue
        add_doc({"key": "file:" + f.name, "title": f.name, "path": str(f), "kind": "file"}, pages)

    warnings = list(dict.fromkeys(warnings))
    embed_model = None
    url = cfg["ollama_url"]
    if embeddings is not False and chunks:
        installed = ollama.models(url)
        want = cfg["embed_model"]
        if installed is None:
            if embeddings:
                raise RagError(f"Ollama is not running at {url}. Start it, or index without --embeddings.")
        elif not ollama.has_model(installed, want):
            msg = f"embedding model {want} not pulled; run: ollama pull {want}"
            if embeddings:
                raise RagError(msg)
            warnings.append(msg + " (using keyword search only)")
        else:
            embed_model = want
            cache = {c["h"]: c["v"] for c in old.get("chunks", []) if c.get("v") and old.get("embed_model") == want}
            todo = [c for c in chunks if _hash(c["text"]) not in cache]
            with httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0)) as cl:
                for start in range(0, len(todo), EMBED_BATCH):
                    batch = todo[start:start + EMBED_BATCH]
                    vecs = ollama.embed(url, want, [c["text"] for c in batch], cl)
                    for c, v in zip(batch, vecs):
                        cache[_hash(c["text"])] = _pack(v)
                    if progress:
                        progress(min(start + EMBED_BATCH, len(todo)), len(todo))
            for c in chunks:
                c["v"] = cache.get(_hash(c["text"]))

    for c in chunks:
        c["h"] = _hash(c["text"])
    data = {
        "version": VERSION,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embed_model": embed_model,
        "paths": paths,
        "docs": docs,
        "chunks": chunks,
    }
    config.write_text_atomic(index_path(), json.dumps(data, ensure_ascii=False))
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "embeddings": embed_model,
        "paths": paths,
        "index": str(index_path()),
        "warnings": warnings,
    }


def retrieve(question: str, k: int = 5, idx: dict | None = None, use_embeddings: bool = True) -> tuple[list[dict], str]:
    """Top-k passages and the retrieval mode used ('bm25' or 'hybrid')."""
    idx = idx or load_index()
    if not idx or not idx.get("chunks"):
        raise RagError("the index is empty. Add papers with `kx library add`, then run `kx rag index`.")
    chunks, docs = idx["chunks"], idx["docs"]
    bm = BM25([tokens(c["text"]) for c in chunks])
    kw = bm.scores(tokens(question))
    ranked_kw = sorted(range(len(chunks)), key=lambda i: -kw[i])
    mode = "bm25"
    fused: dict[int, float] = {}
    for r, i in enumerate(ranked_kw[:50]):
        if kw[i] > 0:
            fused[i] = 1.0 / (60 + r)

    cfg = config.load()
    if use_embeddings and idx.get("embed_model") and any(c.get("v") for c in chunks):
        installed = ollama.models(cfg["ollama_url"])
        if installed is not None and ollama.has_model(installed, idx["embed_model"]):
            try:
                with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as cl:
                    q = ollama.embed(cfg["ollama_url"], idx["embed_model"], [question], cl)[0]
                n = math.sqrt(sum(x * x for x in q)) or 1.0
                q = [x / n for x in q]
                sims = []
                for i, c in enumerate(chunks):
                    if c.get("v"):
                        v = _unpack(c["v"])
                        sims.append((sum(a * b for a, b in zip(q, v)), i))
                sims.sort(reverse=True)
                for r, (_, i) in enumerate(sims[:50]):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r)
                mode = "hybrid"
            except net.SourceError:
                pass

    best = sorted(fused, key=lambda i: -fused[i])[:k]
    out = []
    for n, i in enumerate(best, 1):
        c, d = chunks[i], docs[chunks[i]["doc"]]
        out.append({"n": n, "text": c["text"], "title": d["title"], "key": d["key"],
                    "page": c.get("page"), "path": d.get("path"), "score": round(fused[i], 5)})
    return out, mode


def ask(question: str, k: int = 5, llm: bool = True, model: str | None = None) -> dict:
    """Answer from local passages. Uses Ollama for the answer when it is running;
    otherwise returns the passages themselves."""
    cfg = config.load()
    url = cfg["ollama_url"]
    passages, mode = retrieve(question, k)
    result = {
        "question": question,
        "answer": None,
        "passages": passages,
        "retrieval": mode,
        "model": None,
        "ollama_url": url,
        "local": True,
        "note": None,
    }

    def done(r: dict) -> dict:
        # Only true when no text went to a non-local Ollama (query embedding or answer).
        r["local"] = ollama.is_local(url) or (r["retrieval"] == "bm25" and r["model"] is None)
        return r

    if not passages:
        result["note"] = "No passage in your library matches that question."
        return done(result)
    if not llm:
        return done(result)
    installed = ollama.models(url)
    model = model or cfg["ollama_model"]
    if installed is None:
        result["note"] = f"Ollama is not running at {url}, so here are the best passages. Start Ollama for written answers."
        return done(result)
    if not ollama.has_model(installed, model):
        result["note"] = f"Model {model} is not pulled. Run: ollama pull {model}. Showing passages instead."
        return done(result)
    context = "\n\n".join(
        f"[{p['n']}] ({p['title']}{', p. ' + str(p['page']) if p.get('page') else ''})\n{p['text']}" for p in passages
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
    ]
    try:
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=5.0)) as cl:
            result["answer"] = ollama.chat(url, model, messages, cl) or None
        result["model"] = model
    except net.SourceError as exc:
        result["note"] = f"{exc} Showing passages instead."
    return done(result)
