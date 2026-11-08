"""`kx primers autobuild --manifest primers_catalog.yaml --parallel N`."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .builder import BuildRequest, BuildResult, build, slugify


class ManifestError(ValueError):
    pass


DEFAULTS_KEYS = ("pages", "pdfs", "ppts", "images", "jsons", "watermark", "author",
                 "engine", "model", "verify_citations", "personas", "out_dir",
                 "research_limit")


def load_manifest(path: Path) -> list[BuildRequest]:
    try:
        import yaml
    except ImportError:
        raise ManifestError("autobuild needs PyYAML: pip install 'kinetixai[primers]'")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list):
        raise ManifestError("manifest must be a mapping with a 'tracks' list.")
    defaults = data.get("defaults") or {}
    base_out = Path(defaults.get("out_dir", "./dist/primers"))
    reqs: list[BuildRequest] = []
    for i, tr in enumerate(data["tracks"], 1):
        if not isinstance(tr, dict) or not tr.get("topic"):
            raise ManifestError(f"track #{i} needs at least a 'topic'.")
        merged = {k: tr.get(k, defaults.get(k)) for k in DEFAULTS_KEYS}
        personas = merged.get("personas") or ["student"]
        if isinstance(personas, str):
            personas = [p.strip() for p in personas.split(",") if p.strip()]
        out = Path(tr["out_dir"]) if tr.get("out_dir") else base_out / slugify(tr["topic"])
        reqs.append(BuildRequest(
            topic=str(tr["topic"]),
            source=tr.get("source"),
            category=str(tr.get("category") or ""),
            personas=personas,
            pages=int(merged.get("pages") or 8),
            pdfs=int(merged.get("pdfs") or 1),
            ppts=int(merged.get("ppts") or 0),
            images=int(merged.get("images") or 0),
            jsons=int(merged.get("jsons") or 0),
            watermark=str(merged.get("watermark") or "Rohit Patel | Narmadya Innovations"),
            author=str(merged.get("author") or "Rohit Patel"),
            engine=str(merged.get("engine") or "outline"),
            model=merged.get("model"),
            verify_citations=bool(merged.get("verify_citations")),
            research_limit=int(merged.get("research_limit") or 8),
            out_dir=out,
        ))
    return reqs


def autobuild(manifest: Path, parallel: int = 2,
              log: Callable[[str], None] = lambda s: None) -> list[dict]:
    reqs = load_manifest(manifest)
    results: list[dict] = []

    def one(req: BuildRequest) -> dict:
        try:
            res: BuildResult = build(req)
            return {"topic": req.topic, "ok": True, **res.to_dict()}
        except Exception as exc:
            return {"topic": req.topic, "ok": False, "error": f"{exc.__class__.__name__}: {exc}"}

    parallel = max(1, min(parallel, 8))
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(one, r): r for r in reqs}
        for fut in as_completed(futures):
            r = fut.result()
            log(("✓ " if r["ok"] else "✗ ") + r["topic"]
                + ("" if r["ok"] else f" — {r['error']}"))
            results.append(r)
    order = {req.topic: i for i, req in enumerate(reqs)}
    results.sort(key=lambda r: order.get(r["topic"], 999))
    return results
