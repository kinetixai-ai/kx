"""`kx primers autobuild --manifest primers_catalog.yaml --parallel N`."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .builder import BuildRequest, BuildResult, build, slugify


class ManifestError(ValueError):
    pass


DEFAULTS_KEYS = ("pages", "pdfs", "ppts", "images", "jsons", "watermark", "author",
                 "engine", "model", "api_base", "num_ctx", "mode", "min_sections",
                 "relevance", "verify_citations", "personas", "out_dir",
                 "research_limit", "copy_to", "category", "max_tokens")
# The CLI flag is --json-bundles; accepting only "jsons" meant a manifest
# written from the CLI's own vocabulary was silently ignored.
ALIASES = {"json_bundles": "jsons", "json": "jsons", "pdf": "pdfs",
           "ppt": "ppts", "image": "images", "persona": "personas"}
TRACK_ONLY = {"topic", "source", "category", "personas", "out_dir", "slug"}


def load_manifest(path: Path) -> list[BuildRequest]:
    try:
        import yaml
    except ImportError:
        raise ManifestError("autobuild needs PyYAML: pip install 'kinetixai[primers]'")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("tracks"), list):
        raise ManifestError("manifest must be a mapping with a 'tracks' list.")
    defaults = dict(data.get("defaults") or {})
    notes: list[str] = []

    def canon(block: dict, where: str) -> dict:
        fixed = {}
        for key, value in block.items():
            name = ALIASES.get(key, key)
            if name not in DEFAULTS_KEYS and name not in TRACK_ONLY:
                notes.append(f"{where}: ignoring unknown key {key!r}")
                continue
            fixed[name] = value
        return fixed

    defaults = canon(defaults, "defaults")
    base_out = Path(defaults.get("out_dir", "./dist/primers"))
    reqs: list[BuildRequest] = []
    for i, tr in enumerate(data["tracks"], 1):
        if not isinstance(tr, dict) or not tr.get("topic"):
            raise ManifestError(f"track #{i} needs at least a 'topic'.")
        tr = canon(tr, f"track {i} ({tr.get('topic', 'unnamed')})") | {
            "topic": tr.get("topic"), "source": tr.get("source")}
        merged = {k: tr.get(k, defaults.get(k)) for k in DEFAULTS_KEYS}
        personas = merged.get("personas") or ["student"]
        if isinstance(personas, str):
            personas = [p.strip() for p in personas.split(",") if p.strip()]
        # The folder follows the slug too, so dist/primers/<slug>/<slug>.pdf
        # lines up with the published /primers/<slug>/ page.
        slug = slugify(str(tr["slug"])) if tr.get("slug") else slugify(tr["topic"])
        out = Path(tr["out_dir"]) if tr.get("out_dir") else base_out / slug
        reqs.append(BuildRequest(
            topic=str(tr["topic"]),
            source=tr.get("source"),
            category=str(tr.get("category") or ""),
            slug=str(tr.get("slug") or ""),
            personas=personas,
            pages=int(merged.get("pages") or 8),
            pdfs=int(merged.get("pdfs") or 1),
            ppts=int(merged.get("ppts") or 0),
            images=int(merged.get("images") or 0),
            jsons=int(merged.get("jsons") or 0),
            watermark=str(merged.get("watermark") or "KinetixAI.AI | NARMADYA INNOVATIONS"),
            author=str(merged.get("author") or "KinetixAI.AI"),
            engine=str(merged.get("engine") or "outline"),
            model=merged.get("model"),
            api_base=merged.get("api_base") or None,
            num_ctx=int(merged["num_ctx"]) if merged.get("num_ctx") else None,
            mode=str(merged.get("mode") or "iterative"),
            min_sections=int(merged.get("min_sections") or 5),
            relevance=float(merged["relevance"]) if merged.get("relevance") is not None else 0.18,
            verify_citations=bool(merged.get("verify_citations")),
            research_limit=int(merged.get("research_limit") or 8),
            max_tokens=int(merged.get("max_tokens") or 0),
            copy_to=Path(merged["copy_to"]).expanduser() if merged.get("copy_to") else None,
            out_dir=out,
        ))
    for note in notes:
        print(f"warning: {note}", file=__import__("sys").stderr)
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
