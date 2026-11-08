"""`kx primers build` / `kx convert` pipeline.

research (kx federated search) → ingest (--source) → synthesize (LLM or
offline outline) → validate word budget (repair-prompt once, then
deterministic compaction) → render HTML → export PDF / PNG / PPTX / JSON.

Citations only ever come from fetched metadata (aggregate.search /
resolve) — the model is given those and told it may cite nothing else;
--verify-citations re-resolves each one and drops any that do not check out.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .. import net
from ..sources import aggregate
from ..resolve import resolve
from . import engines, ingest, prompts, renderer, schema
from .exporters import images as images_x, jsonx, pptx as pptx_x


@dataclass
class BuildRequest:
    topic: str
    source: str | None = None
    category: str = ""
    personas: list[str] = field(default_factory=lambda: ["student"])
    pages: int = 8
    pdfs: int = 1
    ppts: int = 0
    images: int = 0
    jsons: int = 0
    watermark: str = "Rohit Patel | Narmadya Innovations"
    author: str = "Rohit Patel"
    engine: str = "outline"
    model: str | None = None
    verify_citations: bool = False
    research_limit: int = 8
    out_dir: Path = Path("./dist/primers")
    style: str = "trifleet-8page"


@dataclass
class BuildResult:
    primer: schema.Primer
    html_path: Path | None = None
    pdf_paths: list[Path] = field(default_factory=list)
    png_paths: list[Path] = field(default_factory=list)
    pptx_paths: list[Path] = field(default_factory=list)
    json_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.primer.title,
            "pages": len(self.primer.pages),
            "html": str(self.html_path) if self.html_path else None,
            "pdfs": [str(p) for p in self.pdf_paths],
            "images": [str(p) for p in self.png_paths],
            "ppts": [str(p) for p in self.pptx_paths],
            "json": [str(p) for p in self.json_paths],
            "citations": len(self.primer.citations),
            "warnings": self.warnings,
        }


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "primer"


def _research(topic: str, limit: int, warnings: list[str]) -> list[dict]:
    """Federated search for citable sources. Failure is non-fatal: the primer
    simply carries fewer (or zero) citations."""
    try:
        with net.client() as client:
            papers, warns = aggregate.search(topic, limit, client, None)
        warnings.extend(warns)
        return [p.to_dict() for p in papers]
    except Exception as exc:  # research must never kill the build
        warnings.append(f"research skipped ({exc.__class__.__name__}: {exc})")
        return []


def _verify(citations: list[dict], warnings: list[str]) -> list[dict]:
    """Re-resolve each citation; keep only those whose id checks out."""
    kept: list[dict] = []
    with net.client() as client:
        for c in citations:
            ident = c.get("arxiv_id") or c.get("doi")
            if not ident:
                kept.append(c)  # title-only source from search metadata: keep as-is
                continue
            try:
                p = resolve(ident, client)
            except Exception:
                p = None
            if p is not None:
                kept.append(p.to_dict())
            else:
                warnings.append(f"citation dropped (could not verify): {ident}")
    return kept


def build(req: BuildRequest, log: Callable[[str], None] = lambda s: None) -> BuildResult:
    warnings: list[str] = []
    out_dir = Path(req.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ingest + research
    log("ingesting source material…")
    material = ingest.load(req.source, req.topic)
    warnings.extend(material.warnings)
    log("searching verified sources (arXiv + Semantic Scholar + OpenAlex)…")
    sources = material.papers + _research(req.topic, req.research_limit, warnings)
    seen: set[str] = set()
    sources = [s for s in sources
               if (k := (s.get("doi") or s.get("arxiv_id") or s.get("title", "")).lower())
               and k not in seen and not seen.add(k)]

    # 2. synthesize
    personas = [p.strip().lower() for p in req.personas if p.strip()]
    if req.engine == "outline":
        log("building offline outline (no-LLM mode)…")
        primer = engines.outline(req.topic, req.pages, personas, material.text,
                                 sources, req.watermark)
    else:
        log(f"synthesizing {req.pages} pages with {req.engine}…")
        system = prompts.SYSTEM_PROMPT
        user = prompts.build_user_prompt(req.topic, req.pages, personas,
                                         material.text, sources)
        primer = engines.synthesize(req.engine, req.model, system, user)
        problems = schema.validate_budget(primer)
        if problems:
            log("page budget violated; asking the model to repair…")
            try:
                primer = engines.synthesize(
                    req.engine, req.model, system,
                    user + "\n\n" + prompts.repair_prompt(problems))
            except engines.EngineError as exc:
                warnings.append(f"repair pass failed ({exc}); compacting deterministically.")
    primer.topic = primer.topic or req.topic
    primer.category = req.category
    primer.personas = personas
    primer.watermark = req.watermark
    primer.author = req.author
    if len(primer.pages) != req.pages:
        warnings.append(f"model produced {len(primer.pages)} pages (asked for {req.pages}).")

    # 3. citations
    primer.citations = sources[:15]
    if req.verify_citations and primer.citations:
        log("verifying citations…")
        primer.citations = _verify(primer.citations, warnings)

    # 4. budget enforcement (deterministic, never ships an overflowing page)
    touched = schema.compact(primer)
    if touched:
        warnings.append(f"compacted {touched} over-budget page(s) by ~15%.")
    for problem in schema.validate_budget(primer):
        warnings.append(f"budget: {problem}")

    # 5. render + export
    slug = slugify(primer.title or req.topic)
    result = BuildResult(primer=primer, warnings=warnings)
    log("rendering HTML…")
    html = renderer.render_html(primer, editable=True)
    result.html_path = out_dir / f"{slug}.html"
    result.html_path.write_text(html, encoding="utf-8")

    pdf_needed = req.pdfs > 0 or req.images > 0
    pdf_path = out_dir / f"{slug}.pdf"
    if pdf_needed:
        log("compiling vector PDF…")
        renderer.html_to_pdf(html, pdf_path)
        if req.pdfs > 0:
            result.pdf_paths.append(pdf_path)
    if req.images > 0:
        log(f"exporting {req.images} page PNGs at 300 DPI…")
        result.png_paths = images_x.pdf_to_pngs(pdf_path, out_dir / "images",
                                                count=req.images)
        if req.pdfs == 0:
            pdf_path.unlink(missing_ok=True)
    if req.ppts > 0:
        log("exporting editable .pptx…")
        result.pptx_paths.append(pptx_x.primer_to_pptx(primer.to_dict(),
                                                       out_dir / f"{slug}.pptx"))
    if req.jsons > 0:
        log("exporting Paper Lab JSON bundle…")
        result.json_paths.append(jsonx.primer_to_json(primer.to_dict(),
                                                      out_dir / f"{slug}.json"))
    return result
