"""`kx primers build` / `kx convert` pipeline.

research → ingest → synthesize → density gate → budget → render → export.

Synthesis runs page by page by default (``mode="iterative"``): each page
gets its own LLM call carrying only that page's archetype and density
quota, and a page that comes back sparse is sent through an expansion
pass before it is allowed near the renderer. That is what lets a 7B
local model fill eight dense pages; single-call mode remains available
for large models (``mode="single"``).

Citations only ever come from fetched metadata, filtered for domain
relevance (``search.filter_sources``) so an ambiguous word like "fusion"
cannot drag in another discipline's papers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from .. import net
from ..resolve import resolve
from ..sources import aggregate
from . import blueprint, engines, ingest, prompts, providers, renderer, schema, search
from .exporters import images as images_x, jsonx, pptx as pptx_x

MODES = ("iterative", "single")


@dataclass
class BuildRequest:
    topic: str
    source: str | None = None
    category: str = ""
    slug: str = ""                  # "" = derive from the title; set it to
                                    # match a published URL exactly
    personas: list[str] = field(default_factory=lambda: ["student"])
    pages: int = 8
    pdfs: int = 1
    ppts: int = 0
    images: int = 0
    jsons: int = 0
    watermark: str = "Narmadya Innovations"
    author: str = "Rohit Patel"
    engine: str = "auto"
    model: str | None = None
    api_base: str | None = None
    num_ctx: int | None = None
    mode: str = "iterative"
    min_sections: int = schema.SECTIONS_MIN
    verify_citations: bool = False
    relevance: float = 0.18
    max_tokens: int = 0             # 0 = the engine's own per-page default.
                                    # A hosted endpoint reserves this against
                                    # the per-minute budget on admission, so a
                                    # cap above that budget is never admitted.
    max_files: int = 0              # 0 = the repo reader's own default
    research_limit: int = 12
    out_dir: Path = Path("./dist/primers")
    copy_to: Path | None = None        # a second home for the finished files
    style: str = "trifleet-8page"


@dataclass
class BuildResult:
    primer: schema.Primer
    html_path: Path | None = None
    pdf_paths: list[Path] = field(default_factory=list)
    png_paths: list[Path] = field(default_factory=list)
    pptx_paths: list[Path] = field(default_factory=list)
    json_paths: list[Path] = field(default_factory=list)
    copied: list[Path] = field(default_factory=list)
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
            "copied": [str(p) for p in self.copied],
            "warnings": self.warnings,
        }

    def files(self) -> list[Path]:
        """Everything this build produced, in the order it is reported."""
        out = list(self.pdf_paths) + list(self.png_paths) + list(self.pptx_paths)
        out += list(self.json_paths)
        if self.html_path:
            out.append(self.html_path)
        return out


# Which import each output needs, and the extra that provides it.
_RENDER_DEPS = [("jinja2", "the HTML renderer")]
_PDF_DEPS = [("weasyprint", "PDF export")]
_IMAGE_DEPS = [("pypdfium2", "PNG export"), ("PIL", "PNG export")]
_PPTX_DEPS = [("pptx", ".pptx export")]


def preflight(req: "BuildRequest") -> None:
    """Check what the requested outputs need BEFORE any model is called.

    Finding out after eight pages of local inference that jinja2 is missing
    throws away everything the run just produced.
    """
    import importlib
    import os

    needed = list(_RENDER_DEPS)
    if req.pdfs > 0 or req.images > 0:
        if os.environ.get("KX_PDF_ENGINE", "").lower() == "playwright":
            needed.append(("playwright", "PDF export"))
        else:
            needed += _PDF_DEPS
    if req.images > 0:
        needed += _IMAGE_DEPS
    if req.ppts > 0:
        needed += _PPTX_DEPS

    # Import rather than find_spec: WeasyPrint installs cleanly on macOS and
    # then fails at import without Homebrew's pango, which find_spec cannot
    # see. A broken install has to be caught here, not after a model run.
    missing: list[tuple[str, str]] = []
    broken: list[tuple[str, str, str]] = []
    for mod, why in needed:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            if mod in str(exc) or "No module named" in str(exc):
                missing.append((mod, why))
            else:
                broken.append((mod, why, str(exc)[:160]))
        except Exception as exc:  # noqa: BLE001 — a system-library failure
            broken.append((mod, why, f"{exc.__class__.__name__}: {exc}"[:160]))
    if not missing and not broken:
        return

    lines: list[str] = []
    if missing:
        names = ", ".join(sorted({m for m, _ in missing}))
        uses = ", ".join(sorted({w for _, w in missing}))
        lines.append(f"{uses} needs {names}, which is not installed.")
        lines.append("  pip  : pip install 'kinetixai[primers]'")
        lines.append("  pipx : pipx inject kinetixai jinja2 pyyaml python-pptx "
                     "pypdf pypdfium2 pillow weasyprint")
    for mod, why, detail in broken:
        lines.append(f"{why} needs {mod}, which is installed but will not import:")
        lines.append(f"  {detail}")
        if mod == "weasyprint":
            lines.append("  macOS: brew install pango gdk-pixbuf libffi")
            lines.append("  or:    pip install playwright && playwright install chromium")
            lines.append("         export KX_PDF_ENGINE=playwright")
    raise MissingDependency("\n".join(lines))


class MissingDependency(RuntimeError):
    pass


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "primer"


# ── research ──────────────────────────────────────────────────────────────

def _research(topic: str, limit: int, warnings: list[str],
              relevance: float = 0.18) -> list[dict]:
    """Federated search, then domain filtering. Never fatal: a build with no
    citations is honest, a build with wrong ones is not."""
    found: list[dict] = []
    try:
        with net.client() as client:
            for query in search.queries_for(topic):
                papers, warns = aggregate.search(query, limit, client, None)
                warnings.extend(w for w in warns if w not in warnings)
                found.extend(p.to_dict() for p in papers)
    except Exception as exc:  # research must never kill the build
        warnings.append(f"research skipped ({exc.__class__.__name__}: {exc})")
        return []

    seen: set[str] = set()
    unique: list[dict] = []
    for p in found:
        key = (p.get("doi") or p.get("arxiv_id") or p.get("title", "")).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)

    kept, notes = search.filter_sources(unique, topic, min_coverage=relevance)
    warnings.extend(notes)
    if unique and not kept:
        warnings.append("no source passed domain filtering; the primer will carry no "
                        "citations (lower --relevance to keep weaker matches)")
    return kept


def _verify(citations: list[dict], warnings: list[str]) -> list[dict]:
    """Re-resolve each citation; keep only those whose id checks out."""
    kept: list[dict] = []
    with net.client() as client:
        for c in citations:
            ident = c.get("arxiv_id") or c.get("doi")
            if not ident:
                kept.append(c)  # title-only source from search metadata
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


# ── synthesis ─────────────────────────────────────────────────────────────

def _synthesize_iterative(req: BuildRequest, sources: list[dict], source_text: str,
                          warnings: list[str], log: Callable[[str], None]) -> schema.Primer:
    specs = blueprint.plan(req.pages)
    total = len(specs)
    primer = schema.Primer(title=req.topic, topic=req.topic, watermark=req.watermark,
                           author=req.author, personas=req.personas)
    titles: list[str] = []
    import time
    started = time.monotonic()
    for i, spec in enumerate(specs, 1):
        page_started = time.monotonic()
        log(f"page {i}/{total} ({spec.role})…")
        user = prompts.build_page_prompt(req.topic, spec, i, total, req.personas,
                                         source_text, sources, titles)
        try:
            page = engines.synthesize_page(req.engine, req.model, prompts.PAGE_SYSTEM_PROMPT,
                                           user, api_base=req.api_base, num_ctx=req.num_ctx,
                                           max_tokens=req.max_tokens or None)
        except engines.EngineError as exc:
            warnings.append(f"page {i} ({spec.role}) failed: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 — keep the pages already written
            warnings.append(f"page {i} ({spec.role}) failed "
                            f"({exc.__class__.__name__}: {exc})")
            continue

        d = schema.density(page, req.min_sections, spec.must_have)
        if not d.ok or d.sections < req.min_sections:
            log(f"page {i} is sparse ({'; '.join(d.reasons())}); expanding…")
            page = _expand(req, page, d, spec, i, total, warnings)

        d = schema.density(page, req.min_sections, spec.must_have)
        if not d.ok or d.sections < req.min_sections:
            warnings.append(f"page {i} '{page.title}' still below the density standard: "
                            + "; ".join(d.reasons()))
        page.must_have = list(spec.must_have)   # remembered for the fitting pass
        primer.pages.append(page)
        titles.append(page.title)
        elapsed = time.monotonic() - page_started
        done = len(primer.pages)
        if done < total:
            eta = (time.monotonic() - started) / done * (total - done)
            log(f"  page {i} took {elapsed:.0f}s; about {eta / 60:.0f} min left")
        else:
            log(f"  page {i} took {elapsed:.0f}s; "
                f"{(time.monotonic() - started) / 60:.1f} min total")

    if not primer.pages:
        # The per-page reasons were the whole diagnosis and used to be
        # thrown away with the warnings; a generic message left the user
        # with nothing to act on.
        reasons = [w for w in warnings if ") failed" in w]
        detail = ("\n  " + "\n  ".join(reasons[:3])) if reasons else ""
        more = f"\n  (and {len(reasons) - 3} more)" if len(reasons) > 3 else ""
        raise engines.EngineError(
            f"every page failed to generate with engine '{req.engine}'."
            + detail + more
            + "\n\nTry `kx primers engines` to see what is ready, pass --engine "
              "explicitly, or use --engine outline for a draft without a model.")
    return primer


def _expand(req: BuildRequest, page: schema.Page, d: schema.Density,
            spec: blueprint.PageSpec, index: int, total: int,
            warnings: list[str]) -> schema.Page:
    """One completion pass over a sparse page. Keeps the better of the two."""
    payload = {"title": page.title, "subtitle": page.subtitle,
               "takeaway": page.takeaway, "blocks": page.blocks}
    user = prompts.expand_prompt(payload, d.reasons(), spec, req.topic, index, total)
    try:
        grown = engines.synthesize_page(req.engine, req.model, prompts.PAGE_SYSTEM_PROMPT,
                                        user, api_base=req.api_base, num_ctx=req.num_ctx)
    except Exception as exc:  # noqa: BLE001 — the original page is still usable
        warnings.append(f"page {index} expansion failed "
                        f"({exc.__class__.__name__}: {exc})")
        return page
    # The expansion is only an improvement if it actually added substance.
    if len(grown.blocks) >= len(page.blocks) and grown.word_count() >= page.word_count():
        return grown
    warnings.append(f"page {index} expansion returned less than the original; kept the original")
    return page


def _synthesize_single(req: BuildRequest, sources: list[dict], source_text: str,
                       warnings: list[str], log: Callable[[str], None]) -> schema.Primer:
    log(f"synthesizing {req.pages} pages in one call with {req.engine}…")
    system = prompts.SYSTEM_PROMPT
    user = prompts.build_user_prompt(req.topic, req.pages, req.personas,
                                     source_text, sources)
    primer = engines.synthesize(req.engine, req.model, system, user,
                                api_base=req.api_base, num_ctx=req.num_ctx)
    problems = schema.validate_budget(primer)
    if problems:
        log("page budget violated; asking the model to repair…")
        try:
            primer = engines.synthesize(req.engine, req.model, system,
                                        user + "\n\n" + prompts.repair_prompt(problems),
                                        api_base=req.api_base, num_ctx=req.num_ctx)
        except engines.EngineError as exc:
            warnings.append(f"repair pass failed ({exc}); compacting deterministically.")
    sparse = schema.validate_density(primer, req.min_sections)
    if sparse:
        warnings.append("single-call mode produced sparse pages; --mode iterative "
                        "generates one page per call and fills them")
        warnings.extend(sparse)
    return primer


# ── pipeline ──────────────────────────────────────────────────────────────

def build(req: BuildRequest, log: Callable[[str], None] = lambda s: None) -> BuildResult:
    warnings: list[str] = []
    if req.mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    preflight(req)          # before any model call, never after
    out_dir = Path(req.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. ingest + research
    log("ingesting source material…")
    material = ingest.load(req.source, req.topic, max_files=req.max_files)
    warnings.extend(material.warnings)
    if material.files:
        log(f"read {len(material.files)} files from {material.name}: "
            + ", ".join(material.files[:6])
            + (f" (+{len(material.files) - 6} more)" if len(material.files) > 6 else ""))
    log("searching verified sources (arXiv + Semantic Scholar + OpenAlex)…")
    sources = material.papers + _research(req.topic, req.research_limit, warnings,
                                          req.relevance)

    # 2. synthesize
    personas = [p.strip().lower() for p in req.personas if p.strip()]
    req.personas = personas
    if req.engine in ("auto", "", None):
        chosen, why = providers.pick()
        if chosen == "outline":
            warnings.append(why)
        log(f"engine: {chosen} ({why})")
        req.engine = chosen
    if req.engine != "outline" and not req.model:
        # Name the model in the log: "it used Ollama" is not reproducible.
        # Hosted endpoints were excluded here, so a pinned default went
        # straight to the API and 404'd on all eight pages. Resolve once,
        # for local and hosted alike, so every page reuses one verified name.
        p_ = providers.get(req.engine)
        if p_ is not None and p_.protocol not in ("mcp", "none"):
            found = providers.choose_model(p_, p_.default_model or None)
            if found:
                req.model = found
                log(f"model: {found}")
    if req.engine == "outline":
        log("building offline outline (no-LLM mode)…")
        primer = engines.outline(req.topic, req.pages, personas, material.text,
                                 sources, req.watermark)
    elif req.mode == "iterative":
        primer = _synthesize_iterative(req, sources, material.text, warnings, log)
    else:
        primer = _synthesize_single(req, sources, material.text, warnings, log)

    primer.topic = primer.topic or req.topic
    primer.category = req.category
    primer.personas = personas
    primer.watermark = req.watermark
    primer.author = req.author
    if len(primer.pages) != req.pages:
        warnings.append(f"produced {len(primer.pages)} pages (asked for {req.pages}).")

    # 3. citations
    primer.citations = sources[:15]
    if req.verify_citations and primer.citations:
        log("verifying citations…")
        primer.citations = _verify(primer.citations, warnings)

    # 4. deterministic clean-up: never ship an overflowing page or stale numbering
    schema.renumber_sections(primer)
    touched = schema.compact(primer)
    if touched:
        warnings.append(f"compacted {touched} over-budget page(s) by ~15%.")
    # The page box clips what does not fit, so fit it here rather than
    # letting a chart the model wrote disappear off the bottom unannounced.
    for i, page in enumerate(primer.pages, 1):
        over = schema.overflow_mm(page)
        if over > 0:
            if over >= 1:
                log(f"page {i} runs {over:.0f}mm past the printable area; fitting…")
            for note in schema.fit_page(page, getattr(page, "must_have", None)):
                warnings.append(f"page {i}: {note}")
    for problem in schema.validate_budget(primer):
        warnings.append(f"budget: {problem}")

    # 5. render + export
    # An explicit slug wins: the files are usually uploaded to a page whose
    # URL already exists, and the model is free to reword the title.
    slug = slugify(req.slug) if req.slug else slugify(primer.title or req.topic)
    result = BuildResult(primer=primer, warnings=warnings)
    log("rendering HTML…")
    try:
        html = renderer.render_html(primer, editable=True)
    except Exception:
        # The pages cost real time to write. Keep them so the run can be
        # rendered later rather than repeated.
        rescue = out_dir / f"{slug}.primer.json"
        try:
            jsonx.primer_to_json(primer.to_dict(), rescue)
        except Exception:  # noqa: BLE001 — the original error is the one to report
            pass
        else:
            log(f"rendering failed; the written pages are saved at {rescue}")
        raise
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

    # 6. optional second home (a notes folder, a shared drive, a repo)
    if req.copy_to:
        log(f"copying to {req.copy_to}…")
        result.copied, notes = _copy_out(result.files(), Path(req.copy_to), slug)
        warnings.extend(notes)
    return result


def _copy_out(files: list[Path], dest: Path, slug: str) -> tuple[list[Path], list[str]]:
    """Copy finished files into dest, keeping the images/ subfolder shape.

    Never raises: a build that succeeded should not be reported as failed
    because a shared drive was not mounted.
    """
    import shutil

    notes: list[str] = []
    dest = dest.expanduser()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return [], [f"could not copy to {dest} ({exc.strerror or exc}); "
                    "the files are still in the output folder"]
    written: list[Path] = []
    for src in files:
        try:
            target = dest / ("images" if src.parent.name == "images" else "") / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            written.append(target)
        except OSError as exc:
            notes.append(f"could not copy {src.name} to {dest} ({exc.strerror or exc})")
    return written, notes


# ── comparing models ─────────────────────────────────────────────────────

def compare_models(req: BuildRequest, models: list[str],
                   log: Callable[[str], None] = lambda s: None) -> list[dict]:
    """Build the same short sample with several models and measure each.

    Picking a model by reading its name is guesswork, and finding out it
    is too slow or too thin after an eight-page run is expensive. This
    builds one page per candidate so the choice is made on evidence.
    """
    import time

    out: list[dict] = []
    base_dir = Path(req.out_dir).expanduser()
    for model in models:
        slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-") or "model"
        trial = replace(req, model=model, pages=req.pages, ppts=0, jsons=0,
                        out_dir=base_dir / slug, copy_to=None)
        log(f"trying {model}…")
        started = time.monotonic()
        row: dict = {"model": model, "engine": req.engine}
        try:
            res = build(trial)
        except Exception as exc:  # noqa: BLE001 — one bad model must not stop the rest
            row.update(ok=False, seconds=round(time.monotonic() - started, 1),
                       error=f"{exc.__class__.__name__}: {exc}")
            log(f"  {model}: failed — {exc}")
            out.append(row)
            continue
        seconds = time.monotonic() - started
        pages = res.primer.pages
        densities = [schema.density(p, req.min_sections, getattr(p, "must_have", None))
                     for p in pages]
        row.update(
            ok=True,
            seconds=round(seconds, 1),
            seconds_per_page=round(seconds / max(1, len(pages)), 1),
            pages=len(pages),
            sections=round(sum(d.sections for d in densities) / max(1, len(densities)), 1),
            words=round(sum(d.words for d in densities) / max(1, len(densities))),
            met_quota=sum(1 for d in densities if d.ok),
            pdf=str(res.pdf_paths[0]) if res.pdf_paths else None,
            html=str(res.html_path) if res.html_path else None,
            warnings=len(res.warnings),
        )
        log(f"  {model}: {row['seconds']}s, {row['sections']} sections/page, "
            f"{row['words']} words/page, {row['met_quota']}/{len(pages)} met the quota")
        out.append(row)
    return out
