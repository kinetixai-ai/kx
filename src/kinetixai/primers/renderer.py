"""Deterministic HTML5 + CSS3 print engine for Primers.

Jinja2 renders ``templates/trifleet_a4.html.j2`` into a self-contained,
editable HTML document; WeasyPrint (default) or Playwright/headless
Chromium (``KX_PDF_ENGINE=playwright``) compiles it into a vector A4 PDF.
Overflow is prevented upstream: schema.validate_budget + schema.compact
keep every <section class="page"> inside 297mm before rendering.
"""
from __future__ import annotations

import os
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_TEMPLATE = "trifleet_a4.html.j2"


class RenderError(RuntimeError):
    pass


def render_html(primer, editable: bool = True, template: str = DEFAULT_TEMPLATE) -> str:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        raise RenderError("rendering needs jinja2: pip install 'kinetixai[primers]'")
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                      autoescape=select_autoescape(["html", "j2"]))
    data = primer.to_dict() if hasattr(primer, "to_dict") else primer
    # attribute access in the template
    class NS(dict):
        __getattr__ = dict.get
    def ns(obj):
        if isinstance(obj, dict):
            return NS({k: ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [ns(v) for v in obj]
        return obj
    return env.get_template(template).render(primer=ns(data), editable=editable)


def html_to_pdf(html: str, out_pdf: Path) -> Path:
    engine = os.environ.get("KX_PDF_ENGINE", "weasyprint").lower()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if engine == "playwright":
        return _pdf_playwright(html, out_pdf)
    return _pdf_weasyprint(html, out_pdf)


def _pdf_weasyprint(html: str, out_pdf: Path) -> Path:
    try:
        from weasyprint import HTML
    except ImportError:
        raise RenderError(
            "PDF export needs WeasyPrint (pip install 'kinetixai[primers]') or "
            "Playwright (set KX_PDF_ENGINE=playwright after `playwright install chromium`).")
    HTML(string=html).write_pdf(str(out_pdf))
    return out_pdf


def _pdf_playwright(html: str, out_pdf: Path) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RenderError("KX_PDF_ENGINE=playwright needs: pip install playwright && playwright install chromium")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.set_content(html, wait_until="networkidle")
        # bounding-box overflow check per page section (spec: compact upstream;
        # here we only report, we never silently clip)
        overflow = page.evaluate(
            """() => [...document.querySelectorAll('section.page')]
                 .map((s, i) => ({i: i + 1, over: s.scrollHeight - s.clientHeight}))
                 .filter(x => x.over > 2)""")
        page.pdf(path=str(out_pdf), width="210mm", height="297mm",
                 print_background=True, margin={"top": "0", "bottom": "0",
                                                "left": "0", "right": "0"})
        browser.close()
    if overflow:
        pages = ", ".join(str(o["i"]) for o in overflow)
        raise RenderError(f"page(s) {pages} overflow 297mm after rendering; "
                          "re-run with fewer words or more pages.")
    return out_pdf
