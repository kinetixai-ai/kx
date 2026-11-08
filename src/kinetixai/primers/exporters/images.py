"""High-DPI PNG export: rasterize the compiled PDF's pages at 300 DPI.

Uses pypdfium2 (small wheel, no system deps) so --images works everywhere
the PDF works — LinkedIn/X cards, lecture slides, poster panels.
"""
from __future__ import annotations

from pathlib import Path


class ImageExportError(RuntimeError):
    pass


def pdf_to_pngs(pdf_path: Path, out_dir: Path, count: int | None = None,
                dpi: int = 300) -> list[Path]:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImageExportError("PNG export needs pypdfium2: pip install 'kinetixai[primers]'")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        n = len(doc) if count is None else min(count, len(doc))
        scale = dpi / 72.0
        stem = Path(pdf_path).stem
        written: list[Path] = []
        for i in range(n):
            page = doc[i]
            bmp = page.render(scale=scale)
            img = bmp.to_pil()
            dest = out_dir / f"{stem}_page{i + 1:02d}_{dpi}dpi.png"
            img.save(dest, format="PNG", dpi=(dpi, dpi))
            written.append(dest)
        return written
    finally:
        doc.close()
