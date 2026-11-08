"""Editable .pptx export (python-pptx) with speaker notes.

One slide per Primer page: title, body content distilled from the page's
blocks, and the page's prose + takeaway as speaker notes. 16:9.
"""
from __future__ import annotations

from pathlib import Path


class PptxExportError(RuntimeError):
    pass


ACCENTS = [(0xD2, 0x23, 0x2A), (0x1D, 0x4E, 0xD8), (0x04, 0x78, 0x57), (0xB4, 0x53, 0x09)]


def _bullets(blocks: list[dict]) -> list[tuple[str, int]]:
    """(text, indent-level) bullet list distilled from page blocks."""
    out: list[tuple[str, int]] = []
    for b in blocks:
        t = b.get("type")
        if t == "section":
            out.append((b.get("title", ""), 0))
        elif t == "bento":
            for c in b.get("cards", []):
                out.append((f"{c.get('title')}: {c.get('body', '')}"[:160], 1))
        elif t == "flow":
            out.append((" → ".join(b.get("steps", [])), 1))
        elif t == "callout":
            out.append((f"[{(b.get('style') or 'note').upper()}] {b.get('title')}: "
                        f"{b.get('body', '')}"[:180], 1))
        elif t == "kv":
            for k, v in b.get("pairs", [])[:4]:
                out.append((f"{k} — {v}"[:140], 1))
        elif t == "table" and b.get("title"):
            out.append((f"Table: {b['title']}", 1))
        elif t == "code" and b.get("title"):
            out.append((f"Code: {b['title']}", 1))
    return out[:12]


def primer_to_pptx(primer_dict: dict, out_path: Path) -> Path:
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Emu, Pt
    except ImportError:
        raise PptxExportError(".pptx export needs python-pptx: pip install 'kinetixai[primers]'")

    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 16:9
    blank = prs.slide_layouts[6]

    # title slide
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Emu(800000), Emu(2200000), Emu(10600000), Emu(2000000))
    tf = tb.text_frame
    tf.text = primer_dict.get("title", "Primer")
    tf.paragraphs[0].font.size = Pt(44)
    tf.paragraphs[0].font.bold = True
    p = tf.add_paragraph()
    p.text = primer_dict.get("subtitle") or primer_dict.get("topic") or ""
    p.font.size = Pt(20)
    p.font.color.rgb = RGBColor(0x5B, 0x63, 0x72)
    p2 = tf.add_paragraph()
    p2.text = primer_dict.get("watermark", "")
    p2.font.size = Pt(12)
    p2.font.color.rgb = RGBColor(*ACCENTS[0])

    for i, page in enumerate(primer_dict.get("pages", [])):
        s = prs.slides.add_slide(blank)
        bar = s.shapes.add_textbox(Emu(600000), Emu(350000), Emu(11000000), Emu(900000))
        tf = bar.text_frame
        tf.text = page.get("title", f"Page {i + 1}")
        tf.paragraphs[0].font.size = Pt(28)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = RGBColor(*ACCENTS[i % 4])

        body = s.shapes.add_textbox(Emu(600000), Emu(1300000), Emu(11000000), Emu(4600000))
        tfb = body.text_frame
        tfb.word_wrap = True
        first = True
        for text, level in _bullets(page.get("blocks", [])):
            para = tfb.paragraphs[0] if first else tfb.add_paragraph()
            first = False
            para.text = ("• " if level else "") + text
            para.level = level
            para.font.size = Pt(18 if level == 0 else 14)
            para.font.bold = level == 0

        if page.get("takeaway"):
            foot = s.shapes.add_textbox(Emu(600000), Emu(6100000), Emu(11000000), Emu(500000))
            ft = foot.text_frame
            ft.text = f"Key takeaway: {page['takeaway']}"
            ft.paragraphs[0].font.size = Pt(13)
            ft.paragraphs[0].font.italic = True

        notes = s.notes_slide.notes_text_frame
        prose = " ".join(b.get("body", "") for b in page.get("blocks", [])
                         if b.get("type") == "section")
        notes.text = (prose[:3500] + ("\n\nKey takeaway: " + page.get("takeaway", "")
                                      if page.get("takeaway") else ""))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return out_path
