"""Tests for kx primers: schema, budget validator, outline engine, exporters, manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kinetixai.primers import schema
from kinetixai.primers.builder import BuildRequest, build, slugify
from kinetixai.primers.catalog import ManifestError, load_manifest
from kinetixai.primers.engines import _parse, outline, EngineError
from kinetixai.primers.exporters.jsonx import primer_to_json
from kinetixai.primers.renderer import render_html

PAGE = {
    "title": "Test page", "subtitle": "sub", "takeaway": "one line",
    "blocks": [
        {"type": "section", "num": "1", "title": "Intro", "body": ("alpha beta. " * 250).strip()},
        {"type": "bento", "columns": 3, "cards": [{"title": "A", "body": "b " * 30}]},
        {"type": "flow", "title": "Pipe", "steps": ["A", "B", "C"]},
        {"type": "kv", "title": "Terms", "pairs": [["warp", "32 threads"]]},
        {"type": "callout", "style": "mnemonic", "title": "M", "body": "expansion"},
        {"type": "code", "title": "t", "lang": "bash", "code": "echo hi"},
    ],
}


def _primer() -> schema.Primer:
    return schema.primer_from_dict({"title": "T", "pages": [PAGE]})


def test_schema_roundtrip():
    p = _primer()
    assert p.pages[0].title == "Test page"
    assert p.to_dict()["pages"][0]["blocks"][0]["type"] == "section"


def test_schema_rejects_unknown_block():
    bad = {"title": "T", "pages": [{"title": "p", "blocks": [{"type": "hologram"}]}]}
    with pytest.raises(schema.PrimerSchemaError):
        schema.primer_from_dict(bad)


def test_budget_validator_flags_thin_and_fat():
    p = _primer()
    problems = schema.validate_budget(p)
    assert any("max" in x or "min" in x for x in problems)


def test_compact_trims_overflow():
    p = _primer()
    before = p.pages[0].word_count()
    assert before > schema.WORDS_MAX
    touched = schema.compact(p)
    assert touched == 1
    assert p.pages[0].word_count() <= before


def test_parse_strips_fences():
    assert _parse('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse('noise {"a": 1} noise') == {"a": 1}
    with pytest.raises(EngineError):
        _parse("no json here")


def test_outline_builds_requested_pages():
    p = outline("Topic X", 3, ["student"], "para one\n\npara two\n\npara three\n\npara four",
                [{"title": "W", "year": 2024, "arxiv_id": "2401.00001"}], "WM")
    assert len(p.pages) == 3
    assert p.watermark == "WM"
    assert "#" not in p.pages[0].blocks[0]["body"]


def test_render_html_contains_spec_laws():
    html = render_html(_primer())
    assert "width: 210mm; height: 297mm" in html
    assert "page-break-after: always" in html
    assert 'contenteditable="true"' in html
    assert "Key takeaway" in html


def test_render_html_escapes_source_content():
    evil = {"title": "T", "pages": [{"title": "<script>alert(1)</script>",
                                     "blocks": [{"type": "section", "title": "s",
                                                 "body": "<img onerror=x>"}]}]}
    html = render_html(schema.primer_from_dict(evil))
    assert "<script>alert(1)</script>" not in html
    assert "<img onerror" not in html


def test_json_bundle_tabs(tmp_path: Path):
    out = primer_to_json(_primer().to_dict(), tmp_path / "b.json")
    data = json.loads(out.read_text())
    assert data["schema"] == "kinetixai.primer/1"
    for tab in ("visualization", "architecture", "mnemonic", "cheatsheet", "flashcards"):
        assert tab in data["tabs"]
    assert data["tabs"]["mnemonic"][0]["mnemonic"] == "M"
    assert data["tabs"]["flashcards"]
    assert data["rag_chunks"]


def test_build_offline_outputs(tmp_path: Path, monkeypatch):
    # research is network-bound: stub it out so the test is offline
    import kinetixai.primers.builder as b
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings: [])
    src = tmp_path / "notes.md"
    src.write_text("alpha one\n\nbeta two\n\ngamma three\n\ndelta four", encoding="utf-8")
    req = BuildRequest(topic="Offline Topic", source=str(src), pages=2, pdfs=1,
                       ppts=1, images=1, jsons=1, engine="outline",
                       out_dir=tmp_path / "dist")
    res = build(req)
    assert res.html_path and res.html_path.is_file()
    assert len(res.pdf_paths) == 1 and res.pdf_paths[0].stat().st_size > 1000
    assert len(res.png_paths) == 1
    assert len(res.pptx_paths) == 1
    assert len(res.json_paths) == 1


def test_slugify():
    assert slugify("GPU & CUDA: Kernel Fusion!") == "gpu-cuda-kernel-fusion"


def test_manifest_roundtrip(tmp_path: Path):
    m = tmp_path / "cat.yaml"
    m.write_text(
        "defaults:\n  pages: 4\n  engine: outline\n"
        "tracks:\n  - topic: One\n    category: gpu\n  - topic: Two\n    pages: 6\n",
        encoding="utf-8")
    reqs = load_manifest(m)
    assert len(reqs) == 2
    assert reqs[0].pages == 4 and reqs[1].pages == 6
    assert reqs[0].engine == "outline"


def test_manifest_rejects_bad_shape(tmp_path: Path):
    m = tmp_path / "bad.yaml"
    m.write_text("just: a mapping\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_manifest(m)


# ── reasoning-model / malformed-reply handling ────────────────────────────

def test_parse_strips_think_blocks():
    assert _parse('<think>I should use {curly} braces</think>\n{"a": 1}') == {"a": 1}
    assert _parse('<thinking>plan</thinking>```json\n{"a": 2}\n```') == {"a": 2}


def test_parse_drops_unclosed_think_block():
    # truncated reasoning model: opening tag, no close, no answer
    with pytest.raises(EngineError, match="no JSON object|empty response"):
        _parse("<think>still reasoning about {this}")


def test_parse_ignores_braces_inside_strings():
    assert _parse('{"body": "a } brace and { another"}')["body"] == "a } brace and { another"


def test_parse_takes_first_complete_object_after_prose():
    assert _parse('Sure! Here it is:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_parse_empty_response_names_the_cause():
    with pytest.raises(EngineError, match="empty response"):
        _parse("   ")


def test_message_text_prefers_content_then_reasoning():
    from kinetixai.primers.engines import _message_text
    assert _message_text({"choices": [{"message": {"content": "x"}}]}, "t") == "x"
    assert _message_text({"choices": [{"message": {"reasoning_content": "y"}}]}, "t") == "y"


def test_message_text_explains_truncation():
    from kinetixai.primers.engines import _message_text
    payload = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    with pytest.raises(EngineError, match="token limit"):
        _message_text(payload, "t")


def test_synthesize_retries_once_then_succeeds(monkeypatch):
    import kinetixai.primers.engines as e
    calls = []

    def fake_complete(engine, model, system, user):
        calls.append(user)
        if len(calls) == 1:
            return "<think>no answer yet"
        return '{"title": "T", "pages": [{"title": "p", "blocks": []}]}'

    monkeypatch.setattr(e, "complete", fake_complete)
    primer = e.synthesize("ollama", None, "sys", "user")
    assert primer.title == "T"
    assert len(calls) == 2
    assert "JSON object ALONE" in calls[1]
