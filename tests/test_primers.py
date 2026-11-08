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
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings, relevance=0.18: [])
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

    def fake_complete(engine, model, system, user, api_base=None):
        calls.append(user)
        if len(calls) == 1:
            return "<think>no answer yet"
        return '{"title": "T", "pages": [{"title": "p", "blocks": []}]}'

    monkeypatch.setattr(e, "complete", fake_complete)
    primer = e.synthesize("ollama", None, "sys", "user")
    assert primer.title == "T"
    assert len(calls) == 2
    assert "JSON object ALONE" in calls[1]


# ── density gate, blueprint, fitting, relevance, engines ──────────────────

from kinetixai.primers import blueprint, search as primer_search  # noqa: E402

DENSE_BLOCKS = [
    {"type": "section", "num": "1", "title": "One", "body": "word " * 45},
    {"type": "section", "num": "2", "title": "Two", "body": "word " * 45},
    {"type": "section", "num": "3", "title": "Three", "body": "word " * 45},
    {"type": "section", "num": "4", "title": "Four", "body": "word " * 45},
    {"type": "section", "num": "5", "title": "Five", "body": "word " * 45},
    {"type": "flow", "title": "Pipe", "steps": ["A", "B", "C"]},
    {"type": "table", "title": "T", "headers": ["a", "b"], "rows": [["1", "2"]]},
    {"type": "bento", "columns": 3, "cards": [{"title": "a", "body": "x"},
                                              {"title": "b", "body": "y"},
                                              {"title": "c", "body": "z"}]},
    {"type": "code", "title": "run", "lang": "bash", "code": "echo hi"},
    {"type": "cloud", "text": "Real impact!"},
]


def _dense_page() -> schema.Page:
    return schema.page_from_dict({"title": "Dense", "takeaway": "t", "blocks":
                                  [dict(b) for b in DENSE_BLOCKS]})


def test_density_passes_a_dense_page():
    d = schema.density(_dense_page())
    assert d.sections == 5
    assert not d.missing_kinds
    assert d.ok, d.reasons()


def test_density_names_what_a_sparse_page_lacks():
    sparse = schema.page_from_dict({"title": "Sparse", "blocks": [
        {"type": "section", "num": "1", "title": "Only", "body": "a few words"}]})
    d = schema.density(sparse)
    assert not d.ok
    reasons = " ".join(d.reasons())
    assert "numbered sections" in reasons and "words" in reasons
    assert "a table" in reasons and "comic cloud" in reasons


def test_density_follows_the_page_archetype():
    """A chart is required on the evaluation page and nowhere else."""
    page = _dense_page()
    assert schema.density(page, must_have=["table", "code", "cloud", "chart"]).missing_kinds
    assert not schema.density(page, must_have=["table", "code", "cloud"]).missing_kinds


def test_new_block_types_round_trip():
    page = schema.page_from_dict({"title": "p", "blocks": [
        {"type": "chart", "title": "c", "bars": [{"label": "a", "value": 2}]},
        {"type": "hierarchy", "title": "h", "levels": [{"label": "Grid", "detail": "all"}]},
        {"type": "steps", "title": "s", "items": ["one", "two"]},
        {"type": "cloud", "text": "hi"}]})
    assert [b["type"] for b in page.blocks] == ["chart", "hierarchy", "steps", "cloud"]


def test_block_aliases_are_accepted():
    page = schema.page_from_dict({"title": "p", "blocks": [
        {"type": "pipeline_flow", "steps": ["a"]},
        {"type": "comparison_table", "headers": ["h"], "rows": [["r"]]},
        {"type": "bento_cards", "cards": [{"title": "t", "body": "b"}]},
        {"type": "callout_cloud", "text": "x"}]})
    assert [b["type"] for b in page.blocks] == ["flow", "table", "bento", "cloud"]


def test_chart_drops_bars_without_numbers():
    page = schema.page_from_dict({"title": "p", "blocks": [
        {"type": "chart", "bars": [{"label": "ok", "value": "2.5"},
                                   {"label": "bad", "value": "n/a"}]}]})
    assert page.blocks[0]["bars"] == [{"label": "ok", "value": 2.5}]


def test_renumber_sections_is_sequential():
    page = schema.page_from_dict({"title": "p", "blocks": [
        {"type": "section", "num": "7", "title": "a", "body": "x"},
        {"type": "flow", "steps": ["s"]},
        {"type": "section", "num": "9", "title": "b", "body": "y"}]})
    primer = schema.Primer(title="t", pages=[page])
    schema.renumber_sections(primer)
    assert [b["num"] for b in page.sections()] == ["1", "2"]


# ── printable height ──────────────────────────────────────────────────────

def test_a_dense_page_fits_the_printable_area():
    assert schema.overflow_mm(_dense_page()) == 0


def test_every_archetype_fits_its_required_blocks():
    """The density quota has to be physically printable on A4."""
    cost = {"bento": 22, "table": 33, "code": 20, "cloud": 21,
            "flow": 23, "steps": 31, "chart": 39, "hierarchy": 38, "callout": 14}
    for spec in blueprint.plan(8):
        visuals = sum(cost.get(k, 0) for k in spec.must_have)
        per_section = (schema.PAGE_BUDGET_MM - visuals) / len(spec.sections)
        assert per_section >= 12, f"{spec.role}: only {per_section:.0f}mm per section"


def test_fit_page_keeps_required_blocks_and_reports_drops():
    page = _dense_page()
    page.blocks.insert(5, {"type": "chart", "title": "extra",
                           "bars": [{"label": "a", "value": 1}]})
    page.blocks.insert(6, {"type": "chart", "title": "extra 2",
                           "bars": [{"label": "b", "value": 2}]})
    for b in page.blocks:
        if b.get("type") == "section":
            b["body"] = "word " * 90
    assert schema.overflow_mm(page) > 0
    notes = schema.fit_page(page, must_have=["bento", "table", "code", "cloud", "flow"])
    assert notes
    assert schema.overflow_mm(page) == 0
    assert {"bento", "table", "code", "cloud", "flow"} <= page.kinds()


def test_fit_page_is_a_no_op_when_the_page_fits():
    page = _dense_page()
    before = list(page.blocks)
    assert schema.fit_page(page) == []
    assert page.blocks == before


# ── blueprint ─────────────────────────────────────────────────────────────

def test_plan_returns_the_full_arc_for_eight_pages():
    roles = [s.role for s in blueprint.plan(8)]
    assert roles == ["overview", "data", "stage1", "stage2", "guardrails",
                     "implementation", "evaluation", "takeaways"]


def test_plan_keeps_overview_and_takeaways_when_shortened():
    for n in (1, 2, 3, 4):
        roles = [s.role for s in blueprint.plan(n)]
        assert len(roles) == n
        assert "overview" in roles
        if n >= 2:
            assert "takeaways" in roles


def test_plan_extends_past_eight_pages():
    assert len(blueprint.plan(11)) == 11
    assert [s.role for s in blueprint.plan(11)][-1] == "takeaways"


def test_page_prompt_block_carries_sections_and_requirements():
    spec = blueprint.plan(8)[6]           # evaluation
    text = blueprint.page_prompt_block(spec, 7, 8, "Kernel Fusion")
    assert "PAGE 7 OF 8" in text and "Kernel Fusion" in text
    assert "Three Honest Columns" in text
    assert "chart" in text


# ── citation relevance ────────────────────────────────────────────────────

GPU_PAPERS = [
    {"title": "Automatic kernel fusion for CUDA deep learning workloads",
     "abstract": "We fuse pointwise CUDA kernels to cut HBM traffic on the GPU.", "citations": 40},
    {"title": "Plasma confinement in tokamak fusion reactors",
     "abstract": "Nuclear fusion energy and deuterium plasma behaviour in ITER.", "citations": 900},
    {"title": "Infrared and visible image fusion via deep networks",
     "abstract": "Image fusion of infrared and visible modalities.", "citations": 300},
    {"title": "Wearable facial sensor fusion for emotion recognition",
     "abstract": "Sensor fusion from wearable EEG and facial sensors.", "citations": 120},
    {"title": "Triton: a compiler for fused GPU kernels",
     "abstract": "A compiler producing fused CUDA kernels for SIMT hardware.", "citations": 500},
]


def test_relevance_filter_drops_other_disciplines():
    kept, notes = primer_search.filter_sources(GPU_PAPERS, "GPU & CUDA: Kernel Fusion")
    titles = " ".join(p["title"] for p in kept)
    assert "CUDA" in titles and "Triton" in titles
    assert "tokamak" not in titles.lower()
    assert "image fusion" not in titles.lower()
    assert "wearable" not in titles.lower()
    assert len(notes) == 3          # every drop is explained, never silent


def test_relevance_filter_can_be_disabled():
    kept, notes = primer_search.filter_sources(GPU_PAPERS, "GPU & CUDA: Kernel Fusion",
                                               min_coverage=0)
    assert len(kept) == len(GPU_PAPERS) and notes == []


def test_kernel_methods_paper_is_not_a_cuda_paper():
    papers = [{"title": "Reproducing kernel Hilbert spaces and the kernel trick",
               "abstract": "Support vector machines and kernel regression."}]
    kept, notes = primer_search.filter_sources(papers, "GPU & CUDA: Kernel Fusion")
    assert kept == [] and notes


def test_queries_pin_an_ambiguous_topic_to_its_field():
    qs = primer_search.queries_for("GPU & CUDA: Kernel Fusion")
    assert qs[0].startswith("GPU")
    assert any("GPU" in q and "Kernel Fusion" in q for q in qs)


# ── engine router ─────────────────────────────────────────────────────────

def test_engine_aliases_resolve():
    from kinetixai.primers.engines import canonical
    assert canonical("vllm") == "local-vllm"
    assert canonical("claude") == "anthropic"
    assert canonical("Ollama") == "ollama"
    assert canonical("none") == "outline"


def test_mcp_engine_without_a_host_explains_itself():
    import kinetixai.primers.engines as e
    e.set_sampler(None)
    with pytest.raises(EngineError, match="MCP server"):
        e.complete("mcp", None, "sys", "user")


def test_mcp_engine_uses_the_installed_sampler():
    import kinetixai.primers.engines as e
    e.set_sampler(lambda system, user: '{"ok": 1}')
    try:
        assert e.complete("mcp", None, "s", "u") == '{"ok": 1}'
        assert e.sampler_available()
    finally:
        e.set_sampler(None)


def test_synthesize_page_accepts_a_wrapped_page():
    import kinetixai.primers.engines as e
    monkey = '{"pages": [{"title": "p", "blocks": [{"type": "section", "title": "s", "body": "b"}]}]}'
    e.set_sampler(lambda system, user: monkey)
    try:
        page = e.synthesize_page("mcp", None, "sys", "user")
        assert page.title == "p"
    finally:
        e.set_sampler(None)


# ── iterative build ───────────────────────────────────────────────────────

def test_iterative_build_calls_the_model_once_per_page(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings, relevance=0.18: [])
    calls: list[str] = []

    def fake_complete(engine, model, system, user, api_base=None):
        calls.append(user)
        return json.dumps({"title": f"Page {len(calls)}", "takeaway": "t",
                           "blocks": [dict(x) for x in DENSE_BLOCKS]})

    monkeypatch.setattr(e, "complete", fake_complete)
    req = BuildRequest(topic="Topic", pages=3, pdfs=0, jsons=1, engine="ollama",
                       mode="iterative", out_dir=tmp_path / "d")
    res = b.build(req)
    assert len(calls) == 3                      # one call per page, not one for all
    assert len(res.primer.pages) == 3
    assert "PAGE 1 OF 3" in calls[0] and "PAGE 3 OF 3" in calls[2]


def test_iterative_build_expands_a_sparse_page(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings, relevance=0.18: [])
    calls: list[str] = []

    def fake_complete(engine, model, system, user, api_base=None):
        calls.append(user)
        if len(calls) == 1:                      # first reply: the sparse-page bug
            return json.dumps({"title": "Thin", "blocks": [
                {"type": "section", "num": "1", "title": "Only", "body": "one short line"}]})
        return json.dumps({"title": "Full", "takeaway": "t",
                           "blocks": [dict(x) for x in DENSE_BLOCKS]})

    monkeypatch.setattr(e, "complete", fake_complete)
    req = BuildRequest(topic="Topic", pages=1, pdfs=0, jsons=0, engine="ollama",
                       mode="iterative", out_dir=tmp_path / "d")
    res = b.build(req)
    assert len(calls) == 2                       # sparse page triggered an expansion
    assert "WHAT IS MISSING" in calls[1]
    assert res.primer.pages[0].title == "Full"
    assert schema.density(res.primer.pages[0]).ok


def test_build_rejects_an_unknown_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="mode must be"):
        build(BuildRequest(topic="t", mode="sideways", out_dir=tmp_path))
