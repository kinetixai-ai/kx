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

    def fake_complete(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
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
    assert canonical("claude") == "anthropic"
    assert canonical("local-vllm") == "vllm"       # the old name still works
    assert canonical("lm-studio") == "lmstudio"
    assert canonical("chatgpt") == "openai"
    assert canonical("grok") == "xai"
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

    def fake_complete(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
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

    def fake_complete(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
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


# ── failure containment and Ollama context ────────────────────────────────

def test_ollama_uses_its_native_api_with_num_ctx(monkeypatch):
    """The OpenAI-compatible endpoint cannot set num_ctx, and Ollama's 4096
    default leaves a small model no room to write a full page."""
    import kinetixai.primers.engines as e
    seen = {}

    class FakeResponse:
        status_code = 200
        text = ""
        @staticmethod
        def json():
            return {"message": {"content": '{"ok": 1}'}}

    class FakeClient:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json=None, headers=None):
            seen["url"] = url
            seen["body"] = json
            return FakeResponse()

    monkeypatch.setattr(e.httpx, "Client", FakeClient)
    e.complete("ollama", "qwen2.5:7b", "sys", "user", num_ctx=16384)
    assert seen["url"].endswith("/api/chat")          # native, not /v1/chat/completions
    assert seen["body"]["options"]["num_ctx"] == 16384
    assert seen["body"]["format"] == "json"


def test_ollama_v1_suffix_is_stripped(monkeypatch):
    import kinetixai.primers.engines as e
    seen = {}

    class FakeResponse:
        status_code = 200
        text = ""
        @staticmethod
        def json():
            return {"message": {"content": "{}"}}

    class FakeClient:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json=None, headers=None):
            seen["url"] = url
            return FakeResponse()

    monkeypatch.setattr(e.httpx, "Client", FakeClient)
    e.complete("ollama", "m", "s", "u", api_base="http://localhost:11434/v1")
    assert seen["url"] == "http://localhost:11434/api/chat"


def test_empty_ollama_reply_blames_the_context_window(monkeypatch):
    import kinetixai.primers.engines as e

    class FakeResponse:
        status_code = 200
        text = ""
        @staticmethod
        def json():
            return {"message": {"content": ""}, "done_reason": "length"}

    class FakeClient:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **kw): return FakeResponse()

    monkeypatch.setattr(e.httpx, "Client", FakeClient)
    with pytest.raises(EngineError, match="num_ctx|context"):
        e.complete("ollama", "m", "s", "u")


def test_network_timeouts_become_readable_errors(monkeypatch):
    """A timeout used to escape as a traceback half way through a build."""
    import httpx
    import kinetixai.primers.engines as e

    class FakeClient:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **kw): raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(e.httpx, "Client", FakeClient)
    with pytest.raises(EngineError, match="timed out"):
        e.complete("ollama", "m", "s", "u")
    with pytest.raises(EngineError, match="timed out"):
        e.complete("vllm", "m", "s", "u")


def test_connection_refused_names_the_fix(monkeypatch):
    import httpx
    import kinetixai.primers.engines as e

    class FakeClient:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, *a, **kw): raise httpx.ConnectError("nope")

    monkeypatch.setattr(e.httpx, "Client", FakeClient)
    with pytest.raises(EngineError, match="ollama serve"):
        e.complete("ollama", "m", "s", "u")


def test_one_failing_page_does_not_lose_the_others(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings, relevance=0.18: [])
    calls = {"n": 0}

    def flaky(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("some unexpected failure")      # not an EngineError
        return json.dumps({"title": f"Page {calls['n']}", "takeaway": "t",
                           "blocks": [dict(x) for x in DENSE_BLOCKS]})

    monkeypatch.setattr(e, "complete", flaky)
    req = BuildRequest(topic="T", pages=3, pdfs=0, jsons=0, engine="ollama",
                       mode="iterative", out_dir=tmp_path / "d")
    res = b.build(req)
    assert len(res.primer.pages) == 2                          # page 2 lost, not the build
    assert any("some unexpected failure" in w for w in res.warnings)


def test_page_skeleton_lists_every_required_block():
    spec = blueprint.plan(8)[6]           # evaluation
    sk = blueprint.page_skeleton(spec)
    for kind in ("table", "code", "cloud", "chart"):
        assert kind in sk
    assert "Do not stop early" in sk


# ── provider registry ─────────────────────────────────────────────────────

from kinetixai.primers import providers as pv  # noqa: E402


def test_registry_covers_local_and_hosted_engines():
    for name in ("ollama", "vllm", "llamacpp", "lmstudio", "jan", "localai"):
        assert pv.PROVIDERS[name].local, name
    for name in ("anthropic", "openai", "gemini", "groq", "openrouter", "together"):
        assert not pv.PROVIDERS[name].local, name
    assert len(pv.NAMES) >= 20


def test_every_provider_is_usable():
    """A registry row with no endpoint and no protocol would fail at run time."""
    for p in pv.PROVIDERS.values():
        if p.protocol in ("mcp", "none"):
            continue
        assert p.base.startswith("http"), p.name
        if not p.local:
            assert p.keys, f"{p.name} is hosted but names no API key"
            # Either a pinned default, or an endpoint kx can ask. A pinned
            # name on a provider that rotates ids is how every page of a
            # Groq build came back 404 model_not_found.
            assert p.default_model or p.protocol == "openai", \
                f"{p.name} has no default model and no discoverable model list"


def test_aliases_are_unique_and_resolve():
    assert len(pv.ALIASES) == len({*pv.ALIASES})
    for alias, target in pv.ALIASES.items():
        assert target in pv.PROVIDERS
        assert alias not in pv.PROVIDERS, f"{alias} is both a name and an alias"


def test_base_url_prefers_override_then_env(monkeypatch):
    p = pv.PROVIDERS["groq"]
    assert pv.base_url(p, "http://custom/v1") == "http://custom/v1"
    monkeypatch.setenv("GROQ_BASE_URL", "http://from-env/v1")
    assert pv.base_url(p) == "http://from-env/v1"
    monkeypatch.delenv("GROQ_BASE_URL")
    assert pv.base_url(p) == p.base


def test_hosted_engine_without_a_key_says_which_one(monkeypatch):
    import kinetixai.primers.engines as e
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(EngineError, match="GROQ_API_KEY"):
        e.complete("groq", None, "s", "u")


def test_openai_compatible_providers_share_one_path(monkeypatch):
    """Adding a backend should be a registry row, not a new code branch."""
    import kinetixai.primers.engines as e
    seen = {}

    class FakeResponse:
        status_code = 200
        text = ""
        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "{}"}}]}

    class FakeClient:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json=None, headers=None):
            seen["url"], seen["model"] = url, json["model"]
            return FakeResponse()

    monkeypatch.setattr(e.httpx, "Client", FakeClient)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    # Groq pins no default, so the served list is what a bare call resolves to.
    monkeypatch.setattr(pv, "models", lambda p, timeout=None: ["openai/gpt-oss-120b"])
    e.complete("groq", None, "s", "u")
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["model"] == "openai/gpt-oss-120b"
    monkeypatch.setenv("TOGETHER_API_KEY", "x")
    e.complete("together", "my-model", "s", "u")
    assert seen["url"].startswith("https://api.together.xyz")
    assert seen["model"] == "my-model"


def test_pick_prefers_a_running_local_server(monkeypatch):
    monkeypatch.setattr(pv, "models",
                        lambda p, timeout=2.0: ["qwen2.5-7b"] if p.name == "lmstudio" else [])
    monkeypatch.setenv("OPENAI_API_KEY", "x")          # a key exists, but local wins
    name, why = pv.pick()
    assert name == "lmstudio" and "LM Studio" in why and "qwen2.5-7b" in why


def test_pick_falls_back_to_a_configured_key(monkeypatch):
    monkeypatch.setattr(pv, "models", lambda p, timeout=2.0: [])
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    name, why = pv.pick()
    assert name == "groq" and "GROQ_API_KEY" in why


def test_pick_explains_itself_when_nothing_is_available(monkeypatch):
    monkeypatch.setattr(pv, "models", lambda p, timeout=2.0: [])
    for p in pv.PROVIDERS.values():
        for env in p.keys:
            monkeypatch.delenv(env, raising=False)
    name, why = pv.pick()
    assert name == "outline"
    assert "kx primers engines" in why          # tells the user where to look


def test_pick_honours_an_explicit_choice():
    assert pv.pick("claude")[0] == "anthropic"
    assert pv.pick("lm-studio")[0] == "lmstudio"


def test_auto_engine_is_resolved_and_reported(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings, relevance=0.18: [])
    monkeypatch.setattr(b.providers, "pick", lambda prefer=None, check_local=True:
                        ("outline", "nothing available"))
    logged: list[str] = []
    req = BuildRequest(topic="T", pages=1, pdfs=0, jsons=0, engine="auto",
                       out_dir=tmp_path / "d")
    res = b.build(req, log=logged.append)
    assert any("engine: outline" in line for line in logged)
    assert req.engine == "outline"


def test_status_rows_cover_every_engine():
    rows = pv.status(check_local=False)
    assert {r["engine"] for r in rows} == set(pv.NAMES)
    for r in rows:
        assert r["kind"] in ("local", "hosted", "agent")


# ── ambiguity markers must match whole words ──────────────────────────────

def test_marker_matching_is_word_boundary_not_substring():
    """'iter' (for ITER, the reactor) must not match 'iterative'."""
    papers = [
        {"title": "GPU implementation of a ray-surface intersection algorithm",
         "abstract": "An iterative CUDA kernel for ray-surface intersection on the GPU."},
        {"title": "CUDA-DClust+: Revisiting Early GPU-Accelerated DBSCAN Clustering",
         "abstract": "We revisit GPU-accelerated DBSCAN with CUDA kernels."},
    ]
    kept, notes = primer_search.filter_sources(papers, "GPU & CUDA: Kernel Fusion")
    assert len(kept) == 2, notes


def test_real_plasma_physics_is_still_dropped():
    papers = [{"title": "Challenges for AI to help deliver fusion energy",
               "abstract": "Tokamak plasma control for nuclear fusion energy at ITER."}]
    kept, notes = primer_search.filter_sources(papers, "GPU & CUDA: Kernel Fusion")
    assert kept == [] and notes


def test_a_strong_topic_match_survives_one_stray_marker():
    """A CUDA paper that happens to say 'plasma' once is still a CUDA paper."""
    papers = [{"title": "CUDA kernel fusion for GPU particle-in-cell simulation",
               "abstract": "Fused CUDA kernels on the GPU for plasma simulation workloads."}]
    kept, _ = primer_search.filter_sources(papers, "GPU & CUDA: Kernel Fusion")
    assert len(kept) == 1


# ── config validation ─────────────────────────────────────────────────────

def test_primer_engine_setting_rejects_nonsense():
    from kinetixai import config
    with pytest.raises(ValueError, match="primer_engine must be"):
        config.validate("primer_engine", "claude# 1. No flags")
    with pytest.raises(ValueError, match="kx primers engines"):
        config.validate("primer_engine", "gpt5-turbo-ultra")


def test_primer_engine_setting_normalises_aliases():
    from kinetixai import config
    assert config.validate("primer_engine", "claude") == "anthropic"
    assert config.validate("primer_engine", "lm-studio") == "lmstudio"
    assert config.validate("primer_engine", "auto") == "auto"


def test_optional_primer_settings_can_be_cleared():
    from kinetixai import config
    assert config.validate("primer_model", "") == ""
    assert config.validate("primer_copy_to", "") == ""


# ── copy-to ───────────────────────────────────────────────────────────────

def test_copy_to_places_every_file_and_keeps_images_folder(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings, relevance=0.18: [])
    dest = tmp_path / "notes"
    req = BuildRequest(topic="Shared Topic", pages=1, pdfs=1, images=1, jsons=1,
                       engine="outline", out_dir=tmp_path / "out", copy_to=dest)
    res = b.build(req)
    names = sorted(p.name for p in dest.rglob("*") if p.is_file())
    assert any(n.endswith(".pdf") for n in names)
    assert any(n.endswith(".json") for n in names)
    assert (dest / "images").is_dir()
    assert len(res.copied) == len(res.files())


def test_copy_to_failure_does_not_fail_the_build(tmp_path: Path, monkeypatch):
    """A missing shared drive must not throw away a primer that was built."""
    import kinetixai.primers.builder as b
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings, relevance=0.18: [])
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("i am a file", encoding="utf-8")
    req = BuildRequest(topic="T", pages=1, pdfs=1, engine="outline",
                       out_dir=tmp_path / "out", copy_to=blocker)
    res = b.build(req)
    assert res.pdf_paths and res.pdf_paths[0].is_file()   # the build still stands
    assert res.copied == []
    assert any("could not copy" in w for w in res.warnings)


# ── wrong server on a busy port, and failures that explain themselves ─────

def _fake_http(monkeypatch, payload, status=200):
    """providers.models imports httpx locally, so patch the module itself."""
    import httpx

    class R:
        status_code = status
        @staticmethod
        def json():
            return payload

    class C:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, headers=None): return R()

    monkeypatch.setattr(httpx, "Client", C)


def test_a_plain_web_server_is_not_an_inference_server(monkeypatch):
    """8080 is a busy port. Anything answering there is not necessarily a model."""
    _fake_http(monkeypatch, {"message": "hello from a dev server"})
    assert pv.models(pv.PROVIDERS["llamacpp"]) == []
    assert pv.reachable(pv.PROVIDERS["llamacpp"]) is False


def test_an_openai_model_list_is_recognised(monkeypatch):
    _fake_http(monkeypatch, {"data": [{"id": "qwen2.5-7b"}, {"id": "phi-4"}]})
    assert pv.models(pv.PROVIDERS["llamacpp"]) == ["qwen2.5-7b", "phi-4"]
    assert pv.reachable(pv.PROVIDERS["llamacpp"]) is True


def test_an_ollama_model_list_is_recognised(monkeypatch):
    _fake_http(monkeypatch, {"models": [{"name": "llama3.1:8b"}]})
    assert pv.models(pv.PROVIDERS["ollama"]) == ["llama3.1:8b"]


def test_a_non_200_is_not_reachable(monkeypatch):
    _fake_http(monkeypatch, {"data": []}, status=404)
    assert pv.models(pv.PROVIDERS["lmstudio"]) == []


def test_local_engine_asks_the_server_which_model_it_serves(monkeypatch):
    """llama.cpp and LM Studio serve whatever is loaded; demanding --model
    for a name the user cannot easily give is the wrong default."""
    import kinetixai.primers.engines as e
    monkeypatch.setattr(e.providers, "models", lambda p, timeout=2.0: ["loaded-model"])
    seen = {}

    class R:
        status_code = 200
        text = ""
        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "{}"}}]}

    class C:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json=None, headers=None):
            seen["model"] = json["model"]
            return R()

    monkeypatch.setattr(e.httpx, "Client", C)
    e.complete("llamacpp", None, "s", "u")
    assert seen["model"] == "loaded-model"


def test_every_page_failing_reports_why(tmp_path: Path, monkeypatch):
    """The per-page reasons are the diagnosis; a generic message is useless."""
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    monkeypatch.setattr(b, "_research", lambda topic, limit, warnings, relevance=0.18: [])

    def always_fails(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
        raise e.EngineError("HTTP 400: no model is loaded")

    monkeypatch.setattr(e, "complete", always_fails)
    req = BuildRequest(topic="T", pages=3, pdfs=0, jsons=0, engine="llamacpp",
                       mode="iterative", out_dir=tmp_path / "d")
    with pytest.raises(e.EngineError) as excinfo:
        b.build(req)
    message = str(excinfo.value)
    assert "no model is loaded" in message          # the real cause survives
    assert "llamacpp" in message                    # and which engine it was
    assert "kx primers engines" in message          # and what to do next


# ── choosing a model the server actually has ──────────────────────────────

def test_embedding_models_are_not_offered_as_chat_models(monkeypatch):
    """nomic-embed-text is installed on most RAG machines and cannot chat."""
    _fake_http(monkeypatch, {"models": [{"name": "nomic-embed-text:latest"},
                                        {"name": "qwen2.5:7b"},
                                        {"name": "bge-m3:latest"}]})
    p = pv.PROVIDERS["ollama"]
    assert pv.chat_models(p) == ["qwen2.5:7b"]


def test_a_configured_model_that_was_never_pulled_is_not_used(monkeypatch):
    """ollama_model defaults to llama3.1 whether or not it exists."""
    _fake_http(monkeypatch, {"models": [{"name": "qwen3.6-32k:latest"}]})
    p = pv.PROVIDERS["ollama"]
    assert pv.choose_model(p, "llama3.1") == "qwen3.6-32k:latest"


def test_a_configured_model_wins_when_the_server_has_it(monkeypatch):
    _fake_http(monkeypatch, {"models": [{"name": "qwen3.6-32k:latest"},
                                        {"name": "qwen2.5:7b"}]})
    p = pv.PROVIDERS["ollama"]
    assert pv.choose_model(p, "qwen2.5:7b") == "qwen2.5:7b"


def test_a_tagless_preference_matches_its_tagged_install(monkeypatch):
    _fake_http(monkeypatch, {"models": [{"name": "llama3.1:8b"}]})
    p = pv.PROVIDERS["ollama"]
    assert pv.choose_model(p, "llama3.1") == "llama3.1:8b"


def test_ollama_engine_uses_a_model_the_server_has(monkeypatch):
    import kinetixai.primers.engines as e
    monkeypatch.setattr(e.providers, "choose_model",
                        lambda p, preferred=None, timeout=2.0: "qwen3.6-32k:latest")
    seen = {}

    class R:
        status_code = 200
        text = ""
        @staticmethod
        def json():
            return {"message": {"content": "{}"}}

    class C:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json=None, headers=None):
            seen["model"] = json["model"]
            return R()

    monkeypatch.setattr(e.httpx, "Client", C)
    e.complete("ollama", None, "s", "u")
    assert seen["model"] == "qwen3.6-32k:latest"


def test_ollama_with_no_chat_model_says_what_to_pull(monkeypatch):
    import kinetixai.primers.engines as e
    monkeypatch.setattr(e.providers, "choose_model",
                        lambda p, preferred=None, timeout=2.0: None)
    with pytest.raises(EngineError, match="ollama pull"):
        e.complete("ollama", None, "s", "u")


# ── dependencies are checked before the model runs, not after ─────────────

def _block_import(monkeypatch, name, error=None):
    """Make one module fail to import, the way a real environment does."""
    import importlib
    real = importlib.import_module

    def fake(mod, *a, **kw):
        if mod == name:
            raise error or ImportError("No module named %r" % name)
        return real(mod, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", fake)


def test_preflight_runs_before_any_model_call(tmp_path: Path, monkeypatch):
    """Discovering a missing renderer after eight pages of local inference
    throws away everything the run produced."""
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e

    called = {"model": False}

    def never(*a, **kw):
        called["model"] = True
        raise AssertionError("the model must not be called before preflight")

    monkeypatch.setattr(e, "complete", never)
    monkeypatch.setattr(b, "_research", lambda *a, **kw: [])
    _block_import(monkeypatch, "jinja2")

    req = BuildRequest(topic="T", pages=8, pdfs=1, engine="ollama",
                       out_dir=tmp_path / "d")
    with pytest.raises(b.MissingDependency, match="jinja2"):
        b.build(req)
    assert called["model"] is False


def test_preflight_names_the_install_command(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    _block_import(monkeypatch, "pptx")
    with pytest.raises(b.MissingDependency, match=r"kinetixai\[primers\]"):
        b.preflight(BuildRequest(topic="T", ppts=1, out_dir=tmp_path))


def test_preflight_only_checks_what_was_asked_for(tmp_path: Path, monkeypatch):
    """No .pptx requested means python-pptx is not required."""
    import kinetixai.primers.builder as b
    _block_import(monkeypatch, "pptx")
    b.preflight(BuildRequest(topic="T", pdfs=1, ppts=0, out_dir=tmp_path))


def test_preflight_catches_an_installed_but_broken_library(tmp_path: Path, monkeypatch):
    """WeasyPrint installs cleanly on macOS and then fails at import without
    Homebrew's pango. find_spec cannot see that; importing can."""
    import kinetixai.primers.builder as b
    _block_import(monkeypatch, "weasyprint",
                  OSError("cannot load library 'libpango-1.0-0': not found"))
    with pytest.raises(b.MissingDependency) as excinfo:
        b.preflight(BuildRequest(topic="T", pdfs=1, out_dir=tmp_path))
    message = str(excinfo.value)
    assert "will not import" in message
    assert "libpango" in message
    assert "brew install pango" in message       # and how to fix it
    assert "KX_PDF_ENGINE=playwright" in message


def test_written_pages_survive_a_rendering_failure(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    monkeypatch.setattr(b, "_research", lambda *a, **kw: [])
    monkeypatch.setattr(b.renderer, "render_html",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tmp_path / "out"
    req = BuildRequest(topic="Rescue Me", pages=1, pdfs=0, engine="outline", out_dir=out)
    with pytest.raises(RuntimeError, match="boom"):
        b.build(req)
    rescued = list(out.glob("*.primer.json"))
    assert rescued, "the generated pages were lost"
    data = json.loads(rescued[0].read_text())
    assert data["pages"]


# ── ingested markdown is prose, not GitHub markup ─────────────────────────

def test_readme_badges_do_not_reach_the_page():
    """A README is written for GitHub. Badges and link syntax used to be
    printed verbatim in the PDF."""
    from kinetixai.primers.ingest import clean_markdown
    readme = (
        "# kx\n\n"
        "[![PyPI](https://img.shields.io/pypi/v/kinetixai)](https://pypi.org/project/kinetixai/)\n"
        "[![ci](https://github.com/o/r/actions/workflows/ci.yml/badge.svg)](https://github.com/o/r)\n\n"
        "<!-- hidden -->\n"
        "Search across [arXiv](https://arxiv.org) and **more**.\n\n"
        "---\n<p>Some <b>HTML</b>.</p>\n"
        "[ref]: https://example.com\n"
    )
    out = clean_markdown(readme)
    for junk in ("img.shields.io", "![", "](", "<!--", "<p>", "<b>", "**", "[ref]:"):
        assert junk not in out, f"{junk!r} survived: {out!r}"
    assert "arXiv" in out and "more" in out and "Some HTML" in out


def test_clean_markdown_keeps_identifiers_with_underscores():
    """Stripping underscores would mangle the config keys READMEs are full of."""
    from kinetixai.primers.ingest import clean_markdown
    out = clean_markdown("Set `primer_copy_to` and **primer_engine** to begin.")
    assert "primer_copy_to" in out and "primer_engine" in out


def test_unbalanced_emphasis_is_swept_up():
    """A README with odd ** markers, or prose cut mid-emphasis, left stray
    markers in the printed page."""
    from kinetixai.primers.ingest import clean_markdown
    assert "**" not in clean_markdown("odd **one and **two** three")
    assert "two" in clean_markdown("odd **one and **two** three")


def test_clean_markdown_survives_empty_and_plain_text():
    from kinetixai.primers.ingest import clean_markdown
    assert clean_markdown("") == ""
    assert clean_markdown("just words") == "just words"


# ── findings from the review pass ─────────────────────────────────────────

def test_doctor_advice_does_not_crash_without_the_primers_extra():
    """The hint fired a NameError for exactly the user it was written for."""
    from kinetixai import doctor
    ch = doctor.run(offline=True)
    ch["extras"]["primers"]["installed"] = False
    lines = doctor.advice(ch)
    assert any("primers" in line.lower() for line in lines)


def test_manifest_accepts_the_cli_flag_spelling(tmp_path: Path):
    """--json-bundles is the flag; a manifest saying json_bundles was ignored."""
    m = tmp_path / "cat.yaml"
    m.write_text("defaults:\n  json_bundles: 1\n  pdf: 1\n"
                 "tracks:\n  - topic: One\n", encoding="utf-8")
    reqs = load_manifest(m)
    assert reqs[0].jsons == 1
    assert reqs[0].pdfs == 1


def test_manifest_reports_a_key_it_ignores(tmp_path: Path, capsys):
    m = tmp_path / "cat.yaml"
    m.write_text("defaults:\n  pagez: 4\ntracks:\n  - topic: One\n", encoding="utf-8")
    load_manifest(m)
    assert "pagez" in capsys.readouterr().err


def test_manifest_carries_the_newer_options(tmp_path: Path):
    m = tmp_path / "cat.yaml"
    m.write_text("defaults:\n  mode: single\n  relevance: 0.5\n  num_ctx: 8192\n"
                 "tracks:\n  - topic: One\n", encoding="utf-8")
    r = load_manifest(m)[0]
    assert r.mode == "single" and r.relevance == 0.5 and r.num_ctx == 8192


def test_rst_readmes_become_prose():
    """A .rst project used to contribute only its metadata."""
    from kinetixai.primers.ingest import _strip_rst
    out = _strip_rst(
        "Project\n=======\n\n.. image:: https://img.shields.io/badge.svg\n"
        "   :target: https://example.com\n\n"
        "See `the docs <https://example.com/docs>`_ and :mod:`mymodule`.\n")
    assert "shields.io" not in out and "image::" not in out
    assert "the docs" in out and "mymodule" in out


# ── reading a repository deeply ───────────────────────────────────────────

from kinetixai.primers import repo as repo_mod  # noqa: E402


def test_subdirectory_urls_are_honoured():
    """A /tree/main/Folder URL means that folder. The path used to be
    discarded, so the repo root README was read instead."""
    r = repo_mod.parse_url("https://github.com/rpaut03l/TS-01/tree/main/DSA%26T")
    assert (r.owner, r.repo, r.ref, r.path) == ("rpaut03l", "TS-01", "main", "DSA&T")
    assert r.is_file is False


def test_every_github_url_form_parses():
    cases = {
        "https://github.com/o/r": ("o", "r", "", "", False),
        "https://github.com/o/r.git": ("o", "r", "", "", False),
        "git@github.com:o/r.git": ("o", "r", "", "", False),
        "https://github.com/o/r/tree/v1.2/docs/guide/": ("o", "r", "v1.2", "docs/guide", False),
        "https://github.com/o/r/blob/dev/src/main.py": ("o", "r", "dev", "src/main.py", True),
    }
    for url, expected in cases.items():
        r = repo_mod.parse_url(url)
        assert (r.owner, r.repo, r.ref, r.path, r.is_file) == expected, url
    assert repo_mod.parse_url("https://example.com/not/github") is None


def test_noise_is_never_read():
    for path in ("node_modules/left-pad/index.js", "dist/bundle.min.js",
                 "package-lock.json", ".venv/lib/python3.12/site-packages/x.py",
                 "assets/logo.png", "model.safetensors", "build/out.o",
                 "src/kinetixai.egg-info/PKG-INFO"):
        assert repo_mod._is_skipped(path) or not repo_mod._is_readable(path), path


def test_source_and_docs_are_read():
    for path in ("src/app.py", "cmd/main.go", "kernel.cu", "docs/guide.md",
                 "notebook.ipynb", "Dockerfile", "Makefile", "pyproject.toml"):
        assert repo_mod._is_readable(path) and not repo_mod._is_skipped(path), path


def test_ranking_puts_explaining_files_first():
    from kinetixai.primers.search import terms
    topic = terms("CLI argument parsing")
    files = [("tests/test_parser.py", 9000), ("README.md", 4000),
             ("src/parser.py", 12000), ("vendor/x.py", 500),
             ("docs/architecture.md", 6000)]
    ranked = sorted(files, key=lambda e: -repo_mod.score(e[0], e[1], topic))
    names = [p for p, _ in ranked]
    assert names[0] == "README.md"
    assert names.index("src/parser.py") < names.index("tests/test_parser.py")
    assert names.index("docs/architecture.md") < names.index("tests/test_parser.py")


def test_notebook_keeps_source_and_drops_output():
    nb = json.dumps({"cells": [
        {"cell_type": "markdown", "source": ["# Title\n", "Prose here.\n"]},
        {"cell_type": "code", "source": ["print('hi')\n"],
         "outputs": [{"text": ["a megabyte of output"]}]},
    ]})
    out = repo_mod._notebook_text(nb)
    assert "Prose here" in out and "print('hi')" in out
    assert "megabyte of output" not in out


def test_a_rejected_token_falls_back_to_anonymous(monkeypatch):
    """A stale GITHUB_TOKEN made PUBLIC repos unreadable: raw.githubusercontent
    answers 404 to a bad Authorization header, which looks like a missing file."""
    import httpx
    seen = []

    class R:
        def __init__(self, code, text=""):
            self.status_code, self.text = code, text
        def json(self):
            return {}

    class C:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, headers=None, params=None):
            seen.append("auth" if (headers or {}).get("Authorization") else "anon")
            return R(404) if (headers or {}).get("Authorization") else R(200, "body")

    monkeypatch.setattr(httpx, "Client", C)
    with httpx.Client() as c:
        r = repo_mod._get(c, "https://raw.githubusercontent.com/o/r/HEAD/README.md",
                          {"Authorization": "Bearer stale"})
    assert r.status_code == 200 and r.text == "body"
    assert seen == ["auth", "anon"]
    assert getattr(r, "kx_token_ignored", False) is True


def test_digest_text_has_metadata_layout_then_files():
    d = repo_mod.RepoDigest(ref=repo_mod.parse_url("https://github.com/o/r"))
    d.metadata = {"description": "A thing", "language": "Python", "stars": 12}
    d.tree = [("src/a.py", 100), ("src/b.py", 200)]
    d.files = [("src/a.py", "def a(): pass")]
    text = d.to_text()
    assert text.index("REPOSITORY o/r") < text.index("LAYOUT")
    assert text.index("LAYOUT") < text.index("--- FILE: src/a.py ---")
    assert "A thing" in text and "language: Python" in text
    assert "def a(): pass" in text


# ── comparing models before committing to one ─────────────────────────────

def test_compare_runs_every_model_and_measures_it(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    monkeypatch.setattr(b, "_research", lambda *a, **kw: [])
    used: list[str] = []

    def fake(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
        used.append(model)
        blocks = [dict(x) for x in DENSE_BLOCKS]
        if model == "thin:1b":                      # a model that writes too little
            blocks = blocks[:2]
        return json.dumps({"title": "P", "takeaway": "t", "blocks": blocks})

    monkeypatch.setattr(e, "complete", fake)
    req = BuildRequest(topic="T", pages=1, pdfs=0, engine="ollama",
                       out_dir=tmp_path / "cmp")
    rows = b.compare_models(req, ["good:7b", "thin:1b"])
    assert [r["model"] for r in rows] == ["good:7b", "thin:1b"]
    assert used[0] == "good:7b" and "thin:1b" in used
    assert all(r["ok"] for r in rows)
    assert rows[0]["met_quota"] == 1 and rows[1]["met_quota"] == 0
    assert rows[0]["sections"] > rows[1]["sections"]
    assert all(isinstance(r["seconds"], float) for r in rows)


def test_compare_writes_each_sample_to_its_own_folder(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    monkeypatch.setattr(b, "_research", lambda *a, **kw: [])
    monkeypatch.setattr(e, "complete", lambda *a, **kw: json.dumps(
        {"title": "P", "takeaway": "t", "blocks": [dict(x) for x in DENSE_BLOCKS]}))
    req = BuildRequest(topic="T", pages=1, pdfs=1, engine="ollama",
                       out_dir=tmp_path / "cmp")
    b.compare_models(req, ["qwen2.5:7b", "llama3.1:8b"])
    folders = sorted(p.name for p in (tmp_path / "cmp").iterdir() if p.is_dir())
    assert folders == ["llama3-1-8b", "qwen2-5-7b"]       # one sample each, comparable


def test_a_failing_model_does_not_stop_the_comparison(tmp_path: Path, monkeypatch):
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    monkeypatch.setattr(b, "_research", lambda *a, **kw: [])

    def fake(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
        if model == "broken":
            raise e.EngineError("model not found")
        return json.dumps({"title": "P", "takeaway": "t",
                           "blocks": [dict(x) for x in DENSE_BLOCKS]})

    monkeypatch.setattr(e, "complete", fake)
    req = BuildRequest(topic="T", pages=1, pdfs=0, engine="ollama",
                       out_dir=tmp_path / "cmp")
    rows = b.compare_models(req, ["broken", "working"])
    assert rows[0]["ok"] is False and "model not found" in rows[0]["error"]
    assert rows[1]["ok"] is True                           # the rest still ran


def test_explicit_slug_names_the_files(tmp_path: Path, monkeypatch):
    """The title is the model's to reword; the slug has to match the URL."""
    import kinetixai.primers.builder as b
    monkeypatch.setattr(b, "_research", lambda *a, **kw: [])
    req = BuildRequest(topic="GPU & CUDA: How Your Model Actually Runs on Silicon",
                       slug="gpu-cuda", engine="outline", pages=1, pdfs=1,
                       jsons=1, out_dir=tmp_path / "d")
    b.build(req)
    names = sorted(p.name for p in (tmp_path / "d").iterdir())
    assert names == ["gpu-cuda.html", "gpu-cuda.json", "gpu-cuda.pdf"]


def test_manifest_slug_sets_folder_and_stem(tmp_path: Path):
    m = tmp_path / "cat.yaml"
    m.write_text(
        "defaults:\n  out_dir: ./dist/primers\n"
        "tracks:\n"
        "  - topic: 'Attention & Transformers'\n    slug: attention-transformers\n"
        "  - topic: 'No slug here'\n",
        encoding="utf-8")
    reqs = load_manifest(m)
    assert reqs[0].slug == "attention-transformers"
    assert reqs[0].out_dir == Path("dist/primers/attention-transformers")
    # Unset slug keeps the old behaviour: derived from the topic.
    assert reqs[1].slug == ""
    assert reqs[1].out_dir == Path("dist/primers/no-slug-here")


def test_shipped_catalog_covers_the_site_with_unique_slugs():
    reqs = load_manifest(Path(__file__).resolve().parents[1] / "primers_catalog.yaml")
    assert len(reqs) == 24                                  # the 24 "Coming soon" pages
    slugs = [r.slug for r in reqs]
    assert all(slugs) and len(set(slugs)) == len(slugs)     # no collisions on upload


def test_hosted_model_list_sends_the_api_key(monkeypatch):
    """Without auth, /models returns nothing and a stale default 404s."""
    import kinetixai.primers.providers as pv
    seen = {}

    class R:
        status_code = 200
        def json(self): return {"data": [{"id": "openai/gpt-oss-120b"}]}

    class C:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, headers=None):
            seen["headers"] = headers or {}
            return R()

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr("httpx.Client", C)
    assert pv.models(pv.get("groq")) == ["openai/gpt-oss-120b"]
    assert seen["headers"].get("Authorization") == "Bearer gsk_test"


def test_speech_and_guard_models_are_not_offered_as_chat(monkeypatch):
    import kinetixai.primers.providers as pv
    catalogue = ["whisper-large-v3", "canopylabs/orpheus-v1-english",
                 "meta-llama/llama-prompt-guard-2-86m", "openai/gpt-oss-safeguard-20b",
                 "openai/gpt-oss-20b", "openai/gpt-oss-120b", "allam-2-7b"]
    monkeypatch.setattr(pv, "models", lambda p, timeout=2.0: catalogue)
    chat = pv.chat_models(pv.get("groq"))
    assert "whisper-large-v3" not in chat
    assert "canopylabs/orpheus-v1-english" not in chat
    assert "meta-llama/llama-prompt-guard-2-86m" not in chat
    assert "openai/gpt-oss-safeguard-20b" not in chat
    # The biggest usable model leads, so an unattended pick is a sane one.
    assert pv.rank_models(chat)[0] == "openai/gpt-oss-120b"


def test_stale_hosted_default_falls_back_to_a_served_model(monkeypatch):
    """The Groq default 404'd on every page; it must not be used blindly."""
    import kinetixai.primers.providers as pv
    monkeypatch.setattr(pv, "models", lambda p, timeout=2.0:
                        ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
    got = pv.choose_model(pv.get("groq"), "llama-3.3-70b-versatile")
    assert got == "openai/gpt-oss-120b"


def test_explicit_model_is_never_second_guessed(monkeypatch):
    import kinetixai.primers.providers as pv
    monkeypatch.setattr(pv, "models", lambda p, timeout=2.0: ["qwen/qwen3.8-27b"])
    assert pv.choose_model(pv.get("groq"), "qwen/qwen3.8-27b") == "qwen/qwen3.8-27b"


def test_hosted_probe_gets_a_longer_timeout_than_localhost():
    """A 2s timeout against an internet API returned [] and the stale pin won."""
    import kinetixai.primers.providers as pv
    assert pv.probe_timeout(pv.get("ollama")) == pv.LOCAL_PROBE_TIMEOUT
    assert pv.probe_timeout(pv.get("groq")) == pv.HOSTED_PROBE_TIMEOUT
    assert pv.HOSTED_PROBE_TIMEOUT > pv.LOCAL_PROBE_TIMEOUT


def test_groq_has_no_pinned_default_model():
    """Groq retires ids, so a pin is a 404 on every page waiting to happen."""
    import kinetixai.primers.providers as pv
    assert pv.get("groq").default_model == ""


def test_a_hosted_engine_resolves_its_model_before_the_pages(tmp_path, monkeypatch):
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    import kinetixai.primers.providers as pv
    monkeypatch.setattr(b, "_research", lambda *a, **kw: [])
    monkeypatch.setattr(pv, "models", lambda p, timeout=None:
                        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
    seen = []

    def fake(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
        seen.append(model)
        return json.dumps({"title": "P", "takeaway": "t",
                           "blocks": [dict(x) for x in DENSE_BLOCKS]})

    monkeypatch.setattr(e, "complete", fake)
    req = BuildRequest(topic="T", pages=2, pdfs=0, engine="groq",
                       out_dir=tmp_path / "d")
    b.build(req)
    # One resolved name, reused by every page — not None, not a stale pin.
    assert set(seen) == {"openai/gpt-oss-120b"}
    assert req.model == "openai/gpt-oss-120b"


def test_legacy_completion_ids_do_not_win_the_ranking(monkeypatch):
    import kinetixai.primers.providers as pv
    monkeypatch.setattr(pv, "models", lambda p, timeout=None:
                        ["babbage-002", "dall-e-3", "davinci-002", "gpt-4o-mini"])
    chat = pv.chat_models(pv.get("openai"))
    assert chat == ["gpt-4o-mini"]


# ── ship_primers.py orchestration ─────────────────────────────────────────

def _ship():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ship_primers", Path(__file__).resolve().parents[1] / "scripts" / "ship_primers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ship_script_covers_every_catalog_track():
    ship = _ship()
    _, tracks = ship.load_catalog()
    assert len(tracks) == 24
    # defaults are merged in, so every track can be built on its own
    assert all(t.get("engine") and t.get("slug") and t.get("watermark") for t in tracks)


def test_ship_build_cmd_carries_watermark_persona_and_slug():
    ship = _ship()
    _, tracks = ship.load_catalog()
    track = next(t for t in tracks if t["slug"] == "gpu-cuda")
    cmd = ship.build_cmd(track, Path("/tmp/out"), "openai/gpt-oss-120b")
    assert "--slug" in cmd and cmd[cmd.index("--slug") + 1] == "gpu-cuda"
    assert cmd[cmd.index("--watermark") + 1] == "Rohit Patel | Narmadya Innovations"
    assert cmd[cmd.index("--model") + 1] == "openai/gpt-oss-120b"
    assert cmd[cmd.index("--out-dir") + 1].endswith("gpu-cuda")
    assert "," in cmd[cmd.index("--persona") + 1]


def test_ship_treats_a_thin_pdf_as_unfinished(tmp_path):
    """A build that technically produced a file is not a shippable primer."""
    ship = _ship()
    track = {"slug": "s", "topic": "T", "pages": 8}
    (tmp_path / "s").mkdir()
    pdf = tmp_path / "s" / "s.pdf"
    assert ship.is_finished(track, tmp_path) == (False, "no pdf")
    pdf.write_bytes(b"x" * 1000)
    ok, why = ship.is_finished(track, tmp_path)
    assert ok is False and "kb" in why


def test_ship_index_entry_points_at_the_published_url(tmp_path):
    ship = _ship()
    slug = "gpu-cuda"
    src = tmp_path / "out" / slug
    (src / "images").mkdir(parents=True)
    (src / f"{slug}.pdf").write_bytes(b"x" * (ship.MIN_PDF_BYTES + 1))
    (src / f"{slug}.pptx").write_bytes(b"x")
    (src / "images" / f"{slug}_page01_300dpi.png").write_bytes(b"x")
    track = {"slug": slug, "topic": "GPU & CUDA", "category": "gpu",
             "pages": 8, "personas": ["student"]}
    index = ship.stage([track], tmp_path / "out", tmp_path / "bundle")
    assert index[0]["url"] == "/primers/gpu-cuda/"
    assert index[0]["pdf"] == "gpu-cuda/gpu-cuda.pdf"
    assert index[0]["cover"].endswith("page01_300dpi.png")
    written = json.loads((tmp_path / "bundle" / "primers_index.json").read_text())
    assert written["count"] == 1


def test_ship_error_report_names_the_cause_not_the_hint():
    """Taking the stderr tail caught kx's 'Try ...' suggestion, not the error."""
    ship = _ship()
    out = ("error: every page failed to generate with engine 'groq'.\n"
           "  page 1 (overview) failed: LLM endpoint https://api.groq.com/openai/v1: "
           'HTTP 429: {"error":{"message":"Rate limit reached"}}\n\n'
           "  (and 5 more)\n\n"
           "Try `kx primers engines` to see what is ready, pass --engine explicitly.")
    got = ship.first_real_error(out)
    assert "429" in got and "Rate limit reached" in got
    assert "Try `kx" not in got
    assert ship.first_real_error("") == "failed with no error output"


def test_verify_model_names_a_project_level_block(monkeypatch):
    """/models lists the model; the project forbids it. Only a call reveals it."""
    import kinetixai.primers.engines as e
    blocked = ('LLM endpoint https://api.groq.com/openai/v1: HTTP 403: '
               '{"error":{"message":"The model `openai/gpt-oss-120b` is blocked at '
               'the project level.","code":"model_permission_blocked_project"}}')

    def fake(*a, **kw):
        raise e.EngineError(blocked)

    monkeypatch.setattr(e, "complete", fake)
    ok, why = e.verify_model("groq", "openai/gpt-oss-120b")
    assert ok is False and why == "blocked at project level"


def test_first_callable_skips_the_blocked_model(monkeypatch):
    import kinetixai.primers.engines as e

    def fake(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
        if model == "openai/gpt-oss-120b":
            raise e.EngineError("HTTP 403: model_permission_blocked_project")
        return "OK"

    monkeypatch.setattr(e, "complete", fake)
    model, report = e.first_callable("groq", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"])
    assert model == "qwen/qwen3.8-27b"
    assert report[0]["callable"] is False and report[1]["callable"] is True


def test_verify_model_never_raises(monkeypatch):
    import kinetixai.primers.engines as e

    def boom(*a, **kw):
        raise RuntimeError("socket exploded")

    monkeypatch.setattr(e, "complete", boom)
    ok, why = e.verify_model("groq", "m")
    assert ok is False and "socket exploded" in why


def test_ship_falls_back_when_verification_itself_breaks(monkeypatch):
    """A broken verify step must not be reported as 'no model can be called'."""
    ship = _ship()
    calls = []

    class R:
        def __init__(self, rc, out, err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "--verify" in cmd:
            return R(2, "", "Error: No such option: --verify")
        return R(0, json.dumps({"models": ["openai/gpt-oss-20b"]}))

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    usable, checks = ship.callable_models("groq")
    assert usable == ["openai/gpt-oss-20b"]       # fell back, did not report none
    assert checks == []
    assert any("--verify" in c for c in calls)


def test_ship_does_not_probe_a_local_server(monkeypatch):
    """Probing ollama loads the weights at full context for an 8-token call."""
    ship = _ship()
    seen = []

    class R:
        def __init__(self, out):
            self.returncode, self.stdout, self.stderr = 0, out, ""

    def fake_run(cmd, **kw):
        seen.append(cmd)
        if "--help" in cmd:
            return R("--slug  Output filename stem.")
        return R(json.dumps({"models": ["qwen3.6-32k:latest"]}))

    monkeypatch.setattr(ship.subprocess, "run", fake_run)
    monkeypatch.setattr(ship.shutil, "which", lambda n: "/usr/local/bin/kx")
    assert ship.preflight("ollama") == ["qwen3.6-32k:latest"]
    assert not any("--verify" in c for c in seen)


def test_ship_streams_page_progress(tmp_path):
    """A silent terminal for 20 minutes is indistinguishable from a hang."""
    import sys
    ship = _ship()
    script = tmp_path / "fake.py"
    script.write_text(
        "import sys\n"
        "for line in ['ingesting source material…', 'page 1/8 (overview)…',\n"
        "             'noise that is not progress', 'rendering HTML…']:\n"
        "    print(line, flush=True)\n", encoding="utf-8")
    proc, out = ship.stream([sys.executable, str(script)], "slug")
    assert proc.returncode == 0
    assert "page 1/8" in out and "rendering HTML" in out


def test_rate_limit_is_retried_then_explained(monkeypatch):
    """A per-minute budget refills; a reservation bigger than it never does."""
    import kinetixai.primers.engines as e
    sleeps, posts = [], []

    class R429:
        status_code, text, headers = 429, '{"error":"Rate limit reached"}', {"retry-after": "2"}

    class C:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json=None, headers=None):
            posts.append(json["max_tokens"])
            return R429()

    monkeypatch.setattr(e.httpx, "Client", C)
    monkeypatch.setattr(e.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(e, "RATE_LIMIT_RETRIES", 2)
    with pytest.raises(e.EngineError) as exc:
        e._openai_compatible("https://api.groq.com/openai/v1", "m", "s", "u",
                             key="k", max_tokens=6000)
    assert len(posts) == 3                       # first try plus two retries
    assert sleeps == [3.0, 3.0]                  # honoured Retry-After, +1s
    assert "6000 output tokens" in str(exc.value)
    assert "--max-tokens" in str(exc.value)


def test_max_tokens_flows_from_request_to_the_call(tmp_path, monkeypatch):
    import kinetixai.primers.builder as b
    import kinetixai.primers.engines as e
    monkeypatch.setattr(b, "_research", lambda *a, **kw: [])
    seen = []

    def fake(engine, model, system, user, api_base=None, max_tokens=None, num_ctx=None):
        seen.append(max_tokens)
        return json.dumps({"title": "P", "takeaway": "t",
                           "blocks": [dict(x) for x in DENSE_BLOCKS]})

    monkeypatch.setattr(e, "complete", fake)
    b.build(BuildRequest(topic="T", pages=1, pdfs=0, engine="outline",
                         out_dir=tmp_path / "a"))        # outline: no model call
    b.build(BuildRequest(topic="T", pages=1, pdfs=0, engine="ollama",
                         max_tokens=3500, out_dir=tmp_path / "b"))
    assert 3500 in seen
