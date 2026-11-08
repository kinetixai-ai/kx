"""Regression tests for the QA/security review findings."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from conftest import make_pdf
from kinetixai import api, cite, cli, config, library, mcp_server
from kinetixai.models import Paper
from kinetixai.rag import index as rag
from kinetixai.rag import ollama

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "papers.json"
TOKEN = "kx_" + "S" * 43


# ── 5. tracebacks never print locals (the token) ──────────────────────────

def test_typer_apps_never_show_locals():
    for t in (cli.app, cli.config_app, cli.library_app, cli.rag_app):
        assert t.pretty_exceptions_show_locals is False
        assert t.pretty_exceptions_enable is False


def test_crash_does_not_leak_the_token(tmp_path):
    script = textwrap.dedent(f"""
        import sys
        from kinetixai import cli, services
        def boom(paper, limit):
            token = "{TOKEN}"
            raise RuntimeError("unexpected")
        services.find_code = boom
        sys.argv = ["kx", "code", "2205.14135"]
        cli.main()
    """)
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=dict(os.environ))
    assert res.returncode != 0 and "RuntimeError" in res.stderr
    assert TOKEN not in res.stderr and TOKEN not in res.stdout


# ── 6. loopback means loopback ────────────────────────────────────────────

@pytest.mark.parametrize("host,ok", [
    ("localhost", True), ("LOCALHOST", True), ("127.0.0.1", True), ("127.9.9.9", True), ("::1", True),
    ("127.attacker.com", False), ("localhost.evil.example", False), ("10.0.0.1", False), ("", False), (None, False),
])
def test_is_loopback_host(host, ok):
    assert api.is_loopback_host(host) is ok


def test_lookalike_hosts_are_not_local():
    assert not ollama.is_local("http://127.attacker.com:11434")
    assert not ollama.is_local("http://localhost.evil.example")
    assert ollama.is_local("http://[::1]:11434")
    with httpx.Client() as c:
        with pytest.raises(api.ApiError):
            api.KinetixApi("http://127.attacker.com", c, TOKEN)


# ── 8. config: clean errors, never tracebacks ─────────────────────────────

@pytest.mark.parametrize("args,code", [
    (["config", "set", "endpoint", "https://[bad"], 2),
    (["config", "set", "endpoint", "kinetixai.ai"], 2),
    (["config", "set", "ollama_url", "http://"], 2),
    (["config", "set", "nonsense", "1"], 2),
    (["config", "set", "token_store", "vault"], 2),
    (["config", "set", "limit", "0"], 2),
    (["config", "get", "nonexistent"], 1),
])
def test_config_rejects_bad_input(args, code):
    res = runner.invoke(cli.app, args)
    assert res.exit_code == code, res.output
    assert res.exception is None or isinstance(res.exception, SystemExit)


def test_config_accepts_good_values():
    assert runner.invoke(cli.app, ["config", "set", "token_store", "file"]).exit_code == 0
    assert runner.invoke(cli.app, ["config", "set", "endpoint", "http://localhost:8793"]).exit_code == 0
    assert json.loads(runner.invoke(cli.app, ["config", "get", "endpoint", "--json"]).stdout) == {
        "endpoint": "http://localhost:8793"}


@pytest.mark.parametrize("content", [
    "[1, 2]", "null", '"text"', '{"endpoint": null}', '{"ollama_url": 5}', '{"endpoint": "https://[bad"}',
    '{"limit": "lots"}', "{not json",
])
def test_bad_config_file_falls_back_to_defaults(content, capsys):
    config._warned.clear()
    config.config_path().parent.mkdir(parents=True, exist_ok=True)
    config.config_path().write_text(content, encoding="utf-8")
    cfg = config.load()
    assert cfg["endpoint"] == config.DEFAULTS["endpoint"]
    assert cfg["ollama_url"] == config.DEFAULTS["ollama_url"] and cfg["limit"] == 10
    assert "warning:" in capsys.readouterr().err
    res = runner.invoke(cli.app, ["config", "show", "--json"])
    assert res.exit_code == 0 and json.loads(res.stdout)["settings"]["endpoint"] == "https://kinetixai.ai"
    # and setting a value repairs the file instead of crashing on it
    assert runner.invoke(cli.app, ["config", "set", "limit", "5"]).exit_code == 0


def test_bad_env_endpoint_ignored(monkeypatch, capsys):
    config._warned.clear()
    monkeypatch.setenv("KINETIXAI_ENDPOINT", "https://[bad")
    assert config.load()["endpoint"] == "https://kinetixai.ai"
    assert "KINETIXAI_ENDPOINT" in capsys.readouterr().err


def test_bad_endpoint_override_is_usage_error():
    res = runner.invoke(cli.app, ["whoami", "--endpoint", "https://[bad"])
    assert res.exit_code == 2 and "not 'https://[bad'" in res.stderr


def test_invalid_url_at_use_is_clean():
    with httpx.Client() as c:
        with pytest.raises(api.ApiError):
            api.KinetixApi("https://[bad", c, None).me()


# ── 8. MCP tools return errors instead of raising ─────────────────────────

def test_mcp_tools_never_raise(monkeypatch):
    def down(*a, **k):
        raise RuntimeError("arXiv: unreachable")
    wrapped = mcp_server._safe(down)
    assert wrapped("x") == {"ok": False, "error": "arXiv: unreachable", "warnings": ["arXiv: unreachable"]}


def test_mcp_get_paper_offline_is_a_clean_result():
    """A real stdio session with networking forced to fail."""
    pytest.importorskip("mcp")
    env = dict(os.environ, HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9",
               ALL_PROXY="", NO_PROXY="", no_proxy="")
    proc = subprocess.Popen([sys.executable, "-m", "kinetixai", "mcp"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)

    def send(msg):
        proc.stdin.write(json.dumps(msg) + "\n")
        proc.stdin.flush()

    def read(want):
        while True:
            line = proc.stdout.readline()
            assert line, proc.stderr.read()
            msg = json.loads(line)
            if msg.get("id") == want:
                return msg
    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
        read(1)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "get_paper", "arguments": {"id": "1706.03762"}}})
        result = read(2)["result"]
        payload = result.get("structuredContent") or json.loads(result["content"][0]["text"])
        assert payload["ok"] is False and payload["warnings"]
    finally:
        proc.stdin.close()
        proc.wait(timeout=20)
    assert "Traceback" not in proc.stderr.read()


# ── should-fix: search, bounds ────────────────────────────────────────────

def test_search_bad_source_and_total_failure(mock_http):
    assert runner.invoke(cli.app, ["search", "x", "--source", "scholar"]).exit_code == 2
    mock_http(lambda r: httpx.Response(503))
    res = runner.invoke(cli.app, ["search", "x"])
    assert res.exit_code == 1 and "unavailable" in res.stderr


@pytest.mark.parametrize("args", [
    ["search", "x", "--limit", "-5"], ["search", "x", "-n", "0"], ["code", "x", "-n", "0"],
    ["models", "x", "-n", "1000"], ["rag", "ask", "x", "-k", "0"], ["library", "search", "x", "-n", "-1"],
])
def test_out_of_range_numbers_are_usage_errors(args):
    assert runner.invoke(cli.app, args).exit_code == 2


def test_help_shows_named_metavars():
    # typer >= 0.2x marks required arguments with braces in the usage line;
    # what matters is a readable upper-case name, not the raw "{query}".
    for args, name in ((["search", "--help"], "QUERY"), (["library", "search", "--help"], "QUERY"),
                       (["rag", "ask", "--help"], "QUESTION"), (["cite", "--help"], "REF")):
        out = runner.invoke(cli.app, args).stdout
        assert name in out and "{query}" not in out and "{question}" not in out and "{ref}" not in out


# ── 3. library keys ───────────────────────────────────────────────────────

def test_cjk_only_titles_get_distinct_keys():
    a, _ = library.add(Paper(title="注意力机制综述", year=2024))
    b, _ = library.add(Paper(title="大型语言模型的对齐", year=2024))
    assert a["key"] != b["key"] and "untitled" not in a["key"]
    assert len(library.load()) == 2


def test_same_title_different_pdfs_are_both_kept(tmp_path):
    one, two = tmp_path / "a" / "paper.pdf", tmp_path / "b" / "paper.pdf"
    one.parent.mkdir()
    two.parent.mkdir()
    make_pdf(one, "first paper about mixtures")
    make_pdf(two, "second paper about routing")
    res = runner.invoke(cli.app, ["library", "add", str(one), str(two), "--copy", "--json"])
    added = json.loads(res.stdout)["added"]
    assert len({e["key"] for e in added}) == 2 and all(e["created"] for e in added)
    copies = sorted(Path(e["pdf_path"]) for e in added)
    assert copies[0] != copies[1] and all(c.exists() for c in copies)
    assert {c.read_bytes() for c in copies} == {one.read_bytes(), two.read_bytes()}
    # adding the same file again is an update, not a duplicate
    again = json.loads(runner.invoke(cli.app, ["library", "add", str(one), "--json"]).stdout)["added"][0]
    assert again["created"] is False and len(library.load()) == 2


def test_import_tags_as_string(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"title": "T", "tags": "abc, def"}, {"title": "U", "tags": 7}]), encoding="utf-8")
    runner.invoke(cli.app, ["library", "add", "--from-file", str(f)])
    tags = sorted(tuple(i["tags"]) for i in library.load())
    assert tags == [("7",), ("abc", "def")]


# ── BibTeX ────────────────────────────────────────────────────────────────

def test_bibtex_export_disambiguates_duplicate_keys():
    base = {"authors": ["Tri Dao"], "year": 2022}
    items = [dict(base, title="FlashAttention one", arxiv_id="2205.00001"),
             dict(base, title="FlashAttention two", arxiv_id="2205.00002"),
             dict(base, title="Other work", arxiv_id="2205.00003")]
    text = cite.bibtex_many(items)
    assert "{dao2022flashattentiona," in text and "{dao2022flashattentionb," in text
    assert "{dao2022other," in text


def test_bibtex_braces_are_escaped_and_balanced():
    b = cite.bibtex({"title": "Unbalanced } brace { and $x_1$ ~ ^", "authors": ["A}B"], "year": 2020})
    body = b.replace("\\{", "").replace("\\}", "")
    assert body.count("{") == body.count("}")
    assert "\\}" in b and "\\_" in b


def test_cli_export_uses_unique_keys():
    for i in (1, 2):
        library.add(Paper(title=f"Same Name {i}", authors=["Ann Lee"], year=2020, arxiv_id=f"2001.0000{i}"))
    out = runner.invoke(cli.app, ["library", "export"]).stdout
    assert "{lee2020samea," in out and "{lee2020sameb," in out


# ── doctor, whoami ────────────────────────────────────────────────────────

def test_doctor_extra_labels_are_spaced():
    out = runner.invoke(cli.app, ["doctor", "--offline"]).stdout
    assert "[keyring] extra " in out and "extrano" not in out and "extraok" not in out


def test_whoami_without_access_label(mock_http):
    user = {"id": "u", "email": "a@b.c", "name": None, "plan": None}
    mock_http(lambda r: httpx.Response(200, json={"ok": True, "user": user}))
    from kinetixai import auth
    auth.save("https://kinetixai.ai", TOKEN)
    out = runner.invoke(cli.app, ["whoami"]).stdout
    assert "Plan: Free" in out and "None" not in out


# ── RAG paths ─────────────────────────────────────────────────────────────

def test_rag_ignores_dotfiles_and_handles_missing_and_forgotten_paths(tmp_path):
    docs = tmp_path / "docs"
    (docs / ".git").mkdir(parents=True)
    (docs / ".git" / "notes.md").write_text("secret git internals", encoding="utf-8")
    (docs / ".hidden.md").write_text("hidden dotfile", encoding="utf-8")
    (docs / "real.md").write_text("visible routing note", encoding="utf-8")
    s = rag.build([str(docs)])
    assert s["documents"] == 1
    assert runner.invoke(cli.app, ["rag", "index", "--path", str(tmp_path / "nope")]).exit_code == 1
    gone = tmp_path / "gone"
    gone.mkdir()
    (gone / "x.md").write_text("temporary", encoding="utf-8")
    rag.build([str(gone)])
    gone.joinpath("x.md").unlink()
    gone.rmdir()
    s = rag.build()
    assert any("no longer exists" in w for w in s["warnings"])
    res = runner.invoke(cli.app, ["rag", "index", "--forget", str(gone), "--json"])
    assert str(gone.resolve()) not in json.loads(res.stdout)["paths"]
    res = runner.invoke(cli.app, ["rag", "index", "--forget-all-paths", "--json"])
    assert json.loads(res.stdout)["paths"] == []


# ── MCP snippets on Windows ───────────────────────────────────────────────

def test_windows_paths_with_spaces_are_quoted(monkeypatch):
    cmd = r"C:\Program Files\kx\kx.exe"
    assert mcp_server.config_snippet("open-webui", cmd)["snippet"] == f'uvx mcpo --port 8000 -- "{cmd}" mcp'
    monkeypatch.setattr(sys, "platform", "win32")
    g = mcp_server.config_snippet("goose", cmd)
    assert g["file"] == r"%APPDATA%\Block\goose\config\config.yaml"
    assert f'"{cmd}" mcp' in g["note"]
    assert json.dumps(cmd) in g["snippet"]
