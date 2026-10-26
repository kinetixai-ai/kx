import json

from typer.testing import CliRunner

from kinetixai import cli, config
from kinetixai.models import Paper, Researcher

runner = CliRunner()


def test_version():
    res = runner.invoke(cli.app, ["version"])
    assert res.exit_code == 0
    assert "kx" in res.stdout


def test_search_json(monkeypatch):
    sample = [Paper(title="FlashAttention", year=2022, citations=5000, source="arXiv")]
    monkeypatch.setattr(cli.aggregate, "search", lambda *a, **k: (sample, []))
    res = runner.invoke(cli.app, ["search", "flash attention", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    assert data[0]["title"] == "FlashAttention"


def test_scholar_rejects_google_scholar():
    res = runner.invoke(cli.app, ["scholar", "https://scholar.google.com/citations?user=x"])
    assert res.exit_code == 2


def test_scholar_json(monkeypatch):
    sample = [Researcher(name="Tri Dao", orcid="0000-0001-2345-6789", source="ORCID")]
    monkeypatch.setattr(cli.orcid, "lookup", lambda *a, **k: sample)
    res = runner.invoke(cli.app, ["scholar", "Tri Dao", "--json"])
    assert res.exit_code == 0
    assert "Tri Dao" in res.stdout


def test_ask(monkeypatch):
    monkeypatch.setattr(
        cli.KinetixApi, "ask",
        lambda self, q: {"ok": True, "answer": "Blocks fit SRAM.", "citations": [{"n": 1, "source": "Dao 2022"}]},
    )
    res = runner.invoke(cli.app, ["ask", "what is flash attention?"])
    assert res.exit_code == 0
    assert "SRAM" in res.stdout


def test_config_roundtrip(tmp_path, monkeypatch):
    cfgfile = tmp_path / "config.json"
    monkeypatch.setattr(config, "config_path", lambda: cfgfile)
    assert runner.invoke(cli.app, ["config", "set", "limit", "25"]).exit_code == 0
    res = runner.invoke(cli.app, ["config", "get", "limit"])
    assert res.stdout.strip() == "25"
