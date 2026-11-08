import json
from pathlib import Path

import httpx
from typer.testing import CliRunner

from conftest import make_pdf
from kinetixai import cli, library
from kinetixai.models import Paper

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "papers.json"
ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2106.09685v2</id><title>LoRA: Low-Rank Adaptation</title>
<summary>We freeze the weights and train low-rank matrices.</summary>
<published>2021-06-17T00:00:00Z</published><author><name>Edward J. Hu</name></author>
<link title="pdf" href="https://arxiv.org/pdf/2106.09685v2" type="application/pdf"/>
</entry></feed>"""


def test_import_fixture_list_and_find():
    res = runner.invoke(cli.app, ["library", "add", "--from-file", str(FIXTURE), "--json"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert len(out["added"]) == 3 and all(a["created"] for a in out["added"])
    items = library.load()
    assert [i["key"] for i in items] == ["arxiv:2205.14135", "arxiv:1706.03762", "arxiv:2106.09685"]
    assert items[0]["tags"] == ["attention", "gpu"]
    assert library.find("2")["key"] == "arxiv:1706.03762"
    assert library.find("arXiv:2205.14135v3")["key"] == "arxiv:2205.14135"
    # re-import updates instead of duplicating
    runner.invoke(cli.app, ["library", "add", "--from-file", str(FIXTURE)])
    assert len(library.load()) == 3


def test_list_filter_search_remove():
    runner.invoke(cli.app, ["library", "add", "--from-file", str(FIXTURE)])
    res = runner.invoke(cli.app, ["library", "list", "--tag", "gpu", "--json"])
    assert [i["key"] for i in json.loads(res.stdout)] == ["arxiv:2205.14135"]
    res = runner.invoke(cli.app, ["library", "search", "low-rank", "--json"])
    assert json.loads(res.stdout)[0]["key"] == "arxiv:2106.09685"
    res = runner.invoke(cli.app, ["library", "remove", "1706.03762", "--json"])
    assert json.loads(res.stdout) == {"removed": "arxiv:1706.03762"}
    assert runner.invoke(cli.app, ["library", "remove", "1706.03762"]).exit_code == 1
    assert len(library.load()) == 2


def test_export_bibtex_to_file_and_json(tmp_path):
    runner.invoke(cli.app, ["library", "add", "--from-file", str(FIXTURE)])
    out = tmp_path / "refs.bib"
    res = runner.invoke(cli.app, ["library", "export", "-o", str(out)])
    assert res.exit_code == 0
    text = out.read_text(encoding="utf-8")
    assert text.count("@") == 3 and "dao2022flashattention" in text
    res = runner.invoke(cli.app, ["library", "export", "--format", "json", "--json"])
    payload = json.loads(res.stdout)
    assert payload["count"] == 3 and json.loads(payload["content"])["papers"][0]["key"] == "arxiv:2205.14135"


def test_add_local_pdf_reads_title(tmp_path):
    pdf = tmp_path / "my notes.pdf"
    make_pdf(pdf, "Speculative decoding drafts tokens with a small model")
    res = runner.invoke(cli.app, ["library", "add", str(pdf), "--tag", "mine", "--json"])
    assert res.exit_code == 0, res.output
    entry = json.loads(res.stdout)["added"][0]
    assert entry["key"].startswith("local:my-notes-") and entry["pdf_path"] == str(pdf.resolve())
    assert entry["tags"] == ["mine"]


def test_add_by_arxiv_id_with_pdf_download(mock_http):
    def handler(req):
        if req.url.host == "export.arxiv.org":
            return httpx.Response(200, text=ATOM)
        if req.url.path.startswith("/pdf/"):
            return httpx.Response(200, content=b"%PDF-1.4 fake")
        return httpx.Response(404)
    mock_http(handler)
    res = runner.invoke(cli.app, ["library", "add", "2106.09685", "--pdf", "--json"])
    assert res.exit_code == 0, res.output
    entry = json.loads(res.stdout)["added"][0]
    assert entry["key"] == "arxiv:2106.09685" and entry["abstract"].startswith("We freeze")
    assert Path(entry["pdf_path"]).name == "2106.09685v2.pdf"
    assert Path(entry["pdf_path"]).read_bytes().startswith(b"%PDF")


def test_add_rejects_unknown_reference():
    res = runner.invoke(cli.app, ["library", "add", "not a paper"])
    assert res.exit_code == 1


def test_corrupt_library_is_reported(tmp_path):
    library.library_path().parent.mkdir(parents=True, exist_ok=True)
    library.library_path().write_text("{nope", encoding="utf-8")
    res = runner.invoke(cli.app, ["library", "list"])
    assert res.exit_code == 1 and "could not read" in res.stderr


def test_unicode_titles_survive_json_roundtrip():
    library.add(Paper(title="Résumé of “quotes” 注意力", authors=["Christopher Ré"], year=2024))
    res = runner.invoke(cli.app, ["library", "list", "--json"])
    assert json.loads(res.stdout)[0]["title"] == "Résumé of “quotes” 注意力"
    assert res.stdout.isascii()  # safe on any console code page
