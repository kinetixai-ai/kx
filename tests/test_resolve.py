import json

import httpx
from typer.testing import CliRunner

from kinetixai import cli
from kinetixai.resolve import arxiv_id, doi
from kinetixai.sources import openalex

runner = CliRunner()
ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/1706.03762v7</id><title>Attention Is All You Need</title>
<summary>  The dominant
 sequence transduction models.</summary>
<published>2017-06-12T00:00:00Z</published><author><name>Ashish Vaswani</name></author>
</entry></feed>"""


def test_arxiv_id_forms():
    assert arxiv_id("2205.14135") == "2205.14135"
    assert arxiv_id("arXiv:2205.14135v2") == "2205.14135v2"
    assert arxiv_id("https://arxiv.org/pdf/2205.14135v2.pdf") == "2205.14135v2"
    assert arxiv_id("https://huggingface.co/papers/2205.14135") == "2205.14135"
    assert arxiv_id("hep-th/9901001") == "hep-th/9901001"
    assert arxiv_id("flash attention") is None


def test_doi_forms():
    assert doi("https://doi.org/10.18653/v1/N19-1423") == "10.18653/v1/N19-1423"
    assert doi("doi:10.1145/3065386") == "10.1145/3065386"
    assert doi("2205.14135") is None


def test_get_arxiv_with_abstract_and_bibtex(mock_http):
    mock_http(lambda r: httpx.Response(200, text=ATOM))
    res = runner.invoke(cli.app, ["get", "1706.03762", "--json"])
    p = json.loads(res.stdout)
    assert p["title"] == "Attention Is All You Need" and p["abstract"] == "The dominant sequence transduction models."
    res = runner.invoke(cli.app, ["get", "1706.03762", "--bibtex"])
    assert res.stdout.startswith("@misc{vaswani2017attention")


def test_get_arxiv_falls_back_to_semantic_scholar(mock_http):
    def handler(req):
        if req.url.host == "export.arxiv.org":
            return httpx.Response(503)
        return httpx.Response(200, json={"title": "Attention Is All You Need", "year": 2017,
                                         "externalIds": {"ArXiv": "1706.03762"}, "abstract": "A."})
    seen = mock_http(handler)
    res = runner.invoke(cli.app, ["get", "1706.03762", "--json"])
    assert res.exit_code == 0 and json.loads(res.stdout)["abstract"] == "A."
    assert seen[-1].url.path.endswith("/paper/arXiv:1706.03762")


def test_get_doi_falls_back_to_openalex(mock_http):
    def handler(req):
        if "semanticscholar" in req.url.host:
            return httpx.Response(404)
        return httpx.Response(200, json={"display_name": "BERT", "publication_year": 2019,
                                         "abstract_inverted_index": {"Pre-training": [0], "BERT": [1]}})
    mock_http(handler)
    res = runner.invoke(cli.app, ["get", "10.18653/v1/N19-1423", "--json"])
    assert json.loads(res.stdout)["abstract"] == "Pre-training BERT"


def test_get_unknown_exits_1(mock_http):
    mock_http(lambda r: httpx.Response(404))
    assert runner.invoke(cli.app, ["get", "not-an-id"]).exit_code == 1


def test_openalex_abstract_rebuild():
    assert openalex._abstract({"world": [1], "hello": [0]}) == "hello world"
    assert openalex._abstract(None) is None
