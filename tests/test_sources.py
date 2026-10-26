import httpx

from kinetixai.models import Paper
from kinetixai.sources import aggregate, arxiv, openalex, semantic_scholar

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2205.14135v2</id>
    <title>FlashAttention: Fast and Memory-Efficient Exact Attention</title>
    <published>2022-05-27T00:00:00Z</published>
    <author><name>Tri Dao</name></author>
    <author><name>Dan Fu</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2205.14135v2" type="application/pdf"/>
  </entry>
</feed>"""


def test_arxiv_parse():
    papers = arxiv._parse(ATOM)
    assert len(papers) == 1
    p = papers[0]
    assert p.title.startswith("FlashAttention")
    assert p.year == 2022
    assert p.arxiv_id == "2205.14135v2"
    assert p.pdf_url.endswith("2205.14135v2")
    assert p.authors == ["Tri Dao", "Dan Fu"]


def test_semantic_scholar_paper():
    raw = {
        "title": "Attention Is All You Need",
        "year": 2017,
        "authors": [{"name": "Ashish Vaswani"}],
        "venue": "NeurIPS",
        "externalIds": {"DOI": "10.5555/abc", "ArXiv": "1706.03762"},
        "url": "https://semanticscholar.org/x",
        "citationCount": 100000,
        "openAccessPdf": {"url": "https://x/pdf"},
    }
    p = semantic_scholar._paper(raw)
    assert p.doi == "10.5555/abc"
    assert p.citations == 100000
    assert p.source == "Semantic Scholar"


def test_openalex_paper():
    raw = {
        "display_name": "BERT",
        "publication_year": 2019,
        "authorships": [{"author": {"display_name": "Jacob Devlin"}}],
        "primary_location": {"source": {"display_name": "NAACL"}, "pdf_url": "https://x/pdf"},
        "doi": "https://doi.org/10.18653/v1/n19-1423",
        "cited_by_count": 90000,
        "id": "https://openalex.org/W123",
    }
    p = openalex._paper(raw)
    assert p.venue == "NAACL"
    assert p.doi == "10.18653/v1/n19-1423"
    assert p.citations == 90000


def _mock_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if "arxiv.org" in host:
            return httpx.Response(200, text=ATOM)
        if "semanticscholar" in host:
            return httpx.Response(200, json={"data": [{
                "title": "FlashAttention", "year": 2022,
                "authors": [{"name": "Tri Dao"}], "venue": "NeurIPS",
                "externalIds": {"ArXiv": "2205.14135"}, "citationCount": 5000,
                "openAccessPdf": None, "url": "https://s2/x",
            }]})
        if "openalex" in host:
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_aggregate_dedup_and_warnings():
    with _mock_client() as client:
        papers, warnings = aggregate.search("flash attention", 5, client)
    # arXiv entry and the S2 entry share the arxiv id -> merged into one
    assert len(papers) == 1
    assert papers[0].citations == 5000  # enriched from Semantic Scholar
    assert warnings == []


def test_aggregate_one_source_down():
    def handler(request):
        if "openalex" in request.url.host:
            return httpx.Response(200, json={"results": []})
        raise httpx.ConnectError("down")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        papers, warnings = aggregate.search("x", 3, client)
    assert any("unavailable" in w for w in warnings)
