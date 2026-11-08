"""Hugging Face and GitHub sources, against recorded-shape fixtures."""
import httpx

from kinetixai.sources import github, huggingface

HF_MODELS = [
    {"id": "meta-llama/Llama-3.1-8B", "downloads": 5_400_000, "likes": 1200,
     "pipeline_tag": "text-generation", "library_name": "transformers",
     "tags": ["transformers", "license:llama3.1", "arxiv:2407.21783"], "lastModified": "2024-09-25T00:00:00.000Z"},
    {"modelId": "someone/no-license", "downloads": 3, "likes": 0, "tags": []},
]
HF_DATASETS = [
    {"id": "HuggingFaceFW/fineweb", "downloads": 200_000, "likes": 2000,
     "tags": ["task_categories:text-generation", "license:odc-by"]},
]
GH_SEARCH = {"total_count": 2, "items": [
    {"full_name": "Dao-AILab/flash-attention", "html_url": "https://github.com/Dao-AILab/flash-attention",
     "stargazers_count": 25000, "description": "Fast and memory-efficient exact attention", "language": "Python",
     "license": {"spdx_id": "BSD-3-Clause", "name": "BSD 3-Clause"}, "pushed_at": "2026-11-01T00:00:00Z"},
    {"full_name": "x/y", "html_url": "https://github.com/x/y", "stargazers_count": 3,
     "license": {"spdx_id": "NOASSERTION", "name": "Other"}},
]}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_models_search_params_and_parse():
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=HF_MODELS)
    with _client(handler) as c:
        items = huggingface.search_models("llama", 5, c, sort="likes", task="text-generation")
    q = seen[0].url.params
    assert seen[0].url.path == "/api/models"
    assert q["search"] == "llama" and q["sort"] == "likes" and q["direction"] == "-1"
    assert q["limit"] == "5" and q["pipeline_tag"] == "text-generation"
    assert items[0].license == "llama3.1" and items[0].task == "text-generation"
    assert items[0].url == "https://huggingface.co/meta-llama/Llama-3.1-8B"
    assert items[1].id == "someone/no-license" and items[1].license is None


def test_models_for_arxiv_uses_filter():
    seen = []
    with _client(lambda r: (seen.append(r), httpx.Response(200, json=[]))[1]) as c:
        huggingface.models_for_arxiv("2407.21783", 3, c)
    assert seen[0].url.params["filter"] == "arxiv:2407.21783"


def test_datasets_parse_task_and_license_and_url():
    with _client(lambda r: httpx.Response(200, json=HF_DATASETS)) as c:
        items = huggingface.search_datasets("fineweb", 3, c)
    d = items[0]
    assert d.kind == "dataset" and d.license == "odc-by" and d.task == "text-generation"
    assert d.url == "https://huggingface.co/datasets/HuggingFaceFW/fineweb"


def test_hf_paper_404_is_none():
    with _client(lambda r: httpx.Response(404)) as c:
        assert huggingface.paper("2101.00001", c) is None


def test_hf_token_sent_when_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    seen = []
    with _client(lambda r: (seen.append(r), httpx.Response(200, json=[]))[1]) as c:
        huggingface.search_models("x", 1, c)
    assert seen[0].headers["authorization"] == "Bearer hf_test"


def test_github_search_for_arxiv_query_and_parse():
    seen = []
    with _client(lambda r: (seen.append(r), httpx.Response(200, json=GH_SEARCH))[1]) as c:
        repos = github.search_for_arxiv("2205.14135", 5, c)
    q = seen[0].url.params
    assert q["q"] == "2205.14135 in:name,description,readme,topics"
    assert q["sort"] == "stars" and q["order"] == "desc"
    assert seen[0].headers["accept"] == "application/vnd.github+json"
    assert "authorization" not in seen[0].headers
    assert repos[0].name == "Dao-AILab/flash-attention" and repos[0].stars == 25000
    assert repos[0].license == "BSD-3-Clause" and repos[1].license == "Other"


def test_parse_repo_url():
    assert github.parse_repo_url("https://github.com/Dao-AILab/flash-attention") == "Dao-AILab/flash-attention"
    assert github.parse_repo_url("https://github.com/a/b.git/tree/main") == "a/b"
    assert github.parse_repo_url("https://gitlab.com/a/b") is None
    assert github.parse_repo_url("") is None
