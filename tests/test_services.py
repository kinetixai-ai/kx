"""find_code / find_models / find_datasets through the CLI, every network call mocked."""
import json

import httpx
from typer.testing import CliRunner

from kinetixai import cli, services

runner = CliRunner()

HF_PAPER = {"id": "2205.14135", "title": "FlashAttention", "upvotes": 12,
            "githubRepo": "https://github.com/Dao-AILab/flash-attention", "githubStars": 20000}
GH = {"items": [
    {"full_name": "lucidrains/x", "html_url": "https://github.com/lucidrains/x", "stargazers_count": 900},
    {"full_name": "Dao-AILab/flash-attention", "html_url": "https://github.com/Dao-AILab/flash-attention",
     "stargazers_count": 25000, "language": "Python"},
]}


def router(req):
    host, path = req.url.host, req.url.path
    if host == "huggingface.co" and path.startswith("/api/papers/"):
        return httpx.Response(200, json=HF_PAPER)
    if host == "huggingface.co" and path == "/api/models":
        return httpx.Response(200, json=[{"id": "org/m", "downloads": 10, "likes": 1, "tags": ["license:mit"]}])
    if host == "huggingface.co" and path == "/api/datasets":
        return httpx.Response(200, json=[{"id": "org/d", "downloads": 5, "likes": 2, "tags": []}])
    if host == "api.github.com":
        return httpx.Response(200, json=GH)
    return httpx.Response(404)


def test_find_code_official_first_and_deduped(mock_http):
    mock_http(router)
    data = services.find_code("https://arxiv.org/abs/2205.14135v2")
    assert data["arxiv_id"] == "2205.14135"
    names = [r["name"] for r in data["repos"]]
    assert names == ["Dao-AILab/flash-attention", "lucidrains/x"]
    assert data["repos"][0]["official"] is True
    assert data["repos"][0]["stars"] == 25000  # fresher GitHub count wins
    assert data["paper"]["title"] == "FlashAttention"


def test_find_code_survives_hf_outage(mock_http):
    def handler(req):
        if req.url.host == "huggingface.co":
            return httpx.Response(503)
        return router(req)
    mock_http(handler)
    data = services.find_code("2205.14135")
    assert [r["name"] for r in data["repos"]][0] == "Dao-AILab/flash-attention"
    assert any("Hugging Face" in w for w in data["warnings"])


def test_find_code_free_text_uses_github_search(mock_http):
    seen = mock_http(router)
    services.find_code("flash attention")
    assert all(r.url.host == "api.github.com" for r in seen)
    assert seen[0].url.params["q"] == "flash attention"


def test_cli_code_json(mock_http):
    mock_http(router)
    res = runner.invoke(cli.app, ["code", "2205.14135", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["repos"][0]["official"] is True


def test_cli_code_all_sources_down_exits_1(mock_http):
    mock_http(lambda r: httpx.Response(503))
    res = runner.invoke(cli.app, ["code", "2205.14135"])
    assert res.exit_code == 1


def test_cli_models_by_arxiv_id_filters(mock_http):
    seen = mock_http(router)
    res = runner.invoke(cli.app, ["models", "arXiv:2205.14135", "--json"])
    assert res.exit_code == 0
    assert seen[0].url.params["filter"] == "arxiv:2205.14135"
    assert json.loads(res.stdout)["models"][0]["license"] == "mit"


def test_cli_models_table(mock_http):
    mock_http(router)
    res = runner.invoke(cli.app, ["models", "bert"])
    assert res.exit_code == 0 and "org/m" in res.stdout


def test_cli_datasets_json(mock_http):
    mock_http(router)
    res = runner.invoke(cli.app, ["datasets", "squad", "--json"])
    assert res.exit_code == 0
    assert json.loads(res.stdout)["datasets"][0]["url"] == "https://huggingface.co/datasets/org/d"


def test_cli_models_rate_limited_is_friendly(mock_http):
    mock_http(lambda r: httpx.Response(429, headers={"retry-after": "600"}))
    res = runner.invoke(cli.app, ["models", "bert"])
    assert res.exit_code == 1
    assert "rate limit" in res.stderr
