import json
from pathlib import Path

from typer.testing import CliRunner

from conftest import make_pdf
from kinetixai import cli, config
from kinetixai.rag import index as rag
from kinetixai.rag import ollama
from kinetixai.rag.bm25 import BM25
from kinetixai.rag.text import chunk, tokens

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "papers.json"


def _seed():
    assert runner.invoke(cli.app, ["library", "add", "--from-file", str(FIXTURE)]).exit_code == 0


def test_bm25_ranks_the_relevant_doc_first():
    docs = [tokens(t) for t in ["the cat sat on the mat", "gpu memory bandwidth for attention kernels",
                                "low rank adaptation of language models"]]
    s = BM25(docs).scores(tokens("attention memory"))
    assert s.index(max(s)) == 1 and s[0] == 0


def test_chunking_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(400)]
    parts = chunk(" ".join(words), size=160, overlap=40)
    assert len(parts) == 3 and parts[1].split()[0] == "w120" and parts[-1].split()[-1] == "w399"
    assert chunk("") == [] and chunk("one two") == ["one two"]


def test_index_and_ask_offline_without_ollama():
    _seed()
    res = runner.invoke(cli.app, ["rag", "index", "--json"])
    assert res.exit_code == 0, res.output
    summary = json.loads(res.stdout)
    assert summary["documents"] == 3 and summary["embeddings"] is None
    res = runner.invoke(cli.app, ["rag", "ask", "what does LoRA freeze?", "-k", "2", "--json"])
    out = json.loads(res.stdout)
    assert out["passages"][0]["key"] == "arxiv:2106.09685"
    assert out["answer"] is None and out["retrieval"] == "bm25" and out["local"] is True
    assert "not running" in out["note"]
    human = runner.invoke(cli.app, ["rag", "ask", "tiling between HBM and SRAM"])
    assert "FlashAttention" in human.stdout and "nothing was sent anywhere" in human.stdout


def test_ask_before_index_is_a_clear_error():
    res = runner.invoke(cli.app, ["rag", "ask", "anything"])
    assert res.exit_code == 1 and "kx rag index" in res.stderr


def test_index_includes_pdfs_and_extra_paths(tmp_path):
    notes = tmp_path / "notes"
    notes.mkdir()
    make_pdf(notes / "paper.pdf", "Mixture of experts routes each token to two experts")
    (notes / "idea.md").write_text("# Idea\nSpeculative decoding with a draft model.", encoding="utf-8")
    (notes / "ignore.bin").write_bytes(b"\x00\x01")
    res = runner.invoke(cli.app, ["rag", "index", "--path", str(notes), "--json"])
    s = json.loads(res.stdout)
    assert s["documents"] == 2 and s["paths"] == [str(notes.resolve())]
    out = rag.ask("which experts does each token go to", llm=False)
    assert out["passages"][0]["title"] == "paper.pdf" and out["passages"][0]["page"] == 1
    # the path is remembered on the next rebuild
    assert rag.build()["documents"] == 2
    st = json.loads(runner.invoke(cli.app, ["rag", "status", "--json"]).stdout)
    assert st["index"]["chunks"] >= 2


class FakeOllama:
    """Stands in for a local Ollama: 3-dim embeddings keyed on a few words."""
    def __init__(self, monkeypatch, url_models=("llama3.1:latest", "nomic-embed-text:latest")):
        self.chats = []
        monkeypatch.setattr(ollama, "models", lambda url, client=None: list(url_models))
        monkeypatch.setattr(ollama, "embed", self.embed)
        monkeypatch.setattr(ollama, "chat", self.chat)

    @staticmethod
    def embed(url, model, texts, client):
        def v(t):
            t = t.lower()
            return [float("rank" in t or "freeze" in t), float("attention" in t), 0.1]
        return [v(t) for t in texts]

    def chat(self, url, model, messages, client):
        self.chats.append((url, model, messages))
        return "LoRA freezes the pretrained weights [1]."


def test_hybrid_retrieval_and_local_answer_with_ollama(monkeypatch):
    fake = FakeOllama(monkeypatch)
    _seed()
    s = rag.build()
    assert s["embeddings"] == "nomic-embed-text"
    assert all(c.get("v") for c in rag.load_index()["chunks"])
    res = runner.invoke(cli.app, ["rag", "ask", "what does it freeze", "--json"])
    out = json.loads(res.stdout)
    assert out["retrieval"] == "hybrid" and out["model"] == "llama3.1"
    assert out["answer"].endswith("[1].") and out["local"] is True
    url, model, messages = fake.chats[0]
    assert url == "http://localhost:11434" and "[1]" in messages[1]["content"]


def test_embeddings_are_cached_between_builds(monkeypatch):
    FakeOllama(monkeypatch)
    calls = []
    real = FakeOllama.embed
    monkeypatch.setattr(ollama, "embed", lambda *a: (calls.append(len(a[2])), real(*a))[1])
    _seed()
    rag.build()
    rag.build()
    assert calls == [3]  # second build re-used every vector


def test_missing_model_falls_back_to_passages(monkeypatch):
    FakeOllama(monkeypatch, url_models=("nomic-embed-text:latest",))
    _seed()
    rag.build()
    out = rag.ask("attention", k=1)
    assert out["answer"] is None and "ollama pull llama3.1" in out["note"]


def test_remote_ollama_is_flagged_not_private(monkeypatch):
    FakeOllama(monkeypatch)
    config.set_value("ollama_url", "http://gpu-box.lan:11434")
    _seed()
    rag.build()
    out = rag.ask("attention", k=1)
    assert out["local"] is False
    human = runner.invoke(cli.app, ["rag", "ask", "attention"])
    assert "gpu-box.lan" in human.stdout


def test_require_embeddings_without_ollama_errors():
    _seed()
    res = runner.invoke(cli.app, ["rag", "index", "--embeddings"])
    assert res.exit_code == 1 and "Ollama is not running" in res.stderr


def test_ollama_host_env_is_normalised(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0")
    assert config.load()["ollama_url"] == "http://localhost:11434"
    assert config.normalise_ollama("10.0.0.5:9999") == "http://10.0.0.5:9999"
    assert ollama.is_local("http://127.0.0.1:11434") and not ollama.is_local("http://10.0.0.5:1")
