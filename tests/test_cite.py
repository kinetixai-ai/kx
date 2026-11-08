import json

from typer.testing import CliRunner

from kinetixai import cli, cite, library
from kinetixai.models import Paper

runner = CliRunner()
P = {"title": "Attention Is All You Need", "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
     "year": 2017, "venue": "NeurIPS", "arxiv_id": "1706.03762v7", "doi": None}


def test_bibtex_article_with_eprint():
    b = cite.bibtex(P)
    assert b.startswith("@article{vaswani2017attention,")
    assert "journal = {NeurIPS}" in b and "eprint = {1706.03762}" in b
    assert "author = {Ashish Vaswani and Noam Shazeer and Niki Parmar}" in b


def test_bibtex_arxiv_only_is_misc_and_escapes():
    b = cite.bibtex({"title": "A & B 100%", "authors": ["Ö Ünal"], "year": 2024, "arxiv_id": "2401.00001"})
    assert b.startswith("@misc{unal2024") and "A \\& B 100\\%" in b


def test_apa_formats():
    a = cite.apa(P)
    assert a.startswith("Vaswani, A., Shazeer, N., & Parmar, N. (2017). Attention Is All You Need.")
    single = cite.apa({"title": "X", "authors": ["Dao, Tri"], "year": 2022, "arxiv_id": "2205.14135"})
    assert single == "Dao, T. (2022). X. arXiv. https://doi.org/10.48550/arXiv.2205.14135"
    assert cite.apa({"title": "Untitled"}).startswith("Untitled. (n.d.).")


def test_apa_particles_and_many_authors():
    assert cite._split_name("Ludwig van Beethoven") == ("van Beethoven", ["Ludwig"])
    many = {"title": "T", "authors": [f"A{i} B{i}" for i in range(25)], "year": 2020}
    assert ". . . B24, A." in cite.apa(many)


def test_cli_cite_from_library_offline():
    library.add(Paper(**{k: v for k, v in P.items()}))
    res = runner.invoke(cli.app, ["cite", "1706.03762", "--format", "apa", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout)["citation"].startswith("Vaswani, A.")
    assert runner.invoke(cli.app, ["cite", "1", "-f", "mla"]).exit_code == 2
