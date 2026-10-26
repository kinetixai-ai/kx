# kx — the KinetixAI command line

Search papers across **arXiv**, **Semantic Scholar** and **OpenAlex**, look up
researchers by **ORCID**, and **ask KinetixAI** grounded questions — all from your
terminal. Runs on Windows, macOS and Linux (Python 3.10+).

## Install

```bash
pip install kinetixai        # gives you the `kx` command
# or, isolated:
pipx install kinetixai
```

## Use

```bash
kx search "flash attention" -n 10
kx search "mixture of experts" --source arxiv --json
kx scholar "Tri Dao"
kx scholar 0000-0002-1825-0097
kx get 2205.14135 --pdf -o ./papers
kx ask "What is the difference between MoE and Mixture of Agents?"
```

Global: `--json` for scriptable output, `--limit/-n`, `--open` (search).

## Configure

```bash
kx config show
kx config set endpoint https://kinetixai.ai
kx config set limit 15
```

`kx ask` talks to the KinetixAI API at the configured `endpoint`. Search and
`scholar` use public APIs and work without an account.

## Sources

| Command | Backs onto |
|---|---|
| `search` | arXiv, Semantic Scholar, OpenAlex (merged + de-duplicated) |
| `scholar` | ORCID public API |
| `get` | arXiv (by id/URL), Semantic Scholar (by DOI) |
| `ask` | KinetixAI API |

Google Scholar has no public API and is not used; Semantic Scholar and OpenAlex
cover the same ground and are allowed.

## Develop

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```
