# kx: the KinetixAI command line

[![PyPI](https://img.shields.io/pypi/v/kinetixai)](https://pypi.org/project/kinetixai/) [![ci](https://github.com/kinetixai-ai/kx/actions/workflows/ci.yml/badge.svg)](https://github.com/kinetixai-ai/kx/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

From a paper to its **code**, **models** and **datasets**, a **local paper
library** with BibTeX/APA export, **private Q&A over your own PDFs** (BM25 in
pure Python, plus Ollama when it is running), and an **MCP server** so
open-source agents can use all of it. macOS, Linux and Windows, Python 3.10+.

## Install

| Where | Command |
|---|---|
| macOS / Linux | `curl -fsSL https://kinetixai.ai/install.sh \| sh` |
| Windows (PowerShell) | `irm https://kinetixai.ai/install.ps1 \| iex` |
| Any OS, PyPI | `pipx install "kinetixai[rag,mcp]"` |
| From source | `pipx install "git+https://github.com/kinetixai-ai/kx.git"` |
| No Python | standalone `kx` binaries: coming with the first release |

The installers never use sudo/Administrator, check for Python 3.10+, verify the
SHA-256 of the wheel they download, keep it in `~/.kinetixai/wheels` (so
`pipx reinstall` / `pipx upgrade` keep working), and use pipx if present
(otherwise a venv in `~/.kinetixai`). They never install from PyPI unless you set
`KX_SOURCE=pypi`: kx is not published there yet, and an unclaimed name must not
be installed by accident. Options: `KX_EXTRAS` (default `rag,mcp`), `KX_SOURCE=site|pypi|git`,
`KX_WHEEL=<local wheel>`, `KX_NO_PIPX=1`, and on Windows `KX_NO_MODIFY_PATH=1`.

Extras: `rag` (pypdf, to read PDFs), `mcp` (the agent server), `keyring` (token
in the OS keychain), `all`. Add one later with `pipx inject kinetixai pypdf`.

Then run `kx doctor`.

## Use

```bash
# papers
kx search "speculative decoding" -n 5
kx get 1706.03762                    # metadata + abstract; --pdf to download, --bibtex
kx cite 1706.03762 --format apa
kx scholar "Tri Dao"

# paper -> code, models, datasets (GitHub + Hugging Face public APIs)
kx code 2205.14135                   # official repo first, then GitHub by stars
kx models 2307.09288                 # models that cite the paper
kx models whisper --task automatic-speech-recognition --sort likes
kx datasets "code generation"

# local library (JSON in your data dir; works offline)
kx library add 2205.14135 2106.09685 --pdf --tag attention
kx library add ~/Downloads/notes.pdf
kx library list
kx library export -o refs.bib        # or --format apa | json

# private RAG over the library (+ any folder)
kx rag index --path ~/papers
kx rag ask "How does FlashAttention reduce HBM traffic?"

# agents (MCP over stdio)
kx mcp --config goose                # also: claude-desktop, cline, continue, openhands, open-webui, cursor
kx mcp                               # run the server

# account (optional; only kx ask uses it)
kx login                             # paste a token from https://kinetixai.ai/account/tokens/
kx whoami
kx ask "What is RLVR?"
kx logout                            # revokes the token and forgets it
```

Every command takes `--json`, placed after the command (`kx search "rope" --json`,
not `kx --json search`): ASCII-safe JSON on stdout, errors on stderr, exit code 1
on failure, 2 on bad usage.

## Local RAG and privacy

`kx rag index` chunks your library (titles, abstracts and PDF text) and any
`--path` folders (PDF, Markdown, text), and stores the index in your data
directory. Retrieval is BM25 in pure Python. If Ollama is running at
`localhost:11434` with `nomic-embed-text` pulled, kx adds embeddings (hybrid
retrieval, cached between runs); `kx rag ask` then answers with your Ollama
model (`kx config set ollama_model qwen2.5`) and cites passages as `[n]`.
Without Ollama you get the best passages with sources. With a local Ollama,
nothing leaves the machine; if `ollama_url`/`OLLAMA_HOST` points elsewhere, kx
says so on every answer.

## MCP tools

`search_papers`, `get_paper`, `find_code`, `find_models`, `find_datasets`,
`library_search`, `rag_ask`. Works with MCP SDK 1.x and 2.x. For Open WebUI,
expose it through mcpo: `uvx mcpo --port 8000 -- kx mcp`.

## Configure

```bash
kx config show
kx config set endpoint https://kinetixai.ai
kx config set ollama_model llama3.1
kx config set embed_model nomic-embed-text
kx config set token_store file       # skip the OS keyring
```

Environment: `KINETIXAI_ENDPOINT`, `KX_TOKEN` (CI), `OLLAMA_HOST`,
`GITHUB_TOKEN` / `HF_TOKEN` (optional, higher rate limits, never stored),
`KX_CONFIG_DIR` / `KX_DATA_DIR` (override the platformdirs locations).

## Token handling

Tokens come from `/account/tokens/`, are shown once, and the server keeps only
a SHA-256 hash. kx stores the token per endpoint in the OS keyring when the
`keyring` extra has a working backend, otherwise in `credentials.json` in the
config dir, created owner-only (0600) on macOS/Linux. kx never prints it,
refuses to send it over plain http except to localhost, and only sends it to
the configured KinetixAI endpoint. A token can call `/api/auth/me`, `/api/ask`,
`/api/rag/*` and `/api/research/*`; it cannot manage tokens, payments or the
account.

## Develop

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q                                      # offline; every HTTP source is mocked
```

Standalone binary: `pip install ".[rag,mcp,build]"` then
`pyinstaller --onefile --name kx --collect-submodules kinetixai --collect-all mcp --copy-metadata mcp packaging/kx_entry.py`.
`python packaging/mcp_smoke.py dist/kx mcp` does a real MCP handshake against it.

## Contributing and support

Bugs and ideas: [open an issue](https://github.com/kinetixai-ai/kx/issues).
Pull requests are welcome; please run `pytest -q` first. Security problems:
see [SECURITY.md](SECURITY.md), not public issues.

## Releasing

Bump the version in `pyproject.toml` and `src/kinetixai/__init__.py`, merge,
then `git tag vX.Y.Z && git push origin vX.Y.Z`. The release workflow
publishes to PyPI (trusted publishing, no stored token), builds binaries for
Linux, macOS and Windows and attaches them with `SHA256SUMS` to the GitHub
release.

## License

MIT. Copyright (c) 2026 Rohit S Patel / Narmadya Innovations.
