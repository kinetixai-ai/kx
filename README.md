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

# Primers: topic or source -> styled PDF / PNG / PPTX / JSON  (pip install 'kinetixai[primers]')
kx primers engines                   # what can write a primer, and what is ready
kx primers build --topic "Types of RAG"            # picks an engine for you
kx primers build --topic "GPU & CUDA: Kernel Fusion" \
  --persona student,professional --pages 8 --pdfs 1 --ppts 1 --images 8 --json-bundles 1 \
  --engine claude --verify-citations --out-dir ./dist/primers/gpu
kx convert ./TriParseGPU_Deck.pptx --style trifleet-8page --pages 8 -o ./TriParseGPU_8p.pdf
kx primers autobuild --manifest ./primers_catalog.yaml --parallel 4   # all 18 tracks
kx primers build --topic "Types of RAG" --copy-to ~/Notes/primers     # second home

# account (optional; only kx ask uses it)
kx login                             # paste a token from https://kinetixai.ai/account/tokens/
kx whoami
kx ask "What is RLVR?"
kx logout                            # revokes the token and forgets it
```

Every command takes `--json`, placed after the command (`kx search "rope" --json`,
not `kx --json search`): ASCII-safe JSON on stdout, errors on stderr, exit code 1
on failure, 2 on bad usage.

## Primers ("Story to Silicon")

### Picking an engine

`kx primers engines` lists every backend and marks the ones ready to use.
With no `--engine`, kx picks one: a local server if it finds one running
(nothing per page), otherwise a provider whose API key is set. It says which
it chose and why.

| Where it runs | Engines |
|---|---|
| Your machine, open source | `ollama`, `vllm`, `llamacpp`, `lmstudio`, `jan`, `localai`, `tgi`, `textgen`, `llamafile` |
| Hosted | `claude`, `openai`, `gemini`, `groq`, `openrouter`, `together`, `deepseek`, `mistral`, `fireworks`, `cerebras`, `xai` |
| No key at all | `mcp` (the agent calling kx writes the pages), `outline` (deterministic draft) |

Any endpoint can be moved with `--api-base` or the provider's `*_BASE_URL`
variable, so a changed URL does not need a new release. Set a default once and
drop the flags:

```bash
kx config set primer_engine claude      # or ollama, groq, lmstudio, mcp ...
kx config set primer_model  claude-sonnet-4-5
kx primers build --topic "Types of RAG"
```

Ollama's 4096-token context is too small for a full page, so kx uses Ollama's
native API and asks for 32768; `--num-ctx` or `KX_OLLAMA_NUM_CTX` changes it.

`--copy-to` (or `kx config set primer_copy_to <folder>`) puts a second copy of
the finished files wherever you keep things — a notes folder, a shared drive, a
repo. If that folder is unreachable the build still succeeds and says so.

Full reference, including every option and a quick start for each kind of
reader: [docs/PRIMERS.md](docs/PRIMERS.md).

### From an agent instead of the terminal

`kx mcp serve` exposes `build_primer`, `convert_document`,
`search_verified_papers` and the research tools over MCP, so Claude Code,
Claude Desktop, Cursor, Gemini CLI, Codex, VS Code, Zed, Goose, Cline,
Continue and OpenHands can drive kx in plain language. `kx mcp install
<client>` prints the configuration for each.

```bash
claude mcp add kx -- kx mcp serve      # then just ask for a primer
```

With `engine: "mcp"` the host's own model writes the pages, so no API key and
no local model are needed. Sampling is optional in the MCP protocol, so a
client that lacks it gets a clear message naming the alternatives.


`kx primers build` researches a topic (federated arXiv + Semantic Scholar +
OpenAlex), optionally ingests a source (`--source` takes a .pptx, .pdf, .md,
a GitHub repo URL, or an arXiv id/DOI), synthesizes a page-by-page layout
model with your chosen engine, and compiles it with a deterministic HTML/CSS
A4 print engine (WeasyPrint by default, `KX_PDF_ENGINE=playwright` for
headless Chromium). Outputs per run: editable self-contained HTML, vector
PDF, 300 DPI PNG page cards, an editable .pptx with speaker notes, and a
Paper Lab JSON bundle (visualization, architecture, mnemonic, cheatsheet and
flashcards tabs, plus bibtex and rag_chunks you can feed straight into
`kx rag`).

Engines: `local-vllm` (any OpenAI-compatible endpoint; set `KX_VLLM_URL`),
`ollama`, `openai`, `anthropic`, `gemini` — or `outline`, a no-LLM mode that
produces clearly labelled draft scaffolding from the source text alone.
Personas (`--persona`): student, phd, researcher, professor, professional,
lab, ai-security — each adds its own callouts (interview traps, related-work
tables, misconception radar, Three Honest Columns benchmarks, reproducibility
blocks, threat matrices).

Honesty guarantees: citations only ever come from fetched metadata, never
from the model; `--verify-citations` re-resolves each one and drops failures;
pages carry a 380–460-word budget with an automatic repair pass and
deterministic compaction, so nothing overflows the 297mm page.

`kx primers autobuild --manifest primers_catalog.yaml --parallel 4` builds
the whole catalog; the repo ships a manifest with all 18 KinetixAI tracks.

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
