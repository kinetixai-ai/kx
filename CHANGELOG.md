# Changelog

## 0.3.0 (unreleased)
- `--copy-to` (and `kx config set primer_copy_to`) copies finished primers to a
  second folder — a notes directory, a shared drive, a repo. A destination that
  cannot be written is a warning, never a lost build.
- `kx config set primer_engine` now rejects an unknown engine and lists the
  valid ones instead of accepting it and failing later.
- Citation filtering matched ambiguous markers as substrings, so "iter" (for
  ITER, the fusion reactor) matched "iterative" and dropped real CUDA papers.
  Markers are whole-word now, and a paper that matches the topic strongly
  survives one stray marker.
- Full Primers reference in docs/PRIMERS.md.
- 22 engines, from one registry: self-hosted and open source (Ollama, vLLM,
  llama.cpp, LM Studio, Jan, LocalAI, TGI, text-generation-webui, llamafile),
  hosted (Claude, OpenAI, Gemini, Groq, OpenRouter, Together, DeepSeek,
  Mistral, Fireworks, Cerebras, xAI), plus `mcp` and `outline`. Endpoints move
  with `--api-base` or a `*_BASE_URL` variable.
- `--engine` is now optional: kx prefers a local server that is running, then
  a provider whose key is set, and reports what it chose. `kx primers engines`
  lists them all with their status. `kx config set primer_engine/primer_model`
  makes the choice stick.
- `kx mcp install` covers Gemini CLI, Codex, VS Code and Zed as well as Claude
  Code, Claude Desktop, Cursor, Goose, Cline, Continue, OpenHands and Open WebUI.
- Primers are now generated page by page (`--mode iterative`, the default):
  each page gets its own LLM call carrying its archetype and a density quota,
  which is what lets a 7B local model fill eight dense pages instead of
  emitting one paragraph per page. `--mode single` keeps the old one-call
  behaviour for large models.
- Pages follow the TriFleetRCA arc: overview, data, two deep dives,
  guardrails, implementation, evaluation, takeaways. Any `--pages N` maps
  onto it. A page that comes back sparse goes through an expansion pass
  before rendering; `--min-sections-per-page` sets the floor.
- New blocks: inline SVG bar charts, hierarchy stacks, numbered step lists,
  and the pink scalloped comic cloud. Pastel bento cards throughout.
- A printable-height model fits each page before rendering. The page box
  clips what does not fit, so content that used to vanish off the bottom is
  now trimmed or dropped deliberately, and every drop is reported.
- Citations are filtered for domain relevance: searching "GPU & CUDA: Kernel
  Fusion" no longer returns tokamak fusion or image-fusion papers. Every
  dropped source is explained; `--relevance 0` disables it.
- Default watermark is "Narmadya Innovations", rendered tilted at the top
  and bottom left. Page badges count the references page, so an 8-page
  primer reads 1/9 through 9/9 rather than 1/8 then 5/5.
- Engine aliases (`vllm`, `claude`) and `--api-base`. New `--engine mcp`
  delegates generation to the host agent via MCP sampling, so no API key is
  needed on the user's machine.
- `kx mcp serve` and `kx mcp install <client>`; the server exposes
  build_primer, convert_document and search_verified_papers alongside the
  research tools. `kx mcp` with no subcommand still runs the server.
- New: `kx primers build` — automated "Story to Silicon" Primer generation.
  Topic or source (.pptx / .pdf / .md, GitHub repo URL, arXiv id/DOI) in;
  styled A4 PDF, 300 DPI PNG page cards, editable .pptx with speaker notes,
  and a Paper Lab JSON bundle (visualization / architecture / mnemonic /
  cheatsheet / flashcards tabs + bibtex + rag_chunks) out.
  Engines: `local-vllm` (OpenAI-compatible), `ollama`, `openai`, `anthropic`,
  `gemini`, or `outline` (deterministic no-LLM draft). Persona modules:
  student, phd, researcher, professor, professional, lab, ai-security.
  Citations come only from fetched arXiv/Semantic Scholar/OpenAlex metadata;
  `--verify-citations` re-resolves and drops anything that fails.
- New: `kx convert` — turn an existing deck, paper or repo into the
  TriFleetRCA 8-page format (`--style trifleet-8page`).
- New: `kx primers autobuild --manifest primers_catalog.yaml --parallel N` —
  batch-build the whole Primer catalog; a ready manifest with all 18
  KinetixAI tracks ships in the repo.
- A4 print engine: Jinja2 template + WeasyPrint (default) or headless
  Chromium via `KX_PDF_ENGINE=playwright`; 380–460-word page budget with a
  repair pass and deterministic compaction so pages never overflow 297mm.
- New optional extra: `pip install 'kinetixai[primers]'`.

## 0.2.1
- Project moved to its own public repository, github.com/kinetixai-ai/kx.
- PyPI page links to source, issues and docs.

## 0.2.0
- First PyPI release: search, get, scholar, cite, code, models, datasets,
  library, local RAG (BM25 + optional Ollama), MCP server, login with API
  tokens, doctor. macOS, Linux and Windows.
