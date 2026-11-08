# Changelog

## 0.3.0 (unreleased)
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
