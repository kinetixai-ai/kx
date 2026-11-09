# Primers — "Story to Silicon"

Turn a topic, a deck, a paper or a repo into a publication-grade A4 primer:
a styled PDF, 300 DPI page cards for slides and social, an editable
PowerPoint with speaker notes, and a JSON bundle of flashcards, cheatsheets
and diagrams.

```bash
pip install 'kinetixai[primers]'
kx primers build --topic "Types of RAG"
```

That is the whole thing. kx finds an engine, researches the topic, writes
eight pages, checks every page fits an A4 sheet, and compiles the PDF.

---

## Contents

- [What you get](#what-you-get)
- [Quick start by persona](#quick-start-by-persona)
- [Choosing an engine](#choosing-an-engine)
- [How a primer is built](#how-a-primer-is-built)
- [The eight pages](#the-eight-pages)
- [Personas](#personas)
- [Converting something you already have](#converting-something-you-already-have)
- [Building a whole catalog](#building-a-whole-catalog)
- [From an agent instead of the terminal](#from-an-agent-instead-of-the-terminal)
- [Where the files go](#where-the-files-go)
- [Every option](#every-option)
- [What kx will not do](#what-kx-will-not-do)
- [Troubleshooting](#troubleshooting)

---

## What you get

| Flag | Output | Good for |
|---|---|---|
| `--pdfs 1` | Vector A4 PDF | Reading, printing, sending |
| `--images 8` | 300 DPI PNG per page | LinkedIn, X, lecture slides, posters |
| `--ppts 1` | Editable `.pptx` with speaker notes | Teaching, talks |
| `--json-bundles 1` | Flashcards, cheatsheet, diagrams, BibTeX, RAG chunks | Study tools, Paper Lab, your own pipeline |
| always | Self-contained editable HTML | Hand-tweak a line, reprint from the browser |

Every page carries numbered sections with coloured pills, pastel card
grids, comparison tables, SVG pipeline diagrams and bar charts, a terminal
block, a comic-cloud callout, a Key Takeaway bar, and your watermark.

---

## Quick start by persona

**Student** — intuition before mathematics, interview traps, flashcards.

```bash
kx primers build --topic "Advanced DSA: Graphs & Heaps" --persona student \
  --pages 8 --pdfs 1 --images 8 --json-bundles 1 --out-dir ./dist/dsa

jq '.tabs.flashcards' ./dist/dsa/*.json        # drill before the exam
kx rag index --path ./dist/dsa && kx rag ask "When does Dijkstra fail?"
```

**Teacher or professor** — a 45-minute lecture, a handout, and the three
misconceptions students actually have.

```bash
kx primers build --topic "Classical ML: Bias-Variance" --persona professor,student \
  --pages 8 --pdfs 1 --ppts 1 --images 8 --json-bundles 1 --out-dir ./dist/lecture
```

Open the `.pptx`, press *View → Notes*: every slide has speaker notes.

**PhD student or researcher** — a related-work table built only from
verified sources, and citations that resolve.

```bash
kx primers build --topic "Sparse Attention Mechanisms" --persona phd,researcher \
  --pages 8 --verify-citations --json-bundles 1 --out-dir ./dist/sparse
jq -r '.bibtex' ./dist/sparse/*.json > citations.bib
```

**Industry engineer or SRE** — honest benchmarks and an observability
recipe, from a deck you already have.

```bash
kx convert ./TriParseGPU_Deck.pptx --persona professional,lab \
  --pages 8 --images 8 -o ./TriParseGPU_8_Pages.pdf
```

**AI security** — a threat matrix with the guarded-pipeline counterpart.

```bash
kx primers build --topic "AI Security & Guarded Runbooks" --persona ai-security \
  --pages 8 --pdfs 1 --out-dir ./dist/ai-sec
```

**A whole department** — every track in one run.

```bash
kx primers autobuild --manifest ./primers_catalog.yaml --parallel 4
```

---

## Choosing an engine

```bash
kx primers engines        # the full list, and which are ready right now
```

With no `--engine`, kx prefers a local server that is actually running
(nothing per page), then a provider whose API key is set, and tells you
which it chose and why.

| Where it runs | Engines | Needs |
|---|---|---|
| Your machine, open source | `ollama`, `vllm`, `llamacpp`, `lmstudio`, `jan`, `localai`, `tgi`, `textgen`, `llamafile` | the server running |
| Hosted | `claude`, `openai`, `gemini`, `groq`, `openrouter`, `together`, `deepseek`, `mistral`, `fireworks`, `cerebras`, `xai` | that provider's API key |
| No key at all | `mcp` — the agent calling kx writes the pages | an MCP host |
| No model at all | `outline` — a deterministic draft from your source | nothing |

Set a default once and stop typing flags:

```bash
kx config set primer_engine claude          # or ollama, groq, lmstudio, mcp ...
kx config set primer_model  claude-sonnet-4-5
kx config set primer_copy_to ~/Notes/primers
kx primers build --topic "Types of RAG"
```

Moved endpoint, self-hosted gateway, or a proxy? `--api-base` overrides any
of them, as does the provider's own `*_BASE_URL` variable.

```bash
kx primers build --topic "NLP & NLU" --engine vllm \
  --api-base http://gpu-box.local:8000/v1 --model Qwen/Qwen2.5-14B-Instruct
```

### Local models: give them room

Ollama defaults to a 4096-token context, which is too small for a full
page, so kx uses Ollama's native API and asks for 32768. Change it with
`--num-ctx` or `KX_OLLAMA_NUM_CTX`. A 7B model works; a 14B or larger one
follows the layout rules more reliably.

---

## How a primer is built

1. **Ingest** — read `--source` if given (`.pptx`, `.pdf`, `.md`, a GitHub
   URL, an arXiv id or DOI).
2. **Research** — search arXiv, Semantic Scholar and OpenAlex, then keep
   only sources that are actually about your topic. Searching *GPU & CUDA:
   Kernel Fusion* will not hand you tokamak fusion or image fusion; every
   dropped source is reported with the reason.
3. **Write, one page per call** — each page gets its own request carrying
   that page's role and a block-by-block checklist. This is what lets a
   small local model fill eight dense pages instead of writing one
   paragraph and stopping.
4. **Check density** — a page that comes back thin goes through an
   expansion pass before it is allowed near the renderer.
5. **Fit the page** — the sheet is a fixed 210×297mm and anything past it
   would be cut, so kx measures each page first, trims prose, and drops
   only redundant blocks, reporting anything it removes.
6. **Compile** — HTML to vector PDF, then PNG, PPTX and JSON.

---

## The eight pages

`--pages 8` gives the full arc. Fewer pages keep the overview and the
takeaways and drop depth from the middle; more repeat the deep-dive shape.

| # | Page | Carries |
|---|---|---|
| 1 | Executive overview & architecture | problem, goal, benefits, pipeline diagram, components, stack |
| 2 | Datasets, workloads & benchmarks | sources, failure modes, environment, construction flow, statistics |
| 3 | Deep dive 1 — ingestion & dataflow | what happens under the hood, step-by-step flow, worked example, complexity |
| 4 | Deep dive 2 — core algorithm | scope mapping, pipeline, side-by-side example, mnemonic, formulation |
| 5 | Guardrails & correctness | failure modes, verification gate, filtering flow, ablation |
| 6 | Local execution & code | hardware table, parameters, steps, runnable code, challenges |
| 7 | Evaluation & honest benchmarks | setup, metrics, bar chart, Three Honest Columns, ablation |
| 8 | Takeaways & production runbook | summary, impact, lessons, future work, copy-paste runbook |

**Three Honest Columns** is enforced, not optional: any speedup is reported
as kernel/compute only, then including host-to-device or network transfer,
then end-to-end wall time. A single flattering number never appears alone.

---

## Personas

Pass one or several: `--persona student,professional`.

| Persona | Adds |
|---|---|
| `student` | plain-language intuition before maths, interview traps, flashcard pairs |
| `phd` | related-work table from verified sources, empirical vs theoretical marking, target venues |
| `researcher` | research-gap framing, hedged as "not found in the surveyed sources" |
| `professor` | lecture-shaped pages, top 3 student misconceptions with corrections |
| `professional` | Three Honest Columns, observability recipe (eBPF, DCGM, Nsight) |
| `lab` | reproducibility block, lock-clock commands, CI accuracy gate |
| `ai-security` | threat matrix and the guarded-pipeline counterpart |

---

## Converting something you already have

```bash
kx convert ./deck.pptx        --pages 8 -o ./deck_8p.pdf     # slides + speaker notes
kx convert ./paper.pdf        --pages 8 --topic "Paper Title"
kx convert ./notes.md         --pages 4
kx convert https://github.com/kinetixai-ai/kx --topic "kx CLI" --pages 6
kx convert 2205.14135         --pages 8 --json-bundles 1      # arXiv id or DOI
```

---

## Building a whole catalog

`primers_catalog.yaml` ships with all 18 KinetixAI tracks. `defaults` apply
to every track; any track key overrides them.

```yaml
defaults:
  pages: 8
  pdfs: 1
  ppts: 1
  images: 8
  json_bundles: 1
  engine: vllm
  model: "Qwen/Qwen2.5-14B-Instruct"
  verify_citations: true
  out_dir: ./dist/primers

tracks:
  - topic: "Attention & Transformers"
    category: core-ai
    personas: [student, phd, professor]
  - topic: "GPU & CUDA"
    category: gpu
    personas: [student, professional, lab]
```

```bash
kx primers autobuild --manifest ./primers_catalog.yaml --parallel 4
kx primers autobuild --manifest ./primers_catalog.yaml --json | jq '.built, .failed'
```

A failing track reports and the others carry on; the exit code is 1 if any
failed, which makes it usable in CI.

---

## From an agent instead of the terminal

`kx mcp serve` exposes kx over the Model Context Protocol, so an agent can
drive it in plain language.

```bash
claude mcp add kx -- kx mcp serve      # Claude Code
kx mcp install cursor                  # prints the config for any other host
```

Supported hosts: Claude Code, Claude Desktop, Cursor, Gemini CLI, Codex,
VS Code (Copilot agent mode), Zed, Goose, Cline, Continue, OpenHands,
Open WebUI.

Then just ask:

> Use the kx tool to build an 8-page primer on GPU kernel fusion for a
> student audience, into ./dist/gpu

| Tool | Does |
|---|---|
| `build_primer` | topic → full primer bundle |
| `convert_document` | deck, paper or repo → primer |
| `search_verified_papers` | domain-filtered sources for a topic |
| `search_papers`, `get_paper`, `find_code`, `find_models`, `find_datasets` | research |
| `library_search`, `rag_ask` | your own local library |

With `engine: "mcp"` the host's own model writes the pages, so no API key
and no local model are needed. Sampling is optional in the MCP protocol; a
host that lacks it gets a clear message naming the alternatives.

---

## Where the files go

`--out-dir` is the working output. `--copy-to` puts a second copy
wherever you keep things — a notes folder, a shared drive, a repo.

```bash
kx primers build --topic "Types of RAG" \
  --out-dir ./dist/rag \
  --copy-to ~/My_Companies_Data/KinetixAI-AI/kx-cli/notes/

kx config set primer_copy_to ~/My_Companies_Data/KinetixAI-AI/kx-cli/notes/
```

The copy keeps the `images/` subfolder. If the destination is unreachable
the build still succeeds and says so — a missing drive never costs you a
primer.

---

## Every option

### `kx primers build`

| Option | Default | Means |
|---|---|---|
| `--topic`, `-t` | required | What the primer is about |
| `--source`, `-s` | — | `.pptx`, `.pdf`, `.md`, GitHub URL, arXiv id or DOI |
| `--persona` | `student` | Comma list; see Personas |
| `--pages` | `8` | Pages, 1–42 |
| `--pdfs` | `1` | PDF variants |
| `--images` | `0` | 300 DPI PNG pages |
| `--ppts` | `0` | Editable `.pptx` decks |
| `--json-bundles` | `0` | Paper Lab JSON bundles |
| `--engine` | config, else auto | See Choosing an engine |
| `--model`, `-m` | provider default | Model name |
| `--api-base` | provider default | Override the endpoint |
| `--num-ctx` | `32768` | Ollama context window |
| `--mode` | `iterative` | `iterative` (dense) or `single` (one call) |
| `--min-sections-per-page` | `5` | Density floor |
| `--relevance` | `0.18` | Citation domain threshold; `0` disables |
| `--verify-citations` | off | Re-resolve each citation, drop failures |
| `--watermark` | `Narmadya Innovations` | Header and footer mark |
| `--author` | `KinetixAI.AI` | Name in the page badge |
| `--category` | — | Catalog category slug |
| `--out-dir`, `-o` | `./dist/primers` | Output folder |
| `--copy-to` | config | Second home for the finished files |
| `--json` | off | Machine-readable result on stdout |

### `kx convert`

Same options, plus `--out/-o` for the exact PDF path and `--style`
(`trifleet-8page`). Takes the source as its argument.

### Settings

```bash
kx config set primer_engine  claude
kx config set primer_model   claude-sonnet-4-5
kx config set primer_copy_to ~/Notes/primers
kx config show
```

---

## What kx will not do

These are deliberate, and they are why the output can be trusted.

- **No invented citations.** Sources come from arXiv, Semantic Scholar and
  OpenAlex. `--verify-citations` re-resolves every one and drops what does
  not check out, naming each.
- **No invented numbers.** The model is told it may use only the figures in
  your sources. With none, it describes the measurement method instead.
- **No single flattering benchmark.** Three Honest Columns is required.
- **No silent loss.** If a page will not fit, kx says what it trimmed or
  dropped. If a source is filtered out, it says why. If an engine fails
  mid-build, the pages already written survive and the failure is reported.
- **No pretending a draft is finished.** `outline` mode labels every page
  "Draft outline — verify before publishing."

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `primers needs extra packages` | `pip install 'kinetixai[primers]'` |
| `set ANTHROPIC_API_KEY to use --engine claude` | export the key, or pick another engine |
| `the LLM endpoint ... is not reachable` | start the server: `ollama serve`, LM Studio's Developer tab, etc. |
| `timed out after 600s` | raise `KX_LLM_TIMEOUT`, lower `--pages`, or use a faster model |
| `Ollama returned no content (done_reason=length)` | raise `--num-ctx` |
| `the model returned no JSON object` | use a non-reasoning model, or a larger one |
| Pages come out thin | raise `--num-ctx`, use a bigger model, keep `--mode iterative` |
| WeasyPrint import errors on macOS | `brew install pango gdk-pixbuf libffi`, or `pip install playwright && playwright install chromium && export KX_PDF_ENGINE=playwright` |
| `semantic unavailable` | Semantic Scholar rate limit; arXiv and OpenAlex still supply sources |
| An off-topic source got through | lower `--relevance` for more, raise it for fewer |

Still stuck: `kx doctor` checks paths, extras, engines and network, and
`kx primers engines` shows what can write a primer right now.
