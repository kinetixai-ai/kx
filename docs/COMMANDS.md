# `kx` command reference

A deep dive into every `kx` command: what it does, where the data comes
from, what it writes to your disk, what leaves your machine, every flag,
and a worked example.

`kx` is the command line for KinetixAI (PyPI package `kinetixai`). It
goes paper → code → models → datasets → your own private Q&A, and it can
compile any of that into a printable Primer.

## How to read this document

Commands are shown in fenced blocks:

```bash
kx search "flash attention" -n 5
```

Every output block is labelled one of two ways:

| Label | Meaning |
| --- | --- |
| **Captured** | Really run and pasted here, trimmed for width. |
| **Illustrative** | Hand-written to show the output *shape*. The code path was read, not run. |

Why the split: this reference was produced in a sandbox where outbound
HTTPS to arXiv, Semantic Scholar, OpenAlex, Hugging Face and
`kinetixai.ai` is blocked by a proxy. GitHub was reachable, and
everything local (library, RAG, Primers, config, MCP) was reachable. So
anything that needs those four services is marked illustrative. Nothing
here is a benchmark and no numbers were invented — where a real number
was unavailable, the field is shown as a placeholder.

## Conventions that apply to every command

**`--json`.** Every command takes `--json` and writes a single JSON
object to stdout. Warnings and progress always go to stderr, never
stdout, so `kx ... --json | jq` is always safe.

**Exit codes.**

| Code | Meaning |
| --- | --- |
| 0 | Worked. |
| 1 | Could not do it: nothing found, every source down, bad token, build failed. |
| 2 | You asked for something impossible: bad flag value, no PDF to download, missing manifest. |
| 130 | You pressed Ctrl-C during a Primer build. |

**Where things live.** Two directories, from `platformdirs`:

| | Linux | macOS | Windows |
| --- | --- | --- | --- |
| Config | `~/.config/kinetixai` | `~/Library/Application Support/kinetixai` | `%LOCALAPPDATA%\kinetixai` |
| Data | `~/.local/share/kinetixai` | `~/Library/Application Support/kinetixai` | `%LOCALAPPDATA%\kinetixai` |

Inside them: `config.json`, `credentials.json`, `library.json`,
`pdfs/`, `rag/index.json`. Override either directory with the
`KX_CONFIG_DIR` and `KX_DATA_DIR` environment variables — useful for a
per-project library. `kx doctor` prints all of these paths.

**Accounts.** Only one command needs `kx login`: `kx ask`. Everything
else — search, scholar, get, cite, code, models, datasets, library, rag,
primers, convert, mcp, config, doctor — works with no account at all.
Two optional environment tokens raise rate limits but are never
required and never stored: `GITHUB_TOKEN` (or `GH_TOKEN`) for GitHub, and
`HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for Hugging Face.

**Optional extras.** `kx` installs small by default. Features that need
more:

| Extra | Packages | Needed by |
| --- | --- | --- |
| `rag` | `pypdf` | reading PDF text for `kx rag index` |
| `primers` | `jinja2`, `weasyprint`, `pypdfium2`, `pillow`, `python-pptx`, `pyyaml` | `kx primers`, `kx convert` |
| `mcp` | `mcp` | `kx mcp serve` |
| `keyring` | `keyring` | storing your token in the OS keychain |

Install with `pip install 'kinetixai[primers]'` or, under pipx,
`pipx inject kinetixai jinja2 pyyaml python-pptx pypdf pypdfium2 pillow weasyprint`.

---

# The commands

## `kx version`

**For:** printing which `kx` you are running.

**Data:** none. Reads the package version in memory.

**Flow:** nothing is fetched and nothing is written. Fully offline.

**Account:** not needed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--json` | JSON on stdout. | Pinning the version in a script or bug report. |

```bash
kx version
```

**Captured**

```
kx 0.3.0
```

---

## `kx doctor`

**For:** showing what `kx` can see on this machine, and what to fix.

**Data:** mostly local — the Python version, OS, config and data paths,
whether a token is stored, which optional extras import, whether Ollama
answers at `ollama_url`, and how many papers and RAG chunks you have.
With network checks on it also pings six endpoints: your configured
KinetixAI endpoint (`/api/auth/me`), arXiv, Semantic Scholar, OpenAlex,
Hugging Face and the GitHub rate-limit endpoint.

**Flow:** reads `config.json`, `library.json` and `rag/index.json`;
calls `GET /api/tags` on Ollama; issues six HTTP GETs with a 6-second
timeout. Writes nothing. The network probes send no identifying data
beyond the `kinetixai-cli/<version>` user agent, and the KinetixAI probe
sends no token.

**Account:** not needed. Doctor tells you if you are not logged in, and
says that only `kx ask` cares.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--offline` | Skip the six network probes. | On a plane, or when you only want the paths and extras. |
| `--json` | JSON on stdout. | Health checks in CI, or pasting into a bug report. |

```bash
kx doctor
```

**Captured** (trimmed; this sandbox blocks everything but GitHub)

```
kx 0.3.0
  Python      3.13.18  ok  /usr/local/bin/python3.13
  OS          Linux 6.18.44-fc-v310 (x86_64)
  config      /root/.config/kinetixai/config.json
  data        /root/.local/share/kinetixai
  library     /root/.local/share/kinetixai/library.json
  rag_index   /root/.local/share/kinetixai/rag/index.json
  kx          /usr/local/bin/kx
  commands    19  ask, cite, code, config, convert, datasets, doctor, get, ...
  Endpoint    https://kinetixai.ai
  Token       no   store: file
  Library     0 papers; RAG index: not built
  [rag] extra      ok  read PDFs for kx rag index
  [mcp] extra      ok  kx mcp server for agents
  [primers] extra  ok  kx primers build and kx convert
  [keyring] extra  no  store your token in the OS keychain
  Ollama      no  http://localhost:11434
  KinetixAI         no  ProxyError
  arXiv             no  ProxyError
  Semantic Scholar  no  ProxyError
  OpenAlex          no  ProxyError
  Hugging Face      no  ProxyError
  GitHub            ok  200 in 523 ms

Suggestions
  - Not logged in (only `kx ask` uses an account): create a token at
    https://kinetixai.ai/account/tokens/ and run `kx login`.
  - Ollama not running: kx rag ask will return passages, not written answers.
  - arXiv unreachable (ProxyError). Check your network, proxy or firewall.
```

Known bug: if the `primers` extra is **not** installed, the advice
builder raises `NameError` instead of printing its hint. See
"Things that do not match the help text" at the end.

---

## `kx search`

**For:** finding papers on a topic across three public indexes at once.

**Data:** three public APIs, no key:

| Source | Endpoint | What it gives |
| --- | --- | --- |
| arXiv | `export.arxiv.org/api/query` (Atom XML) | title, authors, year, abstract, arXiv id, PDF link |
| Semantic Scholar | `api.semanticscholar.org/graph/v1/paper/search` | the above plus DOI, venue, citation count, open-access PDF |
| OpenAlex | `api.openalex.org/works` | the above plus venue and cited-by count |

**Flow:** one HTTP GET per source, in sequence, with the query you
typed. Results are merged in memory on a de-duplication key; when the
same paper comes back from two sources, the kept copy is *enriched* —
missing DOI, arXiv id, PDF URL, venue, year, URL and citation count are
filled from the duplicate, and the `Source` column then reads e.g.
`arXiv, OpenAlex`. The merged list is sorted by citation count, then
year, both descending. Nothing is written to disk. One source failing is
a warning on stderr, not an error; only if *all* of them fail do you get
exit 1 (so "no results" and "everything is down" are different
outcomes).

OpenAlex ships abstracts as an inverted index (`{word: [positions]}`);
`kx` reassembles the sentence locally.

**Account:** not needed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--limit`, `-n` | Max results **per source**, 1–100. Defaults to the `limit` setting (10). | Narrowing a broad topic, or pulling a long list for a review. |
| `--source`, `-s` | `all` (default), `arxiv`, `semantic`, `openalex`. | When one source is slow or rate-limiting you, or you want only arXiv preprints. |
| `--json` | JSON on stdout: `{query, papers[], warnings[]}`. | Piping into `jq`, or into `kx library add`. |
| `--open` | Open the top result's page in your browser. | "Just show me the best hit." |

```bash
kx search "flash attention io-aware" -n 3
```

**Illustrative** (arXiv, Semantic Scholar and OpenAlex are blocked here)

```
  #   Title                               Authors            Year   Cites  Source
  1   FlashAttention: Fast and Memory-     Tri Dao, Daniel    2022    ....  Semantic Scholar,
      Efficient Exact Attention with IO-   Y. Fu, Stefano             arXiv
      Awareness                            Ermon, +2
  2   FlashAttention-2: Faster Attention   Tri Dao            2023    ....  arXiv, OpenAlex
      with Better Parallelism
```

**Captured** — what it really looks like when every source is down:

```
warning: arxiv unavailable (ProxyError)
warning: semantic unavailable (ProxyError)
warning: openalex unavailable (ProxyError)
No results.
```

---

## `kx scholar`

**For:** looking up a researcher and their recent work.

**Data:** the ORCID public API (`pub.orcid.org/v3.0`) only. Not Google
Scholar — Google Scholar has no public API, and if your query contains
`scholar.google` the command says so and exits 2 rather than pretending.

**Flow:** if what you typed matches the ORCID iD pattern
(`0000-0000-0000-0000`), `kx` fetches that one record and reads the
person's name plus the titles of up to ten work groups. Otherwise it
calls ORCID's expanded name search and returns a list of candidates with
name, iD and first affiliation. One GET either way. Nothing is written.

**Account:** not needed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--limit`, `-n` | Max people, 1–100. Default 10. | A common surname. |
| `--json` | JSON on stdout. | Scripting a lab roster. |

```bash
kx scholar 0000-0002-1825-0097
```

**Illustrative** (ORCID is blocked here)

```
Josiah Carberry  0000-0002-1825-0097
  https://orcid.org/0000-0002-1825-0097
    - Toward a Unified Theory of High-Energy Metaphysics
    - Electron Band Structure in Germanium, My Ass
```

---

## `kx get`

**For:** pulling one paper's full metadata by identifier, and optionally
its PDF.

**Data:** identifier-dependent, and deliberately ordered:

- An **arXiv id** (`2205.14135`, `2205.14135v2`, `cs.LG/0703041`), an
  `arxiv.org/abs|pdf|html/...` URL, or a `huggingface.co/papers/...`
  URL → arXiv's API first; if arXiv does not answer, Semantic Scholar's
  `paper/arXiv:<id>` as a fallback.
- A **DOI** (`10.1234/xyz`, or a `doi.org` URL) → Semantic Scholar's
  `paper/DOI:<doi>` first, then OpenAlex's `works/doi:<doi>`.

**Flow:** one or two GETs to resolve the record. With `--pdf`, the PDF
URL that came back in the metadata is streamed to disk: written to
`<out>/<id>.part` and then atomically renamed to `<out>/<id>.pdf`, with
every character outside `A-Za-z0-9._-` replaced by `_` so the name is
safe on Windows too. That download goes directly to whoever hosts the
PDF (arXiv, or the open-access host Semantic Scholar/OpenAlex named) —
not through KinetixAI. Without `--pdf`, nothing is written.

If there is no open PDF, `kx` prints a note and exits 2. It does not go
looking for a paywalled copy.

**Account:** not needed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--pdf` | Also download the PDF. | You want the full text, e.g. to feed `kx rag index`. |
| `--out`, `-o` | Directory for the PDF. Default `.`. | Keeping downloads out of your working folder. |
| `--bibtex` | Print a BibTeX entry instead of the detail view. | Dropping a reference straight into a `.bib` file. |
| `--json` | JSON on stdout. With `--pdf` the object also carries `saved`. | Scripting. |

```bash
kx get 2205.14135
kx get 2205.14135 --pdf -o ~/papers
kx get 10.1038/s41586-021-03819-2 --bibtex
```

**Illustrative** (arXiv blocked here)

```
FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
Tri Dao, Daniel Y. Fu, Stefano Ermon, +2  ·  2022  ·  arXiv
arXiv: 2205.14135
URL: https://arxiv.org/abs/2205.14135

Transformers are slow and memory-hungry on long sequences. ...
saved /home/you/papers/2205.14135.pdf
```

---

## `kx cite`

**For:** printing one citation, BibTeX or APA 7.

**Data:** your **local library first**. `kx cite` looks the reference up
in `library.json` by key, by bare arXiv id or DOI, or by its row number
in `kx library list`. Only if it is not there does it go to the network,
using exactly the same resolver as `kx get`.

**Flow:** library lookup is pure file read, no network. The formatting is
entirely local and deterministic. BibTeX escapes LaTeX specials
(`\ { } & % # $ _ ^ ~`) so a title with an ampersand can never unbalance
the entry; the key is `<surname><year><firstlongtitleword>`, e.g.
`dao2022flashattention`. arXiv-only papers come out as `@misc` with
`eprint` and `archivePrefix`; anything with a venue comes out as
`@article` with `journal`. APA 7 handles up to 20 authors in full and
abbreviates past that with the first 19, an ellipsis and the last.
Nothing is written to disk.

**Account:** not needed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--format`, `-f` | `bibtex` (default) or `apa`. Anything else exits 2. | Whichever your document wants. |
| `--json` | JSON on stdout: `{id, format, citation}`. | Building a reference list in a script. |

```bash
kx cite 1
kx cite 2205.14135 -f apa
```

**Captured** (the paper was already in the library, so this ran offline)

```
@misc{dao2022flashattention,
  title = {{FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness}},
  author = {Tri Dao and Daniel Y. Fu and Stefano Ermon and Atri Rudra and Christopher Re},
  year = {2022},
  eprint = {2205.14135},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2205.14135}
}
```

```
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). FlashAttention:
Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv.
https://doi.org/10.48550/arXiv.2205.14135
```

---

## `kx code`

**For:** finding the code for a paper.

**Data:** two services, in a deliberate order.

1. **Hugging Face** — `huggingface.co/api/papers/<arxiv_id>`. If HF has
   a paper page, `kx` takes the `githubRepo` field from it. That is the
   repo the authors themselves linked, so it is marked official and
   pinned to the top of the list.
2. **GitHub** — `api.github.com/search/repositories`, sorted by stars,
   descending. Given an arXiv id the query is
   `<id> in:name,description,readme,topics`, so it finds repos that
   mention the paper anywhere in their front matter. Given free text,
   the text is the query.

If GitHub also returns the official repo, GitHub's row wins (fresher
star count and description) but keeps the official flag. The final
ordering is: official first, then by stars.

**Flow:** at most two GETs. Nothing is written. Unauthenticated GitHub
search allows roughly 10 requests a minute; set `GITHUB_TOKEN` or
`GH_TOKEN` to raise that. `kx` sends the token if it is in the
environment and never stores it. A 403 with `x-ratelimit-remaining: 0`
is treated as a rate limit and retried politely (never sleeping longer
than 8 seconds) before it becomes an error.

An arXiv id gets much better results than free text, because the id is a
unique string that implementations quote.

**Account:** not needed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--limit`, `-n` | Max repos, 1–100. Default 10. | A short list for a README, or a long one for a survey. |
| `--json` | JSON on stdout: `{query, arxiv_id, paper, repos[], warnings[]}`. | Feeding a dashboard. |

```bash
kx code 2205.14135 -n 3
```

**Captured** (GitHub was reachable; Hugging Face was not, hence the warning)

```
warning: Hugging Face: unreachable (ProxyError). Check your connection or proxy.
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ #   ┃ Repository                    ┃   Stars ┃ Lang   ┃ License      ┃ About           ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 1   │ Dao-AILab/flash-attention     │   25.3k │ Python │ BSD-3-Clause │ Fast and memory-│
│     │                               │         │        │              │ efficient exact │
│     │                               │         │        │              │ attention       │
│ 2   │ lucidrains/PaLM-rlhf-pytorch  │    7.9k │ Python │ MIT          │ Implementation  │
│     │                               │         │        │              │ of RLHF ...     │
│ 3   │ lucidrains/ring-attention-pyt │     548 │ Python │ MIT          │ Ring Attention, │
│     │ orch                          │         │        │              │ from Liu et al. │
└─────┴───────────────────────────────┴─────────┴────────┴──────────────┴─────────────────┘
Open: https://github.com/<repository>
```

Note what the warning means: with Hugging Face down there is no
official-repo marker, so the top row here is "most starred", not
"the authors' own". In this case they happen to be the same repo.

---

## `kx models`

**For:** finding Hugging Face models.

**Data:** the Hugging Face Hub API, `huggingface.co/api/models`, no key.
Two different queries depending on what you type:

- **Free text** → `?search=<text>`, optionally `&pipeline_tag=<task>`.
- **An arXiv id** → `?filter=arxiv:<id>`. Hugging Face tags a model card
  `arxiv:<id>` when it cites a paper, so this is "models that cite this
  paper", not a text match.

**Flow:** one GET, `limit` capped at 100 server-side. The licence is read
from the `license:<name>` tag, falling back to `cardData.license`. Sorting
happens on Hugging Face's side, not locally: `--sort` maps to their
`downloads`, `likes`, `trendingScore` and `lastModified` fields,
descending. Nothing is written. `HF_TOKEN` is sent if set, for higher
rate limits, and never stored.

**Account:** not needed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--limit`, `-n` | Max models, 1–100. Default 10. | |
| `--sort` | `downloads` (default), `likes`, `trending`, `updated`. | `updated` to see what is actively maintained; `trending` for what is new. |
| `--task` | A pipeline tag, e.g. `text-generation`, `automatic-speech-recognition`. Free-text searches only. | Cutting a noisy name search down to one modality. |
| `--json` | JSON on stdout: `{query, arxiv_id, models[], warnings[]}`. | |

```bash
kx models "whisper" --task automatic-speech-recognition -n 3
kx models 2205.14135 --sort likes
```

**Illustrative** (Hugging Face blocked here)

```
  #   Model                      Downloads   Likes  License       Task
  1   openai/whisper-large-v3        ......  .....  apache-2.0    automatic-speech-recognition
  2   openai/whisper-small           ......  .....  apache-2.0    automatic-speech-recognition
```

**Captured** — the failure shape:

```
warning: Hugging Face: unreachable (ProxyError). Check your connection or proxy.
No models found.
```

---

## `kx datasets`

**For:** finding Hugging Face datasets.

**Data and flow:** identical to `kx models`, against
`huggingface.co/api/datasets`. Free text becomes `?search=`; an arXiv id
becomes `?filter=arxiv:<id>`, i.e. datasets whose card cites the paper.
The task column falls back to the `task_categories:<name>` tag, since
datasets have no `pipeline_tag`. Nothing is written.

**Account:** not needed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--limit`, `-n` | Max datasets, 1–100. Default 10. | |
| `--sort` | `downloads` (default), `likes`, `trending`, `updated`. | |
| `--json` | JSON on stdout. | |

There is no `--task` on `datasets`.

```bash
kx datasets "common voice" -n 3
kx datasets 1810.04805 --sort downloads
```

**Illustrative** (Hugging Face blocked here)

```
  #   Dataset                    Downloads   Likes  License       Task
  1   mozilla-foundation/common_     ......  .....  cc0-1.0       automatic-speech-recognition
      voice_17_0
```

---

## `kx library`

**For:** your own list of papers, kept on this machine.

**Data:** one JSON file, `<data dir>/library.json`. **Nothing in the
`library` subcommands touches the network except `add`**, and `add` only
when you hand it an arXiv id or DOI (to resolve metadata) or pass
`--pdf` (to download one).

Each entry is a paper record plus a stable key, tags, an added-at
timestamp, and optionally the path to a PDF on disk. Keys are:

- `arxiv:2205.14135` — arXiv id, version suffix stripped.
- `doi:10.1234/xyz` — DOI, lowercased.
- `local:<slug>-<hash12>` — for a file with no id. The hash is the
  SHA-256 of the **file's bytes** when there is a file, otherwise of
  title + authors + year. So two different PDFs both called `paper.pdf`
  never collide.

Adding a paper that is already there *updates* it: new non-empty fields
overwrite, empty ones never blank out what you had, and tags are
appended.

### `kx library add`

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `REFS` (positional, repeatable) | arXiv ids, DOIs, or paths to local PDFs / notes. | The normal case. |
| `--from-file` | Import papers from a JSON file. Accepts one object, a list, or a `kx library export --format json` dump. Fully offline. | Restoring a library, moving between machines, or seeding from `kx search --json`. |
| `--pdf` | Also download the open PDF into `<data dir>/pdfs/`. | You intend to run `kx rag index` and want full text, not just abstracts. |
| `--copy` | Copy a local file into `<data dir>/pdfs/` instead of referencing it where it sits. The copy's name carries a content hash. | The original lives somewhere you may move or delete. |
| `--tag`, `-t` | Attach a tag. Repeatable. | Grouping by course, project or reading list. |
| `--json` | JSON on stdout: `{added[], failed[], library}`. | |

A non-PDF local file (`.md`, `.txt`) is added as a title-only entry
pointing at the file; a `.pdf` has its title and authors read from the
PDF metadata when `pypdf` is installed, and falls back to the file name.
A reference that is neither an id, a DOI nor an existing file is an
error for that one reference — the rest still get added, and the command
only exits 1 if *nothing* was added.

```bash
kx library add 2205.14135 1706.03762 --pdf -t gpu -t reading-list
kx library add ./notes/kernel-fusion.md
kx library add --from-file papers.json
```

**Captured** (imported from a local JSON file, no network)

```
added arxiv:2205.14135  FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
```

### `kx library list`

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--tag`, `-t` | Only papers with this tag. | |
| `--json` | JSON on stdout. | |

```bash
kx library list
kx library list -t gpu
```

**Captured**

```
┏━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ #   ┃ Key               ┃ Title                                         ┃ Year  ┃ PDF  ┃ Tags    ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━━━━┩
│ 1   │ arxiv:2205.14135  │ FlashAttention: Fast and Memory-Efficient     │ 2022  │      │ gpu     │
│     │                   │ Exact Attention with IO-Awareness             │       │      │         │
└─────┴───────────────────┴───────────────────────────────────────────────┴───────┴──────┴─────────┘
```

The `#` column is a stable handle for `kx cite 1` and
`kx library remove 1`.

### `kx library search`

Not in the brief's command list, but it exists and is useful: a keyword
match over title, authors, abstract and tags, entirely offline. Title
hits count triple.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--limit`, `-n` | Max hits, 1–100. Default 10. | |
| `--json` | JSON on stdout: `{query, papers[], total}`. | |

```bash
kx library search tiling
```

**Captured** — same table shape as `kx library list`, filtered to the
one matching row.

### `kx library remove`

Removes the entry. **Files on disk are left alone** — a downloaded PDF
in `<data dir>/pdfs/` stays there.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--json` | JSON on stdout. | |

```bash
kx library remove 2205.14135
kx library remove 1
```

**Captured**

```
removed arxiv:2205.14135
```

### `kx library export`

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--format`, `-f` | `bibtex` (default), `apa`, `json`. | `bibtex` for LaTeX; `json` for backup or moving machines. |
| `--out`, `-o` | Write to a file instead of stdout. | |
| `--json` | JSON on stdout (the wrapper object, not the export). | |

BibTeX export disambiguates colliding keys with letter suffixes:
`dao2022flash`, `dao2022flasha`, `dao2022flashb`, … The `json` format
round-trips through `kx library add --from-file`.

```bash
kx library export -f bibtex -o refs.bib
kx library export -f json -o backup.json
```

**Captured**

```
wrote 1 entries to /tmp/kxdemo/lib.bib
```

---

## `kx rag`

**For:** asking questions of your own library, on your own machine.

**Data:** one file, `<data dir>/rag/index.json`. It holds the documents,
the text chunks, and optionally one embedding vector per chunk. The
content comes from two places: every entry in your library (its title and
abstract, plus the full text of its PDF if it has one), and any extra
folder or file you point at with `--path`.

**Retrieval is local.** BM25 (Okapi BM25, pure Python, in this
repository) always runs. If Ollama is running *and* your embedding model
is pulled *and* the index was built with embeddings, a second,
vector-similarity ranking runs and the two are fused with reciprocal
rank fusion; the mode is then reported as `hybrid` instead of `bm25`.

**What can leave the machine:** only text sent to Ollama. If Ollama is
on `localhost` (the default) nothing leaves at all, and `kx` says so.
If you have pointed `ollama_url` at another host, then chunk text (during
`index`) and your question (during `ask`) go to that host. `kx rag ask`
reports a `local: true/false` field that is only true when no text went
to a non-local Ollama.

**Account:** not needed, ever. There is no KinetixAI call in `kx rag`.

### `kx rag index`

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--path`, `-p` | An extra folder or file of PDFs/notes to index. **Remembered** — it is stored in the index and re-scanned on every future `kx rag index`. Repeatable. | Pointing at your Obsidian vault, a course folder, a papers directory. |
| `--forget` | Stop indexing one remembered path. | You moved or deleted the folder. |
| `--forget-all-paths` | Stop indexing every remembered path. | Starting over with just your library. |
| `--embeddings` / `--no-embeddings` | Require Ollama embeddings, or skip them. Default: use them if Ollama is up. | `--embeddings` in CI, where a silent fallback to keyword-only would be a surprise. `--no-embeddings` for a fast rebuild. |
| `--json` | JSON on stdout: `{documents, chunks, embeddings, paths, index, warnings[]}`. | |

Folder scanning takes `.pdf`, `.txt`, `.md`, `.markdown`, `.rst` and
`.tex`, recursively, and **skips dotfiles and anything inside a
dot-directory** (`.git`, `.venv`, `.obsidian`). Text is chunked into
overlapping ~160-word windows with 40 words of overlap. PDF pages are
read with `pypdf` (the `rag` extra) and the page number is kept so a
passage can cite `p. 7`. Embeddings are cached by chunk hash between
runs, so re-indexing after adding one paper only embeds the new chunks.

A path that no longer exists is a warning, not an error, and the warning
tells you the exact `--forget` command.

```bash
kx rag index
kx rag index --path ~/papers --path ~/notes
kx rag index --no-embeddings
kx rag index --forget ~/old-notes
```

**Captured** (Ollama not running, so keyword-only)

```
Indexed 2 documents, 2 chunks (BM25 keyword search).
/tmp/kxdemo/data/rag/index.json. Stored on this machine only.
```

### `kx rag ask`

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `-k` | Passages to retrieve, 1–50. Default 5. | More for a broad question, fewer for a precise one. |
| `--no-llm` | Return passages only, even if Ollama is running. | You want the exact source text, not a summary. Also keeps the run entirely offline. |
| `--model`, `-m` | Ollama model for the answer. Defaults to the `ollama_model` setting. | Trying a bigger model on a hard question. |
| `--json` | JSON on stdout: `{question, answer, passages[], retrieval, model, ollama_url, local, note}`. | |

The system prompt tells the model to answer **only** from the numbered
passages, to cite each claim as `[2]`, and to say plainly when the
passages do not contain the answer. If Ollama is not running, or the
model is not pulled, you get the passages and a note saying why — never
a silent hallucination.

```bash
kx rag ask "why does kernel fusion help?" -k 2
kx rag ask "what tiling size did they use?" --no-llm
```

**Captured** (Ollama not running)

```
Ollama is not running at http://localhost:11434, so here are the best passages.
Start Ollama for written answers.

[1] notes.md  file:notes.md
    # Kernel Fusion Notes Fusing elementwise kernels removes round trips to HBM.
    ## Why it helps One launch, one pass over memory. ## Cost Register pressure
    rises and occupancy can drop.

Private: retrieval and answer ran on this machine; nothing was sent anywhere.
```

### `kx rag status`

Also not in the brief's list, but present: shows what is indexed without
rebuilding.

```bash
kx rag status
```

**Captured**

```
2 documents, 2 chunks, built 2026-11-09T06:51:52Z, BM25 only
  /tmp/kxdemo/notes.md
```

---

## `kx primers`

**For:** generating a styled, dense, A4 "Story to Silicon" Primer —
PDF, PNG page cards, an editable PPTX, and a structured JSON bundle.

The brief listed `primers convert` as a subcommand. It is not one.
`convert` is a **top-level** command (`kx convert`). The `primers` group
has exactly three subcommands: `build`, `engines`, `autobuild`.

### How a primer is built

Seven stages, in this order:

1. **Preflight.** Before any model is called, `kx` imports the packages
   the requested outputs need (`jinja2` always; `weasyprint` for PDFs
   and PNGs; `pypdfium2` and `pillow` for PNGs; `python-pptx` for
   decks). It *imports* rather than checks, because WeasyPrint installs
   cleanly on macOS and then fails at import without Homebrew's pango.
   Finding that out after eight pages of local inference would throw the
   whole run away.
2. **Ingest.** Your `--source`, if any. See `kx convert` below for the
   per-type detail.
3. **Research.** Up to three federated searches (arXiv + Semantic
   Scholar + OpenAlex, same code as `kx search`) derived from the topic,
   then local, deterministic domain filtering. Research failing is a
   warning; the build continues with no citations, because "a build with
   no citations is honest, a build with wrong ones is not".
4. **Synthesize.** One LLM call per page by default, carrying only that
   page's archetype and density quota. A page that comes back sparse gets
   one expansion pass, and the better of the two is kept. Pages that
   fail are warnings; the build continues with what it has, and only
   fails outright if *every* page failed.
5. **Citations.** `primer.citations = sources[:15]` — ingested source
   papers first, then researched ones.
6. **Fit.** Section numbers are regenerated, over-budget pages are
   compacted ~15%, and any page still past 297mm is fitted by dropping
   droppable blocks, with a warning naming each drop. Required block
   types are never dropped.
7. **Render and export.** Jinja2 renders a self-contained HTML document;
   WeasyPrint compiles it to a vector A4 PDF. If rendering fails after
   the pages were written, the pages are rescued to
   `<slug>.primer.json` so the run can be rendered later rather than
   repeated.

The domain filter is worth understanding, because it is why citation
lists are sometimes short. It is deterministic and offline — no
embedding model, no extra LLM call — so a citation list can always be
explained. It scores each candidate as *domain-term overlap +
subtopic-term overlap − ambiguity penalty*. A paper that matches only an
ambiguous word from your topic is dropped: "GPU & CUDA: Kernel Fusion"
will not pull in tokamak fusion, infrared image fusion or kernel
methods. Every drop produces a note, so a thin citation list is never a
silent one. `--relevance 0` turns filtering off entirely.

### `kx primers build`

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--topic`, `-t` | **Required.** The topic, e.g. `'GPU & CUDA: Kernel Fusion'`. Also the default title and output slug. | Always. |
| `--source`, `-s` | A `.pptx`/`.pdf`/`.md`/`.txt` file, a GitHub URL, or an arXiv id/DOI to build from. | You have material already; see `kx convert`. |
| `--category` | A catalog category slug, e.g. `gpu`, `rag`, `security`. Metadata only; it appears in the JSON bundle. | Organising a catalog. |
| `--persona` | Comma list from `student,phd,researcher,professor,professional,lab,ai-security`. Default `student`. | Pitching the same topic to a class and to a review board. |
| `--pages` | Exact page count per PDF, 1–42. Default 8. | 8 is the full arc; fewer drops the middle deep-dives; more repeats the deep-dive archetype. |
| `--pdfs` | Styled PDF variants, 0 or more. Default 1. | `0` when you only want PNGs or a deck. |
| `--ppts` | Editable `.pptx` decks, with the prose as speaker notes. Default 0. | You will present it and want to edit slides. |
| `--images` | 300 DPI PNG page cards. Default 0. | Social cards, lecture slides, poster panels. |
| `--json-bundles` | Paper Lab JSON bundles (tabs + bibtex + rag_chunks). Default 0. | Feeding the primer back into a tool, or into `kx rag`. |
| `--watermark` | Running header/footer text. Default `Narmadya Innovations`. | Your own name or lab. |
| `--author` | Author line. Default `KinetixAI.AI`. | |
| `--engine` | Which model writes it. `auto` by default (or your `primer_engine` setting). `kx primers engines` lists all. | `outline` for a no-model draft; a named engine for reproducibility. |
| `--model`, `-m` | Model name for the chosen engine. | Pinning `qwen2.5:14b` instead of whatever is loaded. |
| `--api-base` | Override the engine's endpoint, e.g. `http://localhost:8000/v1`. | A vLLM server on another port or host. |
| `--mode` | `iterative` (default, one call per page) or `single` (one call for everything). | `single` only with a large model; small models produce sparse pages that way. |
| `--num-ctx` | Ollama context window, minimum 2048. | Ollama defaults to 4096, which the prompt alone fills. `kx` asks for 32768 unless you set this. |
| `--min-sections-per-page` | Density floor: numbered sections required per page, 1–9. Default 5. | Lowering it for a very small model; raising it for a reference sheet. |
| `--relevance` | Citation domain-match threshold, 0.0–1.0. Default 0.18. `0` disables filtering. | Lower it when a niche topic gets everything dropped; `0` when you want the raw search results. |
| `--verify-citations` | Re-resolve every citation and drop any that fail. | Publishing. It costs one network call per citation. |
| `--out-dir`, `-o` | Output folder. Default `./dist/primers`. | |
| `--copy-to` | Also copy the finished files here, keeping the `images/` subfolder shape. A permanent default can be set with `kx config set primer_copy_to <folder>`. | A notes folder, a shared drive, a repo. A failed copy is a warning, never a failed build. |
| `--json` | JSON on stdout: titles, page count, and every written path. | |

```bash
kx primers build --topic "Kernel fusion on GPUs" \
  --source ./notes/kernel-fusion.md \
  --engine outline --pages 3 --images 9 \
  --out-dir ./dist/kf
```

**Captured**

```
warning: arxiv unavailable (ProxyError)
warning: semantic unavailable (ProxyError)
warning: openalex unavailable (ProxyError)
warning: budget: page 1 'Kernel fusion on GPUs — overview': 86 words (min 225) — too thin
warning: budget: page 2 'Kernel fusion on GPUs — stage1': 72 words (min 225) — too thin
warning: budget: page 3 'Kernel fusion on GPUs — takeaways': 74 words (min 225) — too thin
built Kernel fusion on GPUs (3 pages, 0 verified citations)
   pdf  ./dist/kf/kernel-fusion-on-gpus.pdf
   png  ./dist/kf/images/kernel-fusion-on-gpus_page01_300dpi.png
   png  ./dist/kf/images/kernel-fusion-on-gpus_page02_300dpi.png
   png  ./dist/kf/images/kernel-fusion-on-gpus_page03_300dpi.png
  html  ./dist/kf/kernel-fusion-on-gpus.html  (editable)
```

Two things to read out of that run. `--images 9` produced **three**
PNGs, because the PDF has three pages — `--images N` is capped at the
page count. And the "too thin" warnings are exactly what `--engine
outline` is: it reorganises your source material into valid page
structure and invents nothing, so with a short `.md` the pages are
genuinely thin. That is the honest result, not a bug.

### `kx primers engines`

**For:** seeing every engine a Primer can be written with, and which are
ready right now.

**Data:** a static registry in the package, plus, with `--check`, a
2-second probe of each local server's model-list endpoint
(`/api/tags` for Ollama, `/models` for the OpenAI-shaped ones). The
probe doubles as the reachability test: a port answering with something
that is not a model list is not an inference server, and 8080 and 1234
are busy ports on a developer's machine. Hosted providers are never
probed; they are reported as ready purely on whether their API key
environment variable is set.

Embedding models are filtered out of the candidate list (`bge-`, `e5-`,
`minilm`, `nomic-embed`, …), because picking one would fail in a way
that looks like the engine being broken.

**Flow:** local HTTP only. No key is ever read from disk — keys come
from the environment, and nothing is logged.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--check` / `--no-check` | Probe local servers. On by default. | `--no-check` when you just want the list and do not want to wait. |
| `--json` | JSON on stdout: every row plus `{auto: {engine, why}}`. | |

```bash
kx primers engines --no-check
```

**Captured** (trimmed)

```
Local / open source
    engine     what it is             needs
?   ollama     Ollama                 not probed
?   vllm       vLLM                   not probed
?   llamacpp   llama.cpp server       not probed
?   lmstudio   LM Studio              not probed
?   jan        Jan                    not probed
?   localai    LocalAI                not probed
?   tgi        Hugging Face TGI       not probed
?   textgen    text-generation-webui  not probed
?   llamafile  llamafile              not probed

Hosted APIs
-   anthropic   Anthropic (Claude)       ANTHROPIC_API_KEY (unset)
-   openai      OpenAI (ChatGPT models)  OPENAI_API_KEY (unset)
-   gemini      Google Gemini            GEMINI_API_KEY (unset)
-   openrouter  OpenRouter               OPENROUTER_API_KEY (unset)
-   groq        Groq                     GROQ_API_KEY (unset)
-   together    Together AI              TOGETHER_API_KEY (unset)
-   deepseek    DeepSeek                 DEEPSEEK_API_KEY (unset)
-   mistral     Mistral                  MISTRAL_API_KEY (unset)
-   fireworks   Fireworks AI             FIREWORKS_API_KEY (unset)
-   cerebras    Cerebras                 CEREBRAS_API_KEY (unset)
-   xai         xAI (Grok)               XAI_API_KEY (unset)

No key needed
*   mcp      The calling agent (MCP sampling)  nothing
*   outline  No model (deterministic draft)    nothing

auto would use: outline  (no model engine found — writing a structural draft. ...)
Set a default: kx config set primer_engine <name>
```

`auto` tries local servers first (a local server costs nothing per page),
then hosted providers whose key is set, and falls back to `outline`.
Aliases are accepted everywhere: `claude` → `anthropic`, `lm-studio` →
`lmstudio`, `local-vllm` → `vllm`, `none`/`offline` → `outline`, `gpt`/`chatgpt`
→ `openai`, `google` → `gemini`, `grok` → `xai`.

### `kx primers autobuild`

**For:** building a whole catalog of primers from a YAML manifest.

**Data:** the manifest you point at (`primers_catalog.yaml` by default),
plus whatever each track's build needs.

**Flow:** the manifest is read once, turned into a list of build
requests, and those run on a thread pool. Each track's output folder is
`<defaults.out_dir>/<slug of topic>` unless the track sets its own
`out_dir`. A track that raises is caught and reported as `✗ <topic> —
<error>`; the other tracks still finish. Results are re-sorted back into
manifest order before printing, so parallel runs give a stable report.
The command exits 1 if any track failed.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--manifest` | The YAML file. Default `./primers_catalog.yaml`. A missing file exits 2. | |
| `--parallel` | Concurrent builds, 1–8. Default 2. | Raise it for `outline` or a hosted API; keep it at 1–2 for one local GPU, which serialises anyway. |
| `--json` | JSON on stdout: `{built, failed, tracks[]}`. | |

Manifest shape — a `defaults` mapping and a `tracks` list, where any
track key overrides the default:

```yaml
defaults:
  pages: 8
  pdfs: 1
  ppts: 1
  images: 8
  jsons: 1
  watermark: "KinetixAI.AI | NARMADYA INNOVATIONS"
  author: "KinetixAI.AI"
  engine: local-vllm
  model: "Qwen/Qwen2.5-14B-Instruct"
  verify_citations: true
  personas: [student, professional]
  out_dir: ./dist/primers

tracks:
  - topic: "Attention & Transformers"
    category: core-ai
    personas: [student, phd, professor]
  - topic: "Types of RAG"
    category: rag
```

Recognised keys are `pages`, `pdfs`, `ppts`, `images`, `jsons`,
`watermark`, `author`, `engine`, `model`, `verify_citations`,
`personas`, `out_dir`, `research_limit`, and per track also `topic`
(required), `source` and `category`. Note that the manifest key for JSON
bundles is **`jsons`**, not `json_bundles` — it does not match the
`--json-bundles` flag name. A track with no `topic` is a hard error
naming the track number.

```bash
kx primers autobuild --manifest ./mini.yaml --parallel 2
```

**Captured** (a two-track manifest with `engine: outline`, `pages: 2`)

```
✓ Kernel Fusion on GPUs
✓ Types of RAG
2/2 tracks built.
```

and the files:

```
dist/mini/kernel-fusion-on-gpus/images/kernel-fusion-on-gpus_page01_300dpi.png
dist/mini/kernel-fusion-on-gpus/images/kernel-fusion-on-gpus_page02_300dpi.png
dist/mini/kernel-fusion-on-gpus/kernel-fusion-on-gpus.html
dist/mini/kernel-fusion-on-gpus/kernel-fusion-on-gpus.json
dist/mini/kernel-fusion-on-gpus/kernel-fusion-on-gpus.pdf
dist/mini/types-of-rag/images/types-of-rag_page01_300dpi.png
dist/mini/types-of-rag/images/types-of-rag_page02_300dpi.png
dist/mini/types-of-rag/types-of-rag.html
dist/mini/types-of-rag/types-of-rag.json
dist/mini/types-of-rag/types-of-rag.pdf
```

---

## `kx convert`

**For:** turning something you already have — a deck, a paper, a repo, a
notes file — into the same 8-page Primer format.

It is `kx primers build` with the source as a positional argument and
the title derived from the source, so everything in the build pipeline
above applies. What differs is the ingestion, which depends on what you
hand it.

| Source | What is read | Where it comes from |
| --- | --- | --- |
| `.pptx` | Every slide: titles, every text frame, every table row (joined with ` \| `), **and the speaker notes**, labelled `(speaker notes)`. Slides are separated as `## Slide N`. | Local file, via `python-pptx`. No network. |
| `.pdf` | Text from the **first 60 pages**. | Local file, via `pypdf`. No network. |
| `.md`, `.markdown`, `.txt` | The whole file, read as UTF-8 with replacement for bad bytes. | Local file. No network. |
| GitHub URL | Repo metadata, then `README.md`. See below. | `api.github.com` and `raw.githubusercontent.com`. |
| arXiv id / DOI / arXiv URL | Title, author list and abstract, plus the full verified metadata record. | Same resolver as `kx get`. |

Any other file extension is an error naming the supported list. A string
that is not an existing file, not a GitHub URL and not an id is an error
too — `kx convert` never silently falls back to treating it as a topic.

**GitHub, specifically.** Two requests, both best-effort:

1. `GET https://api.github.com/repos/<owner>/<repo>` — full name,
   description, primary language and star count. If this does not return
   200 you get the warning `GitHub repo metadata unavailable (HTTP
   <code>)` and the build continues on the README alone.
2. `GET https://raw.githubusercontent.com/<owner>/<repo>/HEAD/README.md`
   — the README text. Only `README.md` at `HEAD` is tried: not
   `README.rst`, not `readme.md`, not `docs/`. If it is missing you get
   `no README.md found; using repo metadata only.`

So a repo with a `README.rst` and no reachable API gives the ingester
*nothing*, and you get a scaffolding-only primer. That is visible in the
warnings, not hidden.

### Flags

Everything from `primers build` except `--category`,
`--min-sections-per-page`, `--relevance` and `--verify-citations`, which
`convert` does not expose. Plus:

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `SOURCE` (positional) | **Required.** The file, URL or id. | Always. |
| `--style` | Output style preset. Default `trifleet-8page`. Recorded on the build request; there is currently only the one preset. | |
| `--topic`, `-t` | Title override. Default: the source's file stem with `_` and `-` turned into spaces, or the source string itself. | A repo called `fa2-kernels` deserves a better title. |
| `--persona` | Default here is `professional`, not `student`. | |
| `--out`, `-o` | An exact **PDF path**. Its folder receives the other artifacts. | You need `report.pdf`, not `my-repo.pdf`. |
| `--out-dir` | An output **folder**, when the exact PDF name does not matter. Same flag as `primers build`. | |
| `--copy-to` | As in `primers build`. | |

`--out` and `--out-dir` together exits 2. `--out` works by building to
the folder and then renaming the slug-named PDF; if an LLM engine
rewrites the primer title, the built file is named from the new title and
the rename silently does not happen (see the caveats at the end).

### Worked example — a GitHub repo into a primer

```bash
kx convert https://github.com/pallets/click \
  --engine outline --pages 4 --images 2 \
  --out-dir ./dist/click
```

**Captured**

```
ingesting source material…
searching verified sources (arXiv + Semantic Scholar + OpenAlex)…
building offline outline (no-LLM mode)…
rendering HTML…
compiling vector PDF…
exporting 2 page PNGs at 300 DPI…
warning: GitHub repo metadata unavailable (HTTP 403).
warning: arxiv unavailable (ProxyError)
warning: semantic unavailable (ProxyError)
warning: openalex unavailable (ProxyError)
warning: budget: page 1 'click — overview': 142 words (min 225) — too thin
warning: budget: page 2 'click — stage1': 85 words (min 225) — too thin
warning: budget: page 3 'click — evaluation': 107 words (min 225) — too thin
warning: budget: page 4 'click — takeaways': 114 words (min 225) — too thin
built click (4 pages, 0 verified citations)
   pdf  ./dist/click/click.pdf
   png  ./dist/click/images/click_page01_300dpi.png
   png  ./dist/click/images/click_page02_300dpi.png
  html  ./dist/click/click.html  (editable)
```

The HTTP 403 on the metadata call is this sandbox's proxy, not a `kx`
bug: `curl https://api.github.com/repos/pallets/click` also returns 403
here while `raw.githubusercontent.com` returns 200. On a normal network
you would get the description, language and star count too. The point
the run proves is the fallback: the README alone was enough to produce
all four pages and both PNGs.

**No-README case, captured** — `python/cpython` has a `README.rst`, so
the `README.md` fetch misses:

```bash
kx convert https://github.com/python/cpython \
  --engine outline --pages 2 --pdfs 0 --images 2 --out-dir ./dist/cpy
```

```
warning: GitHub repo metadata unavailable (HTTP 403).
warning: no README.md found; using repo metadata only.
...
built cpython (2 pages, 0 verified citations)
   png  ./dist/cpy/images/cpython_page01_300dpi.png
   png  ./dist/cpy/images/cpython_page02_300dpi.png
  html  ./dist/cpy/cpython.html  (editable)
```

Both a 403 and a missing README, and it still produced a bundle — just
an empty-scaffolding one. Note also that `--pdfs 0 --images 2` worked:
the PDF is built because PNGs are rasterized from it, then deleted
because you asked for zero PDFs. The folder keeps only the PNGs and the
HTML.

---

## `kx ask`

**For:** a grounded, cited answer from KinetixAI.

**Data:** the KinetixAI API. `POST <endpoint>/api/ask` with
`{"question": ...}`, returning `{answer, citations, grounded}`.

**Flow:** your question goes to `kinetixai.ai` (or whatever `endpoint`
is set to), with your bearer token if you have one. **This is the one
command where your text leaves the machine for KinetixAI.** Nothing is
written to disk.

**Account:** this is the only command that uses one. It will attempt the
call without a token, but a `401` comes back as "not signed in. Run
`kx login`." If you want a private, local alternative, that is `kx rag
ask`.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--json` | JSON on stdout. | |

```bash
kx ask "what does IO-awareness mean for attention kernels?"
```

**Illustrative** (`kinetixai.ai` blocked here)

```
IO-awareness means counting reads and writes between GPU HBM and on-chip
SRAM, not just FLOPs [1]. ...

Sources
  [1] FlashAttention (arXiv:2205.14135)
```

**Captured** — the failure shape when the endpoint is unreachable:

```
error: could not reach https://kinetixai.ai (ProxyError).
```

---

## `kx login`, `kx logout`, `kx whoami`

**For:** signing this machine in to a KinetixAI account. Needed only for
`kx ask`.

**Data:** `GET <endpoint>/api/auth/me` to validate a token and read your
email, name and plan; `DELETE <endpoint>/api/auth/tokens?id=current` to
have a token revoke itself.

**Where the token lives**, in order of preference:

1. `KX_TOKEN` in the environment — used if set, written nowhere.
2. The OS keyring (macOS Keychain, Windows Credential Manager, Secret
   Service on Linux) when the optional `keyring` package is installed and
   has a working backend.
3. `credentials.json` in the config directory, created owner-only
   (`0600`) from the first byte on POSIX.

Tokens are stored **per endpoint**, so a localhost development token is
never sent to `kinetixai.ai` and vice versa. The token is never printed
by any command. `kx` refuses to send a token over plain `http` to a
non-loopback host, and says so.

A token must look like `kx_` followed by 20–128 URL-safe characters, or
`kx login` rejects it before making any network call.

### `kx login`

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--token-stdin` | Read the token from stdin. | CI, or `pass show kx \| kx login --token-stdin`. |
| `--endpoint` | Override the configured endpoint for this call. | Logging in to a development server without changing your config. |
| `--json` | JSON on stdout: `{ok, endpoint, email, store, path}`. The token is not in it. | |

With no flags and no `KX_TOKEN`, it prompts with hidden input.

```bash
kx login
echo "$KX_TOKEN" | kx login --token-stdin
```

**Illustrative**

```
Logged in to https://kinetixai.ai as you@example.com.
Token saved in your OS keyring.
```

### `kx logout`

Revokes the token on the server *and* forgets it locally. If the revoke
call fails for any reason other than the token already being invalid, you
get a warning telling you to revoke it in the web UI — the local copy is
still deleted.

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--local` | Only forget it here; do not revoke it on the server. | The same token is in use on another machine. |
| `--endpoint` | Which endpoint's token. | |
| `--json` | JSON on stdout: `{ok, endpoint, revoked, removed}`. | |

If `KX_TOKEN` is set in your environment, logout warns you to unset it
too — it cannot delete an environment variable from inside the process.

### `kx whoami`

| Flag | What it does | When to reach for it |
| --- | --- | --- |
| `--endpoint` | Which endpoint to ask. | |
| `--json` | JSON on stdout: `{endpoint, user, token_source}`. `token_source` is `env`, `keyring`, `file` or `null`. | Debugging "which token is it actually using?" |

Exits 1 when there is no token, or when the token is no longer valid.

```bash
kx whoami
```

**Captured** (not logged in here)

```
error: not logged in. Create a token at https://kinetixai.ai/account/tokens/ and run `kx login`.
```

**Illustrative** (logged in)

```
you@example.com  (Your Name)
Plan: Free   Endpoint: https://kinetixai.ai
```

---

## `kx config`

**For:** viewing and changing settings.

**Data:** one JSON file, `<config dir>/config.json`. Written atomically
(temp file plus rename). Unknown keys already in the file are kept but
ignored. A value that fails validation on load is ignored with a warning
on stderr and the default is used — a broken config never stops `kx`
from running.

Two environment variables win over the file: `KINETIXAI_ENDPOINT` for
`endpoint`, and `OLLAMA_HOST` for `ollama_url` (and `OLLAMA_HOST` is
normalised: `0.0.0.0` becomes `localhost`, a bare host gets `:11434`, a
missing scheme gets `http://`).

### Settings

| Key | Default | What it does |
| --- | --- | --- |
| `endpoint` | `https://kinetixai.ai` | The API for `ask`, `login`, `logout`, `whoami`. Must be an `http(s)` URL. |
| `limit` | `10` | Default `-n` for `kx search`. Whole number 1–100. |
| `ollama_url` | `http://localhost:11434` | Where Ollama is. Must be an `http(s)` URL. |
| `ollama_model` | `llama3.1` | Model for `kx rag ask` answers. |
| `embed_model` | `nomic-embed-text` | Model for `kx rag index` embeddings. |
| `token_store` | `auto` | `auto` uses the keyring when available; `file` forces `credentials.json`. |
| `primer_engine` | `auto` | Default `--engine` for `primers build` and `convert`. Validated against the engine registry. |
| `primer_model` | `""` | Default `--model`. Can be set back to empty. |
| `primer_copy_to` | `""` | Default `--copy-to` folder. Must be a folder, not a file. Can be set back to empty. |
| `primer_watermark` | `Narmadya Innovations` | **Currently dead.** It validates and stores, but no command reads it; the watermark default is hardcoded in the flag. |

The help text for `kx config set` lists only `endpoint`, `limit`,
`ollama_url`, `ollama_model`, `embed_model`, `token_store`. The four
`primer_*` keys above are also settable; the help text is just out of
date.

| Subcommand | Flags | Notes |
| --- | --- | --- |
| `kx config show` | `--json` | Prints every setting and the config file path. |
| `kx config get KEY` | `--json` | One setting. |
| `kx config set KEY VALUE` | `--json` | Validates first; a bad value exits 2 with a message saying what is allowed. |

```bash
kx config show
kx config get endpoint
kx config set limit 25
kx config set primer_engine ollama
kx config set primer_copy_to ~/Notes/primers
```

**Captured**

```
endpoint = https://kinetixai.ai
limit = 10
ollama_url = http://localhost:11434
ollama_model = llama3.1
embed_model = nomic-embed-text
primer_engine = lmstudio
primer_model =
primer_watermark = Narmadya Innovations
primer_copy_to =
/root/.config/kinetixai/config.json
```

---

## `kx mcp`

**For:** running `kx` as an MCP server, so an agent (Claude Code, Claude
Desktop, Cursor, Goose, Cline, Continue, OpenHands, Open WebUI) can call
it instead of you typing commands.

**Data:** the same `services` layer the CLI uses, so every tool hits
exactly the same sources with exactly the same behaviour. Transport is
stdio. The agent launches `kx mcp serve` as a subprocess; there is no
port and no listening socket.

**Ten tools:**

| Tool | Maps to | Arguments |
| --- | --- | --- |
| `search_papers` | `kx search` | `query`, `limit` (1–50) |
| `get_paper` | `kx get` | `id` |
| `find_code` | `kx code` | `paper`, `limit` |
| `find_models` | `kx models` | `query`, `limit`, `sort` |
| `find_datasets` | `kx datasets` | `query`, `limit`, `sort` |
| `library_search` | `kx library search` | `query`, `limit` |
| `rag_ask` | `kx rag ask` | `question`, `k` (1–20), `use_llm` |
| `search_verified_papers` | the Primer domain filter, standalone | `topic`, `limit`. Returns `papers` plus a `dropped` note per rejected result. |
| `build_primer` | `kx primers build` | `topic`, `pages`, `personas`, `out_dir`, `engine`, `model`, `pdfs`, `images`, `ppts`, `json_bundles`, `watermark` |
| `convert_document` | `kx convert` | `source`, `topic`, plus the same build arguments |

The interesting part is the **`mcp` engine**. For `build_primer` and
`convert_document` the default `engine` is `"mcp"`, which means `kx`
does not call any model itself — it asks the host agent to write each
page via MCP sampling. So the user needs no API key and no local GPU: the
agent they are already talking to does the writing. Sampling is optional
in the MCP protocol and many clients do not implement it; when it is
unavailable `kx` says so rather than failing mid-build, and tells you to
use `--engine ollama`, `vllm`, `claude`, `gemini` or `outline` instead.
`--engine mcp` from a normal shell always fails, with that same message.

| Subcommand / flag | What it does | When to reach for it |
| --- | --- | --- |
| `kx mcp serve` | Run the server on stdio. | What the agent client runs. You rarely type it. |
| `kx mcp` (no subcommand) | Same as `serve`. | |
| `kx mcp install [CLIENT]` | Print the config that registers `kx` with a client. `claude-code` (default), `claude-desktop`, `cursor`, `goose`, `cline`, `continue`, `openhands`, `open-webui`. | Setting it up. It prints; it does not edit your files. |
| `kx mcp --config CLIENT` | The same snippet, from the group callback. | |
| `kx mcp info` / `kx mcp --info` | List the tools and whether the `mcp` package is installed. | Checking the install before debugging the agent. |
| `--json` | JSON on stdout, on `install` and `info`. | |

```bash
kx mcp info
kx mcp install claude-code
kx mcp install claude-desktop
```

**Captured**

```
Tools: search_papers, get_paper, find_code, find_models, find_datasets,
library_search, rag_ask, search_verified_papers, build_primer, convert_document
mcp package installed: True
```

```
# Claude Code (CLI): terminal
claude mcp add kx -- /usr/local/bin/kx mcp serve
# Then `claude mcp list` to verify. Ask Claude Code to build a primer and it will call kx directly.
```

```
# Claude Desktop: ~/.config/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "kinetixai": {
      "command": "/usr/local/bin/kx",
      "args": [
        "mcp",
        "serve"
      ]
    }
  }
}
# Merge into mcpServers, then restart Claude Desktop.
```

The snippet uses the absolute path to the `kx` that printed it, which
matters when `kx` lives in a venv or a pipx shim that the agent's
environment does not have on `PATH`.

---

# End-to-end integration recipes

Each of these is a complete flow. Where a recipe was really run, it says
so.

## 1. A GitHub repository into a primer

**Verified by running it.** GitHub is reachable from the sandbox this
reference was written in.

```bash
kx convert https://github.com/pallets/click \
  --engine outline --pages 4 --images 2 \
  --out-dir ./dist/click
```

What actually happens, in order:

1. `kx` parses `owner/repo` out of the URL.
2. `GET https://api.github.com/repos/pallets/click`. On 200 it keeps the
   full name, description, primary language and star count as the first
   block of source text. On anything else: the warning `GitHub repo
   metadata unavailable (HTTP <code>)`, and on it goes.
3. `GET https://raw.githubusercontent.com/pallets/click/HEAD/README.md`.
   On 200 the whole README becomes the bulk of the source text. On
   anything else: the warning `no README.md found; using repo metadata
   only.` **Only `README.md` at `HEAD` is tried** — a project with
   `README.rst` lands in the no-README path.
4. Research runs on the derived topic (`click`, from the URL's last
   segment) against arXiv, Semantic Scholar and OpenAlex. With `--engine
   outline` the found papers become a "Verified sources" table on page 1.
5. `outline` splits the README into page-sized sections across the
   4-page blueprint. No model is called and nothing is invented.
6. HTML is rendered, WeasyPrint compiles the A4 PDF, and pypdfium2
   rasterizes the first 2 pages to 300 DPI PNGs.

The captured output and the no-README case are both shown under
`kx convert` above.

**Controlling page count and images.** `--pages N` sets the exact page
count, 1–42. `--images N` exports the first N pages as PNGs and is
**capped at the page count** — `--pages 3 --images 9` gives three PNGs,
verified. To get N images and no PDF, pass `--pdfs 0 --images N`: the
PDF is still compiled internally (the PNGs are rasterized from it) and
then deleted. Verified.

**If the repo gives nothing.** You get a scaffolding-only primer, every
page flagged "too thin" in the warnings. Fix it by giving the builder
real material (`--source` a design doc instead, or point at a repo with
a `README.md`) or by using a real engine so the model can write from the
topic.

**With a model instead of `outline`:**

```bash
export ANTHROPIC_API_KEY=...
kx convert https://github.com/pallets/click \
  --engine claude --pages 8 --images 8 --ppts 1 \
  --out ./out/click-primer.pdf
```

## 2. An arXiv id or DOI into a primer, and how the paper becomes citation #1

**Read from the code, not run** — arXiv, Semantic Scholar and OpenAlex
are blocked in this sandbox.

```bash
kx convert 2205.14135 --engine ollama -m qwen2.5:14b \
  --pages 8 --images 8 --json-bundles 1 \
  --out-dir ./dist/flashattention
```

The citation ordering is not a heuristic, it is the code:

1. Ingestion resolves the identifier with the same resolver as `kx get`
   — arXiv first for an arXiv id, Semantic Scholar then OpenAlex for a
   DOI — and keeps the **full verified metadata record** in
   `material.papers`, not just the text.
2. The source text handed to the model is the title, the author list and
   the abstract.
3. Research then runs separately on the topic and returns its own list.
4. The two are concatenated **source papers first**:
   `sources = material.papers + researched`.
5. Citations are `sources[:15]`.

So the paper you named is citation **#1**, by construction, and it never
has to survive the domain filter — the filter only ever runs on the
researched papers. Add `--verify-citations` to re-resolve all fifteen and
drop any that no longer resolve; each drop is reported.

If the identifier cannot be resolved at all, `kx convert` fails with
`could not resolve '<id>' on arXiv/Crossref.` rather than quietly
treating the string as a topic.

The matching `primers build` form, when you want your own title:

```bash
kx primers build --topic "FlashAttention: IO-aware exact attention" \
  --source 2205.14135 --category gpu \
  --persona student,professional --pages 8 --verify-citations
```

## 3. A `.pptx` deck into an 8-page PDF, speaker notes included

**Verified by running it**, with a two-slide deck built locally.

```bash
kx convert ./deck.pptx --engine outline --pages 8 --out-dir ./dist/deck
```

**Captured** (tail)

```
warning: budget: page 4 'deck — stage2': 83 words (min 225) — too thin
...
built deck (8 pages, 0 verified citations)
   pdf  ./dist/deck/deck.pdf
  html  ./dist/deck/deck.html  (editable)
```

The speaker notes really are read. The deck's notes text appeared in the
rendered HTML, checked directly. What the ingester extracts per slide:

- a `## Slide N` marker,
- every paragraph of every text frame,
- every table row, cells joined with ` | `,
- the notes slide's text, prefixed `(speaker notes) `.

That means the thinking you put in the notes — "we page on symptoms, not
causes" — reaches the model, which is usually where the real content of a
deck lives.

Round-tripping back out to a deck:

```bash
kx convert ./deck.pptx --engine ollama --pages 8 --ppts 1 --pdfs 1 \
  --out-dir ./dist/deck
```

The exported `.pptx` is 16:9, one slide per primer page, with the page's
prose and takeaway written into the slide's own speaker notes.

Needs the `primers` extra for `python-pptx`; without it the ingester
says `reading .pptx needs python-pptx: pip install 'kinetixai[primers]'`.

## 4. A local `.pdf` or `.md` into a primer

**Verified by running it** with a `.md` source.

```bash
kx primers build --topic "Kernel fusion on GPUs" \
  --source ./notes/kernel-fusion.md \
  --engine outline --pages 3 --images 9 --out-dir ./dist/kf
```

The captured output is under `kx primers build` above. The `.md` is read
whole; a `.pdf` is read with `pypdf` and **truncated at 60 pages** — so a
200-page thesis contributes only its first 60. If that matters, split the
PDF first.

The difference between `convert` and `build --source` is only the title
and the default persona:

```bash
# title derived from the filename, persona 'professional'
kx convert ./thesis.pdf --pages 8

# your title, persona 'student'
kx primers build --topic "Thesis in eight pages" --source ./thesis.pdf --pages 8
```

A `.pdf` source needs `pypdf` (the `rag` extra). Both still run research
against the three paper sources, so a local PDF can still come out with
real citations attached.

## 5. A topic with no source at all

**Verified by running it.**

```bash
kx primers build --topic "Paging and page faults for SREs" \
  --engine outline --pages 2 --out-dir ./dist/paging
```

**Captured**

```
warning: arxiv unavailable (ProxyError)
warning: semantic unavailable (ProxyError)
warning: openalex unavailable (ProxyError)
warning: budget: page 1 'Paging and page faults for SREs — overview': 110 words (min 225) — too thin
warning: budget: page 2 'Paging and page faults for SREs — takeaways': 72 words (min 225) — too thin
built Paging and page faults for SREs (2 pages, 0 verified citations)
   pdf  ./dist/paging/paging-and-page-faults-for-sres.pdf
  html  ./dist/paging/paging-and-page-faults-for-sres.html  (editable)
```

With no source, `outline` has nothing to reorganise, so it writes
scaffolding and says so on the page ("Draft — verify before
publishing!"). Topic-only is the case that genuinely needs a model:

```bash
kx primers build --topic "Paging and page faults for SREs" \
  --persona professional --engine ollama -m qwen2.5:14b \
  --num-ctx 32768 --pages 8 --images 8
```

Research still runs on the topic, so the primer can carry real citations
even with no source file. If the topic contains an ambiguous word and
every result gets dropped, you will see `no source passed domain
filtering` and the advice to lower `--relevance`.

## 6. Exactly N images at 300 DPI

**Verified by running it.**

```bash
kx convert https://github.com/python/cpython \
  --engine outline --pages 2 --pdfs 0 --images 2 --out-dir ./dist/cpy
```

**Captured file list**

```
dist/cpy/cpython.html
dist/cpy/images/cpython_page01_300dpi.png
dist/cpy/images/cpython_page02_300dpi.png
```

The three things to know:

- **Filenames** are `<slug>_page<NN>_300dpi.png`, zero-padded to two
  digits, in an `images/` subfolder of the output folder. The slug is the
  primer title lowercased with every run of non-alphanumeric characters
  turned into `-`, truncated to 60 characters.
- **`--images N` caps at the page count.** Verified: `--pages 3
  --images 9` produced three PNGs, not nine. It renders pages 1..min(N,
  pages).
- **Images without `--pdfs` works.** Verified: `--pdfs 0 --images 2`
  produced the two PNGs and no PDF. The PDF is compiled internally
  because pypdfium2 rasterizes *from* it, then deleted. So you still
  need WeasyPrint installed even when you want no PDF.

300 DPI is fixed in the exporter; there is no `--dpi` flag. A page from
the run above measures 2481 × 3508 pixels with the DPI tag set to 300,
checked with Pillow.

Needs `pypdfium2` and `pillow` from the `primers` extra. Preflight
checks for them before any model runs.

## 7. A whole catalog from `primers_catalog.yaml`, in parallel

**Verified by running it** with a cut-down two-track manifest.

```bash
kx primers autobuild --manifest ./primers_catalog.yaml --parallel 4
```

What happens: the manifest is parsed once into one build request per
track, `defaults` merged under each track's own keys. Each track goes to
`<defaults.out_dir>/<slug of topic>` unless it sets `out_dir`. The
requests run on a thread pool of `--parallel` workers (1–8). A track that
throws becomes `✗ <topic> — <ErrorClass>: <message>` and the rest carry
on. Results are sorted back into manifest order before printing. Exit 1
if any track failed.

**Captured**, with `engine: outline`, `pages: 2`, `images: 2`,
`jsons: 1` and two tracks:

```
✓ Kernel Fusion on GPUs
✓ Types of RAG
2/2 tracks built.
```

```
dist/mini/kernel-fusion-on-gpus/images/kernel-fusion-on-gpus_page01_300dpi.png
dist/mini/kernel-fusion-on-gpus/images/kernel-fusion-on-gpus_page02_300dpi.png
dist/mini/kernel-fusion-on-gpus/kernel-fusion-on-gpus.html
dist/mini/kernel-fusion-on-gpus/kernel-fusion-on-gpus.json
dist/mini/kernel-fusion-on-gpus/kernel-fusion-on-gpus.pdf
dist/mini/types-of-rag/...
```

Choosing `--parallel`: one local GPU serialises anyway, so 1–2 is right
for Ollama or vLLM. A hosted API or `outline` can take 4–8. Each worker
is a full build, so eight workers means eight WeasyPrint compilations at
once — on a small machine that is the bottleneck, not the model.

Two traps in the manifest format, both confirmed by reading
`catalog.py`:

- The JSON-bundle key is **`jsons`**, not `json_bundles`. It does not
  match the `--json-bundles` flag.
- `autobuild` has no `--out-dir`, `--engine` or `--pages`. Everything
  comes from the manifest. There is no way to override a manifest value
  from the command line.

The watermark default differs too: `autobuild` falls back to `Rohit
Patel | Narmadya Innovations` when the manifest sets none, where
`primers build` defaults to `Narmadya Innovations`.

## 8. Driving all of it from an agent over MCP

**Partly verified:** `kx mcp info` and `kx mcp install` were run; the
agent side was not exercised here.

```bash
# one-time setup
kx mcp install claude-code
claude mcp add kx -- /usr/local/bin/kx mcp serve
claude mcp list
```

For Claude Desktop, Cursor, Goose, Cline, Continue, OpenHands or Open
WebUI, run `kx mcp install <client>` and merge the snippet it prints into
that client's config file — the command names the file in a comment and
does not edit anything itself.

Then you stop typing commands. In the agent:

> Find the three most-cited papers on IO-aware attention, get the
> official implementation of the top one, and build me an eight-page
> primer on it with 300 DPI page images.

The agent will call `search_papers`, then `find_code`, then
`convert_document` or `build_primer`. Three things make this work
properly:

- **`search_verified_papers`** is the domain-filtered search, exposed as
  its own tool. It returns the kept papers *and* a `dropped` note for
  every rejected result, so the agent can tell the user why a citation
  list is short instead of guessing.
- **`engine: "mcp"` is the default** for `build_primer` and
  `convert_document`. The agent itself writes the pages via MCP
  sampling, so the user needs no API key and no GPU. If the client does
  not implement sampling, `kx` returns a clear message naming the
  alternatives rather than failing halfway through a build.
- **`rag_ask` stays local.** The agent can query the user's library, and
  retrieval never leaves the machine (and the answer does not either,
  unless Ollama is pointed somewhere remote).

The tools write files to `out_dir`, relative to wherever the agent
launched the server, and return the paths.

## 9. One research session: search → get → library → rag → ask

**Partly verified:** the `library`, `cite`, `export`, `rag index`, `rag
status` and `rag ask` steps were all really run (offline, via
`--from-file`). The `search` and `get` steps need arXiv, which is
blocked here.

```bash
# 1. find the work. Three sources, merged and de-duplicated.
kx search "IO-aware attention kernels" -n 5

# 2. read one properly, and pull the PDF.
kx get 2205.14135
kx get 2205.14135 --pdf -o ~/papers

# 3. keep it, with the PDF, tagged.
kx library add 2205.14135 1706.03762 --pdf -t attention
kx library list

# 4. index it. Library abstracts + PDF full text + your own notes folder.
kx rag index --path ~/notes

# 5. ask your own papers, on your own machine.
kx rag ask "what memory hierarchy assumption does the tiling rely on?" -k 5

# 6. and when you want the public answer instead of your own, with an account:
kx ask "how does FlashAttention-2 differ from FlashAttention?"

# 7. write it up.
kx cite 1
kx library export -f bibtex -o refs.bib
```

How the data moves through that:

| Step | Network | Disk |
| --- | --- | --- |
| `search` | 3 GETs out | nothing written |
| `get` | 1–2 GETs out | with `--pdf`, one PDF |
| `library add` | 1 resolve per id, plus PDF downloads | `library.json`, `pdfs/` |
| `rag index` | **none**, unless Ollama is remote | `rag/index.json` |
| `rag ask` | **none**, unless Ollama is remote | nothing written |
| `ask` | your question to `kinetixai.ai` | nothing written |
| `cite`, `library export` | none (library hit) | the file you name |

Everything up to step 5 needs no account. Step 6 is the only one that
does.

**Captured**, for steps 3–7 run offline from a local JSON import:

```
added arxiv:2205.14135  FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
Indexed 2 documents, 2 chunks (BM25 keyword search).
2 documents, 2 chunks, built 2026-11-09T06:51:52Z, BM25 only
Ollama is not running at http://localhost:11434, so here are the best passages.
Private: retrieval and answer ran on this machine; nothing was sent anywhere.
wrote 1 entries to /tmp/kxdemo/lib.bib
```

Closing the loop: a primer's JSON bundle carries a `rag_chunks` array and
a `bibtex` string, so a primer you built can be read back into the
library and the index:

```bash
kx primers build --topic "Types of RAG" --json-bundles 1 --out-dir ./dist/rag
kx rag index --path ./dist/rag
```

---

# What leaves your machine

A single table, because this is the question people actually ask.

| Command | Goes out to | What is sent |
| --- | --- | --- |
| `version`, `config`, `cite` (library hit), `library list/search/remove/export`, `library add --from-file` | nothing | — |
| `rag index`, `rag ask`, `rag status` | nothing, if Ollama is on localhost | chunk text / your question, to your own Ollama |
| `doctor` | 6 endpoints | a user-agent string; no token |
| `search`, `get`, `scholar` | arXiv, Semantic Scholar, OpenAlex, ORCID | your query or identifier |
| `code`, `models`, `datasets` | Hugging Face, GitHub | your query or identifier; the env token if set |
| `primers`, `convert` | the three paper sources, plus GitHub for a repo source, plus the chosen engine | your topic as search queries; the prompt and source text to the engine you picked |
| `ask`, `login`, `logout`, `whoami` | your configured `endpoint` | your question / your token |

`--engine outline` calls no model at all. `--engine ollama` with a
localhost URL keeps the text on your machine. A hosted engine sends the
prompt and your source material to that provider.

---

# Things that do not match the help text

Found while verifying this reference, all by reading the source and
confirming at the command line where possible. Listed so you are not
surprised; none of them are fixed in this document.

1. **`primers convert` does not exist.** `convert` is top-level: `kx
   convert`, not `kx primers convert`. The `primers` group has `build`,
   `engines`, `autobuild`.

2. **`kx config set --help` under-lists the settings.** It names six
   keys. Four more are real and validated: `primer_engine`,
   `primer_model`, `primer_copy_to`, `primer_watermark`.

3. **`primer_watermark` is dead.** `kx config set primer_watermark X`
   validates and stores, and `kx config show` prints it, but no command
   reads it. The watermark comes from the `--watermark` flag's hardcoded
   default. Confirmed by grep: the only non-config reference to it is the
   `DEFAULTS` dict itself.

4. **`kx doctor` raises `NameError` when the `primers` extra is
   missing.** In `doctor.advice()` the primers hint appends to `tips`,
   a name that does not exist, instead of `out`. Not triggered in this
   sandbox because `jinja2` is installed. A user without the extra —
   exactly the user that hint is for — gets a traceback instead.

5. **Two commands are undocumented in the brief but real and useful:**
   `kx library search` and `kx rag status`. Both are documented above.

6. **`kx convert --out` can silently not rename.** It builds to the
   folder and then renames `<slug>.pdf` to your path, where the slug
   comes from the *requested* title. If an LLM engine rewrites the
   primer title, the built file is named from the new title, the rename
   finds nothing, and you get `<new-slug>.pdf` in the folder with no
   error and no `saved` line. With `--engine outline` the title is never
   rewritten, so this does not arise.

7. **The autobuild manifest key is `jsons`, not `json_bundles`.** It does
   not match the CLI flag name, and a manifest using `json_bundles` is
   silently ignored (unknown keys are not an error).

8. **`autobuild` has no command-line overrides.** No `--engine`,
   `--pages` or `--out-dir`. If you want to batch-build the same catalog
   with a different engine, you edit the manifest.

9. **GitHub ingestion is `README.md`-only, at `HEAD`.** Not
   `README.rst`, not `readme.md`, not a default-branch name. Verified:
   `python/cpython` falls into the no-README path.

10. **`kx code` without Hugging Face loses the official-repo marker
    silently-ish.** You get a warning that HF is unreachable, but the
    resulting list is "most starred", not "the authors' repo first" —
    which is easy to misread as the same thing.

One genuine surprise, worth calling out separately: `--pdfs 0 --images
2` still requires WeasyPrint, because the PNGs are rasterized from a PDF
that is compiled and then deleted. Asking for "no PDF" does not remove
the PDF dependency.
