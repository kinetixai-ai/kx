"""Read a GitHub repository deeply enough to teach from it.

The first version of this fetched a repo's description and its README,
which is enough to say what a project is and nothing like enough to
explain how it works. A primer that claims to be a deep dive has to be
built from the actual source: the directory layout, the entry points,
the modules that matter for the topic asked about.

So this module walks the tree, scores every file against the topic and
against how much it usually explains, reads the ones worth reading
within a byte budget, and hands back one ordered document: metadata,
the layout, then the files themselves. A subdirectory URL is honoured,
because `/tree/main/DSA%26T` means "this folder", not "this repo".

Private repositories work when GITHUB_TOKEN (or GH_TOKEN) is set; the
token is sent to api.github.com and raw.githubusercontent.com only.
"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from urllib.parse import unquote

import httpx

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"

# Worth reading, by extension.
CODE = {".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt",
        ".c", ".h", ".cc", ".cpp", ".hpp", ".cu", ".cuh", ".rb", ".php", ".swift",
        ".scala", ".sh", ".bash", ".sql", ".r", ".jl", ".lua", ".pl", ".ex", ".exs"}
DOCS = {".md", ".markdown", ".rst", ".txt", ".adoc", ".org"}
NOTEBOOK = {".ipynb"}
CONFIG = {".yaml", ".yml", ".toml", ".cfg", ".ini", ".json"}
READABLE = CODE | DOCS | NOTEBOOK | CONFIG
NAMED = {"dockerfile", "makefile", "cmakelists.txt", "justfile", "procfile"}

# Never worth the budget.
SKIP_DIRS = {"node_modules", ".git", ".github/workflows/cache", "dist", "build",
             "vendor", "venv", ".venv", "env", "__pycache__", "site-packages",
             ".next", ".nuxt", "target", "coverage", ".pytest_cache", ".mypy_cache",
             ".idea", ".vscode", "bower_components", "third_party", "externals",
             ".terraform", "htmlcov", "egg-info", ".tox"}
SKIP_FILES = {"package-lock.json", "yarn.lock", "poetry.lock", "pnpm-lock.yaml",
              "cargo.lock", "composer.lock", "gemfile.lock", "go.sum",
              "requirements.lock", "pdm.lock", "uv.lock"}
SKIP_SUFFIX = (".min.js", ".min.css", ".map", ".lock", ".svg", ".png", ".jpg",
               ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".whl", ".so", ".dylib",
               ".dll", ".bin", ".pt", ".onnx", ".safetensors", ".parquet", ".csv")

# Files that explain a project out of proportion to their size.
EXPLAINS = ("readme", "architecture", "design", "overview", "concepts", "guide",
            "tutorial", "docs/", "doc/", "contributing", "main", "index", "app",
            "__init__", "cli", "core", "engine", "pipeline", "model", "schema")
LESS_USEFUL = ("test", "spec", "fixture", "mock", "example", "sample", "benchmark",
               "migration", "locale", "i18n", "__snapshot")

MAX_FILE_BYTES = 120_000        # one file
DEFAULT_BUDGET = 140_000        # all files together, roughly 35k tokens
DEFAULT_MAX_FILES = 40


class RepoError(RuntimeError):
    pass


@dataclass
class RepoRef:
    owner: str
    repo: str
    ref: str = ""               # branch, tag or sha; empty means default branch
    path: str = ""              # subdirectory or single file, "" means whole repo
    is_file: bool = False

    @property
    def slug(self) -> str:
        base = f"{self.owner}/{self.repo}"
        return f"{base}/{self.path}" if self.path else base


_URL = re.compile(
    r"github\.com[/:]"
    r"(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?"
    r"(?:/(?P<kind>tree|blob)/(?P<ref>[^/]+)(?:/(?P<path>.*))?)?"
    r"/?(?:[?#].*)?$")


def parse_url(url: str) -> RepoRef | None:
    """owner/repo plus the branch and subpath, when the URL carries them."""
    m = _URL.search((url or "").strip())
    if not m:
        return None
    return RepoRef(
        owner=m.group("owner"),
        repo=m.group("repo"),
        ref=unquote(m.group("ref") or ""),
        path=unquote(m.group("path") or "").strip("/"),
        is_file=m.group("kind") == "blob",
    )


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(client: httpx.Client, url: str, headers: dict, params: dict | None = None):
    """GET, and retry without the token if the token is what was rejected.

    A stale or wrong GITHUB_TOKEN in a shell would otherwise make PUBLIC
    repositories unreadable: raw.githubusercontent.com answers 404 to a bad
    Authorization header, which looks exactly like a missing file.
    """
    r = client.get(url, headers=headers, params=params)
    if r.status_code in (401, 403, 404) and "Authorization" in headers:
        bare = {k: v for k, v in headers.items() if k != "Authorization"}
        retry = client.get(url, headers=bare, params=params)
        if retry.status_code == 200:
            retry.kx_token_ignored = True     # noqa: B010 — read by callers below
            return retry
    return r


def _is_skipped(path: str) -> bool:
    lower = path.lower()
    parts = lower.split("/")
    if any(p in SKIP_DIRS for p in parts):
        return True
    if any(seg.endswith(".egg-info") for seg in parts):
        return True
    if parts[-1] in SKIP_FILES:
        return True
    return lower.endswith(SKIP_SUFFIX)


def _is_readable(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    if name in NAMED:
        return True
    dot = name.rfind(".")
    return dot > 0 and name[dot:] in READABLE


def score(path: str, size: int, topic_terms: set[str]) -> float:
    """How much this file is likely to teach about the topic."""
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    s = 0.0

    from .search import terms as split_terms
    overlap = topic_terms & split_terms(lower.replace("/", " ").replace("_", " "))
    s += 3.0 * len(overlap)

    for i, hint in enumerate(EXPLAINS):
        if hint in lower:
            s += 2.5 - (i * 0.05)       # earlier hints matter slightly more
            break
    if name.startswith("readme"):
        s += 4.0
    if lower.endswith((".md", ".rst")):
        s += 1.5
    if any(bad in lower for bad in LESS_USEFUL):
        s -= 2.0
    s -= 0.8 * lower.count("/")          # shallow files tend to be entry points
    if size < 200:
        s -= 1.5                         # a stub teaches nothing
    elif size > 60_000:
        s -= 1.0                         # a giant file eats the whole budget
    return s


@dataclass
class RepoDigest:
    ref: RepoRef
    metadata: dict = field(default_factory=dict)
    tree: list[tuple[str, int]] = field(default_factory=list)   # (path, bytes)
    files: list[tuple[str, str]] = field(default_factory=list)  # (path, text)
    warnings: list[str] = field(default_factory=list)

    def to_text(self, max_tree: int = 300) -> str:
        """One document: what the project is, how it is laid out, then the code."""
        out: list[str] = []
        md = self.metadata
        head = [f"REPOSITORY {self.ref.slug}"]
        if md.get("description"):
            head.append(md["description"])
        facts = [f"{k}: {md[k]}" for k in ("language", "stars", "license", "updated")
                 if md.get(k)]
        if md.get("topics"):
            facts.append("topics: " + ", ".join(md["topics"][:12]))
        if facts:
            head.append(" | ".join(facts))
        out.append("\n".join(head))

        if self.tree:
            lines = [f"  {p}  ({s:,} bytes)" for p, s in self.tree[:max_tree]]
            more = len(self.tree) - len(lines)
            out.append("LAYOUT (%d files read in full below)\n%s%s"
                       % (len(self.files), "\n".join(lines),
                          f"\n  ... and {more} more files" if more > 0 else ""))

        for path, text in self.files:
            out.append(f"--- FILE: {path} ---\n{text}")
        return "\n\n".join(out)


def _notebook_text(raw: str) -> str:
    """Markdown and code from a notebook, without the outputs."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source")
        body = "".join(src) if isinstance(src, list) else str(src or "")
        if not body.strip():
            continue
        kind = cell.get("cell_type")
        parts.append(body if kind == "markdown" else f"```\n{body}\n```")
    return "\n\n".join(parts)


def fetch(url: str, topic: str, client: httpx.Client,
          max_files: int = DEFAULT_MAX_FILES,
          budget: int = DEFAULT_BUDGET) -> RepoDigest:
    """Read a repository (or one folder of it) deeply enough to teach from."""
    ref = parse_url(url)
    if ref is None:
        raise RepoError(f"'{url}' is not a GitHub repository URL.")
    digest = RepoDigest(ref=ref)
    headers = _headers()

    r = _get(client, f"{API}/repos/{ref.owner}/{ref.repo}", headers)
    if getattr(r, "kx_token_ignored", False):
        digest.warnings.append(
            "GITHUB_TOKEN was rejected, so the repository was read anonymously. "
            "Check or unset the token to reach private repositories.")
        headers = {k: v for k, v in headers.items() if k != "Authorization"}
    if r.status_code == 404:
        raise RepoError(
            f"{ref.owner}/{ref.repo} not found. If it is private, set GITHUB_TOKEN "
            "to a token with repo read access.")
    if r.status_code in (401, 403):
        detail = "rate limited" if "rate limit" in r.text.lower() else "not authorised"
        digest.warnings.append(
            f"GitHub API {detail} (HTTP {r.status_code}); reading what is reachable "
            "without it. Set GITHUB_TOKEN to raise the limit or reach a private repo.")
        meta = {}
    else:
        j = r.json()
        meta = {"description": j.get("description"), "language": j.get("language"),
                "stars": j.get("stargazers_count"), "topics": j.get("topics") or [],
                "updated": (j.get("pushed_at") or "")[:10],
                "license": ((j.get("license") or {}).get("spdx_id") or None),
                "default_branch": j.get("default_branch")}
    digest.metadata = {k: v for k, v in meta.items() if v}
    branch = ref.ref or meta.get("default_branch") or "HEAD"

    # A single file URL: just that file.
    if ref.is_file and ref.path:
        text = _read_blob(client, ref, branch, ref.path, headers, digest)
        if text:
            digest.files.append((ref.path, text))
            digest.tree.append((ref.path, len(text)))
        return digest

    entries = _list_tree(client, ref, branch, headers, digest)
    if not entries:
        # Tree listing needs the API, which a restricted network or a rate
        # limit can block while raw file access still works. Falling back to
        # the documents a repo almost always has beats returning nothing.
        _read_known_docs(client, ref, branch, headers, digest)
        return digest

    prefix = (ref.path + "/") if ref.path else ""
    if prefix:
        scoped = [(p, s) for p, s in entries if p.startswith(prefix)]
        if not scoped:
            digest.warnings.append(
                f"nothing found under '{ref.path}' on branch {branch}; "
                "reading the whole repository instead.")
        else:
            entries = scoped

    digest.tree = sorted(entries)
    from .search import terms as split_terms
    topic_terms = split_terms(topic)
    ranked = sorted(entries, key=lambda e: -score(e[0], e[1], topic_terms))

    spent = 0
    for path, size in ranked:
        if len(digest.files) >= max_files or spent >= budget:
            break
        text = _read_blob(client, ref, branch, path, headers, digest)
        if not text:
            continue
        text = text[: min(MAX_FILE_BYTES, budget - spent)]
        digest.files.append((path, text))
        spent += len(text)

    if not digest.files:
        digest.warnings.append(
            "no readable source files found; using repository metadata only.")
    return digest


KNOWN_DOCS = ("README.md", "README.rst", "README.txt", "README", "readme.md",
              "docs/README.md", "docs/index.md", "ARCHITECTURE.md", "DESIGN.md",
              "CONTRIBUTING.md", "docs/overview.md", "pyproject.toml",
              "package.json", "Cargo.toml", "go.mod")


def _read_known_docs(client: httpx.Client, ref: RepoRef, branch: str,
                     headers: dict, digest: RepoDigest) -> None:
    """Best effort when the tree cannot be listed: try the usual names."""
    prefix = (ref.path + "/") if ref.path else ""
    tried = [prefix + name for name in KNOWN_DOCS]
    if prefix:
        tried += list(KNOWN_DOCS)          # fall back to the repo root too
    for path in tried:
        if len(digest.files) >= 4:
            break
        text = _read_blob(client, ref, branch, path, headers, digest)
        if text and text.strip():
            digest.files.append((path, text[:MAX_FILE_BYTES]))
            digest.tree.append((path, len(text)))
    if digest.files:
        digest.warnings.append(
            "read %d known document(s) without the repository tree; the primer "
            "describes the project rather than its code. Set GITHUB_TOKEN, or "
            "run again where api.github.com is reachable, for a deep read."
            % len(digest.files))
    else:
        digest.warnings.append("no readable documents found in the repository.")


def _list_tree(client: httpx.Client, ref: RepoRef, branch: str,
               headers: dict, digest: RepoDigest) -> list[tuple[str, int]]:
    url = f"{API}/repos/{ref.owner}/{ref.repo}/git/trees/{branch}?recursive=1"
    try:
        r = _get(client, url, headers)
    except httpx.HTTPError as exc:
        digest.warnings.append(f"could not list the repository tree ({exc.__class__.__name__}).")
        return []
    if r.status_code != 200:
        digest.warnings.append(
            f"could not list the repository tree (HTTP {r.status_code}); "
            "reading metadata and README only.")
        return []
    data = r.json()
    if data.get("truncated"):
        digest.warnings.append("the repository tree is very large; GitHub truncated it, "
                               "so some files were not considered.")
    out = []
    for node in data.get("tree", []):
        if node.get("type") != "blob":
            continue
        path = node.get("path") or ""
        if _is_skipped(path) or not _is_readable(path):
            continue
        out.append((path, int(node.get("size") or 0)))
    return out


def _read_blob(client: httpx.Client, ref: RepoRef, branch: str, path: str,
               headers: dict, digest: RepoDigest) -> str:
    """Raw first; the API's contents endpoint when the repo is private."""
    auth = {k: v for k, v in headers.items() if k == "Authorization"}
    try:
        r = _get(client, f"{RAW}/{ref.owner}/{ref.repo}/{branch}/{path}", auth)
        if r.status_code == 200:
            return _decode(path, r.text)
        rr = _get(client, f"{API}/repos/{ref.owner}/{ref.repo}/contents/{path}",
                  headers, params={"ref": branch})
        if rr.status_code == 200:
            body = rr.json()
            if body.get("encoding") == "base64" and body.get("content"):
                raw = base64.b64decode(body["content"]).decode("utf-8", "replace")
                return _decode(path, raw)
    except httpx.HTTPError:
        return ""
    except Exception:  # noqa: BLE001 — one unreadable file must not stop a build
        return ""
    return ""


def _decode(path: str, raw: str) -> str:
    if path.lower().endswith(".ipynb"):
        return _notebook_text(raw)
    if path.lower().endswith((".md", ".markdown")):
        from .ingest import clean_markdown
        return clean_markdown(raw)
    if path.lower().endswith(".rst"):
        from .ingest import _strip_rst
        return _strip_rst(raw)
    return raw
