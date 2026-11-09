#!/usr/bin/env python3
"""Build every missing KinetixAI primer and stage it for the website.

One command, no babysitting:

    python3 scripts/ship_primers.py

It reads primers_catalog.yaml, works out which primers are not finished
yet, builds those, verifies each PDF, retries a failed track on a second
model, stages everything into site-bundle/ with an index the site can read,
and writes a report. Re-running it only builds what is still missing, so an
interrupted run resumes instead of starting over.

    --force            rebuild everything, even finished primers
    --only a,b         restrict to these slugs
    --engine groq      override the catalog engine
    --model <id>       override the catalog model
    --parallel 4       concurrent builds (default 4)
    --pages 8          override page count
    --skip-live        leave slugs already published on the site alone
    --dry-run          print the plan and exit
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "primers_catalog.yaml"
SITE = "https://kinetixai.ai/primers"

# A primer that built but came out thin is a failure worth catching here
# rather than after it is uploaded.
MIN_PDF_BYTES = 40_000


# ── plan ──────────────────────────────────────────────────────────────────

def load_catalog() -> tuple[dict, list[dict]]:
    try:
        import yaml
    except ImportError:
        sys.exit("needs PyYAML:  pip install pyyaml")
    data = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    defaults = dict(data.get("defaults") or {})
    tracks = []
    for tr in data.get("tracks") or []:
        merged = {**defaults, **tr}
        if not merged.get("slug"):
            sys.exit(f"track {merged.get('topic')!r} has no slug; add one to {CATALOG.name}")
        tracks.append(merged)
    if not tracks:
        sys.exit(f"no tracks in {CATALOG.name}")
    return defaults, tracks


def pdf_pages(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return None
    try:
        doc = pdfium.PdfDocument(str(path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return None


def is_finished(track: dict, out_root: Path) -> tuple[bool, str]:
    """Already built and plausible? Reports why not, for the log."""
    slug = track["slug"]
    pdf = out_root / slug / f"{slug}.pdf"
    if not pdf.is_file():
        return False, "no pdf"
    size = pdf.stat().st_size
    if size < MIN_PDF_BYTES:
        return False, f"pdf only {size // 1024}kb"
    want = int(track.get("pages") or 8)
    got = pdf_pages(pdf)
    if got is not None and got < want:
        return False, f"{got} pages, wanted {want}"
    return True, f"{size // 1024}kb"


def is_live(slug: str, timeout: float = 6.0) -> bool:
    req = urllib.request.Request(f"{SITE}/{slug}/", method="HEAD",
                                 headers={"User-Agent": "kx-ship-primers"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False


# ── build ─────────────────────────────────────────────────────────────────

def build_cmd(track: dict, out_root: Path, model: str | None) -> list[str]:
    slug = track["slug"]
    cmd = [
        "kx", "primers", "build",
        "--topic", str(track["topic"]),
        "--slug", slug,
        "--pages", str(track.get("pages") or 8),
        "--pdfs", str(track.get("pdfs", 1)),
        "--ppts", str(track.get("ppts", 1)),
        "--images", str(track.get("images", 8)),
        "--json-bundles", str(track.get("jsons", 1)),
        "--engine", str(track["engine"]),
        "--author", str(track.get("author") or "Rohit Patel"),
        "--out-dir", str(out_root / slug),
    ]
    if track.get("category"):
        cmd += ["--category", str(track["category"])]
    personas = track.get("personas") or ["student"]
    if isinstance(personas, str):
        personas = [personas]
    cmd += ["--persona", ",".join(personas)]
    if track.get("watermark"):
        cmd += ["--watermark", str(track["watermark"])]
    if track.get("source"):
        cmd += ["--source", str(track["source"])]
    if track.get("verify_citations"):
        cmd += ["--verify-citations"]
    if model:
        cmd += ["--model", model]
    return cmd


def first_real_error(text: str) -> str:
    """The line that says what went wrong, not the trailing 'try this' hint.

    kx prints a summary, then one line per failed page, then a suggestion.
    Taking the tail caught the suggestion and threw the cause away.
    """
    interesting = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("Try `kx"):
            continue
        if any(k in line for k in ("HTTP ", "failed:", "error:", "Error",
                                   "Traceback", "rate limit", "429", "400",
                                   "401", "403", "404", "timeout", "Timeout")):
            interesting.append(line)
    # The per-page cause is more specific than the summary that precedes it.
    for line in interesting:
        if "failed:" in line or "HTTP " in line:
            return line[:500]
    return (interesting[0][:500] if interesting else "failed with no error output")


def run_one(track: dict, out_root: Path, models: list[str | None],
            logs: Path) -> dict:
    """Build one track, retrying on the next model if the first fails."""
    slug = track["slug"]
    started = time.time()
    attempts: list[dict] = []
    log_path = logs / f"{slug}.log"
    chunks: list[str] = []
    for model in models:
        cmd = build_cmd(track, out_root, model)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = (proc.stdout or "") + (proc.stderr or "")
        chunks.append(f"$ {' '.join(cmd)}\n\n{output}\n")
        log_path.write_text("\n".join(chunks), encoding="utf-8")
        ok, why = is_finished(track, out_root)
        attempt = {"model": model or "auto", "exit": proc.returncode, "ok": ok}
        if ok:
            attempt["detail"] = why
            attempts.append(attempt)
            return {"slug": slug, "topic": track["topic"], "ok": True,
                    "seconds": round(time.time() - started, 1),
                    "detail": why, "attempts": attempts}
        attempt["detail"] = why if "pdf" in why else first_real_error(output)
        attempt["error"] = first_real_error(output)
        attempts.append(attempt)
    return {"slug": slug, "topic": track["topic"], "ok": False,
            "seconds": round(time.time() - started, 1),
            "detail": attempts[-1]["error"], "attempts": attempts,
            "log": str(log_path)}


# ── stage for the site ────────────────────────────────────────────────────

def stage(tracks: list[dict], out_root: Path, bundle: Path) -> list[dict]:
    """Copy finished primers into bundle/<slug>/ and write an index."""
    bundle.mkdir(parents=True, exist_ok=True)
    index = []
    for track in tracks:
        slug = track["slug"]
        src, dest = out_root / slug, bundle / slug
        ok, _ = is_finished(track, out_root)
        if not ok:
            continue
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        pdf = dest / f"{slug}.pdf"
        entry = {
            "slug": slug,
            "title": track["topic"],
            "category": track.get("category", ""),
            "personas": track.get("personas") or ["student"],
            "pages": pdf_pages(pdf) or int(track.get("pages") or 8),
            "pdf": f"{slug}/{slug}.pdf",
            "bytes": pdf.stat().st_size,
            "url": f"/primers/{slug}/",
        }
        for extra, name in (("pptx", f"{slug}.pptx"), ("json", f"{slug}.json"),
                            ("html", f"{slug}.html")):
            if (dest / name).is_file():
                entry[extra] = f"{slug}/{name}"
        covers = sorted((dest / "images").glob("*.png")) if (dest / "images").is_dir() else []
        if covers:
            entry["cover"] = f"{slug}/images/{covers[0].name}"
            entry["images"] = [f"{slug}/images/{p.name}" for p in covers]
        index.append(entry)
    (bundle / "primers_index.json").write_text(
        json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "count": len(index), "primers": index}, indent=2) + "\n",
        encoding="utf-8")
    return index


def write_report(path: Path, results: list[dict], skipped: list[dict],
                 index: list[dict], bundle: Path, elapsed: float) -> None:
    built = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    lines = [
        "# Primer build report",
        "",
        f"- Run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Wall clock: {elapsed / 60:.1f} min",
        f"- Built now: {len(built)}   Failed: {len(failed)}   "
        f"Already done: {len(skipped)}",
        f"- Staged for upload: {len(index)} → `{bundle}`",
        "",
    ]
    if failed:
        lines += ["## Failed", "",
                  "| slug | why |", "|---|---|"]
        lines += [f"| {r['slug']} | {r['detail'][:140]} |" for r in failed]
        lines += ["", f"Per-track logs: `{bundle.parent / 'logs'}`", ""]
    if built:
        lines += ["## Built this run", "", "| slug | pages | size | secs |", "|---|---|---|---|"]
        by_slug = {e["slug"]: e for e in index}
        for r in sorted(built, key=lambda r: r["slug"]):
            e = by_slug.get(r["slug"], {})
            lines.append(f"| {r['slug']} | {e.get('pages', '?')} | "
                         f"{e.get('bytes', 0) // 1024}kb | {r['seconds']} |")
        lines.append("")
    if skipped:
        lines += ["## Skipped (already finished)", ""]
        lines += [f"- {s['slug']} — {s['why']}" for s in skipped]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────

def callable_models(engine: str) -> tuple[list[str], list[dict]]:
    """Models that answer a real request — not merely listed.

    A hosted /models list reports entitlement, not permission: Groq lists a
    model and then returns 403 model_permission_blocked_project for it, so
    clearing a batch run on the list alone fails all 24 tracks.
    """
    cmd = ["kx", "primers", "models", engine, "--verify", "--json"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        # Never report "no model can be called" for what is actually a broken
        # verification step: say what failed, then fall back to the plain list
        # so a bug here can't block a build that would have worked.
        print(f"  verification did not run: `{' '.join(cmd)}` exited {r.returncode}",
              file=sys.stderr)
        detail = (r.stderr or r.stdout or "").strip()
        if detail:
            print("  " + detail.splitlines()[0][:300], file=sys.stderr)
        listed = plain_models(engine)
        if listed:
            print(f"  falling back to the unverified list: {', '.join(listed)}",
                  file=sys.stderr)
        return listed, []
    return (data.get("callable") or []), (data.get("checks") or [])


def plain_models(engine: str) -> list[str]:
    r = subprocess.run(["kx", "primers", "models", engine, "--json"],
                       capture_output=True, text=True)
    try:
        return json.loads(r.stdout).get("models") or []
    except json.JSONDecodeError:
        return []


def preflight(engine: str) -> list[str]:
    if not shutil.which("kx"):
        sys.exit("kx is not on PATH. Run:  pip install -e '.[primers]'")
    probe = subprocess.run(["kx", "primers", "build", "--help"],
                           capture_output=True, text=True)
    if "--slug" not in probe.stdout:
        sys.exit("this kx is too old for --slug. Run:  git pull && pip install -e '.[primers]'")
    if engine in ("outline", "mcp"):
        return []
    # Only hosted engines need permission verification. A local server has no
    # project allowlist, and probing it loads the weights at full context for
    # an 8-token request, which is slow enough to look like a hang.
    local = engine in ("ollama", "vllm", "llamacpp", "lmstudio", "jan",
                       "localai", "tgi", "textgen", "llamafile")
    if local:
        listed = plain_models(engine)
        if not listed:
            print(f"\nengine {engine!r}: the server is not serving a chat model. "
                  f"Start it, or `ollama pull qwen2.5:7b`.", file=sys.stderr)
            sys.exit(1)
        print(f"engine {engine}: {len(listed)} model(s), using {listed[0]}")
        return listed
    usable, checks = callable_models(engine)
    for c in checks:
        print(f"  {'ok  ' if c['callable'] else 'FAIL'} {c['model']:28} {c['detail']}")
    if not usable:
        print(f"\nengine {engine!r}: no model can be called with this key.",
              file=sys.stderr)
        if any("blocked" in c["detail"] for c in checks):
            print("Models are listed but blocked at the project level. Enable them "
                  "in the provider's project settings, then re-run.", file=sys.stderr)
        sys.exit(1)
    print(f"engine {engine}: {len(usable)} callable model(s), using {usable[0]}")
    return usable


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--engine", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--pages", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--skip-live", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--bundle", default="")
    args = ap.parse_args()

    defaults, tracks = load_catalog()
    out_root = Path(args.out_dir or defaults.get("out_dir") or "./dist/primers")
    out_root = (ROOT / out_root).resolve() if not Path(out_root).is_absolute() else Path(out_root)
    bundle = Path(args.bundle) if args.bundle else ROOT / "site-bundle"
    logs = bundle.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if args.engine:
        for t in tracks:
            t["engine"] = args.engine
    if args.pages:
        for t in tracks:
            t["pages"] = args.pages
    engine = tracks[0]["engine"]

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    if wanted:
        unknown = wanted - {t["slug"] for t in tracks}
        if unknown:
            sys.exit(f"--only names unknown slugs: {', '.join(sorted(unknown))}")
        tracks = [t for t in tracks if t["slug"] in wanted]

    todo, skipped = [], []
    for t in tracks:
        if args.skip_live and is_live(t["slug"]):
            skipped.append({"slug": t["slug"], "why": "already published"})
            continue
        ok, why = is_finished(t, out_root)
        if ok and not args.force:
            skipped.append({"slug": t["slug"], "why": why})
        else:
            todo.append(t)

    print(f"catalog: {len(tracks)}   to build: {len(todo)}   skipping: {len(skipped)}")
    for t in todo:
        print(f"  build  {t['slug']:34} {t['topic']}")
    for s in skipped:
        print(f"  skip   {s['slug']:34} {s['why']}")
    if args.dry_run:
        return 0

    # Retry ladder: the chosen model, then a genuinely different one from
    # the served list. [x, None] deduped to a single attempt, so the retry
    # this script advertised never actually ran.
    ladder: list[str | None] = []
    if not todo:
        print("nothing to build.")
    else:
        usable = preflight(engine)
        ladder = [args.model] if args.model else ([usable[0]] if usable else [None])
        for m in usable:
            if m not in ladder and len(ladder) < 2:
                ladder.append(m)
        print("model ladder: " + " → ".join(str(m or "auto") for m in ladder))

    started = time.time()
    results: list[dict] = []
    aborted = False
    if todo:
        # A wrong key, a rate limit or an unparseable model fails every track
        # the same way. Stop after the first few rather than running all 24
        # into the same wall and reporting it 24 times.
        give_up_after = max(2, min(3, len(todo)))
        workers = max(1, min(args.parallel, 8))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_one, t, out_root, ladder, logs): t for t in todo}
            done = 0
            for fut in as_completed(futures):
                r = fut.result()
                done += 1
                mark = "ok  " if r["ok"] else "FAIL"
                print(f"[{done}/{len(todo)}] {mark} {r['slug']:34} "
                      f"{r['seconds']}s  {r['detail'][:110]}")
                results.append(r)
                if (not aborted and len(results) >= give_up_after
                        and not any(x["ok"] for x in results)):
                    aborted = True
                    print(f"\nstopping: the first {len(results)} tracks all failed "
                          f"the same way. Nothing is wrong with the remaining "
                          f"{len(todo) - len(results)} topics — fix the cause and re-run.")
                    print(f"  cause: {results[-1]['detail'][:300]}")
                    print(f"  full log: {results[-1].get('log', logs)}")
                    for f in futures:
                        f.cancel()

    index = stage(tracks, out_root, bundle)
    elapsed = time.time() - started
    report = bundle / "build-report.md"
    write_report(report, results, skipped, index, bundle, elapsed)

    failed = [r for r in results if not r["ok"]]
    print(f"\nstaged {len(index)} primer(s) in {bundle}")
    print(f"index:  {bundle / 'primers_index.json'}")
    print(f"report: {report}")
    if failed:
        print(f"\n{len(failed)} track(s) failed; logs in {logs}")
        for r in failed:
            print(f"  {r['slug']}: {r['detail'][:120]}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; re-run to resume.", file=sys.stderr)
        sys.exit(130)
