"""Citations: BibTeX and APA 7 from a paper dict (library entry or Paper.to_dict())."""
from __future__ import annotations

import re
import unicodedata

FORMATS = ("bibtex", "apa")


def _split_name(name: str) -> tuple[str, list[str]]:
    """'Tri Dao' or 'Dao, Tri' -> ('Dao', ['Tri'])."""
    name = " ".join((name or "").split())
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
        return last, first.split()
    parts = name.split(" ")
    if len(parts) == 1:
        return parts[0], []
    # keep particles with the surname: "Ludwig van Beethoven" -> "van Beethoven"
    i = len(parts) - 1
    while i > 1 and parts[i - 1].lower() in {"van", "von", "de", "del", "der", "da", "di", "la", "le"}:
        i -= 1
    return " ".join(parts[i:]), parts[:i]


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def bibtex_key(p: dict) -> str:
    authors = p.get("authors") or []
    last = _split_name(authors[0])[0] if authors else "anon"
    words = [w for w in re.findall(r"[A-Za-z0-9]+", _ascii(p.get("title") or "")) if len(w) > 3]
    first = words[0].lower() if words else "paper"
    return re.sub(r"[^A-Za-z0-9]", "", _ascii(last)).lower() + str(p.get("year") or "") + first


_BIB_ESC = {"\\": "\\textbackslash{}", "{": "\\{", "}": "\\}", "&": "\\&", "%": "\\%", "#": "\\#",
            "$": "\\$", "_": "\\_", "^": "\\^{}", "~": "\\~{}"}


def _bib_escape(s: str) -> str:
    """Escape LaTeX specials, braces included, so a field can never unbalance the entry."""
    return "".join(_BIB_ESC.get(ch, ch) for ch in (s or ""))


def bibtex(p: dict, key: str | None = None) -> str:
    aid = re.sub(r"v\d+$", "", p.get("arxiv_id") or "") or None
    venue = p.get("venue")
    is_arxiv_only = aid and (not venue or venue.lower() == "arxiv")
    kind = "misc" if is_arxiv_only or not venue else "article"
    fields: list[tuple[str, str]] = [
        ("title", "{" + _bib_escape(p.get("title") or "") + "}"),
        ("author", " and ".join(_bib_escape(a) for a in p.get("authors") or []) or "Unknown"),
    ]
    if p.get("year"):
        fields.append(("year", str(p["year"])))
    if kind == "article" and venue:
        fields.append(("journal", _bib_escape(venue)))
    if aid:
        fields += [("eprint", aid), ("archivePrefix", "arXiv")]
    if p.get("doi"):
        fields.append(("doi", p["doi"]))
    url = p.get("url") or (f"https://arxiv.org/abs/{aid}" if aid else None)
    if url:
        fields.append(("url", url))
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
    return f"@{kind}{{{key or bibtex_key(p)},\n{body}\n}}"


def bibtex_many(items: list[dict]) -> str:
    """Several entries with unique keys: dao2022flash, dao2022flasha, dao2022flashb..."""
    seen: dict[str, int] = {}
    keys = [bibtex_key(i) for i in items]
    dupes = {k for k in keys if keys.count(k) > 1}
    out = []
    for item, k in zip(items, keys):
        if k in dupes:
            n = seen.get(k, 0)
            seen[k] = n + 1
            suffix = ""
            while True:  # a, b, ... z, aa, ab ...
                suffix = chr(97 + n % 26) + suffix
                n = n // 26 - 1
                if n < 0:
                    break
            k = k + suffix
        out.append(bibtex(item, k))
    return "\n\n".join(out)


def _apa_author(name: str) -> str:
    last, firsts = _split_name(name)
    initials = " ".join(f"{f[0]}." for f in firsts if f)
    return f"{last}, {initials}".strip().rstrip(",")


def apa(p: dict) -> str:
    authors = [_apa_author(a) for a in p.get("authors") or [] if a]
    if not authors:
        who = ""
    elif len(authors) == 1:
        who = authors[0]
    elif len(authors) <= 20:
        who = ", ".join(authors[:-1]) + ", & " + authors[-1]
    else:  # APA 7: first 19, ellipsis, last
        who = ", ".join(authors[:19]) + ", . . . " + authors[-1]
    year = f"({p['year']})" if p.get("year") else "(n.d.)"
    title = (p.get("title") or "").strip().rstrip(".")
    aid = re.sub(r"v\d+$", "", p.get("arxiv_id") or "") or None
    venue = p.get("venue")
    if aid and (not venue or venue.lower() == "arxiv"):
        tail = f"arXiv. https://doi.org/10.48550/arXiv.{aid}" if re.match(r"^\d{4}\.\d{4,5}$", aid) else f"arXiv:{aid}."
    else:
        tail = (f"{venue}. " if venue else "") + (f"https://doi.org/{p['doi']}" if p.get("doi") else (p.get("url") or ""))
    head = f"{who} {year}." if who else f"{title}. {year}."
    middle = f" {title}." if who else ""
    return f"{head}{middle} {tail}".strip()


def format_citation(p: dict, fmt: str) -> str:
    if fmt == "bibtex":
        return bibtex(p)
    if fmt == "apa":
        return apa(p)
    raise ValueError(f"unknown format {fmt!r}; use one of {', '.join(FORMATS)}")
