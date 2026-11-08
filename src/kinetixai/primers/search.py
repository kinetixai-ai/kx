"""Domain-relevant source selection for Primers.

Federated search matches words, not meaning: "GPU & CUDA: Kernel Fusion"
pulls back tokamak fusion energy, infrared image fusion and sensor fusion
papers, all of which would then be printed as "verified sources".

This module scores each candidate against the topic and drops the ones
that only share an ambiguous word. The scorer is deterministic and
offline — no embedding model, no extra LLM call — so a build is
reproducible and a citation list can be explained:

  score = domain-term overlap + subtopic-term overlap - ambiguity penalty

A paper must match the topic's *subject* terms, not just one of its
ambiguous words, to survive. ``--relevance 0`` turns filtering off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Words that mean different things in different fields. A paper that
# matches ONLY one of these, with no other topic term, is almost always a
# false positive from another discipline.
AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "fusion": ("tokamak", "plasma", "nuclear fusion", "iter", "confinement", "plasmas",
               "infrared", "visible image", "image fusion", "sensor fusion",
               "multimodal fusion", "facial", "wearable", "eeg", "plenoptic",
               "plasma physics", "plasma-facing", "deuterium", "tritium", "stellarator"),
    "kernel": ("kernel method", "kernel methods", "reproducing kernel", "gaussian process",
               "kernel density", "kernel regression", "svm", "support vector",
               "mercer", "kernel trick", "hilbert space"),
    "attention": ("attention deficit", "adhd", "visual attention span", "attentional bias"),
    "memory": ("working memory", "episodic memory", "memory impairment", "amnesia",
               "human memory", "memory consolidation"),
    "transformer": ("power transformer", "distribution transformer", "transformer oil",
                    "transformer winding", "electrical transformer", "substation"),
    "agent": ("chemical agent", "contrast agent", "antimicrobial agent", "pharmacological"),
    "pipeline": ("oil pipeline", "gas pipeline", "pipeline corrosion", "water pipeline"),
    "scaling": ("scaling of fish", "dental scaling"),
    "diffusion": ("diffusion mri", "molecular diffusion", "thermal diffusion",
                  "diffusion of innovations", "gas diffusion", "dti"),
    "quantization": ("vector quantization of speech", "charge quantization",
                     "flux quantization", "energy quantization"),
    "security": ("food security", "water security", "job security", "social security",
                 "homeland security", "energy security"),
}

STOP = {
    "a", "an", "and", "the", "of", "for", "to", "in", "on", "with", "from", "by", "or",
    "at", "as", "is", "are", "be", "its", "it", "this", "that", "into", "via", "using",
    "toward", "towards", "between", "about", "how", "what", "why", "when", "primer",
    "introduction", "overview", "guide", "deep", "dive", "part", "types",
}


def terms(text: str) -> set[str]:
    """Content words, lowercased, singular-ish."""
    words = re.findall(r"[a-z0-9+#.]{2,}", (text or "").lower())
    out: set[str] = set()
    for w in words:
        if w in STOP:
            continue
        out.add(w)
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            out.add(w[:-1])  # kernels -> kernel
    return out


_WORD_CACHE: dict[str, re.Pattern] = {}


def _marker(term: str) -> re.Pattern:
    """Whole-word match for an ambiguity marker.

    Substring matching was wrong and quietly costly: "iter" (for ITER, the
    fusion reactor) matched "iterative", so real CUDA papers were dropped as
    plasma physics.
    """
    pat = _WORD_CACHE.get(term)
    if pat is None:
        pat = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
        _WORD_CACHE[term] = pat
    return pat


@dataclass
class Scored:
    paper: dict
    score: float
    reason: str

    @property
    def kept(self) -> bool:
        return self.reason == "kept"


def _haystack(paper: dict) -> str:
    return " ".join(str(paper.get(k) or "") for k in ("title", "abstract", "venue"))


def score_paper(paper: dict, topic_terms: set[str], ambiguous_in_topic: set[str]) -> Scored:
    hay = _haystack(paper).lower()
    hay_terms = terms(hay)
    overlap = topic_terms & hay_terms
    # how much of the topic the paper actually covers
    coverage = len(overlap) / max(1, len(topic_terms))

    # A paper that matches only ambiguous words is from another field.
    unambiguous_hits = overlap - ambiguous_in_topic
    if not unambiguous_hits:
        return Scored(paper, coverage, "matched only an ambiguous word "
                                       f"({', '.join(sorted(overlap)) or 'nothing'})")

    # Explicit other-discipline markers for the ambiguous words in this topic.
    # A marker only counts when the paper does not also sit squarely in the
    # topic's own field: "GPU ... iterative solver" is ours, "ITER plasma" is not.
    strong = len(unambiguous_hits) >= 2
    for word in ambiguous_in_topic:
        for marker in AMBIGUOUS.get(word, ()):
            if _marker(marker).search(hay) and not strong:
                return Scored(paper, coverage, f"other field: '{marker}'")

    return Scored(paper, coverage + 0.1 * len(unambiguous_hits), "kept")


def filter_sources(papers: list[dict], topic: str, min_coverage: float = 0.18,
                   limit: int = 15) -> tuple[list[dict], list[str]]:
    """Keep papers that are actually about the topic.

    Returns (kept, notes). Notes explain every drop, so a thin citation
    list is never a silent one.
    """
    topic_terms = terms(topic)
    if not topic_terms or min_coverage <= 0:
        return papers[:limit], []
    ambiguous_in_topic = {w for w in topic_terms if w in AMBIGUOUS}

    scored = [score_paper(p, topic_terms, ambiguous_in_topic) for p in papers]
    kept = [s for s in scored if s.kept and s.score >= min_coverage]
    kept.sort(key=lambda s: (-s.score, -(s.paper.get("citations") or 0)))

    notes: list[str] = []
    dropped = [s for s in scored if s not in kept]
    for s in dropped[:8]:
        title = (s.paper.get("title") or "untitled")[:60]
        why = s.reason if not s.kept else f"weak topic match ({s.score:.2f})"
        notes.append(f"source dropped — {title}: {why}")
    if len(dropped) > 8:
        notes.append(f"source dropped — and {len(dropped) - 8} more off-topic results")
    return [s.paper for s in kept[:limit]], notes


def queries_for(topic: str, max_queries: int = 3) -> list[str]:
    """Search queries that pin an ambiguous topic to its field.

    "GPU & CUDA: Kernel Fusion" searches for the whole phrase, then for the
    subtopic paired with its domain anchor, instead of the bare word that
    matches four other disciplines.
    """
    clean = re.sub(r"[^\w\s+#.-]", " ", topic).strip()
    clean = re.sub(r"\s+", " ", clean)
    out = [clean]
    parts = [p.strip() for p in re.split(r"[:—–-]| and ", topic) if p.strip()]
    if len(parts) >= 2:
        anchor, subject = parts[0], parts[-1]
        out.append(f"{subject} {anchor}")
        out.append(subject)
    return [q for i, q in enumerate(out) if q and q not in out[:i]][:max_queries]
