"""The TriFleetRCA page archetypes.

A primer follows a canonical research-to-production progression rather
than letting the model invent an arc per topic. Each archetype names the
page's role, the numbered sections it should carry, and the visual blocks
it must contain — which is what the per-page prompts and the density gate
are built from.

``plan(pages)`` maps any requested page count onto the progression: 8 is
the full arc, fewer pages drop the deep-dives from the middle, more pages
repeat the deep-dive archetype with a continuing number.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageSpec:
    role: str                       # short name, used in prompts and logs
    title_hint: str                 # how the page title should read
    sections: list[str] = field(default_factory=list)   # the numbered sections
    must_have: list[str] = field(default_factory=list)  # block types required
    note: str = ""                  # extra instruction for this page only


# Every page gets these on top of its own requirements.
BASELINE = ["bento", "table", "code", "cloud"]

OVERVIEW = PageSpec(
    role="overview",
    title_hint="Executive overview and high-level architecture of {topic}",
    sections=[
        "Problem Statement — today's reality",
        "Our Goal / Core Objectives",
        "Key Benefits & Real-World Value",
        "High-Level Architecture (Input → Stage 1 → Stage 2 → Stage 3 → Output)",
        "Core Components / Sub-Systems Breakdown",
        "Tech Stack & Hardware/Software Ecosystem",
        "Open Source, Community & Quick Summary",
    ],
    must_have=BASELINE + ["flow"],
    note="Start with plain-language intuition before any mathematics or code.",
)

DATA = PageSpec(
    role="data",
    title_hint="Datasets, workloads and benchmark construction for {topic}",
    sections=[
        "Data / Workload Sources",
        "Scenarios, Failure Modes & Edge Cases (Type | Description | Example)",
        "Environment & Hardware Setup",
        "Pipeline & Data Construction Flow (Collect → Normalize → Execute)",
        "Workload Statistics & Metrics",
        "Why Real Benchmarks Matter",
    ],
    must_have=BASELINE + ["flow"],
    note="A hierarchy block is welcome in place of the card grid, not in addition.",
)

STAGE1 = PageSpec(
    role="stage1",
    title_hint="Deep dive, stage 1 — ingestion, memory hierarchy and dataflow in {topic}",
    sections=[
        "What Happens Under the Hood",
        "Step-by-Step Execution Flow",
        "Normalization, Masking & Optimization Rules",
        "Concrete Input/Output Transformation Example",
        "Complexity & Bottleneck Analysis",
    ],
    must_have=BASELINE + ["steps"],
)

STAGE2 = PageSpec(
    role="stage2",
    title_hint="Deep dive, stage 2 — core algorithmic reformulation in {topic}",
    sections=[
        "Hierarchical Scope Mapping (wider → medium → narrower)",
        "Algorithmic Pipeline Flow (Step A → Step B → Step C)",
        "Side-by-Side Worked Example (input → intermediate → output)",
        "Mnemonic Box — a memory hook for the stages",
        "Mathematical / Algorithmic Formulation",
    ],
    must_have=BASELINE + ["flow", "callout"],
    note="The mnemonic must be derivable from the content, e.g. first letters of the stages.",
)

GUARDRAILS = PageSpec(
    role="guardrails",
    title_hint="Safety guardrails, edge cases and correctness verification for {topic}",
    sections=[
        "Failure Modes, Pitfalls & Security Risks",
        "Guardrail / Verification Gate Architecture (Input → Gate → Safe vs Rejected)",
        "Filtering & Validation Flow",
        "Concrete Guard / Correctness Rules",
        "Ablation — what breaks without this guard",
    ],
    must_have=BASELINE + ["flow", "callout"],
)

IMPLEMENTATION = PageSpec(
    role="implementation",
    title_hint="Local execution, hardware setup and implementation code for {topic}",
    sections=[
        "Hardware & Runtime Setup (Model/Kernel | Engine | Hardware | Environment)",
        "Inputs & Parameters Specification",
        "Step-by-Step Execution Process",
        "Implementation Code & Output Schema",
        "End-to-End On-Device Architecture",
        "Common Challenges & Solutions, plus Best Practices",
    ],
    must_have=BASELINE + ["steps"],
    note="The code block must be runnable and specific, not pseudocode.",
)

EVALUATION = PageSpec(
    role="evaluation",
    title_hint="Evaluation, honest benchmarks and results for {topic}",
    sections=[
        "Evaluation Setup (Environment | Datasets | Baselines)",
        "Evaluation Metrics (Metric | Definition)",
        "Primary Performance Comparison",
        "Latency / Memory-Traffic Reduction (lower is better)",
        "Three Honest Columns (Benchmark | Kernel Only | + H2D Transfer | End-to-End)",
        "Ablation Study (Method | Speedup | Latency | Notes)",
    ],
    # the chart and the honest-columns table carry this page; a card grid
    # on top of them does not fit an A4 sheet
    must_have=["table", "code", "cloud", "chart"],
    note=("Include ONE chart block (a second will not fit on an A4 page) plus the "
          "Three Honest Columns table, which is mandatory: never report a single "
          "flattering speedup. If the sources give no numbers, describe the "
          "measurement method and leave the table's cells as 'not measured'."),
)

TAKEAWAYS = PageSpec(
    role="takeaways",
    title_hint="Key takeaways, production runbook and next steps for {topic}",
    sections=[
        "What We Achieved / Core Summary",
        "Real-World Production Impact",
        "Lessons Learned & Engineering Principles",
        "Future Work & Open Research Directions",
        "Copy-Paste CLI / Profiling Runbook",
        "Resources, Traps Solved & Final Takeaway",
    ],
    must_have=BASELINE + ["steps"],
    note="The runbook must be copy-pasteable: setup, lock clocks, run, profile.",
)

FULL_ARC = [OVERVIEW, DATA, STAGE1, STAGE2, GUARDRAILS, IMPLEMENTATION, EVALUATION, TAKEAWAYS]

# Which pages survive when fewer than 8 are asked for: the overview and the
# takeaways always stay, then depth is added back in this order.
_PRIORITY = [OVERVIEW, TAKEAWAYS, STAGE1, EVALUATION, IMPLEMENTATION, DATA, STAGE2, GUARDRAILS]


def plan(pages: int) -> list[PageSpec]:
    """The archetype for each page of an N-page primer."""
    pages = max(1, int(pages))
    if pages >= len(FULL_ARC):
        out = list(FULL_ARC)
        # extra pages repeat the deep-dive shape rather than inventing an arc
        i = 3
        while len(out) < pages:
            extra = PageSpec(
                role=f"deepdive{i}",
                title_hint="Deep dive, stage %d — further mechanisms in {topic}" % i,
                sections=list(STAGE1.sections),
                must_have=list(STAGE1.must_have),
            )
            out.insert(len(out) - 2, extra)  # before evaluation and takeaways
            i += 1
        return out[:pages]
    keep = _PRIORITY[:pages]
    return [spec for spec in FULL_ARC if spec in keep]


def page_prompt_block(spec: PageSpec, index: int, total: int, topic: str) -> str:
    """The per-page instruction appended to the system prompt."""
    lines = [
        f"PAGE {index} OF {total} — role: {spec.role}",
        f"Title should read as: {spec.title_hint.format(topic=topic)}",
        "Numbered sections for this page (use these as the section titles, adapted "
        "to the topic; keep their order):",
    ]
    lines += [f"  {i}. {s}" for i, s in enumerate(spec.sections, 1)]
    lines.append("This page MUST include blocks of these types: "
                 + ", ".join(sorted(set(spec.must_have))) + ".")
    if spec.note:
        lines.append("Note: " + spec.note)
    return "\n".join(lines)


# How many of each visual block a page should carry, and in what shape. A
# small model follows a concrete checklist far better than prose rules.
_SHAPE = {
    "bento": "bento with exactly 3 cards",
    "table": "table with 4 columns and 3 rows",
    "flow": "flow with 4 steps",
    "steps": "steps with 4 items",
    "hierarchy": "hierarchy with 4 levels",
    "chart": "chart with 3 or 4 bars (numbers from the sources only)",
    "code": "code block, bash or python, 3 lines",
    "cloud": "cloud with one short punchy line",
    "callout": "callout",
}


def page_skeleton(spec: PageSpec, max_sections: int = 5) -> str:
    """An explicit block-by-block checklist for one page."""
    lines = ["EMIT EXACTLY THIS BLOCK SEQUENCE, IN THIS ORDER:"]
    n = 0
    for title in spec.sections[:max_sections]:
        n += 1
        lines.append(f'{n}. section, num "{n}", title "{title}", body 30-45 words')
    for kind in spec.must_have:
        n += 1
        lines.append(f"{n}. {_SHAPE.get(kind, kind)}")
    lines.append(f"That is {n} blocks. Do not stop early and do not add more.")
    return "\n".join(lines)
