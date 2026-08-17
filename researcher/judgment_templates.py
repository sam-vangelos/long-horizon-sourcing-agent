"""Researcher LLM evaluation templates — Slice 5.

Two templates: facial (compact triage on a snippet) + full (deep
evaluation on a hydrated candidate). Both produce
:class:`shared.schemas.OpusDecision`-shaped output.

Per Researcher Module Spec Opinion 6, the full evaluator MUST populate
``rationale`` + ``confidence`` exactly — those fields land at
``terminal_payload_json["full_decision"]`` and are read by the
source-agnostic
:func:`shared.runtime_state.read_models.extract_save_reason_and_confidence`.

Per Spec Opinion 7 (engineer-vocab leak): the facial fast-exit rationale
must be recruiter-readable (e.g., "Skipped — only 1 paper in last 36
months, below the field-default minimum of 3"), NOT
"papers_in_window=1 < papers_in_window_floor=3".
"""

from __future__ import annotations

from typing import Any

from researcher.discipline_defaults import discipline_label
from researcher.schemas import ResearcherCandidate, ResearcherSnippet


# ---------------------------------------------------------------------------
# Fast-exit rationale builders (recruiter-readable copy)
# ---------------------------------------------------------------------------


def fast_exit_rationale_for_papers(
    *,
    papers_in_window: int,
    papers_in_window_floor: int,
    papers_in_window_months: int,
    discipline: str,
) -> str:
    """Recruiter-readable copy for the papers-in-window fast-exit gate.

    Shape:
        "Skipped — only 1 paper in the last 36 months, below the field-
         default minimum of 3."

    No engineer vocab; no equation form; no `_floor` suffix. The label
    "field-default" hedges between "discipline default" and "universal
    minimum" so we don't leak the resolver layering into the workspace
    card.
    """

    label = discipline_label(discipline) if discipline else "field defaults"
    paper_word = "paper" if papers_in_window == 1 else "papers"
    return (
        f"Skipped — only {papers_in_window} {paper_word} in the last "
        f"{papers_in_window_months} months, below the {label} minimum "
        f"of {papers_in_window_floor}."
    )


def fast_exit_rationale_for_h_index(
    *,
    h_index: int,
    h_index_floor: int,
    discipline: str,
) -> str:
    """Recruiter-readable copy for the h-index fast-exit gate."""

    label = discipline_label(discipline) if discipline else "field defaults"
    return (
        f"Skipped — h-index {h_index}, below the {label} minimum "
        f"of {h_index_floor}."
    )


# ---------------------------------------------------------------------------
# System prompt assemblers
# ---------------------------------------------------------------------------


def assemble_facial_system(brief: Any) -> str:
    """Cacheable facial-triage system prompt; no candidate data."""

    role_title = getattr(brief, "role_title", "") or "(unspecified)"
    capability_areas = _capability_area_names(brief)
    return _FACIAL_SYSTEM_TEMPLATE.format(
        role_title=role_title,
        capability_areas="\n".join(f"  - {n}" for n in capability_areas) or "  - (none)",
    )


def assemble_full_evaluation_system(brief: Any) -> str:
    """Cacheable full-evaluation system prompt; no candidate data."""

    role_title = getattr(brief, "role_title", "") or "(unspecified)"
    capability_areas = _capability_area_block(brief)
    depth = _depth_block(brief)
    return _FULL_SYSTEM_TEMPLATE.format(
        role_title=role_title,
        capability_areas=capability_areas,
        depth_block=depth,
    )


def render_facial_user_prompt(snippet: ResearcherSnippet) -> str:
    """Per-snippet user prompt body."""

    top_papers = "\n".join(
        f"  - {t}" for t in snippet.top_paper_titles[:5]
    ) or "  - (no recent papers)"
    return _FACIAL_USER_TEMPLATE.format(
        name=snippet.name,
        affiliation=snippet.current_affiliation or "(unknown)",
        h_index=snippet.h_index,
        papers_in_window=snippet.papers_in_window,
        top_papers=top_papers,
        arxiv_categories=", ".join(snippet.arxiv_categories) or "(none)",
    )


def render_full_user_prompt(candidate: ResearcherCandidate) -> str:
    """Per-candidate user prompt body for full evaluation."""

    top_papers_lines: list[str] = []
    for paper in candidate.top_papers[:5]:
        first_marker = " [first author]" if paper.is_first_author else ""
        venue = f" — {paper.venue}" if paper.venue else ""
        year = f" ({paper.year})" if paper.year else ""
        cited = f" — cited {paper.citation_count}x" if paper.citation_count else ""
        top_papers_lines.append(
            f"  - {paper.title}{first_marker}{venue}{year}{cited}"
        )
    top_papers = "\n".join(top_papers_lines) or "  - (no top papers in payload)"

    return _FULL_USER_TEMPLATE.format(
        name=candidate.name,
        orcid=candidate.orcid or "(none)",
        affiliations=", ".join(candidate.affiliations) or "(unknown)",
        h_index=candidate.h_index,
        citation_count=candidate.citation_count,
        works_count=candidate.works_count,
        papers_in_window=candidate.papers_in_window,
        top_papers=top_papers,
    )


# ---------------------------------------------------------------------------
# Brief block helpers
# ---------------------------------------------------------------------------


def _capability_area_names(brief: Any) -> list[str]:
    areas = getattr(brief, "capability_areas", None) or []
    names: list[str] = []
    for area in areas:
        name = getattr(area, "name", None)
        if name:
            names.append(str(name))
    return names


def _capability_area_block(brief: Any) -> str:
    areas = getattr(brief, "capability_areas", None) or []
    if not areas:
        return "(none)"
    lines: list[str] = []
    for idx, area in enumerate(areas, start=1):
        name = getattr(area, "name", None) or ""
        description = getattr(area, "description", None) or ""
        lines.append(f"  {idx}. {name}")
        if description:
            lines.append(f"     — {description}")
    return "\n".join(lines)


def _depth_block(brief: Any) -> str:
    depth = getattr(brief, "depth_distinction", None)
    if depth is None:
        return "(no depth_distinction in brief)"
    builder = (
        getattr(depth, "builder_definition", None)
        if not isinstance(depth, dict)
        else depth.get("builder_definition")
    ) or ""
    user = (
        getattr(depth, "user_definition", None)
        if not isinstance(depth, dict)
        else depth.get("user_definition")
    ) or ""
    return f"  Builder: {builder}\n  User: {user}"


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


_FACIAL_SYSTEM_TEMPLATE = """\
You are Cloris, doing facial triage on academic researcher candidates
for the role: {role_title}.

CAPABILITY AREAS the candidate must touch:
{capability_areas}

OUTPUT (JSON object):
  decision: one of FACIAL_YES | FACIAL_NO | FACIAL_BORDERLINE
  rationale: one short recruiter-readable sentence (no engineer vocab,
             no equations, no field-name leakage)
  confidence: float 0.0–1.0

Decision rules:
  - FACIAL_YES: clear evidence of original research that touches one or
    more capability areas, in the last 36 months.
  - FACIAL_NO: zero capability-area overlap, OR no recent publications.
  - FACIAL_BORDERLINE: ambiguous overlap; route to full eval.

Emit ONLY the JSON object — no preamble, no markdown fences.
"""


_FACIAL_USER_TEMPLATE = """\
Candidate snippet:

  Name: {name}
  Current affiliation: {affiliation}
  h-index: {h_index}
  Papers in window: {papers_in_window}
  arXiv categories: {arxiv_categories}
  Top papers:
{top_papers}
"""


_FULL_SYSTEM_TEMPLATE = """\
You are Cloris, doing full evaluation on academic researcher candidates
for the role: {role_title}.

CAPABILITY AREAS:
{capability_areas}

DEPTH DISTINCTION:
{depth_block}

OUTPUT (JSON object):
  decision: one of SAVE | INFERENTIAL_SAVE | TRANSFERABLE_SAVE | REJECT
  path: short slug describing why (e.g., "first_author_at_canonical_venue",
        "transferable_methods", "drive_by_collaborator")
  confidence: float 0.0–1.0
  rationale: 2–4 sentence recruiter-readable explanation. Cite the top
             1–3 papers by title + venue. No engineer vocab; no field-
             name leakage; no equations.

Decision rules:
  - SAVE: clear first-author original research aligned with capability
    areas; recent (last 24–36 months); confidently the right person.
  - INFERENTIAL_SAVE: strong but indirect signal — common-name
    collision flagged at acquisition, or strong adjacent capability
    that needs human verification.
  - TRANSFERABLE_SAVE: methods/topics translate from a different
    sub-field; borderline strong, escalate to recruiter.
  - REJECT: doesn't pass the depth_distinction's builder bar.

Emit ONLY the JSON object — no preamble, no markdown fences.
"""


_FULL_USER_TEMPLATE = """\
Candidate full record:

  Name: {name}
  ORCID: {orcid}
  Affiliations: {affiliations}
  h-index: {h_index} | Citations: {citation_count} | Works: {works_count}
  Papers in window: {papers_in_window}

  Top papers:
{top_papers}
"""
