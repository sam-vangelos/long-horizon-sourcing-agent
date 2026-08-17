"""Designer judgment templates.

Designer Slice 2 upgrades the Slice-1 placeholder evaluator with real
text-based contextualization prompts. The prompts are deliberately
text-only (no images yet); Slice 5 introduces the multimodal vision-
evaluation pipeline that adds image inputs.

Two stages, mirroring the LinkedIn / GitHub evaluators:

- ``assemble_designer_facial_system(brief)`` — fast triage prompt the
  facial judge calls per candidate. Produces FACIAL_YES /
  FACIAL_NO / FACIAL_BORDERLINE based on the snippet text alone
  (display_name, headline, location, fields, tools, top project
  titles). Cheap; runs against most discovered candidates.
- ``assemble_designer_full_system(brief)`` — deep evaluation prompt
  the full judge calls only for candidates that cleared facial. The
  Slice-2 version is text-only contextualization (specialization,
  tool depth, notable projects, client tier inferred from project
  text). Slice 5 layers the vision-evaluation output on top of this
  text-based contextualization in the workspace surface.

Both prompts read from the V2 brief shape:

- ``role_title``, ``role_summary``
- ``capability_areas`` (name + description; optional
  ``behance_specialization_signals`` + ``tool_stack_signals`` are
  appended as recruiter-authored vocabulary)
- ``depth_distinction.builder_definition``
- ``non_fit_patterns`` (when present)
- ``design_rubric.principles[*].name`` (when present — used as the
  vocabulary the contextualization prompt grounds itself in; the
  full vision-LLM scoring against these principles arrives in Slice 5)

Backward-compat: the Slice-1 placeholder helpers
(:func:`designer_facial_judge_placeholder`,
:func:`designer_full_judge_placeholder`) remain for any caller still
on the stub path, with a deprecation note in the docstring.
"""

from __future__ import annotations

from typing import Any

from shared.schemas import OpusDecision


# Re-exported from Slice 1 so the Slice-2 import surface is one module
# rather than two.
PLACEHOLDER_RATIONALE = (
    "no judgment yet — designer pipeline not built (Slice 1 placeholder; "
    "real text-based contextualization arrives in Slice 2, vision evaluation "
    "in Slice 5)"
)


def designer_facial_judge_placeholder(
    *,
    candidate_name: str = "",
    profile_url: str = "",
) -> OpusDecision:
    """Slice-1 placeholder. Retained for backward compat — Slice 2's
    real prompt arrives via :func:`assemble_designer_facial_system`."""

    return OpusDecision(
        stage="facial",
        decision="FACIAL_NO",
        path="placeholder",
        confidence=0.0,
        rationale=PLACEHOLDER_RATIONALE,
        candidate_name=candidate_name,
        profile_url=profile_url,
    )


def designer_full_judge_placeholder(
    *,
    candidate_name: str = "",
    profile_url: str = "",
) -> OpusDecision:
    """Slice-1 placeholder. Retained for backward compat."""

    return OpusDecision(
        stage="full",
        decision="REJECT",
        path="placeholder",
        confidence=0.0,
        rationale=PLACEHOLDER_RATIONALE,
        candidate_name=candidate_name,
        profile_url=profile_url,
    )


# ---------------------------------------------------------------------------
# Slice 2: real text-based prompts
# ---------------------------------------------------------------------------


_FACIAL_VOICE_RULES = """\
You are a senior design director triaging a stack of designer profiles
for a hiring brief. Your job is to fast-decide which profiles deserve
deep evaluation. Decide FACIAL_YES if the snippet shows strong on-brief
signal; FACIAL_NO if the snippet is clearly off-brief; FACIAL_BORDERLINE
if the snippet is plausibly on-brief but the call needs the full
evaluation pass.

Voice:
- Substantive and specific. Reference what's actually in the snippet,
  not what could hypothetically be in the portfolio.
- No marketing copy. No hedging. No "it depends."
- Write the rationale as one short paragraph (1-3 sentences) the
  recruiter can read at a glance.
"""


_FULL_VOICE_RULES = """\
You are a senior design director writing the text-based contextualization
that lands in the recruiter's workspace alongside the visual judgment
(Slice 5 will append the per-principle vision scoring; today, your text
is what the recruiter sees first).

For each candidate, produce a 2-3 sentence contextualization covering:
1. Specialization — what the candidate is most clearly known for.
2. Tool depth — what tool stack they're fluent in.
3. Project tier — the kind of work product (consumer brand systems,
   enterprise dashboards, indie illustration, motion reels, etc.).

Voice:
- Editorial register. Substantive. Recruiter-readable.
- No marketing copy. No hedging.
- Reference the candidate's stated specialization, fields, top project
  titles. Don't invent details.
"""


def assemble_designer_facial_system(brief: dict[str, Any]) -> str:
    """Build the facial-stage system prompt for a designer candidate.

    Slice 2: text-only triage. The vision-evaluation pipeline in
    Slice 5 augments the workspace surface; the facial-stage decision
    remains text-only because triage runs at high volume and the
    vision call is the expensive surface to gate.
    """

    role_block = _role_block(brief)
    capability_block = _capability_block(brief)
    rubric_vocab_block = _rubric_vocab_block(brief)
    non_fit_block = _non_fit_block(brief)

    return (
        f"{_FACIAL_VOICE_RULES}\n"
        "BRIEF CONTEXT\n"
        "-------------\n"
        f"{role_block}\n"
        f"{capability_block}\n"
        f"{rubric_vocab_block}\n"
        f"{non_fit_block}\n"
        "OUTPUT FORMAT\n"
        "-------------\n"
        "Single line, exact format:\n"
        "DECISION: <FACIAL_YES|FACIAL_NO|FACIAL_BORDERLINE>\n"
        "REASON: <one short paragraph; 1-3 sentences; no marketing copy>\n"
    )


def assemble_designer_full_system(brief: dict[str, Any]) -> str:
    """Build the full-stage system prompt for a designer candidate.

    Slice 2: text-only contextualization — produces the prose that
    lands in the recruiter's workspace card alongside the candidate's
    profile fields. Slice 5 will compose this prose with the vision-
    evaluation per-principle output to render the full HITL visual
    review surface.
    """

    role_block = _role_block(brief)
    capability_block = _capability_block(brief)
    rubric_vocab_block = _rubric_vocab_block(brief)
    depth_block = _depth_block(brief)
    non_fit_block = _non_fit_block(brief)

    return (
        f"{_FULL_VOICE_RULES}\n"
        "BRIEF CONTEXT\n"
        "-------------\n"
        f"{role_block}\n"
        f"{capability_block}\n"
        f"{depth_block}\n"
        f"{rubric_vocab_block}\n"
        f"{non_fit_block}\n"
        "OUTPUT FORMAT\n"
        "-------------\n"
        "Exact format:\n"
        "DECISION: <SAVE|REJECT|INFERENTIAL_SAVE>\n"
        "PATH: <specialization-tag-or-NONE>\n"
        "CONFIDENCE: <0.0-1.0>\n"
        "SUMMARY: <2-3 sentence text-based contextualization; "
        "specialization, tool depth, project tier>\n"
    )


# ---------------------------------------------------------------------------
# Brief-block helpers
# ---------------------------------------------------------------------------


def _role_block(brief: dict[str, Any]) -> str:
    title = str(brief.get("role_title") or "").strip()
    summary = str(brief.get("role_summary") or "").strip()
    pieces = []
    if title:
        pieces.append(f"Role: {title}")
    if summary:
        pieces.append(f"Summary: {summary}")
    if not pieces:
        return ""
    return "\n".join(pieces) + "\n"


def _capability_block(brief: dict[str, Any]) -> str:
    capability_areas = brief.get("capability_areas") or []
    if not isinstance(capability_areas, list) or not capability_areas:
        return ""
    lines = ["Capability areas the recruiter cares about:"]
    for ca in capability_areas:
        if not isinstance(ca, dict):
            continue
        name = str(ca.get("name") or "").strip()
        desc = str(ca.get("description") or "").strip()
        if not name:
            continue
        lines.append(f"  - {name}: {desc}")
        spec_signals = ca.get("behance_specialization_signals") or []
        if isinstance(spec_signals, list) and spec_signals:
            cleaned = [str(s).strip() for s in spec_signals if isinstance(s, str) and s.strip()]
            if cleaned:
                lines.append(f"    specialization signals: {', '.join(cleaned)}")
        tool_signals = ca.get("tool_stack_signals") or []
        if isinstance(tool_signals, list) and tool_signals:
            cleaned = [str(s).strip() for s in tool_signals if isinstance(s, str) and s.strip()]
            if cleaned:
                lines.append(f"    tool stack: {', '.join(cleaned)}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"


def _depth_block(brief: dict[str, Any]) -> str:
    depth = brief.get("depth_distinction")
    if not isinstance(depth, dict):
        return ""
    builder = str(depth.get("builder_definition") or "").strip()
    if not builder:
        return ""
    return f"What 'building it' looks like for this role: {builder}\n"


def _rubric_vocab_block(brief: dict[str, Any]) -> str:
    """The principle names the rubric carries — used as on-brief
    vocabulary the prompt can reference. Slice 5 will substitute the
    full per-principle anchor definitions when the vision-evaluation
    prompt forms; today, just the names."""

    rubric = brief.get("design_rubric")
    if not isinstance(rubric, dict):
        return ""
    principles = rubric.get("principles") or []
    if not isinstance(principles, list):
        return ""
    names: list[str] = []
    for principle in principles:
        if not isinstance(principle, dict):
            continue
        name = principle.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    if not names:
        return ""
    return f"Design principles the recruiter weights: {', '.join(names)}.\n"


def _non_fit_block(brief: dict[str, Any]) -> str:
    patterns = brief.get("non_fit_patterns") or []
    if not isinstance(patterns, list) or not patterns:
        return ""
    lines = ["Non-fit patterns the recruiter explicitly rejects:"]
    for pattern in patterns:
        if not isinstance(pattern, dict):
            continue
        label = str(pattern.get("label") or "").strip()
        why_not = str(pattern.get("why_not") or "").strip()
        if label:
            lines.append(f"  - {label}: {why_not}" if why_not else f"  - {label}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n"
