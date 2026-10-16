"""Per-candidate dossier evidence assembly for executive search.

Combines the LinkedIn profile text the existing full-eval pipeline
already produces with off-LinkedIn signal sections from each
adapter. The output is a single string the dossier full-eval prompt
folds into its user message in place of the raw profile text.

Design notes:

- Failed signals (``SignalFailure``) render a one-line "section
  unavailable" placeholder rather than disappearing silently. The
  dossier eval reads this — if 3 of 5 signals failed, the rationale
  honestly reflects the missing context rather than papering over.
- Section headers are recruiter-readable (no engineer vocab) per
  ``docs/cloris-ia-doctrine-cursor.md`` voice/copy rules.
- The function is pure and order-stable: callers control which
  signals to fetch and in what order; assembly preserves that
  ordering verbatim.

Slice 3 ships the assembler with Perplexity as the only signal.
Slices 4-5 will route additional adapters through the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from shared.brief_schema import Brief
from shared.schemas import CandidateProfileSummary

from exec_search.signals import (
    ExecutiveSignalSource,
    SIGNAL_REGISTRY,
    SignalFailure,
    SignalRequestContext,
    SignalResult,
    get_signal_source,
)


@dataclass(frozen=True)
class DossierEvidence:
    """Per-candidate dossier evidence: assembled prompt body + telemetry.

    ``prompt_body`` is what callers fold into the full-eval user
    message. ``signal_outcomes`` is the per-source result dict for
    telemetry (which signals fired, which failed, with detail).
    """

    prompt_body: str
    signal_outcomes: dict[str, SignalResult | SignalFailure]


def _profile_text(candidate: CandidateProfileSummary) -> str:
    """Render the LinkedIn profile summary into the same shape the
    LinkedIn full-eval user message uses.

    Mirrors the body of ``shared.judger.full_judge`` but without the
    surrounding header — the dossier prompt folds this into a section
    of the broader user message.
    """

    exp_lines: list[str] = []
    for e in candidate.experiences or []:
        bullets = "; ".join(e.summary_bullets) if e.summary_bullets else "no details"
        exp_lines.append(f"- {e.title} at {e.company} ({e.start}-{e.end}): {bullets}")

    edu_lines: list[str] = []
    for e in candidate.education or []:
        edu_lines.append(f"- {e.degree} in {e.field}, {e.school} ({e.start}-{e.end})")

    skills = ", ".join(candidate.skills_snippet) if candidate.skills_snippet else "none listed"

    return (
        f"Name: {candidate.name}\n"
        f"Headline: {candidate.headline}\n\n"
        f"Experience:\n"
        f"{chr(10).join(exp_lines) if exp_lines else 'None listed'}\n\n"
        f"Education:\n"
        f"{chr(10).join(edu_lines) if edu_lines else 'None listed'}\n\n"
        f"Skills: {skills}"
    )


def fetch_signals(
    *,
    candidate: CandidateProfileSummary,
    brief: Brief,
    context: SignalRequestContext,
    sources: Iterable[str] | None = None,
) -> dict[str, SignalResult | SignalFailure]:
    """Fetch a set of signals against a candidate, return per-source outcomes.

    ``sources`` controls which adapters to invoke. ``None`` (default)
    means "all registered adapters." The returned dict preserves
    insertion order for deterministic dossier-section ordering.

    Adapter exceptions are caught defensively (an adapter that
    misbehaves and raises despite the contract still degrades to a
    single-source failure rather than aborting the whole dossier).
    """

    if sources is None:
        sources = tuple(SIGNAL_REGISTRY.keys())

    outcomes: dict[str, SignalResult | SignalFailure] = {}
    for source_name in sources:
        try:
            source = get_signal_source(source_name)
        except KeyError:
            outcomes[source_name] = SignalFailure(
                source=source_name,
                reason="unknown_source",
                detail=f"{source_name} is not registered",
            )
            continue
        try:
            outcome = source.fetch(
                candidate=candidate,
                brief=brief,
                context=context,
            )
        except Exception as exc:
            outcomes[source_name] = SignalFailure(
                source=source_name,
                reason="adapter_exception",
                detail=f"{exc.__class__.__name__}: {exc}",
            )
            continue
        outcomes[source_name] = outcome
    return outcomes


def assemble_dossier_evidence(
    *,
    candidate: CandidateProfileSummary,
    brief: Brief,
    context: SignalRequestContext,
    sources: Iterable[str] | None = None,
) -> DossierEvidence:
    """Build the dossier-evaluation user-message body.

    Composition:

      1. Candidate profile (LinkedIn-derived) — required.
      2. Per-source signal sections, in registry order (or the order
         the caller passed via ``sources``). Successful signals render
         their own section text verbatim; failed signals render a
         one-line "section unavailable" placeholder.

    The dossier full-eval prompt expects this body to be the user
    message. The system prompt (assembled by
    :func:`linkedin.judgment_templates.assemble_full_evaluation_system`
    in dossier mode) instructs the LLM to weave the profile +
    signals into a 2-paragraph ``DOSSIER_RATIONALE``.
    """

    outcomes = fetch_signals(
        candidate=candidate,
        brief=brief,
        context=context,
        sources=sources,
    )

    sections: list[str] = []
    sections.append("## Candidate profile\n" + _profile_text(candidate))
    for source_name, outcome in outcomes.items():
        sections.append(_format_outcome_section(source_name, outcome))

    body = "\n\n".join(sections).strip()
    return DossierEvidence(prompt_body=body, signal_outcomes=outcomes)


def _format_outcome_section(
    source_name: str, outcome: SignalResult | SignalFailure
) -> str:
    """Render one signal outcome as a recruiter-readable section."""

    header = f"## Off-LinkedIn signal: {source_name}"
    if isinstance(outcome, SignalResult):
        return f"{header}\n{outcome.section_text}"
    detail = outcome.detail or outcome.reason
    return (
        f"{header}\n[Signal unavailable for this candidate "
        f"({outcome.reason}): {detail}]"
    )


__all__ = (
    "DossierEvidence",
    "assemble_dossier_evidence",
    "fetch_signals",
)
