"""Designer launch-readiness probe — CSE-primary contract (audit Move #14).

Phase 2.2 of the multi-agent-execution plan, refactored for audit
Move #14. Provides a callable ``probe_designer_readiness()`` that the
launch-readiness aggregator at ``cloris/api.py:_readiness_blockers``
dispatches via ``LAUNCHERS["designer"].readiness_probe_fn()``.

Mirrors :mod:`linkedin.health` / :mod:`github.health` so the API layer
can union the four sources without per-source branching at the route
layer (post-Slice 1.1 the dispatch is registry-driven; this slice
fills the registry slot for designer).

Per audit Move #14: Behance is no longer a hard blocker. Adobe
stopped accepting new Behance v2 keys in 2020 — gating Designer
launches on a key the recruiter literally cannot acquire is a dead
end. Google CSE is the new primary acquisition surface; Behance
becomes an optional augment when configured.

Hard blockers (``ready=False``):

- ``ANTHROPIC_API_KEY`` missing — Designer's vision evaluators
  (facial triage + full visual judgment) call Claude. Without it
  the run fails at the first evaluator call.
- BOTH ``GOOGLE_CSE_API_KEY``/``GOOGLE_CSE_ID`` AND ``BEHANCE_API_KEY``
  missing — Designer needs at least one acquisition source. CSE
  alone is sufficient; Behance alone is sufficient; both is the
  richest signal.

Informational notes (``ready=True``, surfaced as soft warnings):

- ``BEHANCE_API_KEY`` missing while CSE is configured — Designer
  runs CSE-only. Per Move #14 this is a supported launch posture.
  The recruiter sees an editorial note acknowledging the tradeoff
  (lower-res thumbnails vs Behance's structured taxonomy).
- ``GOOGLE_CSE_API_KEY``/``GOOGLE_CSE_ID`` missing while Behance is
  configured — Designer runs Behance-only (the legacy posture). The
  recruiter sees a note recommending CSE for richer discovery.

Out of scope: network probing Behance / CSE / Anthropic. Each
client's ``validate_credentials()`` is a real-network call and would
slow the pre-flight; they stay as run-time checks. The probe is
meant to be fast and offline-tolerant.
"""

from __future__ import annotations

import os

from shared import config
from shared.health_types import ReadinessBlocker, ReadinessReport


def probe_designer_readiness(
    *,
    behance_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    google_cse_api_key: str | None = None,
    google_cse_id: str | None = None,
) -> ReadinessReport:
    """Launch-readiness probe for Designer (CSE-primary per Move #14).

    Args:
        behance_api_key: Override the ``BEHANCE_API_KEY`` env var.
            Defaults to the current process environment value.
        anthropic_api_key: Override the ``ANTHROPIC_API_KEY`` env var.
            Defaults to ``shared.config.ANTHROPIC_API_KEY``.
        google_cse_api_key: Override the ``GOOGLE_CSE_API_KEY`` env var.
        google_cse_id: Override the ``GOOGLE_CSE_ID`` env var.

    Returns:
        :class:`ReadinessReport` with ``ready`` true iff
        ANTHROPIC_API_KEY is set AND at least one acquisition source
        (CSE or Behance) is configured. Soft warnings about absent
        sources land in ``informational_notes``.
    """

    blockers: list[ReadinessBlocker] = []
    notes: list[str] = []

    effective_behance = (
        behance_api_key
        if behance_api_key is not None
        else os.environ.get("BEHANCE_API_KEY", "")
    )
    effective_cse_key = (
        google_cse_api_key
        if google_cse_api_key is not None
        else os.environ.get("GOOGLE_CSE_API_KEY", "")
    )
    effective_cse_id = (
        google_cse_id
        if google_cse_id is not None
        else os.environ.get("GOOGLE_CSE_ID", "")
    )
    cse_configured = bool(effective_cse_key and effective_cse_id)

    # Hard block 1: at least one acquisition source must be configured.
    # CSE-primary contract — Behance is no longer required, but the
    # recruiter needs SOME way to discover candidates.
    if not cse_configured and not effective_behance:
        blockers.append(
            ReadinessBlocker(
                kind="config",
                message=(
                    "No Designer acquisition source configured (Google "
                    "CSE or Behance)."
                ),
                remediation=(
                    "Add GOOGLE_CSE_API_KEY + GOOGLE_CSE_ID to your "
                    ".env file (Designer's primary acquisition source "
                    "per the CSE-primary architecture). BEHANCE_API_KEY "
                    "is optional; configure it when available for richer "
                    "structured taxonomy signal — Adobe stopped issuing "
                    "new keys in 2020 so most new customers run "
                    "CSE-only."
                ),
            )
        )

    # Hard block 2: Anthropic key is always required for vision
    # judgment. Same posture across all source mixes.
    effective_anthropic = (
        anthropic_api_key
        if anthropic_api_key is not None
        else config.ANTHROPIC_API_KEY
    )
    if not effective_anthropic:
        blockers.append(
            ReadinessBlocker(
                kind="config",
                message="No Anthropic API key configured.",
                remediation=(
                    "Add ANTHROPIC_API_KEY to your .env file. "
                    "Designer's vision evaluators (batch facial triage "
                    "and full visual judgment) call Claude — without "
                    "a key the run fails at the first evaluator call."
                ),
            )
        )

    # Informational notes — surface source-mix tradeoffs without
    # blocking. The launch UI renders them inline.
    if cse_configured and not effective_behance:
        notes.append(
            "Designer will run CSE-only (Google Custom Search Engine). "
            "Configure BEHANCE_API_KEY for richer structured-taxonomy "
            "signal when you have a key; otherwise CSE is sufficient."
        )
    elif effective_behance and not cse_configured:
        notes.append(
            "Designer will run Behance-only. Configure GOOGLE_CSE_API_KEY "
            "+ GOOGLE_CSE_ID to also pull from personal portfolio sites "
            "(Cargo.site / Squarespace / etc.) — the CSE-primary "
            "architecture is the recommended posture."
        )

    return ReadinessReport(
        ready=not blockers,
        blockers=tuple(blockers),
        informational_notes=tuple(notes),
    )
