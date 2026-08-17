"""Heuristic trigger gate for the candidate-level external-evidence step.

Slice 1 keeps this minimal and dependency-free: a pure function over the already
extracted ``CandidateProfileSummary``. The ``Brief`` is accepted in the signature
so that slice 2/3 can layer in role/brief-aware gating without rewiring callers,
but slice 1 deliberately does not consult it.

Audit Move #23 mirrors the same gate shape for the Researcher module
(``should_request_external_evidence_for_researcher``) so the
researcher full judge can cross-check against arXiv preprints and
news mentions before a final SAVE / NO call. The researcher gate is
keyed off :class:`researcher.schemas.ResearcherCandidate` instead of
``CandidateProfileSummary`` because the publication record is the
load-bearing surface for that module.

The gate never raises and never performs I/O.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shared.brief_schema import Brief
from shared.schemas import CandidateProfileSummary, TriggerDecision

if TYPE_CHECKING:
    from researcher.schemas import ResearcherCandidate

_PHD_TOKENS: tuple[str, ...] = ("phd", "ph.d", "doctor")


def _has_phd_education(summary: CandidateProfileSummary) -> bool:
    for edu in summary.education:
        degree = (edu.degree or "").lower()
        for token in _PHD_TOKENS:
            if token in degree:
                return True
    return False


def _bullet_count(summary: CandidateProfileSummary) -> int:
    return sum(len(exp.summary_bullets) for exp in summary.experiences)


def should_request_external_evidence(
    *,
    summary: CandidateProfileSummary,
    brief: Brief,
) -> TriggerDecision:
    """Decide whether to request external evidence augmentation for this candidate.

    Slice 1 only fires on two heuristics:

    - ``academic_context``: any education entry whose degree mentions PhD.
    - ``sparse_profile``: at most 2 experiences AND fewer than 3 total bullets.

    Anything else returns ``should_run=False`` with ``skip_reason="no_trigger_matched"``.
    The ``brief`` argument is reserved for slice 2/3 and is intentionally unused
    here (kept in the signature so callers don't break when policy lands).
    """

    del brief  # reserved for slice 2/3 — intentionally unused in slice 1.

    experience_count = len(summary.experiences)
    education_count = len(summary.education)
    bullet_total = _bullet_count(summary)
    has_phd = _has_phd_education(summary)

    base_signals: dict = {
        "experience_count": experience_count,
        "education_count": education_count,
        "bullet_total": bullet_total,
        "has_phd": has_phd,
    }

    if has_phd:
        return TriggerDecision(
            should_run=True,
            reason="academic_context",
            skip_reason="",
            signals={**base_signals, "fired": "academic_context"},
        )

    if experience_count <= 2 and bullet_total < 3:
        return TriggerDecision(
            should_run=True,
            reason="sparse_profile",
            skip_reason="",
            signals={**base_signals, "fired": "sparse_profile"},
        )

    return TriggerDecision(
        should_run=False,
        reason="",
        skip_reason="no_trigger_matched",
        signals={**base_signals, "fired": "none"},
    )


# ---------------------------------------------------------------------------
# Researcher gate — audit Move #23
# ---------------------------------------------------------------------------


def should_request_external_evidence_for_researcher(
    *,
    candidate: "ResearcherCandidate",
    brief: Brief | None = None,
) -> TriggerDecision:
    """Decide whether to request external evidence (arXiv preprints +
    news mentions) for a researcher candidate before the full judge.

    Audit Move #23. Mirrors :func:`should_request_external_evidence`
    in shape; differs in the trigger heuristics because researcher
    candidates ride a different signal surface (publication record,
    h-index, ORCID anchor, recent activity).

    Trigger heuristics — fires on ANY of:

    - ``orcid_missing``: the candidate has no ORCID anchor. arXiv +
      news cross-check disambiguates the identity (a "Wei Wang" with
      no ORCID at MIT may be 1 of 5 people; a recent arXiv
      ``cs.CL`` preprint or news mention can pin the right one).
    - ``thin_publication_record``: works_count < 5. Early-career
      researchers may have arXiv preprints not yet indexed in
      OpenAlex; news mentions of awards / talks add useful context.
    - ``recent_burst``: papers_in_window >= 5 with works_count <
      papers_in_window * 3. Suggests a recent activity spike that
      news / arXiv may corroborate (lab move, paper announcement,
      visibility event).

    Brief is reserved for future role-aware gating (e.g., post-trial
    customers may want "always cross-check for senior research staff
    roles"). The gate never raises and never performs I/O.
    """

    del brief  # reserved for future role-aware gating.

    orcid = (candidate.orcid or "").strip()
    works_count = int(candidate.works_count or 0)
    papers_in_window = int(candidate.papers_in_window or 0)
    h_index = int(candidate.h_index or 0)
    has_orcid = bool(orcid)

    base_signals: dict = {
        "has_orcid": has_orcid,
        "works_count": works_count,
        "papers_in_window": papers_in_window,
        "h_index": h_index,
    }

    if not has_orcid:
        return TriggerDecision(
            should_run=True,
            reason="orcid_missing",
            skip_reason="",
            signals={**base_signals, "fired": "orcid_missing"},
        )

    if 0 < works_count < 5:
        return TriggerDecision(
            should_run=True,
            reason="thin_publication_record",
            skip_reason="",
            signals={**base_signals, "fired": "thin_publication_record"},
        )

    if papers_in_window >= 5 and works_count < papers_in_window * 3:
        return TriggerDecision(
            should_run=True,
            reason="recent_burst",
            skip_reason="",
            signals={**base_signals, "fired": "recent_burst"},
        )

    return TriggerDecision(
        should_run=False,
        reason="",
        skip_reason="no_trigger_matched",
        signals={**base_signals, "fired": "none"},
    )
