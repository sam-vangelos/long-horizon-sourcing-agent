"""Combine identity, holistic fit, and Recruiter engagement into SAVE / MANUAL_REVIEW / REJECT."""

from __future__ import annotations

from dataclasses import dataclass

from shared.failures import is_failure_decision
from shared.reconciliation_schemas import RecruiterActivitySnapshot
from shared.schemas import CandidateProfileSummary, OpusDecision


@dataclass(frozen=True)
class ReconciliationGateResult:
    final_action: str
    final_subreason: str
    engagement_blocked_save: bool


FIT_PASS_DECISIONS = frozenset({"SAVE", "SIGNAL_SAVE", "TRANSFERABLE_SAVE"})
FIT_BORDERLINE_DECISIONS = frozenset({"INFERENTIAL_SAVE"})


def engagement_blocks_save(
    *,
    already_saved_card: bool,
    profile_status: RecruiterActivitySnapshot | None,
    novelty_pressure: str,
    reachout_status: str,
) -> tuple[bool, str]:
    """Return (blocked, subreason_if_blocked) when auto-save should not run."""
    if already_saved_card:
        return True, "already_saved_elsewhere"
    if reachout_status == "recent_outbound_contact":
        return True, "already_worked"
    if novelty_pressure == "high":
        return True, "low_novelty"
    if profile_status and profile_status.message_count >= 6:
        return True, "already_worked"
    return False, ""


def decide_final_reconciliation_action(
    *,
    identity_high_confidence: bool,
    identity_name_mismatch: bool,
    had_plausible_cards: bool,
    holistic_decision: OpusDecision | None,
    profile_summary: CandidateProfileSummary | None,
    already_saved_card: bool,
    profile_status: RecruiterActivitySnapshot | None,
    novelty_pressure: str,
    reachout_status: str,
    extraction_failed: bool,
) -> ReconciliationGateResult:
    """Map gates to canonical SAVE / MANUAL_REVIEW / REJECT + structured subreason."""
    if extraction_failed:
        return ReconciliationGateResult(
            final_action="MANUAL_REVIEW",
            final_subreason="tool_failure",
            engagement_blocked_save=False,
        )

    if not had_plausible_cards:
        return ReconciliationGateResult(
            final_action="REJECT",
            final_subreason="no_plausible_profile",
            engagement_blocked_save=False,
        )

    if not identity_high_confidence or identity_name_mismatch:
        return ReconciliationGateResult(
            final_action="MANUAL_REVIEW",
            final_subreason="identity_ambiguous",
            engagement_blocked_save=False,
        )

    blocked, eng_reason = engagement_blocks_save(
        already_saved_card=already_saved_card,
        profile_status=profile_status,
        novelty_pressure=novelty_pressure,
        reachout_status=reachout_status,
    )

    if holistic_decision is None or is_failure_decision(holistic_decision.decision):
        return ReconciliationGateResult(
            final_action="MANUAL_REVIEW",
            final_subreason="tool_failure",
            engagement_blocked_save=blocked,
        )

    decision = holistic_decision.decision

    if decision in FIT_BORDERLINE_DECISIONS:
        return ReconciliationGateResult(
            final_action="MANUAL_REVIEW",
            final_subreason="borderline_fit",
            engagement_blocked_save=blocked,
        )

    if decision == "REJECT" or decision not in FIT_PASS_DECISIONS:
        return ReconciliationGateResult(
            final_action="REJECT",
            final_subreason="fit_reject",
            engagement_blocked_save=blocked,
        )

    # Fit passed (SAVE family strict)
    if blocked:
        action = "MANUAL_REVIEW" if eng_reason == "already_saved_elsewhere" else "REJECT"
        return ReconciliationGateResult(
            final_action=action,
            final_subreason=eng_reason,
            engagement_blocked_save=True,
        )

    # Defensive: require a parsed profile for audit trail
    if profile_summary is None or not (profile_summary.name or "").strip():
        return ReconciliationGateResult(
            final_action="MANUAL_REVIEW",
            final_subreason="tool_failure",
            engagement_blocked_save=False,
        )

    return ReconciliationGateResult(
        final_action="SAVE",
        final_subreason="",
        engagement_blocked_save=False,
    )
