from shared.reconciliation_schemas import RecruiterActivitySnapshot
from shared.recruiter_reconciliation_decision import decide_final_reconciliation_action
from shared.schemas import CandidateProfileSummary, OpusDecision


def _opus(decision: str) -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision=decision,
        path="DIRECT:1.X",
        confidence=0.9,
        rationale="r",
        candidate_name="n",
        profile_url="u",
    )


def test_save_requires_fit_and_engagement():
    summary = CandidateProfileSummary(name="Ada", profile_url="/u", headline="h")
    gate = decide_final_reconciliation_action(
        identity_high_confidence=True,
        identity_name_mismatch=False,
        had_plausible_cards=True,
        holistic_decision=_opus("SAVE"),
        profile_summary=summary,
        already_saved_card=False,
        profile_status=None,
        novelty_pressure="low",
        reachout_status="unworked",
        extraction_failed=False,
    )
    assert gate.final_action == "SAVE"
    assert gate.final_subreason == ""


def test_inferential_save_is_manual_borderline():
    summary = CandidateProfileSummary(name="Ada", profile_url="/u", headline="h")
    gate = decide_final_reconciliation_action(
        identity_high_confidence=True,
        identity_name_mismatch=False,
        had_plausible_cards=True,
        holistic_decision=_opus("INFERENTIAL_SAVE"),
        profile_summary=summary,
        already_saved_card=False,
        profile_status=None,
        novelty_pressure="low",
        reachout_status="unworked",
        extraction_failed=False,
    )
    assert gate.final_action == "MANUAL_REVIEW"
    assert gate.final_subreason == "borderline_fit"


def test_engagement_recent_outbound_rejects():
    summary = CandidateProfileSummary(name="Ada", profile_url="/u", headline="h")
    gate = decide_final_reconciliation_action(
        identity_high_confidence=True,
        identity_name_mismatch=False,
        had_plausible_cards=True,
        holistic_decision=_opus("SAVE"),
        profile_summary=summary,
        already_saved_card=False,
        profile_status=None,
        novelty_pressure="low",
        reachout_status="recent_outbound_contact",
        extraction_failed=False,
    )
    assert gate.final_action == "REJECT"
    assert gate.final_subreason == "already_worked"


def test_already_saved_card_blocks_auto_save():
    summary = CandidateProfileSummary(name="Ada", profile_url="/u", headline="h")
    gate = decide_final_reconciliation_action(
        identity_high_confidence=True,
        identity_name_mismatch=False,
        had_plausible_cards=True,
        holistic_decision=_opus("SAVE"),
        profile_summary=summary,
        already_saved_card=True,
        profile_status=None,
        novelty_pressure="low",
        reachout_status="unworked",
        extraction_failed=False,
    )
    assert gate.final_action == "MANUAL_REVIEW"
    assert gate.final_subreason == "already_saved_elsewhere"


def test_high_message_count_rejects():
    summary = CandidateProfileSummary(name="Ada", profile_url="/u", headline="h")
    status = RecruiterActivitySnapshot(message_count=6, project_count=0, view_count=0)
    gate = decide_final_reconciliation_action(
        identity_high_confidence=True,
        identity_name_mismatch=False,
        had_plausible_cards=True,
        holistic_decision=_opus("SAVE"),
        profile_summary=summary,
        already_saved_card=False,
        profile_status=status,
        novelty_pressure="low",
        reachout_status="messaged",
        extraction_failed=False,
    )
    assert gate.final_action == "REJECT"
    assert gate.final_subreason == "already_worked"
