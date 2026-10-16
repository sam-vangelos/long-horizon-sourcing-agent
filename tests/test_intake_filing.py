"""Unit tests for shared/intake_filing.py (pure filing readiness logic)."""

from __future__ import annotations

from shared.intake_filing import (
    filing_blockers,
    filing_readiness,
    filing_readiness_wire,
    http_status_for_blocker,
    is_conversational_session,
    primary_filing_blocker,
)


def _minimal_ready_draft() -> dict:
    return {
        "role_title": "Senior Tax Associate",
        "capability_areas": [
            {
                "name": "Tax compliance",
                "description": "Owns complex returns end-to-end.",
            }
        ],
        "role_summary": "Owns tax work for growth-stage clients.",
        "depth_distinction": {
            "builder_definition": "Builds the process.",
            "user_definition": "Uses the process.",
            "edge_case_guidance": "Escalate ambiguous filings.",
        },
    }


def test_wizard_at_review_with_valid_draft_has_no_blockers() -> None:
    session = {
        "current_step": "review",
        "state_json": {"v2_draft": _minimal_ready_draft()},
    }
    assert filing_blockers(session) == []
    assert filing_readiness(session).can_file is True


def test_conversational_insufficient_draft_requires_compose() -> None:
    session = {
        "current_step": "conversation",
        "state_json": {
            "messages": [{"role": "cloris", "content": "Hi"}],
            "v2_draft": {"role_title": "TBD"},
        },
    }
    blocker = primary_filing_blocker(session)
    assert blocker is not None
    assert blocker.code == "intake_compose_required"
    assert http_status_for_blocker(blocker) == 422


def test_conversational_ready_draft_not_at_review_requires_compose() -> None:
    session = {
        "current_step": "conversation",
        "state_json": {
            "messages": [{"role": "cloris", "content": "Hi"}],
            "v2_draft": _minimal_ready_draft(),
        },
    }
    blocker = primary_filing_blocker(session)
    assert blocker is not None
    assert blocker.code == "intake_compose_required"


def test_conversational_at_review_with_ready_draft_can_file() -> None:
    session = {
        "current_step": "review",
        "state_json": {
            "conversation_meta": {"turn_count": 3},
            "v2_draft": _minimal_ready_draft(),
        },
    }
    assert is_conversational_session(session) is True
    assert filing_blockers(session) == []


def test_synthesis_running_blocks_before_compose_required() -> None:
    session = {
        "current_step": "conversation",
        "state_json": {
            "messages": [{"role": "cloris", "content": "Hi"}],
            "source_packet_synthesis": {"status": "running"},
        },
    }
    blockers = filing_blockers(session)
    assert [b.code for b in blockers] == [
        "intake_synthesis_in_progress",
        "intake_compose_required",
    ]
    assert http_status_for_blocker(blockers[0]) == 409


def test_filing_readiness_includes_extended_fields() -> None:
    session = {
        "current_step": "review",
        "state_json": {
            "v2_draft": _minimal_ready_draft(),
            "source_packet_synthesis": {"status": "running"},
            "conversation_compose": {"status": "composing"},
        },
    }
    readiness = filing_readiness(session)
    assert readiness.valid_v2_draft is True
    assert readiness.in_flight_synthesis is True
    assert readiness.in_flight_compose is True
    assert readiness.can_file is False

    wire = filing_readiness_wire(session)
    assert wire["valid_v2_draft"] is True
    assert wire["in_flight_synthesis"] is True
    assert "intake_synthesis_in_progress" in wire["blocking_codes"]


def test_insight_deficits_do_not_block_when_v2_valid() -> None:
    session = {
        "current_step": "review",
        "state_json": {
            "v2_draft": _minimal_ready_draft(),
            "intake_insights": {},
        },
    }
    readiness = filing_readiness(session)
    assert readiness.can_file is True
    assert len(readiness.insight_deficits) == 1
    assert readiness.insight_deficits[0]["field"] == "hiring_manager_success_image"


def test_compose_in_progress_returns_409() -> None:
    session = {
        "current_step": "review",
        "state_json": {
            "conversation_compose": {"status": "composing"},
            "v2_draft": _minimal_ready_draft(),
        },
    }
    blocker = primary_filing_blocker(session)
    assert blocker is not None
    assert blocker.code == "intake_compose_in_progress"
    assert http_status_for_blocker(blocker) == 409
