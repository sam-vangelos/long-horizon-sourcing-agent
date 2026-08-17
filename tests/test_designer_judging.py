"""Designer judge parser contracts."""

from __future__ import annotations

import pytest

from designer.judging import designer_full_judge, fail_honest_full_decision
from designer.schemas import DesignerCandidate, DesignerSnippet


def _candidate() -> DesignerCandidate:
    return DesignerCandidate(
        snippet=DesignerSnippet(
            source="google_cse",
            identity_key="cse:example.com/designer",
            display_name="Alex Designer",
            profile_url="https://example.com/designer",
            headline="Product designer",
        )
    )


@pytest.mark.parametrize("decision", ["TRANSFERABLE_SAVE", "SIGNAL_SAVE"])
def test_designer_full_judge_preserves_save_family_decisions(decision: str) -> None:
    parsed = designer_full_judge(
        _candidate(),
        brief={},
        llm_caller=lambda _system, _user: (
            f"DECISION: {decision}\n"
            "PATH: design_systems\n"
            "CONFIDENCE: 0.72\n"
            "SUMMARY: Strong adjacent design-systems signal."
        ),
    )

    assert parsed.decision == decision
    assert parsed.path == "design_systems"
    assert parsed.confidence == 0.72


@pytest.mark.parametrize(
    "raw",
    [
        "DECISION: MAYBE\nPATH: unclear\nCONFIDENCE: 0.8\nSUMMARY: malformed",
        "PATH: missing_decision\nCONFIDENCE: 0.8\nSUMMARY: malformed",
    ],
)
def test_designer_full_judge_malformed_decision_is_parse_failure(raw: str) -> None:
    parsed = designer_full_judge(
        _candidate(),
        brief={},
        llm_caller=lambda _system, _user: raw,
    )

    assert parsed.decision == "PARSE_FAILURE"
    assert parsed.path == "none"
    assert parsed.confidence == 0.0
    assert "PARSE_FAILURE" in parsed.rationale


def test_non_ok_parse_status_never_maps_to_reject() -> None:
    decision, path, confidence, rationale = fail_honest_full_decision(
        {
            "parse_status": "parse_fail",
            "decision": "REJECT",
            "path": "would_have_rejected",
            "confidence": 0.91,
            "rationale": "malformed response should not become negative decision",
        }
    )

    assert decision == "PARSE_FAILURE"
    assert path == "none"
    assert confidence == 0.0
    assert "PARSE_FAILURE" in rationale


def test_designer_full_judge_emits_judge_receipt() -> None:
    parsed = designer_full_judge(
        _candidate(),
        brief={},
        llm_caller=lambda _system, _user: (
            "DECISION: SAVE\n"
            "PATH: product_design\n"
            "CONFIDENCE: 0.84\n"
            "SUMMARY: Strong direct signal."
        ),
    )

    receipt = parsed.prompt_capture["judge_receipt"]
    assert receipt["receipt_type"] == "judge"
    assert receipt["actual_status"] == "ok"
    assert receipt["input_hash"].startswith("sha256:")
    assert receipt["intended_postcondition"]
    assert receipt["actual_detail"] == {
        "parse_status": "ok",
        "final_decision": "SAVE",
    }


def test_designer_full_judge_refusal_emits_refused_receipt() -> None:
    parsed = designer_full_judge(
        _candidate(),
        brief={},
        llm_caller=lambda _system, _user: (
            "I cannot comply with this request or evaluate this candidate."
        ),
    )

    assert parsed.decision == "PARSE_FAILURE"
    assert "REFUSED" in parsed.rationale
    receipt = parsed.prompt_capture["judge_receipt"]
    assert receipt["actual_status"] == "refused"
    assert receipt["actual_detail"]["parse_status"] == "refused"
    assert receipt["actual_detail"]["final_decision"] == "PARSE_FAILURE"
