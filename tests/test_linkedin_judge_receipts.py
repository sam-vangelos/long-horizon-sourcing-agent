"""LinkedIn judge receipt contracts for fail-honest M1A."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import shared.judger as judger
from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    Education,
    Experience,
)


def _brief():
    brief = MagicMock()
    brief.has_v2_schema = True
    brief._new_brief = MagicMock()
    brief._new_brief.capability_area_names.return_value = ["ML evaluation"]
    brief._new_brief.post_save_modifiers = []
    return brief


def _snippet() -> CandidateSnippet:
    return CandidateSnippet(
        name="Jane Doe",
        headline="ML Researcher",
        current_title="Research Scientist",
        current_company="OpenLab",
        location="SF",
        education_snippet="PhD, MIT",
        profile_url="https://www.linkedin.com/in/janedoe",
        source_string_id=1,
        source_string_name="string",
        page=1,
        result_rank=1,
    )


def _summary() -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name="Jane Doe",
        profile_url="https://www.linkedin.com/in/janedoe",
        headline="ML Researcher",
        experiences=[
            Experience(
                title="Research Scientist",
                company="OpenLab",
                start="2020",
                end="present",
                summary_bullets=["LLM evals"],
            )
        ],
        education=[Education(degree="PhD", school="MIT", field="ML")],
        skills_snippet=["python", "pytorch"],
    )


def _full_eval(decision_block: str) -> str:
    return (
        "STEP_1_MATCH: DIRECT\n"
        "STEP_1_AREA: ML evaluation\n"
        "STEP_1_EVIDENCE: Built evaluation workflows.\n"
        "STEP_1_RECENCY: CURRENT\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: Owned tooling and experiments.\n"
        "STEP_3_TRANSFERABILITY: N/A\n"
        "STEP_3_EVIDENCE: Direct match.\n"
        "STEP_4_LEVEL: ALIGNED\n"
        "STEP_5_COHERENCE: COHERENT\n"
        "STEP_6_CALIBER: STRONG\n"
        "CASE_FOR: Direct capability signal.\n"
        "CASE_AGAINST: None.\n"
        "REJECT_REASON: NONE\n"
        "OUTREACH_TIER: STANDARD\n"
        + decision_block
    )


def test_linkedin_facial_parse_failure_emits_parse_fail_receipt() -> None:
    with (
        patch.object(judger, "assemble_facial_system", return_value="SYSTEM"),
        patch.object(
            judger,
            "facial_llm",
            return_value="DECISION: MAYBE\nREASON: malformed verdict.",
        ),
    ):
        decision = judger.facial_judge(_snippet(), _brief())

    assert decision.decision == "PARSE_FAILURE"
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["receipt_type"] == "judge"
    assert receipt["stage"] == "linkedin_facial_judge"
    assert receipt["actual_status"] == "parse_fail"
    assert receipt["input_hash"].startswith("sha256:")
    assert receipt["actual_detail"]["parse_status"] == "parse_fail"
    assert receipt["actual_detail"]["final_decision"] == "PARSE_FAILURE"
    assert "prompt_capture" not in decision.to_dict()


def test_linkedin_facial_refusal_emits_refused_receipt() -> None:
    with (
        patch.object(judger, "assemble_facial_system", return_value="SYSTEM"),
        patch.object(
            judger,
            "facial_llm",
            return_value="I cannot comply with this request or evaluate this profile.",
        ),
    ):
        decision = judger.facial_judge(_snippet(), _brief())

    assert decision.decision == "PARSE_FAILURE"
    assert "REFUSED" in decision.rationale
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["actual_status"] == "refused"
    assert receipt["actual_detail"]["parse_status"] == "refused"
    assert receipt["actual_detail"]["final_decision"] == "PARSE_FAILURE"


def test_linkedin_facial_stamps_usage_context_stage_without_clobbering() -> None:
    captured_contexts: list[dict] = []

    def _facial_llm(*args, **kwargs):
        captured_contexts.append(dict(kwargs["usage_context"]))
        return "DECISION: FACIAL_YES\nREASON: strong builder signal"

    with (
        patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False),
        patch.object(judger, "assemble_facial_system", return_value="SYSTEM"),
        patch.object(judger, "facial_llm", side_effect=_facial_llm),
    ):
        defaulted = judger.facial_judge(
            _snippet(),
            _brief(),
            lane_context={"lane_id": "lane-A"},
        )
        preserved = judger.facial_judge(
            _snippet(),
            _brief(),
            lane_context={"stage": "custom", "lane_id": "lane-B"},
        )

    assert defaulted.decision == "FACIAL_YES"
    assert preserved.decision == "FACIAL_YES"
    assert captured_contexts[0]["stage"] == "facial"
    assert captured_contexts[0]["lane_id"] == "lane-A"
    assert captured_contexts[1]["stage"] == "custom"
    assert captured_contexts[1]["lane_id"] == "lane-B"


def test_linkedin_facial_batch_stamps_usage_context_stage() -> None:
    captured_contexts: list[dict] = []

    def _facial_llm(*args, **kwargs):
        captured_contexts.append(dict(kwargs["usage_context"]))
        return "[1] FACIAL_YES | strong builder signal\n"

    with (
        patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False),
        patch.object(judger, "assemble_facial_batch_system", return_value="SYSTEM"),
        patch.object(judger, "facial_llm", side_effect=_facial_llm),
    ):
        decisions = judger.facial_judge_batch([_snippet()], _brief())

    assert decisions[0].decision == "FACIAL_YES"
    assert len(captured_contexts) == 1
    assert captured_contexts[0]["stage"] == "facial"
    assert captured_contexts[0]["logical_call_id"].startswith("judge-")
    assert captured_contexts[0]["judgment_contract_mode"] == "legacy"
    assert (
        captured_contexts[0]["judgment_contract_version"]
        == "linkedin_facial_legacy_text_v1"
    )
    assert captured_contexts[0]["batch_size"] == 1


def test_linkedin_full_save_emits_ok_receipt() -> None:
    captured_contexts: list[dict] = []

    def _full_llm(*args, **kwargs):
        captured_contexts.append(dict(kwargs["usage_context"]))
        return _full_eval(
            "DECISION: SAVE\n"
            "CONFIDENCE: 0.82\n"
            "POST_SAVE_MODIFIER: NONE\n"
            "SUMMARY: Strong direct signal.\n"
        )

    with (
        patch.object(judger, "assemble_full_evaluation_system", return_value="SYSTEM"),
        patch.object(judger, "opus_llm_cached", side_effect=_full_llm),
    ):
        decision = judger.full_judge(_summary(), _brief())

    assert decision.decision == "SAVE"
    assert decision.outreach_tier == "STANDARD"
    assert len(captured_contexts) == 1
    assert captured_contexts[0]["judgment_contract_mode"] == "legacy"
    assert (
        captured_contexts[0]["judgment_contract_version"]
        == "linkedin_full_legacy_text_v2"
    )
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["stage"] == "linkedin_full_judge"
    assert receipt["actual_status"] == "ok"
    assert receipt["actual_detail"]["parse_status"] == "ok"
    assert receipt["actual_detail"]["final_decision"] == "SAVE"


def test_linkedin_full_refusal_emits_refused_receipt() -> None:
    with (
        patch.object(judger, "assemble_full_evaluation_system", return_value="SYSTEM"),
        patch.object(
            judger,
            "opus_llm_cached",
            return_value="I cannot comply with this request or evaluate this profile.",
        ),
    ):
        decision = judger.full_judge(_summary(), _brief())

    assert decision.decision == "PARSE_FAILURE"
    assert "REFUSED" in decision.rationale
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["actual_status"] == "refused"
    assert receipt["actual_detail"]["parse_status"] == "refused"
    assert receipt["actual_detail"]["final_decision"] == "PARSE_FAILURE"


def test_linkedin_facial_judgment_failure_emits_error_receipt() -> None:
    with (
        patch.object(judger, "assemble_facial_system", return_value="SYSTEM"),
        patch.object(judger, "facial_llm", side_effect=RuntimeError("timeout")),
    ):
        decision = judger.facial_judge(_snippet(), _brief())

    assert decision.decision == "JUDGMENT_FAILURE"
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["actual_status"] == "error"
    assert receipt["actual_detail"]["parse_status"] == "error"
    assert receipt["actual_detail"]["final_decision"] == "JUDGMENT_FAILURE"
