"""A verdict that quotes the candidate is not a refusal.

The full and facial parsers used to run a lowercase substring search for
refusal phrases ("as an ai", "decline to", "refuse to", "cannot assist" …)
over the WHOLE reply before reading anything else, and return
PARSE_FAILURE "REFUSED: model declined to judge …" on a hit. The judge
quotes the candidate's profile as evidence, and profiles say things like
"serving as an AI/ML Cloud Consultant". Every live REFUSED row in
``full_parse_failures.jsonl`` (17 of them, seven of them saves) carried a
complete, valid ``DECISION:`` line — zero were true refusals. One of the 17
tripped on the judge's OWN prose ("the rules explicitly refuse to convert a
title into a save"), not a quotation, so scoping the markers to quoted
evidence would not have fixed the class.

The rule now: refusal detection runs only on a reply the parser cannot read
a verdict from. The receipt layer in ``shared.judger`` had the same class on
a different input — it scanned the prompt capture, which carries the
candidate's profile text, so any parse failure on a profile that said "as an
AI engineer" was labelled ``refused``. It now reads the parser's own REFUSED
label off the rationale and nothing else.

Fixtures are SYNTHETIC: they keep the live replies' shape and trigger
phrases, with invented employers, schools and numbers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import shared.judger as judger
from shared.judgment.templates import (
    _MODEL_REFUSAL_MARKERS,
    parse_facial_response,
    parse_full_evaluation_response,
)
from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    Education,
    Experience,
)

AREA = "Customer-facing technical engagement"
AREAS = (AREA, "Applied model evaluation")

# Shape of the live field-engineering reply that was discarded (it ended
# DECISION: INFERENTIAL_SAVE and tripped on the quoted "as an AI/ML Cloud
# Consultant"). Employer, school, client and numbers are invented.
FORWARD_DEPLOYED_INFERENTIAL_SAVE = """STEP_1_MATCH: DIRECT
STEP_1_AREA: Customer-facing technical engagement
STEP_1_EVIDENCE: The current seat is titled "AI/ML Forward Deployed Engineer / Cloud Consultant - Public Sector," embedded with an engagement team for a civic agency and framed in the About as professional-services delivery ("Currently serving as an AI/ML Cloud Consultant within the Professional Services team at Nimbus Cloud… solution architecture… contributing to robust cloud solutions"). That is the technical voice inside a customer engagement, which is this area's purpose.

STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: Named, owned artifacts: "Designed, built, and deployed an end-to-end log analytics platform… from infrastructure provisioning to automated incident response"; "Authored 12 Terraform modules with 150 automated tests." These are first-person build items, not consumed output.

STEP_3_TRANSFERABILITY: N/A
STEP_3_EVIDENCE: N/A

STEP_1_RECENCY: RECENT
STEP_4_LEVEL: BELOW
STEP_4_EVIDENCE: Every role is an internship held while finishing a BS (Lakeshore State, 2026), so the register sits a rung under the band; the owned scope exceeds assigned-slice intern work, so this is a one-rung level question rather than a level verdict.
STEP_5_COHERENCE: COHERENT
STEP_5_DRIVER: N/A
STEP_6_CALIBER: SOLID
STEP_6_EVIDENCE: (1) Owned an end-to-end analytics platform for a public-sector client, from IaC through automated remediation; (2) authored 12 Terraform modules with 150 automated tests.

STEP_7_NEAR_MISS: LEVEL

CASE_FOR: Direct, recent customer-facing technical work with hands-on pipeline and model-integration construction and named, owned artifacts.
CASE_AGAINST: The entire record is internships and student roles, sitting under the manager's experience sweet spot.

RESOLUTION: Exactly one dimension fails — LEVEL — while capability fit is direct and recent, depth is BUILDER, and caliber is SOLID, so this is the inferential channel's shape.

DECISION: INFERENTIAL_SAVE
CONFIDENCE: 0.45
REJECT_REASON: NONE
OUTREACH_TIER: STANDARD
POST_SAVE_MODIFIER: NONE
SUMMARY: Owned forward-deployed building evidence gives direct capability fit; the internship-only record is a level question for the outreach conversation."""


# Shape of the live strategic-project-lead reply whose trigger was the
# judge's OWN reasoning, not a quotation of the profile.
TITLE_ONLY_REJECT_OWN_PROSE = """STEP_1_MATCH: ADJACENT
STEP_1_AREA: Customer-facing technical engagement
STEP_1_EVIDENCE: The only signal is a "Technical Program Manager" title at a strong employer; no bullet names an artifact, a client, or a system.

STEP_2_DEPTH: UNKNOWN
STEP_2_EVIDENCE: Nothing in the About or the bullets establishes ownership or construction work.

STEP_3_TRANSFERABILITY: NOT_TRANSFERABLE
STEP_3_EVIDENCE: There is no methodology on the page to transfer.

STEP_1_RECENCY: CURRENT
STEP_4_LEVEL: UNCLEAR
STEP_4_EVIDENCE: Title register only; no scope is stated.
STEP_5_COHERENCE: UNCLEAR
STEP_5_DRIVER: N/A
STEP_6_CALIBER: UNKNOWN
STEP_6_EVIDENCE: N/A

STEP_7_NEAR_MISS: NONE

CASE_FOR: A program title at a strong employer.
CASE_AGAINST: The employer-signal rules and the brief's non-fit patterns explicitly refuse to convert a title at a strong employer into a save, and nothing else is on the page.

RESOLUTION: The case against wins; this is a title-only profile missing the technical gate outright.

DECISION: REJECT
CONFIDENCE: 0.25
REJECT_REASON: CAPABILITY_INSUFFICIENT
OUTREACH_TIER: NONE
POST_SAVE_MODIFIER: NONE
SUMMARY: Title-only profile; no owned technical work to evaluate against the gate."""


def _save_quoting(evidence_quote: str) -> str:
    """A valid V2 SAVE whose capability evidence quotes the profile."""

    return f"""STEP_1_MATCH: DIRECT
STEP_1_AREA: Applied model evaluation
STEP_1_EVIDENCE: The About reads "{evidence_quote}" and the current role bullets name an evaluation harness the candidate built and runs.

STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: "Built the regression-eval harness used by four product teams" is a named, owned artifact.

STEP_3_TRANSFERABILITY: N/A
STEP_3_EVIDENCE: N/A

STEP_1_RECENCY: CURRENT
STEP_4_LEVEL: ALIGNED
STEP_4_EVIDENCE: Owns a cross-team system at senior IC scope.
STEP_5_COHERENCE: COHERENT
STEP_5_DRIVER: N/A
STEP_6_CALIBER: STRONG
STEP_6_EVIDENCE: (1) Built the regression-eval harness four teams depend on; (2) led the eval migration that cut release regressions.

STEP_7_NEAR_MISS: NONE

CASE_FOR: Current, owned evaluation infrastructure squarely in the area.
CASE_AGAINST: No public writing to corroborate the scope.

DECISION: SAVE
CONFIDENCE: 0.8
REJECT_REASON: NONE
OUTREACH_TIER: PRIORITY
POST_SAVE_MODIFIER: NONE
SUMMARY: Builds and owns the evaluation harness this role exists to run."""


# Quotations a real profile plausibly carries, one per refusal marker, so the
# whole marker population is exercised, not just the two that fired live.
_QUOTE_FOR_MARKER = {
    "as an ai": "Serving as an AI/ML Cloud Consultant for public-sector clients",
    "i cannot comply": "Wrote the playbook for when I cannot comply with a client's data-residency ask",
    "i can't comply": "Built the escalation path for requests I can't comply with under HIPAA",
    "cannot comply": "Flagged vendors that cannot comply with SOC 2 controls",
    "can't comply": "Rebuilt the intake form so teams that can't comply get an exemption path",
    "unable to comply": "Audited suppliers unable to comply with the new export rules",
    "i cannot assist": "Created the triage bot that tells users when I cannot assist and routes them",
    "i can't assist": "Wrote the fallback copy for when I can't assist a caller directly",
    "cannot assist": "Built handoff tooling for cases the chatbot cannot assist with",
    "can't assist": "Owned the queue for tickets tier-1 can't assist with",
    "decline to": "I decline to chase benchmarks; I ship evaluation harnesses",
    "refuse to": "I refuse to ship a model without a regression gate",
    "not able to evaluate": "Built tooling for prompts the old harness was not able to evaluate",
}


def test_every_refusal_marker_has_a_quotation_fixture():
    assert set(_QUOTE_FOR_MARKER) == set(_MODEL_REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# Full parser: a verdict-carrying reply is never a refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("semantic_v2", [True, False])
def test_full_reply_quoting_as_an_ai_keeps_its_inferential_save(semantic_v2):
    result = parse_full_evaluation_response(
        FORWARD_DEPLOYED_INFERENTIAL_SAVE,
        require_semantic_v2=semantic_v2,
        capability_areas=AREAS,
    )

    assert result.decision == "INFERENTIAL_SAVE"
    assert "REFUSED" not in result.summary
    assert result.outreach_tier == "STANDARD"
    assert result.capability_area == AREA
    assert result.confidence == 0.45


@pytest.mark.parametrize("semantic_v2", [True, False])
def test_full_reply_whose_own_prose_says_refuse_to_keeps_its_reject(semantic_v2):
    result = parse_full_evaluation_response(
        TITLE_ONLY_REJECT_OWN_PROSE,
        require_semantic_v2=semantic_v2,
        capability_areas=AREAS,
    )

    assert result.decision == "REJECT"
    assert result.reject_reason == "CAPABILITY_INSUFFICIENT"
    assert "REFUSED" not in result.case_against


@pytest.mark.parametrize("marker", sorted(_QUOTE_FOR_MARKER))
def test_full_save_quoting_any_refusal_marker_keeps_its_save(marker):
    reply = _save_quoting(_QUOTE_FOR_MARKER[marker])
    assert marker in reply.lower()

    result = parse_full_evaluation_response(
        reply,
        require_semantic_v2=True,
        capability_areas=AREAS,
    )

    assert result.decision == "SAVE"
    assert result.outreach_tier == "PRIORITY"
    assert "REFUSED" not in result.summary


# ---------------------------------------------------------------------------
# Facial parser: same rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            'DECISION: FACIAL_YES\nREASON: Headline reads "Serving as an AI/ML '
            'Cloud Consultant" inside a cloud provider\'s professional-services team.',
            "FACIAL_YES",
        ),
        (
            'DECISION: FACIAL_NO\nREASON: Every role is sales; the headline says '
            '"I decline to write code anymore."',
            "FACIAL_NO",
        ),
        (
            'DECISION: FACIAL_BORDERLINE\nREASON: Headline "I refuse to ship '
            'without evals" fits, but the titles cannot resolve depth.',
            "FACIAL_BORDERLINE",
        ),
        (
            "DECISION: NO\nREASON: The rules refuse to open a profile on a "
            "generic title, and that is all this snippet has.",
            "FACIAL_NO",
        ),
    ],
)
def test_facial_reply_quoting_a_marker_keeps_its_decision(reply, expected):
    result = parse_facial_response(reply)

    assert result.decision == expected
    assert not result.reason.startswith("REFUSED")


def test_facial_reply_without_a_decision_line_still_reads_its_single_class():
    reply = (
        "FACIAL_YES — the headline reads 'serving as an AI engineer' on an "
        "evaluation team, which connects directly to the capability area."
    )

    result = parse_facial_response(reply)

    assert result.decision == "FACIAL_YES"
    assert result.reason == "parsed from raw"


# ---------------------------------------------------------------------------
# True refusals are still refusals
# ---------------------------------------------------------------------------

TRUE_REFUSALS = (
    "As an AI, I can't evaluate this candidate.",
    "I cannot assist with ranking real people for employment decisions.",
    "I'm sorry, but I decline to judge this profile.",
    # A DECISION line that names no verdict is not a verdict.
    "DECISION: I cannot assist with this request.",
    # Nor is one that names a failure token.
    "DECISION: PARSE_FAILURE\nAs an AI, I can't evaluate this candidate.",
)


@pytest.mark.parametrize("reply", TRUE_REFUSALS)
@pytest.mark.parametrize("semantic_v2", [True, False])
def test_full_true_refusal_is_still_refused(reply, semantic_v2):
    result = parse_full_evaluation_response(
        reply,
        require_semantic_v2=semantic_v2,
        capability_areas=AREAS,
    )

    assert result.decision == "PARSE_FAILURE"
    assert result.summary == "REFUSED: model declined to judge full fit"
    assert result.case_against == "REFUSED: model declined to judge full fit"


@pytest.mark.parametrize("reply", TRUE_REFUSALS)
def test_facial_true_refusal_is_still_refused(reply):
    result = parse_facial_response(reply)

    assert result.decision == "PARSE_FAILURE"
    assert result.reason == "REFUSED: model declined to judge facial fit"


def test_unparseable_reply_without_a_marker_stays_an_ordinary_parse_failure():
    full = parse_full_evaluation_response(
        "The profile is thin.", require_semantic_v2=True, capability_areas=AREAS
    )
    facial = parse_facial_response("The profile is thin.")

    assert full.decision == "PARSE_FAILURE"
    assert not full.summary.startswith("REFUSED")
    assert facial.decision == "PARSE_FAILURE"
    assert facial.reason == "could not parse facial decision"


# ---------------------------------------------------------------------------
# Judge receipts: the refusal label comes from the parser, not the profile
# ---------------------------------------------------------------------------


def _brief():
    brief = MagicMock()
    brief.has_v2_schema = True
    brief._new_brief = MagicMock()
    brief._new_brief.capability_area_names.return_value = list(AREAS)
    brief._new_brief.post_save_modifiers = []
    brief._new_brief.dossier_mode = False
    return brief


def _snippet_saying_as_an_ai() -> CandidateSnippet:
    return CandidateSnippet(
        name="Riya Example",
        headline="Serving as an AI/ML Cloud Consultant",
        current_title="Forward Deployed Engineer",
        current_company="Nimbus Cloud",
        location="Austin",
        education_snippet="BS, Lakeshore State",
        profile_url="https://www.linkedin.com/in/riya-example",
        source_string_id=1,
        source_string_name="string",
        page=1,
        result_rank=1,
    )


def _summary_saying_as_an_ai() -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name="Riya Example",
        profile_url="https://www.linkedin.com/in/riya-example",
        headline="Serving as an AI/ML Cloud Consultant",
        experiences=[
            Experience(
                title="Forward Deployed Engineer",
                company="Nimbus Cloud",
                start="2025",
                end="present",
                summary_bullets=["I refuse to ship without a regression gate"],
            )
        ],
        education=[Education(degree="BS", school="Lakeshore State", field="CS")],
        skills_snippet=["python", "terraform"],
    )


def test_facial_parse_failure_on_an_as_an_ai_profile_is_parse_fail_not_refused():
    with (
        patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False),
        patch.object(judger, "assemble_facial_system", return_value="SYSTEM"),
        patch.object(
            judger,
            "facial_llm",
            return_value="DECISION: MAYBE\nREASON: malformed verdict.",
        ),
    ):
        decision = judger.facial_judge(_snippet_saying_as_an_ai(), _brief())

    assert decision.decision == "PARSE_FAILURE"
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["actual_status"] == "parse_fail"
    assert receipt["actual_detail"]["parse_status"] == "parse_fail"


def test_full_parse_failure_on_an_as_an_ai_profile_is_parse_fail_not_refused():
    # A stated verdict that fails the V2 contract (fields missing) is an
    # honest parse failure, and the profile's own words do not relabel it.
    with (
        patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False),
        patch.object(judger, "assemble_full_evaluation_system", return_value="SYSTEM"),
        patch.object(
            judger,
            "opus_llm_cached",
            return_value="DECISION: SAVE\nCONFIDENCE: 0.8\nSUMMARY: Strong.",
        ),
    ):
        decision = judger.full_judge(_summary_saying_as_an_ai(), _brief())

    assert decision.decision == "PARSE_FAILURE"
    assert "REFUSED" not in decision.rationale
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["actual_status"] == "parse_fail"
    assert receipt["actual_detail"]["parse_status"] == "parse_fail"


def test_full_judge_keeps_the_verdict_and_an_ok_receipt_end_to_end():
    with (
        patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False),
        patch.object(judger, "assemble_full_evaluation_system", return_value="SYSTEM"),
        patch.object(
            judger,
            "opus_llm_cached",
            return_value=FORWARD_DEPLOYED_INFERENTIAL_SAVE,
        ),
    ):
        decision = judger.full_judge(_summary_saying_as_an_ai(), _brief())

    assert decision.decision == "INFERENTIAL_SAVE"
    assert decision.outreach_tier == "STANDARD"
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["actual_status"] == "ok"


def test_full_true_refusal_on_an_as_an_ai_profile_is_still_a_refused_receipt():
    with (
        patch("shared.config.SHADOW_FACIAL_MODEL_ENABLED", False),
        patch.object(judger, "assemble_full_evaluation_system", return_value="SYSTEM"),
        patch.object(
            judger,
            "opus_llm_cached",
            return_value="As an AI, I can't evaluate this candidate.",
        ),
    ):
        decision = judger.full_judge(_summary_saying_as_an_ai(), _brief())

    assert decision.decision == "PARSE_FAILURE"
    assert decision.rationale.startswith("REFUSED:")
    receipt = decision.prompt_capture["judge_receipt"]
    assert receipt["actual_status"] == "refused"
    assert receipt["actual_detail"]["parse_status"] == "refused"
