"""Exa source module S4 — judgment wiring.

Exa rides the source-neutral V2 full judge, once per candidate, with no
facial stage. What is Exa's to get right is the wiring around it: which
profile summary and evidence object go in, which keys attribute the
spend, which bucket a verdict lands in, and what must NOT be caught on
the way out.

Most tests fake the judge at the ``exa.judge`` module level. One does
not: ``test_full_judge_rendered_user_message_carries_evidence`` runs the
real ``full_judge_with_external_evidence`` with only the provider call
stubbed, so the assertion is about the prompt a live run would actually
send. No network in either case.
"""

from __future__ import annotations

import json

import pytest

import shared.config as config
import shared.judger as judger
import exa.judge as judge_module
from exa.evidence import (
    EXA_TRIGGER_REASON,
    PROFILE_SOURCE_SUFFIX,
    RENDER_SOURCE_FRAMING,
    SOURCE_FRAMING,
    TRANSFERABILITY_CONTRACT_LINE,
)
from exa.linkedin_render import parse_exa_linkedin_markdown
from exa.judge import (
    BUCKET_FAILURE,
    BUCKET_REJECT,
    BUCKET_REVIEW,
    BUCKET_SAVE,
    SHORT_LIST_DECISIONS,
    TRANSFERABILITY_CONTRACT_REASON,
    JudgmentOutcome,
    classify_verdict,
    exa_lane_context,
    judge_candidate,
)
from exa.schemas import ExaCandidate, ExaEvidence, ExaSearchQuery
from shared.brief_loader import load_brief
from shared.failures import ApiBudgetExhaustedError, parse_failure_decision
from shared.llm_usage import _ATTEMPT_CONTEXT_KEYS
from shared.schemas import OpusDecision


LINKEDIN_URL = "https://www.linkedin.com/in/theo-r-marsh"
FIRM_PAGE = "https://www.kilnventures.com/team"
CAPABILITY_AREA = "Operating leverage"


# Exa's LinkedIn profile rendering — the live shape, synthetic name. This
# is what the people_search lane retrieves, and what S7 taught the record
# to read; the judge's profile half is built from it.
PROFILE_PAGE = """# Theo Marsh

Investor @ Kiln Ventures

San Francisco Bay Area (US)

500 connections • 1,795 followers

## About

Operator turned investor.

Total Experience: 14 years and 8 months

## Experience

### Principal - [Kiln Ventures](https://www.linkedin.com/company/kiln-ventures) (Current)

Feb 2022 - Present (4 years and 6 months)

Department: Education • Level: Senior

Kiln Ventures is a Financial Services company with 30-40 employees.

### [Canvasly](https://www.linkedin.com/company/canvasly)

#### Self-Serve Strategy

Dec 2020 - Feb 2022 (1 year and 2 months)

#### Strategic Finance & Business Operations

Jan 2018 - Dec 2020 (2 years and 11 months)

## Education

### M.B.A at Stanford Graduate School of Business

1982 - 1984 (2 years)
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def brief(tmp_path):
    """The smallest brief that loads as V2."""

    payload = {
        "role_title": "Chief of Staff to the CEO",
        "capability_areas": [
            {
                "name": CAPABILITY_AREA,
                "description": "Has run the founder's operating cadence.",
            }
        ],
        "depth_distinction": {
            "builder_definition": "Ran the cadence, didn't just attend it.",
            "user_definition": "Sat in the meeting and took notes.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(payload))
    return load_brief(path)


def _candidate(**overrides) -> ExaCandidate:
    payload = dict(
        identity_key="linkedin.com/in/theo-r-marsh",
        name="Theo Marsh",
        current_title="Principal",
        current_firm="Kiln Ventures",
        location="SF Bay Area",
        linkedin_url=LINKEDIN_URL,
        evidence=[
            ExaEvidence(
                url=LINKEDIN_URL,
                title="LinkedIn",
                published_date="2026-08-01",
                highlights=["Principal at Kiln Ventures."],
            ),
            ExaEvidence(
                url=FIRM_PAGE,
                title="Team",
                highlights=["Theo Marsh spent four years operating at Canvasly."],
            ),
        ],
        source_query_id=7,
        source_lane="people_search",
        profile_render=parse_exa_linkedin_markdown(PROFILE_PAGE).to_dict(),
    )
    payload.update(overrides)
    return ExaCandidate(**payload)


def _query(**overrides) -> ExaSearchQuery:
    payload = dict(
        id=7,
        name="people_search · Kiln Ventures",
        lane="people_search",
        firm="Kiln Ventures",
        query="Principal at Kiln Ventures",
        include_domains=["kilnventures.com"],
        exclude_terms=["Kiln Pottery"],
    )
    payload.update(overrides)
    return ExaSearchQuery(**payload)


def _decision(decision: str, **overrides) -> OpusDecision:
    payload = dict(
        stage="full",
        decision=decision,
        path="none",
        confidence=None,
        rationale="because",
        candidate_name="WRONG NAME",
        profile_url="https://example.invalid/wrong",
    )
    payload.update(overrides)
    return OpusDecision(**payload)


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------


def test_classify_verdict_short_list_is_save_and_inferential_save() -> None:
    """The bar is enforced here, not trusted to the prompt. Sam's ruling
    2026-09-03: INFERENTIAL_SAVE is a short-list verdict on web evidence."""

    assert SHORT_LIST_DECISIONS == frozenset({"SAVE", "INFERENTIAL_SAVE"})
    assert classify_verdict(_decision("SAVE")) == BUCKET_SAVE
    assert classify_verdict(_decision("INFERENTIAL_SAVE")) == BUCKET_SAVE

    # The remaining hedged saves still read as "worth a look".
    for hedged in ("TRANSFERABLE_SAVE", "SIGNAL_SAVE"):
        assert classify_verdict(_decision(hedged)) == BUCKET_REVIEW

    assert classify_verdict(_decision("REVIEW_INFERRED")) == BUCKET_REVIEW
    assert classify_verdict(_decision("REVIEW_FLAGGED")) == BUCKET_REVIEW

    assert classify_verdict(_decision("REJECT")) == BUCKET_REJECT
    # An unrecognized verdict is a rejection, never a silent save.
    assert classify_verdict(_decision("SOMETHING_NEW")) == BUCKET_REJECT

    # A judgment that did not happen is not a verdict about the person.
    assert classify_verdict(_decision("PARSE_FAILURE")) == BUCKET_FAILURE
    assert classify_verdict(_decision("JUDGMENT_FAILURE")) == BUCKET_FAILURE


# ---------------------------------------------------------------------------
# Lane context
# ---------------------------------------------------------------------------


def test_lane_context_only_uses_surviving_keys() -> None:
    """A key outside the allowlist is dropped silently at log time."""

    context = exa_lane_context(_candidate(), _query())

    assert set(context) <= set(_ATTEMPT_CONTEXT_KEYS)
    assert context["source"] == "exa"
    assert context["module"] == "exa"
    assert context["lane_id"] == "people_search"
    assert context["string_id"] == "7"
    # The Exa-flavored names would not survive the write.
    assert "lane" not in context and "query_id" not in context

    # No query in hand: the record still attributes its own lane and id.
    without_query = exa_lane_context(_candidate())
    assert set(without_query) <= set(_ATTEMPT_CONTEXT_KEYS)
    assert without_query["lane_id"] == "people_search"
    assert without_query["string_id"] == "7"


# ---------------------------------------------------------------------------
# The loop body
# ---------------------------------------------------------------------------


def test_judge_candidate_buckets_the_full_verdict(monkeypatch, brief) -> None:
    calls: list[str] = []

    def fake_full(summary, evidence, b=None, lane_context=None, **kwargs):
        calls.append("full")
        assert summary.name == "Theo Marsh"
        return _decision("SAVE")

    monkeypatch.setattr(judge_module, "full_judge_with_external_evidence", fake_full)

    outcome = judge_candidate(_candidate(), brief, query=_query())

    assert isinstance(outcome, JudgmentOutcome)
    assert calls == ["full"]  # one candidate, one call — serial, no batching
    assert outcome.bucket == BUCKET_SAVE
    assert outcome.full.decision == "SAVE"


def test_judge_candidate_buckets_hedged_save_as_review(monkeypatch, brief) -> None:
    monkeypatch.setattr(
        judge_module,
        "full_judge_with_external_evidence",
        lambda *a, **k: _decision("TRANSFERABLE_SAVE", confidence=0.4),
    )

    outcome = judge_candidate(_candidate(), brief, query=_query())

    assert outcome.bucket == BUCKET_REVIEW
    assert outcome.full.decision == "TRANSFERABLE_SAVE"


def test_judge_candidate_failure_bucket_on_judgment_failure(monkeypatch, brief) -> None:
    monkeypatch.setattr(
        judge_module,
        "full_judge_with_external_evidence",
        lambda *a, **k: _decision("JUDGMENT_FAILURE"),
    )

    outcome = judge_candidate(_candidate(), brief, query=_query())

    assert outcome.bucket == BUCKET_FAILURE
    assert outcome.full.decision == "JUDGMENT_FAILURE"


def _transferability_failure() -> OpusDecision:
    """The PARSE_FAILURE the tool contract raises, rationale and all.

    Built through the production helper, not hand-written, so the test
    reads the same rationale a live failure produces — the validator's
    own reason token, which is what the retry branches on.
    """

    return parse_failure_decision(
        stage="full",
        candidate_name="Theo Marsh",
        profile_url=LINKEDIN_URL,
        reason=TRANSFERABILITY_CONTRACT_REASON,
        detail="ADJACENT",
    )


def test_transferability_parse_failure_retries_once(monkeypatch, brief) -> None:
    """A dropped output field is not a verdict about the person.

    The model reasons to ADJACENT and then writes "N/A" where the
    contract needs TRANSFERABLE or NOT_TRANSFERABLE
    (``shared/judgment/tool_contracts.py:746-748``). On 2026-09-03 that
    took the ENTIRE in-band slate. It is a format slip, which is exactly
    the class a second sample fixes.
    """

    calls: list[dict] = []

    def flaky(summary, evidence, b=None, lane_context=None, **kwargs):
        calls.append({"lane_context": lane_context})
        if len(calls) == 1:
            return _transferability_failure()
        return _decision("SAVE")

    monkeypatch.setattr(judge_module, "full_judge_with_external_evidence", flaky)

    outcome = judge_candidate(_candidate(), brief, query=_query())

    assert len(calls) == 2
    assert outcome.full.decision == "SAVE"
    assert outcome.bucket == BUCKET_SAVE
    # The retry is still attributed to this candidate's lane and query.
    assert calls[1]["lane_context"]["lane_id"] == "people_search"
    assert calls[1]["lane_context"]["string_id"] == "7"
    assert set(calls[1]["lane_context"]) <= set(_ATTEMPT_CONTEXT_KEYS)


def test_transferability_retry_happens_once_and_only_for_that_fault(
    monkeypatch, brief
) -> None:
    """Once, never twice — and never for a fault that would repeat."""

    persistent: list[int] = []

    def always_failing(*args, **kwargs):
        persistent.append(1)
        return _transferability_failure()

    monkeypatch.setattr(
        judge_module, "full_judge_with_external_evidence", always_failing
    )
    outcome = judge_candidate(_candidate(), brief, query=_query())
    # Two calls total: the original and one retry. A loop would spend a
    # campaign's budget on one person.
    assert len(persistent) == 2
    assert outcome.bucket == BUCKET_FAILURE

    # Every OTHER failure returns on the first call. A real parse error
    # and a judgment failure both repeat; retrying them buys nothing.
    for other in (
        # A parse failure that is NOT a validator slip (no
        # JudgmentToolContractError in the rationale) repeats and is not
        # retried; a judgment failure likewise.
        parse_failure_decision(
            stage="full",
            candidate_name="Theo Marsh",
            profile_url=LINKEDIN_URL,
            reason="empty_response",
            detail="",
        ),
        _decision("JUDGMENT_FAILURE"),
        _decision("REJECT"),
        _decision("SAVE"),
    ):
        seen: list[int] = []

        def once(*args, _verdict=other, **kwargs):
            seen.append(1)
            return _verdict

        monkeypatch.setattr(judge_module, "full_judge_with_external_evidence", once)
        judge_candidate(_candidate(), brief, query=_query())
        assert len(seen) == 1, other.decision


def test_budget_exhaustion_propagates(monkeypatch, brief) -> None:
    """Absorbing this into a verdict is how a run spends hours on a dead key."""

    def exhausted(*args, **kwargs):
        raise ApiBudgetExhaustedError("credit balance is too low")

    monkeypatch.setattr(judge_module, "full_judge_with_external_evidence", exhausted)

    with pytest.raises(ApiBudgetExhaustedError):
        judge_candidate(_candidate(), brief, query=_query())


# ---------------------------------------------------------------------------
# Stamping and evidence hand-off
# ---------------------------------------------------------------------------


def test_decisions_carry_candidate_name_and_profile_url(monkeypatch, brief) -> None:
    monkeypatch.setattr(
        judge_module,
        "full_judge_with_external_evidence",
        lambda *a, **k: _decision("SAVE"),
    )

    outcome = judge_candidate(_candidate(), brief, query=_query())
    assert outcome.full.candidate_name == "Theo Marsh"
    assert outcome.full.profile_url == LINKEDIN_URL

    # No LinkedIn URL: the first retrieved page is the record's address,
    # so a receipt is never left with an empty profile_url.
    no_linkedin = _candidate(
        linkedin_url="",
        evidence=[ExaEvidence(url=FIRM_PAGE, title="Team", highlights=["a"])],
    )
    outcome = judge_candidate(no_linkedin, brief, query=_query())
    assert outcome.full.profile_url == FIRM_PAGE

    # A manufactured failure decision is stamped too — the receipt must
    # name the person even when the judgment never happened.
    monkeypatch.setattr(
        judge_module,
        "full_judge_with_external_evidence",
        lambda *a, **k: _decision("PARSE_FAILURE"),
    )
    outcome = judge_candidate(_candidate(), brief, query=_query())
    assert outcome.full.candidate_name == "Theo Marsh"
    assert outcome.full.profile_url == LINKEDIN_URL


def test_full_judge_receives_exa_evidence_object(monkeypatch, brief) -> None:
    captured: dict = {}

    def fake_full(summary, evidence, b=None, lane_context=None, **kwargs):
        captured["evidence"] = evidence
        captured["lane_context"] = lane_context
        captured["brief"] = b
        return _decision("SAVE")

    monkeypatch.setattr(judge_module, "full_judge_with_external_evidence", fake_full)

    judge_candidate(_candidate(), brief, query=_query())

    evidence = captured["evidence"]
    assert evidence.trigger_reason == EXA_TRIGGER_REASON
    assert evidence.raw_provider_model == "exa"
    assert [
        ref.url for block in evidence.external_fact_blocks for ref in block.evidence_refs
    ] == [LINKEDIN_URL, FIRM_PAGE]
    # The lane's declared domain reached the source-quality grader.
    assert [block.source_quality for block in evidence.external_fact_blocks] == [
        "high",
        "high",
    ]
    assert captured["lane_context"]["source"] == "exa"
    assert captured["brief"] is brief


# ---------------------------------------------------------------------------
# End to end, through the real judge
# ---------------------------------------------------------------------------


def _legacy_full_response(decision: str = "SAVE") -> str:
    """A minimal response the V2 legacy text contract accepts.

    Field set derived from ``parse_full_evaluation_response``'s
    ``require_semantic_v2`` branch (shared/judgment/templates.py:1670-1696):
    every semantic wire field and every evidence field must be non-empty.
    """

    return (
        f"STEP_1_MATCH: DIRECT\n"
        f"STEP_1_AREA: {CAPABILITY_AREA}\n"
        "STEP_1_EVIDENCE: Ran the operating cadence at Canvasly.\n"
        "STEP_1_RECENCY: CURRENT\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: Owned strategic finance and self-serve strategy.\n"
        "STEP_3_TRANSFERABILITY: N/A\n"
        "STEP_3_EVIDENCE: Direct match.\n"
        "STEP_4_LEVEL: ALIGNED\n"
        "STEP_5_COHERENCE: COHERENT\n"
        "STEP_6_CALIBER: STRONG\n"
        "CASE_FOR: Operator chapter through hypergrowth.\n"
        "CASE_AGAINST: Investing seat is recent.\n"
        "REJECT_REASON: NONE\n"
        "OUTREACH_TIER: STANDARD\n"
        f"DECISION: {decision}\n"
        "CONFIDENCE: 0.78\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Four operating years at Canvasly, now investing.\n"
    )


def test_full_judge_rendered_user_message_carries_evidence(monkeypatch, brief) -> None:
    """The one honest end-to-end: what the model would actually be sent.

    Only the provider call is stubbed. The system prompt, the profile
    rendering, the external-evidence block, and the parse are the real
    production path. The contract mode is pinned to ``legacy`` so the
    test asserts one rendering rather than whichever the operator's
    environment happens to select.
    """

    monkeypatch.setattr(config, "LINKEDIN_V2_FULL_CONTRACT", "legacy")
    captured: dict = {}

    def fake_llm(system, user, **kwargs):
        captured["system"] = system
        captured["user"] = user
        return _legacy_full_response()

    monkeypatch.setattr(judger, "opus_llm_cached", fake_llm)

    outcome = judge_candidate(_candidate(), brief, query=_query())

    user_message = captured["user"]
    assert "Theo Marsh" in user_message
    # This record HAS a profile section, so it gets the framing that
    # says so. The old "the profile section's emptiness is a source
    # artifact" line would tell the judge to discount a real profile.
    assert RENDER_SOURCE_FRAMING in user_message
    assert SOURCE_FRAMING not in user_message
    for url in _candidate().evidence_urls():
        assert url in user_message
    # Publication dates ride the topic line.
    assert "published 2026-08-01" in user_message
    assert "published unknown" in user_message
    # The facts themselves, not just the citations.
    assert "Theo Marsh spent four years operating at Canvasly." in user_message

    # The profile half of the prompt is a real profile, not the stub that
    # produced 100% PARSE_FAILURE on 2026-09-03. A dated experience line
    # and an About section, and nowhere the sparse-profile placeholder.
    assert "- Principal at Kiln Ventures (Feb 2022-Present)" in user_message
    assert "- Strategic Finance & Business Operations at Canvasly (Jan 2018-Dec 2020)" in (
        user_message
    )
    assert "About:\nOperator turned investor." in user_message
    # Platform arithmetic gets its own labeled line, not a place inside
    # the person's own words.
    assert "Platform-derived tenure (Exa): 14 years and 8 months" in user_message
    assert "- M.B.A in , Stanford Graduate School of Business (1982-1984)" in user_message
    # No section placeholder anywhere: About, Experience and Education
    # all have content. (A position with no bullets of its own still ends
    # ": no details" — inventing them from the firm blurb underneath is
    # exactly the fabrication this module refuses.
    # The dated history, the About text and the education are what stop
    # the sparse-profile read.)
    assert "None listed" not in user_message

    # The page the profile was built from says so, so the judge does not
    # read one page rendered twice as two agreeing sources.
    assert PROFILE_SOURCE_SUFFIX in user_message

    # And the output contract the tool validator enforces is stated.
    assert TRANSFERABILITY_CONTRACT_LINE not in user_message  # dropped 2026-09-03; the retry handles slips

    assert outcome.full.decision == "SAVE"
    assert outcome.bucket == BUCKET_SAVE
    assert outcome.full.candidate_name == "Theo Marsh"
    assert outcome.full.profile_url == LINKEDIN_URL


def test_any_validator_slip_retries_once(monkeypatch, brief) -> None:
    """Broadened 2026-09-03: the resumed CoS run failed in-band people on three
    different validator reasons. Any ``JudgmentToolContractError`` gets one
    resample; a second slip still returns the failure."""

    for reason in (
        "review_fields_on_non_review",
        "recommended_next_step_on_inferred_review",
        "invalid_direct_transferability",
    ):
        calls: list[int] = []

        def slip_then_save(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return parse_failure_decision(
                    stage="full",
                    candidate_name="Theo Marsh",
                    profile_url=LINKEDIN_URL,
                    reason=f"JudgmentToolContractError: {reason}",
                    detail="",
                )
            return _decision("SAVE")

        monkeypatch.setattr(
            judge_module, "full_judge_with_external_evidence", slip_then_save
        )
        outcome = judge_candidate(_candidate(), brief, query=_query())
        assert len(calls) == 2, reason
        assert outcome.bucket == "save", reason
