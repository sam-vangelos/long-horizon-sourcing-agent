"""Phase 1 seam-contract tests — judgment cluster.

These pins/xfails fasten the producer->CONSUMER edge of the judgment seams so
a refactor cannot silently sever the wiring (the unit suite already proves the
halves in isolation). Each test drives a REAL producer with the only true
external boundary mocked — the Anthropic LLM API (``opus_llm_cached`` /
``anthropic.Anthropic``) — and asserts the real consumer observes the value
crossing the seam. Neither the producer nor the consumer nor the seam itself
is mocked.

Seams covered (see the cluster inventory for anchors):

- 1.5 (pin): ``full_judge -> OpusDecision.decision`` drives the orchestrator's
  three-way routing in ``_full_evaluate`` (SAVE -> ``handle_save_decision`` +
  ``candidate_saved``; malformed strict-v2 review -> PARSE_FAILURE;
  JUDGMENT_FAILURE -> ``_finish_runtime_failure_decision``). A separate injected
  malformed review pins the orchestrator's defensive demotion guard.
- 1.2 (pin): ``assemble_full_evaluation_system`` fills every template
  placeholder from ``Brief.*`` blocks — capability-area names, domain verbs,
  non-fit labels — with no unfilled ``{`` placeholder. (The token/transferability
  halves are pinned in ``tests/test_judgment_template_calibration.py``; this
  pins the no-unfilled-placeholder + capability-name-coverage gap.)
- 1.0 (pin): the orchestrator threads ``lane_context`` through
  ``full_judge -> usage_context -> record_llm_usage`` into the cost-log JSONL,
  and ``lane_cost_from_usage_log`` rolls each row up per ``lane_id``.
- 1.1 (xfail): ``stage`` / ``variant_id`` reach the JSONL row but no JSONL cost
  consumer keys on them — only ``lane_id`` is read. Flips when Phase 2 adds a
  per-stage/per-variant cost reader.
- 1.4 (xfail): ``ProfileProbe.record_shadow_outcome`` appends to
  ``shadow_records`` but nothing at runtime drains/persists/reads it into an
  ``AuditSampler`` or artifact. Flips when Phase 2 wires the recorder into a
  consumer the run exposes.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.browser import LinkedInBrowser
from linkedin.judgment_templates import assemble_full_evaluation_system
from shared.brief_loader import load_brief
from shared.judger import full_judge
from shared.llm_usage import llm_usage_session
from shared.runtime_state.linkedin_progress_sync import lane_cost_from_usage_log
from shared.schemas import CandidateProfileSummary, OpusDecision, SearchString
from shared.storage import read_jsonl
from tests.test_linkedin_pipeline import _make_pipeline, _make_snippet

_REPO_ROOT = Path(__file__).resolve().parents[1]
_V2_BRIEF_PATH = _REPO_ROOT / "config" / "FDL-Colombia" / "brief-fdl-colombia-v4.json"


# ---------------------------------------------------------------------------
# Fixtures / model-response builders
#
# The judger's V2 full-eval path is ``opus_llm_cached(system, profile_text) ->
# parse_full_evaluation_response(raw) -> OpusDecision.decision`` (shared/judger.py
# :674,698,725,739). Patching ``shared.judger.opus_llm_cached`` (the imported
# LLM-boundary symbol, judger.py:28) lets the REAL full_judge run end-to-end:
# the canned model text below is parsed by the real parser into a real decision
# string that then crosses the seam into the orchestrator's routing.
# ---------------------------------------------------------------------------


def _save_response() -> str:
    return (
        "STEP_1_MATCH: DIRECT\n"
        "STEP_1_AREA: 1. RL Environments, Verifiers & Post-Training Data Systems\n"
        "STEP_1_EVIDENCE: Built RL verifier systems for post-training data.\n"
        "STEP_1_RECENCY: CURRENT\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: design / build / ship / own.\n"
        "STEP_3_TRANSFERABILITY: N/A\n"
        "STEP_3_EVIDENCE: N/A\n"
        "STEP_4_LEVEL: ALIGNED\n"
        "STEP_5_COHERENCE: COHERENT\n"
        "STEP_6_CALIBER: STRONG\n"
        "CASE_FOR: Direct fit on the core capability with senior scope.\n"
        "CASE_AGAINST: None material.\n"
        "REJECT_REASON: NONE\n"
        "OUTREACH_TIER: STANDARD\n"
        "DECISION: SAVE\n"
        "CONFIDENCE: 0.82\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Strong direct fit on the core capability.\n"
    )


def _invalid_review_inferred_one_signal_response() -> str:
    """REVIEW_INFERRED carrying a single STRUCTURAL_EVIDENCE item.

    Strict legacy v2 shares the tool validator, so this must parse-fail before
    it can reach the orchestrator's defensive review-demotion guard.
    """
    return (
        "STEP_1_MATCH: ADJACENT\n"
        "STEP_1_AREA: 1. RL Environments, Verifiers & Post-Training Data Systems\n"
        "STEP_1_EVIDENCE: Senior technologist with relevant org scope.\n"
        "STEP_1_RECENCY: RECENT\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: Inferred from title path.\n"
        "STEP_3_TRANSFERABILITY: TRANSFERABLE\n"
        "STEP_3_EVIDENCE: Capability transfers from adjacent work.\n"
        "STEP_4_LEVEL: UNCLEAR\n"
        "STEP_5_COHERENCE: UNCLEAR\n"
        "STEP_6_CALIBER: UNKNOWN\n"
        "CASE_FOR: Structural signals justify a human look.\n"
        "CASE_AGAINST: Explicit evidence sparse on profile.\n"
        "REJECT_REASON: NONE\n"
        "OUTREACH_TIER: NONE\n"
        "DECISION: REVIEW_INFERRED\n"
        "CONFIDENCE: 0.48\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "REVIEW_REASON: inferred_high_priority\n"
        "STRUCTURAL_EVIDENCE: only one structural signal\n"
        "SUMMARY: Preserved for a recruiter spot check.\n"
    )


def _reject_response() -> str:
    return (
        "STEP_1_MATCH: NONE\n"
        "STEP_1_AREA: N/A\n"
        "STEP_1_EVIDENCE: Outside the capability area.\n"
        "STEP_1_RECENCY: RECENT\n"
        "STEP_2_DEPTH: USER\n"
        "STEP_2_EVIDENCE: Uses tools; doesn't build them.\n"
        "STEP_3_TRANSFERABILITY: NOT_TRANSFERABLE\n"
        "STEP_3_EVIDENCE: Gap is too wide.\n"
        "STEP_4_LEVEL: BELOW\n"
        "STEP_5_COHERENCE: INCOHERENT\n"
        "STEP_6_CALIBER: WEAK\n"
        "CASE_FOR: Adjacent vocabulary.\n"
        "CASE_AGAINST: Wrong depth and wrong domain.\n"
        "REJECT_REASON: CAPABILITY_INSUFFICIENT\n"
        "OUTREACH_TIER: NONE\n"
        "DECISION: REJECT\n"
        "CONFIDENCE: 0.30\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Adjacent at best; not a save.\n"
    )


def _build_routing_pipeline(output_dir: str):
    """A Pipeline wired for ``_full_evaluate`` routing tests.

    Only the true external boundaries are stubbed (browser/CDP page,
    profile-acquisition service, runtime-state finishers, the LinkedIn
    save-click side-effect). ``brief_obj`` is a REAL V2 brief so the real
    ``full_judge`` takes the V2 structural path; ``opus_llm_cached`` is patched
    per-test. The decision-routing under test is left entirely real.
    """
    pipeline = _make_pipeline(output_dir)
    pipeline.brief_obj = load_brief(str(_V2_BRIEF_PATH))
    pipeline._ensure_services = MagicMock()
    pipeline._acquisition_service = MagicMock()
    summary = MagicMock()
    summary.to_dict.return_value = {
        "current_title": "ML Engineer",
        "headline": "RL Environments",
        "summary": "RL verifiers and post-training data systems",
    }
    pipeline._acquisition_service.extract_profile_summary = AsyncMock(
        return_value=SimpleNamespace(profile_summary=summary)
    )
    pipeline._start_runtime_stage_attempt = MagicMock(return_value=1)
    pipeline._finish_runtime_stage_success = MagicMock()
    pipeline._finish_runtime_failure_decision = MagicMock()
    pipeline._mark_terminal = MagicMock()
    pipeline._bias_monitor = None
    from shared.execution import SideEffectOutcome

    pipeline._side_effects_service = MagicMock()
    # P1.2: _full_evaluate consumes the returned SideEffectOutcome, so the
    # stub returns the real success shape.
    pipeline._side_effects_service.handle_save_decision = AsyncMock(
        return_value=SideEffectOutcome(
            effect_type="linkedin_save",
            status="succeeded",
            payload={},
        )
    )
    pipeline.browser = MagicMock()
    pipeline.browser.page.url = "https://www.linkedin.com/talent/search"
    pipeline.browser.current_profile_identity_fragment.side_effect = (
        lambda: LinkedInBrowser._profile_url_fragment(
            pipeline.browser.page.url
        )
    )

    async def open_profile_by_url(profile_url):
        identity = LinkedInBrowser._profile_url_fragment(profile_url)
        pipeline.browser.page.url = (
            "https://www.linkedin.com/talent/recruiterSearch/profile/"
            f"{identity}"
        )

    pipeline.browser.open_profile_by_url = AsyncMock(
        side_effect=open_profile_by_url
    )
    pipeline.browser.go_back_to_results = AsyncMock()
    pipeline.browser.get_profile_status_summary = AsyncMock(return_value={})
    pipeline._derive_novelty_value = MagicMock(return_value=("high", "rationale"))
    return pipeline


def _run_full_evaluate(pipeline, snippet, search_string, *, llm_return=None, llm_side_effect=None):
    """Drive the real ``_full_evaluate`` with the LLM boundary mocked."""
    judge_patch = patch(
        "shared.judger.opus_llm_cached",
        **(
            {"side_effect": llm_side_effect}
            if llm_side_effect is not None
            else {"return_value": llm_return}
        ),
    )
    with judge_patch, patch(
        "linkedin.orchestrator.human_delay_correlated",
        side_effect=lambda base, channel: base,
    ), patch("linkedin.orchestrator.asyncio.sleep", new=AsyncMock()):
        return asyncio.run(pipeline._full_evaluate(snippet, None, search_string))


def _candidate_saved_events(pipeline) -> list[dict]:
    import json

    log_path = Path(pipeline.log_path)
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("event") == "candidate_saved":
            out.append(rec)
    return out


def _review_demotion_reasons(pipeline) -> list[str]:
    import json

    log_path = Path(pipeline.log_path)
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if (
            isinstance(rec, dict)
            and rec.get("event") == "candidate_review_recorded"
            and rec.get("demoted")
        ):
            out.append(rec.get("reason"))
    return out


# ---------------------------------------------------------------------------
# Seam 1.5 (pin) — full_judge decision string drives the orchestrator routing
#
# PRODUCER: full_judge -> OpusDecision.decision (shared/judger.py:736-756)
# CONSUMER: _full_evaluate three-way routing (orchestrator.py:4485-4516 FAILURE,
#           :4522-4543 REVIEW guard demotion, :4750-4760 SAVE).
#
# The decision is produced by the REAL full_judge (V2 path), not injected —
# only opus_llm_cached (the LLM API) is mocked. Each assertion fails if the
# producer's decision string stops driving the corresponding consumer branch.
# ---------------------------------------------------------------------------


def test_seam_1_5_save_decision_routes_to_handle_save_and_emits_candidate_saved():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _build_routing_pipeline(td)
        snippet = _make_snippet(profile_url="/talent/profile/seam-save")
        search_string = SearchString(id=1, name="t", boolean="x", lane_id="lane-A")

        decision = _run_full_evaluate(
            pipeline, snippet, search_string, llm_return=_save_response()
        )

        # The real parser produced a real SAVE decision that crossed the seam.
        assert decision is not None
        assert decision.decision == "SAVE"
        # CONSUMER: SAVE branch invoked the side-effect exactly once...
        assert pipeline._side_effects_service.handle_save_decision.await_count == 1
        _, kwargs = pipeline._side_effects_service.handle_save_decision.await_args
        assert kwargs["snippet"] is snippet
        pipeline.browser.open_profile_by_url.assert_awaited_once_with(
            snippet.profile_url
        )
        # ...incremented the save-attempt counter...
        assert pipeline.stats["save_attempts"] == 1
        # ...stamped the actuator outcome onto the decision (P1.2)...
        assert decision.save_outcome["persisted"] is True
        # ...emitted NO candidate_saved at the orchestrator layer: the one
        # honest emission (with the linkedin_save flag) lives inside
        # handle_save_decision, which is stubbed here. The flag-less
        # duplicate this test used to pin was deleted in P1.2 — it shadowed
        # the honest event for every consumer.
        saved_events = _candidate_saved_events(pipeline)
        assert saved_events == []
        # ...and did NOT take the failure path.
        assert pipeline._finish_runtime_failure_decision.called is False


def test_seam_1_5_strict_legacy_rejects_review_inferred_thin_evidence():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _build_routing_pipeline(td)
        snippet = _make_snippet(profile_url="/talent/profile/seam-review")
        search_string = SearchString(id=1, name="t", boolean="x", lane_id="lane-A")

        decision = _run_full_evaluate(
            pipeline,
            snippet,
            search_string,
            llm_return=_invalid_review_inferred_one_signal_response(),
        )

        assert decision is not None
        assert decision.decision == "PARSE_FAILURE"
        assert pipeline._finish_runtime_failure_decision.called is True
        assert pipeline.stats["reviewed_demoted"] == 0
        assert pipeline._side_effects_service.handle_save_decision.await_count == 0


def test_seam_1_5_defensive_review_demotion_rejects_injected_thin_evidence():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _build_routing_pipeline(td)
        snippet = _make_snippet(profile_url="/talent/profile/seam-review-defense")
        search_string = SearchString(id=1, name="t", boolean="x", lane_id="lane-A")
        malformed_review = OpusDecision(
            stage="full",
            decision="REVIEW_INFERRED",
            path="ADJACENT:RL Environments",
            confidence=0.48,
            rationale="Only one structural signal.",
            candidate_name=snippet.name,
            profile_url=snippet.profile_url,
            review_reason_code="inferred_high_priority",
            review_structural_evidence=["only one structural signal"],
        )

        with patch(
            "linkedin.orchestrator.full_judge",
            return_value=malformed_review,
        ), patch(
            "linkedin.orchestrator.human_delay_correlated",
            side_effect=lambda base, channel: base,
        ), patch("linkedin.orchestrator.asyncio.sleep", new=AsyncMock()):
            decision = asyncio.run(
                pipeline._full_evaluate(snippet, None, search_string)
            )

        assert decision is malformed_review
        assert decision.decision == "REJECT"
        assert decision.reject_reason == "CAPABILITY_INSUFFICIENT"
        assert decision.outreach_tier == ""
        assert "insufficient_structural_evidence" in _review_demotion_reasons(pipeline)
        assert pipeline.stats["reviewed_demoted"] == 1
        assert pipeline._side_effects_service.handle_save_decision.await_count == 0


def test_seam_1_5_judgment_failure_routes_to_finish_failure_not_save():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _build_routing_pipeline(td)
        snippet = _make_snippet(profile_url="/talent/profile/seam-failure")
        search_string = SearchString(id=1, name="t", boolean="x", lane_id="lane-A")

        # An LLM-boundary exception makes the REAL full_judge return a
        # JUDGMENT_FAILURE decision (shared/judger.py:704-724 except block).
        decision = _run_full_evaluate(
            pipeline,
            snippet,
            search_string,
            llm_side_effect=RuntimeError("boom"),
        )

        assert decision is not None
        assert decision.decision == "JUDGMENT_FAILURE"
        # CONSUMER: failure decisions take the non-terminal finish path...
        assert pipeline._finish_runtime_failure_decision.called is True
        # ...never the terminal stage-success write...
        assert pipeline._finish_runtime_stage_success.called is False
        # ...and never the save side-effect.
        assert pipeline._side_effects_service.handle_save_decision.await_count == 0


# ---------------------------------------------------------------------------
# Seam 1.2 (pin) — assemble_full_evaluation_system fills every placeholder from
# the brief blocks, with no unfilled "{" placeholder.
#
# PRODUCER: assemble_full_evaluation_system (linkedin/judgment_templates.py
#           :521-558) reads brief.capability_area_block / non_fit_block /
#           domain verbs / etc. off Brief.* and renders FULL_EVALUATION_TEMPLATE.
# CONSUMER: full_judge uses the assembled string as the LLM system message
#           (shared/judger.py:674,698).
#
# DEDUP: tests/test_judgment_template_calibration.py:177-211 pins the
# domain_verbs / depth-object / transferability tokens for a populated brief.
# This pins the COMPLEMENTARY uncovered facts: every capability_area_names()
# entry is present, the non_fit labels are present, and the rendered prompt has
# no leftover "{" (i.e. the .format() supplied every placeholder the template
# declares).
# ---------------------------------------------------------------------------


def test_seam_1_2_full_eval_system_renders_every_brief_block_with_no_unfilled_placeholder():
    brief = load_brief(str(_V2_BRIEF_PATH))
    assert brief.has_v2_schema is True
    new_brief = brief._new_brief

    prompt = assemble_full_evaluation_system(new_brief)

    capability_names = new_brief.capability_area_names()
    assert capability_names, "fixture should have capability areas"
    for name in capability_names:
        assert name in prompt, (
            f"capability area name not rendered into full-eval system prompt: {name!r}"
        )

    assert new_brief.domain_verbs, "fixture should populate domain_verbs"
    for verb in new_brief.domain_verbs:
        assert verb in prompt, f"domain verb not rendered into system prompt: {verb!r}"

    assert new_brief.non_fit_patterns, "fixture should have non-fit patterns"
    for nf in new_brief.non_fit_patterns:
        assert nf.label in prompt, (
            f"non-fit label not rendered into system prompt: {nf.label!r}"
        )

    # No leftover placeholder: every "{...}" the template declares was supplied
    # by the assemble_* call. A new template placeholder the assembler does not
    # fill raises KeyError at .format(); a literal stray brace fails this.
    assert "{" not in prompt, "rendered full-eval system prompt has an unfilled placeholder"


# ---------------------------------------------------------------------------
# Seam 1.0 (pin) — lane_context threaded through the judge lands in the
# cost-log JSONL and rolls up per lane_id.
#
# PRODUCER: full_judge(lane_context=...) -> usage_context (shared/judger.py:696)
#           -> opus_llm_cached(usage_context=...) -> record_llm_usage spreads it
#           into the JSONL row (shared/llm_usage.py:199-227).
# CONSUMER: lane_cost_from_usage_log reads record['lane_id'] + estimated_cost_usd
#           (shared/runtime_state/linkedin_progress_sync.py:54-59).
#
# Only the Anthropic client (the network boundary) is mocked; the whole
# threading chain is real. DEDUP: test_run_cost_attribution.py:107-108 proves
# pipeline/run_id context round-trips via cheap_llm, but does NOT pin the
# per-lane rollup from JUDGE lane_context threading. This pins that edge.
# ---------------------------------------------------------------------------


def _fake_anthropic_factory(responses):
    """Build a patch target for ``anthropic.Anthropic`` returning canned messages."""

    seq = iter(responses)

    class _Messages:
        def create(self, **kwargs):
            text, in_tok, out_tok = next(seq)
            return SimpleNamespace(
                content=[SimpleNamespace(text=text)],
                stop_reason="end_turn",
                usage=SimpleNamespace(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            )

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    return _Client


def test_seam_1_0_judge_lane_context_rolls_up_per_lane_in_cost_log():
    import anthropic

    brief = load_brief(str(_V2_BRIEF_PATH))
    summ_a = CandidateProfileSummary(name="Cand A", profile_url="/p/a", headline="RL")
    summ_b = CandidateProfileSummary(name="Cand B", profile_url="/p/b", headline="RL")

    # Two lane-A judge calls and one lane-B call. Distinct token counts keep
    # the rollup meaningful (lane-A cost = sum of its two rows).
    responses = [
        (_save_response(), 1000, 100),
        (_save_response(), 2000, 200),
        (_save_response(), 500, 50),
    ]

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.object(
            anthropic, "Anthropic", _fake_anthropic_factory(responses)
        ), patch(
            "shared.judger.config.FULL_EVAL_MODEL_NAME",
            "claude-opus-4-8",
        ):
            with llm_usage_session(log_path, pipeline="seam-judgment", run_id="run-seam"):
                full_judge(
                    summ_a,
                    brief,
                    lane_context={"lane_id": "lane-A", "stage": "full_eval", "variant_id": "v2"},
                )
                full_judge(
                    summ_a,
                    brief,
                    lane_context={"lane_id": "lane-A", "stage": "full_eval", "variant_id": "v2"},
                )
                full_judge(
                    summ_b,
                    brief,
                    lane_context={"lane_id": "lane-B", "stage": "full_eval", "variant_id": "v1"},
                )

        rows = read_jsonl(log_path)
        assert len(rows) == 3
        # The lane_id threaded by the judge is present on every row.
        assert {r.get("lane_id") for r in rows} == {"lane-A", "lane-B"}

        by_lane = lane_cost_from_usage_log(log_path)
        assert set(by_lane) == {"lane-A", "lane-B"}
        assert by_lane["lane-A"] > 0.0

        # The per-lane rollup equals the sum of the rows whose lane_id matches —
        # i.e. the consumer attributed cost by the lane the producer threaded.
        lane_a_rows = [r for r in rows if r.get("lane_id") == "lane-A"]
        expected_lane_a = round(
            sum(float(r["estimated_cost_usd"] or 0) for r in lane_a_rows), 6
        )
        assert round(by_lane["lane-A"], 6) == expected_lane_a


# ---------------------------------------------------------------------------
# Seam 1.1 (xfail, strict) — stage / variant_id reach the JSONL row but no
# JSONL cost consumer keys on them.
#
# PRODUCER: _lane_context_for_stage emits stage + variant_id alongside lane_id
#           (orchestrator.py:777-781), threaded into the row via record_llm_usage
#           (shared/llm_usage.py:224).
# CONSUMER: NONE for JSONL cost. lane_cost_from_usage_log keys solely on lane_id
#           (linkedin_progress_sync.py:54-59); cost_rollup reads pipeline_end,
#           never the per-call rows. stage/variant_id reach only Langfuse
#           metadata (a no-op when disabled).
#
# Written AS IF a per-stage cost reader existed: two rows sharing one lane but
# carrying distinct stages should roll up to a stage-keyed dict. Today the only
# JSONL cost reader (lane_cost_from_usage_log) returns a LANE-keyed dict, so the
# stage-keyed assertion fails AT the seam (not at an import). Flips to pass when
# Phase 2 adds a per-stage/per-variant cost reader.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Phase 2 per-stage/variant cost attribution -- inert until a JSONL "
    "cost consumer keys on record['stage']/['variant_id'] (today only lane_id "
    "is read by lane_cost_from_usage_log)",
    strict=True,
)
def test_seam_1_1_stage_variant_cost_attribution_from_cost_log():
    from shared.llm_usage import record_llm_usage

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        # Two calls, one lane, two distinct stages — the producer always emits
        # stage + variant_id into the row.
        with llm_usage_session(log_path, pipeline="seam-judgment"):
            record_llm_usage(
                provider="anthropic",
                model="claude-opus-4-6",
                usage={"input_tokens": 1000, "output_tokens": 100},
                usage_context={"lane_id": "L", "stage": "facial", "variant_id": "v2"},
            )
            record_llm_usage(
                provider="anthropic",
                model="claude-opus-4-6",
                usage={"input_tokens": 2000, "output_tokens": 200},
                usage_context={"lane_id": "L", "stage": "full_eval", "variant_id": "v1"},
            )

        rows = read_jsonl(log_path)
        # Producer side is alive: stage + variant_id are on every row.
        assert {r.get("stage") for r in rows} == {"facial", "full_eval"}
        assert {r.get("variant_id") for r in rows} == {"v1", "v2"}

        # The ONLY JSONL cost consumer. A per-stage rollup would key on stage;
        # today it keys on lane_id, so its keys are {"L"}, not {"facial",
        # "full_eval"}. This is the seam assertion that currently fails: the
        # observed value is {"L"}.
        cost_by_stage = lane_cost_from_usage_log(log_path)
        assert set(cost_by_stage) == {"facial", "full_eval"}


# ---------------------------------------------------------------------------
# Seam 1.4 (xfail, strict) — the probe's shadow outcome is recorded but never
# observed by any runtime consumer.
#
# PRODUCER: ProfileProbe.record_shadow_outcome builds a CascadeRecord and
#           appends to self.shadow_records (orchestrator.py:4468-4475;
#           evaluation_cascade.py:154-186).
# CONSUMER: NONE at runtime for activation purposes. P10 actuate #2 (see
#           test_seam_p10_cascade_shadow_record_persisted_to_runtime_events
#           below) made the CascadeRecord real by persisting it to the
#           runtime_state events channel at the checkpoint, so it is no
#           longer discarded/never-observed -- but no AuditSampler /
#           can_activate consumer reads it back at runtime yet; full_judge
#           still runs unconditionally regardless of probe_decision
#           (orchestrator.py:4445 then :4453-4454). Cascade activation stays
#           OFF by design (D8).
#
# Written AS IF the run exposed an AuditSampler populated from the probe's
# shadow outcomes. The producer side is genuinely exercised (a real
# _full_evaluate appends one CascadeRecord), so the test runs to the seam; only
# the "a consumer observed it" assertion fails because the pipeline exposes no
# such AuditSampler attribute. Flips when a runtime AuditSampler is wired from
# the persisted events (a further, out-of-scope Phase 2 step).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="No runtime AuditSampler consumer wired yet -- the shadow record is "
    "now persisted (runtime_state events, P10 actuate #2) but nothing drains "
    "it into an AuditSampler/can_activate at runtime",
    strict=True,
)
def test_seam_1_4_shadow_outcome_observable_to_audit_consumer():
    from linkedin.evaluation_cascade import AuditSampler

    with tempfile.TemporaryDirectory() as td:
        pipeline = _build_routing_pipeline(td)
        snippet = _make_snippet(profile_url="/talent/profile/seam-shadow")
        search_string = SearchString(id=1, name="t", boolean="x", lane_id="lane-A")

        decision = _run_full_evaluate(
            pipeline, snippet, search_string, llm_return=_reject_response()
        )

        # Producer side is alive: the probe ran and recorded one shadow outcome.
        assert decision is not None
        assert len(pipeline._profile_probe.shadow_records) == 1

        # The run must expose an AuditSampler the runtime populated from those
        # shadow outcomes so suppression / false-negative metrics are
        # observable. No such consumer exists today: the attribute is absent
        # (the observed value is None), so this seam assertion fails here.
        runtime_sampler = getattr(pipeline, "_cascade_audit_sampler", None)
        assert isinstance(runtime_sampler, AuditSampler)
        assert runtime_sampler.metrics()["total_candidates"] >= 1


# ---------------------------------------------------------------------------
# Seam P10-actuate-#2 (pin) -- ProfileProbe.record_shadow_outcome's
# CascadeRecord is now persisted to the runtime DB at the same checkpoint
# the 1.4 xfail above documents as historically discarding it.
#
# PRODUCER: ProfileProbe.record_shadow_outcome -> CascadeRecord
#           (evaluation_cascade.py).
# CONSUMER: RuntimeStateStore.record_event via Pipeline._record_runtime_event
#           at the _full_evaluate checkpoint (orchestrator.py, right after
#           record_shadow_outcome) -- a real row in the `events` table,
#           keyed to the string's work_unit_id. No AuditSampler reads it back
#           yet (see 1.4 xfail); this only proves the shadow evidence is real
#           and queryable, per spec ("Activation stays OFF; this only makes
#           the shadow evidence real").
# ---------------------------------------------------------------------------


def test_seam_p10_cascade_shadow_record_persisted_to_runtime_events():
    import json

    with tempfile.TemporaryDirectory() as td:
        pipeline = _build_routing_pipeline(td)
        run_id = pipeline._runtime_state.start_run(
            source="linkedin",
            brief_id="test",
            output_dir=td,
            mode="full",
        )
        pipeline._runtime_run_id = run_id
        snippet = _make_snippet(profile_url="/talent/profile/seam-shadow-persist")
        search_string = SearchString(id=1, name="t", boolean="x", lane_id="lane-A")

        decision = _run_full_evaluate(
            pipeline, snippet, search_string, llm_return=_reject_response()
        )

        assert decision is not None
        assert len(pipeline._profile_probe.shadow_records) == 1

        with pipeline._runtime_state.connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM events WHERE run_id = ? "
                "AND event_type = 'cascade_shadow_recorded'",
                (run_id,),
            ).fetchall()

        assert len(rows) == 1
        payload = json.loads(rows[0]["payload_json"])
        assert payload["profile_url"] == snippet.profile_url
        assert payload["lane_id"] == "lane-A"
        assert payload["probe_decision"] == pipeline._profile_probe.shadow_records[0].probe_decision
