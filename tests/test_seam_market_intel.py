"""Phase 1 seam-contract tests for the market_intel cluster.

These pin the PRODUCER -> CONSUMER edge of each seam so a refactor cannot
silently sever the wiring. Each pin drives a real producer and asserts the
real consumer observes the produced value crossing the seam; the only mocks
are true external boundaries (none are needed here -- every seam in this
cluster is in-process Python). Inert seams are marked strict-xfail and run
far enough to fail AT the seam assertion.

Seam map (see cluster brief):
- 3.0  maybe_build_and_persist_research_packet -> build_research_context_bundle
- 3.1a build_external_research_backend key/provider gate (RuntimeError path)
- 3.1b ExternalResearchResult.sourcing_implications -> _build_artifact.planner_diffs
- 3.2  implication -> _build_gated_planner_diffs_from_implications gate downgrade
- 3.3  market-intel artifact planner_diffs -> load -> strategy prompt (full chain)
- 3.4  ExecutionPlan.consumed_feedback_ids -> diff retirement (INERT, strict-xfail)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import market_intelligence.engine as engine_mod
import market_intelligence.research_agent as research_agent_mod
from market_intelligence.engine import (
    ExternalResearchResult,
    _build_artifact,
    _build_gated_planner_diffs_from_implications,
    _merge_planner_diffs,
    load_lane_feedback_for_strategy,
    resolve_market_intel_artifact_path,
)
from market_intelligence.research_context import (
    build_research_context_bundle,
    maybe_build_and_persist_research_packet,
)
from market_intelligence.schema import MarketEvidenceBatch, MarketIdentity
from shared.brief_loader import load_brief
from shared.storage import read_json, write_json


_MINIMAL_BRIEF_JSON = (
    '{"role_title": "ML Engineer", "geography": "NYC", "role_level": "Senior"}'
)


def _market_identity() -> MarketIdentity:
    return MarketIdentity.from_dict(
        {
            "market_key": "ml_engineer__nyc__senior",
            "role_title": "ML Engineer",
            "role_level": "Senior",
            "geography": "NYC",
            "channels_seen": ["linkedin"],
            "brief_ids_seen": ["ml-eng"],
            "brief_versions_seen": ["2.1"],
        }
    )


def _strategy_brief() -> SimpleNamespace:
    """Duck-typed brief sufficient for linkedin.strategy._build_strategy_user."""
    return SimpleNamespace(
        role_title="ML Engineer",
        role_level="Senior",
        role_summary="Build ML systems",
        geography="NYC",
        linkedin_project="ML Eng",
        linkedin_project_id="123",
        jd_text="",
        intake_notes="",
        search_priorities=[],
        additional_search_terms=[],
        instructions=[],
        permanent_filters={},
        has_v2_schema=False,
        raw={},
    )


def _write_brief(tmp_path) -> object:
    """Write the minimal brief JSON load_brief + derive_market_key accept."""
    brief_path = tmp_path / "brief.json"
    brief_path.write_text(_MINIMAL_BRIEF_JSON)
    return brief_path


def _make_deterministic_summary(identity: MarketIdentity) -> dict:
    """Real deterministic summary (empty evidence) -- same producer _build_artifact
    is fed in production, so the consumed shape is faithful."""
    return engine_mod._build_deterministic_summary(
        market_identity=identity,
        evidence_batches=[],
        previous_artifact=None,
    )


# ---------------------------------------------------------------------------
# Seam 3.0 -- per-run research packet -> research-context bundle
#
# PRODUCER: maybe_build_and_persist_research_packet (research_context.py:773)
#   sets batch.research_context from batch.report (report_analysis.winning_lanes
#   copied verbatim from report["winning_lanes"], research_context.py:682-685,821).
# CONSUMER: build_research_context_bundle (research_context.py:1423) -> _lane_rollup
#   (research_context.py:849) reads batch.research_context["report_analysis"]
#   ["winning_lanes"] and rolls each winner.get("lane") into
#   cross_run_aggregate["lane_trend_summary"]; _select_run_packets feeds
#   run_packets[0]["report_analysis"]["winning_lanes"].
# ---------------------------------------------------------------------------

def test_seam_3_0_research_packet_winning_lane_surfaces_in_bundle(tmp_path):
    output_dir = tmp_path / "run-a"
    output_dir.mkdir()
    batch = MarketEvidenceBatch(
        run_ref="linkedin:output/run-a",
        source="linkedin",
        output_dir=str(output_dir),
        brief_version="2.1",
        generated_at="2026-04-08T10:00:00+00:00",
        report={
            "winning_lanes": [
                {
                    "lane": "ML Platform Builders",
                    "evidence": "12 saves",
                    "why_it_worked": "Strong platform signal",
                }
            ],
            "underperforming_lanes": [],
        },
        metrics_summary={
            "run_count": 1,
            "saved": 12,
            "candidate_volume": 120,
            "save_rate": 0.1,
        },
    )

    # PRODUCER: build + persist the packet (no research_context pre-set).
    assert batch.research_context is None
    produced = maybe_build_and_persist_research_packet(
        batch, reconstruct_report_analysis=False
    )
    # Sanity: producer actually set research_context with the winning lane.
    assert produced.research_context is not None
    assert (
        produced.research_context["report_analysis"]["winning_lanes"][0]["lane"]
        == "ML Platform Builders"
    )

    # CONSUMER: bundle reads batch.research_context across the seam.
    bundle = build_research_context_bundle(_market_identity(), [produced])

    lane_labels = {
        entry["lane"] for entry in bundle["cross_run_aggregate"]["lane_trend_summary"]
    }
    assert "ML Platform Builders" in lane_labels
    assert bundle["run_packets"], "consumer dropped the produced packet"
    packet_lanes = {
        winner["lane"]
        for winner in bundle["run_packets"][0]["report_analysis"]["winning_lanes"]
    }
    assert "ML Platform Builders" in packet_lanes


# ---------------------------------------------------------------------------
# Seam 3.1a -- external-research backend key/provider gate
#
# PRODUCER/GATE: build_external_research_backend (research_agent.py:1139) reads
#   config.MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER + config.PERPLEXITY_API_KEY +
#   config.ANTHROPIC_API_KEY; raises RuntimeError("No external research provider
#   ...") when provider is auto/blank and neither key is set (research_agent.py:1146).
#   The positive "perplexity when key set" branch is already pinned by
#   tests/test_market_intelligence.py::test_build_external_research_backend_
#   prefers_perplexity_when_available; this pins the RuntimeError branch that
#   gates engine.py:993-1003 should_collect_external = bool(backend) and ...
# ---------------------------------------------------------------------------

def test_seam_3_1a_backend_gate_raises_without_keys(monkeypatch):
    monkeypatch.setattr(
        research_agent_mod.config,
        "MARKET_INTEL_EXTERNAL_RESEARCH_PROVIDER",
        "",
        raising=False,
    )
    monkeypatch.setattr(
        research_agent_mod.config, "PERPLEXITY_API_KEY", "", raising=False
    )
    monkeypatch.setattr(
        research_agent_mod.config, "ANTHROPIC_API_KEY", "", raising=False
    )

    with pytest.raises(RuntimeError, match="No external research provider"):
        research_agent_mod.build_external_research_backend()

    # And with a Perplexity key present the gate yields a real backend instance
    # (the value that makes engine.py:994 bool(external_research_backend) True).
    monkeypatch.setattr(
        research_agent_mod.config, "PERPLEXITY_API_KEY", "pplx-test", raising=False
    )
    backend = research_agent_mod.build_external_research_backend()
    assert isinstance(backend, research_agent_mod.PerplexityResearchBackend)


# ---------------------------------------------------------------------------
# Seam 3.1b -- ExternalResearchResult.sourcing_implications -> planner_diffs
#
# PRODUCER: the external backend's .collect() yields an ExternalResearchResult
#   (engine.py:232) carrying sourcing_implications. CONSUMER: _build_artifact
#   (engine.py:2938) reads external_result.sourcing_implications (engine.py:3003-
#   3008), routes them through _build_gated_planner_diffs_from_implications, and
#   writes data["planner_diffs"]=merged (engine.py:3070). With external_result
#   None the same path emits only the previous diffs (engine.py:3015-3017).
#   We construct the REAL produced value (ExternalResearchResult) and assert the
#   consumer observes it crossing into the persisted artifact's planner_diffs.
# ---------------------------------------------------------------------------

def test_seam_3_1b_external_implications_drive_planner_diffs(tmp_path):
    brief = load_brief(str(_write_brief(tmp_path)))
    deterministic_summary = _make_deterministic_summary(_market_identity())

    def _planner_diffs(external_result):
        artifact = _build_artifact(
            brief=brief,
            market_identity=_market_identity(),
            deterministic_summary=deterministic_summary,
            evidence_batches=[],
            previous_artifact=None,
            generated_sections={},
            preserve_previous_narrative=False,
            external_result=external_result,
            section_generation_metadata={},
            delta_since_last_run={},
        )
        return artifact.to_dict()["planner_diffs"]

    # Real produced value: a compilable, internally-evidenced implication.
    external_result = ExternalResearchResult(
        sourcing_implications=[
            {
                "category": "probe_adjacent_pool",
                "recommendation": "Investigate MLOps platform engineers",
                "rationale": "Adjacent pool with transferable skills",
                "priority": "high",
                "supporting_run_refs": ["linkedin:output/run-a"],
            }
        ]
    )

    diffs_with_research = _planner_diffs(external_result)
    assert len(diffs_with_research) == 1, "implication did not cross into planner_diffs"
    assert diffs_with_research[0]["target_type"] == "hypothesis"

    # With no external result the same consumer path emits only prior diffs
    # (here: none), proving planner_diffs is keyed off external_result.
    diffs_without_research = _planner_diffs(None)
    assert diffs_without_research == []


# ---------------------------------------------------------------------------
# Seam 3.2 -- implication -> gated planner diff downgrade
#
# PRODUCER: _build_gated_planner_diffs_from_implications (engine.py:2663) maps
#   each implication via _implication_to_planner_diff (engine.py:2592;
#   add_employer_target -> constraint, dimension="employer_target") then routes
#   it through gate_planner_diff (shared/brief_iteration.py:90). CONSUMER: the
#   gate downgrades employer-inventory constraint diffs lacking internal_evidence
#   to target_type="validation_question" + warning (brief_iteration.py:115-139),
#   and the producer copies that warning into payload["gate_warning"]
#   (engine.py:2683-2684). This pins the implication->gated-edge as a chain,
#   distinct from test_market_intelligence_lane_feedback.py which calls
#   gate_planner_diff directly on a hand-built PlannerDiff.
# ---------------------------------------------------------------------------

def test_seam_3_2_employer_inventory_implication_downgraded_by_gate():
    # >=5 values -> values_text has >=4 commas -> looks_like_company_inventory True
    # (shared/strict_seniority.py:275). No supporting_run_refs -> no internal evidence.
    employer_implication = {
        "category": "add_employer_target",
        "recommendation": "Target buy-side and fintech employers",
        "rationale": "External signal",
        "priority": "high",
        "suggested_values": [
            "Citadel",
            "Two Sigma",
            "Jane Street",
            "Stripe",
            "Plaid",
            "Ramp",
        ],
        # NOTE: no supporting_run_refs -> internal_evidence == []
    }

    downgraded = _build_gated_planner_diffs_from_implications([employer_implication])
    assert len(downgraded) == 1
    assert downgraded[0]["target_type"] == "validation_question"
    assert "gate_warning" in downgraded[0]

    # Same implication WITH supporting_run_refs (internal evidence) stays a constraint.
    evidenced = dict(employer_implication)
    evidenced["supporting_run_refs"] = ["linkedin:output/run-a"]
    kept = _build_gated_planner_diffs_from_implications([evidenced])
    assert len(kept) == 1
    assert kept[0]["target_type"] == "constraint"
    assert "gate_warning" not in kept[0]


# ---------------------------------------------------------------------------
# Seam 3.3 -- artifact planner_diffs -> load -> strategy prompt (full chain)
#
# PRODUCER: market-intel artifact carries data["planner_diffs"] (engine.py:3070,
#   persisted at resolve_market_intel_artifact_path). load_lane_feedback_for_strategy
#   (engine.py:2701) re-reads planner_diffs (engine.py:2716-2723). CONSUMER:
#   linkedin.strategy._build_strategy_user (strategy.py:1249-1259) serializes the
#   diffs into the Opus prompt. The two existing tests pin only halves (load from a
#   hand-written file; prompt from a hand-built feedback list); this pins the
#   continuous write -> resolve -> load -> prompt chain through the real path
#   resolver so the write site and read site cannot disagree on the market key.
# ---------------------------------------------------------------------------

def test_seam_3_3_artifact_diffs_reach_strategy_prompt(tmp_path):
    from linkedin.strategy import _build_strategy_user

    brief_path = _write_brief(tmp_path)

    # PRODUCER: persist an artifact at the resolver-derived path.
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=tmp_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        artifact_path,
        {
            "planner_diffs": [
                {
                    "diff_id": "d1",
                    "action": "add",
                    "target_type": "hypothesis",
                    "payload": {"label": "ML Platform"},
                }
            ]
        },
    )

    # SEAM (load): re-read via the same resolver.
    feedback = load_lane_feedback_for_strategy(brief_path, output_dir=tmp_path)
    assert len(feedback) == 1
    assert feedback[0]["diff_id"] == "d1"

    # CONSUMER: the loaded diffs serialize into the strategy prompt.
    prompt = _build_strategy_user(_strategy_brief(), [], lane_feedback=feedback)
    assert "Lane Feedback Diffs" in prompt
    assert '"d1"' in prompt


# ---------------------------------------------------------------------------
# Seam 3.4 -- ExecutionPlan.consumed_feedback_ids -> diff retirement (INERT)
#
# PRODUCER: ExecutionPlan.consumed_feedback_ids is populated by the strategy LLM
#   and persisted to execution_plan.json (schemas.py:56,82). CONSUMER: NONE today.
#   _merge_planner_diffs (engine.py:2689) re-emits every prior diff unconditionally
#   and load_lane_feedback_for_strategy (engine.py:2719-2723) returns every diff
#   with a diff_id -- neither consults consumed_feedback_ids. The field is
#   write-only telemetry. This test asserts the seam AS IF wired (a consumed diff
#   is retired) and fails at exactly that assertion until Phase 2 adds a
#   consumed-diff retirement step that prunes consumed_feedback_ids from the
#   persisted planner_diffs.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Phase 2 consumed-diff retirement -- inert until a retirement step "
    "prunes ExecutionPlan.consumed_feedback_ids from the persisted planner_diffs "
    "(today _merge_planner_diffs re-emits d1 and load_lane_feedback_for_strategy "
    "re-returns it)",
    strict=True,
)
def test_seam_3_4_consumed_diff_is_retired(tmp_path):
    from shared.schemas import ExecutionPlan

    brief_path = _write_brief(tmp_path)
    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=tmp_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    # PRODUCER (prior run): an artifact whose planner_diffs already contains d1.
    write_json(
        artifact_path,
        {
            "planner_diffs": [
                {
                    "diff_id": "d1",
                    "action": "add",
                    "target_type": "hypothesis",
                    "payload": {"label": "ML Platform"},
                }
            ]
        },
    )

    # PRODUCER (strategy LLM): records d1 as consumed in execution_plan.json.
    plan = ExecutionPlan(strategy_rationale="consumed d1", consumed_feedback_ids=["d1"])
    plan_path = artifact_path.parent / "execution_plan.json"
    write_json(plan_path, plan.to_dict())

    # Re-run the merge/persist path the way the next artifact build would: the
    # prior diffs are merged forward. A Phase-2 retirement step would drop any
    # diff_id present in the persisted plan's consumed_feedback_ids before
    # re-persisting; today nothing does.
    persisted = read_json(artifact_path)
    consumed_ids = set(read_json(plan_path).get("consumed_feedback_ids", []))
    remerged = _merge_planner_diffs(
        list(persisted.get("planner_diffs", []) or []),
        [],  # no new implications this run
    )
    write_json(artifact_path, {"planner_diffs": remerged})

    # CONSUMER: strategy formation re-loads lane feedback for the next run.
    feedback = load_lane_feedback_for_strategy(brief_path, output_dir=tmp_path)
    returned_ids = {item["diff_id"] for item in feedback}

    # SEAM ASSERTION: a consumed diff must not be re-served. Fails today because
    # consumed_feedback_ids is never consulted (returned_ids == {"d1"}).
    assert consumed_ids.isdisjoint(returned_ids), (
        "consumed diff was re-served to strategy: "
        f"{consumed_ids & returned_ids}"
    )
