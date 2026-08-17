"""P7 Stages B-C — lane-key integrity + family-key stability (Wave 3 slice 12).

plans/sourcing-rigor-hardening.md P7: learning keys are validated, stable
identifiers. Stage B: a model-emitted ``domain_lane`` outside the declared
universe (brief ``domain_lane_hints`` ∪ profile lane template ids ∪
``general``) is remapped to the nearest declared slug or kept-but-flagged
``undeclared_lane``; a deterministic lane-collapse warning fires when >60% of
strings share one lane on a ≥2-declared-lane brief, printed AND carried to the
block report. Stage C: ``update_search_memory`` records the Jaccard overlap of
family_key sets between the latest run and the nearest prior same-epoch run;
<0.3 with a nonzero prior renders a run-report warning ("family keys churned;
learning is not accumulating").

Validation is ACTIVE only when the brief/plan declares at least one
non-general lane — on a hint-less brief the model is instructed to derive its
own lanes, so specific-but-undeclared labels are expected, not defects.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from shared.brief_loader import Brief
from shared.brief_schema import DomainLaneHint
from shared.schemas import BlockReport, ExecutionPlan, SearchString
from shared.search_memory import update_search_memory
from linkedin.strategy import _annotate_plan_metadata, form_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brief(**overrides) -> Brief:
    brief = Brief(
        id="lane-key-integrity-test",
        role_title="Payments Platform Engineer",
        role_description="Owns payment orchestration workflows.",
        kit_url="",
        linkedin_project="proj-lane-keys",
        linkedin_project_id="",
        minimum_bar="5+ years payments engineering.",
        archetypes=[{"name": "Payments builder"}],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )
    for key, value in overrides.items():
        setattr(brief, key, value)
    return brief


def _two_lane_hints() -> list[DomainLaneHint]:
    return [
        DomainLaneHint(lane="payments", patterns=["fednow", "orchestration"]),
        DomainLaneHint(lane="capital_markets", patterns=["custody", "post-trade"]),
    ]


def _string(family: str, lane: str, boolean: str = '("payments")') -> dict:
    return {
        "boolean": boolean,
        "rationale": f"{family} population",
        "vocabulary_sources": "mock",
        "family_key": family,
        "novelty_bucket": "canonical",
        "domain_lane": lane,
    }


def _plan(strings: list[dict], sourcing_lanes: list[dict] | None = None) -> ExecutionPlan:
    return ExecutionPlan(
        strategy_rationale="mock",
        generated_strings=strings,
        sourcing_lanes=sourcing_lanes or [],
    )


# ---------------------------------------------------------------------------
# Stage B — lane validation in _annotate_plan_metadata
# ---------------------------------------------------------------------------

def test_undeclared_lane_remaps_to_nearest_declared_slug():
    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    plan = _plan([_string("payments_eng", "payments_engineering")])

    _annotate_plan_metadata(brief, plan)

    item = plan.generated_strings[0]
    assert item["domain_lane"] == "payments"
    assert item["domain_lane_raw"] == "payments_engineering"
    assert not item.get("undeclared_lane")
    # The remap is visible at plan level too (P5: a flag nobody sees is a
    # dead lever) and stays idempotent across re-annotation passes.
    _annotate_plan_metadata(brief, plan)
    remap_warnings = [
        w for w in plan.plan_warnings if w.get("code") == "lane_remapped"
    ]
    assert len(remap_warnings) == 1
    assert "payments_engineering→payments" in remap_warnings[0]["message"]


def test_unmappable_lane_kept_and_flagged_undeclared():
    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    plan = _plan([_string("healthcare", "healthcare_payers")])

    _annotate_plan_metadata(brief, plan)

    item = plan.generated_strings[0]
    assert item["domain_lane"] == "healthcare_payers"
    assert item["undeclared_lane"] is True


def test_declared_hint_profile_and_general_lanes_pass_untouched():
    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    plan = _plan(
        [
            _string("a", "payments"),
            _string("b", "general"),
            _string("c", "supply_chain_lane"),
        ],
        sourcing_lanes=[{"lane_id": "supply_chain_lane"}],
    )

    _annotate_plan_metadata(brief, plan)

    for item in plan.generated_strings:
        assert not item.get("undeclared_lane")
        assert "domain_lane_raw" not in item
    assert [i["domain_lane"] for i in plan.generated_strings] == [
        "payments",
        "general",
        "supply_chain_lane",
    ]


def test_hintless_brief_flags_nothing_through_production_order():
    """No brief hints → the model derives lanes; specific labels are expected,
    not defects. MUST run through form_strategy: the generic role-strategy
    profile always merges at least one lane template into sourcing_lanes, so
    a direct _annotate_plan_metadata call would never exercise the real
    activation order (correctness lens, slice 12)."""
    brief = _make_brief()
    payload = _collapse_payload(["specialty_pharma", "general", "specialty_pharma"])

    with patch("linkedin.strategy.opus_llm", return_value=payload):
        plan = form_strategy(brief, [], prior_run_data={})

    # The generic profile DID merge a lane template (the trap this test pins).
    assert plan.sourcing_lanes, "expected the generic profile to seed a lane"
    for item in plan.generated_strings:
        assert not item.get("undeclared_lane")
        assert "domain_lane_raw" not in item
    assert not [
        w
        for w in plan.plan_warnings
        if w.get("code") in {"undeclared_lane", "lane_remapped"}
    ]


def test_ambiguous_slug_match_keeps_and_flags():
    brief = _make_brief(
        domain_lane_hints=[
            DomainLaneHint(lane="payments_infra", patterns=["rails"]),
            DomainLaneHint(lane="payments_ops", patterns=["chargeback"]),
        ]
    )
    plan = _plan([_string("p", "payments")])

    _annotate_plan_metadata(brief, plan)

    item = plan.generated_strings[0]
    assert item["domain_lane"] == "payments"  # never guess between two
    assert item["undeclared_lane"] is True


def test_adaptation_strings_get_same_lane_validation():
    """Adaptation is a second string-producing path — same validation, when
    the plan context is available (Wave-2 lint-gate precedent)."""
    from shared.schemas import AdaptationResponse
    from linkedin.strategy import _annotate_adaptation_metadata

    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    plan = _plan([])
    adaptation = AdaptationResponse(
        new_strings=[
            _string("a", "payments_engineering"),
            _string("b", "healthcare_payers"),
        ]
    )

    _annotate_adaptation_metadata(brief, adaptation, plan=plan)

    remapped, flagged = adaptation.new_strings
    assert remapped["domain_lane"] == "payments"
    assert remapped["domain_lane_raw"] == "payments_engineering"
    assert flagged["undeclared_lane"] is True


# ---------------------------------------------------------------------------
# Stage B — lane-collapse warning (through the production entry)
# ---------------------------------------------------------------------------

def _collapse_payload(lanes: list[str]) -> dict:
    return {
        "architecture": "dragnet",
        "architecture_rationale": "mock",
        "architecture_success_criteria": [],
        "architecture_pivot_triggers": [],
        "strategy_rationale": "mock",
        "generated_strings": [
            _string(f"family_{i}", lane) for i, lane in enumerate(lanes)
        ],
        "coverage_gaps": [],
        "noise_predictions": [],
    }


def test_lane_collapse_warning_lands_on_plan_and_prints(capsys):
    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    payload = _collapse_payload(
        ["payments", "payments", "payments", "payments", "capital_markets"]
    )

    with patch("linkedin.strategy.opus_llm", return_value=payload):
        plan = form_strategy(brief, [], prior_run_data={})

    warnings = [w for w in plan.plan_warnings if w.get("code") == "lane_collapse"]
    assert len(warnings) == 1
    assert warnings[0]["lane"] == "payments"
    assert "4/5" in warnings[0]["message"]
    assert "lane collapse" in capsys.readouterr().out


def test_no_lane_collapse_warning_below_threshold_or_single_lane_brief():
    # 3/5 (60%) is NOT >60% — no warning.
    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    payload = _collapse_payload(
        ["payments", "payments", "payments", "capital_markets", "capital_markets"]
    )
    with patch("linkedin.strategy.opus_llm", return_value=payload):
        plan = form_strategy(brief, [], prior_run_data={})
    assert not [w for w in plan.plan_warnings if w.get("code") == "lane_collapse"]

    # Single declared lane — collapse is definitional, not a defect.
    brief = _make_brief(
        domain_lane_hints=[DomainLaneHint(lane="payments", patterns=["fednow"])]
    )
    payload = _collapse_payload(["payments", "payments", "payments"])
    with patch("linkedin.strategy.opus_llm", return_value=payload):
        plan = form_strategy(brief, [], prior_run_data={})
    assert not [w for w in plan.plan_warnings if w.get("code") == "lane_collapse"]


def test_plan_warnings_round_trip_through_dict():
    payload = {"strategy_rationale": "m", "plan_warnings": [{"code": "lane_collapse"}]}
    plan = ExecutionPlan.from_dict(payload)
    assert plan.plan_warnings == [{"code": "lane_collapse"}]
    assert plan.to_dict()["plan_warnings"] == [{"code": "lane_collapse"}]


def test_block_report_renders_plan_warnings():
    report = BlockReport(
        block_name="Block 1",
        strings_run=3,
        plan_warnings=["lane collapse: 4/5 strings in 'payments'"],
    )
    text = report.to_summary_text()
    assert "lane collapse: 4/5 strings in 'payments'" in text


# ---------------------------------------------------------------------------
# Stage C — family-key stability in update_search_memory
# ---------------------------------------------------------------------------

def _run_string(family: str, run_id: int, epoch: str = "epoch-1") -> SimpleNamespace:
    return SimpleNamespace(
        family_key=family,
        boolean=f'("{family}")',
        name=family,
        novelty_bucket="canonical",
        domain_lane="general",
        candidates_count=5,
        duplicates_count=0,
        suppressed_prior_session_count=0,
        saves=[],
        facial_yes_count=1,
        facial_no_count=2,
        pages_reviewed=1,
        retrieval_recipe={},
        retrieval_hypothesis_ids=[],
        run_id=run_id,
        brief_epoch=epoch,
    )


def test_key_stability_warns_on_family_churn_across_runs():
    strings = [
        _run_string("alpha", 1),
        _run_string("beta", 1),
        _run_string("gamma", 2),
        _run_string("delta", 2),
    ]
    memory = update_search_memory({}, "proj", strings)

    stability = memory["key_stability"]
    assert stability["jaccard"] == 0.0
    assert stability["warning"] is True
    assert stability["prior_run_id"] == 1
    assert stability["current_run_id"] == 2


def test_key_stability_no_warning_on_high_overlap():
    strings = [
        _run_string("alpha", 1),
        _run_string("beta", 1),
        _run_string("alpha", 2),
        _run_string("beta", 2),
    ]
    memory = update_search_memory({}, "proj", strings)

    stability = memory["key_stability"]
    assert stability["jaccard"] == 1.0
    assert stability["warning"] is False


def test_key_stability_absent_for_first_run_and_epoch_change():
    # Single run: no prior to compare against.
    memory = update_search_memory({}, "proj", [_run_string("alpha", 1)])
    assert "key_stability" not in memory

    # Epoch changed between runs: churn is expected, metric skips.
    strings = [
        _run_string("alpha", 1, epoch="epoch-1"),
        _run_string("gamma", 2, epoch="epoch-2"),
    ]
    memory = update_search_memory({}, "proj", strings)
    assert "key_stability" not in memory


def test_key_stability_survives_incremental_persisted_memory_calls():
    """The established incremental pattern (persisted memory + one run's
    delta per call) must not erase the metric: per-run family history is
    persisted and merged, so run 2's call sees run 1 (correctness lens,
    slice 12)."""
    memory = update_search_memory({}, "proj", [_run_string("alpha", 1)])
    assert "key_stability" not in memory

    memory = update_search_memory(memory, "proj", [_run_string("gamma", 2)])
    stability = memory["key_stability"]
    assert stability["prior_run_id"] == 1
    assert stability["current_run_id"] == 2
    assert stability["jaccard"] == 0.0
    assert stability["warning"] is True

    # And a third incremental call with overlapping keys clears the warning.
    memory = update_search_memory(memory, "proj", [_run_string("gamma", 3)])
    assert memory["key_stability"]["jaccard"] == 1.0
    assert memory["key_stability"]["warning"] is False


def test_lane_markers_survive_search_string_round_trip():
    """P7 Stage B markers persist across the plan-dict→SearchString boundary
    (contract lens, slice 12: explicit-kwarg construction dropped them)."""
    from shared.schemas import SearchString

    ss = SearchString(
        id=1,
        name="s",
        boolean='("x")',
        domain_lane="payments",
        domain_lane_raw="payments_engineering",
        undeclared_lane=True,
    )
    round_tripped = SearchString.from_dict(ss.to_dict())
    assert round_tripped.domain_lane_raw == "payments_engineering"
    assert round_tripped.undeclared_lane is True


def test_key_stability_skips_intermediate_other_epoch_runs():
    """The prior run is the NEAREST same-epoch run, not merely run N-1."""
    strings = [
        _run_string("alpha", 1, epoch="epoch-1"),
        _run_string("other", 2, epoch="epoch-2"),
        _run_string("alpha", 3, epoch="epoch-1"),
    ]
    memory = update_search_memory({}, "proj", strings)

    stability = memory["key_stability"]
    assert stability["prior_run_id"] == 1
    assert stability["current_run_id"] == 3
    assert stability["jaccard"] == 1.0


def test_two_lane_report_aggregates_two_lane_rows():
    """P7 guardrail at the aggregation seam itself: two string_performance
    entries with distinct family_key + domain_lane produce two lane rows.
    The end-to-end artifact variant lives in test_market_intelligence.py but
    skips in checkouts without the optional brief fixture — this one always
    runs (verify-the-verifier)."""
    import market_intelligence.engine as engine_mod
    from market_intelligence.schema import MarketEvidenceBatch

    batch = MarketEvidenceBatch(
        run_ref="linkedin:output/run-a",
        source="linkedin",
        output_dir="output/run-a",
        brief_version="2.1",
        generated_at="2026-07-04T00:00:00+00:00",
        report={
            "string_performance": [
                {
                    "family_key": "payments_orchestration",
                    "domain_lane": "payments",
                    "novelty_bucket": "canonical",
                    "name": "payments string",
                    "candidates_count": 10,
                    "saves_count": 2,
                },
                {
                    "family_key": "capital_markets_trading",
                    "domain_lane": "capital_markets",
                    "novelty_bucket": "edge_case",
                    "name": "capital markets string",
                    "candidates_count": 8,
                    "saves_count": 1,
                },
            ]
        },
    )

    rows = engine_mod._aggregate_lane_intelligence(
        evidence_batches=[batch], previous_artifact=None
    )

    assert len(rows) >= 2
    lane_keys = {row["lane_key"] for row in rows}
    domain_lanes = {row["domain_lane"] for row in rows}
    assert {"payments_orchestration", "capital_markets_trading"} <= lane_keys
    assert {"payments", "capital_markets"} <= domain_lanes


def test_projection_computes_key_stability_across_store_runs(tmp_path):
    """Production seam: the SQLite projection attaches run_id + brief_epoch
    per work unit, so a full memory rebuild still yields the run-over-run
    stability metric (churned keys across two same-epoch runs → warning)."""
    from shared.runtime_state import RuntimeStateStore
    from shared.runtime_state.projections import project_linkedin_search_memory
    from shared.runtime_state.store import LINKEDIN_STRING_KIND
    from shared.schemas import SearchString

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")

    def _run_with_families(run_ordinal: int, families: list[str]) -> None:
        run_id = store.start_run(
            source="linkedin",
            brief_id="brief-stability",
            output_dir=str(tmp_path),
            mode="fresh",
            resume_state={"brief_name": "brief-stability"},
            brief_content_hash="epoch-hash-1",
        )
        for i, family in enumerate(families):
            string = SearchString(
                id=run_ordinal * 100 + i,
                name=family,
                boolean=f'("{family}")',
                status="done",
                result_count=5,
                family_key=family,
                novelty_bucket="canonical",
                domain_lane="general",
                candidates_count=4,
            )
            store.upsert_work_unit(
                run_id=run_id,
                source="linkedin",
                brief_id="brief-stability",
                kind=LINKEDIN_STRING_KIND,
                source_unit_id=str(string.id),
                display_name=string.name,
                ordering_index=i,
                status="done",
                payload=string.to_dict(),
                family_key=string.family_key,
                novelty_bucket=string.novelty_bucket,
                domain_lane=string.domain_lane,
                counters={"result_count": 5, "candidates_discovered": 4},
            )

    _run_with_families(1, ["alpha", "beta"])
    _run_with_families(2, ["gamma", "delta"])

    memory = project_linkedin_search_memory(store, brief_id="brief-stability")

    stability = memory["key_stability"]
    assert stability["jaccard"] == 0.0
    assert stability["warning"] is True
    assert stability["brief_epoch"] == "epoch-hash-1"


# ---------------------------------------------------------------------------
# Production wiring (test-honesty lens, slice 12: every seam below was
# implemented but unlocked — a revert of the wire passed the full suite)
# ---------------------------------------------------------------------------

def test_adapt_after_block_validates_adapted_string_lanes():
    """The REAL adaptation entry threads plan= into lane validation."""
    from linkedin.strategy import adapt_after_block

    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    plan = _plan([])
    report = BlockReport(block_name="Block 1", strings_run=1)
    mock_adaptation = {
        "new_strings": [
            _string("a", "payments_engineering", boolean='("fednow")'),
            _string("b", "healthcare_payers", boolean='("claims")'),
        ],
        "skip_remaining": [],
        "reorder": [],
        "noise_updates": [],
    }

    with patch("linkedin.strategy.opus_llm", return_value=mock_adaptation):
        adaptation = adapt_after_block(
            brief,
            report,
            [],
            execution_plan=plan,
        )

    remapped, flagged = adaptation.new_strings
    assert remapped["domain_lane"] == "payments"
    assert remapped["domain_lane_raw"] == "payments_engineering"
    assert flagged["undeclared_lane"] is True


def test_run_block_adaptation_threads_plan_warnings_into_block_report():
    """The orchestrator's real block-report build carries plan.plan_warnings."""
    import asyncio
    import tempfile

    from shared.schemas import AdaptationResponse, Progress
    from tests.test_linkedin_pipeline import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = _plan([])
        p._execution_plan.plan_warnings = [
            {"code": "lane_collapse", "message": "lane collapse: 4/5 in 'payments'"}
        ]
        seen: dict = {}

        def spy_adapt_fn(brief, report, remaining, **kwargs):
            seen["plan_warnings"] = list(report.plan_warnings)
            return AdaptationResponse(no_change=True)

        done = SearchString(
            id=1,
            name="s1",
            boolean='("x")',
            status="done",
            result_count=10,
            candidates_count=5,
            pages_reviewed=1,
        )
        queued = SearchString(id=2, name="s2", boolean='("y")', status="queued")
        progress = Progress(brief_name="test", strings=[done, queued])
        asyncio.run(
            p._run_block_adaptation("Block 1", [done], progress, spy_adapt_fn)
        )

    assert seen["plan_warnings"] == ["lane collapse: 4/5 in 'payments'"]


def test_snapshot_stamps_key_stability_from_search_memory():
    """_build_run_report_snapshot reads the metric off the memory artifact."""
    import tempfile

    from shared.schemas import Progress
    from tests.test_linkedin_pipeline import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._search_memory = {
            "key_stability": {"jaccard": 0.1, "warning": True, "prior_run_id": 1}
        }
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1, name="s", boolean='("x")', status="done", pages_reviewed=1
                )
            ],
        )
        snapshot = p._build_run_report_snapshot(progress)

    assert snapshot["metrics_summary"]["key_stability"]["jaccard"] == 0.1

    # And absent-from-memory means absent-from-snapshot (only-when-present).
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._search_memory = {}
        snapshot = p._build_run_report_snapshot(
            Progress(
                brief_name="test",
                strings=[
                    SearchString(
                        id=1, name="s", boolean='("x")', status="done", pages_reviewed=1
                    )
                ],
            )
        )
    assert "key_stability" not in snapshot["metrics_summary"]


def test_exploitation_overlay_matches_raw_lane_spelling():
    """Promotion matching accepts the pre-remap spelling: a string remapped to
    'fintech' still matches history proven under raw 'fintech_infra'
    (the resume-boundary splinter, contract lens slice 12)."""
    import tempfile

    from shared.schemas import AdaptationResponse
    from tests.test_linkedin_pipeline import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        ss = SearchString(
            id=7,
            name="fintech string",
            boolean='("fintech")',
            family_key="fam_fintech",
            novelty_bucket="edge_case",
            domain_lane="fintech",
            domain_lane_raw="fintech_infra",
        )
        result = p._apply_exploitation_bias_to_adaptation(
            adaptation=AdaptationResponse(),
            remaining=[ss],
            block_summary={
                "proven_family_keys": [],
                "proven_domain_lanes": ["fintech_infra"],
                "dead_family_keys": [],
                "contaminated_family_keys": [],
                "contaminated_domain_lanes": [],
            },
            checkpoint_mode="normal_block_checkpoint",
        )

    assert 7 in result["promoted_string_ids"]


# ---------------------------------------------------------------------------
# Stage C — run-report rendering
# ---------------------------------------------------------------------------

def _structured_report(metrics_summary: dict):
    from shared.run_report_schema import StructuredRunReport

    return StructuredRunReport(
        schema_version=1,
        run_metadata={"brief_name": "test-brief"},
        metrics_summary=metrics_summary,
        string_performance=[],
        winning_lanes=[],
        underperforming_lanes=[],
        coverage_gaps=[],
        noise_patterns=[],
        saved_candidate_patterns={},
        adaptation_assessment={},
        recommendations={},
        brief_iteration_hints={},
    )


def test_run_report_renders_key_stability_warning():
    from shared.run_report_schema import render_run_report_markdown

    report = _structured_report(
        {
            "key_stability": {
                "jaccard": 0.1,
                "prior_run_id": 1,
                "current_run_id": 2,
                "prior_family_count": 4,
                "current_family_count": 5,
                "warning": True,
            }
        }
    )
    text = render_run_report_markdown(report)
    assert "family keys churned" in text.lower()
    assert "0.1" in text


def test_run_report_silent_without_key_stability_warning():
    from shared.run_report_schema import render_run_report_markdown

    report = _structured_report(
        {"key_stability": {"jaccard": 0.9, "warning": False}}
    )
    text = render_run_report_markdown(report)
    assert "churned" not in text.lower()


def test_format_search_memory_summary_renders_validated_status_and_low_wilson_confidence():
    """P6 residual (Wave 3 slice 14): status stays gated at 2/2 while the
    rendered text carries the volume-aware Wilson confidence beside it —
    'validated' alone reads as far stronger evidence than 2 saves off 6
    candidates actually is. Lives HERE (not test_search_memory.py) because
    that module skips without an optional local brief fixture and this lock
    must always run (test-honesty lens)."""
    import pytest

    from shared.schemas import SearchString
    from shared.search_memory import (
        format_search_memory_summary,
        wilson_lower_bound,
    )

    def _thin_string(sid: int, name: str, save: str) -> SearchString:
        return SearchString(
            id=sid,
            name=name,
            boolean='("edge case anchor")',
            pages_reviewed=1,
            candidates_count=3,
            duplicates_count=0,
            saves=[save],
            family_key="thin_evidence_family",
            novelty_bucket="edge_case",
            domain_lane="general",
            retrieval_recipe={"applied_hypothesis_ids": ["thin_evidence_hyp"]},
            retrieval_hypothesis_ids=["thin_evidence_hyp"],
        )

    memory = update_search_memory(
        {}, "proj", [_thin_string(10, "one", "Ada"), _thin_string(11, "two", "Grace")]
    )

    hyp = memory["hypotheses"]["thin_evidence_hyp"]
    assert hyp["status"] == "validated"  # the 2/2 gate must NOT change
    expected_confidence = wilson_lower_bound(2, 6)
    assert hyp["confidence"] == pytest.approx(expected_confidence)
    assert expected_confidence < 0.2

    rendered = format_search_memory_summary(memory)
    assert "status=validated" in rendered
    assert f"confidence={expected_confidence:.2f}" in rendered


def test_off_geo_warn_is_wired_at_both_save_sites():
    """Wiring lock (test-honesty lens, slice 14): deleting the two
    _warn_if_off_geo_save calls inside the save-outcome branches left the
    whole suite green. The helper's behavior is locked elsewhere; this pins
    the WIRES — the orchestrator module must call the helper at least twice
    outside its own definition (the two review loops' save branches)."""
    import ast
    import inspect

    import linkedin.orchestrator as orch

    tree = ast.parse(inspect.getsource(orch))
    call_sites = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "_warn_if_off_geo_save":
                call_sites += 1
    assert call_sites >= 2, (
        "off-geo WARN telemetry unwired: expected the two save-branch call "
        f"sites, found {call_sites}"
    )


def test_run_report_renders_facial_band_source():
    """The band's provenance renders beside the authored band (Wave 3 slice
    14; the suffix shipped with zero coverage — test-honesty lens)."""
    from shared.run_report_schema import render_run_report_markdown

    report = _structured_report(
        {
            "facial_calibration": {
                "status": "ok",
                "actual_yes_rate": 0.4,
                "authored_low": 0.2,
                "authored_high": 0.9,
                "band_source": "loader_default",
                "deviation_from_band": 0.0,
                "out_of_band": False,
                "calibration_drift_warning": False,
            }
        }
    )
    text = render_run_report_markdown(report)
    assert "band source: loader_default" in text


# ---------------------------------------------------------------------------
# Codex review, Wave 3 (F2): coverage gaps are a SECOND executable surface —
# same lane validation and marker aggregation as generated_strings.
# ---------------------------------------------------------------------------


def _gap(family: str, lane: str, boolean: str = '("claims")') -> dict:
    return {
        "gap": f"{family} population",
        "suggested_boolean": boolean,
        "rationale": f"{family} rationale",
        "family_key": family,
        "novelty_bucket": "edge_case",
        "domain_lane": lane,
    }


def test_executable_coverage_gaps_get_lane_validation():
    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    plan = _plan([])
    plan.coverage_gaps = [
        _gap("h", "healthcare_payers"),
        _gap("p", "payments_engineering"),
        {"gap": "non-executable note with no suggested boolean"},
    ]

    _annotate_plan_metadata(brief, plan)

    flagged, remapped, note = plan.coverage_gaps
    assert flagged["undeclared_lane"] is True
    assert remapped["domain_lane"] == "payments"
    assert remapped["domain_lane_raw"] == "payments_engineering"
    assert "undeclared_lane" not in note  # non-executable gaps untouched


def test_plan_warning_aggregates_cover_coverage_gaps():
    brief = _make_brief(domain_lane_hints=_two_lane_hints())
    plan = _plan([_string("a", "payments")])
    plan.coverage_gaps = [_gap("h", "healthcare_payers")]

    _annotate_plan_metadata(brief, plan)

    undeclared = [
        w for w in plan.plan_warnings if w.get("code") == "undeclared_lane"
    ]
    assert len(undeclared) == 1
    assert "healthcare_payers" in undeclared[0]["message"]


# ---------------------------------------------------------------------------
# Telemetry demotion (2026-07-04): per-string bias context rides the block
# report into the adaptation prompt — the structured replacement for the
# deleted mid-string bias pause. Locks live here because this module is
# fixture-free and always runs.
# ---------------------------------------------------------------------------


def test_run_block_adaptation_threads_bias_context_into_block_report():
    """The orchestrator's real block-report build carries string_context
    keyed by str(s.id) — the monitor keys strings as strings — plus the
    session-level expected band once per report."""
    import asyncio
    import tempfile

    from shared.bias_controls import BiasMonitor, DecisionRecord
    from shared.schemas import AdaptationResponse, Progress
    from tests.test_linkedin_pipeline import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p._execution_plan = _plan([])
        p._bias_monitor = BiasMonitor(
            max_consecutive_saves=1,
            expected_facial_yes_low=0.1,
            expected_facial_yes_high=0.22,
        )
        p._bias_monitor.record_decision(DecisionRecord(
            candidate_id="c1",
            string_id="1",
            stage="full",
            decision="SAVE",
            confidence=0.8,
            capability_area=None,
        ))
        p._bias_monitor.check_alerts("1")

        seen: dict = {}

        def spy_adapt_fn(brief, report, remaining, **kwargs):
            seen["bias"] = [sd.get("bias") for sd in report.string_details]
            seen["band"] = report.bias_expected_band
            seen["summary"] = report.to_summary_text()
            return AdaptationResponse(no_change=True)

        done = SearchString(
            id=1,
            name="s1",
            boolean='("x")',
            status="done",
            result_count=10,
            candidates_count=5,
            pages_reviewed=1,
        )
        queued = SearchString(id=2, name="s2", boolean='("y")', status="queued")
        progress = Progress(brief_name="test", strings=[done, queued])
        asyncio.run(
            p._run_block_adaptation("Block 1", [done], progress, spy_adapt_fn)
        )

    assert seen["band"] == [0.1, 0.22]
    ctx = seen["bias"][0]
    assert ctx is not None
    assert ctx["full_evals"] == 1
    assert ctx["saves"] == 1
    assert ctx["fired_alert_types"] == ["consecutive_saves"]
    # And the model-facing rendering carries both lines.
    assert "Expected opens-for-full-eval band (brief calibration): 10%–22%" in seen["summary"]
    assert "Bias context: 1/1 full evals saved (100%)" in seen["summary"]
    assert "signals: consecutive_saves" in seen["summary"]


def test_block_report_bias_line_renders_only_when_present():
    report = BlockReport(
        block_name="B",
        strings_run=1,
        string_details=[{
            "string_id": 1,
            "boolean": "x",
            "saves": 0,
            "pages_reviewed": 1,
            "result_count": 5,
            "bias": None,
        }],
    )
    text = report.to_summary_text()
    assert "Bias context:" not in text
    assert "Expected opens-for-full-eval band" not in text


def test_adapt_after_block_system_prompt_explains_bias_context_two_sided():
    """The adaptation system prompt names the field and keeps BOTH readings
    (exploitable vein vs loosened bar) so the model reasons rather than
    pattern-matches. No characterization coverage existed for this prompt;
    this is its presence lock."""
    from shared.schemas import AdaptationResponse, BlockReport as BR

    captured: dict = {}

    def fake_opus(system, user_prompt, **kwargs):
        captured["system"] = system
        captured["user"] = user_prompt
        return {"no_change": True}

    brief = _make_brief()
    report = BR(block_name="B", strings_run=0)
    with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
        from linkedin.strategy import adapt_after_block

        result = adapt_after_block(brief, report, [])

    assert isinstance(result, AdaptationResponse)
    assert 'Per-string "Bias context" lines report save density' in captured["system"]
    assert "vein worth exploiting" in captured["system"]
    assert "bar has loosened" in captured["system"]


# ---------------------------------------------------------------------------
# Preflight confidence notes reach the report (2026-07-04): run_metadata
# threading is isinstance-guarded like the provenance sibling, and the
# markdown renders the banner beside the unreviewed-brief stamp.
# ---------------------------------------------------------------------------


def test_snapshot_threads_preflight_confidence_notes_only_when_string():
    import tempfile

    from shared.schemas import Progress
    from tests.test_linkedin_pipeline import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.raw = {
            "preflight_confidence_notes": "Role level is ambiguous — confirm the band."
        }
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1, name="s", boolean='("x")', status="done", pages_reviewed=1
                )
            ],
        )
        snapshot = p._build_run_report_snapshot(progress)
        assert snapshot["run_metadata"]["preflight_confidence_notes"] == (
            "Role level is ambiguous — confirm the band."
        )

    # Absent (or non-string, e.g. a Mock attribute) → key absent, never a
    # coerced repr.
    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        p.brief_obj.raw = {}
        progress = Progress(
            brief_name="test",
            strings=[
                SearchString(
                    id=1, name="s", boolean='("x")', status="done", pages_reviewed=1
                )
            ],
        )
        snapshot = p._build_run_report_snapshot(progress)
        assert "preflight_confidence_notes" not in snapshot["run_metadata"]


def test_run_report_markdown_renders_confidence_notes_banner():
    from shared.run_report_schema import StructuredRunReport, render_run_report_markdown

    report = StructuredRunReport(
        schema_version=1,
        run_metadata={
            "brief_name": "test-brief",
            "preflight_confidence_notes": "Role level is ambiguous — confirm the band.",
        },
        metrics_summary={},
        string_performance=[],
        winning_lanes=[],
        underperforming_lanes=[],
        coverage_gaps=[],
        noise_patterns=[],
        saved_candidate_patterns={},
        adaptation_assessment={},
        recommendations={},
        brief_iteration_hints={},
    )
    text = render_run_report_markdown(report)
    assert "⚠ PREFLIGHT CONFIDENCE NOTES (operator review):" in text
    assert "Role level is ambiguous — confirm the band." in text


# ---------------------------------------------------------------------------
# RC1 (2026-07-04 SPL RCA): the tapped-market playbook binds ONLY when the
# brief declares a worked market; the intersection-anchor rule binds always.
# ---------------------------------------------------------------------------


def test_formation_prompt_omits_tapped_playbook_without_brief_trigger():
    from linkedin.strategy import _build_strategy_system

    system = _build_strategy_system(_make_brief(), has_kit=False)

    assert "Tapped-Market" not in system
    assert "NOVELTY ACCOUNTING" not in system
    # The anchor rule is unconditional (de-prescribed craft principle 3).
    assert "intersect the bridge population" in system
    assert "the industry analog" in system
    assert "must operationalize" in system


def test_formation_prompt_includes_tapped_playbook_when_brief_declares_it():
    from linkedin.strategy import _build_strategy_system

    brief = _make_brief(
        instructions=["The obvious pool for this role is heavily worked; open with edge cases."]
    )
    system = _build_strategy_system(brief, has_kit=False)

    assert "Tapped-Market" in system
    assert "NOVELTY ACCOUNTING" in system
    # The anchor rule still binds alongside the playbook.
    assert "intersect the bridge population" in system


def _captured_adapt_system(brief) -> str:
    from shared.schemas import BlockReport as BR

    captured: dict = {}

    def fake_opus(system, user_prompt, **kwargs):
        captured["system"] = system
        return {"no_change": True}

    with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
        from linkedin.strategy import adapt_after_block

        adapt_after_block(brief, BR(block_name="B", strings_run=0), [])
    return captured["system"]


def test_adaptation_prompt_scopes_tapped_doctrine_to_declaring_briefs():
    plain = _captured_adapt_system(_make_brief())
    assert "If the brief says the obvious pool is tapped" not in plain
    assert "In tapped markets, evaluate BLOCK QUALITY" not in plain
    # Unconditional pieces survive the scoping.
    assert "Every new string must carry at least one anchor" in plain
    assert 'Per-string "Bias context" lines report save density' in plain

    tapped = _captured_adapt_system(
        _make_brief(instructions=["This market is already heavily worked."])
    )
    assert "If the brief says the obvious pool is tapped" in tapped
    assert "In tapped markets, evaluate BLOCK QUALITY" in tapped
    assert 'Per-string "Bias context" lines report save density' in tapped


# ---------------------------------------------------------------------------
# RC2 (2026-07-04 SPL RCA): memory records the discovered pocket — exemplars
# and the winning (refined) boolean — under the SAME family key, and the
# formation guidance is yield-aware (proven veins are opening bets, not
# cleanup). Locks live here because test_search_memory.py skips without the
# committed fixture.
# ---------------------------------------------------------------------------


def _saved_string(**overrides):
    defaults = dict(
        id=1,
        name="s1",
        boolean='("chief of staff") AND ("human data" OR "annotation")',
        status="done",
        pages_reviewed=2,
        candidates_count=10,
        family_key="startup_ops_transfer",
        novelty_bucket="edge_case",
        domain_lane="startup_bizops",
        saves=["A Person", "B Person"],
        save_exemplars=[
            {"title": "VP Business Operations", "company": "Surge AI"},
            {"title": "Human Data Ops Lead", "company": "Meta"},
        ],
        original_boolean='("chief of staff")',
        refinement_stack=['("chief of staff")'],
    )
    defaults.update(overrides)
    return SearchString(**defaults)


def test_update_search_memory_records_exemplars_and_winning_boolean():
    memory = update_search_memory(None, "proj", [_saved_string()])

    entry = memory["families"]["startup_ops_transfer"]
    # The boolean AS EXECUTED when the saves landed — the refined form.
    assert entry["winning_boolean"] == '("chief of staff") AND ("human data" OR "annotation")'
    assert entry["save_exemplars"] == [
        {"title": "VP Business Operations", "company": "Surge AI"},
        {"title": "Human Data Ops Lead", "company": "Meta"},
    ]

    # Merge dedupes and caps at 5, keeping the most recent.
    second = _saved_string(
        id=2,
        save_exemplars=[
            {"title": "VP Business Operations", "company": "Surge AI"},  # dupe
            {"title": "T3", "company": "C3"},
            {"title": "T4", "company": "C4"},
            {"title": "T5", "company": "C5"},
            {"title": "T6", "company": "C6"},
        ],
    )
    memory = update_search_memory(memory, "proj", [second])
    entry = memory["families"]["startup_ops_transfer"]
    assert len(entry["save_exemplars"]) == 5
    assert {"title": "T6", "company": "C6"} in entry["save_exemplars"]
    # Dedupe: the repeated Surge exemplar never mints a second entry; the
    # cap keeps the NEWEST five, so with six uniques the oldest falls off.
    assert {"title": "Human Data Ops Lead", "company": "Meta"} in entry["save_exemplars"]
    assert {"title": "VP Business Operations", "company": "Surge AI"} not in entry[
        "save_exemplars"
    ]


def test_zero_save_string_does_not_touch_winning_boolean():
    memory = update_search_memory(None, "proj", [_saved_string()])
    dud = _saved_string(id=3, boolean='("something else")', saves=[], save_exemplars=[])
    memory = update_search_memory(memory, "proj", [dud])

    entry = memory["families"]["startup_ops_transfer"]
    assert entry["winning_boolean"] == '("chief of staff") AND ("human data" OR "annotation")'


def test_memory_summary_renders_proven_vein_with_exemplars():
    from shared.search_memory import format_search_memory_summary

    memory = update_search_memory(None, "proj", [_saved_string()])
    text = format_search_memory_summary(memory)

    assert "PROVEN VEIN: 2 saves over 2 pages reviewed" in text
    assert "VP Business Operations @ Surge AI" in text
    assert 'winning boolean: ("chief of staff") AND ("human data" OR "annotation")' in text

    # A family with zero saves renders no vein block.
    dud_memory = update_search_memory(
        None, "proj", [_saved_string(saves=[], save_exemplars=[], family_key="dud_family")]
    )
    assert "PROVEN VEIN" not in format_search_memory_summary(dud_memory)


def test_formation_memory_guidance_is_yield_aware():
    from linkedin.strategy import _build_strategy_user

    memory = update_search_memory(None, "proj", [_saved_string()])
    prompt = _build_strategy_user(
        _make_brief(), [], prior_run_data={"search_memory_summary": memory}
    )

    assert "PROVEN VEINS" in prompt
    assert "A vein earns a CHEAP PROBE as its opener" in prompt
    assert "never evidence the pool" in prompt
    assert "Avoid reopening exhausted families early" not in prompt
    # The rendered memory itself carries the pocket.
    assert "VP Business Operations @ Surge AI" in prompt


def test_search_string_round_trips_save_exemplars():
    s = _saved_string()
    restored = SearchString.from_dict(s.to_dict())
    assert restored.save_exemplars == s.save_exemplars


# ---------------------------------------------------------------------------
# Worked example_compounds seam (2026-07-04): a preflight-emitted
# example_compounds field must survive preflight_to_brief_json → _load_v2_brief
# → _build_strategy_system and land in the formation system prompt. This locks
# the WHOLE seam end to end — the preflight field is the only piece that was
# missing; loader hydration (brief_loader.py:400) and the renderer
# (strategy.py:_render_example_compounds_block) already existed.
# ---------------------------------------------------------------------------


def test_example_compounds_round_trip_into_formation_prompt():
    from linkedin.strategy import _build_strategy_system
    from shared.brief_loader import _load_v2_brief
    from shared.preflight_v2 import preflight_to_brief_json
    from tests.test_orchestrator_preflight import _valid_preflight_dict

    data = {
        **_valid_preflight_dict(),
        "example_compounds": [
            {
                "boolean": '("idempotency key" OR "ledger reconciliation")',
                "purpose": "Precision proof-of-practice",
                "novelty_bucket": "canonical",
            },
            {
                "boolean": '("payment orchestration" OR "payment rails") AND ("settlement" OR "chargeback")',
                "purpose": "Multi-angle recall net",
                "novelty_bucket": "edge_case",
            },
        ],
    }

    brief = _load_v2_brief(preflight_to_brief_json(data))
    system = _build_strategy_system(brief, has_kit=False)

    assert "Brief-supplied compound hints" in system
    # A distinctive term from an authored boolean survives into the prompt.
    assert "idempotency key" in system
    assert "Multi-angle recall net" in system


def test_no_example_compounds_omits_formation_block():
    from linkedin.strategy import _build_strategy_system
    from shared.brief_loader import _load_v2_brief
    from shared.preflight_v2 import preflight_to_brief_json
    from tests.test_orchestrator_preflight import _valid_preflight_dict

    brief = _load_v2_brief(preflight_to_brief_json(_valid_preflight_dict()))
    system = _build_strategy_system(brief, has_kit=False)

    assert "Brief-supplied compound hints" not in system


def test_opening_priority_ranks_from_preflight_born_mirrors():
    """RC3: with the mirrors populated (as preflight now emits them), the
    deterministic opening sort actually ranks — canonical strings bucket 2,
    edge strings bucket 0; a mirror-less brief stays neutral (the pre-RC3
    blindness this slice closes)."""
    from linkedin.strategy import _opening_priority

    brief = _make_brief(
        canonical_title_patterns=["strategic project lead"],
        canonical_company_patterns=["scale ai"],
        edge_case_patterns=["rater program manager"],
    )
    canonical_bucket, _ = _opening_priority(
        brief, '("Strategic Project Lead") AND ("delivery")', "direct-hit pool"
    )
    edge_bucket, _ = _opening_priority(
        brief, '("rater program manager") AND ("quality")', "adjacent pool"
    )
    assert canonical_bucket == 2
    assert edge_bucket == 0

    blind = _make_brief()
    assert _opening_priority(blind, '("Strategic Project Lead")', "")[0] == 1


def _discriminating_vocabulary_section(system: str) -> str:
    start = system.index("## Discriminating Vocabulary by Capability Area")
    end = system.index("\n## Permanent Filters", start)
    return system[start:end]


def test_discriminating_vocabulary_compat_section_uses_key_terms_byte_identically():
    from linkedin.strategy import _build_strategy_system

    brief = _make_brief(
        key_terms_by_area={
            "Payments": ["idempotency", "ledger reconciliation"],
        }
    )

    system = _build_strategy_system(brief, has_kit=False)

    assert _discriminating_vocabulary_section(system) == (
        "## Discriminating Vocabulary by Capability Area\n"
        "- Payments: idempotency, ledger reconciliation\n"
        "\n"
        "Use these terms as anchors for Type B precision strings. They are the "
        "specific technical vocabulary that distinguishes qualified candidates "
        "in each area.\n"
    )


def test_discriminating_vocabulary_prefers_candidate_register_terms():
    from linkedin.strategy import _build_strategy_system

    brief = _make_brief(
        key_terms_by_area={"Payments": ["idempotency"]},
        candidate_register_terms_by_area={
            "Payments": ["payment orchestration", "settlement systems"],
        },
    )

    system = _build_strategy_system(brief, has_kit=False)
    section = _discriminating_vocabulary_section(system)

    assert section == (
        "## Discriminating Vocabulary by Capability Area\n"
        "- Payments: payment orchestration, settlement systems\n"
        "\n"
        "Use these terms as candidate self-description anchors for Type B "
        "precision strings. They are the profile vocabulary qualified candidates "
        "plausibly write for each area.\n"
    )
    assert "idempotency" not in section


# ---------------------------------------------------------------------------
# De-prescribed formation prompt (2026-07-05,
# plans/formation-prompt-de-prescribed.md): the no-kit non-layered path
# renders the crafted minimal core; kit/layered paths keep the legacy
# builder; brief data and the JSON contract stay byte-identical to legacy.
# Locks live here — fixture-free, always run.
# ---------------------------------------------------------------------------


def test_formation_prompt_is_deprescribed_on_no_kit_path():
    from linkedin.strategy import _build_strategy_system

    system = _build_strategy_system(_make_brief(), has_kit=False)

    # The crafted core.
    assert (
        "You are a world-class, expert-level talent sourcer and talent researcher."
        in system
    )
    assert "Search the language candidates use about themselves" in system
    assert "Every clause must add information beyond its siblings." in system
    assert "The test for any required term: would essentially" in system
    assert "Enter the pool from different starting surfaces." in system
    assert "ampersand form with its and-form twin" in system
    assert "distinct hypothesis about the population it uniquely reaches" in system
    assert "design a portfolio of 15-30 LinkedIn Recruiter Boolean search strings" in system
    assert "open with probes" in system
    assert "warnings repair nothing" in system
    # Facet semantics + telemetry legend survive the cut, so the unchanged
    # contract's cross-references ("rules above", "the six listed above")
    # still resolve.
    assert "## Structured filters — the executable levers" in system
    assert "Never emit a location facet" in system
    assert "- sniper — distinctive title, large market; precision-led." in system
    # Rules above the data, data above the contract.
    assert system.index("Every clause must add information") < system.index("<brief>")
    assert (
        system.index("<brief>")
        < system.index("</brief>")
        < system.index("Return JSON with this structure")
    )
    # The accreted doctrine is gone from this path.
    assert "Stage 0" not in system
    assert "Mandatory Self-Review" not in system
    assert "### Signal Test" not in system
    assert "composing a PORTFOLIO of searches" not in system
    assert "senior sourcing strategist" not in system
    assert "Generate 15-30 search strings total." not in system
    # Kit mode still runs the legacy builder (dispatch lock).
    legacy_kit = _build_strategy_system(_make_brief(), has_kit=True)
    assert "composing a PORTFOLIO of searches" in legacy_kit
    assert "AND-gate high-signal concepts" not in legacy_kit


def test_deprescribed_prompt_keeps_data_and_contract_byte_identical_to_legacy():
    """Drift guard for the copied contract + relocated data sections: the
    de-prescribed render must carry the legacy render's brief-data slices
    verbatim and its JSON contract byte-for-byte."""
    from linkedin.strategy import (
        _build_strategy_system,
        _build_strategy_system_legacy,
    )

    brief = _make_brief(
        key_terms_by_area={"Payments": ["idempotency", "ledger reconciliation"]},
    )
    new = _build_strategy_system(brief, has_kit=False)
    old = _build_strategy_system_legacy(brief, has_kit=False)

    marker = "Return JSON with this structure"
    assert new[new.index(marker):] == old[old.index(marker):]

    for start, end in (
        ("Role: ", "\n## Minimum Bar"),
        ("## Minimum Bar", "\n## Archetypes"),
        ("## Archetypes", "\n## Discriminating"),
        ("## Discriminating", "\n## Permanent Filters"),
    ):
        piece = old[old.index(start):old.index(end)]
        assert piece and piece in new, (
            f"legacy data slice not verbatim in de-prescribed prompt: {start!r}"
        )


def test_layered_mode_keeps_family_contract_and_omits_shape_freedom():
    from linkedin.strategy import _build_strategy_system

    system = _build_strategy_system(_make_brief(), has_kit=False, use_layered_retrieval=True)

    # Universal principles still render in layered mode.
    assert "Maximum-Inclusion gate" in system
    assert "composing a PORTFOLIO of searches" in system
    # The structural family contract is retained (would contradict shape-freedom).
    assert "Every search family should be expressed as" in system
    assert (
        "No pre-built search kit is available. You will generate compound Boolean strings directly"
        in system
    )
    assert (
        "Your job: design targeted layered retrieval families and rendered search strings from the role requirements, using LinkedIn-compatible Boolean syntax."
        in system
    )
    assert (
        "### 1. DESIGN layered retrieval families\n"
        "This role requires an INTERSECTION of skills.\n"
        "Create retrieval families that AND-gate high-signal layers with domain/seniority qualifiers from the brief.\n\n"
        "You are composing a PORTFOLIO"
    ) in system
    assert (
        "- Use LinkedIn-compatible Boolean syntax: parenthetical groups joined by AND"
        in system
    )
    # ...so the shape-freedom sentence is branch-scoped OUT of layered mode.
    assert "There is no correct number of parentheticals" not in system
    assert "quoted phrases, AND, OR, NOT, and parentheses" not in system
    assert "broad-to-narrow" not in system


def test_adaptation_prompt_carries_portfolio_doctrine():
    system = _captured_adapt_system(_make_brief())

    assert "portfolio of angles, not one shape" in system
    assert "Maximum-Inclusion test to every required AND term" in system
    assert "broad-to-narrow" not in system
    # The anchor rule and bias-context sentence (prior locks) survive untouched.
    assert "Every new string must carry at least one anchor" in system
