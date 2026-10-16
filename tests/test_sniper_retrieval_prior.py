"""CLO-221 S1: strategy labels survive validation and queue construction."""

from __future__ import annotations

import asyncio

from linkedin.adaptation_signal_state import AdaptationGateConfig
from linkedin.strategy import _build_strategy_system, _normalize_precision_classes
from shared.schemas import AdaptationResponse, ExecutionPlan, Progress, SearchString
from tests.test_lane_key_integrity import _make_brief
from tests.test_linkedin_pipeline import _make_pipeline


def _warning_codes(plan: ExecutionPlan) -> list[str]:
    return [warning["code"] for warning in plan.plan_warnings]


def test_strategy_contract_requests_sniper_precision_metadata_in_both_renders():
    for has_kit in (True, False):
        system = _build_strategy_system(_make_brief(), has_kit=has_kit)
        structured = system.index('  - "structured_filters"')
        precision = system.index('  - "precision_class"', structured)
        implication = system.index('  - "evidence_implication"', precision)
        coverage = system.index('- "coverage_gaps"', implication)

        assert structured < precision < implication < coverage
        assert '"sniper", "standard", or "recall"' in system[precision:implication]
        assert 'REQUIRED when precision_class is "sniper"' in system[implication:coverage]


def test_precision_class_normalization_warns_and_is_idempotent():
    plan = ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[
            {
                "precision_class": "SNIPER ",
                "evidence_implication": "Membership proves benchmark practice.",
            },
            {"precision_class": "laser"},
            {"precision_class": "sniper", "evidence_implication": "  "},
        ],
    )

    _normalize_precision_classes(plan)

    assert [item["precision_class"] for item in plan.generated_strings] == [
        "sniper",
        "",
        "standard",
    ]
    assert _warning_codes(plan) == [
        "precision_class_unknown",
        "sniper_downgraded",
    ]

    first_pass = plan.to_dict()
    _normalize_precision_classes(plan)
    assert plan.to_dict() == first_pass


def test_precision_class_normalization_absent_fields_adds_no_warning():
    plan = ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[{"boolean": '"benchmark"'}],
    )

    _normalize_precision_classes(plan)

    assert _warning_codes(plan) == []


def test_search_string_precision_metadata_round_trip_and_legacy_defaults():
    search_string = SearchString(
        id=1,
        name="benchmark",
        boolean='"benchmark"',
        precision_class="sniper",
        evidence_implication="Membership proves benchmark practice.",
    )

    payload = search_string.to_dict()
    assert payload["precision_class"] == "sniper"
    assert payload["evidence_implication"] == "Membership proves benchmark practice."
    assert SearchString.from_dict(payload) == search_string

    legacy = {"id": 2, "name": "legacy", "boolean": '"legacy"'}
    restored = SearchString.from_dict(legacy)
    assert restored.precision_class == ""
    assert restored.evidence_implication == ""


def test_initial_queue_carries_sniper_fields_and_preserves_string_types(tmp_path):
    pipeline = _make_pipeline(str(tmp_path))
    pipeline.state.planner._execution_plan = ExecutionPlan(
        strategy_rationale="test",
        generated_strings=[
            {
                "boolean": '"named benchmark"',
                "rationale": "specific benchmark",
                "family_key": "benchmark",
                "precision_class": "sniper",
                "evidence_implication": "Membership proves named benchmark practice.",
            }
        ],
        coverage_gaps=[
            {
                "suggested_boolean": '"adjacent practice"',
                "gap": "adjacent population",
                "family_key": "adjacent",
            }
        ],
    )

    compound, coverage = pipeline.state.planner._build_ordered_search_strings()

    assert (compound.string_type, compound.precision_class) == ("Precision", "sniper")
    assert compound.evidence_implication == "Membership proves named benchmark practice."
    assert (coverage.string_type, coverage.precision_class) == ("Recall", "")
    assert coverage.evidence_implication == ""


def test_lane_native_queue_defaults_precision_fields_without_changing_type(tmp_path):
    pipeline = _make_pipeline(str(tmp_path))
    record = {
        "lane_id": "benchmark_lane",
        "lane_name": "Benchmark lane",
        "lane_intent": "Find benchmark practitioners",
        "snapshot": {
            "acquisition_mode": "linkedin_hybrid",
            "query_payload": {
                "boolean": '"named benchmark"',
                "structured_filters": {"companies": ["Example"]},
            },
        },
    }

    search_string = pipeline.state.planner._lane_only_search_string(record, next_id=1, batch_num=1)

    assert search_string is not None
    assert search_string.string_type == "Precision"
    assert search_string.precision_class == ""
    assert search_string.evidence_implication == ""


def test_adaptive_strings_inherit_only_from_matching_sniper_family(tmp_path):
    pipeline = _make_pipeline(str(tmp_path))
    pipeline.state.planner._adaptation_gate_config = AdaptationGateConfig(0, 0, 0)
    parent = SearchString(
        id=1,
        name="parent",
        boolean='"parent benchmark"',
        status="done",
        block="Block A",
        family_key="benchmark",
        precision_class="sniper",
        evidence_implication="Membership proves named benchmark practice.",
    )
    progress = Progress(
        brief_name="test",
        strings=[
            parent,
            SearchString(id=2, name="queued", boolean='"queued"', family_key="other"),
        ],
    )
    response = AdaptationResponse(
        new_strings=[
            {
                "boolean": '"matching child"',
                "rationale": "matching child",
                "family_key": "benchmark",
            },
            {
                "boolean": '"unmatched child"',
                "rationale": "unmatched child",
                "family_key": "missing",
            },
            {
                "boolean": '"explicit child"',
                "rationale": "explicit child",
                "family_key": "benchmark",
                "precision_class": "standard",
                "evidence_implication": "",
            },
        ]
    )

    asyncio.run(
        pipeline.state.planner._run_block_adaptation(
            "Block A", [parent], progress, lambda *args, **kwargs: response
        )
    )

    adaptive = [item for item in progress.strings if item.string_type == "Adaptive"]
    assert [item.precision_class for item in adaptive] == ["sniper", "", "standard"]
    assert [item.evidence_implication for item in adaptive] == [
        "Membership proves named benchmark practice.",
        "",
        "",
    ]
