"""Tests mapping RetrievalDesign families to sourcing lane contracts."""

from linkedin.boolean_compiler import compile_constraint
from shared.retrieval_design import (
    RetrievalDesign,
    RetrievalLayerItem,
    retrieval_design_from_payload,
)
from shared.schemas import AdaptationResponse, ExecutionPlan
from shared.sourcing_lanes import (
    populate_execution_plan_lane_payloads,
    search_constraints_from_layer_items,
    search_hypothesis_from_edge_case,
    search_hypothesis_from_retrieval_family,
    search_slice_from_retrieval_family,
    sourcing_lane_from_retrieval_family,
    sourcing_lanes_from_retrieval_design,
)


def _sample_design_payload() -> dict:
    return {
        "families": [
            {
                "family_id": "delivery_builders",
                "label": "Delivery builders",
                "objective": "Broad entry plus builder proof.",
                "priority": 90,
                "enabled": True,
                "variants_to_emit": 1,
                "entry_signals": [
                    {
                        "item_id": "entry_1",
                        "label": "Delivery",
                        "terms": ["deployment engineer", "implementation engineer"],
                    }
                ],
                "capability_proxies": [
                    {
                        "item_id": "cap_1",
                        "label": "Capability",
                        "terms": ["workflow orchestration", "tool calling"],
                    }
                ],
                "reality_filters": [
                    {"item_id": "real_1", "label": "Reality", "terms": ["production", "deployed"]}
                ],
                "context_constraints": [],
                "anti_noise": [
                    {"item_id": "anti_1", "label": "Sales noise", "terms": ["account executive"]}
                ],
                "hypothesis_ids": ["hidden_pool_delivery"],
            }
        ],
        "shared_layers": {},
        "edge_case_hypotheses": [
            {
                "hypothesis_id": "hidden_pool_delivery",
                "label": "Hidden delivery pool",
                "hidden_cohort": "Delivery engineers who do not use FDE titles",
                "why_missed": "They use implementation language rather than canonical titles.",
                "entry_signal_variants": [],
                "capability_proxy_variants": [],
                "reality_filter_variants": [],
                "context_constraint_variants": [],
                "anti_noise_variants": [],
                "validation_rule": "Promote after repeated strong saves.",
                "noise_risks": ["title mismatch"],
            }
        ],
    }


def test_retrieval_family_maps_to_hypothesis_and_slice():
    design = retrieval_design_from_payload(_sample_design_payload())
    family = design.families[0]
    hypothesis = search_hypothesis_from_retrieval_family(family)
    search_slice = search_slice_from_retrieval_family(family)

    assert hypothesis.hypothesis_id == "delivery_builders"
    assert "workflow orchestration" in hypothesis.capability_signals
    assert search_slice.hypothesis_id == "delivery_builders"
    assert search_slice.slice_id == "delivery_builders_slice"
    dimensions = {constraint.dimension for constraint in search_slice.constraints}
    assert "entry_signal" in dimensions
    assert "capability" in dimensions
    entry_values = [
        value
        for constraint in search_slice.constraints
        if constraint.dimension == "entry_signal"
        for value in constraint.values
    ]
    assert "deployment engineer" in entry_values


def test_edge_case_hypothesis_maps_to_search_hypothesis():
    design = retrieval_design_from_payload(_sample_design_payload())
    edge = design.edge_case_hypotheses[0]
    hypothesis = search_hypothesis_from_edge_case(edge)
    assert hypothesis.hypothesis_id == "hidden_pool_delivery"
    assert hypothesis.target_archetype == edge.hidden_cohort
    assert hypothesis.why_this_pool_may_exist == edge.why_missed
    assert "title mismatch" in hypothesis.hidden_pool_risks


def test_sourcing_lane_from_retrieval_family_unifies_identity():
    design = retrieval_design_from_payload(_sample_design_payload())
    lane = sourcing_lane_from_retrieval_family(design.families[0])
    assert lane.lane_id == lane.hypothesis.hypothesis_id == lane.execution.lane_id
    assert lane.lane_id == "delivery_builders"
    assert lane.slice.hypothesis_id == lane.hypothesis.hypothesis_id


def test_sourcing_lanes_from_retrieval_design_counts_enabled_families():
    design = RetrievalDesign.from_dict(_sample_design_payload())
    lanes = sourcing_lanes_from_retrieval_design(design)
    assert len(lanes) == 1


def test_populate_execution_plan_lane_payloads_without_touching_generated_strings():
    generated = [
        {
            "boolean": '("deployment engineer") AND ("workflow orchestration")',
            "family_key": "delivery_builders",
        }
    ]
    plan = ExecutionPlan(
        strategy_rationale="test",
        retrieval_families=_sample_design_payload()["families"],
        generated_strings=list(generated),
    )
    populate_execution_plan_lane_payloads(plan)

    assert plan.search_hypotheses
    assert plan.search_slices
    assert plan.sourcing_lanes
    assert plan.generated_strings == generated
    assert plan.sourcing_lanes[0]["lane_id"] == "delivery_builders"


# ---------------------------------------------------------------------------
# build #2 — structured-filter producer (title/company) on layer items
# ---------------------------------------------------------------------------


def test_layer_item_structured_surface_emits_linkedin_filter():
    item = RetrievalLayerItem(
        item_id="es1",
        label="VP Engineering",
        terms=["VP Engineering"],
        structured_surface="linkedin_title_filter",
    )
    constraints = search_constraints_from_layer_items([item], dimension="entry_signal")
    assert len(constraints) == 1
    assert constraints[0].execution_surface == "linkedin_title_filter"
    # ...and that constraint compiles to a structured title control. The rest of the
    # chain (-> query_payload -> SearchString.structured_filters -> hybrid) is already
    # pinned in test_seam_strategy_execution.py.
    compiled = compile_constraint(constraints[0])
    assert compiled.structured_control["dimension"] == "title"
    assert compiled.structured_control["values"] == ["VP Engineering"]


def test_layer_item_default_surface_is_boolean_keyword():
    # No annotation -> byte-identical to pre-build-#2 behavior (backward compat).
    item = RetrievalLayerItem(item_id="es1", label="ml", terms=["ml"])
    assert item.structured_surface == ""
    constraints = search_constraints_from_layer_items([item], dimension="entry_signal")
    assert constraints[0].execution_surface == "boolean_keyword"


def test_anti_noise_never_emits_structured_surface():
    # A structured annotation on an anti_noise item is forced to soft_hint — negative
    # structured filtering is not wired and would be silently dropped by the
    # positive-only fold in lane_compiler.
    item = RetrievalLayerItem(
        item_id="an1",
        label="intern",
        terms=["intern"],
        structured_surface="linkedin_title_filter",
    )
    constraints = search_constraints_from_layer_items([item], dimension="anti_noise")
    assert constraints[0].execution_surface == "soft_hint"
    assert constraints[0].operator == "exclude"


def test_layer_item_structured_surface_round_trip_and_allowlist():
    # Allowed surface parses; an unknown/typo surface collapses to "" (boolean); the
    # field survives to_dict -> from_value.
    ok = RetrievalLayerItem.from_value(
        {"label": "Stripe", "terms": ["Stripe"], "structured_surface": "linkedin_company_filter"},
        fallback_prefix="es",
        index=0,
    )
    assert ok.structured_surface == "linkedin_company_filter"
    bad = RetrievalLayerItem.from_value(
        {"label": "x", "terms": ["x"], "structured_surface": "linkedin_seniority_filter"},
        fallback_prefix="es",
        index=1,
    )
    assert bad.structured_surface == ""  # not in the allow-list
    restored = RetrievalLayerItem.from_value(ok.to_dict(), fallback_prefix="es", index=0)
    assert restored.structured_surface == "linkedin_company_filter"


# ---------------------------------------------------------------------------
# P2.4 — one carrier: the renderer emits structured filters, never folds them
# ---------------------------------------------------------------------------


def _design_with_company_surface() -> dict:
    payload = _sample_design_payload()
    payload["families"][0]["context_constraints"] = [
        {
            "item_id": "ctx_companies",
            "label": "Target employers",
            "terms": [
                "Nubank",
                "Rappi",
                "Mercado Libre",
                "Bancolombia",
                "Habi",
                "Addi",
            ],
            "structured_surface": "linkedin_company_filter",
        }
    ]
    return payload


def test_render_family_variants_emits_company_surface_as_filters_not_keywords():
    """P2.4: a company-surface context constraint renders a boolean WITHOUT the
    company names and a structured_filters payload WITH the full list (no
    truncation to the keyword group's 2-4 term cap)."""

    from shared.retrieval_design import render_family_variants, retrieval_design_from_payload

    design = retrieval_design_from_payload(_design_with_company_surface())
    variants = render_family_variants(design.families[0], design)

    assert variants
    for variant in variants:
        boolean = variant["boolean"]
        for company in ("Nubank", "Rappi", "Mercado Libre", "Bancolombia", "Habi", "Addi"):
            assert company not in boolean
        assert variant["structured_filters"] == {
            "companies": [
                "Nubank",
                "Rappi",
                "Mercado Libre",
                "Bancolombia",
                "Habi",
                "Addi",
            ]
        }


def test_render_family_variants_without_structured_surfaces_is_byte_identical():
    """P2.4: families with no structured surfaces render exactly as before —
    no structured_filters key, same booleans."""

    from shared.retrieval_design import render_family_variants, retrieval_design_from_payload

    design = retrieval_design_from_payload(_sample_design_payload())
    variants = render_family_variants(design.families[0], design)

    assert variants
    for variant in variants:
        assert "structured_filters" not in variant
        assert '"deployment engineer"' in variant["boolean"]


def test_materialize_plan_keeps_model_strings_ahead_of_rendered():
    """P2.5: with prefer_rendered_strings=True, model-authored generated_strings
    keep queue priority; rendered variants merge AFTER (dedup unchanged)."""

    from linkedin.strategy import _materialize_retrieval_plan

    model_strings = [
        {"boolean": '("model authored one")', "rationale": "m1"},
        {"boolean": '("model authored two")', "rationale": "m2"},
    ]
    plan = ExecutionPlan(
        strategy_rationale="test",
        retrieval_families=_sample_design_payload()["families"],
        generated_strings=list(model_strings),
    )
    _materialize_retrieval_plan(plan, prefer_rendered_strings=True)

    booleans = [item["boolean"] for item in plan.generated_strings]
    assert booleans[0] == '("model authored one")'
    assert booleans[1] == '("model authored two")'
    # Rendered output merged after the model's strings.
    assert len(booleans) > 2
    assert any("deployment engineer" in b for b in booleans[2:])


def test_materialize_plan_families_only_uses_rendered_strings():
    """P2.5: when the model supplied only families, rendered strings are the
    plan — unchanged behavior."""

    from linkedin.strategy import _materialize_retrieval_plan

    plan = ExecutionPlan(
        strategy_rationale="test",
        retrieval_families=_sample_design_payload()["families"],
        generated_strings=[],
    )
    _materialize_retrieval_plan(plan, prefer_rendered_strings=True)

    assert plan.generated_strings
    assert all("deployment engineer" in item["boolean"] or item["boolean"] for item in plan.generated_strings)
    assert any("deployment engineer" in item["boolean"] for item in plan.generated_strings)


def test_materialize_adaptation_keeps_model_strings_ahead_of_rendered():
    """P11.3 verification of P2.5 on the adaptation path: with
    prefer_rendered_strings=True, the adaptation's model-authored new_strings
    keep queue priority; rendered variants merge AFTER (dedup unchanged).
    Mirrors test_materialize_plan_keeps_model_strings_ahead_of_rendered for
    _materialize_retrieval_adaptation — the sibling renderer-defers-to-model
    path that previously had no direct regression coverage."""

    from linkedin.strategy import _materialize_retrieval_adaptation

    model_strings = [
        {"boolean": '("model authored one")', "rationale": "m1"},
        {"boolean": '("model authored two")', "rationale": "m2"},
    ]
    adaptation = AdaptationResponse(
        new_strings=list(model_strings),
        new_retrieval_families=_sample_design_payload()["families"],
    )
    _materialize_retrieval_adaptation(adaptation, prefer_rendered_strings=True)

    booleans = [item["boolean"] for item in adaptation.new_strings]
    assert booleans[0] == '("model authored one")'
    assert booleans[1] == '("model authored two")'
    # Rendered output merged after the model's strings.
    assert len(booleans) > 2
    assert any("deployment engineer" in b for b in booleans[2:])


def test_materialize_adaptation_families_only_uses_rendered_strings():
    """P11.3 verification of P2.5 on the adaptation path: when the model
    supplied only new_retrieval_families (no new_strings), rendered strings
    become the adaptation's new_strings — unchanged behavior."""

    from linkedin.strategy import _materialize_retrieval_adaptation

    adaptation = AdaptationResponse(
        new_strings=[],
        new_retrieval_families=_sample_design_payload()["families"],
    )
    _materialize_retrieval_adaptation(adaptation, prefer_rendered_strings=True)

    assert adaptation.new_strings
    assert any("deployment engineer" in item["boolean"] for item in adaptation.new_strings)
