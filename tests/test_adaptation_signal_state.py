from __future__ import annotations

from unittest.mock import patch

import pytest

from linkedin.adaptation_signal_state import (
    AdaptationGateConfig,
    AdaptationGateDecision,
    AdaptationValidationError,
    MarketSignalPrior,
    ProfileIdAvailabilityContract,
    ProfileIdAvailabilityStatus,
    SearchSignalState,
    apply_adapted_string_firewall,
    coerce_market_signal_prior,
    evaluate_adaptation_gate,
)
from linkedin.strategy import adapt_after_block
from shared.brief_loader import Brief
from shared.schemas import BlockReport, SearchString


def _brief() -> Brief:
    return Brief(
        id="brief-test",
        role_title="AI Builder",
        role_description="Builds applied AI systems.",
        kit_url="",
        linkedin_project="",
        linkedin_project_id="",
        minimum_bar="Has shipped production systems.",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
    )


def _block_report() -> BlockReport:
    return BlockReport(
        block_name="Opening",
        strings_run=2,
        strings_with_saves=1,
        total_results=120,
        total_saves=2,
        zero_save_string_ids=[11],
        string_details=[
            {
                "string_id": 10,
                "boolean": '("research copilot")',
                "result_count": 80,
                "pages_reviewed": 2,
                "candidates": 12,
                "duplicates": 3,
                "saves": 2,
                "facial_yes": 6,
                "facial_no": 6,
                "family_key": "research_workflow",
                "novelty_bucket": "edge_case",
                "domain_lane": "asset_management",
                "saved_profiles": [
                    {"name": "A", "company": "Acme", "profile_id": "1"},
                    {"name": "B", "company": "Acme", "profile_id": "2"},
                ],
            },
            {
                "string_id": 11,
                "boolean": '("frontier lab")',
                "result_count": 40,
                "pages_reviewed": 1,
                "candidates": 4,
                "duplicates": 1,
                "saves": 0,
                "facial_yes": 1,
                "facial_no": 3,
                "family_key": "canonical",
                "novelty_bucket": "canonical",
                "domain_lane": "general",
            },
        ],
    )


def test_search_signal_state_from_block_report_computes_typed_rates() -> None:
    state = SearchSignalState.from_block_report(_block_report())
    payload = state.to_dict()

    assert payload["candidates_seen"] == 16
    assert payload["duplicates_seen"] == 4
    assert payload["triage_pass_rate"]["successes"] == 7
    assert payload["triage_pass_rate"]["total"] == 16
    assert (
        0.0
        < payload["triage_pass_rate"]["lower"]
        < payload["triage_pass_rate"]["upper"]
        < 1.0
    )
    assert payload["novelty_mix"] == {"edge_case_saves": 2, "canonical_saves": 0}
    assert payload["employer_concentration"] == 1.0
    assert payload["overlap"]["available"] is False
    assert payload["overlap"]["profile_id_sample_size"] == 2
    assert payload["overlap"]["profile_id_availability"]["status"] == "unverified"


def test_search_signal_state_aggregates_surface_receipts() -> None:
    """P2.2: block-report surface receipts reach the adaptation signal state.

    Facet requested/applied counts aggregate per dimension and a
    keyword-only fallback string is counted, so the adapting model sees
    actuator health instead of grading a broken actuator's output (FM12).
    """

    report = _block_report()
    report.string_details[0]["surface_receipt"] = {
        "requested_value_counts": {"companies": 4, "titles": 2},
        "applied_value_counts": {"companies": 4, "titles": 0},
        "fell_back_to_keyword": False,
    }
    report.string_details[1]["surface_receipt"] = {
        "requested_value_counts": {"companies": 2},
        "applied_value_counts": {"companies": 0},
        "fell_back_to_keyword": True,
    }

    state = SearchSignalState.from_block_report(report)
    payload = state.to_dict()

    assert payload["structured_actuator"] == {
        "facet_values_requested": {"companies": 6, "titles": 2},
        "facet_values_applied": {"companies": 4, "titles": 0},
        "strings_fell_back_to_keyword": 1,
    }


def test_search_signal_state_requires_verified_profile_id_contract_for_overlap() -> None:
    state = SearchSignalState.from_block_report(
        _block_report(),
        profile_id_contract=ProfileIdAvailabilityContract.verified(
            evidence={"source": "fixture", "stable_profile_id_seen": True},
            verified_at="2026-06-17T00:00:00Z",
        ),
    )
    payload = state.to_dict()

    assert payload["overlap"]["available"] is True
    assert payload["overlap"]["profile_id_sample_size"] == 2
    assert payload["overlap"]["profile_id_availability"]["status"] == (
        ProfileIdAvailabilityStatus.VERIFIED.value
    )


def test_profile_id_contract_rejects_non_iso_timestamp() -> None:
    with pytest.raises(AdaptationValidationError, match="ISO timestamp"):
        ProfileIdAvailabilityContract.verified(
            evidence={"source": "fixture", "stable_profile_id_seen": True},
            verified_at="not-a-timestamp",
        )


def test_adaptation_gate_uses_explicit_sufficiency_and_cooldown_config() -> None:
    state = SearchSignalState.from_block_report(_block_report())

    collect = evaluate_adaptation_gate(
        state,
        AdaptationGateConfig(min_strings=3, min_candidates_seen=20, min_results_seen=200),
    )
    adapt = evaluate_adaptation_gate(
        state,
        AdaptationGateConfig(min_strings=2, min_candidates_seen=10, min_results_seen=100),
    )
    cooldown = evaluate_adaptation_gate(
        state,
        AdaptationGateConfig(
            min_strings=2,
            min_candidates_seen=10,
            min_results_seen=100,
            cooldown_blocks_remaining=1,
        ),
    )

    assert collect.decision == AdaptationGateDecision.COLLECT_MORE_SIGNAL
    assert adapt.decision == AdaptationGateDecision.ADAPT
    assert cooldown.decision == AdaptationGateDecision.COOLDOWN


def test_adaptation_gate_blocks_autonomous_reset_without_product_approval() -> None:
    state = SearchSignalState.from_block_report(_block_report())
    result = evaluate_adaptation_gate(
        state,
        AdaptationGateConfig(
            min_strings=2,
            min_candidates_seen=10,
            min_results_seen=100,
            sprt_lower=0.9,
            allow_autonomous_reset=False,
        ),
    )

    assert result.decision == AdaptationGateDecision.RESET_BLOCKED
    assert result.reasons == ("autonomous reset requires product approval",)


def test_adaptation_gate_config_rejects_malformed_thresholds() -> None:
    with pytest.raises(AdaptationValidationError, match="min_strings"):
        AdaptationGateConfig(
            min_strings=True,
            min_candidates_seen=10,
            min_results_seen=100,
        )

    with pytest.raises(AdaptationValidationError, match="min_candidates_seen"):
        AdaptationGateConfig(
            min_strings=2,
            min_candidates_seen=-1,
            min_results_seen=100,
        )

    with pytest.raises(AdaptationValidationError, match="min_results_seen"):
        AdaptationGateConfig(
            min_strings=2,
            min_candidates_seen=10,
            min_results_seen="100",
        )

    with pytest.raises(AdaptationValidationError, match="cooldown_blocks_remaining"):
        AdaptationGateConfig(
            min_strings=2,
            min_candidates_seen=10,
            min_results_seen=100,
            cooldown_blocks_remaining=-1,
        )

    with pytest.raises(AdaptationValidationError, match="allow_autonomous_reset"):
        AdaptationGateConfig(
            min_strings=2,
            min_candidates_seen=10,
            min_results_seen=100,
            allow_autonomous_reset="yes",
        )

    for value in (-0.1, 1.1, True, "0.5", float("inf")):
        with pytest.raises(AdaptationValidationError, match="sprt_lower"):
            AdaptationGateConfig(
                min_strings=2,
                min_candidates_seen=10,
                min_results_seen=100,
                sprt_lower=value,
            )

    with pytest.raises(AdaptationValidationError, match="sprt_lower cannot exceed"):
        AdaptationGateConfig(
            min_strings=2,
            min_candidates_seen=10,
            min_results_seen=100,
            sprt_lower=0.8,
            sprt_upper=0.2,
        )


def test_market_signal_prior_coerces_advisory_context_to_typed_signals() -> None:
    prior = coerce_market_signal_prior(
        "## Market Intel Advisory Context\n"
        "- [exploit] Lean into the asset-management workflow lane.\n"
        "- [avoid] Deprioritize generic platform vocabulary."
    )

    assert isinstance(prior, MarketSignalPrior)
    assert prior.source == "market_intelligence.live_advisory"
    assert prior.context_hash.startswith("sha256:")
    assert [signal.signal_type for signal in prior.signals] == ["exploit", "avoid"]
    assert prior.signals[0].recommendation == (
        "Lean into the asset-management workflow lane."
    )


def test_market_signal_prior_accepts_structured_mapping_fields() -> None:
    prior = coerce_market_signal_prior(
        {
            "source": "market_intelligence",
            "context_hash": "sha256:abc123",
            "generated_at": "2026-06-17T00:00:00Z",
            "signals": [
                {
                    "signal_type": "exploit",
                    "recommendation": "Lean into the workflow lane.",
                    "evidence_ref_ids": ["market-intel:e1"],
                    "confidence": 0.8,
                }
            ],
        }
    )

    assert isinstance(prior, MarketSignalPrior)
    assert prior.source == "market_intelligence"
    assert prior.context_hash == "sha256:abc123"
    assert prior.generated_at == "2026-06-17T00:00:00Z"
    assert prior.to_dict()["signals"] == [
        {
            "signal_type": "exploit",
            "recommendation": "Lean into the workflow lane.",
            "evidence_ref_ids": ["market-intel:e1"],
            "confidence": 0.8,
        }
    ]


def test_market_signal_prior_mapping_rejects_stringified_fields() -> None:
    with pytest.raises(AdaptationValidationError, match="source must be a string"):
        coerce_market_signal_prior({"source": 123, "signals": []})

    with pytest.raises(
        AdaptationValidationError,
        match=r"signals\[0\] must be an object",
    ):
        coerce_market_signal_prior({"signals": ["not-object"]})

    with pytest.raises(AdaptationValidationError, match="signal_type must be a string"):
        coerce_market_signal_prior(
            {"signals": [{"signal_type": 123, "recommendation": "Lean in."}]}
        )

    with pytest.raises(AdaptationValidationError, match="recommendation must be a string"):
        coerce_market_signal_prior(
            {"signals": [{"signal_type": "exploit", "recommendation": ["Lean in."]}]}
        )

    with pytest.raises(
        AdaptationValidationError,
        match="evidence_ref_ids must contain strings",
    ):
        coerce_market_signal_prior(
            {
                "signals": [
                    {"recommendation": "Lean in.", "evidence_ref_ids": [123]}
                ]
            }
        )

    with pytest.raises(AdaptationValidationError, match="confidence must be numeric"):
        coerce_market_signal_prior(
            {"signals": [{"recommendation": "Lean in.", "confidence": "0.5"}]}
        )


def test_adapted_string_firewall_applies_m1c_normalizer_trace() -> None:
    new_strings = [{"boolean": '("reward model" OR "reward model development")'}]
    trace = apply_adapted_string_firewall(
        new_strings,
        enable_token_subset_pruning=True,
    )
    assert trace.passed is True
    assert trace.dropped == ()
    assert new_strings[0]["boolean"] == '("reward model")'
    assert new_strings[0]["boolean_normalization"]["changed"] is True


def test_adapted_string_firewall_drops_ubiquity_hits_per_string() -> None:
    """P5 (Wave 2): a ubiquity-gate hit drops THAT string and keeps the rest.

    The old batch-wide raise voided the entire adaptation decision (skips,
    reorders, healthy siblings) through the orchestrator's catch-all — the
    2026-06-18 regression class, reopened the moment the gate got a live
    term feed (correctness + contract lenses, Wave 2)."""
    new_strings = [
        {
            "boolean": '("Python") AND ("PyTorch")',
            "rationale": "too generic",
            "family_key": "generic",
        },
        {
            "boolean": '("reward model" OR "preference tuning")',
            "rationale": "healthy",
            "family_key": "healthy",
        },
    ]

    trace = apply_adapted_string_firewall(
        new_strings, ubiquitous_terms={"python", "pytorch"}
    )

    # The offender is gone from the batch IN PLACE; the healthy sibling stays.
    assert [item["family_key"] for item in new_strings] == ["healthy"]
    assert len(trace.dropped) == 1
    assert trace.dropped[0]["code"] == "ubiquitous_and_gate"
    assert trace.dropped[0]["family_key"] == "generic"
    assert trace.passed is True
    assert trace.to_dict()["dropped"][0]["code"] == "ubiquitous_and_gate"


def test_adapted_string_firewall_rejects_malformed_local_rule_inputs() -> None:
    # P5 (Wave 2): the locale/morphology expansion parameters were deleted as
    # dead plumbing (no producer ever existed); the surviving malformed-input
    # guards are structured_filters and ubiquitous_terms.
    with pytest.raises(AdaptationValidationError, match="structured_filters"):
        apply_adapted_string_firewall(
            [
                {
                    "boolean": '("Nubank")',
                    "structured_filters": ["Nubank"],
                }
            ]
        )


def test_adapt_after_block_uses_typed_signal_state() -> None:
    """Malformed optional metadata must never veto the adaptation payload.

    Regression for the 2026-06-18 live failure: a malformed typed action
    (``POPULATION_REFRAME`` with empty parameters) raised out of
    ``adapt_after_block`` and the orchestrator's catch-all discarded the
    ENTIRE adaptation (new_strings, skips, reorders). The typed-action
    vocabulary is deleted (zero consumers); a stray "actions" key in the
    model response is now ignored entirely.
    """

    captured: dict[str, str] = {}

    def fake_opus(
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True,
        max_tokens: int = 8192,
        usage_context: dict | None = None,
        model_name: str | None = None,
    ):
        captured["system"] = system_prompt
        return {
            "actions": [
                {
                    "type": "POPULATION_REFRAME",
                    "parameters": {},
                    "reason": "",
                }
            ],
            "new_strings": [
                {
                    "boolean": (
                        '("workflow automation" OR '
                        '"workflow automation platform")'
                    ),
                    "rationale": "Observed workflow saves.",
                    "family_key": "workflow",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "asset_management",
                }
            ],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
        }

    with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
        adaptation = adapt_after_block(
            _brief(),
            _block_report(),
            [SearchString(id=20, name="Queued", boolean='("queued")')],
        )

    assert "## Typed SearchSignalState" in captured["system"]
    assert "## Typed MarketSignalPrior" not in captured["system"]
    assert "## Block Aggregate Statistics" not in captured["system"]
    assert not hasattr(adaptation, "actions")
    assert adaptation.search_signal_state["triage_pass_rate"]["successes"] == 7
    assert adaptation.adapted_string_firewall["passed"] is True
    assert adaptation.new_strings[0]["boolean"] == '("workflow automation")'
    assert adaptation.new_strings[0]["boolean_normalization"]["changed"] is True
    assert adaptation.new_strings[0]["boolean_normalization"]["findings"][0]["code"] == (
        "token_subset_superstring_pruned"
    )


def test_adapt_after_block_structured_filters_produce_hybrid_string() -> None:
    """P2.3: the adaptation schema's structured_filters slot reaches execution.

    A new string carrying {"companies": [...]} flows through the firewall and
    lane_fields_from_work_unit_item into a hybrid SearchString; disallowed
    dimensions (locations) are dropped deterministically; a keyword-only
    response is unchanged.
    """

    from shared.sourcing_lanes import lane_fields_from_work_unit_item

    def fake_opus(
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True,
        max_tokens: int = 8192,
        usage_context: dict | None = None,
        model_name: str | None = None,
    ):
        return {
            "new_strings": [
                {
                    "boolean": '("reinforcement learning")',
                    "rationale": "Bound the canonical pool by employer.",
                    "family_key": "canonical_pool",
                    "novelty_bucket": "canonical",
                    "domain_lane": "general",
                    "structured_filters": {
                        "companies": ["Nubank", "Rappi"],
                        "locations": ["Colombia"],
                    },
                },
                {
                    "boolean": '("agent evals")',
                    "rationale": "Keyword-only expansion.",
                    "family_key": "evals",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "general",
                },
            ],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
        }

    with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
        adaptation = adapt_after_block(
            _brief(),
            _block_report(),
            [SearchString(id=20, name="Queued", boolean='("queued")')],
        )

    hybrid_item = adaptation.new_strings[0]
    # Locations dropped (never a mid-run lever); companies kept.
    assert hybrid_item["structured_filters"] == {"companies": ["Nubank", "Rappi"]}

    # The consumer path the orchestrator uses produces a hybrid SearchString.
    lane_fields = lane_fields_from_work_unit_item(hybrid_item)
    ss = SearchString(id=99, name="Adaptive", boolean=hybrid_item["boolean"], **lane_fields)
    assert ss.structured_filters.get("companies") == ["Nubank", "Rappi"]
    assert ss.acquisition_mode == "linkedin_hybrid"

    # Keyword-only new string is untouched — no structured_filters key appears.
    keyword_item = adaptation.new_strings[1]
    assert "structured_filters" not in keyword_item
    keyword_fields = lane_fields_from_work_unit_item(keyword_item)
    assert not keyword_fields.get("structured_filters")


def test_adapt_after_block_renders_market_signal_as_typed_prior() -> None:
    captured: dict[str, str] = {}

    def fake_opus(
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True,
        max_tokens: int = 8192,
        usage_context: dict | None = None,
        model_name: str | None = None,
    ):
        captured["system"] = system_prompt
        return {
            "new_strings": [],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
        }

    with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
        adaptation = adapt_after_block(
            _brief(),
            _block_report(),
            [SearchString(id=20, name="Queued", boolean='("queued")')],
            market_intel_advisory_context=(
                "## Market Intel Advisory Context\n"
                "- [exploit] Lean into the workflow lane."
            ),
        )

    assert "## Typed MarketSignalPrior" in captured["system"]
    assert "## Market Intel Advisory Context" not in captured["system"]
    assert '"signal_type": "exploit"' in captured["system"]
    assert adaptation.market_signal_prior["signals"][0]["signal_type"] == "exploit"


def test_adapt_after_block_accepts_explicit_no_change_as_valid_decision() -> None:
    """P11.1: an explicit "no_change": true is a valid decision, not a
    failure — deterministic code enforces it even if the model also sent
    other adaptations alongside it (the decline wins, nothing leaks through)."""

    def fake_opus(
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True,
        max_tokens: int = 8192,
        usage_context: dict | None = None,
        model_name: str | None = None,
    ):
        return {
            "no_change": True,
            # These should all be discarded by the decline, not applied.
            "new_strings": [{"boolean": '("should not appear")', "rationale": "r"}],
            "skip_remaining": [{"string_id": 20, "reason": "should not apply"}],
            "reorder": [{"string_id": 20, "move_to": "next", "reason": "should not apply"}],
            "noise_updates": [{"term": "x", "status": "confirmed_noise", "note": "n"}],
            "pivot_to_architecture": "dragnet",
            "pivot_rationale": "should not apply",
        }

    with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
        adaptation = adapt_after_block(
            _brief(),
            _block_report(),
            [SearchString(id=20, name="Queued", boolean='("queued")')],
        )

    assert adaptation.no_change is True
    assert adaptation.new_strings == []
    assert adaptation.skip_remaining == []
    assert adaptation.reorder == []
    assert adaptation.noise_updates == []
    assert adaptation.pivot_to_architecture == ""
    assert adaptation.pivot_rationale == ""


def test_adapt_after_block_clamps_next_checkpoint_after_within_bounds() -> None:
    """P11.2: next_checkpoint_after is clamped to [2, 8]; both the raw
    request and the applied value are preserved for logging."""

    def make_fake_opus(value):
        def fake_opus(
            system_prompt: str,
            user_prompt: str,
            expect_json: bool = True,
            max_tokens: int = 8192,
            usage_context: dict | None = None,
            model_name: str | None = None,
        ):
            return {"new_strings": [], "next_checkpoint_after": value}

        return fake_opus

    with patch("linkedin.strategy.opus_llm", side_effect=make_fake_opus(20)):
        high = adapt_after_block(
            _brief(), _block_report(), [SearchString(id=20, name="Queued", boolean='("queued")')]
        )
    assert high.next_checkpoint_after == 8
    assert high.next_checkpoint_after_requested == 20

    with patch("linkedin.strategy.opus_llm", side_effect=make_fake_opus(1)):
        low = adapt_after_block(
            _brief(), _block_report(), [SearchString(id=20, name="Queued", boolean='("queued")')]
        )
    assert low.next_checkpoint_after == 2
    assert low.next_checkpoint_after_requested == 1

    with patch("linkedin.strategy.opus_llm", side_effect=make_fake_opus(5)):
        in_range = adapt_after_block(
            _brief(), _block_report(), [SearchString(id=20, name="Queued", boolean='("queued")')]
        )
    assert in_range.next_checkpoint_after == 5
    assert in_range.next_checkpoint_after_requested == 5


def test_adapt_after_block_default_response_shape_is_byte_identical_to_pre_p11() -> None:
    """EXIT GATE (spec sec 12 P11): a model that never uses no_change or
    next_checkpoint_after parses to the same shape as pre-P11 behavior — the
    new levers are present but inert (False / None)."""

    def fake_opus(
        system_prompt: str,
        user_prompt: str,
        expect_json: bool = True,
        max_tokens: int = 8192,
        usage_context: dict | None = None,
        model_name: str | None = None,
    ):
        return {
            "new_strings": [
                {
                    "boolean": '("workflow automation")',
                    "rationale": "Observed workflow saves.",
                    "family_key": "workflow",
                    "novelty_bucket": "edge_case",
                    "domain_lane": "asset_management",
                }
            ],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
        }

    with patch("linkedin.strategy.opus_llm", side_effect=fake_opus):
        adaptation = adapt_after_block(
            _brief(),
            _block_report(),
            [SearchString(id=20, name="Queued", boolean='("queued")')],
        )

    assert adaptation.no_change is False
    assert adaptation.next_checkpoint_after is None
    assert adaptation.next_checkpoint_after_requested is None
    assert [s["boolean"] for s in adaptation.new_strings] == ['("workflow automation")']
    assert adaptation.skip_remaining == []
    assert adaptation.reorder == []
    assert adaptation.noise_updates == []
    assert adaptation.pivot_to_architecture == ""
    assert adaptation.pivot_rationale == ""

