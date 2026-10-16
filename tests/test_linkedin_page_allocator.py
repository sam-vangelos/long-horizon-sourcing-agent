"""Contracts for the pure LinkedIn root-string page allocator."""

from __future__ import annotations

import math

import pytest

from linkedin import page_allocator
from linkedin.page_allocator import (
    AllocationAction,
    AllocatorArm,
    AllocatorPolicyError,
    ArmScore,
    PageObservation,
    allocate_page,
    challenger_clears_friction,
    pool_arm,
)


def _observation(
    *,
    root: int,
    page: int = 1,
    extracted: int = 10,
    slots: int = 10,
    full_expected: int | None = None,
    full_settled: int | None = None,
    priority: int = 0,
    standard: int = 0,
    outreach: int | None = None,
    break_reason: str = "",
    technical_interruption: bool = False,
    off_policy: bool = False,
    variant: str = "root",
) -> PageObservation:
    outreach = priority + standard if outreach is None else outreach
    full_expected = max(2, outreach) if full_expected is None else full_expected
    full_settled = full_expected if full_settled is None else full_settled
    return PageObservation(
        root_string_id=root,
        variant_id=variant,
        page=page,
        slots=slots,
        extracted=extracted,
        full_expected=full_expected,
        full_settled=full_settled,
        priority=priority,
        standard=standard,
        outreach=outreach,
        break_reason=break_reason,
        technical_interruption=technical_interruption,
        off_policy=off_policy,
    )


def _arm(
    root: int,
    queue_priority: int,
    *observations: PageObservation,
    block: str = "Compound Batch 1",
    active_valid_page_count: int | None = None,
    root_has_valid_probe: bool | None = None,
    legacy_unobserved_pages: int = 0,
    exhausted: bool = False,
    terminal: bool = False,
    variant: str = "root",
) -> AllocatorArm:
    matching = [
        observation
        for observation in observations
        if observation.root_string_id == root and observation.teaches_policy
    ]
    return AllocatorArm(
        root_string_id=root,
        block=block,
        queue_priority=queue_priority,
        active_variant_id=variant,
        observations=tuple(observations),
        active_valid_page_count=(
            sum(observation.variant_id == variant for observation in matching)
            if active_valid_page_count is None
            else active_valid_page_count
        ),
        root_has_valid_probe=(
            bool(matching) if root_has_valid_probe is None else root_has_valid_probe
        ),
        legacy_unobserved_pages=legacy_unobserved_pages,
        physically_exhausted=exhausted,
        terminal=terminal,
    )


def test_page_accepts_exact_eighty_percent_boundaries_and_counts_all_extracted():
    observation = _observation(
        root=1,
        slots=10,
        extracted=8,
        full_expected=5,
        full_settled=4,
        priority=1,
        standard=1,
    )

    assert observation.valid
    assert (observation.n, observation.p, observation.e) == (8, 1, 2)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"extracted": 0}, "empty_extraction"),
        ({"extracted": 7, "slots": 10}, "incomplete_extraction"),
        (
            {"full_expected": 5, "full_settled": 3},
            "incomplete_full_reviews",
        ),
        (
            {"priority": 1, "standard": 0, "outreach": 2},
            "tier_outreach_mismatch",
        ),
        (
                {
                    "full_expected": 2,
                    "full_settled": 0,
                    "priority": 1,
                    "outreach": 1,
                },
            "outreach_exceeds_full_settled",
        ),
        ({"technical_interruption": True}, "technical_interruption"),
        ({"priority": -1, "outreach": -1}, "negative_counter"),
    ],
)
def test_page_rejects_incomplete_or_incoherent_currency(overrides, reason):
    observation = _observation(root=1, **overrides)

    assert not observation.valid
    assert reason in observation.invalid_reasons


@pytest.mark.parametrize("break_reason", ["early_exit", "glance_reformulate"])
def test_intentional_partial_relaxes_only_extraction_completeness(break_reason):
    valid = _observation(
        root=1,
        slots=20,
        extracted=5,
        full_expected=5,
        full_settled=4,
        priority=1,
        break_reason=break_reason,
    )
    unsettled = _observation(
        root=1,
        slots=20,
        extracted=5,
        full_expected=5,
        full_settled=3,
        priority=1,
        break_reason=break_reason,
    )

    assert valid.valid
    assert not unsettled.valid
    assert "incomplete_full_reviews" in unsettled.invalid_reasons


def test_any_other_nonempty_break_reason_is_intrinsically_invalid():
    observation = _observation(root=1, break_reason="session_cap")

    assert not observation.valid
    assert "unsupported_break_reason" in observation.invalid_reasons


def test_off_policy_page_can_be_well_formed_but_never_teaches():
    observation = _observation(root=1, priority=1, off_policy=True)

    assert observation.valid
    assert not observation.teaches_policy


def test_observation_round_trip_ignores_derived_diagnostics():
    observation = _observation(
        root=7,
        page=3,
        priority=1,
        standard=1,
        break_reason="early_exit",
        variant="rewrite-1",
    )

    payload = observation.to_dict()

    assert payload["valid"] is True
    assert payload["invalid_reasons"] == []
    assert PageObservation.from_dict(payload) == observation
    assert not hasattr(page_allocator, "PendingPageSettlement")


def test_pool_uses_only_last_two_valid_on_policy_pages_of_active_variant():
    observations = (
        _observation(root=1, page=1, priority=1, variant="rewrite"),
        _observation(root=1, page=2, priority=8, variant="old"),
        _observation(
            root=1,
            page=3,
            extracted=7,
            slots=10,
            priority=7,
            variant="rewrite",
        ),
        _observation(root=1, page=4, priority=8, off_policy=True, variant="rewrite"),
        _observation(root=2, page=4, priority=9, variant="rewrite"),
        _observation(root=1, page=5, standard=1, variant="rewrite"),
        _observation(root=1, page=6, priority=1, standard=1, variant="rewrite"),
    )

    score = pool_arm(_arm(1, 0, *observations, variant="rewrite"))

    assert (score.n, score.priority, score.eligible) == (20, 1, 3)


def test_pool_uses_hard_coded_wilson_95_percent_upper_bounds():
    score = pool_arm(_arm(1, 0, _observation(root=1, priority=1, standard=2)))

    assert score.priority_upper == pytest.approx(0.4041563854975721, abs=1e-15)
    assert score.eligible_upper == pytest.approx(0.6032267800204347, abs=1e-15)


def test_opening_probe_uses_queue_order_and_defers_legacy_deep_roots():
    current = _arm(1, 0, _observation(root=1))
    legacy_deep = _arm(2, 1, legacy_unobserved_pages=7)
    later_clean = _arm(4, 3)
    earlier_clean = _arm(3, 2)

    verdict = allocate_page(
        current_root_id=1,
        arms=[current, legacy_deep, later_clean, earlier_clean],
    )

    assert verdict.action is AllocationAction.SWITCH
    assert verdict.selected_root_id == 3
    assert verdict.reason == "opening_probe"


def test_no_root_receives_a_second_valid_page_before_every_root_has_one():
    current = _arm(
        1,
        0,
        _observation(root=1, page=1),
        _observation(root=1, page=2),
    )
    untouched = _arm(2, 1)

    verdict = allocate_page(current_root_id=1, arms=[current, untouched])

    assert verdict.action is AllocationAction.SWITCH
    assert verdict.selected_root_id == 2
    assert verdict.reason == "opening_probe"


def test_invalid_and_off_policy_pages_do_not_complete_an_opening_probe():
    current = _arm(
        1,
        0,
        _observation(root=1, extracted=7, slots=10),
        _observation(root=1, page=2, off_policy=True),
    )
    sibling = _arm(2, 1)

    verdict = allocate_page(current_root_id=1, arms=[current, sibling])

    assert verdict.action is AllocationAction.CONTINUE
    assert verdict.selected_root_id == 1
    assert verdict.reason == "opening_probe"


def test_rewrite_keeps_root_probe_fairness_but_starts_a_fresh_active_window():
    rewritten = _arm(
        1,
        0,
        _observation(root=1, priority=1, variant="root"),
        variant="rewrite",
    )
    untouched = _arm(2, 1)

    probe = allocate_page(current_root_id=1, arms=[rewritten, untouched])

    assert rewritten.root_has_valid_probe
    assert rewritten.active_valid_page_count == 0
    assert pool_arm(rewritten).n == 0
    assert probe.selected_root_id == 2

    probed_sibling = _arm(2, 1, _observation(root=2))
    after_probe = allocate_page(current_root_id=1, arms=[rewritten, probed_sibling])
    assert after_probe.action is AllocationAction.CONTINUE
    assert after_probe.reason == "friction_hold"
    assert after_probe.floored_root_ids == ()


def test_rank_tie_and_exact_scores_keep_current_despite_queue_priority():
    current = _arm(1, 1, _observation(root=1, priority=1))
    challenger = _arm(2, 0, _observation(root=2, priority=1))

    verdict = allocate_page(current_root_id=1, arms=[challenger, current])

    assert verdict.ranked_root_ids == (2, 1)
    assert verdict.action is AllocationAction.CONTINUE
    assert verdict.selected_root_id == 1
    assert verdict.reason == "friction_hold"


def test_switch_friction_priority_boundary_is_strict_to_one_ulp():
    current = ArmScore(1, 1, 10, 1, 2, 0.40, 0.50)

    assert not challenger_clears_friction(
        current,
        ArmScore(2, 0, 10, 2, 5, 0.50, 0.90),
    )
    assert challenger_clears_friction(
        current,
        ArmScore(
            2,
            0,
            10,
            2,
            5,
            math.nextafter(0.50, math.inf),
            0.90,
        ),
    )


def test_switch_friction_eligible_boundary_is_strict_to_one_ulp():
    current = ArmScore(1, 1, 10, 1, 2, 0.40, 0.50)

    assert not challenger_clears_friction(
        current,
        ArmScore(2, 0, 10, 2, 5, 0.40, 0.70),
    )
    assert challenger_clears_friction(
        current,
        ArmScore(
            2,
            0,
            10,
            2,
            5,
            0.40,
            math.nextafter(0.70, math.inf),
        ),
    )


def test_switch_friction_priority_nearness_is_strict_at_lower_boundary():
    current = ArmScore(1, 1, 10, 1, 2, 0.40, 0.50)
    lower_boundary = current.priority_upper - 1 / current.n

    assert not challenger_clears_friction(
        current,
        ArmScore(2, 0, 10, 0, 5, lower_boundary, 0.90),
    )
    assert challenger_clears_friction(
        current,
        ArmScore(
            2,
            0,
            10,
            0,
            5,
            math.nextafter(lower_boundary, math.inf),
            0.90,
        ),
    )


def test_best_challenger_can_trigger_real_relative_switch():
    current = _arm(1, 0, _observation(root=1, extracted=100, slots=100))
    challenger = _arm(
        2,
        1,
        _observation(root=2, extracted=100, slots=100, priority=10),
    )

    verdict = allocate_page(current_root_id=1, arms=[current, challenger])

    assert verdict.action is AllocationAction.SWITCH
    assert verdict.selected_root_id == 2
    assert verdict.paused_root_ids == (1,)
    assert verdict.reason == "relative_underperformance"


def test_physical_exhaustion_switches_immediately_without_friction():
    current = _arm(1, 0, _observation(root=1, priority=10), exhausted=True)
    sibling = _arm(2, 1, _observation(root=2))

    verdict = allocate_page(current_root_id=1, arms=[current, sibling])

    assert verdict.action is AllocationAction.SWITCH
    assert verdict.selected_root_id == 2
    assert verdict.paused_root_ids == ()
    assert verdict.reason == "physical_exhaustion"


def test_physical_exhaustion_finishes_when_no_runnable_root_remains():
    current = _arm(1, 0, exhausted=True)
    exhausted = _arm(2, 1, exhausted=True)
    terminal = _arm(3, 2, terminal=True)

    verdict = allocate_page(current_root_id=1, arms=[current, exhausted, terminal])

    assert verdict.action is AllocationAction.FINISH
    assert verdict.selected_root_id is None
    assert verdict.reason == "physical_exhaustion"


def test_absolute_floor_waits_for_two_active_valid_pages_on_every_live_arm():
    current = _arm(
        1,
        0,
        _observation(root=1, page=1),
        _observation(root=1, page=2),
    )
    once = _arm(2, 1, _observation(root=2))

    verdict = allocate_page(current_root_id=1, arms=[current, once])

    assert verdict.floored_root_ids == ()


def test_floor_viability_uses_priority_or_exact_eight_percent_eligible():
    exact_eight_percent = _arm(
        1,
        0,
        _observation(root=1, page=1, extracted=25, slots=25, standard=4),
        _observation(root=1, page=2, extracted=25, slots=25),
    )
    priority_viable = _arm(
        2,
        1,
        _observation(root=2, page=1, extracted=100, slots=100, priority=1),
        _observation(root=2, page=2, extracted=100, slots=100),
    )
    below_floor = _arm(
        3,
        2,
        _observation(root=3, page=1, extracted=25, slots=25, standard=3),
        _observation(root=3, page=2, extracted=25, slots=25),
    )
    exhausted_unprobed = _arm(4, 3, exhausted=True)

    verdict = allocate_page(
        current_root_id=1,
        arms=[exact_eight_percent, priority_viable, below_floor, exhausted_unprobed],
    )

    assert verdict.floored_root_ids == (3,)
    assert 1 in verdict.ranked_root_ids
    assert 2 in verdict.ranked_root_ids
    assert 3 not in verdict.ranked_root_ids


def test_floor_finishes_all_nonviable_roots_together():
    weak_one = _arm(
        1,
        0,
        _observation(root=1, page=1),
        _observation(root=1, page=2),
    )
    weak_two = _arm(
        2,
        1,
        _observation(root=2, page=1),
        _observation(root=2, page=2),
    )

    verdict = allocate_page(current_root_id=1, arms=[weak_one, weak_two])

    assert verdict.action is AllocationAction.FLOOR
    assert verdict.selected_root_id is None
    assert verdict.floored_root_ids == (1, 2)
    assert verdict.reason == "allocation_floor"


def test_floored_current_switches_to_viable_root_without_pausing():
    weak = _arm(
        1,
        0,
        _observation(root=1, page=1),
        _observation(root=1, page=2),
    )
    viable = _arm(
        2,
        1,
        _observation(root=2, page=1, priority=1),
        _observation(root=2, page=2),
    )

    verdict = allocate_page(current_root_id=1, arms=[weak, viable])

    assert verdict.action is AllocationAction.SWITCH
    assert verdict.selected_root_id == 2
    assert verdict.paused_root_ids == ()
    assert verdict.floored_root_ids == (1,)
    assert verdict.reason == "allocation_floor"


def test_allocator_rejects_cross_block_scheduling():
    with pytest.raises(AllocatorPolicyError, match="contiguous block"):
        allocate_page(
            current_root_id=1,
            arms=[
                _arm(1, 0, block="Batch 1"),
                _arm(2, 1, block="Batch 2"),
            ],
        )


def test_allocator_rejects_duplicate_or_missing_current_roots():
    with pytest.raises(AllocatorPolicyError, match="duplicate roots"):
        allocate_page(current_root_id=1, arms=[_arm(1, 0), _arm(1, 1)])

    with pytest.raises(AllocatorPolicyError, match="outside allocator frontier"):
        allocate_page(current_root_id=3, arms=[_arm(1, 0), _arm(2, 1)])


def test_allocator_rejects_ambiguous_priorities_terminal_current_and_mismatched_observations():
    with pytest.raises(AllocatorPolicyError, match="duplicate queue priorities"):
        allocate_page(current_root_id=1, arms=[_arm(1, 0), _arm(2, 0)])

    with pytest.raises(AllocatorPolicyError, match="current root is terminal"):
        allocate_page(current_root_id=1, arms=[_arm(1, 0, terminal=True)])

    mismatched = _arm(1, 0, _observation(root=2))
    with pytest.raises(AllocatorPolicyError, match="mismatched observations"):
        allocate_page(current_root_id=1, arms=[mismatched])
