"""Tests for the calibration threshold layer (Slice 3.2 of multi-agent-execution-plan).

Slice card lines 840-882. Pins:

- Per-area threshold (below / at / above) — slice card line 849.
- Confidence-weighted bonus passes through correctly from the
  aggregator's ``weighted_markers_by_area`` field (slice card line 851).
- Saturation regime (raw n_markers ≥ 20) is flagged but does not
  multiply proposals — proposer fires once per area per cycle (slice
  card line 854).
- Per-cycle cap of 5 patches, ranked by ``confidence_weighted_count``
  desc, with deterministic tie-breaking (slice card line 856).
- Telemetry format matches the slice card spec exactly (lines 859-863):
  ``calibration.proposer:eligible n_markers=<N> capability_area=<A>
  confidence_weighted_count=<C> proposed=<true|false>``.
- Areas considered but rejected (below threshold, unattributed, lost
  cycle cut) are *also* logged so post-trial tuning has full data.
- The ``None`` (unattributed) capability_area bucket from the
  aggregator is logged but never proposed.

The threshold layer is a pure function over a ``CalibrationRollup``;
these tests build rollups directly rather than walking the DB through
the aggregator. End-to-end aggregator coverage already lives in
``tests/test_calibration_aggregator.py``.
"""

from __future__ import annotations

import pytest

from market_intelligence.calibration_thresholds import (
    EligibleArea,
    HIGH_CONFIDENCE_THRESHOLD,
    MAX_PATCHES_PER_CYCLE,
    MIN_MARKERS_PER_AREA,
    SATURATION_MARKER_COUNT,
    select_eligible_areas,
)
from shared.runtime_state.calibration import (
    CalibrationRollup,
    CalibrationRollupKey,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _rollup_with_areas(
    *,
    by_capability_area: dict[str | None, int],
    weighted_markers_by_area: dict[str | None, int] | None = None,
) -> CalibrationRollup:
    """Build a minimal ``CalibrationRollup`` for threshold testing.

    Threshold logic only consumes ``by_capability_area`` and
    ``weighted_markers_by_area``; the other fields are populated with
    empty / placeholder values so the dataclass round-trips. If the
    threshold layer ever grows to consume more fields, this builder is
    where the test surface widens.

    ``weighted_markers_by_area`` defaults to a copy of
    ``by_capability_area`` (i.e., no high-confidence bonus markers); pass
    a separate dict when the test cares about the bonus path.
    """

    if weighted_markers_by_area is None:
        weighted_markers_by_area = dict(by_capability_area)
    return CalibrationRollup(
        brief_id="brief-test",
        source=None,
        total_markers=sum(by_capability_area.values()),
        counts={},
        by_marker_value={},
        by_capability_area=dict(by_capability_area),
        by_confidence_quartile={},
        by_terminal_decision={},
        weighted_markers_by_area=dict(weighted_markers_by_area),
    )


# ---------------------------------------------------------------------------
# Per-area threshold (below / at / above)
# ---------------------------------------------------------------------------


def test_below_threshold_not_eligible() -> None:
    """Weighted count < ``MIN_MARKERS_PER_AREA`` (5) → no proposals."""

    rollup = _rollup_with_areas(by_capability_area={"Area Below": 4})

    proposed = select_eligible_areas(rollup)

    assert proposed == []


def test_at_threshold_is_eligible() -> None:
    """The threshold is ≥, not >: weighted == 5 surfaces a proposal.

    Slice card line 849: '≥5 markers (any kind) before eligibility.'
    A common off-by-one would make this fail — pin it explicitly.
    """

    rollup = _rollup_with_areas(by_capability_area={"Area At": 5})

    proposed = select_eligible_areas(rollup)

    assert len(proposed) == 1
    assert proposed[0].capability_area == "Area At"
    assert proposed[0].confidence_weighted_count == 5
    assert proposed[0].n_markers == 5


def test_above_threshold_is_eligible() -> None:
    """Weighted count > MIN_MARKERS_PER_AREA → eligible."""

    rollup = _rollup_with_areas(by_capability_area={"Area Above": 12})

    proposed = select_eligible_areas(rollup)

    assert len(proposed) == 1
    assert proposed[0].capability_area == "Area Above"
    assert proposed[0].confidence_weighted_count == 12
    assert proposed[0].saturated is False


def test_high_confidence_bonus_lifts_below_floor_to_eligible() -> None:
    """Eligibility gates on the *weighted* count, not raw n_markers.

    3 high-confidence ``wrong`` markers contribute 6 to the weighted
    count (2x bonus) — that's above the floor even though the raw
    count is below. This is the entire point of the confidence-weighted
    rule (slice card line 851).
    """

    rollup = _rollup_with_areas(
        by_capability_area={"Bonus Lift": 3},
        weighted_markers_by_area={"Bonus Lift": 6},
    )

    proposed = select_eligible_areas(rollup)

    assert len(proposed) == 1
    assert proposed[0].capability_area == "Bonus Lift"
    assert proposed[0].n_markers == 3
    assert proposed[0].confidence_weighted_count == 6


def test_raw_count_alone_does_not_qualify_when_weighted_below_floor() -> None:
    """Inverse of the bonus-lift case: if the aggregator surfaces a
    weighted count below the floor (e.g., a future schema where the
    bonus rule changes), raw n_markers ≥ 5 is NOT enough.

    Defensive: keeps the threshold layer reading the weighted count,
    not the raw count, regardless of how the aggregator computes the
    weight in future revisions.
    """

    rollup = _rollup_with_areas(
        by_capability_area={"Area X": 6},
        weighted_markers_by_area={"Area X": 3},
    )

    proposed = select_eligible_areas(rollup)

    assert proposed == []


# ---------------------------------------------------------------------------
# Saturation regime
# ---------------------------------------------------------------------------


def test_saturation_above_20_flagged_but_one_proposal_per_area() -> None:
    """Slice card line 854: above ≥20 markers, the proposer still
    fires once per area per cycle. Saturation is a regime flag for
    telemetry / future tuning, not a multiplier.
    """

    rollup = _rollup_with_areas(
        by_capability_area={"Heavy": 50},
        weighted_markers_by_area={"Heavy": 80},
    )

    proposed = select_eligible_areas(rollup)

    assert len(proposed) == 1
    assert proposed[0].saturated is True
    assert proposed[0].n_markers == 50
    assert proposed[0].confidence_weighted_count == 80


def test_saturation_boundary_at_20_markers() -> None:
    """``n_markers >= 20`` is saturated; ``n_markers == 19`` is not.
    Pins the boundary so a future ``>`` regression is caught.
    """

    rollup_19 = _rollup_with_areas(by_capability_area={"Edge Sub": 19})
    rollup_20 = _rollup_with_areas(by_capability_area={"Edge At": 20})

    proposed_19 = select_eligible_areas(rollup_19)
    proposed_20 = select_eligible_areas(rollup_20)

    assert proposed_19[0].saturated is False
    assert proposed_20[0].saturated is True


def test_saturation_constant_value() -> None:
    """Pin the saturation cap value — any future change should be
    deliberate and flag this assertion as part of the diff.
    """

    assert SATURATION_MARKER_COUNT == 20


# ---------------------------------------------------------------------------
# Per-cycle cap (max 5 patches per cycle, ranked by signal strength)
# ---------------------------------------------------------------------------


def test_per_cycle_cap_caps_at_max_patches() -> None:
    """7 areas above threshold → only ``MAX_PATCHES_PER_CYCLE`` (5)
    proposed. Slice card line 856.
    """

    rollup = _rollup_with_areas(
        by_capability_area={f"Area {idx}": 6 for idx in range(7)}
    )

    proposed = select_eligible_areas(rollup)

    assert len(proposed) == MAX_PATCHES_PER_CYCLE


def test_per_cycle_cap_ranks_by_weighted_count_desc() -> None:
    """When more areas are eligible than the cap allows, the top N are
    selected by ``confidence_weighted_count`` desc — the proxy for
    'strongest signal' since the bonus rule already encodes
    high-confidence error density.
    """

    rollup = _rollup_with_areas(
        by_capability_area={
            "Top": 5,
            "Strong": 5,
            "Mid": 5,
            "Weakish": 5,
            "Weak": 5,
            "LosesCut1": 5,
            "LosesCut2": 5,
        },
        weighted_markers_by_area={
            "Top": 30,
            "Strong": 20,
            "Mid": 12,
            "Weakish": 9,
            "Weak": 7,
            "LosesCut1": 6,
            "LosesCut2": 5,
        },
    )

    proposed = select_eligible_areas(rollup)

    assert [area.capability_area for area in proposed] == [
        "Top",
        "Strong",
        "Mid",
        "Weakish",
        "Weak",
    ]
    assert "LosesCut1" not in {area.capability_area for area in proposed}
    assert "LosesCut2" not in {area.capability_area for area in proposed}


def test_ranking_tie_break_by_raw_n_markers() -> None:
    """Areas with the same weighted count but more raw n_markers have
    more underlying evidence — they win the tie. Determinism matters
    for stable proposals across reflection cycles.
    """

    rollup = _rollup_with_areas(
        by_capability_area={"Wide Evidence": 12, "Narrow Evidence": 6},
        weighted_markers_by_area={"Wide Evidence": 12, "Narrow Evidence": 12},
    )

    proposed = select_eligible_areas(rollup)

    assert [area.capability_area for area in proposed] == [
        "Wide Evidence",
        "Narrow Evidence",
    ]


def test_ranking_tie_break_by_capability_area_name() -> None:
    """When weighted count and n_markers tie, sort by area name asc.
    Pure determinism guard — without it, the same rollup could shuffle
    proposed areas across runs because dict iteration order in CPython
    is insertion-ordered but the aggregator's iteration order isn't a
    contract.
    """

    rollup = _rollup_with_areas(
        by_capability_area={"Charlie": 6, "Alpha": 6, "Bravo": 6}
    )

    proposed = select_eligible_areas(rollup)

    assert [area.capability_area for area in proposed] == [
        "Alpha",
        "Bravo",
        "Charlie",
    ]


# ---------------------------------------------------------------------------
# Unattributed (None) bucket
# ---------------------------------------------------------------------------


def test_unattributed_area_never_proposed() -> None:
    """Slice card lines 38-40: the aggregator surfaces pre-V2 LinkedIn
    rows + facial-only saves with ``capability_area=None``. The
    threshold layer never proposes these because the brief-patch
    translator (Slice 3.3) needs a real area name to attach a
    ``non_fit_pattern`` / ``depth_distinction`` / ``calibration_examples``
    entry.
    """

    rollup = _rollup_with_areas(by_capability_area={None: 50})

    proposed = select_eligible_areas(rollup)

    assert proposed == []


def test_unattributed_does_not_block_real_areas() -> None:
    """The unattributed bucket coexists with real areas in the same
    rollup; eligibility decisions on real areas are independent.
    """

    rollup = _rollup_with_areas(
        by_capability_area={None: 30, "Real Area": 8}
    )

    proposed = select_eligible_areas(rollup)

    assert len(proposed) == 1
    assert proposed[0].capability_area == "Real Area"


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


def test_no_areas_returns_empty() -> None:
    """Empty rollup → empty proposals, no exception."""

    rollup = _rollup_with_areas(by_capability_area={})

    proposed = select_eligible_areas(rollup)

    assert proposed == []


def test_returned_areas_are_immutable_dataclasses() -> None:
    """``EligibleArea`` is frozen so callers can't mutate proposals
    in-place after the threshold layer hands them off."""

    rollup = _rollup_with_areas(by_capability_area={"Area": 6})
    proposed = select_eligible_areas(rollup)

    assert len(proposed) == 1
    with pytest.raises(Exception):  # FrozenInstanceError under dataclasses
        proposed[0].capability_area = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Telemetry — exact spec format
# ---------------------------------------------------------------------------


def _proposer_lines(stderr_text: str) -> list[str]:
    """Pull the spec-format lines out of captured stderr.

    Each line starts with the ``[market-intel]`` prefix from
    ``_emit_stage`` followed by the spec event name. Other ``[market-intel]``
    lines (none today, but some may land later) are filtered out so
    these tests don't false-positive on unrelated logs.
    """

    return [
        line
        for line in stderr_text.splitlines()
        if "calibration.proposer:eligible" in line
    ]


def test_telemetry_emitted_for_eligible_proposed_area(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One area passes threshold AND fits the cycle cap → spec line
    with ``proposed=true`` emitted to stderr. Pin the exact format.
    """

    rollup = _rollup_with_areas(
        by_capability_area={"Foundation Models Research": 6},
        weighted_markers_by_area={"Foundation Models Research": 8},
    )

    select_eligible_areas(rollup)

    lines = _proposer_lines(capsys.readouterr().err)

    assert lines == [
        "[market-intel] calibration.proposer:eligible "
        "n_markers=6 "
        'capability_area="Foundation Models Research" '
        "confidence_weighted_count=8 "
        "proposed=true"
    ]


def test_telemetry_emitted_for_below_threshold_area(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Areas below threshold are still logged so post-trial tuning has
    visibility into the threshold's fit. Slice card line 859 framing:
    'so post-trial tuning has data'.
    """

    rollup = _rollup_with_areas(
        by_capability_area={"Below": 3},
        weighted_markers_by_area={"Below": 4},
    )

    select_eligible_areas(rollup)

    lines = _proposer_lines(capsys.readouterr().err)

    assert lines == [
        "[market-intel] calibration.proposer:eligible "
        "n_markers=3 "
        'capability_area="Below" '
        "confidence_weighted_count=4 "
        "proposed=false"
    ]


def test_telemetry_emitted_for_eligible_but_lost_cycle_cut(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Areas that pass per-area threshold but lose the ``MAX_PATCHES_PER_CYCLE``
    cut are logged with ``proposed=false`` — distinct from below-threshold
    rejection by their ``confidence_weighted_count >= MIN_MARKERS_PER_AREA``.
    """

    rollup = _rollup_with_areas(
        by_capability_area={
            "A": 5,
            "B": 5,
            "C": 5,
            "D": 5,
            "E": 5,
            "F-loses": 5,
        },
        weighted_markers_by_area={
            "A": 30,
            "B": 25,
            "C": 20,
            "D": 15,
            "E": 10,
            "F-loses": 5,
        },
    )

    select_eligible_areas(rollup)

    lines = _proposer_lines(capsys.readouterr().err)

    proposed_true = [line for line in lines if "proposed=true" in line]
    proposed_false = [line for line in lines if "proposed=false" in line]

    assert len(proposed_true) == MAX_PATCHES_PER_CYCLE
    assert len(proposed_false) == 1
    assert 'capability_area="F-loses"' in proposed_false[0]
    assert "confidence_weighted_count=5" in proposed_false[0]


def test_telemetry_emitted_for_unattributed_bucket(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ``None`` capability_area bucket is logged as
    ``capability_area="<unattributed>"`` so operators can see how much
    volume falls into the pre-V2 / facial-only-save bucket and decide
    whether the brief-patch translator should grow to handle it.
    """

    rollup = _rollup_with_areas(by_capability_area={None: 12})

    select_eligible_areas(rollup)

    lines = _proposer_lines(capsys.readouterr().err)

    assert lines == [
        "[market-intel] calibration.proposer:eligible "
        "n_markers=12 "
        'capability_area="<unattributed>" '
        "confidence_weighted_count=12 "
        "proposed=false"
    ]


def test_telemetry_emits_one_line_per_area(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No area double-logs: eligible areas emit once with their final
    ``proposed=`` decision; rejected areas emit once with
    ``proposed=false``.
    """

    rollup = _rollup_with_areas(
        by_capability_area={"Eligible": 6, "Below": 3, None: 8}
    )

    select_eligible_areas(rollup)

    lines = _proposer_lines(capsys.readouterr().err)

    assert len(lines) == 3


def test_telemetry_silent_on_empty_rollup(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Empty rollup → no proposer lines. The threshold layer doesn't
    fire telemetry for non-events.
    """

    rollup = _rollup_with_areas(by_capability_area={})

    select_eligible_areas(rollup)

    lines = _proposer_lines(capsys.readouterr().err)

    assert lines == []


# ---------------------------------------------------------------------------
# Constants & contract
# ---------------------------------------------------------------------------


def test_threshold_constants_match_slice_card() -> None:
    """Pin the slice card's stated tuning starts so a casual edit
    surfaces in the diff. Tuning is welcome — slice card line 847 calls
    these 'best-guess starts; tune from telemetry' — but the change
    should be visible in PR review.
    """

    assert MIN_MARKERS_PER_AREA == 5
    assert SATURATION_MARKER_COUNT == 20
    assert MAX_PATCHES_PER_CYCLE == 5
    assert HIGH_CONFIDENCE_THRESHOLD == 0.7


def test_eligible_area_dataclass_shape() -> None:
    """Pin the ``EligibleArea`` field shape so Slice 3.3 (translator)
    has a stable contract to consume.
    """

    area = EligibleArea(
        capability_area="Area",
        n_markers=10,
        confidence_weighted_count=12,
        saturated=False,
    )

    assert area.capability_area == "Area"
    assert area.n_markers == 10
    assert area.confidence_weighted_count == 12
    assert area.saturated is False


def test_rollup_with_no_weighted_field_fallback_to_raw() -> None:
    """If a future aggregator drops the weighted-by-area update (or a
    test fixture forgets to populate it), the threshold layer falls
    back to the raw count rather than KeyErroring out. The aggregator
    today writes both in lock-step at calibration.py:315-326; this is a
    resilience guard, not a sanctioned use case.

    Constructed manually so the weighted field is empty but the raw
    field has data.
    """

    rollup = CalibrationRollup(
        brief_id="brief",
        source=None,
        total_markers=6,
        counts={
            CalibrationRollupKey(
                capability_area="Fallback Area",
                marker_value="useful",
                confidence_quartile="q3",
                terminal_decision="SAVE",
            ): 6
        },
        by_marker_value={"useful": 6},
        by_capability_area={"Fallback Area": 6},
        by_confidence_quartile={"q3": 6},
        by_terminal_decision={"SAVE": 6},
        weighted_markers_by_area={},
    )

    proposed = select_eligible_areas(rollup)

    assert len(proposed) == 1
    assert proposed[0].capability_area == "Fallback Area"
    assert proposed[0].confidence_weighted_count == 6
