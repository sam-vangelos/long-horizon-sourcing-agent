"""Threshold layer between calibration aggregation and brief-patch proposal.

The aggregator is a read-only primitive over canonical per-source state; this
module applies editorial policy to its rollup. Keeping that policy in
``market_intelligence`` also keeps the translator, reflection integration, and
Designer rubric-refinement caller together.

Threshold posture:

- A capability area needs ``MIN_MARKERS_PER_AREA`` weighted markers (5).
- ``wrong`` and ``off_rubric`` markers above
  ``HIGH_CONFIDENCE_THRESHOLD`` (0.7) count twice in the aggregator's weighted
  total.
- ``SATURATION_MARKER_COUNT`` (20) marks an area saturated at 20 or more raw
  markers for telemetry; eligibility remains per-area and the per-cycle cap
  still applies.
- At most ``MAX_PATCHES_PER_CYCLE`` (5) areas surface; the translator may emit
  multiple patterns for one area. Rank by weighted count DESC, raw marker count
  DESC, and area name ASC. Losing the cap is reported as
  ``proposed=false``.

Emit one ``[market-intel] calibration.proposer:eligible`` line per area,
including rejected and unattributed areas, so threshold decisions remain
observable. The unattributed ``None`` bucket is logged as ``<unattributed>``
and never proposed because a brief patch needs a real capability area.

Brief-patch translation and reflection ingestion remain caller responsibilities.
Cross-source policy stays upstream. Threshold constants are explicit so later
calibration can tune them from collected telemetry.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from shared.runtime_state.calibration import (
    CalibrationRollup,
    HIGH_CONFIDENCE_THRESHOLD,  # noqa: F401  re-exported for callers
)


# Per-area eligibility floor. The confidence-weighted count must reach 5 before
# an area can produce a brief patch; raw marker count and weighted count are distinct.
MIN_MARKERS_PER_AREA: int = 5

# Saturation regime threshold. At or above 20 raw markers, an area is marked
# saturated for telemetry while the per-cycle cap remains the behavioral limit.
# The regime marker supports later volume-scaling calibration.
SATURATION_MARKER_COUNT: int = 20

# Per-reflection-cycle cap. Surface at most 5 areas; the translator may emit
# multiple patterns for one area. Rank by ``confidence_weighted_count`` descending,
# then the deterministic tie-breakers below.
MAX_PATCHES_PER_CYCLE: int = 5


@dataclass(frozen=True)
class EligibleArea:
    """Capability area cleared to produce a brief patch this cycle.

    ``confidence_weighted_count`` is the ranking key — passed through
    from the aggregator's ``weighted_markers_by_area``. ``saturated``
    flags the regime (``n_markers >= SATURATION_MARKER_COUNT``) so
    downstream consumers can render saturation-aware copy if useful;
    today it's primarily for telemetry / future tuning.
    """

    capability_area: str
    n_markers: int
    confidence_weighted_count: int
    saturated: bool


# Sentinel string for the aggregator's ``None`` capability_area bucket
# in telemetry. Quoted so a future grep on ``capability_area="..."``
# captures both real area names (which can contain spaces) and the
# unattributed bucket uniformly.
_UNATTRIBUTED_LABEL: str = "<unattributed>"


def _emit_stage(message: str) -> None:
    """Stderr line with the ``[market-intel]`` prefix.

    Mirrors ``market_intelligence.engine._emit_stage`` (engine.py:243-244)
    in shape and prefix so operators can grep one stream for both
    engine stage logs and proposer telemetry.
    """

    print(f"[market-intel] {message}", file=sys.stderr, flush=True)


def _format_capability_area(area: str | None) -> str:
    """Render a capability area for telemetry — quoted, with sentinel.

    Real area names can contain spaces (``"Foundation Models Research"``);
    the ``None`` bucket from the aggregator becomes ``"<unattributed>"``.
    Quoting keeps the ``key="value"`` shape parseable when a future log
    consumer wants structured extraction.
    """

    if area is None:
        return f'"{_UNATTRIBUTED_LABEL}"'
    return f'"{area}"'


def _emit_proposer_line(
    *,
    n_markers: int,
    capability_area: str | None,
    confidence_weighted_count: int,
    proposed: bool,
) -> None:
    """Emit one line in the proposer telemetry format.

    Format::

        calibration.proposer:eligible n_markers=<N> capability_area=<A>
        confidence_weighted_count=<C> proposed=<true|false>

    Emit the line for every area considered, whether eligible or rejected, so
    threshold fit remains observable.
    """

    _emit_stage(
        "calibration.proposer:eligible "
        f"n_markers={n_markers} "
        f"capability_area={_format_capability_area(capability_area)} "
        f"confidence_weighted_count={confidence_weighted_count} "
        f"proposed={'true' if proposed else 'false'}"
    )


def select_eligible_areas(rollup: CalibrationRollup) -> list[EligibleArea]:
    """Apply the per-area threshold and per-cycle cap, then emit telemetry.

    Return eligible areas sorted by weighted count DESC, raw marker count DESC,
    and area name ASC, capped at ``MAX_PATCHES_PER_CYCLE`` areas. The function is pure over the
    rollup apart from one telemetry line per area considered; the caller feeds
    the result to the brief-patch translator once per reflection cycle.

    Unattributed and below-threshold areas are logged with ``proposed=false``.
    Keeping those lines makes it possible to distinguish below-threshold,
    unattributed, and cycle-cap rejection without changing the returned set.
    """

    n_per_area = dict(rollup.by_capability_area)
    weighted_per_area = dict(rollup.weighted_markers_by_area)

    candidates: list[EligibleArea] = []
    rejected: list[tuple[str | None, int, int]] = []

    for area, n_markers in n_per_area.items():
        # If a future aggregator change drops the synchronized update of
        # ``weighted_markers_by_area``, fall back to the raw count rather
        # than KeyError out — keeps the threshold layer resilient if the
        # aggregator surface drifts. Today the two are written in lock-step
        # at calibration.py:315-326.
        weighted = weighted_per_area.get(area, n_markers)

        if area is None:
            rejected.append((area, n_markers, weighted))
            continue
        if weighted < MIN_MARKERS_PER_AREA:
            rejected.append((area, n_markers, weighted))
            continue

        candidates.append(
            EligibleArea(
                capability_area=area,
                n_markers=n_markers,
                confidence_weighted_count=weighted,
                saturated=n_markers >= SATURATION_MARKER_COUNT,
            )
        )

    # Sort by signal strength: weighted count desc, then raw n_markers
    # desc (areas with the same weighted count but more raw volume have
    # more underlying evidence), then area name asc for determinism on
    # tied keys. Determinism matters because the per-cycle cap is a
    # top-N selection — without a deterministic tie-break, the same
    # rollup could surface different patches across runs.
    candidates.sort(
        key=lambda area: (
            -area.confidence_weighted_count,
            -area.n_markers,
            area.capability_area,
        )
    )

    proposed = candidates[: MAX_PATCHES_PER_CYCLE]
    proposed_set = {area.capability_area for area in proposed}

    for area in candidates:
        is_proposed = area.capability_area in proposed_set
        _emit_proposer_line(
            n_markers=area.n_markers,
            capability_area=area.capability_area,
            confidence_weighted_count=area.confidence_weighted_count,
            proposed=is_proposed,
        )
    for area_name, n_markers, weighted in rejected:
        _emit_proposer_line(
            n_markers=n_markers,
            capability_area=area_name,
            confidence_weighted_count=weighted,
            proposed=False,
        )

    return proposed


__all__ = [
    "EligibleArea",
    "HIGH_CONFIDENCE_THRESHOLD",
    "MAX_PATCHES_PER_CYCLE",
    "MIN_MARKERS_PER_AREA",
    "SATURATION_MARKER_COUNT",
    "select_eligible_areas",
]
