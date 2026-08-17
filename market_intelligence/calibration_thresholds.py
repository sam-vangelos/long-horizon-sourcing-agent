"""Threshold layer gating calibration aggregator → brief-patch proposer.

Phase 3.2 of the multi-agent execution plan
(``plans/multi-agent-execution-plan.md`` lines 840-882; correction 3c at
lines 57-62). Sits between the aggregator (Slice 3.1 at
``shared/runtime_state/calibration.py``) and the brief-patch translator
(Slice 3.4 at ``market_intelligence/calibration_to_brief.py``).

## Why this lives in ``market_intelligence/``, not ``shared/runtime_state/``

The slice card explicitly leaves the file location as a judgment call:
extend ``shared/runtime_state/calibration.py`` OR new
``market_intelligence/calibration_thresholds.py`` — pick whichever
reads cleaner. Three reasons to land it here:

1. The aggregator is a *runtime-state read primitive*: it opens the
   per-source SQLite ``mode=ro`` and walks ``judgment_accuracy`` rows.
   Its identity (per ``AGENTS.md`` runtime-state discipline) is "pure
   read over canonical state". Threshold logic is *editorial policy* —
   "given the rollup, decide which areas are eligible to surface
   patches this cycle". Mixing policy into the runtime-state primitive
   would erode that boundary.

2. Every downstream consumer lives in ``market_intelligence/``: the
   brief-patch translator (Slice 3.3 at
   ``market_intelligence/calibration_to_brief.py``), reflection
   integration (Slice 3.4 at ``market_intelligence/reflection.py``),
   and the rubric-refinement caller (Slice 3.5 invoking
   ``market_intelligence/design_market_intelligence.py:212``). The
   threshold layer is the upstream gate for those three; co-locating
   keeps the consumption tree in one directory.

3. The slice card explicitly cites the
   ``HALLUCINATION_OVERLAP_THRESHOLD = 0.30`` posture at
   ``market_intelligence/brief_polish.py:59`` as the substrate to
   mirror. That threshold is module-scope constant + per-call telemetry
   so post-trial tuning is a one-line tweak with data behind it. Same
   shape applies here. Brief polish lives in ``market_intelligence/``;
   matching the substrate's neighborhood matches the substrate's
   pattern.

The aggregator does carry one Slice-3.2-specific extension
(``weighted_markers_by_area`` field + ``HIGH_CONFIDENCE_THRESHOLD``
constant) — but only because the row walk already has raw confidence
in hand and surfacing the precomputed weighted count avoids a second
DB walk from this module. That extension is pure data, not policy.

## Threshold posture (best-guess starts; tune from telemetry)

Per the slice card lines 847-858:

- Per capability area: weighted count ≥ ``MIN_MARKERS_PER_AREA`` (5).
- Confidence weighting: ``> HIGH_CONFIDENCE_THRESHOLD`` (0.7) ``wrong``
  / ``off_rubric`` markers count 2x. Computed by the aggregator into
  ``CalibrationRollup.weighted_markers_by_area``.
- Saturation cap: above ``SATURATION_MARKER_COUNT`` (20) raw markers
  in an area, the proposer fires once per area per cycle (i.e., the
  threshold doesn't keep raising as volume grows). Today that's
  implicit because eligibility is per-area; the constant exists so
  telemetry distinguishes saturated from non-saturated regimes for
  post-trial tuning of a future scaling threshold.
- Per-cycle cap: ``MAX_PATCHES_PER_CYCLE`` (5), ranked by
  ``confidence_weighted_count`` descending (then by raw n_markers,
  then by area name for determinism). Areas above per-area threshold
  but losing the cycle cut are logged with ``proposed=false``.

## Telemetry

One stderr line per area considered, in the exact format the slice
card prescribes::

    calibration.proposer:eligible n_markers=<N> capability_area=<A>
    confidence_weighted_count=<C> proposed=<true|false>

Emitted via the ``[market-intel]`` prefix the
``market_intelligence/engine.py:243`` ``_emit_stage`` pattern uses, so
operators can grep one stream for both engine and proposer output.

Areas below the per-area threshold *also* log this line with
``proposed=false`` — without those, post-trial tuning has no signal
about whether the threshold is too tight. ``proposed=false`` covers
three rejection reasons (below threshold, unattributed area, lost the
cycle cut); operators distinguish them by computing
``confidence_weighted_count >= MIN_MARKERS_PER_AREA`` from the line
itself.

The unattributed-area bucket (``capability_area=None`` from the
aggregator — pre-V2 LinkedIn rows or facial-only saves) is logged as
``capability_area="<unattributed>"`` and never proposed: the brief-
patch translator (Slice 3.3) needs a real area name to attach a
``non_fit_pattern`` / ``depth_distinction`` / ``calibration_examples``
entry, and surfacing unattributed volume as a patch would feed
nonsense back into the brief.

## Out of scope for this slice

- Brief-patch translation (Slice 3.3) — this module gates *which* areas
  produce patches; the translator decides *what* the patch looks like.
- Reflection ingestion (Slice 3.4) — caller wires this in.
- Threshold tuning — the constants here are starting points. Once the
  ``calibration.proposer:eligible`` log accumulates real data (post-
  Phase 3 deployment), a dedicated calibration-tuning pass adjusts
  them.
- Cross-source merging — the aggregator and this layer both operate on
  a single per-source state SQLite (or a cross-source rollup the
  caller already merged). Cross-source policy lives upstream of this
  layer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from shared.runtime_state.calibration import (
    CalibrationRollup,
    HIGH_CONFIDENCE_THRESHOLD,  # noqa: F401  re-exported for callers
)


# Per-area eligibility floor. Confidence-weighted count must reach this
# value before the area is eligible to produce a brief patch this cycle.
# Slice card line 849: "≥5 markers (any kind) before eligibility."
MIN_MARKERS_PER_AREA: int = 5

# Saturation regime threshold. Above this raw-marker count, an area is
# considered saturated — today the only behavioral consequence is that
# the per-cycle cap still fires once per area (which it does for any
# eligible area), but logging the regime separately gives post-trial
# tuning the data needed to decide whether eligibility should scale
# with volume. Slice card line 854.
SATURATION_MARKER_COUNT: int = 20

# Per-reflection-cycle cap. Even if 12 areas hit the per-area
# threshold, surface at most this many patches; ranked by
# ``confidence_weighted_count`` descending. Slice card line 856.
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
    """One line in the spec-mandated proposer telemetry format.

    Format (slice card lines 859-863)::

        calibration.proposer:eligible n_markers=<N> capability_area=<A>
        confidence_weighted_count=<C> proposed=<true|false>

    Emitted for every area considered (eligible or not) so post-trial
    tuning has full visibility into the threshold's fit at trial volume.
    """

    _emit_stage(
        "calibration.proposer:eligible "
        f"n_markers={n_markers} "
        f"capability_area={_format_capability_area(capability_area)} "
        f"confidence_weighted_count={confidence_weighted_count} "
        f"proposed={'true' if proposed else 'false'}"
    )


def select_eligible_areas(rollup: CalibrationRollup) -> list[EligibleArea]:
    """Apply per-area threshold + per-cycle cap; emit telemetry.

    Returns the per-area subset eligible to produce patches this cycle,
    ranked by signal strength (``confidence_weighted_count`` desc, then
    raw n_markers desc, then capability_area asc for determinism), capped
    at ``MAX_PATCHES_PER_CYCLE``.

    Pure function over a rollup. The caller (Slice 3.4 at
    ``market_intelligence/reflection.py:526``) is responsible for calling
    this once per reflection cycle and feeding the result to the brief-
    patch translator (Slice 3.3).

    Telemetry is a side effect: one
    ``calibration.proposer:eligible`` stderr line per area considered.
    Emitting here (not at the caller) keeps the spec format pinned to
    the place that knows ``proposed=true|false``; if a caller wanted
    silent eligibility, it would have to suppress stderr — and we'd
    lose the post-trial tuning signal. Worth the coupling.
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
