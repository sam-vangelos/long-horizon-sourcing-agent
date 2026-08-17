"""Tests for the calibration aggregator (Slice 3.1 of multi-agent-execution-plan).

Pins the rollup math, the per-axis breakdowns, and the edge cases the
aggregator's downstream consumers (Slice 3.2 thresholding, 3.3 brief-
patch translation) will lean on:

- Single-marker happy path with full V2 ``full_decision`` payload.
- Multi-marker counts across capability_area / quartile / decision axes.
- Quartile boundary math (``[0,0.25)``, ``[0.25,0.5)`` …).
- ``confidence`` missing from the payload buckets to ``"unknown"``.
- Pre-V2 rows without ``capability_area`` collapse to the explicit
  ``None`` bucket (the legacy LinkedIn ``OpusDecision.path`` namespace
  intentionally does NOT leak into the V2 capability-area axis; see
  the rationale in ``shared/runtime_state/calibration.py`` module
  docstring).
- Source filter scopes correctly when the same brief carries both
  LinkedIn and GitHub markers in one DB.
- Zero-marker edge case (candidates exist, none marked).
- Missing DB file collapses to an empty rollup.
- Unknown ``judgment_accuracy`` values (legacy/imported rows) skip
  silently rather than poisoning the rollup.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from shared.runtime_state.calibration import (
    CalibrationRollupKey,
    QUARTILE_LABELS,
    QUARTILE_UNKNOWN,
    aggregate_calibration_markers,
    confidence_quartile,
)
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    db_path = tmp_path / "runtime_state.sqlite3"
    return RuntimeStateStore(db_path)


def _seed_saved_candidate(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
    capability_area: str | None,
    confidence: float | None,
    terminal_decision: str = "SAVE",
    judgment_accuracy: str | None = None,
) -> int:
    """Walk a candidate from discovered → full_terminal with a V2 payload.

    The full_decision shape mirrors the wire contract every module shares
    (see ``shared/runtime_state/read_models.py:782-806`` and
    ``linkedin/judgment_templates.py:FullEvaluationResult``). Returns the
    candidate row id so callers can stamp ``judgment_accuracy`` via the
    store's validated setter.
    """

    run_id = _ensure_run(store, source=source, brief_id=brief_id)

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=f"Candidate {identity_key}",
        profile_url=f"https://example.test/{identity_key}",
    )

    for new_state in (
        "snippet_extracted",
        "facial_started",
        "facial_terminal",
        "full_started",
    ):
        store.set_candidate_state(
            run_id=run_id,
            source=source,
            brief_id=brief_id,
            identity_key=identity_key,
            new_state=new_state,
        )

    full_decision: dict[str, object] = {
        "decision": terminal_decision,
        "rationale": f"rationale for {identity_key}",
    }
    if confidence is not None:
        full_decision["confidence"] = confidence
    if capability_area is not None:
        full_decision["capability_area"] = capability_area

    store.set_candidate_state(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_terminal",
        terminal_decision=terminal_decision,
        terminal_payload={"full_decision": full_decision},
    )

    candidate_id = _candidate_id(
        store, source=source, brief_id=brief_id, identity_key=identity_key
    )
    if judgment_accuracy is not None:
        store.set_candidate_judgment_accuracy(candidate_id, judgment_accuracy)
    return candidate_id


def _ensure_run(
    store: RuntimeStateStore, *, source: str, brief_id: str
) -> int:
    """Reuse a single per-source run for fixture brevity."""

    runs = store.list_runs(source=source, brief_id=brief_id)
    if runs:
        return int(runs[0]["id"])
    return store.start_run(
        source=source,
        brief_id=brief_id,
        output_dir=str(store.db_path.parent),
        mode="fresh",
        resume_state={"brief_name": brief_id},
    )


def _candidate_id(
    store: RuntimeStateStore,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
) -> int:
    candidate = store.get_candidate(
        source=source, brief_id=brief_id, identity_key=identity_key
    )
    assert candidate is not None, f"candidate not found: {identity_key}"
    return int(candidate["id"])


# ---------------------------------------------------------------------------
# Quartile math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, QUARTILE_UNKNOWN),
        (0.0, "q1"),
        (0.249999, "q1"),
        (0.25, "q2"),
        (0.49, "q2"),
        (0.5, "q3"),
        (0.74, "q3"),
        (0.75, "q4"),
        (1.0, "q4"),
    ],
)
def test_confidence_quartile_boundaries(value: float | None, expected: str) -> None:
    """Static absolute bands: ``q1=[0,0.25)``, ``q2=[0.25,0.5)``,
    ``q3=[0.5,0.75)``, ``q4=[0.75,1.0]``. Boundary cases land on the
    upper bucket per the closed-on-left convention. ``None`` → ``unknown``
    so callers don't have to special-case missing confidence."""

    assert confidence_quartile(value) == expected


def test_quartile_labels_constant_matches_implementation() -> None:
    """If a future slice changes the quartile shape, this fails first."""

    assert QUARTILE_LABELS == ("q1", "q2", "q3", "q4")
    assert QUARTILE_UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# Rollup math
# ---------------------------------------------------------------------------


def test_missing_db_file_returns_empty_rollup(tmp_path: Path) -> None:
    """Passive observer must not crash on a missing DB; collapses to
    an empty rollup the way other read helpers do."""

    rollup = aggregate_calibration_markers(
        tmp_path / "does_not_exist.sqlite3", brief_id="brief-1"
    )

    assert rollup.total_markers == 0
    assert rollup.counts == {}
    assert rollup.by_marker_value == {}
    assert rollup.by_capability_area == {}
    assert rollup.by_confidence_quartile == {}
    assert rollup.by_terminal_decision == {}
    assert rollup.weighted_markers_by_area == {}


def test_zero_markers_when_judgment_accuracy_unset(tmp_path: Path) -> None:
    """Edge case explicitly named in the slice spec: candidates exist,
    but none of them have ``judgment_accuracy`` set yet. The rollup is
    empty, not "all candidates" — ``judgment_accuracy IS NOT NULL`` is
    the filter."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-zero",
        identity_key="li-1",
        capability_area="Foundation Models Research",
        confidence=0.8,
    )
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-zero",
        identity_key="li-2",
        capability_area="Applied AI Engineering",
        confidence=0.4,
    )

    rollup = aggregate_calibration_markers(store.db_path, brief_id="brief-zero")

    assert rollup.total_markers == 0
    assert rollup.counts == {}


def test_single_marker_full_payload(tmp_path: Path) -> None:
    """One candidate, fully-populated V2 ``full_decision`` payload, one
    marker → exactly one key in the rollup; the per-axis breakdowns
    each carry a single entry."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        capability_area="Foundation Models Research",
        confidence=0.82,
        terminal_decision="SAVE",
        judgment_accuracy="useful",
    )

    rollup = aggregate_calibration_markers(store.db_path, brief_id="brief-1")

    expected_key = CalibrationRollupKey(
        capability_area="Foundation Models Research",
        marker_value="useful",
        confidence_quartile="q4",
        terminal_decision="SAVE",
    )
    assert rollup.brief_id == "brief-1"
    assert rollup.source is None
    assert rollup.total_markers == 1
    assert rollup.counts == {expected_key: 1}
    assert rollup.by_marker_value == {"useful": 1}
    assert rollup.by_capability_area == {"Foundation Models Research": 1}
    assert rollup.by_confidence_quartile == {"q4": 1}
    assert rollup.by_terminal_decision == {"SAVE": 1}
    # ``useful`` markers never get the high-confidence weight bonus, even
    # at q4 — only ``wrong`` / ``off_rubric`` do. Pin the contract.
    assert rollup.weighted_markers_by_area == {"Foundation Models Research": 1}


def test_multi_marker_breakdown_math(tmp_path: Path) -> None:
    """Multiple markers across capability_area, quartile, and decision
    axes. Asserts the canonical full-key ``counts`` matches the per-axis
    breakdowns for the same data — the breakdowns are a one-pass rollup
    of ``counts``, not a re-query."""

    store = _make_store(tmp_path)
    fixtures = [
        # (identity_key, capability_area, confidence, decision, marker)
        ("li-a", "Foundation Models Research", 0.85, "SAVE", "useful"),
        ("li-b", "Foundation Models Research", 0.80, "SAVE", "useful"),
        ("li-c", "Foundation Models Research", 0.40, "SAVE", "wrong"),
        ("li-d", "Applied AI Engineering", 0.78, "SAVE", "off_rubric"),
        ("li-e", "Applied AI Engineering", 0.60, "REJECT", "wrong"),
        ("li-f", "Applied AI Engineering", 0.10, "REJECT", "useful"),
    ]
    for identity_key, area, conf, decision, marker in fixtures:
        _seed_saved_candidate(
            store,
            source="linkedin",
            brief_id="brief-multi",
            identity_key=identity_key,
            capability_area=area,
            confidence=conf,
            terminal_decision=decision,
            judgment_accuracy=marker,
        )

    rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-multi"
    )

    assert rollup.total_markers == 6

    # Two q4 ``useful`` markers on Foundation Models Research SAVEs
    # collapse to one key with count 2 — the rollup is a true count, not
    # a list of events.
    foundation_useful_q4_save = CalibrationRollupKey(
        capability_area="Foundation Models Research",
        marker_value="useful",
        confidence_quartile="q4",
        terminal_decision="SAVE",
    )
    assert rollup.counts[foundation_useful_q4_save] == 2

    foundation_wrong_q2_save = CalibrationRollupKey(
        capability_area="Foundation Models Research",
        marker_value="wrong",
        confidence_quartile="q2",
        terminal_decision="SAVE",
    )
    assert rollup.counts[foundation_wrong_q2_save] == 1

    applied_off_rubric_q4_save = CalibrationRollupKey(
        capability_area="Applied AI Engineering",
        marker_value="off_rubric",
        confidence_quartile="q4",
        terminal_decision="SAVE",
    )
    assert rollup.counts[applied_off_rubric_q4_save] == 1

    applied_wrong_q3_reject = CalibrationRollupKey(
        capability_area="Applied AI Engineering",
        marker_value="wrong",
        confidence_quartile="q3",
        terminal_decision="REJECT",
    )
    assert rollup.counts[applied_wrong_q3_reject] == 1

    applied_useful_q1_reject = CalibrationRollupKey(
        capability_area="Applied AI Engineering",
        marker_value="useful",
        confidence_quartile="q1",
        terminal_decision="REJECT",
    )
    assert rollup.counts[applied_useful_q1_reject] == 1

    assert rollup.by_marker_value == {"useful": 3, "wrong": 2, "off_rubric": 1}
    assert rollup.by_capability_area == {
        "Foundation Models Research": 3,
        "Applied AI Engineering": 3,
    }
    assert rollup.by_confidence_quartile == {"q1": 1, "q2": 1, "q3": 1, "q4": 3}
    assert rollup.by_terminal_decision == {"SAVE": 4, "REJECT": 2}

    assert sum(rollup.by_marker_value.values()) == rollup.total_markers
    assert sum(rollup.by_capability_area.values()) == rollup.total_markers
    assert sum(rollup.by_confidence_quartile.values()) == rollup.total_markers
    assert sum(rollup.by_terminal_decision.values()) == rollup.total_markers


def test_missing_capability_area_buckets_to_none(tmp_path: Path) -> None:
    """Pre-V2 LinkedIn payloads (``OpusDecision.path``) and facial-only
    saves don't carry ``capability_area``; the aggregator surfaces those
    rows with a ``None`` capability_area bucket so the threshold layer
    (Slice 3.2) can decide whether to drop or report unattributed
    volume."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-noattr",
        identity_key="li-attr",
        capability_area="Applied AI Engineering",
        confidence=0.6,
        judgment_accuracy="useful",
    )
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-noattr",
        identity_key="li-noattr",
        capability_area=None,
        confidence=0.6,
        judgment_accuracy="wrong",
    )

    rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-noattr"
    )

    assert rollup.by_capability_area == {
        "Applied AI Engineering": 1,
        None: 1,
    }


def test_missing_confidence_buckets_to_unknown(tmp_path: Path) -> None:
    """Facial-only saves (no ``full_decision`` confidence) and any
    payload that doesn't carry a numeric confidence land in the
    ``unknown`` quartile bucket — distinct from ``q1`` so callers can
    tell "low confidence" apart from "no confidence reported"."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-noconf",
        identity_key="li-noconf",
        capability_area="Applied AI Engineering",
        confidence=None,
        judgment_accuracy="off_rubric",
    )

    rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-noconf"
    )

    assert rollup.by_confidence_quartile == {"unknown": 1}
    expected_key = CalibrationRollupKey(
        capability_area="Applied AI Engineering",
        marker_value="off_rubric",
        confidence_quartile="unknown",
        terminal_decision="SAVE",
    )
    assert rollup.counts == {expected_key: 1}


def test_legacy_linkedin_path_does_not_leak_into_capability_area(
    tmp_path: Path,
) -> None:
    """Pre-V2 LinkedIn rows that wrote ``OpusDecision.path`` (e.g.,
    ``"pedigree"``, ``"direct_experience"``) live in a different
    namespace from V2 capability-area names. Surfacing them as
    capability_area would feed the brief-patch translator (Slice 3.3)
    bad attributions. Asserts the aggregator drops the legacy ``path``
    field rather than aliasing it onto the V2 axis."""

    store = _make_store(tmp_path)
    run_id = _ensure_run(store, source="linkedin", brief_id="brief-legacy")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-legacy",
        identity_key="li-legacy",
        display_name="Legacy Pat",
        profile_url="https://example.test/li-legacy",
    )
    for new_state in (
        "snippet_extracted",
        "facial_started",
        "facial_terminal",
        "full_started",
    ):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="brief-legacy",
            identity_key="li-legacy",
            new_state=new_state,
        )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-legacy",
        identity_key="li-legacy",
        new_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload={
            "full_decision": {
                "decision": "SAVE",
                "rationale": "legacy path-encoded fixture",
                "confidence": 0.8,
                # Legacy LinkedIn path enum (NOT a V2 capability_area name).
                "path": "direct_experience",
            }
        },
    )
    store.set_candidate_judgment_accuracy(
        _candidate_id(
            store,
            source="linkedin",
            brief_id="brief-legacy",
            identity_key="li-legacy",
        ),
        "useful",
    )

    rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-legacy"
    )

    assert rollup.total_markers == 1
    assert rollup.by_capability_area == {None: 1}


def test_source_filter_scopes_correctly(tmp_path: Path) -> None:
    """A multi-source brief may carry markers across LinkedIn and GitHub
    candidates in the same DB. Source filter scopes the rollup."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-multi",
        identity_key="li-1",
        capability_area="Foundation Models Research",
        confidence=0.8,
        judgment_accuracy="useful",
    )
    _seed_saved_candidate(
        store,
        source="github",
        brief_id="brief-multi",
        identity_key="gh-1",
        capability_area="Applied AI Engineering",
        confidence=0.4,
        judgment_accuracy="wrong",
    )

    full_rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-multi"
    )
    li_rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-multi", source="linkedin"
    )
    gh_rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-multi", source="github"
    )

    assert full_rollup.total_markers == 2
    assert li_rollup.total_markers == 1
    assert li_rollup.source == "linkedin"
    assert li_rollup.by_marker_value == {"useful": 1}
    assert li_rollup.by_capability_area == {"Foundation Models Research": 1}
    assert gh_rollup.total_markers == 1
    assert gh_rollup.source == "github"
    assert gh_rollup.by_marker_value == {"wrong": 1}
    assert gh_rollup.by_capability_area == {"Applied AI Engineering": 1}


def test_unknown_marker_value_is_dropped(tmp_path: Path) -> None:
    """Imported legacy or out-of-band rows might carry a
    ``judgment_accuracy`` value not in the writer-validated set
    (``store.py:660-666``). The aggregator drops those rows defensively
    so one bad value can't poison the rollup. Bypasses the validated
    setter via direct UPDATE because that's the only way to reproduce
    the imported-legacy shape."""

    store = _make_store(tmp_path)
    candidate_id_known = _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-unknown",
        identity_key="li-known",
        capability_area="Applied AI Engineering",
        confidence=0.6,
        judgment_accuracy="useful",
    )
    candidate_id_unknown = _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-unknown",
        identity_key="li-unknown",
        capability_area="Applied AI Engineering",
        confidence=0.6,
    )

    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE candidates SET judgment_accuracy = ?, "
            "judgment_accuracy_at = ? WHERE id = ?",
            ("legacy_garbage", "2026-05-04T00:00:00+00:00", candidate_id_unknown),
        )

    rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-unknown"
    )

    assert rollup.total_markers == 1
    assert rollup.by_marker_value == {"useful": 1}
    assert candidate_id_known != candidate_id_unknown


def test_malformed_terminal_payload_collapses_axes_safely(
    tmp_path: Path,
) -> None:
    """A row whose ``terminal_payload_json`` is non-JSON, non-dict, or
    missing keys collapses to ``capability_area=None`` and
    ``confidence_quartile="unknown"`` rather than raising. Imported
    legacy data may carry oddly-shaped payloads; the aggregator must not
    crash on a single bad row."""

    store = _make_store(tmp_path)
    candidate_id = _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-malformed",
        identity_key="li-malformed",
        capability_area="Applied AI Engineering",
        confidence=0.6,
        judgment_accuracy="wrong",
    )

    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE candidates SET terminal_payload_json = ? WHERE id = ?",
            ("this is not valid json", candidate_id),
        )

    rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-malformed"
    )

    assert rollup.total_markers == 1
    expected_key = CalibrationRollupKey(
        capability_area=None,
        marker_value="wrong",
        confidence_quartile="unknown",
        terminal_decision="SAVE",
    )
    assert rollup.counts == {expected_key: 1}


def test_top_level_confidence_fallback_is_read(tmp_path: Path) -> None:
    """The wire helper at
    ``shared.runtime_state.read_models.extract_save_reason_and_confidence``
    falls back to a top-level ``confidence`` when ``full_decision`` isn't
    present (legacy LinkedIn shape per
    ``test_runtime_projections.py:305``). The aggregator inherits that
    fallback and must surface the quartile correctly so legacy rows
    don't all collapse to the ``unknown`` bucket."""

    store = _make_store(tmp_path)
    run_id = _ensure_run(store, source="linkedin", brief_id="brief-toplevel")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-toplevel",
        identity_key="li-toplevel",
        display_name="Toplevel Pat",
        profile_url="https://example.test/li-toplevel",
    )
    for new_state in (
        "snippet_extracted",
        "facial_started",
        "facial_terminal",
        "full_started",
    ):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="brief-toplevel",
            identity_key="li-toplevel",
            new_state=new_state,
        )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-toplevel",
        identity_key="li-toplevel",
        new_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload={"confidence": 0.62, "source_string_id": 1},
    )
    store.set_candidate_judgment_accuracy(
        _candidate_id(
            store,
            source="linkedin",
            brief_id="brief-toplevel",
            identity_key="li-toplevel",
        ),
        "useful",
    )

    rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-toplevel"
    )

    assert rollup.by_confidence_quartile == {"q3": 1}


def test_brief_filter_excludes_other_briefs(tmp_path: Path) -> None:
    """One DB can carry candidates from multiple briefs (a per-source
    state dir is brief-scoped today, but the aggregator's filter is
    explicit so a future shared-state-dir layout doesn't silently leak
    markers across briefs)."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-A",
        identity_key="li-a",
        capability_area="Area A",
        confidence=0.8,
        judgment_accuracy="useful",
    )
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-B",
        identity_key="li-b",
        capability_area="Area B",
        confidence=0.8,
        judgment_accuracy="wrong",
    )

    rollup_a = aggregate_calibration_markers(store.db_path, brief_id="brief-A")
    rollup_b = aggregate_calibration_markers(store.db_path, brief_id="brief-B")

    assert rollup_a.total_markers == 1
    assert rollup_a.by_marker_value == {"useful": 1}
    assert rollup_b.total_markers == 1
    assert rollup_b.by_marker_value == {"wrong": 1}


# ---------------------------------------------------------------------------
# Weighted-marker math (Slice 3.2 surface)
# ---------------------------------------------------------------------------


def test_high_confidence_wrong_marker_double_weighted(tmp_path: Path) -> None:
    """``wrong`` marker with confidence > 0.7 contributes 2 to the
    per-area weighted count (per execution-plan correction 3c)."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-w",
        identity_key="li-1",
        capability_area="Applied AI Engineering",
        confidence=0.85,
        judgment_accuracy="wrong",
    )

    rollup = aggregate_calibration_markers(store.db_path, brief_id="brief-w")

    assert rollup.by_capability_area == {"Applied AI Engineering": 1}
    assert rollup.weighted_markers_by_area == {"Applied AI Engineering": 2}


def test_high_confidence_off_rubric_marker_double_weighted(tmp_path: Path) -> None:
    """``off_rubric`` mirrors ``wrong`` for the bonus — Cloris was sure
    but the recruiter said the read missed the rubric."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-or",
        identity_key="li-1",
        capability_area="Applied AI Engineering",
        confidence=0.92,
        judgment_accuracy="off_rubric",
    )

    rollup = aggregate_calibration_markers(store.db_path, brief_id="brief-or")

    assert rollup.weighted_markers_by_area == {"Applied AI Engineering": 2}


def test_high_confidence_useful_marker_not_double_weighted(tmp_path: Path) -> None:
    """``useful`` is NOT in the bonus set even at high confidence —
    "Cloris was right and the recruiter agreed" is positive signal but
    not the kind of signal the threshold layer wants to escalate."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-u",
        identity_key="li-1",
        capability_area="Applied AI Engineering",
        confidence=0.99,
        judgment_accuracy="useful",
    )

    rollup = aggregate_calibration_markers(store.db_path, brief_id="brief-u")

    assert rollup.weighted_markers_by_area == {"Applied AI Engineering": 1}


def test_low_confidence_wrong_not_double_weighted(tmp_path: Path) -> None:
    """The bonus is gated on the confidence cut, not the marker alone.
    A ``wrong`` marker at confidence 0.4 contributes 1, not 2."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-lw",
        identity_key="li-1",
        capability_area="Applied AI Engineering",
        confidence=0.4,
        judgment_accuracy="wrong",
    )

    rollup = aggregate_calibration_markers(store.db_path, brief_id="brief-lw")

    assert rollup.weighted_markers_by_area == {"Applied AI Engineering": 1}


def test_high_confidence_threshold_is_strictly_greater_than(tmp_path: Path) -> None:
    """``> 0.7`` is strict — confidence == 0.7 exactly contributes 1, not 2.
    Pins the boundary so a future ``>=`` regression is caught."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-edge",
        identity_key="li-eq",
        capability_area="Applied AI Engineering",
        confidence=0.7,
        judgment_accuracy="wrong",
    )
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-edge",
        identity_key="li-just-over",
        capability_area="Applied AI Engineering",
        confidence=0.71,
        judgment_accuracy="wrong",
    )

    rollup = aggregate_calibration_markers(store.db_path, brief_id="brief-edge")

    # Two raw markers; one at 0.7 (weight 1), one at 0.71 (weight 2).
    assert rollup.by_capability_area == {"Applied AI Engineering": 2}
    assert rollup.weighted_markers_by_area == {"Applied AI Engineering": 3}


def test_unknown_confidence_does_not_qualify_for_bonus(tmp_path: Path) -> None:
    """No confidence reported (facial-only saves) → marker contributes
    1 even if it's ``wrong``. The bonus needs an explicit numeric
    confidence; ``None`` is not "implicitly high"."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-noconf-wrong",
        identity_key="li-1",
        capability_area="Applied AI Engineering",
        confidence=None,
        judgment_accuracy="wrong",
    )

    rollup = aggregate_calibration_markers(
        store.db_path, brief_id="brief-noconf-wrong"
    )

    assert rollup.weighted_markers_by_area == {"Applied AI Engineering": 1}


def test_weighted_markers_aggregate_per_area(tmp_path: Path) -> None:
    """Mix of bonus and non-bonus markers in the same area — weighted
    count sums per-row contributions."""

    store = _make_store(tmp_path)
    fixtures = [
        # (identity, marker, confidence, expected_weight)
        ("li-1", "wrong", 0.85, 2),         # high-conf wrong → 2
        ("li-2", "off_rubric", 0.80, 2),    # high-conf off_rubric → 2
        ("li-3", "useful", 0.90, 1),        # high-conf useful → 1
        ("li-4", "wrong", 0.50, 1),         # low-conf wrong → 1
        ("li-5", "overstated_depth", 0.95, 1),  # not in bonus set → 1
    ]
    for identity, marker, conf, _weight in fixtures:
        _seed_saved_candidate(
            store,
            source="linkedin",
            brief_id="brief-mix",
            identity_key=identity,
            capability_area="Applied AI Engineering",
            confidence=conf,
            judgment_accuracy=marker,
        )

    rollup = aggregate_calibration_markers(store.db_path, brief_id="brief-mix")

    # 2+2+1+1+1 == 7 weighted, 5 raw markers.
    assert rollup.by_capability_area == {"Applied AI Engineering": 5}
    assert rollup.weighted_markers_by_area == {"Applied AI Engineering": 7}


def test_aggregator_does_not_mutate_db(tmp_path: Path) -> None:
    """Read-only invariant: the aggregator opens via ``mode=ro`` so the
    kernel forbids writes. Pin the byte-equivalence of the DB after a
    rollup to catch any future regression where a caller accidentally
    writes via this path."""

    store = _make_store(tmp_path)
    _seed_saved_candidate(
        store,
        source="linkedin",
        brief_id="brief-ro",
        identity_key="li-ro",
        capability_area="Applied AI Engineering",
        confidence=0.8,
        judgment_accuracy="useful",
    )

    before = _row_snapshot(store.db_path)
    aggregate_calibration_markers(store.db_path, brief_id="brief-ro")
    after = _row_snapshot(store.db_path)

    assert before == after


def _row_snapshot(db_path: Path) -> str:
    """Compact JSON snapshot of the candidates table for byte-equivalence
    assertions. Uses a separate read-only connection so this helper
    can't accidentally mutate either."""

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in conn.execute(
                "SELECT id, brief_id, source, identity_key, "
                "current_lifecycle_state, terminal_decision, "
                "terminal_payload_json, judgment_accuracy, "
                "judgment_accuracy_at, last_seen_at "
                "FROM candidates ORDER BY id"
            ).fetchall()
        ]
    finally:
        conn.close()
    return json.dumps(rows, sort_keys=True)
