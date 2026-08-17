"""Designer Slice 7 — recruiter annotation primitives.

Pins:

- ``ExcludedAssetStore`` is append-only with idempotent active-
  exclusion behavior (re-excluding an already-excluded asset is a
  no-op; re-excluding after revoke produces a new row).
- ``PrincipleFeedbackStore.record`` validates marker enum and raises
  on unknown markers.
- ``feedback_marker_distribution`` rolls up counts per principle
  for the Slice-9 reflection polish input.
- ``compute_re_eval_asset_set`` is a pure function that filters the
  original asset set by the recruiter-excluded URL set.
- (Slice 3.6) ``record_designer_principle_feedback`` mirrors a unified
  ``judgment_accuracy`` value to the canonical ``runtime_state.sqlite3``
  candidate row, with the per-principle detail in
  ``terminal_payload_json["principle_markers"]``. The Designer's
  three-value enum maps to the canonical five-value enum at
  :data:`DESIGNER_MARKER_TO_JUDGMENT_ACCURACY`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from designer.recruiter_annotations import (
    DESIGNER_MARKER_TO_JUDGMENT_ACCURACY,
    RECOGNIZED_FEEDBACK_MARKERS,
    ExcludedAssetStore,
    PrincipleFeedbackStore,
    compute_re_eval_asset_set,
    record_designer_principle_feedback,
)
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# ExcludedAssetStore
# ---------------------------------------------------------------------------


def test_exclude_persists_active_exclusion(tmp_path: Path) -> None:
    store = ExcludedAssetStore(tmp_path / "annotations.sqlite3")
    excluded = store.exclude(
        candidate_identity_key="behance:joe",
        asset_url="https://example.com/img1.jpg",
        reason="That's their old portfolio.",
    )
    assert excluded.candidate_identity_key == "behance:joe"
    assert excluded.asset_url == "https://example.com/img1.jpg"
    assert excluded.reason == "That's their old portfolio."
    assert excluded.revoked_at is None


def test_exclude_is_idempotent_on_active_exclusion(tmp_path: Path) -> None:
    """Excluding the same asset twice without revoking returns the
    existing row — no duplicate rows."""

    store = ExcludedAssetStore(tmp_path / "annotations.sqlite3")
    first = store.exclude(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    second = store.exclude(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    assert first.excluded_id == second.excluded_id

    actives = store.active_exclusions_for_candidate("behance:joe")
    assert len(actives) == 1


def test_revoke_marks_existing_exclusion_revoked_at(tmp_path: Path) -> None:
    store = ExcludedAssetStore(tmp_path / "annotations.sqlite3")
    store.exclude(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    revoked = store.revoke(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    assert revoked is True
    assert (
        store.active_exclusion(
            candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
        )
        is None
    )


def test_revoke_returns_false_when_no_active_exclusion(tmp_path: Path) -> None:
    store = ExcludedAssetStore(tmp_path / "annotations.sqlite3")
    assert (
        store.revoke(
            candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
        )
        is False
    )


def test_re_exclude_after_revoke_produces_new_row(tmp_path: Path) -> None:
    """Recruiter intent over time: each exclude → revoke → re-exclude
    cycle leaves a distinct row in the append-only log."""

    store = ExcludedAssetStore(tmp_path / "annotations.sqlite3")
    first = store.exclude(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    store.revoke(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    second = store.exclude(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    assert first.excluded_id != second.excluded_id


def test_active_exclusions_for_candidate_omits_revoked(tmp_path: Path) -> None:
    store = ExcludedAssetStore(tmp_path / "annotations.sqlite3")
    store.exclude(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    store.exclude(
        candidate_identity_key="behance:joe", asset_url="https://x.com/b.jpg"
    )
    store.revoke(
        candidate_identity_key="behance:joe", asset_url="https://x.com/a.jpg"
    )
    actives = store.active_exclusions_for_candidate("behance:joe")
    assert {ex.asset_url for ex in actives} == {"https://x.com/b.jpg"}


# ---------------------------------------------------------------------------
# PrincipleFeedbackStore
# ---------------------------------------------------------------------------


def test_recognized_markers_set() -> None:
    assert RECOGNIZED_FEEDBACK_MARKERS == frozenset(
        {"useful_guidance", "wrong_shallow", "off_rubric"}
    )


def test_record_persists_marker(tmp_path: Path) -> None:
    store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")
    marker = store.record(
        candidate_identity_key="behance:joe",
        principle_name="Visual hierarchy",
        marker="useful_guidance",
        note="Strong primary focal point read.",
    )
    assert marker.marker == "useful_guidance"
    assert marker.principle_name == "Visual hierarchy"
    assert marker.note == "Strong primary focal point read."


def test_record_rejects_unknown_marker(tmp_path: Path) -> None:
    store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")
    with pytest.raises(ValueError, match="Unknown feedback marker"):
        store.record(
            candidate_identity_key="behance:joe",
            principle_name="Visual hierarchy",
            marker="thumbs_up",  # not in the recognized set
        )


def test_markers_for_candidate_returns_history(tmp_path: Path) -> None:
    store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")
    store.record(
        candidate_identity_key="behance:joe",
        principle_name="Visual hierarchy",
        marker="useful_guidance",
    )
    store.record(
        candidate_identity_key="behance:joe",
        principle_name="Typographic refinement",
        marker="off_rubric",
    )
    history = store.markers_for_candidate("behance:joe")
    assert len(history) == 2
    assert {m.principle_name for m in history} == {
        "Visual hierarchy",
        "Typographic refinement",
    }


def test_feedback_marker_distribution_rolls_up_counts(tmp_path: Path) -> None:
    store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")
    # Visual hierarchy: 2× useful_guidance.
    store.record(
        candidate_identity_key="behance:a",
        principle_name="Visual hierarchy",
        marker="useful_guidance",
    )
    store.record(
        candidate_identity_key="behance:b",
        principle_name="Visual hierarchy",
        marker="useful_guidance",
    )
    # Typographic refinement: 1× off_rubric.
    store.record(
        candidate_identity_key="behance:a",
        principle_name="Typographic refinement",
        marker="off_rubric",
    )
    distribution = store.feedback_marker_distribution()
    assert distribution["Visual hierarchy"]["useful_guidance"] == 2
    assert distribution["Typographic refinement"]["off_rubric"] == 1


# ---------------------------------------------------------------------------
# compute_re_eval_asset_set
# ---------------------------------------------------------------------------


def test_compute_re_eval_asset_set_drops_excluded_urls() -> None:
    original = [
        (0, "https://x.com/a.jpg"),
        (1, "https://x.com/b.jpg"),
        (2, "https://x.com/c.jpg"),
        (3, "https://x.com/d.jpg"),
    ]
    excluded = {"https://x.com/b.jpg", "https://x.com/d.jpg"}
    out = compute_re_eval_asset_set(
        original_assets=original, excluded_asset_urls=excluded
    )
    assert out == [0, 2]


def test_compute_re_eval_asset_set_preserves_order() -> None:
    original = [
        (5, "https://x.com/e.jpg"),
        (6, "https://x.com/f.jpg"),
        (1, "https://x.com/a.jpg"),
    ]
    out = compute_re_eval_asset_set(
        original_assets=original, excluded_asset_urls=set()
    )
    assert out == [5, 6, 1]


def test_compute_re_eval_asset_set_empty_when_all_excluded() -> None:
    original = [(0, "https://x.com/a.jpg")]
    excluded = {"https://x.com/a.jpg"}
    out = compute_re_eval_asset_set(
        original_assets=original, excluded_asset_urls=excluded
    )
    assert out == []


# ---------------------------------------------------------------------------
# record_designer_principle_feedback (Slice 3.6 reconciliation bridge)
# ---------------------------------------------------------------------------


def _make_runtime_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _seed_designer_candidate_for_bridge(
    store: RuntimeStateStore,
    *,
    brief_id: str = "brief-d",
    identity_key: str = "behance:joe",
    capability_area: str | None = "Visual hierarchy",
    confidence: float | None = 0.82,
) -> int:
    """Walk a designer candidate to ``full_terminal`` with a V2-shape
    payload so the unified-rollup contract is exercised end-to-end."""

    run_id = store.start_run(
        source="designer",
        brief_id=brief_id,
        output_dir=str(store.db_path.parent),
        mode="fresh",
        resume_state={"brief_name": brief_id},
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="designer",
        brief_id=brief_id,
        identity_key=identity_key,
        display_name="Joe Designer",
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
            source="designer",
            brief_id=brief_id,
            identity_key=identity_key,
            new_state=new_state,
        )
    full_decision: dict[str, object] = {
        "decision": "SAVE",
        "rationale": "Strong portfolio.",
    }
    if confidence is not None:
        full_decision["confidence"] = confidence
    if capability_area is not None:
        full_decision["capability_area"] = capability_area
    store.set_candidate_state(
        run_id=run_id,
        source="designer",
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload={"full_decision": full_decision},
    )
    candidate = store.get_candidate(
        source="designer", brief_id=brief_id, identity_key=identity_key
    )
    assert candidate is not None
    return int(candidate["id"])


def test_designer_marker_mapping_covers_three_recognized_values() -> None:
    """The mapping must cover every Designer-recognized marker so the
    bridge can never raise a KeyError on a well-formed input. The
    inverse — five-value canonical enum keys NOT being reachable from
    Designer (overstated_depth / understated_depth) — is also pinned
    so a future schema-axis change fires the test before regressing
    production."""

    assert set(DESIGNER_MARKER_TO_JUDGMENT_ACCURACY.keys()) == RECOGNIZED_FEEDBACK_MARKERS
    assert DESIGNER_MARKER_TO_JUDGMENT_ACCURACY == {
        "useful_guidance": "useful",
        "wrong_shallow": "wrong",
        "off_rubric": "off_rubric",
    }


def test_bridge_writes_to_both_substrates(tmp_path: Path) -> None:
    """End-to-end: the per-state-dir log records the verbatim Designer
    marker (with original three-value nuance), and the canonical store
    gets the mapped five-value enum + the per-principle detail in
    ``terminal_payload_json``. Both writes execute on a single call."""

    runtime_store = _make_runtime_store(tmp_path)
    feedback_store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")
    candidate_id = _seed_designer_candidate_for_bridge(runtime_store)

    marker = record_designer_principle_feedback(
        runtime_state_store=runtime_store,
        principle_feedback_store=feedback_store,
        source="designer",
        brief_id="brief-d",
        identity_key="behance:joe",
        principle_name="Visual hierarchy",
        marker="wrong_shallow",
        note="Hierarchy reads as muddled, not clear.",
    )

    # Per-state-dir log carries the original Designer marker verbatim.
    history = feedback_store.markers_for_candidate("behance:joe")
    assert len(history) == 1
    assert history[0].marker == "wrong_shallow"
    assert history[0].principle_name == "Visual hierarchy"
    assert history[0].note == "Hierarchy reads as muddled, not clear."
    # Returned marker matches the persisted row (id + marked_at preserved).
    assert marker.marker_id == history[0].marker_id
    assert marker.marked_at == history[0].marked_at

    # Canonical store gets the unified five-value enum.
    with sqlite3.connect(str(runtime_store.db_path)) as conn:
        row = conn.execute(
            "SELECT judgment_accuracy, judgment_accuracy_at, terminal_payload_json "
            "FROM candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
    assert row[0] == "wrong"  # mapped from wrong_shallow
    assert row[1] is not None  # writer stamps a timestamp

    # Per-principle detail in terminal_payload_json — the full_decision
    # block (which the calibration aggregator reads for capability_area
    # and confidence) is preserved verbatim.
    payload = json.loads(row[2])
    assert payload["full_decision"]["capability_area"] == "Visual hierarchy"
    assert payload["full_decision"]["confidence"] == 0.82
    assert payload["principle_markers"] == [
        {
            "principle_name": "Visual hierarchy",
            "marker": "wrong_shallow",  # original Designer enum, not the mapped value
            "note": "Hierarchy reads as muddled, not clear.",
            "marked_at": history[0].marked_at,
        }
    ]


def test_bridge_writes_unified_rollup_readable_marker(tmp_path: Path) -> None:
    """End-to-end Slice 3.6 contract: a Designer feedback marker is
    visible to ``aggregate_calibration_markers`` via the unified
    ``judgment_accuracy`` column, with no Designer-specific code path
    in the aggregator."""

    from shared.runtime_state.calibration import aggregate_calibration_markers

    runtime_store = _make_runtime_store(tmp_path)
    feedback_store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")
    _seed_designer_candidate_for_bridge(
        runtime_store,
        brief_id="brief-rollup",
        identity_key="behance:roll",
        capability_area="Typographic refinement",
        confidence=0.88,
    )

    record_designer_principle_feedback(
        runtime_state_store=runtime_store,
        principle_feedback_store=feedback_store,
        source="designer",
        brief_id="brief-rollup",
        identity_key="behance:roll",
        principle_name="Typographic refinement",
        marker="off_rubric",
    )

    rollup = aggregate_calibration_markers(
        runtime_store.db_path, brief_id="brief-rollup"
    )
    assert rollup.total_markers == 1
    assert rollup.by_marker_value == {"off_rubric": 1}
    assert rollup.by_capability_area == {"Typographic refinement": 1}
    # ``off_rubric`` at confidence > 0.7 earns the high-confidence
    # weight bonus (calibration.py:127-135). Pin that the bridge
    # preserves the rollup's weighting math — the canonical column is
    # the only input the aggregator reads, so the mapped enum value
    # must place this in the bonus set.
    assert rollup.weighted_markers_by_area == {"Typographic refinement": 2}


def test_bridge_appends_multiple_markers_to_principle_markers_list(
    tmp_path: Path,
) -> None:
    """Recruiter feedback over time accumulates — multiple bridge calls
    on the same candidate produce a list, not a clobber. The latest
    column value wins (matches set_candidate_judgment_accuracy's
    last-write-wins semantics)."""

    runtime_store = _make_runtime_store(tmp_path)
    feedback_store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")
    candidate_id = _seed_designer_candidate_for_bridge(runtime_store)

    record_designer_principle_feedback(
        runtime_state_store=runtime_store,
        principle_feedback_store=feedback_store,
        source="designer",
        brief_id="brief-d",
        identity_key="behance:joe",
        principle_name="Visual hierarchy",
        marker="useful_guidance",
    )
    record_designer_principle_feedback(
        runtime_state_store=runtime_store,
        principle_feedback_store=feedback_store,
        source="designer",
        brief_id="brief-d",
        identity_key="behance:joe",
        principle_name="Color system coherence",
        marker="off_rubric",
    )

    with sqlite3.connect(str(runtime_store.db_path)) as conn:
        row = conn.execute(
            "SELECT judgment_accuracy, terminal_payload_json "
            "FROM candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
    payload = json.loads(row[1])
    assert [m["principle_name"] for m in payload["principle_markers"]] == [
        "Visual hierarchy",
        "Color system coherence",
    ]
    assert row[0] == "off_rubric"  # most-recent write wins on the column


def test_bridge_propagates_unknown_marker_validation_error(tmp_path: Path) -> None:
    """The :class:`PrincipleFeedbackStore` validation gate fires first,
    before the canonical store sees any write. A 422-mappable
    ``ValueError`` surfaces; the canonical row stays unchanged so the
    rollup never sees a half-written marker."""

    runtime_store = _make_runtime_store(tmp_path)
    feedback_store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")
    candidate_id = _seed_designer_candidate_for_bridge(runtime_store)

    with pytest.raises(ValueError, match="Unknown feedback marker"):
        record_designer_principle_feedback(
            runtime_state_store=runtime_store,
            principle_feedback_store=feedback_store,
            source="designer",
            brief_id="brief-d",
            identity_key="behance:joe",
            principle_name="Visual hierarchy",
            marker="thumbs_up",
        )

    with sqlite3.connect(str(runtime_store.db_path)) as conn:
        row = conn.execute(
            "SELECT judgment_accuracy, terminal_payload_json "
            "FROM candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
    assert row[0] is None  # canonical column never written
    assert "principle_markers" not in json.loads(row[1])  # payload untouched


def test_bridge_propagates_unknown_candidate_error(tmp_path: Path) -> None:
    """The canonical store gate raises on unknown candidate. The
    feedback log will have already accepted the row — the docstring's
    promise: per-state-dir is primary, canonical is the mirror, so
    divergence on canonical-store failure is recoverable but never
    silent."""

    runtime_store = _make_runtime_store(tmp_path)
    feedback_store = PrincipleFeedbackStore(tmp_path / "annotations.sqlite3")

    with pytest.raises(ValueError, match="candidate not found"):
        record_designer_principle_feedback(
            runtime_state_store=runtime_store,
            principle_feedback_store=feedback_store,
            source="designer",
            brief_id="brief-ghost",
            identity_key="behance:ghost",
            principle_name="Visual hierarchy",
            marker="useful_guidance",
        )

    # Per the docstring contract: per-state-dir log was written first;
    # canonical-store failure does NOT roll it back. Recovery is replay
    # of the log into the canonical store, not the inverse.
    history = feedback_store.markers_for_candidate("behance:ghost")
    assert len(history) == 1


def test_non_designer_run_unchanged(tmp_path: Path) -> None:
    """The plan's test-plan acceptance criterion: a non-Designer run
    is unchanged when the bridge isn't called. Pinned by exercising
    the existing LinkedIn calibration path (set_candidate_judgment_accuracy
    + aggregate_calibration_markers) and asserting the rollup matches
    the pre-Slice-3.6 shape — no ``principle_markers`` leak, no
    behavior shift."""

    from shared.runtime_state.calibration import aggregate_calibration_markers

    runtime_store = _make_runtime_store(tmp_path)
    # Walk a LinkedIn candidate the existing way (no Designer bridge call).
    run_id = runtime_store.start_run(
        source="linkedin",
        brief_id="brief-li",
        output_dir=str(runtime_store.db_path.parent),
        mode="fresh",
        resume_state={"brief_name": "brief-li"},
    )
    runtime_store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-li",
        identity_key="li-1",
        display_name="LI Pat",
        profile_url="https://example.test/li-1",
    )
    for new_state in (
        "snippet_extracted",
        "facial_started",
        "facial_terminal",
        "full_started",
    ):
        runtime_store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="brief-li",
            identity_key="li-1",
            new_state=new_state,
        )
    runtime_store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-li",
        identity_key="li-1",
        new_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload={
            "full_decision": {
                "decision": "SAVE",
                "rationale": "On-brief.",
                "confidence": 0.8,
                "capability_area": "Foundation Models Research",
            }
        },
    )
    candidate = runtime_store.get_candidate(
        source="linkedin", brief_id="brief-li", identity_key="li-1"
    )
    assert candidate is not None
    runtime_store.set_candidate_judgment_accuracy(int(candidate["id"]), "useful")

    rollup = aggregate_calibration_markers(
        runtime_store.db_path, brief_id="brief-li"
    )
    assert rollup.total_markers == 1
    assert rollup.by_marker_value == {"useful": 1}
    assert rollup.by_capability_area == {"Foundation Models Research": 1}

    # Crucially: the non-Designer payload has no ``principle_markers``
    # key. The bridge is the only write site that creates it; all
    # existing callers continue to write the historical shape.
    with sqlite3.connect(str(runtime_store.db_path)) as conn:
        row = conn.execute(
            "SELECT terminal_payload_json FROM candidates WHERE id = ?",
            (int(candidate["id"]),),
        ).fetchone()
    assert "principle_markers" not in json.loads(row[0])
