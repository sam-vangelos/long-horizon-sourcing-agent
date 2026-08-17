"""Tests for ``RuntimeStateStore.record_candidate_principle_marker``.

Slice 3.6 of ``plans/multi-agent-execution-plan.md``. Pins the
two-write-in-one-transaction shape: ``judgment_accuracy`` column gets
the unified five-value enum, ``terminal_payload_json`` gets the
per-principle nuance appended under ``principle_markers``.

Generic store primitive — caller (Designer today; future modules may
reuse) owns the per-principle metadata shape. Tests pin invariants the
calibration aggregator at ``shared/runtime_state/calibration.py`` leans
on:

- The column accepts the same five-value set as
  :meth:`set_candidate_judgment_accuracy` (store.py:660-666); anything
  outside that set raises ``ValueError`` and never writes.
- ``terminal_payload_json`` is parsed, the marker is appended to a
  ``principle_markers`` list, and existing keys (e.g.,
  ``full_decision``) are preserved verbatim — the aggregator's read of
  ``full_decision.capability_area`` must keep working post-write.
- Multiple calls append in chronological order (recruiter feedback over
  time is signal — Slice 9 reflection polish reads the history).
- Unknown candidates raise without partial writes.
- ``judgment_accuracy_at`` and ``last_seen_at`` move forward together.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _seed_designer_candidate(
    store: RuntimeStateStore,
    *,
    source: str = "designer",
    brief_id: str = "brief-d",
    identity_key: str = "behance:joe",
    capability_area: str | None = "Visual hierarchy",
    confidence: float | None = 0.82,
    terminal_decision: str = "SAVE",
) -> int:
    """Walk a candidate to ``full_terminal`` with a V2 ``full_decision``
    payload. Mirrors ``tests/test_calibration_aggregator.py``'s seeder
    so the rollup-readable contract is exercised on the same shape."""

    runs = store.list_runs(source=source, brief_id=brief_id)
    if runs:
        run_id = int(runs[0]["id"])
    else:
        run_id = store.start_run(
            source=source,
            brief_id=brief_id,
            output_dir=str(store.db_path.parent),
            mode="fresh",
            resume_state={"brief_name": brief_id},
        )

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

    candidate = store.get_candidate(
        source=source, brief_id=brief_id, identity_key=identity_key
    )
    assert candidate is not None
    return int(candidate["id"])


def _read_payload(store: RuntimeStateStore, candidate_id: int) -> dict:
    with sqlite3.connect(str(store.db_path)) as conn:
        row = conn.execute(
            "SELECT terminal_payload_json FROM candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
    assert row is not None
    return json.loads(row[0])


def _read_judgment_accuracy(
    store: RuntimeStateStore, candidate_id: int
) -> tuple[str | None, str | None]:
    with sqlite3.connect(str(store.db_path)) as conn:
        row = conn.execute(
            "SELECT judgment_accuracy, judgment_accuracy_at "
            "FROM candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()
    assert row is not None
    return row[0], row[1]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_records_marker_and_judgment_accuracy_in_one_call(tmp_path: Path) -> None:
    """The unified-rollup contract: ``judgment_accuracy`` set, the
    per-principle metadata appended to ``terminal_payload_json``, and
    the existing ``full_decision`` payload preserved verbatim."""

    store = _make_store(tmp_path)
    candidate_id = _seed_designer_candidate(store)

    store.record_candidate_principle_marker(
        source="designer",
        brief_id="brief-d",
        identity_key="behance:joe",
        judgment_accuracy="useful",
        principle_marker={
            "principle_name": "Visual hierarchy",
            "marker": "useful_guidance",
            "note": "Strong primary focal point.",
            "marked_at": "2026-05-04T17:00:00+00:00",
        },
    )

    accuracy, accuracy_at = _read_judgment_accuracy(store, candidate_id)
    assert accuracy == "useful"
    assert accuracy_at is not None  # writer stamps a timestamp

    payload = _read_payload(store, candidate_id)
    # The full_decision dict the aggregator reads from must be untouched.
    assert payload["full_decision"]["capability_area"] == "Visual hierarchy"
    assert payload["full_decision"]["confidence"] == 0.82
    # The principle marker lands in a sibling list so the aggregator
    # path (judgment_accuracy + full_decision) keeps working unchanged.
    assert payload["principle_markers"] == [
        {
            "principle_name": "Visual hierarchy",
            "marker": "useful_guidance",
            "note": "Strong primary focal point.",
            "marked_at": "2026-05-04T17:00:00+00:00",
        }
    ]


def test_appends_in_chronological_order(tmp_path: Path) -> None:
    """Recruiter feedback over time is signal — every call appends to
    the list in order, never replaces. Slice 9 reflection polish reads
    the history; clobbering would lose calibration drift."""

    store = _make_store(tmp_path)
    _seed_designer_candidate(store)

    store.record_candidate_principle_marker(
        source="designer",
        brief_id="brief-d",
        identity_key="behance:joe",
        judgment_accuracy="useful",
        principle_marker={"principle_name": "Visual hierarchy", "marker": "useful_guidance"},
    )
    store.record_candidate_principle_marker(
        source="designer",
        brief_id="brief-d",
        identity_key="behance:joe",
        judgment_accuracy="off_rubric",
        principle_marker={"principle_name": "Typographic refinement", "marker": "off_rubric"},
    )
    store.record_candidate_principle_marker(
        source="designer",
        brief_id="brief-d",
        identity_key="behance:joe",
        judgment_accuracy="wrong",
        principle_marker={"principle_name": "Conceptual strength", "marker": "wrong_shallow"},
    )

    candidate = store.get_candidate(
        source="designer", brief_id="brief-d", identity_key="behance:joe"
    )
    assert candidate is not None
    payload = _read_payload(store, int(candidate["id"]))
    principles = [m["principle_name"] for m in payload["principle_markers"]]
    assert principles == [
        "Visual hierarchy",
        "Typographic refinement",
        "Conceptual strength",
    ]
    # Most-recent write wins on the column — matches
    # set_candidate_judgment_accuracy's "last write wins" semantics.
    accuracy, _ = _read_judgment_accuracy(store, int(candidate["id"]))
    assert accuracy == "wrong"


def test_works_against_empty_terminal_payload(tmp_path: Path) -> None:
    """Default ``terminal_payload_json`` is ``"{}"``; the marker creates
    the ``principle_markers`` list rather than failing on a missing
    key."""

    store = _make_store(tmp_path)
    run_id = store.start_run(
        source="designer",
        brief_id="brief-empty",
        output_dir=str(store.db_path.parent),
        mode="fresh",
        resume_state={"brief_name": "brief-empty"},
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="designer",
        brief_id="brief-empty",
        identity_key="behance:lee",
        display_name="Lee",
        profile_url="https://example.test/lee",
    )

    store.record_candidate_principle_marker(
        source="designer",
        brief_id="brief-empty",
        identity_key="behance:lee",
        judgment_accuracy="useful",
        principle_marker={"principle_name": "Color system coherence", "marker": "useful_guidance"},
    )

    candidate = store.get_candidate(
        source="designer", brief_id="brief-empty", identity_key="behance:lee"
    )
    assert candidate is not None
    payload = _read_payload(store, int(candidate["id"]))
    assert "full_decision" not in payload  # nothing to preserve here
    assert payload["principle_markers"] == [
        {"principle_name": "Color system coherence", "marker": "useful_guidance"}
    ]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_unknown_judgment_accuracy(tmp_path: Path) -> None:
    """Mirrors ``set_candidate_judgment_accuracy``'s gate: any value
    outside the writer-validated five-value set raises and never
    writes. The calibration aggregator's allowed set
    (``calibration.py:108-116``) is downstream of this gate; protecting
    here means a future legacy import can't sneak in."""

    store = _make_store(tmp_path)
    candidate_id = _seed_designer_candidate(store)
    payload_before = _read_payload(store, candidate_id)
    accuracy_before, _ = _read_judgment_accuracy(store, candidate_id)

    with pytest.raises(ValueError, match="invalid judgment_accuracy"):
        store.record_candidate_principle_marker(
            source="designer",
            brief_id="brief-d",
            identity_key="behance:joe",
            judgment_accuracy="thumbs_up",
            principle_marker={"principle_name": "Visual hierarchy", "marker": "x"},
        )

    # Both writes must roll back together — no partial state.
    assert _read_payload(store, candidate_id) == payload_before
    assert _read_judgment_accuracy(store, candidate_id)[0] == accuracy_before


def test_raises_for_unknown_candidate(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="candidate not found"):
        store.record_candidate_principle_marker(
            source="designer",
            brief_id="brief-d",
            identity_key="behance:ghost",
            judgment_accuracy="useful",
            principle_marker={"principle_name": "Visual hierarchy", "marker": "useful_guidance"},
        )


def test_recovers_from_corrupt_terminal_payload(tmp_path: Path) -> None:
    """Defensive: if a row's ``terminal_payload_json`` was hand-edited
    to non-JSON, the writer rebuilds rather than raising. Important
    because Designer's per-principle write is recruiter-driven and
    must not surface a 500 just because someone fat-fingered the DB
    earlier in the lifecycle."""

    store = _make_store(tmp_path)
    candidate_id = _seed_designer_candidate(store)
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE candidates SET terminal_payload_json = ? WHERE id = ?",
            ("not-json-at-all", candidate_id),
        )
        conn.commit()

    store.record_candidate_principle_marker(
        source="designer",
        brief_id="brief-d",
        identity_key="behance:joe",
        judgment_accuracy="useful",
        principle_marker={"principle_name": "Visual hierarchy", "marker": "useful_guidance"},
    )

    payload = _read_payload(store, candidate_id)
    assert payload == {
        "principle_markers": [
            {"principle_name": "Visual hierarchy", "marker": "useful_guidance"}
        ]
    }
