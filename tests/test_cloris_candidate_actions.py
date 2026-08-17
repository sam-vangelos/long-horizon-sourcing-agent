"""Tests for the Phase C, slice C3 candidate-action mutations
(``RuntimeStateStore.append_candidate_note`` /
``set_candidate_user_status``) and the corresponding API endpoints.

Pins the contract:

- Schema v8 adds ``candidates.notes`` (JSON array) and
  ``candidates.user_status`` (nullable text).
- ``append_candidate_note`` appends, never overwrites; whitespace-only
  bodies raise ``ValueError``.
- ``set_candidate_user_status(None)`` clears the override; non-null
  values write through verbatim.
- The aggregator surfaces parsed notes + user_status.
- Endpoints map missing candidate id → 404, empty / invalid input → 422.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from cloris.control_plane import aggregate_candidate_detail
from shared.runtime_state.store import CURRENT_SCHEMA_VERSION, RuntimeStateStore


def test_schema_adds_notes_user_status_and_judgment_accuracy_columns(
    tmp_path: Path,
) -> None:
    """Schema migration adds the v8 columns (``notes``, ``user_status``)
    and the v9 columns (``judgment_accuracy``, ``judgment_accuracy_at``).
    Later versions follow from ongoing ``CURRENT_SCHEMA_VERSION`` bumps.
    Pins to ``CURRENT_SCHEMA_VERSION`` so the assertion tracks the
    constant rather than going stale on the next bump."""

    db_path = tmp_path / "runtime_state.sqlite3"
    RuntimeStateStore(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
    }
    conn.close()

    assert CURRENT_SCHEMA_VERSION == "12"
    assert "notes" in columns
    assert "user_status" in columns
    assert "judgment_accuracy" in columns
    assert "judgment_accuracy_at" in columns


def _seed_candidate(tmp_path: Path) -> tuple[Path, RuntimeStateStore, int]:
    state_dir = tmp_path / "linkedin" / "key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-1",
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": "brief-1"},
    )
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
    )
    snippet_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        stage="snippet",
        display_name="Pat Doe",
        profile_url="https://linkedin.com/in/pat",
    )
    store.finish_attempt_success(
        attempt_id=snippet_id,
        new_state="snippet_extracted",
        payload={},
        run_id=run_id,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        new_state="facial_started",
    )
    facial_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        stage="facial",
    )
    store.finish_attempt_success(
        attempt_id=facial_id,
        new_state="facial_terminal",
        payload={},
        run_id=run_id,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        new_state="full_started",
    )
    full_id = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-1",
        stage="full",
    )
    store.finish_attempt_success(
        attempt_id=full_id,
        new_state="full_terminal",
        terminal_decision="SAVE",
        payload={"confidence": 0.9, "save_reason": "Strong fit"},
        run_id=run_id,
    )
    conn = sqlite3.connect(f"file:{store.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT id FROM candidates WHERE identity_key='li-1'").fetchone()
        return state_dir, store, int(row["id"])
    finally:
        conn.close()


def test_append_candidate_note_persists_in_canonical_state(tmp_path: Path) -> None:
    state_dir, store, candidate_id = _seed_candidate(tmp_path)

    count_after = store.append_candidate_note(candidate_id, "First note")
    assert count_after == 1

    detail = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail is not None
    assert len(detail.notes) == 1
    assert detail.notes[0].body == "First note"
    assert detail.notes[0].created_at  # non-empty timestamp


def test_append_candidate_note_is_append_only(tmp_path: Path) -> None:
    state_dir, store, candidate_id = _seed_candidate(tmp_path)

    store.append_candidate_note(candidate_id, "First")
    store.append_candidate_note(candidate_id, "Second")
    store.append_candidate_note(candidate_id, "Third")

    detail = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail is not None
    assert [n.body for n in detail.notes] == ["First", "Second", "Third"]


def test_append_candidate_note_rejects_whitespace_body(tmp_path: Path) -> None:
    _, store, candidate_id = _seed_candidate(tmp_path)

    with pytest.raises(ValueError):
        store.append_candidate_note(candidate_id, "   ")


def test_append_candidate_note_raises_for_unknown_id(tmp_path: Path) -> None:
    _, store, _ = _seed_candidate(tmp_path)

    with pytest.raises(ValueError):
        store.append_candidate_note(99999, "Some note")


def test_set_candidate_user_status_writes_and_clears(tmp_path: Path) -> None:
    state_dir, store, candidate_id = _seed_candidate(tmp_path)

    store.set_candidate_user_status(candidate_id, "shortlist")
    detail_after_set = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail_after_set is not None
    assert detail_after_set.user_status == "shortlist"

    store.set_candidate_user_status(candidate_id, None)
    detail_after_clear = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail_after_clear is not None
    assert detail_after_clear.user_status is None


def test_set_candidate_user_status_raises_for_unknown_id(tmp_path: Path) -> None:
    _, store, _ = _seed_candidate(tmp_path)

    with pytest.raises(ValueError):
        store.set_candidate_user_status(99999, "shortlist")


def test_user_status_persists_across_aggregator_reads(tmp_path: Path) -> None:
    """Confirm the user_status field survives the read path that the
    candidate-detail surface goes through, even after the row is touched
    by additional notes."""

    state_dir, store, candidate_id = _seed_candidate(tmp_path)

    store.set_candidate_user_status(candidate_id, "contacted")
    store.append_candidate_note(candidate_id, "Reached out via email")

    detail = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail is not None
    assert detail.user_status == "contacted"
    assert len(detail.notes) == 1


# ---------------------------------------------------------------------------
# Phase C-bis Slice 0.5: closed-loop feedback substrate (judgment_accuracy)
# ---------------------------------------------------------------------------


def test_set_candidate_judgment_accuracy_writes_and_clears(
    tmp_path: Path,
) -> None:
    """Setting ``judgment_accuracy`` persists the value AND a timestamp;
    clearing (``None``) clears both. Distinct from ``user_status`` —
    both columns live independently on the row."""

    state_dir, store, candidate_id = _seed_candidate(tmp_path)

    # Initially, both judgment_accuracy and judgment_accuracy_at are NULL.
    detail_before = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail_before is not None
    assert detail_before.judgment_accuracy is None
    assert detail_before.judgment_accuracy_at is None

    store.set_candidate_judgment_accuracy(candidate_id, "useful")

    detail_after_set = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail_after_set is not None
    assert detail_after_set.judgment_accuracy == "useful"
    # Timestamp tracks the value — set together, cleared together.
    assert detail_after_set.judgment_accuracy_at is not None

    # And the user_status column is independently NULL — these are
    # schema-distinct fields by design.
    assert detail_after_set.user_status is None

    store.set_candidate_judgment_accuracy(candidate_id, None)

    detail_after_clear = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail_after_clear is not None
    assert detail_after_clear.judgment_accuracy is None
    assert detail_after_clear.judgment_accuracy_at is None


def test_set_candidate_judgment_accuracy_rejects_unknown_value(
    tmp_path: Path,
) -> None:
    """Unknown values raise ``ValueError`` at the store layer — the
    validation lives close to the data so any caller (API, CLI, future
    backfill) hits the same gate."""

    _, store, candidate_id = _seed_candidate(tmp_path)

    with pytest.raises(ValueError, match="invalid judgment_accuracy"):
        store.set_candidate_judgment_accuracy(candidate_id, "garbage")


def test_set_candidate_judgment_accuracy_raises_for_unknown_id(
    tmp_path: Path,
) -> None:
    _, store, _ = _seed_candidate(tmp_path)

    with pytest.raises(ValueError, match="not found"):
        store.set_candidate_judgment_accuracy(99999, "useful")


def test_judgment_accuracy_and_user_status_are_independent(
    tmp_path: Path,
) -> None:
    """The two columns must be independently settable. Setting one
    leaves the other untouched — that's the whole point of keeping them
    schema-distinct: pipeline action and judgment calibration are
    orthogonal signals."""

    _, store, candidate_id = _seed_candidate(tmp_path)

    store.set_candidate_user_status(candidate_id, "shortlist")
    store.set_candidate_judgment_accuracy(candidate_id, "wrong")

    detail = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail is not None
    assert detail.user_status == "shortlist"
    assert detail.judgment_accuracy == "wrong"

    # Clearing one leaves the other intact.
    store.set_candidate_judgment_accuracy(candidate_id, None)

    detail = aggregate_candidate_detail(
        brief_id="brief-1",
        candidate_id=candidate_id,
        state_root=tmp_path,
    )
    assert detail is not None
    assert detail.user_status == "shortlist"
    assert detail.judgment_accuracy is None


def test_judgment_accuracy_endpoint_returns_422_on_unknown_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API contract: the PATCH endpoint validates the input against the
    allowed set and returns 422 with a structured error body. This is
    the gate the future Phase D Next Run Learning surface will rely on
    when constructing valid PATCH payloads."""

    from fastapi.testclient import TestClient

    from cloris.app import create_app
    import shared.output_paths

    _, _, candidate_id = _seed_candidate(tmp_path)
    # Redirect the lazy STATE_ROOT lookup that aggregate_* helpers do
    # when state_root=None. Patches the attribute the late import binds
    # against, so every code path under the API endpoint sees tmp_path.
    monkeypatch.setattr(shared.output_paths, "STATE_ROOT", tmp_path)

    client = TestClient(create_app())
    response = client.patch(
        f"/api/candidate/brief-1/{candidate_id}/judgment-accuracy",
        json={"judgment_accuracy": "garbage"},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"] == "invalid_judgment_accuracy"
    assert "useful" in body["detail"]["allowed"]
    assert "wrong" in body["detail"]["allowed"]


def test_judgment_accuracy_endpoint_writes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: PATCH writes the value, GET reflects it. This pins
    that the read/write paths converge on the same column and the new
    field is on the wire by default (Phase D will bind the UI to it)."""

    from fastapi.testclient import TestClient

    from cloris.app import create_app
    import shared.output_paths

    _, _, candidate_id = _seed_candidate(tmp_path)
    monkeypatch.setattr(shared.output_paths, "STATE_ROOT", tmp_path)

    client = TestClient(create_app())
    response = client.patch(
        f"/api/candidate/brief-1/{candidate_id}/judgment-accuracy",
        json={"judgment_accuracy": "useful"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["judgment_accuracy"] == "useful"
    assert body["judgment_accuracy_at"] is not None

    # GET reflects it too.
    get_response = client.get(f"/api/candidate/brief-1/{candidate_id}")
    assert get_response.status_code == 200
    assert get_response.json()["judgment_accuracy"] == "useful"
