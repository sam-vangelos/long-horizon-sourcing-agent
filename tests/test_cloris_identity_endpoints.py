"""Tests for the Phase G Slice G2 identity reconciliation endpoints
(`GET /api/brief/{brief_id}/identity/pending`, `POST .../decision`,
`POST .../unlink`).

Pins the wire contract:
- Empty brief → empty decisions list, persons_total ≥ 0, NOT 500.
- Pending list reflects F3 ambiguous-name pending rows.
- POST decision merge → re-points links, locks the survivor person.
- POST decision keep_separate → stamps row, leaves links untouched.
- POST decision on already-resolved row → 422.
- POST decision on unknown id → 404.
- POST unlink → splits candidate into fresh person row, locks new link.
- POST unlink on unknown candidate → no-op (204), defensive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloris.app import create_app


def _seed_state_dir(state_root: Path, *, source: str, state_key: str) -> Path:
    state_dir = state_root / source / state_key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _seed_candidate(
    state_dir: Path,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
    display_name: str,
    profile_url: str = "",
    terminal_payload: dict | None = None,
) -> int:
    from shared.runtime_state.store import RuntimeStateStore

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    candidate_id = store.ensure_candidate(
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=display_name,
        profile_url=profile_url,
        initial_state="full_terminal",
    )
    if terminal_payload is not None:
        with store.connect() as conn:
            conn.execute(
                "UPDATE candidates SET terminal_payload_json = ? WHERE id = ?",
                (json.dumps(terminal_payload), candidate_id),
            )
    return candidate_id


@pytest.fixture()
def isolated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with STATE_ROOT and IDENTITY_ROOT isolated to tmp_path."""

    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    identity_root = state_root / "_identity"
    identity_root.mkdir(parents=True, exist_ok=True)
    import shared.output_paths as output_paths

    monkeypatch.setattr(output_paths, "STATE_ROOT", state_root)
    monkeypatch.setattr(output_paths, "IDENTITY_ROOT", identity_root)
    return TestClient(create_app()), state_root


def _seed_pending_brief(state_root: Path, brief_id: str) -> tuple[int, int]:
    """Seed two same-name candidates → F3 writes a pending_merge_decisions row."""

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key=brief_id + "-a")
    cand_a = _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-a",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-12345/",
    )
    li_dir_b = _seed_state_dir(
        state_root, source="linkedin", state_key=brief_id + "-b"
    )
    cand_b = _seed_candidate(
        li_dir_b,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-b",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-67890/",
    )
    return cand_a, cand_b


def test_pending_endpoint_empty_for_unknown_brief(isolated_client):
    client, _ = isolated_client
    res = client.get("/api/brief/unknown_brief/identity/pending")
    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-identity-pending-1"
    assert body["brief_id"] == "unknown_brief"
    assert body["persons_total"] == 0
    assert body["decisions"] == []


def test_pending_endpoint_returns_ambiguous_pair(isolated_client):
    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_ambig")

    res = client.get("/api/brief/brief_g2_ambig/identity/pending")
    assert res.status_code == 200
    body = res.json()
    assert body["persons_total"] == 2
    assert len(body["decisions"]) == 1
    decision = body["decisions"][0]
    assert decision["decision_id"] >= 1
    # Editorial signal_summary, not raw enum:
    assert decision["signal_summary"]
    assert "_" not in decision["signal_summary"]
    # Person evidence on both sides:
    assert decision["person_a"]["canonical_name"] == "John Smith"
    assert decision["person_b"]["canonical_name"] == "John Smith"
    # Confidence float NOT exposed to wire (R14 hygiene):
    assert "confidence" not in decision


def test_pending_endpoint_does_not_materialize_full_person_read_model(
    isolated_client, monkeypatch
):
    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_count_only")

    import shared.identity_resolution_service as identity_service

    def fail_full_read(*_args, **_kwargs):
        raise AssertionError("pending endpoint should use count-only person read")

    monkeypatch.setattr(
        identity_service,
        "brief_persons_with_evidence",
        fail_full_read,
    )

    res = client.get("/api/brief/brief_g2_count_only/identity/pending")
    assert res.status_code == 200
    body = res.json()
    assert body["persons_total"] == 2
    assert len(body["decisions"]) == 1


def test_pending_endpoint_excludes_confidence_from_wire(isolated_client):
    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_no_confidence")

    res = client.get("/api/brief/brief_g2_no_confidence/identity/pending")
    assert res.status_code == 200
    decisions = res.json()["decisions"]
    assert decisions
    for d in decisions:
        assert "confidence" not in d
        assert "evidence_json" not in d


def test_decision_merge_re_points_links_and_drops_pending(isolated_client):
    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_merge")

    pending = client.get("/api/brief/brief_g2_merge/identity/pending").json()
    decision_id = pending["decisions"][0]["decision_id"]

    res = client.post(
        "/api/brief/brief_g2_merge/identity/decision",
        json={"decision_id": decision_id, "choice": "merge"},
    )
    assert res.status_code == 204

    # After merge: the pending list collapses (decision is terminal)
    after = client.get("/api/brief/brief_g2_merge/identity/pending").json()
    assert after["decisions"] == []
    # Persons collapsed from 2 to 1 by the merge:
    assert after["persons_total"] == 1


def test_decision_keep_separate_clears_pending_without_merging(isolated_client):
    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_keep")

    pending = client.get("/api/brief/brief_g2_keep/identity/pending").json()
    decision_id = pending["decisions"][0]["decision_id"]

    res = client.post(
        "/api/brief/brief_g2_keep/identity/decision",
        json={"decision_id": decision_id, "choice": "keep_separate"},
    )
    assert res.status_code == 204

    after = client.get("/api/brief/brief_g2_keep/identity/pending").json()
    assert after["decisions"] == []
    # Persons stay at 2 — keep_separate doesn't merge:
    assert after["persons_total"] == 2


def test_decision_keep_separate_already_resolved_returns_422(isolated_client):
    """When a keep_separate stamp lands but the row stays in the DB, a
    second decision attempt hits the ``already_resolved`` 422 branch.

    Note: the merge case CAN'T hit this branch because the F3 schema
    declares ``pending_merge_decisions.person_b ON DELETE CASCADE`` —
    when ``record_recruiter_merge`` drops the merged-away person, the
    pending row vanishes with it (the contract becomes 404). That's
    an F3 substrate limitation; this test pins the keep_separate path
    where the row is preserved.
    """

    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_terminal")

    pending = client.get("/api/brief/brief_g2_terminal/identity/pending").json()
    decision_id = pending["decisions"][0]["decision_id"]
    client.post(
        "/api/brief/brief_g2_terminal/identity/decision",
        json={"decision_id": decision_id, "choice": "keep_separate"},
    )

    res = client.post(
        "/api/brief/brief_g2_terminal/identity/decision",
        json={"decision_id": decision_id, "choice": "merge"},
    )
    assert res.status_code == 422
    body = res.json()
    assert body["detail"]["error"] == "identity_decision_already_resolved"


def test_decision_merge_then_retry_returns_404_due_to_fk_cascade(isolated_client):
    """Pin the F3 schema's CASCADE behavior: merging deletes person_b,
    which cascade-deletes the pending_merge_decisions row. A retry of
    the same decision_id hits ``not_found`` (404), NOT ``already_resolved``
    (422). G2's frontend handles both shapes editorially.
    """

    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_cascade")

    pending = client.get("/api/brief/brief_g2_cascade/identity/pending").json()
    decision_id = pending["decisions"][0]["decision_id"]
    client.post(
        "/api/brief/brief_g2_cascade/identity/decision",
        json={"decision_id": decision_id, "choice": "merge"},
    )

    res = client.post(
        "/api/brief/brief_g2_cascade/identity/decision",
        json={"decision_id": decision_id, "choice": "keep_separate"},
    )
    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "identity_decision_not_found"


def test_decision_unknown_id_returns_404(isolated_client):
    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_unknown")

    res = client.post(
        "/api/brief/brief_g2_unknown/identity/decision",
        json={"decision_id": 99999, "choice": "merge"},
    )
    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "identity_decision_not_found"


def test_decision_invalid_choice_returns_422(isolated_client):
    client, state_root = isolated_client
    _seed_pending_brief(state_root, "brief_g2_bad_choice")

    res = client.post(
        "/api/brief/brief_g2_bad_choice/identity/decision",
        json={"decision_id": 1, "choice": "delete"},
    )
    # Pydantic Literal mismatch → 422
    assert res.status_code == 422


def test_unlink_splits_candidate_into_new_person(isolated_client):
    client, state_root = isolated_client
    cand_a, cand_b = _seed_pending_brief(state_root, "brief_g2_unlink")

    # First merge so both candidates point at one person:
    pending = client.get("/api/brief/brief_g2_unlink/identity/pending").json()
    decision_id = pending["decisions"][0]["decision_id"]
    client.post(
        "/api/brief/brief_g2_unlink/identity/decision",
        json={"decision_id": decision_id, "choice": "merge"},
    )

    after_merge = client.get("/api/brief/brief_g2_unlink/identity/pending").json()
    assert after_merge["persons_total"] == 1

    # Now unlink one candidate:
    res = client.post(
        "/api/brief/brief_g2_unlink/identity/unlink",
        json={"source": "linkedin", "state_key": "brief_g2_unlink-b", "candidate_id": cand_b},
    )
    assert res.status_code == 204

    after_unlink = client.get("/api/brief/brief_g2_unlink/identity/pending").json()
    # Split → back to 2 persons:
    assert after_unlink["persons_total"] == 2


def test_unlink_unknown_candidate_returns_204_idempotent(isolated_client):
    client, _ = isolated_client
    res = client.post(
        "/api/brief/brief_unknown/identity/unlink",
        json={"source": "linkedin", "state_key": "nope", "candidate_id": 999999},
    )
    assert res.status_code == 204
