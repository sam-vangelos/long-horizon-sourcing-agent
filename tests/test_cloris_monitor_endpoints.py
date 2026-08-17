"""Tests for the Phase G Slice G3 Live Monitor endpoints
(`GET /api/monitor/index`, `GET /api/run/{source}/{state_key}/{run_id}/telemetry`).

Pins:
- Index returns active runs only (filters by worker_alive=True).
- Telemetry returns recent attempts + events, bounded by limits.
- Telemetry on unknown state_dir → 404.
- Telemetry on a state_dir with no DB returns empty rows, not 500.
- Attempt rows preserve raw enums (operational register).
- Event payload_summary truncates oversized payloads.
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


@pytest.fixture()
def client_with_isolated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    import shared.output_paths as output_paths

    monkeypatch.setattr(output_paths, "STATE_ROOT", state_root)
    return TestClient(create_app()), state_root


def _seed_run_with_attempts(
    state_root: Path,
    *,
    source: str,
    state_key: str,
    attempts: int = 3,
    events: int = 2,
) -> tuple[int, int]:
    """Seed a state_dir with a run + N attempts + M events.

    Returns ``(run_id, candidate_id)`` so tests can assert against the
    same ids the endpoints return.
    """

    state_dir = state_root / source / state_key
    state_dir.mkdir(parents=True, exist_ok=True)

    from shared.runtime_state.store import RuntimeStateStore

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source=source,
        mode="fresh",
        brief_id="test_brief",
        output_dir=str(state_dir),
    )
    candidate_id = store.ensure_candidate(
        source=source,
        brief_id="test_brief",
        identity_key="ident-1",
        display_name="Test Person",
        profile_url="https://example.com/profile",
        initial_state="full_terminal",
    )

    with store.connect() as conn:
        for i in range(attempts):
            conn.execute(
                """
                INSERT INTO candidate_attempts(
                    run_id, candidate_id, work_unit_id, stage, attempt_number,
                    status, failure_kind, failure_reason, payload_json,
                    source_cursor_json, started_at, ended_at
                )
                VALUES (?, ?, NULL, ?, ?, ?, ?, NULL, '{}', '{}', ?, ?)
                """,
                (
                    run_id,
                    candidate_id,
                    "facial",
                    i + 1,
                    "succeeded" if i < attempts - 1 else "failed",
                    "browser_disconnect" if i == attempts - 1 else None,
                    f"2026-05-01T00:0{i}:00Z",
                    f"2026-05-01T00:0{i}:30Z" if i < attempts - 1 else None,
                ),
            )
        for i in range(events):
            conn.execute(
                """
                INSERT INTO events(
                    run_id, work_unit_id, candidate_id, attempt_id,
                    event_type, payload_json, created_at
                )
                VALUES (?, NULL, ?, NULL, ?, ?, ?)
                """,
                (
                    run_id,
                    candidate_id,
                    "lifecycle_transition",
                    json.dumps({"i": i, "msg": "x" * 300}),
                    f"2026-05-01T01:0{i}:00Z",
                ),
            )

    return run_id, candidate_id


def test_monitor_index_returns_empty_when_no_active_runs(
    client_with_isolated_root,
):
    client, _ = client_with_isolated_root
    res = client.get("/api/monitor/index")
    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-monitor-index-1"
    assert body["active_runs"] == []


def test_telemetry_returns_attempts_and_events_for_known_run(
    client_with_isolated_root,
):
    client, state_root = client_with_isolated_root
    run_id, _ = _seed_run_with_attempts(
        state_root, source="linkedin", state_key="li-test-g3"
    )

    res = client.get(f"/api/run/linkedin/li-test-g3/{run_id}/telemetry")
    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-run-telemetry-1"
    assert body["source"] == "linkedin"
    assert body["state_key"] == "li-test-g3"
    assert body["run_id"] == run_id
    assert body["attempts_total"] == 3
    # `start_run` auto-writes a run_start event, so the seeded 2 + 1 auto = 3.
    assert body["events_total"] == 3
    assert len(body["attempts"]) == 3
    assert len(body["events"]) == 3


def test_telemetry_attempts_sorted_most_recent_first(
    client_with_isolated_root,
):
    client, state_root = client_with_isolated_root
    run_id, _ = _seed_run_with_attempts(
        state_root, source="linkedin", state_key="li-sorted-g3", attempts=4, events=0
    )

    res = client.get(f"/api/run/linkedin/li-sorted-g3/{run_id}/telemetry")
    assert res.status_code == 200
    attempts = res.json()["attempts"]
    assert attempts[0]["attempt_number"] == 4
    assert attempts[-1]["attempt_number"] == 1


def test_telemetry_preserves_raw_failure_kind_enum(client_with_isolated_root):
    client, state_root = client_with_isolated_root
    run_id, _ = _seed_run_with_attempts(
        state_root, source="linkedin", state_key="li-enum-g3", attempts=1, events=0
    )

    res = client.get(f"/api/run/linkedin/li-enum-g3/{run_id}/telemetry")
    body = res.json()
    # Last attempt was seeded as failed/browser_disconnect; preserve raw
    # enum (operational register, NOT editorial).
    assert body["attempts"][0]["failure_kind"] == "browser_disconnect"
    assert body["attempts"][0]["status"] == "failed"


def test_telemetry_truncates_oversized_event_payload(client_with_isolated_root):
    client, state_root = client_with_isolated_root
    run_id, _ = _seed_run_with_attempts(
        state_root, source="linkedin", state_key="li-trunc-g3", attempts=0, events=1
    )

    res = client.get(f"/api/run/linkedin/li-trunc-g3/{run_id}/telemetry")
    body = res.json()
    # start_run auto-writes a run_start event; we seed 1 oversized event.
    assert len(body["events"]) == 2
    # Find the seeded oversized lifecycle_transition (not the auto run_start):
    oversized = next(
        e for e in body["events"] if e["event_type"] == "lifecycle_transition"
    )
    summary = oversized["payload_summary"] or ""
    # 240-char limit + ellipsis sentinel:
    assert summary.endswith("…")
    assert len(summary) <= 241


def test_telemetry_unknown_state_dir_returns_404(client_with_isolated_root):
    client, _ = client_with_isolated_root
    res = client.get("/api/run/linkedin/unknown-key/1/telemetry")
    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "state_dir_not_found"


def test_telemetry_state_dir_without_db_returns_empty(
    client_with_isolated_root,
):
    """Edge case: state_dir exists but no runtime_state.sqlite3 yet
    (e.g. worker just spawned, hasn't written its first row). The
    monitor should render empty, not 500."""

    client, state_root = client_with_isolated_root
    state_dir = state_root / "linkedin" / "li-empty-g3"
    state_dir.mkdir(parents=True, exist_ok=True)

    res = client.get("/api/run/linkedin/li-empty-g3/1/telemetry")
    assert res.status_code == 200
    body = res.json()
    assert body["attempts"] == []
    assert body["events"] == []
    assert body["attempts_total"] == 0
