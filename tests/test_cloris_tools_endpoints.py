"""Tests for the Phase G Slice G4 tools framework
(`GET /api/tools`, `POST /api/tools/{tool_id}`, `GET /api/tools/jobs/{job_id}`).

Pins:
- Catalog lists every registered tool with editorial pitch + cli_command.
- Tier C / cli_only tools 422 on POST run (catalog already shows them).
- Unknown tool_id → 404.
- Invalid args (Pydantic validation failure) → 422.
- Sync execution returns exit_code + tails.
- Async execution returns job_id; status pollable.
- Job status 404 on unknown job_id.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloris.app import create_app


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_tools_index_returns_registered_catalog(client):
    res = client.get("/api/tools")
    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-tools-index-1"
    tool_ids = {t["tool_id"] for t in body["tools"]}
    # Tier A:
    assert "iterate_brief" in tool_ids
    assert "update_market_intel" in tool_ids
    assert "run_recruiter_identity_resolver" in tool_ids
    assert "backfill_clean_linkedin_urls" in tool_ids
    # Tier B:
    assert "backfill_brief_snapshot" in tool_ids
    assert "gemini_consult" in tool_ids
    # Tier B/C cli_only:
    assert "audit_surfaces" in tool_ids
    # 12+ entries total:
    assert len(body["tools"]) >= 10


def test_tools_index_carries_editorial_pitch_and_cli_command(client):
    res = client.get("/api/tools")
    body = res.json()
    iterate = next(t for t in body["tools"] if t["tool_id"] == "iterate_brief")
    assert iterate["pitch"]
    assert "_" not in iterate["pitch"]  # no raw enums leak into editorial
    assert iterate["cli_command"].startswith("tools/iterate_brief.py")


def test_tools_index_includes_schema_fields_for_a_tier(client):
    res = client.get("/api/tools")
    body = res.json()
    iterate = next(t for t in body["tools"] if t["tool_id"] == "iterate_brief")
    field_names = {f["name"] for f in iterate["schema_fields"]}
    assert "brief_path" in field_names
    assert "report_path" in field_names


def test_unknown_tool_id_returns_404(client):
    res = client.post("/api/tools/nonexistent_tool", json={"args": {}})
    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "tool_not_found"


def test_cli_only_tool_returns_422(client):
    """Tier C / cli_only tools shouldn't be runnable via the API — the
    catalog surfaces them as documentation entries only."""

    res = client.post("/api/tools/audit_surfaces", json={"args": {}})
    assert res.status_code == 422
    body = res.json()
    assert body["detail"]["error"] == "tool_cli_only"
    assert body["detail"]["cli_command"]


def test_invalid_args_returns_422(client):
    """Pydantic Literal mismatch on workflow_mode → 422."""

    res = client.post(
        "/api/tools/run_recruiter_identity_resolver",
        json={"args": {"workflow_mode": "invalid"}},
    )
    assert res.status_code == 422
    assert res.json()["detail"]["error"] == "tool_args_invalid"


def test_async_tool_returns_job_id(client):
    """An async tool should return a job_id immediately. The actual
    subprocess will fail (no real brief path), but the framework
    contract is what we're pinning."""

    res = client.post(
        "/api/tools/run_recruiter_identity_resolver",
        json={"args": {"workflow_mode": "dry_run"}},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-tool-async-1"
    assert body["tool_id"] == "run_recruiter_identity_resolver"
    assert body["job_id"]


def test_backfill_clean_linkedin_urls_runs_in_process(monkeypatch, tmp_path):
    async def fail_subprocess(*_args, **_kwargs):
        raise AssertionError("backfill runner should not spawn a subprocess")

    monkeypatch.setattr(
        "cloris.tools_runtime.asyncio.create_subprocess_exec",
        fail_subprocess,
    )
    state_root = tmp_path / "state" / "linkedin"
    state_dir = state_root / "brief-1"
    state_dir.mkdir(parents=True)
    db_path = state_dir / "runtime_state.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE candidates (id INTEGER PRIMARY KEY, source TEXT, profile_url TEXT)"
        )
        conn.execute(
            "INSERT INTO candidates (source, profile_url) VALUES (?, ?)",
            (
                "linkedin",
                "https://www.linkedin.com/in/example/?miniProfileUrn=abc&trackingId=def",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    with TestClient(create_app()) as live_client:
        res = live_client.post(
            "/api/tools/backfill_clean_linkedin_urls",
            json={"args": {"state_root": str(state_root), "dry_run": True}},
        )
        assert res.status_code == 200
        job_id = res.json()["job_id"]

        status = None
        for _ in range(40):
            poll = live_client.get(f"/api/tools/jobs/{job_id}")
            assert poll.status_code == 200
            status = poll.json()
            if status["status"] in {"succeeded", "failed", "purged"}:
                break
            time.sleep(0.05)

    assert status is not None
    assert status["status"] == "succeeded"
    assert status["exit_code"] == 0
    assert "DRY RUN" in status["stdout_tail"]
    assert "would update 1" in status["stdout_tail"]

    conn = sqlite3.connect(str(db_path))
    try:
        stored_url = conn.execute("SELECT profile_url FROM candidates").fetchone()[0]
    finally:
        conn.close()
    assert "trackingId=def" in stored_url


def test_unknown_job_id_returns_404(client):
    res = client.get("/api/tools/jobs/no_such_job")
    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "tool_job_not_found"
