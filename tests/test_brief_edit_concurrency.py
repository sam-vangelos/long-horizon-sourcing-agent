"""Tests for brief edit optimistic locking (ETag-style via last_modified)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _v2_brief(role: str = "Test Role") -> dict:
    return {
        "role_title": role,
        "id": role.lower().replace(" ", "_"),
        "linkedin_project_id": role.lower().replace(" ", "_"),
        "capability_areas": [
            {
                "name": "Engineering",
                "description": "Ships customer-facing systems.",
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns architecture.",
            "user_definition": "Maintains features.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }


@pytest.fixture()
def brief_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with an isolated tmp config dir containing one valid brief."""
    from cloris import api as cloris_api
    from cloris.app import create_app

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)

    brief_dir = config_dir / "concurrency-role"
    brief_dir.mkdir()
    brief_file = brief_dir / "brief.json"
    brief_file.write_text(json.dumps(_v2_brief("Concurrency Role")))

    from shared.output_paths import derive_brief_id

    brief_id = derive_brief_id(brief_path=str(brief_file))
    client = TestClient(create_app())
    return client, brief_file, brief_id


def _edit_body(last_modified=None) -> dict:
    # Keep id/linkedin_project_id stable so derive_brief_id returns the same
    # brief_id after the PUT (the ID is content-derived from these fields).
    v2 = _v2_brief("Concurrency Role Updated")
    v2["id"] = "concurrency_role"
    v2["linkedin_project_id"] = "concurrency_role"
    body: dict = {
        "v2_data": v2,
        "preserved_legacy": {},
        "dropped_legacy_keys": [],
    }
    if last_modified is not None:
        body["last_modified"] = last_modified
    return body


def _current_mtime_iso(path: Path) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def test_put_without_last_modified_succeeds(brief_client):
    """PUT with no last_modified field skips staleness check (backward compat)."""
    client, _, brief_id = brief_client
    response = client.put(f"/api/brief/{brief_id}", json=_edit_body())
    assert response.status_code == 200


def test_put_with_matching_last_modified_succeeds(brief_client):
    """PUT echoing the correct mtime from a GET succeeds."""
    client, brief_file, brief_id = brief_client
    mtime_iso = _current_mtime_iso(brief_file)

    response = client.put(f"/api/brief/{brief_id}", json=_edit_body(mtime_iso))
    assert response.status_code == 200


def test_put_with_stale_last_modified_returns_409(brief_client):
    """PUT with a stale last_modified returns 409 Conflict."""
    client, brief_file, brief_id = brief_client
    stale_mtime = _current_mtime_iso(brief_file)

    # Simulate a concurrent edit: rewrite the file to advance its mtime.
    time.sleep(0.01)
    brief_file.write_text(brief_file.read_text())

    response = client.put(f"/api/brief/{brief_id}", json=_edit_body(stale_mtime))
    assert response.status_code == 409

    detail = response.json()["detail"]
    assert detail["error"] == "stale_edit"
    assert "client_last_modified" in detail
    assert "server_last_modified" in detail
    assert detail["client_last_modified"] == stale_mtime
    assert detail["client_last_modified"] != detail["server_last_modified"]


def test_409_has_user_facing_message(brief_client):
    """409 detail includes a user-readable message."""
    client, brief_file, brief_id = brief_client
    stale_mtime = _current_mtime_iso(brief_file)
    time.sleep(0.01)
    brief_file.write_text(brief_file.read_text())

    response = client.put(f"/api/brief/{brief_id}", json=_edit_body(stale_mtime))
    assert response.status_code == 409
    assert "modified" in response.json()["detail"]["message"].lower()
