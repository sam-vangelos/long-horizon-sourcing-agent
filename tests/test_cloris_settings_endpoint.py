"""Tests for the Phase G Slice G5 settings transparency endpoint.

Pins:
- Credentials are boolean-only (R14 hygiene).
- Governor limits read as constants from shared/governor.py.
- Save destinations summarized from V2 source_config per brief.
- Editorial register: explainers + pitches present, no raw enums.
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
def client():
    return TestClient(create_app())


def test_settings_returns_credentials_as_boolean_only(client):
    res = client.get("/api/settings")
    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-settings-1"
    assert isinstance(body["credentials"], list)
    assert len(body["credentials"]) >= 4
    for c in body["credentials"]:
        assert isinstance(c["present"], bool)
        # Never expose the actual value:
        assert "value" not in c
        assert "secret" not in c
        # Editorial pitch present:
        assert c["pitch"]
        assert c["label"]


def test_settings_governor_renders_constants(client):
    res = client.get("/api/settings")
    body = res.json()
    governor = body["governor"]
    names = {g["name"] for g in governor}
    assert "MAX_PROFILE_OPENS_PER_SESSION" in names
    assert "MAX_PROFILE_OPENS_PER_24H" in names
    # CLO-153: the daily session-count cap is removed; the settings surface
    # must not advertise a gate that no longer exists.
    assert "MAX_SESSIONS_PER_DAY" not in names
    # Each limit carries an editorial explainer:
    for g in governor:
        assert g["explainer"]
        assert g["label"]


def test_settings_governor_session_limit_matches_shared_module(client):
    """Belt-and-suspenders: the wire value should match the runtime constant."""

    res = client.get("/api/settings")
    body = res.json()
    import shared.governor as gov

    profile_session = next(
        g for g in body["governor"] if g["name"] == "MAX_PROFILE_OPENS_PER_SESSION"
    )
    assert profile_session["value"] == int(gov.MAX_PROFILE_OPENS_PER_SESSION)


def test_settings_save_destinations_is_a_list(client):
    """The shape pin: save_destinations is a list of summaries (may be
    empty in test env if no briefs are seeded)."""

    res = client.get("/api/settings")
    body = res.json()
    assert isinstance(body["save_destinations"], list)


def test_settings_save_destinations_includes_authored_briefs(
    client, tmp_path, monkeypatch
):
    """Bug regression: the typo ``api_list_briefs()`` (vs ``api_briefs``)
    inside ``api_settings`` raised ``NameError`` at runtime, but the
    surrounding bare ``except Exception`` swallowed it and silently
    degraded ``save_destinations`` to ``[]`` on every call.

    The previous shape test (``..._is_a_list``) tolerated empty, so the
    bug passed the existing suite. This test pins the fan-out: when an
    authored brief exists in the catalog, ``/api/settings`` must surface
    a summary for it.

    Note on path resolution: this test seeds a brief inside a temp
    directory and must monkeypatch BOTH ``_CONFIG_DIR`` and
    ``_PROJECT_ROOT`` so ``_scan_authored_briefs`` produces a
    project-relative path that resolves under the temp tree, AND
    ``chdir`` so ``derive_brief_id`` (which opens the relative path
    via ``read_json``) finds the seeded brief at the patched relative
    location.
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    brief_path = config_dir / "brief-settings-fan-out-test.json"
    brief_path.write_text(
        json.dumps(
            {
                "role_title": "Settings Fan-Out Test Role",
                "linkedin_project_id": "9999000099990000",
                "permanent_filters": {},
                "preferred_filters": {},
            }
        )
    )

    monkeypatch.setattr("cloris.api._paths._CONFIG_DIR", config_dir)
    monkeypatch.setattr("cloris.api._paths._CONFIG_PARENT", tmp_path)
    monkeypatch.setattr("cloris.api._paths._PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)

    res = client.get("/api/settings")
    assert res.status_code == 200
    save_destinations = res.json()["save_destinations"]
    assert len(save_destinations) >= 1, (
        "/api/settings save_destinations is empty even though one brief "
        "is seeded — outer try/except is swallowing an exception again."
    )
    role_titles = [d.get("role_title") for d in save_destinations]
    assert "Settings Fan-Out Test Role" in role_titles


def test_settings_includes_cdp_url(client):
    res = client.get("/api/settings")
    body = res.json()
    assert "cdp_url" in body
    assert isinstance(body["cdp_url"], str)
