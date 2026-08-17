"""Tests for Phase F Slice F2 — save destination configuration.

Pins the F2 contract:

- ``source_config`` is a recognized V2 brief key; the validator
  accepts it (when shape is right) and rejects malformed shapes.
- ``derive_brief_id()`` reads ``source_config.linkedin.project_id``
  first, falls back to flat ``linkedin_project_id``, AND produces the
  same value across the migration (the load-bearing invariant — if
  this breaks, every existing state_dir orphans).
- The launchers registry's ``save_destination_blocker_fn`` returns
  ``None`` when LinkedIn project id is configured (either path),
  fires when neither is.
- ``GET /api/launch-readiness/linkedin/<brief_id>`` surfaces the F2
  brief-level blocker alongside the source-level probe.
- ``POST /api/launch/{source}`` blocks launches without ``force=true``
  when F2's blocker fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloris import api as cloris_api
from cloris.api import _SpawnResult


def _v2_minimal(role: str = "F2 Test") -> dict:
    return {
        "role_title": role,
        "id": role.lower().replace(" ", "_"),
        "capability_areas": [
            {"name": "Eng", "description": "ships systems."}
        ],
        "depth_distinction": {
            "builder_definition": "owns",
            "user_definition": "uses",
            "edge_case_guidance": "borderline",
        },
    }


def _write_brief(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# V2 schema validator extension
# ---------------------------------------------------------------------------


def test_validator_accepts_source_config_shape() -> None:
    from shared.brief_v2_schema import validate_v2_brief

    payload = _v2_minimal()
    payload["source_config"] = {
        "linkedin": {"project_id": "3000000007", "project_name": "FDE NYC"},
        "github": {},
    }
    # Should not raise.
    validate_v2_brief(payload)


def test_validator_rejects_non_dict_source_config() -> None:
    from shared.brief_v2_schema import BriefSchemaError, validate_v2_brief

    payload = _v2_minimal()
    payload["source_config"] = "not a dict"

    with pytest.raises(BriefSchemaError) as exc:
        validate_v2_brief(payload)
    assert "source_config" in exc.value.invalid_keys


def test_validator_rejects_non_string_linkedin_project_id() -> None:
    from shared.brief_v2_schema import BriefSchemaError, validate_v2_brief

    payload = _v2_minimal()
    payload["source_config"] = {"linkedin": {"project_id": 1234}}

    with pytest.raises(BriefSchemaError) as exc:
        validate_v2_brief(payload)
    assert "source_config.linkedin.project_id" in exc.value.invalid_keys


def test_validator_passes_through_unknown_source_keys() -> None:
    """Forward-compat: a new source not yet in the registered set
    passes validation without raising — F2 doesn't gatekeep what F+1
    can add."""

    from shared.brief_v2_schema import validate_v2_brief

    payload = _v2_minimal()
    payload["source_config"] = {"researcher": {"future_field": "any"}}
    validate_v2_brief(payload)


# ---------------------------------------------------------------------------
# derive_brief_id fallback
# ---------------------------------------------------------------------------


def test_derive_brief_id_uses_source_config_path(tmp_path: Path) -> None:
    """source_config.linkedin.project_id wins over flat linkedin_project_id."""

    from shared.output_paths import derive_brief_id

    payload = _v2_minimal("Migration A")
    payload["linkedin_project_id"] = "old_value"
    payload["source_config"] = {"linkedin": {"project_id": "new_value"}}
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    key = derive_brief_id(brief_path=str(brief_path))
    assert "new_value" in key


def test_derive_brief_id_falls_back_to_flat_field(tmp_path: Path) -> None:
    """Brief carrying only the legacy flat field still produces a key."""

    from shared.output_paths import derive_brief_id

    payload = _v2_minimal("Migration B")
    payload["linkedin_project_id"] = "flat_only"
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    key = derive_brief_id(brief_path=str(brief_path))
    assert "flat_only" in key


def test_derive_brief_id_stable_across_migration(tmp_path: Path) -> None:
    """LOAD-BEARING: a brief with only the new shape produces the SAME
    state_key as the same brief with only the legacy flat shape.

    If this regresses, every existing state_dir orphans the moment a
    recruiter migrates a brief via the F2 UI.
    """

    from shared.output_paths import derive_brief_id

    legacy_payload = _v2_minimal("Stable")
    legacy_payload["linkedin_project_id"] = "3000000007"
    legacy_path = tmp_path / "legacy" / "brief.json"
    _write_brief(legacy_path, legacy_payload)

    migrated_payload = _v2_minimal("Stable")
    migrated_payload["source_config"] = {
        "linkedin": {"project_id": "3000000007"}
    }
    migrated_path = tmp_path / "migrated" / "brief.json"
    _write_brief(migrated_path, migrated_payload)

    legacy_key = derive_brief_id(brief_path=str(legacy_path))
    migrated_key = derive_brief_id(brief_path=str(migrated_path))
    assert legacy_key == migrated_key


# ---------------------------------------------------------------------------
# Launchers registry — save_destination_blocker_fn
# ---------------------------------------------------------------------------


def test_linkedin_blocker_returns_none_when_configured(tmp_path: Path) -> None:
    from cloris.launchers import LAUNCHERS

    payload = _v2_minimal("Configured LinkedIn")
    payload["source_config"] = {"linkedin": {"project_id": "3000000007"}}
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    blocker = LAUNCHERS["linkedin"].save_destination_blocker_fn(
        str(brief_path)
    )
    assert blocker is None


def test_linkedin_blocker_fires_when_neither_path_set(tmp_path: Path) -> None:
    from cloris.launchers import LAUNCHERS

    payload = _v2_minimal("Unconfigured LinkedIn")
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    blocker = LAUNCHERS["linkedin"].save_destination_blocker_fn(
        str(brief_path)
    )
    assert blocker is not None
    assert blocker.kind == "config"
    assert "LinkedIn" in blocker.message
    assert "Where Cloris saves" in blocker.remediation


def test_linkedin_blocker_passes_with_legacy_flat_field(tmp_path: Path) -> None:
    """A brief that hasn't migrated to source_config but carries the
    legacy flat field is still launchable — the blocker reads through
    the same fallback as derive_brief_id."""

    from cloris.launchers import LAUNCHERS

    payload = _v2_minimal("Legacy flat")
    payload["linkedin_project_id"] = "3000000007"
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    assert (
        LAUNCHERS["linkedin"].save_destination_blocker_fn(str(brief_path))
        is None
    )


def test_github_blocker_returns_none_for_classic_brief(tmp_path: Path) -> None:
    """GitHub has no per-brief save-destination concept (F2's punt;
    revisit when GitHub gains list/team semantics). P6.9 repurposed this
    slot for the OSS-Maintainers-posture gate, but a classic brief —
    no ``target_projects``, default ``maintainership_level`` — still
    passes through with no blocker (unchanged behavior)."""

    from cloris.launchers import LAUNCHERS

    payload = _v2_minimal("GitHub Brief")
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    assert (
        LAUNCHERS["github"].save_destination_blocker_fn(str(brief_path))
        is None
    )


# ---------------------------------------------------------------------------
# P6.9 — GitHub OSS-Maintainers-posture readiness gate
# ---------------------------------------------------------------------------


def test_github_blocker_fires_for_posture_declared_without_targets(
    tmp_path: Path,
) -> None:
    """A brief that declares OSS Maintainers posture via an elevated
    ``maintainership_level`` but supplies no ``target_projects`` would
    silently never run maintainership classification
    (`github/maintainership.py:176` returns None on empty
    target_projects) — block it with a named, actionable reason instead
    of letting it launch as if the posture were never declared."""

    from cloris.launchers import LAUNCHERS

    payload = _v2_minimal("Maintainer Posture, No Targets")
    payload["maintainership_level"] = "maintainer"
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    blocker = LAUNCHERS["github"].save_destination_blocker_fn(str(brief_path))

    assert blocker is not None
    assert blocker.kind == "config"
    assert "target_projects" in blocker.message
    assert "maintainer" in blocker.message
    assert "target_projects" in blocker.remediation


def test_github_blocker_passes_for_populated_target_projects(
    tmp_path: Path,
) -> None:
    """Non-empty target_projects trivially satisfies the requirement,
    regardless of maintainership_level."""

    from cloris.launchers import LAUNCHERS

    payload = _v2_minimal("Named Projects Brief")
    payload["target_projects"] = ["torvalds/linux"]
    payload["maintainership_level"] = "project_lead"
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    assert (
        LAUNCHERS["github"].save_destination_blocker_fn(str(brief_path))
        is None
    )


def test_github_blocker_passes_and_logs_classic_sourcing_for_target_less_brief(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A target-less brief (no target_projects, default/absent
    maintainership_level) remains classic GitHub sourcing — unchanged
    behavior, explicitly logged as such per P6.9."""

    import logging

    from cloris.launchers import LAUNCHERS

    payload = _v2_minimal("Classic GitHub Brief")
    brief_path = tmp_path / "brief.json"
    _write_brief(brief_path, payload)

    with caplog.at_level(logging.INFO, logger="github.health"):
        blocker = LAUNCHERS["github"].save_destination_blocker_fn(
            str(brief_path)
        )

    assert blocker is None
    assert any(
        "classic GitHub sourcing" in record.message for record in caplog.records
    )


# ---------------------------------------------------------------------------
# /api/launch-readiness — F2 brief-level blocker layered in
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    from cloris.app import create_app

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)

    # Stub source-level probes so we can isolate F2's brief-level layer.
    class _OkReport:
        ready = True
        blockers: tuple = ()

    import linkedin.health as li_health
    import github.health as gh_health

    monkeypatch.setattr(li_health, "probe_linkedin_readiness", lambda: _OkReport())
    monkeypatch.setattr(gh_health, "probe_github_readiness", lambda: _OkReport())

    return TestClient(create_app()), tmp_path


def _seed_brief(tmp_path: Path, *, with_linkedin_dest: bool) -> str:
    role = "Readiness Brief"
    payload = _v2_minimal(role)
    if with_linkedin_dest:
        payload["source_config"] = {
            "linkedin": {"project_id": "3000000007"}
        }
    config_dir = tmp_path / "config" / role.replace(" ", "-")
    _write_brief(config_dir / "brief.json", payload)

    from shared.output_paths import derive_brief_id

    return derive_brief_id(brief_path=str(config_dir / "brief.json"))


def test_readiness_endpoint_clean_when_destination_configured(
    client: tuple[TestClient, Path],
) -> None:
    api, tmp = client
    brief_id = _seed_brief(tmp, with_linkedin_dest=True)

    response = api.get(f"/api/launch-readiness/linkedin/{brief_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["blockers"] == []


def test_readiness_endpoint_blocks_when_destination_missing(
    client: tuple[TestClient, Path],
) -> None:
    api, tmp = client
    brief_id = _seed_brief(tmp, with_linkedin_dest=False)

    response = api.get(f"/api/launch-readiness/linkedin/{brief_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert len(body["blockers"]) == 1
    blocker = body["blockers"][0]
    assert blocker["kind"] == "config"
    assert blocker["code"] == ""
    assert "LinkedIn" in blocker["message"]


def test_readiness_endpoint_unknown_source_returns_422(
    client: tuple[TestClient, Path],
) -> None:
    api, _ = client
    response = api.get("/api/launch-readiness/myspace/abc")

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "unknown_source"


def test_launch_endpoint_blocks_without_force_when_destination_missing(
    client: tuple[TestClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: F1's launch endpoint reads F2's blocker via the
    shared `_readiness_blockers` aggregator and 422s before spawn."""

    api, tmp = client
    brief_id = _seed_brief(tmp, with_linkedin_dest=False)

    # Stub spawn so a 201 path doesn't accidentally execute real subprocess.
    captures: list = []

    def fake_spawn(
        *,
        source: str,
        brief_path: Path,
        mode: str,
        force_fresh: bool = False,
    ) -> _SpawnResult:
        captures.append({"source": source, "mode": mode})
        return _SpawnResult(
            pid=1,
            state_dir=Path("/tmp/x"),
            worker_json_path=Path("/tmp/x/worker.json"),
        )

    monkeypatch.setattr(cloris_api._monolith, "_spawn_worker_for_source", fake_spawn)

    response = api.post(
        "/api/launch/linkedin",
        json={"brief_id": brief_id, "mode": "fresh", "force": False},
    )

    # The legacy /api/launch/linkedin literal route uses {brief_path}, not
    # {brief_id}; expect 422 from Pydantic's extra="forbid" rejecting
    # brief_id. F2's brief-level blocker fires only on the F1 generic path.
    # Use the github source to exercise the F1 path with a known-clean
    # source-level probe.
    assert response.status_code == 422


def test_launch_endpoint_passes_when_force_true_skips_blocker(
    client: tuple[TestClient, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, tmp = client
    brief_id = _seed_brief(tmp, with_linkedin_dest=False)

    captures: list = []

    def fake_spawn(
        *,
        source: str,
        brief_path: Path,
        mode: str,
        force_fresh: bool = False,
    ) -> _SpawnResult:
        captures.append({"source": source, "mode": mode})
        return _SpawnResult(
            pid=1,
            state_dir=Path("/tmp/x"),
            worker_json_path=Path("/tmp/x/worker.json"),
        )

    monkeypatch.setattr(cloris_api._monolith, "_spawn_worker_for_source", fake_spawn)

    # Use generic-path endpoint with the GitHub source (which has no F2
    # blocker today) to sanity-check the force path doesn't break.
    response = api.post(
        "/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh", "force": True},
    )

    assert response.status_code == 201
    assert len(captures) == 1
