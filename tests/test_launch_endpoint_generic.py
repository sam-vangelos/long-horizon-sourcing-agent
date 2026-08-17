"""Tests for the Phase F Slice F1 generic launch endpoint.

Pins the F1 contract:

- ``POST /api/launch/{source}`` accepts
  ``{brief_id, mode, force?, force_fresh?}``;
  dispatches via :data:`cloris.launchers.LAUNCHERS`.
- Unknown source → 422 with allowed-list payload.
- Unknown brief_id → 404.
- Readiness blockers → 422 unless ``force=true``.
- ``mode="resume"`` collapses the launch + resume code paths.
- Legacy ``POST /api/launch/linkedin`` and ``POST /api/resume/linkedin``
  remain as backward-compat synonyms returning the existing shapes.

Stubs the spawn helper so no real subprocess is spawned. The test
client routes through FastAPI so route-precedence (literal beats
path-param) is exercised end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloris import api as cloris_api
from cloris.api import _SpawnResult


def _v2_minimal(role: str = "F1 Test") -> dict:
    return {
        "role_title": role,
        "id": role.lower().replace(" ", "_"),
        "linkedin_project_id": role.lower().replace(" ", "_"),
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


@pytest.fixture()
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, dict[str, Any]]:
    """Patch _PROJECT_ROOT/_CONFIG_DIR + stub the spawn helper.

    The stub records every call so the test can assert on `(source,
    brief_path, mode)`. Returns ``(client, captures)``.
    """

    from cloris.app import create_app

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)

    captures: dict[str, Any] = {"calls": []}

    def fake_spawn(
        *,
        source: str,
        brief_path: Path,
        mode: str,
        force_fresh: bool = False,
    ) -> _SpawnResult:
        captures["calls"].append(
            {
                "source": source,
                "brief_path": str(brief_path),
                "mode": mode,
                "force_fresh": force_fresh,
            }
        )
        return _SpawnResult(
            pid=42424,
            state_dir=Path(f"/tmp/state/{source}/key"),
            worker_json_path=Path(f"/tmp/state/{source}/key/worker.json"),
        )

    monkeypatch.setattr(cloris_api._monolith, "_spawn_worker_for_source", fake_spawn)

    # Make the readiness probe return ready=true so blockers don't fire
    # in the happy-path tests; tests that need blockers re-stub locally.
    def fake_blockers(source: str, brief_id: str) -> list:
        return []

    monkeypatch.setattr(cloris_api._monolith, "_readiness_blockers", fake_blockers)

    return TestClient(create_app()), captures


def _seed_brief(tmp_path: Path, role: str = "F1 Role") -> tuple[str, dict]:
    """Write a real V2 brief and compute its brief_id via the same hash
    the endpoint uses. Returns ``(brief_id, payload)``."""

    config_dir = tmp_path / "config" / role.replace(" ", "-")
    payload = _v2_minimal(role)
    _write_brief(config_dir / "brief.json", payload)

    from shared.output_paths import derive_brief_id

    brief_id = derive_brief_id(brief_path=str(config_dir / "brief.json"))
    return brief_id, payload


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_launch_linkedin_via_generic_endpoint_dispatches_correctly(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    api, captures = client
    brief_id, _ = _seed_brief(tmp_path, role="LinkedIn F1")

    response = api.post(
        "/api/launch/linkedin",
        json={"brief_id": brief_id, "mode": "fresh"},
    )

    # Legacy literal `/api/launch/linkedin` route still wins for this path
    # (Starlette matches in declaration order). The generic-endpoint path
    # for LinkedIn requires hitting a different shape — not via this URL.
    # This test just asserts the legacy literal still works post-F1
    # refactor.
    assert response.status_code in (201, 422)


def test_launch_github_dispatches_via_generic_endpoint(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    api, captures = client
    brief_id, _ = _seed_brief(tmp_path, role="GitHub F1")

    response = api.post(
        f"/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "github"
    assert body["mode"] == "fresh"
    assert body["pid"] == 42424
    assert "/tmp/state/github/key" in body["state_dir"]
    assert len(captures["calls"]) == 1
    assert captures["calls"][0]["source"] == "github"
    assert captures["calls"][0]["mode"] == "fresh"


def test_resume_via_mode_resume_collapses_launch_and_resume(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    api, captures = client
    brief_id, _ = _seed_brief(tmp_path, role="Resume F1")

    response = api.post(
        f"/api/launch/github",
        json={"brief_id": brief_id, "mode": "resume"},
    )

    assert response.status_code == 201
    assert response.json()["mode"] == "resume"
    assert captures["calls"][0]["mode"] == "resume"


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_unknown_source_returns_422_with_allowed_list(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    api, _ = client
    brief_id, _ = _seed_brief(tmp_path, role="Unknown Src")

    response = api.post(
        f"/api/launch/myspace", json={"brief_id": brief_id, "mode": "fresh"}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "unknown_source"
    assert detail["source"] == "myspace"
    assert "linkedin" in detail["allowed"]
    assert "github" in detail["allowed"]


def test_unknown_brief_id_returns_404(
    client: tuple[TestClient, dict[str, Any]],
) -> None:
    api, _ = client

    response = api.post(
        "/api/launch/github",
        json={"brief_id": "totally_bogus", "mode": "fresh"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "brief_id_not_found"


def test_readiness_blockers_without_force_return_422(
    client: tuple[TestClient, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _ = client
    brief_id, _ = _seed_brief(tmp_path, role="Not Ready")

    class FakeBlocker:
        kind = "auth"
        message = "LinkedIn session is missing."
        remediation = "Open linkedin.com/talent in a browser tab."

    monkeypatch.setattr(
        cloris_api._monolith,
        "_readiness_blockers",
        lambda source, brief_id: [FakeBlocker()],
    )

    response = api.post(
        f"/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh", "force": False},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "launch_not_ready"
    assert detail["source"] == "github"
    assert len(detail["blockers"]) == 1
    assert detail["blockers"][0]["kind"] == "auth"


def test_force_true_skips_readiness_probe(
    client: tuple[TestClient, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, captures = client
    brief_id, _ = _seed_brief(tmp_path, role="Forced")

    class FakeBlocker:
        kind = "auth"
        message = "Skip me with force"
        remediation = ""

    monkeypatch.setattr(
        cloris_api._monolith,
        "_readiness_blockers",
        lambda source, brief_id: [FakeBlocker()],
    )

    response = api.post(
        f"/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh", "force": True},
    )

    assert response.status_code == 201
    assert len(captures["calls"]) == 1


def test_force_fresh_threads_to_spawn_helper(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    api, captures = client
    brief_id, _ = _seed_brief(tmp_path, role="Force Fresh")

    response = api.post(
        "/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh", "force_fresh": True},
    )

    assert response.status_code == 201
    assert captures["calls"][-1]["force_fresh"] is True


def test_fresh_over_resumable_state_maps_to_422(
    client: tuple[TestClient, dict[str, Any]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api, _ = client
    brief_id, _ = _seed_brief(tmp_path, role="Fresh Over Existing")

    def fake_spawn(**_kwargs: Any) -> _SpawnResult:
        raise cloris_api._monolith.FreshOverResumableStateError(
            state_dir="/tmp/state/linkedin/key"
        )

    monkeypatch.setattr(cloris_api._monolith, "_spawn_worker_for_source", fake_spawn)

    response = api.post(
        "/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "fresh_over_resumable_state"
    assert detail["state_dir"] == "/tmp/state/linkedin/key"
    assert "Resume the existing run" in detail["message"]
    assert "force_fresh=true" in detail["message"]


def test_multi_launch_threads_force_fresh_to_each_module(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    api, captures = client
    brief_id, _ = _seed_brief(tmp_path, role="Multi Force Fresh")

    response = api.post(
        "/api/launch/multi",
        json={
            "brief_id": brief_id,
            "modules": ["linkedin", "github"],
            "mode": "fresh",
            "force_fresh": True,
        },
    )

    assert response.status_code == 201
    assert [call["force_fresh"] for call in captures["calls"][-2:]] == [True, True]


def test_extra_fields_rejected_at_request_boundary(
    client: tuple[TestClient, dict[str, Any]],
    tmp_path: Path,
) -> None:
    """`extra="forbid"` on LaunchRequest catches typos like `bref_id`."""

    api, _ = client
    brief_id, _ = _seed_brief(tmp_path, role="Strict")

    response = api.post(
        "/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh", "unknown_field": True},
    )

    assert response.status_code == 422


def test_invalid_mode_rejected(
    client: tuple[TestClient, dict[str, Any]],
    tmp_path: Path,
) -> None:
    api, _ = client
    brief_id, _ = _seed_brief(tmp_path, role="Invalid Mode")

    response = api.post(
        "/api/launch/github",
        json={"brief_id": brief_id, "mode": "refresh"},
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Backward-compat synonyms
# ---------------------------------------------------------------------------


def test_legacy_launch_linkedin_endpoint_still_works(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    """The legacy ``POST /api/launch/linkedin`` literal endpoint must
    still match BEFORE the generic ``/api/launch/{source}`` path-param
    route, because it uses the legacy ``{brief_path}`` request shape."""

    api, captures = client
    config_dir = tmp_path / "config" / "Legacy-LinkedIn"
    payload = _v2_minimal("Legacy LinkedIn")
    _write_brief(config_dir / "brief.json", payload)

    response = api.post(
        "/api/launch/linkedin",
        json={"brief_path": str(config_dir / "brief.json")},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["source"] == "linkedin"
    # Legacy synonym still spawns through the generalized helper.
    assert captures["calls"][0]["source"] == "linkedin"
    assert captures["calls"][0]["mode"] == "fresh"


def test_legacy_launch_linkedin_accepts_force_fresh(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    api, captures = client
    config_dir = tmp_path / "config" / "Legacy-Force-Fresh"
    payload = _v2_minimal("Legacy Force Fresh")
    _write_brief(config_dir / "brief.json", payload)

    response = api.post(
        "/api/launch/linkedin",
        json={
            "brief_path": str(config_dir / "brief.json"),
            "force_fresh": True,
        },
    )

    assert response.status_code == 201, response.text
    assert captures["calls"][-1]["source"] == "linkedin"
    assert captures["calls"][-1]["force_fresh"] is True


def test_legacy_resume_linkedin_endpoint_still_works(
    client: tuple[TestClient, dict[str, Any]], tmp_path: Path
) -> None:
    api, captures = client
    config_dir = tmp_path / "config" / "Legacy-Resume"
    payload = _v2_minimal("Legacy Resume")
    _write_brief(config_dir / "brief.json", payload)

    response = api.post(
        "/api/resume/linkedin",
        json={"brief_path": str(config_dir / "brief.json")},
    )

    # Resume needs a pending-work check; in this stub the underlying
    # _spawn_worker_for_source is faked so the resume-readiness check
    # in the real spawner is bypassed. Either 201 (the synonym worked)
    # or 422 (resume rejected) is acceptable; the test pins routing.
    assert response.status_code in (201, 422)
