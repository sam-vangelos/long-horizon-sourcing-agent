"""Tests for the Phase D Slice D9 launch-readiness HTTP endpoint
(``GET /api/launch-readiness/{source}/{brief_id}``). Pins the contract:

- Returns a structured ``LaunchReadinessResponse`` with ``ready`` +
  ``blockers`` echoed back.
- Wraps :func:`linkedin.health.probe_linkedin_readiness` and
  :func:`github.health.probe_github_readiness` (substrate from D-prep).
- 422 ``unknown_source`` for sources outside the registered set.
- The ``brief_id`` path segment is opaque for Phase D — captured but
  not consulted. Phase F's per-brief save-destination check will start
  using it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cloris.app import create_app
import linkedin.health as linkedin_health
import github.health as github_health


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def _stub_ready(monkeypatch) -> None:
    """Force both probes to report ready=True with no blockers."""

    monkeypatch.setattr(
        linkedin_health,
        "probe_linkedin_readiness",
        lambda **_kwargs: linkedin_health.ReadinessReport(ready=True, blockers=()),
    )
    monkeypatch.setattr(
        github_health,
        "probe_github_readiness",
        lambda **_kwargs: github_health.ReadinessReport(ready=True, blockers=()),
    )


def test_linkedin_readiness_returns_ready_when_probe_passes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_ready(monkeypatch)

    response = client.get("/api/launch-readiness/linkedin/some-brief-id")

    assert response.status_code == 200
    body = response.json()
    assert body["slice"] == "v0-launch-readiness-1"
    assert body["source"] == "linkedin"
    assert body["brief_id"] == "some-brief-id"
    assert body["ready"] is True
    assert body["blockers"] == []


def test_linkedin_readiness_surfaces_blockers(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = linkedin_health.ReadinessBlocker(
        kind="net",
        message="Cloris can't reach Chrome over CDP.",
        remediation="Run ./launch-chrome.sh --force, open linkedin.com/talent.",
    )
    monkeypatch.setattr(
        linkedin_health,
        "probe_linkedin_readiness",
        lambda **_kwargs: linkedin_health.ReadinessReport(
            ready=False, blockers=(blocker,)
        ),
    )
    monkeypatch.setattr(
        github_health,
        "probe_github_readiness",
        lambda **_kwargs: github_health.ReadinessReport(ready=True, blockers=()),
    )

    response = client.get("/api/launch-readiness/linkedin/brief-1")

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert len(body["blockers"]) == 1
    blocker_out = body["blockers"][0]
    assert blocker_out["kind"] == "net"
    assert blocker_out["code"] == ""
    assert "Chrome" in blocker_out["message"]
    assert "launch-chrome" in blocker_out["remediation"]


def test_github_readiness_dispatches_to_github_probe(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = github_health.ReadinessBlocker(
        kind="config",
        message="No GitHub token configured.",
        remediation="Add GITHUB_TOKEN to your .env file.",
    )
    monkeypatch.setattr(
        github_health,
        "probe_github_readiness",
        lambda **_kwargs: github_health.ReadinessReport(
            ready=False, blockers=(blocker,)
        ),
    )
    monkeypatch.setattr(
        linkedin_health,
        "probe_linkedin_readiness",
        lambda **_kwargs: linkedin_health.ReadinessReport(ready=True, blockers=()),
    )

    response = client.get("/api/launch-readiness/github/brief-1")

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "github"
    assert body["ready"] is False
    assert body["blockers"][0]["kind"] == "config"
    assert body["blockers"][0]["code"] == ""


def test_unknown_source_returns_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe an unknown source. Uses a synthetic name so this test
    keeps testing rejection as the registered source set grows
    (researcher, designer, …)."""

    _stub_ready(monkeypatch)

    response = client.get(
        "/api/launch-readiness/nonexistent_source_for_test_only/brief-1"
    )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"] == "unknown_source"
    assert body["detail"]["source"] == "nonexistent_source_for_test_only"
    assert "linkedin" in body["detail"]["allowed"]
    assert "github" in body["detail"]["allowed"]


def test_brief_id_with_slashes_is_passed_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``:path`` URL converter accepts encoded slashes so a brief
    identifier like ``config/<brief>/brief.json`` rides cleanly. For
    Phase D the brief_id is captured but not consulted by the probes."""

    _stub_ready(monkeypatch)

    response = client.get(
        "/api/launch-readiness/linkedin/config/Forward-Deployed-Engineer-NYC/brief-fde.json"
    )

    assert response.status_code == 200
    body = response.json()
    assert (
        body["brief_id"]
        == "config/Forward-Deployed-Engineer-NYC/brief-fde.json"
    )
    assert body["ready"] is True
