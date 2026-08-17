"""Tests for the Phase 0 chrome-launcher API surface
(``GET /api/chrome-status``, ``POST /api/chrome-relaunch``).

Pins the wire contract that the welcome surface's polling loop and
the Settings "Re-open Chrome" affordance depend on. The actual Chrome
spawning logic in ``cloris.chrome_launcher`` is exercised separately
by direct module tests; here we only assert that the API layer
correctly translates the launcher's :class:`ChromeStatus` into the
:class:`ChromeStatusResponse` wire shape.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloris import chrome_launcher
from cloris.app import create_app
from cloris.chrome_launcher import ChromeStatus


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _stub_status(state: str, message: str) -> ChromeStatus:
    return ChromeStatus(
        state=state,  # type: ignore[arg-type]
        cdp_url="http://127.0.0.1:9222",
        profile_dir="/tmp/test/chrome-profile",
        message=message,
    )


def test_chrome_status_translates_healthy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Healthy launcher state surfaces as ``state="healthy"`` on the wire,
    with the launcher's recruiter-readable message echoed verbatim so
    the welcome surface can render it without translation."""

    monkeypatch.setattr(
        chrome_launcher,
        "status",
        lambda: _stub_status("healthy", "Chrome is open and ready."),
    )

    resp = client.get("/api/chrome-status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slice"] == "v0-chrome-status-1"
    assert body["state"] == "healthy"
    assert body["cdp_url"] == "http://127.0.0.1:9222"
    assert body["profile_dir"] == "/tmp/test/chrome-profile"
    assert body["message"] == "Chrome is open and ready."


def test_chrome_status_translates_unhealthy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unhealthy state lets the welcome surface render the
    "Re-open Chrome" affordance with the launcher's diagnostic
    sentence as supporting copy."""

    monkeypatch.setattr(
        chrome_launcher,
        "status",
        lambda: _stub_status(
            "unhealthy",
            "Cloris hasn't opened its Chrome window yet — click the "
            "re-open Chrome control to spawn it.",
        ),
    )

    resp = client.get("/api/chrome-status")
    assert resp.status_code == 200
    assert resp.json()["state"] == "unhealthy"


def test_chrome_status_translates_missing_chrome(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """missing_chrome is an editorial state — the recipient hasn't
    installed Chrome at all. The welcome surface should treat this as
    a hard block (no relaunch button — "go install Chrome first")."""

    monkeypatch.setattr(
        chrome_launcher,
        "status",
        lambda: _stub_status(
            "missing_chrome",
            "Cloris couldn't find Google Chrome on this Mac. Install "
            "Chrome from google.com/chrome and reopen Cloris.",
        ),
    )

    resp = client.get("/api/chrome-status")
    assert resp.status_code == 200
    assert resp.json()["state"] == "missing_chrome"


def test_chrome_relaunch_calls_ensure_running_with_force(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "Re-open Chrome" affordance must invoke
    :func:`chrome_launcher.ensure_running` with ``force=True`` —
    otherwise an ``unhealthy`` state where CDP is reachable-but-
    contextless would short-circuit and never recycle."""

    received: dict[str, Any] = {}

    def fake_ensure_running(*, force: bool = False) -> ChromeStatus:
        received["force"] = force
        return _stub_status("healthy", "Chrome is open and ready.")

    monkeypatch.setattr(chrome_launcher, "ensure_running", fake_ensure_running)

    resp = client.post("/api/chrome-relaunch")
    assert resp.status_code == 200
    assert received == {"force": True}
    assert resp.json()["state"] == "healthy"


def test_chrome_open_linkedin_calls_non_destructive_opener(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LinkedIn readiness CTA opens a Recruiter tab without force-recycling Chrome."""

    received: dict[str, Any] = {}

    def fake_open_linkedin_recruiter() -> ChromeStatus:
        received["called"] = True
        return _stub_status(
            "healthy",
            "LinkedIn Recruiter is opening in Cloris Chrome.",
        )

    monkeypatch.setattr(
        chrome_launcher, "open_linkedin_recruiter", fake_open_linkedin_recruiter
    )

    resp = client.post("/api/chrome-open-linkedin")
    assert resp.status_code == 200
    assert received == {"called": True}
    assert resp.json()["state"] == "healthy"
    assert resp.json()["message"] == "LinkedIn Recruiter is opening in Cloris Chrome."
