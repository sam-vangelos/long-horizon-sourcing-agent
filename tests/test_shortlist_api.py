"""Tests for Executive Search Slice 7 — shortlist surface (downgraded scope).

Pins the contract for the read-side shortlist API. The Cloris-native
``shortlist_entries`` table + ``AbstractSaveDestination`` write path
depend on multi-module-foundation Slices 6-7 (workspace tables +
AbstractSaveDestination), which are NOT shipped — Slice 7's scope is
explicitly downgraded per the spec's "downgrade or absorb" rule.

What Slice 7 ships:

- :class:`ShortlistResponse` wire shape with the saves-shape alarm
  fields.
- ``GET /api/shortlist/{brief_id}`` route returning the response.
- ``saves_shape_alarm`` flips when ``len(candidates) > 25``.
- 404 when no state_dir holds the brief.

What Slice 7 explicitly defers:

- Save-destination write path (waits for mfm Slice 7).
- ``shortlist_entries`` table (waits for mfm Slice 6).
- Frontend ``Shortlist.svelte`` page (a follow-up; this slice ships
  the read API only).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from cloris.app import create_app
from cloris.models import (
    EXEC_SEARCH_SAVES_SHAPE_THRESHOLD,
    CandidateCardSummary,
    LatestRunRef,
    ShortlistResponse,
    WorkspaceResponse,
)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def _candidate(idx: int) -> CandidateCardSummary:
    return CandidateCardSummary(
        candidate_id=idx,
        source="exec_search",
        identity_key=f"id-{idx}",
        display_name=f"Candidate {idx}",
        profile_url=f"https://linkedin.com/in/candidate-{idx}",
        terminal_decision="SAVE",
    )


def _workspace_response(*, n_candidates: int) -> WorkspaceResponse:
    return WorkspaceResponse(
        brief_id="exec-vp-engineering",
        sources=["exec_search"],
        brief_role_title="VP Engineering",
        brief_linkedin_project=None,
        latest_run=LatestRunRef(
            source="exec_search",
            state_key="exec-vp-engineering",
            run_id=42,
        ),
        total_saves=n_candidates,
        candidates=[_candidate(i) for i in range(n_candidates)],
    )


# ---------------------------------------------------------------------------
# Threshold constant
# ---------------------------------------------------------------------------


def test_threshold_constant_is_25() -> None:
    assert EXEC_SEARCH_SAVES_SHAPE_THRESHOLD == 25


# ---------------------------------------------------------------------------
# /api/shortlist 404
# ---------------------------------------------------------------------------


def test_shortlist_returns_404_when_brief_has_no_runs(client: TestClient) -> None:
    with patch(
        "cloris.api.candidate_routes.aggregate_workspace",
        return_value=None,
    ):
        response = client.get("/api/shortlist/missing-brief")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"]["error"] == "shortlist_not_found"
    assert body["detail"]["brief_id"] == "missing-brief"


# ---------------------------------------------------------------------------
# Saves-shape alarm
# ---------------------------------------------------------------------------


def test_shortlist_alarm_silent_below_threshold(client: TestClient) -> None:
    """A reasonable exec search returns ~10 candidates. Alarm stays
    silent."""

    with patch(
        "cloris.api.candidate_routes.aggregate_workspace",
        return_value=_workspace_response(n_candidates=10),
    ):
        response = client.get("/api/shortlist/exec-vp-engineering")
    assert response.status_code == 200
    body = response.json()
    assert body["saves_shape_alarm"] is False
    assert body["saves_shape_alarm_threshold"] == 25
    assert len(body["candidates"]) == 10


def test_shortlist_alarm_silent_at_threshold(client: TestClient) -> None:
    """Exactly 25 saves: still acceptable — alarm fires above 25."""

    with patch(
        "cloris.api.candidate_routes.aggregate_workspace",
        return_value=_workspace_response(n_candidates=25),
    ):
        response = client.get("/api/shortlist/exec-vp-engineering")
    body = response.json()
    assert body["saves_shape_alarm"] is False


def test_shortlist_alarm_fires_above_threshold(client: TestClient) -> None:
    """A run that produced > 25 saves looks more like a high-volume
    search than an exec search. Alarm fires."""

    with patch(
        "cloris.api.candidate_routes.aggregate_workspace",
        return_value=_workspace_response(n_candidates=40),
    ):
        response = client.get("/api/shortlist/exec-vp-engineering")
    assert response.status_code == 200
    body = response.json()
    assert body["saves_shape_alarm"] is True
    assert body["saves_shape_alarm_threshold"] == 25
    assert len(body["candidates"]) == 40


# ---------------------------------------------------------------------------
# Wire shape compat
# ---------------------------------------------------------------------------


def test_shortlist_response_carries_brief_metadata(client: TestClient) -> None:
    with patch(
        "cloris.api.candidate_routes.aggregate_workspace",
        return_value=_workspace_response(n_candidates=3),
    ):
        response = client.get("/api/shortlist/exec-vp-engineering")
    body = response.json()
    assert body["brief_id"] == "exec-vp-engineering"
    assert body["brief_role_title"] == "VP Engineering"
    assert body["sources"] == ["exec_search"]
    assert body["total_saves"] == 3
    assert body["candidates"][0]["display_name"] == "Candidate 0"


def test_shortlist_response_model_round_trip() -> None:
    """Pydantic round-trip on the wire shape is stable across versions."""

    payload = {
        "brief_id": "x",
        "sources": ["exec_search"],
        "brief_role_title": "VP Eng",
        "total_saves": 0,
        "candidates": [],
        "saves_shape_alarm": False,
        "saves_shape_alarm_threshold": 25,
    }
    out = ShortlistResponse(**payload)
    assert out.brief_id == "x"
    assert out.sources == ["exec_search"]
    assert out.saves_shape_alarm is False
    assert out.saves_shape_alarm_threshold == 25
