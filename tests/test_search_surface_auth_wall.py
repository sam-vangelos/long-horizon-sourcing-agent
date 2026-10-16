"""CLO-243 integration: search_surface's fail-soft re-anchor paths never swallow a wall.

``reanchor_search_surface`` and the page-local ``_reanchor_recovery_page``
are fail-soft on purpose: a navigation that cannot repair the surface leaves
the following gates to refuse on their own evidence. A LinkedIn sign-in wall
is not a repair failure, though. Swallowed there, the readiness gate that
follows refuses generically (``search_surface_unready``), the evaluation loop
reads that as a facet strike against the string, re-anchors into the same
wall for the next string, and the run ends three strings later as
``facet_surface_unreconcilable`` with the wall's name lost.

These tests drive the real ``SearchSurface`` methods; only the browser is a
stand-in, and its ``navigate_to_search`` raises the typed ``AuthWallError``
exactly as ``LinkedInBrowser.navigate_to_search`` does on a wall page.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.auth_wall import AuthWallError
from linkedin.browser import FacetReconciliationError
from linkedin.evaluation_loop import EvaluationLoop
from linkedin.evaluation_state import EvaluationRunState
from linkedin.search_intelligence import bootstrap_experiment_state
from shared import config
from shared.schemas import SearchString
from tests.test_linkedin_pipeline import _facet_containment_pipeline

WALL_URL = "https://www.linkedin.com/enterprise-authentication/sessions"
SEARCH_URL = "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"


def _wall() -> AuthWallError:
    return AuthWallError("session_conflict", url=WALL_URL)


def test_a_wall_met_while_reanchoring_after_a_facet_strike_ends_the_run_by_name():
    """Production path: EvaluationLoop.run_full -> facet strike -> the real
    reanchor_search_surface -> navigate_to_search lands on the wall.

    RED without the re-raise: the wall is swallowed, every string strikes on
    the generic readiness refusal, and the run stops three strings later as
    facet_surface_unreconcilable (a GovernorLimitReached that names no wall).
    """
    with tempfile.TemporaryDirectory() as td:
        p = _facet_containment_pipeline(td, string_count=3)
        p.state._safety.record_governor_limit = MagicMock()
        wall = _wall()
        p.state.context.browser.navigate_to_search = AsyncMock(side_effect=wall)
        processed_ids: list[int] = []

        async def fake_process(search_string, _progress):
            processed_ids.append(search_string.id)
            raise FacetReconciliationError("search_surface_unready")

        p.state.driver._process_string = fake_process

        with patch.object(
            config, "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True
        ), pytest.raises(AuthWallError) as raised:
            asyncio.run(p.run_full(resume=True))

        assert raised.value is wall
        # One string, one re-anchor: nothing after the wall touches the page.
        assert processed_ids == [1]
        p.state.context.browser.navigate_to_search.assert_awaited_once_with(SEARCH_URL)
        p.state._safety.record_governor_limit.assert_not_called()
        with sqlite3.connect(Path(td, "runtime_state.sqlite3")) as conn:
            run = conn.execute(
                "SELECT status, stop_reason, stop_reason_detail FROM runs "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert run[0] == "error"
        assert run[1] == "fatal_runtime_error"
        # Slice 1's detail carries slice 2's name for the wall.
        assert run[2].startswith(
            "runtime: AuthWallError: recruiter_reauth_required: session_conflict: "
        )


def test_a_marked_surface_reanchor_into_a_wall_raises_before_the_readiness_gate():
    """Opening path: a zero-result string marked the surface, the next
    opening re-anchors first, and the navigation lands on the wall.

    RED without the re-raise: the wall is swallowed and the readiness gate's
    generic refusal (a FacetReconciliationError) is what escapes.
    """
    with tempfile.TemporaryDirectory() as td:
        p = _facet_containment_pipeline(td, string_count=1)
        p.state.reviewer.attempts._record_runtime_event = MagicMock()
        wall = _wall()
        browser = p.state.context.browser
        browser.navigate_to_search = AsyncMock(side_effect=wall)
        browser.assert_search_surface_ready = AsyncMock(
            side_effect=FacetReconciliationError("search_surface_unready")
        )
        search_string = MagicMock(id=41, acquisition_mode="linkedin_boolean")
        filters = MagicMock(companies=[], titles=[])
        filters.is_empty.return_value = True
        experiment_state = MagicMock()
        experiment_state.active_variant.structured_filters = filters

        p.state.surface.mark_surface_for_reanchor("zero_results")
        with patch.object(
            config, "LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True
        ), pytest.raises(AuthWallError) as raised:
            asyncio.run(
                p.state.surface._apply_opening_search(
                    search_string, experiment_state, "python"
                )
            )

        assert raised.value is wall
        browser.assert_search_surface_ready.assert_not_awaited()
        # The flag is spent even though the re-anchor raised.
        assert p.state.surface.surface_reanchor_pending() == ""
        # The opening's failure receipt files the wall under the readiness
        # step the re-anchor serves, never under the compile-time location
        # check that had already passed.
        payload = p.state.reviewer.attempts._record_runtime_event.call_args.kwargs[
            "payload"
        ]
        assert payload["failure_gate"] == "search_surface_unready"


def test_page_local_recovery_reanchor_reraises_the_wall_itself():
    """_reanchor_recovery_page catches RuntimeError around its navigation,
    and AuthWallError is a RuntimeError. The wall must leave as itself, not
    as the health check's SurfaceOwnershipLostError raised over it.
    """
    loop = EvaluationLoop(EvaluationRunState())
    loop.state.surface.budget.enabled = True
    owner = SearchString(id=1, name="owner", boolean="x")
    loop.state._experiment_states = {1: bootstrap_experiment_state(owner)}
    wall = _wall()
    loop.state.context.browser = SimpleNamespace(
        navigate_to_search=AsyncMock(side_effect=wall),
        page=SimpleNamespace(url=WALL_URL),
    )
    loop.state.surface._get_project_url = lambda: SEARCH_URL
    loop.state.surface._check_page_recovery_health = AsyncMock()
    loop.state.surface.geography = SimpleNamespace(
        reset_location_applied=MagicMock(),
        _apply_session_location_filter=AsyncMock(),
    )
    loop.state.surface._apply_opening_search = AsyncMock()

    with pytest.raises(AuthWallError) as raised:
        asyncio.run(loop.state.surface._reanchor_recovery_page(owner, 2, "root"))

    assert raised.value is wall
    loop.state.surface._check_page_recovery_health.assert_not_awaited()
    loop.state.surface.geography.reset_location_applied.assert_not_called()
    loop.state.surface._apply_opening_search.assert_not_awaited()


@pytest.mark.parametrize("stage", ["connect", "navigate"])
def test_reconnect_never_retries_into_a_wall(stage):
    """_attempt_reconnect swallows every failure and tries again; a wall
    must end it on first sight instead of six more navigations into it."""
    loop = EvaluationLoop(EvaluationRunState())
    wall = _wall()
    browser = SimpleNamespace(
        disconnect=AsyncMock(),
        connect=AsyncMock(side_effect=wall if stage == "connect" else None),
        navigate_to_search=AsyncMock(side_effect=wall),
    )
    loop.state.context.browser = browser

    with pytest.raises(AuthWallError) as raised:
        asyncio.run(
            loop.state.surface._attempt_reconnect(
                recovery_url=SEARCH_URL, max_attempts=3, wait_seconds=0
            )
        )

    assert raised.value is wall
    assert browser.connect.await_count == 1
