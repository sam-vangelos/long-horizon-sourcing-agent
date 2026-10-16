"""CLO-205: profile-panel mount-wedge containment.

The 2026-08-18 Brazil run failed 21 consecutive panel mounts over 28 minutes:
the panel container never re-appeared after one ordinary close, the wedge
survived every SPA navigation, and the run answered only with per-candidate
retries and mislabeled breaker pauses until the operator killed it. These
tests lock the containment seams added in response:

- the panel is identified by class OR LinkedIn's data-live-test marker;
- every panel close records WHICH strategy closed it (the nuclear close was
  unattributable before);
- a final extraction failure captures forensics (screenshot + HTML + context)
  into the state dir, bounded and fail-soft;
- three consecutive mount failures trigger ONE document reload; six trip the
  governor with a short cooldown so the session ends cleanly.
"""

from linkedin.evaluation_loop import EvaluationLoop
from linkedin.evaluation_state import EvaluationRunState

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin.browser import PROFILE_PANEL_SELECTOR, LinkedInBrowser
from shared.governor import UNGOVERNED_FOR_TESTS


# ---------------------------------------------------------------------------
# Panel identity
# ---------------------------------------------------------------------------


@pytest.mark.structural
def test_panel_selector_accepts_class_and_live_test_marker():
    assert "div.profile-slidein__container" in PROFILE_PANEL_SELECTOR
    assert "data-live-test-profile-slidein" in PROFILE_PANEL_SELECTOR


# ---------------------------------------------------------------------------
# Close-strategy attribution (go_back_to_results)
# ---------------------------------------------------------------------------


def _browser_with_recorder(events):
    return LinkedInBrowser(
        governor=UNGOVERNED_FOR_TESTS,
        timing_recorder=lambda event, payload: events.append((event, payload)),
    )


def test_go_back_records_button_close():
    events = []
    browser = _browser_with_recorder(events)
    panel = MagicMock()
    panel.is_visible = AsyncMock(side_effect=[True, False])
    page = MagicMock()
    page.locator.return_value = panel
    page.evaluate = AsyncMock(return_value=True)
    page.wait_for_timeout = AsyncMock()
    browser._page = page

    asyncio.run(browser.go_back_to_results())

    assert ("panel_close_strategy", {"strategy": "button", "panel_closed": True}) in events


def test_go_back_records_and_announces_nuclear_close(capsys):
    events = []
    browser = _browser_with_recorder(events)
    panel = MagicMock()
    # entry (open), after Escape (still open), final verification (closed)
    panel.is_visible = AsyncMock(side_effect=[True, True, False])
    page = MagicMock()
    page.locator.return_value = panel
    # S1 close-button miss, S3 goes nuclear, then the outlet restore call
    page.evaluate = AsyncMock(side_effect=[False, "nuclear", None])
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    browser._press_key = AsyncMock()
    browser.reload_search_page = AsyncMock(return_value=True)

    asyncio.run(browser.go_back_to_results())

    assert ("panel_close_strategy", {"strategy": "nuclear", "panel_closed": True}) in events
    assert "nuclear dismiss" in capsys.readouterr().out
    # CLO-205 live acceptance: a nuclear close deterministically wedges the
    # panel path, so the close itself pays the document reload up front.
    browser.reload_search_page.assert_awaited_once()


def test_go_back_button_close_never_reloads():
    events = []
    browser = _browser_with_recorder(events)
    panel = MagicMock()
    panel.is_visible = AsyncMock(side_effect=[True, False])
    page = MagicMock()
    page.locator.return_value = panel
    page.evaluate = AsyncMock(return_value=True)
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    browser.reload_search_page = AsyncMock(return_value=True)

    asyncio.run(browser.go_back_to_results())

    browser.reload_search_page.assert_not_awaited()


def test_go_back_no_panel_emits_nothing():
    events = []
    browser = _browser_with_recorder(events)
    panel = MagicMock()
    panel.is_visible = AsyncMock(return_value=False)
    page = MagicMock()
    page.locator.return_value = panel
    browser._page = page

    asyncio.run(browser.go_back_to_results())

    assert events == []


# ---------------------------------------------------------------------------
# Failure forensics
# ---------------------------------------------------------------------------


def _forensics_page():
    page = MagicMock()
    page.screenshot = AsyncMock()
    page.content = AsyncMock(return_value="<html>wedged</html>")
    page.url = "https://www.linkedin.com/talent/hire/123/discover"
    page.viewport_size = {"width": 1440, "height": 900}
    return page


def test_forensics_capture_writes_html_and_context(tmp_path):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS, state_dir=tmp_path)
    browser._page = _forensics_page()

    stem = asyncio.run(browser.capture_failure_forensics("panel-mount-1"))

    assert stem is not None
    forensics = tmp_path / "forensics"
    files = sorted(p.name for p in forensics.iterdir())
    assert any(name.endswith(".html") for name in files)
    context_files = [p for p in forensics.iterdir() if p.suffix == ".json"]
    context = json.loads(context_files[0].read_text())
    assert context["label"] == "panel-mount-1"
    assert context["url"].startswith("https://www.linkedin.com/talent")


def test_forensics_prunes_to_newest_captures(tmp_path):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS, state_dir=tmp_path)
    browser._page = _forensics_page()

    for i in range(8):
        asyncio.run(browser.capture_failure_forensics(f"panel-mount-{i}"))

    stems = {
        p.name.rsplit(".", 1)[0] for p in (tmp_path / "forensics").iterdir()
    }
    assert len(stems) <= LinkedInBrowser._FORENSICS_KEEP


def test_forensics_never_raises_when_page_is_broken(tmp_path):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS, state_dir=tmp_path)
    page = MagicMock()
    page.screenshot = AsyncMock(side_effect=RuntimeError("target closed"))
    page.content = AsyncMock(side_effect=RuntimeError("target closed"))
    page.url = "https://www.linkedin.com/talent/hire/123"
    page.viewport_size = None
    browser._page = page

    # Screenshot and HTML both fail; the context JSON still lands, nothing raises.
    stem = asyncio.run(browser.capture_failure_forensics("panel-mount-1"))
    assert stem is not None


def test_forensics_without_state_dir_is_a_noop():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = _forensics_page()
    assert asyncio.run(browser.capture_failure_forensics("x")) is None


# ---------------------------------------------------------------------------
# Document reload
# ---------------------------------------------------------------------------


def test_reload_search_page_success():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    page.reload = AsyncMock()
    page.locator.return_value.first.wait_for = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    browser._page = page

    assert asyncio.run(browser.reload_search_page()) is True
    page.reload.assert_awaited_once()


def test_reload_search_page_failure_returns_false_without_raising(capsys):
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    page.reload = AsyncMock(side_effect=RuntimeError("net::ERR_ABORTED"))
    browser._page = page

    assert asyncio.run(browser.reload_search_page()) is False
    assert "reload failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# F11 — post-reload owned-facet verification (facet lifecycle flag)
#
# reload_search_page's docstring claims the URL preserves every facet; that
# is an assumption, not evidence. Under the flag the browser remembers the
# last successfully reconciled company/title target and re-verifies it
# (read-only) after every document reload. A mismatch makes the reload
# report failure so panel/pagination recovery escalate instead of resuming
# candidate work on an unproven sidebar. No mutation happens mid-string —
# a removal would re-run the search and corrupt the page position.
# ---------------------------------------------------------------------------


def _reloadable_browser():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    page.reload = AsyncMock()
    page.locator.return_value.first.wait_for = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    return browser


def test_flagged_reload_fails_closed_on_owned_facet_mismatch():
    """RED against the unverified reload: it returned True without ever
    reading the sidebar, so recovery resumed candidate work on it."""
    from unittest.mock import patch

    from linkedin.browser import FacetReconciliationError

    browser = _reloadable_browser()
    browser._owned_facet_target = {"companies": {"Acme"}, "titles": set()}
    browser.reconcile_facets = AsyncMock(
        side_effect=FacetReconciliationError("postcondition_mismatch")
    )

    with patch(
        "linkedin.browser.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True
    ):
        assert asyncio.run(browser.reload_search_page()) is False

    browser.reconcile_facets.assert_awaited_once_with(
        companies={"Acme"}, titles=set(), verify_only=True
    )


def test_flagged_reload_verifies_and_passes():
    from unittest.mock import patch

    browser = _reloadable_browser()
    browser._owned_facet_target = {"companies": set(), "titles": {"Staff Engineer"}}
    browser.reconcile_facets = AsyncMock()

    with patch(
        "linkedin.browser.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True
    ):
        assert asyncio.run(browser.reload_search_page()) is True

    browser.reconcile_facets.assert_awaited_once_with(
        companies=set(), titles={"Staff Engineer"}, verify_only=True
    )


def test_flag_off_reload_never_verifies_facets():
    from unittest.mock import patch

    browser = _reloadable_browser()
    browser._owned_facet_target = {"companies": {"Acme"}, "titles": set()}
    browser.reconcile_facets = AsyncMock()

    with patch(
        "linkedin.browser.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", False
    ):
        assert asyncio.run(browser.reload_search_page()) is True

    browser.reconcile_facets.assert_not_awaited()


def test_flagged_reload_without_owner_skips_verification():
    """Before any reconcile has claimed the surface (no owner yet), a reload
    must not invent a target — clearing earlier than the desired owner is
    known was explicitly rejected by the plan."""
    from unittest.mock import patch

    browser = _reloadable_browser()
    browser.reconcile_facets = AsyncMock()

    with patch(
        "linkedin.browser.config.LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED", True
    ):
        assert asyncio.run(browser.reload_search_page()) is True

    browser.reconcile_facets.assert_not_awaited()


def test_reconcile_success_records_owned_facet_target():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    page.wait_for_timeout = AsyncMock()
    browser._page = page
    browser.read_applied_company_chips = AsyncMock(return_value=["Acme"])
    browser.read_applied_title_chips = AsyncMock(return_value=[])

    asyncio.run(browser.reconcile_facets(companies={"Acme"}, titles=set()))

    assert browser._owned_facet_target == {"companies": {"Acme"}, "titles": set()}


# ---------------------------------------------------------------------------
# Governor: short cooldown pass-through
# ---------------------------------------------------------------------------


def test_force_backoff_threads_custom_cooldown(monkeypatch):
    from shared import governor as governor_module
    from shared.governor import GovernorLimitReached, SessionGovernor

    recorded = {}

    def _capture(reason, cooldown_seconds=None):
        recorded["reason"] = reason
        recorded["cooldown_seconds"] = cooldown_seconds

    monkeypatch.setattr(
        governor_module.cooldown, "record_forced_backoff", _capture
    )
    governor = SessionGovernor()
    governor.force_backoff("profile_panel_unavailable (6)", cooldown_seconds=1800)

    assert recorded == {
        "reason": "profile_panel_unavailable (6)",
        "cooldown_seconds": 1800,
    }
    with pytest.raises(GovernorLimitReached):
        governor.check_profile_open_or_raise()


# ---------------------------------------------------------------------------
# Orchestrator policy: streak -> one reload -> governor trip
# ---------------------------------------------------------------------------

_MOUNT_ERROR = Exception(
    "Locator.wait_for: Timeout 10000ms exceeded.\n"
    "Call log:\n"
    '  - waiting for locator("div.profile-slidein__container, '
    'div[data-live-test-profile-slidein]") to be visible'
)


def _pipeline_stub(tmp_path):
    from linkedin.orchestrator import Pipeline

    pipeline = EvaluationLoop(EvaluationRunState())
    pipeline.state.surface._panel_mount_failstreak = 0
    pipeline.state.surface._panel_mount_reload_used = False
    pipeline.state.context.log_path = tmp_path / "run_log.jsonl"
    pipeline.state.context.browser = SimpleNamespace(
        capture_failure_forensics=AsyncMock(return_value=None),
        reload_search_page=AsyncMock(return_value=True),
    )
    pipeline.state.context._governor = MagicMock()
    return pipeline


def _run_log_events(tmp_path):
    path = tmp_path / "run_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["event"] for line in path.read_text().splitlines()]


def test_mount_failstreak_reloads_once_then_trips_governor(tmp_path):
    from linkedin.orchestrator import (
        _PANEL_DOWN_COOLDOWN_SECONDS,
        _PANEL_MOUNT_TRIP_AT,
    )

    pipeline = _pipeline_stub(tmp_path)
    snippet = SimpleNamespace(name="Marco Nemetz Bochernitsan")

    for _ in range(_PANEL_MOUNT_TRIP_AT):
        asyncio.run(pipeline.state.surface._note_profile_extraction_failure(snippet, _MOUNT_ERROR))

    pipeline.state.context.browser.reload_search_page.assert_awaited_once()
    pipeline.state.context._governor.force_backoff.assert_called_once()
    args, kwargs = pipeline.state.context._governor.force_backoff.call_args
    assert "profile_panel_unavailable" in args[0]
    assert kwargs["cooldown_seconds"] == _PANEL_DOWN_COOLDOWN_SECONDS

    events = _run_log_events(tmp_path)
    assert "panel_recovery_started" in events
    assert "panel_recovered" in events
    assert "profile_panel_unavailable" in events


def test_successful_extraction_resets_the_streak(tmp_path):
    pipeline = _pipeline_stub(tmp_path)
    snippet = SimpleNamespace(name="X")

    for _ in range(2):
        asyncio.run(pipeline.state.surface._note_profile_extraction_failure(snippet, _MOUNT_ERROR))
    assert pipeline.state.surface._panel_mount_failstreak == 2

    pipeline.state.surface._note_profile_extraction_success()
    assert pipeline.state.surface._panel_mount_failstreak == 0
    assert pipeline.state.surface._panel_mount_reload_used is False

    # A fresh wedge episode gets its own reload budget.
    from linkedin.orchestrator import _PANEL_MOUNT_RELOAD_AT

    for _ in range(_PANEL_MOUNT_RELOAD_AT):
        asyncio.run(pipeline.state.surface._note_profile_extraction_failure(snippet, _MOUNT_ERROR))
    pipeline.state.context.browser.reload_search_page.assert_awaited_once()


def test_non_mount_extraction_failure_captures_without_streak(tmp_path):
    pipeline = _pipeline_stub(tmp_path)
    snippet = SimpleNamespace(name="X")
    read_error = Exception(
        "Locator.wait_for: Timeout 10000ms exceeded.\n"
        'Call log:\n  - waiting for locator("div.profile__main-container").first'
    )

    asyncio.run(pipeline.state.surface._note_profile_extraction_failure(snippet, read_error))

    assert pipeline.state.surface._panel_mount_failstreak == 0
    pipeline.state.context.browser.capture_failure_forensics.assert_awaited_once()
    label = pipeline.state.context.browser.capture_failure_forensics.await_args.args[0]
    assert label.startswith("profile-error-")


def test_containment_is_fail_soft_when_browser_misbehaves(tmp_path):
    pipeline = _pipeline_stub(tmp_path)
    pipeline.state.context.browser.capture_failure_forensics = AsyncMock(
        side_effect=RuntimeError("boom")
    )
    snippet = SimpleNamespace(name="X")

    # Must not raise even though every capture explodes.
    asyncio.run(pipeline.state.surface._note_profile_extraction_failure(snippet, _MOUNT_ERROR))
    assert pipeline.state.surface._panel_mount_failstreak == 1


# ---------------------------------------------------------------------------
# CLO-218: pagination escalation — position-verified reload before giving up
# ---------------------------------------------------------------------------


def _pagination_pipeline(*, next_results, signatures, reload_ok=True):
    from linkedin.orchestrator import Pipeline

    pipeline = EvaluationLoop(EvaluationRunState())
    pipeline.state.context.browser = SimpleNamespace(
        go_to_next_page=AsyncMock(side_effect=next_results),
        scroll_to_load_all_results=AsyncMock(return_value=0),
        capture_failure_forensics=AsyncMock(return_value=None),
        reload_search_page=AsyncMock(return_value=reload_ok),
        _peek_top_card_signature=AsyncMock(side_effect=signatures),
    )
    return pipeline


def _run_pagination(pipeline):
    from unittest.mock import patch

    with patch("linkedin.evaluation_loop.asyncio.sleep", new=AsyncMock()):
        return asyncio.run(
            pipeline.state.surface._go_to_next_page_with_transient_retry(
                result_count=460, page_num=1
            )
        )


def test_pagination_reload_with_stable_position_retries_and_advances():
    pipeline = _pagination_pipeline(
        next_results=[False, False, False, True],
        signatures=[("Ada L", "/a"), ("Ada L", "/a")],
    )
    has_next, transient = _run_pagination(pipeline)
    assert (has_next, transient) == (True, False)
    assert pipeline.state.context.browser.go_to_next_page.await_count == 4
    pipeline.state.context.browser.reload_search_page.assert_awaited_once()
    pipeline.state.context.browser.capture_failure_forensics.assert_awaited_once()


def test_pagination_reload_that_moves_the_page_never_click_advances():
    # A reload that resets to page 1 must NOT click Next blind — that would
    # corrupt page accounting. Falls through to the transient signal.
    pipeline = _pagination_pipeline(
        next_results=[False, False, False],
        signatures=[("Ada L", "/a"), ("Zed Q", "/z")],
    )
    has_next, transient = _run_pagination(pipeline)
    assert (has_next, transient) == (False, True)
    assert pipeline.state.context.browser.go_to_next_page.await_count == 3


def test_pagination_unreadable_signature_never_click_advances():
    # ("", "") is the peek's fail-soft value; it proves nothing about
    # position, so no in-place retry.
    pipeline = _pagination_pipeline(
        next_results=[False, False, False],
        signatures=[("", ""), ("", "")],
    )
    has_next, transient = _run_pagination(pipeline)
    assert (has_next, transient) == (False, True)
    assert pipeline.state.context.browser.go_to_next_page.await_count == 3
