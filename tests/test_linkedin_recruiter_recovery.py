"""Tests for P6 Recruiter recovery state machine."""

import asyncio
import pytest
import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from linkedin.browser import LinkedInBrowser
from shared import config
from shared.governor import UNGOVERNED_FOR_TESTS
from linkedin.recruiter_recovery import (
    RECRUITER_HEALTH_STATES,
    RecoveryResult,
    RecruiterRecoverySnapshot,
    _CAPTURED_RATE_LIMIT_PATTERNS,
    capture_recovery_snapshot,
    detect_recruiter_health,
    load_capture_vocabulary,
    recover_recruiter_context,
    replay_search_context,
    verify_recruiter_context,
)


def _make_browser(
    url: str = "",
    project_id: str | None = None,
    *,
    title: str = "",
    body_text: str = "",
) -> LinkedInBrowser:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value=url)
    page.title = AsyncMock(return_value=title)
    body_locator = MagicMock()
    body_locator.inner_text = AsyncMock(return_value=body_text)
    page.locator = MagicMock(return_value=body_locator)
    browser._page = page
    browser._project_id = project_id
    browser.refresh_active_tab = AsyncMock(return_value=True)
    browser._bind_existing_recruiter_page = AsyncMock(return_value=True)
    browser.navigate_to_search = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.connect = AsyncMock()
    return browser


# ---------------------------------------------------------------------------
# RecruiterRecoverySnapshot
# ---------------------------------------------------------------------------


def test_snapshot_round_trip():
    snap = RecruiterRecoverySnapshot(
        run_id=1,
        work_unit_id="wu-1",
        lane_id="lane-a",
        search_url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
        current_page=3,
        advanced_search_controls={"location": "NYC"},
        keyword_boolean='"AI" AND "ML"',
    )
    d = snap.to_dict()
    assert d["run_id"] == 1
    assert d["work_unit_id"] == "wu-1"
    assert d["lane_id"] == "lane-a"
    assert d["project_id"] == "123"
    assert d["keyword_boolean"] == '"AI" AND "ML"'
    assert d["advanced_search_controls"] == {"location": "NYC"}


def test_snapshot_preserves_context():
    """Work unit and lane context survive the snapshot boundary."""
    snap = RecruiterRecoverySnapshot(
        work_unit_id="wu-42",
        lane_id="senior-ml",
        keyword_boolean="deep learning",
    )
    assert snap.work_unit_id == "wu-42"
    assert snap.lane_id == "senior-ml"


# ---------------------------------------------------------------------------
# capture_recovery_snapshot
# ---------------------------------------------------------------------------


def test_capture_snapshot_reads_browser_state():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/999/discover/recruiterSearch",
        project_id="999",
    )
    snap = capture_recovery_snapshot(
        browser, run_id=5, work_unit_id="wu-x", lane_id="eng", keyword_boolean="python"
    )
    assert snap.project_id == "999"
    assert snap.search_url.endswith("recruiterSearch")
    assert snap.run_id == 5
    assert snap.keyword_boolean == "python"


def test_capture_snapshot_handles_broken_page():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(side_effect=RuntimeError("Target closed"))
    browser._page = page
    browser._project_id = "42"
    snap = capture_recovery_snapshot(browser)
    assert snap.search_url == ""
    assert snap.project_id == "42"


def test_capture_snapshot_to_dict_is_json_safe_with_mock_browser():
    browser = MagicMock()
    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/talent/search")
    browser.page = page
    browser.get_current_search_url = MagicMock(return_value=MagicMock())
    snap = capture_recovery_snapshot(browser)
    json.dumps(snap.to_dict())
    assert snap.project_id == ""


# ---------------------------------------------------------------------------
# verify_recruiter_context
# ---------------------------------------------------------------------------


def test_verify_context_healthy_match():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    snap = RecruiterRecoverySnapshot(project_id="123")
    assert verify_recruiter_context(browser, snap) is True


def test_verify_context_wrong_project():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/999/discover/recruiterSearch"
    )
    snap = RecruiterRecoverySnapshot(project_id="123")
    assert verify_recruiter_context(browser, snap) is False


def test_verify_context_non_linkedin():
    browser = _make_browser(url="https://www.google.com/")
    snap = RecruiterRecoverySnapshot(project_id="123")
    assert verify_recruiter_context(browser, snap) is False


def test_verify_context_no_project_requirement():
    browser = _make_browser(url="https://www.linkedin.com/talent/search")
    snap = RecruiterRecoverySnapshot(project_id="")
    assert verify_recruiter_context(browser, snap) is True


# ---------------------------------------------------------------------------
# recover_recruiter_context
# ---------------------------------------------------------------------------


def test_recover_healthy_returns_immediately():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
    )
    snap = RecruiterRecoverySnapshot(project_id="123")

    with patch("linkedin.recruiter_recovery.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(recover_recruiter_context(browser, snap))

    assert result.success is True
    assert result.attempts == 0
    assert result.reason == "already_healthy"
    assert result.context_verified is True


def test_recover_aw_snap_rebinds_and_verifies():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    call_count = 0

    def url_side_effect():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return "chrome-error://chromewebdata/"
        return "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"

    type(page).url = PropertyMock(side_effect=url_side_effect)
    browser._page = page
    browser._project_id = "123"
    browser.refresh_active_tab = AsyncMock(return_value=True)
    browser._bind_existing_recruiter_page = AsyncMock(return_value=True)
    browser.navigate_to_search = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.connect = AsyncMock()

    snap = RecruiterRecoverySnapshot(
        project_id="123",
        search_url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        work_unit_id="wu-7",
        lane_id="ml-lane",
    )

    with patch("linkedin.recruiter_recovery.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(recover_recruiter_context(browser, snap))

    assert result.success is True
    assert result.health_before == "aw_snap"
    assert result.context_verified is True
    assert result.attempts >= 1


def test_recover_fails_closed_when_context_unverifiable():
    browser = _make_browser(
        url="https://www.google.com/",
    )
    snap = RecruiterRecoverySnapshot(project_id="123")

    with patch("linkedin.recruiter_recovery.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(recover_recruiter_context(browser, snap, max_attempts=2))

    assert result.success is False
    assert result.context_verified is False
    assert result.reason == "context_verification_failed"


# ---------------------------------------------------------------------------
# P8.2: blocked_or_rate_limited backs off — never refresh-and-retry.
# ---------------------------------------------------------------------------


def test_recover_backs_off_on_rate_limit_without_any_navigation():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
        title="LinkedIn Recruiter",
        body_text="You have exceeded the rate limit. Please try again later.",
    )
    browser._governor = MagicMock()
    snap = RecruiterRecoverySnapshot(project_id="123")

    with patch("linkedin.recruiter_recovery.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        result = asyncio.run(recover_recruiter_context(browser, snap, max_attempts=3))

    assert result.success is False
    assert result.health_before == "blocked_or_rate_limited"
    assert result.health_after == "blocked_or_rate_limited"
    assert result.attempts == 0
    assert "backoff" in result.reason
    # The whole point of P8.2: no refresh-and-retry mechanics run at all.
    browser.refresh_active_tab.assert_not_awaited()
    browser.disconnect.assert_not_awaited()
    browser.connect.assert_not_awaited()
    browser.navigate_to_search.assert_not_awaited()
    browser._bind_existing_recruiter_page.assert_not_awaited()
    sleep_mock.assert_not_awaited()
    # Back-off is enforced through the browser's governor, not a local flag.
    browser._governor.force_backoff.assert_called_once_with("blocked_or_rate_limited")


def test_recover_backs_off_on_rate_limit_records_event():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
        title="LinkedIn",
        body_text="429 Too Many Requests",
    )
    snap = RecruiterRecoverySnapshot(project_id="123")
    events = []

    result = asyncio.run(
        recover_recruiter_context(
            browser, snap, event_recorder=lambda t, p: events.append((t, p))
        )
    )

    assert result.success is False
    event_types = [e[0] for e in events]
    assert "recruiter_recovery_backoff" in event_types
    assert "recruiter_recovery_attempted" not in event_types


def test_recover_records_structured_events():
    browser = _make_browser(url="https://www.google.com/")
    snap = RecruiterRecoverySnapshot(project_id="123")

    events = []

    def recorder(event_type: str, payload: dict):
        events.append((event_type, payload))

    with patch("linkedin.recruiter_recovery.asyncio.sleep", new=AsyncMock()):
        asyncio.run(
            recover_recruiter_context(browser, snap, max_attempts=1, event_recorder=recorder)
        )

    event_types = [e[0] for e in events]
    assert "recruiter_recovery_attempted" in event_types
    assert "recruiter_recovery_failed" in event_types


def test_recover_records_success_event():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
    )
    page = MagicMock()
    type(page).url = PropertyMock(
        side_effect=[
            RuntimeError("Target crashed"),
            "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
            "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
            "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
            "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        ]
    )
    browser._page = page

    snap = RecruiterRecoverySnapshot(project_id="123")
    events = []

    with patch("linkedin.recruiter_recovery.asyncio.sleep", new=AsyncMock()):
        asyncio.run(
            recover_recruiter_context(browser, snap, event_recorder=lambda t, p: events.append((t, p)))
        )

    event_types = [e[0] for e in events]
    assert "recruiter_recovery_attempted" in event_types
    assert "recruiter_recovery_succeeded" in event_types


# ---------------------------------------------------------------------------
# RecoveryResult
# ---------------------------------------------------------------------------


def test_recovery_result_to_dict():
    r = RecoveryResult(
        success=True,
        health_before="aw_snap",
        health_after="healthy",
        attempts=2,
        context_verified=True,
        reason="recovered",
    )
    d = r.to_dict()
    assert d["success"] is True
    assert d["health_before"] == "aw_snap"
    assert d["attempts"] == 2


# ---------------------------------------------------------------------------
# RECRUITER_HEALTH_STATES constant
# ---------------------------------------------------------------------------


def test_health_states_match_spec():
    expected = {
        "healthy", "aw_snap", "target_crashed", "slow_or_unresponsive",
        "logged_out", "wrong_surface", "lost_project_context",
        "stale_search_context", "cdp_attach_failed", "blocked_or_rate_limited",
    }
    assert RECRUITER_HEALTH_STATES == expected


def test_replay_search_context_reapplies_keyword_boolean():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
    )
    browser.enter_search_string = AsyncMock(
        return_value=MagicMock(typing_result=MagicMock(), results_wait_ms=100)
    )
    snap = RecruiterRecoverySnapshot(
        project_id="123",
        keyword_boolean='"ML" AND "engineer"',
    )
    ok, reason = asyncio.run(replay_search_context(browser, snap))
    assert ok is True
    assert reason


def test_recover_healthy_replay_fails_returns_failure_without_full_recovery():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
    )
    browser.enter_search_string = AsyncMock(side_effect=RuntimeError("sidebar broken"))
    browser.refresh_active_tab = AsyncMock()
    browser.disconnect = AsyncMock()
    browser.connect = AsyncMock()
    snap = RecruiterRecoverySnapshot(
        project_id="123",
        keyword_boolean="python",
        search_url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
    )
    result = asyncio.run(recover_recruiter_context(browser, snap))
    assert result.success is False
    assert result.context_verified is True
    assert result.attempts == 0
    assert "failed" in result.reason
    browser.refresh_active_tab.assert_not_awaited()
    browser.disconnect.assert_not_awaited()


# ---------------------------------------------------------------------------
# Wave 0 Slice 0.2: page-capture hook on wrong_surface (linkedin.com URL)
# ---------------------------------------------------------------------------


def test_linkedin_url_classified_wrong_surface_is_captured(tmp_path):
    browser = _make_browser(
        url="https://www.linkedin.com/enterprise-authentication/sessions",
        project_id="12345",
    )
    browser._state_dir = tmp_path

    capture_mock = AsyncMock()
    with patch("linkedin.page_capture.capture_page_state", capture_mock):
        result = asyncio.run(detect_recruiter_health(browser))

    assert result == "wrong_surface"
    capture_mock.assert_awaited_once_with(
        browser, tmp_path, reason="health_unclassified"
    )


def test_off_linkedin_wrong_surface_does_not_capture(tmp_path):
    browser = _make_browser(url="https://www.google.com/", project_id="12345")
    browser._state_dir = tmp_path

    capture_mock = AsyncMock()
    with patch("linkedin.page_capture.capture_page_state", capture_mock):
        result = asyncio.run(detect_recruiter_health(browser))

    assert result == "wrong_surface"
    capture_mock.assert_not_awaited()


def test_blank_url_wrong_surface_does_not_capture(tmp_path):
    browser = _make_browser(url="about:blank", project_id="12345")
    browser._state_dir = tmp_path

    capture_mock = AsyncMock()
    with patch("linkedin.page_capture.capture_page_state", capture_mock):
        result = asyncio.run(detect_recruiter_health(browser))

    assert result == "wrong_surface"
    capture_mock.assert_not_awaited()


def test_recover_replays_search_context_after_verify():
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
    )
    browser.enter_search_string = AsyncMock(
        return_value=MagicMock(typing_result=MagicMock(), results_wait_ms=100)
    )
    snap = RecruiterRecoverySnapshot(
        project_id="123",
        keyword_boolean="python",
        search_url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
    )
    with patch("linkedin.recruiter_recovery.asyncio.sleep", new=AsyncMock()):
        result = asyncio.run(recover_recruiter_context(browser, snap))
    assert result.success is True
    browser.enter_search_string.assert_awaited()


# ---------------------------------------------------------------------------
# Wave 3 Slice 3.1: live health classifier on the happy path
# ---------------------------------------------------------------------------


def _orchestrator_mod():
    return importlib.import_module("linkedin.orchestrator")


def _make_pipeline_for_health_check(tmp_path: Path, *, governor):
    orch = _orchestrator_mod()
    with (
        patch.object(orch, "load_brief") as load_brief,
        patch.object(orch, "init_judger"),
        patch.object(orch, "LinkedInBrowser"),
    ):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.permanent_filters = {}
        brief.needs_preflight.return_value = False
        load_brief.return_value = brief

        brief_path = tmp_path / "brief.json"
        brief_path.write_text('{"id": "test"}')
        pipeline = orch.Pipeline(
            brief_path=str(brief_path),
            output_dir=str(tmp_path),
            governor=governor,
        )

    page = MagicMock()
    type(page).url = PropertyMock(
        return_value="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    pipeline.browser.page = page
    pipeline.browser.check_and_recover = AsyncMock(return_value=False)
    pipeline._maybe_cadence_pause = AsyncMock(return_value=None)
    return pipeline


def _configure_governor_paths(monkeypatch, tmp_path):
    import shared.cooldown as cooldown

    governor_dir = tmp_path / "governor"
    monkeypatch.setattr(cooldown, "GOVERNOR_DIR", governor_dir)
    monkeypatch.setattr(cooldown, "DAILY_STATS_FILE", governor_dir / "daily_stats.json")
    monkeypatch.setattr(cooldown, "SESSIONS_LOG", governor_dir / "sessions.jsonl")
    return governor_dir


def test_blocked_page_on_happy_path_trips_force_backoff(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED", True)

    governor = MagicMock()
    pipeline = _make_pipeline_for_health_check(tmp_path, governor=governor)

    detect_mock = AsyncMock(return_value="blocked_or_rate_limited")
    with patch("linkedin.recruiter_recovery.detect_recruiter_health", detect_mock):
        asyncio.run(pipeline._ensure_browser_healthy())

    detect_mock.assert_awaited_once_with(pipeline.browser)
    governor.force_backoff.assert_called_once_with("blocked_or_rate_limited")


def test_force_backoff_from_happy_path_persists_across_process_restart(monkeypatch, tmp_path):
    import shared.cooldown as cooldown
    from shared.governor import SessionGovernor

    _configure_governor_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED", True)

    governor = SessionGovernor()
    pipeline = _make_pipeline_for_health_check(tmp_path, governor=governor)

    with patch(
        "linkedin.recruiter_recovery.detect_recruiter_health",
        new=AsyncMock(return_value="blocked_or_rate_limited"),
    ):
        asyncio.run(pipeline._ensure_browser_healthy())

    active = cooldown.get_active_backoff()
    assert active is not None
    assert active["reason"] == "blocked_or_rate_limited"

    fresh = SessionGovernor()
    ok, reason = fresh.can_start_session()
    assert ok is False
    assert "blocked_or_rate_limited" in reason


def test_flag_off_leaves_the_disconnect_only_path_unchanged(monkeypatch, tmp_path):
    _configure_governor_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(config, "LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED", False)

    governor = MagicMock()
    pipeline = _make_pipeline_for_health_check(tmp_path, governor=governor)

    detect_mock = AsyncMock(return_value="blocked_or_rate_limited")
    with patch("linkedin.recruiter_recovery.detect_recruiter_health", detect_mock):
        asyncio.run(pipeline._ensure_browser_healthy())

    detect_mock.assert_not_awaited()
    governor.force_backoff.assert_not_called()


# ---------------------------------------------------------------------------
# Slice A3: rate-limit vocabulary — precise 429 matching + capture loader
# ---------------------------------------------------------------------------


def _write_capture_tree(
    tmp_path: Path,
    *,
    title: str = "",
    body_lines: list[str] | None = None,
    meta_json: str | None = None,
    capture_name: str = "20260101T000000-rate-limit",
) -> Path:
    capture_dir = tmp_path / "captures" / capture_name
    capture_dir.mkdir(parents=True)
    if meta_json is not None:
        (capture_dir / "meta.json").write_text(meta_json, encoding="utf-8")
    else:
        (capture_dir / "meta.json").write_text(
            json.dumps({"title": title, "url": "https://www.linkedin.com/"}),
            encoding="utf-8",
        )
    if body_lines is not None:
        (capture_dir / "body.txt").write_text("\n".join(body_lines), encoding="utf-8")
    return capture_dir


def _classify(title: str = "", body_text: str = "") -> str:
    """Classify a healthy Recruiter URL carrying the given page content."""
    browser = _make_browser(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
        project_id="123",
        title=title,
        body_text=body_text,
    )
    return asyncio.run(detect_recruiter_health(browser))


def test_pattern_list_only_contains_strings_present_in_captures():
    """No capture has ever recorded a real interstitial, so nothing may be promoted.

    A guard that checked the promoted strings against a capture tree the test
    itself wrote would pass for any string the fixture happened to contain,
    which is not provenance. Until a real capture corpus exists to diff
    against, the honest invariant is that the list is empty: promoting a
    string must land together with the capture that justifies it, and that
    commit is what changes this assertion.
    """
    assert _CAPTURED_RATE_LIMIT_PATTERNS == ()


def test_capture_vocabulary_reader_surfaces_capture_text(tmp_path):
    known_phrase = "http error 429 too many requests"
    _write_capture_tree(
        tmp_path,
        title="Rate Limited",
        body_lines=[known_phrase, "Please try again later."],
    )

    vocabulary = load_capture_vocabulary(tmp_path)
    assert known_phrase in vocabulary
    assert "rate limited" in vocabulary


def test_follower_count_does_not_trip_rate_limit():
    assert (
        _classify(
            title="Jane Doe | LinkedIn",
            body_text="Software Engineer at Acme Corp. 1,429 followers",
        )
        == "healthy"
    )


@pytest.mark.parametrize(
    "body_text",
    [
        "Software Engineer at Stripe. Built the API rate limiting layer for 1M QPS",
        "I work on rate limiters, load shedding and backpressure",
        "Distributed rate limit design | Redis",
        "Senior SRE - designed the rate limiting service",
    ],
)
def test_engineer_profile_copy_does_not_trip_rate_limit(body_text):
    """"rate limiting" is ordinary engineering vocabulary, not a block on us.

    The Recruiter body carries candidate headlines and About text, so a bare
    "rate limit" substring reads a backend engineer's profile as a rate-limit
    interstitial. This classifier runs on every page and a hit costs a
    six-hour persisted backoff, so the body demands notice framing.
    """
    assert _classify(title="Jane Doe | LinkedIn", body_text=body_text) == "healthy"


@pytest.mark.parametrize(
    "body_text",
    [
        "HTTP ERROR 429",
        "Error code: 429",
        "HTTP/1.1 429",
        "status=429",
        "429. That's an error.",
        "429 Too Many Requests",
    ],
)
def test_http_429_renderings_trip_rate_limit(body_text):
    """Servers print 429 with punctuation far more often than as a tidy phrase."""
    assert _classify(body_text=body_text) == "blocked_or_rate_limited"


@pytest.mark.parametrize(
    "body_text",
    [
        "You have exceeded the rate limit. Please try again later.",
        "You've been rate limited. Try again in a few minutes.",
        "We are temporarily limiting your activity. You have reached the rate limit.",
    ],
)
def test_rate_limit_notices_trip(body_text):
    assert _classify(body_text=body_text) == "blocked_or_rate_limited"


@pytest.mark.parametrize("title", ["Rate limit exceeded", "Too Many Requests"])
def test_rate_limit_titles_trip(title):
    """An interstitial owns the page, so its title is matched sensitively."""
    assert _classify(title=title) == "blocked_or_rate_limited"


def test_capture_loader_returns_empty_when_no_captures(tmp_path):
    assert load_capture_vocabulary(tmp_path) == ()


def test_capture_loader_tolerates_malformed_capture(tmp_path):
    capture_dir = tmp_path / "captures" / "20260101T000001-broken"
    capture_dir.mkdir(parents=True)
    (capture_dir / "meta.json").write_text("{not valid json", encoding="utf-8")
    (capture_dir / "body.txt").write_text("still readable body line", encoding="utf-8")

    vocabulary = load_capture_vocabulary(tmp_path)
    assert "still readable body line" in vocabulary
