import asyncio
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from linkedin.browser import LinkedInBrowser
from shared.governor import UNGOVERNED_FOR_TESTS
from shared import config
from shared.schemas import OpusDecision, SearchString
from shared.storage import read_jsonl
from tests.test_linkedin_pipeline import (
    _PROJECT_SEARCH_URL,
    _make_pipeline,
    _make_snippet,
)


def test_expand_all_readmore_uses_trimmed_dwell_and_settle():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = MagicMock()
    browser._page.wait_for_timeout = AsyncMock()
    browser._ghost_click_locator = AsyncMock()

    matching = []
    for _ in range(20):
        el = MagicMock()
        el.inner_text = AsyncMock(return_value="See more")
        el.is_visible = AsyncMock(return_value=True)
        matching.append(el)

    locator = MagicMock()
    locator.all = AsyncMock(return_value=matching)
    container = MagicMock()
    container.locator.return_value = locator

    with patch("linkedin.browser.human_delay_correlated", side_effect=lambda base, channel: base), patch(
        "linkedin.browser.asyncio.sleep",
        new=AsyncMock(),
    ) as sleep_mock:
        asyncio.run(browser._expand_all_readmore(container))

    container.locator.assert_called_once_with("a, button, [role=\"button\"]")
    assert browser._ghost_click_locator.await_count == 15
    assert sleep_mock.await_args_list == [
        call(config.LINKEDIN_PROFILE_EXPAND_CLICK_DWELL_SECONDS)
        for _ in range(15)
    ]
    browser._page.wait_for_timeout.assert_awaited_once_with(
        int(config.LINKEDIN_PROFILE_EXPAND_SETTLE_SECONDS * 1000)
    )


def test_expand_all_readmore_skips_hidden_and_irrelevant_controls():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = MagicMock()
    browser._page.wait_for_timeout = AsyncMock()
    browser._ghost_click_locator = AsyncMock()

    visible_match = MagicMock()
    visible_match.inner_text = AsyncMock(return_value="Read more")
    visible_match.is_visible = AsyncMock(return_value=True)

    hidden_match = MagicMock()
    hidden_match.inner_text = AsyncMock(return_value="See more")
    hidden_match.is_visible = AsyncMock(return_value=False)

    irrelevant = MagicMock()
    irrelevant.inner_text = AsyncMock(return_value="Connect")
    irrelevant.is_visible = AsyncMock(return_value=True)

    locator = MagicMock()
    locator.all = AsyncMock(return_value=[visible_match, hidden_match, irrelevant])
    container = MagicMock()
    container.locator.return_value = locator

    with patch("linkedin.browser.human_delay_correlated", side_effect=lambda base, channel: base), patch(
        "linkedin.browser.asyncio.sleep",
        new=AsyncMock(),
    ):
        asyncio.run(browser._expand_all_readmore(container))

    browser._ghost_click_locator.assert_awaited_once_with(visible_match)
    browser._page.wait_for_timeout.assert_awaited_once()


def test_linkedin_save_linger_uses_configured_bounds_and_scroll_range():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.test_mode = False
        pipeline._runtime_bridge = None
        # The slide-in a live save is issued from: the profile panel opened
        # OVER the brief's own project search view, so the URL carries both the
        # profile identity and the project (docs/linkedin-recruiter-dom-map.md
        # §"the URL changes to include /recruiterSearch/profile/{memberID}").
        # `page` — not `_page` — is what the pre-save project guard reads.
        pipeline.browser.page = pipeline.browser._page
        pipeline.browser._page.url = f"{_PROJECT_SEARCH_URL}/profile/cadence-save"
        pipeline.browser.current_profile_identity_fragment.side_effect = (
            lambda: LinkedInBrowser._profile_url_fragment(
                pipeline.browser._page.url
            )
        )
        # Wave G: the service probes the candidate's card, not the panel.
        pipeline.browser.is_already_saved_on_card = AsyncMock(return_value=False)
        pipeline.browser.save_candidate = AsyncMock(return_value=True)
        pipeline.browser.scroll_for_linger = AsyncMock(return_value=120)
        pipeline.browser.scroll_restore = AsyncMock()

        snippet = _make_snippet(profile_url="/talent/profile/cadence-save")
        runtime_search_string = SearchString(id=1, name="test", boolean="foo")
        captured = []

        def fake_delay(base, channel):
            captured.append((base, channel))
            return 99.0

        with patch("linkedin.side_effects.human_delay_correlated", side_effect=fake_delay), patch(
            "linkedin.side_effects.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock, patch("linkedin.side_effects.random.randint", return_value=2) as randint_mock:
            outcome = asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=snippet,
                    runtime_search_string=runtime_search_string,
                    attempt_id=11,
                )
            )

        assert outcome.status == "succeeded"
        assert captured == [(config.LINKEDIN_SAVE_LINGER_BASE_SECONDS, "save_linger")]
        randint_mock.assert_called_once_with(
            config.LINKEDIN_SAVE_LINGER_MIN_CHUNKS_BACK,
            config.LINKEDIN_SAVE_LINGER_MAX_CHUNKS_BACK,
        )
        sleep_mock.assert_awaited_once_with(config.LINKEDIN_SAVE_LINGER_MAX_SECONDS)
        pipeline.browser.scroll_for_linger.assert_awaited_once_with(2)
        pipeline.browser.scroll_restore.assert_awaited_once_with(120)


def test_linkedin_save_linger_rejects_mismatched_fake_panel_before_save():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.test_mode = False
        pipeline._runtime_bridge = None
        pipeline.browser._page.url = (
            "https://www.linkedin.com/talent/recruiterSearch/profile/someone-else"
        )
        pipeline.browser.current_profile_identity_fragment.side_effect = (
            lambda: LinkedInBrowser._profile_url_fragment(
                pipeline.browser._page.url
            )
        )
        pipeline.browser.is_already_saved_on_card = AsyncMock()
        pipeline.browser.save_candidate = AsyncMock()

        with pytest.raises(RuntimeError, match="profile identity"):
            asyncio.run(
                pipeline._side_effects_service.handle_save_decision(
                    snippet=_make_snippet(
                        profile_url="/talent/profile/cadence-save"
                    ),
                    runtime_search_string=SearchString(
                        id=1,
                        name="test",
                        boolean="foo",
                    ),
                    attempt_id=11,
                )
            )

        pipeline.browser.is_already_saved_on_card.assert_not_awaited()
        pipeline.browser.save_candidate.assert_not_awaited()


def test_full_evaluate_reject_uses_trimmed_close_and_panel_dwell():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline._ensure_services = MagicMock()
        pipeline._acquisition_service = MagicMock()
        pipeline._acquisition_service.extract_profile_summary = AsyncMock(
            return_value=SimpleNamespace(profile_summary=MagicMock(to_dict=MagicMock(return_value={"ok": True})))
        )
        pipeline._start_runtime_stage_attempt = MagicMock(return_value=17)
        pipeline._finish_runtime_stage_success = MagicMock()
        pipeline._mark_terminal = MagicMock()
        pipeline._bias_monitor = None
        pipeline.browser.go_back_to_results = AsyncMock()
        snippet = _make_snippet(profile_url="/talent/profile/reject-cadence")

        captured = []

        def fake_delay(base, channel):
            captured.append((base, channel))
            if channel == "reject_close":
                return 99.0
            return base

        with patch(
            "linkedin.orchestrator.full_judge",
            return_value=OpusDecision(
                stage="full",
                decision="REJECT",
                path="none",
                confidence=0.3,
                rationale="Not a fit",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            ),
        ), patch("linkedin.orchestrator.human_delay_correlated", side_effect=fake_delay), patch(
            "linkedin.orchestrator.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock:
            decision = asyncio.run(pipeline._full_evaluate(snippet, None, SearchString(id=1, name="test", boolean="x")))

        assert decision is not None
        assert decision.decision == "REJECT"
        assert captured == [
            (config.LINKEDIN_REJECT_CLOSE_BASE_SECONDS, "reject_close"),
            (config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS, "panel_close"),
        ]
        assert sleep_mock.await_args_list == [
            call(config.LINKEDIN_REJECT_CLOSE_MAX_SECONDS),
            call(config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS),
        ]


def test_full_evaluate_failure_decision_uses_trimmed_panel_close():
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        summary = MagicMock()
        summary.to_dict.return_value = {"ok": True}
        pipeline._ensure_services = MagicMock()
        pipeline._acquisition_service = MagicMock()
        pipeline._acquisition_service.extract_profile_summary = AsyncMock(
            return_value=SimpleNamespace(profile_summary=summary)
        )
        pipeline._start_runtime_stage_attempt = MagicMock(return_value=19)
        pipeline._finish_runtime_failure_decision = MagicMock()
        pipeline._bias_monitor = None
        pipeline.browser.go_back_to_results = AsyncMock()
        snippet = _make_snippet(profile_url="/talent/profile/failure-cadence")

        captured = []

        def fake_delay(base, channel):
            captured.append((base, channel))
            return base

        with patch(
            "linkedin.orchestrator.full_judge",
            return_value=OpusDecision(
                stage="full",
                decision="PARSE_FAILURE",
                path="none",
                confidence=0.0,
                rationale="[PARSE_FAILURE: bad output]",
                candidate_name=snippet.name,
                profile_url=snippet.profile_url,
            ),
        ), patch("linkedin.orchestrator.human_delay_correlated", side_effect=fake_delay), patch(
            "linkedin.orchestrator.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock:
            decision = asyncio.run(pipeline._full_evaluate(snippet, None, SearchString(id=1, name="test", boolean="x")))

        assert decision is not None
        assert decision.decision == "PARSE_FAILURE"
        assert captured == [(config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS, "panel_close")]
        sleep_mock.assert_awaited_once_with(config.LINKEDIN_PANEL_CLOSE_SETTLE_SECONDS)


def test_cadence_pause_timing_is_separate_registered_and_default_off(capsys):
    assert config.LINKEDIN_TIMING_TELEMETRY_ENABLED is False
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        events = []
        pipeline._timing_recorder = (
            lambda event, payload: events.append((event, payload))
        )
        pipeline._last_pause_time = time.time() - 120

        with patch(
            "linkedin.orchestrator.config.CADENCE_INTERVAL_MINUTES",
            1.0,
        ), patch(
            "linkedin.orchestrator.human_delay",
            side_effect=[0.0, 2.5],
        ), patch(
            "linkedin.orchestrator.asyncio.sleep",
            new=AsyncMock(),
        ) as sleep_mock:
            asyncio.run(pipeline._maybe_cadence_pause())

        sleep_mock.assert_awaited_once_with(2.5)
        assert events[0][0] == "cadence_pause_timing"
        assert events[0][1]["pause_seconds"] == 2.5
        assert events[0][1]["elapsed_ms"] >= 0
        assert [record["event"] for record in read_jsonl(pipeline.log_path)] == [
            "cadence_pause"
        ]
        assert "cadence_pause_timing" not in capsys.readouterr().out


def test_cadence_interval_clamp_mass_under_five_percent(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_CADENCE_READ_FIX_ENABLED", True)
    from shared.human_timing import reset_human_timing

    reset_human_timing("cadence_interval")
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        low = config.CADENCE_INTERVAL_MINUTES * 0.3
        high = config.CADENCE_INTERVAL_MINUTES * 4.0
        clamp_count = 0
        for _ in range(20000):
            sample = pipeline._sample_cadence_interval()
            if abs(sample - low) < 1e-9 or abs(sample - high) < 1e-9:
                clamp_count += 1
        assert clamp_count / 20000 < 0.05


def test_cadence_pause_clamp_mass_under_five_percent(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_CADENCE_READ_FIX_ENABLED", True)
    from shared.human_timing import reset_human_timing

    reset_human_timing("cadence_pause")
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        low = config.CADENCE_PAUSE_SECONDS * 0.3
        high = config.CADENCE_PAUSE_SECONDS * 4.0
        clamp_count = 0
        for _ in range(20000):
            sample = pipeline._sample_cadence_pause()
            if abs(sample - low) < 1e-9 or abs(sample - high) < 1e-9:
                clamp_count += 1
        assert clamp_count / 20000 < 0.05


def test_cadence_flag_off_uses_legacy_human_delay(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_CADENCE_READ_FIX_ENABLED", False)
    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        with patch("linkedin.orchestrator.human_delay", side_effect=lambda low, high: low) as human_delay_mock, patch(
            "linkedin.orchestrator.human_delay_correlated"
        ) as correlated_mock:
            pipeline._sample_cadence_interval()
            pipeline._sample_cadence_pause()

        human_delay_mock.assert_has_calls(
            [
                call(
                    config.CADENCE_INTERVAL_MINUTES * 0.8,
                    config.CADENCE_INTERVAL_MINUTES * 1.4,
                ),
                call(
                    config.CADENCE_PAUSE_SECONDS * 0.6,
                    config.CADENCE_PAUSE_SECONDS * 2.0,
                ),
            ]
        )
        correlated_mock.assert_not_called()
