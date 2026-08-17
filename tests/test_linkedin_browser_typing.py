import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

from linkedin.browser import LinkedInBrowser
from shared.governor import UNGOVERNED_FOR_TESTS
from linkedin.input_backends import TypingPlan, TypingResult, TypingStep


def test_enter_search_string_uses_backend_typing_and_never_calls_fill():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser.go_back_to_results = AsyncMock()
    browser.require_recruiter_tab = AsyncMock()
    browser._wait_for_search_results_ready = AsyncMock(return_value=1350)
    browser._peek_results_count_text = AsyncMock(return_value="100")
    browser._peek_top_card_signature = AsyncMock(return_value=("Ada", "/talent/profile/ada"))
    browser._ghost_click_locator = AsyncMock()
    browser._press_key = AsyncMock()
    browser._press_combo = AsyncMock()
    browser._input_backend.type_text = AsyncMock(
        return_value=TypingResult(
            transport="playwright_keyboard",
            duration_ms=2400,
            typo_count=1,
            used_correction=True,
            fallback_char_count=0,
        )
    )
    browser._page = MagicMock()
    browser._page.wait_for_timeout = AsyncMock()

    textarea = MagicMock()
    textarea.is_visible = AsyncMock(return_value=True)
    textarea.wait_for = AsyncMock()
    textarea.input_value = AsyncMock(return_value="")
    textarea.fill = AsyncMock()

    def locator_factory(selector):
        if selector == 'textarea[id*="free-text-single-value-input"]':
            return MagicMock(first=textarea)
        raise AssertionError(f"Unexpected selector: {selector}")

    browser._page.locator.side_effect = locator_factory
    plan = TypingPlan(steps=[TypingStep(kind="char", value="f", delay_seconds=0.0, source_index=0)])

    with patch("linkedin.browser.asyncio.sleep", new=AsyncMock()), patch(
        "linkedin.browser.human_delay_correlated",
        side_effect=lambda base, channel: base,
    ):
        result = asyncio.run(browser.enter_search_string("foo", typing_plan=plan))

    assert result.typing_result.duration_ms == 2400
    textarea.fill.assert_not_awaited()
    browser._input_backend.type_text.assert_awaited_once_with(
        browser.page,
        textarea,
        "foo",
        plan=plan,
    )
    browser._press_combo.assert_awaited_once_with("Meta+A")
    browser._press_key.assert_has_awaits([call("Backspace"), call("Enter")])


def test_wait_for_visible_poll_uses_readiness_polling():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = MagicMock()
    browser._page.wait_for_timeout = AsyncMock()
    locator = MagicMock()
    locator.is_visible = AsyncMock(side_effect=[False, False, True])

    visible = asyncio.run(
        browser._wait_for_visible_poll(locator, timeout_ms=1200, interval_ms=100)
    )

    assert visible is True
    browser._page.wait_for_timeout.assert_has_awaits([call(100), call(100)])


def test_wait_for_search_results_ready_uses_bounded_polling():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = MagicMock()
    browser._page.wait_for_timeout = AsyncMock()
    browser._peek_results_count_text = AsyncMock(side_effect=["100", "100", "120"])
    browser._peek_top_card_signature = AsyncMock(
        side_effect=[("Ada", "/talent/profile/ada")] * 3
    )

    waited_ms = asyncio.run(
        browser._wait_for_search_results_ready(
            previous_count_text="100",
            previous_top_card_signature=("Ada", "/talent/profile/ada"),
            min_wait_ms=1200,
            interval_ms=150,
            timeout_ms=3000,
        )
    )

    assert waited_ms == 1500
    browser._page.wait_for_timeout.assert_has_awaits([call(1200), call(150), call(150)])
