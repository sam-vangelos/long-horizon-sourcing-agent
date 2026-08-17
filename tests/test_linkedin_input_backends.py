"""Tests for LinkedIn input backend selection and wiring."""

import asyncio
import random
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from shared import config
from linkedin.input_backends import (
    AwayInputBackend,
    ConcurrentInputBackend,
    TypingPlan,
    TypingStep,
    _sample_key_dwell,
    build_boolean_typing_plan,
    create_input_backend,
    normalize_input_mode,
)


def test_normalize_input_mode_aliases():
    assert normalize_input_mode("concurrent") == "concurrent"
    assert normalize_input_mode("ghost-cursor") == "concurrent"
    assert normalize_input_mode("takeover") == "away"
    assert normalize_input_mode("afk") == "away"


def test_create_input_backend_concurrent():
    backend = create_input_backend("concurrent")
    assert isinstance(backend, ConcurrentInputBackend)


def test_create_input_backend_away():
    with patch("linkedin.input_backends._CoreGraphicsBridge", return_value=MagicMock()):
        backend = create_input_backend("away")
    assert isinstance(backend, AwayInputBackend)


def test_away_input_backend_press_key_returns_false_for_unknown_key():
    fake_bridge = MagicMock()
    with patch("linkedin.input_backends._CoreGraphicsBridge", return_value=fake_bridge):
        backend = AwayInputBackend()

    page = MagicMock()
    handled = asyncio.run(backend.press_key(page, "ArrowDown"))
    assert handled is False
    fake_bridge.post_key.assert_not_called()


def test_build_boolean_typing_plan_short_strings_have_no_typos():
    plan = build_boolean_typing_plan("foo AND bar", rng=random.Random(7))
    assert plan.typo_count == 0
    assert all(step.kind != "backspace" for step in plan.steps)


def test_build_boolean_typing_plan_skips_boolean_operators_and_slows_after_correction():
    text = "ALPHABRAVO AND CHARLIEDELTA"
    operator_start = text.index("AND")
    operator_range = range(operator_start, operator_start + 3)

    with patch.object(config, "LINKEDIN_SEARCH_TYPING_MEDIUM_TYPO_PROBABILITY", 1.0), patch.object(
        config,
        "LINKEDIN_SEARCH_TYPING_LONG_TYPO_PROBABILITY",
        1.0,
    ), patch.object(
        config,
        "LINKEDIN_SEARCH_TYPING_SECOND_TYPO_PROBABILITY",
        0.0,
    ), patch.object(config, "LINKEDIN_SEARCH_TYPING_CHAR_MIN_SECONDS", 0.06), patch.object(
        config,
        "LINKEDIN_SEARCH_TYPING_CHAR_MAX_SECONDS",
        0.06,
    ):
        plan = build_boolean_typing_plan(text, rng=random.Random(3))

    assert plan.typo_count == 1
    assert all(idx not in operator_range for idx in plan.typo_positions)
    backspace_index = next(i for i, step in enumerate(plan.steps) if step.kind == "backspace")
    following_chars = [
        step for step in plan.steps[backspace_index + 1 :]
        if step.kind == "char" and not step.is_correction
    ][:4]
    assert len(following_chars) == 4
    assert all(step.delay_seconds > 0.06 for step in following_chars)


def test_concurrent_input_backend_types_character_by_character():
    backend = ConcurrentInputBackend()
    page = MagicMock()
    page.keyboard.type = AsyncMock()
    page.keyboard.press = AsyncMock()
    page.keyboard.insert_text = AsyncMock()
    locator = MagicMock()
    plan = TypingPlan(
        steps=[
            TypingStep(kind="char", value="A", delay_seconds=0.0, source_index=0),
            TypingStep(kind="backspace", delay_seconds=0.0, source_index=0),
            TypingStep(kind="pause", delay_seconds=0.0, source_index=0),
            TypingStep(kind="char", value="b", delay_seconds=0.0, source_index=1),
        ],
        typo_positions=(0,),
    )

    result = asyncio.run(backend.type_text(page, locator, "Ab", plan=plan))

    assert result.transport == "playwright_keyboard"
    assert result.typo_count == 1
    page.keyboard.type.assert_has_awaits([call("A", delay=0), call("b", delay=0)])
    page.keyboard.press.assert_awaited_once_with("Backspace")
    page.keyboard.insert_text.assert_not_awaited()


def test_away_input_backend_supports_combo_and_boolean_ascii_chars():
    fake_bridge = MagicMock()
    with patch("linkedin.input_backends._CoreGraphicsBridge", return_value=fake_bridge), patch(
        "linkedin.input_backends.asyncio.sleep",
        new=AsyncMock(),
    ):
        backend = AwayInputBackend()
        page = MagicMock()
        page.keyboard.insert_text = AsyncMock()
        plan = TypingPlan(
            steps=[
                TypingStep(kind="char", value="(", delay_seconds=0.0, source_index=0),
                TypingStep(kind="char", value="&", delay_seconds=0.0, source_index=1),
                TypingStep(kind="backspace", delay_seconds=0.0, source_index=1),
            ],
            typo_positions=(1,),
        )
        combo_handled = asyncio.run(backend.press_combo(page, "Meta+A"))
        result = asyncio.run(backend.type_text(page, MagicMock(), "(&", plan=plan))

    assert combo_handled is True
    assert result.transport == "coregraphics_keyboard"
    assert result.fallback_char_count == 0
    assert fake_bridge.post_key.call_count > 0
    page.keyboard.insert_text.assert_not_awaited()


def test_away_input_backend_records_single_char_fallback():
    fake_bridge = MagicMock()
    with patch("linkedin.input_backends._CoreGraphicsBridge", return_value=fake_bridge), patch(
        "linkedin.input_backends.asyncio.sleep",
        new=AsyncMock(),
    ):
        backend = AwayInputBackend()
        page = MagicMock()
        page.keyboard.insert_text = AsyncMock()
        plan = TypingPlan(
            steps=[TypingStep(kind="char", value="é", delay_seconds=0.0, source_index=0)]
        )
        result = asyncio.run(backend.type_text(page, MagicMock(), "é", plan=plan))

    assert result.fallback_char_count == 1
    page.keyboard.insert_text.assert_awaited_once_with("é")


def test_concurrent_click_rechecks_commit_hook_after_pointer_move():
    backend = ConcurrentInputBackend()
    backend._cursor = MagicMock()
    backend._cursor.move = AsyncMock()
    page = MagicMock()
    page.mouse.down = AsyncMock()
    page.mouse.up = AsyncMock()
    locator = MagicMock()
    locator.element_handle = AsyncMock(return_value=MagicMock())

    def about_to_commit():
        backend._cursor.move.assert_awaited_once()
        raise RuntimeError("profile identity mismatch during save")

    with pytest.raises(RuntimeError, match="profile identity"):
        asyncio.run(
            backend.click_locator(
                page,
                locator,
                about_to_commit=about_to_commit,
            )
        )

    page.mouse.down.assert_not_awaited()


def test_away_click_rechecks_commit_hook_after_pointer_move():
    fake_bridge = MagicMock()
    with patch(
        "linkedin.input_backends._CoreGraphicsBridge",
        return_value=fake_bridge,
    ), patch("linkedin.input_backends.asyncio.sleep", new=AsyncMock()):
        backend = AwayInputBackend()
        backend._locator_screen_point = AsyncMock(return_value=(10, 20))
        backend._move_mouse = AsyncMock()

        def about_to_commit():
            backend._move_mouse.assert_awaited_once()
            raise RuntimeError("profile identity mismatch during save")

        with pytest.raises(RuntimeError, match="profile identity"):
            asyncio.run(
                backend.click_locator(
                    MagicMock(),
                    MagicMock(),
                    about_to_commit=about_to_commit,
                )
            )

    fake_bridge.post_mouse.assert_not_called()


def test_pipeline_passes_input_mode_to_browser():
    with tempfile.TemporaryDirectory() as td, \
         patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser") as mock_browser:
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        mock_brief.return_value = brief

        brief_path = Path(td) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        Pipeline(brief_path=str(brief_path), output_dir=td, input_mode="away")

        # P8.1: Pipeline also passes its governor into LinkedInBrowser now.
        assert mock_browser.call_args.kwargs["input_mode"] == "away"
        assert mock_browser.call_args.kwargs["governor"] is not None


def test_dwell_is_sampled_not_constant():
    random.seed(42)
    samples = [_sample_key_dwell(ch) for ch in "abcdefghijklmnopqrstuvwxyz "]
    assert len(set(samples)) > 1
    mean = sum(samples) / len(samples)
    variance = sum((value - mean) ** 2 for value in samples) / len(samples)
    assert variance > 1e-6


def test_dwell_distribution_is_right_skewed_within_human_band():
    random.seed(12345)
    samples = [_sample_key_dwell("e") for _ in range(5000)]
    samples.sort()
    median = samples[len(samples) // 2]
    mean = sum(samples) / len(samples)
    assert 0.06 <= median <= 0.15
    assert mean > median
    assert all(0.03 <= value <= 0.30 for value in samples)


def test_some_bigrams_produce_overlapping_down_up():
    backend = ConcurrentInputBackend()
    page = MagicMock()
    events: list[tuple[str, str]] = []

    async def record_down(key: str) -> None:
        events.append(("down", key))

    async def record_up(key: str) -> None:
        events.append(("up", key))

    page.keyboard.down = AsyncMock(side_effect=record_down)
    page.keyboard.up = AsyncMock(side_effect=record_up)
    page.keyboard.type = AsyncMock()
    page.keyboard.insert_text = AsyncMock()
    locator = MagicMock()
    plan = TypingPlan(
        steps=[
            TypingStep(kind="char", value=ch, delay_seconds=0.0, source_index=idx)
            for idx, ch in enumerate("abcdefgh")
        ]
    )

    with patch.object(config, "LINKEDIN_TYPING_DWELL_ENABLED", True), patch(
        "linkedin.input_backends.asyncio.sleep",
        new=AsyncMock(),
    ), patch("linkedin.input_backends.random.random", side_effect=[0.0] * 20):
        asyncio.run(backend.type_text(page, locator, "abcdefgh", plan=plan))

    overlap_found = False
    for index in range(1, len(events)):
        prev_kind, prev_key = events[index - 1]
        kind, key = events[index]
        if prev_kind == "down" and kind == "down" and prev_key != key:
            overlap_found = True
            break
        if prev_kind == "down" and kind == "up":
            continue
    assert overlap_found, f"expected overlapping down/up sequence, got {events}"


def test_flag_off_preserves_current_atomic_dispatch_byte_for_byte():
    backend = ConcurrentInputBackend()
    page = MagicMock()
    page.keyboard.type = AsyncMock()
    page.keyboard.down = AsyncMock()
    page.keyboard.up = AsyncMock()
    page.keyboard.insert_text = AsyncMock()
    locator = MagicMock()
    plan = TypingPlan(
        steps=[
            TypingStep(kind="char", value="f", delay_seconds=0.0, source_index=0),
            TypingStep(kind="char", value="o", delay_seconds=0.0, source_index=1),
            TypingStep(kind="char", value="o", delay_seconds=0.0, source_index=2),
        ]
    )

    with patch.object(config, "LINKEDIN_TYPING_DWELL_ENABLED", False):
        asyncio.run(backend.type_text(page, locator, "foo", plan=plan))

    page.keyboard.type.assert_has_awaits(
        [call("f", delay=0), call("o", delay=0), call("o", delay=0)]
    )
    page.keyboard.down.assert_not_awaited()
    page.keyboard.up.assert_not_awaited()
    page.keyboard.insert_text.assert_not_awaited()


def test_overlap_up_failure_does_not_stick_or_double_press():
    backend = ConcurrentInputBackend()
    page = MagicMock()
    down_calls: list[str] = []
    up_calls: list[str] = []
    up_attempts = 0

    async def record_down(key: str) -> None:
        down_calls.append(key)

    async def record_up(key: str) -> None:
        nonlocal up_attempts
        up_attempts += 1
        up_calls.append(key)
        if key == "a" and up_attempts == 1:
            raise RuntimeError("up failed after overlap down")

    page.keyboard.down = AsyncMock(side_effect=record_down)
    page.keyboard.up = AsyncMock(side_effect=record_up)
    page.keyboard.type = AsyncMock()
    page.keyboard.insert_text = AsyncMock()
    locator = MagicMock()
    plan = TypingPlan(
        steps=[
            TypingStep(kind="char", value="a", delay_seconds=0.0, source_index=0),
            TypingStep(kind="char", value="b", delay_seconds=0.0, source_index=1),
        ]
    )

    with patch.object(config, "LINKEDIN_TYPING_DWELL_ENABLED", True), patch(
        "linkedin.input_backends.asyncio.sleep",
        new=AsyncMock(),
    ), patch("linkedin.input_backends.random.random", return_value=0.0):
        asyncio.run(backend.type_text(page, locator, "ab", plan=plan))

    # ch ('a') release attempted after the failed up — best-effort cleanup, not stuck.
    assert up_calls.count("a") >= 2
    # ch was not re-typed atomically (keydown already registered the character).
    page.keyboard.type.assert_not_awaited()
    page.keyboard.insert_text.assert_not_awaited()
    # overlap carry preserved: next_char ('b') downed once via overlap, not again.
    assert down_calls == ["a", "b"]


# ---------------------------------------------------------------------------
# Away mode screen-point math. `box` is viewport-relative, `screenX/screenY`
# locate the window, and the gap between them is browser chrome. Until
# 2026-08-03 nothing reconciled the two, so every OS-level click landed high by
# the height of the tab strip and omnibox and away mode had never worked.
# ---------------------------------------------------------------------------


def _screen_point(
    *,
    box,
    screen_x=100.0,
    screen_y=200.0,
    outer_w=1440.0,
    inner_w=1440.0,
    outer_h=900.0,
    inner_h=812.0,
):
    fake_bridge = MagicMock()
    with patch(
        "linkedin.input_backends._CoreGraphicsBridge", return_value=fake_bridge
    ):
        backend = AwayInputBackend()
    locator = MagicMock()
    locator.scroll_into_view_if_needed = AsyncMock()
    locator.bounding_box = AsyncMock(return_value=box)
    page = MagicMock()
    page.evaluate = AsyncMock(
        return_value={
            "screenX": screen_x,
            "screenY": screen_y,
            "outerWidth": outer_w,
            "outerHeight": outer_h,
            "innerWidth": inner_w,
            "innerHeight": inner_h,
        }
    )
    return asyncio.run(backend._locator_screen_point(page, locator))


def test_screen_point_adds_the_browser_chrome_offset():
    """A viewport y of 0 sits at the BOTTOM of the chrome, not the window top."""
    point = _screen_point(box={"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0})
    # 900 - 812 = 88px of tab strip + omnibox above the viewport.
    assert point == (100.0, 200.0 + 88.0)


def test_screen_point_targets_the_element_centre():
    point = _screen_point(box={"x": 40.0, "y": 60.0, "width": 20.0, "height": 10.0})
    assert point == (100.0 + 40.0 + 10.0, 200.0 + 88.0 + 60.0 + 5.0)


def test_screen_point_offset_tracks_the_measured_chrome_not_a_constant():
    """A bookmarks bar or extension row changes the offset; it is measured."""
    tall = _screen_point(
        box={"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
        outer_h=900.0,
        inner_h=760.0,
    )
    short = _screen_point(
        box={"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
        outer_h=900.0,
        inner_h=812.0,
    )
    assert tall[1] - short[1] == 52.0


def test_screen_point_adds_half_the_horizontal_chrome_as_left_border():
    point = _screen_point(
        box={"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
        outer_w=1450.0,
        inner_w=1440.0,
    )
    assert point[0] == 100.0 + 5.0


def test_screen_point_never_subtracts_when_inner_exceeds_outer():
    """Fullscreen and some zoom states report inner >= outer; clamp at zero."""
    point = _screen_point(
        box={"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0},
        outer_h=800.0,
        inner_h=900.0,
        outer_w=1400.0,
        inner_w=1500.0,
    )
    assert point == (100.0, 200.0)


def test_screen_point_returns_none_without_a_bounding_box():
    assert _screen_point(box=None) is None
