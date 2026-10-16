"""Tests for LinkedIn input backend selection and wiring."""

from linkedin.orchestrator import Pipeline as PipelineFacade

import asyncio
import hashlib
import random
import statistics
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
    build_note_typing_plan,
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
         patch("linkedin.evaluation_loop.load_brief") as mock_brief, \
         patch("linkedin.evaluation_loop.init_judger"), \
         patch("linkedin.evaluation_loop.LinkedInBrowser") as mock_browser:
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        mock_brief.return_value = brief

        brief_path = Path(td) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        PipelineFacade(brief_path=str(brief_path), output_dir=td, input_mode="away")._loop

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
# Note typing cadence (CLO-355). A note plan's char delay is the whole
# key-down to next-key-down interval, so the key hold happens INSIDE it. Before
# this, the backend held each key ~90ms and then slept the full delay on top,
# and a 500-char note took ~104s with dwell on. The boolean plan keeps the old
# stacked semantics byte for byte.
# ---------------------------------------------------------------------------

_NOTE_500 = (
    "Staff ML engineer who built the ranking stack for a consumer marketplace "
    "from scratch, then led its move to learned retrieval. At Instacart he owned "
    "the two-tower model that serves search, and his own bullets name the offline "
    "evaluation harness he wrote to gate every launch. Before that he spent three "
    "years on fraud models at Stripe, where he shipped the first streaming feature "
    "store. Strong on production systems, lighter on research papers and talks. "
    "Based in Seattle and open to a hybrid setup."
)
_BOOLEAN_75 = '("machine learning" OR "ML engineer") AND (ranking OR retrieval) NOT intern'


class _RecordingKeyboard:
    """Plain async keyboard: cheaper than AsyncMock across 100k keystrokes."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def down(self, key: str) -> None:
        self.events.append(("down", key))

    async def up(self, key: str) -> None:
        self.events.append(("up", key))

    async def type(self, text: str, delay: int = 0) -> None:
        self.events.append(("type", text))

    async def press(self, key: str) -> None:
        self.events.append(("press", key))

    async def insert_text(self, text: str) -> None:
        self.events.append(("insert", text))


def _run_concurrent_plan(plan, text, *, dwell, seed=None):
    """Type ``plan`` through the concurrent backend; return (sleeps, key events).

    ``seed`` pins the global ``random`` the backend draws dwell and bigram
    overlap from, and the prior global state is restored afterwards.
    """
    page = MagicMock()
    page.keyboard = _RecordingKeyboard()
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        with patch.object(config, "LINKEDIN_TYPING_DWELL_ENABLED", dwell), patch(
            "linkedin.input_backends.asyncio.sleep", new=fake_sleep
        ):
            asyncio.run(
                ConcurrentInputBackend().type_text(page, MagicMock(), text, plan=plan)
            )
    finally:
        random.setstate(state)
    return slept, page.keyboard.events


def _fingerprint(values) -> str:
    return hashlib.sha256(
        "|".join(repr(round(value, 9)) for value in values).encode()
    ).hexdigest()


def test_note_plan_char_interval_absorbs_the_key_hold():
    """Per char, dwell on: time spent is max(interval, hold), never hold + interval."""
    text = "abcdefghijklmnop"
    plan = build_note_typing_plan(text, rng=random.Random(5))
    char_delays = [step.delay_seconds for step in plan.steps if step.kind == "char"]
    assert len(char_delays) == len(text)
    assert all(0.03 < delay < 0.20 for delay in char_delays)
    # Alternate a hold shorter than every interval with one longer than every
    # interval, so both sides of the max() are exercised whatever the knobs are.
    holds = [0.03 if idx % 2 == 0 else 0.20 for idx in range(len(text))]

    with patch(
        "linkedin.input_backends._sample_key_dwell", side_effect=list(holds)
    ), patch("linkedin.input_backends.random.random", return_value=1.0):
        slept, events = _run_concurrent_plan(plan, text, dwell=True)

    expected = sum(max(delay, hold) for delay, hold in zip(char_delays, holds))
    assert sum(slept) == pytest.approx(expected, abs=1e-9)
    # A hold longer than the interval leaves no remainder to sleep, so there is
    # no zero-length sleep either: one sleep per short hold plus its remainder,
    # one per long hold.
    assert len(slept) == len(text) + len(text) // 2
    assert [key for kind, key in events if kind == "down"] == list(text)


def test_note_plan_uses_note_knobs_and_marks_interval_semantics():
    with patch.object(config, "LINKEDIN_NOTE_TYPING_CHAR_MIN_SECONDS", 0.07), patch.object(
        config, "LINKEDIN_NOTE_TYPING_CHAR_MAX_SECONDS", 0.07
    ), patch.object(config, "LINKEDIN_SEARCH_TYPING_CHAR_MIN_SECONDS", 0.5), patch.object(
        config, "LINKEDIN_SEARCH_TYPING_CHAR_MAX_SECONDS", 0.5
    ):
        plan = build_note_typing_plan("Two words. Then more", rng=random.Random(2))

    assert plan.interval_includes_dwell is True
    assert all(
        step.delay_seconds == 0.07 for step in plan.steps if step.kind == "char"
    )
    assert build_boolean_typing_plan("foo AND bar").interval_includes_dwell is False
    assert TypingPlan(steps=[]).interval_includes_dwell is False


def test_note_500_chars_types_in_about_half_the_old_time_with_dwell_on():
    """Seeded simulation through the real backend: mean over 200 notes of 500 chars.

    The pre-fix cadence averaged ~104s here. The floor is the key holds alone:
    ~44s per 500 chars at the current dwell distribution, which no interval
    setting can go under without shortening the hold itself.
    """
    assert len(_NOTE_500) == 500
    totals = []
    for index in range(200):
        plan = build_note_typing_plan(_NOTE_500, rng=random.Random(10_000 + index))
        slept, events = _run_concurrent_plan(plan, _NOTE_500, dwell=True, seed=index)
        totals.append(sum(slept))
        if index == 0:
            # The composer must receive exactly the note (browser.py compares it).
            assert "".join(key for kind, key in events if kind == "down") == _NOTE_500
    mean = statistics.mean(totals)
    assert 48.0 <= mean <= 58.0, f"mean note typing time {mean:.1f}s"
    # Still a human cadence, not a metronome: per-note totals vary, and
    # individual sleeps are spread out.
    assert statistics.pstdev(totals) > 0.3


def test_boolean_plan_and_backend_timing_are_unchanged():
    """Fingerprints captured on the pre-CLO-355 code (a69bfa3) with fixed seeds.

    Covers the plan's delays and the backend's exact sleep sequence and key
    events with dwell on and off, including a seed whose plan carries a typo
    and its backspace correction.
    """
    golden = {
        11: {
            "plan": ("290b823de4c1b246f7818dabf081c41a562d9f34528068bf7d42855a10b4f16b", 84),
            True: (
                "ef2a4901ff138af4b8a749b1649897b42699101beebad36374bdc4bc11704bbf",
                159,
                15.558717787,
                "6051f9e9dec779d43e5ae4a2958d010e138dd7317a1f529eafbe8543e1db4196",
            ),
            False: (
                "290b823de4c1b246f7818dabf081c41a562d9f34528068bf7d42855a10b4f16b",
                84,
                8.996020299,
                "2142b3e2a83313a29f0519cf6a0e6d12d1796a4057e71b809c20275716d36624",
            ),
        },
        3: {
            "plan": ("618de8cea0884aaf4546dd8d42acea4b5dd2ddb3d7f5f10a5cb84cf7aded1859", 85),
            True: (
                "98984995fcb229d8b1b02d35fd3862f7f2dd23627a5eee8512cda506c30fae46",
                161,
                15.428734091,
                "c8c4a7a3273a0debedabfb83a118f4e5806013278ae7d9360a06e5b808e8b43f",
            ),
            False: (
                "618de8cea0884aaf4546dd8d42acea4b5dd2ddb3d7f5f10a5cb84cf7aded1859",
                85,
                9.011549917,
                "19da1189acb37dab977e28ed5c6b5121e0239409ad6ce6003c5bd49fb5c0dad8",
            ),
        },
    }
    for seed, expected in golden.items():
        plan = build_boolean_typing_plan(_BOOLEAN_75, rng=random.Random(seed))
        delays = [step.delay_seconds for step in plan.steps]
        assert (_fingerprint(delays), len(plan.steps)) == expected["plan"], seed
        for dwell in (True, False):
            slept, events = _run_concurrent_plan(plan, _BOOLEAN_75, dwell=dwell, seed=seed)
            sleep_fp, count, total, events_fp = expected[dwell]
            assert round(sum(slept), 9) == total, (seed, dwell, sum(slept))
            assert len(slept) == count, (seed, dwell)
            assert _fingerprint(slept) == sleep_fp, (seed, dwell)
            assert hashlib.sha256(repr(events).encode()).hexdigest() == events_fp, (seed, dwell)


def test_away_backend_applies_the_interval_semantics_to_held_keys_only():
    """CoreGraphics chars post down/up with no hold, so the whole interval sleeps.

    Backspace is the one away-mode key that holds (``_post_key_tap``); under an
    interval plan its hold comes out of the interval, under a boolean plan it
    does not.
    """
    fake_bridge = MagicMock()
    steps = [
        TypingStep(kind="char", value="a", delay_seconds=0.08, source_index=0),
        TypingStep(kind="backspace", delay_seconds=0.08, source_index=0),
    ]
    results = {}
    for interval_plan in (True, False):
        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        with patch(
            "linkedin.input_backends._CoreGraphicsBridge", return_value=fake_bridge
        ), patch("linkedin.input_backends.asyncio.sleep", new=fake_sleep), patch(
            "linkedin.input_backends.time.sleep"
        ) as blocking_sleep, patch(
            "linkedin.input_backends._sample_key_dwell", return_value=0.05
        ), patch.object(config, "LINKEDIN_TYPING_DWELL_ENABLED", True):
            backend = AwayInputBackend()
            page = MagicMock()
            page.keyboard.insert_text = AsyncMock()
            plan = TypingPlan(steps=list(steps), interval_includes_dwell=interval_plan)
            asyncio.run(backend.type_text(page, MagicMock(), "a", plan=plan))
        blocking_sleep.assert_called_once_with(0.05)
        results[interval_plan] = slept

    assert results[False] == [0.08, 0.08]
    assert results[True] == [0.08, pytest.approx(0.03)]


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
