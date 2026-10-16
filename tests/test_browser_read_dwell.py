"""Profile-read dwell budgeting (2026-07-05 SPL run).

The per-chunk profile-read dwell was uncapped log-normal: a single section
could dwell 60s and a long profile compounded to 3+ minutes — inhuman and
slow. _clamp_profile_read_dwell caps each section and budgets the whole
profile; these lock the timing policy (pure, no browser needed).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.browser import LinkedInBrowser, _ProfileReadBudgetExhausted
from linkedin.profile_sections import SectionAnchor
from linkedin.timing_telemetry import TIMING_EVENT_SCHEMAS
from shared import config
from shared.governor import UNGOVERNED_FOR_TESTS


def _clamp(dwell, elapsed):
    return LinkedInBrowser._clamp_profile_read_dwell(
        dwell,
        elapsed,
        max_chunk=LinkedInBrowser._MAX_CHUNK_DWELL,
        max_total=LinkedInBrowser._MAX_PROFILE_READ,
    )


def test_single_section_dwell_capped_at_max_chunk():
    assert _clamp(50.0, 0.0) == LinkedInBrowser._MAX_CHUNK_DWELL
    assert _clamp(3.0, 0.0) == 3.0  # under the cap → untouched (variance preserved)


def test_negative_or_zero_dwell_is_zero():
    assert _clamp(-3.0, 0.0) == 0.0
    assert _clamp(0.0, 0.0) == 0.0


def test_whole_profile_budget_clamps_the_tail():
    # Near the ceiling, the next dwell is clipped to the remaining budget.
    remaining = LinkedInBrowser._MAX_PROFILE_READ - 32.0
    assert _clamp(8.0, 32.0) == remaining
    # Once the budget is spent, lingering stops entirely.
    assert _clamp(8.0, LinkedInBrowser._MAX_PROFILE_READ) == 0.0
    assert _clamp(8.0, LinkedInBrowser._MAX_PROFILE_READ + 5.0) == 0.0


def test_budget_never_exceeded_across_a_long_profile():
    # Simulate many capped chunks accumulating — total lingering cannot pass
    # the ceiling no matter how many chunks or how large each raw sample.
    elapsed = 0.0
    for _ in range(40):
        elapsed += _clamp(100.0, elapsed)  # every raw dwell wildly over the cap
    assert elapsed == LinkedInBrowser._MAX_PROFILE_READ


@pytest.mark.structural
def test_dwell_constants_are_the_lowered_bounded_values():
    # Guards against a silent revert to the uncapped 1.5-4.0 / 2-4x regime.
    assert LinkedInBrowser._BASE_CHUNK_DWELL_LOW == 1.0
    assert LinkedInBrowser._BASE_CHUNK_DWELL_HIGH == 2.5
    assert LinkedInBrowser._MAX_CHUNK_DWELL == 8.0
    assert LinkedInBrowser._MAX_PROFILE_READ == 35.0


def test_exhausted_budget_ends_the_read_rather_than_scrolling_with_zero_delay(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_CADENCE_READ_FIX_ENABLED", True)
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._profile_read_elapsed = LinkedInBrowser._MAX_PROFILE_READ
    with patch("linkedin.browser.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(_ProfileReadBudgetExhausted):
            asyncio.run(browser._profile_read_dwell(5.0))
    sleep_mock.assert_not_awaited()


def test_flag_off_exhausted_budget_does_not_raise(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_CADENCE_READ_FIX_ENABLED", False)
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._profile_read_elapsed = LinkedInBrowser._MAX_PROFILE_READ
    with patch("linkedin.browser.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        asyncio.run(browser._profile_read_dwell(5.0))
    sleep_mock.assert_not_awaited()


def _section_read_anchors() -> list[SectionAnchor]:
    return [
        SectionAnchor(name="about", heading_text="about", offset=100.0),
        SectionAnchor(name="experience", heading_text="experience", offset=500.0),
        SectionAnchor(name="education", heading_text="education", offset=1200.0),
    ]


def _run_section_read(browser, anchors, *, style):
    """Drive _read_section_directed with the traversal style forced, capturing
    every scroll delta. `about`-skip is disabled so the path is deterministic."""
    scroll_deltas: list[int] = []

    async def mock_scroll(delta, *, channel):
        scroll_deltas.append(delta)
        return 1

    browser._human_scroll = mock_scroll
    browser._profile_read_dwell = AsyncMock()

    real_plan = browser._plan_section_traversal
    browser._plan_section_traversal = lambda ts, **_: real_plan(ts, style=style)

    with patch("linkedin.browser.human_delay_correlated", return_value=1.0), patch(
        "linkedin.browser.random.random", return_value=0.99
    ), patch("linkedin.browser.random.randint", return_value=2):
        asyncio.run(browser._read_section_directed(anchors, 0.5))
    return scroll_deltas


# --- The traversal planner: pure, so the policy is pinned without a browser ---


def test_every_style_preserves_the_dwell_budget():
    """Whatever the path, the dwell fractions sum to 1.0 — the redesign's core
    promise that variety costs no dwell time."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    for style in ("straight", "backtracker", "experience_first", "overshoot"):
        with patch("linkedin.browser.random.random", return_value=0.99):
            _, stops = browser._plan_section_traversal(
                _section_read_anchors(), style=style
            )
        assert abs(sum(frac for _, frac, _ in stops) - 1.0) < 1e-9, style


def test_straight_style_is_offset_sorted_even_for_misordered_anchors():
    """The old forward-only guarantee survives inside the `straight` style: a
    misordered anchor list is read top to bottom, never with an upward hop."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    anchors = [
        SectionAnchor(name="about", heading_text="about", offset=100.0),
        SectionAnchor(name="experience", heading_text="experience", offset=1200.0),
        SectionAnchor(name="education", heading_text="education", offset=500.0),
    ]
    with patch("linkedin.browser.random.random", return_value=0.99):
        style, stops = browser._plan_section_traversal(anchors, style="straight")
    assert style == "straight"
    offsets = [section.offset for section, _, _ in stops]
    assert offsets == sorted(offsets) == [100.0, 500.0, 1200.0]


def test_experience_first_visits_experience_before_about():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    with patch("linkedin.browser.random.random", return_value=0.99):
        style, stops = browser._plan_section_traversal(
            _section_read_anchors(), style="experience_first"
        )
    names = [section.name for section, _, _ in stops]
    assert style == "experience_first"
    assert names.index("experience") < names.index("about")


def test_backtracker_revisits_experience():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    with patch("linkedin.browser.random.random", return_value=0.99):
        style, stops = browser._plan_section_traversal(
            _section_read_anchors(), style="backtracker"
        )
    names = [section.name for section, _, _ in stops]
    assert style == "backtracker"
    assert names.count("experience") == 2


def test_a_style_that_cannot_apply_degrades_to_straight():
    """One section can't be reordered or revisited, so the label must not lie."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    one = [SectionAnchor(name="experience", heading_text="experience", offset=500.0)]
    for requested in ("backtracker", "experience_first"):
        style, stops = browser._plan_section_traversal(one, style=requested)
        assert style == "straight"
        assert len(stops) == 1


# --- The executor: the varied path reaches the wheel as real up/down scrolls ---


def test_straight_read_never_hops_backward(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_SECTION_DIRECTED_READ_ENABLED", True)
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    deltas = _run_section_read(browser, _section_read_anchors(), style="straight")
    assert deltas, "expected at least one forward hop"
    assert all(delta >= 0 for delta in deltas)
    assert deltas[0] == 100  # about first
    assert 400 in deltas     # then down to experience


def test_backtracker_read_produces_a_real_upward_scroll(monkeypatch):
    """The end-to-end proof the rigidity is gone: a backtracker read reaches the
    wheel engine with a genuine negative (upward) delta."""
    monkeypatch.setattr(config, "LINKEDIN_SECTION_DIRECTED_READ_ENABLED", True)
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    deltas = _run_section_read(browser, _section_read_anchors(), style="backtracker")
    assert any(delta < 0 for delta in deltas), deltas


def test_read_records_its_traversal_style_for_telemetry(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_SECTION_DIRECTED_READ_ENABLED", True)
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    _run_section_read(browser, _section_read_anchors(), style="experience_first")
    assert browser._section_traversal_style == "experience_first"


def test_medium_interest_budget_lands_in_target_band():
    budget = LinkedInBrowser._interest_read_budget_seconds(0.5)
    assert 8.0 <= budget <= 13.0


def test_low_interest_budget_is_a_fraction_of_high_interest_budget():
    low = LinkedInBrowser._interest_read_budget_seconds(0.0)
    high = LinkedInBrowser._interest_read_budget_seconds(1.0)
    assert low < high * 0.5


def test_out_of_range_interest_is_clamped_to_the_top_of_the_scale():
    """Pins the input clamp on its own.

    Asserting only that the budget stays under _MAX_PROFILE_READ proves
    nothing while the top of the scale is 15s against a 35s ceiling — either
    clamp can be deleted with that assertion still green.
    """
    top = LinkedInBrowser._interest_read_budget_seconds(1.0)
    assert LinkedInBrowser._interest_read_budget_seconds(5.0) == top
    assert LinkedInBrowser._interest_read_budget_seconds(-3.0) == (
        LinkedInBrowser._interest_read_budget_seconds(0.0)
    )


def test_budget_is_capped_by_max_profile_read(monkeypatch):
    """Pins the ceiling on its own, by lowering it under the scale."""
    monkeypatch.setattr(LinkedInBrowser, "_MAX_PROFILE_READ", 5.0)
    assert LinkedInBrowser._interest_read_budget_seconds(1.0) == 5.0


def test_budget_scale_endpoints_are_pinned():
    assert LinkedInBrowser._interest_read_budget_seconds(0.0) == 3.0
    assert LinkedInBrowser._interest_read_budget_seconds(1.0) == 15.0


def test_flag_off_runs_the_legacy_patterns_unchanged(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_SECTION_DIRECTED_READ_ENABLED", False)
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    container = MagicMock()
    container.wait_for = AsyncMock()
    container.evaluate = AsyncMock(side_effect=[1200, 400])
    page = MagicMock()
    page.locator.return_value.first = container
    browser._page = page

    browser._read_focused = AsyncMock()
    browser._read_skipper = AsyncMock()
    browser._read_skimmer = AsyncMock()
    browser._read_section_hopper = AsyncMock()
    browser._read_section_directed = AsyncMock()
    browser._profile_read_dwell = AsyncMock()

    with patch("linkedin.browser.random.choices", return_value=["skipper"]):
        asyncio.run(browser.simulate_profile_read())

    browser._read_section_directed.assert_not_awaited()
    legacy_calls = (
        browser._read_focused.await_count
        + browser._read_skipper.await_count
        + browser._read_skimmer.await_count
        + browser._read_section_hopper.await_count
    )
    assert legacy_calls == 1
    browser._read_skipper.assert_awaited_once()


def test_profile_read_timing_schema_still_validates(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_SECTION_DIRECTED_READ_ENABLED", True)
    events: list[tuple[str, dict]] = []
    browser = LinkedInBrowser(
        governor=UNGOVERNED_FOR_TESTS,
        timing_recorder=lambda event, payload: events.append((event, dict(payload))),
    )
    container = MagicMock()
    container.wait_for = AsyncMock()
    container.evaluate = AsyncMock(side_effect=[1200, 400])
    page = MagicMock()
    page.locator.return_value.first = container
    browser._page = page
    browser._human_scroll = AsyncMock(return_value=1)
    browser._profile_read_dwell = AsyncMock()

    anchors = _section_read_anchors()

    with patch("linkedin.browser.locate_sections", new=AsyncMock(return_value=anchors)):
        asyncio.run(browser.simulate_profile_read(interest=0.5))

    assert events
    event_name, payload = events[-1]
    assert event_name == "profile_read_timing"
    schema = TIMING_EVENT_SCHEMAS["profile_read_timing"]
    for field, expected_type in schema.items():
        assert field in payload
        assert isinstance(payload[field], expected_type)
    # Addendum D: the pattern now carries the traversal style after a "|".
    assert payload["pattern"].startswith("section:about>experience>education")
    style = payload["pattern"].split("|", 1)[1]
    assert style in {"straight", "backtracker", "experience_first", "overshoot"}
    assert payload["chunk_count"] == 3


# --- Results-list scroll (Addendum D): easing bursts + a stepped return ---


def test_results_scroll_chunk_eases_as_the_list_settles():
    """Bursts shrink monotonically as `stable` rises — big while cards load,
    short approaching the bottom. Pins the band bounds, not just the ordering."""
    with patch("linkedin.browser.random.randint", side_effect=lambda a, b: a):
        loading = LinkedInBrowser._results_scroll_chunk(0)
        easing = LinkedInBrowser._results_scroll_chunk(1)
        settling = LinkedInBrowser._results_scroll_chunk(3)
    assert loading == 400 and easing == 240 and settling == 150
    with patch("linkedin.browser.random.randint", side_effect=lambda a, b: b):
        assert LinkedInBrowser._results_scroll_chunk(0) == 700
    # every band is strictly smaller than the one before it, at both extremes
    for pick in (min, max):
        with patch("linkedin.browser.random.randint", side_effect=lambda a, b: pick(a, b)):
            bands = [LinkedInBrowser._results_scroll_chunk(s) for s in (0, 1, 2)]
        assert bands[0] > bands[1] > bands[2], bands


def test_stepped_return_accounts_for_the_whole_distance():
    """The uneven return must sum to exactly -total_scrolled — a stepped return
    that loses or overshoots pixels would leave the list mispositioned."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    deltas: list[int] = []

    async def mock_scroll(delta, *, channel):
        deltas.append(delta)
        return 1

    browser._human_scroll = mock_scroll

    with patch("linkedin.browser.human_delay_correlated", return_value=0.01):
        asyncio.run(browser._scroll_results_to_top(3000))

    assert deltas, "expected at least one return scroll"
    assert all(delta < 0 for delta in deltas)
    assert sum(deltas) == -3000
    assert 2 <= len(deltas) <= 3


def test_stepped_return_is_a_noop_scroll_when_nothing_was_scrolled():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    scroll = AsyncMock()
    browser._human_scroll = scroll
    with patch("linkedin.browser.human_delay_correlated", return_value=0.01):
        asyncio.run(browser._scroll_results_to_top(0))
    scroll.assert_not_awaited()
