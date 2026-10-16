"""Phase 1 seam-contract tests — recovery_browser cluster.

These pin the PRODUCER->CONSUMER edge of each seam so a refactor cannot
silently sever the wiring.

Only true external boundaries are mocked: the Playwright/CDP page and the
network-bound browser I/O (``enter_search_string`` et al.). The producers
(``_resolve_save_trigger``, ``compile_recovery_plan_from_snapshot``,
``compile_structured_filters_to_plan``) and consumers
(``save_candidate``, ``apply_advanced_search_plan``,
``LinkedInSearchMutationExecutor.apply_variant``) are exercised for real;
every assertion lands on a real value crossing the seam, never on a mock's
return value.

Seam coverage map:
  4.2  pin   — _resolve_save_trigger -> save_candidate (fallback is the clicked one)
  4.3  SKIP  — already pinned (see module docstring on _ghost_click_locator)
  4.1  split — replay_search_context -> compile_recovery_plan_from_snapshot
               -> apply_advanced_search_plan (keyword wired / non-keyword dropped)
  4.4  split — compile_structured_filters_to_plan -> apply_advanced_search_plan
               via LinkedInSearchMutationExecutor (location/title -> unsupported)
  4.0  same-run recovery — canonical checkpoint coverage lives in
                            test_disconnect_path_characterization.py;
                            recovery events remain diagnostic-only
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from linkedin.advanced_search import (
    AdvancedSearchPlan,
    apply_advanced_search_plan,
    compile_recovery_plan_from_snapshot,
)
from linkedin.browser import LinkedInBrowser
from shared.governor import UNGOVERNED_FOR_TESTS
from shared import config
from linkedin.recruiter_recovery import (
    RecruiterRecoverySnapshot,
    replay_search_context,
)


# ---------------------------------------------------------------------------
# Shared fixtures (mirror tests/test_linkedin_browser_advanced_search.py and
# tests/test_linkedin_recruiter_recovery.py construction patterns).
# ---------------------------------------------------------------------------


def _recruiter_search_url() -> str:
    return "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"


def _make_browser_on_search(*, project_id: str | None = "123") -> LinkedInBrowser:
    """A LinkedInBrowser whose only mocked surface is the Playwright page URL.

    Used for the replay seams where the producer (compile_*_from_snapshot) and
    the consumer (apply_advanced_search_plan) must both run for real; the only
    network/browser boundary that gets stubbed is ``enter_search_string``.
    """
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value=_recruiter_search_url())
    browser._page = page
    browser._project_id = project_id
    return browser


# ===========================================================================
# Seam 4.2 (pin) — _resolve_save_trigger emits the semantic fallback and
# save_candidate CLICKS that fallback locator. Pins consumption of the
# fallback, not the resolver in isolation (which is already covered by
# test_resolve_save_trigger_uses_semantic_fallback_on_class_rotation).
# ===========================================================================


def _tx_browser(*, confirmations, tx_result=None, tx_error=None):
    """A browser whose ONLY stubbed surface is the atomic card transaction.

    Resolution and the click now happen inside one browser-side callback
    (`_CARD_SAVE_TRANSACTION_JS`), which is executed for real against a real DOM
    by tools/verify_card_tx.mjs. What these tests own is the PYTHON side: how
    many times actuation may be attempted, and in what order the guard runs.
    """
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    page.url = "https://www.linkedin.com/talent/recruiterSearch/profile/seam-save"
    page.wait_for_timeout = AsyncMock(return_value=None)
    browser._page = page
    browser.set_required_project_id(None)

    events = []

    async def _tx(fragment, *, dry_run):
        if dry_run:
            events.append("resolve")
            return {"ok": True, "dispatched": False}
        events.append("dispatch")
        if tx_error is not None:
            raise tx_error
        return tx_result if tx_result is not None else {"ok": True, "dispatched": True}

    browser._card_save_transaction = _tx

    seq = list(confirmations)

    async def _probe(_fragment):
        events.append("confirm")
        return seq.pop(0) if seq else (confirmations[-1] if confirmations else None)

    browser.is_already_saved_on_card = _probe
    return browser, events


def test_an_unrendered_card_is_scrolled_into_view_before_being_refused():
    """Virtualisation: `card_not_found` must trigger a scroll and ONE re-resolve.

    Measured against live Recruiter 2026-07-27: 21 of 25 result slots were empty
    `profile-list__occlusion-area` placeholders until the list was scrolled, so a
    candidate who IS on the page reads as absent. The browser-side transaction
    cannot see past that on its own — it queries, it does not scroll. Without
    this retry the run would refuse the large majority of its saves.

    The scroll lives in the RETRYABLE resolve phase and presses nothing, which is
    what keeps the commit transaction a pure verify-and-click.
    """
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    page.url = "https://www.linkedin.com/talent/recruiterSearch/profile/seam-save"
    page.wait_for_timeout = AsyncMock(return_value=None)
    browser._page = page
    browser.set_required_project_id(None)

    events = []
    dry_results = [
        {"ok": False, "reason": "card_not_found"},   # unrendered
        {"ok": True, "dispatched": False},           # after the scroll
    ]

    async def _tx(fragment, *, dry_run):
        if dry_run:
            events.append("resolve")
            return dry_results.pop(0) if dry_results else {"ok": True, "dispatched": False}
        events.append("dispatch")
        return {"ok": True, "dispatched": True}

    async def _scroll(fragment):
        events.append("scroll")
        return 0

    browser._card_save_transaction = _tx
    browser._find_result_slot_by_fragment = _scroll
    browser.is_already_saved_on_card = AsyncMock(side_effect=[False, True])

    assert asyncio.run(browser.save_candidate()) is True
    # Scrolled BETWEEN the two resolves, and only then dispatched — once.
    assert events.count("scroll") == 1, events
    assert events.index("scroll") < events.index("dispatch"), events
    assert events.count("dispatch") == 1, events


def test_a_genuinely_absent_card_still_refuses_after_the_scroll():
    """The scroll retry must not turn a real absence into an endless attempt."""
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    page.url = "https://www.linkedin.com/talent/recruiterSearch/profile/seam-save"
    page.wait_for_timeout = AsyncMock(return_value=None)
    browser._page = page
    browser.set_required_project_id(None)

    events = []

    async def _tx(fragment, *, dry_run):
        if dry_run:
            events.append("resolve")
            return {"ok": False, "reason": "card_not_found"}
        events.append("dispatch")
        return {"ok": True, "dispatched": True}

    async def _scroll(fragment):
        events.append("scroll")
        return None

    browser._card_save_transaction = _tx
    browser._find_result_slot_by_fragment = _scroll
    browser.is_already_saved_on_card = AsyncMock(return_value=False)

    assert asyncio.run(browser.save_candidate()) is False
    assert "dispatch" not in events, events
    assert browser._last_save_failure_reason == "save_card_not_found"


def test_save_dispatches_at_most_once_when_the_card_never_confirms():
    """One logical save, one physical press. Pinned because it was three.

    Executed against the pre-fix code: `_retry` wrapped the actuation, so a
    dispatch followed by an unconfirmed read re-entered the click path and
    pressed again — 3 presses for one save, in BOTH the "card still says
    unsaved" and "card unreadable" cases.
    """
    for confirm, want_reason in ((False, "save_not_persisted"), (None, "save_not_confirmed")):
        browser, events = _tx_browser(confirmations=[confirm, confirm, confirm, confirm])
        assert asyncio.run(browser.save_candidate()) is False
        assert events.count("dispatch") == 1, (
            f"confirm={confirm!r} produced {events.count('dispatch')} presses: {events}"
        )
        assert browser._last_save_failure_reason == want_reason


def test_save_dispatches_at_most_once_when_the_transaction_itself_errors():
    """A failure mid-dispatch may already have clicked: never retry it, and never
    record it as a clean did-not-persist.

    Both shapes. A dead target must still propagate (the run has to stop), but
    it must carry the ambiguity with it — this is the one state where Recruiter
    can hold a save that local state cannot confirm.
    """
    browser, events = _tx_browser(
        confirmations=[False], tx_error=RuntimeError("protocol error: malformed reply")
    )
    assert asyncio.run(browser.save_candidate()) is False
    assert events.count("dispatch") == 1, events
    assert browser._last_save_failure_reason == "save_not_confirmed"

    browser, events = _tx_browser(
        confirmations=[False], tx_error=RuntimeError("Target crashed")
    )
    with pytest.raises(RuntimeError, match="Target crashed"):
        asyncio.run(browser.save_candidate())
    assert events.count("dispatch") == 1, events
    assert browser._last_save_failure_reason == "save_not_confirmed"


def test_commit_guard_runs_before_the_press_not_after():
    """Guard-then-press, in one ordered list.

    The previous version of this test asserted the guard ran and that a click
    happened, but never their ORDER — an adversarial review moved before_click()
    to after the click and the test still passed. One event list, one assertion.
    """
    browser, events = _tx_browser(confirmations=[False, True])
    order = []

    def guard():
        order.append("guard")
        events.append("guard")

    assert asyncio.run(browser.save_candidate(before_click=guard)) is True
    assert "guard" in events and "dispatch" in events
    assert events.index("guard") < events.index("dispatch"), events


def test_a_refusal_inside_the_transaction_never_presses():
    """The browser-side transaction can refuse; that must not be a click."""
    for reason, want in (
        ("project_mismatch", "save_project_mismatch"),
        ("card_not_found", "save_card_not_found"),
        ("card_ambiguous", "save_card_ambiguous"),
        ("trigger_ambiguous", "save_trigger_ambiguous"),
    ):
        browser, events = _tx_browser(
            confirmations=[False],
            tx_result={"ok": False, "reason": reason},
        )
        assert asyncio.run(browser.save_candidate()) is False
        assert browser._last_save_failure_reason == want
        # It reached actuation and the BROWSER refused; no confirmation follows.
        assert "confirm" not in events[events.index("dispatch"):], events


def test_already_saved_short_circuits_without_dispatching():
    browser, events = _tx_browser(confirmations=[True])
    assert asyncio.run(browser.save_candidate()) is True
    assert "dispatch" not in events, events
    assert browser._last_save_failure_reason is None


def test_save_candidate_propagates_internal_probe_failure_before_click():
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    browser._page = MagicMock(
        url="https://www.linkedin.com/talent/recruiterSearch/profile/seam-save"
    )
    # Wave G: the probe is the candidate's own CARD, not the panel.
    browser.is_already_saved_on_card = AsyncMock(
        side_effect=RuntimeError("browser context closed")
    )
    browser._ghost_click_locator = AsyncMock()

    with pytest.raises(RuntimeError, match="browser context closed"):
        asyncio.run(browser.save_candidate())

    browser.is_already_saved_on_card.assert_awaited_once()
    browser._ghost_click_locator.assert_not_awaited()
    assert browser._last_save_failure_reason == "save_probe_failed"


# ===========================================================================
# Seam 4.3 — SKIPPED (already pinned).
# tests/test_linkedin_browser_advanced_search.py:
#   - test_ghost_click_locator_returns_true_after_playwright_fallback (396-414)
#   - test_ghost_click_locator_no_fallback_click_when_ghost_succeeds (417-429)
# pin the exact no-double-click consumer guarantee. Not re-pinned here.
# ===========================================================================


# ===========================================================================
# Seam 4.1 (split) — replay_search_context -> compile_recovery_plan_from_snapshot
# -> apply_advanced_search_plan. The keyword Boolean is the only dimension wired
# to the live sidebar; every non-keyword dimension is silently dropped to
# unsupported_controls today (STABLE_NOW_CONTROLS == {'keywords'}).
# ===========================================================================


def _snapshot_with_location_and_keyword() -> RecruiterRecoverySnapshot:
    return RecruiterRecoverySnapshot(
        project_id="123",
        keyword_boolean='"ML" AND "engineer"',
        advanced_search_controls={
            "controls": [
                {"dimension": "locations", "values": ["NYC"]},
            ],
            "acquisition_mode": "linkedin_hybrid",
        },
    )


def test_replay_search_context_reapplies_keyword_boolean_to_browser():
    """PIN (current wired behavior): the keyword Boolean from the snapshot
    crosses into the browser via enter_search_string.

    Producer: replay_search_context -> compile_recovery_plan_from_snapshot
    builds a plan whose only stable_now control is 'keywords'. Consumer:
    apply_advanced_search_plan -> _apply_stable_control awaits
    browser.enter_search_string with the keyword Boolean. enter_search_string
    is the network/browser boundary and is the only thing stubbed.
    """
    browser = _make_browser_on_search()
    browser.enter_search_string = AsyncMock(
        return_value=MagicMock(typing_result=MagicMock(), results_wait_ms=100)
    )
    browser.apply_location_filter = AsyncMock(return_value=True)
    snapshot = _snapshot_with_location_and_keyword()

    ok, reason = asyncio.run(replay_search_context(browser, snapshot))

    assert ok is True
    browser.enter_search_string.assert_awaited()
    # The REAL keyword Boolean (not a mock sentinel) crossed the seam.
    awaited_values = [c.args[0] for c in browser.enter_search_string.await_args_list if c.args]
    assert '"ML" AND "engineer"' in awaited_values
    assert reason


def test_replay_search_context_applies_graduated_location_control():
    """Recovery replay re-applies a graduated 'locations' control to the browser.

    Producer: compile_recovery_plan_from_snapshot reconstructs a 'locations' control
    from the snapshot. Consumer: apply_advanced_search_plan now classifies 'locations'
    as stable_now (hop 4, 2026-05-29 live smoke) and routes it to _apply_stable_control
    -> browser.apply_location_filter, so it lands in applied_controls — recovery restores
    the location filter, not just the keyword Boolean. apply_location_filter is the
    browser boundary and is stubbed; the live DOM is verified by
    tools/hop4_location_smoke.py.
    """
    browser = _make_browser_on_search()
    browser.enter_search_string = AsyncMock(
        return_value=MagicMock(typing_result=MagicMock(), results_wait_ms=100)
    )
    browser.apply_location_filter = AsyncMock(return_value=True)
    snapshot = _snapshot_with_location_and_keyword()

    # The exact producer->consumer edge replay_search_context runs internally.
    plan = compile_recovery_plan_from_snapshot(snapshot)
    assert any(c.dimension == "locations" for c in plan.controls), (
        "producer must reconstruct the 'locations' control from the snapshot"
    )

    result = asyncio.run(apply_advanced_search_plan(browser, plan))

    assert "locations" in result.applied_controls
    assert set(result.applied_controls) == {"keywords", "locations"}
    browser.apply_location_filter.assert_awaited_once()


# ===========================================================================
# Seam 4.4 (split, Phase-2 bridge hop 4) —
# compile_structured_filters_to_plan -> browser.apply_advanced_search_plan
# -> module apply_advanced_search_plan, driven through the real
# LinkedInSearchMutationExecutor. Only the keyword Boolean mutates the live
# sidebar; structured title/location controls are requested but land in
# unsupported_controls.
# ===========================================================================


def _make_hybrid_browser() -> LinkedInBrowser:
    """A real LinkedInBrowser with only network/browser-boundary methods
    stubbed. apply_advanced_search_plan stays REAL so the compiled plan
    genuinely crosses into the advanced_search consumer.
    """
    browser = _make_browser_on_search()
    browser.go_back_to_results = AsyncMock(return_value=None)
    browser.enter_search_string = AsyncMock(
        return_value=MagicMock(typing_result=MagicMock(), results_wait_ms=100)
    )
    browser.get_results_count_text = AsyncMock(return_value="120")
    browser.get_results_count = AsyncMock(return_value=120)
    browser.get_card_snapshot = AsyncMock(return_value={})
    # locations graduated to stable_now (hop 4); stub the browser boundary so the
    # plan routes through it without driving the live DOM (verified by the live smoke).
    browser.apply_location_filter = AsyncMock(return_value=True)
    # job_titles + companies graduated to stable_now (slice H, 2026-05-31); stub those
    # boundaries too so a plan carrying them routes through without the live DOM.
    browser.apply_title_filter = AsyncMock(return_value=True)
    browser.apply_company_filter = AsyncMock(return_value=True)
    return browser


def _make_mutation_executor(browser):
    from linkedin.search_mutation import (
        LinkedInSearchMutationDeps,
        LinkedInSearchMutationExecutor,
    )

    deps = LinkedInSearchMutationDeps(
        browser=browser,
        log_path=Path("test.log"),
        get_input_mode=lambda: "concurrent",
        get_runtime_run_id=lambda: None,
        get_runtime_state=lambda: None,
        get_search_mutation_budget_used=lambda: 0,
        set_search_mutation_budget_used=lambda _v: None,
    )
    return LinkedInSearchMutationExecutor(deps)


def _hybrid_variant_with_title_and_location():
    from linkedin.search_intelligence import (
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )

    return LinkedInSearchVariant(
        variant_id="hybrid-1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"ML" AND "engineer"',
        structured_filters=LinkedInStructuredFilters(
            titles=["VP Engineering"],
            sidebar_filters={"locations": ["NYC"]},
        ),
    )


def _experiment_state():
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
    )

    return LinkedInExperimentState(
        root_string_id=1,
        intent=LinkedInSearchIntent(root_boolean='"ML" AND "engineer"'),
        mode="experiment",
    )


def test_mutation_executor_routes_structured_plan_through_real_apply():
    """PIN + PIN-of-current-reality through one real apply.

    PIN: compile_structured_filters_to_plan (advanced_search.py:195-225, called
    at search_mutation.py:183-189) emits keyword + location + job_titles
    controls; the keyword Boolean reaches the live sidebar via enter_search_string.

    Post-slice-H reality: 'locations' (hop 4) AND 'job_titles' (slice H, 2026-05-31)
    both graduated to stable_now, so both route through their browser-boundary apply
    methods (stubbed) into applied_controls; nothing requested drops, so this is a full
    structured apply, not a partial. apply_advanced_search_plan stays REAL — it delegates
    to the module function — so the compiled plan truly crosses the seam; only the
    browser-boundary methods are stubbed.
    """
    browser = _make_hybrid_browser()
    executor = _make_mutation_executor(browser)
    experiment_state = _experiment_state()
    variant = _hybrid_variant_with_title_and_location()
    experiment_state.variants[variant.variant_id] = variant

    from shared.schemas import SearchString

    search_string = SearchString(id=1, name="hybrid", boolean=variant.boolean)

    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0):
        result = asyncio.run(
            executor.apply_variant(
                search_string=search_string,
                experiment_state=experiment_state,
                variant=variant,
                acquisition_mode="linkedin_hybrid",
            )
        )

    # The structured plan really crossed into the consumer (no mock between).
    assert result.structured_apply is not None
    # locations (hop 4) + job_titles (slice H) both graduated -> both applied; nothing
    # requested dropped, so this is a full structured apply (not a keyword-only partial).
    assert set(result.structured_apply["applied_controls"]) == {"keywords", "locations", "job_titles"}
    assert result.structured_apply["unsupported_controls"] == []
    assert result.hybrid_partial is False
    # The keyword Boolean (alongside the graduated structured dims) hit the live sidebar.
    awaited_values = [c.args[0] for c in browser.enter_search_string.await_args_list if c.args]
    assert '"ML" AND "engineer"' in awaited_values


# ===========================================================================
# Wave 3 Slice 3.1 — SWG ladder reconciliation under live health classifier
# ===========================================================================


def _swg_page_mocks(*, heading_stays_visible: bool):
    """Mocks for a persistent 'Something went wrong' page."""
    page = MagicMock()
    type(page).url = PropertyMock(return_value="https://www.linkedin.com/talent/search")
    page.wait_for_timeout = AsyncMock()
    page.reload = AsyncMock()
    page.goto = AsyncMock()

    error_heading = MagicMock()
    if heading_stays_visible:
        error_heading.is_visible = AsyncMock(return_value=True)
    else:
        error_heading.is_visible = AsyncMock(side_effect=[True, False])

    try_again_btn = MagicMock()
    try_again_btn.is_visible = AsyncMock(return_value=True)
    try_again_btn.click = AsyncMock()

    def locator_side_effect(selector):
        mock_loc = MagicMock()
        if 'text="Something went wrong"' in selector:
            mock_loc.first = error_heading
        elif 'button:has-text("Try again")' in selector:
            mock_loc.first = try_again_btn
        else:
            mock_loc.first = MagicMock()
        return mock_loc

    page.locator = MagicMock(side_effect=locator_side_effect)
    return page, try_again_btn


def test_something_went_wrong_retries_once_then_classifies(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED", True)

    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page, try_again_btn = _swg_page_mocks(heading_stays_visible=True)
    browser._page = page

    recovered = asyncio.run(browser.check_and_recover())

    try_again_btn.click.assert_awaited_once()
    page.reload.assert_not_awaited()
    page.goto.assert_not_awaited()
    assert recovered is False


def test_flag_off_leaves_the_disconnect_only_path_unchanged(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_LIVE_HEALTH_CLASSIFIER_ENABLED", False)

    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page, try_again_btn = _swg_page_mocks(heading_stays_visible=True)
    browser._page = page

    recovered = asyncio.run(browser.check_and_recover())

    try_again_btn.click.assert_awaited_once()
    page.reload.assert_awaited_once()
    page.goto.assert_awaited_once()
    assert recovered is True
