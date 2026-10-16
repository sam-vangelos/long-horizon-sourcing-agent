"""Offline recovery acceptance: actual loop, canonical writes, and actuator truth."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluation_loop_replay import assert_recording, replay
from linkedin.evaluation_loop import EvaluationLoop
from linkedin.evaluation_state import EvaluationRunState
from linkedin.evaluation_errors import (
    GeographyRegimeError, PanelRecoveryError, ProjectContextMismatchError,
    SurfaceOwnershipLostError,
)
from shared.governor import GovernorLimitReached, OperatorStopRequested, SessionExpired
from shared.schemas import SearchString
from linkedin.search_intelligence import bootstrap_experiment_state
from tests.test_full_eval_pipelining import _make_pipeline

FIXTURES = Path(__file__).parent / "fixtures" / "evaluation_loop_recovery"


def rows(effects, table):
    return next(item["value"]["rows"] for item in effects
                if item["effect"] == "runtime.rows" and item["value"]["table"] == table)


def recovery_attempts(effects):
    return [row["payload_json"] for row in rows(effects, "events")
            if row["event_type"] == "page_local_recovery"
            and row["payload_json"]["status"] == "reanchoring"]


def assert_recovery_outcome(name, effects):
    escaped = next(item["value"] for item in effects if item["effect"] == "escape")
    expected_error = {
        "pagination_limit": "TransientPaginationError", "render_limit": "PageRenderFailedError",
        "detection_stop": "GovernorLimitReached", "detection_page": "GovernorLimitReached",
        "panel_limit": "PanelRecoveryError", "save_limit": "UnconfirmedSaveError",
        "geography_stop": "GeographyRegimeError", "surface_stop": "SurfaceOwnershipLostError",
        "project_stop": "ProjectContextMismatchError",
    }.get(name)
    assert (escaped or {}).get("type") == expected_error, (name, escaped)
    if name == "save_limit":
        assert escaped["message"] == "LinkedIn save was not durably confirmed"
    attempts = recovery_attempts(effects)
    expected_attempts = (2 if name.endswith("_limit") else
                         0 if name.endswith("_stop") or name == "detection_page" else 1)
    assert len(attempts) == expected_attempts, (name, attempts)
    assert [item["session_attempts"] for item in attempts] == list(range(1, expected_attempts + 1))
    if name.endswith("_limit"):
        assert [item["page_attempts"] for item in attempts] == [1, 2]
    full = [row for row in rows(effects, "candidate_attempts") if row["stage"] == "full"]
    save_cases = {"render_second_page", "render_recovered", "panel_recovered", "save_confirmed", "panel_limit", "save_limit", "panel_before_save"}
    full_calls = sum(item["effect"] == "llm.full" for item in effects)
    presses = [item for item in effects if item["effect"] == "browser.call"
               and item["value"]["method"] == "save_candidate"]
    assert len(presses) == (1 if name in save_cases else 0), name
    assert full_calls == (1 if name in save_cases else 0), name
    assert [row["status"] for row in full] == (
        ["failed"] if name == "save_limit" else ["succeeded"] if name in save_cases else []
    ), name
    stats = next(item["value"] for item in effects if item["effect"] == "stats")
    assert stats["saved"] == (1 if name in save_cases - {"save_limit"} else 0), name
    if name in {"save_confirmed", "save_limit"}:
        saves = [row for row in rows(effects, "side_effects") if row["effect_type"] == "linkedin_save"]
        assert len(saves) == 1
        assert saves[0]["attempt_count"] == (2 if name == "save_confirmed" else 3)
        assert saves[0]["status"] == ("succeeded" if name == "save_confirmed" else "failed")
    if name == "detection_page":
        assert sum(item["effect"] == "governor.force_backoff" for item in effects) == 1
        assert not any(item["effect"] == "browser.call" and item["value"]["method"] in {"go_to_next_page", "reload", "reload_search_page"} for item in effects)
    # Recovery may navigate a completed page again but cannot rejudge it.
    if name.startswith("pagination_"):
        assert sum(item["effect"] == "llm.facial_batch" for item in effects) == (
            2 if name == "pagination_recovered" else 1
        )
    if name == "render_second_page":
        assert sum(item["effect"] == "llm.facial_batch" for item in effects) == 2
        unit = max(rows(effects, "work_units"), key=lambda row: row["run_id"])
        assert unit["checkpoint_json"]["experiment_state"]["variants"]["root"]["allocator_page_cursor"] == 3
    for item in effects:
        if item["effect"] == "console" and "[page-recovery]" in item["value"]:
            assert "\n" not in item["value"]


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda path: path.stem)
def test_page_local_recovery_recording(path, tmp_path):
    recording = json.loads(path.read_text())
    effects = replay(recording["input"], tmp_path)
    assert_recovery_outcome(recording["name"], effects)
    assert_recording(recording["effects"], effects, name=recording["name"])


def test_flag_is_read_once_at_pipeline_construction(tmp_path, monkeypatch):
    monkeypatch.delenv("LINKEDIN_PAGE_LOCAL_RECOVERY_ENABLED", raising=False)
    (tmp_path / "off").mkdir()
    (tmp_path / "on").mkdir()
    off = _make_pipeline(str(tmp_path / "off"))
    monkeypatch.setenv("LINKEDIN_PAGE_LOCAL_RECOVERY_ENABLED", "true")
    on = _make_pipeline(str(tmp_path / "on"))
    monkeypatch.setenv("LINKEDIN_PAGE_LOCAL_RECOVERY_ENABLED", "false")
    assert off.state.surface.budget.enabled is False
    assert on.state.surface.budget.enabled is True


def budget_loop():
    state = EvaluationRunState()
    state.surface.budget.enabled = True
    state._experiment_states = {}
    loop = EvaluationLoop(state)
    loop.state.surface._check_page_recovery_authority = AsyncMock()
    loop.state.surface._check_page_recovery_health = AsyncMock()
    loop.state.surface._reanchor_recovery_page = AsyncMock(return_value=True)
    loop.state.surface.attempts = MagicMock()
    loop.state.surface.attempts._record_runtime_event = MagicMock()
    loop.state.surface._session_geography_values = lambda: []
    return loop


def add_owner(loop, number):
    owner = SearchString(id=number, name=f"owner {number}", boolean="test")
    loop.state._experiment_states[owner.id] = bootstrap_experiment_state(owner)
    return owner


def test_recovery_success_spends_shared_page_and_session_budget():
    loop = budget_loop()
    owner = add_owner(loop, 1)
    async def run():
        for fault in ("render", "panel"):
            assert await loop.state.surface._try_page_recovery(owner=owner, page_num=1, fault=fault)
        assert not await loop.state.surface._try_page_recovery(owner=owner, page_num=1, fault="save confirmation")
        for number in range(2, 6):
            assert await loop.state.surface._try_page_recovery(owner=add_owner(loop, number), page_num=1, fault="pagination")
        assert not await loop.state.surface._try_page_recovery(owner=add_owner(loop, 6), page_num=1, fault="render")
    asyncio.run(run())
    assert loop.state.surface._reanchor_recovery_page.await_count == 6
    assert loop.state.surface.budget.used == 6


@pytest.mark.parametrize("danger", [
    GeographyRegimeError("geo"), SurfaceOwnershipLostError("surface"),
    ProjectContextMismatchError("project"), GovernorLimitReached("blocked_or_rate_limited"),
    OperatorStopRequested(), SessionExpired("expired"),
])
def test_panel_wrapper_never_retries_a_dangerous_first_error(danger, tmp_path):
    loop = budget_loop()
    owner = add_owner(loop, 1)
    loop.state.context.browser = SimpleNamespace(go_back_to_results=AsyncMock(side_effect=[danger, None]))
    loop.state.context.log_path = tmp_path / "panel-events.jsonl"
    with patch("linkedin.search_surface.log_event"):
        with pytest.raises(type(danger)) as caught:
            asyncio.run(loop.state.surface._recover_stuck_profile_panel(candidate_name="x", page_num=1, search_string=owner))
    assert caught.value is danger
    loop.state.context.browser.go_back_to_results.assert_awaited_once()
    loop.state.surface._reanchor_recovery_page.assert_not_awaited()
    assert loop.state.surface.budget.used == 0


def test_recovery_propagates_danger_in_exception_chain():
    loop = budget_loop()
    owner = add_owner(loop, 1)
    original = GeographyRegimeError("lost geography")
    wrapper = PanelRecoveryError("panel failed")
    wrapper.__cause__ = original
    with pytest.raises(GeographyRegimeError) as caught:
        asyncio.run(loop.state.surface._try_page_recovery(owner=owner, page_num=1, fault="panel", error=wrapper))
    assert caught.value is original
    assert loop.state.surface.budget.used == 0
    loop.state.surface._reanchor_recovery_page.assert_not_awaited()


@pytest.mark.parametrize("probe", [True, False, None])
@pytest.mark.parametrize("cached", [False, True])
def test_confirmation_only_requires_an_exact_probe_and_never_presses(tmp_path, probe, cached):
    from linkedin.side_effects import LinkedInSideEffectsDeps, LinkedInSideEffectsService
    from tests.test_full_eval_pipelining import FakeBrowser, _snippet

    snippet = _snippet("Confirm", 1)
    browser = FakeBrowser()
    browser.show_profile(snippet.profile_url)
    browser.is_already_saved_on_card = AsyncMock(return_value=probe)
    browser.save_candidate = AsyncMock(side_effect=AssertionError("confirmation pressed Save"))
    guard = MagicMock()
    stats = {"saved": 0}
    service = LinkedInSideEffectsService(LinkedInSideEffectsDeps(
        browser=browser, stats=stats, saved_urls={snippet.profile_url} if cached else set(),
        log_path=tmp_path / "log.jsonl", get_test_mode=lambda: False,
        get_runtime_bridge=lambda: None, get_runtime_run_id=lambda: None,
        assert_project_context=guard,
    ))
    outcome = asyncio.run(service.handle_save_decision(
        snippet=snippet, runtime_search_string=SearchString(id=7, name="owner", boolean="test"),
        attempt_id=None, confirmation_only=True,
    ))
    assert outcome.status == ("succeeded" if probe is True else "failed")
    browser.is_already_saved_on_card.assert_awaited_once()
    browser.save_candidate.assert_not_awaited()
    assert guard.call_count == 2
    assert stats["saved"] == 0


def test_geography_check_cannot_self_heal_and_reset_a_seeked_page():
    loop = EvaluationLoop(EvaluationRunState())
    loop.state.surface.budget.enabled = True
    browser = SimpleNamespace(read_applied_location_chips=AsyncMock(return_value=[]), page_num=3)
    loop.state.context.browser = browser
    loop.state.surface._check_page_recovery_health = AsyncMock()
    loop.state.surface._session_geography_values = lambda: ["United States"]
    loop.state.surface._honor_irreversible_side_effect_boundary = MagicMock()
    loop.state.surface._verify_owned_facets_after_reload = AsyncMock()

    async def self_heal():
        browser.page_num = 1

    loop.state.surface._verify_session_geography_chips = AsyncMock(side_effect=self_heal)
    with pytest.raises(GeographyRegimeError):
        asyncio.run(loop.state.surface._check_page_recovery_authority(SearchString(id=1, name="owner", boolean="x")))
    assert browser.page_num == 3
    loop.state.surface._verify_session_geography_chips.assert_not_awaited()


def _reload_verification_browser(*, rail: list[str], substituted: dict[str, str],
                                 owned: set[str], unapplied: list[str] | None = None,
                                 cache: dict[str, str] | None = None):
    """A browser whose rail is `rail` and whose last reconcile said what landed.

    `cache` is the resolution sidecar, in memory and keyed the way the sidecar
    keys it (`facet_name_key` of the request).
    """
    from linkedin.browser import LinkedInBrowser

    browser = LinkedInBrowser.__new__(LinkedInBrowser)
    browser._page = SimpleNamespace(wait_for_timeout=AsyncMock())
    browser._state_dir = None
    browser._facet_name_resolutions = {
        "companies": dict(cache or {}),
        "titles": {},
    }
    browser.last_facet_reconciliation = {
        "unapplied": {"companies": list(unapplied or []), "titles": []},
        "substituted": {"companies": dict(substituted), "titles": {}},
    }
    browser._owned_facet_target = {"companies": set(owned), "titles": set()}
    browser.read_applied_company_chips = AsyncMock(return_value=list(rail))
    browser.read_applied_title_chips = AsyncMock(return_value=[])
    return browser


def _verify_after_reload(loop, owner):
    return loop.state.surface._verify_owned_facets_after_reload(
        search_string=owner, boundary="zero_slot_reload"
    )


def test_post_reload_verification_accepts_the_name_that_actually_landed(tmp_path):
    """The check must not refuse the rail its own reconcile proved.

    "Mercado Livre" is taken as LinkedIn's "Mercado Livre Brasil", so the
    owned target holds the landed spelling while the variant still asks for
    the request. The subset guard (`owned <= request`) rejected the owned
    target because a landed spelling is not one of the requested strings,
    fell back to the request, and `verify_only` never substitutes — so the
    rail key "mercado livre brasil" was compared with "mercado livre", a
    correct page raised, and the strike handler navigated off a live result
    surface to repair nothing.

    RED against the subset guard: FacetReconciliationError.
    """
    loop = budget_loop()
    loop.state.context.log_path = tmp_path / "facet-events.jsonl"
    owner = add_owner(loop, 1)
    filters = loop.state._experiment_states[owner.id].active_variant.structured_filters
    filters.companies = ["Mercado Livre"]
    browser = _reload_verification_browser(
        rail=["Mercado Livre Brasil"],
        substituted={"Mercado Livre": "Mercado Livre Brasil"},
        owned={"Mercado Livre Brasil"},
    )
    loop.state.context.browser = browser

    asyncio.run(_verify_after_reload(loop, owner))

    assert browser.last_facet_reconciliation["failure_gate"] == ""


def test_post_reload_verification_accepts_a_cached_substitution(tmp_path):
    """The sidecar is the other way a landed spelling is accounted for.

    A string whose reconcile short-circuited on an already-applied pill has no
    substitution on its receipt, so the cache is what says "Mercado Livre
    Brasil" is this request's spelling. Neither existing post-reload test
    seeded a cache, so deleting the cache loop kept both of them green.
    """
    loop = budget_loop()
    loop.state.context.log_path = tmp_path / "facet-events.jsonl"
    owner = add_owner(loop, 1)
    filters = loop.state._experiment_states[owner.id].active_variant.structured_filters
    filters.companies = ["Mercado Livre"]
    browser = _reload_verification_browser(
        rail=["Mercado Livre Brasil"],
        substituted={},
        owned={"Mercado Livre Brasil"},
        cache={"mercado livre": "Mercado Livre Brasil"},
    )
    loop.state.context.browser = browser

    asyncio.run(_verify_after_reload(loop, owner))

    assert browser.last_facet_reconciliation["failure_gate"] == ""


def test_a_cached_spelling_cannot_account_for_a_pill_the_receipt_calls_unapplied(
    tmp_path,
):
    """The receipt beats the sidecar about what this string actually landed.

    A wrong typeahead take writes "banco do brasil" -> "Banco" to the sidecar.
    The next string's reconcile refuses the name and records it `unapplied`,
    which forgets the entry — but any receipt that still lists the request
    under `unapplied` is the rail's word that no pill was proved for it, so
    the cached spelling must not be what blesses a "Banco" pill on reload.

    RED against the unconditional cache loop: the check accepted the owned
    target and the verification passed.
    """
    from linkedin.browser import FacetReconciliationError

    loop = budget_loop()
    loop.state.context.log_path = tmp_path / "facet-events.jsonl"
    owner = add_owner(loop, 1)
    filters = loop.state._experiment_states[owner.id].active_variant.structured_filters
    filters.companies = ["Banco do Brasil"]
    browser = _reload_verification_browser(
        rail=["Banco"],
        substituted={},
        owned={"Banco"},
        unapplied=["Banco do Brasil"],
        cache={"banco do brasil": "Banco"},
    )
    loop.state.context.browser = browser

    with pytest.raises(FacetReconciliationError):
        asyncio.run(_verify_after_reload(loop, owner))


def test_post_reload_verification_still_refuses_a_target_this_string_never_proved(
    tmp_path,
):
    """The guard survives: an owned set the receipt cannot account for is not
    this string's evidence, so the request stands and the mismatch raises."""
    from linkedin.browser import FacetReconciliationError

    loop = budget_loop()
    loop.state.context.log_path = tmp_path / "facet-events.jsonl"
    owner = add_owner(loop, 1)
    filters = loop.state._experiment_states[owner.id].active_variant.structured_filters
    filters.companies = ["Mercado Livre"]
    browser = _reload_verification_browser(
        rail=["Lojas Riachuelo"], substituted={}, owned={"Lojas Riachuelo"},
    )
    loop.state.context.browser = browser

    with pytest.raises(FacetReconciliationError):
        asyncio.run(_verify_after_reload(loop, owner))


def test_challenge_after_navigation_prevents_all_restore_mutations():
    loop = EvaluationLoop(EvaluationRunState())
    loop.state.surface.budget.enabled = True
    owner = SearchString(id=1, name="owner", boolean="x")
    loop.state._experiment_states = {1: bootstrap_experiment_state(owner)}
    loop.state.context.browser = SimpleNamespace(
        navigate_to_search=AsyncMock(),
        page=SimpleNamespace(url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch",
                             title=AsyncMock(return_value="Too many requests"),
                             locator=lambda _: SimpleNamespace(inner_text=AsyncMock(return_value=""))),
    )
    loop.state.context._governor = SimpleNamespace(force_backoff=MagicMock())
    loop.state.surface.geography = SimpleNamespace(reset_location_applied=MagicMock(),
                                                    _apply_session_location_filter=AsyncMock())
    loop.state.surface._get_project_url = lambda: loop.state.context.browser.page.url
    loop.state.surface._honor_irreversible_side_effect_boundary = MagicMock()
    loop.state.surface._apply_opening_search = AsyncMock()
    with pytest.raises(GovernorLimitReached):
        asyncio.run(loop.state.surface._reanchor_recovery_page(owner, 3, "root"))
    loop.state.surface.geography.reset_location_applied.assert_not_called()
    loop.state.surface.geography._apply_session_location_filter.assert_not_awaited()
    loop.state.surface._apply_opening_search.assert_not_awaited()
    loop.state.context._governor.force_backoff.assert_called_once_with("blocked_or_rate_limited")


def test_swallowed_next_failure_is_classified_before_any_retry():
    loop = EvaluationLoop(EvaluationRunState())
    loop.state.surface.budget.enabled = True
    loop.state.context.browser = SimpleNamespace(go_to_next_page=AsyncMock(return_value=False),
                                         scroll_to_load_all_results=AsyncMock(),
                                         reload_search_page=AsyncMock())
    loop.state.surface._check_page_recovery_health = AsyncMock(side_effect=[None, GovernorLimitReached("blocked_or_rate_limited")])
    with pytest.raises(GovernorLimitReached):
        asyncio.run(loop.state.surface._attempt_page_advance(result_count=50, page_num=1))
    loop.state.context.browser.go_to_next_page.assert_awaited_once()
    loop.state.context.browser.scroll_to_load_all_results.assert_not_awaited()
    loop.state.context.browser.reload_search_page.assert_not_awaited()


@pytest.mark.parametrize("page_offset", [None, 25])
def test_missing_or_ambiguous_first_page_proof_refuses_recovery(tmp_path, page_offset):
    spec = json.loads((FIXTURES / "pagination_limit.json").read_text())["input"]
    spec["page_offset"] = page_offset
    effects = replay(spec, tmp_path)
    escaped = next(item["value"] for item in effects if item["effect"] == "escape")
    assert escaped["type"] == "TransientPaginationError"
    assert len(recovery_attempts(effects)) == 2
    # Only legacy attempts ran: neither reanchor could prove page one.
    assert sum(item["effect"] == "browser.call" and item["value"]["method"] == "go_to_next_page" for item in effects) == 4


def test_unreadable_or_unchanged_identity_never_proves_a_seek():
    loop = budget_loop()
    owner = add_owner(loop, 1)
    loop.state.context.browser = SimpleNamespace(go_to_next_page=AsyncMock(return_value=True))
    for identities in (("",), ("same", "same"), ("before", "")):
        loop.state.surface._recovery_top_identity = AsyncMock(side_effect=identities)
        assert asyncio.run(loop.state.surface._advance_recovery_page(owner)) is False
    assert loop.state.context.browser.go_to_next_page.await_count == 2


def test_exact_geography_restore_cannot_start_a_resolution_model(tmp_path):
    from linkedin.geography_gate import GeographyGateDeps, GeographyGateService
    browser = SimpleNamespace(apply_location_filter=AsyncMock(return_value=False))
    brief = SimpleNamespace(permanent_filters={"Location": "United States"}, geography_source="preflight")
    service = GeographyGateService(GeographyGateDeps(
        get_browser=lambda: browser, get_brief_obj=lambda: brief,
        log_path=tmp_path / "log.jsonl", stats={},
    ))
    service._resolve_and_reapply_geography = AsyncMock(side_effect=AssertionError("recovery reinterpreted geography"))
    with pytest.raises(GeographyRegimeError):
        asyncio.run(service._apply_session_location_filter(allow_resolution=False))
    service._resolve_and_reapply_geography.assert_not_awaited()


@pytest.mark.parametrize("kind", ["runtime", "browser"])
def test_navigation_hydration_and_transport_failures_spend_bounded_attempts(kind):
    from rebrowser_playwright.async_api import Error
    loop = budget_loop()
    owner = add_owner(loop, 1)
    failure = RuntimeError("sidebar elements not found after 16s") if kind == "runtime" else Error("net::ERR_ABORTED")
    loop.state.context.browser = SimpleNamespace(navigate_to_search=AsyncMock(side_effect=failure))
    loop.state.surface._get_project_url = lambda: "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    loop.state.surface._reanchor_recovery_page = type(loop.state.surface)._reanchor_recovery_page.__get__(loop.state.surface)
    assert asyncio.run(loop.state.surface._try_page_recovery(owner=owner, page_num=2, fault="render")) is False
    assert loop.state.context.browser.navigate_to_search.await_count == 2
    assert loop.state.surface.budget.used == 2


def test_persistence_failure_during_recovery_does_not_retry_browser_work():
    loop = budget_loop()
    owner = add_owner(loop, 1)
    failure = OSError("recorded disk full")
    loop.state.surface.attempts._record_runtime_event = MagicMock(side_effect=failure)
    with pytest.raises(OSError) as caught:
        asyncio.run(loop.state.surface._try_page_recovery(owner=owner, page_num=1, fault="render"))
    assert caught.value is failure
    loop.state.surface._reanchor_recovery_page.assert_not_awaited()


def test_unverifiable_structured_controls_do_not_broaden_the_search():
    loop = budget_loop()
    owner = add_owner(loop, 1)
    loop.state._experiment_states[owner.id].active_variant.structured_filters.skills = ["unsupported skill control"]
    assert asyncio.run(loop.state.surface._try_page_recovery(owner=owner, page_num=1, fault="panel")) is False
    assert loop.state.surface.budget.used == 0
    loop.state.surface._reanchor_recovery_page.assert_not_awaited()


def test_historical_panel_recovery_keeps_checkpoint_ownership_separate(tmp_path):
    from shared.schemas import Progress

    loop = budget_loop()
    owner = add_owner(loop, 1)
    progress = Progress(brief_name="test", strings=[owner], current_page=5)
    loop.state.context.browser = SimpleNamespace(go_back_to_results=AsyncMock(side_effect=RuntimeError("close miss")))
    loop.state.context.log_path = tmp_path / "panel-events.jsonl"
    loop.state.surface._try_page_recovery = AsyncMock(return_value=True)
    loop.state._allocator_service._checkpoint_progress = MagicMock()
    with patch("linkedin.search_surface.log_event"):
        asyncio.run(loop.state.surface._recover_stuck_profile_panel(
            candidate_name="Historical candidate", page_num=2, progress=progress,
            search_string=None, recovery_owner=owner,
        ))
    assert loop.state.surface._try_page_recovery.await_args.kwargs["owner"] is owner
    assert loop.state.surface._try_page_recovery.await_args.kwargs["page_num"] == 2
    assert progress.current_page == 5
    loop.state._allocator_service._checkpoint_progress.assert_not_called()


def test_cancellation_cannot_start_recovery_during_paid_worker_cleanup():
    loop = EvaluationLoop(EvaluationRunState())
    loop.state.context.browser = object()

    async def scenario():
        task = asyncio.current_task()
        task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        with pytest.raises(asyncio.CancelledError):
            await loop.state.surface._check_page_recovery_health()

    with patch("linkedin.recruiter_recovery.detect_recruiter_health", new=AsyncMock(side_effect=AssertionError("recovery began after cancellation"))) as health:
        asyncio.run(scenario())
    health.assert_not_awaited()
