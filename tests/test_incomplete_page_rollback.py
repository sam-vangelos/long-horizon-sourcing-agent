"""Behavior locks for incomplete-page rollback and session expiry."""

import asyncio
import importlib
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from linkedin.browser import LinkedInBrowser
from linkedin.search_intelligence import bootstrap_experiment_state
from shared.governor import UNGOVERNED_FOR_TESTS
from shared.schemas import Progress, SearchString


def _orchestrator_mod():
    """Resolve linkedin.orchestrator at call time, not collection time.

    tests/test_adapter_services.py pops the module from sys.modules and
    reimports it; a Pipeline bound at import here would then resolve its
    globals through the ORPHANED module while ``patch("linkedin.
    orchestrator.X")`` targets the fresh one — the patch lands in the
    wrong namespace and a real LinkedInBrowser gets constructed.
    """

    return importlib.import_module("linkedin.orchestrator")


_PROJECT_SEARCH_URL = (
    "https://www.linkedin.com/talent/hire/test-project/discover/recruiterSearch"
)
_SESSION_EXPIRED_MESSAGE = "LinkedIn session expired — re-authenticate and resume."


def _make_pipeline(tmp_path: Path):
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
        return orch.Pipeline(brief_path=str(brief_path), output_dir=str(tmp_path))


def _prepare_first_page(
    pipeline,
    search_string: SearchString,
    *,
    no_results_visible: bool = False,
) -> tuple[Progress, object]:
    progress = Progress(brief_name="test", strings=[search_string])
    state = bootstrap_experiment_state(search_string)
    pipeline._experiment_states[search_string.id] = state

    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=no_results_visible)
    page = MagicMock()
    type(page).url = PropertyMock(return_value=_PROJECT_SEARCH_URL)
    page.locator.return_value.first = no_results
    pipeline.browser.page = page
    pipeline.browser.get_results_count_text = AsyncMock(return_value="100")
    pipeline.browser.get_results_count = AsyncMock(return_value=100)

    pipeline._prepare_active_allocator_dispatch = MagicMock(return_value=None)
    pipeline._check_allocator_pre_spend = MagicMock()
    pipeline._ensure_browser_healthy = AsyncMock()
    pipeline._apply_opening_search = AsyncMock()
    pipeline._allocator_active_enabled = MagicMock(return_value=False)
    pipeline._validated_owner_pending_full_snippets = MagicMock(return_value=[])
    pipeline._checkpoint_progress = MagicMock()
    return progress, state


def test_mid_page_failure_restores_armed_state_and_propagates(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    search_string = SearchString(
        id=1,
        name="test",
        boolean="foo",
        status="queued",
        notes="before",
    )
    progress, state = _prepare_first_page(pipeline, search_string)
    state.allocator_last_verdict = {"action": "continue", "reason": "before"}
    state.allocator_causality = {"queue_depth": 2}
    state.active_variant.allocator_valid_page_count = 2
    state.active_variant.allocator_completed_observation_count = 3

    armed = {}
    arm = pipeline._arm_incomplete_page_rollback

    def capture_arm(armed_string, armed_state):
        arm(armed_string, armed_state)
        snapshot = pipeline._incomplete_page_rollbacks[armed_string.id]
        armed["experiment_state"] = deepcopy(snapshot["experiment_state"])
        armed["search_string"] = deepcopy(snapshot["search_string"])

    pipeline._arm_incomplete_page_rollback = MagicMock(side_effect=capture_arm)
    failure = RuntimeError("crash during page")

    async def fail_mid_page(**_kwargs):
        state.mode = "mutated"
        state.mutations_used = 99
        state.allocator_last_verdict = {"action": "stop"}
        state.allocator_causality = {"mutated": True}
        state.active_variant.allocator_valid_page_count = 99
        state.active_variant.allocator_completed_observation_count = 99
        state.active_variant.allocator_page_cursor = 99
        search_string.boolean = "mutated"
        search_string.status = "done"
        search_string.result_count = 999
        search_string.pages_reviewed = 99
        search_string.notes = "mutated"
        raise failure

    pipeline._review_page_sequentially = AsyncMock(side_effect=fail_mid_page)

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(pipeline._process_string(search_string, progress))

    expected_string = SearchString.from_dict(armed["search_string"])
    expected_variant = armed["experiment_state"]["variants"][
        armed["experiment_state"]["active_variant_id"]
    ]
    assert raised.value is failure
    assert pipeline._experiment_states[search_string.id] is state
    assert state.to_dict() == armed["experiment_state"]
    assert state.allocator_last_verdict == armed["experiment_state"][
        "allocator_last_verdict"
    ]
    assert (
        state.active_variant.allocator_valid_page_count
        == expected_variant["allocator_valid_page_count"]
    )
    assert (
        state.active_variant.allocator_completed_observation_count
        == expected_variant["allocator_completed_observation_count"]
    )
    assert (
        state.active_variant.allocator_page_cursor
        == expected_variant["allocator_page_cursor"]
    )
    assert (
        search_string.boolean,
        search_string.status,
        search_string.result_count,
        search_string.pages_reviewed,
        search_string.notes,
    ) == (
        expected_string.boolean,
        expected_string.status,
        expected_string.result_count,
        expected_string.pages_reviewed,
        expected_string.notes,
    )
    assert pipeline._incomplete_page_rollbacks == {}


def test_clean_page_completion_discards_rollback_snapshot(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    search_string = SearchString(id=2, name="test", boolean="foo")
    progress, _state = _prepare_first_page(
        pipeline,
        search_string,
        no_results_visible=True,
    )
    arm = MagicMock(wraps=pipeline._arm_incomplete_page_rollback)
    pipeline._arm_incomplete_page_rollback = arm
    pipeline._checkpoint_allocator_exhaustion_transition = MagicMock(
        return_value=None
    )
    pipeline._sync_bounded_page_stats_for_checkpoint = MagicMock()
    pipeline._hydrate_search_string_metadata = MagicMock()

    assert asyncio.run(pipeline._process_string(search_string, progress)) is None

    arm.assert_called_once_with(search_string, pipeline._experiment_states[2])
    assert pipeline._incomplete_page_rollbacks == {}


def test_corrupt_rollback_snapshot_raises_actual_guard_message(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    search_string = SearchString(id=3, name="test", boolean="foo")
    state = bootstrap_experiment_state(search_string)
    pipeline._arm_incomplete_page_rollback(search_string, state)
    pipeline._incomplete_page_rollbacks[search_string.id][
        "experiment_state"
    ] = None

    with pytest.raises(RuntimeError) as raised:
        pipeline._restore_incomplete_page_rollback(search_string)

    assert str(raised.value) == "invalid incomplete-page rollback snapshot"
    assert pipeline._incomplete_page_rollbacks == {}


def _browser_at(url: str) -> LinkedInBrowser:
    browser = LinkedInBrowser(governor=UNGOVERNED_FOR_TESTS)
    page = MagicMock()
    type(page).url = PropertyMock(return_value=url)
    error_heading = MagicMock()
    error_heading.is_visible = AsyncMock(return_value=False)
    page.locator.return_value.first = error_heading
    browser._page = page
    return browser


def test_session_expiry_propagates_through_pipeline_health_check():
    login_url = "https://www.linkedin.com/uas/login?session_redirect=/talent/search"
    browser = _browser_at(login_url)
    check_and_recover = browser.check_and_recover
    browser.check_and_recover = AsyncMock(wraps=check_and_recover)
    browser.navigate_to_search = AsyncMock()

    pipeline_cls = _orchestrator_mod().Pipeline
    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.browser = browser
    pipeline._maybe_cadence_pause = AsyncMock()
    pipeline._record_last_good_url = MagicMock()
    pipeline._apply_session_location_filter = AsyncMock()

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(pipeline._ensure_browser_healthy())

    assert str(raised.value) == _SESSION_EXPIRED_MESSAGE
    browser.check_and_recover.assert_awaited_once()
    browser.navigate_to_search.assert_not_awaited()
    pipeline._apply_session_location_filter.assert_not_awaited()


def test_non_login_recruiter_url_does_not_report_session_expiry():
    browser = _browser_at(_PROJECT_SEARCH_URL)

    assert asyncio.run(browser.check_and_recover()) is False
