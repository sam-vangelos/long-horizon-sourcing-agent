"""Wave 3.2 (part A): zero-results-after-prior-results negative signal."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.governor import UNGOVERNED_FOR_TESTS
from shared.schemas import Progress, SearchString


def _orchestrator_mod():
    return importlib.import_module("linkedin.orchestrator")


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
        pipeline = orch.Pipeline(
            brief_path=str(brief_path),
            output_dir=str(tmp_path),
            governor=UNGOVERNED_FOR_TESTS,
        )
    return pipeline


def _search_string(*, string_id: int = 7) -> SearchString:
    return SearchString(
        id=string_id,
        name="test-string",
        boolean="engineer AND python",
        pages_reviewed=0,
    )


def _progress() -> Progress:
    return Progress(brief_name="test")


async def _drive_zero_results_entry(
    pipeline,
    search_string: SearchString,
    progress: Progress,
    *,
    prior_results: bool,
) -> None:
    if prior_results:
        pipeline._string_ids_with_results.add(search_string.id)

    pipeline._prepare_active_allocator_dispatch = MagicMock(return_value=None)
    pipeline._check_allocator_pre_spend = MagicMock()
    pipeline._ensure_browser_healthy = AsyncMock()
    pipeline._allocator_active_enabled = MagicMock(return_value=True)
    pipeline._checkpoint_allocator_exhaustion_transition = MagicMock(return_value=None)

    experiment_state = MagicMock()
    experiment_state.current_boolean.return_value = search_string.boolean
    experiment_state.active_allocator_page_cursor.return_value = 0
    pipeline._experiment_state_for = MagicMock(return_value=experiment_state)

    pipeline._apply_opening_search = AsyncMock()
    pipeline._record_last_good_url = MagicMock()
    pipeline.browser.page = MagicMock()
    pipeline.browser.page.url = (
        "https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    pipeline.browser.get_results_count_text = AsyncMock(return_value="0")
    pipeline.browser.get_results_count = AsyncMock(return_value=0)

    await pipeline._process_string_impl(search_string, progress)


def test_zero_results_after_prior_results_classifies_as_suspicious(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    search_string = _search_string()
    progress = _progress()

    with patch.object(_orchestrator_mod(), "log_event") as log_event_mock:
        asyncio.run(
            _drive_zero_results_entry(
                pipeline,
                search_string,
                progress,
                prior_results=True,
            )
        )

    suspicious_calls = [
        call
        for call in log_event_mock.call_args_list
        if call.args[1] == "zero_results_after_prior_results"
    ]
    assert len(suspicious_calls) == 1
    assert suspicious_calls[0].kwargs == {
        "string_id": search_string.id,
        "result_count": 0,
    }
    assert search_string.notes.endswith(" Skipped: no results on entry.")
    pipeline._checkpoint_allocator_exhaustion_transition.assert_called_once()


def test_first_time_zero_results_is_not_flagged(tmp_path):
    pipeline = _make_pipeline(tmp_path)
    search_string = _search_string()
    progress = _progress()

    with patch.object(_orchestrator_mod(), "log_event") as log_event_mock:
        asyncio.run(
            _drive_zero_results_entry(
                pipeline,
                search_string,
                progress,
                prior_results=False,
            )
        )

    suspicious_calls = [
        call
        for call in log_event_mock.call_args_list
        if len(call.args) > 1 and call.args[1] == "zero_results_after_prior_results"
    ]
    assert suspicious_calls == []
    assert search_string.notes.endswith(" Skipped: no results on entry.")
    pipeline._checkpoint_allocator_exhaustion_transition.assert_called_once()


def test_zero_results_event_is_registered_in_contracts():
    import shared.contracts

    assert "zero_results_after_prior_results" in shared.contracts.RUN_LOG_EVENTS
