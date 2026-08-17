from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin import acquisition, search_mutation, side_effects, work_units
from linkedin.acquisition import LinkedInAcquisitionDeps, LinkedInAcquisitionService
from linkedin.search_mutation import (
    LinkedInSearchMutationDeps,
    LinkedInSearchMutationExecutor,
)
from linkedin.side_effects import LinkedInSideEffectsDeps, LinkedInSideEffectsService
from linkedin.work_units import LinkedInWorkUnitDeps, LinkedInWorkUnitService
from shared.schemas import Progress, SearchString


class _Browser:
    pass


async def _ensure_browser_healthy() -> None:
    return None


def _card_acquisition_service(tmp_path) -> LinkedInAcquisitionService:
    browser = MagicMock()
    browser.focus_card_for_review = AsyncMock()
    browser.get_card_snapshot = AsyncMock(
        return_value={
            "innertext": "Select Ada Lovelace\nAda Lovelace\nML Engineer",
            "name": "Ada Lovelace",
            "url": "/talent/profile/ada",
        }
    )
    return LinkedInAcquisitionService(
        LinkedInAcquisitionDeps(
            browser=browser,
            log_path=tmp_path / "run_log.jsonl",
            ensure_browser_healthy=_ensure_browser_healthy,
        )
    )


@pytest.mark.parametrize(
    "message",
    ["invalid api key", "browser context closed"],
)
def test_card_extraction_propagates_systemic_failures(tmp_path, message) -> None:
    service = _card_acquisition_service(tmp_path)

    with patch(
        "linkedin.acquisition.extract_snippet_from_card_innertext",
        side_effect=RuntimeError(message),
    ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
        with pytest.raises(RuntimeError, match=message):
            asyncio.run(
                service.extract_card_snippet(
                    SearchString(id=1, name="builders", boolean="foo"),
                    page_num=1,
                    card_index=0,
                )
            )


def test_card_extraction_keeps_json_parse_failure_local(tmp_path) -> None:
    service = _card_acquisition_service(tmp_path)

    with patch(
        "linkedin.acquisition.extract_snippet_from_card_innertext",
        side_effect=RuntimeError("Could not parse JSON from LLM response: malformed"),
    ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
        result = asyncio.run(
            service.extract_card_snippet(
                SearchString(id=1, name="builders", boolean="foo"),
                page_num=1,
                card_index=0,
            )
        )

    assert result is None
    assert service.last_card_extraction_error is not None


def test_card_without_extractable_name_is_not_an_extraction_failure(tmp_path) -> None:
    service = _card_acquisition_service(tmp_path)

    with patch(
        "linkedin.acquisition.extract_snippet_from_card_innertext",
        return_value=None,
    ), patch("linkedin.acquisition.human_delay_correlated", return_value=0.0):
        result = asyncio.run(
            service.extract_card_snippet(
                SearchString(id=1, name="builders", boolean="foo"),
                page_num=1,
                card_index=0,
            )
        )

    assert result is None
    assert service.last_card_extraction_error is None


def test_linkedin_collaborator_modules_do_not_type_against_pipeline() -> None:
    for module in (acquisition, search_mutation, side_effects, work_units):
        source = inspect.getsource(module)
        assert "from linkedin.orchestrator import Pipeline" not in source
        assert '"Pipeline"' not in source
        assert "'Pipeline'" not in source


def test_linkedin_collaborators_construct_from_explicit_deps(tmp_path) -> None:
    browser = _Browser()
    runtime = {"run_id": None, "progress": None, "states": {}, "memory": {}, "budget": 0}

    def set_run_id(run_id: int | None) -> None:
        runtime["run_id"] = run_id

    def set_progress(progress) -> None:
        runtime["progress"] = progress

    def set_states(states) -> None:
        runtime["states"] = states

    def set_memory(memory) -> None:
        runtime["memory"] = memory

    def set_budget(budget: int) -> None:
        runtime["budget"] = budget

    services = [
        LinkedInAcquisitionService(
            LinkedInAcquisitionDeps(
                browser=browser,
                log_path=tmp_path / "run_log.jsonl",
                ensure_browser_healthy=_ensure_browser_healthy,
            )
        ),
        LinkedInSearchMutationExecutor(
            LinkedInSearchMutationDeps(
                browser=browser,
                log_path=tmp_path / "run_log.jsonl",
                get_input_mode=lambda: "concurrent",
                get_runtime_run_id=lambda: runtime["run_id"],
                get_runtime_state=lambda: None,
                get_search_mutation_budget_used=lambda: runtime["budget"],
                set_search_mutation_budget_used=set_budget,
            )
        ),
        LinkedInSideEffectsService(
            LinkedInSideEffectsDeps(
                browser=browser,
                stats={"saved": 0},
                saved_urls=set(),
                log_path=tmp_path / "run_log.jsonl",
                get_test_mode=lambda: True,
                get_runtime_bridge=lambda: None,
                get_runtime_run_id=lambda: runtime["run_id"],
            )
        ),
        LinkedInWorkUnitService(
            LinkedInWorkUnitDeps(
                ensure_runtime_state=lambda: None,
                get_runtime_bridge=lambda: None,
                get_runtime_run_id=lambda: runtime["run_id"],
                set_runtime_run_id=set_run_id,
                get_progress=lambda: runtime["progress"],
                set_progress=set_progress,
                stats={"saved": 0, "rejected": 0},
                get_experiment_states=lambda: runtime["states"],
                set_experiment_states=set_states,
                get_search_memory=lambda: runtime["memory"],
                set_search_memory=set_memory,
                reset_restart_history=lambda: None,
            )
        ),
    ]

    assert [type(service).__name__ for service in services] == [
        "LinkedInAcquisitionService",
        "LinkedInSearchMutationExecutor",
        "LinkedInSideEffectsService",
        "LinkedInWorkUnitService",
    ]


def test_work_unit_checkpoint_keeps_commit_when_memory_refresh_fails() -> None:
    runtime_bridge = MagicMock()
    runtime_bridge.load_search_memory.side_effect = RuntimeError(
        "injected memory refresh failure"
    )
    memory = {"families": {"existing": {}}}
    set_memory = MagicMock()
    progress = Progress(
        brief_name="test",
        strings=[SearchString(id=1, name="builders", boolean="foo")],
    )
    service = LinkedInWorkUnitService(
        LinkedInWorkUnitDeps(
            ensure_runtime_state=lambda: None,
            get_runtime_bridge=lambda: runtime_bridge,
            get_runtime_run_id=lambda: 7,
            set_runtime_run_id=lambda _run_id: None,
            get_progress=lambda: progress,
            set_progress=lambda _progress: None,
            stats={"saved": 0, "rejected": 0},
            get_experiment_states=lambda: {},
            set_experiment_states=lambda _states: None,
            get_search_memory=lambda: memory,
            set_search_memory=set_memory,
            reset_restart_history=lambda: None,
        )
    )

    checkpoint = service.checkpoint_progress(progress)

    runtime_bridge.sync_progress.assert_called_once_with(
        7,
        progress,
        experiment_states={},
    )
    set_memory.assert_not_called()
    assert checkpoint is not None
    assert checkpoint.status == "synced"
