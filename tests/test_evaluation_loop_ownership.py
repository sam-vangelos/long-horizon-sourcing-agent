"""The caller sees a facade; execution owns one declared state holder."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

from linkedin.evaluation_loop import EvaluationLoop
from linkedin.evaluation_state import EvaluationRunState
from linkedin.orchestrator import Pipeline
import pytest


@pytest.mark.structural
def test_evaluation_implementation_has_left_pipeline():
    implementations = {
        "run_full", "_process_string_impl", "_review_page_sequentially",
        "_review_page_batch", "_evaluate_snippet", "_full_evaluate",
        "_process_pipelined_full_evaluations", "_open_and_extract",
        "_complete_full_evaluation", "_run_block_adaptation",
    }
    assert implementations.intersection(vars(Pipeline)) == {"run_full"}
    assert not issubclass(Pipeline, EvaluationLoop)
    assert "__getattr__" not in vars(Pipeline)
    assert "__getattr__" not in vars(EvaluationLoop)


def test_day_cycle_references_share_the_loop_state_and_delegate():
    state = EvaluationRunState()
    loop = EvaluationLoop(state)
    facade = Pipeline.__new__(Pipeline)
    facade._loop = loop
    state.context.stats = {"saved": 3}
    state.context.browser = object()
    assert loop.state is state
    assert facade.stats is state.context.stats
    assert facade.browser is state.context.browser
    for name in ("_operator_stop_event", "_session_expired", "_pause_requested", "_resume_event"):
        assert not hasattr(facade, name)
        event = asyncio.Event()
        setattr(facade, name, event)
        assert getattr(state.context, name) is event
    loop.run_full = AsyncMock()
    asyncio.run(facade.run_full(resume=True, restart_string_ids=[7]))
    loop.run_full.assert_awaited_once_with(True, None, [7])


def test_execution_owners_have_scoped_state_and_concrete_dependencies():
    from dataclasses import fields
    from linkedin.runtime_attempts import RuntimeAttemptDeps

    state = EvaluationRunState()
    assert state.driver.reviewer is state.reviewer
    assert state.driver.pages is state.pages
    assert state.driver.surface is state.surface
    assert state.reviewer.ledger is state.candidates
    assert state.pages.reviewer is state.reviewer
    assert state.surface.allocator is state._allocator_service
    assert state._allocator_service.context is state.context
    assert state._allocator_service.candidates is state.candidates
    assert state._allocator_service.work_units is None
    assert state._allocator_service.events is state.reviewer.attempts
    assert state.surface.attempts is state.reviewer.attempts
    assert state.planner.events is state.reviewer.attempts
    for owner in (state.driver, state.pages, state.reviewer, state.surface, state.planner):
        assert owner.context is state.context
        assert not hasattr(owner, "state")
    assert not hasattr(state.context, "_experiment_states")
    assert not hasattr(state.context, "_progress")
    assert {f.name for f in fields(RuntimeAttemptDeps)} == {
        "get_runtime_bridge", "get_runtime_run_id", "get_runtime_state",
    }
    for name in ("_process_string_impl", "_review_page_batch", "_full_evaluate",
                 "_run_full_queue", "_run_block_adaptation", "_checkpoint_progress_impl"):
        assert name not in vars(EvaluationLoop)


def test_allocator_events_read_late_runtime_and_log_bindings(monkeypatch, tmp_path):
    from shared.schemas import SearchString

    state = EvaluationRunState()
    allocator = state._allocator_service
    # Construction must not access unbound runtime fields or create day controls.
    assert not hasattr(state.context, "_runtime_state")
    assert not hasattr(state.context, "_operator_stop_event")
    owner = SearchString(id=1, name="owner", boolean="one")
    log_event = MagicMock()
    monkeypatch.setattr("linkedin.allocator_state.log_event", log_event)
    for run_id in (1, 2):
        store = MagicMock()
        store.get_work_unit_id.return_value = 10 + run_id
        state.context._runtime_state = store
        state.context._runtime_run_id = run_id
        state.context.log_path = tmp_path / f"run-{run_id}.jsonl"
        allocator._emit_allocator_event_after_sync(
            event_type="page_allocator_shadow_checkpoint",
            search_string=owner,
            payload={"checkpoint_index": run_id},
        )
        store.record_event.assert_called_once_with(
            run_id=run_id,
            work_unit_id=10 + run_id,
            event_type="page_allocator_shadow_checkpoint",
            payload={"checkpoint_index": run_id},
        )
        log_event.assert_called_with(
            state.context.log_path, "page_allocator_shadow_checkpoint",
            checkpoint_index=run_id,
        )
