"""Historical full work cannot borrow the active page's observation objects."""
from tests.evaluation_test_adapters import patch_evaluation_delay
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from linkedin.page_review import PageReviewContext
from linkedin.search_intelligence import bootstrap_experiment_state
from shared import config
from tests.test_full_eval_pipelining import (
    _make_pipeline, _snippet, _decision, _wire_common_pipeline,
    _wire_real_runtime, _full_attempt_rows,
)
from tests.test_resume_recovery_containment import _seed_facial_yes, _track_pending


@pytest.mark.parametrize("window", [1, 2])
@pytest.mark.parametrize("interrupt", [False, True])
def test_historical_review_keeps_live_context_and_committed_cursor(tmp_path, monkeypatch, window, interrupt):
    monkeypatch.setattr(config, "FULL_EVAL_PIPELINE_ENABLED", window == 2)
    monkeypatch.setattr(config, "LINKEDIN_PAGE_ALLOCATOR_MODE", "shadow")
    monkeypatch.setattr(config, "LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False)
    monkeypatch.setattr(config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", False)
    p = _make_pipeline(str(tmp_path))
    _wire_common_pipeline(p)
    historical = _snippet("Historical", 1)
    owner, progress = _wire_real_runtime(p, output_dir=str(tmp_path), snippets=[historical])
    experiment = bootstrap_experiment_state(owner)
    experiment.set_active_allocator_page_cursor(3)
    p.state._experiment_states[owner.id] = experiment
    progress.current_string_id, progress.current_page = owner.id, 3
    p.state._progress = progress
    _seed_facial_yes(p, owner, historical)
    _track_pending(p, historical, owner)
    # Use the production checkpoint implementation and SQLite bridge.
    del p.state._allocator_service._checkpoint_progress
    p.state._allocator_service._checkpoint_progress(progress)
    p.state.pages._reset_page_observation(25, search_string=owner, page_num=3)
    live = p.state.page.review
    live._latest_page_observed["extracted"] = 11
    live._allocator_page_expected_keys.add("live-expected")
    live._allocator_page_settled_keys.add("live-settled")
    live._allocator_page_off_policy = True
    observed, expected, settled = (live._latest_page_observed,
                                   live._allocator_page_expected_keys,
                                   live._allocator_page_settled_keys)
    before = (dict(observed), set(expected), set(settled),
              live._allocator_page_identity, live._allocator_page_off_policy)
    p.state.context.browser.find_result_slot_by_profile_url = AsyncMock(return_value=0)
    contexts = []
    original = p.state.reviewer._process_resumed_pending_full_evaluations

    async def capture_context(**kwargs):
        context = kwargs["page_context"]
        assert isinstance(context, PageReviewContext) and context is not live
        contexts.append(context)
        result = await original(**kwargs)
        if interrupt:
            raise RuntimeError("after historical settlement")
        return result

    p.state.reviewer._process_resumed_pending_full_evaluations = AsyncMock(side_effect=capture_context)
    verdict = _decision(historical, "REJECT")
    with patch("linkedin.candidate_review.full_judge", return_value=verdict) as judge, \
         patch_evaluation_delay(return_value=0.0):
        call = p.state.driver._recover_owner_pending_full_evaluations(
            progress=progress, search_string=owner, first_incomplete_page=3,
            string_stats=p.state.candidates._fresh_string_stats(),
        )
        if interrupt:
            with pytest.raises(RuntimeError, match="after historical settlement"):
                asyncio.run(call)
        else:
            assert asyncio.run(call) == 1
    judge.assert_called_once()
    assert len(contexts) == 1
    assert contexts[0].observation()["full_expected"] == 1
    assert contexts[0].observation()["full_settled"] == 1
    assert live._latest_page_observed is observed
    assert live._allocator_page_expected_keys is expected
    assert live._allocator_page_settled_keys is settled
    assert (observed, expected, settled, live._allocator_page_identity,
            live._allocator_page_off_policy) == before
    assert [row["status"] for row in _full_attempt_rows(p)] == ["succeeded"]
    assert p.state.candidates._resume_pending_full_decisions == {}
    durable = p.state.context._runtime_bridge.load_progress(p.state.context._runtime_run_id)
    assert (durable.current_string_id, durable.current_page) == (owner.id, 3)
    states = p.state.context._runtime_bridge.load_experiment_states(p.state.context._runtime_run_id, progress=durable)
    assert states[owner.id].active_allocator_page_cursor() == 3
