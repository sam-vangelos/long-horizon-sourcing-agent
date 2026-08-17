"""Offline integration checks for TUR-14 shadow allocator orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin import orchestrator as orchestrator_module
from linkedin.page_allocator import (
    AllocationAction,
    PageObservation,
    allocate_page,
)
from linkedin.search_intelligence import (
    LinkedInSearchVariant,
    bootstrap_experiment_state,
)
from shared.governor import OperatorStopRequested
from shared.schemas import CandidateSnippet, OpusDecision, Progress, SearchString
from tools import shadow_replay


def _make_pipeline(output_dir: Path):
    """Mirror the existing pipeline-test fixture without opening a browser."""

    with (
        patch.object(orchestrator_module, "load_brief") as mock_brief,
        patch.object(orchestrator_module, "init_judger"),
        patch.object(orchestrator_module, "LinkedInBrowser"),
    ):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.permanent_filters = {}
        brief.needs_preflight.return_value = False
        mock_brief.return_value = brief

        brief_path = output_dir / "brief.json"
        brief_path.write_text('{"id": "test"}')
        return orchestrator_module.Pipeline(
            brief_path=str(brief_path),
            output_dir=str(output_dir),
        )


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        orchestrator_module.config,
        "LINKEDIN_PAGE_ALLOCATOR_MODE",
        "shadow",
    )
    return _make_pipeline(tmp_path)


@pytest.fixture
def active_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        orchestrator_module.config,
        "LINKEDIN_PAGE_ALLOCATOR_MODE",
        "active",
    )
    monkeypatch.setattr(
        orchestrator_module.config,
        "LINKEDIN_TOTAL_PAGE_CAP",
        1,
    )
    return _make_pipeline(tmp_path)


def _root(
    root_id: int,
    *,
    status: str = "queued",
    pages_reviewed: int = 0,
    block: str = "Compound Batch 1",
) -> SearchString:
    return SearchString(
        id=root_id,
        name=f"root-{root_id}",
        boolean=f"query-{root_id}",
        status=status,
        pages_reviewed=pages_reviewed,
        block=block,
    )


def _install_states(pipeline, roots: list[SearchString]) -> None:
    for root in roots:
        state = bootstrap_experiment_state(root)
        state.active_variant.pages_reviewed = root.pages_reviewed
        pipeline._experiment_states[root.id] = state


def _snippet(root_id: int, page: int, slug: str) -> CandidateSnippet:
    return CandidateSnippet(
        name=slug,
        headline="",
        current_title="",
        current_company="",
        location="Somewhere",
        education_snippet="",
        profile_url=f"/talent/profile/{slug}",
        source_string_id=root_id,
        source_string_name=f"root-{root_id}",
        page=page,
        result_rank=1,
    )


def _decision(
    snippet: CandidateSnippet,
    disposition: str,
    *,
    tier: str = "",
) -> OpusDecision:
    return OpusDecision(
        stage="full",
        decision=disposition,
        path="DIRECT:test" if disposition != "REJECT" else "NONE",
        confidence=0.8,
        rationale="test",
        candidate_name=snippet.name,
        profile_url=snippet.profile_url,
        outreach_tier=tier,
        reject_reason="NON_FIT" if disposition == "REJECT" else "",
    )


def _open_page(
    pipeline,
    root: SearchString,
    *,
    page: int = 1,
    slots: int = 5,
    extracted: int = 5,
) -> None:
    state = pipeline._experiment_states[root.id]
    state.set_active_allocator_page_cursor(page)
    pipeline._reset_page_observation(
        slots,
        search_string=root,
        page_num=page,
    )
    pipeline._note_page_observation("extracted", extracted)


def _stage_page(
    pipeline,
    progress: Progress,
    root: SearchString,
    *,
    page: int = 1,
) -> PageObservation:
    pipeline._stage_allocator_page_checkpoint(
        progress=progress,
        search_string=root,
        experiment_state=pipeline._experiment_states[root.id],
        page_num=page,
        page_observed=pipeline._page_observation(),
    )
    observation = pipeline._pending_allocator_checkpoint["observation"]
    assert isinstance(observation, PageObservation)
    return observation


def _valid_observation(root_id: int, *, variant_id: str = "root") -> PageObservation:
    return PageObservation(
        root_string_id=root_id,
        variant_id=variant_id,
        page=1,
        slots=5,
        extracted=5,
        full_expected=1,
        full_settled=1,
        priority=1,
        standard=0,
        outreach=1,
    )


def _stage_active_opening_switch(pipeline):
    current = _root(1, status="in_progress")
    sibling = _root(2)
    third = _root(3)
    later_block = _root(4, block="Compound Batch 2")
    roots = [current, sibling, third, later_block]
    progress = Progress(
        brief_name="test",
        strings=roots,
        current_string_id=current.id,
        current_page=1,
    )
    _install_states(pipeline, roots)
    _open_page(pipeline, current)
    _stage_page(pipeline, progress, current)
    verdict = pipeline._pending_allocator_checkpoint["verdict"]
    assert verdict.action is AllocationAction.SWITCH
    assert verdict.selected_root_id == sibling.id
    assert verdict.paused_root_ids == (current.id,)
    pipeline._experiment_states[current.id].set_active_allocator_page_cursor(2)
    return progress, current, sibling, third, later_block


def test_page_local_currency_uses_exact_expected_and_settled_candidates(pipeline):
    root = _root(1, status="in_progress")
    progress = Progress(brief_name="test", strings=[root], current_string_id=1)
    _install_states(pipeline, [root])
    _open_page(pipeline, root)

    snippets = [_snippet(1, 1, f"candidate-{index}") for index in range(5)]
    for snippet in snippets:
        pipeline._note_page_full_review_expected(snippet)

    outcomes = [
        _decision(snippets[0], "SAVE", tier="PRIORITY"),
        _decision(snippets[1], "SAVE", tier="STANDARD"),
        _decision(snippets[2], "REVIEW_FLAGGED"),
        _decision(snippets[3], "REJECT"),
    ]
    for snippet, decision in zip(snippets, outcomes):
        pipeline._finish_runtime_stage_success(
            attempt_id=None,
            stage="full",
            snippet=snippet,
            decision=decision,
        )

    # Retries and work settling outside the exact open page cannot earn currency.
    pipeline._finish_runtime_stage_success(
        attempt_id=None,
        stage="full",
        snippet=snippets[0],
        decision=outcomes[0],
    )
    outside = _snippet(1, 2, "outside-page")
    pipeline._note_page_full_review_expected(outside)
    pipeline._finish_runtime_stage_success(
        attempt_id=None,
        stage="full",
        snippet=outside,
        decision=_decision(outside, "SAVE", tier="PRIORITY"),
    )

    observation = _stage_page(pipeline, progress, root)

    assert (
        observation.full_expected,
        observation.full_settled,
        observation.priority,
        observation.standard,
        observation.outreach,
    ) == (5, 4, 1, 1, 2)
    assert observation.full_settled * 5 == observation.full_expected * 4
    assert observation.valid


def test_save_without_valid_tier_invalidates_page_currency(pipeline):
    root = _root(1, status="in_progress")
    progress = Progress(brief_name="test", strings=[root], current_string_id=1)
    _install_states(pipeline, [root])
    _open_page(pipeline, root, slots=1, extracted=1)
    snippet = _snippet(1, 1, "missing-tier")
    pipeline._note_page_full_review_expected(snippet)
    pipeline._finish_runtime_stage_success(
        attempt_id=None,
        stage="full",
        snippet=snippet,
        decision=_decision(snippet, "SAVE"),
    )

    observation = _stage_page(pipeline, progress, root)

    assert (observation.priority, observation.standard, observation.outreach) == (
        0,
        0,
        1,
    )
    assert "tier_outreach_mismatch" in observation.invalid_reasons


def test_legacy_deep_bootstrap_divergence_makes_first_page_off_policy(pipeline):
    current = _root(1, status="in_progress", pages_reviewed=3)
    untouched = _root(2)
    progress = Progress(
        brief_name="test",
        strings=[current, untouched],
        current_string_id=current.id,
    )
    _install_states(pipeline, [current, untouched])

    pipeline._check_allocator_pre_spend(progress=progress, current=current)

    state = pipeline._experiment_states[current.id]
    assert state.allocator_shadow_diverged is True
    assert state.allocator_causality["reason"] == "bootstrap_mismatch"
    assert state.allocator_causality["expected_root_id"] == untouched.id

    _open_page(pipeline, current)
    observation = _stage_page(pipeline, progress, current)
    assert observation.valid and observation.off_policy
    assert not observation.teaches_policy

    state.set_active_allocator_page_cursor(2)
    pipeline._work_unit_service = MagicMock()
    pipeline._emit_allocator_event_after_sync = MagicMock()
    pipeline._checkpoint_progress(progress, search_string=current, page_num=1)

    assert state.active_variant.allocator_completed_observation_count == 1
    assert state.active_variant.allocator_valid_page_count == 0
    assert state.active_variant.allocator_observations == []
    assert state.allocator_causality["reason"] == "bootstrap_mismatch"
    payload = pipeline._emit_allocator_event_after_sync.call_args.kwargs["payload"]
    assert payload["shadow_diverged"] is True
    assert payload["divergence_after_observation"] is False
    assert payload["divergence_reason"] == "bootstrap_mismatch"

    pipeline._check_allocator_pre_spend(progress=progress, current=current)
    assert state.allocator_causality["reason"] == "bootstrap_mismatch"


def test_prior_switch_rejects_done_paused_root_even_when_selected_root_matches(
    pipeline,
):
    paused = _root(1, status="in_progress", pages_reviewed=3)
    selected = _root(2)
    progress = Progress(
        brief_name="test",
        strings=[paused, selected],
        current_string_id=paused.id,
    )
    _install_states(pipeline, [paused, selected])

    arms = pipeline._allocator_arms(progress=progress, current=paused)
    verdict = allocate_page(current_root_id=paused.id, arms=arms)
    assert verdict.action is AllocationAction.SWITCH
    assert verdict.selected_root_id == selected.id
    expectation = pipeline._allocator_expectation(
        progress=progress,
        current=paused,
        arms=arms,
        verdict=verdict,
        sequence=1,
    )
    pipeline._experiment_states[paused.id].allocator_frontier_expectation = expectation

    # The counterfactual active transition is selected-first with the paused
    # root queued at the block tail.
    paused.status = "queued"
    selected.status = "in_progress"
    progress.strings[:] = [selected, paused]
    progress.current_string_id = selected.id
    aligned, reason = pipeline._allocator_frontier_alignment(
        progress=progress,
        current=selected,
        expectation=expectation,
        require_selected=True,
    )
    assert (aligned, reason) == (True, "aligned")

    # Marking the paused root done is a lifecycle divergence even though
    # execution moved to the allocator-selected root.
    paused.status = "done"
    pipeline._check_allocator_pre_spend(progress=progress, current=selected)

    selected_state = pipeline._experiment_states[selected.id]
    assert expectation["selected_root_id"] == selected.id
    assert selected_state.allocator_shadow_diverged is True
    assert selected_state.allocator_causality["reason"] == (
        "frontier_disposition_changed"
    )


def test_same_root_rewrite_keeps_frontier_alignment_and_root_probe(pipeline):
    root = _root(1, status="in_progress")
    progress = Progress(brief_name="test", strings=[root], current_string_id=1)
    _install_states(pipeline, [root])
    state = pipeline._experiment_states[root.id]
    state.record_allocator_observation(_valid_observation(root.id))

    prior_arms = pipeline._allocator_arms(progress=progress, current=root)
    prior_verdict = allocate_page(current_root_id=root.id, arms=prior_arms)
    state.allocator_frontier_expectation = pipeline._allocator_expectation(
        progress=progress,
        current=root,
        arms=prior_arms,
        verdict=prior_verdict,
        sequence=1,
    )

    rewrite = LinkedInSearchVariant(
        variant_id="rewrite-1",
        parent_variant_id="root",
        root_string_id=root.id,
        boolean="query-1 AND focused",
        variant_kind="precision",
    )
    state.variants[rewrite.variant_id] = rewrite
    state.activate_variant(rewrite.variant_id)

    pipeline._check_allocator_pre_spend(progress=progress, current=root)
    arms = pipeline._allocator_arms(progress=progress, current=root)
    verdict = allocate_page(current_root_id=root.id, arms=arms)

    assert state.allocator_shadow_diverged is False
    assert state.allocator_causality["aligned"] is True
    assert arms[0].root_has_valid_probe is True
    assert arms[0].active_variant_id == rewrite.variant_id
    assert arms[0].active_valid_page_count == 0
    assert arms[0].observations == ()
    assert verdict.selected_root_id == root.id
    assert verdict.reason != "opening_probe"


def test_pending_page_commits_only_at_cursor_n_plus_one_and_emits_after_sync(
    pipeline,
):
    root = _root(1, status="in_progress")
    progress = Progress(brief_name="test", strings=[root], current_string_id=1)
    _install_states(pipeline, [root])
    state = pipeline._experiment_states[root.id]
    _open_page(pipeline, root)
    _stage_page(pipeline, progress, root)

    order: list[str] = []
    service = MagicMock()
    service.checkpoint_progress.side_effect = lambda *_args, **_kwargs: order.append(
        "sync"
    )
    pipeline._work_unit_service = service
    pipeline._emit_allocator_event_after_sync = MagicMock(
        side_effect=lambda **_kwargs: order.append("event")
    )

    pipeline._checkpoint_progress(progress, search_string=root, page_num=1)
    assert state.active_variant.allocator_completed_observation_count == 0
    assert pipeline._pending_allocator_checkpoint is not None
    pipeline._emit_allocator_event_after_sync.assert_not_called()

    order.clear()
    state.set_active_allocator_page_cursor(2)
    pipeline._checkpoint_progress(progress, search_string=root, page_num=1)

    assert state.active_variant.allocator_completed_observation_count == 1
    assert state.active_variant.allocator_valid_page_count == 1
    assert pipeline._pending_allocator_checkpoint is None
    assert order == ["sync", "event"]


def test_failed_service_checkpoint_restores_allocator_state_and_emits_nothing(
    pipeline,
):
    root = _root(1, status="in_progress")
    progress = Progress(brief_name="test", strings=[root], current_string_id=1)
    _install_states(pipeline, [root])
    state = pipeline._experiment_states[root.id]
    _open_page(pipeline, root)
    _stage_page(pipeline, progress, root)
    state.set_active_allocator_page_cursor(2)
    before = state.to_dict()

    service = MagicMock()
    service.checkpoint_progress.side_effect = RuntimeError("injected sync failure")
    pipeline._work_unit_service = service
    pipeline._emit_allocator_event_after_sync = MagicMock()

    with pytest.raises(RuntimeError, match="injected sync failure"):
        pipeline._checkpoint_progress(progress, search_string=root, page_num=1)

    assert state.to_dict() == before
    assert pipeline._pending_allocator_checkpoint is not None
    pipeline._emit_allocator_event_after_sync.assert_not_called()


def test_malformed_pending_shadow_state_cannot_block_canonical_checkpoint(
    pipeline,
):
    root = _root(1, status="in_progress")
    progress = Progress(brief_name="test", strings=[root], current_string_id=1)
    _install_states(pipeline, [root])
    state = pipeline._experiment_states[root.id]
    pipeline._pending_allocator_checkpoint = {
        "kind": "exhaustion",
        "root_string_id": "not-an-int",
    }
    pipeline._work_unit_service = MagicMock()
    pipeline._emit_allocator_event_after_sync = MagicMock()

    pipeline._checkpoint_progress(progress, search_string=root, page_num=1)

    pipeline._work_unit_service.checkpoint_progress.assert_called_once_with(
        progress,
        search_string=root,
        page_num=1,
    )
    assert pipeline._pending_allocator_checkpoint is None
    assert state.allocator_shadow_diverged is True
    assert state.allocator_causality["reason"] == "checkpoint_apply:ValueError"
    assert state.allocator_causality["trace_poison_reason"] == (
        "checkpoint_apply:ValueError"
    )
    pipeline._emit_allocator_event_after_sync.assert_called_once()
    emitted = pipeline._emit_allocator_event_after_sync.call_args.kwargs
    assert emitted["event_type"] == "page_allocator_shadow_poison"
    report = shadow_replay.summarize_page_allocator_replay(
        [{"event": emitted["event_type"], **emitted["payload"]}]
    )
    assert report["evaluable"] is False
    assert report["poison"]["reason"] == (
        "allocator_shadow_poison:checkpoint_apply:ValueError"
    )


def test_allocator_poison_survives_sync_failure_and_emits_once_after_retry(
    pipeline,
):
    root = _root(1, status="in_progress")
    progress = Progress(brief_name="test", strings=[root], current_string_id=1)
    _install_states(pipeline, [root])
    pipeline._pending_allocator_checkpoint = {
        "kind": "exhaustion",
        "root_string_id": root.id,
        "page": 1,
    }
    service = MagicMock()
    service.checkpoint_progress.side_effect = [
        RuntimeError("injected canonical sync failure"),
        None,
        None,
    ]
    pipeline._work_unit_service = service
    pipeline._emit_allocator_event_after_sync = MagicMock()

    with pytest.raises(RuntimeError, match="canonical sync failure"):
        pipeline._checkpoint_progress(progress, search_string=root, page_num=1)

    assert pipeline._pending_allocator_checkpoint is None
    assert pipeline._pending_allocator_poison is not None
    assert pipeline._experiment_states[root.id].allocator_shadow_diverged is True
    pipeline._emit_allocator_event_after_sync.assert_not_called()

    pipeline._checkpoint_progress(progress, search_string=root, page_num=1)
    assert pipeline._pending_allocator_poison is None
    assert pipeline._experiment_states[root.id].allocator_shadow_diverged is True
    pipeline._emit_allocator_event_after_sync.assert_called_once()
    assert (
        pipeline._emit_allocator_event_after_sync.call_args.kwargs["event_type"]
        == "page_allocator_shadow_poison"
    )

    pipeline._checkpoint_progress(progress, search_string=root, page_num=1)
    pipeline._emit_allocator_event_after_sync.assert_called_once()

    pipeline._reset_page_observation(1, search_string=root, page_num=1)
    assert pipeline._allocator_page_off_policy is True


def test_completed_page_sync_failure_rolls_back_legacy_and_shadow_state(
    pipeline,
):
    root = _root(1, status="in_progress")
    progress = Progress(brief_name="test", strings=[root], current_string_id=1)
    _install_states(pipeline, [root])
    state = pipeline._experiment_states[root.id]

    no_results = MagicMock()
    no_results.is_visible = AsyncMock(return_value=False)
    locator = MagicMock()
    locator.first = no_results
    pipeline.browser.page = MagicMock(
        url="https://www.linkedin.com/talent/hire/123/discover/recruiterSearch"
    )
    pipeline.browser.page.locator.return_value = locator
    pipeline.browser.enter_search_string = AsyncMock()
    pipeline.browser.get_results_count_text = AsyncMock(return_value="100")
    pipeline.browser.get_results_count = AsyncMock(return_value=100)
    pipeline._ensure_browser_healthy = AsyncMock()

    async def review_page(**_kwargs):
        pipeline._reset_page_observation(
            1,
            search_string=root,
            page_num=1,
        )
        pipeline._note_page_observation("extracted")
        pipeline._latest_page_preview_snippets = []

    pipeline._review_page_sequentially = AsyncMock(side_effect=review_page)
    pipeline._evaluate_variant_lifecycle = MagicMock(return_value=None)
    pipeline._assess_string_state = AsyncMock(
        return_value={"decision": "stop", "rationale": "done", "page": 1}
    )
    pipeline._maybe_discover_fallback_candidates = MagicMock()
    pipeline._emit_allocator_event_after_sync = MagicMock()

    durable_state: dict = {}
    observed_checkpoints: list[tuple[int, str, int, int]] = []
    service = MagicMock()

    def checkpoint(*_args, **_kwargs):
        observed_checkpoints.append(
            (
                state.active_allocator_page_cursor(),
                root.status,
                root.pages_reviewed,
                state.active_variant.allocator_completed_observation_count,
            )
        )
        if len(observed_checkpoints) == 1:
            durable_state.update(state.to_dict())
            return
        raise RuntimeError("injected completed-page sync failure")

    service.checkpoint_progress.side_effect = checkpoint
    pipeline._work_unit_service = service

    with pytest.raises(RuntimeError, match="completed-page sync failure"):
        asyncio.run(pipeline._process_string(root, progress))

    assert observed_checkpoints == [
        (1, "in_progress", 0, 0),
        (2, "done", 0, 1),
    ]
    assert state.to_dict() == durable_state
    assert (root.status, root.pages_reviewed, root.notes) == (
        "in_progress",
        0,
        "",
    )
    assert pipeline._pending_allocator_checkpoint is None
    pipeline._emit_allocator_event_after_sync.assert_not_called()


@pytest.mark.parametrize("with_sibling", [True, False])
def test_physical_exhaustion_checkpoints_switch_or_finish_after_sync(
    pipeline,
    with_sibling: bool,
):
    current = _root(1, status="in_progress")
    sibling = _root(2)
    roots = [current, sibling] if with_sibling else [current]
    progress = Progress(
        brief_name="test",
        strings=roots,
        current_string_id=current.id,
    )
    _install_states(pipeline, roots)

    pipeline._stage_allocator_exhaustion(
        progress=progress,
        search_string=current,
        page_num=3,
    )
    pending = pipeline._pending_allocator_checkpoint
    assert pending is not None
    verdict = pending["verdict"]
    expectation = pending["expectation"]
    if with_sibling:
        assert verdict.action is AllocationAction.SWITCH
        assert verdict.selected_root_id == sibling.id
        assert verdict.paused_root_ids == ()
        assert expectation["segment"]["root_ids"] == [sibling.id, current.id]
        assert expectation["expected_status_by_root"] == {
            str(current.id): "terminal",
            str(sibling.id): "in_progress",
        }
    else:
        assert verdict.action is AllocationAction.FINISH
        assert verdict.selected_root_id is None
        assert expectation["segment"]["root_ids"] == [current.id]
        assert expectation["expected_status_by_root"] == {
            str(current.id): "terminal"
        }

    current.status = "done"
    order: list[str] = []
    pipeline._work_unit_service = MagicMock()
    pipeline._work_unit_service.checkpoint_progress.side_effect = (
        lambda *_args, **_kwargs: order.append("sync")
    )
    pipeline._emit_allocator_event_after_sync = MagicMock(
        side_effect=lambda **_kwargs: order.append("event")
    )

    pipeline._checkpoint_progress(progress, search_string=current)

    assert order == ["sync", "event"]
    emitted = pipeline._emit_allocator_event_after_sync.call_args.kwargs
    assert emitted["event_type"] == "page_allocator_shadow_exhaustion"
    assert emitted["payload"]["root_string_id"] == current.id
    assert emitted["payload"]["page"] == 3
    assert pipeline._experiment_states[current.id].allocator_last_verdict[
        "action"
    ] == ("switch" if with_sibling else "finish")
    if with_sibling:
        assert [item.id for item in progress.strings] == [current.id, sibling.id]
        assert emitted["payload"]["shadow_diverged"] is True
        assert emitted["payload"]["divergence_reason"] == (
            "segment_signature_changed"
        )
    else:
        assert emitted["payload"]["shadow_diverged"] is False
        assert emitted["payload"]["divergence_reason"] == ""


def test_floor_expectation_requires_every_root_terminal(pipeline):
    current = _root(1, status="in_progress")
    sibling = _root(2)
    progress = Progress(
        brief_name="test",
        strings=[current, sibling],
        current_string_id=current.id,
    )
    _install_states(pipeline, [current, sibling])
    for root in (current, sibling):
        state = pipeline._experiment_states[root.id]
        for page in (1, 2):
            state.record_allocator_observation(
                PageObservation(
                    root_string_id=root.id,
                    variant_id="root",
                    page=page,
                    slots=10,
                    extracted=10,
                    full_expected=0,
                    full_settled=0,
                    priority=0,
                    standard=0,
                    outreach=0,
                )
            )

    arms = pipeline._allocator_arms(progress=progress, current=current)
    verdict = allocate_page(current_root_id=current.id, arms=arms)
    assert verdict.action is AllocationAction.FLOOR
    expectation = pipeline._allocator_expectation(
        progress=progress,
        current=current,
        arms=arms,
        verdict=verdict,
        sequence=1,
    )
    assert expectation["expected_status_by_root"] == {
        str(current.id): "terminal",
        str(sibling.id): "terminal",
    }

    current.status = "done"
    sibling.status = "done"
    assert pipeline._allocator_frontier_alignment(
        progress=progress,
        current=current,
        expectation=expectation,
        require_selected=False,
    ) == (True, "aligned")

    sibling.status = "queued"
    assert pipeline._allocator_frontier_alignment(
        progress=progress,
        current=current,
        expectation=expectation,
        require_selected=False,
    ) == (False, "frontier_disposition_changed")


def test_shadow_switch_verdict_does_not_mutate_queue_authority(pipeline):
    current = _root(1, status="in_progress")
    sibling = _root(2)
    later_block = _root(3, block="Compound Batch 2")
    progress = Progress(
        brief_name="test",
        strings=[current, sibling, later_block],
        current_string_id=current.id,
        current_page=1,
    )
    _install_states(pipeline, [current, sibling, later_block])
    _open_page(pipeline, current)
    _stage_page(pipeline, progress, current)
    pending_verdict = pipeline._pending_allocator_checkpoint["verdict"]
    assert pending_verdict.action is AllocationAction.SWITCH
    assert pending_verdict.selected_root_id == sibling.id
    queue_before = progress.to_dict()

    pipeline._experiment_states[current.id].set_active_allocator_page_cursor(2)
    pipeline._work_unit_service = MagicMock()
    pipeline._emit_allocator_event_after_sync = MagicMock()
    pipeline._checkpoint_progress(progress, search_string=current, page_num=1)

    assert progress.to_dict() == queue_before
    assert pipeline._experiment_states[current.id].allocator_last_verdict[
        "action"
    ] == "switch"


def test_active_phase_one_then_actuation_rotates_only_the_current_block(
    active_pipeline,
):
    progress, current, sibling, third, later_block = (
        _stage_active_opening_switch(active_pipeline)
    )
    queue_before = progress.to_dict()
    service = MagicMock()
    active_pipeline._work_unit_service = service
    active_pipeline._emit_allocator_event_after_sync = MagicMock()

    active_pipeline._checkpoint_progress(
        progress,
        search_string=current,
        page_num=1,
    )

    owner_state = active_pipeline._experiment_states[current.id]
    assert progress.to_dict() == queue_before
    assert owner_state.allocator_last_verdict["mode"] == "active"
    assert owner_state.allocator_last_verdict["actuation_required"] is True
    assert owner_state.allocator_last_verdict["actuated"] is False
    phase_one_event = (
        active_pipeline._emit_allocator_event_after_sync.call_args.kwargs
    )
    assert phase_one_event["event_type"] == "page_allocator_active_checkpoint"
    assert phase_one_event["payload"]["actuated"] is False

    # Simulate a process crash/reload at the transaction boundary: only
    # serialized canonical state survives into phase two.
    progress = Progress.from_dict(progress.to_dict())
    active_pipeline._experiment_states = {
        root_id: type(state).from_dict(state.to_dict())
        for root_id, state in active_pipeline._experiment_states.items()
    }
    owner_state = active_pipeline._experiment_states[current.id]

    verdict = active_pipeline._resume_active_allocator_actuation(progress)

    assert verdict is not None
    assert verdict.action is AllocationAction.SWITCH
    assert [item.id for item in progress.strings] == [
        sibling.id,
        third.id,
        current.id,
        later_block.id,
    ]
    assert [item.status for item in progress.strings] == [
        "in_progress",
        "queued",
        "queued",
        "queued",
    ]
    assert progress.current_string_id == sibling.id
    assert progress.current_page == 1
    assert owner_state.allocator_last_verdict["actuated"] is True
    assert service.checkpoint_progress.call_count == 2
    service.checkpoint_progress.assert_called_with(
        progress,
        search_string=None,
    )
    actuation_event = (
        active_pipeline._emit_allocator_event_after_sync.call_args.kwargs
    )
    assert actuation_event["event_type"] == "page_allocator_active_actuation"
    assert actuation_event["payload"]["segment_root_ids"] == [
        sibling.id,
        third.id,
        current.id,
    ]


def test_active_actuation_sync_failure_restores_exact_in_memory_preimage(
    active_pipeline,
):
    progress, current, *_rest = _stage_active_opening_switch(active_pipeline)
    service = MagicMock()
    active_pipeline._work_unit_service = service
    active_pipeline._emit_allocator_event_after_sync = MagicMock()
    active_pipeline._checkpoint_progress(
        progress,
        search_string=current,
        page_num=1,
    )
    progress_before = progress.to_dict()
    state_refs = dict(active_pipeline._experiment_states)
    states_before = {
        root_id: state.to_dict() for root_id, state in state_refs.items()
    }
    service.reset_mock()
    service.checkpoint_progress.side_effect = RuntimeError(
        "injected actuation sync failure"
    )
    active_pipeline._emit_allocator_event_after_sync.reset_mock()

    with pytest.raises(RuntimeError, match="actuation sync failure"):
        active_pipeline._resume_active_allocator_actuation(progress)

    assert progress.to_dict() == progress_before
    for root_id, state in state_refs.items():
        assert active_pipeline._experiment_states[root_id] is state
        assert state.to_dict() == states_before[root_id]
    assert (
        active_pipeline._experiment_states[current.id]
        .allocator_last_verdict["actuated"]
        is False
    )
    service.checkpoint_progress.assert_called_once_with(
        progress,
        search_string=None,
    )
    active_pipeline._emit_allocator_event_after_sync.assert_not_called()


def test_active_floor_terminalizes_block_and_persists_ready_adaptation(
    active_pipeline,
):
    current = _root(1, status="in_progress")
    sibling = _root(2)
    progress = Progress(
        brief_name="test",
        strings=[current, sibling],
        current_string_id=current.id,
        current_page=2,
    )
    _install_states(active_pipeline, [current, sibling])
    for root in (current, sibling):
        state = active_pipeline._experiment_states[root.id]
        for page in (1, 2):
            state.record_allocator_observation(
                PageObservation(
                    root_string_id=root.id,
                    variant_id="root",
                    page=page,
                    slots=10,
                    extracted=10,
                    full_expected=0,
                    full_settled=0,
                    priority=0,
                    standard=0,
                    outreach=0,
                )
            )
    arms = active_pipeline._allocator_arms(
        progress=progress,
        current=current,
    )
    verdict = allocate_page(current_root_id=current.id, arms=arms)
    assert verdict.action is AllocationAction.FLOOR
    active_pipeline._stage_active_allocator_dispatch(
        progress=progress,
        current=current,
        arms=arms,
        verdict=verdict,
    )
    service = MagicMock()

    def set_pending(live_progress, block_name, strings, *, ready):
        live_progress.pending_block_name = block_name
        live_progress.pending_block_string_ids = [item.id for item in strings]
        live_progress.pending_block_ready = ready

    service.set_pending_block_adaptation.side_effect = set_pending
    active_pipeline._work_unit_service = service
    active_pipeline._checkpoint_progress(progress)

    actuated = active_pipeline._resume_active_allocator_actuation(progress)
    assert actuated is not None
    assert actuated.action is AllocationAction.FLOOR
    assert actuated.floored_root_ids == verdict.floored_root_ids
    assert [root.status for root in progress.strings] == ["done", "done"]
    assert progress.current_string_id is None
    assert progress.current_page == 0
    assert progress.pending_block_string_ids == [current.id, sibling.id]
    assert progress.pending_block_ready is True


def test_active_total_page_cap_stops_before_actuation(active_pipeline):
    progress, current, *_rest = _stage_active_opening_switch(active_pipeline)
    service = MagicMock()
    active_pipeline._work_unit_service = service
    active_pipeline._emit_allocator_event_after_sync = MagicMock()
    active_pipeline._checkpoint_progress(
        progress,
        search_string=current,
        page_num=1,
    )
    queue_before = progress.to_dict()
    resume_actuation = MagicMock(
        wraps=active_pipeline._resume_active_allocator_actuation
    )
    active_pipeline._resume_active_allocator_actuation = resume_actuation

    with pytest.raises(OperatorStopRequested, match="total_page_cap_reached"):
        active_pipeline._finish_completed_page_allocator_boundary(
            progress=progress,
            search_string=current,
            page_num=1,
        )

    assert progress.to_dict() == queue_before
    assert (
        active_pipeline._experiment_states[current.id]
        .allocator_last_verdict["actuated"]
        is False
    )
    resume_actuation.assert_not_called()
    assert active_pipeline._emit_allocator_event_after_sync.call_count == 1
    assert (
        active_pipeline._emit_allocator_event_after_sync.call_args.kwargs[
            "event_type"
        ]
        == "page_allocator_active_checkpoint"
    )


def test_off_mode_allocator_staging_is_a_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        orchestrator_module.config,
        "LINKEDIN_PAGE_ALLOCATOR_MODE",
        "off",
    )
    monkeypatch.setattr(
        orchestrator_module.config,
        "LINKEDIN_TOTAL_PAGE_CAP",
        1,
    )
    off_pipeline = _make_pipeline(tmp_path)
    current = _root(1, status="in_progress")
    sibling = _root(2)
    progress = Progress(
        brief_name="test",
        strings=[current, sibling],
        current_string_id=current.id,
        current_page=1,
    )
    _install_states(off_pipeline, [current, sibling])
    _open_page(off_pipeline, current)
    progress_before = progress.to_dict()
    states_before = {
        root_id: state.to_dict()
        for root_id, state in off_pipeline._experiment_states.items()
    }

    off_pipeline._stage_allocator_page_checkpoint(
        progress=progress,
        search_string=current,
        experiment_state=off_pipeline._experiment_states[current.id],
        page_num=1,
        page_observed=off_pipeline._page_observation(),
    )
    off_pipeline._stage_allocator_exhaustion(
        progress=progress,
        search_string=current,
        page_num=1,
    )
    off_pipeline._work_unit_service = MagicMock()
    off_pipeline._emit_allocator_event_after_sync = MagicMock()
    off_pipeline._checkpoint_progress(
        progress,
        search_string=current,
        page_num=1,
    )

    assert off_pipeline._allocator_page_identity == (1, "root", 1)
    assert off_pipeline._pending_allocator_checkpoint is None
    assert progress.to_dict() == progress_before
    assert {
        root_id: state.to_dict()
        for root_id, state in off_pipeline._experiment_states.items()
    } == states_before
    off_pipeline._emit_allocator_event_after_sync.assert_not_called()
