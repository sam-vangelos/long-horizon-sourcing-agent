"""Unit tests for the extracted LinkedIn AllocatorStateService cluster."""

from __future__ import annotations

from dataclasses import replace

from shared.schemas import CandidateSnippet, Progress, SearchString

from linkedin.allocator_state import AllocatorStateDeps, AllocatorStateService
from linkedin.page_allocator import (
    AllocationAction,
    AllocationVerdict,
    AllocatorArm,
)
from linkedin.search_intelligence import bootstrap_experiment_state


def _make_service(
    experiment_states: dict | None = None,
) -> AllocatorStateService:
    states = experiment_states if experiment_states is not None else {}
    deps = AllocatorStateDeps(
        get_experiment_states=lambda: states,
        get_allocator_page_identity=lambda: None,
        get_pending_allocator_checkpoint=lambda: None,
    )
    return AllocatorStateService(deps)


def test_allocator_terminal_status_and_expected_statuses():
    """Terminal classification and expected-status projection match today's values."""
    service = _make_service()
    assert service._allocator_terminal_status("done") is True
    assert service._allocator_terminal_status("skipped") is True
    assert service._allocator_terminal_status("error") is True
    assert service._allocator_terminal_status("in_progress") is False

    strings = [
        SearchString(
            id=1,
            name="s1",
            boolean="a",
            block="block1",
            status="in_progress",
        ),
        SearchString(
            id=2,
            name="s2",
            boolean="b",
            block="block1",
            status="queued",
        ),
    ]
    current = strings[0]
    progress = Progress(brief_name="test", strings=strings, current_string_id=1)
    arms = [
        AllocatorArm(
            root_string_id=1,
            block="block1\x1f0:2\x1f1,2",
            queue_priority=0,
            active_variant_id="v1",
            terminal=False,
        ),
        AllocatorArm(
            root_string_id=2,
            block="block1\x1f0:2\x1f1,2",
            queue_priority=1,
            active_variant_id="v2",
            terminal=False,
        ),
    ]
    verdict = AllocationVerdict(
        action=AllocationAction.CONTINUE,
        current_root_id=1,
        selected_root_id=1,
        reason="",
    )

    statuses = service._allocator_expected_statuses(
        progress=progress,
        current=current,
        arms=arms,
        verdict=verdict,
    )
    assert statuses == {"1": "in_progress", "2": "queued"}


def test_allocator_verdict_from_payload_and_requires_actuation():
    """Verdict rehydration and actuation gating match today's control semantics."""
    service = _make_service()
    payload = {
        "action": "continue",
        "current_root_id": 1,
        "selected_root_id": 1,
        "paused_root_ids": [],
        "floored_root_ids": [],
        "ranked_root_ids": [1],
        "reason": "keep going",
    }
    verdict = service._allocator_verdict_from_payload(payload)
    assert verdict.action is AllocationAction.CONTINUE
    assert verdict.current_root_id == 1
    assert verdict.selected_root_id == 1
    assert verdict.reason == "keep going"
    assert service._allocator_verdict_requires_actuation(verdict) is False

    switch_verdict = AllocationVerdict(
        action=AllocationAction.SWITCH,
        current_root_id=1,
        selected_root_id=2,
        reason="switch",
    )
    assert service._allocator_verdict_requires_actuation(switch_verdict) is True


def test_allocator_methods_are_pure_with_unchanged_inputs():
    """Repeated calls with unchanged inputs return equal results and mutate nothing."""
    holder: dict = {"states": {}}
    service = AllocatorStateService(
        AllocatorStateDeps(
            get_experiment_states=lambda: holder["states"],
            get_allocator_page_identity=lambda: None,
            get_pending_allocator_checkpoint=lambda: None,
        )
    )
    strings = [
        SearchString(
            id=1,
            name="s1",
            boolean="a",
            block="block1",
            status="done",
        ),
    ]
    progress = Progress(brief_name="test", strings=strings)
    arms = [
        AllocatorArm(
            root_string_id=1,
            block="block1\x1f0:1\x1f1",
            queue_priority=0,
            active_variant_id="v1",
            terminal=True,
        ),
    ]
    verdict = AllocationVerdict(
        action=AllocationAction.FINISH,
        current_root_id=1,
        selected_root_id=None,
        reason="",
    )

    first_statuses = service._allocator_expected_statuses(
        progress=progress,
        current=strings[0],
        arms=arms,
        verdict=verdict,
    )
    second_statuses = service._allocator_expected_statuses(
        progress=progress,
        current=strings[0],
        arms=arms,
        verdict=verdict,
    )
    assert first_statuses == second_statuses == {"1": "terminal"}

    payload = {
        "action": "finish",
        "current_root_id": 1,
        "selected_root_id": None,
        "paused_root_ids": [],
        "floored_root_ids": [],
        "ranked_root_ids": [1],
        "reason": "",
    }
    first_verdict = service._allocator_verdict_from_payload(payload)
    second_verdict = service._allocator_verdict_from_payload(payload)
    assert first_verdict == second_verdict
    assert holder["states"] == {}


def test_service_reads_experiment_states_live_not_snapshotted():
    """get_experiment_states must read live pipeline state, not a snapshot at construction."""
    holder: dict = {"states": {}}
    service = AllocatorStateService(
        AllocatorStateDeps(
            get_experiment_states=lambda: holder["states"],
            get_allocator_page_identity=lambda: None,
            get_pending_allocator_checkpoint=lambda: None,
        )
    )
    assert service._allocator_run_diverged() is False

    state = bootstrap_experiment_state(
        SearchString(id=1, name="s1", boolean="a"),
    )
    state.allocator_shadow_diverged = True
    holder["states"] = {1: state}

    assert service._allocator_run_diverged() is True

    deps = replace(
        service.deps,
        get_experiment_states=lambda: holder["states"],
    )
    rebound_service = AllocatorStateService(deps)
    assert rebound_service._allocator_run_diverged() is True


def test_service_reads_page_identity_live_not_snapshotted():
    """get_allocator_page_identity must read live pipeline state, not a snapshot."""
    holder: dict = {"identity": (1, "v1", 1)}
    service = AllocatorStateService(
        AllocatorStateDeps(
            get_experiment_states=lambda: {},
            get_allocator_page_identity=lambda: holder["identity"],
            get_pending_allocator_checkpoint=lambda: None,
        )
    )
    snippet = CandidateSnippet(
        name="n",
        headline="h",
        current_title="t",
        current_company="c",
        location="l",
        education_snippet="e",
        profile_url="u",
        source_string_id=1,
        source_string_name="s",
        page=1,
        result_rank=1,
    )
    assert service._allocator_page_matches(snippet) is True

    holder["identity"] = (2, "v2", 1)
    assert service._allocator_page_matches(snippet) is False
