"""Unit tests for the extracted LinkedIn RuntimeAttemptService cluster."""

from __future__ import annotations

from unittest.mock import MagicMock

from shared.schemas import CandidateSnippet, SearchString

from linkedin.runtime_attempts import RuntimeAttemptDeps, RuntimeAttemptService


def _make_snippet(**kwargs) -> CandidateSnippet:
    defaults = {
        "name": "Test Person",
        "headline": "",
        "current_title": "",
        "current_company": "",
        "location": "Somewhere",
        "education_snippet": "",
        "profile_url": "/talent/profile/test123",
        "source_string_id": 1,
        "source_string_name": "test",
        "page": 1,
        "result_rank": 1,
    }
    defaults.update(kwargs)
    return CandidateSnippet(**defaults)


def _make_service(
    *,
    runtime_run_id: int | None = None,
    runtime_bridge: MagicMock | None = None,
    runtime_state: MagicMock | None = None,
) -> RuntimeAttemptService:
    bridge = runtime_bridge if runtime_bridge is not None else MagicMock()
    state = runtime_state if runtime_state is not None else MagicMock()
    deps = RuntimeAttemptDeps(
        get_runtime_bridge=lambda: bridge,
        get_runtime_run_id=lambda: runtime_run_id,
        get_runtime_state=lambda: state,
        get_in_flight_urls=lambda: set(),
        get_resume_pending_full_decisions=lambda: {},
        get_resume_pending_full_snippets=lambda: {},
        get_resume_pending_full_owner_ids=lambda: {},
        funnel_candidate_key=lambda snippet: snippet.profile_url,
        note_page_full_review_settled=lambda **kwargs: None,
        record_outreach_tier_outcome=lambda **kwargs: None,
        variant_id_for_search_string=lambda search_string: "",
    )
    return RuntimeAttemptService(deps)


def test_record_runtime_event_forwards_to_bridge():
    """Runtime events reach the canonical store with the expected type and payload."""
    bridge_holder: dict = {"bridge": MagicMock()}
    state = MagicMock()
    state.get_work_unit_id.return_value = 99
    holder: dict = {"run_id": 42}

    service = RuntimeAttemptService(
        RuntimeAttemptDeps(
            get_runtime_bridge=lambda: bridge_holder["bridge"],
            get_runtime_run_id=lambda: holder["run_id"],
            get_runtime_state=lambda: state,
            get_in_flight_urls=lambda: set(),
            get_resume_pending_full_decisions=lambda: {},
            get_resume_pending_full_snippets=lambda: {},
            get_resume_pending_full_owner_ids=lambda: {},
            funnel_candidate_key=lambda snippet: snippet.profile_url,
            note_page_full_review_settled=lambda **kwargs: None,
            record_outreach_tier_outcome=lambda **kwargs: None,
            variant_id_for_search_string=lambda search_string: "",
            )
    )
    search_string = SearchString(
        id=7,
        name="test",
        boolean="foo",
        status="in_progress",
    )
    payload = {"reason": "probe"}

    service._record_runtime_event(
        search_string=search_string,
        event_type="test_event",
        payload=payload,
    )

    state.get_work_unit_id.assert_called_once_with(
        42,
        kind="linkedin_string",
        source_unit_id="7",
    )
    state.record_event.assert_called_once_with(
        run_id=42,
        work_unit_id=99,
        event_type="test_event",
        payload=payload,
    )


def test_start_runtime_stage_attempt_guard_and_success():
    """Without a runtime run id the service returns None; with one it returns the attempt id."""
    bridge = MagicMock()
    bridge.start_stage_attempt.return_value = 101
    search_string = SearchString(
        id=1,
        name="test",
        boolean="foo",
        status="in_progress",
    )
    snippet = _make_snippet()

    no_run_service = _make_service(runtime_run_id=None, runtime_bridge=bridge)
    assert (
        no_run_service._start_runtime_stage_attempt(
            search_string=search_string,
            snippet=snippet,
            stage="facial",
        )
        is None
    )
    bridge.start_stage_attempt.assert_not_called()

    with_run_service = _make_service(runtime_run_id=55, runtime_bridge=bridge)
    attempt_id = with_run_service._start_runtime_stage_attempt(
        search_string=search_string,
        snippet=snippet,
        stage="facial",
        payload={"lane": "a"},
    )
    assert attempt_id == 101
    bridge.start_stage_attempt.assert_called_once_with(
        run_id=55,
        search_string=search_string,
        snippet=snippet,
        stage="facial",
        payload={"lane": "a"},
    )


def test_service_reads_runtime_state_live_not_snapshotted():
    """get_runtime_state must read live pipeline state, not a value snapshotted at construction.

    The staleness mode this locks is REBINDING, not in-place mutation. In
    production ``__init__`` always sets ``self._runtime_state`` and the lazy
    re-init in ``_ensure_runtime_state`` is ``is None``-guarded, but test
    fixtures rebind ``pipeline._runtime_state`` (e.g. to ``None``) and the
    accessor must see that rebind; a snapshot field would keep pointing at the
    object captured at construction.

    Note: in-place mutation of one mock proves nothing — a snapshotted VALUE
    field holds that same object, so mutation is visible under both designs.
    """
    first_state = MagicMock(name="first_state")
    second_state = MagicMock(name="second_state")
    holder: dict = {"state": first_state}

    service = RuntimeAttemptService(
        RuntimeAttemptDeps(
            get_runtime_bridge=lambda: MagicMock(),
            get_runtime_run_id=lambda: 1,
            get_runtime_state=lambda: holder["state"],
            get_in_flight_urls=lambda: set(),
            get_resume_pending_full_decisions=lambda: {},
            get_resume_pending_full_snippets=lambda: {},
            get_resume_pending_full_owner_ids=lambda: {},
            funnel_candidate_key=lambda snippet: snippet.profile_url,
            note_page_full_review_settled=lambda **kwargs: None,
            record_outreach_tier_outcome=lambda **kwargs: None,
            variant_id_for_search_string=lambda search_string: "",
            )
    )

    service._record_runtime_event(
        search_string=None,
        event_type="first",
        payload={"n": 1},
    )
    first_state.record_event.assert_called_once()

    holder["state"] = second_state
    service._record_runtime_event(
        search_string=None,
        event_type="second",
        payload={"n": 2},
    )
    second_state.record_event.assert_called_once_with(
        run_id=1,
        work_unit_id=None,
        event_type="second",
        payload={"n": 2},
    )
    assert first_state.record_event.call_count == 1
