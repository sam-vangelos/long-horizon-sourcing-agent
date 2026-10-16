"""Pending queries preserve the driver's strict and tolerant recovery contracts."""

import asyncio
import pytest

from linkedin.evaluation_state import EvaluationRunState
from linkedin.search_intelligence import bootstrap_experiment_state
from shared import config
from shared.schemas import Progress, SearchString
from tests.test_linkedin_pipeline import _make_snippet


def test_surface_change_preserves_owner_order_and_tolerates_incomplete_entries(monkeypatch):
    monkeypatch.setattr(config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", True)
    state = EvaluationRunState()
    ledger = state.candidates
    owner = SearchString(id=1, name="owner", boolean="one")
    first = _make_snippet(name="first", source_string_id=1, page=2)
    # Tolerant settlement does not impose recovery's strict identity validation.
    second = _make_snippet(name="second", source_string_id=99, page=1)
    ledger._resume_pending_full_decisions = {
        "second": "invalid", "missing": "FACIAL_YES", "first": "FACIAL_YES",
        "foreign": "FACIAL_YES",
    }
    ledger._resume_pending_full_owner_ids = {
        "first": 1, "missing": 1, "second": 1, "foreign": 2, "stale": 1,
    }
    ledger._resume_pending_full_snippets = {"first": first, "second": second}
    calls = []

    def abandon(**kwargs):
        calls.append((kwargs["key"], kwargs["snippet"]))
        ledger.forget_pending(kwargs["key"])

    monkeypatch.setattr(state.driver, "_abandon_unrecoverable_pending_full", abandon)
    state.driver._settle_owner_pending_full_before_surface_change(
        search_string=owner, reason="surface_replaced_by_drift",
    )
    assert calls == [("first", first), ("second", second)]
    assert ledger._resume_pending_full_decisions == {"foreign": "FACIAL_YES"}
    assert ledger._resume_pending_full_owner_ids == {"foreign": 2, "stale": 1}
    assert ledger._resume_pending_full_snippets == {}


@pytest.mark.parametrize("missing_map", ["decisions", "snippets", "owner_ids"])
def test_surface_change_with_missing_metadata_stays_a_noop(monkeypatch, missing_map):
    monkeypatch.setattr(config, "LINKEDIN_RESUME_RECOVERY_CONTAINMENT_ENABLED", True)
    state = EvaluationRunState()
    ledger = state.candidates
    ledger._resume_pending_full_decisions = {"key": "FACIAL_YES"}
    ledger._resume_pending_full_owner_ids = {"key": 1}
    ledger._resume_pending_full_snippets = {"key": _make_snippet(source_string_id=1)}
    setattr(ledger, "_resume_pending_full_" + missing_map, None)
    before = dict(vars(ledger))
    state.driver._settle_owner_pending_full_before_surface_change(
        search_string=SearchString(id=1, name="owner", boolean="one"),
        reason="surface_replaced_by_drift",
    )
    assert vars(ledger) == before
    if missing_map != "decisions":
        with pytest.raises(RuntimeError, match="metadata is incomplete"):
            ledger.validated_pending(owner_id=1, queue_ids={1})


def test_finish_pending_preserves_reporting_order_checkpoint_and_error_chain(monkeypatch):
    state = EvaluationRunState()
    ledger, driver = state.candidates, state.driver
    owner = SearchString(id=1, name="owner", boolean="one", status="done")
    progress = Progress(brief_name="test", strings=[owner])
    first = _make_snippet(name="first", profile_url="/talent/profile/first", source_string_id=1)
    second = _make_snippet(name="second", profile_url="/talent/profile/second", source_string_id=1)
    ledger._resume_pending_full_decisions = {
        "second": "FACIAL_YES", "missing": "FACIAL_YES", "first": "FACIAL_YES",
    }
    ledger._resume_pending_full_owner_ids = {"first": 1, "second": 1, "missing": 1}
    ledger._resume_pending_full_snippets = {"first": first, "second": second}
    experiment = bootstrap_experiment_state(owner)
    monkeypatch.setattr(driver, "_experiment_state_for", lambda _: experiment)
    events = []

    def checkpoint(actual_progress, **kwargs):
        assert actual_progress is progress
        assert owner.status == "in_progress"
        events.append("checkpoint")
        if len(events) > 1:
            second.name = "after checkpoint"

    async def open_search(*args):
        events.append("open")

    recovery_error = RuntimeError("recovery failed")

    async def recover(**kwargs):
        events.append("recover")
        raise recovery_error

    monkeypatch.setattr(state._allocator_service, "_checkpoint_progress", checkpoint)
    monkeypatch.setattr(state.surface, "_apply_opening_search", open_search)
    monkeypatch.setattr(driver, "_recover_owner_pending_full_evaluations", recover)
    with pytest.raises(RuntimeError) as caught:
        asyncio.run(driver.finish_pending_reviews(owner, progress))
    assert str(caught.value) == (
        "outstanding full review(s) for string #1: "
        "after checkpoint (/talent/profile/second), missing, first (/talent/profile/first)"
    )
    assert caught.value.__cause__ is recovery_error
    assert events == ["checkpoint", "open", "recover", "checkpoint"]
    assert owner.status == "in_progress"
