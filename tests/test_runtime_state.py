"""Tests for the SQLite-backed canonical runtime state store."""

from __future__ import annotations

import json
import sqlite3

import pytest

from github.schemas import GitHubProgress, GitHubSearchQuery
from shared.runtime_state import RuntimeStateLock, RuntimeStateStore
from shared.runtime_state.linkedin_progress_sync import sync_linkedin_progress
from shared.runtime_state.store import LINKEDIN_STRING_KIND
from shared.schemas import Progress, SearchString


def _make_store(tmp_path):
    return RuntimeStateStore(tmp_path / "runtime_state.sqlite3")


def _start_run(store: RuntimeStateStore, tmp_path, *, source: str = "github", brief_id: str = "brief-1") -> int:
    return store.start_run(
        source=source,
        brief_id=brief_id,
        output_dir=str(tmp_path),
        mode="fresh",
        resume_state={"brief_name": brief_id},
    )


def test_bootstrap_is_idempotent(tmp_path):
    store = _make_store(tmp_path)
    store.initialize()

    with store.connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        # Phase 1.5 bumped schema_version to "4" for the stop_reason_detail
        # column + legacy normalization path. Pin against
        # CURRENT_SCHEMA_VERSION rather than a literal so the test tracks
        # the constant rather than going stale on the next bump.
        from shared.runtime_state.store import CURRENT_SCHEMA_VERSION

        assert row["value"] == CURRENT_SCHEMA_VERSION


def test_rejects_invalid_state_transition(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice",
        profile_url="https://github.com/alice",
    )

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        store.set_candidate_state(
            run_id=run_id,
            source="github",
            brief_id="brief-1",
            identity_key="alice",
            new_state="full_terminal",
            terminal_decision="SAVE",
        )


def test_snippet_extracted_may_skip_facial_for_exec_search_seeds(tmp_path):
    # Exec Search candidates skip facial triage — seeds come from vetted
    # target-company lists. This locks in the widened transition.
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source="exec_search", brief_id="brief-x")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="exec_search",
        brief_id="brief-x",
        identity_key="casey",
        display_name="Casey Operator",
        profile_url="https://linkedin.com/in/casey-operator",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="exec_search",
        brief_id="brief-x",
        identity_key="casey",
        new_state="snippet_extracted",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="exec_search",
        brief_id="brief-x",
        identity_key="casey",
        new_state="full_started",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="exec_search",
        brief_id="brief-x",
        identity_key="casey",
        new_state="full_terminal",
        terminal_decision="SAVE",
    )
    candidate = store.get_candidate(
        source="exec_search", brief_id="brief-x", identity_key="casey"
    )
    assert candidate["current_lifecycle_state"] == "full_terminal"


def test_same_transition_is_idempotent(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice",
        profile_url="https://github.com/alice",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        new_state="snippet_extracted",
    )
    store.set_candidate_state(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        new_state="snippet_extracted",
    )

    candidate = store.get_candidate(source="github", brief_id="brief-1", identity_key="alice")
    assert candidate["current_lifecycle_state"] == "snippet_extracted"


def test_discovery_does_not_reopen_full_terminal_candidate(tmp_path):
    """A repeat discovery must not reset a terminal candidate to discovered.

    Resume/re-run paths can encounter an identity already decided in an earlier
    run. The shared store is the last guard before module loops re-pay full
    evaluation; it must preserve the terminal lifecycle until an explicit
    terminal-clear path runs.
    """

    store = _make_store(tmp_path)
    first_run = _start_run(store, tmp_path)
    second_run = _start_run(store, tmp_path)

    store.record_candidate_discovery(
        run_id=first_run,
        work_unit_id=None,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice",
        profile_url="https://github.com/alice",
    )
    store.set_candidate_state(
        run_id=first_run,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        new_state="snippet_extracted",
    )
    store.set_candidate_state(
        run_id=first_run,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        new_state="full_started",
    )
    store.set_candidate_state(
        run_id=first_run,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        new_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload={"why": "already evaluated"},
    )

    store.record_candidate_discovery(
        run_id=second_run,
        work_unit_id=None,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice Updated",
        profile_url="https://github.com/alice-updated",
    )

    candidate = store.get_candidate(
        source="github", brief_id="brief-1", identity_key="alice"
    )
    assert candidate["current_lifecycle_state"] == "full_terminal"
    assert candidate["terminal_decision"] == "SAVE"
    assert json.loads(candidate["terminal_payload_json"]) == {
        "why": "already evaluated"
    }


def test_discovery_does_not_reopen_failed_terminal_candidate(tmp_path):
    store = _make_store(tmp_path)
    first_run = _start_run(store, tmp_path)
    second_run = _start_run(store, tmp_path)

    store.record_candidate_discovery(
        run_id=first_run,
        work_unit_id=None,
        source="researcher",
        brief_id="brief-1",
        identity_key="paper:1",
        display_name="Paper One",
    )
    store.mark_candidate_terminal_runtime(
        run_id=first_run,
        source="researcher",
        brief_id="brief-1",
        identity_key="paper:1",
        decision="JUDGMENT_FAILURE",
        payload={"reason": "provider unavailable"},
    )

    store.record_candidate_discovery(
        run_id=second_run,
        work_unit_id=None,
        source="researcher",
        brief_id="brief-1",
        identity_key="paper:1",
        display_name="Paper One",
    )

    candidate = store.get_candidate(
        source="researcher", brief_id="brief-1", identity_key="paper:1"
    )
    assert candidate["current_lifecycle_state"] == "failed_terminal"
    assert candidate["terminal_decision"] == "JUDGMENT_FAILURE"


def test_identity_uniqueness_is_scoped_by_brief_and_source(tmp_path):
    store = _make_store(tmp_path)
    first = store.ensure_candidate(
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice",
        profile_url="https://github.com/alice",
    )
    second = store.ensure_candidate(
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice A.",
        profile_url="https://github.com/alice",
    )
    third = store.ensure_candidate(
        source="linkedin",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice",
        profile_url="https://linkedin.com/in/alice",
    )

    assert first == second
    assert third != first


def test_reconciles_orphaned_attempts_on_startup(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice",
        profile_url="https://github.com/alice",
    )
    attempt_id = store.start_attempt(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        stage="facial",
        payload={"candidate_record": {"username": "alice"}},
        source_cursor={"query_id": 1},
        display_name="Alice",
        profile_url="https://github.com/alice",
    )
    assert attempt_id > 0

    reconciled = store.reconcile_open_attempts(source="github", brief_id="brief-1")
    assert reconciled == 1

    candidate = store.get_candidate(source="github", brief_id="brief-1", identity_key="alice")
    assert candidate["current_lifecycle_state"] == "failed_retryable"

    with store.connect() as conn:
        row = conn.execute(
            "SELECT status, failure_kind, failure_reason FROM candidate_attempts WHERE id = ?",
            (attempt_id,),
        ).fetchone()
        assert row["status"] == "reconciled"
        assert row["failure_kind"] == "orphaned_attempt"
        assert "interrupted" in row["failure_reason"]


def test_runtime_lock_enforces_single_writer(tmp_path):
    first = RuntimeStateLock(tmp_path)
    second = RuntimeStateLock(tmp_path)

    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            second.acquire()
    finally:
        first.release()


def test_finish_run_persists_stop_reason(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    store.finish_run(run_id, "interrupted", stop_reason="governor_limit")

    run = store.get_run(run_id)
    assert run["status"] == "interrupted"
    assert run["stop_reason"] == "governor_limit"


def _begin_save_side_effect(store, run_id, identity_key="/talent/profile/ada"):
    return store.begin_candidate_side_effect(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key=identity_key,
        attempt_id=None,
        effect_type="linkedin_save",
        idempotency_key="save",
        payload={"search_string_id": 1},
    )


def test_reconciles_pending_candidate_side_effects_and_allows_manual_replay(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source="linkedin")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="/talent/profile/ada",
        display_name="Ada",
        profile_url="/talent/profile/ada",
    )

    started = _begin_save_side_effect(store, run_id)
    assert started["should_execute"] is True

    reconciled = store.reconcile_pending_side_effects(source="linkedin", brief_id="brief-1")
    assert reconciled == 1

    # P1.1: crash-interrupted pending rows become 'interrupted' (retryable),
    # with the original payload preserved and the interruption noted under
    # its own key — NOT 'failed' with a clobbered payload.
    rows = store.list_candidate_side_effects(source="linkedin", brief_id="brief-1")
    assert rows[0]["status"] == "interrupted"
    payload = json.loads(rows[0]["payload_json"])
    assert payload["search_string_id"] == 1
    assert payload["interruption"]["reason"] == "interrupted"

    # An interrupted row retries WITHOUT consuming an attempt.
    retried = _begin_save_side_effect(store, run_id)
    assert retried["should_execute"] is True
    assert int(retried["side_effect"]["attempt_count"]) == 1

    invalidated = store.invalidate_candidate_side_effects(
        source="linkedin",
        brief_id="brief-1",
        identity_key="/talent/profile/ada",
        effect_type="linkedin_save",
    )
    assert invalidated == 1

    replay = _begin_save_side_effect(store, run_id)
    assert replay["should_execute"] is True


def test_linkedin_save_retries_without_cap_and_reopens_historical_permanent(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source="linkedin")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="/talent/profile/ada",
        display_name="Ada",
        profile_url="/talent/profile/ada",
    )
    _terminal_save(store, run_id, "brief-1", "/talent/profile/ada")

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id="brief-1",
            identity_key="/talent/profile/ada",
            new_state="full_started",
        )

    started = _begin_save_side_effect(store, run_id)
    assert started["should_execute"] is True
    side_effect_id = int(started["side_effect"]["id"])
    assert started["side_effect"]["idempotency_key"] == "save"

    current = started
    for expected_attempt in range(1, 5):
        assert int(current["side_effect"]["id"]) == side_effect_id
        assert int(current["side_effect"]["attempt_count"]) == expected_attempt
        store.complete_candidate_side_effect(
            side_effect_id=side_effect_id,
            status="failed",
            payload={"failure_reason": "save_not_persisted"},
        )
        current = _begin_save_side_effect(store, run_id)
        assert current["should_execute"] is True

    # Historical versions could strand this same receipt permanently.
    with store.connect() as conn:
        conn.execute(
            "UPDATE side_effects SET status = 'failed_permanent' WHERE id = ?",
            (side_effect_id,),
        )

    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id="brief-1",
        identity_key="/talent/profile/ada",
        new_state="full_started",
    )
    reopened = _begin_save_side_effect(store, run_id)
    assert reopened["should_execute"] is True
    assert int(reopened["side_effect"]["id"]) == side_effect_id
    assert reopened["side_effect"]["idempotency_key"] == "save"
    assert int(reopened["side_effect"]["attempt_count"]) == 6
    candidate = store.get_candidate(
        source="linkedin",
        brief_id="brief-1",
        identity_key="/talent/profile/ada",
    )
    assert candidate["current_lifecycle_state"] == "full_started"
    assert candidate["terminal_decision"] is None


def test_other_side_effect_still_stops_after_three_attempts(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source="github")
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="github",
        brief_id="brief-1",
        identity_key="ada",
        display_name="Ada",
        profile_url="https://github.com/ada",
    )

    def begin():
        return store.begin_candidate_side_effect(
            run_id=run_id,
            source="github",
            brief_id="brief-1",
            identity_key="ada",
            attempt_id=None,
            effect_type="github_outreach",
            idempotency_key="outreach",
        )

    current = begin()
    for expected_attempt in range(1, 4):
        assert current["should_execute"] is True
        assert int(current["side_effect"]["attempt_count"]) == expected_attempt
        store.complete_candidate_side_effect(
            side_effect_id=int(current["side_effect"]["id"]),
            status="failed",
        )
        current = begin()

    assert current["should_execute"] is False
    assert current["side_effect"]["status"] == "failed_permanent"


def test_record_candidate_discovery_normalizes_linkedin_url(tmp_path):
    """Phase C-bis 0.4: defense-in-depth URL normalization at the store
    layer. The acquisition path already strips tracking params, but any
    future code path that bypasses acquisition (manual backfill, a
    different module) gets the same scrubbing on insert. The normalizer
    is idempotent."""

    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source="linkedin")

    dirty_url = (
        "https://www.linkedin.com/in/pat-doe?"
        "miniProfileUrn=urn%3Ali%3Afsd_profile%3AACoAAA"
        "&trackingId=abc123"
        "&searchEntityType=PEOPLE"
        "&position=4"
        "&searchId=xyz789"
    )

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="linkedin",
        brief_id="brief-1",
        identity_key="li-pat-doe",
        display_name="Pat Doe",
        profile_url=dirty_url,
    )

    with store.connect() as conn:
        row = conn.execute(
            "SELECT profile_url FROM candidates "
            "WHERE source='linkedin' AND brief_id='brief-1' "
            "AND identity_key='li-pat-doe'"
        ).fetchone()

    assert row is not None
    # Tracking params stripped; trailing slash absent; lowercased.
    assert row["profile_url"] == "https://www.linkedin.com/in/pat-doe"


def test_record_candidate_discovery_does_not_normalize_github_url(tmp_path):
    """Negative case: the normalizer is LinkedIn-specific. GitHub URLs
    pass through untouched, so the defense-in-depth is scoped and won't
    surprise other modules."""

    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source="github")

    github_url = "https://github.com/alice?ref=tracking"

    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source="github",
        brief_id="brief-1",
        identity_key="alice",
        display_name="Alice",
        profile_url=github_url,
    )

    with store.connect() as conn:
        row = conn.execute(
            "SELECT profile_url FROM candidates "
            "WHERE source='github' AND brief_id='brief-1' "
            "AND identity_key='alice'"
        ).fetchone()

    assert row is not None
    # Untouched — the LinkedIn-specific normalizer is gated by source.
    assert row["profile_url"] == github_url


def _terminal_reject(store, run_id, brief_id, identity_key):
    for state in ("snippet_extracted", "full_started"):
        store.set_candidate_state(
            run_id=run_id, source="linkedin", brief_id=brief_id,
            identity_key=identity_key, new_state=state,
        )
    store.set_candidate_state(
        run_id=run_id, source="linkedin", brief_id=brief_id,
        identity_key=identity_key, new_state="full_terminal",
        terminal_decision="REJECT",
    )


def _terminal_save(store, run_id, brief_id, identity_key):
    for state in ("snippet_extracted", "full_started"):
        store.set_candidate_state(
            run_id=run_id, source="linkedin", brief_id=brief_id,
            identity_key=identity_key, new_state=state,
        )
    store.set_candidate_state(
        run_id=run_id, source="linkedin", brief_id=brief_id,
        identity_key=identity_key, new_state="full_terminal",
        terminal_decision="SAVE",
    )


def test_suppression_is_brief_version_aware(tmp_path):
    """P3.1 red-first: a candidate rejected under brief v1 was suppressed
    forever, keyed on the version-stable brief_id. Now: re-eligible under v2;
    a SAVED candidate stays suppressed regardless of revision."""

    store = _make_store(tmp_path)
    brief_id = "brief-1"
    run_v1 = store.start_run(
        source="linkedin", brief_id=brief_id, output_dir=str(tmp_path),
        mode="fresh", brief_content_hash="hash-v1",
    )
    for key, name in (("/in/ada", "Ada"), ("/in/bob", "Bob")):
        store.record_candidate_discovery(
            run_id=run_v1, work_unit_id=None, source="linkedin",
            brief_id=brief_id, identity_key=key, display_name=name,
            profile_url=key,
        )
    _terminal_reject(store, run_v1, brief_id, "/in/ada")
    _terminal_save(store, run_v1, brief_id, "/in/bob")

    # Same brief content: both suppressed (today's behavior preserved).
    assert store.is_dedup_blocked(source="linkedin", brief_id=brief_id, identity_key="/in/ada")
    assert store.is_dedup_blocked(source="linkedin", brief_id=brief_id, identity_key="/in/bob")

    # Brief revised: the REJECT re-opens, the SAVE stays suppressed.
    store.start_run(
        source="linkedin", brief_id=brief_id, output_dir=str(tmp_path),
        mode="fresh", brief_content_hash="hash-v2",
    )
    assert not store.is_dedup_blocked(source="linkedin", brief_id=brief_id, identity_key="/in/ada")
    assert store.is_dedup_blocked(source="linkedin", brief_id=brief_id, identity_key="/in/bob")
    keys = store.list_terminal_identity_keys(source="linkedin", brief_id=brief_id)
    assert "/in/ada" not in keys
    assert "/in/bob" in keys

    # The re-eligibility is logged with the old hash.
    with store.connect() as conn:
        events = conn.execute(
            "SELECT payload_json FROM events WHERE event_type = 'candidate_re_eligible'"
        ).fetchall()
    payloads = [json.loads(e["payload_json"]) for e in events]
    assert any(
        p["identity_key"] == "/in/ada" and p["old_brief_content_hash"] == "hash-v1"
        for p in payloads
    )
    assert not any(p["identity_key"] == "/in/bob" for p in payloads)


def test_legacy_terminal_rows_backfill_and_reopen_on_next_revision(tmp_path):
    """P3.1: pre-hash rows adopt the next run's hash (behavior preserved),
    then re-open on the revision after that. failed_terminal is retryable."""

    store = _make_store(tmp_path)
    brief_id = "brief-1"
    legacy_run = store.start_run(
        source="linkedin", brief_id=brief_id, output_dir=str(tmp_path), mode="fresh",
    )
    store.record_candidate_discovery(
        run_id=legacy_run, work_unit_id=None, source="linkedin",
        brief_id=brief_id, identity_key="/in/cleo", display_name="Cleo",
        profile_url="/in/cleo",
    )
    store.mark_candidate_terminal_runtime(
        run_id=legacy_run, source="linkedin", brief_id=brief_id,
        identity_key="/in/cleo", decision="INSUFFICIENT_DATA",
    )

    # v1 run backfills the legacy row: still suppressed under v1.
    store.start_run(
        source="linkedin", brief_id=brief_id, output_dir=str(tmp_path),
        mode="fresh", brief_content_hash="hash-v1",
    )
    assert store.is_dedup_blocked(source="linkedin", brief_id=brief_id, identity_key="/in/cleo")

    # v2 run re-opens it.
    run_v2 = store.start_run(
        source="linkedin", brief_id=brief_id, output_dir=str(tmp_path),
        mode="fresh", brief_content_hash="hash-v2",
    )
    assert not store.is_dedup_blocked(source="linkedin", brief_id=brief_id, identity_key="/in/cleo")

    # failed_terminal has an outgoing transition again: re-evaluation is legal.
    store.set_candidate_state(
        run_id=run_v2, source="linkedin", brief_id=brief_id,
        identity_key="/in/cleo", new_state="snippet_extracted",
    )
    candidate = store.get_candidate(source="linkedin", brief_id=brief_id, identity_key="/in/cleo")
    assert candidate["current_lifecycle_state"] == "snippet_extracted"


def _upsert_work_unit(
    store: RuntimeStateStore, run_id: int, *, status: str, validate_status: bool = True
) -> int:
    return store.upsert_work_unit(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        kind="github_query",
        source_unit_id="q1",
        display_name="Query 1",
        ordering_index=1,
        status=status,
        validate_status=validate_status,
    )


def test_finish_run_rejects_unknown_status(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with pytest.raises(ValueError, match="compleeted"):
        store.finish_run(run_id, "compleeted")


def test_upsert_work_unit_rejects_unknown_status(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with pytest.raises(ValueError, match="que ued"):
        _upsert_work_unit(store, run_id, status="que ued")


def test_upsert_work_unit_tolerates_unknown_status_when_validation_disabled(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    work_unit_id = store.upsert_work_unit(
        run_id=run_id,
        source="github",
        brief_id="brief-1",
        kind="github_query",
        source_unit_id="q1",
        display_name="Query 1",
        ordering_index=1,
        status="unexpected_legacy",
        validate_status=False,
    )

    unit = store.get_work_unit_by_source_id(run_id, kind="github_query", source_unit_id="q1")
    assert unit is not None
    assert int(unit["id"]) == work_unit_id
    assert unit["status"] == "unexpected_legacy"


def test_upsert_work_unit_still_rejects_unknown_status_by_default(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with pytest.raises(ValueError, match="bogus"):
        store.upsert_work_unit(
            run_id=run_id,
            source="github",
            brief_id="brief-1",
            kind="github_query",
            source_unit_id="q1",
            display_name="Query 1",
            ordering_index=1,
            status="bogus",
        )


def test_upsert_work_unit_replays_unknown_status_already_stored_on_row(tmp_path):
    """Forward-compat round-trip: a status persisted via the hydration opt-out
    must survive a validated checkpoint of the same row — replayed, never
    rejected."""
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    _upsert_work_unit(store, run_id, status="future_status", validate_status=False)
    work_unit_id = _upsert_work_unit(store, run_id, status="future_status")

    unit = store.get_work_unit_by_source_id(run_id, kind="github_query", source_unit_id="q1")
    assert unit is not None
    assert int(unit["id"]) == work_unit_id
    assert unit["status"] == "future_status"


def test_upsert_work_unit_rejects_unknown_status_that_differs_from_stored(tmp_path):
    """Replay tolerance is exact-match only: a row holding a valid status must
    not accept a freshly minted unknown one."""
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    _upsert_work_unit(store, run_id, status="queued")
    with pytest.raises(ValueError, match="minted_status"):
        _upsert_work_unit(store, run_id, status="minted_status")


def test_sync_linkedin_progress_rejects_minted_unknown_status(tmp_path):
    """CLO-162 red-proof: the hot checkpoint writer validates by default.
    Before the fix it passed validate_status=False unconditionally, so this
    write landed silently."""
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source="linkedin")

    progress = Progress(
        brief_name="brief-1",
        strings=[SearchString(id=1, name="s1", boolean="a", status="exploded")],
    )
    with pytest.raises(ValueError, match="exploded"):
        sync_linkedin_progress(
            store=store,
            run_id=run_id,
            brief_id="brief-1",
            progress=progress,
            rebuild_artifacts=lambda _run_id: None,
            work_unit_metrics=lambda _s: {},
        )


def test_sync_linkedin_progress_hydration_optout_then_validated_replay(tmp_path):
    """The legacy importer syncs with validate_status=False (rows may carry
    forward-compat statuses); every later validated checkpoint of the same
    rows must replay those statuses without raising."""
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source="linkedin")

    progress = Progress(
        brief_name="brief-1",
        strings=[SearchString(id=1, name="s1", boolean="a", status="future_status")],
    )
    kwargs = dict(
        store=store,
        run_id=run_id,
        brief_id="brief-1",
        progress=progress,
        rebuild_artifacts=lambda _run_id: None,
        work_unit_metrics=lambda _s: {},
    )
    sync_linkedin_progress(validate_status=False, **kwargs)
    sync_linkedin_progress(**kwargs)

    unit = store.get_work_unit_by_source_id(
        run_id, kind=LINKEDIN_STRING_KIND, source_unit_id="1"
    )
    assert unit is not None
    assert unit["status"] == "future_status"


def test_sync_github_progress_rejects_minted_unknown_status(tmp_path):
    """The GitHub checkpoint writer validates by default too (it previously
    passed validate_status=False unconditionally)."""
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with pytest.raises(ValueError, match="exploded"):
        store.sync_github_progress(
            run_id,
            GitHubProgress(
                brief_name="brief-1",
                queries=[
                    GitHubSearchQuery(
                        id=1,
                        name="q1",
                        query="language:python",
                        channel="user_search",
                        status="exploded",
                    )
                ],
            ),
        )


@pytest.mark.parametrize(
    "status",
    sorted(
        {
            "completed",
            "interrupted",
            "error",
            "governor_limit_reached",
            "abandoned",
            "succeeded",
            "failed",
            "running",
        }
    ),
)
def test_finish_run_accepts_every_live_status(tmp_path, status):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    store.finish_run(run_id, status)

    run = store.get_run(run_id)
    assert run["status"] == status


@pytest.mark.parametrize(
    "status",
    sorted({"queued", "in_progress", "done", "skipped", "error"}),
)
def test_upsert_work_unit_accepts_every_live_status(tmp_path, status):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    work_unit_id = _upsert_work_unit(store, run_id, status=status)

    unit = store.get_work_unit_by_source_id(run_id, kind="github_query", source_unit_id="q1")
    assert unit is not None
    assert int(unit["id"]) == work_unit_id
    assert unit["status"] == status


def test_fresh_db_writes_schema_migration_ledger(tmp_path):
    from shared.runtime_state.store import CURRENT_SCHEMA_VERSION

    store = _make_store(tmp_path)

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT version, applied_at FROM schema_migrations"
        ).fetchall()
        schema_row = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()

    assert len(rows) >= 1
    assert all(row["version"] and row["applied_at"] for row in rows)
    assert schema_row["value"] == CURRENT_SCHEMA_VERSION


def test_record_candidate_discovery_rejects_empty_identity_key(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with pytest.raises(ValueError):
        store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=None,
            source="github",
            brief_id="brief-1",
            identity_key="",
            display_name="Alice",
        )


def test_record_candidate_discovery_rejects_unknown_source(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with pytest.raises(ValueError, match="linkdin"):
        store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=None,
            source="linkdin",
            brief_id="brief-1",
            identity_key="alice",
            display_name="Alice",
        )


@pytest.mark.parametrize(
    "source",
    sorted({"linkedin", "github", "exec_search", "designer", "researcher"}),
)
def test_record_candidate_discovery_accepts_every_known_source(tmp_path, source):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path, source=source)

    candidate_id = store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source=source,
        brief_id="brief-1",
        identity_key=f"id-{source}",
        display_name="Alice",
    )

    assert candidate_id > 0
    candidate = store.get_candidate(
        source=source,
        brief_id="brief-1",
        identity_key=f"id-{source}",
    )
    assert candidate is not None


def test_record_candidate_discovery_rejects_non_serializable_payload(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with pytest.raises(ValueError, match="payload is not JSON-serializable"):
        store.record_candidate_discovery(
            run_id=run_id,
            work_unit_id=None,
            source="github",
            brief_id="brief-1",
            identity_key="alice",
            display_name="Alice",
            payload={"x": object()},
        )

    candidate = store.get_candidate(
        source="github",
        brief_id="brief-1",
        identity_key="alice",
    )
    assert candidate is None


def test_upsert_work_unit_rejects_unknown_source(tmp_path):
    store = _make_store(tmp_path)
    run_id = _start_run(store, tmp_path)

    with pytest.raises(ValueError, match="linkdin"):
        store.upsert_work_unit(
            run_id=run_id,
            source="linkdin",
            brief_id="brief-1",
            kind="github_query",
            source_unit_id="q1",
            display_name="Query 1",
            ordering_index=1,
            status="queued",
        )


def test_schemas_from_dict_logs_dropped_keys(caplog):
    import logging

    from shared.schemas import SearchString

    with caplog.at_level(logging.DEBUG, logger="shared.schemas"):
        obj = SearchString.from_dict(
            {"id": 1, "name": "test", "boolean": "(foo)", "bogus_key": 1}
        )

    assert not hasattr(obj, "bogus_key")
    assert "bogus_key" in caplog.text


def test_reopen_does_not_duplicate_ledger_rows(tmp_path):
    db_path = tmp_path / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)
    with store.connect() as conn:
        first_count = conn.execute(
            "SELECT COUNT(*) AS c FROM schema_migrations"
        ).fetchone()["c"]
        first_versions = [
            row["version"]
            for row in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]

    store2 = RuntimeStateStore(db_path)
    with store2.connect() as conn:
        second_count = conn.execute(
            "SELECT COUNT(*) AS c FROM schema_migrations"
        ).fetchone()["c"]
        second_versions = [
            row["version"]
            for row in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]

    assert second_count == first_count
    assert second_versions == first_versions
