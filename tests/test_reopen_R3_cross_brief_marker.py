"""Reopen Stage 3 (R3, the "remembers" half): the inline cross-brief presence
marker on the candidate-detail surface.

Verifies ``cloris.api._monolith._cross_brief_presence_for_candidate`` — the
read-only join that powers the "you've seen them before" marker:

    candidate (source, state_key, candidate_id)
      -> candidate_persons.person_id            [identity DB]
      -> recruiter_for_brief(brief_id)          [recruiter spine]
      -> candidate_history(recruiter_id, person)[recruiter spine]
      -> CrossBriefPresence | None

The load-bearing properties under test:

1. **Only recurs.** The marker attaches ONLY when the person was sighted in
   at least one OTHER brief (``other_briefs_count`` >= 1). A first-seen-here
   person gets ``None`` — no marker, no "seen in 1 brief" noise.
2. **The elsewhere count is honest.** ``other_briefs_count`` subtracts the
   current brief from ``times_surfaced`` (the recruiter is looking at this
   candidate, so they were sighted here).
3. **Verdict-free (invariant D-B).** ``last_lifecycle_state`` is the pipeline
   state Cloris last observed, NOT a recruiter judgment. The model carries no
   verdict field at all — this test pins that the projection is presence-only.
4. **Fail-soft.** Missing link / unprovisioned recruiter / no source_run ->
   ``None``, never a raise (the candidate page must not 500 on this enrichment).

Seed idiom (candidate_persons link + sighting history) mirrors
``tests/test_reopen_refactorX5.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cloris.models import CandidateDetailResponse, LatestRunRef
from shared.runtime_state.identity_store import IdentityStore
from shared.runtime_state.recruiter_store import RecruiterStore


def _link_candidate_person(
    identity_db_path: Path,
    *,
    person_id: int,
    source: str,
    state_key: str,
    candidate_id: int,
    brief_id: str,
) -> None:
    """Write a ``candidate_persons`` link the way the resolver does. Copied
    from ``test_reopen_refactorX5.py`` so this verification is self-contained."""

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO persons"
            "(id, canonical_name, canonical_handle, created_at, last_seen_at) "
            "VALUES (?, ?, '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (person_id, f"Person {person_id}"),
        )
        conn.execute(
            "INSERT OR REPLACE INTO candidate_persons"
            "(source, state_key, candidate_id, person_id, brief_id, link_kind, "
            "match_signal_json, recruiter_locked, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'auto_strong', '{}', 0, "
            "'2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (source, state_key, candidate_id, person_id, brief_id),
        )


def _detail(
    *, source: str, state_key: str, candidate_id: int, brief_id: str
) -> CandidateDetailResponse:
    """A minimal CandidateDetailResponse carrying just the fields the marker
    resolution reads: source, brief_id, candidate_id, and a source_run whose
    state_key completes the candidate_persons key."""

    return CandidateDetailResponse(
        source=source,  # type: ignore[arg-type]
        brief_id=brief_id,
        candidate_id=candidate_id,
        identity_key=f"key-{candidate_id}",
        display_name="Cand",
        profile_url="https://example.test/c",
        source_run=LatestRunRef(
            source=source,  # type: ignore[arg-type]
            state_key=state_key,
            run_id=1,
        ),
    )


@pytest.fixture()
def _paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the production path resolvers at tmp DBs and reset the recruiter
    resolver to the Stage-1 default (Sam -> id 1)."""

    identity_db = tmp_path / "_identity" / "identity.sqlite3"
    recruiter_db = tmp_path / "_recruiter" / "recruiter.sqlite3"
    identity_db.parent.mkdir(parents=True, exist_ok=True)
    recruiter_db.parent.mkdir(parents=True, exist_ok=True)

    # The enrichment helper calls these with NO injection, so patch the
    # resolvers themselves (patch at the call site module — _monolith imports
    # them lazily from shared.output_paths inside the function).
    import shared.output_paths as op

    monkeypatch.setattr(op, "resolve_identity_db_path", lambda: identity_db)
    monkeypatch.setattr(op, "resolve_recruiter_db_path", lambda: recruiter_db)

    from shared.recruiter_context import reset_recruiter_id_resolver

    reset_recruiter_id_resolver()
    yield identity_db, recruiter_db
    reset_recruiter_id_resolver()


def test_marker_present_when_person_recurs_across_briefs(_paths) -> None:
    identity_db, recruiter_db = _paths
    from cloris.api._monolith import _cross_brief_presence_for_candidate

    source, state_key, cid = "linkedin", "key-a", 5
    person_id = 42
    current_brief = "brief-b"

    # Recruiter owns the current brief (the Stage-2 link the resolver uses).
    # upsert_recruiter materializes the recruiters row (the recruiter_briefs FK
    # target) and returns its real id — don't hardcode 1.
    rstore = RecruiterStore(recruiter_db)
    from shared.recruiter_context import DEFAULT_RECRUITER_HANDLE

    rid = rstore.upsert_recruiter(DEFAULT_RECRUITER_HANDLE)
    rstore.link_brief(rid, current_brief)

    # Person sighted in TWO briefs: brief-a (elsewhere) then brief-b (here).
    rstore.record_candidate_sighting(
        rid, person_id, brief_id="brief-a", lifecycle_state="sourced"
    )
    rstore.record_candidate_sighting(
        rid, person_id, brief_id=current_brief, lifecycle_state="screened"
    )

    # Link the on-screen candidate to that person.
    _link_candidate_person(
        identity_db,
        person_id=person_id,
        source=source,
        state_key=state_key,
        candidate_id=cid,
        brief_id=current_brief,
    )

    presence = _cross_brief_presence_for_candidate(
        _detail(
            source=source, state_key=state_key, candidate_id=cid, brief_id=current_brief
        )
    )

    assert presence is not None, "a person seen in another brief must get a marker"
    assert presence.times_surfaced == 2
    assert presence.other_briefs_count == 1, "exactly one OTHER brief (brief-a)"
    assert presence.first_seen_brief == "brief-a"
    assert presence.last_seen_brief == current_brief
    # Invariant D-B: the marker carries the last pipeline STATE, never a verdict.
    assert presence.last_lifecycle_state == "screened"
    assert not hasattr(presence, "verdict")
    assert not hasattr(presence, "decision")


def test_no_marker_when_first_seen_here(_paths) -> None:
    identity_db, recruiter_db = _paths
    from cloris.api._monolith import _cross_brief_presence_for_candidate

    source, state_key, cid = "linkedin", "key-a", 7
    person_id = 99
    current_brief = "brief-only"

    rstore = RecruiterStore(recruiter_db)
    from shared.recruiter_context import DEFAULT_RECRUITER_HANDLE

    rid = rstore.upsert_recruiter(DEFAULT_RECRUITER_HANDLE)
    rstore.link_brief(rid, current_brief)
    # Sighted ONLY in the current brief — times_surfaced == 1.
    rstore.record_candidate_sighting(
        rid, person_id, brief_id=current_brief, lifecycle_state="sourced"
    )
    _link_candidate_person(
        identity_db,
        person_id=person_id,
        source=source,
        state_key=state_key,
        candidate_id=cid,
        brief_id=current_brief,
    )

    presence = _cross_brief_presence_for_candidate(
        _detail(
            source=source, state_key=state_key, candidate_id=cid, brief_id=current_brief
        )
    )
    assert presence is None, "first-seen-here must NOT render a marker"


def test_fail_soft_when_no_candidate_person_link(_paths) -> None:
    identity_db, recruiter_db = _paths
    from cloris.api._monolith import _cross_brief_presence_for_candidate

    # No candidate_persons link seeded → person unresolvable → None, no raise.
    rstore = RecruiterStore(recruiter_db)
    from shared.recruiter_context import DEFAULT_RECRUITER_HANDLE

    rstore.link_brief(rstore.upsert_recruiter(DEFAULT_RECRUITER_HANDLE), "brief-x")
    presence = _cross_brief_presence_for_candidate(
        _detail(source="linkedin", state_key="key-a", candidate_id=1, brief_id="brief-x")
    )
    assert presence is None


def test_no_marker_when_source_run_absent(_paths) -> None:
    from cloris.api._monolith import _cross_brief_presence_for_candidate

    # state_key rides on source_run; without it the candidate_persons key is
    # incomplete, so the helper degrades to no marker (never a raise).
    detail = CandidateDetailResponse(
        source="linkedin",
        brief_id="brief-x",
        candidate_id=1,
        identity_key="k",
        display_name="C",
        profile_url="u",
        source_run=None,
    )
    assert _cross_brief_presence_for_candidate(detail) is None
