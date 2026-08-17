"""Tests for `shared/identity_resolution_service.py` (Phase F Slice F3).

Pins the F3 contract:
- Walks per-state-dir DBs read-only; writes only to identity DB.
- `auto_strong` for handle matches and singletons.
- `auto_medium` for name + corroborating signal.
- Ambiguous name-only matches → pending_merge_decisions, NOT auto-merge.
- Cross-brief same handle → ONE persons row, TWO brief_persons rows.
- recruiter_locked links never auto-overwritten.
- Pending decisions are terminal once decided.
- describe_merge_signal returns editorial prose per kind.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _seed_state_dir(
    state_root: Path,
    *,
    source: str,
    state_key: str,
) -> Path:
    state_dir = state_root / source / state_key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _seed_candidate(
    state_dir: Path,
    *,
    source: str,
    brief_id: str,
    identity_key: str,
    display_name: str,
    profile_url: str = "",
    terminal_payload: dict | None = None,
) -> int:
    from shared.runtime_state.store import RuntimeStateStore

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    candidate_id = store.ensure_candidate(
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=display_name,
        profile_url=profile_url,
        initial_state="full_terminal",
    )
    if terminal_payload is not None:
        with store.connect() as conn:
            conn.execute(
                "UPDATE candidates SET terminal_payload_json = ? WHERE id = ?",
                (json.dumps(terminal_payload), candidate_id),
            )
    return candidate_id


@pytest.fixture()
def env(tmp_path: Path) -> tuple[Path, Path]:
    """Returns (state_root, identity_db_path) under tmp_path."""

    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    identity_db = tmp_path / "_identity" / "identity.sqlite3"
    return state_root, identity_db


def test_handle_match_across_sources_merges_to_auto_strong(env):
    state_root, identity_db = env
    brief_id = "brief_handle_match"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_proj_1")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-1",
        display_name="Eri Barrett",
        profile_url="https://www.linkedin.com/in/eri-barrett/",
    )

    gh_dir = _seed_state_dir(state_root, source="github", state_key="gh_proj_1")
    _seed_candidate(
        gh_dir,
        source="github",
        brief_id=brief_id,
        identity_key="erosika",
        display_name="erosika",
        profile_url="https://github.com/erosika",
        terminal_payload={
            "candidate_record": {
                "user": {"name": "Eri Barrett"},
                "contact": {"linkedin_url": "https://www.linkedin.com/in/eri-barrett"},
            }
        },
    )

    from shared.identity_resolution_service import (
        brief_persons_with_evidence,
        resolve_persons_for_brief,
    )

    result = resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    assert result.persons_total == 1
    assert result.candidates_linked == 2
    assert result.auto_strong == 2
    assert result.pending_merges == 0

    persons = brief_persons_with_evidence(brief_id, identity_db_path=identity_db)
    assert len(persons) == 1
    assert persons[0].canonical_handle == "eri-barrett"
    assert {link.source for link in persons[0].sources} == {"linkedin", "github"}
    assert all(link.link_kind == "auto_strong" for link in persons[0].sources)


def test_name_with_corroboration_merges_to_auto_medium(env):
    state_root, identity_db = env
    brief_id = "brief_name_corroborated"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_proj_2")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-2",
        display_name="Pat Smith",
        profile_url="https://www.linkedin.com/in/patsmith/",
    )

    gh_dir = _seed_state_dir(state_root, source="github", state_key="gh_proj_2")
    _seed_candidate(
        gh_dir,
        source="github",
        brief_id=brief_id,
        identity_key="patsmith",
        display_name="patsmith",
        profile_url="https://github.com/patsmith",
        terminal_payload={
            "candidate_record": {"user": {"name": "Pat Smith"}, "contact": {}}
        },
    )

    from shared.identity_resolution_service import (
        brief_persons_with_evidence,
        resolve_persons_for_brief,
    )

    result = resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    # Pass 1 (handle) catches it because LinkedIn handle "patsmith"
    # equals GitHub username "patsmith" — but that's only via Pass 2's
    # name+corroboration path. Pass 1 requires both candidates to carry
    # the same NON-EMPTY linkedin_handle; GitHub's linkedin_handle is
    # empty here (no contact.linkedin_url). So this should hit Pass 2.
    assert result.persons_total == 1
    assert result.auto_medium == 2
    assert result.auto_strong == 0
    assert result.pending_merges == 0

    persons = brief_persons_with_evidence(brief_id, identity_db_path=identity_db)
    assert len(persons) == 1
    assert {link.link_kind for link in persons[0].sources} == {"auto_medium"}


def test_name_only_ambiguous_writes_pending_merge_not_auto(env):
    state_root, identity_db = env
    brief_id = "brief_name_only"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_proj_3")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-3",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-12345/",
    )
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-4",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-67890/",
    )

    from shared.identity_resolution_service import (
        brief_persons_with_evidence,
        resolve_persons_for_brief,
    )

    result = resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    # Two distinct LinkedIn handles → Pass 1 yields nothing.
    # Two same names + no GitHub username corroboration → Pass 2 ambiguous.
    # Result: TWO singleton persons + ONE pending_merge row.
    assert result.persons_total == 2
    assert result.pending_merges == 1
    assert result.auto_medium == 0

    persons = brief_persons_with_evidence(brief_id, identity_db_path=identity_db)
    assert len(persons) == 2


def test_cross_brief_same_handle_one_person_two_memberships(env):
    state_root, identity_db = env

    li_dir_a = _seed_state_dir(state_root, source="linkedin", state_key="brief_a")
    _seed_candidate(
        li_dir_a,
        source="linkedin",
        brief_id="brief_a",
        identity_key="li-a",
        display_name="Eri Barrett",
        profile_url="https://www.linkedin.com/in/eri-barrett/",
    )
    li_dir_b = _seed_state_dir(state_root, source="linkedin", state_key="brief_b")
    _seed_candidate(
        li_dir_b,
        source="linkedin",
        brief_id="brief_b",
        identity_key="li-b",
        display_name="Eri Barrett",
        profile_url="https://www.linkedin.com/in/eri-barrett/",
    )

    from shared.identity_resolution_service import resolve_persons_for_brief
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        "brief_a", identity_db_path=identity_db, state_root=state_root
    )
    resolve_persons_for_brief(
        "brief_b", identity_db_path=identity_db, state_root=state_root
    )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        person_count = conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"]
        membership_count = conn.execute(
            "SELECT COUNT(*) AS n FROM brief_persons"
        ).fetchone()["n"]
    # Same human → ONE persons row.
    assert person_count == 1
    # But TWO brief memberships.
    assert membership_count == 2


def test_resolve_is_idempotent(env):
    state_root, identity_db = env
    brief_id = "brief_idempotent"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_idem")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-x",
        display_name="Idem Person",
        profile_url="https://www.linkedin.com/in/idem-person/",
    )

    from shared.identity_resolution_service import resolve_persons_for_brief
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        person_count = conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"]
        link_count = conn.execute(
            "SELECT COUNT(*) AS n FROM candidate_persons"
        ).fetchone()["n"]
    assert person_count == 1
    assert link_count == 1


def test_no_handle_singleton_resolve_is_idempotent(env):
    state_root, identity_db = env
    brief_id = "brief_no_handle_idempotent"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_no_handle")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-no-handle",
        display_name="No Handle Person",
        profile_url="",
    )

    from shared.identity_resolution_service import (
        brief_persons_with_evidence,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        person_count = conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"]
        membership_count = conn.execute(
            "SELECT COUNT(*) AS n FROM brief_persons WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()["n"]
        link_count = conn.execute(
            "SELECT COUNT(*) AS n FROM candidate_persons WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()["n"]

    assert person_count == 1
    assert membership_count == 1
    assert link_count == 1
    persons = brief_persons_with_evidence(brief_id, identity_db_path=identity_db)
    assert len(persons) == 1
    assert len(persons[0].sources) == 1


def test_no_handle_ambiguous_resolve_is_idempotent(env):
    state_root, identity_db = env
    brief_id = "brief_no_handle_ambiguous"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_no_handle_ambig")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-no-handle-a",
        display_name="Common Name",
        profile_url="",
    )
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-no-handle-b",
        display_name="Common Name",
        profile_url="",
    )

    from shared.identity_resolution_service import (
        pending_decisions_for_brief,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        person_count = conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"]
        membership_count = conn.execute(
            "SELECT COUNT(*) AS n FROM brief_persons WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()["n"]
        link_count = conn.execute(
            "SELECT COUNT(*) AS n FROM candidate_persons WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()["n"]
        pending_count = conn.execute(
            "SELECT COUNT(*) AS n FROM pending_merge_decisions WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()["n"]

    assert person_count == 2
    assert membership_count == 2
    assert link_count == 2
    assert pending_count == 1
    decisions = pending_decisions_for_brief(brief_id, identity_db_path=identity_db)
    assert len(decisions) == 1
    assert len(decisions[0].person_a.sources) == 1
    assert len(decisions[0].person_b.sources) == 1


def test_identity_read_models_ignore_orphaned_memberships_and_pending(env):
    state_root, identity_db = env
    brief_id = "brief_orphan_read_model"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_orphan")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-orphan-a",
        display_name="Orphan Test",
        profile_url="",
    )
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-orphan-b",
        display_name="Orphan Test",
        profile_url="",
    )

    from shared.identity_resolution_service import (
        brief_persons_with_evidence,
        pending_decisions_for_brief,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        orphan_id = conn.execute(
            """
            INSERT INTO persons(canonical_name, canonical_handle, created_at, last_seen_at)
            VALUES ('Old Orphan', '', '2026-05-24T00:00:00Z', '2026-05-24T00:00:00Z')
            """
        ).lastrowid
        conn.execute(
            """
            INSERT INTO brief_persons(brief_id, person_id, first_seen_at)
            VALUES (?, ?, '2026-05-24T00:00:00Z')
            """,
            (brief_id, orphan_id),
        )
        linked_id = conn.execute(
            "SELECT person_id FROM candidate_persons WHERE brief_id = ? LIMIT 1",
            (brief_id,),
        ).fetchone()["person_id"]
        conn.execute(
            """
            INSERT INTO pending_merge_decisions(
                brief_id, person_a, person_b, confidence, evidence_json, created_at
            )
            VALUES (?, ?, ?, 0.55, '{}', '2026-05-24T00:00:00Z')
            """,
            (brief_id, linked_id, orphan_id),
        )

    persons = brief_persons_with_evidence(brief_id, identity_db_path=identity_db)
    decisions = pending_decisions_for_brief(brief_id, identity_db_path=identity_db)

    assert len(persons) == 2
    assert all(person.sources for person in persons)
    assert len(decisions) == 1
    assert all(decision.person_a.sources for decision in decisions)
    assert all(decision.person_b.sources for decision in decisions)


def test_recruiter_locked_link_not_overwritten(env):
    state_root, identity_db = env
    brief_id = "brief_locked"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_lock")
    li_candidate_id = _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-lock",
        display_name="Locked Person",
        profile_url="https://www.linkedin.com/in/locked-person/",
    )

    from shared.identity_resolution_service import (
        record_recruiter_unlink,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    # Recruiter explicitly unlinks (splits off) — sets recruiter_locked=1.
    record_recruiter_unlink(
        source="linkedin",
        state_key="li_lock",
        candidate_id=li_candidate_id,
        identity_db_path=identity_db,
    )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        before = conn.execute(
            "SELECT person_id, link_kind, recruiter_locked FROM candidate_persons "
            "WHERE source='linkedin' AND state_key='li_lock'"
        ).fetchone()
    assert int(before["recruiter_locked"]) == 1
    assert before["link_kind"] == "manual"
    locked_person_id = int(before["person_id"])

    # Re-run resolver — the locked link must NOT be re-pointed.
    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    with store.connect() as conn:
        after = conn.execute(
            "SELECT person_id, link_kind, recruiter_locked FROM candidate_persons "
            "WHERE source='linkedin' AND state_key='li_lock'"
        ).fetchone()
    assert int(after["person_id"]) == locked_person_id
    assert after["link_kind"] == "manual"
    assert int(after["recruiter_locked"]) == 1


def test_pending_decision_is_terminal_not_re_prompted(env):
    state_root, identity_db = env
    brief_id = "brief_terminal_pending"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_pending")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-pa",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-a/",
    )
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-pb",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-b/",
    )

    from shared.identity_resolution_service import (
        record_recruiter_merge,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        pending = conn.execute(
            "SELECT id, person_a, person_b FROM pending_merge_decisions "
            "WHERE brief_id = ? AND recruiter_decision IS NULL",
            (brief_id,),
        ).fetchone()
    assert pending is not None
    person_a, person_b = int(pending["person_a"]), int(pending["person_b"])

    record_recruiter_merge(
        brief_id=brief_id,
        person_a=person_a,
        person_b=person_b,
        decision="keep_separate",
        identity_db_path=identity_db,
    )

    # Re-run resolver — no new pending row should be inserted.
    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    with store.connect() as conn:
        rows = conn.execute(
            "SELECT recruiter_decision FROM pending_merge_decisions WHERE brief_id = ?",
            (brief_id,),
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["recruiter_decision"] == "keep_separate"


def test_recruiter_merge_collapses_persons(env):
    state_root, identity_db = env
    brief_id = "brief_recruiter_merge"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_merge")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-m1",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-1/",
    )
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-m2",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-2/",
    )

    from shared.identity_resolution_service import (
        record_recruiter_merge,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    store = IdentityStore(identity_db)
    with store.connect() as conn:
        pending = conn.execute(
            "SELECT person_a, person_b FROM pending_merge_decisions WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()
    record_recruiter_merge(
        brief_id=brief_id,
        person_a=int(pending["person_a"]),
        person_b=int(pending["person_b"]),
        decision="merge",
        identity_db_path=identity_db,
    )

    with store.connect() as conn:
        person_count = conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()["n"]
        link_kinds = {
            row["link_kind"]
            for row in conn.execute(
                "SELECT link_kind FROM candidate_persons WHERE brief_id = ?",
                (brief_id,),
            ).fetchall()
        }
    assert person_count == 1
    assert link_kinds == {"manual"}


def test_describe_merge_signal_per_kind(env):
    from shared.identity_resolution_service import describe_merge_signal

    assert (
        describe_merge_signal("auto_strong", {"kind": "linkedin_handle"})
        == "Same LinkedIn handle on both saves."
    )
    assert (
        describe_merge_signal("auto_medium", {"kind": "name_with_corroboration"})
        == "Names match; corroborating GitHub link."
    )
    assert (
        describe_merge_signal("manual", {"kind": "linkedin_handle"})
        == "Recruiter merged these candidates."
    )
    assert (
        describe_merge_signal("auto_strong", {"kind": "name_only_ambiguous"})
        == "Same name; recruiter review suggested before merging."
    )
    assert describe_merge_signal("auto_strong", {"kind": "singleton"}) == ""


def test_singleton_candidate_gets_its_own_person(env):
    state_root, identity_db = env
    brief_id = "brief_singleton"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_solo")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-solo",
        display_name="Solo Person",
        profile_url="https://www.linkedin.com/in/solo-person/",
    )

    from shared.identity_resolution_service import (
        brief_persons_with_evidence,
        resolve_persons_for_brief,
    )

    result = resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )
    assert result.persons_total == 1
    assert result.candidates_linked == 1
    assert result.auto_strong == 1
    assert result.pending_merges == 0

    persons = brief_persons_with_evidence(brief_id, identity_db_path=identity_db)
    assert len(persons) == 1
    assert persons[0].sources[0].link_kind == "auto_strong"


def test_resolver_walks_state_root_readonly(env):
    """Smoke: resolver opens per-state-dir DBs read-only.

    Pinned by trying a write to the underlying DB after resolve completes
    (the per-state-dir DB must still accept writes — the resolver should
    not have left an exclusive lock).
    """

    state_root, identity_db = env
    brief_id = "brief_readonly"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_ro")
    li_candidate_id = _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-ro",
        display_name="ReadOnly Test",
        profile_url="https://www.linkedin.com/in/ro-test/",
    )

    from shared.identity_resolution_service import resolve_persons_for_brief
    from shared.runtime_state.store import RuntimeStateStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    # Per-state-dir DB still writable (resolver did not poison it).
    store = RuntimeStateStore(li_dir / "runtime_state.sqlite3")
    with store.connect() as conn:
        conn.execute(
            "UPDATE candidates SET display_name = ? WHERE id = ?",
            ("Updated Name", li_candidate_id),
        )


def test_brief_persons_with_evidence_returns_empty_for_unknown_brief(env):
    _, identity_db = env

    from shared.identity_resolution_service import brief_persons_with_evidence

    persons = brief_persons_with_evidence(
        "nonexistent_brief", identity_db_path=identity_db
    )
    assert persons == []


def test_brief_person_count_matches_evidence_read_model(env):
    state_root, identity_db = env
    brief_id = "brief_count_contract"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_count")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-count-a",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-count-a/",
    )
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-count-b",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-count-b/",
    )

    from shared.identity_resolution_service import (
        brief_person_count,
        brief_persons_with_evidence,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    # A stale membership without candidate links should not inflate the count,
    # matching brief_persons_with_evidence() semantics.
    store = IdentityStore(identity_db)
    with store.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO persons(canonical_name, canonical_handle, created_at, last_seen_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "Stale Person",
                "",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO brief_persons(brief_id, person_id, first_seen_at)
            VALUES (?, ?, ?)
            """,
            (brief_id, cursor.lastrowid, "2026-01-01T00:00:00+00:00"),
        )

    persons = brief_persons_with_evidence(brief_id, identity_db_path=identity_db)
    assert len(persons) == 2
    assert brief_person_count(brief_id, identity_db_path=identity_db) == len(persons)
    assert brief_person_count("nonexistent_brief", identity_db_path=identity_db) == 0
