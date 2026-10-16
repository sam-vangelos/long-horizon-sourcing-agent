"""Tests for P3.8 — identity decision durability.

Pins three invariants that `tests/test_identity_resolution_service.py`
doesn't cover:

- A recruiter `keep_separate` decision must survive a later re-run in
  which a NEW corroborating signal (e.g. a handle match) appears between
  the two split persons — automation must not silently re-merge what a
  human split.
- A GitHub candidate whose LinkedIn URL was discovered via profile
  README (`linkedin_url_source == "readme"`) must not auto-merge at
  `auto_strong` on a handle match; it routes to `pending_merge_decisions`
  instead. Blog-sourced matches keep today's auto_strong behavior.

Harness mirrors `tests/test_identity_resolution_service.py` (same
fixture shape, same tmp-sqlite state-dir seeding).
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


def _set_profile_url(state_dir: Path, *, candidate_id: int, profile_url: str) -> None:
    """Directly mutate a seeded candidate's profile_url — simulates a
    fresh acquisition run discovering a new/changed URL for an existing
    candidate row (the "new corroborating signal" scenario).
    """

    from shared.runtime_state.store import RuntimeStateStore

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    with store.connect() as conn:
        conn.execute(
            "UPDATE candidates SET profile_url = ? WHERE id = ?",
            (profile_url, candidate_id),
        )


@pytest.fixture()
def env(tmp_path: Path) -> tuple[Path, Path]:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    identity_db = tmp_path / "_identity" / "identity.sqlite3"
    return state_root, identity_db


# ---------------------------------------------------------------------------
# (a) keep_separate must survive a new corroborating signal.
# ---------------------------------------------------------------------------


def test_keep_separate_survives_new_handle_corroboration(env):
    state_root, identity_db = env
    brief_id = "brief_keep_separate_durable"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_ks")
    cand_a = _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-ks-a",
        display_name="John Smith",
        profile_url="https://www.linkedin.com/in/john-smith-a/",
    )
    cand_b = _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-ks-b",
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
            "SELECT person_a, person_b FROM pending_merge_decisions WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()
    assert pending is not None
    person_a, person_b = int(pending["person_a"]), int(pending["person_b"])

    # Recruiter looks at the pair and says: these are different humans.
    record_recruiter_merge(
        brief_id=brief_id,
        person_a=person_a,
        person_b=person_b,
        decision="keep_separate",
        identity_db_path=identity_db,
    )

    with store.connect() as conn:
        before_a = conn.execute(
            "SELECT person_id, recruiter_locked FROM candidate_persons "
            "WHERE source='linkedin' AND state_key='li_ks' AND candidate_id=?",
            (cand_a,),
        ).fetchone()
        before_b = conn.execute(
            "SELECT person_id, recruiter_locked FROM candidate_persons "
            "WHERE source='linkedin' AND state_key='li_ks' AND candidate_id=?",
            (cand_b,),
        ).fetchone()
    # The keep_separate decision must have locked BOTH sides of the pair
    # immediately (same transaction), before any new signal arrives.
    assert int(before_a["recruiter_locked"]) == 1
    assert int(before_b["recruiter_locked"]) == 1
    assert int(before_a["person_id"]) == person_a or int(before_a["person_id"]) == person_b
    assert int(before_b["person_id"]) == person_a or int(before_b["person_id"]) == person_b
    assert int(before_a["person_id"]) != int(before_b["person_id"])

    # A NEW corroborating signal arrives: candidate B's profile URL now
    # matches candidate A's exactly (e.g. a re-scrape resolves a vanity
    # redirect). Pre-fix, Pass 1's handle grouping would silently
    # re-point B onto A's person.
    _set_profile_url(
        li_dir,
        candidate_id=cand_b,
        profile_url="https://www.linkedin.com/in/john-smith-a/",
    )

    resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    with store.connect() as conn:
        after_a = conn.execute(
            "SELECT person_id, recruiter_locked FROM candidate_persons "
            "WHERE source='linkedin' AND state_key='li_ks' AND candidate_id=?",
            (cand_a,),
        ).fetchone()
        after_b = conn.execute(
            "SELECT person_id, recruiter_locked FROM candidate_persons "
            "WHERE source='linkedin' AND state_key='li_ks' AND candidate_id=?",
            (cand_b,),
        ).fetchone()
        person_count = conn.execute(
            "SELECT COUNT(*) AS n FROM persons WHERE id IN (?, ?)",
            (person_a, person_b),
        ).fetchone()["n"]

    # Both original person rows still exist, distinct, and neither
    # candidate got re-pointed by the new handle match.
    assert person_count == 2
    assert int(after_a["person_id"]) == int(before_a["person_id"])
    assert int(after_b["person_id"]) == int(before_b["person_id"])
    assert int(after_a["person_id"]) != int(after_b["person_id"])
    assert int(after_a["recruiter_locked"]) == 1
    assert int(after_b["recruiter_locked"]) == 1


# ---------------------------------------------------------------------------
# (b) readme-sourced LinkedIn URL must not auto-merge at auto_strong;
#     blog-sourced keeps today's behavior.
# ---------------------------------------------------------------------------


def test_readme_sourced_handle_match_routes_to_pending_not_auto_strong(env):
    state_root, identity_db = env
    brief_id = "brief_readme_pending"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_readme")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-readme",
        display_name="Jamie Rivera",
        profile_url="https://www.linkedin.com/in/jamie-rivera/",
    )

    gh_dir = _seed_state_dir(state_root, source="github", state_key="gh_readme")
    _seed_candidate(
        gh_dir,
        source="github",
        brief_id=brief_id,
        identity_key="jrivera-oss",
        display_name="jrivera-oss",
        profile_url="https://github.com/jrivera-oss",
        terminal_payload={
            "candidate_record": {
                "user": {"name": "Jamie Rivera"},
                "contact": {
                    "linkedin_url": "https://www.linkedin.com/in/jamie-rivera",
                    "linkedin_url_source": "readme",
                },
            }
        },
    )

    from shared.identity_resolution_service import (
        brief_persons_with_evidence,
        resolve_persons_for_brief,
    )
    from shared.runtime_state.identity_store import IdentityStore

    result = resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    # NOT merged into one auto_strong person — two persons, one pending.
    assert result.persons_total == 2
    assert result.pending_merges == 1

    persons = brief_persons_with_evidence(brief_id, identity_db_path=identity_db)
    assert len(persons) == 2
    for person in persons:
        for link in person.sources:
            match_signal = link.match_signal
            assert match_signal.get("kind") != "linkedin_handle", (
                "readme-sourced handle match must not use the auto_strong "
                "linkedin_handle merge path"
            )

    store = IdentityStore(identity_db)
    with store.connect() as conn:
        pending = conn.execute(
            "SELECT recruiter_decision FROM pending_merge_decisions WHERE brief_id = ?",
            (brief_id,),
        ).fetchone()
    assert pending is not None
    assert pending["recruiter_decision"] is None


def test_blog_sourced_handle_match_still_auto_merges(env):
    state_root, identity_db = env
    brief_id = "brief_blog_still_merges"

    li_dir = _seed_state_dir(state_root, source="linkedin", state_key="li_blog")
    _seed_candidate(
        li_dir,
        source="linkedin",
        brief_id=brief_id,
        identity_key="li-blog",
        display_name="Casey Nolan",
        profile_url="https://www.linkedin.com/in/casey-nolan/",
    )

    gh_dir = _seed_state_dir(state_root, source="github", state_key="gh_blog")
    _seed_candidate(
        gh_dir,
        source="github",
        brief_id=brief_id,
        identity_key="cnolan-dev",
        display_name="cnolan-dev",
        profile_url="https://github.com/cnolan-dev",
        terminal_payload={
            "candidate_record": {
                "user": {"name": "Casey Nolan"},
                "contact": {
                    "linkedin_url": "https://www.linkedin.com/in/casey-nolan",
                    "linkedin_url_source": "blog",
                },
            }
        },
    )

    from shared.identity_resolution_service import resolve_persons_for_brief

    result = resolve_persons_for_brief(
        brief_id, identity_db_path=identity_db, state_root=state_root
    )

    assert result.persons_total == 1
    assert result.candidates_linked == 2
    assert result.auto_strong == 2
    assert result.pending_merges == 0
