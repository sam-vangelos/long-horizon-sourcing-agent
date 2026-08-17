"""Tests for the recruiter store (reopen Stage 1 — the durable recruiter primitive)."""

from __future__ import annotations

import pytest

from shared.runtime_state.recruiter_store import (
    RELATIONSHIP_OWNER,
    SIGNAL_ARCHETYPE_PREFERENCE,
    SIGNAL_PRINCIPLE_FEEDBACK,
    RecruiterStore,
)


@pytest.fixture
def store(tmp_path) -> RecruiterStore:
    return RecruiterStore(tmp_path / "_recruiter" / "recruiter.sqlite3")


# --- init / schema ----------------------------------------------------------

def test_initialize_creates_schema_and_version(store: RecruiterStore) -> None:
    with store.connect() as conn:
        tables = {
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"recruiters", "recruiter_briefs", "recruiter_taste_signals",
                "recruiter_candidate_history", "meta"} <= tables
        ver = conn.execute(
            "SELECT value FROM meta WHERE key='recruiter_schema_version'"
        ).fetchone()
        assert ver["value"] == "1"


# --- recruiters (durable entity, idempotent upsert) -------------------------

def test_upsert_recruiter_is_idempotent_by_handle(store: RecruiterStore) -> None:
    a = store.upsert_recruiter("Jordan.Rivera@example.com", display_name="Jordan")
    b = store.upsert_recruiter("jordan.rivera@example.com")  # same handle, normalized
    assert a == b  # one row, not two
    rec = store.get_recruiter(a)
    assert rec["canonical_handle"] == "jordan.rivera@example.com"
    assert rec["display_name"] == "Jordan"  # preserved when 2nd call passes ''


def test_upsert_empty_handle_raises(store: RecruiterStore) -> None:
    with pytest.raises(ValueError):
        store.upsert_recruiter("   ")


def test_get_recruiter_by_handle(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    assert store.get_recruiter_by_handle("A@B.com")["id"] == rid
    assert store.get_recruiter_by_handle("nope@x.com") is None


# --- recruiter_briefs (membership) ------------------------------------------

def test_link_brief_idempotent_and_listed(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, "brief-1", relationship=RELATIONSHIP_OWNER)
    store.link_brief(rid, "brief-1")  # duplicate ignored
    store.link_brief(rid, "brief-2")
    assert store.briefs_for_recruiter(rid) == ["brief-1", "brief-2"]


def test_brief_cascade_on_recruiter_delete(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    store.link_brief(rid, "brief-1")
    with store.connect() as conn:
        conn.execute("DELETE FROM recruiters WHERE id = ?", (rid,))
    assert store.briefs_for_recruiter(rid) == []  # FK ON DELETE CASCADE


# --- taste signals (the compounding axis) -----------------------------------

def test_record_and_read_active_taste_signal(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    sid = store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="designer",
        payload={"principle": "Typographic refinement", "delta": -1},
        source_brief_id="brief-1",
        confidence=0.8,
    )
    active = store.active_taste_signals(rid, domain="designer")
    assert len(active) == 1
    assert active[0]["id"] == sid
    assert active[0]["payload"]["principle"] == "Typographic refinement"
    assert active[0]["confidence"] == 0.8


def test_unknown_signal_kind_raises(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    with pytest.raises(ValueError, match="unknown signal_kind"):
        store.record_taste_signal(rid, signal_kind="bogus", domain="designer")


def test_superseded_signal_excluded_from_active(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    old = store.record_taste_signal(rid, signal_kind=SIGNAL_ARCHETYPE_PREFERENCE, domain="github")
    new = store.record_taste_signal(rid, signal_kind=SIGNAL_ARCHETYPE_PREFERENCE, domain="github")
    store.supersede_signal(old, new)
    active = store.active_taste_signals(rid, domain="github")
    assert [s["id"] for s in active] == [new]  # old is gone, history preserved in table


def test_active_signals_domain_filter(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    store.record_taste_signal(rid, signal_kind=SIGNAL_PRINCIPLE_FEEDBACK, domain="designer")
    store.record_taste_signal(rid, signal_kind=SIGNAL_ARCHETYPE_PREFERENCE, domain="github")
    assert len(store.active_taste_signals(rid)) == 2          # all domains
    assert len(store.active_taste_signals(rid, domain="designer")) == 1


# --- candidate history (cross-brief accretion) ------------------------------

def test_candidate_sighting_accretes_across_briefs(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    store.record_candidate_sighting(rid, 42, brief_id="brief-1", lifecycle_state="full_terminal", recruiter_action="shortlist")
    store.record_candidate_sighting(rid, 42, brief_id="brief-2", lifecycle_state="full_terminal", recruiter_action="contacted")
    h = store.candidate_history(rid, 42)
    assert h["times_surfaced"] == 2
    assert h["first_seen_brief"] == "brief-1"   # preserved
    assert h["last_seen_brief"] == "brief-2"    # updated
    assert h["last_recruiter_action"] == "contacted"


def test_candidate_sighting_action_coalesce(store: RecruiterStore) -> None:
    # a later sighting with no action must NOT clobber a prior action
    rid = store.upsert_recruiter("a@b.com")
    store.record_candidate_sighting(rid, 7, brief_id="b1", recruiter_action="replied")
    store.record_candidate_sighting(rid, 7, brief_id="b2", recruiter_action=None)
    h = store.candidate_history(rid, 7)
    assert h["last_recruiter_action"] == "replied"  # preserved via COALESCE
    assert h["times_surfaced"] == 2


def test_candidate_history_absent_returns_none(store: RecruiterStore) -> None:
    rid = store.upsert_recruiter("a@b.com")
    assert store.candidate_history(rid, 999) is None
