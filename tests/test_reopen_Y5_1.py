"""Behavioral tests for reopen Refactor Y.5.1 — the PER-SOURCE divergence metric,
the per-domain flippability gate for Refactor Y.

Refactor Y graduates one domain (source) at a time onto the
``recruiter_candidates`` current-state authority. The all-sources
``divergence_report`` (Refactor X) is too coarse for that: it asks "does the
authority agree with the brief-keyed truth across EVERY source?", which conflates
domains. ``divergence_report_for_source(recruiter_id, D)`` isolates the slice
attributable to source D so a single domain can be proved flippable while others
are still dirty.

What these pin (the load-bearing properties):

- SOURCE FILTER: ``_candidate_person_links(person, source='designer')`` returns
  only designer links; ``source=None`` is the unchanged all-sources read.
- CLEAN: a person with a designer candidate, authority filled from designer
  (last_source='designer'), matching -> diverged_for_source('designer') == 0.
- REAL DIVERGENCE: mutate the designer candidate so the authority (still
  last_source='designer') disagrees -> diverged_for_source == 1, flippable False.
- THE FALSE-DIVERGENCE TRAP (the critical test): a person with BOTH a designer
  link AND a github link, authority won from github (last_source='github', more
  recent). ``divergence_report_for_source('designer')`` must NOT count this as
  diverged even though designer's own state disagrees with the authority — it
  lands in ``attributed_elsewhere`` (the authority legitimately speaks for the
  newer source). A naive "D-link person whose authority disagrees with D = diverged"
  WOULD flag it; this test fails that naive impl and passes the correct one.
- PER-SOURCE ISOLATION: domain A clean while domain B has a real divergence ->
  A flippable, B not, measured independently against the same recruiter.
- ADDITIVE: ``divergence_report`` (no source) behaves identically to before;
  ``ensure_candidate`` is untouched.

Seeding helpers (``_seed_candidate`` / ``_seed_brief_persons`` /
``_link_candidate_person``) + the ``tmp_path`` db/state-root fixtures are copied
from ``test_reopen_refactorX.py`` so the per-state-dir layout
(``<root>/state/<source>/<key>/runtime_state.sqlite3``) and the soft cross-DB
``candidate_persons`` key match exactly what the fill joins on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.runtime_state.identity_store import IdentityStore
from shared.runtime_state.recruiter_candidate_fill import (
    _candidate_person_links,
    divergence_report,
    divergence_report_for_source,
    fill_recruiter_candidate,
    resolve_person_current_state,
)
from shared.runtime_state.recruiter_store import RecruiterStore
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirrored from test_reopen_refactorX.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def identity_db(tmp_path: Path) -> Path:
    return tmp_path / "_identity" / "identity.sqlite3"


@pytest.fixture()
def recruiter_db(tmp_path: Path) -> Path:
    return tmp_path / "_recruiter" / "recruiter.sqlite3"


@pytest.fixture()
def state_root(tmp_path: Path) -> Path:
    return tmp_path / "state"


@pytest.fixture(autouse=True)
def _reset_resolver() -> None:
    """Every test starts + ends on the Stage-1 default resolver (mirrors X)."""

    from shared.recruiter_context import reset_recruiter_id_resolver

    reset_recruiter_id_resolver()
    yield
    reset_recruiter_id_resolver()


def _seed_brief_persons(
    identity_db_path: Path, brief_id: str, person_ids: list[int]
) -> None:
    """Write ``persons`` + ``brief_persons`` (copied from test_reopen_refactorX)."""

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        for pid in person_ids:
            conn.execute(
                "INSERT OR IGNORE INTO persons"
                "(id, canonical_name, canonical_handle, created_at, last_seen_at) "
                "VALUES (?, ?, '', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
                (pid, f"Person {pid}"),
            )
            conn.execute(
                "INSERT OR IGNORE INTO brief_persons(brief_id, person_id, first_seen_at) "
                "VALUES (?, ?, '2026-01-01T00:00:00Z')",
                (brief_id, pid),
            )


def _seed_candidate(
    state_root: Path,
    *,
    source: str,
    state_key: str,
    brief_id: str,
    identity_key: str,
    lifecycle_state: str,
    terminal_decision: str | None,
    last_seen_at: str,
) -> int:
    """Create a per-state-dir candidate row and pin its current-state +
    ``last_seen_at`` (the merge key). Copied from test_reopen_refactorX."""

    per_state_dir = state_root / source / state_key
    per_state_dir.mkdir(parents=True, exist_ok=True)
    store = RuntimeStateStore(per_state_dir / "runtime_state.sqlite3")
    cand_id = store.ensure_candidate(
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=f"Cand {identity_key}",
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE candidates SET current_lifecycle_state = ?, "
            "terminal_decision = ?, last_seen_at = ? WHERE id = ?",
            (lifecycle_state, terminal_decision, last_seen_at, cand_id),
        )
    return cand_id


def _link_candidate_person(
    identity_db_path: Path,
    *,
    person_id: int,
    source: str,
    state_key: str,
    candidate_id: int,
    brief_id: str,
) -> None:
    """Write a ``candidate_persons`` link (copied from test_reopen_refactorX)."""

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


# ---------------------------------------------------------------------------
# SOURCE FILTER — _candidate_person_links(source=...) restricts to that source
# ---------------------------------------------------------------------------


def test_candidate_person_links_source_filter(
    identity_db: Path, state_root: Path
) -> None:
    """A person with a designer link AND a github link: filtering by
    source='designer' returns only the designer link; source=None returns both
    (the unchanged all-sources read)."""

    cid_d = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-d",
        brief_id="brief-d",
        identity_key="ik-d",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-01-01T00:00:00+00:00",
    )
    cid_g = _seed_candidate(
        state_root,
        source="github",
        state_key="key-g",
        brief_id="brief-g",
        identity_key="ik-g",
        lifecycle_state="full_terminal",
        terminal_decision="REJECT",
        last_seen_at="2026-02-01T00:00:00+00:00",
    )
    _link_candidate_person(
        identity_db,
        person_id=10,
        source="designer",
        state_key="key-d",
        candidate_id=cid_d,
        brief_id="brief-d",
    )
    _link_candidate_person(
        identity_db,
        person_id=10,
        source="github",
        state_key="key-g",
        candidate_id=cid_g,
        brief_id="brief-g",
    )

    # source-filtered -> only the designer link.
    designer_links = _candidate_person_links(10, identity_db, source="designer")
    assert [l.source for l in designer_links] == ["designer"]
    assert designer_links[0].candidate_id == cid_d

    github_links = _candidate_person_links(10, identity_db, source="github")
    assert [l.source for l in github_links] == ["github"]
    assert github_links[0].candidate_id == cid_g

    # source=None (default) -> both links, unchanged all-sources behavior.
    all_links = _candidate_person_links(10, identity_db)
    assert {l.source for l in all_links} == {"designer", "github"}
    assert len(all_links) == 2

    # And the source-scoped resolve returns the source-D-only current-state.
    truth_d = resolve_person_current_state(
        10, source="designer", identity_db_path=identity_db, state_root=state_root
    )
    assert truth_d is not None
    assert truth_d.source == "designer"
    assert truth_d.terminal_decision == "SAVE"


# ---------------------------------------------------------------------------
# CLEAN — authority filled from designer, matching -> diverged_for_source == 0
# ---------------------------------------------------------------------------


def test_source_divergence_clean_when_authority_from_source_matches(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-100",
        brief_id="brief-100",
        identity_key="ik-100",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, "brief-100", [100])
    _link_candidate_person(
        identity_db,
        person_id=100,
        source="designer",
        state_key="key-100",
        candidate_id=cid,
        brief_id="brief-100",
    )
    # Fill the authority from the (only) designer link.
    assert fill_recruiter_candidate(
        rid,
        100,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    auth = store.recruiter_candidate(rid, 100)
    assert auth["last_source"] == "designer"

    rep = divergence_report_for_source(
        rid,
        "designer",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert rep.matched == 1
    assert rep.diverged_for_source == 0
    assert rep.missing_for_source == 0
    assert rep.orphan_for_source == 0
    assert rep.attributed_elsewhere == 0
    assert rep.flippable is True


# ---------------------------------------------------------------------------
# REAL DIVERGENCE — authority (last_source='designer') disagrees with designer
# ---------------------------------------------------------------------------


def test_source_divergence_one_when_authority_from_source_disagrees(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-110",
        brief_id="brief-110",
        identity_key="ik-110",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, "brief-110", [110])
    _link_candidate_person(
        identity_db,
        person_id=110,
        source="designer",
        state_key="key-110",
        candidate_id=cid,
        brief_id="brief-110",
    )
    assert fill_recruiter_candidate(
        rid,
        110,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )

    # Mutate the authority out from under the designer row, KEEPING
    # last_source='designer' so it still claims to speak for designer — this is
    # a REAL source-D divergence (not attributed-elsewhere).
    store.upsert_recruiter_candidate(
        rid,
        110,
        current_lifecycle_state="full_terminal",
        terminal_decision="REJECT",  # designer row still says SAVE
        terminal_payload_json="{}",
        source="designer",
        identity_key="ik-110",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )

    rep = divergence_report_for_source(
        rid,
        "designer",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert rep.diverged_for_source == 1
    assert rep.diverged_person_ids == (110,)
    assert rep.matched == 0
    assert rep.attributed_elsewhere == 0
    assert rep.flippable is False


# ---------------------------------------------------------------------------
# THE FALSE-DIVERGENCE TRAP — the critical test
# ---------------------------------------------------------------------------


def test_false_divergence_trap_attributed_elsewhere_not_diverged(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """A person with BOTH a designer link AND a github link. The github row is
    MORE RECENT, so the all-sources merge wins github and the authority's
    last_source == 'github'. The designer row's own decision DISAGREES with the
    authority (designer SAVE vs github REJECT).

    A naive per-source impl ("designer-link person whose authority disagrees
    with designer's state = diverged") WOULD flag this as a designer divergence
    and wrongly block designer's flip. The CORRECT impl recognizes the authority
    is attributable to a legitimately newer source -> attributed_elsewhere,
    diverged_for_source == 0. This single test discriminates the two impls.
    """

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    # Designer row: SAVE, OLDER.
    cid_d = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-200d",
        brief_id="brief-200d",
        identity_key="ik-200d",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-01-01T00:00:00+00:00",
    )
    # Github row: REJECT, NEWER -> wins the all-sources merge.
    cid_g = _seed_candidate(
        state_root,
        source="github",
        state_key="key-200g",
        brief_id="brief-200g",
        identity_key="ik-200g",
        lifecycle_state="full_terminal",
        terminal_decision="REJECT",
        last_seen_at="2026-09-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, "brief-200d", [200])
    _link_candidate_person(
        identity_db,
        person_id=200,
        source="designer",
        state_key="key-200d",
        candidate_id=cid_d,
        brief_id="brief-200d",
    )
    _link_candidate_person(
        identity_db,
        person_id=200,
        source="github",
        state_key="key-200g",
        candidate_id=cid_g,
        brief_id="brief-200g",
    )

    # Fill the authority via the all-sources merge -> github wins (newer).
    assert fill_recruiter_candidate(
        rid,
        200,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    auth = store.recruiter_candidate(rid, 200)
    assert auth["last_source"] == "github"  # authority speaks for github, not designer
    assert auth["terminal_decision"] == "REJECT"

    # Sanity: designer's OWN current-state disagrees with the authority. This is
    # exactly the condition a naive impl trips on.
    truth_d = resolve_person_current_state(
        200, source="designer", identity_db_path=identity_db, state_root=state_root
    )
    assert truth_d.terminal_decision == "SAVE"
    assert truth_d.terminal_decision != auth["terminal_decision"]

    rep = divergence_report_for_source(
        rid,
        "designer",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    # The correct classification: NOT a designer divergence.
    assert rep.diverged_for_source == 0, rep.to_dict()
    assert 200 not in rep.diverged_person_ids
    assert rep.attributed_elsewhere == 1
    assert rep.attributed_elsewhere_person_ids == (200,)
    assert rep.matched == 0
    # And designer is therefore NOT blocked from flipping by this person.
    assert rep.flippable is True

    # Cross-check: from GITHUB's vantage, the authority DOES speak for github and
    # agrees with github's state -> matched, github clean too.
    rep_g = divergence_report_for_source(
        rid,
        "github",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert rep_g.matched == 1
    assert rep_g.diverged_for_source == 0
    assert rep_g.attributed_elsewhere == 0
    assert rep_g.flippable is True


# ---------------------------------------------------------------------------
# PER-SOURCE ISOLATION — A clean (flippable) while B has a real divergence
# ---------------------------------------------------------------------------


def test_per_source_isolation_a_flippable_b_blocked(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """Two domains under ONE recruiter, each with its own person. Domain A
    (designer) is clean; domain B (github) has a real divergence. Measured
    independently: A flippable, B not — proving a single domain can graduate
    while another is still dirty."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    # Domain A: person 310, designer, authority matches -> clean.
    cid_a = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-310",
        brief_id="brief-310",
        identity_key="ik-310",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, "brief-310", [310])
    _link_candidate_person(
        identity_db,
        person_id=310,
        source="designer",
        state_key="key-310",
        candidate_id=cid_a,
        brief_id="brief-310",
    )
    assert fill_recruiter_candidate(
        rid, 310, identity_db_path=identity_db,
        recruiter_db_path=recruiter_db, state_root=state_root,
    )

    # Domain B: person 320, github, then mutate the authority (keep
    # last_source='github') to a real divergence.
    cid_b = _seed_candidate(
        state_root,
        source="github",
        state_key="key-320",
        brief_id="brief-320",
        identity_key="ik-320",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, "brief-320", [320])
    _link_candidate_person(
        identity_db,
        person_id=320,
        source="github",
        state_key="key-320",
        candidate_id=cid_b,
        brief_id="brief-320",
    )
    assert fill_recruiter_candidate(
        rid, 320, identity_db_path=identity_db,
        recruiter_db_path=recruiter_db, state_root=state_root,
    )
    store.upsert_recruiter_candidate(
        rid,
        320,
        current_lifecycle_state="full_terminal",
        terminal_decision="REJECT",  # github row still says SAVE
        terminal_payload_json="{}",
        source="github",
        identity_key="ik-320",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )

    rep_a = divergence_report_for_source(
        rid, "designer", identity_db_path=identity_db,
        recruiter_db_path=recruiter_db, state_root=state_root,
    )
    rep_b = divergence_report_for_source(
        rid, "github", identity_db_path=identity_db,
        recruiter_db_path=recruiter_db, state_root=state_root,
    )

    # A measured independently of B: clean + flippable.
    assert rep_a.matched == 1
    assert rep_a.diverged_for_source == 0
    assert rep_a.flippable is True
    # A's population is designer-only: B's person 320 is NOT in it.
    assert 320 not in rep_a.diverged_person_ids
    assert 320 not in rep_a.attributed_elsewhere_person_ids

    # B has a real divergence: not flippable.
    assert rep_b.diverged_for_source == 1
    assert rep_b.diverged_person_ids == (320,)
    assert rep_b.flippable is False
    # B's population is github-only: A's person 310 is NOT in it.
    assert 310 not in rep_b.diverged_person_ids


# ---------------------------------------------------------------------------
# MISSING / ORPHAN for source — the two non-blocking-vs-blocking edges
# ---------------------------------------------------------------------------


def test_source_divergence_missing_and_orphan_for_source(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """missing_for_source: a designer-link person with a live designer state but
    NO authority row (fill never ran). orphan_for_source: an authority row
    last_source='designer' whose designer link points at a candidate row that no
    longer resolves -> source-D has no live state behind the authority."""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    # Person 410: live designer candidate + a sighting (so it's in the
    # population), but NO authority row -> missing_for_source.
    cid_m = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-410",
        brief_id="brief-410",
        identity_key="ik-410",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-05-01T00:00:00+00:00",
    )
    _link_candidate_person(
        identity_db,
        person_id=410,
        source="designer",
        state_key="key-410",
        candidate_id=cid_m,
        brief_id="brief-410",
    )
    store.record_candidate_sighting_once(rid, 410, brief_id="brief-410")

    # Person 420: authority row last_source='designer', and a designer
    # candidate_persons link, BUT the link points at a candidate id that does
    # not exist in the per-state-dir DB -> resolve(source='designer') is None
    # (stale D roll-up) -> orphan_for_source.
    _link_candidate_person(
        identity_db,
        person_id=420,
        source="designer",
        state_key="key-420",
        candidate_id=999999,  # no such candidate row
        brief_id="brief-420",
    )
    # Make the state-dir DB exist (with the candidates table) but without id
    # 999999, so the read is a clean miss, not an unreadable-DB skip.
    _seed_candidate(
        state_root,
        source="designer",
        state_key="key-420",
        brief_id="brief-420",
        identity_key="ik-420-other",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-05-01T00:00:00+00:00",
    )
    store.upsert_recruiter_candidate(
        rid,
        420,
        current_lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        terminal_payload_json="{}",
        source="designer",
        identity_key="ik-420",
        last_seen_at="2026-05-01T00:00:00+00:00",
    )

    rep = divergence_report_for_source(
        rid,
        "designer",
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert rep.missing_for_source == 1
    assert rep.missing_person_ids == (410,)
    assert rep.orphan_for_source == 1
    assert rep.orphan_person_ids == (420,)
    # An orphan blocks flippability (stale D roll-up); missing does not by
    # itself, but here both are present so the gate is closed.
    assert rep.flippable is False


# ---------------------------------------------------------------------------
# ADDITIVE — divergence_report (no source) unchanged; ensure_candidate untouched
# ---------------------------------------------------------------------------


def test_all_sources_divergence_report_unchanged_by_y51(
    identity_db: Path, recruiter_db: Path, state_root: Path
) -> None:
    """The all-sources ``divergence_report`` behaves exactly as in Refactor X:
    a person with a designer candidate, authority filled, matches -> matched==1,
    diverged==0; mutate -> diverged==1. (Same scenario as the X test, re-run
    here to prove Y.5.1's additions did not perturb it.)"""

    store = RecruiterStore(recruiter_db)
    rid = store.upsert_recruiter("a@b.com")

    cid = _seed_candidate(
        state_root,
        source="designer",
        state_key="key-500",
        brief_id="brief-500",
        identity_key="ik-500",
        lifecycle_state="full_terminal",
        terminal_decision="SAVE",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    _seed_brief_persons(identity_db, "brief-500", [500])
    _link_candidate_person(
        identity_db,
        person_id=500,
        source="designer",
        state_key="key-500",
        candidate_id=cid,
        brief_id="brief-500",
    )
    assert fill_recruiter_candidate(
        rid, 500, identity_db_path=identity_db,
        recruiter_db_path=recruiter_db, state_root=state_root,
    )

    clean = divergence_report(
        rid,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert clean.matched == 1
    assert clean.diverged == 0
    assert clean.missing_in_authority == 0
    assert clean.orphan_in_authority == 0
    # No source field on the all-sources report (the additive surface is the
    # separate SourceDivergenceReport).
    assert not hasattr(clean, "source")

    store.upsert_recruiter_candidate(
        rid,
        500,
        current_lifecycle_state="full_terminal",
        terminal_decision="REJECT",
        terminal_payload_json="{}",
        source="designer",
        identity_key="ik-500",
        last_seen_at="2026-04-01T00:00:00+00:00",
    )
    mutated = divergence_report(
        rid,
        identity_db_path=identity_db,
        recruiter_db_path=recruiter_db,
        state_root=state_root,
    )
    assert mutated.diverged == 1
    assert mutated.diverged_person_ids == (500,)


def test_ensure_candidate_untouched_by_y51(tmp_path: Path) -> None:
    """ADDITIVE proof at the writer: ``ensure_candidate`` still behaves exactly
    as before — a discovered candidate write is independent of the recruiter
    store and the (new) per-source divergence path. Also grep-confirms store.py
    carries no reference to the per-source metric."""

    per_state_dir = tmp_path / "iso" / "designer" / "key-1"
    per_state_dir.mkdir(parents=True, exist_ok=True)
    rss = RuntimeStateStore(per_state_dir / "runtime_state.sqlite3")
    cand_id = rss.ensure_candidate(
        source="designer",
        brief_id="brief-additive",
        identity_key="ik-additive",
        display_name="Additive Person",
    )
    assert cand_id > 0
    row = rss.get_candidate(
        source="designer", brief_id="brief-additive", identity_key="ik-additive"
    )
    assert row["current_lifecycle_state"] == "discovered"

    # store.py (the brief-keyed writer) carries no reference to the recruiter
    # authority OR the new per-source metric — Y.5.1 lives entirely outside it.
    store_src = (
        Path(__file__).resolve().parents[1] / "shared" / "runtime_state" / "store.py"
    ).read_text()
    assert "divergence_report_for_source" not in store_src
    assert "recruiter_candidate_fill" not in store_src
