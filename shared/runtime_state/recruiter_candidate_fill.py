"""Reopen Refactor X: the recruiter-candidate current-state fill + divergence metric.

The recruiter — not the brief — is Cloris's durable entity (reopen, LOCKED Sam
2026-06-01). Stage 1 gave the recruiter a per-(recruiter, person) ACCRETION log
(``recruiter_candidate_history`` — times_surfaced, first/last brief). Refactor X
adds the missing other half: a per-(recruiter, person) CURRENT-STATE authority
(``recruiter_candidates``), the resolved single lifecycle a person is in *for a
recruiter*, rolled up across every brief/source that person maps to.

The join the Stage-2 backfill deferred
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tools/backfill_recruiter_store.py`` explicitly punted the candidate-history
roll-up: *"It requires a (source, state_key, person_id) join that ``candidates``
does not carry."* That is exactly true of the brief-keyed ``candidates`` table —
but ``candidate_persons`` (identity DB) DOES carry it. The full join, all
cross-DB:

    person_id
      -> candidate_persons(source, state_key, candidate_id)     [identity.sqlite3]
      -> output/state/<source>/<state_key>/runtime_state.sqlite3 [per-state-dir]
      -> candidates WHERE id = candidate_id
         (current_lifecycle_state, terminal_decision, terminal_payload_json,
          last_seen_at)
      -> MOST-RECENT by last_seen_at across the person's links
      -> upsert into recruiter_candidates                       [recruiter.sqlite3]

``state_key`` is the load-bearing seam: ``identity_resolution_service`` writes
``candidate_persons.state_key = state_dir.name`` (service.py:258 -> :287/:628),
the same value ``cloris.control_plane.enumerate_state_dirs`` yields as the
directory name. So a link row names exactly the per-state-dir DB and the row id
inside it — no scanning, a direct open + point lookup.

Why a sibling module (not a method on ``RecruiterStore``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identical rationale to ``recruiter_sighting.py``: this orchestration spans THREE
DBs ``RecruiterStore`` deliberately does not import — the identity DB (read
``candidate_persons``), the per-state-dir DBs (read ``candidates`` read-only),
and the recruiter DB (write). ``RecruiterStore`` mirrors ``IdentityStore``: a
single-DB store with no cross-DB imports. The recruiter-DB-local primitive — the
``upsert_recruiter_candidate`` write — lives on the store; only the cross-DB
join lives here.

Merge strategy (the contract's flagged decision)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A person can carry candidate rows across multiple briefs/sources in different
terminal states. "Which one is current?" is a genuine product question. The
default — MOST-RECENT by the candidate row's ``last_seen_at`` — is the
defensible, reversible choice, and it lives in a single documented helper
(:func:`_pick_current_state`) so swapping it (e.g. to a terminal-decision
priority order, or recruiter-locked-wins) is a one-function change, not a
rewrite. This is a materialized roll-up, not a source of truth: if the merge
policy changes, re-running the fill re-derives every row.

Fail-soft: like the 3a hook, the fill NEVER raises into a read path. A person
with zero live candidate rows is skipped (no authority row written); an
unreadable / pre-candidates-table state-dir is skipped silently and the other
links still contribute.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CandidateLink:
    """One ``candidate_persons`` link: which per-state-dir candidate row."""

    source: str
    state_key: str
    candidate_id: int


@dataclass(frozen=True)
class CandidateCurrentState:
    """The resolved current-state of one per-state-dir candidate row.

    ``last_seen_at`` is the merge key. ``terminal_payload_json`` is kept as the
    raw stored string (not parsed) so the upsert writes it verbatim into
    ``recruiter_candidates`` — same column, same encoding as the source.

    ``identity_key`` / ``brief_id`` are the REAL columns off the brief-keyed
    ``candidates`` row (verified against the live schema: candidates carries
    both). ``identity_key`` is what gets recorded as ``last_identity_key`` so the
    authority points at a genuine brief-keyed row, not a synthetic composite;
    ``brief_id`` lets a future per-brief diff locate the exact source row.
    """

    source: str
    state_key: str
    candidate_id: int
    brief_id: str
    identity_key: str
    current_lifecycle_state: str
    terminal_decision: str | None
    terminal_payload_json: str
    last_seen_at: str | None


def _candidate_person_links(
    person_id: int, identity_db_path: Path | None, *, source: str | None = None
) -> list[_CandidateLink]:
    """Read every ``candidate_persons`` link for a person (all briefs/sources).

    Person-scoped, NOT brief-scoped: the authority is the person's current
    state across everything they map to, so the merge sees every candidate row.
    Reuses ``IdentityStore`` (the canonical reader of this table).

    ``source`` (Refactor Y.5.1): when given, restrict the links to that source
    only — ``AND source = ?``. This is the seam the per-source divergence report
    leans on to derive a person's source-D-only current-state. ``source=None``
    (the default, every existing caller) is byte-for-byte the all-sources query
    it has always been — no behavior change.
    """

    from shared.runtime_state.identity_store import IdentityStore

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        if source is None:
            rows = conn.execute(
                "SELECT source, state_key, candidate_id FROM candidate_persons "
                "WHERE person_id = ? "
                "ORDER BY source, state_key, candidate_id",
                (person_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT source, state_key, candidate_id FROM candidate_persons "
                "WHERE person_id = ? AND source = ? "
                "ORDER BY source, state_key, candidate_id",
                (person_id, source),
            ).fetchall()
    return [
        _CandidateLink(
            source=str(r["source"]),
            state_key=str(r["state_key"]),
            candidate_id=int(r["candidate_id"]),
        )
        for r in rows
    ]


def _read_candidate_current_state(
    link: _CandidateLink, state_root: Path
) -> CandidateCurrentState | None:
    """Open the per-state-dir DB read-only and fetch the linked candidate row.

    Discovery mirrors ``enumerate_state_dirs`` exactly: the DB lives at
    ``state_root/<source>/<state_key>/runtime_state.sqlite3`` (the link's
    ``state_key`` IS the directory name, by construction at
    ``identity_resolution_service`` write time). Read-only via the
    ``file:...?mode=ro`` URI form, the F3 contract for never mutating
    per-state-dir DBs (matches ``_iter_candidates_for_brief`` service.py:261).

    Returns ``None`` (the link does not contribute) when: the DB file is
    missing, the candidates table predates this state-dir / is unreadable, or
    the row id is absent (a stale link to a deleted candidate). All fail-soft —
    a single unreadable link must not sink the others.
    """

    db_path = state_root / link.source / link.state_key / "runtime_state.sqlite3"
    if not db_path.exists():
        return None

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    conn.row_factory = sqlite3.Row
    try:
        try:
            row = conn.execute(
                "SELECT brief_id, identity_key, current_lifecycle_state, "
                "terminal_decision, terminal_payload_json, last_seen_at "
                "FROM candidates WHERE id = ?",
                (link.candidate_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            # State dir predates the candidates table or is otherwise
            # unreadable — skip silently, exactly like the resolver's walk.
            return None
    finally:
        conn.close()

    if row is None:
        return None

    return CandidateCurrentState(
        source=link.source,
        state_key=link.state_key,
        candidate_id=link.candidate_id,
        brief_id=str(row["brief_id"] or ""),
        identity_key=str(row["identity_key"] or ""),
        current_lifecycle_state=str(row["current_lifecycle_state"] or ""),
        terminal_decision=row["terminal_decision"],
        terminal_payload_json=str(row["terminal_payload_json"] or "{}"),
        last_seen_at=row["last_seen_at"],
    )


def _pick_current_state(
    candidates: list[CandidateCurrentState],
) -> CandidateCurrentState | None:
    """THE MERGE POLICY (swappable). Default: MOST-RECENT by ``last_seen_at``.

    When a person carries candidate rows across several briefs/sources, this
    picks the one whose brief-keyed ``candidates`` row was most recently
    touched. ``last_seen_at`` is an ISO-8601 UTC string written by
    ``_utc_now()`` everywhere it is set, so lexical ``max`` is chronological
    max. A ``NULL`` last_seen_at (shouldn't happen — every write sets it, but
    defensive) sorts earliest so a real timestamp always wins.

    Ties resolve DETERMINISTICALLY. Two links can share an identical
    ``last_seen_at`` — common in tests (hardcoded second-precision stamps) and
    possible in bulk/backfill writes that copy a stamp verbatim. The tiebreaker
    is the ``(source, state_key, candidate_id)`` triple — the ``candidate_persons``
    PRIMARY KEY, globally unique and stable (``candidate_id`` alone is unique only
    WITHIN a per-state-dir DB, so two links from different state-dirs could
    collide on it). Determinism lives HERE, in the policy helper, so it holds
    regardless of the order the caller passes ``candidates`` in — not solely on
    the SQL read's ORDER BY (which is belt-and-suspenders for the one current
    caller, not the contract).

    This is the contract's flagged decision. Most-recent is the defensible,
    reversible default; the roll-up is materialized, so changing this function
    and re-running the fill re-derives every authority row. To swap the policy
    (terminal-decision priority, recruiter-locked-wins, source precedence),
    change ONLY this function.
    """

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda c: (c.last_seen_at or "", c.source, c.state_key, c.candidate_id),
    )


def resolve_person_current_state(
    person_id: int,
    *,
    source: str | None = None,
    identity_db_path: Path | None = None,
    state_root: Path | None = None,
) -> CandidateCurrentState | None:
    """Resolve a person's MERGED current-state across all their candidate rows.

    The full cross-DB join + merge for one person:
      1. read the person's ``candidate_persons`` links (identity DB),
      2. open each linked per-state-dir DB read-only, fetch the candidate's
         current-state (skip any that are missing/unreadable),
      3. pick the most-recent via :func:`_pick_current_state`.

    ``source`` (Refactor Y.5.1): when given, restrict the merge to that source's
    links only — the result is the person's source-D-only current-state (the
    most-recent candidate row WITHIN source D), used by
    :func:`divergence_report_for_source` to ask "what does source D alone say
    this person's state is?". ``source=None`` (every existing caller) merges
    across all sources, unchanged.

    Returns ``None`` when the person has zero live candidate rows for the
    requested scope (caller skips — no authority row should exist for a person
    with nothing resolved).
    """

    if state_root is None:
        from shared.output_paths import STATE_ROOT

        state_root = STATE_ROOT

    links = _candidate_person_links(person_id, identity_db_path, source=source)
    states: list[CandidateCurrentState] = []
    for link in links:
        st = _read_candidate_current_state(link, state_root)
        if st is not None:
            states.append(st)
    return _pick_current_state(states)


def fill_recruiter_candidate(
    recruiter_id: int,
    person_id: int,
    *,
    identity_db_path: Path | None = None,
    recruiter_db_path: Path | None = None,
    state_root: Path | None = None,
) -> bool:
    """Resolve + upsert ONE person's current-state into ``recruiter_candidates``.

    Returns ``True`` when an authority row was written/updated, ``False`` when
    the person had no live candidate row (skipped — see
    :func:`resolve_person_current_state`). Fail-soft is the caller's job (the
    3a hook wraps this in try/except per person); this function does the join
    and the write.
    """

    from shared.runtime_state.recruiter_store import RecruiterStore

    current = resolve_person_current_state(
        person_id,
        identity_db_path=identity_db_path,
        state_root=state_root,
    )
    if current is None:
        return False

    if recruiter_db_path is None:
        from shared.output_paths import resolve_recruiter_db_path

        recruiter_db_path = resolve_recruiter_db_path()

    store = RecruiterStore(recruiter_db_path)
    store.upsert_recruiter_candidate(
        recruiter_id,
        person_id,
        current_lifecycle_state=current.current_lifecycle_state,
        terminal_decision=current.terminal_decision,
        terminal_payload_json=current.terminal_payload_json,
        source=current.source,
        identity_key=current.identity_key,
        last_seen_at=current.last_seen_at,
    )
    return True


# ---------------------------------------------------------------------------
# The divergence metric — the gate for Refactor Y.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DivergenceReport:
    """Health of the current-state authority vs the brief-keyed source of truth.

    Refactor Y can only lean on ``recruiter_candidates`` once it provably agrees
    with the brief-keyed ``candidates`` rows. This report is that gate —
    queryable per recruiter.

    Fields:
      matched              -- authority row's (state, decision) == the
                              most-recent brief-keyed row for that person
      diverged             -- authority row exists but its (state, decision)
                              disagrees with the most-recent brief-keyed row
      missing_in_authority -- person has live candidate rows but NO authority
                              row (the fill never ran / failed for them)
      orphan_in_authority  -- authority row exists but the person has ZERO live
                              candidate rows behind it (stale roll-up)
    """

    recruiter_id: int
    matched: int = 0
    diverged: int = 0
    missing_in_authority: int = 0
    orphan_in_authority: int = 0
    diverged_person_ids: tuple[int, ...] = ()
    missing_person_ids: tuple[int, ...] = ()
    orphan_person_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return {
            "recruiter_id": self.recruiter_id,
            "matched": self.matched,
            "diverged": self.diverged,
            "missing_in_authority": self.missing_in_authority,
            "orphan_in_authority": self.orphan_in_authority,
            "diverged_person_ids": list(self.diverged_person_ids),
            "missing_person_ids": list(self.missing_person_ids),
            "orphan_person_ids": list(self.orphan_person_ids),
        }


def _person_ids_for_recruiter(
    recruiter_id: int, recruiter_db_path: Path
) -> tuple[set[int], set[int]]:
    """Return ``(authority_person_ids, history_person_ids)`` for a recruiter.

    The population to check is every person this recruiter has touched. Two
    sources name them: ``recruiter_candidates`` (the authority) and
    ``recruiter_candidate_history`` (the sighting accretion — what the recruiter
    has actually been shown). Their union is the full population; the set
    difference is what the divergence counts key off.
    """

    from shared.runtime_state.recruiter_store import RecruiterStore

    store = RecruiterStore(recruiter_db_path)
    with store.connect() as conn:
        auth = {
            int(r["person_id"])
            for r in conn.execute(
                "SELECT person_id FROM recruiter_candidates WHERE recruiter_id = ?",
                (recruiter_id,),
            ).fetchall()
        }
        hist = {
            int(r["person_id"])
            for r in conn.execute(
                "SELECT person_id FROM recruiter_candidate_history WHERE recruiter_id = ?",
                (recruiter_id,),
            ).fetchall()
        }
    return auth, hist


def divergence_report(
    recruiter_id: int,
    *,
    identity_db_path: Path | None = None,
    recruiter_db_path: Path | None = None,
    state_root: Path | None = None,
) -> DivergenceReport:
    """Compare ``recruiter_candidates`` against the brief-keyed source of truth.

    For each person in the recruiter's population (authority rows ∪ sighting
    history), re-derive the brief-keyed current-state via the SAME cross-DB
    join the fill uses (:func:`resolve_person_current_state`), then classify:

      - brief-keyed state exists + authority row exists + (state, decision)
        agree           -> matched
      - both exist but disagree            -> diverged
      - brief-keyed exists, authority absent -> missing_in_authority
      - authority exists, brief-keyed absent -> orphan_in_authority

    A person with neither an authority row nor any live brief-keyed candidate
    (history-only, e.g. sighted before any candidate row landed) is counted in
    no bucket — there is nothing to reconcile.
    """

    if recruiter_db_path is None:
        from shared.output_paths import resolve_recruiter_db_path

        recruiter_db_path = resolve_recruiter_db_path()
    if state_root is None:
        from shared.output_paths import STATE_ROOT

        state_root = STATE_ROOT

    from shared.runtime_state.recruiter_store import RecruiterStore

    store = RecruiterStore(recruiter_db_path)
    auth_ids, hist_ids = _person_ids_for_recruiter(recruiter_id, recruiter_db_path)
    population = sorted(auth_ids | hist_ids)

    matched = 0
    diverged: list[int] = []
    missing: list[int] = []
    orphan: list[int] = []

    for person_id in population:
        truth = resolve_person_current_state(
            person_id,
            identity_db_path=identity_db_path,
            state_root=state_root,
        )
        authority = store.recruiter_candidate(recruiter_id, person_id)

        if truth is None and authority is None:
            # History-only person with no live candidate row — nothing to
            # reconcile.
            continue
        if truth is None and authority is not None:
            orphan.append(person_id)
            continue
        if truth is not None and authority is None:
            missing.append(person_id)
            continue

        # Both present — compare the materialized state to the truth.
        assert truth is not None and authority is not None
        same_state = (
            str(authority["current_lifecycle_state"] or "")
            == truth.current_lifecycle_state
        )
        same_decision = authority["terminal_decision"] == truth.terminal_decision
        if same_state and same_decision:
            matched += 1
        else:
            diverged.append(person_id)

    return DivergenceReport(
        recruiter_id=recruiter_id,
        matched=matched,
        diverged=len(diverged),
        missing_in_authority=len(missing),
        orphan_in_authority=len(orphan),
        diverged_person_ids=tuple(diverged),
        missing_person_ids=tuple(missing),
        orphan_person_ids=tuple(orphan),
    )


# ---------------------------------------------------------------------------
# The PER-SOURCE divergence metric — the per-domain gate for Refactor Y.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceDivergenceReport:
    """Per-SOURCE health of the current-state authority — the flippability gate.

    Refactor Y graduates one domain (source) at a time: a domain D can flip to
    leaning on ``recruiter_candidates`` only once the authority provably agrees
    with D's brief-keyed rows FOR THE PERSONS D IS RESPONSIBLE FOR. That is a
    strictly narrower question than :class:`DivergenceReport` (all-sources): it
    isolates the slice of the population attributable to source D.

    The false-divergence trap this report exists to avoid
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    A person can carry a source-D link AND a more-recent link from another
    source. The authority's single merged row legitimately reflects that newer
    OTHER source (``last_source != D``). Such a person is NOT a source-D
    divergence — D's own state may be perfectly self-consistent; the authority
    simply isn't currently speaking for D. A naive "every person with a D link
    whose authority disagrees with D's state = diverged" would FALSELY flag it
    and wrongly block D's flip. So we classify ONLY where the authority's
    current-state is attributable to D (``authority.last_source == D``); a D-link
    person whose authority won from another source lands in ``attributed_elsewhere``
    — surfaced for the operator, but NOT a divergence and NOT a flip blocker.

    Fields (every count keyed off persons WITH a source-D link):
      matched               -- authority.last_source == D AND its (state,
                               decision) == source-D's own current-state
      diverged_for_source   -- authority.last_source == D but its (state,
                               decision) disagrees with source-D's current-state
                               (a REAL source-D divergence — the flip blocker)
      missing_for_source    -- source-D has a live current-state for the person
                               but there is NO authority row at all
      orphan_for_source     -- authority.last_source == D but source-D has ZERO
                               live current-state behind it (stale D roll-up)
      attributed_elsewhere  -- person has a source-D link but the authority's
                               current-state is attributable to a DIFFERENT,
                               legitimately more-recent source (NOT a divergence)

    The Y flippability condition for domain D:
        ``diverged_for_source == 0 AND orphan_for_source == 0``
    (``missing_for_source`` means the fill never ran for those persons — fix by
    running it, not a contradiction; ``attributed_elsewhere`` is by-design and
    never blocks). The two counts that DO block are the two that mean the
    authority, where it claims to speak for D, is wrong or stale.
    """

    recruiter_id: int
    source: str
    matched: int = 0
    diverged_for_source: int = 0
    missing_for_source: int = 0
    orphan_for_source: int = 0
    attributed_elsewhere: int = 0
    diverged_person_ids: tuple[int, ...] = ()
    missing_person_ids: tuple[int, ...] = ()
    orphan_person_ids: tuple[int, ...] = ()
    attributed_elsewhere_person_ids: tuple[int, ...] = ()

    @property
    def flippable(self) -> bool:
        """The Y gate: D can flip iff no real divergence and no stale D orphan."""
        return self.diverged_for_source == 0 and self.orphan_for_source == 0

    def to_dict(self) -> dict:
        return {
            "recruiter_id": self.recruiter_id,
            "source": self.source,
            "matched": self.matched,
            "diverged_for_source": self.diverged_for_source,
            "missing_for_source": self.missing_for_source,
            "orphan_for_source": self.orphan_for_source,
            "attributed_elsewhere": self.attributed_elsewhere,
            "flippable": self.flippable,
            "diverged_person_ids": list(self.diverged_person_ids),
            "missing_person_ids": list(self.missing_person_ids),
            "orphan_person_ids": list(self.orphan_person_ids),
            "attributed_elsewhere_person_ids": list(
                self.attributed_elsewhere_person_ids
            ),
        }


def _person_ids_with_source_link(
    population: list[int], source: str, identity_db_path: Path | None
) -> list[int]:
    """Of ``population``, the persons carrying at least one source-D link.

    The per-source population. ``candidate_persons`` links are person-scoped,
    not recruiter-scoped, so the recruiter scoping comes from ``population``
    (the recruiter's authority ∪ sighting-history ids, the same population
    :func:`divergence_report` walks); this narrows it to those with a link in
    source D. Reuses ``IdentityStore`` (the canonical reader), one point query
    per person — the population is already small and bounded per recruiter.
    """

    from shared.runtime_state.identity_store import IdentityStore

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    store = IdentityStore(identity_db_path)
    out: list[int] = []
    with store.connect() as conn:
        for person_id in population:
            row = conn.execute(
                "SELECT 1 FROM candidate_persons "
                "WHERE person_id = ? AND source = ? LIMIT 1",
                (person_id, source),
            ).fetchone()
            if row is not None:
                out.append(person_id)
    return out


def divergence_report_for_source(
    recruiter_id: int,
    source: str,
    *,
    identity_db_path: Path | None = None,
    recruiter_db_path: Path | None = None,
    state_root: Path | None = None,
) -> SourceDivergenceReport:
    """Per-SOURCE divergence — the per-domain flippability gate for Refactor Y.

    Additive sibling of :func:`divergence_report` (which is UNCHANGED). For each
    person in the recruiter's population WHO CARRIES A SOURCE-D LINK:

      1. ``truth_D`` = source-D-only current-state via
         :func:`resolve_person_current_state` ``(source=D)`` — what D alone says.
      2. ``authority`` = the single merged ``recruiter_candidates`` row
         (:meth:`RecruiterStore.recruiter_candidate`).

    Then classify — the false-divergence guard is the ``last_source == D`` test:

      - authority absent, ``truth_D`` present       -> missing_for_source
      - ``authority.last_source != D``               -> attributed_elsewhere
        (the authority legitimately reflects a more-recent OTHER source; NOT a
        divergence, NOT a flip blocker)
      - ``authority.last_source == D``, ``truth_D`` absent -> orphan_for_source
      - ``authority.last_source == D`` and (state, decision) agree -> matched
      - ``authority.last_source == D`` and (state, decision) disagree
                                                     -> diverged_for_source

    A person with a source-D link but neither an authority row nor any live
    source-D state is counted in no bucket — nothing to reconcile (mirrors the
    all-sources report's history-only skip).
    """

    if recruiter_db_path is None:
        from shared.output_paths import resolve_recruiter_db_path

        recruiter_db_path = resolve_recruiter_db_path()
    if state_root is None:
        from shared.output_paths import STATE_ROOT

        state_root = STATE_ROOT

    from shared.runtime_state.recruiter_store import RecruiterStore

    store = RecruiterStore(recruiter_db_path)
    auth_ids, hist_ids = _person_ids_for_recruiter(recruiter_id, recruiter_db_path)
    full_population = sorted(auth_ids | hist_ids)
    population = _person_ids_with_source_link(
        full_population, source, identity_db_path
    )

    matched = 0
    diverged: list[int] = []
    missing: list[int] = []
    orphan: list[int] = []
    elsewhere: list[int] = []

    for person_id in population:
        truth_d = resolve_person_current_state(
            person_id,
            source=source,
            identity_db_path=identity_db_path,
            state_root=state_root,
        )
        authority = store.recruiter_candidate(recruiter_id, person_id)

        if authority is None:
            if truth_d is not None:
                # Source D has a live state for a person D is responsible for,
                # but the fill never produced an authority row.
                missing.append(person_id)
            # else: no authority and no live source-D state — nothing to
            # reconcile (history-only / stale-link person). No bucket.
            continue

        if str(authority["last_source"] or "") != source:
            # The authority's current-state is attributable to a DIFFERENT,
            # legitimately more-recent source. Source D is not speaking for this
            # person right now — NOT a source-D divergence (the false-divergence
            # trap). Surface it, but it does not block D's flip.
            elsewhere.append(person_id)
            continue

        # authority.last_source == D: the authority claims to speak for D, so
        # source-D's own current-state is the thing it must agree with.
        if truth_d is None:
            # Authority says "current state, from D" but D has no live state
            # behind it — a stale D roll-up.
            orphan.append(person_id)
            continue

        same_state = (
            str(authority["current_lifecycle_state"] or "")
            == truth_d.current_lifecycle_state
        )
        same_decision = authority["terminal_decision"] == truth_d.terminal_decision
        if same_state and same_decision:
            matched += 1
        else:
            diverged.append(person_id)

    return SourceDivergenceReport(
        recruiter_id=recruiter_id,
        source=source,
        matched=matched,
        diverged_for_source=len(diverged),
        missing_for_source=len(missing),
        orphan_for_source=len(orphan),
        attributed_elsewhere=len(elsewhere),
        diverged_person_ids=tuple(diverged),
        missing_person_ids=tuple(missing),
        orphan_person_ids=tuple(orphan),
        attributed_elsewhere_person_ids=tuple(elsewhere),
    )
