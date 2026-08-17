"""Reopen Y.5.3: re-sync the recruiter current-state authority after a
candidate mutation.

The recruiter — not the brief — is Cloris's durable entity (reopen, LOCKED Sam
2026-06-01). Refactor X built ``recruiter_candidates``: the per-(recruiter,
person) CURRENT-STATE authority, a cross-DB roll-up of the brief-keyed
``candidates`` rows a person maps to. Stage 3a fills it on every read-path
re-resolution. But a recruiter can ALSO mutate a candidate directly — append a
note, set ``user_status`` (shortlist / parked / contacted / hidden), set
``judgment_accuracy`` — through the three brief-keyed PATCH/POST handlers in
``cloris.api._monolith``. Those handlers write the per-state-dir ``candidates``
row and bump its ``last_seen_at``; without this hook the authority would lag the
mutation until the next read-path resolution happened to re-run the fill. This
module closes that gap: immediately after a brief-keyed mutation lands, it
re-derives that one person's authority row.

Why a thin module separate from ``RuntimeStateStore`` (the per-state-dir store)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identical rationale to ``recruiter_sighting.py`` / ``recruiter_candidate_fill.py``.
The sync spans THREE databases the per-state-dir store deliberately does not
import — the identity DB (read ``candidate_persons`` for the person link), the
recruiter DB (resolve the owning recruiter + write the authority), and (inside
the reused fill) the OTHER per-state-dir DBs that person maps to. Crucially the
authority sync is keyed on ``(source, state_key, candidate_id) -> person_id``,
NOT on ``candidate_id`` alone: ``candidate_id`` is unique only WITHIN a
per-state-dir DB (``candidate_persons`` PK is ``(source, state_key,
candidate_id)``, identity_store.py:121), so the per-state-dir
``RuntimeStateStore`` setters — which carry no ``source`` / ``state_key`` — must
NOT reach cross-DB. That boundary is exactly why this lives in a sibling module
fired from the HANDLER (which knows the state_dir, hence the source + state_key),
not on the store setter. Same reason ``recruiter_sighting`` is a separate module.

The source/state_key seam (verified, the load-bearing split)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The handler resolves the state_dir via
``_find_state_dir_for_candidate(brief_id, candidate_id)``, which walks
``state_dirs_for_brief_id`` -> ``enumerate_state_dirs``. That yields
``(source, state_dir)`` where ``state_dir = state_root / source / state_key``
(control_plane.py:322,325-327). So the split is unambiguous and NOT assumed:

    state_key = state_dir.name
    source    = state_dir.parent.name

This is the same value ``candidate_persons.state_key`` carries
(``identity_resolution_service`` writes ``state_key = state_dir.name``), so the
triple ``(source, state_key, candidate_id)`` is an exact point lookup into
``candidate_persons``.

Reuse, don't reinvent
~~~~~~~~~~~~~~~~~~~~~~~

The re-sync is ``fill_recruiter_candidate(recruiter_id, person_id)`` from
``recruiter_candidate_fill.py`` (Refactor X) — it re-reads the (now-mutated)
candidate row across ALL the person's links, re-derives the most-recent
current-state, and upserts the authority. We do NOT reinvent the upsert and do
NOT clobber the derived ``current_lifecycle_state`` / ``terminal_decision``: the
fill re-derives them. (Note: notes / user_status / judgment_accuracy do not
themselves move ``current_lifecycle_state`` or ``terminal_decision`` — those
mutations touch other columns — so the re-derived authority row is generally
unchanged in those two fields. The hook still fires unconditionally: it keeps
``last_seen_at`` / ``updated_at`` current and is the correct seam for any future
mutation that DOES move lifecycle state, with zero handler change.)

Fail-soft (load-bearing): a missing person link is a clean ``False``, not a
raise; any genuine failure mid-sync is the CALLER's to swallow (the three
handlers wrap the call in try/except and log at debug — see ``_monolith.py``).
The brief-keyed setter already succeeded before this runs, so a broken authority
must NEVER break a recruiter's mutation. Mirrors ``recruiter_sighting`` exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _split_source_state_key(state_dir: Path) -> tuple[str, str]:
    """Split a per-state-dir path into ``(source, state_key)``.

    The canonical layout ``enumerate_state_dirs`` constructs is
    ``state_root/<source>/<state_key>/`` (control_plane.py:322,325-327), so
    ``state_dir.name`` is the state_key (the directory name
    ``candidate_persons.state_key`` mirrors) and ``state_dir.parent.name`` is
    the source. Not assumed — derived from how the launcher names the tree and
    how the identity resolver writes the link's ``state_key``.
    """

    state_dir = Path(state_dir)
    return state_dir.parent.name, state_dir.name


def _person_id_for_candidate(
    source: str,
    state_key: str,
    candidate_id: int,
    identity_db_path: Path | None,
) -> int | None:
    """Resolve ``(source, state_key, candidate_id) -> person_id`` via the
    ``candidate_persons`` link (the soft cross-DB key, identity DB).

    A point lookup on the ``candidate_persons`` PRIMARY KEY
    ``(source, state_key, candidate_id)`` (identity_store.py:121). Returns
    ``None`` when the candidate carries no identity link yet — a candidate the
    resolver hasn't merged into a person (no authority row should exist for it).
    Reuses ``IdentityStore`` (the canonical reader of this table).
    """

    from shared.runtime_state.identity_store import IdentityStore

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        row = conn.execute(
            "SELECT person_id FROM candidate_persons "
            "WHERE source = ? AND state_key = ? AND candidate_id = ?",
            (source, state_key, candidate_id),
        ).fetchone()
    return int(row["person_id"]) if row is not None else None


def _resolve_recruiter_id(brief_id: str, recruiter_db_path: Path | None) -> int:
    """Resolve the recruiter the authority sync should land under.

    Mirrors ``recruiter_sighting._resolve_recruiter_id`` exactly. Primary: the
    brief's owner via ``recruiter_for_brief`` (the Stage-2 link). Fallback: the
    ambient resolver (``get_current_recruiter_id``) plus a LAZY ``link_brief``
    so the next resolution goes through the primary path — a brief that had runs
    but was never backfilled into ``recruiter_briefs``.
    """

    from shared.runtime_state.recruiter_store import RecruiterStore

    if recruiter_db_path is None:
        from shared.output_paths import resolve_recruiter_db_path

        recruiter_db_path = resolve_recruiter_db_path()

    store = RecruiterStore(recruiter_db_path)
    recruiter_id = store.recruiter_for_brief(brief_id)
    if recruiter_id is not None:
        return recruiter_id

    from shared.recruiter_context import get_current_recruiter_id

    recruiter_id = get_current_recruiter_id()
    store.link_brief(recruiter_id, brief_id)
    return recruiter_id


def sync_candidate_mutation(
    brief_id: str,
    candidate_id: int,
    state_dir: Path,
    *,
    identity_db_path: Path | None = None,
    recruiter_db_path: Path | None = None,
    state_root: Path | None = None,
) -> bool:
    """Re-sync the recruiter current-state authority for the person behind a
    just-mutated candidate.

    Call this immediately AFTER a brief-keyed candidate setter
    (``append_candidate_note`` / ``set_candidate_user_status`` /
    ``set_candidate_judgment_accuracy``) succeeds in a mutation handler, BEFORE
    returning the response. The full chain:

      1. ``(source, state_key)`` from the state_dir path (the verified split).
      2. ``person_id`` from ``candidate_persons`` WHERE
         ``(source, state_key, candidate_id)`` — ``None`` -> nothing to sync.
      3. ``recruiter_id`` via ``recruiter_for_brief`` (or the ambient resolver
         + lazy link).
      4. ``fill_recruiter_candidate(recruiter_id, person_id)`` (Refactor X) —
         re-reads the now-mutated candidate row across the person's links,
         re-derives the most-recent current-state, upserts the authority.

    Returns ``True`` when the authority was re-synced, ``False`` when the
    candidate has no person link yet (a clean skip — NOT a failure). A missing
    link is the only ``False`` path; every OTHER failure mode is the CALLER's to
    swallow — this function does not wrap step 3/4 in try/except, because the
    three handlers already wrap the whole call (mirroring
    ``recruiter_sighting``'s fail-soft contract). The brief-keyed mutation has
    already committed; a broken authority must never break the recruiter's
    write.

    ``identity_db_path`` / ``recruiter_db_path`` / ``state_root`` are injectable
    for tests; each defaults to the live resolver / ``STATE_ROOT`` when omitted,
    so the production handler call passes positionals only.
    """

    source, state_key = _split_source_state_key(state_dir)

    person_id = _person_id_for_candidate(
        source, state_key, candidate_id, identity_db_path
    )
    if person_id is None:
        # The candidate isn't linked to a person yet (the resolver hasn't merged
        # it). Nothing to re-sync — a clean skip, not a failure.
        return False

    recruiter_id = _resolve_recruiter_id(brief_id, recruiter_db_path)

    from shared.runtime_state.recruiter_candidate_fill import fill_recruiter_candidate

    # Re-derive + upsert the authority. fill_recruiter_candidate re-reads the
    # (now-mutated) candidate row across ALL the person's links, re-picks the
    # most-recent current-state, and upserts — so it owns the derived
    # current_lifecycle_state / terminal_decision; we never set them here.
    fill_recruiter_candidate(
        recruiter_id,
        person_id,
        identity_db_path=identity_db_path,
        recruiter_db_path=recruiter_db_path,
        state_root=state_root,
    )
    return True
