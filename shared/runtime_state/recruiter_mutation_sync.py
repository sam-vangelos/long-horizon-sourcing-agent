"""Re-sync the recruiter current-state authority after a candidate mutation.

The recruiter is Cloris's durable entity across briefs. ``recruiter_candidates``
stores one ``(recruiter, person)`` current-state row assembled from the
brief-keyed candidate rows that person maps to. Candidate mutation handlers
write a state-dir candidate row and update ``last_seen_at``; this hook then
re-derives that person's authority row.

This module is separate from ``RuntimeStateStore`` because the sync spans
three database boundaries: the identity database supplies the
``candidate_persons`` link, the recruiter database owns the authority, and
the shared fill reads the other state-dir databases for that person. The
lookup is keyed by ``(source, state_key, candidate_id)`` rather than
``candidate_id`` alone, since candidate ids are only unique within a
state-dir. The mutation handler knows ``source`` and ``state_key`` from its
``state_dir``; a store setter does not, so the cross-database sync belongs at
the handler seam.

The handler derives ``source`` from ``state_dir.parent.name`` and
``state_key`` from ``state_dir.name``. That pair matches the identity link,
which makes the candidate lookup exact. Re-sync uses
``fill_recruiter_candidate(recruiter_id, person_id)`` to re-read all linked
candidate rows and derive ``current_lifecycle_state`` and
``terminal_decision``; this module never overwrites those derived fields.

A missing person link returns ``False`` as a clean skip. Other sync failures
are left for the caller to swallow because the brief-keyed mutation has
already succeeded; an authority failure must never break that recruiter
write.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _split_source_state_key(state_dir: Path) -> tuple[str, str]:
    """Split a per-state-dir path into ``(source, state_key)``.

    The layout is ``state_root/<source>/<state_key>/``. The directory name
    supplies ``state_key`` and the parent directory supplies ``source``;
    ``candidate_persons.state_key`` uses the same state-key value.
    """

    state_dir = Path(state_dir)
    return state_dir.parent.name, state_dir.name


def _person_id_for_candidate(
    source: str,
    state_key: str,
    candidate_id: int,
    identity_db_path: Path | None,
) -> int | None:
    """Resolve ``(source, state_key, candidate_id) -> person_id`` via
    the ``candidate_persons`` link in the identity database.

    The lookup uses the table's primary key. ``None`` means the candidate
    has no person link yet, so no recruiter authority row exists to sync.
    ``IdentityStore`` remains the canonical reader for this table.
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

    First use the brief's ``recruiter_for_brief`` owner. If the brief has no
    recruiter row yet, use the ambient recruiter id and lazily link the brief
    so later resolutions take the primary path.
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
    """Re-sync the recruiter current-state authority for a mutated candidate.

    Call this immediately after a brief-keyed candidate setter
    (``append_candidate_note`` / ``set_candidate_user_status`` /
    ``set_candidate_judgment_accuracy``) succeeds and before returning the
    response. The chain is:

      1. derive ``(source, state_key)`` from ``state_dir``;
      2. resolve ``person_id`` from ``candidate_persons`` (no link means no
         authority row to sync);
      3. resolve ``recruiter_id`` from the brief owner or ambient fallback;
      4. call ``fill_recruiter_candidate(recruiter_id, person_id)`` to read
         all linked candidate rows, derive current state, and upsert authority.

    Returns ``True`` when the authority was re-synced and ``False`` when the
    candidate has no person link. The missing link is the only ``False`` path;
    the caller owns other failure handling because the brief-keyed mutation
    has already committed. The injectable database paths and ``state_root``
    default to the live resolvers for production callers.
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
