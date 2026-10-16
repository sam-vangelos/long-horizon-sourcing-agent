"""Record recruiter candidate sightings after identity resolution.

Once ``resolve_persons_for_brief`` has resolved a brief's people, record one
sighting per person for the recruiter who owns the brief. This thin orchestration
module joins the global identity DB, recruiter DB, and recruiter-context seam;
the single-DB ``RecruiterStore`` remains free of cross-DB imports.

Both read call sites may run the hook repeatedly. Reuse the resolver's
``brief_persons`` rows and ``record_candidate_sighting_once`` so the same
(recruiter, person, brief) is a no-op while a new brief accretes. The hook never raises into a read path: one call site invokes the resolver
outside a try/except, so a raise here would fail the endpoint. Any identity or
recruiter-store failure is debug-logged and the read continues; the sighting
is a calibration side effect, not part of the response contract.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _resolved_person_ids(brief_id: str, identity_db_path: Path | None) -> list[int]:
    """Read the persons the resolver already wrote for this brief.

    Consumes ``brief_persons`` (the per-brief person-membership table, PK
    ``(brief_id, person_id)``, already deduped) — NOT ``candidate_persons``,
    which is the per-candidate link at the wrong grain (one person can carry
    several candidate links across sources). Does NOT run resolution; the
    caller fires this only after ``resolve_persons_for_brief`` has populated
    the table.
    """

    from shared.runtime_state.identity_store import IdentityStore

    if identity_db_path is None:
        from shared.output_paths import resolve_identity_db_path

        identity_db_path = resolve_identity_db_path()

    store = IdentityStore(identity_db_path)
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT person_id FROM brief_persons WHERE brief_id = ?",
            (brief_id,),
        ).fetchall()
    return [int(r["person_id"]) for r in rows]


def _resolve_recruiter_id(brief_id: str, recruiter_db_path: Path | None) -> int:
    """Resolve the recruiter a sighting should land under.

    Primary: the brief's owner via ``recruiter_for_brief`` (the Stage-2 link),
    so the sighting follows the brief even if the ambient resolver later
    changes. Fallback: the ambient resolver (Stage-1 implicit recruiter / a
    Phase-2 authenticated principal) plus a LAZY ``link_brief`` so the next
    call resolves directly through the primary path — the brief had runs but
    was never backfilled into ``recruiter_briefs``.
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
    # Lazily own the brief so subsequent sightings (and the recruiter
    # dashboard's brief list) resolve through the primary path.
    store.link_brief(recruiter_id, brief_id)
    return recruiter_id


def record_sightings_for_brief(
    brief_id: str,
    *,
    identity_db_path: Path | None = None,
    recruiter_db_path: Path | None = None,
    state_root: Path | None = None,
) -> int:
    """Record one idempotent sighting per resolved person on ``brief_id``.

    Call this immediately AFTER ``resolve_persons_for_brief(brief_id)`` at its
    read-path call sites. Reads the persons the resolver wrote
    (``brief_persons``), resolves the owning recruiter, accrues a sighting per
    person via the idempotent ledger gate, AND (reopen Refactor X) fills the
    per-(recruiter, person) CURRENT-STATE authority (``recruiter_candidates``)
    by resolving that person's merged current lifecycle across every brief/source
    they map to. Returns the number of NEW sightings recorded this call (0 on a
    re-resolution of an already-sighted brief — the steady state on repeated
    reads); the X fill runs every call (it is an idempotent upsert, not gated by
    the sighting ledger, because a person's current-state can change between
    reads even when the sighting itself is a no-op).

    The sighting's own ``lifecycle_state`` is recorded as ``""``: the per-brief
    person grain has no single lifecycle state (a person can map to candidates in
    different terminal states across sources), and ``brief_persons`` carries
    none. Refactor X is exactly the resolution of that — it does the wrong-grain-
    avoiding cross-DB join (``candidate_persons`` -> per-state-dir ``candidates``,
    most-recent merge) into a SEPARATE authority table, leaving the sighting
    accretion semantics untouched.

    ``state_root`` (additive) lets the X fill locate per-state-dir DBs; defaults
    to ``STATE_ROOT`` when omitted, mirroring the rest of the seam.

    Fail-soft: never raises. A failure mid-way may leave a partial set of
    sightings recorded, but the ledger makes that safe — the next read tops up
    the rest and never double-counts what already landed. The X fill is wrapped
    per person with the same posture: a fill failure logs + continues, never
    breaking the sighting or the read path.
    """

    try:
        person_ids = _resolved_person_ids(brief_id, identity_db_path)
    except Exception as exc:  # noqa: BLE001 — must never break the read path
        log.debug(
            "Recruiter sighting skipped, persons unreadable for brief %r (%s): %s",
            brief_id,
            type(exc).__name__,
            exc,
        )
        return 0

    if not person_ids:
        return 0

    try:
        recruiter_id = _resolve_recruiter_id(brief_id, recruiter_db_path)
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "Recruiter sighting skipped, recruiter unresolved for brief %r (%s): %s",
            brief_id,
            type(exc).__name__,
            exc,
        )
        return 0

    from shared.runtime_state.recruiter_store import RecruiterStore

    if recruiter_db_path is None:
        from shared.output_paths import resolve_recruiter_db_path

        recruiter_db_path = resolve_recruiter_db_path()

    from shared.runtime_state.recruiter_candidate_fill import fill_recruiter_candidate

    store = RecruiterStore(recruiter_db_path)
    recorded = 0
    for person_id in person_ids:
        try:
            if store.record_candidate_sighting_once(
                recruiter_id,
                person_id,
                brief_id=brief_id,
                lifecycle_state="",
            ):
                recorded += 1
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "Recruiter sighting failed for person %s on brief %r (%s): %s",
                person_id,
                brief_id,
                type(exc).__name__,
                exc,
            )

        # Reopen Refactor X: ALSO fill the current-state authority for this
        # person. Separate try/except from the sighting so a fill failure
        # never loses the sighting (and vice versa), and the fill runs even
        # when the sighting was a ledger no-op — current-state can move
        # between reads while the sighting stays recorded-once.
        try:
            fill_recruiter_candidate(
                recruiter_id,
                person_id,
                identity_db_path=identity_db_path,
                recruiter_db_path=recruiter_db_path,
                state_root=state_root,
            )
        except Exception as exc:  # noqa: BLE001 — never break the read path
            log.debug(
                "Recruiter candidate-fill failed for person %s on brief %r (%s): %s",
                person_id,
                brief_id,
                type(exc).__name__,
                exc,
            )
    return recorded
