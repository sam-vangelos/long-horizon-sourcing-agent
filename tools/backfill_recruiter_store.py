"""Idempotent backfill for the recruiter store (reopen Stage 2, Part VIII).

Three independent, idempotent backfills:

1. ``backfill_recruiter_briefs`` — enumerate authored briefs and record
   recruiter→brief membership for the Stage-1 recruiter. Uses the SAME
   brief discovery + ``derive_brief_id`` machinery the launch path uses
   (``cloris.api.briefs._scan_authored_briefs`` + ``derive_brief_id``), so
   backfilled membership keys match the ids ``_link_recruiter_brief_fail_soft``
   writes at launch. Idempotent via ``link_brief``'s ``INSERT OR IGNORE``.

2. ``replay_write_intentions`` — scan every per-state-dir
   ``runtime_state.sqlite3``, find committed write-intentions that never
   completed their fail-soft recruiter-store mirror (``completed_at IS
   NULL``), replay each into ``recruiter_taste_signals``, and mark it
   complete. Idempotent: completed intentions are never re-read, and the
   replay dedups against already-active signals by (signal_kind, domain,
   payload) so a crash between the recruiter write and the
   mark-complete doesn't double-write on the next run.

3. ``backfill_recruiter_candidates`` (reopen Refactor X.5) — whole-population
   backfill of the ``recruiter_candidates`` CURRENT-STATE authority for
   already-sighted persons. The per-(recruiter, person) sighting hook
   (Stage 3a) fills the authority going forward, but persons sighted BEFORE
   Refactor X landed have a ``recruiter_candidate_history`` accretion row and
   no authority row. This arm closes that gap: for each recruiter, enumerate
   the persons it has actually been shown (``SELECT DISTINCT person_id FROM
   recruiter_candidate_history``) and run the EXISTING per-person primitive
   ``recruiter_candidate_fill.fill_recruiter_candidate`` over each — same
   cross-DB join + idempotent upsert the hook uses, no new join. Convergence
   proof: after the backfill, ``divergence_report(recruiter).missing_in_authority``
   is 0 (every sighted person now has an authority row OR was a legitimate
   skip — a person with NO live candidate row, which the fill returns False
   for; those are NOT missing-corruption, they genuinely have no current-state
   and the divergence metric counts them in no bucket). Idempotent on the
   ``(recruiter_id, person_id)`` PK; re-running re-derives identical rows.
   Touches ONLY ``recruiter_candidates`` — never the accretion log's
   ``times_surfaced``, never the brief-keyed candidates path.

4. ``migrate_intake_prefs`` (reopen recruiter learns-half, R6.4) — one-time
   migration of the legacy intake-DB ``recruiter_meta_preferences`` blob into
   durable ``recruiter_taste_signals``. The pre-primitive learned calibration
   lived in the SINGLE intake DB (``output/intake/intake_sessions.sqlite3``)
   meta key, NOT in any per-state-dir DB; each bucket that crossed the active
   threshold AND has a summary becomes one ``principle_feedback`` /
   ``intake_synthesis`` signal. Append-only (the legacy meta row is read-only
   here) and idempotent via a ``(signal_kind, domain, payload)`` fingerprint, so
   a re-run writes nothing. Operator-invoked, not auto-run.

Run all backfills:  python -m tools.backfill_recruiter_store
Run one:            python -m tools.backfill_recruiter_store --only briefs
                    python -m tools.backfill_recruiter_store --only intentions
                    python -m tools.backfill_recruiter_store --only candidates
                    python -m tools.backfill_recruiter_store --only intake-prefs
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class BackfillResult:
    """Counts from a backfill run (for logging + the idempotency test)."""

    briefs_linked: int = 0
    briefs_seen: int = 0
    intentions_replayed: int = 0
    intentions_seen: int = 0
    intentions_skipped_duplicate: int = 0
    # Refactor X.5 — candidate authority backfill.
    candidate_recruiters_processed: int = 0
    candidate_persons_seen: int = 0
    candidate_authority_filled: int = 0
    candidate_skipped_no_current_state: int = 0
    candidate_errors: int = 0
    # The convergence proof: divergence_report per recruiter, post-backfill.
    candidate_divergence: list[dict] = field(default_factory=list)
    # Reopen recruiter learns-half, R6.4 — legacy intake-prefs migration.
    intake_prefs_buckets_seen: int = 0
    intake_prefs_signals_written: int = 0
    intake_prefs_skipped_below_threshold: int = 0
    intake_prefs_skipped_no_summary: int = 0
    intake_prefs_skipped_duplicate: int = 0
    # Reopen recruiter learns-half, R6.4prime — the bucketless BARE top-level
    # summary arm (distinct from the per-bucket counters above): seen when the
    # blob carries a non-empty top-level ``summary``, written when migrated,
    # skipped_duplicate when the identical-text fingerprint already exists.
    intake_prefs_bare_summary_seen: int = 0
    intake_prefs_bare_summary_written: int = 0
    intake_prefs_bare_summary_skipped_duplicate: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "briefs_linked": self.briefs_linked,
            "briefs_seen": self.briefs_seen,
            "intentions_replayed": self.intentions_replayed,
            "intentions_seen": self.intentions_seen,
            "intentions_skipped_duplicate": self.intentions_skipped_duplicate,
            "candidate_recruiters_processed": self.candidate_recruiters_processed,
            "candidate_persons_seen": self.candidate_persons_seen,
            "candidate_authority_filled": self.candidate_authority_filled,
            "candidate_skipped_no_current_state": (
                self.candidate_skipped_no_current_state
            ),
            "candidate_errors": self.candidate_errors,
            "candidate_divergence": list(self.candidate_divergence),
            "intake_prefs_buckets_seen": self.intake_prefs_buckets_seen,
            "intake_prefs_signals_written": self.intake_prefs_signals_written,
            "intake_prefs_skipped_below_threshold": (
                self.intake_prefs_skipped_below_threshold
            ),
            "intake_prefs_skipped_no_summary": self.intake_prefs_skipped_no_summary,
            "intake_prefs_skipped_duplicate": self.intake_prefs_skipped_duplicate,
            "intake_prefs_bare_summary_seen": self.intake_prefs_bare_summary_seen,
            "intake_prefs_bare_summary_written": (
                self.intake_prefs_bare_summary_written
            ),
            "intake_prefs_bare_summary_skipped_duplicate": (
                self.intake_prefs_bare_summary_skipped_duplicate
            ),
            "notes": list(self.notes),
        }


def _resolve_default_db_path() -> Path:
    from shared.output_paths import resolve_recruiter_db_path

    return resolve_recruiter_db_path()


def _resolve_default_config_dir() -> Path:
    from cloris.api import _paths

    return _paths._CONFIG_DIR


def _resolve_default_state_root() -> Path:
    from shared.output_paths import STATE_ROOT

    return STATE_ROOT


def _ensure_recruiter(store, recruiter_id: int | None) -> int:
    """Return the recruiter id to backfill under.

    When ``recruiter_id`` is None, resolve through the Stage-2 seam
    (``get_current_recruiter_id``), which get-or-creates the Stage-1
    recruiter. When an explicit id is given (tests), the caller owns
    provisioning the row.
    """

    if recruiter_id is not None:
        return recruiter_id
    from shared.recruiter_context import get_current_recruiter_id

    return get_current_recruiter_id()


def backfill_recruiter_briefs(
    *,
    config_dir: Path | None = None,
    db_path: Path | None = None,
    recruiter_id: int | None = None,
    result: BackfillResult | None = None,
) -> BackfillResult:
    """Link every authored brief to the recruiter. Idempotent."""

    from cloris.api.briefs import _scan_authored_briefs
    from cloris.api import _paths
    from shared.output_paths import derive_brief_id
    from shared.runtime_state.recruiter_store import RecruiterStore

    result = result or BackfillResult()
    config_dir = config_dir or _resolve_default_config_dir()
    store = RecruiterStore(db_path or _resolve_default_db_path())
    rid = _ensure_recruiter(store, recruiter_id)

    existing_before = set(store.briefs_for_recruiter(rid))

    for info in _scan_authored_briefs(config_dir):
        abs_path = _paths._CONFIG_PARENT / info.path
        try:
            brief_id = derive_brief_id(brief_path=str(abs_path))
        except Exception:  # noqa: BLE001 — a malformed brief shouldn't abort the rest
            result.notes.append(f"skipped unparseable brief at {info.path}")
            continue
        result.briefs_seen += 1
        store.link_brief(rid, brief_id)  # INSERT OR IGNORE — idempotent
        if brief_id not in existing_before:
            result.briefs_linked += 1
            existing_before.add(brief_id)

    return result


def replay_write_intentions(
    *,
    state_root: Path | None = None,
    db_path: Path | None = None,
    recruiter_id: int | None = None,
    result: BackfillResult | None = None,
) -> BackfillResult:
    """Replay incomplete per-state-dir write-intentions into the store.

    Idempotent: an intention is only read while ``completed_at IS NULL``,
    and the recruiter write is deduped against active signals by
    (signal_kind, domain, payload) so a re-run after a partial failure
    doesn't double-write.
    """

    from cloris.control_plane import enumerate_state_dirs
    from shared.runtime_state.recruiter_store import RecruiterStore
    from shared.runtime_state.store import RuntimeStateStore

    result = result or BackfillResult()
    state_root = state_root or _resolve_default_state_root()
    recruiter_store = RecruiterStore(db_path or _resolve_default_db_path())
    rid = _ensure_recruiter(recruiter_store, recruiter_id)

    # Pre-index active signals per domain so dedup is O(1) per intention.
    active_index: dict[str, set[str]] = {}

    def _signal_fingerprint(signal_kind: str, domain: str, payload: dict) -> str:
        return json.dumps(
            {"signal_kind": signal_kind, "domain": domain, "payload": payload},
            sort_keys=True,
        )

    def _active_fingerprints(domain: str) -> set[str]:
        if domain not in active_index:
            active_index[domain] = {
                _signal_fingerprint(
                    str(sig.get("signal_kind", "")),
                    domain,
                    sig.get("payload") if isinstance(sig.get("payload"), dict) else {},
                )
                for sig in recruiter_store.active_taste_signals(rid, domain=domain)
            }
        return active_index[domain]

    for _source, state_dir in enumerate_state_dirs(state_root):
        per_state_db = state_dir / "runtime_state.sqlite3"
        if not per_state_db.exists():
            continue
        store = RuntimeStateStore(per_state_db)
        for intention in store.list_incomplete_write_intentions():
            result.intentions_seen += 1
            signal_kind = str(intention.get("signal_kind", ""))
            domain = str(intention.get("domain", ""))
            payload = (
                intention.get("payload")
                if isinstance(intention.get("payload"), dict)
                else {}
            )
            fingerprint = _signal_fingerprint(signal_kind, domain, payload)
            seen = _active_fingerprints(domain)
            if fingerprint in seen:
                # Already mirrored (a prior partial run wrote the signal but
                # crashed before marking complete) — just mark complete now.
                store.mark_write_intention_complete(int(intention["id"]))
                result.intentions_skipped_duplicate += 1
                continue
            try:
                recruiter_store.record_taste_signal(
                    rid,
                    signal_kind=signal_kind,
                    domain=domain,
                    payload=payload,
                    source_brief_id=intention.get("source_brief_id"),
                    confidence=float(intention.get("confidence", 0.5)),
                )
            except Exception as exc:  # noqa: BLE001 — leave incomplete for next run
                result.notes.append(
                    f"replay failed for intention {intention.get('id')}: {exc}"
                )
                continue
            seen.add(fingerprint)
            store.mark_write_intention_complete(int(intention["id"]))
            result.intentions_replayed += 1

    return result


def backfill_recruiter_candidates(
    recruiter_id: int | None = None,
    *,
    recruiter_db_path: Path | None = None,
    identity_db_path: Path | None = None,
    state_root: Path | None = None,
    result: BackfillResult | None = None,
) -> BackfillResult:
    """Whole-population backfill of the ``recruiter_candidates`` authority.

    Reopen Refactor X.5. For each recruiter, enumerate the persons it has been
    SHOWN (``recruiter_candidate_history`` — the sighting accretion) and run the
    EXISTING per-person primitive
    :func:`recruiter_candidate_fill.fill_recruiter_candidate` over each. That
    primitive resolves the person's merged current-state across every
    brief/source they map to (the cross-DB join) and idempotently upserts the
    authority row; it returns ``True`` when a row was written and ``False`` when
    the person has no live candidate row (a legitimate skip — there is no
    current-state to materialize). No new cross-DB join is introduced here.

    Recruiter scope: an explicit ``recruiter_id`` processes just that one; ``None``
    processes EVERY recruiter in the ``recruiters`` table. (This deliberately does
    NOT route through ``_ensure_recruiter`` like the briefs/intentions arms — a
    whole-population backfill must never get-or-CREATE a recruiter; it operates on
    recruiters that already exist.)

    Fail-soft per person: one person's fill raising must not abort the rest, so
    each call is guarded; a raise is tallied as an error and the walk continues.

    Convergence proof: after a recruiter's persons are processed,
    :func:`recruiter_candidate_fill.divergence_report` is run and its counts are
    appended to ``result.candidate_divergence``. ``missing_in_authority`` is the
    gate — it should be 0, because the divergence metric's "missing" bucket is
    exactly "person has a live brief-keyed current-state but no authority row,"
    and this backfill writes an authority row for every such person. A person the
    fill skipped (no live candidate row) lands in NO divergence bucket, so a clean
    backfill yields ``missing_in_authority == 0`` with no caveat; any nonzero is a
    real bug.

    Idempotent: keyed on the ``(recruiter_id, person_id)`` PK via
    ``upsert_recruiter_candidate`` — re-running re-derives identical rows. Touches
    ONLY ``recruiter_candidates``; never the accretion log's ``times_surfaced``,
    never the brief-keyed candidates path.
    """

    from shared.runtime_state.recruiter_candidate_fill import (
        divergence_report,
        fill_recruiter_candidate,
    )
    from shared.runtime_state.recruiter_store import RecruiterStore

    result = result or BackfillResult()
    recruiter_db_path = recruiter_db_path or _resolve_default_db_path()
    store = RecruiterStore(recruiter_db_path)

    # Resolve the recruiter scope. Explicit id -> just it; None -> every
    # recruiter that already exists (no get-or-create, unlike the other arms).
    if recruiter_id is not None:
        recruiter_ids = [recruiter_id]
    else:
        with store.connect() as conn:
            recruiter_ids = [
                int(r["id"])
                for r in conn.execute(
                    "SELECT id FROM recruiters ORDER BY id"
                ).fetchall()
            ]

    for rid in recruiter_ids:
        result.candidate_recruiters_processed += 1

        # The population is every person this recruiter has been SHOWN — the
        # sighting accretion, NOT the authority (which is what we're filling).
        with store.connect() as conn:
            person_ids = [
                int(r["person_id"])
                for r in conn.execute(
                    "SELECT DISTINCT person_id FROM recruiter_candidate_history "
                    "WHERE recruiter_id = ? ORDER BY person_id",
                    (rid,),
                ).fetchall()
            ]

        for person_id in person_ids:
            result.candidate_persons_seen += 1
            try:
                wrote = fill_recruiter_candidate(
                    rid,
                    person_id,
                    identity_db_path=identity_db_path,
                    recruiter_db_path=recruiter_db_path,
                    state_root=state_root,
                )
            except Exception as exc:  # noqa: BLE001 — one person must not sink the run
                result.candidate_errors += 1
                result.notes.append(
                    f"candidate fill failed for recruiter {rid} person "
                    f"{person_id}: {exc}"
                )
                log.warning(
                    "candidate fill failed for recruiter %s person %s: %s",
                    rid,
                    person_id,
                    exc,
                )
                continue
            if wrote:
                result.candidate_authority_filled += 1
            else:
                # No live candidate row — a legitimate skip, NOT missing-corruption.
                result.candidate_skipped_no_current_state += 1

        # Convergence proof for this recruiter (missing_in_authority should be 0).
        report = divergence_report(
            rid,
            identity_db_path=identity_db_path,
            recruiter_db_path=recruiter_db_path,
            state_root=state_root,
        )
        result.candidate_divergence.append(report.to_dict())

    return result


def migrate_intake_prefs(
    *,
    intake_db_path: Path | None = None,
    db_path: Path | None = None,
    recruiter_id: int | None = None,
    result: BackfillResult | None = None,
) -> BackfillResult:
    """Migrate legacy intake-DB ``recruiter_meta_preferences`` into taste signals.

    Reopen recruiter learns-half, R6.4. Before the recruiter primitive existed,
    Cloris's only learned calibration lived in the SINGLE intake DB
    (``output/intake/intake_sessions.sqlite3``) under the meta key
    ``recruiter_meta_preferences`` — NOT in any per-state-dir
    ``runtime_state.sqlite3``. That prefs blob has the shape::

        {"override_counts": {bucket: int}, "summaries": {bucket: text},
         "active_voice": bool}

    For each bucket that has crossed the active threshold
    (``override_counts[bucket] >= TASTE_SIGNAL_ACTIVE_THRESHOLD``) AND carries a
    non-empty ``summaries[bucket]`` — i.e. exactly the buckets the legacy reader
    would emit a line for — this records one durable taste signal::

        record_taste_signal(rid, signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
                             domain="intake_synthesis",
                             payload={"bucket": bucket, "summary": summary})

    ``signal_kind`` is ``principle_feedback`` (a member of ``KNOWN_SIGNAL_KINDS``);
    ``intake_synthesis`` is the DOMAIN, never the kind (passing it as the kind
    raises ``ValueError``).

    R6.4prime — SCOPE GAP CLOSED. Beyond the per-bucket arm above, this also
    migrates the legacy blob's BARE top-level ``summary`` (the ungated line the
    live reader emits via ``add(global_prefs.get("summary"))``, which earlier
    R6.4 left behind) into a BUCKETLESS ``{"summary": text}`` signal — the shape
    R6.1prime's reader projects into its bare top-level line. The bare arm is
    UNGATED (no threshold), so a blob with only a bare summary and no
    over-threshold bucket still migrates exactly one signal. Counted under the
    ``intake_prefs_bare_summary_*`` fields.

    Idempotent: deduped by a ``(signal_kind, domain, payload)`` fingerprint
    against the recruiter's already-active signals (the same dedup the
    write-intentions replay uses), so a re-run with the UNCHANGED blob writes
    nothing new — both arms share the one ``active_fingerprints`` set. A
    DIVERGENT bare summary (operator edited it before re-running) has a different
    fingerprint and APPENDS a fresh signal (the path is append-only +
    soft-superseded, never deduped across text changes); the reader's
    last-writer-wins then surfaces the newest. The migration is one-time, so the
    steady state is the identical-text no-op.

    Append-only: this NEVER mutates the legacy meta row — the intake DB blob is
    read-only here. Re-reading it after migration yields the identical blob.

    Empty / absent intake DB: opening a ``RuntimeStateStore`` over a missing path
    creates an empty ``meta`` table, so the meta read returns ``None`` and zero
    signals migrate with no error.

    Operator-invoked, not auto-run; reached via
    ``python -m tools.backfill_recruiter_store --only intake-prefs``.
    """

    from shared.output_paths import resolve_intake_db_path
    from shared.recruiter_overrides import (
        TASTE_SIGNAL_ACTIVE_THRESHOLD,
        _read_meta,
    )
    from shared.runtime_state.recruiter_store import (
        SIGNAL_PRINCIPLE_FEEDBACK,
        RecruiterStore,
    )
    from shared.runtime_state.store import RuntimeStateStore

    result = result or BackfillResult()
    recruiter_store = RecruiterStore(db_path or _resolve_default_db_path())
    rid = _ensure_recruiter(recruiter_store, recruiter_id)

    domain = "intake_synthesis"

    def _signal_fingerprint(payload: dict) -> str:
        return json.dumps(
            {
                "signal_kind": SIGNAL_PRINCIPLE_FEEDBACK,
                "domain": domain,
                "payload": payload,
            },
            sort_keys=True,
        )

    # Pre-index active principle_feedback signals in this domain so dedup is O(1)
    # per bucket and a re-run is a no-op.
    active_fingerprints = {
        _signal_fingerprint(
            sig.get("payload") if isinstance(sig.get("payload"), dict) else {}
        )
        for sig in recruiter_store.active_taste_signals(rid, domain=domain)
        if sig.get("signal_kind") == SIGNAL_PRINCIPLE_FEEDBACK
    }

    # Read the SINGLE intake DB meta key. Constructing the store over an absent
    # path creates an empty meta table; the read then returns None -> zero work.
    intake_store = RuntimeStateStore(intake_db_path or resolve_intake_db_path())
    prefs = _read_meta(intake_store, "recruiter_meta_preferences")
    if not isinstance(prefs, dict):
        return result

    counts = prefs.get("override_counts")
    counts = counts if isinstance(counts, dict) else {}
    summaries = prefs.get("summaries")
    summaries = summaries if isinstance(summaries, dict) else {}

    for bucket in sorted(counts):
        result.intake_prefs_buckets_seen += 1
        try:
            count = int(counts.get(bucket, 0))
        except (TypeError, ValueError):
            count = 0
        if count < TASTE_SIGNAL_ACTIVE_THRESHOLD:
            result.intake_prefs_skipped_below_threshold += 1
            continue
        summary = summaries.get(bucket)
        if not isinstance(summary, str) or not summary.strip():
            result.intake_prefs_skipped_no_summary += 1
            continue
        payload = {"bucket": bucket, "summary": summary.strip()}
        fingerprint = _signal_fingerprint(payload)
        if fingerprint in active_fingerprints:
            result.intake_prefs_skipped_duplicate += 1
            continue
        try:
            recruiter_store.record_taste_signal(
                rid,
                signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
                domain=domain,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001 — one bucket must not sink the run
            result.notes.append(f"intake-prefs migrate failed for {bucket!r}: {exc}")
            continue
        active_fingerprints.add(fingerprint)
        result.intake_prefs_signals_written += 1

    # Reopen recruiter learns-half, R6.4prime. Migrate the legacy blob's BARE
    # top-level ``summary`` (the ungated line the live reader emits via
    # ``add(global_prefs.get("summary"))``, distinct from the per-bucket
    # ``summaries`` map handled by the loop above) into a BUCKETLESS
    # ``{"summary": text}`` principle_feedback signal — the shape R6.1prime's
    # reader projects into its bare top-level line. UNGATED: unlike the per-bucket
    # arm there is no threshold for the bare summary (the live reader emits it
    # outside the active_voice gate), so a blob carrying only a bare summary and
    # no over-threshold bucket still migrates exactly this one signal.
    #
    # Idempotent on IDENTICAL text via the SAME ``(signal_kind, domain,
    # payload)`` fingerprint + ``active_fingerprints`` set the per-bucket arm
    # uses — a re-run with the unchanged blob writes nothing and counts one
    # skipped_duplicate. A DIVERGENT bare summary (the operator edited it before
    # re-running) has a different fingerprint and APPENDS a fresh signal (H3 —
    # the forward/migration path is append-only + soft-superseded, never deduped
    # across text changes; the reader's last-writer-wins ORDER BY created_at then
    # surfaces the newest). The migration is one-time/operator-invoked, so the
    # expected steady state is the identical-text no-op.
    bare = prefs.get("summary")
    if isinstance(bare, str) and bare.strip():
        result.intake_prefs_bare_summary_seen += 1
        bare_payload = {"summary": bare.strip()}
        bare_fingerprint = _signal_fingerprint(bare_payload)
        if bare_fingerprint in active_fingerprints:
            result.intake_prefs_bare_summary_skipped_duplicate += 1
        else:
            try:
                recruiter_store.record_taste_signal(
                    rid,
                    signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
                    domain=domain,
                    payload=bare_payload,
                )
            except Exception as exc:  # noqa: BLE001 — bare summary must not sink the run
                result.notes.append(f"intake-prefs migrate failed for bare summary: {exc}")
            else:
                active_fingerprints.add(bare_fingerprint)
                result.intake_prefs_bare_summary_written += 1

    return result


def run_backfill(
    *,
    only: str | None = None,
    config_dir: Path | None = None,
    state_root: Path | None = None,
    db_path: Path | None = None,
    recruiter_id: int | None = None,
    intake_db_path: Path | None = None,
) -> BackfillResult:
    """Run all backfills (or one, when ``only`` is set). Idempotent."""

    result = BackfillResult()
    if only in (None, "briefs"):
        backfill_recruiter_briefs(
            config_dir=config_dir,
            db_path=db_path,
            recruiter_id=recruiter_id,
            result=result,
        )
    if only in (None, "intentions"):
        replay_write_intentions(
            state_root=state_root,
            db_path=db_path,
            recruiter_id=recruiter_id,
            result=result,
        )
    if only in (None, "candidates"):
        backfill_recruiter_candidates(
            recruiter_id,
            recruiter_db_path=db_path,
            state_root=state_root,
            result=result,
        )
    if only in (None, "intake-prefs"):
        migrate_intake_prefs(
            intake_db_path=intake_db_path,
            db_path=db_path,
            recruiter_id=recruiter_id,
            result=result,
        )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Idempotent backfill for the recruiter store: recruiter→brief "
            "membership + write-intentions replay + candidate current-state "
            "authority (reopen Stage 2 + Refactor X.5)."
        )
    )
    parser.add_argument(
        "--only",
        choices=("briefs", "intentions", "candidates", "intake-prefs"),
        default=None,
        help="Run only one backfill (default: all four).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_backfill(only=args.only)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
