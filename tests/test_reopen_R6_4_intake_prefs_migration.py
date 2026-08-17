"""R6.4 — migrate legacy intake-DB ``recruiter_meta_preferences`` -> taste signals.

Reopen recruiter learns-half. The pre-primitive learned calibration lived in the
SINGLE intake DB (``output/intake/intake_sessions.sqlite3``) under the meta key
``recruiter_meta_preferences`` — NOT in any per-state-dir ``runtime_state.sqlite3``.
``migrate_intake_prefs`` reads THAT meta key and writes one
``principle_feedback``/``intake_synthesis`` signal per bucket that crossed the
active threshold AND has a summary.

The four checklist cases:
- a seeded over-threshold bucket migrates into exactly one signal with the right
  kind/domain/payload, and the kind does NOT raise;
- a re-run is idempotent (still exactly one — fingerprint dedup);
- a sub-threshold (count<2) bucket migrates nothing;
- an empty/absent intake DB migrates zero with no error.
Plus: the legacy meta row is read-only (untouched after migration).
"""

from __future__ import annotations

from pathlib import Path

from shared.recruiter_overrides import (
    TASTE_SIGNAL_ACTIVE_THRESHOLD,
    get_recruiter_preferences,
    put_recruiter_preferences,
)
from shared.runtime_state.recruiter_store import (
    SIGNAL_PRINCIPLE_FEEDBACK,
    RecruiterStore,
)
from shared.runtime_state.store import RuntimeStateStore
from tools.backfill_recruiter_store import migrate_intake_prefs


def _recruiter_db(tmp_path: Path) -> tuple[Path, int]:
    db_path = tmp_path / "_recruiter" / "recruiter.sqlite3"
    store = RecruiterStore(db_path)
    rid = store.upsert_recruiter("operator@example.com", display_name="Sam")
    return db_path, rid


def _seed_intake_prefs(tmp_path: Path, body: dict) -> Path:
    """Write a ``recruiter_meta_preferences`` blob into a tmp intake DB.

    Uses the canonical meta-key writer (``put_recruiter_preferences``) over a
    ``RuntimeStateStore`` — the same write path the live intake reader reads.
    """

    intake_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    store = RuntimeStateStore(intake_db_path)
    put_recruiter_preferences(store, body)
    return intake_db_path


def test_threshold_assumption() -> None:
    """The migration gates on the same threshold the R6.1 reader activates at."""

    assert TASTE_SIGNAL_ACTIVE_THRESHOLD == 2


def test_migrates_over_threshold_bucket_once_and_is_idempotent(
    tmp_path: Path,
) -> None:
    db_path, rid = _recruiter_db(tmp_path)
    intake_db_path = _seed_intake_prefs(
        tmp_path,
        {
            "override_counts": {"role_title": 3},
            "summaries": {"role_title": "Prefer concise titles."},
            "active_voice": True,
        },
    )

    res = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res.intake_prefs_signals_written == 1
    assert res.intake_prefs_skipped_duplicate == 0

    store = RecruiterStore(db_path)
    signals = store.active_taste_signals(rid, domain="intake_synthesis")
    assert len(signals) == 1
    sig = signals[0]
    assert sig["signal_kind"] == SIGNAL_PRINCIPLE_FEEDBACK
    assert sig["domain"] == "intake_synthesis"
    assert sig["payload"] == {
        "bucket": "role_title",
        "summary": "Prefer concise titles.",
    }

    # Re-run: idempotent — still exactly one signal, the re-run wrote nothing.
    res2 = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res2.intake_prefs_signals_written == 0
    assert res2.intake_prefs_skipped_duplicate == 1
    assert (
        len(RecruiterStore(db_path).active_taste_signals(rid, domain="intake_synthesis"))
        == 1
    )


def test_sub_threshold_bucket_migrates_nothing(tmp_path: Path) -> None:
    db_path, rid = _recruiter_db(tmp_path)
    intake_db_path = _seed_intake_prefs(
        tmp_path,
        {
            # count below TASTE_SIGNAL_ACTIVE_THRESHOLD (2) -> not migrated.
            "override_counts": {"x": 1},
            "summaries": {"x": "Sub-threshold summary."},
        },
    )

    res = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res.intake_prefs_signals_written == 0
    assert res.intake_prefs_skipped_below_threshold == 1
    assert (
        RecruiterStore(db_path).active_taste_signals(rid, domain="intake_synthesis")
        == []
    )


def test_over_threshold_without_summary_migrates_nothing(tmp_path: Path) -> None:
    """A bucket at threshold but with no summary is not active_voice-eligible."""

    db_path, rid = _recruiter_db(tmp_path)
    intake_db_path = _seed_intake_prefs(
        tmp_path,
        {"override_counts": {"role_title": 5}, "summaries": {}},
    )
    res = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res.intake_prefs_signals_written == 0
    assert res.intake_prefs_skipped_no_summary == 1


def test_absent_intake_db_migrates_zero_no_error(tmp_path: Path) -> None:
    db_path, rid = _recruiter_db(tmp_path)
    # Path that does not exist yet — the store creates an empty meta table.
    intake_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    assert not intake_db_path.exists()

    res = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res.intake_prefs_signals_written == 0
    assert res.intake_prefs_buckets_seen == 0
    assert (
        RecruiterStore(db_path).active_taste_signals(rid, domain="intake_synthesis")
        == []
    )


def test_legacy_meta_row_untouched_append_only(tmp_path: Path) -> None:
    """The migration is append-only: the legacy meta blob is byte-identical after."""

    db_path, rid = _recruiter_db(tmp_path)
    body = {
        "override_counts": {"role_title": 3, "x": 1},
        "summaries": {"role_title": "Prefer concise titles."},
        "active_voice": True,
    }
    intake_db_path = _seed_intake_prefs(tmp_path, body)

    migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )

    after = get_recruiter_preferences(RuntimeStateStore(intake_db_path))
    assert after == body  # legacy row unchanged


# --- R6.4prime: the bucketless BARE top-level summary arm --------------------
#
# Persists the executor's claimed-but-uncommitted harness. The bare arm is
# UNGATED (no threshold) and migrates the legacy blob's top-level ``summary``
# into a BUCKETLESS {summary} signal — the shape R6.1prime's reader projects
# into the bare top-level line. Idempotent on IDENTICAL text only (H3): a
# divergent re-run APPENDS.


def test_bare_only_blob_migrates_one_bucketless_signal(tmp_path: Path) -> None:
    """A blob with only a bare summary (no over-threshold bucket) migrates 1.

    The bare arm is ungated, so it fires even with zero per-bucket signals; the
    new ``intake_prefs_bare_summary_*`` counters surface it in to_dict.
    """

    db_path, rid = _recruiter_db(tmp_path)
    intake_db_path = _seed_intake_prefs(tmp_path, {"summary": "Be concise."})

    res = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res.intake_prefs_bare_summary_seen == 1
    assert res.intake_prefs_bare_summary_written == 1
    assert res.intake_prefs_signals_written == 0  # no per-bucket arm fired
    # The 3 new counters are serialized in to_dict (operator-gate visibility).
    d = res.to_dict()
    assert d["intake_prefs_bare_summary_seen"] == 1
    assert d["intake_prefs_bare_summary_written"] == 1
    assert d["intake_prefs_bare_summary_skipped_duplicate"] == 0

    store = RecruiterStore(db_path)
    sigs = store.active_taste_signals(rid, domain="intake_synthesis")
    assert len(sigs) == 1
    assert sigs[0]["signal_kind"] == SIGNAL_PRINCIPLE_FEEDBACK
    assert sigs[0]["payload"] == {"summary": "Be concise."}  # BUCKETLESS


def test_bare_summary_identical_rerun_is_idempotent(tmp_path: Path) -> None:
    """Identical-text re-run: 0 written, 1 skipped_duplicate, still 1 row."""

    db_path, rid = _recruiter_db(tmp_path)
    intake_db_path = _seed_intake_prefs(tmp_path, {"summary": "Be concise."})

    migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    res2 = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res2.intake_prefs_bare_summary_written == 0
    assert res2.intake_prefs_bare_summary_skipped_duplicate == 1
    assert (
        len(
            RecruiterStore(db_path).active_taste_signals(
                rid, domain="intake_synthesis"
            )
        )
        == 1
    )


def test_bare_summary_divergent_rerun_appends(tmp_path: Path) -> None:
    """H3: a DIVERGENT bare summary on re-run APPENDS (different fingerprint).

    The migration dedups identical-text only; an operator-edited summary has a
    different payload JSON, so it is a fresh append — not deduped, not mutated.
    """

    db_path, rid = _recruiter_db(tmp_path)
    intake_db_path = _seed_intake_prefs(tmp_path, {"summary": "First take."})
    migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )

    # Operator edits the bare summary, then re-runs.
    _seed_intake_prefs(tmp_path, {"summary": "Second take."})
    res2 = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res2.intake_prefs_bare_summary_written == 1  # appended, not deduped
    sigs = RecruiterStore(db_path).active_taste_signals(
        rid, domain="intake_synthesis"
    )
    assert len(sigs) == 2  # both bare rows present (append-only)


def test_bare_and_over_threshold_bucket_migrate_two_distinct_signals(
    tmp_path: Path,
) -> None:
    """A blob with BOTH a bare summary and an over-threshold bucket migrates 2.

    The two payload shapes are disjoint: {summary} (bucketless) and
    {bucket, summary}. Different fingerprints, both arms fire, no dedup collision.
    """

    db_path, rid = _recruiter_db(tmp_path)
    intake_db_path = _seed_intake_prefs(
        tmp_path,
        {
            "override_counts": {"role_title": 3},
            "summaries": {"role_title": "Prefer concise titles."},
            "summary": "Be concise.",
            "active_voice": True,
        },
    )
    res = migrate_intake_prefs(
        intake_db_path=intake_db_path, db_path=db_path, recruiter_id=rid
    )
    assert res.intake_prefs_signals_written == 1  # the per-bucket arm
    assert res.intake_prefs_bare_summary_written == 1  # the bare arm
    sigs = RecruiterStore(db_path).active_taste_signals(
        rid, domain="intake_synthesis"
    )
    payloads = sorted((s["payload"] for s in sigs), key=lambda p: "bucket" in p)
    assert payloads == [
        {"summary": "Be concise."},
        {"bucket": "role_title", "summary": "Prefer concise titles."},
    ]
