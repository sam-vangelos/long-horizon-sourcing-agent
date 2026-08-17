from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def _cloris_models_importable() -> bool:
    """Whether ``cloris.models`` imports in this interpreter.

    The model module uses PEP-604 ``X | None`` annotations that Pydantic must
    eval at import; on Python < 3.10 without ``eval_type_backport`` that raises
    ``TypeError`` (not ``ImportError``), so ``pytest.importorskip`` can't catch
    it. The PUT-endpoint tests below need the request model, so they skip when
    it can't import — the same env-tolerance the repo already applies to optional
    deps. They run green on 3.10+ / CI.
    """

    try:
        import cloris.models  # noqa: F401

        return True
    except (ImportError, TypeError):
        return False


_NEEDS_MODELS = pytest.mark.skipif(
    not _cloris_models_importable(),
    reason="cloris.models not importable in this interpreter (PEP-604 eval on <3.10)",
)

from shared.recruiter_overrides import (
    OVERRIDE_SUMMARY_THRESHOLD,
    get_recruiter_preferences,
    record_override_for_field_path,
    recruiter_taste_signals_for_extract,
    recruiter_voice_line_for_extract,
)
from shared.runtime_state.recruiter_store import (
    SIGNAL_ARCHETYPE_PREFERENCE,
    SIGNAL_PRINCIPLE_FEEDBACK,
    RecruiterStore,
)
from shared.runtime_state.store import RuntimeStateStore


def _recruiter_store(tmp_path: Path) -> tuple[RecruiterStore, int]:
    store = RecruiterStore(tmp_path / "_recruiter" / "recruiter.sqlite3")
    rid = store.upsert_recruiter("operator@example.com", display_name="Sam")
    return store, rid


def test_record_override_counts_by_field_bucket(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    record_override_for_field_path(store, "capability_areas[0].description")
    record_override_for_field_path(store, "capability_areas[1].name")
    prefs = get_recruiter_preferences(store)
    assert prefs["override_counts"]["capability_areas"] == 2


def test_override_threshold_refreshes_summary(tmp_path: Path) -> None:
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    with patch("shared.recruiter_overrides._summarize_bucket", return_value="Prefer sharper capability language."):
        for _ in range(OVERRIDE_SUMMARY_THRESHOLD):
            record_override_for_field_path(store, "capability_areas[0].description")
    line = recruiter_voice_line_for_extract(store, {})
    assert "Prefer sharper capability language." in line


# --- R6.1: the dark reader sourced from durable taste signals ----------------


def test_taste_signals_for_extract_projects_and_activates(tmp_path: Path) -> None:
    """A single active signal must yield a NON-empty ``bucket: summary`` line.

    Guards the empty-output trap: taste-signal payloads carry no ``summaries``
    map and no ``active_voice`` flag, so a reader that didn't project + synthesize
    the gate would return "" forever. Also proves ``principle_feedback`` is a
    valid kind (the record call must not raise).
    """

    store, rid = _recruiter_store(tmp_path)
    sid = store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"bucket": "role_title", "summary": "Prefer concise titles."},
    )
    assert isinstance(sid, int)  # the kind did not raise

    line = recruiter_taste_signals_for_extract(store, rid)
    assert line  # NON-empty: the projection + active_voice synthesis fired
    assert "role_title: Prefer concise titles." in line


def test_taste_signals_for_extract_empty_when_no_signals(tmp_path: Path) -> None:
    """No active signal -> "" (no spurious activation), matching the meta reader."""

    store, rid = _recruiter_store(tmp_path)
    assert recruiter_taste_signals_for_extract(store, rid) == ""


def test_taste_signals_for_extract_merges_session_prefs(tmp_path: Path) -> None:
    """The session_state meta_preferences merge is preserved like the live reader."""

    store, rid = _recruiter_store(tmp_path)
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"bucket": "role_title", "summary": "Prefer concise titles."},
    )
    session_state = {
        "meta_preferences": {
            "summary": "Lead with seniority.",
            "summaries": {"seniority": "Name the level explicitly."},
            "active_voice": True,
        }
    }
    line = recruiter_taste_signals_for_extract(store, rid, session_state)
    assert "Lead with seniority." in line  # session bare summary
    assert "seniority: Name the level explicitly." in line  # session bucket
    assert "role_title: Prefer concise titles." in line  # global (signal) bucket


def test_taste_signals_for_extract_session_active_voice_false_gates_buckets(
    tmp_path: Path,
) -> None:
    """A session that turns active_voice off suppresses bucket lines, as the live
    reader does (the gate is ``global_active and session_active``)."""

    store, rid = _recruiter_store(tmp_path)
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"bucket": "role_title", "summary": "Prefer concise titles."},
    )
    session_state = {"meta_preferences": {"active_voice": False}}
    line = recruiter_taste_signals_for_extract(store, rid, session_state)
    assert "role_title:" not in line  # bucket lines gated off
    assert line == ""


def test_taste_signals_for_extract_ignores_other_domains_and_kinds(
    tmp_path: Path,
) -> None:
    """Only ``principle_feedback`` signals in the ``intake_synthesis`` domain feed
    the reader; a different domain or a different kind contributes nothing."""

    store, rid = _recruiter_store(tmp_path)
    # Wrong domain.
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="candidate_review",
        payload={"bucket": "role_title", "summary": "From another domain."},
    )
    # Right domain, wrong kind.
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_ARCHETYPE_PREFERENCE,
        domain="intake_synthesis",
        payload={"bucket": "archetype", "summary": "From another kind."},
    )
    assert recruiter_taste_signals_for_extract(store, rid) == ""


# --- R6.1prime: the bucketless BARE top-level summary line -------------------
#
# Persists the executor's claimed-but-uncommitted harness. The dark reader is
# wired NOWHERE in prod (recruiter_voice_line_for_extract is still the live
# reader at intake.py:297 / intake_synthesis.py); these guard the reader's
# behavior for the future R6.3 repoint. H1 is the load-bearing hazard: the
# bucketless branch must sit BEFORE the bucket guard, else a bare-only recruiter
# emits "" post-flip.


def test_taste_signals_bucketless_only_emits_naked_line_active_voice_false(
    tmp_path: Path,
) -> None:
    """A bucketless ``{"summary": text}`` signal must emit the NAKED line.

    The H1 regression this guards: if the bucketless branch were placed AFTER the
    bucket guard (``if not isinstance(bucket, str)...continue``), a bucketless
    payload (bucket=None) would hit the guard and ``continue`` — the branch would
    be DEAD and this reader would return "" for a bare-only recruiter. Placed
    before the guard, the naked summary is captured into ``global_bare_summary``
    and emitted UNGATED (no ``active_voice`` required, no bucket prefix).
    """

    store, rid = _recruiter_store(tmp_path)
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"summary": "Be concise."},  # NO bucket key
    )
    line = recruiter_taste_signals_for_extract(store, rid)
    assert line == "Be concise."  # naked, non-empty
    assert ":" not in line  # no bucket prefix -> active_voice gate did NOT fire


def test_taste_signals_bare_and_bucket_emits_naked_line_first(
    tmp_path: Path,
) -> None:
    """Bare summary + per-bucket signal -> naked line FIRST, then the bucket line.

    Mirrors the live reader, which emits ``add(global_prefs.get("summary"))``
    before the gated per-bucket loop. The two payload shapes are disjoint (a
    bucketless payload has no bucket key; a bucketed one always does), so no
    double-emit.
    """

    store, rid = _recruiter_store(tmp_path)
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"summary": "Be concise."},
    )
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"bucket": "role_title", "summary": "Prefer concise titles."},
    )
    line = recruiter_taste_signals_for_extract(store, rid)
    assert line == "Be concise.\nrole_title: Prefer concise titles."


def test_taste_signals_two_divergent_bare_summaries_last_writer_wins(
    tmp_path: Path,
) -> None:
    """Two bucketless signals -> the freshest (last by created_at) wins.

    ``active_taste_signals`` is ORDER BY created_at, and the bucketless branch
    overwrites ``global_bare_summary`` on each iteration, so the newest naked
    summary lands. Both texts share no bucket, so neither is gated.
    """

    store, rid = _recruiter_store(tmp_path)
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"summary": "First take."},
    )
    store.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"summary": "Second take."},
    )
    line = recruiter_taste_signals_for_extract(store, rid)
    assert line == "Second take."  # last-writer-wins


# --- R6.2prime: the DARK forward writes (record_override + PUT endpoint) ------
#
# H2 (load-bearing): the fail-soft try/except MUST wrap the FULL RecruiterStore
# construction (it runs mkdir+DDL and can raise BEFORE record_taste_signal). An
# unwrapped raise would propagate past the meta write, skipping the summary
# refresh + count increment. H3: the forward write is APPEND-ONLY (bare INSERT,
# no dedup).


def _signals(store: RecruiterStore, rid: int) -> list[dict]:
    return store.active_taste_signals(rid, domain="intake_synthesis")


def test_record_override_sub_threshold_writes_no_spine_signal(
    tmp_path: Path, monkeypatch
) -> None:
    """One edit (below the 2-event threshold) -> ZERO durable spine signals."""

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    rstore, rid = _recruiter_store(tmp_path)
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )
    with patch(
        "shared.recruiter_overrides._summarize_bucket", return_value="S."
    ):
        record_override_for_field_path(store, "role_title.name")  # 1 edit
    assert _signals(rstore, rid) == []  # sub-threshold: no spine row


def test_record_override_threshold_writes_one_bucket_signal(
    tmp_path: Path, monkeypatch
) -> None:
    """The 2nd edit crosses threshold -> exactly ONE {bucket, summary} signal."""

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    rstore, rid = _recruiter_store(tmp_path)
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )
    with patch(
        "shared.recruiter_overrides._summarize_bucket",
        return_value="Prefer concise titles.",
    ):
        for _ in range(OVERRIDE_SUMMARY_THRESHOLD):
            record_override_for_field_path(store, "role_title.name")
    sigs = _signals(rstore, rid)
    assert len(sigs) == 1
    assert sigs[0]["signal_kind"] == SIGNAL_PRINCIPLE_FEEDBACK
    assert sigs[0]["domain"] == "intake_synthesis"
    assert sigs[0]["payload"] == {
        "bucket": "role_title",
        "summary": "Prefer concise titles.",
    }


def test_record_override_forward_write_is_append_only_no_dedup(
    tmp_path: Path, monkeypatch
) -> None:
    """H3: three threshold crossings accrete THREE identical rows, not one.

    The forward write is a bare INSERT with no dedup — only the one-time
    migration dedups (identical-text only). Six edits at threshold 2 cross three
    times.
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    rstore, rid = _recruiter_store(tmp_path)
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )
    with patch(
        "shared.recruiter_overrides._summarize_bucket", return_value="Same text."
    ):
        for _ in range(3 * OVERRIDE_SUMMARY_THRESHOLD):
            record_override_for_field_path(store, "role_title.name")
    assert len(_signals(rstore, rid)) == 3  # append-only, no dedup


def test_record_override_fail_soft_on_record_raise_keeps_meta(
    tmp_path: Path, monkeypatch
) -> None:
    """H2: a raising ``record_taste_signal`` must NOT defeat the meta write.

    The spine write is fail-soft — the meta count increment + summary refresh
    must still land even when the durable write blows up.
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    rstore, rid = _recruiter_store(tmp_path)
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )

    def _boom(*a, **k):
        raise RuntimeError("record blew up")

    monkeypatch.setattr(
        "shared.runtime_state.recruiter_store.RecruiterStore.record_taste_signal",
        _boom,
    )
    with patch(
        "shared.recruiter_overrides._summarize_bucket", return_value="Kept."
    ):
        for _ in range(OVERRIDE_SUMMARY_THRESHOLD):
            record_override_for_field_path(store, "role_title.name")
    prefs = get_recruiter_preferences(store)
    assert prefs["override_counts"]["role_title"] == 2  # meta count incremented
    assert prefs["summaries"]["role_title"] == "Kept."  # summary refreshed
    assert prefs["active_voice"] is True


def test_record_override_fail_soft_on_construction_raise_keeps_meta(
    tmp_path: Path, monkeypatch
) -> None:
    """H2 (the load-bearing arm): a raise in ``RecruiterStore.__init__`` — which
    runs mkdir+DDL BEFORE record_taste_signal — must be caught too.

    This proves the try/except wraps CONSTRUCTION, not just the write. A branch
    that wrapped only the write would let a constructor raise propagate past the
    meta write at the bottom of record_override_for_field_path, skipping the
    count increment + summary refresh.
    """

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    rstore, rid = _recruiter_store(tmp_path)
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )

    def _boom_init(self, *a, **k):
        raise OSError("mkdir/DDL blew up in __init__")

    monkeypatch.setattr(
        "shared.runtime_state.recruiter_store.RecruiterStore.__init__", _boom_init
    )
    with patch(
        "shared.recruiter_overrides._summarize_bucket", return_value="Survived."
    ):
        for _ in range(OVERRIDE_SUMMARY_THRESHOLD):
            record_override_for_field_path(store, "role_title.name")
    prefs = get_recruiter_preferences(store)
    assert prefs["override_counts"]["role_title"] == 2  # meta survives __init__ raise
    assert prefs["summaries"]["role_title"] == "Survived."
    assert prefs["active_voice"] is True


# --- R6.2prime: the PUT /api/recruiter/preferences DARK bucketless write ------
#
# Exercised by calling the endpoint function directly (the route handler is a
# plain function) so the test does not depend on a built app / pypdf / env
# gates. _intake_store is redirected to a tmp RuntimeStateStore; the recruiter
# DB path + recruiter-id resolver are redirected so the spine write lands in
# tmp. H2: the same fail-soft-wraps-construction proof as record_override.


def _put_prefs(req):
    from cloris.api.intake import put_recruiter_preferences_endpoint

    return put_recruiter_preferences_endpoint(req)


def _prefs_request(**fields):
    from cloris.models import RecruiterPreferencesRequest

    return RecruiterPreferencesRequest(**fields)


@_NEEDS_MODELS
def test_put_preferences_writes_one_bucketless_signal_and_saves_meta(
    tmp_path: Path, monkeypatch
) -> None:
    """PUT {summary} -> exactly ONE bucketless {summary} signal + the meta saved.

    The forward write is BUCKETLESS (no bucket key) so the dark reader's
    bucketless branch projects it into the bare top-level line.
    """

    intake_store = RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")
    rstore, rid = _recruiter_store(tmp_path)
    monkeypatch.setattr("cloris.api.intake._intake_store", lambda: intake_store)
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )

    resp = _put_prefs(_prefs_request(summary="Be concise."))
    assert resp.preferences["summary"] == "Be concise."  # meta saved (200-equivalent)

    sigs = _signals(rstore, rid)
    assert len(sigs) == 1
    assert sigs[0]["signal_kind"] == SIGNAL_PRINCIPLE_FEEDBACK
    assert sigs[0]["domain"] == "intake_synthesis"
    assert sigs[0]["payload"] == {"summary": "Be concise."}  # NO bucket key


@_NEEDS_MODELS
def test_put_preferences_no_summary_writes_no_signal(
    tmp_path: Path, monkeypatch
) -> None:
    """A PUT that sets no bare summary writes ZERO spine signals."""

    intake_store = RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")
    rstore, rid = _recruiter_store(tmp_path)
    monkeypatch.setattr("cloris.api.intake._intake_store", lambda: intake_store)
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )

    resp = _put_prefs(_prefs_request(active_voice=True))  # no summary field set
    assert resp.preferences["active_voice"] is True
    assert _signals(rstore, rid) == []


@_NEEDS_MODELS
def test_put_preferences_fail_soft_on_construction_raise_still_saves_meta(
    tmp_path: Path, monkeypatch
) -> None:
    """H2: a raise in ``RecruiterStore.__init__`` must NOT turn the PUT into a 500.

    The meta write is hoisted ABOVE the spine write, and the fail-soft try/except
    wraps the full construction, so the response still carries the saved prefs.
    """

    intake_store = RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")
    rstore, rid = _recruiter_store(tmp_path)
    monkeypatch.setattr("cloris.api.intake._intake_store", lambda: intake_store)
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )

    def _boom_init(self, *a, **k):
        raise OSError("mkdir/DDL blew up in __init__")

    monkeypatch.setattr(
        "shared.runtime_state.recruiter_store.RecruiterStore.__init__", _boom_init
    )

    resp = _put_prefs(_prefs_request(summary="Be concise."))
    assert resp.preferences["summary"] == "Be concise."  # meta saved despite raise
    # The recruiter store is broken (every construct raises), so no signal landed,
    # but the live PUT path is unharmed — that is the whole point of fail-soft.
