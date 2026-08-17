"""R6.3 — THE FLIP: synthesis reads the recruiter SPINE, fail-closed to the blob.

These are the regression pins for the only R6 slice that changes LIVE synthesis
behavior. Until R6.3, the two synthesis seams
(``cloris.api.intake._refresh_source_packet_artifacts`` and
``cloris.api.intake_synthesis``) called ``recruiter_voice_line_for_extract`` —
the per-intake-DB ``recruiter_meta_preferences`` reader, which is structurally
incapable of carrying a correction made on one brief into synthesis on the next
(each brief is a different intake DB). After the flip they call
``resolve_intake_preferences``, which reads the durable cross-brief recruiter
SPINE first and falls back to the legacy blob only when the spine is empty.

The cross-brief pin proves the thing the OLD reader could not do: a correction
recorded while authoring ONE brief (a per-bucket critique-commit forward write
and a bucketless ``PUT /api/recruiter/preferences`` forward write) reaches
``synthesize_v2_from_source_packet`` while authoring a SECOND, different brief —
carried on the recruiter entity, not the per-intake-DB blob.

The fail-closed pins prove the locked safety behavior: an empty spine never
silently wipes a recruiter's prefs to ``""`` — it falls back to the legacy
reader; only when BOTH sources are empty is the result ``""``.

LIVE-PATH discipline: the cross-brief pin drives the REAL repointed seam
(``_refresh_source_packet_artifacts``), not a proxy. It monkeypatches
``synthesize_v2_from_source_packet`` only to CAPTURE the ``recruiter_preferences``
kwarg the seam computed (the flip's whole observable), then raises a sentinel so
the downstream distillation/gap work is skipped — the kwarg is already captured.
The recruiter store + acting recruiter resolve through the same seams the forward
writes use (``resolve_recruiter_db_path`` / ``get_current_recruiter_id``), so the
write on brief A and the read on brief B hit one durable spine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.recruiter_overrides import (
    OVERRIDE_SUMMARY_THRESHOLD,
    put_recruiter_preferences,
    record_override_for_field_path,
    resolve_intake_preferences,
)
from shared.runtime_state.recruiter_store import RecruiterStore
from shared.runtime_state.store import RuntimeStateStore


def _cloris_models_importable() -> bool:
    """Whether ``cloris.models`` imports here (PEP-604 eval on <3.10 raises).

    Mirrors ``tests/test_recruiter_overrides.py``: the bucketless half of the
    cross-brief seed drives the real ``PUT /api/recruiter/preferences`` endpoint,
    which needs ``RecruiterPreferencesRequest``. Skip when the model can't import
    (the same env-tolerance the repo already applies to optional deps); runs green
    on 3.10+/CI.
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


class _SynthesisCaptured(Exception):
    """Sentinel raised by the captured synthesize stub once the kwarg is recorded."""


def _recruiter_store(tmp_path: Path) -> tuple[RecruiterStore, int]:
    store = RecruiterStore(tmp_path / "_recruiter" / "recruiter.sqlite3")
    rid = store.upsert_recruiter("operator@example.com", display_name="Sam")
    return store, rid


def _redirect_recruiter_spine(monkeypatch, rstore: RecruiterStore, rid: int) -> None:
    """Point both forward-write seams AND the repointed reader at one tmp spine.

    The forward writes (``record_override_for_field_path``, the PUT endpoint) and
    the repointed seam all resolve the recruiter store via
    ``shared.output_paths.resolve_recruiter_db_path`` and the acting recruiter via
    ``shared.recruiter_context.get_current_recruiter_id`` (lazy imports, so the
    source-module attributes are what they bind at call time). Redirect both to a
    single tmp store/id — that single store is the cross-brief carrier.
    """

    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path", lambda: rstore.db_path
    )
    monkeypatch.setattr(
        "shared.recruiter_context.get_current_recruiter_id", lambda: rid
    )


def _brief_b_state(jd_text: str) -> dict:
    """A minimal but VALID source-packet state for the seam's brief-B synthesis.

    A non-empty ``job_description_text`` makes ``compose_source_packet_text``
    produce non-empty ``source_text``, so the seam's empty-source 422 guard is
    skipped and it reaches the captured ``synthesize_v2_from_source_packet``.
    """

    return {"source_packet": {"job_description_text": jd_text}}


# --- The cross-brief LIVE-path pin (the flip's whole reason for being) --------


@_NEEDS_MODELS
def test_crossbrief_correction_reaches_synthesis_via_spine(
    tmp_path: Path, monkeypatch
) -> None:
    """A correction on brief A reaches synthesis on brief B — via the SPINE.

    Seeds the recruiter spine through the REAL product forward writes while
    "authoring brief A":
      * a per-bucket critique-commit correction — ``record_override_for_field_path``
        crossing the 2-event threshold forward-writes a ``{bucket, summary}``
        ``principle_feedback`` signal (the exact call the critique/commit handler
        makes at ``cloris/api/intake.py``);
      * a bucketless ``PUT /api/recruiter/preferences`` bare summary —
        forward-writes a bucketless ``{summary}`` signal.

    Then drives the REPOINTED seam (``_refresh_source_packet_artifacts``) on a
    SECOND brief whose own intake DB is empty, and asserts the captured
    ``recruiter_preferences`` carries BOTH the per-bucket line (``role_title: ...``)
    AND the naked bare line (no bucket prefix). The OLD per-intake-DB reader could
    not do this: brief B's intake DB never saw brief A's corrections.
    """

    from cloris.api import intake as intake_mod
    from cloris.api.intake import put_recruiter_preferences_endpoint
    from cloris.models import RecruiterPreferencesRequest

    rstore, rid = _recruiter_store(tmp_path)
    _redirect_recruiter_spine(monkeypatch, rstore, rid)

    # --- author brief A: seed the spine via the real forward writes -----------
    brief_a_store = RuntimeStateStore(tmp_path / "brief_a" / "intake_sessions.sqlite3")
    # Per-bucket forward write (critique/commit path). _summarize_bucket is the
    # LLM-or-fallback summarizer; pin it to a deterministic sentence so the
    # asserted line is stable and no LLM is touched.
    monkeypatch.setattr(
        "shared.recruiter_overrides._summarize_bucket",
        lambda bucket, total: "Prefer concise, concrete titles.",
    )
    for _ in range(OVERRIDE_SUMMARY_THRESHOLD):
        record_override_for_field_path(brief_a_store, "role_title.name")

    # Bucketless forward write (PUT /api/recruiter/preferences bare summary). The
    # endpoint writes its meta blob through _intake_store; point that at brief A's
    # store so the bucketless spine signal is the only thing that can cross briefs.
    monkeypatch.setattr(intake_mod, "_intake_store", lambda: brief_a_store)
    resp = put_recruiter_preferences_endpoint(
        RecruiterPreferencesRequest(summary="Lead with the hardest problem.")
    )
    assert resp.preferences["summary"] == "Lead with the hardest problem."

    # --- author brief B: a DIFFERENT, empty intake DB -------------------------
    brief_b_store = RuntimeStateStore(tmp_path / "brief_b" / "intake_sessions.sqlite3")
    monkeypatch.setattr(intake_mod, "_intake_store", lambda: brief_b_store)

    captured: dict[str, str] = {}

    def _capture(*, recruiter_preferences: str = "", **_kwargs):
        captured["recruiter_preferences"] = recruiter_preferences
        raise _SynthesisCaptured

    # The seam does `from shared.source_packet_synthesis import
    # synthesize_v2_from_source_packet` at call time — patch the source module.
    monkeypatch.setattr(
        "shared.source_packet_synthesis.synthesize_v2_from_source_packet", _capture
    )

    state = _brief_b_state(
        "Staff Platform Engineer\n\nOwn the developer platform and ship "
        "internal systems with strong reliability patterns."
    )
    with pytest.raises(_SynthesisCaptured):
        intake_mod._refresh_source_packet_artifacts(state=state, session_id=999)

    prefs = captured["recruiter_preferences"]
    # The per-bucket line — carried from brief A's critique-commit, on the spine.
    assert "role_title: Prefer concise, concrete titles." in prefs, prefs
    # The naked bare line — carried from brief A's PUT, UNGATED by active_voice.
    assert "Lead with the hardest problem." in prefs, prefs
    # The bare line is genuinely bare (no bucket prefix on its own line).
    bare_lines = [ln for ln in prefs.splitlines() if ln == "Lead with the hardest problem."]
    assert bare_lines, prefs


# --- Fail-closed: empty spine must NOT wipe prefs, falls back to the blob -----


def test_failclosed_empty_spine_falls_back_to_legacy_blob(
    tmp_path: Path,
) -> None:
    """Empty spine + non-empty legacy meta blob -> the OLD reader's output.

    The locked safety behavior: the flip must never silently wipe a recruiter's
    learned calibration to ``""`` just because the durable spine hasn't accreted
    it yet (a pre-primitive recruiter, or a fail-soft forward write that never
    landed). With no signals in the recruiter store, ``resolve_intake_preferences``
    must return exactly what ``recruiter_voice_line_for_extract`` returns over the
    legacy intake-DB blob.
    """

    rstore, rid = _recruiter_store(tmp_path)  # empty spine: no taste signals
    intake_store = RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")
    # A non-empty legacy blob: an active per-bucket summary the old reader emits.
    put_recruiter_preferences(
        intake_store,
        {
            "summary": "Bias toward builders.",
            "summaries": {"role_title": "Prefer concise titles."},
            "active_voice": True,
        },
    )

    from shared.recruiter_overrides import recruiter_voice_line_for_extract

    expected = recruiter_voice_line_for_extract(intake_store, {})
    assert expected.strip()  # guard: the legacy blob really is non-empty

    out = resolve_intake_preferences(intake_store, rstore, rid, None)
    assert out == expected  # fail-closed: legacy output, NOT ""
    assert "Bias toward builders." in out
    assert "role_title: Prefer concise titles." in out


def test_failclosed_both_empty_returns_empty(tmp_path: Path) -> None:
    """Empty spine + empty legacy blob -> ``""`` (no spurious activation)."""

    rstore, rid = _recruiter_store(tmp_path)  # empty spine
    intake_store = RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")
    # No meta blob written: the legacy reader returns "" too.
    assert resolve_intake_preferences(intake_store, rstore, rid, None) == ""


def test_populated_spine_wins_over_legacy_blob(tmp_path: Path) -> None:
    """When the spine is non-empty it wins outright — the blob is not consulted.

    The inverse of fail-closed: once calibration lives on the spine, the legacy
    per-intake-DB blob is irrelevant. A signal-sourced bucket line must appear and
    the (differently-worded) legacy blob line must NOT, proving the spine — not the
    blob — drove the output.
    """

    from shared.runtime_state.recruiter_store import SIGNAL_PRINCIPLE_FEEDBACK

    rstore, rid = _recruiter_store(tmp_path)
    rstore.record_taste_signal(
        rid,
        signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
        domain="intake_synthesis",
        payload={"bucket": "role_title", "summary": "From the spine."},
    )
    intake_store = RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")
    put_recruiter_preferences(
        intake_store,
        {"summaries": {"role_title": "From the blob."}, "active_voice": True},
    )

    out = resolve_intake_preferences(intake_store, rstore, rid, None)
    assert "role_title: From the spine." in out
    assert "From the blob." not in out  # the blob was never read


# --- Seam-level fail-soft: a recruiter-store CONSTRUCTION failure must not 500 -


def test_seam_failsoft_on_recruiter_store_construction_error(
    tmp_path: Path, monkeypatch
) -> None:
    """If RecruiterStore(...) construction raises at the seam, synthesis must NOT
    500 — it degrades to the legacy intake-blob reader.

    R6.3's whole purpose is fail-closed: the flip must never break synthesis.
    The preflight helper is internally safe, but the recruiter-store CONSTRUCTION
    happens at the seam (RecruiterStore.__init__ runs mkdir + DDL and can raise on
    a disk/DDL failure) — pre-flip the seam touched no recruiter store, so this is
    a NET-NEW failure mode the seam wraps in try/except -> legacy reader. This pin
    injects the construction failure (which the cross-brief/fail-closed pins, on a
    writable tmp_path, never exercise).
    """

    from cloris.api import intake as intake_mod

    # A legacy blob in brief B's intake DB — the fallback target. If the seam
    # 500s instead of falling back, this never reaches synthesis.
    brief_b_store = RuntimeStateStore(tmp_path / "brief_b" / "intake_sessions.sqlite3")
    put_recruiter_preferences(
        brief_b_store,
        {"summary": "Legacy fallback summary.", "active_voice": True},
    )
    monkeypatch.setattr(intake_mod, "_intake_store", lambda: brief_b_store)

    # Resolution succeeds, construction raises — exactly the exposed seam raise
    # (get_current_recruiter_id is internally hardened; only RecruiterStore(...)
    # is the unguarded call the seam wraps).
    monkeypatch.setattr(
        "shared.output_paths.resolve_recruiter_db_path",
        lambda: tmp_path / "_recruiter" / "recruiter.sqlite3",
    )
    monkeypatch.setattr("shared.recruiter_context.get_current_recruiter_id", lambda: 1)

    def _boom(*_a, **_k):
        raise OSError("simulated mkdir/DDL failure on the recruiter DB")

    monkeypatch.setattr("shared.runtime_state.recruiter_store.RecruiterStore", _boom)

    captured: dict[str, str] = {}

    def _capture(*, recruiter_preferences: str = "", **_kwargs):
        captured["recruiter_preferences"] = recruiter_preferences
        raise _SynthesisCaptured

    monkeypatch.setattr(
        "shared.source_packet_synthesis.synthesize_v2_from_source_packet", _capture
    )

    state = _brief_b_state(
        "Staff Backend Engineer\n\nOwn core services and reliability."
    )
    # The seam reaches synthesis (the captured stub) WITHOUT propagating the
    # construction OSError — i.e. it did NOT 500; it fell back.
    with pytest.raises(_SynthesisCaptured):
        intake_mod._refresh_source_packet_artifacts(state=state, session_id=998)

    # And the preferences it computed are the LEGACY reader's output (the
    # fail-soft fell back), not "" and not a propagated error.
    assert captured["recruiter_preferences"] == "Legacy fallback summary."
