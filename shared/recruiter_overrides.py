"""Recruiter correction memory for intake authoring."""

from __future__ import annotations

import json
import logging
from typing import Any

from shared.runtime_state.store import RuntimeStateStore

log = logging.getLogger(__name__)

# Reopen recruiter learns-half, R6.2prime: dropped 10 -> 2 so a bucket the meta
# blob summarizes is the SAME bucket the durable spine activates at
# (TASTE_SIGNAL_ACTIVE_THRESHOLD below). The meta write stays the live behavior;
# the forward spine write added in record_override_for_field_path is dark.
OVERRIDE_SUMMARY_THRESHOLD = 2


def get_recruiter_preferences(store: RuntimeStateStore) -> dict[str, Any]:
    raw = _read_meta(store, "recruiter_meta_preferences")
    return raw if isinstance(raw, dict) else {}


def put_recruiter_preferences(
    store: RuntimeStateStore,
    body: dict[str, Any],
) -> dict[str, Any]:
    with store.connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("recruiter_meta_preferences", json.dumps(body)),
        )
    return body


def record_override_for_field_path(
    store: RuntimeStateStore,
    field_path: str,
) -> None:
    """Record one recruiter correction and summarize every N events."""

    bucket = _bucket_for_field_path(field_path)
    prefs = dict(get_recruiter_preferences(store))
    counts = dict(prefs.get("override_counts") or {})
    counts[bucket] = int(counts.get(bucket, 0)) + 1
    prefs["override_counts"] = counts
    if counts[bucket] % OVERRIDE_SUMMARY_THRESHOLD == 0:
        summaries = dict(prefs.get("summaries") or {})
        summaries[bucket] = _summarize_bucket(bucket, counts[bucket])
        prefs["summaries"] = summaries
        prefs["active_voice"] = True
        # Reopen recruiter learns-half, R6.2prime (DARK forward write). Mirror
        # the freshly-summarized bucket into the durable recruiter primitive as
        # a {bucket, summary} principle_feedback signal, so cross-brief
        # calibration accretes on the recruiter entity rather than only in this
        # intake DB's meta blob. Fail-soft BY CONSTRUCTION (H2): the try/except
        # wraps the FULL RecruiterStore construction too — __init__ runs
        # mkdir + DDL (recruiter_store.py) and can raise BEFORE the write; since
        # the meta write below has NOT run yet, an escaping raise would skip the
        # meta refresh + count increment. Wrapping both keeps the live meta path
        # whole. APPEND-ONLY (H3): a bare INSERT, no dedup — every
        # threshold crossing accretes one auditable row (signals are
        # append-only + soft-superseded, never mutated in place). Lazy imports
        # so importing this module never drags the recruiter store in.
        try:
            from shared.output_paths import resolve_recruiter_db_path
            from shared.recruiter_context import get_current_recruiter_id
            from shared.runtime_state.recruiter_store import (
                SIGNAL_PRINCIPLE_FEEDBACK,
                RecruiterStore,
            )

            rid = get_current_recruiter_id()
            RecruiterStore(resolve_recruiter_db_path()).record_taste_signal(
                rid,
                signal_kind=SIGNAL_PRINCIPLE_FEEDBACK,
                domain=INTAKE_SYNTHESIS_DOMAIN,
                payload={"bucket": bucket, "summary": summaries[bucket]},
            )
        except Exception:  # noqa: BLE001 — fail-soft; meta write must still land
            log.debug(
                "R6.2prime spine write failed for bucket %r (fail-soft)",
                bucket,
                exc_info=True,
            )
    put_recruiter_preferences(store, prefs)


def recruiter_voice_line_for_extract(
    store: RuntimeStateStore,
    state: dict[str, Any],
) -> str:
    """Return concise preference lines to include during future synthesis."""

    lines: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            lines.append(text)

    global_prefs = get_recruiter_preferences(store)
    session_prefs = state.get("meta_preferences") if isinstance(state, dict) else None
    if isinstance(session_prefs, dict):
        add(session_prefs.get("summary"))
    add(global_prefs.get("summary"))

    global_active = bool(global_prefs.get("active_voice"))
    session_active = True
    if isinstance(session_prefs, dict) and "active_voice" in session_prefs:
        session_active = bool(session_prefs.get("active_voice"))
    if global_active and session_active:
        for prefs in (session_prefs, global_prefs):
            if not isinstance(prefs, dict):
                continue
            summaries = prefs.get("summaries")
            if not isinstance(summaries, dict):
                continue
            for key in sorted(summaries):
                value = summaries.get(key)
                if isinstance(value, str) and value.strip():
                    add(f"{key}: {value.strip()}")
    return "\n".join(lines)


# Reopen recruiter learns-half, R6.1: the GLOBAL prefs read from the durable
# recruiter primitive (``recruiter_taste_signals``) instead of the per-intake-DB
# ``recruiter_meta_preferences`` meta key. A bucket becomes active at the 2nd
# correction (down from the meta-key path's 10th), so cross-brief calibration
# surfaces sooner once a recruiter has corrected the same bucket twice.
TASTE_SIGNAL_ACTIVE_THRESHOLD = 2

# The domain every intake-synthesis taste signal is filed + read under. This is
# the DOMAIN, not the signal_kind — using it as a kind would raise ValueError in
# ``record_taste_signal`` (it is not in ``KNOWN_SIGNAL_KINDS``).
INTAKE_SYNTHESIS_DOMAIN = "intake_synthesis"


def recruiter_taste_signals_for_extract(
    recruiter_store: Any,
    recruiter_id: int,
    session_state: dict[str, Any] | None = None,
) -> str:
    """Per-bucket-summary contract of :func:`recruiter_voice_line_for_extract`, sourced from signals.

    Reopen recruiter learns-half, R6.1. The live reader
    (:func:`recruiter_voice_line_for_extract`) reads the GLOBAL prefs from the
    intake DB ``recruiter_meta_preferences`` meta key. This reader sources them
    instead from the durable recruiter primitive —
    ``recruiter_store.active_taste_signals(recruiter_id, domain="intake_synthesis")``
    — so calibration compounds across briefs on the recruiter entity rather than
    being trapped in one intake DB.

    SCOPE GAP — CLOSED by R6.1prime/R6.2prime/R6.4prime. This reader now
    reproduces BOTH halves of the live reader's global projection: the per-bucket
    ``summaries`` lines AND the bare top-level ``summary`` line. The bare line is
    sourced from a BUCKETLESS ``{"summary": text}`` ``principle_feedback`` signal
    — the shape the ``PUT /api/recruiter/preferences`` forward write (R6.2prime,
    ``cloris/api/intake.py``) emits when a recruiter sets ``summary``
    (``cloris/models.py`` ``summary`` field), and the shape R6.4prime's
    bare-summary migration backfills from the legacy blob's top-level
    ``summary``. The signal loop captures it into ``global_bare_summary``
    (before the per-bucket guard, H1) and the global-prefs build sets
    ``global_prefs["summary"]`` from it, so the previously-dead
    ``add(global_prefs.get("summary"))`` below now goes live for the
    signal-sourced global prefs — exactly as it always was for the
    ``session_state`` path. The bare line stays UNGATED by ``active_voice``
    (mirrors the live reader), so a recruiter with only a bare summary and no
    per-bucket signal still emits the naked line with zero bucket lines.

    Each active signal carries a ``{bucket, summary}`` payload. Taste-signal rows
    have NO ``summaries`` map and NO ``active_voice`` flag by default, and the
    emission branch below (mirrored byte-for-byte from the live reader) only emits
    ``f"{bucket}: {summary}"`` lines when ``active_voice`` is True against a
    ``{summaries: {bucket: text}}`` shape. So this reader PROJECTS each payload
    into that shape and synthesizes ``active_voice=True`` whenever any active
    signal projects a non-empty ``(bucket, summary)``. Without that projection the
    reader would return ``""`` for every recruiter forever — a silent dead reader.

    The ``session_state`` (per-session in-progress prefs) merge is preserved
    exactly as the live reader does it: the session's ``meta_preferences`` are
    read, its bare ``summary`` and per-bucket ``summaries`` are emitted, and its
    ``active_voice`` gate is honored.

    This reader is wired NOWHERE in production (the live call-sites at
    ``cloris/api/intake.py`` and ``cloris/api/intake_synthesis.py`` still call
    :func:`recruiter_voice_line_for_extract`). Repointing them is R6.3, out of
    scope here.
    """

    from shared.runtime_state.recruiter_store import SIGNAL_PRINCIPLE_FEEDBACK

    lines: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if text and text not in seen:
            seen.add(text)
            lines.append(text)

    # Project active signals -> the {summaries, active_voice} shape the gated
    # emission below understands. ``signal_kind`` is asserted so a future kind
    # mixed into this domain can't silently leak a non-feedback payload.
    projected_summaries: dict[str, str] = {}
    # R6.1prime: the bare top-level summary, sourced from a bucketless
    # {"summary": text} signal (the shape R6.2prime's PUT-endpoint write and
    # R6.4prime's bare-summary migration emit). Last bucketless writer wins —
    # active_taste_signals is ORDER BY created_at, so the freshest naked summary
    # lands. Stays None when no bucketless signal exists, so a bucketed-only
    # recruiter emits no bare line.
    global_bare_summary: str | None = None
    try:
        active = recruiter_store.active_taste_signals(
            recruiter_id, domain=INTAKE_SYNTHESIS_DOMAIN
        )
    except Exception:  # noqa: BLE001 — a read failure must not break synthesis
        active = []
    for signal in active:
        if not isinstance(signal, dict):
            continue
        if signal.get("signal_kind") != SIGNAL_PRINCIPLE_FEEDBACK:
            continue
        payload = signal.get("payload")
        if not isinstance(payload, dict):
            continue
        bucket = payload.get("bucket")
        summary = payload.get("summary")
        # R6.1prime (H1): the bucketless branch MUST precede the bucket guard
        # below. A bucketless payload has bucket=None, which fails the guard's
        # isinstance(bucket, str) check and `continue`s — so a branch placed
        # AFTER the guard would be DEAD and a bare-only recruiter would emit ""
        # once the live call-sites repoint (R6.3). Captured here, before the
        # guard, the naked summary survives.
        if (
            payload.get("bucket") is None
            and isinstance(summary, str)
            and summary.strip()
        ):
            global_bare_summary = summary.strip()
            continue
        if not isinstance(bucket, str) or not isinstance(summary, str):
            continue
        bucket = bucket.strip()
        summary = summary.strip()
        if bucket and summary:
            # Last writer wins within a single read — active_taste_signals is
            # ordered by created_at, so the freshest summary for a bucket lands.
            projected_summaries[bucket] = summary

    global_prefs: dict[str, Any] = {
        "summaries": projected_summaries,
        # Synthesize the gate: active only when a non-empty projection exists, so
        # an empty signal set yields "" (no spurious activation) exactly as the
        # meta-key reader does when no bucket has crossed threshold. UNCHANGED by
        # R6.1prime — the bare summary is an UNGATED top-level line (the live
        # reader emits add(global_prefs.get("summary")) outside the active_voice
        # gate), so a bare-only recruiter is non-empty even with active_voice
        # False and zero bucket lines.
        "active_voice": bool(projected_summaries),
    }
    # R6.1prime: surface the bucketless summary as the bare top-level line. Set
    # ONLY when present so a bucketed-only recruiter keeps no "summary" key (the
    # add() below then no-ops, matching the pre-flip meta reader). This is what
    # makes the previously-dead add(global_prefs.get("summary")) at the bottom
    # go live for the signal-sourced global prefs.
    if global_bare_summary is not None:
        global_prefs["summary"] = global_bare_summary

    state = session_state if isinstance(session_state, dict) else {}
    session_prefs = state.get("meta_preferences") if isinstance(state, dict) else None
    if isinstance(session_prefs, dict):
        add(session_prefs.get("summary"))
    add(global_prefs.get("summary"))

    global_active = bool(global_prefs.get("active_voice"))
    session_active = True
    if isinstance(session_prefs, dict) and "active_voice" in session_prefs:
        session_active = bool(session_prefs.get("active_voice"))
    if global_active and session_active:
        for prefs in (session_prefs, global_prefs):
            if not isinstance(prefs, dict):
                continue
            summaries = prefs.get("summaries")
            if not isinstance(summaries, dict):
                continue
            for key in sorted(summaries):
                value = summaries.get(key)
                if isinstance(value, str) and value.strip():
                    add(f"{key}: {value.strip()}")
    return "\n".join(lines)


def resolve_intake_preferences(
    intake_store: RuntimeStateStore,
    recruiter_store: Any,
    recruiter_id: int,
    session_state: dict[str, Any] | None = None,
) -> str:
    """The R6.3 FLIP preflight: read the recruiter SPINE, fail-closed to the blob.

    Reopen recruiter learns-half, R6.3. This is the helper the two live
    synthesis seams (``cloris/api/intake.py`` ``_refresh_source_packet_artifacts``
    and ``cloris/api/intake_synthesis.py``) call instead of
    :func:`recruiter_voice_line_for_extract` directly. After this flip, synthesis
    sources recruiter calibration from the durable recruiter primitive
    (cross-brief, :func:`recruiter_taste_signals_for_extract`) rather than the
    per-intake-DB ``recruiter_meta_preferences`` blob — so a correction made on
    one brief reaches synthesis on the next.

    FAIL-CLOSED (the locked behavior, not the structure): the SPINE is tried
    first. When it yields anything non-empty, that wins. When the spine is empty
    — a recruiter who accreted calibration before the primitive existed, or whose
    forward-write never landed (the writes are fail-soft) — fall back to the OLD
    per-intake-DB reader rather than silently wiping the recruiter's prefs to "".
    Only when BOTH are empty do we return "". The flip therefore never *loses*
    learned calibration; at worst it keeps reading the legacy source until the
    spine has caught up.

    The fallback is logged so a regression — the spine going quietly empty for a
    recruiter who should have signals — is observable rather than silent.
    """

    spine = recruiter_taste_signals_for_extract(
        recruiter_store, recruiter_id, session_state
    )
    if spine.strip():
        return spine

    legacy = recruiter_voice_line_for_extract(intake_store, session_state or {})
    if legacy.strip():
        log.info(
            "R6.3 intake-prefs fail-closed fallback: recruiter spine empty for "
            "recruiter_id=%r, reading legacy intake-DB meta blob instead",
            recruiter_id,
        )
        return legacy
    return ""


def _read_meta(store: RuntimeStateStore, key: str) -> Any:
    with store.connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return None


def _bucket_for_field_path(path: str) -> str:
    head = path.strip().split(".", 1)[0].split("[", 1)[0]
    return head or "general"


def _summarize_bucket(bucket: str, total: int) -> str:
    from pathlib import Path

    import shared.config as shared_config
    from market_intelligence.briefing_polish import _has_llm_access
    from shared.llm_clients import opus_llm_cached
    from shared.llm_usage import llm_usage_session

    if not _has_llm_access():
        return (
            f"For {bucket}, follow this recruiter's repeated correction pattern "
            f"({total} corrections tracked)."
        )
    base = Path(getattr(shared_config, "OUTPUT_DIR", Path("."))) / "intake_logs"
    base.mkdir(parents=True, exist_ok=True)
    prompt = (
        f"The recruiter has corrected Cloris-authored brief fields grouped as "
        f"{bucket!r} {total} times. Write one concrete sentence for future "
        f"brief synthesis. Return JSON only: {{\"sentence\": \"...\"}}."
    )
    try:
        with llm_usage_session(base / "preference_summary.jsonl", stage="preference_summary"):
            raw = opus_llm_cached(
                "You compress recruiter correction history into one actionable sentence.",
                prompt,
                expect_json=True,
                max_tokens=400,
                usage_context={"stage": "preference_summary", "bucket": bucket},
            )
    except Exception:
        return f"For {bucket}, weight this recruiter's repeated manual fixes."
    if isinstance(raw, dict) and isinstance(raw.get("sentence"), str):
        return raw["sentence"].strip()[:500]
    return f"For {bucket}, follow this recruiter's established correction pattern."
