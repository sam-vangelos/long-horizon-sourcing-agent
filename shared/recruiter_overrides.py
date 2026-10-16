"""Recruiter correction memory for intake authoring."""

from __future__ import annotations

import json
import logging
from typing import Any

from shared.runtime_state.store import RuntimeStateStore

log = logging.getLogger(__name__)

# Keep the meta-summary threshold aligned with the durable signal activation
# threshold. The meta write remains the live path; the signal write is optional
# and fail-soft.
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
        # Mirror the summarized bucket into the durable recruiter primitive as a
        # {bucket, summary} principle_feedback signal so calibration can span
        # briefs. The try/except includes RecruiterStore construction (mkdir and
        # DDL) as well as the write; a failure leaves the meta refresh and count
        # update intact. The INSERT is append-only with no dedup, and active
        # signals use soft supersession rather than in-place mutation. Lazy
        # imports keep this module independent at import time.
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


# The signal-backed reader takes global preferences from the durable recruiter
# primitive rather than the per-intake meta key. A bucket becomes active after
# the second correction, so the same recruiter's calibration can span briefs.
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
    """Return preference lines from active recruiter signals.

    This preserves the shape of :func:`recruiter_voice_line_for_extract` while
    reading ``recruiter_store.active_taste_signals`` for the recruiter and
    ``INTAKE_SYNTHESIS_DOMAIN``. Bucketed ``principle_feedback`` payloads are
    projected into ``summaries`` and activate the bucketed voice when a
    non-empty projection exists. A bucketless ``{"summary": text}`` payload
    becomes the bare global summary and remains ungated by that bucketed voice.

    The bucketless summary is captured before the bucket type guard. Active
    signals are ordered by ``created_at``, so the last signal wins within a
    bucket or for the bare summary. The session state's meta preferences are
    merged with the same summary and ``active_voice`` rules as the regular
    reader. A signal read failure is fail-soft and leaves session preferences
    available. Callers reach this reader through
    :func:`resolve_intake_preferences`, which falls back to
    :func:`recruiter_voice_line_for_extract` when this reader returns ``""``.
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

    # Project active principle_feedback signals into the {summaries,
    # active_voice} shape consumed by the emission branch. Filter on signal_kind so
    # another payload kind in this domain cannot leak into recruiter-facing text.
    projected_summaries: dict[str, str] = {}
    # A bucketless {"summary": text} signal supplies the bare top-level summary.
    # active_taste_signals is ordered by created_at, so the freshest bare summary
    # wins. Leave this value None when no bucketless signal exists.
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
        # Handle bucketless payloads before the bucket type guard: bucket=None
        # would otherwise fail the string check and be discarded. This preserves
        # the bare summary for signal-backed preferences.
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
        # active_voice reflects non-empty bucket projections only. An empty signal
        # set therefore emits no bucket lines. The bare summary is emitted outside
        # this gate, so it remains visible without bucketed signals.
        "active_voice": bool(projected_summaries),
    }
    # Add the bucketless summary only when present. Bucketed-only preferences
    # keep no top-level ``summary`` key, while a bare signal remains a visible
    # recruiter-facing line.
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
    """Read signal-backed recruiter preferences with a fail-closed blob fallback.

    Try the durable recruiter primitive first. A non-empty signal result wins;
    when it is empty, use the per-intake meta reader so fail-soft signal writes
    cannot erase learned preferences. Return an empty string only when both
    sources are empty. Log the fallback so an unexpectedly empty signal read
    remains observable.
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
