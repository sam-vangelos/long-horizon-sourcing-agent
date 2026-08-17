"""Background synthesis scheduler for intake source-packet uploads.

Companion to :func:`cloris.api.intake.upload_source_packet_files_endpoint`.

Splits the long-running source-packet synthesis off the HTTP request
thread so the upload route can return immediately with
``state_json.source_packet_synthesis.status == "running"``. Recruiters
see the uploaded file row instantly; the draft sidebar polls until the
status flips to ``ready`` or ``failed``.

Concurrency contract
--------------------

- At most one active worker per ``session_id`` is *registered* in the
  task table; a new schedule arriving while a task is in flight launches
  a fresh worker and supersedes the registry entry.
- We do **not** cancel the in-flight worker — letting the LLM call
  finish and discarding its result avoids tearing down requests with
  side effects. The revision check on commit makes the older task's
  write a no-op.
- The worker writes **only** synthesis-owned fields
  (``v2_draft``, ``v2_draft_polish_meta``, ``field_provenance``,
  ``gap_questions``, ``retrieval_meta``, ``distillation``) plus the
  ``source_packet_synthesis`` meta block. Everything else under
  ``state_json`` (recruiter manual edits, conversation state, gap-answer
  overlays) is preserved across the commit.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("cloris.api.intake_synthesis")


SYNTHESIS_OWNED_FIELDS: tuple[str, ...] = (
    "v2_draft",
    "v2_draft_polish_meta",
    "field_provenance",
    "gap_questions",
    "retrieval_meta",
    "distillation",
    # Synthesis writes one sub-key under intake_insights
    # (``hiring_manager_success_image``). The whole bag is named owned so
    # the synthesis-vs-conversation merge contract stays honest about what
    # synthesis can touch — but the actual write merges sub-key by sub-key
    # via ``merge_intake_insights`` so a recruiter-corrected picture
    # (locked via ``intake_insights.hiring_manager_success_image`` in
    # ``manually_edited_keys``) survives synthesis re-runs.
    "intake_insights",
)

SYNTHESIS_STATUS_IDLE = "idle"
SYNTHESIS_STATUS_RUNNING = "running"
SYNTHESIS_STATUS_READY = "ready"
SYNTHESIS_STATUS_FAILED = "failed"

# Recruiter-safe error copy. Never include raw stack traces here — this
# string lands verbatim in the intake draft sidebar.
SYNTHESIS_GENERIC_ERROR = (
    "We couldn't update the draft from this file. "
    "Try a different file or paste the text directly."
)
SYNTHESIS_EMPTY_PACKET_ERROR = (
    "We couldn't find any text in this upload. "
    "Try a different file or paste the text directly."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_synthesis_state(state: dict[str, Any]) -> dict[str, Any]:
    """Idempotently initialize ``state_json.source_packet_synthesis``.

    Returns the inner block so callers can mutate it. New sessions land
    with ``status="idle"`` and ``revision=0``; pre-existing sessions
    keep whatever prior runs wrote, with defensive defaults filled in.
    """

    block = state.get("source_packet_synthesis")
    if not isinstance(block, dict):
        block = {
            "status": SYNTHESIS_STATUS_IDLE,
            "error": None,
            "started_at": None,
            "completed_at": None,
            "revision": 0,
        }
        state["source_packet_synthesis"] = block
    block.setdefault("status", SYNTHESIS_STATUS_IDLE)
    block.setdefault("error", None)
    block.setdefault("started_at", None)
    block.setdefault("completed_at", None)
    block.setdefault("revision", 0)
    return block


def bump_synthesis_revision(state: dict[str, Any]) -> int:
    """Increment ``source_packet_synthesis.revision`` and return the new value.

    Used by both the async upload route (before spawning a worker) and
    by the sync ``source_packet`` / ``answer_questions`` routes (right
    after they synchronously rewrite synthesis-owned fields) so any
    concurrently-running background worker's later commit becomes a
    no-op via the revision check.
    """

    block = ensure_synthesis_state(state)
    next_revision = int(block.get("revision") or 0) + 1
    block["revision"] = next_revision
    return next_revision


class SynthesisInputError(RuntimeError):
    """Raised by :func:`refresh_source_packet_artifacts_pure` when the input is unusable."""


@dataclass(frozen=True)
class SynthesisProduct:
    """One synthesis pass's product — committed under the revision guard."""

    v2_draft: dict[str, Any]
    v2_draft_polish_meta: dict[str, Any]
    field_provenance: dict[str, Any]
    gap_questions: list[dict[str, Any]]
    retrieval_meta: dict[str, Any]
    distillation: dict[str, Any]
    role_title: str | None
    intake_insights: dict[str, Any]


def refresh_source_packet_artifacts_pure(
    *,
    state_snapshot: dict[str, Any],
    session_id: int,
) -> SynthesisProduct:
    """Compute one synthesis pass without touching durable state.

    Mirrors :func:`cloris.api.intake._refresh_source_packet_artifacts`
    but takes a read-only snapshot and returns the product. The caller
    commits the synthesis-owned fields under a revision guard.
    """

    from market_intelligence.brief_distillation import distill_brief
    from shared.brief_corpus import build_exemplar_block
    from shared.gap_questions import generate_gap_questions
    from shared.output_paths import resolve_recruiter_db_path
    from shared.recruiter_context import get_current_recruiter_id
    from shared.recruiter_overrides import (
        recruiter_voice_line_for_extract,
        resolve_intake_preferences,
    )
    from shared.runtime_state.recruiter_store import RecruiterStore
    from shared.source_packet import compose_source_packet_text
    from shared.source_packet_synthesis import synthesize_v2_from_source_packet

    from cloris.api.intake import _intake_store, _source_files_from_state

    source_packet = state_snapshot.get("source_packet")
    if not isinstance(source_packet, dict):
        source_packet = {}

    files = _source_files_from_state(source_packet)
    gap_answer_history = state_snapshot.get("gap_answer_history")
    if not isinstance(gap_answer_history, list):
        gap_answer_history = []

    jd_text = (
        source_packet.get("job_description_text")
        if isinstance(source_packet.get("job_description_text"), str)
        else ""
    )
    intake_notes = (
        source_packet.get("intake_notes_text")
        if isinstance(source_packet.get("intake_notes_text"), str)
        else ""
    )
    source_text = compose_source_packet_text(
        job_description_text=jd_text,
        intake_notes_text=intake_notes,
        files=files,
        gap_answer_history=gap_answer_history,
    )
    if not source_text.strip():
        raise SynthesisInputError("empty_source_packet")

    store = _intake_store()
    exemplar_block, used_ids = build_exemplar_block(store, source_text)
    # Reopen recruiter learns-half, R6.3 (THE FLIP). Mirror the JSON-paste seam in
    # ``cloris.api.intake._refresh_source_packet_artifacts``: read recruiter
    # calibration from the durable cross-brief SPINE, fail-closed to the legacy
    # per-intake-DB blob. ``state_snapshot`` is this path's read-only session.
    # Fail-soft to the legacy reader if recruiter-store resolution/construction
    # raises (mkdir + DDL in __init__) — the flip must never 500 synthesis.
    try:
        rid = get_current_recruiter_id()
        recruiter_store = RecruiterStore(resolve_recruiter_db_path())
        preferences = resolve_intake_preferences(
            store, recruiter_store, rid, state_snapshot
        )
    except Exception:  # noqa: BLE001 — degrade to legacy, never break synthesis
        log.warning(
            "R6.3: recruiter-store resolution failed; falling back to the "
            "legacy intake-blob reader for synthesis preferences",
            exc_info=True,
        )
        preferences = recruiter_voice_line_for_extract(store, state_snapshot)
    current_v2 = (
        state_snapshot.get("v2_draft")
        if isinstance(state_snapshot.get("v2_draft"), dict)
        else None
    )
    current_provenance = (
        state_snapshot.get("field_provenance")
        if isinstance(state_snapshot.get("field_provenance"), dict)
        else None
    )

    result = synthesize_v2_from_source_packet(
        source_text=source_text,
        job_description_text=jd_text,
        intake_notes_text=intake_notes,
        current_v2_draft=current_v2,
        field_provenance=current_provenance,
        geography=source_packet.get("geography")
        if isinstance(source_packet.get("geography"), str)
        else None,
        exemplar_block=exemplar_block,
        recruiter_preferences=preferences,
        session_id=session_id,
    )

    gap_questions = generate_gap_questions(
        v2_draft=result.v2_draft,
        field_provenance=result.field_provenance,
    )
    retrieval_meta = {
        "source": "brief_corpus",
        "used_brief_ids": used_ids,
        "exemplar_count": len(used_ids),
    }
    distillation = distill_brief(
        v2_draft=result.v2_draft,
        field_provenance=result.field_provenance,
        source_text=source_text,
        session_id=session_id,
    ).to_state_dict()

    role_value = (
        result.v2_draft.get("role_title")
        if isinstance(result.v2_draft, dict)
        else None
    )
    role_title = role_value if isinstance(role_value, str) and role_value else None

    return SynthesisProduct(
        v2_draft=result.v2_draft,
        v2_draft_polish_meta=result.to_polish_meta_dict(),
        field_provenance=result.field_provenance,
        gap_questions=gap_questions,
        retrieval_meta=retrieval_meta,
        distillation=distillation,
        role_title=role_title,
        intake_insights=dict(result.intake_insights or {}),
    )


# ---------------------------------------------------------------------
# Background task registry.
#
# One process-level dict keyed by ``session_id``. A new schedule replaces
# the prior entry but does not cancel the prior worker; the revision
# check on commit guarantees stale tasks are no-ops, so cancellation is
# unnecessary (and unsafe for an in-flight LLM request).
# ---------------------------------------------------------------------

_registry_lock = threading.Lock()
_active_threads: dict[int, threading.Thread] = {}


def schedule_source_packet_synthesis(
    *,
    session_id: int,
    expected_revision: int,
) -> threading.Thread:
    """Spawn a background thread to run synthesis for the session.

    Returns the thread handle. Tests can call :func:`wait_for_synthesis`
    to block until the registered worker finishes.

    A real :class:`threading.Thread` (rather than an asyncio task) makes
    this safe under both Uvicorn (long-lived event loop) and Starlette
    ``TestClient`` (per-request loops that end before background tasks
    finish).
    """

    thread = threading.Thread(
        target=_synthesis_worker,
        args=(session_id, expected_revision),
        name=f"cloris-synthesis-{session_id}-{expected_revision}",
        daemon=True,
    )
    with _registry_lock:
        _active_threads[session_id] = thread
    thread.start()
    return thread


def _synthesis_worker(session_id: int, expected_revision: int) -> None:
    from cloris import intake_sessions as _intake_sessions
    from cloris.api.intake import _intake_store

    try:
        store = _intake_store()
        snapshot = _intake_sessions.get_intake_session(
            store=store, session_id=session_id
        )
        if snapshot is None:
            log.info(
                "synthesis_worker session_gone session_id=%s revision=%s",
                session_id,
                expected_revision,
            )
            return
        snapshot_state = snapshot.get("state_json") or {}
        if not isinstance(snapshot_state, dict):
            snapshot_state = {}

        try:
            product = refresh_source_packet_artifacts_pure(
                state_snapshot=snapshot_state, session_id=session_id
            )
        except SynthesisInputError as exc:
            log.info(
                "synthesis_worker empty_packet session_id=%s revision=%s detail=%s",
                session_id,
                expected_revision,
                exc,
            )
            _commit_failure(
                session_id=session_id,
                expected_revision=expected_revision,
                error_copy=SYNTHESIS_EMPTY_PACKET_ERROR,
            )
            return
        except Exception as exc:  # noqa: BLE001 - worker must never raise
            log.warning(
                "synthesis_worker compute_failed session_id=%s revision=%s error=%s",
                session_id,
                expected_revision,
                exc,
                exc_info=True,
            )
            _commit_failure(
                session_id=session_id,
                expected_revision=expected_revision,
                error_copy=SYNTHESIS_GENERIC_ERROR,
            )
            return

        _commit_success(
            session_id=session_id,
            expected_revision=expected_revision,
            product=product,
        )
    finally:
        with _registry_lock:
            current = _active_threads.get(session_id)
            if current is threading.current_thread():
                _active_threads.pop(session_id, None)


def _commit_success(
    *,
    session_id: int,
    expected_revision: int,
    product: SynthesisProduct,
) -> None:
    from cloris import intake_sessions as _intake_sessions
    from cloris.api.intake import _intake_store

    store = _intake_store()
    latest = _intake_sessions.get_intake_session(store=store, session_id=session_id)
    if latest is None:
        log.info(
            "synthesis_worker commit_skip_session_gone session_id=%s",
            session_id,
        )
        return
    state = latest.get("state_json") or {}
    if not isinstance(state, dict):
        state = {}

    block = ensure_synthesis_state(state)
    current_revision = int(block.get("revision") or 0)
    if current_revision != expected_revision:
        log.info(
            "synthesis_worker stale_write_dropped session_id=%s "
            "expected_revision=%s current_revision=%s",
            session_id,
            expected_revision,
            current_revision,
        )
        return

    from shared.intake_conversation.insights import merge_intake_insights

    state["v2_draft"] = product.v2_draft
    state["v2_draft_polish_meta"] = product.v2_draft_polish_meta
    state["field_provenance"] = product.field_provenance
    state["gap_questions"] = product.gap_questions
    state["retrieval_meta"] = product.retrieval_meta
    state["distillation"] = product.distillation

    # Synthesis-vs-conversation merge: insights merge sub-key by sub-key
    # via the shared helper so a recruiter-corrected picture survives.
    if product.intake_insights:
        meta = (
            state.get("conversation_meta")
            if isinstance(state.get("conversation_meta"), dict)
            else {}
        )
        locks = meta.get("manually_edited_keys") if isinstance(meta.get("manually_edited_keys"), list) else []
        current_insights = (
            state.get("intake_insights")
            if isinstance(state.get("intake_insights"), dict)
            else {}
        )
        state["intake_insights"] = merge_intake_insights(
            current_insights,
            product.intake_insights,
            manually_edited_keys=[str(x) for x in locks],
        )

    block["status"] = SYNTHESIS_STATUS_READY
    block["error"] = None
    block["completed_at"] = _utc_now()

    _intake_sessions.patch_intake_session(
        store=store,
        session_id=session_id,
        state_json=state,
        role_title=product.role_title,
    )


def _commit_failure(
    *,
    session_id: int,
    expected_revision: int,
    error_copy: str,
) -> None:
    from cloris import intake_sessions as _intake_sessions
    from cloris.api.intake import _intake_store

    store = _intake_store()
    latest = _intake_sessions.get_intake_session(store=store, session_id=session_id)
    if latest is None:
        return
    state = latest.get("state_json") or {}
    if not isinstance(state, dict):
        state = {}

    block = ensure_synthesis_state(state)
    current_revision = int(block.get("revision") or 0)
    if current_revision != expected_revision:
        log.info(
            "synthesis_worker stale_failure_dropped session_id=%s "
            "expected_revision=%s current_revision=%s",
            session_id,
            expected_revision,
            current_revision,
        )
        return

    block["status"] = SYNTHESIS_STATUS_FAILED
    block["error"] = error_copy
    block["completed_at"] = _utc_now()

    _intake_sessions.patch_intake_session(
        store=store,
        session_id=session_id,
        state_json=state,
    )


def wait_for_synthesis(session_id: int, timeout: float = 30.0) -> bool:
    """Block until the registered synthesis worker for ``session_id`` finishes.

    Used by tests to deterministically wait for the background task.
    Returns ``True`` if the thread joined within ``timeout``, ``False``
    if no worker was registered or the join timed out.
    """

    with _registry_lock:
        thread = _active_threads.get(session_id)
    if thread is None:
        return False
    thread.join(timeout=timeout)
    return not thread.is_alive()


__all__ = [
    "SYNTHESIS_EMPTY_PACKET_ERROR",
    "SYNTHESIS_GENERIC_ERROR",
    "SYNTHESIS_OWNED_FIELDS",
    "SYNTHESIS_STATUS_FAILED",
    "SYNTHESIS_STATUS_IDLE",
    "SYNTHESIS_STATUS_READY",
    "SYNTHESIS_STATUS_RUNNING",
    "SynthesisInputError",
    "SynthesisProduct",
    "bump_synthesis_revision",
    "ensure_synthesis_state",
    "refresh_source_packet_artifacts_pure",
    "schedule_source_packet_synthesis",
    "wait_for_synthesis",
]
