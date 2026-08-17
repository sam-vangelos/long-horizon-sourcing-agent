"""Background compose scheduler for intake conversation-to-brief recovery.

Companion to :func:`cloris.api.intake.compose_from_conversation_endpoint`.

Splits the long-running transcript composition off the HTTP request thread
so the route can return immediately with
``state_json.conversation_compose.status == "composing"``. Recruiters
poll (or use ``GET .../compose_jobs/current``) until status flips to
``ready`` or ``failed``.

Concurrency contract mirrors :mod:`cloris.api.intake_synthesis`: revision
guards drop stale worker commits; superseded registry entries do not
cancel in-flight LLM calls.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

log = logging.getLogger("cloris.api.intake_compose")

COMPOSE_STATUS_IDLE = "idle"
COMPOSE_STATUS_COMPOSING = "composing"
COMPOSE_STATUS_READY = "ready"
COMPOSE_STATUS_FAILED = "failed"

# Recruiter-safe error copy — never include raw stack traces here.
COMPOSE_GENERIC_ERROR = (
    "We couldn't build the brief from this conversation. "
    "Try again or add more detail about the role."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_compose_state(state: dict[str, Any]) -> dict[str, Any]:
    """Idempotently initialize ``state_json.conversation_compose``."""

    block = state.get("conversation_compose")
    if not isinstance(block, dict):
        block = {
            "status": COMPOSE_STATUS_IDLE,
            "error": None,
            "started_at": None,
            "completed_at": None,
            "revision": 0,
            "result": None,
        }
        state["conversation_compose"] = block
    block.setdefault("status", COMPOSE_STATUS_IDLE)
    block.setdefault("error", None)
    block.setdefault("started_at", None)
    block.setdefault("completed_at", None)
    block.setdefault("revision", 0)
    if "result" not in block:
        block["result"] = None
    return block


def bump_compose_revision(state: dict[str, Any]) -> int:
    """Increment ``conversation_compose.revision`` and return the new value."""

    block = ensure_compose_state(state)
    next_revision = int(block.get("revision") or 0) + 1
    block["revision"] = next_revision
    return next_revision


@dataclass(frozen=True)
class ComposeProduct:
    """One compose pass's product — committed under the revision guard."""

    compose_status: Literal["composed", "deficits"]
    deficits: list[dict[str, str]]
    missing_keys: list[str]
    invalid_keys: list[str]
    insight_deficits: list[dict[str, str]]
    v2_draft: dict[str, Any]
    insight_updates: dict[str, Any]
    metadata: dict[str, Any]
    role_title: str | None
    current_step: str | None
    conversation_meta: dict[str, Any] | None


def compose_from_conversation_pure(
    *,
    state_snapshot: dict[str, Any],
    session_role_title: str | None,
    session_id: int,
) -> ComposeProduct:
    """Compute one compose pass without touching durable state."""

    from shared.intake_conversation.composer import compose_from_conversation
    from shared.intake_conversation.insights import (
        HIRING_MANAGER_PICTURE_KEY,
        HIRING_MANAGER_PICTURE_LOCK_PATH,
    )

    messages = (
        state_snapshot.get("messages")
        if isinstance(state_snapshot.get("messages"), list)
        else []
    )
    typed_messages = [m for m in messages if isinstance(m, dict)]
    current_v2 = (
        state_snapshot.get("v2_draft")
        if isinstance(state_snapshot.get("v2_draft"), dict)
        else {}
    )
    current_insights = (
        state_snapshot.get("intake_insights")
        if isinstance(state_snapshot.get("intake_insights"), dict)
        else {}
    )
    meta = (
        state_snapshot.get("conversation_meta")
        if isinstance(state_snapshot.get("conversation_meta"), dict)
        else {}
    )
    locks = (
        meta.get("manually_edited_keys")
        if isinstance(meta.get("manually_edited_keys"), list)
        else []
    )
    locks_set = {str(x) for x in locks}
    source_packet = (
        state_snapshot.get("source_packet")
        if isinstance(state_snapshot.get("source_packet"), dict)
        else {}
    )

    result = compose_from_conversation(
        messages=typed_messages,
        current_v2_draft=current_v2,
        source_packet=source_packet,
        manually_edited_keys=list(locks_set),
        role_title_hint=session_role_title
        if isinstance(session_role_title, str)
        else None,
        session_id=session_id,
        current_intake_insights=current_insights,
    )

    role_title: str | None = None
    current_step: str | None = None
    updated_meta: dict[str, Any] | None = None

    if result.status == "composed":
        current_step = "review"
        role_value = result.v2_draft.get("role_title")
        role_title = role_value if isinstance(role_value, str) and role_value else None

    picture_update = (result.insight_updates or {}).get(HIRING_MANAGER_PICTURE_KEY)
    if (
        isinstance(picture_update, dict)
        and bool(picture_update.get("corrected_by_recruiter"))
        and HIRING_MANAGER_PICTURE_LOCK_PATH not in locks_set
    ):
        updated_meta = dict(meta)
        updated_meta["manually_edited_keys"] = sorted(
            locks_set | {HIRING_MANAGER_PICTURE_LOCK_PATH}
        )

    return ComposeProduct(
        compose_status=result.status,
        deficits=list(result.deficits),
        missing_keys=list(result.missing_keys),
        invalid_keys=list(result.invalid_keys),
        insight_deficits=list(result.insight_deficits),
        v2_draft=result.v2_draft,
        insight_updates=dict(result.insight_updates or {}),
        metadata=dict(result.metadata or {}),
        role_title=role_title,
        current_step=current_step,
        conversation_meta=updated_meta,
    )


def should_run_compose_synchronously() -> bool:
    """Return True when compose should finish before the HTTP response returns."""

    if os.getenv("CLORIS_CERTIFY", "").strip().lower() in {"1", "true", "yes"}:
        return True
    if os.getenv("CLORIS_CERTIFY_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    if os.getenv("CLORIS_DISABLE_INTAKE_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return True
    from market_intelligence.briefing_polish import _has_llm_access

    return not _has_llm_access()


_registry_lock = threading.Lock()
_active_threads: dict[int, threading.Thread] = {}


def schedule_compose_from_conversation(
    *,
    session_id: int,
    expected_revision: int,
) -> threading.Thread:
    """Spawn a background thread to compose the brief for the session."""

    thread = threading.Thread(
        target=_compose_worker,
        args=(session_id, expected_revision),
        name=f"cloris-compose-{session_id}-{expected_revision}",
        daemon=True,
    )
    with _registry_lock:
        _active_threads[session_id] = thread
    thread.start()
    return thread


def run_compose_worker_inline(session_id: int, expected_revision: int) -> None:
    """Run the compose worker on the caller thread (cert / deterministic tests)."""

    _compose_worker(session_id, expected_revision)


def _compose_worker(session_id: int, expected_revision: int) -> None:
    from cloris import intake_sessions as _intake_sessions
    from cloris.api.intake import _intake_store

    try:
        store = _intake_store()
        snapshot = _intake_sessions.get_intake_session(
            store=store, session_id=session_id
        )
        if snapshot is None:
            log.info(
                "compose_worker session_gone session_id=%s revision=%s",
                session_id,
                expected_revision,
            )
            return
        snapshot_state = snapshot.get("state_json") or {}
        if not isinstance(snapshot_state, dict):
            snapshot_state = {}

        try:
            product = compose_from_conversation_pure(
                state_snapshot=snapshot_state,
                session_role_title=snapshot.get("role_title")
                if isinstance(snapshot.get("role_title"), str)
                else None,
                session_id=session_id,
            )
        except Exception as exc:  # noqa: BLE001 - worker must never raise
            log.warning(
                "compose_worker compute_failed session_id=%s revision=%s error=%s",
                session_id,
                expected_revision,
                exc,
                exc_info=True,
            )
            _commit_failure(
                session_id=session_id,
                expected_revision=expected_revision,
                error_copy=COMPOSE_GENERIC_ERROR,
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
    product: ComposeProduct,
) -> None:
    from cloris import intake_sessions as _intake_sessions
    from cloris.api.intake import _intake_store
    from shared.v2_draft_undo import push_v2_undo

    store = _intake_store()
    latest = _intake_sessions.get_intake_session(store=store, session_id=session_id)
    if latest is None:
        log.info(
            "compose_worker commit_skip_session_gone session_id=%s",
            session_id,
        )
        return
    state = latest.get("state_json") or {}
    if not isinstance(state, dict):
        state = {}

    block = ensure_compose_state(state)
    current_revision = int(block.get("revision") or 0)
    if current_revision != expected_revision:
        log.info(
            "compose_worker stale_write_dropped session_id=%s "
            "expected_revision=%s current_revision=%s",
            session_id,
            expected_revision,
            current_revision,
        )
        return

    state["conversation_compose_meta"] = product.metadata
    if product.compose_status == "composed":
        current_v2 = state.get("v2_draft") if isinstance(state.get("v2_draft"), dict) else {}
        if isinstance(current_v2, dict):
            push_v2_undo(state)
        state["v2_draft"] = product.v2_draft

    if product.insight_updates:
        from shared.intake_conversation.insights import merge_intake_insights

        current_insights = (
            state.get("intake_insights")
            if isinstance(state.get("intake_insights"), dict)
            else {}
        )
        meta = (
            state.get("conversation_meta")
            if isinstance(state.get("conversation_meta"), dict)
            else {}
        )
        locks = (
            meta.get("manually_edited_keys")
            if isinstance(meta.get("manually_edited_keys"), list)
            else []
        )
        state["intake_insights"] = merge_intake_insights(
            current_insights,
            product.insight_updates,
            manually_edited_keys={str(x) for x in locks},
        )

    if product.conversation_meta is not None:
        state["conversation_meta"] = product.conversation_meta

    block["status"] = COMPOSE_STATUS_READY
    block["error"] = None
    block["completed_at"] = _utc_now()
    block["result"] = _compose_result_payload_from_product(product)

    _intake_sessions.patch_intake_session(
        store=store,
        session_id=session_id,
        current_step=product.current_step,
        state_json=state,
        role_title=product.role_title,
    )


def _compose_result_payload_from_product(product: ComposeProduct) -> dict[str, Any]:
    return {
        "compose_status": product.compose_status,
        "deficits": product.deficits,
        "missing_keys": product.missing_keys,
        "invalid_keys": product.invalid_keys,
        "insight_deficits": product.insight_deficits,
    }


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

    block = ensure_compose_state(state)
    current_revision = int(block.get("revision") or 0)
    if current_revision != expected_revision:
        log.info(
            "compose_worker stale_failure_dropped session_id=%s "
            "expected_revision=%s current_revision=%s",
            session_id,
            expected_revision,
            current_revision,
        )
        return

    block["status"] = COMPOSE_STATUS_FAILED
    block["error"] = error_copy
    block["completed_at"] = _utc_now()
    block["result"] = None

    _intake_sessions.patch_intake_session(
        store=store,
        session_id=session_id,
        state_json=state,
    )


def wait_for_compose(session_id: int, timeout: float = 30.0) -> bool:
    """Block until the registered compose worker for ``session_id`` finishes."""

    with _registry_lock:
        thread = _active_threads.get(session_id)
    if thread is None:
        return False
    thread.join(timeout=timeout)
    return not thread.is_alive()


__all__ = [
    "COMPOSE_GENERIC_ERROR",
    "COMPOSE_STATUS_COMPOSING",
    "COMPOSE_STATUS_FAILED",
    "COMPOSE_STATUS_IDLE",
    "COMPOSE_STATUS_READY",
    "ComposeProduct",
    "bump_compose_revision",
    "compose_from_conversation_pure",
    "ensure_compose_state",
    "run_compose_worker_inline",
    "schedule_compose_from_conversation",
    "should_run_compose_synchronously",
    "wait_for_compose",
]
