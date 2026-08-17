"""Conversation query, mute, and ambient SSE HTTP routes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator

from fastapi import Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from cloris.models import (
    ConversationCitationDebug,
    ConversationMuteRequest,
    ConversationMuteResponse,
    ConversationQueryRequest,
    ConversationQueryResponse,
)

from ._sse import sse_pack as _conversation_sse_pack
from .routing import router

log = logging.getLogger("cloris.api")

_CONVERSATION_QUERY_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_CONVERSATION_QUERY_LOCK = threading.Lock()
_CONVERSATION_QUERY_LIMIT = 10
_CONVERSATION_QUERY_WINDOW_S = 60.0


def _conversation_query_rate_blocked(brief_id: str) -> bool:
    """Return True when the next query for ``brief_id`` must yield HTTP 429."""

    now = time.monotonic()
    bid = brief_id.strip()
    with _CONVERSATION_QUERY_LOCK:
        dq = _CONVERSATION_QUERY_BUCKETS[bid]
        while dq and now - dq[0] > _CONVERSATION_QUERY_WINDOW_S:
            dq.popleft()
        if len(dq) >= _CONVERSATION_QUERY_LIMIT:
            return True
        dq.append(now)
        return False


@router.post(
    "/api/conversation/{brief_id}/query",
    response_model=ConversationQueryResponse,
)
def api_conversation_query(
    brief_id: str,
    body: ConversationQueryRequest,
    x_cloris_conversation_debug: str | None = Header(
        None, alias="X-Cloris-Conversation-Debug"
    ),
) -> ConversationQueryResponse:
    """Recruiter Q&A grounded in telemetry (read-only companion)."""

    if _conversation_query_rate_blocked(brief_id.strip()):
        raise HTTPException(status_code=429, detail="Cloris needs a moment.")

    dbg = (x_cloris_conversation_debug or "").strip() == "1"
    from cloris.conversation.agent import ConversationAgent

    agent = ConversationAgent()
    result = agent.answer(
        brief_id=brief_id,
        message=body.message,
        debug_citations=dbg,
    )
    citations = None
    if dbg and result.citations_debug:
        citations = [
            ConversationCitationDebug(
                source=c["source"],
                state_key=c["state_key"],
                signal_ref=c["signal_ref"],
            )
            for c in result.citations_debug
        ]
    return ConversationQueryResponse(
        assistant_text=result.assistant_text,
        kind=result.kind,  # type: ignore[arg-type]
        degraded_reason=result.degraded_reason,
        citations_debug=citations,
    )


@router.patch(
    "/api/conversation/{brief_id}/mute",
    response_model=ConversationMuteResponse,
)
def api_conversation_mute_patch(
    brief_id: str, request_body: ConversationMuteRequest
) -> ConversationMuteResponse:
    from shared.output_paths import resolve_orchestration_db_path
    from shared.runtime_state.orchestration_store import OrchestrationStateStore

    store = OrchestrationStateStore(resolve_orchestration_db_path())
    store.set_conversation_ambient_muted(
        brief_id=brief_id.strip(),
        muted=request_body.ambient_muted,
    )
    muted = store.get_conversation_ambient_muted(brief_id=brief_id.strip())
    return ConversationMuteResponse(
        brief_id=brief_id.strip(), ambient_muted=muted
    )


@router.get(
    "/api/conversation/{brief_id}/mute",
    response_model=ConversationMuteResponse,
)
def api_conversation_mute_get(brief_id: str) -> ConversationMuteResponse:
    from shared.output_paths import resolve_orchestration_db_path
    from shared.runtime_state.orchestration_store import OrchestrationStateStore

    store = OrchestrationStateStore(resolve_orchestration_db_path())
    muted = store.get_conversation_ambient_muted(brief_id=brief_id.strip())
    return ConversationMuteResponse(
        brief_id=brief_id.strip(), ambient_muted=muted
    )


@router.get("/api/conversation/{brief_id}/stream")
async def api_conversation_stream(
    brief_id: str, request: Request
) -> StreamingResponse:
    """Server-Sent Events: ambient narration for an active brief."""

    disabled = os.getenv(
        "CLORIS_CONVERSATION_SSE_DISABLED", ""
    ).strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        raise HTTPException(
            status_code=404, detail={"error": "sse_disabled"}
        )

    async def gen() -> AsyncIterator[str]:
        last_ping = time.monotonic()
        yield _conversation_sse_pack("ping", {"ts": last_ping})

        from cloris.conversation.agent import ConversationAgent
        from cloris.conversation.cost_governor import (
            BATCH_DEBOUNCE_S,
            default_narration_governor,
        )
        from cloris.conversation.event_source import (
            build_tailers_for_brief,
            poll_significant_events,
            summarize_significant_batches,
        )
        from shared.output_paths import resolve_orchestration_db_path
        from shared.runtime_state.orchestration_store import OrchestrationStateStore

        bid = brief_id.strip()
        store_inst = OrchestrationStateStore(resolve_orchestration_db_path())
        narrator = ConversationAgent(store=store_inst)
        cursors = build_tailers_for_brief(bid)
        for cursor in cursors.values():
            pth = cursor.path
            try:
                if pth.is_file():
                    cursor.offset_bytes = pth.stat().st_size
            except OSError:
                cursor.offset_bytes = 0
        gov = default_narration_governor()
        buffer_events: list = []
        degraded_sent = False
        deadline: float | None = None
        while True:
            if await request.is_disconnected():
                break
            try:
                now = time.monotonic()
                if now - last_ping >= 15.0:
                    last_ping = now
                    yield _conversation_sse_pack("ping", {"ts": now})
                buffer_events.extend(poll_significant_events(cursors))
                if store_inst.get_conversation_ambient_muted(brief_id=bid):
                    buffer_events.clear()
                    deadline = None
                    await asyncio.sleep(3.0)
                    continue
                if buffer_events and deadline is None:
                    deadline = time.monotonic() + BATCH_DEBOUNCE_S
                await asyncio.sleep(3.0)
                now = time.monotonic()
                if (
                    not buffer_events
                    or deadline is None
                    or now < deadline
                ):
                    continue
                pending = buffer_events
                buffer_events = []
                deadline = None

                gate = gov.allow_call(bid)
                if not gate.allowed:
                    yield _conversation_sse_pack(
                        "narration",
                        {
                            "text": "",
                            "kind": "suppressed",
                            "suppressed_reason": gate.suppressed_reason,
                        },
                    )
                    continue
                digest = summarize_significant_batches(pending)
                try:
                    line = narrator.narrate_ambient_batch(
                        brief_id=bid,
                        events_digest=digest,
                    )
                except Exception as exc:
                    log.warning(
                        "conversation ambient narration failed brief_id=%s error=%s",
                        bid,
                        exc,
                        exc_info=True,
                    )
                    line = ""
                if await request.is_disconnected():
                    break
                if not line.strip():
                    if not degraded_sent:
                        degraded_sent = True
                        log.info(
                            "conversation SSE degraded brief_id=%s reason=empty_narration",
                            bid,
                        )
                        yield _conversation_sse_pack(
                            "degraded",
                            {"message": "Cloris is processing."},
                        )
                    continue
                degraded_sent = False
                tid = store_inst.get_or_create_conversation_thread(
                    brief_id=bid,
                )
                store_inst.insert_conversation_turn(
                    thread_id=tid,
                    role="assistant",
                    content=line,
                    kind="ambient",
                )
                yield _conversation_sse_pack(
                    "narration",
                    {"text": line, "kind": "ambient"},
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "conversation SSE degraded after error brief_id=%s error=%s",
                    bid,
                    exc,
                    exc_info=True,
                )
                yield _conversation_sse_pack(
                    "degraded",
                    {"message": "Cloris is processing."},
                )
                await asyncio.sleep(5.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
