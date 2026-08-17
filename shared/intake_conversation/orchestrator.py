"""Streaming orchestrator: one Opus call per recruiter turn.

The orchestrator wraps :func:`shared.llm_clients.opus_llm_cached_stream`
with the conversational-intake system + user prompt builders, and
provides the failure-mode contract the C5 SSE endpoint depends on:

- Yields ``("delta", text)`` for each token chunk.
- Yields exactly one ``("usage", usage_dict)`` at end-of-stream.
- On exception **before** any delta: yields a single
  ``("degraded", {"reason": "provider_failed", "any_delta": False})``
  marker plus a synthetic usage tuple. No fallback delta is emitted
  — the C5 endpoint translates the marker into a structurally distinct
  ``degraded`` SSE event so the recruiter sees a recoverable banner
  rather than a normal-shaped Cloris turn that says
  "Lost my train of thought" (audit finding F-2).
- On exception **mid-stream** (after some deltas already streamed):
  yields a ``("delta", LLM_PARTIAL_INTERRUPT)`` so the partial Cloris
  turn ends with a recruiter-readable cutoff phrase, then a
  ``("degraded", {"any_delta": True, ...})`` marker, then the synthetic
  usage tuple. The endpoint commits the partial Cloris turn AND emits
  the degraded SSE event so the recruiter knows the turn was cut short.

The orchestrator does NOT open a :func:`shared.llm_usage.llm_usage_session`
context — that responsibility belongs to the C5 endpoint, which scopes
the cost log to the session id.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from shared.intake_conversation import ConversationMessage
from shared.intake_conversation.prompts import (
    LLM_PARTIAL_INTERRUPT,
    build_orchestrator_system_prompt,
    build_orchestrator_user_prompt,
)
from shared.llm_clients import opus_llm_cached_stream


log = logging.getLogger(__name__)


# Synthetic usage payload returned when the stream errors. Zero token
# counts so the C5 cost-rollup adds nothing for the failed turn — the
# fallback message is free; the recruiter doesn't pay for our outage.
_EMPTY_USAGE: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
}


# Wire reason for the degraded marker. Kept as a module constant so the
# C5 endpoint and tests can import the same string.
DEGRADED_REASON_PROVIDER_FAILED = "provider_failed"


def stream_next_turn(
    *,
    messages: list[ConversationMessage],
    v2_draft: dict[str, Any],
    source_packet: dict[str, Any] | None,
    sufficiency_state: tuple[bool, list[str]],
    dropped_turn: bool = False,
    cap_state: Literal["normal", "soft", "hard"] = "normal",
    session_id: int | None = None,
):
    """Stream Cloris's next turn as ``(kind, payload)`` tuples.

    Wrap in :func:`shared.llm_usage.llm_usage_session` at the call site::

        with llm_usage_session(
            log_path=...,
            stage="intake_conversation_orchestrator",
            session_id=session_id,
        ):
            for kind, payload in stream_next_turn(...):
                ...

    ``session_id`` is accepted for future telemetry stitching but is
    currently unused inside the function — it travels through the
    ``usage_context`` of the underlying LLM call only via the active
    :func:`llm_usage_session` (sessions stash session-scoped context for
    every record_llm_usage call inside the block).
    """

    system_prompt = build_orchestrator_system_prompt(
        sufficiency_state=sufficiency_state,
        dropped_turn=dropped_turn,
        cap_state=cap_state,
    )
    user_prompt = build_orchestrator_user_prompt(
        messages=messages,
        v2_draft=v2_draft,
        source_packet=source_packet,
    )

    usage_context = {
        "stage": "intake_conversation_orchestrator",
        "session_id": session_id,
        "cap_state": cap_state,
        "dropped_turn": dropped_turn,
    }

    any_delta = False
    try:
        for kind, payload in opus_llm_cached_stream(
            system_prompt,
            user_prompt,
            usage_context=usage_context,
        ):
            if kind == "delta":
                any_delta = True
            yield (kind, payload)
    except Exception as exc:  # noqa: BLE001 — orchestrator MUST NOT propagate
        # Provider failures are a UX problem the C5 endpoint must
        # surface as a structurally distinct ``degraded`` SSE event
        # (audit finding F-2). The orchestrator's contract is
        # "always yield something usable" — but "usable" here means
        # the endpoint can distinguish a failed turn from a normal
        # Cloris turn. We therefore emit a ``degraded`` marker rather
        # than a fallback delta so the recruiter never sees a
        # full-shaped Cloris turn that says "Lost my train of thought".
        # The endpoint owns the recruiter-facing copy.
        log.warning(
            "intake_conversation_orchestrator stream failed "
            "session_id=%s any_delta=%s exc=%s",
            session_id,
            any_delta,
            exc,
            exc_info=True,
        )
        if any_delta:
            # Partial output: terminate the partial Cloris turn with a
            # plain-prose interrupt phrase so the persisted transcript
            # reads cleanly, then mark the turn degraded.
            yield ("delta", LLM_PARTIAL_INTERRUPT)
        yield (
            "degraded",
            {
                "reason": DEGRADED_REASON_PROVIDER_FAILED,
                "any_delta": any_delta,
            },
        )
        yield ("usage", dict(_EMPTY_USAGE))


__all__ = ["stream_next_turn", "DEGRADED_REASON_PROVIDER_FAILED"]
