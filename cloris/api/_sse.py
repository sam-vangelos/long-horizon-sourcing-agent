"""Shared SSE framing helpers.

Pre-Phase-C5 these lived in :mod:`cloris.api.conversation`. Lifted here
behavior-preserving so the conversational-intake endpoint
(:mod:`cloris.api.intake`) can reuse the same wire format without a
cross-module import that would muddy ownership.

Single function: :func:`sse_pack`. Same byte output as the previous
``_conversation_sse_pack`` — preserves field order (``event:`` before
``data:``), JSON serialization options (separators=(',', ':')), and
the trailing double blank lines that close an SSE frame.
"""

from __future__ import annotations

import json
from typing import Any


def sse_pack(event: str | None, payload: dict[str, Any]) -> str:
    """Format a Server-Sent Events frame.

    Output shape::

        event: <event>            (omitted when event is None/empty)
        data: <json-serialized payload>
        <blank line>
        <blank line>

    The trailing blank lines close the frame per the SSE spec; some
    proxies coalesce events without them.
    """

    chunks: list[str] = []
    if event:
        chunks.append(f"event: {event}")
    chunks.append(f"data: {json.dumps(payload, separators=(',', ':'))}")
    chunks.append("")
    chunks.append("")
    return "\n".join(chunks)


__all__ = ["sse_pack"]
