"""Conversational intake orchestrator package.

Replaces the deterministic chapter-wizard intake at ``#/brief/new`` with a
streaming conversational orchestrator. See ``plans/conversational-intake.md``
and ``~/.cursor/plans/conversational_intake_implementation_*.plan.md`` for
the design contract.

Structure:

- :mod:`shared.intake_conversation.state` — pure helpers that mutate
  ``intake_sessions.state_json`` in well-typed, repo-grep-able ways.
- :mod:`shared.intake_conversation.orchestrator` — Opus streaming turn loop
  (one Opus call per recruiter message, prompt-cached system block).
- :mod:`shared.intake_conversation.extractor` — cheap_llm pass that pulls
  structured updates out of the dialogue and merges them into ``v2_draft``.
- :mod:`shared.intake_conversation.sufficiency` — deterministic check that
  decides whether the brief is filed-able.
- :mod:`shared.intake_conversation.prompts` — system + user prompt builders;
  voice content per ``~/Downloads/Sam_Vangelos_Voice_Guide_v2.md``.

Cap constants are exported here (rather than in a sub-module) so they can
be patched from one well-known location during tests + tuned via env vars
without touching code.
"""

from __future__ import annotations

import os
from typing import Any, Literal, NotRequired, TypedDict


class ConversationMessage(TypedDict, total=False):
    """One turn in the conversational intake transcript.

    Stored inside ``intake_sessions.state_json["messages"]`` as a JSON list.
    Roles are ``"cloris"`` and ``"recruiter"`` (not ``"assistant"`` /
    ``"user"``) so the transcripts are grep-friendly across the repo and
    don't collide with the existing ``cloris/api/conversation.py`` chat
    surface, which is a separate concept (run-time narration, not intake).

    ``meta`` is optional and used for per-message debugging context — cost
    in USD on Cloris turns, model name, whether the message was a fallback
    after an LLM exception, and whether the previous turn was dropped
    mid-stream. ``meta`` is never required for protocol correctness;
    callers must tolerate its absence.
    """

    role: Literal["cloris", "recruiter"]
    content: str
    ts: str  # ISO-8601 UTC, matches `cloris.intake_sessions._utc_now`
    meta: NotRequired[dict[str, Any]]


# Soft / hard caps. Tuneable via env vars for the Northwind trial. The orchestrator
# system prompt swaps in a soft-cap reminder at SOFT_CAP_TURNS / SOFT_CAP_USD
# and forces composition at HARD_CAP_TURNS / HARD_CAP_USD. Mirrors the
# explicit-numeric-constants pattern from `shared/governor.py:16-27`.
SOFT_CAP_TURNS: int = int(os.environ.get("INTAKE_SOFT_CAP_TURNS", "20"))
SOFT_CAP_USD: float = float(os.environ.get("INTAKE_SOFT_CAP_USD", "1.0"))
HARD_CAP_TURNS: int = int(os.environ.get("INTAKE_HARD_CAP_TURNS", "40"))
HARD_CAP_USD: float = float(os.environ.get("INTAKE_HARD_CAP_USD", "2.0"))


def cap_state_for(
    *, turn_count: int, cost_usd_running_total: float
) -> str:
    """Classify a conversation's cap state.

    Returns one of:

    - ``"normal"`` — under the soft cap on both turns and cost.
    - ``"soft"`` — at or above the soft cap on either turns or cost,
      but still under the hard cap. The orchestrator's system prompt
      gets a soft-cap reminder block; Cloris is told to take a natural
      close if one is near, but doesn't force composition.
    - ``"hard"`` — at or above the hard cap on either turns or cost.
      The orchestrator's system prompt forces a wrap; the C5 endpoint
      additionally forces ``is_ready_to_compose`` to ``(True, [])``
      regardless of v2_draft state, so the recruiter sees the brief
      after this turn.

    Mirrors :func:`shared.governor.GovernorLimitReached`'s explicit-
    numeric-constants pattern. Caps are env-tunable via the four
    ``INTAKE_*_CAP_*`` constants exported above.
    """

    if turn_count >= HARD_CAP_TURNS or cost_usd_running_total >= HARD_CAP_USD:
        return "hard"
    if turn_count >= SOFT_CAP_TURNS or cost_usd_running_total >= SOFT_CAP_USD:
        return "soft"
    return "normal"


__all__ = [
    "ConversationMessage",
    "SOFT_CAP_TURNS",
    "SOFT_CAP_USD",
    "HARD_CAP_TURNS",
    "HARD_CAP_USD",
    "cap_state_for",
]
