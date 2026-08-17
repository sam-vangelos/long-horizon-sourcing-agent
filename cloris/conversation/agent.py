"""Conversation agent — live-state Q&A in Cloris editorial register."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cloris.conversation import voice_discipline
from cloris.conversation.context import build_live_context_and_citations
from cloris.conversation.context import deterministic_status_answer
from cloris.conversation.prompts import (
    build_conversation_system_prompt,
    build_conversation_user_prompt,
)
from market_intelligence.briefing_polish import _has_llm_access
from shared import output_paths
from shared.llm_clients import cheap_llm
from shared.observability import observe
from shared.runtime_state.orchestration_store import OrchestrationStateStore


@dataclass(frozen=True)
class ConversationQueryResult:
    assistant_text: str
    kind: str
    degraded_reason: str | None = None
    citations_debug: list[dict[str, str]] | None = None


def _sanitize_prose_candidate(raw: str, *, deterministic_fallback: str) -> str:
    text = " ".join((raw or "").strip().split())
    if len(text) < 12:
        return deterministic_fallback
    violations = voice_discipline.voice_violations(text)
    if violations:
        return deterministic_fallback
    return text


class ConversationAgent:
    """Brief-scoped recruiter chat grounded in SQLite + JSONL telemetry."""

    def __init__(
        self,
        *,
        store: OrchestrationStateStore | None = None,
        state_root: Path | None = None,
    ) -> None:
        self._store = store or OrchestrationStateStore(
            output_paths.resolve_orchestration_db_path()
        )
        self._state_root = state_root

    @observe(name="conversation.query")
    def answer(
        self,
        *,
        brief_id: str,
        message: str,
        debug_citations: bool = False,
        history_limit: int = 24,
    ) -> ConversationQueryResult:
        """Return recruiter-facing prose; LLM faults use ``kind=\"degraded\"``."""

        ctx, citations = build_live_context_and_citations(
            brief_id.strip(), state_root=self._state_root
        )

        try:
            thread_id = self._store.get_or_create_conversation_thread(
                brief_id=brief_id.strip()
            )
            self._store.insert_conversation_turn(
                thread_id=thread_id,
                role="user",
                content=message.strip(),
                kind="ok",
            )
            prior = self._store.list_conversation_turns(
                thread_id=thread_id, limit=int(history_limit)
            )
            prior_for_prompt = [
                {"role": t["role"], "content": t["content"]} for t in prior[:-1]
            ]
        except Exception:
            return ConversationQueryResult(
                assistant_text="Conversation unavailable — runs continue.",
                kind="degraded",
                degraded_reason="persistence_error",
                citations_debug=citations if debug_citations else None,
            )

        det_text = deterministic_status_answer(ctx)

        if not _has_llm_access():
            body = _sanitize_prose_candidate(
                det_text,
                deterministic_fallback=(
                    "Cloris is processing — I don't have enough live signal "
                    "to answer precisely yet."
                ),
            )
            self._persist_assistant(thread_id, body, "ok")
            return ConversationQueryResult(
                assistant_text=body,
                kind="ok",
                citations_debug=citations if debug_citations else None,
            )

        system = build_conversation_system_prompt()
        user = build_conversation_user_prompt(
            recruiter_question=message,
            context_bundle=ctx,
            prior_turns=prior_for_prompt,
        )

        usage_context = {
            "stage": "conversation_query",
            "brief_id": brief_id.strip(),
            "thread_id": thread_id,
        }

        try:
            out = cheap_llm(
                system,
                user,
                expect_json=False,
                usage_context=usage_context,
            )
            raw_text = out if isinstance(out, str) else str(out)
        except Exception:
            degraded = ConversationQueryResult(
                assistant_text=(
                    "Cloris is processing — I couldn't finish that read just now."
                ),
                kind="degraded",
                degraded_reason="llm_raise",
                citations_debug=citations if debug_citations else None,
            )
            self._persist_assistant(
                thread_id, degraded.assistant_text, "degraded"
            )
            return degraded

        sanitized = _sanitize_prose_candidate(
            raw_text, deterministic_fallback=det_text
        )
        voice_cascade = bool(raw_text.strip()) and sanitized == det_text
        kind = "degraded" if voice_cascade else "ok"
        degraded_reason = "voice_or_empty_cascade" if voice_cascade else None

        result = ConversationQueryResult(
            assistant_text=sanitized,
            kind=kind,
            degraded_reason=degraded_reason,
            citations_debug=citations if debug_citations else None,
        )
        self._persist_assistant(
            thread_id,
            result.assistant_text,
            kind,
            trace_ref={"degraded_reason": degraded_reason}
            if degraded_reason
            else None,
        )
        return result

    def _persist_assistant(
        self,
        thread_id: int,
        content: str,
        kind: str,
        *,
        trace_ref: dict[str, Any] | None = None,
    ) -> None:
        self._store.insert_conversation_turn(
            thread_id=thread_id,
            role="assistant",
            content=content,
            kind=kind,
            trace_ref=trace_ref,
        )

    @observe(name="conversation.narrate_batch")
    def narrate_ambient_batch(
        self,
        *,
        brief_id: str,
        events_digest: dict[str, Any],
    ) -> str:
        """Produce one SSE narration line — fail-soft."""

        baseline = "Cloris is processing."
        if not _has_llm_access():
            return baseline

        ctx, _ = build_live_context_and_citations(
            brief_id.strip(), state_root=self._state_root
        )
        anchors = deterministic_status_answer(ctx)
        system = (
            "You are Cloris. Write one concise sentence about live sourcing "
            "telemetry (extend to two sentences only if strictly necessary). "
            "First-person editorial voice.\n"
            "Use DIGEST.events / DIGEST.lines only plus SUMMARY anchors for "
            "grounding — cite concrete event names.\n"
            "Forbidden chatter: apologies, reassurance, clichés "
            '("great question", "let me check", "hope this helps"). '
            "Plain prose only — no bullets, no JSON."
        )
        user_blob = json.dumps(
            {"digest": events_digest, "summary_anchor": anchors},
            indent=2,
            sort_keys=True,
        )
        usage_context = {
            "stage": "conversation_narrate_batch",
            "brief_id": brief_id.strip(),
        }

        try:
            out = cheap_llm(
                system,
                user_blob,
                expect_json=False,
                usage_context=usage_context,
            )
            raw_text = out if isinstance(out, str) else str(out)
        except Exception:
            return baseline

        sanitized = _sanitize_prose_candidate(
            raw_text, deterministic_fallback=baseline
        )
        return sanitized


__all__ = ["ConversationAgent", "ConversationQueryResult"]
