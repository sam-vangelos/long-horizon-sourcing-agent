"""System + user prompts for the brief-detail conversation companion.

Voice extracts inherit the chief-of-staff register from current IA doctrine; the
JSON contract differs — this path returns short prose, not synthesis JSON.
"""

from __future__ import annotations

import json
from typing import Any

from cloris.chief_of_staff.prompts import build_chief_of_staff_system_prompt


def build_conversation_system_prompt() -> str:
    """Layer recruiter-QA constraints on top of the editorial voice block."""

    base = build_chief_of_staff_system_prompt()
    return f"""{base}

MODE — Brief companion (recruiter asks; you answer in running prose):
- You respond in 1–4 short sentences, first person, narrating what Cloris is doing and what the telemetry shows.
- The recruiter's question is labeled in the user message. Answer that question directly using ONLY the structured CONTEXT values.
- Do NOT address the recruiter with flattery or assistant tropes (no "Great question", "I'd be happy to", "Let me check", "I hope this helps").
- Do NOT invent run ids, counts, or statuses not present in CONTEXT.
- Translate engine identifiers the same way as the chief-of-staff rules: humanize underscores, never paste snake_case lane keys.
Return PLAIN PROSE ONLY — no JSON, no markdown fences, no numbered lists unless the question truly requires enumeration."""


def build_conversation_user_prompt(
    *,
    recruiter_question: str,
    context_bundle: dict[str, Any],
    prior_turns: list[dict[str, str]],
) -> str:
    prior_lines = [
        f"{t['role'].upper()}: {t['content']}"
        for t in prior_turns[-12:]
        if t.get("content")
    ]
    prior_block = "\n".join(prior_lines) if prior_lines else "(no prior turns)"
    dumped = json.dumps(context_bundle, indent=2, sort_keys=True)
    return (
        f"RECRUITER QUESTION:\n{recruiter_question.strip()}\n\n"
        f"CONTEXT (structured — cite only values present here):\n{dumped}\n\n"
        f"PRIOR THREAD (same brief):\n{prior_block}\n"
    )


__all__ = [
    "build_conversation_system_prompt",
    "build_conversation_user_prompt",
]
