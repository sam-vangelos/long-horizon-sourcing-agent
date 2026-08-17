"""Exec-search dossier full evaluation."""

from __future__ import annotations

import re
from typing import Any, Callable

from shared.brief_loader import Brief
from shared.schemas import CandidateProfileSummary, OpusDecision


LLMCaller = Callable[[str, str], str | dict[str, Any]]


def exec_search_full_judge(
    *,
    candidate: CandidateProfileSummary,
    brief: Brief,
    dossier_prompt_body: str,
    llm_caller: LLMCaller | None = None,
) -> OpusDecision:
    system = _system_prompt(brief)
    if llm_caller is None:
        from shared.llm_clients import opus_llm_cached

        def _default(system_prompt: str, user_prompt: str) -> str:
            usage_context = {
                "stage": "exec_search_full_judge",
                "source": "exec_search",
                "candidate_name": candidate.name,
                "profile_url": candidate.profile_url,
                "brief_id": getattr(brief, "id", None),
                "role_title": getattr(brief, "role_title", ""),
            }
            return str(
                opus_llm_cached(
                    system_prompt,
                    user_prompt,
                    expect_json=False,
                    max_tokens=4096,
                    usage_context=usage_context,
                )
            )

        llm_caller = _default
    raw = llm_caller(system, dossier_prompt_body)
    decision, path, confidence, rationale = _parse_response(raw)
    return OpusDecision(
        stage="full",
        decision=decision,
        path=path,
        confidence=confidence,
        rationale=rationale,
        candidate_name=candidate.name,
        profile_url=candidate.profile_url,
    )


def _system_prompt(brief: Brief) -> str:
    return f"""You are evaluating an executive-search dossier against a hiring thesis.

Role: {brief.role_title}
Minimum bar: {brief.minimum_bar}

Use the candidate profile plus public-web signals. Reward clear scope,
company/stage fit, leadership trajectory, and thesis-specific evidence.
Reject thin, off-scope, or title-inflated profiles.

Output exact fields:
DECISION: <SAVE|REJECT|INFERENTIAL_SAVE|SIGNAL_SAVE>
PATH: <company_scope|career_path|market_thesis|none>
CONFIDENCE: <0.0-1.0>
DOSSIER_RATIONALE: <two concise paragraphs grounded in the dossier>
"""


def _parse_response(raw: str | dict[str, Any]) -> tuple[str, str, float, str]:
    if isinstance(raw, dict):
        decision = str(raw.get("decision") or "").strip().upper()
        path = str(raw.get("path") or "exec_search_dossier").strip()
        confidence_raw = raw.get("confidence", 0.0)
        rationale = str(raw.get("rationale") or raw.get("dossier_rationale") or "").strip()
    else:
        text = str(raw)
        decision = _line_value(text, "DECISION").upper()
        path = _line_value(text, "PATH") or "exec_search_dossier"
        confidence_raw = _line_value(text, "CONFIDENCE") or "0.0"
        rationale = _multiline_value(text, "DOSSIER_RATIONALE") or text.strip()
    if decision not in {"SAVE", "REJECT", "INFERENTIAL_SAVE", "SIGNAL_SAVE"}:
        decision = "REJECT"
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    return decision, path, confidence, rationale or "No dossier rationale returned."


def _line_value(text: str, label: str) -> str:
    match = re.search(rf"^{label}\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _multiline_value(text: str, label: str) -> str:
    match = re.search(
        rf"^{label}\s*:\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""
