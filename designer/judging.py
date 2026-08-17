"""Designer text-stage judgment helpers."""

from __future__ import annotations

import re
from typing import Any, Callable

from designer.judgment_templates import (
    assemble_designer_facial_system,
    assemble_designer_full_system,
)
from designer.schemas import DesignerCandidate, DesignerSnippet
from shared.receipts import ReceiptStatus, build_receipt
from shared.schemas import OpusDecision


LLMCaller = Callable[[str, str], str | dict[str, Any]]

_FACIAL_DECISIONS = {"FACIAL_YES", "FACIAL_NO", "FACIAL_BORDERLINE"}
_FULL_DECISIONS = {
    "SAVE",
    "REJECT",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
}


def designer_facial_judge(
    snippet: DesignerSnippet,
    *,
    brief: dict[str, Any],
    llm_caller: LLMCaller | None = None,
) -> OpusDecision:
    system = assemble_designer_facial_system(brief)
    user = _snippet_user_prompt(snippet)
    if llm_caller is None:
        from shared.llm_clients import facial_llm

        def _default(system_prompt: str, user_prompt: str) -> str:
            usage_context = {
                "stage": "designer_facial_judge",
                "source": "designer",
                "candidate_name": snippet.display_name,
                "profile_url": snippet.profile_url,
                "brief_id": brief.get("id") or brief.get("role_title"),
            }
            return str(
                facial_llm(
                    system_prompt,
                    user_prompt,
                    expect_json=False,
                    max_tokens=1024,
                    usage_context=usage_context,
                )
            )

        llm_caller = _default
    raw = llm_caller(system, user)
    decision, rationale = _parse_facial_response(raw)
    parse_status = "ok" if decision in _FACIAL_DECISIONS else "parse_fail"
    if parse_status != "ok" and _looks_like_model_refusal(raw):
        parse_status = "refused"
        rationale = "REFUSED: model declined to judge designer facial fit."
    if parse_status != "ok":
        decision = "PARSE_FAILURE"
    return OpusDecision(
        stage="facial",
        decision=decision,
        path="designer_text_triage" if decision != "FACIAL_NO" else "none",
        confidence=0.8 if decision not in {"FACIAL_NO", "PARSE_FAILURE"} else 0.0,
        rationale=rationale,
        candidate_name=snippet.display_name,
        profile_url=snippet.profile_url,
        prompt_capture={
            "judge_receipt": _judge_receipt(
                stage="facial",
                raw=raw,
                parse_status=parse_status,
                final_decision=decision,
                intended_postcondition=(
                    "designer facial judge response parses to a bounded decision"
                ),
            ),
        },
    )


def designer_full_judge(
    candidate: DesignerCandidate,
    *,
    brief: dict[str, Any],
    llm_caller: LLMCaller | None = None,
) -> OpusDecision:
    system = assemble_designer_full_system(brief)
    user = _candidate_user_prompt(candidate)
    if llm_caller is None:
        from shared.llm_clients import opus_llm_cached

        def _default(system_prompt: str, user_prompt: str) -> str:
            usage_context = {
                "stage": "designer_full_judge",
                "source": "designer",
                "candidate_name": candidate.snippet.display_name,
                "profile_url": candidate.snippet.profile_url,
                "brief_id": brief.get("id") or brief.get("role_title"),
            }
            return str(
                opus_llm_cached(
                    system_prompt,
                    user_prompt,
                    expect_json=False,
                    max_tokens=2048,
                    usage_context=usage_context,
                )
            )

        llm_caller = _default
    raw = llm_caller(system, user)
    parsed = _parse_full_response(raw)
    decision, path, confidence, rationale = fail_honest_full_decision(parsed)
    return OpusDecision(
        stage="full",
        decision=decision,
        path=path,
        confidence=confidence,
        rationale=rationale,
        candidate_name=candidate.snippet.display_name,
        profile_url=candidate.snippet.profile_url,
        prompt_capture={
            "judge_receipt": _judge_receipt(
                stage="full",
                raw=raw,
                parse_status=parsed["parse_status"],
                final_decision=decision,
                intended_postcondition=(
                    "designer full judge response parses to a bounded decision"
                ),
            ),
        },
    )


def _snippet_user_prompt(snippet: DesignerSnippet) -> str:
    return "\n".join(
        [
            f"Name: {snippet.display_name}",
            f"Profile: {snippet.profile_url}",
            f"Source: {snippet.source}",
            f"Location: {snippet.location}",
            f"Headline: {snippet.headline}",
            f"Fields: {', '.join(snippet.fields)}",
            f"Tools: {', '.join(snippet.tools)}",
            f"Top projects: {', '.join(snippet.top_project_titles)}",
        ]
    )


def _candidate_user_prompt(candidate: DesignerCandidate) -> str:
    snippet = candidate.snippet
    projects: list[str] = []
    for project in candidate.project_summaries:
        if not project.title:
            continue
        line = f"{project.title} ({', '.join(project.fields)})"
        if project.description:
            line += f" — {project.description[:300]}"
        projects.append(line)
    return "\n".join(
        [
            _snippet_user_prompt(snippet),
            f"Project summaries: {'; '.join(projects)}",
        ]
    )


def _parse_facial_response(raw: str | dict[str, Any]) -> tuple[str, str]:
    if isinstance(raw, dict):
        decision = str(raw.get("decision") or "").strip().upper()
        rationale = str(raw.get("rationale") or raw.get("reason") or "").strip()
    else:
        text = str(raw)
        decision = _line_value(text, "DECISION").upper()
        rationale = _line_value(text, "REASON") or text.strip()
    return decision, rationale or "Designer text triage returned no rationale."


def _parse_full_response(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        decision = str(raw.get("decision") or "").strip().upper()
        path = str(raw.get("path") or "designer_text_context").strip()
        confidence_raw = raw.get("confidence", 0.0)
        rationale = str(raw.get("rationale") or raw.get("summary") or "").strip()
    else:
        text = str(raw)
        decision = _line_value(text, "DECISION").upper()
        path = _line_value(text, "PATH") or "designer_text_context"
        confidence_raw = _line_value(text, "CONFIDENCE") or "0.0"
        rationale = _line_value(text, "SUMMARY") or text.strip()
    parse_status = "ok" if decision in _FULL_DECISIONS else "parse_fail"
    if parse_status != "ok" and _looks_like_model_refusal(raw):
        parse_status = "refused"
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    return {
        "parse_status": parse_status,
        "decision": decision,
        "path": path,
        "confidence": confidence,
        "rationale": rationale or "Designer full judge returned no summary.",
    }


def fail_honest_full_decision(parsed: dict[str, Any]) -> tuple[str, str, float, str]:
    """Map parse status and parsed scoring to the final full-stage decision.

    M1A invariant: a non-OK parse status must never become ``REJECT``. Keep this
    as a small deterministic function so tests can pin the decision rule directly.
    """

    if parsed.get("parse_status") != "ok":
        if parsed.get("parse_status") == "refused":
            return (
                "PARSE_FAILURE",
                "none",
                0.0,
                "REFUSED: model declined to judge designer full fit.",
            )
        return (
            "PARSE_FAILURE",
            "none",
            0.0,
            f"PARSE_FAILURE: unrecognized or missing designer full decision "
            f"({parsed.get('decision') or 'missing'}).",
        )
    return (
        str(parsed["decision"]),
        str(parsed.get("path") or "designer_text_context"),
        float(parsed.get("confidence") or 0.0),
        str(parsed.get("rationale") or "Designer full judge returned no summary."),
    )


def _judge_receipt(
    *,
    stage: str,
    raw: str | dict[str, Any],
    parse_status: str,
    final_decision: str,
    intended_postcondition: str,
) -> dict[str, Any]:
    if parse_status == "ok":
        status = ReceiptStatus.OK
    elif parse_status == "refused":
        status = ReceiptStatus.REFUSED
    else:
        status = ReceiptStatus.PARSE_FAIL
    receipt = build_receipt(
        receipt_type="judge",
        stage=f"designer_{stage}_judge",
        input_payload={"raw": raw, "stage": stage},
        actual_status=status,
        intended_postcondition=intended_postcondition,
        actual_detail={
            "parse_status": parse_status,
            "final_decision": final_decision,
        },
        producer="designer.judging",
        version_pins={"designer_judging": "fail-honest-v1"},
    )
    return receipt.to_dict()


_MODEL_REFUSAL_MARKERS = (
    "as an ai",
    "i cannot comply",
    "i can't comply",
    "cannot comply",
    "can't comply",
    "unable to comply",
    "i cannot assist",
    "i can't assist",
    "cannot assist",
    "can't assist",
    "decline to",
    "refuse to",
    "not able to evaluate",
)


def _looks_like_model_refusal(raw: str | dict[str, Any]) -> bool:
    if isinstance(raw, dict):
        return any(_looks_like_model_refusal(value) for value in raw.values())
    text = str(raw or "").lower()
    return any(marker in text for marker in _MODEL_REFUSAL_MARKERS)


def _line_value(text: str, label: str) -> str:
    match = re.search(rf"^{label}\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""
