"""Deterministic recommendation pitch assembly for Designer saves.

D5b of the Designer go-live plan. No LLM call — the pitch is
assembled from the terminal payload's ``full_decision`` and
``visual_judgment`` blocks using rule-based headline selection and
evidence extraction.

Returns ``None`` for REJECT candidates. For SAVE / INFERENTIAL_SAVE,
produces a dict with ``headline``, ``summary``, ``evidence_bullets``,
and ``caveats``.
"""

from __future__ import annotations

from typing import Any


def assemble_recommendation_pitch(
    terminal_payload: dict[str, Any],
    role_title: str = "",
) -> dict[str, Any] | None:
    """Build a recruiter-facing recommendation pitch from terminal evidence.

    Returns ``None`` when the candidate was rejected (no pitch to
    render). For saves, returns::

        {
            "headline": str,
            "summary": str,
            "evidence_bullets": list[str],
            "caveats": list[str],
        }
    """

    full_decision = terminal_payload.get("full_decision") or {}
    decision = full_decision.get("decision", "")
    if decision == "REJECT":
        return None

    visual_judgment = terminal_payload.get("visual_judgment") or {}
    principles = visual_judgment.get("principles") or []
    fallback_reason = visual_judgment.get("fallback_reason") or ""
    overall_verdict = visual_judgment.get("overall_verdict", "")

    # Classify principles by score.
    strong: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    for p in principles:
        score = p.get("score", 0)
        if score >= 2:
            strong.append(p)
        else:
            weak.append(p)

    # Headline rules — ordered by specificity.
    headline = _build_headline(
        strong=strong,
        role_title=role_title,
        fallback_reason=fallback_reason,
        overall_verdict=overall_verdict,
        decision=decision,
    )

    # Summary: full_decision rationale verbatim (already editorial register).
    summary = full_decision.get("rationale", "") or ""

    # Evidence bullets from strong principles.
    evidence_bullets: list[str] = []
    for p in strong:
        name = p.get("name", "")
        anchor = p.get("anchor", "")
        reasoning = p.get("reasoning", "")
        image_ids = p.get("image_ids", [])
        image_ref = f" [images {image_ids}]" if image_ids else ""
        evidence_bullets.append(f"{name} ({anchor}): {reasoning}{image_ref}")

    # Caveats from weak principles + anchor drift + fallback reason.
    caveats: list[str] = []
    for p in weak:
        name = p.get("name", "")
        anchor = p.get("anchor", "")
        reasoning = p.get("reasoning", "")
        caveats.append(f"{name} ({anchor}): {reasoning}")
    for p in principles:
        if not p.get("anchor_consistency_pass", True):
            name = p.get("name", "")
            caveats.append(f"{name}: model anchor drift flagged")
    if fallback_reason:
        caveats.append(f"Visual evidence limited: {fallback_reason}")

    return {
        "headline": headline,
        "summary": summary,
        "evidence_bullets": evidence_bullets,
        "caveats": caveats,
    }


def _build_headline(
    *,
    strong: list[dict[str, Any]],
    role_title: str,
    fallback_reason: str,
    overall_verdict: str,
    decision: str,
) -> str:
    title = role_title or "the role"

    if decision in {"INFERENTIAL_SAVE"} and fallback_reason:
        return f"Promising text signal \u2014 visual evidence was limited"

    if len(strong) >= 3:
        return f"Strong fit for {title}"

    if 1 <= len(strong) <= 2:
        top_name = strong[0].get("name", "")
        return f"Worth a conversation \u2014 {top_name} stands out"

    if overall_verdict == "yes":
        return "Visual review supports a conversation"

    return f"Worth a closer look for {title}"
