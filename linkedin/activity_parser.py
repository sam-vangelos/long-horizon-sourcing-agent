"""Deterministic parsing of LinkedIn Recruiter activity surfaces."""

from __future__ import annotations

import re

from shared.reconciliation_schemas import RecruiterActivitySnapshot

_MESSAGE_RE = re.compile(r"(\d+)\s+messages?\b", re.IGNORECASE)
_PROJECT_RE = re.compile(r"(?:in\s+)?(\d+)\s+projects?\b", re.IGNORECASE)
_VIEW_RE = re.compile(r"(\d+)\s+views?\b", re.IGNORECASE)
_SAVED_BY_RE = re.compile(r"Saved by\s+(.+?)(?:\s+on\s+|\n|$)", re.IGNORECASE)
_STOP_HEADERS_EXACT = {
    "summary",
    "experience",
    "recruiting tools",
    "similar profiles",
    "projects",
    "messages",
    "quick add",
    "email",
    "phone number",
    "message",
    "log activity",
}
_STOP_HEADER_PREFIXES = (
    "greenhouse (",
)


def _matches_header(line: str, header: str) -> bool:
    lowered = (line or "").strip().lower()
    target = header.strip().lower()
    return lowered == target or lowered.startswith(f"{target} ")


def _is_stop_header(line: str) -> bool:
    lowered = (line or "").strip().lower()
    return lowered in _STOP_HEADERS_EXACT or any(lowered.startswith(prefix) for prefix in _STOP_HEADER_PREFIXES)


def _matches_stop_header(line: str, stop_headers: set[str]) -> bool:
    lowered = (line or "").strip().lower()
    if lowered in stop_headers:
        return True
    return any(lowered.startswith(header) for header in stop_headers if header.endswith("("))


def _coerce_first_int(pattern: re.Pattern[str], text: str) -> int:
    match = pattern.search(text or "")
    if not match:
        return 0
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return 0


def extract_recruiter_activity_from_card_text(text: str) -> RecruiterActivitySnapshot:
    text = text or ""
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    activity_lines = [
        line for line in raw_lines
        if "message" in line.lower()
        or "project" in line.lower()
        or "view" in line.lower()
        or line.lower().startswith("saved by ")
        or line.lower().startswith("activity")
    ]
    saved_by_match = _SAVED_BY_RE.search(text)
    snapshot = RecruiterActivitySnapshot(
        message_count=_coerce_first_int(_MESSAGE_RE, text),
        project_count=_coerce_first_int(_PROJECT_RE, text),
        view_count=_coerce_first_int(_VIEW_RE, text),
        saved_by=(saved_by_match.group(1).strip() if saved_by_match else ""),
        raw_activity_text=" | ".join(activity_lines),
        reachout_status="messaged" if _coerce_first_int(_MESSAGE_RE, text) > 0 else "",
    )
    return snapshot


def extract_profile_recent_activity_lines(text: str, *, limit: int = 12) -> list[str]:
    text = text or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def _collect_after(header: str) -> list[str]:
        try:
            start = next(idx for idx, line in enumerate(lines) if _matches_header(line, header))
        except StopIteration:
            return []
        collected: list[str] = []
        for line in lines[start + 1:]:
            if _is_stop_header(line):
                break
            collected.append(line)
            if len(collected) >= limit:
                break
        return collected

    recent = _collect_after("most recent activity")
    if recent:
        return recent
    recruiting = _collect_after("recruiting activity")
    return recruiting[:limit]


def extract_profile_status_summary(text: str) -> dict:
    text = text or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def _collect_after(header: str, stop_headers: set[str]) -> list[str]:
        try:
            start = next(idx for idx, line in enumerate(lines) if _matches_header(line, header))
        except StopIteration:
            return []
        collected: list[str] = []
        for line in lines[start + 1:]:
            lowered = line.lower()
            if _matches_stop_header(lowered, stop_headers):
                break
            collected.append(line)
        return collected

    snapshot = extract_recruiter_activity_from_card_text(text)
    recent_activity = extract_profile_recent_activity_lines(text)
    last_outbound_lines = _collect_after(
        "last outbound contact",
        {"sequences", "projects", "quick add", "greenhouse (", "email", "phone number", "message", "log activity"},
    )
    sequence_lines = _collect_after(
        "sequences",
        {"projects", "quick add", "greenhouse (", "email", "phone number", "message", "log activity"},
    )
    last_outbound = " ".join(last_outbound_lines)
    sequences = sequence_lines[:1]

    reachout_status = snapshot.reachout_status or ("recent_outbound_contact" if last_outbound else "")
    return {
        "message_count": snapshot.message_count,
        "project_count": snapshot.project_count,
        "view_count": snapshot.view_count,
        "saved_by": snapshot.saved_by,
        "raw_activity_text": snapshot.raw_activity_text,
        "recent_activity": recent_activity,
        "last_outbound_contact": last_outbound,
        "reachout_status": reachout_status,
        "sequences": sequences,
    }
