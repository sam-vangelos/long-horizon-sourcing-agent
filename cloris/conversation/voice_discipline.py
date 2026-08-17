"""Shared voice / trope checks for the conversation companion."""

from __future__ import annotations

from market_intelligence.briefing_polish import (
    BANNED_BRIEFING_TOKENS,
    SNAKE_CASE_IDENTIFIER_RE,
)

CHATBOT_TROPE_SUBSTRINGS: tuple[str, ...] = (
    "great question",
    "i'd be happy to",
    "i would be happy to",
    "let me check",
    "let's check",
    "i hope this helps",
    "happy to help",
    "feel free to",
)

__all__ = [
    "BANNED_BRIEFING_TOKENS",
    "CHATBOT_TROPE_SUBSTRINGS",
    "SNAKE_CASE_IDENTIFIER_RE",
    "voice_violations",
]


def voice_violations(text: str) -> list[str]:
    """Return human-readable violation codes if ``text`` breaks register rules."""

    issues: list[str] = []
    lowered = text.lower()
    for token in BANNED_BRIEFING_TOKENS:
        if token in lowered:
            issues.append(f"banned_token:{token}")
    for trope in CHATBOT_TROPE_SUBSTRINGS:
        if trope in lowered:
            issues.append(f"trope:{trope}")
    match = SNAKE_CASE_IDENTIFIER_RE.search(text)
    if match:
        issues.append(f"snake_case:{match.group(0)}")
    return issues
