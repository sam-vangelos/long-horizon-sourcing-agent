"""Query validation layer and search space exhaustion governor.

Validates and repairs LLM-generated GitHub search queries before API execution.
Tracks per-channel exhaustion state to avoid burning API budget on mined-out
search spaces.

Two public interfaces:
1. validate_batch() — validate/repair queries between LLM output and execution
2. ExhaustionState — per-channel tracking of search space depletion
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from github.schemas import GitHubSearchQuery
from shared.brief_loader import Brief


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Channels that get query validation (freeform query strings).
# registry_maintainer_discovery is deliberately excluded: it is structured
# (target_ecosystem + target_packages), not a freeform search string.
_VALIDATED_CHANNELS = {"user_search", "code_search", "topic_search"}

# Natural language filler words to strip from user_search queries
_NL_FILLER = re.compile(
    r"\b(very|ultra|highly|active|experienced|skilled|senior|junior|"
    r"strong|top|best|excellent|great|amazing|talented|proficient|expert|"
    r"passionate|dedicated|enthusiastic)\b",
    re.IGNORECASE,
)

# GitHub user search qualifiers
_USER_QUALIFIERS = re.compile(
    r"(language|location|followers|repos|created|type|fullname|in):"
)

# Code search qualifiers
_CODE_QUALIFIERS = re.compile(
    r"(language|extension|repo|path|filename|org|user):"
)

# Topic/repo search qualifiers
_TOPIC_QUALIFIERS = re.compile(
    r"(topic|language|stars|pushed|forks|created|size|archived|mirror|template):"
)

# Descriptive phrases → qualifier conversions for user_search
# Patterns match compound forms like "ultra-low followers", "very high followers"
_PHRASE_CONVERSIONS = [
    (re.compile(r"(?:\w+[\s-]+)*low[\s-]*followers?\b", re.IGNORECASE), "followers:<20"),
    (re.compile(r"(?:\w+[\s-]+)*high[\s-]*followers?\b", re.IGNORECASE), "followers:>100"),
    (re.compile(r"(?:\w+[\s-]+)*many[\s-]*repos?\b", re.IGNORECASE), "repos:>30"),
    (re.compile(r"(?:\w+[\s-]+)*few[\s-]*repos?\b", re.IGNORECASE), "repos:<10"),
    (re.compile(r"(?:\w+[\s-]+)*new[\s-]*accounts?\b", re.IGNORECASE), "created:>2022-01-01"),
    (re.compile(r"(?:\w+[\s-]+)*old[\s-]*accounts?\b", re.IGNORECASE), "created:<2018-01-01"),
]

# Quoted string pattern
_QUOTED = re.compile(r'"[^"]+"')


@dataclass
class ValidationResult:
    """Result of validating a single query."""
    query_id: int
    channel: str
    original_query: str
    repaired_query: str
    status: str  # "accepted" | "repaired" | "rejected" | "duplicate"
    reason: str = ""


def validate_batch(
    queries: list[GitHubSearchQuery],
    brief: Brief,
    executed_queries: set[str],
) -> tuple[list[GitHubSearchQuery], list[ValidationResult]]:
    """Validate and repair a batch of queries.

    Returns (accepted_queries, all_results). Accepted includes valid and
    repaired queries. Rejected/duplicate queries are excluded.
    """
    accepted = []
    results = []

    # Extract geography from brief for injection
    geo = brief.permanent_filters.get("Location", "")

    for query in queries:
        # Structured channels pass through without validation
        if query.channel not in _VALIDATED_CHANNELS:
            accepted.append(query)
            results.append(ValidationResult(
                query_id=query.id,
                channel=query.channel,
                original_query=query.query,
                repaired_query=query.query,
                status="accepted",
                reason="structured channel — no validation needed",
            ))
            continue

        # Dedup check
        normalized = _normalize(query.query)
        if normalized in executed_queries:
            results.append(ValidationResult(
                query_id=query.id,
                channel=query.channel,
                original_query=query.query,
                repaired_query=query.query,
                status="duplicate",
                reason="query already executed",
            ))
            continue

        # Channel-specific validation
        if query.channel == "user_search":
            result = _validate_user_search(query, geo)
        elif query.channel == "code_search":
            result = _validate_code_search(query)
        elif query.channel == "topic_search":
            result = _validate_topic_search(query)
        else:
            result = ValidationResult(
                query_id=query.id,
                channel=query.channel,
                original_query=query.query,
                repaired_query=query.query,
                status="accepted",
            )

        results.append(result)

        if result.status in ("accepted", "repaired"):
            # Apply repair if needed
            if result.repaired_query != query.query:
                query.query = result.repaired_query
            # Final dedup check after repair
            repaired_normalized = _normalize(query.query)
            if repaired_normalized in executed_queries:
                result.status = "duplicate"
                result.reason = "query matches executed query after repair"
                continue
            accepted.append(query)

    return accepted, results


def _validate_user_search(query: GitHubSearchQuery, geo: str) -> ValidationResult:
    """Validate a user_search query. Repair if possible."""
    original = query.query
    q = original

    # Convert descriptive phrases to qualifiers BEFORE stripping filler
    # (so compound phrases like "ultra-low followers" match)
    for pattern, replacement in _PHRASE_CONVERSIONS:
        q = pattern.sub(replacement, q)

    # Strip natural language filler
    q = _NL_FILLER.sub("", q)

    # Collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()

    # Inject location if brief has geography and query lacks it
    if geo and not re.search(r"location:", q, re.IGNORECASE):
        q = f"{q} location:{geo}"

    # Check: must have at least one qualifier OR quoted terms
    has_qualifier = bool(_USER_QUALIFIERS.search(q))
    has_quoted = bool(_QUOTED.search(q))

    if not has_qualifier and not has_quoted:
        return ValidationResult(
            query_id=query.id,
            channel="user_search",
            original_query=original,
            repaired_query=q,
            status="rejected",
            reason="no GitHub qualifiers or quoted terms after repair",
        )

    if q != original:
        return ValidationResult(
            query_id=query.id,
            channel="user_search",
            original_query=original,
            repaired_query=q,
            status="repaired",
            reason="stripped filler / converted phrases / injected location",
        )

    return ValidationResult(
        query_id=query.id,
        channel="user_search",
        original_query=original,
        repaired_query=q,
        status="accepted",
    )


def _validate_code_search(query: GitHubSearchQuery) -> ValidationResult:
    """Validate a code_search query."""
    original = query.query
    q = original.strip()

    has_qualifier = bool(_CODE_QUALIFIERS.search(q))
    has_quoted = bool(_QUOTED.search(q))

    if not has_qualifier and not has_quoted:
        return ValidationResult(
            query_id=query.id,
            channel="code_search",
            original_query=original,
            repaired_query=q,
            status="rejected",
            reason="no code qualifiers or quoted strings — purely natural language",
        )

    return ValidationResult(
        query_id=query.id,
        channel="code_search",
        original_query=original,
        repaired_query=q,
        status="accepted",
    )


def _validate_topic_search(query: GitHubSearchQuery) -> ValidationResult:
    """Validate a topic_search query. Attempt bare-keyword → topic: conversion."""
    original = query.query
    q = original.strip()

    has_qualifier = bool(_TOPIC_QUALIFIERS.search(q))
    has_quoted = bool(_QUOTED.search(q))

    if not has_qualifier and not has_quoted:
        # Try converting bare keywords to topic: format
        words = q.split()
        if words and all(re.match(r"^[\w-]+$", w) for w in words):
            # Convert space-separated keywords to topic qualifiers
            q = " ".join(f"topic:{w}" for w in words)
            has_qualifier = True

    if not has_qualifier and not has_quoted:
        return ValidationResult(
            query_id=query.id,
            channel="topic_search",
            original_query=original,
            repaired_query=q,
            status="rejected",
            reason="no topic/repo qualifiers or quoted terms after repair",
        )

    if q != original:
        return ValidationResult(
            query_id=query.id,
            channel="topic_search",
            original_query=original,
            repaired_query=q,
            status="repaired",
            reason="converted bare keywords to topic: qualifiers",
        )

    return ValidationResult(
        query_id=query.id,
        channel="topic_search",
        original_query=original,
        repaired_query=q,
        status="accepted",
    )


def _normalize(query: str) -> str:
    """Normalize a query string for dedup comparison."""
    return re.sub(r"\s+", " ", query.lower().strip())


# ---------------------------------------------------------------------------
# Search Space Exhaustion Governor
# ---------------------------------------------------------------------------

@dataclass
class ChannelStats:
    """Per-channel tracking for exhaustion detection."""
    queries_run: int = 0
    total_candidates: int = 0
    total_saves: int = 0
    total_pre_dedup: int = 0
    total_dedup_filtered: int = 0
    zero_result_streak: int = 0    # consecutive queries returning 0 new candidates
    zero_save_streak: int = 0      # consecutive queries with 0 saves
    status: str = "active"         # "active" | "degraded" | "exhausted"

    @property
    def dedup_rate(self) -> float:
        if self.total_pre_dedup == 0:
            return 0.0
        return self.total_dedup_filtered / self.total_pre_dedup


class ExhaustionState:
    """Tracks per-channel search space exhaustion."""

    def __init__(self):
        self.channels: dict[str, ChannelStats] = {}

    def _ensure_channel(self, channel: str) -> ChannelStats:
        if channel not in self.channels:
            self.channels[channel] = ChannelStats()
        return self.channels[channel]

    def record_query_result(
        self,
        channel: str,
        saves: int,
        candidates: int,
        pre_dedup: int,
        post_dedup: int,
    ):
        """Record results from a completed query."""
        cs = self._ensure_channel(channel)
        cs.queries_run += 1
        cs.total_candidates += candidates
        cs.total_saves += saves
        cs.total_pre_dedup += pre_dedup
        cs.total_dedup_filtered += (pre_dedup - post_dedup)

        # Update streaks
        if candidates == 0:
            cs.zero_result_streak += 1
        else:
            cs.zero_result_streak = 0

        if saves == 0:
            cs.zero_save_streak += 1
        else:
            cs.zero_save_streak = 0

        # Re-evaluate channel status
        cs.status = self.evaluate_channel(channel)

    def evaluate_channel(self, channel: str) -> str:
        """Evaluate channel status based on exhaustion thresholds."""
        cs = self.channels.get(channel)
        if not cs:
            return "active"

        # Exhausted conditions (most aggressive first)
        if cs.zero_result_streak >= 3:
            return "exhausted"
        if cs.zero_save_streak >= 10:
            return "exhausted"
        if cs.dedup_rate > 0.7 and cs.queries_run >= 5:
            return "exhausted"

        # Degraded conditions
        if cs.zero_save_streak >= 5 and cs.queries_run >= 5:
            return "degraded"
        if channel == "user_search" and cs.dedup_rate > 0.5 and cs.queries_run >= 10:
            return "degraded"

        return "active"

    def get_exhausted_channels(self) -> list[str]:
        return [ch for ch, cs in self.channels.items() if cs.status == "exhausted"]

    def get_degraded_channels(self) -> list[str]:
        return [ch for ch, cs in self.channels.items() if cs.status == "degraded"]

    def to_adaptation_context(self) -> str:
        """Format exhaustion state for inclusion in LLM adaptation prompt."""
        if not self.channels:
            return "No channel data yet."

        lines = []
        for channel, cs in sorted(self.channels.items()):
            status_icon = {"active": "OK", "degraded": "DEGRADED", "exhausted": "EXHAUSTED"}.get(cs.status, "?")
            lines.append(
                f"- {channel}: [{status_icon}] "
                f"{cs.queries_run} queries, {cs.total_saves} saves, "
                f"{cs.total_candidates} candidates, "
                f"dedup_rate={cs.dedup_rate:.0%}, "
                f"zero_save_streak={cs.zero_save_streak}, "
                f"zero_result_streak={cs.zero_result_streak}"
            )
        return "\n".join(lines)
