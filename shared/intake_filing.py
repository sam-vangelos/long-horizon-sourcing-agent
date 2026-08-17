"""Server-authoritative intake filing readiness (pure contract helper).

Evaluates whether an intake session can be filed without importing HTTP
frameworks. ``cloris/api/intake.py`` maps :class:`FilingBlocker` codes to
HTTP status codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.brief_v2_schema import BriefSchemaError, validate_v2_brief
from shared.intake_conversation.insights import (
    HIRING_MANAGER_PICTURE_KEY,
    is_missing_hiring_manager_success_image,
)
from shared.intake_conversation.sufficiency import is_ready_to_compose

_BLOCKER_MESSAGES: dict[str, str] = {
    "intake_synthesis_in_progress": (
        "Source upload synthesis is still running. Wait for the "
        "draft to finish updating before filing."
    ),
    "intake_compose_in_progress": (
        "Brief composition is still running. Wait a moment, "
        "then try filing again."
    ),
    "intake_compose_required": (
        "This conversation still needs a composed brief before filing. "
        "Compose from the transcript first, then file."
    ),
}


@dataclass(frozen=True)
class FilingBlocker:
    code: str
    message: str


@dataclass(frozen=True)
class FilingReadiness:
    can_file: bool
    blocking_codes: tuple[str, ...]
    valid_v2_draft: bool
    missing_keys: tuple[str, ...]
    invalid_keys: tuple[str, ...]
    in_flight_synthesis: bool
    in_flight_compose: bool
    insight_deficits: tuple[dict[str, str], ...]


def _session_state(session: dict[str, Any]) -> dict[str, Any]:
    state = session.get("state_json")
    return state if isinstance(state, dict) else {}


def is_conversational_session(session: dict[str, Any]) -> bool:
    """True when the session uses the conversational intake path."""

    current_step = session.get("current_step")
    if current_step == "conversation":
        return True
    state = _session_state(session)
    if isinstance(state.get("conversation_meta"), dict):
        return True
    messages = state.get("messages")
    return isinstance(messages, list) and len(messages) > 0


def _needs_compose_from_transcript(session: dict[str, Any]) -> bool:
    if not is_conversational_session(session):
        return False
    state = _session_state(session)
    v2_draft = state.get("v2_draft")
    if not isinstance(v2_draft, dict):
        return True
    ready, _ = is_ready_to_compose(v2_draft)
    if not ready:
        return True
    return session.get("current_step") != "review"


def filing_blockers(session: dict[str, Any]) -> list[FilingBlocker]:
    """Return ordered filing blockers for ``session`` (may be empty)."""

    state = _session_state(session)
    blockers: list[FilingBlocker] = []

    synthesis_block = state.get("source_packet_synthesis")
    if (
        isinstance(synthesis_block, dict)
        and synthesis_block.get("status") == "running"
    ):
        blockers.append(
            FilingBlocker(
                code="intake_synthesis_in_progress",
                message=_BLOCKER_MESSAGES["intake_synthesis_in_progress"],
            )
        )

    compose_block = state.get("conversation_compose")
    if isinstance(compose_block, dict) and compose_block.get("status") == "composing":
        blockers.append(
            FilingBlocker(
                code="intake_compose_in_progress",
                message=_BLOCKER_MESSAGES["intake_compose_in_progress"],
            )
        )

    if _needs_compose_from_transcript(session):
        blockers.append(
            FilingBlocker(
                code="intake_compose_required",
                message=_BLOCKER_MESSAGES["intake_compose_required"],
            )
        )

    return blockers


def primary_filing_blocker(session: dict[str, Any]) -> FilingBlocker | None:
    blockers = filing_blockers(session)
    return blockers[0] if blockers else None


def _insight_deficits_from_state(state: dict[str, Any]) -> list[dict[str, str]]:
    insights = state.get("intake_insights")
    if not isinstance(insights, dict):
        insights = {}
    picture = insights.get(HIRING_MANAGER_PICTURE_KEY)
    if is_missing_hiring_manager_success_image(picture):
        return [
            {
                "field": "hiring_manager_success_image",
                "reason": (
                    "Cloris is still forming the hiring-manager picture — "
                    "keep talking to sharpen it, or file with this gap noted."
                ),
            }
        ]
    return []


def _v2_draft_validation(
    state: dict[str, Any],
) -> tuple[bool, tuple[str, ...], tuple[str, ...]]:
    v2_draft = state.get("v2_draft")
    if not isinstance(v2_draft, dict):
        return False, ("v2_draft",), ()
    try:
        validate_v2_brief(v2_draft)
    except BriefSchemaError as exc:
        return False, tuple(exc.missing_keys), tuple(exc.invalid_keys)
    return True, (), ()


def filing_readiness(session: dict[str, Any]) -> FilingReadiness:
    """Non-throwing readiness summary for session GET responses."""

    state = _session_state(session)
    blockers = filing_blockers(session)
    codes = tuple(blocker.code for blocker in blockers)
    valid_v2, missing_keys, invalid_keys = _v2_draft_validation(state)
    synthesis_block = state.get("source_packet_synthesis")
    compose_block = state.get("conversation_compose")
    in_flight_synthesis = (
        isinstance(synthesis_block, dict)
        and synthesis_block.get("status") == "running"
    )
    in_flight_compose = (
        isinstance(compose_block, dict) and compose_block.get("status") == "composing"
    )
    deficits = tuple(_insight_deficits_from_state(state))
    return FilingReadiness(
        can_file=not codes,
        blocking_codes=codes,
        valid_v2_draft=valid_v2,
        missing_keys=missing_keys,
        invalid_keys=invalid_keys,
        in_flight_synthesis=in_flight_synthesis,
        in_flight_compose=in_flight_compose,
        insight_deficits=deficits,
    )


def filing_readiness_wire(session: dict[str, Any]) -> dict[str, Any]:
    """JSON-serializable filing readiness for API wire models."""

    readiness = filing_readiness(session)
    return {
        "can_file": readiness.can_file,
        "blocking_codes": list(readiness.blocking_codes),
        "valid_v2_draft": readiness.valid_v2_draft,
        "missing_keys": list(readiness.missing_keys),
        "invalid_keys": list(readiness.invalid_keys),
        "in_flight_synthesis": readiness.in_flight_synthesis,
        "in_flight_compose": readiness.in_flight_compose,
        "insight_deficits": [
            dict(item) for item in readiness.insight_deficits
        ],
    }


def http_status_for_blocker(blocker: FilingBlocker) -> int:
    if blocker.code == "intake_compose_required":
        return 422
    return 409


__all__ = [
    "FilingBlocker",
    "FilingReadiness",
    "filing_blockers",
    "filing_readiness",
    "filing_readiness_wire",
    "http_status_for_blocker",
    "is_conversational_session",
    "primary_filing_blocker",
]
