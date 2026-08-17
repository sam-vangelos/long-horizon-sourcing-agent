"""Typed, provider-specific policy for Fireworks judgment calls.

The public dataclasses in this module are deliberately inert.  Merely importing
or constructing one does not change routing; callers must pass it explicitly to
one of the Fireworks-capable helpers in :mod:`shared.llm_clients`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Mapping


FIREWORKS_GLM_5P2_STANDARD_MODEL = "accounts/fireworks/models/glm-5p2"
FIREWORKS_GLM_5P2_FAST_MODEL = "accounts/fireworks/routers/glm-5p2-fast"
FIREWORKS_RETRYABLE_STATUS_CODES = frozenset({408, 429, 502, 503, 504, 520})

FireworksReasoningEffort = Literal["high", "max"]
FireworksResponseTransport = Literal["complete", "stream"]


class FireworksDeadlineExceeded(TimeoutError):
    """The total wall-clock budget for one logical Fireworks call expired."""


@dataclass(frozen=True, slots=True)
class FireworksStagePolicy:
    """Fully explicit retry/reasoning policy for one logical stage call."""

    stage: str
    reasoning_effort: FireworksReasoningEffort | None = None
    attempt_timeout_seconds: float = 300.0
    total_deadline_seconds: float = 1500.0
    max_attempts: int = 5
    response_transport: FireworksResponseTransport = "complete"
    prompt_cache_key: str | None = None
    retry_status_codes: frozenset[int] = field(
        default_factory=lambda: FIREWORKS_RETRYABLE_STATUS_CODES
    )

    def __post_init__(self) -> None:
        stage = str(self.stage or "").strip()
        if not stage:
            raise ValueError("FireworksStagePolicy.stage must be non-empty")
        object.__setattr__(self, "stage", stage)
        if self.reasoning_effort not in {None, "high", "max"}:
            raise ValueError("reasoning_effort must be None, 'high', or 'max'")
        try:
            attempt_timeout_seconds = float(self.attempt_timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("attempt_timeout_seconds must be a finite number > 0") from exc
        if not math.isfinite(attempt_timeout_seconds) or attempt_timeout_seconds <= 0:
            raise ValueError("attempt_timeout_seconds must be finite and > 0")
        object.__setattr__(self, "attempt_timeout_seconds", attempt_timeout_seconds)
        try:
            total_deadline_seconds = float(self.total_deadline_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("total_deadline_seconds must be a finite number > 0") from exc
        if not math.isfinite(total_deadline_seconds) or total_deadline_seconds <= 0:
            raise ValueError("total_deadline_seconds must be finite and > 0")
        object.__setattr__(self, "total_deadline_seconds", total_deadline_seconds)
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.response_transport not in {"complete", "stream"}:
            raise ValueError("response_transport must be 'complete' or 'stream'")
        if self.prompt_cache_key is not None:
            key = self.prompt_cache_key.strip()
            if not key:
                raise ValueError("prompt_cache_key must be non-empty when provided")
            object.__setattr__(self, "prompt_cache_key", key)
        statuses = frozenset(int(value) for value in self.retry_status_codes)
        if not statuses:
            raise ValueError("retry_status_codes must not be empty")
        object.__setattr__(self, "retry_status_codes", statuses)


@dataclass(frozen=True, slots=True)
class FireworksToolContract:
    """One forced, client-side function tool used as an output envelope."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    strict: bool = True

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("FireworksToolContract.name must be non-empty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("FireworksToolContract.parameters must be a mapping")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", str(self.description or "").strip())
        # Detach the transport schema from a mutable caller-owned dictionary.
        object.__setattr__(self, "parameters", copy.deepcopy(dict(self.parameters)))

    def tool_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": copy.deepcopy(dict(self.parameters)),
                "strict": bool(self.strict),
            },
        }

    def forced_choice(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name}}


def build_fireworks_prompt_cache_key(
    *,
    stage: str,
    model: str,
    contract_version: str,
    stable_prefix: str,
    lane_id: str = "",
    batch_slot: int | None = None,
) -> str:
    """Return a stable, opaque affinity key containing no caller text.

    Only the digest crosses the provider boundary.  Candidate evidence must not
    be passed as ``stable_prefix``; callers use the shared rubric/brief/tool
    prefix that precedes candidate-specific content.
    """

    payload = {
        "version": "fireworks-affinity-v1",
        "stage": str(stage or "").strip(),
        "model": str(model or "").strip(),
        "contract_version": str(contract_version or "").strip(),
        "stable_prefix_sha256": hashlib.sha256(
            str(stable_prefix or "").encode("utf-8")
        ).hexdigest(),
        "lane": str(lane_id or "").strip(),
        "batch_slot": batch_slot,
    }
    if not payload["stage"] or not payload["model"] or not payload["contract_version"]:
        raise ValueError("stage, model, and contract_version must be non-empty")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"cloris-fw-v1-{digest[:32]}"


def fireworks_serving_path(model: str) -> str:
    normalized = str(model or "").lower()
    return "fast" if "/routers/" in normalized and "fast" in normalized else "standard"


def effective_reasoning_effort(
    model: str,
    requested: FireworksReasoningEffort | None,
) -> str | None:
    """Document the GLM-5.2 effective tier without inventing other defaults."""

    if requested is not None:
        return requested
    if "glm-5p2" in str(model or "").lower():
        return "max"
    return None


def parse_retry_after_seconds(
    value: object,
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse an HTTP Retry-After delay-seconds or HTTP-date value."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        try:
            target = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        seconds = (target - current).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(seconds, 0.0)


__all__ = [
    "FIREWORKS_GLM_5P2_FAST_MODEL",
    "FIREWORKS_GLM_5P2_STANDARD_MODEL",
    "FIREWORKS_RETRYABLE_STATUS_CODES",
    "FireworksDeadlineExceeded",
    "FireworksResponseTransport",
    "FireworksStagePolicy",
    "FireworksToolContract",
    "build_fireworks_prompt_cache_key",
    "effective_reasoning_effort",
    "fireworks_serving_path",
    "parse_retry_after_seconds",
]
