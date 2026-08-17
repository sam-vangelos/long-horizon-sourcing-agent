"""Shared launch-readiness types used across source health probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

BlockerKind = Literal["auth", "config", "net"]


@dataclass(frozen=True)
class ReadinessBlocker:
    """One reason the launch isn't ready. Each carries editorial remediation."""

    kind: BlockerKind
    message: str
    remediation: str
    code: str = ""


@dataclass(frozen=True)
class ReadinessReport:
    """Outcome of a readiness probe. ``ready`` is true iff blockers is empty."""

    ready: bool
    blockers: tuple[ReadinessBlocker, ...]
    informational_notes: tuple[str, ...] = field(default_factory=tuple)
