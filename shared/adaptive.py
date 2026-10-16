"""Source-agnostic adaptive sourcing decisions.

This module deliberately models only cross-source control concepts:
what was scouted, what signal/noise was observed, and what action the
source adapter decided to take. Channel-specific details stay in
``source_payload`` so LinkedIn Booleans, GitHub API channels, research
venue filters, design rubrics, and exec-search company lanes can keep
their native language.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol


ADAPTATION_EVENT_TYPE = "adaptation_decision"


class AdaptiveAction(StrEnum):
    """Common actions an adaptive sourcing loop can take."""

    COMMIT = "commit"
    CONTINUE = "continue"
    ABANDON = "abandon"
    NARROW = "narrow"
    BROADEN = "broaden"
    EXPERIMENT = "experiment"
    REORDER = "reorder"
    SKIP = "skip"


@dataclass(frozen=True)
class SignalMarker:
    """Positive evidence observed while scouting a lane/work unit."""

    kind: str
    label: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SignalMarker":
        return cls(
            kind=str(payload.get("kind") or ""),
            label=str(payload.get("label") or ""),
            count=int(payload.get("count") or 0),
            examples=[str(item) for item in payload.get("examples") or []],
        )


@dataclass(frozen=True)
class NoiseMarker:
    """Negative or misleading evidence observed while scouting."""

    kind: str
    label: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NoiseMarker":
        return cls(
            kind=str(payload.get("kind") or ""),
            label=str(payload.get("label") or ""),
            count=int(payload.get("count") or 0),
            examples=[str(item) for item in payload.get("examples") or []],
        )


@dataclass(frozen=True)
class ChannelExhaustion:
    """API/channel exhaustion or decay that changed the plan."""

    channel: str
    reason: str
    retry_after_seconds: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChannelExhaustion":
        retry_after = payload.get("retry_after_seconds")
        return cls(
            channel=str(payload.get("channel") or ""),
            reason=str(payload.get("reason") or ""),
            retry_after_seconds=int(retry_after) if retry_after is not None else None,
        )


@dataclass(frozen=True)
class ScoutMetrics:
    """Cross-source scout metrics used to justify adaptation."""

    work_units_run: int = 0
    candidates_discovered: int = 0
    candidates_enriched: int = 0
    facial_yes: int = 0
    facial_no: int = 0
    facial_borderline: int = 0
    saves: int = 0
    rejects: int = 0
    insufficient: int = 0
    signal_markers: list[SignalMarker] = field(default_factory=list)
    noise_markers: list[NoiseMarker] = field(default_factory=list)
    exhaustion: list[ChannelExhaustion] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScoutMetrics":
        return cls(
            work_units_run=int(payload.get("work_units_run") or 0),
            candidates_discovered=int(payload.get("candidates_discovered") or 0),
            candidates_enriched=int(payload.get("candidates_enriched") or 0),
            facial_yes=int(payload.get("facial_yes") or 0),
            facial_no=int(payload.get("facial_no") or 0),
            facial_borderline=int(payload.get("facial_borderline") or 0),
            saves=int(payload.get("saves") or 0),
            rejects=int(payload.get("rejects") or 0),
            insufficient=int(payload.get("insufficient") or 0),
            signal_markers=[
                SignalMarker.from_dict(item)
                for item in payload.get("signal_markers") or []
                if isinstance(item, dict)
            ],
            noise_markers=[
                NoiseMarker.from_dict(item)
                for item in payload.get("noise_markers") or []
                if isinstance(item, dict)
            ],
            exhaustion=[
                ChannelExhaustion.from_dict(item)
                for item in payload.get("exhaustion") or []
                if isinstance(item, dict)
            ],
        )


@dataclass(frozen=True)
class AdaptationDecision:
    """A persisted source-native adaptation decision with common metadata."""

    source: str
    action: AdaptiveAction
    lane: str
    rationale: str
    metrics: ScoutMetrics = field(default_factory=ScoutMetrics)
    work_unit_kind: str = ""
    work_unit_family: str = ""
    inserted_work_units: list[str] = field(default_factory=list)
    skipped_work_units: list[str] = field(default_factory=list)
    reordered_work_units: list[str] = field(default_factory=list)
    source_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdaptationDecision":
        return cls(
            source=str(payload.get("source") or ""),
            action=AdaptiveAction(str(payload.get("action") or AdaptiveAction.CONTINUE.value)),
            lane=str(payload.get("lane") or ""),
            rationale=str(payload.get("rationale") or ""),
            metrics=ScoutMetrics.from_dict(payload.get("metrics") or {}),
            work_unit_kind=str(payload.get("work_unit_kind") or ""),
            work_unit_family=str(payload.get("work_unit_family") or ""),
            inserted_work_units=[str(item) for item in payload.get("inserted_work_units") or []],
            skipped_work_units=[str(item) for item in payload.get("skipped_work_units") or []],
            reordered_work_units=[str(item) for item in payload.get("reordered_work_units") or []],
            source_payload=dict(payload.get("source_payload") or {}),
        )


class SupportsRuntimeEvents(Protocol):
    def record_event(
        self,
        *,
        event_type: str,
        payload: dict | None = None,
        run_id: int | None = None,
        work_unit_id: int | None = None,
        candidate_id: int | None = None,
        attempt_id: int | None = None,
    ) -> None:
        ...


def record_adaptation_decision(
    store: SupportsRuntimeEvents,
    *,
    run_id: int,
    decision: AdaptationDecision,
    work_unit_id: int | None = None,
) -> None:
    """Persist an adaptation decision through the canonical runtime event log."""

    store.record_event(
        event_type=ADAPTATION_EVENT_TYPE,
        run_id=run_id,
        work_unit_id=work_unit_id,
        payload=decision.to_dict(),
    )
