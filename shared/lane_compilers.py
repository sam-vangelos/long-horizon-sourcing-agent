"""Cross-module lane compiler contracts — P9.

Defines a source-neutral protocol for compiling SourcingLane / LaneVariant
into source-native executable searches.  The shared layer owns the protocol
and result envelope; each source adapter owns compilation, evidence
interpretation, and save semantics.

No source-specific imports belong in this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from shared.sourcing_lanes import LaneVariant, SourcingLane


@dataclass(frozen=True)
class LaneCompilerFinding:
    """A single structured finding from compiling or linting a lane."""

    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    dimension: str | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.dimension is not None:
            d["dimension"] = self.dimension
        if self.source is not None:
            d["source"] = self.source
        return d


@dataclass(frozen=True)
class ExecutableSearch:
    """Source-native executable search produced by a LaneCompiler.

    ``query_payload`` is opaque to shared code — each source adapter defines
    its internal structure.  The shared layer can serialize it (via
    ``to_dict``) without understanding the payload schema.
    """

    source: str
    acquisition_mode: str
    display_name: str
    query_payload: Mapping[str, Any] = field(default_factory=dict)
    lane_id: str = ""
    variant_id: str = ""
    unsupported_dimensions: tuple[str, ...] = ()
    warnings: tuple[LaneCompilerFinding, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "acquisition_mode": self.acquisition_mode,
            "display_name": self.display_name,
            "query_payload": dict(self.query_payload),
            "lane_id": self.lane_id,
            "variant_id": self.variant_id,
            "unsupported_dimensions": list(self.unsupported_dimensions),
            "warnings": [w.to_dict() for w in self.warnings],
        }


@runtime_checkable
class LaneCompiler(Protocol):
    """Protocol that source adapters implement to compile lanes."""

    source: str

    def compile(
        self,
        lane: SourcingLane,
        variant: LaneVariant | None = None,
    ) -> ExecutableSearch: ...

    def lint(
        self,
        lane: SourcingLane,
        variant: LaneVariant | None = None,
    ) -> list[LaneCompilerFinding]: ...
