"""Off-LinkedIn signal interface for executive-search dossier evaluation.

Slice 3 of the executive-search module ships the interface, the
:class:`SIGNAL_REGISTRY` dict, and the Perplexity adapter (Slice 3
baseline). Slices 4-5 add news / Crunchbase / PitchBook adapters that
extend the same interface.

Design contract:

- Per-source adapters implement :class:`ExecutiveSignalSource`. Each
  adapter is responsible for ITS source's API surface and failure
  modes; the registry just dispatches.
- Adapters NEVER raise out to callers. A failed signal returns a
  :class:`SignalFailure`, not an exception. The dossier eval degrades
  gracefully (signal section redacted) rather than failing the whole
  evaluation.
- Signal output is recruiter-readable prose. Adapters render their
  source's payload into a single dossier-section string. The
  per-section formatter is the adapter's responsibility, not the
  evidence-assembly module's.
- The registry is module-scope and immutable post-import; sources
  cannot be added at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from shared.brief_schema import Brief
from shared.schemas import CandidateProfileSummary


@dataclass(frozen=True)
class SignalRequestContext:
    """Per-call context for a signal fetch.

    Carries the per-search trigger reason (why this signal is being
    fetched), the brief id (for telemetry), and arbitrary identity
    hints (e.g., known company tickers, published board memberships)
    that the adapter can use to tighten its query.
    """

    brief_id: str
    trigger_reason: str = "dossier_full_eval"
    identity_hints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalResult:
    """Successful per-source signal acquisition.

    ``section_text`` is the recruiter-readable prose the
    evidence-assembly module folds into the dossier prompt's user
    message. ``citations`` are the source URLs / IDs the adapter
    pulled from. ``raw_payload`` is the unmodified API response for
    debugging; downstream consumers should not rely on its shape.
    """

    source: str
    section_text: str
    citations: tuple[str, ...] = ()
    raw_payload: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SignalFailure:
    """Typed failure result from a signal adapter.

    Mirrors :class:`shared.schemas.ExternalEvidenceFailure` shape.
    Adapters return this instead of raising so a single source's
    outage doesn't break the dossier eval — the evidence-assembly
    module renders a "signal unavailable" section in its place.

    ``reason`` is a short machine-readable token (e.g.
    ``"disabled_by_config"``, ``"quota_exhausted"``, ``"timeout"``,
    ``"upstream_5xx"``); ``detail`` is recruiter-readable.
    """

    source: str
    reason: str
    detail: str = ""


class ExecutiveSignalSource(Protocol):
    """One off-LinkedIn signal source for dossier evaluation."""

    name: str

    def fetch(
        self,
        *,
        candidate: CandidateProfileSummary,
        brief: Brief,
        context: SignalRequestContext,
    ) -> SignalResult | SignalFailure:
        ...


# Module-scope source registry. Adding a new signal source is a
# single-line append below. Keep keys lowercase to match the URL /
# CLI flag conventions used elsewhere.
#
# Slice 3 ships only the Perplexity baseline. Slice 4 appends
# ``"news"``, Slice 5 appends ``"crunchbase"`` and ``"pitchbook"``.
# The registry is read-only post-import; tests import this dict by
# name to assert membership.
def _build_default_registry() -> dict[str, ExecutiveSignalSource]:
    """Build the registry lazily so import order is robust against
    optional dependencies in individual adapters (later slices)."""

    from exec_search.signals.crunchbase import CrunchbaseSignalSource
    from exec_search.signals.news import NewsSignalSource
    from exec_search.signals.perplexity import PerplexitySignalSource
    from exec_search.signals.pitchbook import PitchBookSignalSource

    return {
        PerplexitySignalSource.name: PerplexitySignalSource(),
        NewsSignalSource.name: NewsSignalSource(),
        CrunchbaseSignalSource.name: CrunchbaseSignalSource(),
        PitchBookSignalSource.name: PitchBookSignalSource(),
    }


SIGNAL_REGISTRY: dict[str, ExecutiveSignalSource] = _build_default_registry()


def get_signal_source(name: str) -> ExecutiveSignalSource:
    """Return the registered signal adapter for ``name``.

    Raises :class:`KeyError` for unknown sources; callers that need
    structured fallback should ``name in SIGNAL_REGISTRY`` first.
    """

    return SIGNAL_REGISTRY[name]


def known_signal_sources() -> tuple[str, ...]:
    """Return the registered signal source names in stable ascending order."""

    return tuple(sorted(SIGNAL_REGISTRY.keys()))


__all__ = (
    "ExecutiveSignalSource",
    "SIGNAL_REGISTRY",
    "SignalFailure",
    "SignalRequestContext",
    "SignalResult",
    "get_signal_source",
    "known_signal_sources",
)
