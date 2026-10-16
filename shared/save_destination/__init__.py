"""Save destination abstraction.

Per ``docs/cloris-save-destination-abstraction.md``. Separates "how a
save is recorded" from "what a save means." The two destinations
are LinkedIn Recruiter (``linkedin_recruiter``) and the Cloris-native
candidate workspace (``candidate_workspace``); further modules
extend the abstraction without re-inventing parallel save semantics.

``LinkedInRecruiterSaveDestination`` is an interface-only adapter: its
``save`` raises, and LinkedIn saves still go through
``linkedin/side_effects.py``'s ``handle_save_decision`` directly.
``CandidateWorkspaceSaveDestination`` writes to the
``workspace_entries`` / ``workspace_review_events`` /
``workspace_outreach_artifacts`` tables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


SAVE_DECISION_FAMILY: frozenset[str] = frozenset({
    "SAVE",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
})


@dataclass
class SaveResult:
    """Outcome of one save destination's persistence attempt.

    ``destination`` is the destination's canonical name (matching
    :attr:`AbstractSaveDestination.name`). ``status`` is one of
    ``"succeeded"`` / ``"failed"`` / ``"skipped"``. ``side_effect_id``
    is the row id in ``runtime_state.sqlite3:side_effects`` when the
    destination wrote one (LinkedIn does; the workspace destination's
    write to ``workspace_entries`` is its own row, not a side-effect).
    ``payload`` carries destination-specific details for downstream
    inspection. ``error`` is set on ``failed`` outcomes.
    """

    destination: str
    status: str
    side_effect_id: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AbstractSaveDestination(ABC):
    """Pluggable destination for SAVE-family decisions.

    A module's side-effects service dispatches each SAVE-family
    decision (``SAVE`` / ``INFERENTIAL_SAVE`` / ``TRANSFERABLE_SAVE`` /
    ``SIGNAL_SAVE``) to one or more destinations declared in the brief.
    Each destination implements its own persistence, side-effect, and
    idempotency semantics.

    Subclasses MUST set :attr:`name` to the canonical destination
    identifier (e.g., ``"linkedin_recruiter"``,
    ``"candidate_workspace"``). The brief's ``save_destinations``
    field declares which destinations apply per
    brief; the orchestrator dispatches to each in order.
    """

    name: str

    @abstractmethod
    def supports(self, *, brief: Any, source: str) -> bool:
        """Whether this destination is applicable for the given brief + source.

        ``LinkedInRecruiterSaveDestination.supports()`` returns False
        when ``source != "linkedin"`` because clicking in Recruiter
        only makes sense for LinkedIn-discovered candidates.
        ``CandidateWorkspaceSaveDestination.supports()`` returns True
        for every (brief, source) — the workspace is universal.
        """

    @abstractmethod
    def save(
        self,
        *,
        envelope: Any,
        decision: Any,
        evidence_payload: dict[str, Any],
        attempt_id: int | None,
    ) -> SaveResult:
        """Record the save.

        Implementations are responsible for:

        - **Idempotency** — don't double-save on retry. Each
          destination has its own idempotency seam (LinkedIn uses the
          ``side_effects`` row keyed on ``(candidate_id, effect_type,
          idempotency_key)``; the workspace uses
          ``workspace_entries.UNIQUE(brief_id, candidate_id)``).
        - **Side-effect record creation** in
          ``runtime_state.sqlite3:side_effects`` when the destination
          performs an external action (LinkedIn does; the workspace
          destination doesn't — its write to ``workspace_entries`` is
          its own audit trail).
        - **Destination-specific writes** — browser click, workspace
          row insert, artifact append.
        - **Failure handling** — return ``SaveResult(status="failed", ...)``
          on recoverable errors; raise on unrecoverable. Recoverable
          failures don't abort the broader candidate flow; the
          orchestrator surfaces them via the existing
          ``SideEffectOutcome`` plumbing.
        """


__all__ = [
    "AbstractSaveDestination",
    "SaveResult",
    "SAVE_DECISION_FAMILY",
]
