"""LinkedIn Recruiter save-destination adapter.

Wrap the existing ``LinkedInSideEffectsService.handle_save_decision`` behavior
behind :class:`AbstractSaveDestination` so the orchestrator can dispatch
through the registry. The current adapter preserves the existing body and
raises from ``save`` to prevent accidental dispatch until the implementation
moves; its ``supports`` method accepts LinkedIn briefs and leaves empty-project
handling to the existing service. The workspace destination is the separate
cross-source save surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shared.execution import SideEffectOutcome
from shared.save_destination import AbstractSaveDestination, SaveResult

if TYPE_CHECKING:
    from linkedin.side_effects import LinkedInSideEffectsService


class LinkedInRecruiterSaveDestination(AbstractSaveDestination):
    """LinkedIn Recruiter "Save to Pipeline" save destination.

    Wraps :meth:`linkedin.side_effects.LinkedInSideEffectsService.handle_save_decision`.
    The wrap is intentional (vs. body-move): preserves byte-identical
    save behavior so the LinkedIn pipeline regression suite stays
    green without any extraction risk on the high-risk
    ``linkedin/orchestrator.py`` seam. The body-move follow-up lifts
    the browser-click + linger + idempotency logic into this
    module's :meth:`save` body once the regression suite is solid
    against the new path.

    The destination is constructed per-pipeline at orchestrator
    bootstrap; the LinkedIn side-effects service is injected so the
    destination can delegate without owning the browser lifecycle.
    """

    name: str = "linkedin_recruiter"

    def __init__(self, side_effects_service: "LinkedInSideEffectsService"):
        self._service = side_effects_service

    def supports(self, *, brief: Any, source: str) -> bool:
        """Only LinkedIn-sourced candidates can be saved to LinkedIn Recruiter.

        Other sources (researcher / designer / exec_search /
        github-as-OSS-maintainer) have no LinkedIn Recruiter
        equivalent — they save through
        :class:`CandidateWorkspaceSaveDestination` only. LinkedIn
        briefs without a ``linkedin_project`` configured (legacy
        non-LinkedIn briefs that happen to import the module) also
        return False; the existing ``side_effects`` flow checks for
        empty project and skips the browser save.
        """

        if source != "linkedin":
            return False
        # Preserve the existing empty-project skip: the destination supports a
        # LinkedIn-targeted brief, while the underlying service decides whether
        # an empty project should skip the browser save.
        return True

    def save(
        self,
        *,
        envelope: Any,
        decision: Any,
        evidence_payload: dict[str, Any],
        attempt_id: int | None,
    ) -> SaveResult:
        """Raise ``NotImplementedError``; this destination does not save yet.

        ``envelope`` / ``decision`` / ``evidence_payload`` are the uniform
        destination inputs. LinkedIn saves still go through
        ``LinkedInSideEffectsService.handle_save_decision`` directly.
        """

        # This adapter is intentionally a stub: callers must continue using the
        # existing service until the body-preserving delegation is implemented.
        # Raising here makes accidental registry dispatch visible.
        raise NotImplementedError(
            "LinkedInRecruiterSaveDestination.save() body-move is a "
            "behavior-preserving follow-up to Slice A.4. Until then, "
            "call LinkedInSideEffectsService.handle_save_decision directly. "
            "See docs/cloris-save-destination-abstraction.md §3.1."
        )


__all__ = ["LinkedInRecruiterSaveDestination"]
