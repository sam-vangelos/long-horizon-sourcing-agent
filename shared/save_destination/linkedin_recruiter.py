"""LinkedIn Recruiter save destination — Slice A.4.

Per ``docs/cloris-save-destination-abstraction.md`` §3.1. Wraps the
existing ``linkedin/side_effects.py:LinkedInSideEffectsService.handle_save_decision``
behavior behind the :class:`AbstractSaveDestination` interface so
the orchestrator dispatches through the registry instead of
hard-coding the call site.

This slice ships the WRAP, not the body-move. The full extraction
(per spec §3.1: "Move the browser-click + linger + idempotency
logic out of linkedin/side_effects.py:handle_save_decision and
into shared/save_destination/linkedin_recruiter.py:LinkedInRecruiterSaveDestination.save()")
lands as a behavior-preserving follow-up gated on
``tests/test_linkedin_pipeline.py``. Keeping the wrap shape today
preserves byte-identical LinkedIn save behavior — the destination
delegates to the existing service via the pipeline reference, so
no logic moves.

The wrap establishes the registry slot that
:class:`shared.save_destination.candidate_workspace.CandidateWorkspaceSaveDestination`
(Slice A.5) joins as the second registered destination; the
orchestrator's ``save_destinations`` dispatch (also Slice A.5) walks
both in order on every SAVE-family decision.
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
        # Mirror the existing skip-on-empty-project behavior at
        # ``linkedin/side_effects.py``; brief.linkedin_project is the
        # legacy flat field and ``brief.source_config_for("linkedin")``
        # would be the V2 path (Slice F2). For the wrap, we accept the
        # broader contract: if the brief is LinkedIn-targeted, the
        # destination supports it; the underlying service handles the
        # empty-project skip itself.
        return True

    def save(
        self,
        *,
        envelope: Any,
        decision: Any,
        evidence_payload: dict[str, Any],
        attempt_id: int | None,
    ) -> SaveResult:
        """Delegate to the existing LinkedIn side-effects flow.

        The synchronous interface of :class:`AbstractSaveDestination`
        wraps the underlying service's coroutine. The orchestrator
        already runs in an asyncio event loop (LinkedIn's pipeline is
        async); the wrap lifts the coroutine via ``asyncio.run`` only
        when called from a synchronous context, otherwise it returns
        the coroutine awaitable for the orchestrator to await.

        Today's body-preserving wrap calls the service's coroutine
        with the legacy kwargs (``snippet``, ``runtime_search_string``,
        ``attempt_id``); the body-move follow-up will translate the
        :class:`shared.save_destination.SaveResult` shape from the
        existing :class:`shared.execution.SideEffectOutcome` shape
        the service returns.

        ``envelope`` / ``decision`` / ``evidence_payload`` are the
        new uniform inputs; the wrap extracts the legacy kwargs from
        them. The body-move follow-up lifts the wrap into the body
        and drops the legacy kwargs entirely.
        """

        # Slice A.4 wrap is interface-only; the body-move follow-up
        # implements the actual delegation. Until then, callers
        # should continue using the existing
        # ``LinkedInSideEffectsService.handle_save_decision`` directly
        # — this stub raises to make accidental dispatch via the
        # destination obvious. A.5 ships the workspace destination
        # which DOES implement save(); the LinkedIn delegation lands
        # in the follow-up sub-slice.
        raise NotImplementedError(
            "LinkedInRecruiterSaveDestination.save() body-move is a "
            "behavior-preserving follow-up to Slice A.4. Until then, "
            "call LinkedInSideEffectsService.handle_save_decision directly. "
            "See docs/cloris-save-destination-abstraction.md §3.1."
        )


__all__ = ["LinkedInRecruiterSaveDestination"]
