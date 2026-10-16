"""Stable exception classes shared by the loop and its compatibility facade.

Legacy public qualified names stay under linkedin.orchestrator so existing
tracebacks, imports, and pickled exceptions retain their identity contract.
"""
from __future__ import annotations

class GeographyRegimeError(RuntimeError):
    """Raised when the brief's session geography cannot be applied AND verified.

    Geography is a fail-closed precondition of searching, not a monitored
    outcome (plans/sourcing-rigor-hardening.md). Failing soft (log one line,
    proceed boolean-only) lets every save come back off-geography. Same
    doctrine as PreflightRegimeError: a run on the wrong pool is worse than
    no run.
    """

    def __init__(self, *args, retryable: bool = False):
        super().__init__(*args)
        self.retryable = retryable


class PageRenderFailedError(RuntimeError):
    """Raised when a results page failed to render reviewable card slots."""


class PanelRecoveryError(RuntimeError):
    """Raised when bounded profile-panel recovery cannot restore results state."""


class TransientPaginationError(RuntimeError):
    """Raised when result-count evidence says a missing next page is retryable."""


class SurfaceOwnershipLostError(RuntimeError):
    """A mid-string browser recovery re-navigated the search surface (F11).

    The re-navigation loses the owner's exact facet state and page cursor —
    the recruiterSearch URL does not carry the pagination cursor, and facet
    restore after navigation is LinkedIn's saved-search state, not a proven
    copy of the active variant. Continuing would read candidates from an
    unverified pool and position. Raised only under
    LINKEDIN_FACET_LIFECYCLE_CLEARING_ENABLED, and only when no owner
    restore follows in the same flow; it is a NAMED, resumable stop: the
    string stays in_progress and the resume path re-enters through
    _apply_opening_search (full reconcile + committed-cursor fast-forward).
    """


class UnconfirmedSaveError(RuntimeError):
    """A failed save outcome retaining the browser's structured refusal reason."""

    def __init__(self, failure_reason: str | None = None):
        super().__init__("LinkedIn save was not durably confirmed")
        self.failure_reason = failure_reason


class ProjectContextMismatchError(RuntimeError):
    """Raised when the live Recruiter page belongs to a different project.

    A Recruiter save click files the candidate into whichever project the
    visible page belongs to, so a page bound from another project's tab lands
    physical saves in the wrong pipeline. Same fail-closed doctrine as
    GeographyRegimeError: a save into the wrong pipeline is worse than no save.
    """


for _exception in (
    GeographyRegimeError,
    PageRenderFailedError,
    PanelRecoveryError,
    TransientPaginationError,
    SurfaceOwnershipLostError,
    ProjectContextMismatchError,
):
    _exception.__module__ = "linkedin.orchestrator"
del _exception
