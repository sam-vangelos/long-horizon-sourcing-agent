"""Shared evaluation classifiers and constants; no execution state."""

from __future__ import annotations

import logging
from linkedin.acquisition import (
    _is_browser_disconnect_error as _acquisition_is_browser_disconnect_error,
)
from linkedin.browser import _recruiter_page_project_id
import re as _re
from shared.schemas import OpusDecision
from linkedin.page_allocator import AllocatorPolicyError
from linkedin.facial_batching import FacialBatchContractError
from linkedin.judgment_tool_contracts import JudgmentToolContractError
from shared.failures import (
    ApiBudgetExhaustedError,
    classify_runtime_failure,
    is_api_budget_exhausted_error,
)
from shared.contracts import LEGACY_REVIEW_DECISIONS
from shared.governor import GovernorLimitReached, OperatorStopRequested, SessionExpired
from linkedin.evaluation_errors import (
    GeographyRegimeError,
    PageRenderFailedError,
    PanelRecoveryError,
    TransientPaginationError,
    SurfaceOwnershipLostError,
    ProjectContextMismatchError,
)

logger = logging.getLogger("linkedin.orchestrator")


_PANEL_RECOVERY_MAX_ATTEMPTS = 2


_PANEL_MOUNT_RELOAD_AT = 3


_PANEL_MOUNT_TRIP_AT = 6


_PANEL_DOWN_COOLDOWN_SECONDS = 30 * 60


_PANEL_MOUNT_TIMEOUT_RE = _re.compile(
    r"Timeout[\s\S]*profile-slidein|profile-slidein[\s\S]*Timeout"
)


_LINKEDIN_RESULTS_PAGE_SIZE = 25


def _is_browser_disconnect_error(error: BaseException | str) -> bool:
    return _acquisition_is_browser_disconnect_error(
        error,
        include_render_failures=True,
    )


_PROFILE_PAGE_RE = _re.compile(r"/profile/")


_SEARCH_PAGE_RE = _re.compile(r"/discover/|/talent/search")


def _is_recruiter_search_page(url: str) -> bool:
    """True only for a Recruiter SEARCH view — the page with the filter sidebar.

    Treating a profile-under-discover page as a search page skips run-start
    navigation and every filter apply silently gets an empty typeahead.
    """
    text = str(url or "")
    if _PROFILE_PAGE_RE.search(text):
        return False
    return bool(_SEARCH_PAGE_RE.search(text))


def _is_foreign_project_page(url: str, expected_project_id: object) -> bool:
    """True when the page is not PROVABLY the brief's Recruiter project.

    E4: `_bind_existing_recruiter_page` (linkedin/browser.py) binds the first
    healthy Recruiter tab without matching the brief, and a save click files the
    candidate into whichever project the page belongs to — so a tab from another
    project silently saves into the wrong pipeline.

    The rule is asymmetric, and the asymmetry is the correction to E4's original
    "absence of either id is not a mismatch", which was a bypass rather than a
    carve-out: a page with no project id in its URL (the global /talent/search
    view, a bare /talent/profile/<id> page) answered False, so run-start skipped
    navigation and the pre-save boundary passed a projectless page through to
    the click.

    - Brief pins no project → nothing can be violated; every page is accepted,
      which is the one place the permissive carve-out was ever justified.
    - Brief pins a project → the page must NAME that project. An absent or
      unparseable page project is UNVERIFIED, and unverified is a mismatch:
      navigate at run-start, refuse at the pre-save boundary.
    """
    expected = str(expected_project_id or "").strip()
    if not expected:
        return False
    actual = _recruiter_page_project_id(url)
    if not actual:
        return True
    return actual != expected


_FULL_ID_MISMATCH_RECOVERY_CEILING = 3


_FULL_ID_MISMATCH_ACTUAL_ID_MAX_CHARS = 64


_FACIAL_READ_INTEREST = {"FACIAL_YES": 0.9, "FACIAL_BORDERLINE": 0.35}


_DEFAULT_READ_INTEREST = 0.5


def _is_containable_facial_page_error(error: BaseException) -> bool:
    """True only for facial provider/transport and attribution-contract faults."""

    if not isinstance(error, Exception):
        return False
    if isinstance(
        error,
        (
            ApiBudgetExhaustedError,
            GovernorLimitReached,
            OperatorStopRequested,
            SessionExpired,
            GeographyRegimeError,
            PageRenderFailedError,
            PanelRecoveryError,
            TransientPaginationError,
            SurfaceOwnershipLostError,
            ProjectContextMismatchError,
            AllocatorPolicyError,
        ),
    ):
        return False
    if _is_browser_disconnect_error(error):
        return False
    if is_api_budget_exhausted_error(error):
        return False
    if isinstance(error, (FacialBatchContractError, JudgmentToolContractError)):
        return True
    if str(error).startswith("Fireworks tool contract "):
        return True
    classification = classify_runtime_failure(error, source="judgment")
    return classification.domain in {"network", "provider"}


def is_legacy_review_decision(decision: OpusDecision) -> bool:
    """True when a verdict still names a retired review outcome.

    The P4 structural-evidence guard this replaced (2026-09-16) decided
    whether a review outcome had earned its parking spot. There is no
    parking spot any more: the agent is the reviewer, so the only
    question left is whether a verdict is speaking the retired
    vocabulary at all — and the answer routes it to REJECT.
    """

    return decision.decision in LEGACY_REVIEW_DECISIONS


def _clear_review_evidence(decision: OpusDecision) -> None:
    """Clear the retired review_* fields on an ``OpusDecision`` in place.

    Used by the orchestrator's retirement backstop so the canonical row
    never carries half-populated review metadata after the decision
    flips to ``REJECT``.
    """

    decision.review_reason_code = ""
    decision.review_structural_evidence = []
    decision.review_recommended_next_step = ""
    decision.review_ambiguity_reason = ""
