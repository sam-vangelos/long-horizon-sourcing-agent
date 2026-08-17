"""Executive Search launch-readiness probe.

Phase 2.2 of the multi-agent-execution plan. Provides a callable
``probe_exec_search_readiness()`` that the launch-readiness aggregator
at ``cloris/api/_monolith.py:_readiness_blockers`` dispatches via
``LAUNCHERS["exec_search"].readiness_probe_fn()``.

Mirrors :mod:`linkedin.health` / :mod:`github.health` so the API layer
can union the four sources without per-source branching at the route
layer (post-Slice 1.1 the dispatch is registry-driven; this slice fills
the registry slot for exec_search).

What we surface as blockers:

- ``ANTHROPIC_API_KEY`` missing — config blocker. The dossier-eval
  pipeline (Slice 2+ — currently a Slice 1 stub) extends the LinkedIn
  full-eval branch with a ``DOSSIER_RATIONALE:`` block; both sides
  call Claude. Without a key, the eventual run fails at the first
  judgment call.

What we deliberately do NOT block on:

- ``CRUNCHBASE_API_KEY`` / ``NEWSAPI_KEY`` / ``PITCHBOOK_API_KEY``
  — these are off-LinkedIn signal adapters that gracefully degrade
  to ``SignalFailure(reason="disabled_no_api_key")`` (see
  ``exec_search/signals/crunchbase.py:172-180``,
  ``exec_search/signals/news.py:286-294``,
  ``exec_search/signals/pitchbook.py:166-174``). The recruiter sees
  an "honest signal-unavailable placeholder" in the dossier section
  rather than a launch-blocking gate. Surfacing them as blockers
  would be over-strict — partial dossier coverage is the explicit
  product contract.
- LinkedIn CDP reachability. The exec_search pipeline reuses the
  LinkedIn evaluation pipeline downstream; CDP readiness is checked
  by :func:`linkedin.health.probe_linkedin_readiness` when the
  recruiter launches LinkedIn directly. Cross-source coupling at
  pre-flight time isn't the right shape — the exec_search probe
  reports its own blockers and the LinkedIn pre-flight reports
  LinkedIn's.

Out of scope: network probing Anthropic. Pre-flight is meant to be
fast and offline-tolerant.
"""

from __future__ import annotations

from shared import config
from shared.health_types import ReadinessBlocker, ReadinessReport


def probe_exec_search_readiness(
    *,
    anthropic_api_key: str | None = None,
) -> ReadinessReport:
    """Launch-readiness probe for Executive Search.

    Args:
        anthropic_api_key: Override the ``ANTHROPIC_API_KEY`` env var.
            Defaults to ``shared.config.ANTHROPIC_API_KEY``.

    Returns:
        :class:`ReadinessReport` with ``ready`` true iff every check
        passed.
    """

    blockers: list[ReadinessBlocker] = []

    effective_anthropic = (
        anthropic_api_key
        if anthropic_api_key is not None
        else config.ANTHROPIC_API_KEY
    )
    if not effective_anthropic:
        blockers.append(
            ReadinessBlocker(
                kind="config",
                message="No Anthropic API key configured.",
                remediation=(
                    "Add ANTHROPIC_API_KEY to your .env file. "
                    "Executive Search's dossier evaluation extends "
                    "the LinkedIn full-judge branch — without a key "
                    "the run fails at the first judgment call."
                ),
            )
        )

    return ReadinessReport(ready=not blockers, blockers=tuple(blockers))
