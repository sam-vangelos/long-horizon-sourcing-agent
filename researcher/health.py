"""Researcher launch-readiness probe.

Phase 2.2 of the multi-agent-execution plan. Provides a callable
``probe_researcher_readiness()`` that the launch-readiness aggregator
at ``cloris/api.py:_readiness_blockers`` dispatches via
``LAUNCHERS["researcher"].readiness_probe_fn()``.

Mirrors :mod:`linkedin.health` / :mod:`github.health` so the API layer
can union the four sources without per-source branching at the route
layer (post-Slice 1.1 the dispatch is registry-driven; this slice fills
the registry slot for researcher).

What we surface as blockers:

- ``ANTHROPIC_API_KEY`` missing — config blocker. The pipeline's
  facial / full / strategy judges (``shared.judger.researcher_*``)
  call Opus; a missing key fails the run at first judge call.

What we deliberately do NOT block on:

- ``OPENALEX_POLITE_POOL_EMAIL`` missing — etiquette signal, not a
  hard requirement. OpenAlex is queryable without it; the polite
  pool just gives more generous rate limits and an identifiable
  User-Agent. The orchestrator at
  ``researcher/session_orchestrator.py:75`` reads the env var and
  passes ``""`` when unset, which OpenAlex accepts. Surfacing this
  as a launch blocker would gate the recruiter on a non-fatal
  config item; matches the linkedin/github precedent of "only
  block on what would actually fail the run."

Out of scope: network probing OpenAlex / Anthropic. Pre-flight is
meant to be fast and offline-tolerant; transient outages are caught
at request time by ``researcher.sources._rate_limit`` and surfaced
as run-level errors, not launch-readiness blockers.
"""

from __future__ import annotations

from shared import config
from shared.health_types import ReadinessBlocker, ReadinessReport


def probe_researcher_readiness(
    *,
    anthropic_api_key: str | None = None,
) -> ReadinessReport:
    """Launch-readiness probe for Researcher.

    Args:
        anthropic_api_key: Override the ANTHROPIC_API_KEY env var.
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
                    "Researcher's identity, facial, and full judges "
                    "all call Claude — without a key the run fails at "
                    "the first judgment step."
                ),
            )
        )

    return ReadinessReport(ready=not blockers, blockers=tuple(blockers))
