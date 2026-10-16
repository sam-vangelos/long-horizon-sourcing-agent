"""GitHub launch-readiness probe.

Provides synchronous ``probe_github_readiness()`` for the API's brief-agnostic
readiness surface. It checks credentials through the existing async client,
handles environment and event-loop setup, and converts runtime failures into
editorial blockers. Its report shape matches ``linkedin.health`` so callers
can combine sources without route-specific branching.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request

from github import config as github_config
from github.hubs.crates import API_BASE as _CRATES_API_BASE
from github.hubs.crates import USER_AGENT as _CRATES_USER_AGENT
from github.hubs.npm import REGISTRY_BASE as _NPM_REGISTRY_BASE
from github.hubs.npm import USER_AGENT as _NPM_USER_AGENT
from shared.health_types import ReadinessBlocker, ReadinessReport
from github.hubs.hf import API_BASE as _HF_API_BASE
from github.hubs.hf import USER_AGENT as _HF_USER_AGENT
from github.hubs.pypi import API_BASE as _PYPI_API_BASE
from github.hubs.pypi import USER_AGENT as _PYPI_USER_AGENT

logger = logging.getLogger(__name__)

# Elevated ``maintainership_level`` values explicitly declare OSS-maintainer
# posture. ``target_projects`` and the level validate independently; an empty
# target list loads like an absent key, so elevated level is the real posture
# marker used by this readiness gate. The default contributor level preserves
# classic GitHub sourcing.
_ELEVATED_MAINTAINERSHIP_LEVELS = frozenset({"maintainer", "project_lead"})


def github_target_projects_blocker(
    *,
    target_projects: list[str] | None,
    maintainership_level: str,
) -> ReadinessBlocker | None:
    """Return a blocker when elevated OSS-maintainer posture lacks targets.

    This per-brief check performs no network or token work and runs through the
    launcher registry's brief-aware blocker slot. Non-empty ``target_projects``
    returns no blocker. Empty targets plus ``maintainer`` or ``project_lead``
    returns a named blocker, because the maintainership classifier returns
    ``None`` whenever ``target_projects`` is empty and the brief would otherwise
    run as if the posture were never declared. Empty targets plus default or unset contributor
    posture preserves classic GitHub sourcing.
    """

    if target_projects:
        return None

    if maintainership_level in _ELEVATED_MAINTAINERSHIP_LEVELS:
        return ReadinessBlocker(
            kind="config",
            message=(
                f"This brief declares maintainership_level="
                f"{maintainership_level!r} (OSS Maintainers posture) but "
                "lists no target_projects."
            ),
            remediation=(
                "Add at least one \"owner/repo\" to target_projects in the "
                "brief, or set maintainership_level back to \"contributor\" "
                "for classic GitHub sourcing. Without target_projects, "
                "maintainership classification never runs and "
                "maintainership_level has no effect."
            ),
        )

    logger.info(
        "github readiness: classic GitHub sourcing (no target_projects, "
        "maintainership_level=%r) — OSS Maintainers posture gate not "
        "applicable",
        maintainership_level,
    )
    return None


async def _async_validate(token: str) -> tuple[bool, str | None]:
    """Probe /rate_limit using the existing GitHubClient.

    Returns ``(ok, error_message)``. ``ok`` is True iff the token validated
    and the API returned 200. On any failure, ``error_message`` carries
    the underlying RuntimeError text so the caller can map it to an
    editorial remediation.
    """

    # Late import to avoid pulling aiohttp / certifi at module import time
    # (the API layer imports github.health on every readiness check; the
    # heavy GitHubClient deps only load when a real probe fires).
    try:
        from github.client import GitHubClient
    except ImportError as exc:
        return (False, f"GitHubClient import failed: {exc}")

    try:
        async with GitHubClient(token=token) as client:
            await client.validate_credentials()
        return (True, None)
    except RuntimeError as exc:
        return (False, str(exc))
    except Exception as exc:
        return (False, f"unexpected error: {exc}")


def probe_github_readiness(*, token: str | None = None) -> ReadinessReport:
    """Launch-readiness probe for GitHub.

    Synchronous wrapper. Calls :class:`github.client.GitHubClient`'s
    ``validate_credentials()`` under ``asyncio.run`` so a sync FastAPI
    route handler can call this directly.

    Args:
        token: Override the GITHUB_TOKEN env var. Defaults to
            ``shared.config.GITHUB_TOKEN`` (the same source GitHubClient
            uses).

    Returns:
        :class:`ReadinessReport` with ``ready`` true iff the token
        present + valid + GitHub API reachable.
    """

    blockers: list[ReadinessBlocker] = []

    effective_token = token or github_config.GITHUB_TOKEN
    if not effective_token:
        blockers.append(
            ReadinessBlocker(
                kind="config",
                message="No GitHub token configured.",
                remediation=(
                    "Add GITHUB_TOKEN to your .env file. "
                    "You can create one at github.com/settings/tokens "
                    "with the `read:org` and `read:user` scopes."
                ),
            )
        )
        return ReadinessReport(ready=False, blockers=tuple(blockers))

    try:
        ok, err = asyncio.run(_async_validate(effective_token))
    except RuntimeError:
        # asyncio.run can't run inside an existing loop. The API layer is
        # sync, so this shouldn't happen in production — surface a config
        # blocker so the recruiter sees something rather than a crash.
        blockers.append(
            ReadinessBlocker(
                kind="config",
                message="GitHub readiness probe ran from an async context.",
                remediation=(
                    "This is an internal error. Please report it; "
                    "the launch was not blocked by your setup."
                ),
            )
        )
        return ReadinessReport(ready=False, blockers=tuple(blockers))

    if not ok:
        # Map RuntimeError text to a Cloris-voice remediation. The
        # underlying validate_credentials raises RuntimeError on any
        # non-200 — most commonly 401 (bad token) or network issue.
        message_lower = (err or "").lower()
        if "401" in message_lower or "preflight failed" in message_lower:
            blockers.append(
                ReadinessBlocker(
                    kind="auth",
                    message="GitHub rejected your token.",
                    remediation=(
                        "Your GITHUB_TOKEN may have been revoked or expired. "
                        "Generate a new token at github.com/settings/tokens "
                        "and update your .env file."
                    ),
                )
            )
        else:
            blockers.append(
                ReadinessBlocker(
                    kind="net",
                    message="Cloris couldn't reach the GitHub API.",
                    remediation=(
                        "Check your internet connection. "
                        f"Underlying error: {err or 'unknown'}"
                    ),
                )
            )
        return ReadinessReport(ready=False, blockers=tuple(blockers))

    return ReadinessReport(ready=True, blockers=())


_REGISTRY_PROBE_TIMEOUT_SECONDS = 5.0


def _sync_registry_http_probe(*, url: str, user_agent: str) -> bool:
    """Cheap synchronous GET for per-hub registry reachability.

    Strategy formation runs inside the pipeline's event loop, where
    ``asyncio.run`` would raise — do not call the async hub-client
    ``probe()`` here.
    """

    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent},
        method="GET",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=_REGISTRY_PROBE_TIMEOUT_SECONDS,
        ) as response:
            return int(getattr(response, "status", 200) or 200) == 200
    except Exception:
        return False


def probe_npm_registry() -> bool:
    """Return True iff the npm registry answers a cheap ping."""

    return _sync_registry_http_probe(
        url=f"{_NPM_REGISTRY_BASE}/-/ping",
        user_agent=_NPM_USER_AGENT,
    )


def probe_crates_registry() -> bool:
    """Return True iff the crates.io API root answers.

    crates.io crawler policy applies even to probes — send the named
    hub-client User-Agent.
    """

    return _sync_registry_http_probe(
        url=f"{_CRATES_API_BASE}/summary",
        user_agent=_CRATES_USER_AGENT,
    )


def probe_pypi_registry() -> bool:
    """Return True iff pypi.org answers a cheap project JSON probe."""

    return _sync_registry_http_probe(
        url=f"{_PYPI_API_BASE}/six/json",
        user_agent=_PYPI_USER_AGENT,
    )


def probe_hf_hub() -> bool:
    """Return True iff the Hugging Face Hub API answers a cheap models listing."""

    return _sync_registry_http_probe(
        url=f"{_HF_API_BASE}/models?limit=1",
        user_agent=_HF_USER_AGENT,
    )
