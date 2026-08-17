"""GitHub launch-readiness probe.

Phase D Slice D-prep-B. Provides a sync ``probe_github_readiness()`` the
API layer (`GET /api/launch-readiness/github/{brief_id}`) wraps to surface
readiness failures BEFORE a worker fires off.

The underlying check uses :class:`github.client.GitHubClient`'s existing
``validate_credentials()`` async method (which probes ``/rate_limit``).
This wrapper handles the env-var presence check, asyncio-loop boilerplate,
and translates RuntimeErrors into editorial blockers.

Mirrors the shape of :mod:`linkedin.health` so the API endpoint can
union the two without per-source branching at the route layer.
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

logger = logging.getLogger(__name__)

# OSS Maintainers module (Slice 2, `shared/brief_v2_schema.py:317-324`):
# `maintainership_level` values above the default "contributor" are the
# recruiter's explicit declaration of OSS-Maintainers posture — a brief
# can carry this WITHOUT `target_projects` (the two fields validate
# independently; see `shared/brief_v2_schema.py:481-500`). `target_projects`
# itself is not a separate posture flag: an empty list is indistinguishable
# from an absent key once the brief is loaded (both default to `[]` at
# `shared/brief_loader.py:478` / `shared/brief_schema.py:364`), so
# "declared-but-empty target_projects" cannot be represented on the wire.
# `_ELEVATED_MAINTAINERSHIP_LEVELS` is therefore the real posture marker
# P6.9 gates on. Mirrors the same posture test `shared/role_strategy.py:705`
# uses to route a brief to the `oss_maintainer` evaluation archetype.
_ELEVATED_MAINTAINERSHIP_LEVELS = frozenset({"maintainer", "project_lead"})


def github_target_projects_blocker(
    *,
    target_projects: list[str] | None,
    maintainership_level: str,
) -> ReadinessBlocker | None:
    """P6.9 module-readiness gate — OSS-Maintainers posture needs targets.

    This is a per-brief check (no network, no token), so it does not run
    inside :func:`probe_github_readiness` (that probe is brief-agnostic —
    see its docstring). It's dispatched instead from the launcher
    registry's brief-aware slot: ``cloris.launchers._github_save_destination_blocker``,
    registered as ``LAUNCHERS["github"].save_destination_blocker_fn``.

    Two real states, one impossible state:

    - ``target_projects`` non-empty ⇒ OSS Maintainers posture, requirement
      trivially satisfied ⇒ no blocker. ``maintainership.py:184`` runs
      classification normally.
    - ``target_projects`` empty AND ``maintainership_level`` elevated
      (``"maintainer"`` or ``"project_lead"`` — see
      ``shared/brief_v2_schema.py:317-324``) ⇒ the recruiter declared OSS
      Maintainers posture but gave the classifier nothing to check
      against. ``maintainership.py:176`` returns ``None`` unconditionally
      when ``target_projects`` is empty, so this brief would silently run
      as if the posture were never declared. Block with a named,
      recruiter-actionable reason instead.
    - ``target_projects`` empty AND ``maintainership_level`` is the
      default ``"contributor"`` (or unset) ⇒ classic GitHub sourcing,
      unchanged behavior — logged, not blocked. There is no fourth state:
      "posture declared via an explicitly-empty ``target_projects``" is
      not representable (an absent key and an empty list both load as
      ``[]`` — see module docstring note above `_ELEVATED_MAINTAINERSHIP_LEVELS`).
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
