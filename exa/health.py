"""Exa launch-readiness probe.

Provides a sync ``probe_exa_readiness()`` the launcher and preflight
surfaces wrap to surface readiness failures BEFORE a run fires off.

The underlying check uses :class:`exa.client.ExaClient`'s
``validate_credentials()`` async method (the cheapest real search).
This wrapper handles the env-var presence check, asyncio-loop
boilerplate, and translates failures into editorial blockers.

Mirrors the shape of :mod:`github.health` so a readiness surface can
union the sources without per-source branching.
"""

from __future__ import annotations

import asyncio
import logging

from exa import config as exa_config
from shared.health_types import ReadinessBlocker, ReadinessReport

logger = logging.getLogger(__name__)


async def _async_validate(api_key: str) -> tuple[bool, str | None, str | None]:
    """Probe Exa with the cheapest real search using the existing client.

    Returns ``(ok, error_message, failure_kind)`` where ``failure_kind``
    is one of ``"auth"``, ``"credits"``, ``"net"`` or ``None``. Exa's
    typed errors carry the distinction, so the caller maps a named
    failure rather than pattern-matching prose the way GitHub must.
    """

    # Late import to avoid pulling aiohttp / certifi at module import time
    # (a readiness surface imports exa.health on every check; the heavy
    # ExaClient deps only load when a real probe fires).
    try:
        from exa.client import ExaAuthError, ExaClient, ExaCreditsError
    except ImportError as exc:
        return (False, f"ExaClient import failed: {exc}", "net")

    try:
        async with ExaClient(api_key=api_key) as client:
            await client.validate_credentials()
        return (True, None, None)
    except ExaAuthError as exc:
        return (False, str(exc), "auth")
    except ExaCreditsError as exc:
        return (False, str(exc), "credits")
    except RuntimeError as exc:
        return (False, str(exc), "net")
    except Exception as exc:
        return (False, f"unexpected error: {exc}", "net")


def probe_exa_readiness(*, api_key: str | None = None) -> ReadinessReport:
    """Launch-readiness probe for Exa.

    Synchronous wrapper. Calls :class:`exa.client.ExaClient`'s
    ``validate_credentials()`` under ``asyncio.run`` so a sync caller
    can use this directly.

    Args:
        api_key: Override the EXA_API_KEY env var. Defaults to
            ``exa.config.EXA_API_KEY`` (the same source ExaClient uses).

    Returns:
        :class:`ReadinessReport` with ``ready`` true iff the key is
        present + valid + the Exa API is reachable + credits remain.
    """

    blockers: list[ReadinessBlocker] = []

    effective_key = api_key or exa_config.EXA_API_KEY
    if not effective_key:
        blockers.append(
            ReadinessBlocker(
                kind="config",
                message="No Exa API key configured.",
                remediation=(
                    "Add EXA_API_KEY to your .env file. "
                    "Keys live at dashboard.exa.ai."
                ),
            )
        )
        return ReadinessReport(ready=False, blockers=tuple(blockers))

    try:
        ok, err, failure_kind = asyncio.run(_async_validate(effective_key))
    except RuntimeError:
        # asyncio.run can't run inside an existing loop. Callers are sync,
        # so this shouldn't happen in production — surface a config blocker
        # so the recruiter sees something rather than a crash.
        blockers.append(
            ReadinessBlocker(
                kind="config",
                message="Exa readiness probe ran from an async context.",
                remediation=(
                    "This is an internal error. Please report it; "
                    "the launch was not blocked by your setup."
                ),
            )
        )
        return ReadinessReport(ready=False, blockers=tuple(blockers))

    if not ok:
        if failure_kind == "auth":
            blockers.append(
                ReadinessBlocker(
                    kind="auth",
                    message="Exa rejected your API key.",
                    remediation=(
                        "Your EXA_API_KEY may have been revoked or rotated. "
                        "Generate a new key at dashboard.exa.ai and update "
                        "your .env file."
                    ),
                )
            )
        elif failure_kind == "credits":
            blockers.append(
                ReadinessBlocker(
                    kind="config",
                    message="Exa account is out of credits.",
                    remediation=(
                        "Top up the Exa account at dashboard.exa.ai "
                        "(Billing) before launching. Cloris stopped before "
                        "spending anything."
                    ),
                )
            )
        else:
            blockers.append(
                ReadinessBlocker(
                    kind="net",
                    message="Cloris couldn't reach the Exa API.",
                    remediation=(
                        "Check your internet connection. "
                        f"Underlying error: {err or 'unknown'}"
                    ),
                )
            )
        return ReadinessReport(ready=False, blockers=tuple(blockers))

    return ReadinessReport(ready=True, blockers=())


__all__ = ["probe_exa_readiness"]
