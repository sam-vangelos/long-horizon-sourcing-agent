"""LinkedIn launch-readiness probe.

Phase D Slice D-prep-A. Provides a callable ``probe_linkedin_readiness()``
that the API layer (`GET /api/launch-readiness/linkedin/{brief_id}`) wraps
to surface readiness failures BEFORE a worker fires off.

This is intentionally distinct from
:meth:`linkedin.orchestrator.Pipeline._ensure_browser_healthy`, which is a
*recovery* loop for an already-running session. Launch readiness is a
pre-flight check: can we even start? It does NOT spin up a browser; it
asks whether the recruiter has the prerequisites in place.

LinkedIn architecture: Cloris connects to Chrome over CDP at
``config.CDP_URL``. The recruiter launches Chrome separately with
``./launch-chrome.sh --force`` and opens linkedin.com/talent. A healthy
launch needs:

1. CDP endpoint reachable.
2. At least one attachable browser page target.
3. (Best-effort) a LinkedIn page loaded in the Cloris Chrome profile — if
   not, the worker will try to navigate but may stall on auth.

Each blocker carries an editorial remediation string the UI renders as
italic prose. Failures NEVER read as red error chips — Cloris-voice
("Cloris can't reach LinkedIn — your browser session may have ended.").
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from shared import config
from shared.health_types import ReadinessBlocker, ReadinessReport


def _probe_cdp_endpoint(cdp_url: str, timeout: float = 2.0) -> bool:
    """Try a synchronous HTTP GET against the CDP base URL.

    Chrome's DevTools Protocol exposes ``/json/version`` as a quick liveness
    endpoint. If it responds 200, Chrome is up and CDP is listening.
    Returns ``False`` on any network error or non-200 response.
    """

    probe_url = cdp_url.rstrip("/") + "/json/version"
    try:
        with urllib.request.urlopen(probe_url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _probe_linkedin_targets(cdp_url: str, timeout: float = 2.0) -> tuple[bool, bool]:
    """Inspect CDP targets and report ``(has_page_target, has_linkedin_url)``.

    This is intentionally HTTP-only. Readiness only needs a target inventory;
    attaching Playwright here is slower and can trip rebrowser frame/session
    errors before the actual worker has launched.

    - ``has_page_target`` is True iff CDP exposes at least one browser page or
      webview target Cloris can plausibly bind to.
    - ``has_linkedin_url`` is True iff any target URL is on linkedin.com.
    """

    list_url = cdp_url.rstrip("/") + "/json/list"
    try:
        with urllib.request.urlopen(list_url, timeout=timeout) as resp:
            if resp.status != 200:
                return (False, False)
            targets = json.load(resp)
    except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, OSError):
        return (False, False)

    if not isinstance(targets, list):
        return (False, False)

    has_page_target = False
    has_linkedin = False
    for target in targets:
        if not isinstance(target, dict):
            continue
        target_type = str(target.get("type") or "").strip().lower()
        url = str(target.get("url") or "").strip()
        if target_type in {"page", "webview"}:
            has_page_target = True
        try:
            host = urllib.parse.urlparse(url).netloc.lower()
        except ValueError:
            host = ""
        if host == "linkedin.com" or host.endswith(".linkedin.com"):
            has_linkedin = True

    return (has_page_target, has_linkedin)


def probe_linkedin_readiness(*, cdp_url: str | None = None) -> ReadinessReport:
    """Launch-readiness probe for LinkedIn.

    Args:
        cdp_url: Override the CDP endpoint. Defaults to
            ``shared.config.CDP_URL`` (env-overridable; default
            ``http://127.0.0.1:9222``).

    Returns:
        :class:`ReadinessReport` with ``ready`` true iff every check passed.
    """

    cdp = cdp_url or config.CDP_URL
    blockers: list[ReadinessBlocker] = []

    # Step 1: synchronous CDP liveness check (no playwright dependency).
    if not _probe_cdp_endpoint(cdp):
        blockers.append(
            ReadinessBlocker(
                kind="net",
                message="Cloris can't reach its Chrome window.",
                remediation=(
                    "Open Cloris's Chrome window, open linkedin.com/talent, "
                    "wait a few seconds, then retry."
                ),
                code="no_browser_session",
            )
        )
        # No point checking contexts if CDP itself is down.
        return ReadinessReport(ready=False, blockers=tuple(blockers))

    # Step 2: inspect the CDP target list without attaching Playwright.
    has_context, has_linkedin = _probe_linkedin_targets(cdp)

    if not has_context:
        blockers.append(
            ReadinessBlocker(
                kind="auth",
                message="Chrome is open, but Cloris couldn't find a usable tab.",
                remediation=(
                    "Open Cloris's Chrome window and navigate to linkedin.com/talent. "
                    "Cloris can only attach to its dedicated Chrome profile."
                ),
                code="no_browser_session",
            )
        )
        return ReadinessReport(ready=False, blockers=tuple(blockers))

    if not has_linkedin:
        # Soft warning, not a hard block — the worker can still navigate.
        # We surface it as an "auth" blocker so the recruiter sees it,
        # but Phase D D9 may decide to downgrade this to a non-blocking
        # warning if real-world false-blocks are common.
        blockers.append(
            ReadinessBlocker(
                kind="auth",
                message="Chrome is open, but no LinkedIn page is loaded.",
                remediation=(
                    "Open linkedin.com/talent in Cloris's Chrome window so Cloris "
                    "can confirm the attachable browser is signed in."
                ),
                code="no_linkedin_page",
            )
        )

    return ReadinessReport(ready=not blockers, blockers=tuple(blockers))
