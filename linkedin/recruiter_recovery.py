"""Recruiter recovery state machine — P6.

Detects browser/context health problems (Aw Snap, target crash, lost project,
stale search, CDP detach, login session loss) and orchestrates structured
recovery with context verification before resuming work.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from linkedin.browser import LinkedInBrowser

RecruiterHealthState = Literal[
    "healthy",
    "aw_snap",
    "target_crashed",
    "slow_or_unresponsive",
    "logged_out",
    "wrong_surface",
    "lost_project_context",
    "stale_search_context",
    "cdp_attach_failed",
    "blocked_or_rate_limited",
]

RECRUITER_HEALTH_STATES: frozenset[str] = frozenset({
    "healthy",
    "aw_snap",
    "target_crashed",
    "slow_or_unresponsive",
    "logged_out",
    "wrong_surface",
    "lost_project_context",
    "stale_search_context",
    "cdp_attach_failed",
    "blocked_or_rate_limited",
})

_CHROME_ERROR_PATTERNS = ("chrome-error://", "chrome://crash", "chrome://kill")
_AW_SNAP_PATTERNS = ("aw, snap", "aw snap", "page crashed", "target crashed")
_LOGIN_PATTERNS = ("/login", "/uas/login")
_CHALLENGE_PATTERNS = ("/checkpoint/", "/challenge/")
# Matched against the page TITLE only. An interstitial owns the whole page, so
# its title is high-signal and carries none of the candidate copy the body does.
_RATE_LIMIT_TITLE_PATTERNS = ("rate limit", "too many requests")

# Matched against body text too. Specific enough that ordinary profile copy
# does not contain them.
_RATE_LIMIT_BODY_PATTERNS = ("too many requests",)

# A "429" carrying a status qualifier is not profile copy in any realistic
# rendering, so these are safe against the body. Bare "429" is not: it is a
# follower count far more often than a status ("1,429 followers"), and this
# classifier now runs on every page, where a false positive costs a six-hour
# backoff. The separator class is deliberately wide because servers print
# "Error code: 429", "HTTP/1.1 429", "status=429" and "429. That's an error."
# far more often than the tidy "HTTP ERROR 429" shape.
_RATE_LIMIT_STATUS_REGEXES = (
    re.compile(
        r"\b(?:http(?:/\d(?:\.\d)?)?|error|status)\b[\s:=/]{0,3}"
        r"(?:code\b[\s:=]{0,3})?429\b"
    ),
    re.compile(r"\bcode\b[\s:=]{0,3}429\b"),
    re.compile(r"\b429\b\s*[.:,—-]?\s*(?:too many requests|that'?s an error)"),
)

# "rate limit" is ordinary engineering vocabulary — "built the API rate limiting
# layer", "I work on rate limiters" — and the Recruiter body carries candidate
# headlines and About text, so the bare substring reads a backend engineer's
# profile as a block on us. In the body it counts only when the phrasing is a
# notice addressed to the reader.
_RATE_LIMIT_NOTICE_REGEXES = (
    re.compile(r"\byou\b[^.]{0,60}\b(?:rate[- ]?limit|too many requests)"),
    re.compile(
        r"\b(?:exceeded|reached|hit)\b[^.]{0,40}\b(?:rate[- ]?limit|request limit)"
    ),
    re.compile(
        r"\brate[- ]?limit(?:ed|ing)?\b[^.]{0,60}"
        r"\b(?:try again|temporarily|please wait)"
    ),
)

# Phrases promoted from our own captures after human review. EMPTY on purpose:
# the AIEL campaign has produced zero capture directories to date, so no real
# LinkedIn interstitial copy has ever been observed. Public sources paraphrase
# this copy rather than quote it, and guessing at it would install exactly the
# kind of false positive the "429" fix above removes. Promote a string here
# only after reading it in a real capture.
_CAPTURED_RATE_LIMIT_PATTERNS: tuple[str, ...] = ()

_CAPTURE_BODY_READ_CAP = 20_000
_CAPTURE_BODY_LINE_MAX_LEN = 200
_CAPTURE_BODY_LINE_MAX_COUNT = 50


def load_capture_vocabulary(state_dir: Path) -> tuple[str, ...]:
    """Rate-limit phrases observed in our own page captures.

    The curated list below is the ONLY thing the classifier consults; this
    reader exists so a human can diff proposed additions against what the
    captures actually contain before promoting any of them. Nothing here
    feeds the classifier automatically — a string LinkedIn shows once must
    not silently become a six-hour backoff trigger.
    """
    captures_dir = state_dir / "captures"
    if not captures_dir.is_dir():
        return ()

    seen: set[str] = set()
    ordered: list[str] = []

    def _add(phrase: str) -> None:
        lowered = phrase.strip().lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            ordered.append(lowered)

    for capture_subdir in sorted(captures_dir.iterdir()):
        if not capture_subdir.is_dir():
            continue

        meta_path = capture_subdir / "meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = None
            if isinstance(meta, dict):
                title = meta.get("title")
                if isinstance(title, str):
                    _add(title)

        body_path = capture_subdir / "body.txt"
        if body_path.is_file():
            try:
                raw = body_path.read_text(encoding="utf-8", errors="replace")[
                    :_CAPTURE_BODY_READ_CAP
                ]
            except Exception:
                raw = ""
            for line in raw.splitlines()[:_CAPTURE_BODY_LINE_MAX_COUNT]:
                stripped = line.strip()
                if stripped and len(stripped) <= _CAPTURE_BODY_LINE_MAX_LEN:
                    _add(stripped)

    return tuple(ordered)


def _matches_rate_limit_vocabulary(title: str, body: str) -> bool:
    """True when the page carries a rate-limit / temporary-block signal.

    Title and body are held to different standards on purpose — see the
    pattern definitions above. Anything promoted from a real capture is
    treated as title-strength evidence, because that is where an interstitial
    announces itself.
    """
    title_text = (title or "").lower()
    body_text = (body or "").lower()

    if any(p in title_text for p in _RATE_LIMIT_TITLE_PATTERNS):
        return True
    if any(p in title_text for p in _CAPTURED_RATE_LIMIT_PATTERNS):
        return True

    if any(p in body_text for p in _RATE_LIMIT_BODY_PATTERNS):
        return True

    combined = f"{title_text} {body_text}"
    if any(r.search(combined) for r in _RATE_LIMIT_STATUS_REGEXES):
        return True
    return any(r.search(body_text) for r in _RATE_LIMIT_NOTICE_REGEXES)


def _snapshot_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _snapshot_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            _snapshot_str(key): _json_safe(item)
            for key, item in value.items()
            if _snapshot_str(key)
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return _snapshot_str(value)


@dataclass(frozen=True)
class RecruiterRecoverySnapshot:
    """Durable context captured before recovery so it can be replayed after."""

    run_id: int | None = None
    work_unit_id: str | None = None
    lane_id: str = ""
    search_url: str = ""
    project_id: str = ""
    current_page: int = 0
    advanced_search_controls: dict[str, Any] = field(default_factory=dict)
    keyword_boolean: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "work_unit_id": _snapshot_str(self.work_unit_id),
            "lane_id": _snapshot_str(self.lane_id),
            "search_url": _snapshot_str(self.search_url),
            "project_id": _snapshot_str(self.project_id),
            "current_page": _snapshot_int(self.current_page),
            "advanced_search_controls": _json_safe(self.advanced_search_controls),
            "keyword_boolean": _snapshot_str(self.keyword_boolean),
        }


@dataclass
class RecoveryResult:
    """Structured outcome of a recovery attempt."""

    success: bool
    health_before: str = "healthy"
    health_after: str = "healthy"
    attempts: int = 0
    context_verified: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "health_before": self.health_before,
            "health_after": self.health_after,
            "attempts": self.attempts,
            "context_verified": self.context_verified,
            "reason": self.reason,
        }


async def _peek_page_diagnostic_text(browser: LinkedInBrowser) -> str:
    """Best-effort page title + top-of-body text for content-based health
    classification.

    Resilient by design (mirrors LinkedInBrowser._peek_results_count_text):
    any failure — crashed target, detached page, slow render, a mocked page
    in tests — is swallowed and returns "" rather than raising, so callers
    can always fall back to URL-based classification.
    """
    title, body = await _peek_page_diagnostic_parts(browser)
    return " ".join(p for p in (title, body) if p)


async def _peek_page_diagnostic_parts(browser: LinkedInBrowser) -> tuple[str, str]:
    """Lowercased ``(title, body_prefix)``, kept apart for classification.

    They are not interchangeable. A rate-limit interstitial owns the whole
    page, so its title is a high-signal, low-noise place to match on. The body
    is not: on a Recruiter surface it carries candidate headlines and About
    text, where phrases like "rate limiting" are ordinary engineering
    vocabulary rather than a notice about us. Joining the two and matching
    them the same way is how a backend engineer's profile becomes a six-hour
    backoff.

    Same fail-soft contract as the caller above: every failure returns empty
    strings rather than raising, so URL-based classification still runs.
    """
    title = ""
    body_text = ""
    try:
        title = (await browser.page.title()) or ""
    except Exception:
        pass
    try:
        body_text = (
            await browser.page.locator("body").inner_text(timeout=500)
        ) or ""
    except Exception:
        pass
    return title.lower(), body_text[:2000].lower()


async def detect_recruiter_health(browser: LinkedInBrowser) -> RecruiterHealthState:
    """Classify the current browser/page state into a health category.

    Read-only inspection of page URL and rendered content — no mutations.

    P8.2: rate-limit / temporary-block interstitials are classified from
    page CONTENT (title/body text) and checked BEFORE the "linkedin.com/talent"
    wrong-surface check. LinkedIn does not reliably encode a rate-limit signal
    in the URL — the page can keep a /talent URL (or redirect off of one)
    while the actual notice only appears in the rendered title/body. Matching
    the URL alone (the prior behavior) let wrong_surface short-circuit before
    the rate-limit pattern was ever checked, so a blocked session was
    misclassified and retried/refreshed instead of backed off.
    """
    try:
        url = browser.page.url
    except Exception as exc:
        text = str(exc).lower()
        if any(p in text for p in ("target crashed", "target closed", "page crashed", "session closed")):
            return "target_crashed"
        return "cdp_attach_failed"

    lowered = (url or "").lower()

    if any(p in lowered for p in _CHROME_ERROR_PATTERNS):
        return "aw_snap"

    if any(p in lowered for p in _AW_SNAP_PATTERNS):
        return "aw_snap"

    peek_title, peek_body = await _peek_page_diagnostic_parts(browser)
    if (peek_title or peek_body) and _matches_rate_limit_vocabulary(
        peek_title, peek_body
    ):
        return "blocked_or_rate_limited"

    if not lowered or lowered in ("about:blank", "about:newtab"):
        return "wrong_surface"

    if any(p in lowered for p in _CHALLENGE_PATTERNS):
        return "blocked_or_rate_limited"

    if any(p in lowered for p in _LOGIN_PATTERNS):
        return "logged_out"

    if "linkedin.com/talent" not in lowered:
        if "linkedin.com" in lowered and getattr(browser, "_state_dir", None):
            try:
                from linkedin.page_capture import capture_page_state

                await capture_page_state(
                    browser, browser._state_dir, reason="health_unclassified"
                )
            except Exception:
                pass
        return "wrong_surface"

    project_match = re.search(r"/talent/hire/(\d+)", url)
    expected_project = _snapshot_str(getattr(browser, "_project_id", None))

    if expected_project and project_match:
        if project_match.group(1) != expected_project:
            return "lost_project_context"
    elif expected_project and not project_match:
        if "/talent/hire/" not in lowered and "/recruitersearch" not in lowered:
            return "lost_project_context"

    return "healthy"


def capture_recovery_snapshot(
    browser: LinkedInBrowser,
    *,
    run_id: int | None = None,
    work_unit_id: str | None = None,
    lane_id: str = "",
    keyword_boolean: str = "",
    current_page: int = 0,
    search_url: str = "",
    advanced_search_controls: dict[str, Any] | None = None,
) -> RecruiterRecoverySnapshot:
    """Snapshot current browser context for replay after recovery."""
    resolved_url = _snapshot_str(search_url)
    if not resolved_url:
        try:
            resolved_url = _snapshot_str(browser.page.url)
        except Exception:
            resolved_url = ""
    if not resolved_url:
        try:
            resolved_url = _snapshot_str(browser.get_current_search_url())
        except Exception:
            resolved_url = ""
    project_id = _snapshot_str(getattr(browser, "_project_id", ""))
    return RecruiterRecoverySnapshot(
        run_id=run_id,
        work_unit_id=_snapshot_str(work_unit_id),
        lane_id=_snapshot_str(lane_id),
        search_url=resolved_url,
        project_id=project_id,
        current_page=_snapshot_int(current_page),
        advanced_search_controls=_json_safe(advanced_search_controls or {}),
        keyword_boolean=_snapshot_str(keyword_boolean),
    )


def verify_recruiter_context(
    browser: LinkedInBrowser,
    snapshot: RecruiterRecoverySnapshot,
) -> bool:
    """Confirm the browser page matches the snapshot's project/search surface."""
    try:
        url = browser.page.url
    except Exception:
        return False

    lowered = (url or "").lower()
    if "linkedin.com/talent" not in lowered:
        return False

    if snapshot.project_id:
        m = re.search(r"/talent/hire/(\d+)", url)
        if not m or m.group(1) != snapshot.project_id:
            return False

    if snapshot.search_url:
        snapshot_lower = snapshot.search_url.lower()
        if "/recruitersearch" in snapshot_lower or "/discover/" in snapshot_lower:
            if "/recruitersearch" not in lowered and "/discover/" not in lowered:
                return False

    return True


async def replay_search_context(
    browser: LinkedInBrowser,
    snapshot: RecruiterRecoverySnapshot,
) -> tuple[bool, str]:
    """Reapply keyword Boolean and supported advanced controls from a snapshot."""
    from linkedin.advanced_search import (
        compile_recovery_plan_from_snapshot,
        apply_advanced_search_plan,
    )

    if not snapshot.keyword_boolean and not snapshot.advanced_search_controls:
        return True, "no_search_context_to_replay"

    plan = compile_recovery_plan_from_snapshot(snapshot)
    if not plan.controls and snapshot.keyword_boolean:
        try:
            await browser.enter_search_string(snapshot.keyword_boolean)
            return True, "keyword_boolean_replayed"
        except Exception as exc:
            return False, f"keyword_replay_failed:{exc}"

    result = await apply_advanced_search_plan(browser, plan)
    if result.success:
        return True, result.reason or "search_context_replayed"

    if snapshot.keyword_boolean:
        try:
            await browser.enter_search_string(snapshot.keyword_boolean)
            if result.unsupported_controls and not result.failed_controls:
                return True, "boolean_fallback_after_unsupported_controls"
            if result.fallback_to_boolean:
                return True, "boolean_fallback_after_stable_control_failure"
        except Exception as exc:
            return False, f"boolean_fallback_failed:{exc}"

    if result.failed_controls:
        return False, result.reason or "stable_control_replay_failed"
    if result.unsupported_controls and snapshot.keyword_boolean:
        return True, "boolean_only_after_unsupported_controls"
    return False, result.reason or "search_context_replay_failed"


async def recover_recruiter_context(
    browser: LinkedInBrowser,
    snapshot: RecruiterRecoverySnapshot,
    *,
    max_attempts: int = 3,
    event_recorder: Any | None = None,
) -> RecoveryResult:
    """Orchestrate full recovery: detect, rebind, verify, reapply.

    ``event_recorder``, when provided, must be a callable accepting
    ``(event_type: str, payload: dict)`` for structured event persistence.
    """
    health_before = await detect_recruiter_health(browser)

    if health_before == "blocked_or_rate_limited":
        # P8.2: never refresh-and-retry on rate-limit — back off instead.
        # Trip the browser's governor (if any) so the run stops now rather
        # than the normal cap-based cadence; no refresh_active_tab(),
        # disconnect()/connect(), or navigate_to_search() call is made.
        governor = getattr(browser, "_governor", None)
        if governor is not None:
            try:
                governor.force_backoff("blocked_or_rate_limited")
            except Exception:
                pass
        if event_recorder:
            event_recorder(
                "recruiter_recovery_backoff",
                {"health_before": health_before, "snapshot": snapshot.to_dict()},
            )
        return RecoveryResult(
            success=False,
            health_before=health_before,
            health_after=health_before,
            attempts=0,
            context_verified=False,
            reason="blocked_or_rate_limited_backoff",
        )

    if health_before == "healthy":
        if verify_recruiter_context(browser, snapshot):
            if snapshot.keyword_boolean or snapshot.advanced_search_controls:
                replay_ok, replay_reason = await replay_search_context(browser, snapshot)
                if replay_ok:
                    return RecoveryResult(
                        success=True,
                        health_before=health_before,
                        health_after="healthy",
                        attempts=0,
                        context_verified=True,
                        reason=replay_reason or "already_healthy",
                    )
                return RecoveryResult(
                    success=False,
                    health_before=health_before,
                    health_after="healthy",
                    attempts=0,
                    context_verified=True,
                    reason=replay_reason or "replay_failed_on_healthy_browser",
                )
            else:
                return RecoveryResult(
                    success=True,
                    health_before=health_before,
                    health_after="healthy",
                    attempts=0,
                    context_verified=True,
                    reason="already_healthy",
                )

    if event_recorder:
        event_recorder(
            "recruiter_recovery_attempted",
            {"health_before": health_before, "snapshot": snapshot.to_dict()},
        )

    for attempt in range(1, max_attempts + 1):
        try:
            refreshed = await browser.refresh_active_tab()
        except Exception:
            refreshed = False

        if not refreshed:
            try:
                await browser.disconnect()
            except Exception:
                pass
            try:
                await browser.connect()
            except Exception:
                await asyncio.sleep(2)
                continue

        try:
            rebound = await browser._bind_existing_recruiter_page()
        except Exception:
            rebound = False

        if not rebound:
            await asyncio.sleep(2)
            continue

        if snapshot.search_url and "linkedin.com/talent" in snapshot.search_url:
            health_now = await detect_recruiter_health(browser)
            if health_now != "healthy":
                try:
                    await browser.navigate_to_search(snapshot.search_url)
                except Exception:
                    await asyncio.sleep(1)
                    continue

        if verify_recruiter_context(browser, snapshot):
            replay_ok, replay_reason = await replay_search_context(browser, snapshot)
            if not replay_ok:
                await asyncio.sleep(1)
                continue
            health_after = await detect_recruiter_health(browser)
            result = RecoveryResult(
                success=True,
                health_before=health_before,
                health_after=health_after,
                attempts=attempt,
                context_verified=True,
                reason=replay_reason or "recovered",
            )
            if event_recorder:
                event_recorder("recruiter_recovery_succeeded", result.to_dict())
            return result

        await asyncio.sleep(1)

    health_after = await detect_recruiter_health(browser)
    result = RecoveryResult(
        success=False,
        health_before=health_before,
        health_after=health_after,
        attempts=max_attempts,
        context_verified=False,
        reason="context_verification_failed",
    )
    if event_recorder:
        event_recorder("recruiter_recovery_failed", result.to_dict())
    return result
