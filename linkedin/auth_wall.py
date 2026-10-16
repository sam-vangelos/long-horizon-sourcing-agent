"""LinkedIn sign-in walls, named instead of crashed into.

When the Recruiter tab lands on a LinkedIn auth or session page, nothing a
sourcing run does can succeed until a human signs in or picks a session.
Read as a missing sidebar, LinkedIn's one-session-only page
(``/enterprise-authentication/sessions``, "We have detected multiple
sign-ins") spends a campaign's whole absorb budget on new sessions that land
on the same wall, and then a traceback ends the process with no reason.

This module gives that condition a type and a name:

- ``classify_auth_wall`` is a pure function from (URL, body text) to a
  variant, or None for any page that is not a wall;
- ``AuthWallError`` carries the variant, and its first message line is
  ``recruiter_reauth_required: <variant>: <plain sentence>``;
- the helpers below let the browser, recovery and geography seams detect a
  wall, capture evidence, and raise, while touching nothing on the page.

Cloris never clicks anything on these pages — no "Confirm", no "Continue on
this session", no sign-in. Clearing a wall is the operator's move.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Iterable
from urllib.parse import urlparse

# The stable session-end reason every auth-wall variant maps to.
RECRUITER_REAUTH_REQUIRED = "recruiter_reauth_required"
# A checkpoint is LinkedIn questioning the seat, which the health classifier
# already treats as a detected block; naming it at any seam trips the same
# persisted backoff, so a relaunch cannot re-probe the seat at once.
_CHECKPOINT_VARIANT = "checkpoint"
_DETECTION_BACKOFF_REASON = "blocked_or_rate_limited"

AUTH_WALL_SENTENCES: dict[str, str] = {
    "session_conflict": (
        "LinkedIn Recruiter says another sign-in to this seat is active and "
        "only one session is allowed; Cloris stopped without choosing a "
        "session — open Recruiter, choose the session, then relaunch with "
        "resume."
    ),
    "enterprise_reauth": (
        "LinkedIn Recruiter is asking this seat to sign in again; Cloris "
        "stopped without signing in — sign in to Recruiter, then relaunch "
        "with resume."
    ),
    "login_redirect": (
        "LinkedIn sent the Recruiter tab to its sign-in page; Cloris stopped "
        "without signing in — sign in to Recruiter, then relaunch with resume."
    ),
    "checkpoint": (
        "LinkedIn put a security checkpoint in front of this seat; Cloris "
        "stopped without answering it and set the detection backoff — clear "
        "the checkpoint in Recruiter, then relaunch with resume once the "
        "backoff ends."
    ),
}

_ENTERPRISE_AUTH_PATH = "/enterprise-authentication/"
_SESSION_CONFLICT_PATH = "/enterprise-authentication/sessions"
_LOGIN_PATHS = ("/login", "/uas/login")
_CHECKPOINT_PATHS = ("/checkpoint/", "/challenge/")
# Read off a real wall capture (tests/fixtures/auth_wall/). Matched
# only when the URL is not a Recruiter surface: on /talent/ pages the body is
# candidate copy, and no phrase there may end a day cycle.
_SESSION_CONFLICT_BODY_MARKERS = (
    "we have detected multiple sign-ins",
    "only one session is allowed at a time",
    "other session detected",
)
# How much of the page body a wall check reads. The wall pages are short
# (the captured one is ~600 characters); the cap keeps a misfire cheap.
_WALL_BODY_READ_CHARS = 4000
_URL_IN_TEXT = re.compile(r"https?://[^\s'\"<>]+")
_NAMED_IN_TEXT = re.compile(
    rf"\b{RECRUITER_REAUTH_REQUIRED}\b(?::\s*([a-z_]+))?"
)


def _url_path(url: str) -> str:
    """The lowercased path of ``url``; '' when it has none or cannot parse."""
    try:
        return (urlparse(str(url or "").strip()).path or "").lower()
    except Exception:
        return ""


def classify_auth_wall(url: str, body_text: str | None) -> str | None:
    """Name the LinkedIn sign-in wall the page shows, or None if it shows none.

    Variants, in precedence order:

    - ``session_conflict``: the one-session-only page
      (``/enterprise-authentication/sessions``), or its body markers ("We have
      detected multiple sign-ins", "Only one session is allowed at a time");
    - ``enterprise_reauth``: any other ``/enterprise-authentication/`` page;
    - ``checkpoint``: ``/checkpoint/`` or ``/challenge/``;
    - ``login_redirect``: ``/login`` or ``/uas/login``.

    Only the URL's path is read, never its query, so a search URL that
    carries a redirect target can never classify as a wall. A Recruiter
    surface (``/talent/...``) is never a wall, whatever its body says.
    """
    path = _url_path(url)
    if path.startswith("/talent/") or path == "/talent":
        return None
    if path.startswith(_SESSION_CONFLICT_PATH):
        return "session_conflict"
    page_body = (body_text or "").lower()
    if page_body and any(
        marker in page_body for marker in _SESSION_CONFLICT_BODY_MARKERS
    ):
        return "session_conflict"
    if _ENTERPRISE_AUTH_PATH in path:
        return "enterprise_reauth"
    if any(fragment in path for fragment in _CHECKPOINT_PATHS):
        return "checkpoint"
    if any(fragment in path for fragment in _LOGIN_PATHS):
        return "login_redirect"
    return None


def redact_wall_url(url: str) -> str:
    """Scheme, host and path only.

    The wall URLs carry the seat's ``accountId`` and ``appInstanceId`` in the
    query; those never belong in an exception message, a session record or a
    console line.
    """
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return ""
    if not parsed.netloc:
        return parsed.path or ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


class AuthWallError(RuntimeError):
    """The Recruiter tab is on a LinkedIn sign-in wall; the run must stop.

    ``variant`` is one of the ``classify_auth_wall`` names. The first message
    line is ``recruiter_reauth_required: <variant>: <plain sentence>`` so a
    traceback, a session record and a grep all read the same name.
    """

    def __init__(self, variant: str, *, url: str = ""):
        self.variant = variant if variant in AUTH_WALL_SENTENCES else "enterprise_reauth"
        self.sentence = AUTH_WALL_SENTENCES[self.variant]
        self.url = redact_wall_url(url)
        message = f"{RECRUITER_REAUTH_REQUIRED}: {self.variant}: {self.sentence}"
        if self.url:
            message += f"\n  Page: {self.url}"
        super().__init__(message)


def _variant_from_text(text: str) -> tuple[str | None, bool]:
    """``(variant, specific)`` named in an exception's text; ``(None, False)``
    when the text names no wall.

    Two shapes count, both written by our own code rather than read off a
    page: our own name for the condition (an ``AuthWallError`` message that
    lost its type to a wrapper leads with ``recruiter_reauth_required:
    <variant>:``, and a recovery refusal quotes the health state
    ``recruiter_reauth_required``), and a URL on LinkedIn's
    ``/enterprise-authentication/`` path quoted in a navigation failure
    ("Page URL: https://www.linkedin.com/enterprise-authentication/...").

    ``specific`` is False when the text proves a wall but not which one (the
    bare health-state name, or an unparseable enterprise-auth URL); the
    variant is then the ``enterprise_reauth`` fallback.
    """
    named = _NAMED_IN_TEXT.search(text)
    if named is not None:
        variant = named.group(1) or ""
        if variant in AUTH_WALL_SENTENCES:
            return variant, True
        return "enterprise_reauth", False
    if _ENTERPRISE_AUTH_PATH not in text:
        return None, False
    for match in _URL_IN_TEXT.finditer(text):
        variant = classify_auth_wall(match.group(0), None)
        if variant is not None:
            return variant, True
    return "enterprise_reauth", False


def auth_wall_variant_of(exc: BaseException | None) -> str | None:
    """The wall variant behind ``exc``, or None when it is not an auth wall.

    Walks the ``__cause__`` / ``__context__`` chain, because a wall can be
    re-raised inside another handler (a recovery path that catches the
    navigation failure and raises its own error keeps the wall as context).
    A link that proves a wall without naming which one does not end the
    walk: a specific variant further down the chain wins, and the
    ``enterprise_reauth`` fallback is used only when none is found.
    """
    seen: set[int] = set()
    fallback: str | None = None
    current = exc
    while current is not None and id(current) not in seen and len(seen) < 16:
        seen.add(id(current))
        if isinstance(current, AuthWallError):
            return current.variant
        try:
            variant, specific = _variant_from_text(str(current))
        except Exception:
            variant, specific = None, False
        if variant is not None:
            if specific:
                return variant
            fallback = fallback or variant
        current = current.__cause__ or current.__context__
    return fallback


def auth_wall_operator_line(exc: BaseException | None) -> str:
    """The one console line the day cycle prints when a wall stops it."""
    variant = auth_wall_variant_of(exc) or "enterprise_reauth"
    return f"{AUTH_WALL_SENTENCES[variant]} ({RECRUITER_REAUTH_REQUIRED}: {variant})"


async def read_wall_body(page: Any) -> str:
    """A bounded read of the page body. Never raises; '' on any failure."""
    try:
        body = page.locator("body")
        if inspect.isawaitable(body):
            # Playwright's locator() is synchronous; an awaitable here is a
            # stand-in page that cannot answer. Close it rather than leak it.
            getattr(body, "close", lambda: None)()
            return ""
        text = await body.inner_text(timeout=500)
    except Exception:
        return ""
    return str(text or "")[:_WALL_BODY_READ_CHARS]


async def detect_auth_wall(
    browser: Any,
    *,
    url: str | None = None,
    read_body: bool = True,
) -> tuple[str | None, str]:
    """``(variant, url)`` for the browser's current page. Never raises.

    Reads the URL (unless given) and, when ``read_body``, a bounded prefix
    of the body. Reading is all it does.
    """
    if url is None:
        try:
            url = str(browser.page.url or "")
        except Exception:
            url = ""
    body = None
    if read_body:
        try:
            body = await read_wall_body(browser.page)
        except Exception:
            body = None
    try:
        return classify_auth_wall(url, body), url
    except Exception:
        return None, url


async def capture_auth_wall(browser: Any, variant: str) -> None:
    """Capture page evidence as ``auth_wall_<variant>``. Never raises."""
    state_dir = getattr(browser, "_state_dir", None)
    if state_dir is None:
        return
    try:
        from linkedin.page_capture import capture_page_state

        await capture_page_state(
            browser, state_dir, reason=f"auth_wall_{variant}"
        )
    except Exception:
        pass


def trip_detection_backoff(browser: Any, variant: str) -> None:
    """For a checkpoint, trip the browser's governor exactly as recovery
    does for ``blocked_or_rate_limited``. Never raises; no-op otherwise."""
    if variant != _CHECKPOINT_VARIANT:
        return
    governor = getattr(browser, "_governor", None)
    if governor is None:
        return
    try:
        governor.force_backoff(_DETECTION_BACKOFF_REASON)
    except Exception:
        pass


async def live_auth_wall_error(
    browser: Any, cause: BaseException | None = None
) -> AuthWallError | None:
    """The typed wall for the browser's live page, or None. Never raises.

    The backstop for a failure that reached the session boundary without
    naming the wall (a fail-soft caller swallowed it, then a later gate
    refused generically). Reads, captures and trips the checkpoint backoff
    exactly as ``raise_if_auth_wall`` does; returns instead of raising.
    """
    if browser is None:
        return None
    try:
        await raise_if_auth_wall(browser, cause=cause)
    except AuthWallError as wall:
        return wall
    except Exception:
        return None
    return None


async def raise_if_auth_wall(
    browser: Any,
    *,
    url: str | None = None,
    read_body: bool = True,
    skip_variants: Iterable[str] = (),
    cause: BaseException | None = None,
) -> None:
    """Raise ``AuthWallError`` when the current page is a sign-in wall.

    Reads the page, captures evidence on a wall, and raises; touches nothing.
    A checkpoint also trips the governor's detection backoff first. Returns
    quietly for any other page, and for variants in ``skip_variants`` (a
    caller that leaves one variant to another path). Every read failure
    counts as "not a wall": this check may only ever ADD a name to a failure,
    never replace one with a fault of its own.
    """
    variant, url = await detect_auth_wall(browser, url=url, read_body=read_body)
    if variant is None or variant in set(skip_variants):
        return
    await capture_auth_wall(browser, variant)
    trip_detection_backoff(browser, variant)
    error = AuthWallError(variant, url=url)
    if cause is not None:
        raise error from cause
    raise error
