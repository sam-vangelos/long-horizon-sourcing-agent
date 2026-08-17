"""Cloris's Chrome launcher — safe CDP ensure_running.

Replaces ``launch-chrome.sh`` for the trial-day .app where the
recipient cannot reasonably be asked to open a Terminal. Two
critical differences from the shell script:

1. **Profile path is Cloris-namespaced** (``<user_data_dir>/chrome-
   profile``). The shell script used ``~/.chrome-cdp`` — a leak from
   dev-script naming. The dedicated profile keeps Cloris's LinkedIn
   session isolated from the recipient's everyday Chrome (their
   bookmarks, password autofill, work tabs).

2. **Never blasts ``Google Chrome`` globally.** The shell script
   ran ``pkill -9 -f "Google Chrome"`` to clear stuck instances —
   acceptable on a developer's machine, **not** acceptable on a
   customer machine where the recipient probably has their personal
   Chrome with unsaved work open. This module only kills Chrome
   processes whose ``--user-data-dir=`` argv exactly matches Cloris's
   profile path.

This module is also wired into ``cloris.app.run_app`` so the .app
auto-launches Chrome on boot. The ``launch-chrome.sh`` script
remains as a thin compatibility shim that delegates here, so dev
muscle memory keeps working.

Public surface:

- :func:`is_healthy` — non-destructive CDP liveness probe.
- :func:`ensure_running` — full ensure-CDP-Chrome-is-up flow with
  optional ``force=True`` to recycle an unhealthy instance.
- :func:`open_linkedin_recruiter` — non-destructively open LinkedIn
  Recruiter in the Cloris Chrome profile.
- :func:`status` — structured :class:`ChromeStatus` for the API
  surface.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from shared import config
from shared.user_data_dir import chrome_profile_dir

log = logging.getLogger("cloris.chrome_launcher")


CHROME_APP_NAME = "Google Chrome"
CHROME_DEFAULT_BIN_PATH = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)
LINKEDIN_RECRUITER_SEARCH_URL = "https://www.linkedin.com/talent/search"
DEFAULT_CDP_PORT = 9222
LOCAL_SCAFFOLD_TAB_MARKER = "/cloris/frontend/scaffolds/"
_HEALTH_PROBE_TIMEOUT_SECONDS = 1.5
_LAUNCH_WAIT_TIMEOUT_SECONDS = 25.0
_LAUNCH_WAIT_POLL_INTERVAL_SECONDS = 0.5
_KILL_GRACE_SECONDS = 1.0
_KILL_FORCE_SECONDS = 1.5


ChromeState = Literal[
    "healthy",
    "spawning",
    "unhealthy",
    "missing_chrome",
    "unsupported_platform",
]


@dataclass(frozen=True)
class ChromeStatus:
    """Structured status report for the API surface.

    ``state`` is the high-level summary:

    - ``healthy`` — CDP responding, ready for worker attach.
    - ``spawning`` — we just kicked Chrome off; not yet healthy.
    - ``unhealthy`` — CDP unreachable and we did not / could not
      relaunch.
    - ``missing_chrome`` — Google Chrome.app isn't installed at the
      expected path. The recipient needs to install Chrome before
      Cloris can do anything LinkedIn-shaped.
    - ``unsupported_platform`` — non-macOS host. Cloris's .app is
      macOS-only; this state exists so callers (e.g. tests on Linux
      CI, or a future Linux build) get a meaningful status rather
      than a crash.

    ``cdp_url`` is the URL workers should attach to. ``profile_dir``
    is the on-disk path of the dedicated Chrome profile so the UI can
    surface it as part of the relational disclosure. ``message`` is
    a recruiter-readable one-sentence summary.
    """

    state: ChromeState
    cdp_url: str
    profile_dir: str
    message: str


def cdp_url() -> str:
    """The CDP base URL Cloris's workers attach to.

    Threaded through ``shared.config.CDP_URL`` so tests / overrides
    keep working. Defaults to ``http://127.0.0.1:9222`` per the
    long-standing project convention.
    """

    return config.CDP_URL


def _cdp_port() -> int:
    """Parse the port out of :func:`cdp_url`, falling back to 9222."""

    url = cdp_url()
    if ":" in url:
        try:
            return int(url.rsplit(":", 1)[-1].split("/")[0])
        except ValueError:
            pass
    return DEFAULT_CDP_PORT


def is_healthy(*, timeout: float = _HEALTH_PROBE_TIMEOUT_SECONDS) -> bool:
    """Non-destructive CDP liveness probe.

    Hits ``<cdp_url>/json/version`` and returns True iff the response
    is HTTP 200. Mirrors :func:`linkedin.health._probe_cdp_endpoint`
    intentionally — keeping the probe definition local to the launcher
    means the launcher can be exercised in isolation (e.g. by the
    welcome surface) without depending on the LinkedIn readiness
    module.
    """

    probe_url = cdp_url().rstrip("/") + "/json/version"
    try:
        with urllib.request.urlopen(probe_url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def status() -> ChromeStatus:
    """Compose a :class:`ChromeStatus` snapshot for the wire.

    Used by ``GET /api/chrome-status`` and by the welcome-surface
    polling loop in the .app to drive the "Opening Chrome..." →
    "Sign into LinkedIn" transition. Pure read; never spawns or
    kills.
    """

    profile = str(chrome_profile_dir())

    if sys.platform != "darwin":
        return ChromeStatus(
            state="unsupported_platform",
            cdp_url=cdp_url(),
            profile_dir=profile,
            message=(
                "Cloris's Chrome auto-launcher only runs on macOS. "
                "Connect a CDP-enabled Chrome at "
                f"{cdp_url()} manually to proceed."
            ),
        )

    if not _chrome_installed():
        return ChromeStatus(
            state="missing_chrome",
            cdp_url=cdp_url(),
            profile_dir=profile,
            message=(
                "Cloris couldn't find Google Chrome on this Mac. "
                "Install Chrome from google.com/chrome and reopen Cloris."
            ),
        )

    if is_healthy():
        return ChromeStatus(
            state="healthy",
            cdp_url=cdp_url(),
            profile_dir=profile,
            message=(
                "Chrome is open and ready. Sign into LinkedIn Recruiter in "
                "the Cloris Chrome window if you haven't already."
            ),
        )

    return ChromeStatus(
        state="unhealthy",
        cdp_url=cdp_url(),
        profile_dir=profile,
        message=(
            "Cloris hasn't opened its Chrome window yet — click the "
            "re-open Chrome control to spawn it."
        ),
    )


def ensure_running(*, force: bool = False) -> ChromeStatus:
    """Ensure CDP-enabled Chrome is up on the dedicated Cloris profile.

    Default semantics (``force=False``):

    1. If CDP is already healthy, no-op and return ``healthy``.
    2. Otherwise, kill any *Cloris-profile* Chrome processes (only —
       the recipient's personal Chrome is untouched), then spawn a
       fresh Chrome on the dedicated profile, and wait for CDP to
       come up.

    ``force=True`` skips the "already healthy" short-circuit and
    always recycles. Use when the readiness probe has surfaced
    something the silent ``is_healthy`` check can't see (e.g. CDP
    is up but no contexts attached because the user closed the last
    tab).

    Returns the post-action :class:`ChromeStatus`. On
    ``unsupported_platform`` and ``missing_chrome`` the function
    short-circuits without trying to spawn anything.
    """

    profile = chrome_profile_dir()

    if sys.platform != "darwin":
        return status()

    if not _chrome_installed():
        return status()

    if not force and is_healthy():
        _close_local_scaffold_tabs()
        return ChromeStatus(
            state="healthy",
            cdp_url=cdp_url(),
            profile_dir=str(profile),
            message="Chrome is open and ready.",
        )

    _kill_cloris_chrome_only(profile)

    _spawn_chrome(profile)

    deadline = time.monotonic() + _LAUNCH_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if is_healthy():
            _close_local_scaffold_tabs()
            return ChromeStatus(
                state="healthy",
                cdp_url=cdp_url(),
                profile_dir=str(profile),
                message=(
                    "Chrome is open. Sign into LinkedIn Recruiter in the "
                    "Cloris Chrome window if you haven't already."
                ),
            )
        time.sleep(_LAUNCH_WAIT_POLL_INTERVAL_SECONDS)

    return ChromeStatus(
        state="spawning",
        cdp_url=cdp_url(),
        profile_dir=str(profile),
        message=(
            "Cloris asked Chrome to open but it's taking longer than "
            "expected. If a Chrome window is open, sign into LinkedIn and "
            "retry; otherwise click the re-open Chrome control."
        ),
    )


def open_linkedin_recruiter(
    *,
    target_url: str = LINKEDIN_RECRUITER_SEARCH_URL,
) -> ChromeStatus:
    """Open LinkedIn Recruiter in the dedicated Cloris Chrome profile.

    This is the non-destructive counterpart to :func:`ensure_running(force=True)`.
    It never recycles Chrome and never touches personal Chrome. If CDP is already
    up, it asks Chrome's local DevTools endpoint to create a tab directly. If
    CDP is not up, it first performs the normal safe ``ensure_running`` flow.
    """

    snapshot = ensure_running(force=False)
    if snapshot.state != "healthy":
        return snapshot

    opened = _open_url_via_cdp(target_url)
    if not opened and sys.platform == "darwin" and _chrome_installed():
        opened = _open_url_with_macos(chrome_profile_dir(), target_url)

    if opened:
        return ChromeStatus(
            state="healthy",
            cdp_url=snapshot.cdp_url,
            profile_dir=snapshot.profile_dir,
            message="LinkedIn Recruiter is opening in Cloris Chrome.",
        )

    return snapshot


def _chrome_installed() -> bool:
    """True iff Google Chrome is at the canonical macOS install path."""

    return Path(CHROME_DEFAULT_BIN_PATH).exists()


def _spawn_chrome(profile: Path) -> None:
    """Spawn Chrome with CDP enabled on the dedicated profile.

    Uses ``open -na`` rather than invoking the binary directly. This
    matches what the original ``launch-chrome.sh`` did (per
    ``launch-chrome.sh:67``: "Launch via macOS open instead of
    invoking the app binary directly. This has proven more reliable
    after force-killing a prior CDP session.") The macOS launchd
    semantics around ``open -na`` reset some state that direct
    binary invocation does not, which is the difference between a
    clean cold-start and a Chrome that stalls during init.
    """

    port = _cdp_port()
    args = [
        "open",
        "-na",
        CHROME_APP_NAME,
        "--args",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile}",
        "--disable-session-crashed-bubble",
        "--no-first-run",
        "about:blank",
    ]
    log.info("cloris.chrome_launcher: spawning Chrome on CDP port %d", port)
    subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _open_url_via_cdp(url: str) -> bool:
    """Open a new tab through Chrome's local DevTools HTTP endpoint."""

    encoded = urllib.parse.quote(url, safe=":/?&=%")
    target = cdp_url().rstrip("/") + f"/json/new?{encoded}"
    req = urllib.request.Request(target, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=_HEALTH_PROBE_TIMEOUT_SECONDS) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _open_url_with_macos(profile: Path, url: str) -> bool:
    """Fallback opener using the same Cloris profile and CDP arguments."""

    args = [
        "open",
        "-na",
        CHROME_APP_NAME,
        "--args",
        f"--remote-debugging-port={_cdp_port()}",
        f"--user-data-dir={profile}",
        "--disable-session-crashed-bubble",
        "--no-first-run",
        url,
    ]
    try:
        result = subprocess.run(
            args,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("cloris.chrome_launcher: failed to open %s: %r", url, exc)
        return False
    return result.returncode == 0


def _close_local_scaffold_tabs() -> int:
    """Close raw local scaffold tabs in the dedicated Cloris Chrome profile.

    During design-system iteration it is easy to open the source scaffold HTMLs
    in the same CDP profile Cloris uses for LinkedIn. Chrome can restore those
    tabs on the next app launch, making the user-visible profile look like an
    old UI even though the packaged app server is current. Only local file tabs
    under the repo scaffold directory are closed; LinkedIn and ordinary web
    tabs are left alone.
    """

    list_url = cdp_url().rstrip("/") + "/json/list"
    try:
        with urllib.request.urlopen(
            list_url,
            timeout=_HEALTH_PROBE_TIMEOUT_SECONDS,
        ) as resp:
            targets = json.load(resp)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return 0

    if not isinstance(targets, list):
        return 0

    closed = 0
    for target in targets:
        if not isinstance(target, dict):
            continue
        tab_url = target.get("url")
        tab_id = target.get("id")
        if not isinstance(tab_url, str) or not isinstance(tab_id, str):
            continue
        if not (
            tab_url.startswith("file://")
            and LOCAL_SCAFFOLD_TAB_MARKER in tab_url
        ):
            continue

        close_url = (
            cdp_url().rstrip("/")
            + "/json/close/"
            + urllib.parse.quote(tab_id, safe="")
        )
        try:
            with urllib.request.urlopen(
                close_url,
                timeout=_HEALTH_PROBE_TIMEOUT_SECONDS,
            ):
                closed += 1
        except (urllib.error.URLError, TimeoutError, OSError):
            log.debug(
                "cloris.chrome_launcher: failed to close stale scaffold tab %s",
                tab_url,
            )

    if closed:
        log.info(
            "cloris.chrome_launcher: closed %d local scaffold tab(s)",
            closed,
        )
    return closed


def _kill_cloris_chrome_only(profile: Path) -> None:
    """Terminate ONLY Chrome processes whose argv references our profile.

    The recipient's personal Chrome (under
    ``~/Library/Application Support/Google/Chrome``) is untouched.

    Implementation:

    1. ``ps -A -o pid,command`` to list every running process with
       its full argv string.
    2. Filter to PIDs whose argv contains ``--user-data-dir=<profile>``
       (the exact profile path Cloris uses). This is conservative: a
       process accidentally named with a similar substring cannot
       satisfy the equality check on the user-data-dir argument.
    3. Send SIGTERM, wait :data:`_KILL_GRACE_SECONDS`, then SIGKILL
       any survivors.
    """

    pids = list(_pids_for_profile(profile))
    if not pids:
        return

    log.info(
        "cloris.chrome_launcher: terminating %d Cloris-profile Chrome "
        "process(es): %s",
        len(pids),
        pids,
    )
    _kill_pids(pids, signal_name="TERM")
    time.sleep(_KILL_GRACE_SECONDS)

    survivors = list(_pids_for_profile(profile))
    if survivors:
        log.info(
            "cloris.chrome_launcher: SIGKILL'ing %d unresponsive "
            "Cloris-profile Chrome process(es): %s",
            len(survivors),
            survivors,
        )
        _kill_pids(survivors, signal_name="KILL")
        time.sleep(_KILL_FORCE_SECONDS)


def _pids_for_profile(profile: Path) -> Iterable[int]:
    """Yield PIDs of Chrome processes whose argv references ``profile``.

    Equality on the ``--user-data-dir=<path>`` argv token, not a
    substring match — defensive against a personal Chrome whose
    profile path happens to share a prefix.
    """

    if shutil.which("ps") is None:
        return iter(())

    needle = f"--user-data-dir={profile}"
    try:
        result = subprocess.run(
            ["ps", "-A", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning(
            "cloris.chrome_launcher: failed to enumerate processes: %r", exc
        )
        return iter(())

    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # `ps -o pid=,command=` emits ``"<pid>  <cmdline>"`` with no
        # header. Split on first whitespace; the rest is the full
        # argv joined with spaces.
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_str, cmdline = parts
        if needle not in cmdline:
            continue
        try:
            pids.append(int(pid_str))
        except ValueError:
            continue
    return pids


def _kill_pids(pids: Iterable[int], *, signal_name: str) -> None:
    """SIGTERM/SIGKILL the given pids via ``/bin/kill``."""

    for pid in pids:
        try:
            subprocess.run(
                ["/bin/kill", f"-{signal_name}", str(pid)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            log.warning(
                "cloris.chrome_launcher: kill -%s %d failed: %r",
                signal_name,
                pid,
                exc,
            )


def _serialize_status(s: ChromeStatus) -> dict[str, str]:
    """Wire-shape for ``GET /api/chrome-status``.

    Kept as a module-level helper so the API layer can call
    :func:`status` without depending on a Pydantic round-trip.
    """

    return {
        "state": s.state,
        "cdp_url": s.cdp_url,
        "profile_dir": s.profile_dir,
        "message": s.message,
    }


def _main(argv: list[str] | None = None) -> int:
    """CLI entrypoint — replaces ``launch-chrome.sh`` for dev.

    Behavior:

    - No args: equivalent to ``launch-chrome.sh`` (ensure_running,
      no-op if healthy).
    - ``--force``: equivalent to ``launch-chrome.sh --force`` (always
      recycle).
    - ``--status``: emit the current :class:`ChromeStatus` as JSON
      and exit. Used by smoke scripts and the dev cheat sheet.
    """

    import argparse

    parser = argparse.ArgumentParser(
        prog="cloris.chrome_launcher",
        description="Ensure CDP-enabled Chrome is running on Cloris's "
        "dedicated profile.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recycle Chrome even if CDP is already healthy.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Emit the current Chrome status as JSON and exit.",
    )
    args = parser.parse_args(argv)

    if args.status:
        print(json.dumps(_serialize_status(status()), indent=2))
        return 0

    result = ensure_running(force=args.force)
    print(json.dumps(_serialize_status(result), indent=2))
    return 0 if result.state == "healthy" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
