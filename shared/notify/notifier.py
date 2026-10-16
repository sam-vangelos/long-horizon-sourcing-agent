"""Alarm delivery to the operator.

``notify_sam(kind, title, body, dedupe_key)`` sends one alarm and returns
``True`` when at least one transport delivered it. It never raises.

Transports, in order:

1. **macOS Notification Center**, always attempted on macOS. ``osascript``
   runs with an argv list (no shell). The title and body are passed as
   trailing script arguments after ``--`` and read inside an
   ``on run argv`` handler, so no caller text is ever spliced into
   AppleScript source: quotes and backslashes cannot break or inject.
2. **Slack DM**, attempted when the macOS Keychain holds the generic
   passwords ``cloris.notify.slack``/``bot_token`` and
   ``cloris.notify.slack``/``user_id``. The token is read with
   ``security find-generic-password -w`` (the argv names only the service
   and account, never the secret), held in memory for the two API calls
   (``conversations.open`` then ``chat.postMessage``), and never logged,
   written to disk, or passed as an argument. Absent items skip Slack with
   one log line.

Dedupe: a small JSON file (``notify-dedupe.json``) in the state dir records
every ``dedupe_key`` that reached at least one transport, so the same alarm
goes out once. A delivery that reached nothing is not recorded and the next
caller retries it. Dedupe is per transport: when one transport delivered but
a configured one failed (Slack down, rate-limited), the key is recorded with
that transport pending, and :func:`retry_pending` (called by the watchdog on
every scan) or the next ``notify_sam`` with the same key retries only the
pending transport, up to ``MAX_TRANSPORT_RETRIES`` times within a day. A
transport that is not configured (no Keychain items, not macOS) returns
``None`` from its sender and is never pending. The check-send-record sequence
runs under an ``flock`` so two callers cannot both send the same key.

If the state dir is unusable, dedupe moves to a per-user temp dir
(``$TMPDIR/cloris-watchdog-<uid>``) so a 2-minute watchdog cannot repeat one
alarm every scan. Only when both are unusable does dedupe fail open: the
alarm still goes out every scan. That needs both Application Support and the
temp dir to be unwritable (a full disk, say), which is itself worth hearing
about. Delivery never fails closed.

Slack delivery is at-least-once. If ``chat.postMessage`` succeeds and the
process dies before the dedupe record lands, the next scan sends the same
DM again. For alarms that is the right side to err on. Gmail-style
exactly-once machinery is deliberately not built here.

Kill switch: ``CLORIS_NOTIFY_DISABLED=1`` makes ``notify_sam`` a no-op.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger("cloris.notify")

DISABLE_ENV = "CLORIS_NOTIFY_DISABLED"
STATE_DIR_ENV = "CLORIS_WATCHDOG_HOME"
DEFAULT_STATE_DIR = Path.home() / "Library" / "Application Support" / "cloris-watchdog"
DEDUPE_FILENAME = "notify-dedupe.json"
LOCK_FILENAME = "notify.lock"
DEDUPE_RETENTION = timedelta(days=30)
LOCK_WAIT_SECONDS = 30.0
MAX_TRANSPORT_RETRIES = 10
PENDING_RETRY_WINDOW = timedelta(days=1)

OSASCRIPT = "/usr/bin/osascript"
SECURITY = "/usr/bin/security"
KEYCHAIN_SERVICE = "cloris.notify.slack"
KEYCHAIN_TOKEN_ACCOUNT = "bot_token"
KEYCHAIN_USER_ACCOUNT = "user_id"
SLACK_API = "https://slack.com/api/"
SUBPROCESS_TIMEOUT_SECONDS = 10.0
HTTP_TIMEOUT_SECONDS = 10.0

TITLE_LIMIT = 120
BODY_LIMIT = 600
SLACK_LIMIT = 3000

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def default_state_dir() -> Path:
    """Where the dedupe file lives: ``$CLORIS_WATCHDOG_HOME`` or the default."""

    override = os.environ.get(STATE_DIR_ENV, "").strip()
    return Path(override).expanduser() if override else DEFAULT_STATE_DIR


def _fallback_state_dir() -> Path:
    """Dedupe home when the configured state dir is unusable."""

    return Path(tempfile.gettempdir()) / f"cloris-watchdog-{os.getuid()}"


def _usable_state_dir(directory: Path) -> Path:
    """``directory`` if it can hold the dedupe files, else the temp fallback."""

    for candidate in (directory, _fallback_state_dir()):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            with open(candidate / LOCK_FILENAME, "a"):
                pass
        except OSError:
            continue
        if candidate != directory:
            logger.warning("notify: state dir unusable; dedupe falls back to the temp dir")
        return candidate
    return directory


# --- public entry -----------------------------------------------------------


def notify_sam(
    kind: str,
    title: str,
    body: str,
    dedupe_key: str,
    *,
    state_dir: Path | str | None = None,
) -> bool:
    """Send one alarm to the operator; ``True`` if any transport delivered it now.

    Returns ``False`` without sending when ``dedupe_key`` was already
    delivered everywhere it could be, when notifications are disabled, or
    when the arguments are unusable. A key that is recorded with a pending
    transport retries just that transport. Never raises.
    """

    try:
        if _disabled():
            return False
        if not all(isinstance(v, str) and v.strip() for v in (kind, title, dedupe_key)):
            logger.warning("notify: refused an alarm with no kind, title, or dedupe key")
            return False
        body = body if isinstance(body, str) else ""
        directory = _usable_state_dir(
            Path(state_dir) if state_dir is not None else default_state_dir()
        )
        with _locked(directory):
            sent = _load_sent(directory)
            if dedupe_key in sent:
                return _retry_record(directory, sent, dedupe_key)
            delivered, failed = _deliver(title, body)
            if delivered:
                record: dict[str, Any] = {
                    "at": _now_iso(),
                    "kind": kind,
                    "via": delivered,
                }
                if failed:
                    record.update(pending=failed, title=title, body=body, retries=0)
                sent[dedupe_key] = record
                _save_sent(directory, sent)
                logger.info("notify: %s alarm delivered via %s", kind, "+".join(delivered))
            else:
                logger.warning("notify: %s alarm reached no transport; will retry", kind)
            return bool(delivered)
    except Exception as exc:  # noqa: BLE001 - an alarm path must never raise
        logger.warning("notify: alarm failed (%s)", type(exc).__name__)
        return False


def retry_pending(*, state_dir: Path | str | None = None) -> int:
    """Retry every recorded alarm with a pending transport; return how many landed.

    The watchdog calls this on every scan, so a transient Slack failure is
    retried within minutes even when the condition behind the alarm is gone.
    Never raises.
    """

    try:
        if _disabled():
            return 0
        directory = _usable_state_dir(
            Path(state_dir) if state_dir is not None else default_state_dir()
        )
        with _locked(directory):
            sent = _load_sent(directory)
            keys = [key for key, record in sent.items() if _is_pending(record)]
            return sum(1 for key in keys if _retry_record(directory, sent, key))
    except Exception as exc:  # noqa: BLE001
        logger.warning("notify: pending retry failed (%s)", type(exc).__name__)
        return 0


def already_sent(dedupe_key: str, *, state_dir: Path | str | None = None) -> bool:
    """Whether ``dedupe_key`` has reached the operator on some transport. Never raises."""

    try:
        directory = Path(state_dir) if state_dir is not None else default_state_dir()
        if dedupe_key in _load_sent(directory):
            return True
        fallback = _fallback_state_dir()
        return fallback != directory and dedupe_key in _load_sent(fallback)
    except Exception:  # noqa: BLE001
        return False


def _disabled() -> bool:
    return os.environ.get(DISABLE_ENV, "").strip() in {"1", "true", "yes"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_pending(record: Any) -> bool:
    if not isinstance(record, dict) or not record.get("pending"):
        return False
    if int(record.get("retries") or 0) >= MAX_TRANSPORT_RETRIES:
        return False
    try:
        recorded = datetime.fromisoformat(record["at"])
    except (KeyError, TypeError, ValueError):
        return False
    return datetime.now(timezone.utc) - recorded < PENDING_RETRY_WINDOW


def _retry_record(directory: Path, sent: dict[str, Any], key: str) -> bool:
    """Retry one record's pending transports (caller holds the lock)."""

    record = sent.get(key)
    if not _is_pending(record):
        return False
    pending = [name for name in record["pending"] if name in _SENDERS]
    delivered, failed = _deliver(
        str(record.get("title") or ""), str(record.get("body") or ""), only=pending
    )
    record["retries"] = int(record.get("retries") or 0) + 1
    record["via"] = list(record.get("via") or []) + delivered
    if failed:
        record["pending"] = failed
    else:
        for field in ("pending", "title", "body", "retries"):
            record.pop(field, None)
    sent[key] = record
    _save_sent(directory, sent)
    if delivered:
        logger.info("notify: pending %s alarm delivered via %s", record.get("kind"), "+".join(delivered))
    return bool(delivered)


def _call_macos(title: str, body: str) -> bool | None:
    return _send_macos(title, body)


def _call_slack(title: str, body: str) -> bool | None:
    return _send_slack(_slack_text(title, body))


_SENDERS: dict[str, Callable[[str, str], "bool | None"]] = {
    "macos": _call_macos,
    "slack": _call_slack,
}


def _deliver(
    title: str, body: str, *, only: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Try each transport; return (delivered, failed-and-configured).

    A sender returning ``None`` is not configured here and lands in neither
    list. A sender that raises counts as failed.
    """

    delivered: list[str] = []
    failed: list[str] = []
    for name, sender in _SENDERS.items():
        if only is not None and name not in only:
            continue
        try:
            outcome = sender(title, body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("notify: %s transport failed (%s)", name, type(exc).__name__)
            outcome = False
        if outcome is True:
            delivered.append(name)
        elif outcome is not None:
            failed.append(name)
    return delivered, failed


# --- dedupe state -----------------------------------------------------------


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """Hold an exclusive flock on the state dir; proceed unlocked on failure."""

    handle = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(directory / LOCK_FILENAME, "a")
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    logger.warning("notify: dedupe lock busy; sending unlocked")
                    break
                time.sleep(0.2)
    except OSError as exc:
        logger.warning("notify: dedupe state unavailable (%s); sending anyway", type(exc).__name__)
    try:
        yield
    finally:
        if handle is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _load_sent(directory: Path) -> dict[str, Any]:
    try:
        raw = json.loads((directory / DEDUPE_FILENAME).read_text())
    except (OSError, ValueError):
        return {}
    sent = raw.get("sent") if isinstance(raw, dict) else None
    return dict(sent) if isinstance(sent, dict) else {}


def _save_sent(directory: Path, sent: dict[str, Any]) -> None:
    cutoff = datetime.now(timezone.utc) - DEDUPE_RETENTION
    kept: dict[str, Any] = {}
    for key, record in sent.items():
        try:
            if datetime.fromisoformat(record["at"]) < cutoff:
                continue
        except (KeyError, TypeError, ValueError):
            pass
        kept[key] = record
    target = directory / DEDUPE_FILENAME
    tmp = directory / f"{DEDUPE_FILENAME}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps({"sent": kept}, indent=2, sort_keys=True))
        os.replace(tmp, target)
    except OSError as exc:
        logger.warning("notify: could not record dedupe (%s); alarm may repeat", type(exc).__name__)
        with contextlib.suppress(OSError):
            tmp.unlink()


# --- text hygiene -----------------------------------------------------------


def _clean_text(text: object, limit: int) -> str:
    """Printable single-line text, bounded, never starting with ``-``."""

    raw = text if isinstance(text, str) else str(text or "")
    cleaned = "".join(
        " " if ch in "\r\n\t" else ch
        for ch in raw
        if ch in "\r\n\t" or (ord(ch) >= 32 and ord(ch) != 127)
    ).strip()
    if cleaned.startswith("-"):
        cleaned = "–" + cleaned[1:]
    return cleaned[:limit]


def _slack_text(title: str, body: str) -> str:
    def esc(value: str) -> str:
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    text = f"*{esc(_clean_text(title, TITLE_LIMIT))}*\n{esc(body or '')}"
    return text[:SLACK_LIMIT]


# --- macOS transport --------------------------------------------------------

_APPLESCRIPT = (
    "-e", "on run argv",
    "-e", 'display notification (item 2 of argv) with title (item 1 of argv) sound name "Basso"',
    "-e", "end run",
)


def _send_macos(title: str, body: str, *, runner: Runner = subprocess.run) -> bool | None:
    """Post a Notification Center banner via osascript.

    ``True`` on success, ``False`` on failure, ``None`` off macOS (no such
    transport, so nothing to retry).
    """

    if sys.platform != "darwin":
        return None
    argv = [
        OSASCRIPT,
        *_APPLESCRIPT,
        "--",
        _clean_text(title, TITLE_LIMIT),
        _clean_text(body, BODY_LIMIT),
    ]
    try:
        completed = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("notify: osascript failed to run (%s)", type(exc).__name__)
        return False
    if completed.returncode != 0:
        logger.warning("notify: osascript exited %s", completed.returncode)
        return False
    return True


# --- Slack transport --------------------------------------------------------


def _keychain_secret(account: str, *, runner: Runner = subprocess.run) -> str | None:
    """Read one generic password from the login Keychain; ``None`` if absent."""

    try:
        completed = runner(
            [SECURITY, "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", account, "-w"],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = (completed.stdout or "").strip()
    return value or None


def _ssl_context() -> ssl.SSLContext:
    try:
        from shared.tls import client_ssl_context  # noqa: PLC0415 - optional dependency

        return client_ssl_context()
    except Exception:  # noqa: BLE001 - fall back to the platform default
        return ssl.create_default_context()


def _slack_call(
    method: str,
    token: str,
    payload: dict[str, Any],
    *,
    opener: Callable[..., Any],
    context: ssl.SSLContext | None,
) -> dict[str, Any] | None:
    request = urllib.request.Request(
        SLACK_API + method,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=HTTP_TIMEOUT_SECONDS, context=context) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - never echo the request (it holds the token)
        logger.warning("notify: Slack %s failed (%s)", method, type(exc).__name__)
        return None
    if not isinstance(parsed, dict) or not parsed.get("ok"):
        error = parsed.get("error") if isinstance(parsed, dict) else "bad_response"
        logger.warning("notify: Slack %s refused (%s)", method, _clean_text(error, 80))
        return None
    return parsed


def _send_slack(
    text: str,
    *,
    keychain_runner: Runner = subprocess.run,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bool | None:
    """DM the operator on Slack.

    ``True`` on success, ``False`` on failure (worth a retry), ``None`` when
    skipped because the Keychain items are absent (not configured).
    """

    token = _keychain_secret(KEYCHAIN_TOKEN_ACCOUNT, runner=keychain_runner)
    user_id = _keychain_secret(KEYCHAIN_USER_ACCOUNT, runner=keychain_runner)
    if not token or not user_id:
        logger.info(
            "notify: Slack DM skipped (no Keychain items %s/%s and %s/%s; see docs/watchdog.md)",
            KEYCHAIN_SERVICE,
            KEYCHAIN_TOKEN_ACCOUNT,
            KEYCHAIN_SERVICE,
            KEYCHAIN_USER_ACCOUNT,
        )
        return None
    context = _ssl_context()
    opened = _slack_call(
        "conversations.open", token, {"users": user_id}, opener=opener, context=context
    )
    channel = (opened or {}).get("channel") or {}
    channel_id = channel.get("id") if isinstance(channel, dict) else None
    if not channel_id:
        return False
    posted = _slack_call(
        "chat.postMessage",
        token,
        {"channel": channel_id, "text": text, "unfurl_links": False},
        opener=opener,
        context=context,
    )
    return posted is not None
