"""Cloris first-launch onboarding state.

Owns the ``acknowledged.json`` and ``.env`` files under
``cloris_user_data_dir()`` plus the read/write semantics the API
layer wires up at ``/api/onboarding/*`` and the welcome-screen flow
in ``cloris/frontend/src/components/Welcome.svelte``.

Two artifacts:

1. **``<user_data_dir>/.env``** — credentials. Append-or-update
   semantics for individual keys via :func:`upsert_credential`. The
   file is chmod 600 (owner-only) on every write so a curious
   process spawned by another user on the same machine can't read
   the recipient's Anthropic key. ``shared.config`` loads this file
   on import (layered with the project-root ``.env`` for dev — see
   ``shared/config.py``).

2. **``<user_data_dir>/acknowledged.json``** — the recipient's
   first-launch acknowledgment of the LinkedIn-Recruiter relational
   bargain (see the welcome screen copy + the IT-defensibility
   README). Carries the acknowledgment timestamp + the Cloris version
   that was running when the recipient acknowledged so a major bump
   that changes the operational surface can require re-acknowledgment.

The welcome screen gates ``Continue`` on:

- A non-empty ``ANTHROPIC_API_KEY`` in the user-data ``.env``, AND
- An ``acknowledged.json`` whose ``acknowledged_at`` is non-empty.

When both are true, :func:`onboarding_status` returns
``welcome_complete=True`` and the frontend gate falls through to the
existing home route. The welcome screen is not shown on subsequent
launches unless the recipient explicitly returns to it from Settings
(or a major version bump invalidates the acknowledgment, which is a
later policy hook — not implemented in Phase 0).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from shared.user_data_dir import (
    acknowledgment_file_path,
    env_file_path,
)

log = logging.getLogger("cloris.onboarding")


# Mapping from the wire-side credential key (lowercased, snake_case)
# to the canonical environment variable name. Only credentials the
# welcome / Settings surface allows the recipient to edit are listed
# here — server-only / dev-only env vars cannot be set through the
# API. The keys mirror ``_CREDENTIAL_LABELS`` in ``cloris/api.py``.
_EDITABLE_CREDENTIALS: dict[str, str] = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
    "perplexity_api_key": "PERPLEXITY_API_KEY",
}


_ENV_LINE_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


class UnknownCredentialKeyError(Exception):
    """Raised when the API receives a credential key not in
    :data:`_EDITABLE_CREDENTIALS`. The route maps this to HTTP 422
    so a typo in the welcome / Settings UI surfaces clearly rather
    than silently writing a junk key into the recipient's .env."""

    def __init__(self, key: str) -> None:
        super().__init__(
            f"unknown credential key {key!r}; allowed: "
            f"{sorted(_EDITABLE_CREDENTIALS.keys())}"
        )
        self.key = key
        self.allowed = tuple(sorted(_EDITABLE_CREDENTIALS.keys()))


@dataclass(frozen=True)
class OnboardingStatus:
    """Read-side wire shape for ``GET /api/onboarding/status``.

    The frontend gate uses ``welcome_complete`` directly: True →
    fall through to the home route, False → render Welcome.svelte.
    The other fields let the welcome screen render the right initial
    state (API key field empty vs prefilled-and-locked, acknowledgment
    box checked vs blank).
    """

    welcome_complete: bool
    anthropic_present: bool
    acknowledged: bool
    acknowledged_at: str | None
    env_path: str
    acknowledgment_path: str


def is_editable_credential(key: str) -> bool:
    """True iff ``key`` is one of the credential keys the recipient
    is allowed to set through the welcome / Settings surface."""

    return key in _EDITABLE_CREDENTIALS


def env_var_name(key: str) -> str:
    """Map wire-side credential key to canonical env var name. Raises
    :class:`UnknownCredentialKeyError` for unrecognized keys."""

    if key not in _EDITABLE_CREDENTIALS:
        raise UnknownCredentialKeyError(key)
    return _EDITABLE_CREDENTIALS[key]


def upsert_credential(key: str, value: str) -> Path:
    """Insert-or-update a credential in the user-data ``.env``.

    Atomicity: writes to a tmp file then ``os.replace`` so a reader
    can never observe a partial file. Permissions: the file is
    ``chmod 0o600`` after the replace so the credential is
    owner-readable only.

    Also sets ``os.environ[env_var]`` immediately so the in-process
    ``shared.config`` symbols pick up the new value on the next
    attribute read (the symbols are bound at import time, but
    consumers that read them at call time — e.g. the LLM clients —
    see the new value).

    Returns the absolute path of the env file written.
    """

    if not is_editable_credential(key):
        raise UnknownCredentialKeyError(key)

    target_var = env_var_name(key)
    assert re.fullmatch(r"[A-Z][A-Z0-9_]*", target_var)
    cleaned = (value or "").strip()
    if any(c in cleaned for c in "\r\n\x00"):
        raise ValueError(
            f"credential {key!r} cannot contain CR, LF, or NUL characters"
        )
    if not cleaned:
        raise ValueError(
            f"credential {key!r} cannot be empty; pass a non-blank value or "
            "use a delete endpoint when one exists"
        )

    env_path = env_file_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: list[str] = []
    if env_path.exists():
        try:
            existing_lines = env_path.read_text().splitlines()
        except OSError as exc:
            log.warning(
                "cloris.onboarding: existing .env unreadable, replacing: %r",
                exc,
            )
            existing_lines = []

    new_lines: list[str] = []
    replaced = False
    for line in existing_lines:
        match = _ENV_LINE_PATTERN.match(line)
        if match is not None and match.group(1) == target_var:
            new_lines.append(f"{target_var}={cleaned}")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{target_var}={cleaned}")

    payload = "\n".join(new_lines) + "\n"

    tmp_path = env_path.with_suffix(env_path.suffix + ".tmp")
    tmp_path.write_text(payload)
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, env_path)
    # Defensive: re-chmod after the replace too. ``os.replace`` on
    # macOS preserves the source's permissions but a reader on a
    # different filesystem may not, and mode 600 is cheap to assert
    # twice.
    os.chmod(env_path, 0o600)

    os.environ[target_var] = cleaned
    config_mod = sys.modules.get("shared.config")
    if config_mod is not None:
        setattr(config_mod, target_var, cleaned)
    if target_var == "ANTHROPIC_API_KEY":
        health_mod = sys.modules.get("cloris.anthropic_health")
        clear_cache = getattr(health_mod, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()

    log.info(
        "cloris.onboarding: wrote credential %s to %s (mode 0o600)",
        target_var,
        env_path,
    )
    return env_path


def _parse_env_var(env_text: str, var_name: str) -> str | None:
    """Return the value of ``var_name`` from a ``.env``-formatted
    string, or ``None`` if absent / empty. Matches the format
    :func:`upsert_credential` writes (no surrounding quotes, no
    inline comments)."""

    for raw in env_text.splitlines():
        match = _ENV_LINE_PATTERN.match(raw)
        if match is None or match.group(1) != var_name:
            continue
        eq_idx = raw.index("=")
        value = raw[eq_idx + 1 :].strip()
        if not value:
            return None
        return value
    return None


def credential_present(key: str) -> bool:
    """True iff a non-empty value for ``key`` is observable.

    Reads from ``os.environ`` first (live in-process state), then
    falls back to the user-data ``.env`` on disk. The on-disk read is
    important for the welcome-screen status endpoint: the credential
    may have been set by a previous launch and not yet re-loaded into
    this process's ``os.environ``.
    """

    if not is_editable_credential(key):
        return False
    var_name = _EDITABLE_CREDENTIALS[key]

    live = os.environ.get(var_name, "").strip()
    if live:
        return True

    env_path = env_file_path()
    if not env_path.exists():
        return False
    try:
        env_text = env_path.read_text()
    except OSError:
        return False
    return _parse_env_var(env_text, var_name) is not None


def record_acknowledgment(*, cloris_version: str) -> Path:
    """Write the recipient's first-launch acknowledgment to disk.

    The ``acknowledged.json`` payload carries the acknowledgment
    timestamp and the Cloris version that was running. Idempotent:
    re-acknowledging overwrites the timestamp + version with the
    current values. The frontend never re-acknowledges silently —
    this is only called from an explicit user action.

    Returns the path of the file written.
    """

    path = acknowledgment_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "acknowledged_at": datetime.now(timezone.utc).isoformat(),
        "cloris_version": cloris_version,
        "message": (
            "Recipient acknowledged Cloris's LinkedIn-Recruiter operational "
            "surface (Cloris uses recipient's Chrome to read and save "
            "candidates from their Recruiter account, at human pace, only "
            "when explicitly started)."
        ),
    }

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp_path, path)
    log.info("cloris.onboarding: wrote acknowledgment to %s", path)
    return path


def acknowledgment_record() -> dict | None:
    """Return the parsed acknowledgment payload, or ``None`` if absent
    / malformed. Never raises — a corrupt acknowledgment file collapses
    to "not acknowledged" and the welcome screen will surface again."""

    path = acknowledgment_file_path()
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def is_acknowledged() -> bool:
    """True iff the acknowledgment file is present with a non-empty
    ``acknowledged_at`` timestamp."""

    record = acknowledgment_record()
    if record is None:
        return False
    ts = record.get("acknowledged_at")
    return isinstance(ts, str) and ts.strip() != ""


def onboarding_status() -> OnboardingStatus:
    """Return the welcome-gate status for the API surface.

    ``welcome_complete`` is true iff Anthropic credentials are
    present AND the acknowledgment is recorded. The frontend uses
    this single boolean to decide whether to render Welcome.svelte
    or fall through to the route table.
    """

    record = acknowledgment_record()
    acknowledged_at = (
        str(record.get("acknowledged_at"))
        if record is not None and isinstance(record.get("acknowledged_at"), str)
        else None
    )
    anthropic_present = credential_present("anthropic_api_key")
    acknowledged = acknowledged_at is not None and acknowledged_at.strip() != ""

    return OnboardingStatus(
        welcome_complete=anthropic_present and acknowledged,
        anthropic_present=anthropic_present,
        acknowledged=acknowledged,
        acknowledged_at=acknowledged_at,
        env_path=str(env_file_path()),
        acknowledgment_path=str(acknowledgment_file_path()),
    )
