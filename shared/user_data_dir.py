"""Resolve Cloris's writable per-user data directory.

In a frozen macOS app, writable state — environment files, output state
directories, runtime SQLite databases, and the dedicated CDP Chrome profile —
lives at ``~/Library/Application Support/Cloris/`` because the signed bundle
is read-only. In repository development, state remains under
``PROJECT_ROOT/output`` and ``.env`` remains at ``PROJECT_ROOT/.env``.

The frozen-vs-dev split uses ``getattr(sys, 'frozen', False)``; PyInstaller
and py2app both set it. Tests and power users can select the user-data path
with ``CLORIS_USER_DATA_DIR``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen_app() -> bool:
    """True iff running inside a PyInstaller / py2app frozen bundle.

    Both freezers set ``sys.frozen``. PyInstaller additionally sets
    ``sys._MEIPASS``; we don't rely on the latter because py2app does
    not, and the user-data resolution should fire identically under
    either freezer.
    """

    return bool(getattr(sys, "frozen", False))


def _env_override_dir() -> Path | None:
    """Return ``CLORIS_USER_DATA_DIR`` as a Path if set, else ``None``.

    Lets tests and power-users force the user-data layout without
    faking frozen state. The path is expanded (``~`` honored) but not
    otherwise validated; the caller will create it on first use.
    """

    raw = os.environ.get("CLORIS_USER_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return None


def _platform_default_dir() -> Path:
    """Platform-canonical user-data directory for Cloris.

    macOS: ``~/Library/Application Support/Cloris``. Apple's HFS+/APFS
    convention is used for persistent per-user state.

    Linux: ``$XDG_DATA_HOME/Cloris`` (XDG Base Directory spec), with a
    ``~/.local/share/Cloris`` fallback when ``XDG_DATA_HOME`` is unset.

    Other platforms use ``~/.cloris`` so the function remains total rather
    than raising on an unfamiliar platform.
    """

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Cloris"
    if sys.platform.startswith("linux"):
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        if xdg:
            return Path(xdg).expanduser() / "Cloris"
        return Path.home() / ".local" / "share" / "Cloris"
    return Path.home() / ".cloris"


def cloris_user_data_dir(*, ensure_exists: bool = True) -> Path:
    """Return Cloris's user-data root, creating it if it doesn't exist.

    Override priority:

    1. ``CLORIS_USER_DATA_DIR`` env var (testing + power users).
    2. Platform-canonical default (see :func:`_platform_default_dir`).

    The created directory has the system's default umask permissions.
    Sensitive contents (the API-key ``.env`` file) are chmod'd to 600
    by their writer; this function does not touch the parent
    permissions because ``~/Library/Application Support/<App>`` is a
    standard, conventionally-shared location and locking it down
    would be both surprising and unenforceable across processes.
    """

    target = _env_override_dir() or _platform_default_dir()
    if ensure_exists:
        target.mkdir(parents=True, exist_ok=True)
    return target


def should_use_user_data_dir() -> bool:
    """True iff Cloris should resolve writable state through the
    user-data dir instead of PROJECT_ROOT.

    A frozen app uses the user-data directory because its bundle is read-only
    and ``PROJECT_ROOT`` resolves inside the extracted resources. Repository
    development keeps ``PROJECT_ROOT/output``. Tests and power users opt in
    outside a frozen bundle with ``CLORIS_USER_DATA_DIR``.
    """

    if is_frozen_app():
        return True
    return _env_override_dir() is not None


def env_file_path() -> Path:
    """Return the path of the user-data ``.env`` file.

    The loader in ``shared/config.py`` checks this path first when
    :func:`should_use_user_data_dir` is true, and the API-key entry surface
    writes here. The file stays at the user-data root beside ``output/``,
    ``chrome-profile/``, and ``acknowledged.json`` so it is easy to inspect.
    """

    return cloris_user_data_dir() / ".env"


def output_dir() -> Path:
    """Return the writable output directory Cloris should use.

    When :func:`should_use_user_data_dir` is true (frozen app or
    explicit override), this is ``<user_data_dir>/output``. Otherwise
    it falls back to ``<PROJECT_ROOT>/output`` for the dev workflow.

    Created on first call so callers can immediately write into it
    without an ``exists_ok=True`` dance.
    """

    if should_use_user_data_dir():
        target = cloris_user_data_dir() / "output"
    else:
        # Project-root fallback. This module lives at
        # ``shared/user_data_dir.py``; the project root is the parent
        # of ``shared/``.
        project_root = Path(__file__).resolve().parent.parent
        target = project_root / "output"
    target.mkdir(parents=True, exist_ok=True)
    return target


def chrome_profile_dir() -> Path:
    """Return the dedicated CDP Chrome profile directory.

    A frozen app uses ``<user_data_dir>/chrome-profile`` for the spawned
    Chrome ``--user-data-dir``, leaving the recipient's everyday Chrome
    profile untouched. Development uses the same location when the
    user-data directory is selected; otherwise it keeps ``~/.chrome-cdp``
    for existing ``launch-chrome.sh`` sessions.
    """

    if should_use_user_data_dir():
        target = cloris_user_data_dir() / "chrome-profile"
    else:
        target = Path.home() / ".chrome-cdp"
    target.mkdir(parents=True, exist_ok=True)
    return target


def acknowledgment_file_path() -> Path:
    """Return the path where the first-launch recipient acknowledgment is recorded.

    The welcome surface writes a JSON blob here when the recipient confirms
    that Cloris may operate on the Recruiter account. The file's presence
    gates the welcome screen on later launches. It lives at the user-data
    root, beside the other user-facing state files.
    """

    return cloris_user_data_dir() / "acknowledged.json"
