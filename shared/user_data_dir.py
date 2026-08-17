"""Cloris user-data directory resolution.

Production (frozen .app on macOS): Cloris's writable state — env vars,
output state dirs, runtime_state SQLite databases, the dedicated CDP
Chrome profile — lives at ``~/Library/Application Support/Cloris/``.
The signed app bundle is read-only, so PROJECT_ROOT-relative paths are
not writable.

Development (running from the repo as ``python -m cloris start``):
state stays under ``PROJECT_ROOT/output`` the way it always has, and
``.env`` loads from ``PROJECT_ROOT/.env``. No relocation, no behavior
change. Tests that don't opt in see the long-standing PROJECT_ROOT
layout.

The frozen-vs-dev split is detected via ``getattr(sys, 'frozen',
False)`` — PyInstaller and py2app both set that. Tests can force the
user-data path on with the ``CLORIS_USER_DATA_DIR`` environment
variable, which also lets a power-user dev keep state out of the repo
without touching code.
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
    convention, where every app writes its persistent per-user state.
    Cloris's macOS .app is the primary distribution target.

    Linux: ``$XDG_DATA_HOME/Cloris`` (XDG Base Directory spec), with a
    ``~/.local/share/Cloris`` fallback when ``XDG_DATA_HOME`` is unset.
    Linux coverage is cheap and lets CI tests on Linux exercise the
    same relocation semantics that the .app uses on macOS.

    Other platforms (Windows, ...): falls back to ``~/.cloris`` for
    now. Windows is not a Phase 1 distribution target; this fallback
    keeps the function total rather than raising on unfamiliar
    platforms.
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

    Production .app: yes — the bundle is read-only and PROJECT_ROOT
    resolves to a path inside ``Cloris.app/Contents/Resources/`` after
    PyInstaller extraction.

    Dev (running from the repo): no. PROJECT_ROOT/output is the
    long-standing convention; tests and existing flows expect it,
    and silently relocating state on dev machines would surprise
    every contributor.

    Tests / power users: opt in via ``CLORIS_USER_DATA_DIR``. When the
    env var is set, the loader will treat the configured path as
    canonical even outside a frozen bundle.
    """

    if is_frozen_app():
        return True
    return _env_override_dir() is not None


def env_file_path() -> Path:
    """Return the path of the user-data ``.env`` file.

    The loader in ``shared/config.py`` checks this path first when
    :func:`should_use_user_data_dir` is true, and the in-product API
    key entry surface (Phase 0 ``apikey-ui`` slice) writes here.

    The file is intentionally located at the user-data root rather
    than under a subdirectory so a recipient inspecting their own
    machine sees one obvious file (next to ``output/``,
    ``chrome-profile/``, ``acknowledged.json``) rather than an
    arbitrary nesting.
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

    Production .app: ``<user_data_dir>/chrome-profile``. The
    ``cloris.chrome_launcher`` module uses this as the
    ``--user-data-dir`` it passes to the spawned Chrome instance, so
    the recipient's everyday Chrome profile (under
    ``~/Library/Application Support/Google/Chrome``) is never
    touched.

    Dev: same path resolution, so a developer's CDP Chrome instance
    also lands under their user-data dir if they opt in. Otherwise
    :func:`should_use_user_data_dir` is false and we fall back to
    the historical ``~/.chrome-cdp`` path used by ``launch-chrome.sh``
    so existing dev sessions keep working without re-login.
    """

    if should_use_user_data_dir():
        target = cloris_user_data_dir() / "chrome-profile"
    else:
        target = Path.home() / ".chrome-cdp"
    target.mkdir(parents=True, exist_ok=True)
    return target


def acknowledgment_file_path() -> Path:
    """Return the path where the first-launch recipient acknowledgment
    is recorded.

    The welcome surface (Phase 0 ``apikey-ui`` slice) writes a JSON
    blob here when the recipient checks the "I understand and want
    Cloris to operate on my Recruiter account" box. The presence of
    this file is what gates the welcome screen on subsequent
    launches.

    Lives at the user-data root, not in a subdirectory, for the same
    inspectability reason as :func:`env_file_path`.
    """

    return cloris_user_data_dir() / "acknowledged.json"
