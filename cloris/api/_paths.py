"""Repository layout constants for the Cloris HTTP surface.

Single module for path sentinels so tests can monkeypatch
``cloris.api._paths._PROJECT_ROOT`` (and siblings) once and every API
submodule that imports these names sees the override."""

from __future__ import annotations

from pathlib import Path

from cloris.worker import BriefPathNotFoundError

_CLORIS_PKG_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _CLORIS_PKG_ROOT.parent
_DIST_DIR = _CLORIS_PKG_ROOT / "frontend" / "dist"
_FRONTEND_SRC_DIR = _CLORIS_PKG_ROOT / "frontend" / "src"
_FRONTEND_SCAFFOLDS_DIR = _CLORIS_PKG_ROOT / "frontend" / "scaffolds"


def _resolve_config_dir() -> Path:
    """``config/`` lives under project root in dev; under user-data when frozen."""

    from shared.user_data_dir import cloris_user_data_dir, should_use_user_data_dir

    if should_use_user_data_dir():
        return cloris_user_data_dir() / "config"
    return _PROJECT_ROOT / "config"


_CONFIG_DIR = _resolve_config_dir()
# Parent of ``config/`` — use for ``Path.relative_to`` and joining catalog-relative
# ``brief.path`` strings (``config/<slug>/brief.json``) so frozen apps resolve
# writes under ``~/Library/Application Support/Cloris/`` instead of the bundle.
_CONFIG_PARENT = _CONFIG_DIR.parent


def resolve_brief_path_contained(raw: str) -> Path:
    """Resolve a JSON brief path while containing it to ``config/``."""

    raw_path = Path(raw)
    resolved = (
        raw_path if raw_path.is_absolute() else _CONFIG_PARENT / raw_path
    ).resolve()
    # MIGRATION NOTE: state-dir preflight briefs intentionally remain unavailable
    # to raw-path API routes; launch scripts bypass the API and the ID route never
    # resolved them.
    if not resolved.is_relative_to(_CONFIG_DIR) or resolved.suffix.lower() != ".json":
        raise BriefPathNotFoundError(raw)
    return resolved
