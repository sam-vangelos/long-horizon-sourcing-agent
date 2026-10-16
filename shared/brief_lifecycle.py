"""Helpers for classifying and discovering brief files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

BriefLifecycle = Literal["active", "draft", "archived", "superseded"]

ALL_BRIEF_LIFECYCLES: tuple[BriefLifecycle, ...] = (
    "active",
    "draft",
    "archived",
    "superseded",
)

_ARCHIVE_DIR_MARKERS = {"archive", "archives", "archived"}
_SUPERSEDED_DIR_MARKERS = {"superseded"}


def is_brief_path(path: Path) -> bool:
    """Return True when the path looks like a brief JSON file."""
    return path.suffix.lower() == ".json" and path.name.startswith("brief-")


def classify_brief_path(path: Path) -> BriefLifecycle:
    """Infer brief lifecycle from its filename and location."""
    if not is_brief_path(path):
        raise ValueError(f"Path does not look like a brief: {path}")

    name = path.name.lower()
    stem = path.stem.lower()
    parent_markers = {part.lower() for part in path.parts[:-1]}

    if parent_markers & _ARCHIVE_DIR_MARKERS or "archived" in stem:
        return "archived"
    if parent_markers & _SUPERSEDED_DIR_MARKERS or "superseded" in stem:
        return "superseded"
    if ".bak-" in name or stem.endswith("-draft") or "-draft-" in stem:
        return "draft"
    return "active"


def iter_brief_files(config_dir: Path, *, recursive: bool = True) -> list[Path]:
    """Return every brief JSON file under the config tree."""
    pattern = "brief-*.json"
    candidates = config_dir.rglob(pattern) if recursive else config_dir.glob(pattern)
    return sorted(path for path in candidates if path.is_file())


def discover_briefs(
    config_dir: Path,
    *,
    include: tuple[BriefLifecycle, ...] = ("active",),
    recursive: bool = True,
) -> list[Path]:
    """Return briefs matching the requested lifecycle states."""
    include_set = set(include)
    return [
        path
        for path in iter_brief_files(config_dir, recursive=recursive)
        if classify_brief_path(path) in include_set
    ]


def summarize_brief_lifecycle(
    config_dir: Path,
    *,
    recursive: bool = True,
) -> dict[BriefLifecycle, list[Path]]:
    """Group briefs by lifecycle for inventory and hygiene checks."""
    summary: dict[BriefLifecycle, list[Path]] = {
        lifecycle: [] for lifecycle in ALL_BRIEF_LIFECYCLES
    }
    for path in iter_brief_files(config_dir, recursive=recursive):
        summary[classify_brief_path(path)].append(path)
    return summary
