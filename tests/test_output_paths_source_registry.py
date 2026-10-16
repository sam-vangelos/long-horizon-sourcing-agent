"""Pins shared-side source allowlist against the launcher registry."""

from __future__ import annotations

from pathlib import Path

from cloris.launchers import known_sources
from shared.output_paths import KNOWN_STATE_SOURCES, enumerate_state_dirs


def test_enumerate_state_dirs_ignores_unregistered_directories(tmp_path: Path) -> None:
    """Unregistered top-level dirs (scratch) must not be enumerated as sources."""

    (tmp_path / "linkedin" / "b1").mkdir(parents=True)
    (tmp_path / "github" / "b1").mkdir(parents=True)
    (tmp_path / "scratch" / "b1").mkdir(parents=True)
    (tmp_path / "_identity" / "x").mkdir(parents=True)
    (tmp_path / "orchestration" / "x").mkdir(parents=True)

    discovered_sources = {source for source, _ in enumerate_state_dirs(tmp_path)}

    assert discovered_sources == {"linkedin", "github"}


def test_shared_source_allowlist_covers_launcher_registry() -> None:
    """Every registered launcher source must appear in KNOWN_STATE_SOURCES."""

    launcher_sources = set(known_sources())
    shared_sources = set(KNOWN_STATE_SOURCES)

    assert launcher_sources <= shared_sources, (
        "Launcher registry has sources not in KNOWN_STATE_SOURCES — "
        "update KNOWN_STATE_SOURCES in shared/output_paths.py when a launcher "
        "is registered"
    )
