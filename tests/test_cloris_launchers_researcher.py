"""Researcher module Slice 1 — launcher registry coverage.

Asserts the contract that `cloris/launchers/__init__.py` exposes a
fully-populated `LauncherEntry` for the `"researcher"` source so the
generic launch path (POST /api/launch/{source}) can spawn the worker
without LinkedIn- or GitHub-specific branching.

Per Researcher Module Spec Opinion 4 (workspace is the only save
destination), the save_destination_blocker_fn returns None — there is
no per-brief destination input the recruiter must fill in before
launch. Slice 7 adds discipline + research_topics inputs; those are
evaluation inputs, not save-destination configuration.
"""

from __future__ import annotations

from pathlib import Path

from cloris.launchers import (
    LAUNCHERS,
    LauncherEntry,
    get_launcher,
    known_sources,
)


def test_researcher_entry_present_in_launchers_dict() -> None:
    assert "researcher" in LAUNCHERS
    entry = LAUNCHERS["researcher"]
    assert isinstance(entry, LauncherEntry)


def test_known_sources_includes_researcher() -> None:
    sources = known_sources()
    assert "researcher" in sources
    # Stable sort contract — known_sources returns sorted ascending.
    assert sources == tuple(sorted(sources))


def test_get_launcher_returns_researcher_entry() -> None:
    entry = get_launcher("researcher")
    assert isinstance(entry, LauncherEntry)
    assert entry is LAUNCHERS["researcher"]


def test_researcher_save_destination_blocker_returns_none(tmp_path: Path) -> None:
    """Workspace is always available — no readiness blocker fires.

    Per Researcher Module Spec Opinion 4: researchers without LinkedIn
    profiles can't be saved to LinkedIn Recruiter; every saved researcher
    is a `candidates` row with SAVE-class `terminal_decision`. The
    blocker should return None for any brief path (the function does
    not even read the brief content).
    """

    entry = LAUNCHERS["researcher"]
    # Even pointing at a non-existent path should return None — no
    # per-brief destination check is needed for researcher.
    nonexistent = tmp_path / "does_not_exist.json"
    assert entry.save_destination_blocker_fn(str(nonexistent)) is None


def test_researcher_orchestrator_argv_shape() -> None:
    """The argv produced must be in the form
    [python, -m, researcher.session_orchestrator, --brief, <path>,
     --state-dir, <dir>]
    so the worker's in-process dispatch and the frozen-app dispatch
    both find the orchestrator.

    Reopen P7.5(b): ``--resume`` is NEVER appended, even when
    ``resume=True`` — researcher's CLI now treats ``--resume`` as a hard
    "not implemented" error rather than silently re-running from
    scratch, so the launcher must not hand it a flag its own CLI will
    refuse. ``resume`` stays a required kwarg only so this function's
    signature matches every other source's ``orchestrator_argv_fn``.
    """

    entry = LAUNCHERS["researcher"]
    argv_no_resume = entry.orchestrator_argv_fn(
        "/path/to/brief.json",
        "/path/to/state_dir",
        resume=False,
    )
    assert argv_no_resume[1:4] == ["-m", "researcher.session_orchestrator", "--brief"]
    assert "--brief" in argv_no_resume
    assert "/path/to/brief.json" in argv_no_resume
    assert "--state-dir" in argv_no_resume
    assert "/path/to/state_dir" in argv_no_resume
    assert "--resume" not in argv_no_resume

    argv_resume = entry.orchestrator_argv_fn(
        "/path/to/brief.json",
        "/path/to/state_dir",
        resume=True,
    )
    assert "--resume" not in argv_resume
    assert argv_resume == argv_no_resume
