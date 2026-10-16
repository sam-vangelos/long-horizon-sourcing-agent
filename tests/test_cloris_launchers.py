"""Tests for the per-source launcher registry (`cloris.launchers`).

Pins the registry contract:

- :data:`LAUNCHERS` carries the canonical set of registered sources.
- :func:`known_sources` returns them in stable ascending order (used
  by the API layer for 422 ``allowed`` payloads).
- Each :class:`LauncherEntry` exposes the four callables Slice F1
  spawn helpers expect; the entry's defaults (no
  ``save_destination_blocker_fn``) resolve to a no-op.
- The Slice 1 stub for executive search produces the argv shape
  `cloris.worker` execvp's into.

These tests cover the launcher surface end-to-end without spawning
real subprocesses; the launch endpoint integration test lives in
:mod:`tests.test_launch_endpoint_generic`.
"""

from __future__ import annotations

import sys

from cloris.launchers import (
    LAUNCHERS,
    LauncherEntry,
    SaveDestinationBlocker,
    get_launcher,
    known_sources,
)


def test_known_sources_includes_exec_search() -> None:
    sources = known_sources()
    assert "exec_search" in sources


def test_known_sources_returns_stable_ascending_order() -> None:
    sources = known_sources()
    assert sources == tuple(sorted(sources))
    assert len(set(sources)) == len(sources)


def test_known_sources_carries_the_canonical_registered_set() -> None:
    """Every key in `LAUNCHERS` is exposed via `known_sources`."""

    assert set(known_sources()) == set(LAUNCHERS.keys())


def test_get_launcher_returns_exec_search_entry() -> None:
    entry = get_launcher("exec_search")
    assert isinstance(entry, LauncherEntry)


def test_exec_search_save_destination_blocker_default_is_none() -> None:
    """Slice 1: exec_search has no per-brief destination configuration.

    The Cloris-native shortlist destination ships in Slice 7. Until
    then the default no-op blocker applies (no readiness gate fires).
    """

    entry = get_launcher("exec_search")
    blocker = entry.save_destination_blocker_fn("/tmp/does-not-matter.json")
    assert blocker is None


def test_exec_search_orchestrator_argv_shape(tmp_path) -> None:
    """argv shape matches the stub's argparse contract.

    Validates the seam between `cloris.worker` execvp and
    `exec_search.session_orchestrator.main`.
    """

    entry = get_launcher("exec_search")
    brief_path = tmp_path / "brief.json"
    brief_path.write_text("{}")
    state_dir = tmp_path / "state"

    argv = entry.orchestrator_argv_fn(
        str(brief_path),
        str(state_dir),
        resume=False,
    )
    assert argv[0] == sys.executable
    assert argv[1] == "-m"
    assert argv[2] == "exec_search.session_orchestrator"
    assert "--brief" in argv
    assert "--state-dir" in argv
    assert str(brief_path) in argv
    assert str(state_dir) in argv
    assert "--resume" not in argv


def test_exec_search_orchestrator_argv_includes_resume_flag(tmp_path) -> None:
    entry = get_launcher("exec_search")
    brief_path = tmp_path / "brief.json"
    brief_path.write_text("{}")
    state_dir = tmp_path / "state"

    argv = entry.orchestrator_argv_fn(
        str(brief_path),
        str(state_dir),
        resume=True,
    )
    assert "--resume" in argv


def test_exec_search_state_key_slugifies_role_title(tmp_path) -> None:
    """state_key derives from `brief.id` (V2 loader sets that to role_title).

    Mirrors the GitHub posture: the V2 brief loader populates the compat
    Brief's `id` field from `role_title`. State_key slugifies that
    deterministically.
    """

    import json

    brief_path = tmp_path / "EXEC-SEARCH-BRIEF.json"
    brief_path.write_text(
        json.dumps(
            {
                "id": "vp-eng-2026",
                "role_title": "VP Engineering",
                "capability_areas": [
                    {"name": "x", "description": "y"}
                ],
                "depth_distinction": {
                    "builder_definition": "a",
                    "user_definition": "b",
                    "edge_case_guidance": "c",
                },
            }
        )
    )
    entry = get_launcher("exec_search")
    state_key = entry.state_key_fn(str(brief_path))
    assert state_key == "vp_engineering"


def test_exec_search_state_dir_lives_under_state_exec_search(tmp_path) -> None:
    """state_dir lives under output/state/exec_search/<state_key>/.

    Distinct from output/state/linkedin/ so confidential briefs don't
    aggregate into the LinkedIn home view.
    """

    import json

    brief_path = tmp_path / "brief.json"
    brief_path.write_text(
        json.dumps(
            {
                "id": "test-brief",
                "role_title": "VP Engineering",
                "capability_areas": [
                    {"name": "x", "description": "y"}
                ],
                "depth_distinction": {
                    "builder_definition": "a",
                    "user_definition": "b",
                    "edge_case_guidance": "c",
                },
            }
        )
    )
    entry = get_launcher("exec_search")
    state_dir = entry.state_dir_fn(str(brief_path))
    assert state_dir.parent.name == "exec_search"
    assert state_dir.name == "vp_engineering"


def test_save_destination_blocker_dataclass_shape() -> None:
    """SaveDestinationBlocker is the shape the API readiness aggregator expects."""

    blocker = SaveDestinationBlocker(
        kind="config",
        message="m",
        remediation="r",
    )
    assert blocker.kind == "config"
    assert blocker.message == "m"
    assert blocker.remediation == "r"
