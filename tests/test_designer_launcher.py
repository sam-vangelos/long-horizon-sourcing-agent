"""Designer module Slice 1 — launcher registry + state-key + state-dir.

Pins the launch-path contract for `target_modules: ["designer"]` briefs:

- `LAUNCHERS["designer"]` is registered and exposes the four required
  callables (state_key_fn, state_dir_fn, orchestrator_argv_fn,
  save_destination_blocker_fn).
- `designer_state_key` derives a stable canonical key from brief
  content (brief.id / role_title / filename stem fallback).
- `resolve_designer_state_dir` lives under `output/state/designer/<key>/`.
- `_designer_orchestrator_argv` produces a parseable argv that
  `designer.session_orchestrator.main` accepts without raising.
- `LAUNCHERS["designer"].progress_kind == DESIGNER_BEHANCE_QUERY_KIND`
  so the workspace progress aggregator doesn't silently skip Designer
  state-dirs (Slice 1.5 of the multi-agent-execution plan moved this
  contract from the ``_progress_kind_for_source`` ladder in
  ``cloris/control_plane.py`` to the launcher registry).
- `_dispatch_in_process` accepts `source="designer"` for the
  frozen-app fallback.
- `"designer"` is registered in the launcher registry (now the
  single source-of-truth for the registered-source list — Slice 1.5
  removed the duplicate ``_SOURCES`` literal in ``control_plane.py``).
- `Source` Pydantic literal in `cloris/models.py` accepts `"designer"`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cloris.launchers import (
    LAUNCHERS,
    SaveDestinationBlocker,
    _designer_orchestrator_argv,
    _designer_save_destination_blocker,
    get_launcher,
    known_sources,
)
from cloris.models import CandidateCardSummary
from shared.runtime_state.store import (
    DESIGNER_BEHANCE_QUERY_KIND,
    DESIGNER_CSE_QUERY_KIND,
)


# ---------------------------------------------------------------------------
# Registry membership
# ---------------------------------------------------------------------------


def test_designer_is_registered_launcher() -> None:
    assert "designer" in LAUNCHERS
    assert "designer" in known_sources()


def test_designer_launcher_carries_four_callables() -> None:
    entry = get_launcher("designer")
    assert callable(entry.state_key_fn)
    assert callable(entry.state_dir_fn)
    assert callable(entry.orchestrator_argv_fn)
    assert callable(entry.save_destination_blocker_fn)


def test_designer_save_destination_blocker_returns_none() -> None:
    """Designer saves land in the workspace; no per-brief destination
    config to gate on. The blocker is a no-op (returns None)."""

    result = _designer_save_destination_blocker("/nonexistent/brief.json")
    assert result is None


def test_designer_is_registered_in_known_sources() -> None:
    """Slice 1.5 (multi-agent-execution Phase 1) removed the duplicate
    ``_SOURCES`` literal in ``cloris/control_plane.py``; the launcher
    registry's ``known_sources()`` is now the single source-of-truth.
    """

    assert "designer" in known_sources()


def test_progress_kind_for_designer_returns_behance_query() -> None:
    """Workspace progress aggregator needs a non-empty kind for Designer
    state-dirs — otherwise progress lands silently as 0/0. Slice 1.5
    moved this contract from the ``_progress_kind_for_source`` ladder
    to the launcher registry.
    """

    assert get_launcher("designer").progress_kind == DESIGNER_BEHANCE_QUERY_KIND


def test_designer_work_unit_kinds_are_distinct() -> None:
    """Behance-channel and CSE-channel work-units carry different kinds
    so projection layers can disambiguate per-source progress."""

    assert DESIGNER_BEHANCE_QUERY_KIND == "designer_behance_query"
    assert DESIGNER_CSE_QUERY_KIND == "designer_cse_query"
    assert DESIGNER_BEHANCE_QUERY_KIND != DESIGNER_CSE_QUERY_KIND


# ---------------------------------------------------------------------------
# state-key + state-dir resolution
# ---------------------------------------------------------------------------


def _write_minimal_designer_brief(path: Path, *, role: str = "Senior product designer") -> None:
    payload = {
        "id": role.lower().replace(" ", "_"),
        "role_title": role,
        "target_modules": ["designer"],
        "capability_areas": [
            {"name": "Surface design", "description": "Ships shipped product."}
        ],
        "depth_distinction": {
            "builder_definition": "Owns surface end-to-end.",
            "user_definition": "Iterates on shipped surfaces.",
            "edge_case_guidance": "Borderline = surface ownership.",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_designer_state_key_uses_brief_id(tmp_path: Path) -> None:
    brief_path = tmp_path / "config" / "designer-test" / "brief.json"
    _write_minimal_designer_brief(brief_path, role="Senior product designer")

    from shared.output_paths import designer_state_key

    key = designer_state_key(brief_path=str(brief_path))
    # `senior_product_designer` ← slugified brief.id.
    assert key == "senior_product_designer"


def test_resolve_designer_state_dir_lands_under_designer_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path = tmp_path / "config" / "designer-test" / "brief.json"
    _write_minimal_designer_brief(brief_path)

    # Redirect OUTPUT_ROOT so the test doesn't pollute the real
    # output/state tree. shared.config.OUTPUT_DIR is the live root.
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("shared.config.OUTPUT_DIR", output_root)
    monkeypatch.setattr("shared.output_paths.OUTPUT_ROOT", output_root)
    monkeypatch.setattr("shared.output_paths.STATE_ROOT", output_root / "state")

    from shared.output_paths import resolve_designer_state_dir

    state_dir = resolve_designer_state_dir(brief_path=str(brief_path))
    assert "designer" in state_dir.parts
    assert state_dir.name == "senior_product_designer"
    assert state_dir.exists()


# ---------------------------------------------------------------------------
# Orchestrator argv shape — parseable by designer.session_orchestrator.main
# ---------------------------------------------------------------------------


def test_designer_orchestrator_argv_round_trips_through_stub(tmp_path: Path) -> None:
    """The argv builder produces argv that the Slice 1 stub's `main()`
    can parse without raising. This is the launch-chain smoke test."""

    brief_path = tmp_path / "brief.json"
    _write_minimal_designer_brief(brief_path)
    state_dir = tmp_path / "state"

    argv = _designer_orchestrator_argv(
        brief_path=str(brief_path),
        state_dir=str(state_dir),
        resume=False,
    )

    # argv shape: [python, -m, designer.session_orchestrator, --brief, ..., --state-dir, ...]
    assert argv[0] == sys.executable
    assert argv[1] == "-m"
    assert argv[2] == "designer.session_orchestrator"
    assert "--brief" in argv
    assert "--state-dir" in argv
    assert "--resume" not in argv

    # The orchestrator-cli args (everything after `-m MODULE`) parse cleanly.
    from designer.session_orchestrator import _build_arg_parser

    parser = _build_arg_parser()
    parsed = parser.parse_args(argv[3:])
    assert parsed.brief == str(brief_path)
    assert parsed.state_dir == str(state_dir)
    assert parsed.resume is False


def test_designer_orchestrator_argv_threads_resume_flag(tmp_path: Path) -> None:
    brief_path = tmp_path / "brief.json"
    _write_minimal_designer_brief(brief_path)

    argv = _designer_orchestrator_argv(
        brief_path=str(brief_path),
        state_dir=str(tmp_path / "state"),
        resume=True,
    )
    assert "--resume" in argv


# ---------------------------------------------------------------------------
# Worker frozen-app fallback dispatch — accepts source="designer"
# ---------------------------------------------------------------------------


def test_dispatch_in_process_accepts_designer_source(tmp_path: Path) -> None:
    """Frozen .app bundles can't execvp — they fall back to in-process
    dispatch in `cloris.worker._dispatch_in_process`. Each registered
    source needs an explicit elif branch (no DISPATCH_TABLE constant)."""

    brief_path = tmp_path / "brief.json"
    _write_minimal_designer_brief(brief_path)
    state_dir = tmp_path / "state"

    argv = _designer_orchestrator_argv(
        brief_path=str(brief_path),
        state_dir=str(state_dir),
        resume=False,
    )

    from cloris.worker import _dispatch_in_process

    exit_code = _dispatch_in_process("designer", argv)
    # Slice 1 stub exits 0 on success.
    assert exit_code == 0


def test_dispatch_in_process_unknown_source_returns_2() -> None:
    """Defensive: unknown source still rejects with 2, not crash."""

    from cloris.worker import _dispatch_in_process

    exit_code = _dispatch_in_process("nonexistent", [sys.executable, "-m", "x"])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Wire literal: CandidateCardSummary.source accepts "designer"
# ---------------------------------------------------------------------------


def test_candidate_card_summary_accepts_designer_source() -> None:
    """A Designer-evaluated candidate must land cleanly on the wire."""

    summary = CandidateCardSummary(
        candidate_id=1,
        source="designer",
        identity_key="behance:exampledesigner",
        display_name="Example Designer",
        profile_url="https://www.behance.net/exampledesigner",
        terminal_decision="SAVE",
        save_reason="Strong product surface design with clear typographic refinement.",
        confidence=0.78,
    )
    assert summary.source == "designer"
