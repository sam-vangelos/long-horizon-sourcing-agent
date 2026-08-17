"""Researcher module Slice 1 — workspace API tolerates researcher rows.

Per Researcher Module Spec Opinion 4: every saved researcher is a
`candidates` row with SAVE-class `terminal_decision` (no external
side_effects table), and the workspace surface aggregates from the
existing `candidates` table source-agnostically.

This test asserts:

- `cloris.control_plane.aggregate_workspace` discovers a researcher
  state_dir under `output/state/researcher/<key>/` (Researcher Slice 1
  registered `"researcher"` in the launcher registry; multi-agent-
  execution Slice 1.5 made the registry the single source-of-truth
  for the registered-source list — `cloris/control_plane.py` now
  iterates `known_sources()` instead of the deleted `_SOURCES`).
- A SAVE-class candidate row with `source="researcher"` round-trips
  through the workspace aggregator without crashing.
- `CandidateCardSummary.source` accepts `"researcher"` (Slice 1
  widened the Pydantic Literal).
- The `full_decision.rationale` + `full_decision.confidence` write
  contract per Researcher Module Spec Opinion 6 surfaces correctly
  through `extract_save_reason_and_confidence` (the substrate's
  source-agnostic read model).
"""

from __future__ import annotations

from pathlib import Path

from cloris.control_plane import aggregate_workspace
from shared.runtime_state.store import RuntimeStateStore


def _build_state_dir(tmp_path: Path, source: str, key: str) -> Path:
    state_dir = tmp_path / source / key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _seed_run(
    state_dir: Path,
    *,
    source: str,
    brief_id: str,
    brief_snapshot: dict | None = None,
) -> tuple[RuntimeStateStore, int]:
    import json

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    snapshot_json = (
        json.dumps(brief_snapshot) if brief_snapshot is not None else None
    )
    run_id = store.start_run(
        source=source,
        brief_id=brief_id,
        output_dir=str(state_dir),
        mode="fresh",
        resume_state={"brief_name": brief_id},
        brief_snapshot_json=snapshot_json,
    )
    return store, run_id


def _save_researcher_candidate(
    store: RuntimeStateStore,
    *,
    run_id: int,
    brief_id: str,
    identity_key: str,
    display_name: str,
    profile_url: str,
    decision: str,
    full_decision_payload: dict,
) -> None:
    """Walk a researcher candidate to full_terminal with a SAVE-class
    decision and a `full_decision`-shaped terminal payload (the wire
    contract per Spec Opinion 6).
    """

    source = "researcher"
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=None,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        display_name=display_name,
        profile_url=profile_url,
    )

    snippet_id = store.start_attempt(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        stage="snippet",
        display_name=display_name,
        profile_url=profile_url,
    )
    store.finish_attempt_success(
        attempt_id=snippet_id,
        new_state="snippet_extracted",
        payload={},
        run_id=run_id,
    )

    store.set_candidate_state(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="facial_started",
    )
    facial_id = store.start_attempt(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        stage="facial",
    )
    store.finish_attempt_success(
        attempt_id=facial_id,
        new_state="facial_terminal",
        payload={},
        run_id=run_id,
    )

    store.set_candidate_state(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        new_state="full_started",
    )
    full_id = store.start_attempt(
        run_id=run_id,
        source=source,
        brief_id=brief_id,
        identity_key=identity_key,
        stage="full",
    )
    # The wire contract: write the full_decision dict at
    # `terminal_payload["full_decision"]` so
    # `extract_save_reason_and_confidence` reads `rationale` and
    # `confidence` from the canonical path. Researcher Module Spec
    # Opinion 6 makes this non-negotiable.
    store.finish_attempt_success(
        attempt_id=full_id,
        new_state="full_terminal",
        terminal_decision=decision,
        payload={"full_decision": full_decision_payload},
        run_id=run_id,
    )


def test_workspace_aggregator_discovers_researcher_state_dir(tmp_path: Path) -> None:
    """A state_dir under output/state/researcher/<key>/ is enumerated
    by the workspace aggregator (`"researcher"` is registered in
    `cloris.launchers.LAUNCHERS`; the aggregator iterates
    `known_sources()` per multi-agent-execution Slice 1.5).
    """

    state_dir = _build_state_dir(tmp_path, "researcher", "head-of-applied-ai-lab")
    store, run_id = _seed_run(
        state_dir,
        source="researcher",
        brief_id="brief-researcher-1",
        brief_snapshot={"role_title": "Head of Applied AI Lab"},
    )
    _save_researcher_candidate(
        store,
        run_id=run_id,
        brief_id="brief-researcher-1",
        identity_key="orcid:0000-0001-2345-6789",
        display_name="Dr. Jane Researcher",
        profile_url="https://orcid.org/0000-0001-2345-6789",
        decision="SAVE",
        full_decision_payload={
            "rationale": (
                "First-author at NeurIPS 2024 on RLHF reward modeling; "
                "h-index 14; recent publications align with the brief's "
                "post-training capability area."
            ),
            "confidence": 0.91,
        },
    )

    workspace = aggregate_workspace(
        brief_id="brief-researcher-1", state_root=tmp_path
    )

    assert workspace is not None, (
        "Workspace aggregator must discover researcher state dirs. "
        "After multi-agent-execution Slice 1.5, the registered-source "
        "list lives in `cloris.launchers.known_sources()` (the "
        "duplicate `_SOURCES` literal in control_plane.py was removed)."
    )
    assert "researcher" in workspace.sources
    assert workspace.total_saves == 1


def test_researcher_card_round_trips_save_reason_and_confidence(
    tmp_path: Path,
) -> None:
    """The CandidateCardSummary populated from a researcher row carries
    the rationale + confidence written under `full_decision`. This pins
    Spec Opinion 6's wire contract for the researcher source.
    """

    state_dir = _build_state_dir(tmp_path, "researcher", "key-2")
    store, run_id = _seed_run(
        state_dir,
        source="researcher",
        brief_id="brief-researcher-2",
        brief_snapshot={"role_title": "Frontier-lab Researcher"},
    )
    _save_researcher_candidate(
        store,
        run_id=run_id,
        brief_id="brief-researcher-2",
        identity_key="openalex:A1234567890",
        display_name="Wei Wang",
        profile_url="https://openalex.org/A1234567890",
        decision="INFERENTIAL_SAVE",
        full_decision_payload={
            "rationale": "Common name — multiple OpenAlex IDs; manual review needed.",
            "confidence": 0.62,
        },
    )

    workspace = aggregate_workspace(
        brief_id="brief-researcher-2", state_root=tmp_path
    )

    assert workspace is not None
    assert len(workspace.candidates) == 1
    card = workspace.candidates[0]
    assert card.source == "researcher"
    assert card.identity_key == "openalex:A1234567890"
    assert card.display_name == "Wei Wang"
    assert card.profile_url == "https://openalex.org/A1234567890"
    assert card.terminal_decision == "INFERENTIAL_SAVE"
    assert card.confidence == 0.62
    assert card.save_reason == (
        "Common name — multiple OpenAlex IDs; manual review needed."
    )


def test_researcher_card_summary_pydantic_literal_widened() -> None:
    """The Pydantic `CandidateCardSummary.source` Literal must include
    `"researcher"` (Slice 1 widening at `cloris/models.py:586`). If the
    Literal is too narrow, constructing a researcher card raises a
    Pydantic ValidationError before the test even reaches assertions.
    """

    from cloris.models import CandidateCardSummary

    card = CandidateCardSummary(
        candidate_id=1,
        source="researcher",
        identity_key="orcid:0000-0001-2345-6789",
        display_name="Dr. Jane Researcher",
        profile_url="https://orcid.org/0000-0001-2345-6789",
        terminal_decision="SAVE",
    )
    assert card.source == "researcher"
