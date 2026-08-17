"""Researcher Slice 8 — end-to-end smoke fixture.

Exercises the full Researcher pipeline against a recorded OpenAlex
response, with stub LLMs for strategy + facial + full eval. Verifies:

- A real V2 brief file on disk loads cleanly via the brief loader.
- The session orchestrator's `main()` entrypoint wires the pipeline
  end-to-end against the runtime-state store.
- ≥10 SAVE-class candidates land in `runtime_state.sqlite3:candidates`
  with `terminal_payload_json` carrying
  `full_decision.rationale` + `full_decision.confidence` (Spec
  Opinion 6 wire contract).
- The workspace surface (`aggregate_workspace`) returns researcher
  cards with `display_name` + `save_reason` + `confidence` populated
  source-agnostically.

The smoke is observation-only — no real API access required. Per
Spec Slice 8 it documents that the pipeline can run end-to-end given
a real-shape brief file; production wiring happens at first-customer
trial.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from cloris.control_plane import aggregate_workspace
from researcher.orchestrator import ResearcherPipeline
from researcher.sources.openalex import OpenAlexClient  # noqa: F401  (smoke import)
from shared.brief_loader import load_brief
from shared.runtime_state.researcher import ResearcherRuntimeStateBridge
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixture brief — minimal V2 shape that the loader accepts.
# ---------------------------------------------------------------------------


def _write_fixture_brief(brief_path: Path) -> None:
    brief = {
        "id": "head-of-applied-ai-lab",
        "role_title": "Head of Applied AI Lab",
        "role_summary": (
            "Lead a frontier-lab applied AI team. Owns post-training, "
            "inference systems, and agent infrastructure."
        ),
        "capability_areas": [
            {
                "name": "Post-training research",
                "description": (
                    "Publishes original work on RLHF / DPO / SFT at canonical "
                    "ML venues (NeurIPS / ICML / ICLR)."
                ),
            },
            {
                "name": "Inference systems",
                "description": (
                    "Designs quantization, distillation, and serving systems "
                    "for production-grade LLM inference."
                ),
            },
            {
                "name": "Agent infrastructure",
                "description": (
                    "Designs reasoning + tool-use systems for agent loops at "
                    "production scale."
                ),
            },
        ],
        "depth_distinction": {
            "builder_definition": (
                "First-author publications at canonical ML venues in the last "
                "36 months; cited by frontier labs."
            ),
            "user_definition": (
                "Cites foundational papers but doesn't publish original work."
            ),
            "edge_case_guidance": "Borderline = full eval.",
        },
        "non_fit_patterns": [],
        "target_modules": ["researcher"],
        "source_config": {
            "researcher": {
                "research_topics": [
                    "RLHF",
                    "post-training",
                    "agent infrastructure",
                    "inference systems",
                ],
                "conference_allowlist": [
                    "NeurIPS",
                    "ICML",
                    "ICLR",
                    "COLM",
                    "TMLR",
                ],
                "discipline": "ml_general",
            }
        },
    }
    brief_path.write_text(json.dumps(brief, indent=2))


# ---------------------------------------------------------------------------
# Stubs — recorded OpenAlex response + scripted LLM
# ---------------------------------------------------------------------------


class _RecordedOpenAlexClient:
    """Returns 12 author records across 2 queries — exceeds the spec's
    "≥10 SAVE-class candidates" threshold while staying small enough to
    fit in the test fixture."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_authors(self, **kwargs: Any) -> dict:
        self.calls.append(kwargs)
        concepts = kwargs.get("concept_ids") or []
        key = ",".join(concepts) if concepts else ""
        # Salt ORCID + author_id by the concept key so two queries don't
        # produce duplicate candidates (the candidates table's UNIQUE
        # constraint dedups by identity_key).
        concept_salt = abs(hash(key)) % 9999
        return {
            "meta": {"next_cursor": ""},
            "results": [
                _author_payload(
                    author_id=f"A{key}-{i}",
                    name=f"Researcher {key}-{i}",
                    orcid=(
                        f"0000-{concept_salt:04d}-{i:04d}-{i:04d}"
                        if i % 2 == 0
                        else ""
                    ),
                    h_index=10 + i,
                    concept_id=concepts[0] if concepts else "C1",
                )
                for i in range(6)
            ],
        }


def _author_payload(
    *,
    author_id: str,
    name: str,
    orcid: str,
    h_index: int,
    concept_id: str,
) -> dict:
    return {
        "id": f"https://openalex.org/{author_id}",
        "display_name": name,
        "orcid": orcid,
        "summary_stats": {"h_index": h_index},
        "cited_by_count": h_index * 30,
        "works_count": h_index + 8,
        "counts_by_year": [
            {"year": 2024, "works_count": 4},
            {"year": 2023, "works_count": 4},
            {"year": 2022, "works_count": 3},
        ],
        "last_known_institutions": [
            {"display_name": "MIT", "country_code": "US"}
        ],
        "x_concepts": [{"id": f"https://openalex.org/{concept_id}"}],
    }


def _strategy_response(_system: str, _user: str) -> dict:
    return {
        "strategy_rationale": "Cover post-training + inference axes.",
        "generated_strings": [
            {
                "id": 1,
                "name": "Post-training: NeurIPS",
                "topic_concepts": ["C1"],
                "venue_filter": ["NeurIPS"],
                "min_year": 2023,
                "min_citations": 10,
                "ror_country_filter": [],
            },
            {
                "id": 2,
                "name": "Inference systems: ICML",
                "topic_concepts": ["C2"],
                "venue_filter": ["ICML"],
                "min_year": 2023,
                "min_citations": 10,
                "ror_country_filter": [],
            },
        ],
        "architecture": "concept_first",
    }


def _facial_yes(_system: str, _user: str) -> dict:
    return {
        "decision": "FACIAL_YES",
        "rationale": "Recent first-author work in capability area",
        "confidence": 0.9,
    }


def _full_save(_system: str, _user: str) -> dict:
    return {
        "decision": "SAVE",
        "path": "first_author_at_canonical_venue",
        "confidence": 0.92,
        "rationale": (
            "Strong first-author at canonical venues; aligns with "
            "post-training and inference capability areas."
        ),
    }


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_smoke_e2e_real_brief_file_lands_saves_on_workspace(
    tmp_path: Path,
) -> None:
    """The full chain: brief file → load → strategy → acquire →
    disambiguate → facial → full → runtime state → workspace surface.

    The workspace aggregator iterates `known_sources()` from the
    launcher registry and looks for state dirs at
    `state_root/<source>/<state_key>/`, so the test nests the runtime
    DB under that source-rooted layout. (Multi-agent-execution Slice
    1.5 made `cloris.launchers.known_sources()` the single
    source-of-truth; the duplicate `_SOURCES` literal in
    `cloris/control_plane.py` was removed.)
    """

    brief_path = tmp_path / "brief.json"
    _write_fixture_brief(brief_path)

    state_root = tmp_path / "output_state"
    state_dir = state_root / "researcher" / "head-of-applied-ai-lab"
    state_dir.mkdir(parents=True)
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)

    brief = load_brief(str(brief_path))
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
        brief_path=str(brief_path),
    )
    run_id = bridge.start_or_resume_run(resume=False)

    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=_RecordedOpenAlexClient(),
        facial_llm_caller=_facial_yes,
        full_llm_caller=_full_save,
        strategy_llm_caller=_strategy_response,
    )
    stats = pipeline.run(run_id=run_id)

    # Per Spec Slice 8: ≥10 SAVE-class candidates land. Each query
    # returned 6 candidates × 2 queries = 12 candidates discovered.
    assert stats.candidates_discovered == 12
    assert stats.facial_yes == 12
    assert stats.saves == 12
    assert stats.saves >= 10  # The spec's explicit threshold

    # Per Spec Slice 8: workspace surface returns researcher cards
    # source-agnostically with display_name + save_reason + confidence
    # populated.
    workspace = aggregate_workspace(brief_id=brief.id, state_root=state_root)
    assert workspace is not None
    assert workspace.total_saves == 12
    assert "researcher" in workspace.sources
    assert workspace.brief_role_title == "Head of Applied AI Lab"

    # Every card has the wire-contract fields populated.
    for card in workspace.candidates:
        assert card.source == "researcher"
        assert card.display_name
        assert card.save_reason and "Strong first-author" in card.save_reason
        assert card.confidence == 0.92
        assert card.identity_key.startswith(
            ("orcid:", "openalex:")
        ), f"unexpected identity_key shape: {card.identity_key!r}"


def test_smoke_session_orchestrator_main_runs_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """The session_orchestrator.main() CLI entry point wires the full
    pipeline. We monkeypatch ``build_pipeline`` to inject our stubs so
    the smoke doesn't hit real OpenAlex / Opus.
    """

    from researcher import session_orchestrator as so

    brief_path = tmp_path / "brief.json"
    _write_fixture_brief(brief_path)
    state_dir = tmp_path / "state"

    # Inject stub pipeline factory so main() doesn't reach the real
    # OpenAlex client at call time.
    def _fake_build_pipeline(
        *,
        brief_path: Path,
        state_dir: Path,
        openalex_polite_pool_email: str = "",
        facial_llm_caller=None,
        full_llm_caller=None,
        strategy_llm_caller=None,
    ):
        brief = load_brief(str(brief_path))
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        bridge = ResearcherRuntimeStateBridge(
            store=store,
            output_dir=state_dir,
            brief_id=brief.id,
            brief_name=brief.role_title,
            brief_path=str(brief_path),
        )
        run_id = bridge.start_or_resume_run(resume=False)
        pipeline = ResearcherPipeline(
            brief=brief,
            bridge=bridge,
            openalex_client=_RecordedOpenAlexClient(),
            facial_llm_caller=_facial_yes,
            full_llm_caller=_full_save,
            strategy_llm_caller=_strategy_response,
        )
        return pipeline, run_id

    monkeypatch.setattr(so, "build_pipeline", _fake_build_pipeline)

    rc = so.main(
        [
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "researcher.session_orchestrator: starting" in out
    assert "researcher.session_orchestrator: done" in out
    assert "saves=12" in out


def test_smoke_full_decision_rationale_lands_at_canonical_path(
    tmp_path: Path,
) -> None:
    """Spec Opinion 6 — the wire contract: terminal_payload carries
    `full_decision` with `rationale` + `confidence` written verbatim.

    This is the most critical assertion across the module — every
    save-reason render on the workspace + candidate-detail surfaces
    reads from this path.
    """

    brief_path = tmp_path / "brief.json"
    _write_fixture_brief(brief_path)
    state_root = tmp_path / "output_state"
    state_dir = state_root / "researcher" / "head-of-applied-ai-lab"
    state_dir.mkdir(parents=True)
    db_path = state_dir / "runtime_state.sqlite3"
    store = RuntimeStateStore(db_path)

    brief = load_brief(str(brief_path))
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)
    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=_RecordedOpenAlexClient(),
        facial_llm_caller=_facial_yes,
        full_llm_caller=_full_save,
        strategy_llm_caller=_strategy_response,
    )
    pipeline.run(run_id=run_id)

    # Read the candidate row directly to inspect terminal_payload_json.
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT terminal_payload_json FROM candidates "
        "WHERE source='researcher' AND terminal_decision='SAVE'"
    ).fetchall()
    conn.close()

    assert len(rows) == 12
    for row in rows:
        payload = json.loads(row["terminal_payload_json"])
        assert "full_decision" in payload
        assert payload["full_decision"]["decision"] == "SAVE"
        assert payload["full_decision"]["confidence"] == 0.92
        assert "Strong first-author" in payload["full_decision"]["rationale"]
