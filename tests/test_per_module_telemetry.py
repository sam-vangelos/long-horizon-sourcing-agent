"""Per-module telemetry — Move #6.

Asserts that each non-LinkedIn module's orchestrator emits the canonical
``log_event`` envelope (``pipeline_start`` + ``pipeline_end`` at minimum;
``string_complete`` for per-string success; ``string_error`` for per-string
failure) so cross-module observability is no longer LinkedIn-only.

Mirrors :mod:`linkedin.orchestrator`'s namespace and payload shape — see
``linkedin/orchestrator.py:717-1441`` for the canonical pattern. The event
names used here must all live in
:data:`shared.contracts.RUN_LOG_EVENTS` (the frozen contract); these tests
are the equivalent of ``tests/test_phase0_contracts.py`` 's vocabulary
guard, scoped to the new emitters.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researcher.orchestrator import ResearcherPipeline
from shared.contracts import RUN_LOG_EVENTS
from shared.runtime_state.researcher import ResearcherRuntimeStateBridge
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_events(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    events: list[dict] = []
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# Researcher
# ---------------------------------------------------------------------------


class _StubOpenAlexClient:
    def __init__(self, responses_by_concept: dict[str, list[dict]]) -> None:
        self.responses = responses_by_concept

    def search_authors(self, **kwargs: Any) -> dict:
        concepts = kwargs.get("concept_ids") or []
        key = ",".join(concepts) if concepts else ""
        return {"meta": {"next_cursor": ""}, "results": self.responses.get(key, [])}


def _author_payload(*, author_id: str, name: str, h_index: int = 12) -> dict:
    return {
        "id": f"https://openalex.org/{author_id}",
        "display_name": name,
        "summary_stats": {"h_index": h_index},
        "cited_by_count": 200,
        "works_count": 25,
        "counts_by_year": [
            {"year": 2024, "works_count": 4},
            {"year": 2023, "works_count": 4},
        ],
        "last_known_institutions": [{"display_name": "MIT", "country_code": "US"}],
        "x_concepts": [{"id": "https://openalex.org/C1"}],
    }


def _researcher_brief() -> SimpleNamespace:
    return SimpleNamespace(
        id="researcher-telemetry-test",
        role_title="Researcher",
        capability_areas=[
            SimpleNamespace(name="Post-training", description="RLHF/DPO/SFT.")
        ],
        depth_distinction=SimpleNamespace(
            builder_definition="First-author publications.",
            user_definition="Cites without publishing.",
            edge_case_guidance="Borderline = full eval.",
        ),
        _new_brief={"source_config": {"researcher": {"discipline": "ml_general"}}},
    )


def _researcher_strategy() -> dict:
    return {
        "generated_strings": [
            {
                "id": 1,
                "name": "q1",
                "topic_concepts": ["C1"],
                "ror_country_filter": ["US"],
            },
        ],
    }


def _facial_yes(_s: str, _u: str) -> dict:
    return {
        "decision": "FACIAL_YES",
        "rationale": "Recent on-thesis work.",
        "confidence": 0.9,
    }


def _full_save(_s: str, _u: str) -> dict:
    return {
        "decision": "SAVE",
        "path": "first_author_at_canonical_venue",
        "confidence": 0.92,
        "rationale": "Strong first-author publications.",
    }


def test_researcher_pipeline_emits_canonical_envelope(tmp_path: Path) -> None:
    """ResearcherPipeline.run() must emit ``pipeline_start`` /
    ``string_complete`` / ``pipeline_end`` to a run_log.jsonl colocated
    with the runtime-state store, mirroring LinkedIn's pattern.
    """

    state_dir = tmp_path / "researcher" / "key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _researcher_brief()
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)
    client = _StubOpenAlexClient(
        {"C1": [_author_payload(author_id="A1", name="Alice")]}
    )
    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=client,
        facial_llm_caller=_facial_yes,
        full_llm_caller=_full_save,
        strategy_llm_caller=lambda _s, _u: _researcher_strategy(),
    )
    pipeline.run(run_id=run_id)

    events = _read_events(state_dir / "run_log.jsonl")
    names = [e["event"] for e in events]

    assert names[0] == "pipeline_start"
    assert events[0]["mode"] == "full"
    assert "string_complete" in names
    assert names[-1] == "pipeline_end"

    string_complete = next(e for e in events if e["event"] == "string_complete")
    assert string_complete["string_id"] == 1
    assert string_complete["candidates_discovered"] == 1
    assert string_complete["saves_count"] == 1

    pipeline_end = events[-1]
    assert pipeline_end["queries_total"] == 1
    assert pipeline_end["queries_completed"] == 1
    assert pipeline_end["saves"] == 1

    for event in events:
        assert event["event"] in RUN_LOG_EVENTS, (
            f"researcher emitted unknown event {event['event']!r}; "
            "extend shared.contracts.RUN_LOG_EVENTS first"
        )


def test_researcher_pipeline_emits_string_error_on_query_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-query exception must surface as ``string_error`` with the
    exception class name + an elapsed_ms field, then re-raise so the
    pipeline aborts (behavior preservation)."""

    state_dir = tmp_path / "researcher" / "key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _researcher_brief()
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)

    class _BoomClient:
        def search_authors(self, **_kwargs: Any) -> dict:
            raise RuntimeError("openalex_unreachable")

    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=_BoomClient(),
        facial_llm_caller=_facial_yes,
        full_llm_caller=_full_save,
        strategy_llm_caller=lambda _s, _u: _researcher_strategy(),
    )

    with pytest.raises(RuntimeError, match="openalex_unreachable"):
        pipeline.run(run_id=run_id)

    events = _read_events(state_dir / "run_log.jsonl")
    names = [e["event"] for e in events]
    assert "pipeline_start" in names
    assert "string_error" in names
    assert "pipeline_error" in names
    assert names[-1] == "pipeline_end"

    string_error = next(e for e in events if e["event"] == "string_error")
    assert string_error["error_class"] == "RuntimeError"
    assert "openalex_unreachable" in string_error["error"]
    assert "elapsed_ms" in string_error


# ---------------------------------------------------------------------------
# Designer
# ---------------------------------------------------------------------------


def test_designer_session_orchestrator_emits_pipeline_envelope(
    tmp_path: Path,
) -> None:
    """Designer emits the canonical run envelope plus its run-end hook event."""

    from designer.session_orchestrator import main

    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps({"id": "designer-stub"}))
    state_dir = tmp_path / "state"

    rc = main(
        [
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
        ]
    )
    assert rc == 0

    events = _read_events(state_dir / "run_log.jsonl")
    names = [e["event"] for e in events]
    assert names[0] == "pipeline_start"
    assert events[0]["mode"] == "full"
    assert "pipeline_end" in names
    assert names[-1] == "designer_run_end"
    assert events[-1]["status"] == "ok"
    for event in events:
        assert event["event"] in RUN_LOG_EVENTS


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


def test_github_orchestrator_string_complete_event_is_in_run_log_contract() -> None:
    """The GitHub orchestrator emits ``string_complete`` and
    ``string_error`` per query (Move #6). End-to-end coverage lives in
    the GitHub pipeline test suite; this guard asserts the event names
    we added are part of the frozen RUN_LOG_EVENTS contract so they
    won't drift.
    """

    assert "string_complete" in RUN_LOG_EVENTS
    assert "string_error" in RUN_LOG_EVENTS
    assert "pipeline_start" in RUN_LOG_EVENTS
    assert "pipeline_end" in RUN_LOG_EVENTS
    assert "pipeline_error" in RUN_LOG_EVENTS
