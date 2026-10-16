"""Pre-relaunch market-intelligence fixes (plans/market-intel-diagnosis-2026-10-06.md).

Two behaviours, pinned before the req 1238 v2 relaunch so its own MI folds are
not wasted:

1. The internal-synthesis prompt no longer asks for a ``market_thesis.reasoning``
   field filled "FIRST". Anthropic's classifier refuses that request with
   ``stop_details.category = reasoning_extraction`` (9 of 13 calls); the same
   call without the field ends normally.
2. The post-run fold (``linkedin.run_report.enrich_linkedin_run_snapshot``)
   does not replace a snapshot's reconstructed ``report_analysis`` with an
   empty ``raw_only`` packet when the run has no ``run-report.json``.
"""

from __future__ import annotations

import json

from market_intelligence import finalize_run_snapshot, resolve_market_intel_artifact_path
import market_intelligence.agent_backends as agent_backends_mod
from market_intelligence.research_prompts import build_internal_synthesis_system_prompt
from shared.output_paths import resolve_linkedin_state_dir
from shared.runtime_state import RuntimeStateStore
from shared.storage import read_json, write_json
from tests.test_market_intelligence import (
    SOURCE_BRIEF,
    _require_source_brief_path,
    _write_linkedin_inputs,
)


# -- 1. Synthesis prompt no longer requests extractable reasoning ------------


def test_internal_synthesis_prompt_does_not_request_market_thesis_reasoning():
    prompt = build_internal_synthesis_system_prompt()
    assert '"reasoning"' not in prompt
    assert "`reasoning`" not in prompt
    assert "FIRST" not in prompt


def test_internal_synthesis_prompt_keeps_market_thesis_contract():
    """Removing the field must leave every key the validator requires."""
    prompt = build_internal_synthesis_system_prompt()
    for key in ("summary", "supply_assessment", "competition_assessment", "external_context"):
        assert f'"{key}"' in prompt


def test_normalized_market_thesis_tolerates_a_stray_reasoning_key():
    """A model that still emits ``reasoning`` must not break normalization."""
    normalized = agent_backends_mod._normalize_draft_sections(
        {
            "market_thesis": {
                "reasoning": "left over from an older prompt",
                "summary": "Supply is moderate across the two productive lanes.",
                "supply_assessment": "moderate",
                "competition_assessment": "medium",
                "external_context": [],
            }
        }
    )
    thesis = normalized["market_thesis"]
    assert "reasoning" not in thesis
    assert thesis["supply_assessment"] == "moderate"


# -- 2. Post-run fold keeps reconstructed report analysis --------------------


def _finalize_reportless_run(tmp_path):
    """A finished LinkedIn run with no run-report.json, snapshotted."""
    brief_path = tmp_path / SOURCE_BRIEF.name
    write_json(brief_path, read_json(_require_source_brief_path()))
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "output" / "state" / "linkedin" / "9000000001",
    )
    _write_linkedin_inputs(
        state_dir,
        brief_path,
        version="2.1",
        generated_at="2026-04-08T12:00:00+00:00",
        include_report_input=False,
        include_report=False,
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="9000000001",
        output_dir=str(state_dir),
        mode="full_run",
        resume_state={"brief_name": "head-ai-lab"},
    )
    with store.connect() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, started_at = ?, ended_at = ? WHERE id = ?",
            (
                "governor_limit_reached",
                "2026-04-08T10:00:00+00:00",
                "2026-04-08T13:00:00+00:00",
                run_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO work_units(
                run_id, source, brief_id, kind, source_unit_id, display_name, ordering_index, status,
                payload_json, checkpoint_json, metrics_json, family_key, novelty_bucket, domain_lane,
                result_count, candidates_discovered, facial_yes_count, facial_no_count, saves_count,
                rejected_count, notes, started_at, ended_at
            )
            VALUES (?, 'linkedin', ?, 'linkedin_string', '2', 'Research copilot lane', 1, 'done',
                    ?, ?, ?, 'research_copilot_asset_mgmt', 'edge_case', 'asset_management',
                    526, 22, 10, 12, 8, 14, 'Strong lane', ?, ?)
            """,
            (
                run_id,
                "9000000001",
                json.dumps({"boolean": "research copilot lane"}),
                json.dumps({"pages_reviewed": 4, "duplicates_count": 1}),
                json.dumps({"pages_reviewed": 4, "duplicates_count": 1}),
                "2026-04-08T10:00:00+00:00",
                "2026-04-08T12:00:00+00:00",
            ),
        )
    run_dir = finalize_run_snapshot(
        source="linkedin",
        brief_path=brief_path,
        state_dir=state_dir,
        run_id=run_id,
    )
    return brief_path, run_dir, run_id


def test_post_run_fold_keeps_reconstructed_report_analysis(tmp_path, monkeypatch):
    from linkedin import run_report

    # No network: every LLM stage takes its heuristic path.
    monkeypatch.setattr(agent_backends_mod, "_has_llm_access", lambda: False)

    brief_path, run_dir, run_id = _finalize_reportless_run(tmp_path)
    assert not (run_dir / "run-report.json").exists()
    packet_path = run_dir / "market-intel-research-input.json"
    snapshot_packet = read_json(packet_path)
    assert snapshot_packet["context_metadata"]["context_quality"] == "reconstructed_report"
    assert snapshot_packet["report_analysis"]["coverage_gaps"] or snapshot_packet[
        "report_analysis"
    ]["winning_lanes"]

    log_path = tmp_path / "run.log.jsonl"
    run_report.enrich_linkedin_run_snapshot(
        runtime_run_id=run_id,
        brief_path=str(brief_path),
        run_dir=run_dir,
        log_path=log_path,
        with_external_research=False,
    )

    events = [json.loads(line)["event"] for line in log_path.read_text().splitlines() if line]
    assert "market_intel_updated" in events, events

    folded_packet = read_json(packet_path)
    assert folded_packet["context_metadata"]["context_quality"] == "reconstructed_report"
    assert folded_packet["context_metadata"]["analysis_provenance"] == "reconstructed_from_raw"
    assert folded_packet["report_analysis"]["winning_lanes"], (
        "post-run fold replaced the reconstructed report_analysis with an empty packet"
    )

    artifact = read_json(resolve_market_intel_artifact_path(brief_path, output_dir=run_dir))
    run_records = artifact["evidence_index"]["runs"]
    assert run_records
    assert run_records[0]["context_quality"] == "reconstructed_report"
