"""Phase 1 seam-contract tests — runtime_cost cluster.

Pins the producer->CONSUMER edge of three cost/decision seams so a
refactor cannot silently sever the wiring, and marks the genuinely
inert cost-rollup seam as a strict-xfail so the board stops claiming
the usage JSONL feeds the cross-module rollup.

Seams covered:

* 2.0 (pin)  — ``LinkedInRuntimeStateBridge.sync_progress`` reads the
  run's ``token-cost-log.jsonl`` via ``lane_cost_from_usage_log`` and
  attributes per-lane cost into ``work_units.metrics_json['cost_usd']``;
  ``lane_metrics_for_run`` sums that into ``LaneMetricsRow.cost_usd``.
  Existing ``test_lane_metrics.py::test_sync_writes_usage_log_cost_into_work_unit_metrics``
  pins only the inner ``sync_linkedin_progress`` with ``lane_cost_usd``
  handed in — it bypasses the bridge's file read. This drives the bridge.
* 2.1 (pin)  — ``write_linkedin_stage_projections`` materializes
  ``final_judgments.jsonl`` from ``candidate_attempts.payload_json``
  ``['full_decision']``; ``linkedin.run_report.load_run_report_decisions``
  reads that file back into ``candidate_name`` / ``decision`` rows.
* 2.2 (xfail) — ``record_llm_usage`` writes per-call ``estimated_cost_usd``
  rows to the usage JSONL, but ``aggregate_cost_for_run`` reads only
  ``run_log.jsonl`` (``pipeline_end.cost_usd``) + designer telemetry,
  never the usage JSONL. The usage JSONL never reaches the cross-module
  rollup, and ``write_cost_rollup_sidecar`` has zero production callers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from shared.cost_rollup import (
    aggregate_cost_for_run,
    write_cost_rollup_sidecar,
)
from shared.llm_usage import llm_usage_session, record_llm_usage
from shared.runtime_state.lane_metrics import lane_metrics_for_run
from shared.runtime_state.linkedin import LinkedInRuntimeStateBridge
from shared.runtime_state.linkedin_progress_sync import lane_cost_from_usage_log
from shared.runtime_state.projections import write_linkedin_stage_projections
from shared.runtime_state.store import RuntimeStateStore
from shared.schemas import Progress, SearchString
from linkedin.run_report import load_run_report_decisions


# ---------------------------------------------------------------------------
# Shared seeding helpers — mirror tests/test_lane_metrics.py.
# ---------------------------------------------------------------------------


def _start_run(store: RuntimeStateStore, *, brief_id: str) -> int:
    return store.start_run(
        source="linkedin",
        brief_id=brief_id,
        output_dir=str(store.db_path.parent),
        mode="fresh",
        resume_state={"brief_name": "test"},
    )


def _write_usage_log(
    log_path: Path,
    *,
    brief_id: str,
    rows: list[tuple[str, int, int]],
) -> None:
    """Write a ``token-cost-log.jsonl`` exactly the way the orchestrator's
    judge calls do — ``llm_usage_session`` opens the sink, each
    ``record_llm_usage`` appends one row carrying ``estimated_cost_usd``
    and the ``lane_id`` from ``usage_context`` (shared/llm_usage.py:214-227).
    ``rows`` is ``(lane_id, input_tokens, output_tokens)``.
    """

    with llm_usage_session(log_path, module="linkedin", brief_id=brief_id):
        for lane_id, input_tokens, output_tokens in rows:
            record_llm_usage(
                provider="anthropic",
                model="claude-opus",
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
                usage_context={"lane_id": lane_id, "stage": "full_eval"},
            )


# ===========================================================================
# Seam 2.0 — bridge file-read -> sync_linkedin_progress -> metrics_json
#            -> lane_metrics_for_run.cost_usd  (PIN)
# ===========================================================================


def test_seam_bridge_usage_log_cost_reaches_lane_metrics(tmp_path: Path):
    """PIN: the BRIDGE (not the inner sync helper) reads the run's
    ``token-cost-log.jsonl`` and attributes per-lane cost so the read
    model surfaces it.

    Producer entry is ``bridge.sync_progress(run_id, progress)`` — the
    bridge reads ``self.output_dir/'token-cost-log.jsonl'`` at
    shared/runtime_state/linkedin.py:127, rolls it up with
    ``lane_cost_from_usage_log``, and forwards ``lane_cost_usd=`` into
    ``sync_linkedin_progress`` (linkedin.py:138). The inner helper then
    writes ``metrics['cost_usd']`` for one work unit per lane
    (linkedin_progress_sync.py:99-101). Consumer is
    ``lane_metrics_for_run`` summing ``metrics_json`` cost via
    ``_coerce_cost`` (lane_metrics.py:344-347) into
    ``LaneMetricsRow.cost_usd``.

    The existing ``test_lane_metrics.py:684`` pins the inner
    ``sync_linkedin_progress`` with ``lane_cost_usd`` handed in directly,
    bypassing the bridge's file read — this test exercises the bridge edge.
    """

    brief_id = "brief-bridge-cost-seam"
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")

    # The bridge reads exactly this filename relative to output_dir.
    usage_log = tmp_path / "token-cost-log.jsonl"
    _write_usage_log(
        usage_log,
        brief_id=brief_id,
        rows=[
            ("ml-infra", 1000, 500),   # 0.0525
            ("ml-infra", 2000, 800),   # 0.09   -> ml-infra total 0.1425
            ("platform", 500, 100),    # 0.015
        ],
    )

    # Sanity on the producer-side roll-up the bridge will compute.
    lane_cost = lane_cost_from_usage_log(usage_log)
    assert lane_cost["ml-infra"] > lane_cost["platform"] > 0
    expected_ml = lane_cost["ml-infra"]
    expected_platform = lane_cost["platform"]

    bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id=brief_id,
        brief_name="test",
    )
    run_id = _start_run(store, brief_id=brief_id)

    progress = Progress(
        brief_name="test",
        strings=[
            SearchString(id=1, name="s1", boolean="a", lane_id="ml-infra", lane_name="ML Infra"),
            SearchString(id=2, name="s2", boolean="b", lane_id="ml-infra", lane_name="ML Infra"),
            SearchString(id=3, name="s3", boolean="c", lane_id="platform", lane_name="Platform"),
        ],
    )

    # Drive the bridge — NOT sync_linkedin_progress directly. The bridge's
    # own file read is the seam under test.
    bridge.sync_progress(run_id, progress)

    rows = lane_metrics_for_run(store.db_path, run_id=run_id)
    cost_by_lane = {r.lane_id: r.cost_usd for r in rows}

    assert cost_by_lane["ml-infra"] is not None
    assert cost_by_lane["platform"] is not None
    assert cost_by_lane["ml-infra"] == pytest.approx(expected_ml)
    assert cost_by_lane["platform"] == pytest.approx(expected_platform)
    assert cost_by_lane["ml-infra"] > cost_by_lane["platform"]
    # Two strings share ml-infra; the lane total must not be doubled.
    assert cost_by_lane["ml-infra"] < expected_ml * 2


# ===========================================================================
# Seam 2.1 — write_linkedin_stage_projections -> final_judgments.jsonl
#            -> load_run_report_decisions  (PIN)
# ===========================================================================


def test_seam_full_decision_projection_reaches_run_report(tmp_path: Path):
    """PIN: a full-stage decision written into
    ``candidate_attempts.payload_json['full_decision']`` survives the
    projection writer and is read back by the run-report consumer.

    Producer: ``write_linkedin_stage_projections`` -> ``_project_attempt_payloads``
    selects ``stage='full'`` ``payload_json`` and extracts ``full_decision``
    (projections.py:401-429), materializing ``final_judgments.jsonl``
    (:356-367). Consumer: ``load_run_report_decisions`` reads that file ->
    ``candidate_name`` / ``decision`` / ``path`` / ``confidence`` /
    ``rationale`` (linkedin/run_report.py:143-168), feeding the
    saved/rejected debrief summaries.

    ``finish_attempt_success(payload=...)`` is what lands in
    ``candidate_attempts.payload_json`` (store.py:1414/1417), so the
    full-stage attempt carries the ``full_decision`` envelope there.
    """

    brief_id = "brief-final-judgment-seam"
    candidate_name = "Ada Lovelace"
    profile_url = "https://linkedin.com/in/ada-lovelace"

    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    run_id = _start_run(store, brief_id=brief_id)

    work_unit_id = store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        kind="linkedin_string",
        source_unit_id="1",
        display_name="builders",
        ordering_index=1,
        status="done",
        payload={"id": 1, "name": "builders", "boolean": "ml"},
        checkpoint={},
        metrics={},
        counters={},
    )

    # Walk the candidate through the real lifecycle to a succeeded full
    # attempt, mirroring tests/test_lane_metrics.py::_seed_candidate_terminal,
    # but carry the full_decision envelope in the FULL attempt's payload so
    # the projection has something to extract.
    store.record_candidate_discovery(
        run_id=run_id,
        work_unit_id=work_unit_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=profile_url,
        display_name=candidate_name,
        profile_url=profile_url,
    )
    for state in ("snippet_extracted", "facial_started"):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id=brief_id,
            identity_key=profile_url,
            new_state=state,
            last_work_unit_id=work_unit_id,
        )
    facial_attempt = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=profile_url,
        stage="facial",
        work_unit_id=work_unit_id,
    )
    store.finish_attempt_success(
        attempt_id=facial_attempt,
        new_state="facial_terminal",
        terminal_decision=None,
        payload={"facial_decision": {"decision": "FACIAL_YES"}},
        run_id=run_id,
    )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=profile_url,
        new_state="full_started",
        last_work_unit_id=work_unit_id,
    )
    full_attempt = store.start_attempt(
        run_id=run_id,
        source="linkedin",
        brief_id=brief_id,
        identity_key=profile_url,
        stage="full",
        work_unit_id=work_unit_id,
    )
    full_decision = {
        "candidate_name": candidate_name,
        "decision": "SAVE",
        "path": profile_url,
        "confidence": 0.9,
        "rationale": "Strong infra background; ships.",
    }
    store.finish_attempt_success(
        attempt_id=full_attempt,
        new_state="full_terminal",
        terminal_decision="SAVE",
        # payload= is what write_linkedin_stage_projections reads back.
        payload={"full_decision": full_decision},
        terminal_payload={"full_decision": full_decision},
        run_id=run_id,
    )

    # Producer: writes output_dir/final_judgments.jsonl.
    write_linkedin_stage_projections(
        store,
        brief_id=brief_id,
        output_dir=tmp_path,
        run_id=run_id,
    )
    final_path = tmp_path / "final_judgments.jsonl"
    assert final_path.exists()

    # Consumer: reads it back and filters to SAVE.
    decisions = load_run_report_decisions(final_path, {"SAVE"})
    assert len(decisions) == 1
    assert decisions[0]["candidate_name"] == candidate_name
    assert decisions[0]["decision"] == "SAVE"
    assert decisions[0]["path"] == profile_url
    assert decisions[0]["confidence"] == pytest.approx(0.9)


# ===========================================================================
# Seam 2.2 — record_llm_usage usage JSONL -> aggregate_cost_for_run
# ===========================================================================


def test_seam_usage_log_cost_reaches_cross_module_rollup(tmp_path: Path):
    """CONTRACT: a state-dir whose only cost signal is the usage
    JSONL should contribute its summed ``estimated_cost_usd`` to the
    cross-module ``CostRollup.total_usd``.

    Producer: ``record_llm_usage`` appends per-call rows with
    ``estimated_cost_usd`` to ``token-cost-log.jsonl`` (shared/llm_usage.py:214-227).
    Consumer: ``aggregate_cost_for_run`` falls back to summing the usage
  log when no ``pipeline_end.cost_usd`` is present, excluding shadow_stage
  rows so primary spend is not contaminated by shadow evaluation.
    """

    state_dir = tmp_path / "linkedin"
    state_dir.mkdir()
    usage_log = state_dir / "token-cost-log.jsonl"
    _write_usage_log(
        usage_log,
        brief_id="brief-rollup-seam",
        rows=[
            ("ml-infra", 1000, 500),   # 0.0525
            ("ml-infra", 2000, 800),   # 0.09
            ("platform", 500, 100),    # 0.015  -> total 0.1575
        ],
    )

    # Producer-side ground truth: the usage JSONL really carries cost
    # (positive, non-zero). No brittle literal here — the only line that
    # may fail is the seam assertion below.
    expected_total = sum(lane_cost_from_usage_log(usage_log).values())
    assert expected_total > 0

    rollup = aggregate_cost_for_run({"linkedin": state_dir})

    # Seam assertion: the rollup reflects the usage-JSONL cost.
    assert rollup.total_usd == pytest.approx(expected_total)


def test_seam_write_cost_rollup_sidecar_has_production_caller():
    """CONTRACT: some finalize/worker path under the production
    package tree should invoke ``write_cost_rollup_sidecar`` so the
    rollup gets persisted next to a run's artifacts.

    ``write_cost_rollup_sidecar`` is imported above (so a regression fails
    at the seam assertion, not at import). It is real and works in
    isolation — proven here — AND both orchestrator finalize paths call it.

    Wired by A4+O5 (Wave 5): the AST scan of the production package tree
    finds the call sites in linkedin/orchestrator.py and
    github/orchestrator.py. Before that wiring this was a strict xfail —
    the sidecar existed but no run ever persisted it.
    """

    # Prove the producer is callable in isolation (not the seam — just
    # guards against the symbol disappearing and faking the xfail).
    sidecar_path = write_cost_rollup_sidecar(
        aggregate_cost_for_run({}),
        run_dir=Path("/tmp"),
        filename="cost_rollup_seam_probe.json",
    )
    assert sidecar_path.name == "cost_rollup_seam_probe.json"

    repo_root = Path(__file__).resolve().parent.parent
    production_dirs = [
        repo_root / "linkedin",
        repo_root / "shared",
        repo_root / "market_intelligence",
        repo_root / "cloris",
        repo_root / "github",
    ]
    callers: list[str] = []
    for base in production_dirs:
        if not base.exists():
            continue
        for py_file in base.rglob("*.py"):
            # The definition + __all__ export live in cost_rollup.py itself;
            # that is not a caller.
            if py_file.resolve() == (repo_root / "shared" / "cost_rollup.py").resolve():
                continue
            try:
                tree = ast.parse(py_file.read_text())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = (
                        func.id if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute)
                        else None
                    )
                    if name == "write_cost_rollup_sidecar":
                        callers.append(str(py_file.relative_to(repo_root)))

    # Seam assertion: both orchestrator finalize paths persist the sidecar.
    assert callers, "write_cost_rollup_sidecar has no production call site"
    caller_set = set(callers)
    assert "linkedin/orchestrator.py" in caller_set
    assert "github/orchestrator.py" in caller_set
