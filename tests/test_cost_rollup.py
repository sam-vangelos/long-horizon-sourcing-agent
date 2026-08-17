"""Tests for shared.cost_rollup — audit Move #10.

Asserts:

- aggregate_cost_for_run reads the last cumulative cost_usd from each
  module's pipeline_end events in run_log.jsonl (each pipeline_end
  re-reads the whole token-cost log, so summing would double-count).
- Designer additionally surfaces a cost_telemetry.json artifact;
  when run_log.jsonl is empty the aggregator falls back to it.
- Modules whose state-dir is missing or yields no signal land in the
  rollup's `missing` list (not silently dropped).
- The sidecar JSON shape is stable so downstream readers (run-summary
  surface) can rely on it.
- Reading per-module pipeline_end events tolerates malformed JSON
  lines without raising.
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.cost_rollup import (
    CostRollup,
    ModuleCost,
    _sum_token_cost_log_usd,
    aggregate_cost_for_run,
    write_cost_rollup_sidecar,
)


def _write_run_log(state_dir: Path, *events: dict) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "run_log.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def test_rollup_sums_cost_usd_from_pipeline_end_events(tmp_path: Path) -> None:
    linkedin_dir = tmp_path / "linkedin"
    github_dir = tmp_path / "github"
    _write_run_log(
        linkedin_dir,
        {"event": "pipeline_start", "mode": "full"},
        {"event": "pipeline_end", "cost_usd": 3.5},
    )
    _write_run_log(
        github_dir,
        {"event": "pipeline_start", "mode": "autonomous"},
        {"event": "pipeline_end", "cost_usd": 1.25},
    )

    rollup = aggregate_cost_for_run(
        {"linkedin": linkedin_dir, "github": github_dir}
    )
    assert rollup.total_usd == 4.75
    assert [mc.module for mc in rollup.by_module] == ["linkedin", "github"]
    assert rollup.by_module[0].cost_usd == 3.5
    assert rollup.by_module[1].cost_usd == 1.25
    assert rollup.missing == []


def test_rollup_treats_missing_state_dir_as_missing(tmp_path: Path) -> None:
    linkedin_dir = tmp_path / "linkedin"
    _write_run_log(
        linkedin_dir,
        {"event": "pipeline_end", "cost_usd": 2.0},
    )
    rollup = aggregate_cost_for_run(
        {"linkedin": linkedin_dir, "github": tmp_path / "no-such-dir"}
    )
    assert rollup.total_usd == 2.0
    assert rollup.missing == ["github"]
    assert len(rollup.by_module) == 1


def test_rollup_treats_state_dir_without_cost_signal_as_missing(
    tmp_path: Path,
) -> None:
    """A state-dir that exists but has no cost_usd in pipeline_end (and
    no designer cost_telemetry.json) lands in the missing list. The
    rollup distinguishes "module ran but didn't surface cost" from
    "module ran and reported $0" — only the run-log-with-cost path
    counts."""

    researcher_dir = tmp_path / "researcher"
    _write_run_log(
        researcher_dir,
        {"event": "pipeline_start", "mode": "full"},
        # pipeline_end without cost_usd ⇒ no cost signal surfaced
        {"event": "pipeline_end", "queries_total": 5},
    )
    rollup = aggregate_cost_for_run({"researcher": researcher_dir})
    assert rollup.total_usd == 0.0
    assert rollup.missing == ["researcher"]


def test_rollup_handles_zero_cost_signal_as_present_not_missing(
    tmp_path: Path,
) -> None:
    """A module that explicitly publishes cost_usd=0.0 (e.g., the
    designer Slice-1 stub) is "present, ran, cost zero" — not missing."""

    designer_dir = tmp_path / "designer"
    _write_run_log(
        designer_dir,
        {"event": "pipeline_end", "status": "ok", "cost_usd": 0.0},
    )
    rollup = aggregate_cost_for_run({"designer": designer_dir})
    assert rollup.total_usd == 0.0
    assert len(rollup.by_module) == 1
    assert rollup.by_module[0].cost_usd == 0.0
    assert rollup.missing == []


def test_rollup_falls_back_to_designer_cost_telemetry(tmp_path: Path) -> None:
    """When designer's run_log doesn't carry an aggregated cost, the
    aggregator falls back to the cost_telemetry.json artifact."""

    designer_dir = tmp_path / "designer"
    designer_dir.mkdir(parents=True)
    # No pipeline_end with cost_usd in run_log.
    _write_run_log(designer_dir, {"event": "pipeline_start", "mode": "full"})
    (designer_dir / "cost_telemetry.json").write_text(
        json.dumps(
            {
                "primary_pass_usd": 1.20,
                "cross_check_usd": 0.30,
                "total_usd": 1.50,
            }
        )
    )

    rollup = aggregate_cost_for_run({"designer": designer_dir})
    assert rollup.total_usd == 1.5
    assert rollup.by_module[0].cost_usd == 1.5
    assert "cost_telemetry.json.total_usd" in rollup.by_module[0].sources


def test_rollup_handles_designer_cost_telemetry_with_partial_fields(
    tmp_path: Path,
) -> None:
    """Designer cost_telemetry without total_usd but with primary +
    cross-check gets summed by the aggregator."""

    designer_dir = tmp_path / "designer"
    designer_dir.mkdir(parents=True)
    (designer_dir / "cost_telemetry.json").write_text(
        json.dumps(
            {"primary_pass_usd": 0.75, "cross_check_usd": 0.25}
        )
    )

    rollup = aggregate_cost_for_run({"designer": designer_dir})
    assert rollup.total_usd == 1.0


def test_rollup_uses_last_cumulative_pipeline_end_cost(tmp_path: Path) -> None:
    """Resume/retry runs emit one pipeline_end per session; each carries
    the cumulative whole-log total (not an incremental slice). The rollup
    takes the last numeric cost_usd, never sums across sessions."""

    linkedin_dir = tmp_path / "linkedin"
    _write_run_log(
        linkedin_dir,
        {"event": "pipeline_start", "mode": "full"},
        {"event": "pipeline_end", "cost_usd": 1.0},
        {"event": "pipeline_start", "mode": "full_run_resume"},
        {"event": "pipeline_end", "cost_usd": 0.5},
    )
    rollup = aggregate_cost_for_run({"linkedin": linkedin_dir})
    assert rollup.total_usd == 0.5


def test_pipeline_end_cost_takes_last_cumulative_not_sum(tmp_path: Path) -> None:
    """Two sessions: session 1 spent $1.00; session 2 brought cumulative
    total to $3.00. Summing [1.0, 3.0] would yield $4.00; last wins at $3.00."""

    linkedin_dir = tmp_path / "linkedin"
    _write_run_log(
        linkedin_dir,
        {"event": "pipeline_end", "cost_usd": 1.0},
        {"event": "pipeline_end", "cost_usd": 3.0},
    )
    rollup = aggregate_cost_for_run({"linkedin": linkedin_dir})
    assert rollup.total_usd == 3.0
    assert rollup.total_usd != 4.0


def test_pipeline_end_cost_skips_rows_without_cost_then_uses_last(
    tmp_path: Path,
) -> None:
    """Rows lacking cost_usd are skipped; the last row with a numeric
    cost_usd wins. When every pipeline_end omits cost_usd the module
    lands in missing — never a fabricated $0."""

    linkedin_dir = tmp_path / "linkedin"
    _write_run_log(
        linkedin_dir,
        {"event": "pipeline_end", "cost_usd": 1.0},
        {"event": "pipeline_end", "stats": {"saved": 2}},
        {"event": "pipeline_end", "cost_usd": 2.5},
    )
    rollup = aggregate_cost_for_run({"linkedin": linkedin_dir})
    assert rollup.total_usd == 2.5
    assert rollup.missing == []

    no_cost_dir = tmp_path / "no-cost"
    _write_run_log(
        no_cost_dir,
        {"event": "pipeline_end", "stats": {"saved": 1}},
        {"event": "pipeline_end", "queries_total": 5},
    )
    rollup_none = aggregate_cost_for_run({"linkedin": no_cost_dir})
    assert rollup_none.total_usd == 0.0
    assert rollup_none.missing == ["linkedin"]


def test_rollup_tolerates_malformed_run_log_lines(tmp_path: Path) -> None:
    state_dir = tmp_path / "linkedin"
    state_dir.mkdir(parents=True)
    log_path = state_dir / "run_log.jsonl"
    log_path.write_text(
        '{"event": "pipeline_start"}\n'
        "not-json-at-all\n"
        '{"event": "pipeline_end", "cost_usd": 2.5}\n'
    )
    rollup = aggregate_cost_for_run({"linkedin": state_dir})
    assert rollup.total_usd == 2.5


def test_sidecar_round_trips(tmp_path: Path) -> None:
    """The CostRollup.to_dict shape persists correctly to JSON and is
    stable for the run-summary surface to read."""

    rollup = CostRollup(
        total_usd=4.75,
        by_module=[
            ModuleCost(
                module="linkedin",
                cost_usd=3.5,
                sources=["run_log.pipeline_end.cost_usd"],
            ),
            ModuleCost(
                module="github",
                cost_usd=1.25,
                sources=["run_log.pipeline_end.cost_usd"],
            ),
        ],
        missing=["researcher"],
    )
    path = write_cost_rollup_sidecar(rollup, run_dir=tmp_path)
    assert path.exists()

    parsed = json.loads(path.read_text())
    assert parsed["total_usd"] == 4.75
    assert len(parsed["by_module"]) == 2
    assert parsed["by_module"][0]["module"] == "linkedin"
    assert parsed["by_module"][0]["cost_usd"] == 3.5
    assert parsed["by_module"][0]["sources"] == [
        "run_log.pipeline_end.cost_usd"
    ]
    assert parsed["missing"] == ["researcher"]


def test_aggregator_preserves_state_dirs_insertion_order(tmp_path: Path) -> None:
    """The by_module list mirrors the state_dirs dict insertion order
    so the run-summary surface renders modules in a stable sequence."""

    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    for d in (a, b, c):
        _write_run_log(d, {"event": "pipeline_end", "cost_usd": 1.0})

    rollup = aggregate_cost_for_run({"a": a, "b": b, "c": c})
    assert [mc.module for mc in rollup.by_module] == ["a", "b", "c"]


def test_shared_helpers_are_the_single_implementation() -> None:
    """A4: orchestrators bind the shared helpers — no local redefinitions."""

    import github.orchestrator as github_orchestrator
    import linkedin.orchestrator as linkedin_orchestrator
    import shared.cost_rollup as cost_rollup

    assert (
        linkedin_orchestrator._sum_token_cost_log_usd
        is cost_rollup._sum_token_cost_log_usd
    )
    assert (
        github_orchestrator._sum_token_cost_log_usd
        is cost_rollup._sum_token_cost_log_usd
    )
    assert (
        linkedin_orchestrator._cost_per_save_usd
        is cost_rollup._cost_per_save_usd
    )
    assert (
        github_orchestrator._cost_per_save_usd
        is cost_rollup._cost_per_save_usd
    )


def test_per_module_orchestrators_emit_cost_usd_in_pipeline_end(
    tmp_path: Path,
) -> None:
    """End-to-end smoke: the pipeline_end events the per-module
    orchestrators write per audit Move #6 + Move #10 carry the
    ``cost_usd`` field the aggregator reads. Designer's stub
    publishes 0.0; the contract is the field's presence + parse-
    ability, not the dollar value."""

    from designer.session_orchestrator import main as designer_main

    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps({"id": "designer-stub"}))
    state_dir = tmp_path / "designer-state"

    rc = designer_main(
        [
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
        ]
    )
    assert rc == 0

    rollup = aggregate_cost_for_run({"designer": state_dir})
    assert "designer" in [mc.module for mc in rollup.by_module]
    assert rollup.by_module[0].cost_usd == 0.0
    assert rollup.by_module[0].sources == [
        "run_log.pipeline_end.cost_usd"
    ]


def test_github_primary_cost_excludes_shadow_rows(tmp_path: Path) -> None:
    """GitHub primary cost must exclude shadow_stage rows, matching LinkedIn."""

    log_path = tmp_path / "token-cost-log.jsonl"
    rows = [
        {"provider": "anthropic", "estimated_cost_usd": 1.0},
        {"provider": "fireworks", "estimated_cost_usd": 2.0},
        {
            "provider": "fireworks",
            "shadow_stage": "facial_shadow",
            "estimated_cost_usd": 4.0,
        },
    ]
    with log_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    assert _sum_token_cost_log_usd(
        log_path,
        exclude_rows_with=("shadow_stage",),
    ) == 3.0
