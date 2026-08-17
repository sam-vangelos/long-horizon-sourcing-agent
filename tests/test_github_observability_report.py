"""P4.2/P4.3 — GitHub session report renders cost_usd/cost_per_save_usd/
run_health from final_stats.

github.orchestrator writes ``cost_usd``/``cost_per_save_usd`` (top-level,
only when the run's token-cost-log.jsonl has a usable total) and
``run_health`` (status-gated dict from shared.observability_monitors) into
``self.stats``, which lands in ``session_final.final_stats`` in the
metrics-layer JSONL. github.observability.report.generate_report never
rendered any of it. Same silence discipline as
shared.run_report_schema.render_run_report_markdown: no affirmative $0.00,
no "Run Health" section when the monitor never produced a verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

from github.observability.report import generate_report


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _generate(tmp_path: Path, final_stats: dict, bias_summary: dict | None = None) -> str:
    metrics_path = tmp_path / "metrics.jsonl"
    event = {"event": "session_final", "final_stats": final_stats, "queries_completed": 0}
    if bias_summary is not None:
        event["bias_summary"] = bias_summary
    _write_jsonl(metrics_path, [event])
    report_path = tmp_path / "report.md"
    generate_report(
        strategy_path=tmp_path / "strategy.jsonl",
        graph_path=tmp_path / "graph.json",
        candidates_path=tmp_path / "candidates.json",
        metrics_path=metrics_path,
        report_path=report_path,
        session_id="test-session",
        duration_seconds=120.0,
    )
    return report_path.read_text()


def test_report_renders_cost_when_present(tmp_path):
    markdown = _generate(
        tmp_path,
        {"saved": 2, "facial_yes": 4, "cost_usd": 3.5, "cost_per_save_usd": 1.75},
    )

    assert "$3.5000" in markdown
    assert "$1.7500" in markdown


def test_report_omits_cost_when_absent():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        markdown = _generate(Path(td), {"saved": 2, "facial_yes": 4})

    assert "cost_usd" not in markdown
    assert "$0.00" not in markdown
    assert "Cost" not in markdown


def test_report_renders_run_health_quiet_line_when_healthy(tmp_path):
    markdown = _generate(
        tmp_path,
        {
            "saved": 0,
            "facial_yes": 0,
            "run_health": {"status": "ok", "degraded": False, "degraded_reasons": []},
        },
    )

    assert "Run Health" in markdown
    assert "Degraded: No" in markdown


def test_report_renders_run_health_reasons_when_degraded(tmp_path):
    markdown = _generate(
        tmp_path,
        {
            "saved": 0,
            "facial_yes": 0,
            "run_health": {
                "status": "ok",
                "degraded": True,
                "degraded_reasons": ["green_but_useless"],
            },
        },
    )

    assert "Run Health" in markdown
    assert "Degraded: Yes" in markdown
    assert "green_but_useless" in markdown


def test_report_omits_run_health_when_no_runtime_state(tmp_path):
    markdown = _generate(
        tmp_path,
        {"saved": 0, "facial_yes": 0, "run_health": {"status": "no_runtime_state"}},
    )

    assert "Run Health" not in markdown


def test_report_renders_bias_summary_when_present(tmp_path):
    """P6.4 follow-up: orchestrator.py passes a real bias_summary to
    SessionObserver.on_session_end; the report renderer must actually
    surface it, not silently drop it on the floor."""
    markdown = _generate(
        tmp_path,
        {"saved": 1, "facial_yes": 2},
        bias_summary={
            "total_decisions": 5,
            "facial_yes_rate": 0.5,
            "save_rate": 0.25,
            "parse_failures": 1,
            "parse_failure_rate": 0.2,
            "alerts_fired": ["consecutive_saves:1"],
        },
    )

    assert "Bias Monitor" in markdown
    assert "Total decisions**: 5" in markdown
    assert "Facial YES rate**: 50.0%" in markdown
    assert "Save rate**: 25.0%" in markdown
    assert "consecutive_saves:1" in markdown


def test_report_omits_bias_summary_when_absent(tmp_path):
    markdown = _generate(tmp_path, {"saved": 0, "facial_yes": 0}, bias_summary=None)

    assert "Bias Monitor" not in markdown


def test_report_omits_bias_summary_when_monitor_saw_nothing(tmp_path):
    """A monitor that ran but recorded zero decisions must not render an
    empty section — mirrors the run_health 'no verdict' silence pattern."""
    markdown = _generate(
        tmp_path,
        {"saved": 0, "facial_yes": 0},
        bias_summary={"total_decisions": 0},
    )

    assert "Bias Monitor" not in markdown
