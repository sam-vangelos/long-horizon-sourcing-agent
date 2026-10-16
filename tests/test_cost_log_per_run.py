"""Per-run scoping for token-cost-log.jsonl rows and cost sums."""

from __future__ import annotations

import json
from pathlib import Path

from shared.llm_usage import llm_usage_session, record_llm_usage
from shared.storage import append_jsonl


def _write_rows(path: Path, rows: list[dict]) -> None:
    for row in rows:
        append_jsonl(path, row)


def test_cost_log_scopes_to_single_run(tmp_path: Path) -> None:
    from github.orchestrator import _sum_token_cost_log_usd

    log_path = tmp_path / "token-cost-log.jsonl"
    _write_rows(
        log_path,
        [
            {"provider": "anthropic", "estimated_cost_usd": 0.10, "run_id": 101},
            {"provider": "anthropic", "estimated_cost_usd": 0.20, "run_id": 202},
            {"provider": "anthropic", "estimated_cost_usd": 0.05},
        ],
    )

    assert _sum_token_cost_log_usd(log_path, run_id=101) == 0.10
    assert _sum_token_cost_log_usd(log_path, run_id=202) == 0.20
    assert _sum_token_cost_log_usd(log_path, run_id=None) == 0.35


def test_cost_log_scopes_to_single_run_fails_on_filter_revert(tmp_path: Path) -> None:
    """Guardrail: reverting the run-id filter must make this test fail."""
    from github.orchestrator import _sum_token_cost_log_usd

    log_path = tmp_path / "token-cost-log.jsonl"
    _write_rows(
        log_path,
        [
            {"provider": "anthropic", "estimated_cost_usd": 0.10, "run_id": 101},
            {"provider": "anthropic", "estimated_cost_usd": 0.20, "run_id": 202},
        ],
    )

    scoped = _sum_token_cost_log_usd(log_path, run_id=101)
    whole_file = _sum_token_cost_log_usd(log_path, run_id=None)
    assert scoped != whole_file


def test_llm_usage_session_base_context_writes_run_id(tmp_path: Path) -> None:
    log_path = tmp_path / "token-cost-log.jsonl"

    with llm_usage_session(log_path, module="github", brief_id="brief-1", run_id=4242):
        record_llm_usage(
            provider="anthropic",
            model="claude-opus",
            usage={"input_tokens": 10, "output_tokens": 5},
        )

    row = json.loads(log_path.read_text().splitlines()[0])
    assert row["run_id"] == 4242
    assert row["module"] == "github"
    assert row["brief_id"] == "brief-1"
