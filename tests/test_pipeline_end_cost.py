"""P4.2 — run-level cost for LinkedIn and GitHub pipeline_end.

Both orchestrators previously either omitted cost entirely (LinkedIn) or
hardcoded ``cost_usd=0.0`` despite metering real spend (GitHub). Both now
sum the run's ``token-cost-log.jsonl`` at finalize, and never emit an
affirmative ``0.0``/``cost_usd`` when the log has no usable cost signal.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from shared.storage import append_jsonl


def _write_usage_rows(path: Path, costs: list[float | None]) -> None:
    for cost in costs:
        append_jsonl(path, {"provider": "anthropic", "estimated_cost_usd": cost})


# ---------------------------------------------------------------------------
# LinkedIn helpers
# ---------------------------------------------------------------------------


def test_linkedin_sum_token_cost_log_usd_missing_file_returns_none(tmp_path):
    from linkedin.orchestrator import _sum_token_cost_log_usd

    assert _sum_token_cost_log_usd(tmp_path / "does-not-exist.jsonl") is None


def test_linkedin_sum_token_cost_log_usd_sums_rows(tmp_path):
    from linkedin.orchestrator import _sum_token_cost_log_usd

    log_path = tmp_path / "token-cost-log.jsonl"
    _write_usage_rows(log_path, [0.0525, 0.09, 0.015])

    assert _sum_token_cost_log_usd(log_path) == 0.1575


def test_linkedin_sum_token_cost_log_usd_returns_none_when_all_costs_unknown(tmp_path):
    """A JSONL that exists but never resolved a rate (None cost on every
    row) must NOT read back as an affirmative $0 — it's missing data."""
    from linkedin.orchestrator import _sum_token_cost_log_usd

    log_path = tmp_path / "token-cost-log.jsonl"
    _write_usage_rows(log_path, [None, None])

    assert _sum_token_cost_log_usd(log_path) is None


def test_linkedin_cost_per_save_usd_guards_div_by_zero():
    from linkedin.orchestrator import _cost_per_save_usd

    assert _cost_per_save_usd(1.5, 0) is None
    assert _cost_per_save_usd(None, 3) is None
    assert _cost_per_save_usd(3.0, 2) == 1.5


def _make_linkedin_pipeline(output_dir: str):
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test"
        brief.linkedin_project_id = "test-project"
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        mock_brief.return_value = brief

        brief_path = Path(output_dir) / "brief.json"
        brief_path.write_text('{"id": "test"}')

        from linkedin.orchestrator import Pipeline

        return Pipeline(brief_path=str(brief_path), output_dir=output_dir)


def test_linkedin_pipeline_end_stats_includes_cost_when_jsonl_present():
    with tempfile.TemporaryDirectory() as td:
        p = _make_linkedin_pipeline(td)
        p.stats.update({"saved": 2})
        _write_usage_rows(Path(td) / "token-cost-log.jsonl", [0.5, 1.0])

        payload = p._pipeline_end_stats()

        assert payload["cost_usd"] == 1.5
        assert payload["cost_per_save_usd"] == 0.75
        # Base stats must still be present (additive, not replaced).
        assert payload["saved"] == 2


def test_linkedin_pipeline_end_stats_omits_cost_when_jsonl_absent():
    with tempfile.TemporaryDirectory() as td:
        p = _make_linkedin_pipeline(td)

        payload = p._pipeline_end_stats()

        assert "cost_usd" not in payload
        assert "cost_per_save_usd" not in payload


def test_linkedin_cost_summary_for_report_status_marker_when_no_data():
    with tempfile.TemporaryDirectory() as td:
        p = _make_linkedin_pipeline(td)

        summary = p._cost_summary_for_report()

        assert summary == {"status": "no_cost_data"}


def test_linkedin_cost_summary_for_report_omits_cost_per_save_when_no_saves():
    with tempfile.TemporaryDirectory() as td:
        p = _make_linkedin_pipeline(td)
        _write_usage_rows(Path(td) / "token-cost-log.jsonl", [2.0])

        summary = p._cost_summary_for_report()

        assert summary["status"] == "ok"
        assert summary["cost_usd"] == 2.0
        assert summary["cost_per_save_usd"] is None


# ---------------------------------------------------------------------------
# GitHub helpers
# ---------------------------------------------------------------------------


def test_github_sum_token_cost_log_usd_missing_file_returns_none(tmp_path):
    from github.orchestrator import _sum_token_cost_log_usd

    assert _sum_token_cost_log_usd(tmp_path / "does-not-exist.jsonl") is None


def test_github_sum_token_cost_log_usd_sums_rows(tmp_path):
    from github.orchestrator import _sum_token_cost_log_usd

    log_path = tmp_path / "token-cost-log.jsonl"
    _write_usage_rows(log_path, [0.02, 0.03])

    assert _sum_token_cost_log_usd(log_path) == 0.05


def test_github_cost_per_save_usd_guards_div_by_zero():
    from github.orchestrator import _cost_per_save_usd

    assert _cost_per_save_usd(1.5, 0) is None
    assert _cost_per_save_usd(None, 3) is None
    assert _cost_per_save_usd(4.0, 4) == 1.0


# ---------------------------------------------------------------------------
# Shadow-judge cost honesty (GLM promotion): the PRIMARY run cost must exclude
# rows tagged with ``shadow_stage``, not all provider="fireworks" rows, because
# Fireworks now also serves GLM primary calls.
# ---------------------------------------------------------------------------


def _write_provider_rows(log_path, rows):
    import json

    with open(log_path, "w") as fh:
        for provider, cost in rows:
            fh.write(json.dumps({"provider": provider, "estimated_cost_usd": cost}) + "\n")


def test_linkedin_primary_cost_includes_fireworks_primary_excludes_shadow_stage_rows(tmp_path):
    from linkedin.orchestrator import _sum_token_cost_log_usd

    p = _make_linkedin_pipeline(str(tmp_path))
    p.stats.update({"saved": 3})
    log_path = tmp_path / "token-cost-log.jsonl"
    rows = [
        {"provider": "anthropic", "estimated_cost_usd": 1.0},
        {"provider": "fireworks", "estimated_cost_usd": 2.0},
        {
            "provider": "fireworks",
            "shadow_stage": "facial_shadow",
            "estimated_cost_usd": 4.0,
        },
        {
            "provider": "fireworks",
            "shadow_stage": "full_shadow",
            "estimated_cost_usd": 8.0,
        },
    ]
    for row in rows:
        append_jsonl(log_path, row)

    summary = p._cost_summary_for_report()
    payload = p._pipeline_end_stats()

    assert summary["status"] == "ok"
    assert summary["cost_usd"] == 3.0
    assert summary["cost_per_save_usd"] == 1.0
    assert payload["cost_usd"] == 3.0
    assert payload["cost_per_save_usd"] == 1.0
    assert _sum_token_cost_log_usd(
        log_path,
        provider_filter="fireworks",
        field_equals={"shadow_stage": "facial_shadow"},
    ) == 4.0


def test_linkedin_provider_filter_sums_only_shadow_rows(tmp_path):
    from linkedin.orchestrator import _sum_token_cost_log_usd

    log_path = tmp_path / "token-cost-log.jsonl"
    _write_provider_rows(
        log_path,
        [("anthropic", 1.0), ("fireworks", 0.4), ("fireworks", 0.1)],
    )

    assert _sum_token_cost_log_usd(log_path, provider_filter="fireworks") == 0.5


def test_linkedin_provider_filter_returns_none_when_no_shadow_rows(tmp_path):
    from linkedin.orchestrator import _sum_token_cost_log_usd

    log_path = tmp_path / "token-cost-log.jsonl"
    _write_provider_rows(log_path, [("anthropic", 1.0)])

    assert _sum_token_cost_log_usd(log_path, provider_filter="fireworks") is None


def test_linkedin_field_equals_filters_by_shadow_stage(tmp_path):
    """field_equals splits per-tier shadow spend when two shadow tiers
    (facial_shadow / full_shadow) share provider="fireworks" rows."""
    from linkedin.orchestrator import _sum_token_cost_log_usd
    import json

    log_path = tmp_path / "token-cost-log.jsonl"
    with open(log_path, "w") as fh:
        fh.write(json.dumps({"provider": "fireworks", "shadow_stage": "facial_shadow", "estimated_cost_usd": 0.1}) + "\n")
        fh.write(json.dumps({"provider": "fireworks", "shadow_stage": "full_shadow", "estimated_cost_usd": 0.9}) + "\n")
        fh.write(json.dumps({"provider": "fireworks", "estimated_cost_usd": 0.5}) + "\n")  # no shadow_stage at all
        fh.write(json.dumps({"provider": "anthropic", "estimated_cost_usd": 2.0}) + "\n")

    assert _sum_token_cost_log_usd(
        log_path, provider_filter="fireworks", field_equals={"shadow_stage": "facial_shadow"}
    ) == 0.1
    assert _sum_token_cost_log_usd(
        log_path, provider_filter="fireworks", field_equals={"shadow_stage": "full_shadow"}
    ) == 0.9
    # Without field_equals, all three fireworks rows still sum together —
    # unchanged legacy behavior when the caller doesn't opt into per-stage
    # filtering.
    assert _sum_token_cost_log_usd(log_path, provider_filter="fireworks") == 1.5


def test_linkedin_field_equals_returns_none_when_no_rows_match(tmp_path):
    from linkedin.orchestrator import _sum_token_cost_log_usd
    import json

    log_path = tmp_path / "token-cost-log.jsonl"
    with open(log_path, "w") as fh:
        fh.write(json.dumps({"provider": "fireworks", "shadow_stage": "facial_shadow", "estimated_cost_usd": 0.1}) + "\n")

    assert _sum_token_cost_log_usd(
        log_path, provider_filter="fireworks", field_equals={"shadow_stage": "full_shadow"}
    ) is None


def test_linkedin_providerless_rows_count_as_primary(tmp_path):
    """Legacy rows without a provider field are primary spend, not shadow."""
    from linkedin.orchestrator import _sum_token_cost_log_usd
    import json

    log_path = tmp_path / "token-cost-log.jsonl"
    with open(log_path, "w") as fh:
        fh.write(json.dumps({"estimated_cost_usd": 2.0}) + "\n")

    assert _sum_token_cost_log_usd(
        log_path, exclude_rows_with=("shadow_stage",)
    ) == 2.0
