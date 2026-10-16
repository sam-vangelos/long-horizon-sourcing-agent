"""Tests for GitHub side-effect output routing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from github.side_effects import GitHubSideEffectsService


def test_export_saved_candidates_csv_routes_to_exports_root(tmp_path):
    state_dir = tmp_path / "output" / "state" / "github" / "brief123"
    state_dir.mkdir(parents=True, exist_ok=True)
    expected_csv = (
        tmp_path / "output" / "exports" / "github" / "brief123" / "saved_candidates.csv"
    )

    pipeline = SimpleNamespace(
        stats={"saved": 1},
        output_dir=state_dir,
        brief_obj=SimpleNamespace(id="brief123"),
        _observer=SimpleNamespace(
            console=SimpleNamespace(
                emit_info=MagicMock(),
                emit_warn=MagicMock(),
            )
        ),
        _runtime_run_id=None,
    )

    service = GitHubSideEffectsService(pipeline)

    with patch("github.export.export_saved_candidates_csv", return_value=expected_csv) as mock_export:
        outcome = service.export_saved_candidates_csv()

    assert outcome is not None
    assert outcome.status == "succeeded"
    assert Path(outcome.payload["path"]) == expected_csv
    assert mock_export.call_args.args[0] == state_dir
    assert Path(mock_export.call_args.kwargs["csv_path"]) == expected_csv
