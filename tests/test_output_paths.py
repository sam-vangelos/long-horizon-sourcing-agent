"""Tests for the shared output-directory contract."""

from __future__ import annotations

from pathlib import Path

import pytest

import shared.output_paths as output_paths
from market_intelligence.engine import resolve_market_intel_artifact_path
from shared.storage import read_json, write_json


ROOT = Path(__file__).parent.parent
SOURCE_BRIEF = ROOT / "config" / "brief-head-ai-lab-nyc-v2.json"


def test_output_path_classification_contract():
    root = Path("/tmp/example/output")
    assert output_paths.classify_output_location(root) == "output_root"
    assert output_paths.classify_output_location(root / "state" / "linkedin" / "foo") == "state_dir"
    assert output_paths.classify_output_location(root / "runs" / "linkedin" / "foo" / "2026-04-08T13-40-00Z__run-184") == "run_dir"
    assert output_paths.classify_output_location(root / "market_intelligence" / "market-key") == "market_dir"
    assert output_paths.classify_output_location(root / "exports" / "github" / "brief") == "exports_dir"
    assert output_paths.classify_output_location(root / "archive" / "linkedin" / "brief") == "archive_dir"


@pytest.mark.skipif(not SOURCE_BRIEF.is_file(), reason="Optional Head AI Lab V2 brief JSON not under config/")
def test_brief_resolves_to_state_dir_and_run_dir(tmp_path, monkeypatch):
    brief_path = tmp_path / SOURCE_BRIEF.name
    write_json(brief_path, read_json(SOURCE_BRIEF))
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path / "output")

    state_dir = output_paths.resolve_linkedin_state_dir(brief_path=brief_path)
    assert state_dir == tmp_path / "output" / "state" / "linkedin" / "3000000006"
    assert output_paths.is_state_dir(state_dir)

    run_dir = output_paths.resolve_run_dir(
        source="linkedin",
        brief_id="3000000006",
        run_stamp="2026-04-08T13-40-00Z",
        run_id=184,
        output_root=state_dir,
    )
    assert run_dir == tmp_path / "output" / "runs" / "linkedin" / "3000000006" / "2026-04-08T13-40-00Z__run-184"
    assert output_paths.is_run_dir(run_dir)


@pytest.mark.skipif(not SOURCE_BRIEF.is_file(), reason="Optional Head AI Lab V2 brief JSON not under config/")
def test_market_intel_artifact_path_uses_market_root_from_run_dir(tmp_path):
    brief_path = tmp_path / SOURCE_BRIEF.name
    write_json(brief_path, read_json(SOURCE_BRIEF))
    run_dir = tmp_path / "output" / "runs" / "linkedin" / "3000000006" / "2026-04-08T13-40-00Z__run-184"
    run_dir.mkdir(parents=True)

    artifact_path = resolve_market_intel_artifact_path(brief_path, output_dir=run_dir)

    expected_market_key = output_paths.derive_market_key_from_brief(brief_path=brief_path)
    assert artifact_path == tmp_path / "output" / "market_intelligence" / expected_market_key / "market-intel.json"
