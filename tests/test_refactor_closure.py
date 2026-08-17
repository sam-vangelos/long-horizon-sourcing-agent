"""Closure-phase regressions for runtime-state-first behavior and contracts."""

from __future__ import annotations

import json

import pytest

from shared.runtime_state import (
    ARTIFACT_CONTRACTS,
    ArtifactOwnership,
    GitHubRuntimeStateBridge,
    LinkedInRuntimeStateBridge,
    RuntimeStateBridge,
    RuntimeStateStore,
    classify_artifact,
)
from tests.test_linkedin_pipeline import _make_pipeline as _make_linkedin_pipeline


def test_runtime_bridges_share_the_common_protocol(tmp_path):
    store = RuntimeStateStore(tmp_path / "runtime_state.sqlite3")
    github_bridge = GitHubRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="brief-1",
        brief_name="brief-1",
    )
    linkedin_bridge = LinkedInRuntimeStateBridge(
        store=store,
        output_dir=tmp_path,
        brief_id="brief-1",
        brief_name="brief-1",
    )

    assert isinstance(github_bridge, RuntimeStateBridge)
    assert isinstance(linkedin_bridge, RuntimeStateBridge)


def test_artifact_registry_classifies_projection_and_direct_outputs():
    assert classify_artifact("progress.json") == ArtifactOwnership.PROJECTION_OWNED
    assert classify_artifact("candidate_history-brief-1.jsonl") == ArtifactOwnership.PROJECTION_OWNED
    assert classify_artifact("search_memory-brief-1.json") == ArtifactOwnership.PROJECTION_OWNED
    assert classify_artifact("outreach.jsonl") == ArtifactOwnership.DIRECT_SIDE_EFFECT
    assert classify_artifact("bias_monitor-test-project.json") == ArtifactOwnership.DIRECT_SIDE_EFFECT
    assert classify_artifact("execution_plan.json") == ArtifactOwnership.ANALYTICAL_DEBUG
    assert any(contract.pattern == "progress.json" for contract in ARTIFACT_CONTRACTS)


def test_github_bridge_resume_ignores_stale_progress_projection(tmp_path):
    output_dir = tmp_path / "github-output"
    output_dir.mkdir()
    (output_dir / "progress.json").write_text(
        json.dumps(
            {
                "brief_name": "stale-brief",
                "queries": [{"id": 99, "name": "stale", "query": "stale", "channel": "user_search"}],
            }
        )
    )

    store = RuntimeStateStore(output_dir / "runtime_state.sqlite3")
    bridge = GitHubRuntimeStateBridge(
        store=store,
        output_dir=output_dir,
        brief_id="brief-1",
        brief_name="brief-1",
    )

    run_id, progress = bridge.start_or_resume_run(resume=True)

    assert run_id > 0
    assert progress.brief_name == "brief-1"
    assert progress.queries == []

    projected = json.loads((output_dir / "progress.json").read_text())
    assert projected["brief_name"] == "brief-1"
    assert projected["queries"] == []


def test_linkedin_progress_bootstrap_requires_runtime_state(tmp_path):
    pipeline = _make_linkedin_pipeline(str(tmp_path))
    (tmp_path / "progress.json").write_text(
        json.dumps(
            {
                "brief_name": "wrong-brief",
                "strings": [{"id": 99, "name": "stale", "boolean": "stale"}],
            }
        )
    )

    with pytest.raises(
        RuntimeError,
        match="resume requires canonical or legacy runtime state",
    ):
        pipeline._load_or_create_progress()
