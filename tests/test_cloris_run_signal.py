"""Tests for the homescreen live run signal endpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cloris.app import create_app
from shared.storage import log_event


@pytest.fixture()
def client_with_isolated_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True, exist_ok=True)
    import shared.output_paths as output_paths

    monkeypatch.setattr(output_paths, "STATE_ROOT", state_root)
    return TestClient(create_app()), state_root


def _state_dir(state_root: Path, source: str = "linkedin", state_key: str = "li-signal") -> Path:
    state_dir = state_root / source / state_key
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _write_alive_sidecar(state_dir: Path) -> None:
    (state_dir / "worker.json").write_text(
        json.dumps({"pid": os.getpid(), "mode": "fresh"}),
        encoding="utf-8",
    )


def test_run_signal_reports_strategy_phase_before_plan_exists(client_with_isolated_root):
    client, state_root = client_with_isolated_root
    state_dir = _state_dir(state_root)
    _write_alive_sidecar(state_dir)
    (state_dir / "live-console.log").write_text(
        "\n".join(
            [
                "Connected to browser",
                "--- Phase 2: Strategy Formation (Opus) ---",
                "  Strategizing... (Opus is synthesizing compound search strings)",
            ]
        ),
        encoding="utf-8",
    )

    res = client.get("/api/run-signal/linkedin/li-signal")

    assert res.status_code == 200
    body = res.json()
    assert body["slice"] == "v0-run-signal-1"
    assert body["active"] is True
    assert body["phase"] == "strategizing"
    assert body["headline"] == "Strategizing"
    assert "compound strings" in body["detail"]
    assert body["strategy_strings"] == []


def test_run_signal_surfaces_strategy_output_and_recent_activity(client_with_isolated_root):
    client, state_root = client_with_isolated_root
    state_dir = _state_dir(state_root, state_key="li-plan")
    _write_alive_sidecar(state_dir)
    (state_dir / "live-console.log").write_text(
        "\n".join(
            [
                "  Strategy complete.",
                "  2 compound strings synthesized from JD context",
                "--- Execution (2 strings) ---",
            ]
        ),
        encoding="utf-8",
    )
    (state_dir / "execution_plan.json").write_text(
        json.dumps(
            {
                "strategy_rationale": "Use Spanish vocabulary first, then precision post-training strings.",
                "architecture": "negative_space",
                "architecture_rationale": "Avoid generic annotation and analytics-heavy pools.",
                "generated_strings": [
                    {
                        "boolean": "(aprendizaje OR evaluacion) AND Python",
                        "rationale": "Spanish-language profile vocabulary catches missed Colombian builders.",
                        "family_key": "spanish_vocab_ml_builders",
                        "domain_lane": "spanish_language_profiles",
                        "novelty_bucket": "edge_case",
                    },
                    {
                        "boolean": "(reward model OR RLHF) AND PyTorch",
                        "rationale": "Precision post-training query.",
                        "family_key": "rl_posttraining_direct",
                        "domain_lane": "rl_posttraining",
                        "novelty_bucket": "canonical",
                    },
                ],
                "coverage_gaps": [{"suggested_boolean": "x"}],
            }
        ),
        encoding="utf-8",
    )
    log_event(state_dir / "run_log.jsonl", "pipeline_start", mode="full_run")
    log_event(state_dir / "run_log.jsonl", "string_results", string_id=1, result_count_text="3K+")
    log_event(state_dir / "run_log.jsonl", "candidate_saved", name="Test Candidate")

    res = client.get("/api/run-signal/linkedin/li-plan")

    assert res.status_code == 200
    body = res.json()
    assert body["phase"] == "searching"
    assert body["headline"] == "Searching LinkedIn"
    assert body["strategy_architecture"] == "Negative Space"
    assert body["generated_string_count"] == 2
    assert body["coverage_gap_count"] == 1
    assert body["strategy_strings"][0]["label"] == "Spanish Language Profiles"
    assert "Spanish-language" in body["strategy_strings"][0]["rationale"]
    assert body["recent_events"][0]["label"] == "Saved a candidate"
    assert body["recent_events"][0]["detail"] == "Test Candidate"


def test_run_signal_filters_low_level_browser_error_language(client_with_isolated_root):
    client, state_root = client_with_isolated_root
    state_dir = _state_dir(state_root, state_key="li-errors")
    _write_alive_sidecar(state_dir)
    (state_dir / "live-console.log").write_text("--- Execution (1 strings) ---", encoding="utf-8")
    log_event(
        state_dir / "run_log.jsonl",
        "profile_browser_disconnect",
        error="Target page, context or browser has been closed",
    )
    log_event(state_dir / "run_log.jsonl", "string_results", string_id=4, result_count_text="12")

    res = client.get("/api/run-signal/linkedin/li-errors")

    assert res.status_code == 200
    body_text = json.dumps(res.json()).lower()
    assert "browser" not in body_text
    assert "target page" not in body_text
    assert res.json()["recent_events"][0]["label"] == "Opened string #4"


def test_run_signal_unknown_state_dir_returns_404(client_with_isolated_root):
    client, _ = client_with_isolated_root

    res = client.get("/api/run-signal/linkedin/not-here")

    assert res.status_code == 404
    assert res.json()["detail"]["error"] == "state_dir_not_found"


# ---------------------------------------------------------------------------
# Audit finding F-3: canonical SQLite lifecycle takes precedence over
# stale projection text. Without this, a worker that crashed (or
# already finished) leaves a ``Strategizing...`` console log + alive-PID
# sidecar that fakes recruiter-visible "live" state for the next launch.
# ---------------------------------------------------------------------------


def _seed_canonical_run(state_dir: Path, status: str) -> None:
    """Create a ``runtime_state.sqlite3`` whose latest run is ``status``."""

    from shared.runtime_state.store import RuntimeStateStore

    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="brief-live-signal-canonical",
        output_dir=str(state_dir),
        mode="fresh",
    )
    if status != "running":
        store.finish_run(run_id, status, stop_reason="test_seed")


@pytest.mark.parametrize(
    "canonical_status",
    [
        "completed",
        "succeeded",
        "error",
        "failed",
        "governor_limit_reached",
        "interrupted",
        "abandoned",
    ],
)
def test_run_signal_collapses_strategizing_when_canonical_terminal(
    client_with_isolated_root, caplog, canonical_status: str
):
    """Stale projection still mentions ``Strategizing...`` and worker.json
    points at this process's PID; canonical SQLite says the latest run is
    terminal. Wire must collapse to inactive/finished and log drift."""

    client, state_root = client_with_isolated_root
    slug = canonical_status.replace("_", "-")
    state_dir = _state_dir(state_root, state_key=f"li-term-{slug}")
    _write_alive_sidecar(state_dir)
    (state_dir / "live-console.log").write_text(
        "  Strategizing... (Opus is synthesizing compound search strings)\n",
        encoding="utf-8",
    )
    _seed_canonical_run(state_dir, status=canonical_status)

    with caplog.at_level("WARNING", logger="cloris.live_signal"):
        res = client.get(f"/api/run-signal/linkedin/li-term-{slug}")

    assert res.status_code == 200
    body = res.json()
    assert body["active"] is False
    assert body["phase"] == "completed"
    assert body["lifecycle"] == "finished"
    assert body["headline"] != "Strategizing"
    assert any(
        "projection/canonical drift" in record.message for record in caplog.records
    )


def test_run_signal_trusts_projection_when_canonical_running(
    client_with_isolated_root,
):
    """When canonical agrees the run is ``running``, the projection-derived
    phase still carries the recruiter-facing detail (no regression on
    the happy path)."""

    client, state_root = client_with_isolated_root
    state_dir = _state_dir(state_root, state_key="li-canonical-running")
    _write_alive_sidecar(state_dir)
    (state_dir / "live-console.log").write_text(
        "  Strategizing... (Opus is synthesizing compound search strings)\n",
        encoding="utf-8",
    )
    _seed_canonical_run(state_dir, status="running")

    res = client.get("/api/run-signal/linkedin/li-canonical-running")

    assert res.status_code == 200
    body = res.json()
    assert body["active"] is True
    assert body["phase"] == "strategizing"
    assert body["headline"] == "Strategizing"


def test_run_signal_falls_back_to_projections_when_no_canonical_db(
    client_with_isolated_root,
):
    """Pre-runtime-state state dirs (no ``runtime_state.sqlite3``) keep
    the existing projection-only behavior so the change doesn't regress
    legacy fixtures."""

    client, state_root = client_with_isolated_root
    state_dir = _state_dir(state_root, state_key="li-legacy-no-db")
    _write_alive_sidecar(state_dir)
    (state_dir / "live-console.log").write_text(
        "  Strategizing... (Opus is synthesizing compound search strings)\n",
        encoding="utf-8",
    )

    res = client.get("/api/run-signal/linkedin/li-legacy-no-db")

    assert res.status_code == 200
    body = res.json()
    assert body["phase"] == "strategizing"
