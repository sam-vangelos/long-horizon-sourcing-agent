"""Reopen P7.1 / P7.2 / P7.3 / P7.5 — consolidation mechanics.

Covers the hardening-and-consolidation-spec.md §8 items not already
pinned by an existing suite (existing suites were updated in place where
they collided with these fixes — see test_reopen_Y5_2.py,
test_reopen_Y5_6_F1.py, test_cloris_launchers_researcher.py, and
test_researcher_external_evidence_gate.py):

- P7.1: the launch boundary refuses sunset sources (designer,
  exec_search) with a typed 409, even under ``force=true``; linkedin
  and github are unaffected; the recruiter-facing copy says "subagent",
  never "module".
- LinkedIn execution boundary: ``linkedin.run`` retains browser-free
  rejudging and rejects every retired sourcing flag.
- P7.3: a launch that never observes its worker's sidecar returns HTTP
  502 ``worker_did_not_start``, not a 201-with-pid.
- P7.5: researcher's successful runs call ``finish_run`` (no more
  reconciling to abandoned); ``--resume`` exits with a clear error
  instead of silently re-running from scratch.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloris import api as cloris_api
from cloris.api._monolith import (
    BriefPathNotFoundError,
    SourceSunsetError,
    WorkerDidNotStartError,
)
from cloris.launchers import LAUNCHERS


# ---------------------------------------------------------------------------
# P7.1 — launch boundary honors launchability
# ---------------------------------------------------------------------------


def test_registry_marks_designer_and_exec_search_sunset() -> None:
    """Registry-level pin: designer and exec_search are launchable=False
    / sunset=True; every other registered source defaults to launchable
    (in particular linkedin and github, which must stay unaffected)."""

    assert LAUNCHERS["designer"].launchable is False
    assert LAUNCHERS["designer"].sunset is True
    assert LAUNCHERS["exec_search"].launchable is False
    assert LAUNCHERS["exec_search"].sunset is True

    for source in ("linkedin", "github", "researcher"):
        assert LAUNCHERS[source].launchable is True, source
        assert LAUNCHERS[source].sunset is False, source


@pytest.mark.parametrize("source", ["designer", "exec_search"])
def test_spawn_worker_for_source_raises_source_sunset_error(source: str) -> None:
    """The gate fires at the single spawn choke point, before the
    brief-path-existence check — the path below does not exist, yet we
    get SourceSunsetError, not BriefPathNotFoundError."""

    from cloris.api import _spawn_worker_for_source

    with pytest.raises(SourceSunsetError) as excinfo:
        _spawn_worker_for_source(
            source=source,
            brief_path=Path("/nonexistent/brief.json"),
            mode="fresh",
        )
    assert excinfo.value.source == source


def _v2_minimal(role: str) -> dict:
    return {
        "role_title": role,
        "id": role.lower().replace(" ", "_"),
        "linkedin_project_id": role.lower().replace(" ", "_"),
        "capability_areas": [{"name": "Eng", "description": "ships systems."}],
        "depth_distinction": {
            "builder_definition": "owns",
            "user_definition": "uses",
            "edge_case_guidance": "borderline",
        },
    }


def _seed_brief(config_dir: Path, role: str) -> str:
    bdir = config_dir / role.replace(" ", "-")
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / "brief.json").write_text(json.dumps(_v2_minimal(role)))

    from shared.output_paths import derive_brief_id

    return derive_brief_id(brief_path=str(bdir / "brief.json"))


@pytest.fixture()
def api_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    """Real app + REAL spawn path (not stubbed), mirroring
    tests/test_reopen_Y5_2.py's fixture, so the sunset gate and the
    wait_for_sidecar honesty gate are both exercised for real."""

    from cloris.app import create_app

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    monkeypatch.setattr(
        cloris_api._monolith, "_readiness_blockers", lambda source, brief_id: []
    )

    class _FakeProcess:
        def __init__(self, pid: int = 88888) -> None:
            self.pid = pid

    def _fake_popen(*args: Any, **kwargs: Any) -> _FakeProcess:
        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(
        cloris_api._monolith, "wait_for_sidecar", lambda *a, **k: True
    )

    return TestClient(create_app()), config_dir


@pytest.mark.parametrize("source", ["designer", "exec_search"])
def test_sunset_source_launch_returns_409(
    api_client: tuple[TestClient, Path], source: str
) -> None:
    api, config_dir = api_client
    brief_id = _seed_brief(config_dir, role=f"Sunset {source}")

    resp = api.post(
        f"/api/launch/{source}",
        json={"brief_id": brief_id, "mode": "fresh"},
    )

    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "source_sunset"
    assert detail["source"] == source
    assert "subagent" in detail["message"].lower()
    assert "module" not in detail["message"].lower()


@pytest.mark.parametrize("source", ["designer", "exec_search"])
def test_sunset_source_force_true_does_not_bypass_gate(
    api_client: tuple[TestClient, Path], source: str
) -> None:
    """force=true skips readiness PROBES only — the sunset gate is not a
    readiness probe and must still refuse."""

    api, config_dir = api_client
    brief_id = _seed_brief(config_dir, role=f"Sunset Forced {source}")

    resp = api.post(
        f"/api/launch/{source}",
        json={"brief_id": brief_id, "mode": "fresh", "force": True},
    )

    assert resp.status_code == 409, resp.text
    assert resp.json()["detail"]["error"] == "source_sunset"


def test_github_launch_unaffected_by_sunset_gate(
    api_client: tuple[TestClient, Path],
) -> None:
    api, config_dir = api_client

    brief_id = _seed_brief(config_dir, role="Unaffected github")
    resp = api.post(
        "/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh"},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["pid"] == 88888


def test_linkedin_launch_unaffected_by_sunset_gate(
    api_client: tuple[TestClient, Path],
) -> None:
    """linkedin's launch entry point is the legacy literal
    ``/api/launch/linkedin`` route (Starlette route-declaration order
    matches it before the generic ``/api/launch/{source}`` path-param
    route — see tests/test_launch_endpoint_generic.py), which takes
    ``{brief_path}`` rather than ``{brief_id, mode}``."""

    api, config_dir = api_client

    bdir = config_dir / "Unaffected-linkedin"
    bdir.mkdir(parents=True, exist_ok=True)
    brief_path = bdir / "brief.json"
    brief_path.write_text(json.dumps(_v2_minimal("Unaffected linkedin")))

    resp = api.post("/api/launch/linkedin", json={"brief_path": str(brief_path)})
    assert resp.status_code == 201, resp.text
    assert resp.json()["pid"] == 88888


# ---------------------------------------------------------------------------
# P7.3 — launch success is observed, not asserted
# ---------------------------------------------------------------------------


def test_worker_did_not_start_returns_502(
    api_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When wait_for_sidecar times out (returns False), the endpoint
    must report HTTP 502 ``worker_did_not_start`` — never a 201-with-pid
    for a worker we never observed starting."""

    api, config_dir = api_client
    monkeypatch.setattr(
        cloris_api._monolith, "wait_for_sidecar", lambda *a, **k: False
    )
    brief_id = _seed_brief(config_dir, role="Never Started")

    resp = api.post(
        "/api/launch/github",
        json={"brief_id": brief_id, "mode": "fresh"},
    )

    assert resp.status_code == 502, resp.text
    detail = resp.json()["detail"]
    assert detail["error"] == "worker_did_not_start"
    assert detail["source"] == "github"
    assert detail["pid"] == 88888
    assert detail["message"] == "worker did not start"


def test_legacy_launch_linkedin_502_on_worker_did_not_start(
    api_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy ``POST /api/launch/linkedin`` synonym route must map
    the same failure the same way — it calls _spawn_worker_for_source
    directly, bypassing _launch_for_source_impl's except chain."""

    api, config_dir = api_client
    monkeypatch.setattr(
        cloris_api._monolith, "wait_for_sidecar", lambda *a, **k: False
    )
    bdir = config_dir / "Legacy-Never-Started"
    bdir.mkdir(parents=True, exist_ok=True)
    brief_path = bdir / "brief.json"
    brief_path.write_text(json.dumps(_v2_minimal("Legacy Never Started")))

    resp = api.post("/api/launch/linkedin", json={"brief_path": str(brief_path)})

    assert resp.status_code == 502, resp.text
    assert resp.json()["detail"]["error"] == "worker_did_not_start"


def test_spawn_worker_for_source_raises_worker_did_not_start_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit-level pin, independent of the FastAPI layer: the spawn
    helper itself raises when the sidecar is never observed."""

    from cloris.api import _spawn_worker_for_source

    monkeypatch.setattr(
        cloris_api._monolith, "wait_for_sidecar", lambda *a, **k: False
    )

    class _FakeProcess:
        pid = 77777

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _FakeProcess())

    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps(_v2_minimal("Unit Never Started")))

    with pytest.raises(WorkerDidNotStartError) as excinfo:
        _spawn_worker_for_source(source="github", brief_path=brief_path, mode="fresh")

    assert excinfo.value.source == "github"
    assert excinfo.value.pid == 77777


# ---------------------------------------------------------------------------
# LinkedIn rejudge-only CLI boundary
# ---------------------------------------------------------------------------


def _write_linkedin_brief(config_dir: Path, role: str) -> Path:
    brief_dir = config_dir / role.replace(" ", "-")
    brief_dir.mkdir(parents=True, exist_ok=True)
    brief_path = brief_dir / "brief.json"
    brief_path.write_text(json.dumps(_v2_minimal(role)))
    return brief_path


def test_linkedin_run_executes_only_browser_free_rejudging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linkedin.orchestrator
    import linkedin.run

    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps(_v2_minimal("Rejudge")))
    snippets = tmp_path / "snippets.jsonl"
    snippets.write_text("")
    state_dir = tmp_path / "state"
    observed: dict[str, Any] = {}

    class RejudgeOnlyPipeline:
        def __init__(self, **kwargs: Any) -> None:
            observed["kwargs"] = kwargs

        async def rejudge_from_file(self, path: str) -> None:
            observed["snippets"] = path

        async def run_full(self, **_kwargs: Any) -> None:
            raise AssertionError("linkedin.run acquired sourcing authority")

    monkeypatch.setattr(linkedin.orchestrator, "Pipeline", RejudgeOnlyPipeline)
    monkeypatch.setattr(linkedin.run, "enable_console_tee", lambda path: observed.setdefault("tee", path))

    linkedin.run.main(
        [
            "--brief",
            str(brief),
            "--rejudge-from",
            str(snippets),
            "--state-dir",
            str(state_dir),
        ]
    )

    assert observed["kwargs"] == {
        "brief_path": str(brief),
        "output_dir": str(state_dir),
    }
    assert observed["snippets"] == str(snippets)
    assert observed["tee"] == state_dir


@pytest.mark.parametrize(
    "retired_flag",
    ["--full-run", "--resume", "--test-single-page", "--search-config"],
)
def test_linkedin_run_rejects_retired_sourcing_flags(
    tmp_path: Path,
    retired_flag: str,
) -> None:
    import linkedin.run

    brief = tmp_path / "brief.json"
    brief.write_text("{}")
    snippets = tmp_path / "snippets.jsonl"
    snippets.write_text("")

    with pytest.raises(SystemExit) as exc_info:
        linkedin.run.main(
            [
                "--brief",
                str(brief),
                "--rejudge-from",
                str(snippets),
                retired_flag,
            ]
        )

    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# P7.5 — researcher honesty minimums
# ---------------------------------------------------------------------------


def _researcher_stub_brief():
    from types import SimpleNamespace

    return SimpleNamespace(
        id="p7-researcher-honesty",
        role_title="Researcher",
        capability_areas=[
            SimpleNamespace(name="Post-training", description="RLHF/DPO/SFT.")
        ],
        depth_distinction=SimpleNamespace(
            builder_definition="First-author publications.",
            user_definition="Cites without publishing.",
            edge_case_guidance="Borderline = full eval.",
        ),
        _new_brief={"source_config": {"researcher": {"discipline": "ml_general"}}},
    )


def _researcher_strategy_response() -> dict:
    return {
        "generated_strings": [
            {
                "id": 1,
                "name": "q1",
                "topic_concepts": ["C1"],
                "ror_country_filter": ["US"],
            },
        ],
    }


class _EmptyOpenAlexClient:
    def search_authors(self, **kwargs: Any) -> dict:
        return {"meta": {"next_cursor": ""}, "results": []}


class _BoomingOpenAlexClient:
    """Raises from the query-execution path (researcher/acquisition.py's
    ``execute_query`` has no fail-soft wrapper around this call, unlike
    the strategy-formation LLM caller, which falls back to a heuristic
    plan on failure and would not exercise ``run()``'s except branch)."""

    def search_authors(self, **kwargs: Any) -> dict:
        raise RuntimeError("simulated openalex failure")


def test_researcher_pipeline_calls_finish_run_on_success(tmp_path: Path) -> None:
    from researcher.orchestrator import ResearcherPipeline
    from shared.runtime_state.researcher import ResearcherRuntimeStateBridge
    from shared.runtime_state.store import RuntimeStateStore

    state_dir = tmp_path / "researcher" / "key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _researcher_stub_brief()
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)
    assert store.get_run(run_id)["status"] == "running"

    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=_EmptyOpenAlexClient(),
        strategy_llm_caller=lambda _s, _u: _researcher_strategy_response(),
    )
    pipeline.run(run_id=run_id)

    row = store.get_run(run_id)
    assert row["status"] != "running", (
        "researcher run never called finish_run — the reconciler will "
        "finalize this successful run as abandoned"
    )
    assert row["status"] == "completed"


def test_researcher_pipeline_calls_finish_run_with_error_status_on_exception(
    tmp_path: Path,
) -> None:
    from researcher.orchestrator import ResearcherPipeline
    from shared.runtime_state.researcher import ResearcherRuntimeStateBridge
    from shared.runtime_state.store import RuntimeStateStore

    state_dir = tmp_path / "researcher" / "key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    brief = _researcher_stub_brief()
    bridge = ResearcherRuntimeStateBridge(
        store=store,
        output_dir=state_dir,
        brief_id=brief.id,
        brief_name=brief.role_title,
    )
    run_id = bridge.start_or_resume_run(resume=False)

    pipeline = ResearcherPipeline(
        brief=brief,
        bridge=bridge,
        openalex_client=_BoomingOpenAlexClient(),
        strategy_llm_caller=lambda _s, _u: _researcher_strategy_response(),
    )

    with pytest.raises(RuntimeError, match="simulated openalex failure"):
        pipeline.run(run_id=run_id)

    row = store.get_run(run_id)
    assert row["status"] == "error"


def test_researcher_session_orchestrator_resume_exits_with_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    from researcher import session_orchestrator

    brief_path = _write_linkedin_brief(tmp_path, "Researcher Resume Guard")

    def _fail_if_called(**_kwargs: Any):
        raise AssertionError(
            "build_pipeline must not be called when --resume is passed — "
            "the flag must exit before doing any work"
        )

    monkeypatch.setattr(session_orchestrator, "build_pipeline", _fail_if_called)

    exit_code = session_orchestrator.main(
        [
            "--brief",
            str(brief_path),
            "--state-dir",
            str(tmp_path / "state"),
            "--resume",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "resume not implemented for researcher" in captured.err.lower()


def test_researcher_orchestrator_argv_never_appends_resume_regardless_of_flag() -> None:
    """cloris/launchers/__init__.py must not hand researcher's CLI a flag
    its own CLI now hard-refuses."""

    entry = LAUNCHERS["researcher"]
    for resume in (True, False):
        argv = entry.orchestrator_argv_fn(
            "/path/to/brief.json", "/path/to/state_dir", resume=resume
        )
        assert "--resume" not in argv
