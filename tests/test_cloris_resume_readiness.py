"""Phase 1.3 tests for the server-side resume readiness precheck.

The bug being fixed: ``POST /api/resume/linkedin`` used to spawn a
worker even when ``progress.json`` had no queued work. The worker
exited cleanly seconds later, but the API had already returned 201
and the UI rendered a "Resumed" success message that was a lie.

After Phase 1.3, ``_spawn_linkedin_worker(mode="resume")`` consults
``read_models.has_pending_work(state_dir)`` before Popen:

- ``True`` ⇒ proceed normally.
- ``False`` ⇒ raise ``NoPendingWorkError`` (HTTP 422).
- ``None`` ⇒ proceed (preserve the orchestrator's bias toward
  attempting resume on missing/malformed inputs; documented divergence).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cloris import api as cloris_api
from cloris.api import (
    LaunchLinkedInRequest,
    NoPendingWorkError,
    _spawn_linkedin_worker,
)
from cloris.app import create_app
from shared.runtime_state.store import RuntimeStateStore


class _StubPopenWithSidecar:
    """Same shape as the integration stub in test_cloris_launch_lock.

    Drops a sidecar at the resolved state_dir so wait_for_sidecar
    short-circuits and the helper returns immediately.
    """

    instances: list["_StubPopenWithSidecar"] = []

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = list(argv)
        self.kwargs = dict(kwargs)
        sd_index = argv.index("--state-dir") + 1
        state_dir = Path(argv[sd_index])
        self.pid = 4242
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "worker.json").write_text(
            json.dumps({"pid": self.pid, "source": "linkedin"})
        )
        _StubPopenWithSidecar.instances.append(self)


def _make_brief(tmp_path: Path) -> Path:
    brief_path = tmp_path / "brief.json"
    brief_path.write_text('{"role_title": "Resume Test"}')
    return brief_path


@pytest.fixture(autouse=True)
def _contain_briefs_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 containment: this suite writes brief fixtures under tmp_path, which
    the config/ containment boundary rejects. Treat tmp_path as config/."""

    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", tmp_path)


@pytest.fixture
def isolated_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Pin resolve_linkedin_state_dir to tmp_path so the resume tests
    don't pollute the real output/state/ tree."""

    state_dir = tmp_path / "state" / "linkedin" / "resume-key"
    import shared.output_paths as output_paths_mod

    monkeypatch.setattr(
        output_paths_mod,
        "resolve_linkedin_state_dir",
        lambda *, brief_path, **kwargs: state_dir,
    )
    monkeypatch.setattr(
        output_paths_mod,
        "derive_brief_id",
        lambda *, brief_path, **kwargs: "resume-key",
    )
    return state_dir


def test_resume_with_empty_progress_raises_no_pending_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    """progress.json with no queued strings ⇒ has_pending_work=False ⇒
    NoPendingWorkError ⇒ no Popen invocation."""

    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    (isolated_state_dir / "progress.json").write_text(
        json.dumps({"strings": []})
    )

    _StubPopenWithSidecar.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    with pytest.raises(NoPendingWorkError) as excinfo:
        _spawn_linkedin_worker(req, mode="resume")

    assert excinfo.value.state_dir == str(isolated_state_dir)
    assert _StubPopenWithSidecar.instances == [], (
        "no Popen call should fire when there's nothing to resume"
    )


def test_resume_with_queued_string_proceeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    (isolated_state_dir / "progress.json").write_text(
        json.dumps({"strings": [{"id": 1, "status": "queued"}]})
    )

    _StubPopenWithSidecar.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    result = _spawn_linkedin_worker(req, mode="resume")

    assert result.pid == 4242
    assert len(_StubPopenWithSidecar.instances) == 1
    argv = _StubPopenWithSidecar.instances[0].argv
    assert argv[-2:] == ["--mode", "resume"]


def test_resume_with_missing_progress_proceeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    """Missing progress.json ⇒ has_pending_work=None (unknown) ⇒
    proceed. Preserves the orchestrator's bias toward attempting resume
    on missing data; the read model would otherwise mask a recoverable
    edge case where progress.json hasn't been written yet."""

    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately do NOT write progress.json.

    _StubPopenWithSidecar.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    result = _spawn_linkedin_worker(req, mode="resume")

    assert result.pid == 4242
    assert len(_StubPopenWithSidecar.instances) == 1


def test_fresh_launch_without_generated_brief_does_not_check_pending_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    """Fresh launches still proceed over completed work when the state dir
    does not carry a generated brief artifact."""

    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    (isolated_state_dir / "progress.json").write_text(
        json.dumps({"strings": []})
    )

    _StubPopenWithSidecar.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    result = _spawn_linkedin_worker(req, mode="fresh")
    assert result.pid == 4242


def test_fresh_launch_with_generated_brief_and_pending_work_requires_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    """Generated brief + pending work means flagless fresh is unsafe:
    reject before Popen so the API cannot return a fake success."""

    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    (isolated_state_dir / "preflight_v2_brief.json").write_text("{}")
    (isolated_state_dir / "progress.json").write_text(
        json.dumps({"strings": [{"id": 1, "status": "queued"}]})
    )

    _StubPopenWithSidecar.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    with pytest.raises(
        cloris_api._monolith.FreshOverResumableStateError
    ) as excinfo:
        _spawn_linkedin_worker(req, mode="fresh")

    assert excinfo.value.state_dir == str(isolated_state_dir)
    assert _StubPopenWithSidecar.instances == [], (
        "no Popen call should fire without explicit fresh consent"
    )


def test_fresh_launch_with_execution_plan_and_pending_work_requires_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    (isolated_state_dir / "execution_plan.json").write_text("{}")
    (isolated_state_dir / "progress.json").write_text(
        json.dumps({"strings": [{"id": 1, "status": "queued"}]})
    )

    _StubPopenWithSidecar.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    with pytest.raises(
        cloris_api._monolith.FreshOverResumableStateError
    ) as excinfo:
        _spawn_linkedin_worker(req, mode="fresh")

    assert excinfo.value.state_dir == str(isolated_state_dir)
    assert _StubPopenWithSidecar.instances == []


def test_certification_fresh_launch_cannot_bypass_pending_work_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    (isolated_state_dir / "execution_plan.json").write_text("{}")
    store = RuntimeStateStore(isolated_state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="resume-key",
        output_dir=str(isolated_state_dir),
        mode="fresh",
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="resume-key",
        kind="linkedin_string",
        source_unit_id="1",
        display_name="queued search",
        ordering_index=0,
        status="queued",
    )
    initial_run_count = len(store.list_runs(source="linkedin", brief_id="resume-key"))
    monkeypatch.setenv("CLORIS_CERTIFY_STUB_RUNNERS", "true")

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    with pytest.raises(cloris_api._monolith.FreshOverResumableStateError):
        _spawn_linkedin_worker(req, mode="fresh")

    assert len(store.list_runs(source="linkedin", brief_id="resume-key")) == initial_run_count
    assert not (isolated_state_dir / "worker.json").exists()


def test_fresh_launch_with_generated_brief_and_unknown_pending_work_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    """Unknown resumability cannot authorize destructive regeneration."""

    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    (isolated_state_dir / "preflight_v2_brief.json").write_text("{}")
    # Deliberately do NOT write progress.json.

    _StubPopenWithSidecar.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    with pytest.raises(
        cloris_api._monolith.FreshOverResumableStateError
    ) as excinfo:
        _spawn_linkedin_worker(req, mode="fresh")

    assert excinfo.value.state_dir == str(isolated_state_dir)
    assert _StubPopenWithSidecar.instances == []


def test_fresh_launch_with_force_fresh_threads_fresh_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_state_dir: Path,
) -> None:
    brief_path = _make_brief(tmp_path)
    isolated_state_dir.mkdir(parents=True, exist_ok=True)
    (isolated_state_dir / "execution_plan.json").write_text("{}")
    (isolated_state_dir / "progress.json").write_text(
        json.dumps({"strings": [{"id": 1, "status": "queued"}]})
    )

    _StubPopenWithSidecar.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)

    req = LaunchLinkedInRequest(brief_path=str(brief_path), force_fresh=True)
    result = _spawn_linkedin_worker(req, mode="fresh")

    assert result.pid == 4242
    assert len(_StubPopenWithSidecar.instances) == 1
    argv = _StubPopenWithSidecar.instances[0].argv
    assert argv[-1] == "--fresh"
    assert argv.count("--fresh") == 1
    assert "--resume" not in argv


def test_resume_route_returns_422_on_no_pending_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route maps NoPendingWorkError to HTTP 422 with a clean error
    body so the frontend can surface "nothing to resume" rather than the
    fake-success message that fired before Phase 1.3."""

    def fake_resume(req: Any) -> None:
        raise NoPendingWorkError(state_dir="/tmp/state/linkedin/key")

    monkeypatch.setattr(cloris_api._monolith, "resume_linkedin_worker", fake_resume)

    client = TestClient(create_app())
    response = client.post(
        "/api/resume/linkedin", json={"brief_path": "/tmp/brief.json"}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "no_pending_work"
    assert detail["state_dir"] == "/tmp/state/linkedin/key"
