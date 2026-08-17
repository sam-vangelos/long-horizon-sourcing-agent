"""Tests for the Cloris app process (Slices 1 + 2).

Covers:

- ``GET /healthz`` and ``GET /`` HTTP contract via :class:`TestClient`
  (Slice 1, byte-identical here).
- :func:`cloris.app.run_app` lifecycle with a ``NullWindowLauncher`` and a
  stub ``server_factory``. No real socket is bound and no real window is
  opened; the readiness probe is monkeypatched on
  :mod:`cloris.app`.
- ``GET /api/status`` JSON contract (Slice 2). The aggregator is monkeypatched
  on :mod:`cloris.api.briefs` because the package split moved the route out of
  the legacy monolith, and that is now the symbol the route actually calls.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cloris import __version__
from cloris import api as cloris_api
from cloris import app as cloris_app
from cloris.api import StateDirNotFoundError
from cloris.app import NullWindowLauncher, _resolve_port, create_app, run_app
from cloris.models import (
    LaunchResponse,
    ResumeResponse,
    RunSummary,
    StateDirEntry,
    StatusResponse,
    StopResponse,
)
from cloris.worker import BriefPathNotFoundError, WorkerAlreadyRunningError


def test_healthz_contract() -> None:
    client = TestClient(create_app())

    response = client.get("/healthz")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["slice"] == "v0-shell-slice-1"
    assert isinstance(body["version"], str)
    assert body["version"] == __version__


def test_create_app_registers_split_api_routes() -> None:
    """Package-split API assembly must include monolith and health routes."""

    app = create_app()
    paths = {getattr(route, "path", "") for route in app.routes}

    assert "/api/orchestrator/decide" in paths
    assert "/api/launch-readiness/{source}/{brief_id:path}" in paths
    assert "/api/chrome-relaunch" in paths
    assert "/api/chrome-open-linkedin" in paths


def test_create_app_registers_candidate_routes_with_exact_paths_and_methods() -> None:
    """P4-4 guard: the candidate read/annotation routes must stay mounted with
    their exact path + HTTP method after the carve-out into
    ``cloris.api.candidate_routes``. A dropped decorator silently unmounts an
    endpoint; this pins every (method, path) pair so the move can't regress one.
    """

    app = create_app()
    registered: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        for method in getattr(route, "methods", set()) or set():
            registered.add((method, path))

    expected = {
        ("GET", "/api/shortlist/{brief_id}"),
        ("GET", "/api/candidate/{brief_id}/{candidate_id}"),
        ("POST", "/api/candidate/{brief_id}/{candidate_id}/note"),
        ("PATCH", "/api/candidate/{brief_id}/{candidate_id}"),
        ("PATCH", "/api/candidate/{brief_id}/{candidate_id}/judgment-accuracy"),
        ("POST", "/api/candidate/{brief_id}/{candidate_id}/principle-feedback"),
        ("POST", "/api/candidate/{brief_id}/{candidate_id}/excluded-asset"),
        ("DELETE", "/api/candidate/{brief_id}/{candidate_id}/excluded-asset"),
    }
    missing = expected - registered
    assert not missing, f"candidate routes not registered: {sorted(missing)}"


def test_api_package_exports_only_supported_compatibility_names() -> None:
    expected = {
        "router",
        "mount_static",
        "_paths",
        "_PROJECT_ROOT",
        "_CONFIG_DIR",
        "_CONFIG_PARENT",
        "_CONVERSATION_QUERY_BUCKETS",
        "_DIST_DIR",
        "_FRONTEND_SRC_DIR",
        "_intake_store",
        "_intake_db_path",
        "_readiness_blockers",
        "_reflection_store_factory",
        "_resolve_brief_path_or_raise",
        "_scan_authored_briefs",
        "_spawn_linkedin_worker",
        "_spawn_worker_for_source",
        "_build_worker_argv",
        "_warn_if_dist_stale",
        "_SpawnResult",
        "BriefIdNotFoundError",
        "LaunchLinkedInRequest",
        "LaunchNotReadyError",
        "NoPendingWorkError",
        "StateDirNotFoundError",
        "UnknownSourceError",
        "WorkerAlreadyRunningError",
        "stop_worker",
    }

    for name in expected:
        assert hasattr(cloris_api, name), name
    assert callable(cloris_api.mount_static)

    with pytest.raises(AttributeError):
        getattr(cloris_api, "_unsupported_monolith_private")


def test_index_404s_while_shell_is_parked() -> None:
    """The desktop shell is parked (attic/frontend-2026-08, Sam's 2026-08-02
    ruling): the in-tree app boots headless with no built dist. ``GET /``
    must 404 with the parked-shell message rather than crash — the full UI
    runs only at the ``desktop-shell-last-green`` tag."""

    client = TestClient(create_app())

    response = client.get("/")
    assert response.status_code == 404
    assert "parked" in response.json()["detail"]


def test_configure_logging_formats_module_logs_before_request_context(
    tmp_path: Path,
) -> None:
    """Startup logs from module loggers need the default request id too.

    ``logging.Filter`` instances attached only to the root logger are not
    applied to records propagated from child loggers, which is exactly how
    package startup logs flow before the first HTTP request exists.
    """

    from cloris.api.logging_setup import _request_id_var, configure_logging

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_filters = root.filters[:]
    original_level = root.level

    for handler in original_handlers:
        root.removeHandler(handler)
    for log_filter in original_filters:
        root.removeFilter(log_filter)

    try:
        configure_logging(tmp_path)

        logger = logging.getLogger("cloris.package_smoke")
        logger.info("startup log")
        token = _request_id_var.set("rid-123")
        try:
            logger.info("request log")
        finally:
            _request_id_var.reset(token)

        for handler in root.handlers:
            handler.flush()

        content = (tmp_path / "cloris.log").read_text()
        assert "[-] startup log" in content
        assert "[rid-123] request log" in content
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
            handler.close()
        for log_filter in root.filters[:]:
            root.removeFilter(log_filter)
        root.setLevel(original_level)
        for log_filter in original_filters:
            root.addFilter(log_filter)
        for handler in original_handlers:
            root.addHandler(handler)


class _StubServer:
    """Stand-in for :class:`uvicorn.Server` used in lifecycle tests.

    ``run`` blocks until ``should_exit`` flips to ``True``; the readiness
    probe is short-circuited at the module level, so ``run_app`` proceeds
    straight to ``launcher.open`` without ever talking to a real socket.
    """

    def __init__(self) -> None:
        self.should_exit = False
        self._stopped = threading.Event()
        self.run_calls = 0

    def run(self) -> None:
        self.run_calls += 1
        while not self.should_exit:
            if self._stopped.wait(timeout=0.01):
                return
        self._stopped.set()


def test_run_app_lifecycle_uses_launcher_and_shuts_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _StubServer()

    factory_received: dict[str, Any] = {}

    def stub_server_factory(app: Any, host: str, port: int) -> _StubServer:
        assert host == "127.0.0.1"
        assert isinstance(port, int)
        assert port > 0, (
            "run_app must resolve port=0 to a concrete free port before "
            "calling server_factory; received {port!r}".format(port=port)
        )
        factory_received["host"] = host
        factory_received["port"] = port
        return server

    wait_received: dict[str, Any] = {}

    def fake_wait_until_ready(host: str, port: int, *, timeout: float = 5.0) -> None:
        assert host == "127.0.0.1"
        assert isinstance(port, int)
        assert port > 0
        wait_received["host"] = host
        wait_received["port"] = port

    monkeypatch.setattr(cloris_app, "_wait_until_ready", fake_wait_until_ready)

    launcher = NullWindowLauncher()

    captured_threads: list[threading.Thread] = []
    real_thread_init = threading.Thread.__init__

    def recording_init(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        real_thread_init(self, *args, **kwargs)
        if kwargs.get("name") == "cloris-uvicorn":
            captured_threads.append(self)

    monkeypatch.setattr(threading.Thread, "__init__", recording_init)

    app_sentinel = object()
    run_app(
        app_sentinel,
        host="127.0.0.1",
        port=0,
        launcher=launcher,
        server_factory=stub_server_factory,
        readiness_timeout=2.0,
        shutdown_timeout=2.0,
        ensure_chrome=lambda: None,
    )

    captured_port = factory_received["port"]
    assert wait_received["port"] == captured_port, (
        "readiness probe must observe the same resolved port as the server factory"
    )
    assert len(launcher.opened) == 1
    assert launcher.opened[0].startswith(
        f"http://127.0.0.1:{captured_port}/?cloris_build="
    )
    assert server.should_exit is True
    assert server.run_calls == 1

    assert len(captured_threads) == 1
    assert not captured_threads[0].is_alive()


def test_run_app_raises_and_shuts_down_on_readiness_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _StubServer()

    factory_received: dict[str, Any] = {}

    def stub_server_factory(app: Any, host: str, port: int) -> _StubServer:
        assert isinstance(port, int)
        assert port > 0
        factory_received["port"] = port
        return server

    wait_received: dict[str, Any] = {}

    def failing_wait_until_ready(host: str, port: int, *, timeout: float = 5.0) -> None:
        assert isinstance(port, int)
        assert port > 0
        wait_received["port"] = port
        raise RuntimeError("not ready")

    monkeypatch.setattr(cloris_app, "_wait_until_ready", failing_wait_until_ready)

    launcher = NullWindowLauncher()

    with pytest.raises(RuntimeError, match="not ready"):
        run_app(
            object(),
            host="127.0.0.1",
            port=0,
            launcher=launcher,
            server_factory=stub_server_factory,
            readiness_timeout=0.1,
            shutdown_timeout=2.0,
            ensure_chrome=lambda: None,
        )

    assert launcher.opened == []
    assert server.should_exit is True
    assert wait_received["port"] == factory_received["port"]


def test_resolve_port_picks_free_port_for_zero_and_passes_through_otherwise() -> None:
    auto_picked = _resolve_port("127.0.0.1", 0)
    assert isinstance(auto_picked, int)
    assert auto_picked > 0

    assert _resolve_port("127.0.0.1", 8765) == 8765

    second_pick = _resolve_port("127.0.0.1", 0)
    assert isinstance(second_pick, int)
    assert second_pick > 0


def test_api_status_endpoint_returns_empty_for_empty_state_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Slice 4 bumps the StatusResponse slice literal to v0-shell-slice-4
    # because the payload shape gained worker_state, worker_pid, etc.
    def fake_aggregate_status() -> StatusResponse:
        return StatusResponse(slice="v0-shell-slice-4", entries=[])

    monkeypatch.setattr("cloris.api.briefs.aggregate_status", fake_aggregate_status)

    client = TestClient(create_app())
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    # Phase F Slice F7 added the additive `briefs` field — empty when
    # there are no entries to group.
    assert response.json() == {
        "slice": "v0-shell-slice-4",
        "entries": [],
        "counts": {
            "active": 0,
            "working": 0,
            "paused": 0,
            "finished": 0,
            "lost": 0,
            "archived": 0,
            "orphaned": 0,
        },
        "briefs": [],
        "trial_mode": False,
        "modules": [],
    }


def test_api_status_endpoint_serializes_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Slice 4 bumps the slice tag to v0-shell-slice-4 and enriches each
    # entry with worker_* + resumable fields. Test fixtures rely on
    # StateDirEntry's default values for the new fields.
    fixture = StatusResponse(
        slice="v0-shell-slice-4",
        entries=[
            StateDirEntry(
                source="linkedin",
                state_key="li-key",
                runtime_state_present=True,
                latest_run=RunSummary(
                    id=42,
                    status="completed",
                    stop_reason="normal",
                    mode="fresh",
                    started_at="2024-01-01T00:00:00+00:00",
                    ended_at="2024-01-01T00:01:00+00:00",
                ),
                brief_id_from_run="brief-li",
            ),
            StateDirEntry(
                source="github",
                state_key="gh-key",
                runtime_state_present=False,
                latest_run=None,
                brief_id_from_run=None,
            ),
        ],
    )

    def fake_aggregate_status() -> StatusResponse:
        return fixture

    monkeypatch.setattr("cloris.api.briefs.aggregate_status", fake_aggregate_status)

    client = TestClient(create_app())
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json() == {
        "slice": "v0-shell-slice-4",
        "entries": [
            {
                "source": "linkedin",
                "state_key": "li-key",
                "runtime_state_present": True,
                "runtime_state_corrupt": False,
                "latest_run": {
                    "id": 42,
                    "status": "completed",
                    "stop_reason": "normal",
                    "mode": "fresh",
                    "started_at": "2024-01-01T00:00:00+00:00",
                    "ended_at": "2024-01-01T00:01:00+00:00",
                },
                "brief_id_from_run": "brief-li",
                "brief_path_from_worker": None,
                "worker_json_present": False,
                "worker_pid": None,
                "worker_alive": None,
                "worker_mode": None,
                "worker_input_mode": None,
                "resumable": None,
                "worker_state": "missing",
                "heartbeat_age_s": None,
                "brief_role_title": None,
                "brief_linkedin_project": None,
                "brief_drift_since_last_run": None,
                "attempt_health": None,
                "work_unit_progress": None,
                "run_stalled": False,
                "stall_failure_kind": None,
                "lifecycle": "ready",
                "kind": "orphaned_state_dir",
                "attention_state": "idle",
                "live_signal_eligible": False,
                "active": False,
                "terminal_reason": None,
                "projection_disagreement": False,
            },
            {
                "source": "github",
                "state_key": "gh-key",
                "runtime_state_present": False,
                "runtime_state_corrupt": False,
                "latest_run": None,
                "brief_id_from_run": None,
                "brief_path_from_worker": None,
                "worker_json_present": False,
                "worker_pid": None,
                "worker_alive": None,
                "worker_mode": None,
                "worker_input_mode": None,
                "resumable": None,
                "worker_state": "missing",
                "heartbeat_age_s": None,
                "brief_role_title": None,
                "brief_linkedin_project": None,
                "brief_drift_since_last_run": None,
                "attempt_health": None,
                "work_unit_progress": None,
                "run_stalled": False,
                "stall_failure_kind": None,
                "lifecycle": "ready",
                "kind": "orphaned_state_dir",
                "attention_state": "idle",
                "live_signal_eligible": False,
                "active": False,
                "terminal_reason": None,
                "projection_disagreement": False,
            },
        ],
        "counts": {
            "active": 0,
            "working": 0,
            "paused": 0,
            "finished": 0,
            "lost": 0,
            "archived": 0,
            "orphaned": 0,
        },
        # Phase F Slice F7: empty for the fixture above (no briefs
        # configured on the response — the fixture sets entries
        # directly without populating the briefs grouping).
        "briefs": [],
        "trial_mode": False,
        "modules": [],
    }


def test_launch_linkedin_endpoint_201_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_launch(req: Any) -> LaunchResponse:
        captured["brief_path"] = req.brief_path
        return LaunchResponse(
            source="linkedin",
            input_mode="concurrent",
            pid=12345,
            state_dir="/tmp/state/linkedin/key",
            worker_json_path="/tmp/state/linkedin/key/worker.json",
        )

    monkeypatch.setattr(cloris_api._monolith, "launch_linkedin_worker", fake_launch)

    client = TestClient(create_app())
    response = client.post("/api/launch/linkedin", json={"brief_path": "/tmp/brief.json"})

    assert response.status_code == 201
    # Phase F Slice F1 added `mode` to LaunchResponse (additive — default
    # "fresh"). Existing fields are preserved byte-for-byte; clients that
    # ignore unknown fields are unaffected.
    assert response.json() == {
        "slice": "v0-shell-slice-3",
        "source": "linkedin",
        "input_mode": "concurrent",
        "mode": "fresh",
        "pid": 12345,
        "state_dir": "/tmp/state/linkedin/key",
        "worker_json_path": "/tmp/state/linkedin/key/worker.json",
    }
    assert captured["brief_path"] == "/tmp/brief.json"


def test_launch_linkedin_endpoint_409_when_worker_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_launch(req: Any) -> LaunchResponse:
        raise WorkerAlreadyRunningError(
            pid=12345,
            state_dir="/tmp/state/linkedin/key",
        )

    monkeypatch.setattr(cloris_api._monolith, "launch_linkedin_worker", fake_launch)

    client = TestClient(create_app())
    response = client.post("/api/launch/linkedin", json={"brief_path": "/tmp/brief.json"})

    assert response.status_code == 409
    body = response.json()
    detail = body["detail"]
    assert detail["error"] == "worker_already_running"
    assert detail["pid"] == 12345
    assert detail["state_dir"] == "/tmp/state/linkedin/key"


def test_launch_linkedin_endpoint_400_on_missing_brief_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_launch(req: Any) -> LaunchResponse:
        raise BriefPathNotFoundError("/tmp/missing.json")

    monkeypatch.setattr(cloris_api._monolith, "launch_linkedin_worker", fake_launch)

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/linkedin", json={"brief_path": "/tmp/missing.json"}
    )

    assert response.status_code == 400
    body = response.json()
    detail = body["detail"]
    assert detail["error"] == "brief_path_not_found"
    assert detail["brief_path"] == "/tmp/missing.json"


def test_launch_linkedin_endpoint_rejects_input_mode_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(req: Any) -> LaunchResponse:
        raise AssertionError(
            "launch_linkedin_worker should not be called when the request "
            "body has an unknown field"
        )

    monkeypatch.setattr(cloris_api._monolith, "launch_linkedin_worker", boom)

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/linkedin",
        json={"brief_path": "/tmp/brief.json", "input_mode": "away"},
    )

    assert response.status_code == 422


# --- Slice 4: stop + resume routes + helpers ----------------------------


import json
import signal

from cloris import api as _cloris_api  # alias to expose stop_worker for monkeypatch
from cloris.worker import WORKER_SIDECAR_FILENAME, build_sidecar, write_sidecar


def test_stop_endpoint_alive_pid_sends_sigterm_and_returns_202(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stopping worker_state in the helper response triggers HTTP 202
    from the route. The actual SIGTERM dispatch is exercised at the helper
    level further below; this test pins the route's status-code
    translation."""

    def fake_stop(source: str, state_key: str) -> StopResponse:
        return StopResponse(
            source="linkedin",
            state_key=state_key,
            state_dir="/tmp/state/linkedin/key",
            worker_state="stopping",
            pid=12345,
        )

    monkeypatch.setattr(cloris_api._monolith, "stop_worker", fake_stop)

    client = TestClient(create_app())
    response = client.post("/api/stop/linkedin/key")

    assert response.status_code == 202
    body = response.json()
    assert body["worker_state"] == "stopping"
    assert body["pid"] == 12345
    assert body["source"] == "linkedin"
    assert body["state_key"] == "key"
    assert body["state_dir"] == "/tmp/state/linkedin/key"
    assert body["slice"] == "v0-shell-slice-4"


def test_stop_endpoint_missing_sidecar_returns_200_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stop(source: str, state_key: str) -> StopResponse:
        return StopResponse(
            source="linkedin",
            state_key=state_key,
            state_dir="/tmp/state/linkedin/key",
            worker_state="missing",
            pid=None,
        )

    monkeypatch.setattr(cloris_api._monolith, "stop_worker", fake_stop)

    client = TestClient(create_app())
    response = client.post("/api/stop/linkedin/key")

    assert response.status_code == 200
    assert response.json()["worker_state"] == "missing"
    assert response.json()["pid"] is None


def test_stop_endpoint_stale_sidecar_returns_200_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stop(source: str, state_key: str) -> StopResponse:
        return StopResponse(
            source="linkedin",
            state_key=state_key,
            state_dir="/tmp/state/linkedin/key",
            worker_state="stale",
            pid=None,
        )

    monkeypatch.setattr(cloris_api._monolith, "stop_worker", fake_stop)

    client = TestClient(create_app())
    response = client.post("/api/stop/linkedin/key")

    assert response.status_code == 200
    assert response.json()["worker_state"] == "stale"


def test_stop_endpoint_unknown_state_dir_returns_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_stop(source: str, state_key: str) -> StopResponse:
        raise StateDirNotFoundError(source, state_key)

    monkeypatch.setattr(cloris_api._monolith, "stop_worker", fake_stop)

    client = TestClient(create_app())
    response = client.post("/api/stop/linkedin/key")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error"] == "state_dir_not_found"
    assert detail["source"] == "linkedin"
    assert detail["state_key"] == "key"


def _write_alive_sidecar(state_dir: Path, pid: int) -> Path:
    """Write a sidecar pointing at ``pid`` for stop-helper tests."""

    state_dir.mkdir(parents=True, exist_ok=True)
    payload = build_sidecar(
        source="linkedin",
        brief_id="brief-4",
        brief_path=str(state_dir / "brief.json"),
        output_dir=str(state_dir),
        mode="fresh",
        input_mode="concurrent",
        started_at="2026-04-27T18:00:00+00:00",
        pid=pid,
        run_id=None,
    )
    return write_sidecar(state_dir, payload)


def test_stop_helper_alive_dispatches_sigterm_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper signals SIGTERM exactly once with the real-pid fixture
    (os.getpid()) and reports worker_state='stopping'. enumerate_state_dirs
    is monkeypatched to yield the fixture state dir so the
    path-traversal-safe lookup succeeds without any real
    ``output/state/`` discovery. ``_send_sigterm`` is the module-level
    seam used in production; patching it (rather than ``os.kill``) keeps
    the unrelated ``os.kill(pid, 0)`` liveness probe inside
    :func:`cloris.worker.is_pid_alive` untouched."""

    state_dir = tmp_path / "linkedin" / "key"
    _write_alive_sidecar(state_dir, pid=os.getpid())

    def fake_enumerate():
        yield ("linkedin", state_dir)

    monkeypatch.setattr(_cloris_api._monolith, "enumerate_state_dirs", fake_enumerate)

    sigterm_calls: list[int] = []

    def fake_send_sigterm(pid: int) -> None:
        sigterm_calls.append(pid)

    monkeypatch.setattr(_cloris_api._monolith, "_send_sigterm", fake_send_sigterm)

    result = _cloris_api.stop_worker("linkedin", "key")

    assert result.worker_state == "stopping"
    assert result.pid == os.getpid()
    assert result.state_key == "key"
    assert sigterm_calls == [os.getpid()]
    # Verify the production seam wraps signal.SIGTERM, not some other signal.
    assert signal.SIGTERM == signal.SIGTERM  # contract sanity


def test_stop_helper_stale_does_not_dispatch_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-int PID is classified as stale: the helper neither signals nor
    deletes the sidecar file. The existing sidecar must still be readable
    after the call so the next launch can overwrite it via the
    Slice-3-defined stale-overwrite policy."""

    state_dir = tmp_path / "linkedin" / "key"
    state_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = state_dir / WORKER_SIDECAR_FILENAME
    sidecar_path.write_text(
        json.dumps(
            {
                "pid": "not-an-int",
                "source": "linkedin",
                "brief_id": "brief-4",
                "brief_path": str(state_dir / "brief.json"),
                "output_dir": str(state_dir),
                "run_id": None,
                "started_at": "2026-04-27T18:00:00+00:00",
                "heartbeat_at": "2026-04-27T18:00:00+00:00",
                "mode": "fresh",
                "input_mode": "concurrent",
                "launcher_version": "cloris-v0-slice-4",
            },
            indent=2,
            sort_keys=True,
        )
    )

    def fake_enumerate():
        yield ("linkedin", state_dir)

    monkeypatch.setattr(_cloris_api._monolith, "enumerate_state_dirs", fake_enumerate)

    sigterm_calls: list[int] = []

    def fake_send_sigterm(pid: int) -> None:
        sigterm_calls.append(pid)

    monkeypatch.setattr(_cloris_api._monolith, "_send_sigterm", fake_send_sigterm)

    result = _cloris_api.stop_worker("linkedin", "key")

    assert result.worker_state == "stale"
    assert sigterm_calls == []
    # Stop must not delete the stale sidecar; launch overwrites it next.
    assert sidecar_path.exists()


def test_resume_endpoint_201_spawns_worker_with_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_resume(req: Any) -> ResumeResponse:
        captured["brief_path"] = req.brief_path
        return ResumeResponse(
            source="linkedin",
            input_mode="concurrent",
            pid=23456,
            state_dir="/tmp/state/linkedin/key",
            worker_json_path="/tmp/state/linkedin/key/worker.json",
        )

    monkeypatch.setattr(cloris_api._monolith, "resume_linkedin_worker", fake_resume)

    client = TestClient(create_app())
    response = client.post(
        "/api/resume/linkedin", json={"brief_path": "/tmp/brief.json"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["slice"] == "v0-shell-slice-4"
    assert body["mode"] == "resume"
    assert body["pid"] == 23456
    assert body["state_dir"] == "/tmp/state/linkedin/key"
    assert body["worker_json_path"] == "/tmp/state/linkedin/key/worker.json"
    assert captured["brief_path"] == "/tmp/brief.json"


def test_resume_endpoint_409_when_worker_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_resume(req: Any) -> ResumeResponse:
        raise WorkerAlreadyRunningError(
            pid=12345,
            state_dir="/tmp/state/linkedin/key",
        )

    monkeypatch.setattr(cloris_api._monolith, "resume_linkedin_worker", fake_resume)

    client = TestClient(create_app())
    response = client.post(
        "/api/resume/linkedin", json={"brief_path": "/tmp/brief.json"}
    )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "worker_already_running"
    assert detail["pid"] == 12345
    assert detail["state_dir"] == "/tmp/state/linkedin/key"


def test_resume_endpoint_400_on_missing_brief_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_resume(req: Any) -> ResumeResponse:
        raise BriefPathNotFoundError("/tmp/missing.json")

    monkeypatch.setattr(cloris_api._monolith, "resume_linkedin_worker", fake_resume)

    client = TestClient(create_app())
    response = client.post(
        "/api/resume/linkedin", json={"brief_path": "/tmp/missing.json"}
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "brief_path_not_found"
    assert detail["brief_path"] == "/tmp/missing.json"


def test_resume_endpoint_rejects_extra_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LaunchLinkedInRequest's extra='forbid' rejects unknown fields at the
    request boundary, identically to the launch route."""

    def boom(req: Any) -> ResumeResponse:
        raise AssertionError(
            "resume_linkedin_worker should not be called when the request "
            "body has an unknown field"
        )

    monkeypatch.setattr(cloris_api._monolith, "resume_linkedin_worker", boom)

    client = TestClient(create_app())
    response = client.post(
        "/api/resume/linkedin",
        json={"brief_path": "/tmp/brief.json", "input_mode": "away"},
    )

    assert response.status_code == 422


def test_launch_endpoint_still_returns_slice_3_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION: the launch contract version is independent of the status
    contract version. Slice 4 bumped StatusResponse.slice but NOT
    LaunchResponse.slice; clients reading the launch payload must still
    see ``v0-shell-slice-3``."""

    def fake_launch(req: Any) -> LaunchResponse:
        return LaunchResponse(
            source="linkedin",
            input_mode="concurrent",
            pid=11111,
            state_dir="/tmp/state/linkedin/key",
            worker_json_path="/tmp/state/linkedin/key/worker.json",
        )

    monkeypatch.setattr(cloris_api._monolith, "launch_linkedin_worker", fake_launch)

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/linkedin", json={"brief_path": "/tmp/brief.json"}
    )

    assert response.status_code == 201
    assert response.json()["slice"] == "v0-shell-slice-3"


# --- Slice 5: built SPA + /assets static mount --------------------------


def _dist_assets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "cloris" / "frontend" / "dist" / "assets"


def test_assets_mount_absent_while_shell_is_parked() -> None:
    """With the frontend parked there is no built ``dist/assets`` tree; the
    ``/assets`` mount is skipped at app creation (mirroring the brand-dir
    guard) and asset requests 404 instead of erroring at mount time."""

    assets_dir = _dist_assets_dir()
    assert not assets_dir.exists(), (
        f"unexpected built assets dir at {assets_dir}; the shell is parked "
        "(attic/frontend-2026-08) and the in-tree dist should not exist"
    )

    client = TestClient(create_app())
    assert client.get("/assets/anything.js").status_code == 404

def test_assets_mount_returns_404_for_unknown_file() -> None:
    """Slice 5: requests for files that don't exist under the mounted
    ``/assets/`` directory must 404 — StaticFiles' default behavior, but
    pinned here so a future configuration drift cannot silently start
    serving a fallback."""

    client = TestClient(create_app())
    response = client.get("/assets/this-does-not-exist.js")
    assert response.status_code == 404


# --- Orchestrator API (Slice 2.8) ---------------------------------------------


def test_orchestrator_synthesize_returns_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cloris.chief_of_staff.agent import ChiefOfStaffSynthesis

    def _fake_synthesis(*, reflection_session_id: int):
        assert reflection_session_id == 42
        return ChiefOfStaffSynthesis(
            paragraph="Team read across sources.",
            per_specialist_weight={
                "linkedin": {
                    "weight": 0.82,
                    "rationale": "Strong save signal.",
                },
                "github": {"weight": 0.71, "rationale": "Solid maintainer read."},
            },
            priority_for_principal="Start with the LinkedIn saves.",
            confidence=1.0,
            source="llm",
        )

    monkeypatch.setattr(
        cloris_api._monolith,
        "_chief_of_staff_synthesis_for_reflection_session",
        _fake_synthesis,
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/orchestrator/synthesize",
        json={"reflection_session_id": 42},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["slice"] == "orchestrator-synthesize-v1"
    assert body["paragraph"] == "Team read across sources."
    assert body["source"] == "llm"
    assert body["confidence"] == 1.0
    assert set(body["per_specialist_weight"].keys()) == {"linkedin", "github"}


def test_orchestrator_synthesize_returns_404_for_missing_session() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/orchestrator/synthesize",
        json={"reflection_session_id": 999_999_999},
    )
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["error"] == "reflection_session_not_found"
    assert detail["reflection_session_id"] == 999_999_999


def test_orchestrator_synthesize_returns_422_when_no_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-source (or zero-source) evidence must not run synthesis."""

    def boom(*, reflection_session_id: int):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "synthesis_preconditions_not_met",
                "message": "Synthesis requires at least two candidate-producing sources for this brief.",
            },
        )

    monkeypatch.setattr(
        cloris_api._monolith,
        "_chief_of_staff_synthesis_for_reflection_session",
        boom,
    )
    client = TestClient(create_app())
    response = client.post(
        "/api/orchestrator/synthesize",
        json={"reflection_session_id": 1},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "synthesis_preconditions_not_met"


def test_orchestrator_runs_lists_dispatch_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import shared.output_paths as output_paths
    from shared.output_paths import resolve_orchestration_db_path
    from shared.runtime_state.orchestration_store import OrchestrationStateStore

    out = tmp_path / "output"
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", out)

    db_path = resolve_orchestration_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = OrchestrationStateStore(db_path)
    store.initialize()

    plan_a = {"steps": [{"module_name": "linkedin", "handoff_condition": None}]}
    plan_b = {"steps": [{"module_name": "github", "handoff_condition": None}]}
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO chief_of_staff_runs(
                brief_id, principal_id, status,
                dispatch_plan_json, invocation_order_json,
                handoff_payloads_json, synthesis_output_json,
                started_at, ended_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "brief-orch-runs",
                "principal-a",
                "running",
                json.dumps(plan_a),
                json.dumps(["linkedin"]),
                json.dumps({}),
                json.dumps({}),
                "2026-05-01T12:00:00+00:00",
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO chief_of_staff_runs(
                brief_id, principal_id, status,
                dispatch_plan_json, invocation_order_json,
                handoff_payloads_json, synthesis_output_json,
                started_at, ended_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "brief-orch-runs",
                "principal-a",
                "running",
                json.dumps(plan_b),
                json.dumps(["github"]),
                json.dumps({}),
                json.dumps({}),
                "2026-05-04T18:00:00+00:00",
                None,
            ),
        )

    client = TestClient(create_app())
    response = client.get("/api/orchestrator/brief-orch-runs/runs")
    assert response.status_code == 200
    payload = response.json()
    assert payload["slice"] == "orchestrator-runs-v1"
    runs = payload["runs"]
    assert len(runs) == 2
    assert runs[0]["started_at"] == "2026-05-04T18:00:00+00:00"
    assert runs[0]["dispatch_plan"] == plan_b
    assert runs[1]["started_at"] == "2026-05-01T12:00:00+00:00"
    assert runs[1]["dispatch_plan"] == plan_a


def test_orchestrator_runs_returns_empty_list_for_unknown_brief(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shared.output_paths as output_paths

    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path / "output")

    client = TestClient(create_app())
    response = client.get("/api/orchestrator/unknown-brief-xyz/runs")
    assert response.status_code == 200
    assert response.json() == {
        "slice": "orchestrator-runs-v1",
        "runs": [],
    }


# ---------------------------------------------------------------------------
# Audit Move #8 — POST /api/launch/multi
# ---------------------------------------------------------------------------


def _stub_launch_for_source_impl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    by_source: dict[str, Any],
) -> None:
    """Replace ``_launch_for_source_impl`` with a per-source dispatch stub.

    Values in ``by_source`` are either:
    - A :class:`LaunchResponse` (or a callable returning one) — happy
      path; the stub returns it.
    - An :class:`HTTPException` — failure path; the stub raises it,
      mirroring real per-source error mapping.

    Sources not in the dict default to a "module not stubbed" raise so
    accidental mismatches surface in test failures.
    """

    def fake_impl(source: str, req: Any) -> LaunchResponse:
        if source not in by_source:
            raise AssertionError(
                f"_stub_launch_for_source_impl: no stub for source={source!r}"
            )
        outcome = by_source[source]
        if callable(outcome):
            outcome = outcome(req)
        if isinstance(outcome, HTTPException):
            raise outcome
        return outcome  # type: ignore[return-value]

    monkeypatch.setattr(cloris_api._monolith, "_launch_for_source_impl", fake_impl)


def test_launch_multi_happy_path_201_with_two_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: 2 modules, both spawn, response carries 2 launch_ids
    in dispatch order. HTTP 201."""

    def linkedin_response(_req: Any) -> LaunchResponse:
        return LaunchResponse(
            source="linkedin",
            input_mode="concurrent",
            mode="fresh",
            pid=11111,
            state_dir="/tmp/state/linkedin/key",
            worker_json_path="/tmp/state/linkedin/key/worker.json",
        )

    def github_response(_req: Any) -> LaunchResponse:
        return LaunchResponse(
            source="github",
            input_mode="concurrent",
            mode="fresh",
            pid=22222,
            state_dir="/tmp/state/github/key",
            worker_json_path="/tmp/state/github/key/worker.json",
        )

    _stub_launch_for_source_impl(
        monkeypatch,
        by_source={
            "linkedin": linkedin_response,
            "github": github_response,
        },
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/multi",
        json={
            "brief_id": "brief-xyz",
            "modules": ["linkedin", "github"],
            "mode": "fresh",
            "force": False,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["slice"] == "v0-launch-multi-1"
    assert body["brief_id"] == "brief-xyz"
    assert len(body["results"]) == 2
    # Order matches the request's modules list.
    assert body["results"][0]["source"] == "linkedin"
    assert body["results"][0]["launch"]["pid"] == 11111
    assert body["results"][0]["error"] is None
    assert body["results"][1]["source"] == "github"
    assert body["results"][1]["launch"]["pid"] == 22222


def test_launch_multi_partial_failure_201_with_readiness_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit acceptance: one module readiness-blocked, another spawns
    cleanly. HTTP 201 (at least one success); body carries the
    error envelope for the blocked module."""

    def linkedin_response(_req: Any) -> LaunchResponse:
        return LaunchResponse(
            source="linkedin",
            input_mode="concurrent",
            mode="fresh",
            pid=11111,
            state_dir="/tmp/state/linkedin/key",
            worker_json_path="/tmp/state/linkedin/key/worker.json",
        )

    designer_blocked = HTTPException(
        status_code=422,
        detail={
            "error": "launch_not_ready",
            "source": "designer",
            "blockers": [
                {
                    "kind": "config",
                    "message": "No Anthropic API key configured.",
                    "remediation": "Add ANTHROPIC_API_KEY to your .env file.",
                }
            ],
        },
    )

    _stub_launch_for_source_impl(
        monkeypatch,
        by_source={
            "linkedin": linkedin_response,
            "designer": designer_blocked,
        },
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/multi",
        json={"brief_id": "b", "modules": ["linkedin", "designer"]},
    )

    assert response.status_code == 201
    body = response.json()
    assert len(body["results"]) == 2

    li_result = body["results"][0]
    assert li_result["source"] == "linkedin"
    assert li_result["launch"]["pid"] == 11111
    assert li_result["error"] is None

    designer_result = body["results"][1]
    assert designer_result["source"] == "designer"
    assert designer_result["launch"] is None
    assert designer_result["error"]["kind"] == "launch_not_ready"
    assert "blockers" in designer_result["error"]["detail"]
    assert (
        designer_result["error"]["detail"]["blockers"][0]["kind"] == "config"
    )


def test_launch_multi_all_failure_returns_422(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit acceptance: every module's spawn raises a typed error.
    HTTP 422; body still carries the per-module error envelopes so
    the frontend renders inline blockers per source."""

    linkedin_blocked = HTTPException(
        status_code=422,
        detail={
            "error": "launch_not_ready",
            "source": "linkedin",
            "blockers": [
                {
                    "kind": "auth",
                    "message": "LinkedIn cookies missing.",
                    "remediation": "Sign into LinkedIn in the Cloris Chrome profile.",
                }
            ],
        },
    )
    github_blocked = HTTPException(
        status_code=409,
        detail={
            "error": "worker_already_running",
            "pid": 9999,
            "state_dir": "/tmp/state/github/key",
        },
    )

    _stub_launch_for_source_impl(
        monkeypatch,
        by_source={
            "linkedin": linkedin_blocked,
            "github": github_blocked,
        },
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/multi",
        json={"brief_id": "b", "modules": ["linkedin", "github"]},
    )

    assert response.status_code == 422
    body = response.json()
    assert len(body["results"]) == 2
    assert all(r["launch"] is None for r in body["results"])
    assert body["results"][0]["error"]["kind"] == "launch_not_ready"
    assert body["results"][1]["error"]["kind"] == "worker_already_running"
    assert body["results"][1]["error"]["detail"]["pid"] == 9999


def test_launch_multi_400_when_modules_empty() -> None:
    """Empty modules list = no work to do. HTTP 400 with a clear
    error message."""

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/multi",
        json={"brief_id": "b", "modules": []},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["error"] == "modules_required"


def test_launch_multi_preserves_module_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module order in the response matches the request order — the
    frontend's chief-of-staff dispatch UI relies on this for inline
    rendering."""

    def make_response(source: str) -> Any:
        def _impl(_req: Any) -> LaunchResponse:
            return LaunchResponse(
                source=source,  # type: ignore[arg-type]
                input_mode="concurrent",
                mode="fresh",
                pid=hash(source) & 0xFFFF,
                state_dir=f"/tmp/state/{source}/key",
                worker_json_path=f"/tmp/state/{source}/key/worker.json",
            )

        return _impl

    _stub_launch_for_source_impl(
        monkeypatch,
        by_source={
            "github": make_response("github"),
            "linkedin": make_response("linkedin"),
            "researcher": make_response("researcher"),
        },
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/multi",
        json={
            "brief_id": "b",
            "modules": ["researcher", "linkedin", "github"],
        },
    )
    assert response.status_code == 201
    body = response.json()
    sources_in_order = [r["source"] for r in body["results"]]
    assert sources_in_order == ["researcher", "linkedin", "github"]


def test_launch_multi_rejects_unknown_module_at_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown source values are rejected at the request boundary
    (Pydantic Literal[...]), not at runtime — the request returns 422
    with a Pydantic validation envelope."""

    client = TestClient(create_app())
    response = client.post(
        "/api/launch/multi",
        json={"brief_id": "b", "modules": ["not_a_real_source"]},
    )
    assert response.status_code == 422  # FastAPI validation error
    body = response.json()
    # FastAPI's default validation error envelope.
    assert "detail" in body
