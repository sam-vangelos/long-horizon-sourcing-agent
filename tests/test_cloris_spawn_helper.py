"""Phase 1.2 regression tests for the launch/resume spawn helper.

After deduplication, both ``launch_linkedin_worker`` and
``resume_linkedin_worker`` route through ``_spawn_linkedin_worker``. The
single permitted difference between fresh and resume launches is a trailing
``["--mode", "resume"]`` argv pair, except explicit fresh consent which
adds ``--fresh``. Anything else is drift and must fail.

Phase 1.1 will wrap the read-sidecar + Popen critical section in a state-dir
launch lock; these tests pin the helper's behavior independently of locking,
so we can verify each layer in isolation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cloris import api as cloris_api
from cloris.api import (
    LaunchLinkedInRequest,
    _build_worker_argv,
    _spawn_linkedin_worker,
)


@pytest.fixture(autouse=True)
def _short_circuit_sidecar_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 1.1 wired wait_for_sidecar into _spawn_linkedin_worker; the
    stub Popen here never writes a sidecar, so the production wait would
    block for the full 2-second default. Replace the wait with a no-op
    so spawn-helper tests stay fast and don't depend on real worker
    side effects.

    The wait helper itself is exercised by test_cloris_launch_lock.
    """

    monkeypatch.setattr(cloris_api._monolith, "wait_for_sidecar", lambda *a, **k: True)


@pytest.fixture(autouse=True)
def _contain_briefs_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 containment: this suite writes brief fixtures under tmp_path, which
    the config/ containment boundary rejects. Treat tmp_path as config/."""

    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", tmp_path)


class _StubPopen:
    """Recording stand-in for :class:`subprocess.Popen`.

    Captures argv + the detach kwargs verbatim so tests can assert on them
    without ever creating a real subprocess. ``pid`` is a fixed sentinel.
    """

    instances: list["_StubPopen"] = []

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = list(argv)
        self.kwargs = dict(kwargs)
        self.pid = 99999
        _StubPopen.instances.append(self)


def _make_brief(tmp_path: Path) -> Path:
    """Write a minimal brief JSON the resolver will accept.

    The brief loader needs at least a role_title; resolve_linkedin_state_dir
    falls back to brief.id or brief_path.stem when linkedin_project_id is
    absent, so the stem-derived state_key is stable across runs.
    """

    brief_path = tmp_path / "brief.json"
    brief_path.write_text('{"role_title": "Test Role"}')
    return brief_path


def test_build_worker_argv_fresh_omits_mode_flag() -> None:
    argv = _build_worker_argv(
        brief_path="/tmp/brief.json",
        brief_id="brief-id",
        state_dir=Path("/tmp/state/linkedin/brief-id"),
        mode="fresh",
    )
    assert "--mode" not in argv
    assert "resume" not in argv
    assert "--fresh" not in argv
    assert argv[1:3] == ["-m", "cloris.worker"]
    # Phase F Slice F1: argv now begins with `--source <source>` so the
    # cloris.worker wrapper can dispatch via cloris.launchers. Default
    # source is "linkedin" for backward compat.
    assert argv[3:5] == ["--source", "linkedin"]
    assert argv[5:7] == ["--brief", "/tmp/brief.json"]
    assert argv[7:9] == ["--brief-id", "brief-id"]
    assert argv[9:11] == ["--state-dir", "/tmp/state/linkedin/brief-id"]


def test_build_worker_argv_resume_appends_mode_pair() -> None:
    argv = _build_worker_argv(
        brief_path="/tmp/brief.json",
        brief_id="brief-id",
        state_dir=Path("/tmp/state/linkedin/brief-id"),
        mode="resume",
    )
    assert argv[-2:] == ["--mode", "resume"]
    assert "--fresh" not in argv


def test_build_worker_argv_fresh_consent_appends_fresh_flag() -> None:
    argv = _build_worker_argv(
        brief_path="/tmp/brief.json",
        brief_id="brief-id",
        state_dir=Path("/tmp/state/linkedin/brief-id"),
        mode="fresh",
        fresh=True,
    )

    assert argv[-1] == "--fresh"
    assert argv.count("--fresh") == 1
    assert "--mode" not in argv


def test_build_worker_argv_diff_is_only_mode_resume_pair() -> None:
    """The single permitted difference between fresh and resume argvs is the
    trailing ``["--mode", "resume"]`` pair. Anything else is drift."""

    common_kwargs = dict(
        brief_path="/tmp/brief.json",
        brief_id="brief-id",
        state_dir=Path("/tmp/state/linkedin/brief-id"),
    )
    fresh = _build_worker_argv(mode="fresh", **common_kwargs)
    resume = _build_worker_argv(mode="resume", **common_kwargs)

    assert resume[: len(fresh)] == fresh
    assert resume[len(fresh) :] == ["--mode", "resume"]


def test_spawn_linkedin_worker_fresh_uses_no_mode_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path = _make_brief(tmp_path)
    _StubPopen.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopen)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    result = _spawn_linkedin_worker(req, mode="fresh")

    assert result.pid == 99999
    assert _StubPopen.instances, "subprocess.Popen was not invoked"
    argv = _StubPopen.instances[0].argv
    assert "--mode" not in argv
    assert "resume" not in argv
    assert "--fresh" not in argv


def test_spawn_linkedin_worker_fresh_with_consent_appends_fresh_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path = _make_brief(tmp_path)
    _StubPopen.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopen)

    req = LaunchLinkedInRequest(brief_path=str(brief_path), force_fresh=True)
    result = _spawn_linkedin_worker(req, mode="fresh")

    assert result.pid == 99999
    assert _StubPopen.instances, "subprocess.Popen was not invoked"
    argv = _StubPopen.instances[0].argv
    assert argv[-1] == "--fresh"
    assert argv.count("--fresh") == 1
    assert "--mode" not in argv


def test_spawn_linkedin_worker_resume_appends_mode_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    brief_path = _make_brief(tmp_path)
    _StubPopen.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopen)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    result = _spawn_linkedin_worker(req, mode="resume")

    assert result.pid == 99999
    assert _StubPopen.instances, "subprocess.Popen was not invoked"
    argv = _StubPopen.instances[0].argv
    assert argv[-2:] == ["--mode", "resume"]
    assert "--fresh" not in argv


def test_spawn_linkedin_worker_uses_detach_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The detach-related Popen kwargs are part of the contract: closing the
    Cloris UI process must NOT take down the worker. Phase 1.2 must not
    accidentally drop ``start_new_session`` or the stdio redirects
    while consolidating the helper."""

    import subprocess

    brief_path = _make_brief(tmp_path)
    _StubPopen.instances.clear()
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopen)

    req = LaunchLinkedInRequest(brief_path=str(brief_path))
    result = _spawn_linkedin_worker(req, mode="fresh")

    kwargs = _StubPopen.instances[0].kwargs
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is kwargs["stderr"]
    assert kwargs["stdout"] is not subprocess.DEVNULL
    stderr_path = Path(kwargs["stderr"].name)
    assert stderr_path == result.state_dir / "worker.stderr.log"
    assert stderr_path.exists()
    assert kwargs["close_fds"] is True


def test_spawn_worker_stderr_log_captures_argparse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker that fails before sidecar write leaves stderr in the
    per-state-dir worker stderr log."""

    import subprocess
    import sys

    brief_path = _make_brief(tmp_path)
    state_dir = tmp_path / "state" / "linkedin" / "argparse-error"

    import shared.output_paths as output_paths_mod

    monkeypatch.setattr(
        output_paths_mod,
        "resolve_linkedin_state_dir",
        lambda *, brief_path, **kwargs: state_dir,
    )
    monkeypatch.setattr(
        output_paths_mod,
        "derive_brief_id",
        lambda *, brief_path, **kwargs: "argparse-error",
    )

    def invalid_worker_argv(**_kwargs: Any) -> list[str]:
        return [sys.executable, "-m", "cloris.worker", "--source", "linkedin"]

    real_popen = subprocess.Popen
    processes: list[subprocess.Popen] = []

    def recording_popen(argv: list[str], **kwargs: Any) -> subprocess.Popen:
        process = real_popen(argv, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(cloris_api._monolith, "_build_worker_argv", invalid_worker_argv)
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", recording_popen)
    from cloris.launch_lock import wait_for_sidecar

    monkeypatch.setattr(cloris_api._monolith, "wait_for_sidecar", wait_for_sidecar)
    monkeypatch.setattr(cloris_api._monolith, "DEFAULT_SIDECAR_WAIT_TIMEOUT_S", 0.2)

    with pytest.raises(cloris_api._monolith.WorkerDidNotStartError):
        _spawn_linkedin_worker(
            LaunchLinkedInRequest(brief_path=str(brief_path)),
            mode="fresh",
        )

    assert len(processes) == 1
    processes[0].wait(timeout=5)

    stderr_path = state_dir / "worker.stderr.log"
    assert stderr_path.exists()
    stderr_text = stderr_path.read_text()
    assert "usage: cloris.worker" in stderr_text
    assert "the following arguments are required" in stderr_text
