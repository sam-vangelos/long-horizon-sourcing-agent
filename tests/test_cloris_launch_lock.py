"""Phase 1.1 tests for the per-state-dir launch lock.

Covers:

- ``state_dir_launch_lock`` mutual exclusion across two threads (the
  production case: two Cloris UI processes attempting concurrent launches
  for the same brief; threads stand in for processes within one test).
- Lock release on body exit (normal and exceptional).
- Timeout path: a held lock past the deadline raises
  ``LaunchLockTimeoutError``, which the route maps to HTTP 503.
- ``wait_for_sidecar`` polling: returns ``True`` once the sidecar exists
  with the expected PID, ``False`` if the sidecar never materializes
  before the timeout.
- Integration: ``_spawn_linkedin_worker`` raises
  ``WorkerAlreadyRunningError`` even when two threads race the lock — one
  wins, sees no sidecar, "spawns" (stub Popen + stub sidecar write); the
  other waits, then sees the alive PID and raises 409.
"""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from shared.runtime_state.store import RuntimeStateStore
from shared.safety.stop_reasons import RunStopReason
from cloris import api as cloris_api
from cloris.api import LaunchLinkedInRequest, _spawn_linkedin_worker
from cloris.launch_lock import (
    LaunchLockTimeoutError,
    state_dir_launch_lock,
    wait_for_sidecar,
)
from cloris.worker import WORKER_SIDECAR_FILENAME


@pytest.fixture(autouse=True)
def _contain_briefs_to_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """S3 containment: this suite writes brief fixtures under tmp_path, which
    the config/ containment boundary rejects. Treat tmp_path as config/."""

    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", tmp_path)


# --- state_dir_launch_lock --------------------------------------------------


def test_lock_serializes_two_threads(tmp_path: Path) -> None:
    """The second thread must wait until the first releases."""

    state_dir = tmp_path / "state"
    order: list[str] = []
    second_can_enter = threading.Event()
    first_in_critical = threading.Event()

    def first() -> None:
        with state_dir_launch_lock(state_dir, timeout=2.0):
            first_in_critical.set()
            order.append("first_enter")
            # Hold the lock until the test signals release.
            second_can_enter.wait(timeout=2.0)
            order.append("first_exit")

    def second() -> None:
        first_in_critical.wait(timeout=2.0)
        with state_dir_launch_lock(state_dir, timeout=2.0):
            order.append("second_enter")

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    # Let the second thread sit waiting for the lock briefly so we can
    # verify ordering: it should not enter before "first_exit".
    time.sleep(0.1)
    second_can_enter.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    assert order == ["first_enter", "first_exit", "second_enter"]


def test_lock_released_on_exception(tmp_path: Path) -> None:
    """Exceptions inside the body must not leak the lock."""

    state_dir = tmp_path / "state"

    with pytest.raises(RuntimeError, match="boom"):
        with state_dir_launch_lock(state_dir, timeout=1.0):
            raise RuntimeError("boom")

    # If the lock had leaked, this acquisition would time out.
    with state_dir_launch_lock(state_dir, timeout=0.5):
        pass


def test_lock_timeout_raises(tmp_path: Path) -> None:
    """A second acquisition while the first is held times out cleanly."""

    state_dir = tmp_path / "state"
    holding = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with state_dir_launch_lock(state_dir, timeout=2.0):
            holding.set()
            release.wait(timeout=2.0)

    holder = threading.Thread(target=hold)
    holder.start()
    holding.wait(timeout=2.0)

    try:
        with pytest.raises(LaunchLockTimeoutError) as excinfo:
            with state_dir_launch_lock(state_dir, timeout=0.2):
                pytest.fail("should not enter critical section while held")
        assert excinfo.value.timeout == 0.2
        assert str(state_dir) in excinfo.value.state_dir
    finally:
        release.set()
        holder.join(timeout=2.0)


def test_lock_creates_state_dir_if_missing(tmp_path: Path) -> None:
    """The lock helper must mkdir parents — launch is the first thing
    that runs against a brand-new state dir."""

    state_dir = tmp_path / "fresh" / "state"
    assert not state_dir.exists()
    with state_dir_launch_lock(state_dir, timeout=0.5):
        assert state_dir.is_dir()
        assert (state_dir / ".launch.lock").exists()


# --- wait_for_sidecar -------------------------------------------------------


def test_wait_for_sidecar_returns_true_when_sidecar_appears(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    def write_sidecar_after(delay_s: float) -> None:
        time.sleep(delay_s)
        (state_dir / WORKER_SIDECAR_FILENAME).write_text(
            json.dumps({"pid": 4242, "source": "linkedin"})
        )

    writer = threading.Thread(target=write_sidecar_after, args=(0.05,))
    writer.start()
    try:
        observed = wait_for_sidecar(state_dir, expected_pid=4242, timeout=1.0)
        assert observed is True
    finally:
        writer.join(timeout=1.0)


def test_wait_for_sidecar_returns_false_on_timeout(tmp_path: Path) -> None:
    """If the sidecar never appears, return False rather than hanging.
    The lock release then proceeds; the next launcher's sidecar read
    sees nothing and proceeds with its own spawn — exactly the
    user-approved Option B behavior for stale/missing sidecars."""

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    observed = wait_for_sidecar(state_dir, expected_pid=4242, timeout=0.15)
    assert observed is False


def test_wait_for_sidecar_ignores_wrong_pid(tmp_path: Path) -> None:
    """A sidecar with a different PID (e.g., stale from an earlier run)
    is treated as 'not yet ready' until the expected PID's worker writes
    the sidecar. The wait keeps polling."""

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / WORKER_SIDECAR_FILENAME).write_text(
        json.dumps({"pid": 9999, "source": "linkedin"})
    )

    observed = wait_for_sidecar(state_dir, expected_pid=4242, timeout=0.15)
    assert observed is False


def test_wait_for_sidecar_tolerates_malformed_json(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / WORKER_SIDECAR_FILENAME).write_text("not json")

    # First call: malformed sidecar -> keeps polling -> times out cleanly.
    observed = wait_for_sidecar(state_dir, expected_pid=4242, timeout=0.1)
    assert observed is False


# --- integration with _spawn_linkedin_worker -------------------------------


class _StubPopenWithSidecar:
    """Popen stub that writes a sidecar at the spawn-time PID, so the
    real ``wait_for_sidecar`` integration path can resolve quickly.

    Tests using this stub exercise the lock-and-wait flow end-to-end
    without ever creating a real subprocess.
    """

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = list(argv)
        self.kwargs = dict(kwargs)
        # Find --state-dir in argv to know where to drop the sidecar.
        sd_index = argv.index("--state-dir") + 1
        state_dir = Path(argv[sd_index])
        self.pid = 4242
        # Worker would normally do this; mimic it so wait_for_sidecar
        # returns immediately.
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / WORKER_SIDECAR_FILENAME).write_text(
            json.dumps({"pid": self.pid, "source": "linkedin"})
        )


def _make_brief(tmp_path: Path) -> Path:
    brief_path = tmp_path / "brief.json"
    brief_path.write_text('{"role_title": "Test Role"}')
    return brief_path


def test_spawn_lock_serializes_two_concurrent_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two concurrent ``_spawn_linkedin_worker`` calls for the same brief:
    one wins (writes sidecar via stub Popen), the other sees the alive
    PID and raises ``WorkerAlreadyRunningError`` — never a double-spawn.

    The state_dir resolver is monkey-patched to a tmp_path-rooted dir so
    the test (a) doesn't pollute real ``output/state/linkedin/`` and (b)
    starts clean — without the patch, a stale sidecar from a prior run
    confuses the lock-race assertion.
    """

    brief_path = _make_brief(tmp_path)
    state_dir = tmp_path / "state" / "linkedin" / "test-key"
    # Patch the lazy imports inside _spawn_linkedin_worker so they resolve
    # to a tmp_path-isolated state dir. The from-import inside the helper
    # reads from shared.output_paths each call, so patching the source
    # module is sufficient.
    import shared.output_paths as output_paths_mod

    monkeypatch.setattr(
        output_paths_mod,
        "resolve_linkedin_state_dir",
        lambda *, brief_path, **kwargs: state_dir,
    )
    monkeypatch.setattr(
        output_paths_mod,
        "derive_brief_id",
        lambda *, brief_path, **kwargs: "test-key",
    )

    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)
    # Make the alive-pid check return True for the stub PID so the second
    # caller sees an alive worker.
    monkeypatch.setattr(cloris_api._monolith, "is_pid_alive", lambda pid: pid == 4242)

    results: list[Any] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        req = LaunchLinkedInRequest(brief_path=str(brief_path))
        barrier.wait(timeout=2.0)
        try:
            results.append(_spawn_linkedin_worker(req, mode="fresh"))
        except BaseException as exc:  # noqa: BLE001 - test capture only
            errors.append(exc)

    t1 = threading.Thread(target=attempt)
    t2 = threading.Thread(target=attempt)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    # Exactly one success, exactly one WorkerAlreadyRunning.
    assert len(results) == 1, f"results={results} errors={errors}"
    assert len(errors) == 1, f"results={results} errors={errors}"
    assert isinstance(errors[0], cloris_api.WorkerAlreadyRunningError)
    assert errors[0].pid == 4242


def test_spawn_treats_same_brief_alive_sidecar_as_idempotent_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A duplicate launch click for the same brief should not look failed."""

    brief_path = _make_brief(tmp_path)
    state_dir = tmp_path / "state" / "linkedin" / "test-key"
    state_dir.mkdir(parents=True)
    (state_dir / WORKER_SIDECAR_FILENAME).write_text(
        json.dumps(
            {
                "pid": 9999,
                "source": "linkedin",
                "brief_id": "test-key",
                "brief_path": str(brief_path),
                "mode": "fresh",
                "input_mode": "concurrent",
            }
        )
    )

    import shared.output_paths as output_paths_mod

    monkeypatch.setattr(
        output_paths_mod,
        "resolve_linkedin_state_dir",
        lambda *, brief_path, **kwargs: state_dir,
    )
    monkeypatch.setattr(
        output_paths_mod,
        "derive_brief_id",
        lambda *, brief_path, **kwargs: "test-key",
    )
    monkeypatch.setattr(cloris_api._monolith, "is_pid_alive", lambda pid: pid == 9999)

    result = _spawn_linkedin_worker(
        LaunchLinkedInRequest(brief_path=str(brief_path)),
        mode="fresh",
    )

    assert result.pid == 9999
    assert result.state_dir == state_dir


def test_spawn_drains_terminal_browser_recovery_sidecar_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminal browser-recovery locks should not block a fresh launch."""

    brief_path = _make_brief(tmp_path)
    state_dir = tmp_path / "state" / "linkedin" / "test-key"
    state_dir.mkdir(parents=True)
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="test-key",
        output_dir=str(state_dir),
        mode="fresh",
    )
    store.finish_run(
        run_id,
        "interrupted",
        stop_reason=RunStopReason.BROWSER_DISCONNECT_UNRECOVERED,
    )
    (state_dir / WORKER_SIDECAR_FILENAME).write_text(
        json.dumps({"pid": 9999, "source": "linkedin"})
    )

    import shared.output_paths as output_paths_mod

    monkeypatch.setattr(
        output_paths_mod,
        "resolve_linkedin_state_dir",
        lambda *, brief_path, **kwargs: state_dir,
    )
    monkeypatch.setattr(
        output_paths_mod,
        "derive_brief_id",
        lambda *, brief_path, **kwargs: "test-key",
    )
    monkeypatch.setattr(cloris_api._monolith.subprocess, "Popen", _StubPopenWithSidecar)
    monkeypatch.setattr(cloris_api._monolith, "is_pid_alive", lambda pid: pid == 9999)
    monkeypatch.setattr(cloris_api._monolith.reconciler, "is_pid_alive", lambda pid: pid == 9999)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(
        cloris_api._monolith.reconciler.os,
        "kill",
        lambda pid, sig: sent.append((pid, sig)),
    )

    result = _spawn_linkedin_worker(
        LaunchLinkedInRequest(brief_path=str(brief_path)),
        mode="fresh",
    )

    assert result.pid == 4242
    assert sent == [(9999, signal.SIGTERM)]
    sidecar = json.loads((state_dir / WORKER_SIDECAR_FILENAME).read_text())
    assert sidecar["pid"] == 4242
