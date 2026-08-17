"""Tests for the Cloris detached worker (Slice 3).

Pin the contract of :mod:`cloris.worker` without spawning a real LinkedIn
run:

- ``build_sidecar`` returns the exact spec field set with
  ``heartbeat_at == started_at`` and ``launcher_version == LAUNCHER_VERSION``.
- ``write_sidecar`` is atomic and the written JSON is sort-keyed.
- ``read_sidecar`` collapses missing/malformed sidecars to ``None`` instead
  of raising.
- ``is_pid_alive`` correctly distinguishes self/dead/invalid inputs.
- ``build_session_orchestrator_argv`` produces the exact command shape, with
  ``--resume`` / ``--fresh`` only present when requested.
- ``main`` writes ``worker.json`` then calls ``_exec`` once with the right
  argv (the test process is never replaced because ``_exec`` is monkeypatched).
- ``main`` rejects ``--input-mode away`` at the wrapper boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cloris import worker as worker_mod
from cloris.worker import (
    LAUNCHER_VERSION,
    WORKER_SIDECAR_FILENAME,
    build_session_orchestrator_argv,
    build_sidecar,
    is_pid_alive,
    main,
    read_sidecar,
    write_sidecar,
)


def test_build_sidecar_field_contract() -> None:
    payload = build_sidecar(
        source="linkedin",
        brief_id="brief-1",
        brief_path="/tmp/brief.json",
        output_dir="/tmp/state/linkedin/brief-1",
        mode="fresh",
        input_mode="concurrent",
        started_at="2026-04-27T18:00:00+00:00",
        pid=12345,
        run_id=None,
    )

    assert set(payload.keys()) == {
        "pid",
        "source",
        "brief_id",
        "brief_path",
        "output_dir",
        "run_id",
        "started_at",
        "heartbeat_at",
        "mode",
        "input_mode",
        "launcher_version",
    }
    assert payload["pid"] == 12345
    assert payload["source"] == "linkedin"
    assert payload["brief_id"] == "brief-1"
    assert payload["brief_path"] == "/tmp/brief.json"
    assert payload["output_dir"] == "/tmp/state/linkedin/brief-1"
    assert payload["mode"] == "fresh"
    assert payload["input_mode"] == "concurrent"
    assert payload["started_at"] == "2026-04-27T18:00:00+00:00"
    assert payload["heartbeat_at"] == payload["started_at"]
    assert payload["launcher_version"] == LAUNCHER_VERSION
    assert payload["launcher_version"] == "cloris-v0-slice-4"
    assert payload["run_id"] is None


def test_write_and_read_sidecar_round_trip(tmp_path: Path) -> None:
    payload = build_sidecar(
        source="linkedin",
        brief_id="brief-1",
        brief_path=str(tmp_path / "brief.json"),
        output_dir=str(tmp_path),
        mode="fresh",
        input_mode="concurrent",
        started_at="2026-04-27T18:00:00+00:00",
        pid=999,
    )

    sidecar_path = write_sidecar(tmp_path, payload)

    assert sidecar_path == tmp_path / WORKER_SIDECAR_FILENAME
    assert sidecar_path.exists()

    raw = sidecar_path.read_text()
    parsed = json.loads(raw)
    assert parsed == payload

    expected_serialization = json.dumps(payload, indent=2, sort_keys=True)
    assert raw == expected_serialization

    reread = read_sidecar(tmp_path)
    assert reread == payload


def test_read_sidecar_missing_returns_none(tmp_path: Path) -> None:
    assert read_sidecar(tmp_path) is None


def test_read_sidecar_malformed_returns_none(tmp_path: Path) -> None:
    (tmp_path / WORKER_SIDECAR_FILENAME).write_bytes(b"not json")

    assert read_sidecar(tmp_path) is None


def test_read_sidecar_non_object_returns_none(tmp_path: Path) -> None:
    (tmp_path / WORKER_SIDECAR_FILENAME).write_text(json.dumps([1, 2, 3]))

    assert read_sidecar(tmp_path) is None


def test_is_pid_alive_self() -> None:
    assert is_pid_alive(os.getpid()) is True


def test_is_pid_alive_dead() -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()

    if is_pid_alive(child.pid) is True:
        pytest.skip(
            "PID was recycled before probe; portability fallback exercised by "
            "test_is_pid_alive_handles_invalid_input."
        )
    assert is_pid_alive(child.pid) is False


def test_is_pid_alive_handles_invalid_input() -> None:
    assert is_pid_alive(-1) is False
    assert is_pid_alive("not a pid") is False
    assert is_pid_alive(None) is False


def test_build_session_orchestrator_argv_default_concurrent() -> None:
    argv = build_session_orchestrator_argv(
        brief_path="/tmp/brief.json",
        state_dir="/tmp/state/linkedin/brief-1",
    )

    assert argv[:3] == [sys.executable, "-m", "linkedin.session_orchestrator"]
    assert argv[3:5] == ["--brief", "/tmp/brief.json"]
    assert argv[5:7] == ["--state-dir", "/tmp/state/linkedin/brief-1"]
    assert argv[7:9] == ["--input-mode", "concurrent"]
    assert argv[9:] == ["--single-session"]
    assert "--resume" not in argv
    assert "--fresh" not in argv


def test_build_session_orchestrator_argv_with_resume() -> None:
    argv = build_session_orchestrator_argv(
        brief_path="/tmp/brief.json",
        state_dir="/tmp/state/linkedin/brief-1",
        resume=True,
    )

    assert argv[:3] == [sys.executable, "-m", "linkedin.session_orchestrator"]
    assert argv[-1] == "--resume"
    assert argv.count("--resume") == 1
    assert "--fresh" not in argv


def test_build_session_orchestrator_argv_with_fresh() -> None:
    argv = build_session_orchestrator_argv(
        brief_path="/tmp/brief.json",
        state_dir="/tmp/state/linkedin/brief-1",
        fresh=True,
    )

    assert argv[:3] == [sys.executable, "-m", "linkedin.session_orchestrator"]
    assert argv[-1] == "--fresh"
    assert argv.count("--fresh") == 1
    assert "--resume" not in argv


def test_worker_main_writes_sidecar_then_execs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"id": "brief-1"}))

    captured_argv: list[list[str]] = []

    def recorder(argv: list[str]) -> None:
        captured_argv.append(list(argv))

    frozen_iso = "2026-04-27T18:00:00+00:00"

    monkeypatch.setattr(worker_mod, "_exec", recorder)
    monkeypatch.setattr(worker_mod, "_now", lambda: frozen_iso)

    import shared.output_paths as output_paths

    monkeypatch.setattr(
        output_paths,
        "resolve_linkedin_state_dir",
        lambda **_: tmp_path,
    )

    rc = main(["--brief", str(brief), "--brief-id", "brief-1"])
    assert rc == 0

    sidecar_path = tmp_path / WORKER_SIDECAR_FILENAME
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text())
    assert payload["pid"] == os.getpid()
    assert payload["brief_id"] == "brief-1"
    assert payload["source"] == "linkedin"
    assert payload["mode"] == "fresh"
    assert payload["input_mode"] == "concurrent"
    assert payload["started_at"] == frozen_iso
    assert payload["heartbeat_at"] == frozen_iso
    assert payload["launcher_version"] == "cloris-v0-slice-4"
    assert payload["run_id"] is None
    assert payload["brief_path"] == str(brief)
    assert payload["output_dir"] == str(tmp_path)

    assert len(captured_argv) == 1
    argv = captured_argv[0]
    assert argv[:5] == [
        sys.executable,
        "-m",
        "linkedin.session_orchestrator",
        "--brief",
        str(brief),
    ]
    assert "--state-dir" in argv
    assert argv[argv.index("--state-dir") + 1] == str(tmp_path)
    assert "--input-mode" in argv
    assert argv[argv.index("--input-mode") + 1] == "concurrent"
    assert "--single-session" in argv


def test_worker_main_rejects_input_mode_away(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--brief", "/tmp/x", "--brief-id", "x", "--input-mode", "away"])

    assert exc_info.value.code != 0


# --- Slice 4: --mode resume + LAUNCHER_VERSION bump ---------------------


def test_launcher_version_constant_is_slice_4() -> None:
    """Slice 4 bumps LAUNCHER_VERSION so reconciliation against older sidecars
    can distinguish slice-3 vs slice-4 worker writes."""

    assert worker_mod.LAUNCHER_VERSION == "cloris-v0-slice-4"


def test_worker_main_resume_mode_writes_sidecar_with_mode_resume_and_passes_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--mode resume threads --resume into the orchestrator argv exactly once
    and the sidecar truthfully records mode == 'resume'."""

    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"id": "brief-x"}))

    captured_argv: list[list[str]] = []

    def recorder(argv: list[str]) -> None:
        captured_argv.append(list(argv))

    monkeypatch.setattr(worker_mod, "_exec", recorder)
    monkeypatch.setattr(worker_mod, "_now", lambda: "2026-04-27T18:00:00+00:00")

    rc = main(
        [
            "--brief",
            str(brief),
            "--brief-id",
            "x",
            "--state-dir",
            str(tmp_path),
            "--mode",
            "resume",
        ]
    )
    assert rc == 0

    sidecar_path = tmp_path / WORKER_SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text())
    assert payload["mode"] == "resume"
    assert payload["launcher_version"] == "cloris-v0-slice-4"

    assert len(captured_argv) == 1
    argv = captured_argv[0]
    assert argv.count("--resume") == 1
    assert "--fresh" not in argv


def test_worker_main_default_mode_is_fresh_after_slice_4_bump(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default --mode stays 'fresh' after the slice-4 bump and --resume is
    NOT threaded into the orchestrator argv."""

    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"id": "brief-x"}))

    captured_argv: list[list[str]] = []

    def recorder(argv: list[str]) -> None:
        captured_argv.append(list(argv))

    monkeypatch.setattr(worker_mod, "_exec", recorder)
    monkeypatch.setattr(worker_mod, "_now", lambda: "2026-04-27T18:00:00+00:00")

    rc = main(
        [
            "--brief",
            str(brief),
            "--brief-id",
            "x",
            "--state-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0

    sidecar_path = tmp_path / WORKER_SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text())
    assert payload["mode"] == "fresh"

    assert len(captured_argv) == 1
    argv = captured_argv[0]
    assert "--resume" not in argv
    assert "--fresh" not in argv


def test_worker_main_threads_fresh_consent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--fresh threads explicit consent into the orchestrator argv exactly once
    while the sidecar still records mode == 'fresh'."""

    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"id": "brief-x"}))

    captured_argv: list[list[str]] = []

    def recorder(argv: list[str]) -> None:
        captured_argv.append(list(argv))

    monkeypatch.setattr(worker_mod, "_exec", recorder)
    monkeypatch.setattr(worker_mod, "_now", lambda: "2026-04-27T18:00:00+00:00")

    rc = main(
        [
            "--brief",
            str(brief),
            "--brief-id",
            "x",
            "--state-dir",
            str(tmp_path),
            "--fresh",
        ]
    )
    assert rc == 0

    sidecar_path = tmp_path / WORKER_SIDECAR_FILENAME
    payload = json.loads(sidecar_path.read_text())
    assert payload["mode"] == "fresh"

    assert len(captured_argv) == 1
    argv = captured_argv[0]
    assert argv.count("--fresh") == 1
    assert "--resume" not in argv


# --- Phase 0 worker-binary slice: frozen-app in-process dispatch -------


def test_worker_main_dispatches_in_process_when_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frozen .app context: instead of ``execvp``-ing into a python
    interpreter (the bundle has none), the worker calls the
    orchestrator's ``main()`` in-process. Same PID, sidecar's ``pid``
    field stays truthful for any later stop/probe."""

    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"id": "brief-x"}))

    exec_calls: list[list[str]] = []

    def exec_recorder(argv: list[str]) -> None:
        exec_calls.append(list(argv))

    dispatch_calls: list[tuple[str, list[str]]] = []

    def dispatch_recorder(source: str, orchestrator_argv: list[str]) -> int:
        dispatch_calls.append((source, list(orchestrator_argv)))
        return 0

    monkeypatch.setattr(worker_mod, "_exec", exec_recorder)
    monkeypatch.setattr(
        worker_mod, "_dispatch_in_process", dispatch_recorder
    )
    monkeypatch.setattr(worker_mod, "_is_frozen", lambda: True)
    monkeypatch.setattr(worker_mod, "_now", lambda: "2026-04-27T18:00:00+00:00")

    rc = main(
        [
            "--brief",
            str(brief),
            "--brief-id",
            "x",
            "--state-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0

    # Sidecar still written before the in-process dispatch.
    payload = json.loads((tmp_path / WORKER_SIDECAR_FILENAME).read_text())
    assert payload["pid"] == os.getpid()

    # ``_exec`` MUST NOT be called when frozen; we'd be trying to
    # exec a python interpreter that doesn't exist in the bundle.
    assert exec_calls == []

    # ``_dispatch_in_process`` got the source and the same
    # orchestrator argv ``execvp`` would have received.
    assert len(dispatch_calls) == 1
    source, orch_argv = dispatch_calls[0]
    assert source == "linkedin"
    assert orch_argv[:3] == [
        sys.executable,
        "-m",
        "linkedin.session_orchestrator",
    ]


def test_dispatch_in_process_invokes_orchestrator_main_with_sliced_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_dispatch_in_process`` must slice off the python invocation
    prefix and invoke the orchestrator's ``main()`` with sys.argv
    populated from the orchestrator-cli-args portion. The orchestrator
    reads sys.argv via ``argparse.parse_args()`` so this is the
    contract that keeps it working under the frozen bundle."""

    captured: dict[str, list[str]] = {}

    def fake_orchestrator_main() -> int:
        captured["argv"] = list(sys.argv)
        return 7

    fake_orch = type("FakeOrch", (), {"main": staticmethod(fake_orchestrator_main)})

    import linkedin
    monkeypatch.setattr(linkedin, "session_orchestrator", fake_orch, raising=False)
    # Clear cached import so the module-local ``from linkedin import
    # session_orchestrator as orch`` picks up our fake.
    monkeypatch.delitem(sys.modules, "linkedin.session_orchestrator", raising=False)
    sys.modules["linkedin.session_orchestrator"] = fake_orch  # type: ignore[assignment]

    rc = worker_mod._dispatch_in_process(
        "linkedin",
        [
            sys.executable,
            "-m",
            "linkedin.session_orchestrator",
            "--brief",
            "/tmp/brief.json",
            "--state-dir",
            "/tmp/state",
            "--input-mode",
            "concurrent",
        ],
    )

    assert rc == 7
    assert captured["argv"] == [
        "linkedin.session_orchestrator",
        "--brief",
        "/tmp/brief.json",
        "--state-dir",
        "/tmp/state",
        "--input-mode",
        "concurrent",
    ]


def test_dispatch_in_process_rejects_unknown_source() -> None:
    """An unknown source returns exit code 2 — same surface the unknown-
    source branch in ``main`` uses, so the spawn-side aggregator gets
    a consistent failure signal.

    Uses a synthetic source name that is not registered in
    ``cloris/launchers``. As parallel module work lands, the registered
    set grows (researcher, designer, …); pick a name guaranteed not to
    collide.
    """

    rc = worker_mod._dispatch_in_process(
        "nonexistent_source_for_test_only",
        [sys.executable, "-m", "nonexistent_source_for_test_only", "--brief", "x"],
    )
    assert rc == 2


def test_dispatch_in_process_rejects_malformed_argv() -> None:
    """Argv shorter than ``[python, -m, MODULE]`` returns exit code 2
    rather than indexing past the end. Defensive against a registry
    bug that would otherwise crash the worker."""

    rc = worker_mod._dispatch_in_process("linkedin", [sys.executable, "-m"])
    assert rc == 2


# --- Slice 1.2: registry-driven in-process dispatch for every source ---


@pytest.mark.parametrize(
    "source, module_dotpath",
    [
        ("linkedin", "linkedin.session_orchestrator"),
        ("github", "github.session_orchestrator"),
        ("researcher", "researcher.session_orchestrator"),
        ("designer", "designer.session_orchestrator"),
        ("exec_search", "exec_search.session_orchestrator"),
    ],
)
def test_frozen_in_process_dispatch_routes_every_registered_source_via_registry(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    module_dotpath: str,
) -> None:
    """Mock-frozen execution: every source in
    :data:`cloris.launchers.LAUNCHERS` dispatches in-process via its
    registered ``in_process_dispatch_fn``.

    Slice 1.2 closes the silent regression at the legacy ladder where
    ``exec_search`` fell through to the "no in-process dispatch" stderr
    path even though the source was registered. The parametrize set
    includes ``exec_search`` precisely to pin that regression closed —
    a future ladder reintroduction would fail this test.

    The fake orchestrator returns 7; the registry-routed dispatcher
    must surface that exit code unchanged. A return of 2 (the unknown-
    source / malformed-argv stderr branch) means the dispatch fell
    through and the regression is back.
    """

    captured: dict[str, list[str]] = {}

    def fake_orchestrator_main() -> int:
        captured["argv"] = list(sys.argv)
        return 7

    fake_orch = type(
        "FakeOrch", (), {"main": staticmethod(fake_orchestrator_main)}
    )

    import importlib

    parent_pkg = importlib.import_module(source)
    monkeypatch.setattr(parent_pkg, "session_orchestrator", fake_orch, raising=False)
    # Use ``setitem`` (NOT ``delitem`` + direct assignment) so monkeypatch
    # records and restores the original ``sys.modules`` slot on teardown.
    # Otherwise the ``FakeOrch`` leaks into later tests that do
    # ``from <source> import session_orchestrator as so`` — observed via
    # researcher / designer smoke tests failing with
    # ``AttributeError: <class 'FakeOrch'> has no attribute 'build_pipeline'``.
    monkeypatch.setitem(sys.modules, module_dotpath, fake_orch)

    rc = worker_mod._dispatch_in_process(
        source,
        [
            sys.executable,
            "-m",
            module_dotpath,
            "--brief",
            "/tmp/brief.json",
            "--state-dir",
            "/tmp/state",
        ],
    )

    assert rc == 7, (
        f"source={source!r} did not route to its registered dispatcher "
        f"(rc={rc}); registry-driven dispatch is regressed."
    )
    assert captured["argv"][0] == module_dotpath
    assert captured["argv"][1:] == [
        "--brief",
        "/tmp/brief.json",
        "--state-dir",
        "/tmp/state",
    ]


def test_frozen_in_process_dispatch_has_a_callable_for_every_registered_source() -> None:
    """Defense in depth: every registered source has a non-None
    ``in_process_dispatch_fn`` on its ``LauncherEntry`` so the frozen
    .app worker never falls through to the "no in-process dispatch"
    stderr path.

    The parametrized test above exercises the routing end-to-end with
    a fake orchestrator. This one is a registry-shape assertion: if a
    new source lands without populating ``in_process_dispatch_fn``, it
    will silently regress to the "no in-process dispatch" stderr path
    under the frozen .app — same class of bug as the pre-1.2 exec_search
    regression. Catch it at registration time, not at frozen-app
    runtime.
    """

    from cloris.launchers import LAUNCHERS, known_sources

    for source in known_sources():
        entry = LAUNCHERS[source]
        assert entry.in_process_dispatch_fn is not None, (
            f"{source!r} has no in_process_dispatch_fn registered. "
            "Frozen .app worker would silently fall through; populate the "
            "field on this source's LauncherEntry."
        )
