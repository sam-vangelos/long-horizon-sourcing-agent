"""Tests for LinkedIn session orchestrator resume bookkeeping."""

import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from linkedin.session_orchestrator import (
    _classify_session_exception,
    _parse_restart_strings_arg,
    _run_day_cycle_with_browser_lock,
    _resume_has_pending_work,
    _assert_resume_target_exists,
    ResumeTargetHasNoRun,
)
from shared.governor import SessionGovernor
from shared.output_paths import resolve_linkedin_state_dir


@pytest.fixture(autouse=True)
def _isolate_linkedin_browser_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linkedin.session_orchestrator as so
    from shared.runtime_state import RuntimeStateLock

    monkeypatch.setattr(
        so,
        "_linkedin_browser_lock",
        lambda: RuntimeStateLock(
            tmp_path,
            filename="recruiter-browser.lock",
            resource_name="LinkedIn Recruiter browser",
        ),
    )


def _write_minimal_brief(path: Path, *, project_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "role_title": "Test Role",
                "linkedin_project": "Test Project",
                "linkedin_project_id": project_id,
                "search_priorities": ["One good lane"],
                "capability_areas": [
                    {
                        "name": "Technical leadership",
                        "description": "Builder requirement",
                        "builder_signals": ["Built it"],
                        "user_signals": ["Did not build it"],
                        "key_terms": ["builder"],
                    }
                ],
                "location": "New York City",
            }
        )
    )


def _seed_runtime_state(
    state_dir: Path,
    *,
    statuses: tuple[str, ...],
    pending_block_string_ids: tuple[int, ...] = (),
) -> Path:
    db_path = state_dir / "runtime_state.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                resume_state_json TEXT NOT NULL
            );
            CREATE TABLE work_units (
                id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """
        )
        cursor = conn.execute(
            "INSERT INTO runs(source, resume_state_json) VALUES ('linkedin', ?)",
            (json.dumps({"pending_block_string_ids": pending_block_string_ids}),),
        )
        for status in statuses:
            conn.execute(
                "INSERT INTO work_units(run_id, source, kind, status) "
                "VALUES (?, 'linkedin', 'linkedin_string', ?)",
                (cursor.lastrowid, status),
            )
    return db_path


def test_resume_has_pending_work_when_progress_missing():
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        project_id = f"missing-{Path(td).name}"
        _write_minimal_brief(brief_path, project_id=project_id)
        assert _resume_has_pending_work(str(brief_path)) is True


def test_assert_resume_target_refuses_a_state_dir_with_no_run():
    """--resume against a runless dir must REFUSE, not degrade to a fresh run.

    The trap this locks: has_pending_work() answers None for a dir it cannot read,
    _resume_has_pending_work maps None -> True as a fail-safe, and the orchestrator
    then computes resume_existing_state False (no runtime state) and starts a BRAND
    NEW run — live searching and live physical saves — with --resume silently
    discarded. Nothing else in the stack refuses this.
    """
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        _write_minimal_brief(brief_path, project_id=f"norun-{Path(td).name}")
        # The fail-safe says "go", which is exactly why the gate is needed.
        assert _resume_has_pending_work(str(brief_path)) is True
        with pytest.raises(ResumeTargetHasNoRun, match="no run exists"):
            _assert_resume_target_exists(str(brief_path))


def test_assert_resume_target_allows_a_dir_with_legacy_progress():
    """A legacy progress.json IS a resumable run — the gate must not block it."""
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        _write_minimal_brief(brief_path, project_id=f"legacy-{Path(td).name}")
        progress_path = (
            resolve_linkedin_state_dir(brief_path=brief_path) / "progress.json"
        )
        progress_path.write_text(json.dumps({"strings": [{"id": 1, "status": "queued"}]}))
        _assert_resume_target_exists(str(brief_path))  # must not raise


def test_resume_has_pending_work_false_when_queue_exhausted_in_derived_state_dir():
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        project_id = f"exhausted-{Path(td).name}"
        _write_minimal_brief(brief_path, project_id=project_id)
        progress_path = resolve_linkedin_state_dir(brief_path=brief_path) / "progress.json"
        progress_path.write_text(json.dumps({
            "strings": [
                {"id": 1, "status": "done"},
                {"id": 2, "status": "skipped"},
            ]
        }))

        assert _resume_has_pending_work(str(brief_path)) is False


def test_resume_has_pending_work_false_when_queue_exhausted_with_explicit_output_dir():
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        project_id = f"explicit-{Path(td).name}"
        _write_minimal_brief(brief_path, project_id=project_id)
        output_dir = Path(td) / "custom-state"
        progress_path = resolve_linkedin_state_dir(
            brief_path=brief_path,
            state_dir=output_dir,
        ) / "progress.json"
        progress_path.write_text(json.dumps({
            "strings": [
                {"id": 1, "status": "done"},
                {"id": 2, "status": "skipped"},
            ]
        }))

        assert _resume_has_pending_work(str(brief_path), str(output_dir)) is False


def test_resume_has_pending_work_true_when_any_string_is_queued_in_derived_state_dir():
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        project_id = f"queued-{Path(td).name}"
        _write_minimal_brief(brief_path, project_id=project_id)
        progress_path = resolve_linkedin_state_dir(brief_path=brief_path) / "progress.json"
        progress_path.write_text(json.dumps({
            "strings": [
                {"id": 1, "status": "done"},
                {"id": 2, "status": "queued"},
            ]
        }))

        assert _resume_has_pending_work(str(brief_path)) is True


def test_resume_has_pending_work_true_when_any_string_is_in_progress_in_derived_state_dir():
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        project_id = f"in-progress-{Path(td).name}"
        _write_minimal_brief(brief_path, project_id=project_id)
        progress_path = resolve_linkedin_state_dir(brief_path=brief_path) / "progress.json"
        progress_path.write_text(json.dumps({
            "strings": [
                {"id": 1, "status": "done"},
                {"id": 2, "status": "in_progress"},
            ]
        }))

        assert _resume_has_pending_work(str(brief_path)) is True


@pytest.mark.parametrize("status", ["error", "legacy-unknown"])
def test_resume_has_pending_work_true_for_any_nonterminal_projection_status(status):
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        project_id = f"nonterminal-{Path(td).name}"
        _write_minimal_brief(brief_path, project_id=project_id)
        progress_path = resolve_linkedin_state_dir(brief_path=brief_path) / "progress.json"
        progress_path.write_text(json.dumps({
            "strings": [
                {"id": 1, "status": "done"},
                {"id": 2, "status": status},
            ]
        }))

        assert _resume_has_pending_work(str(brief_path)) is True


def test_resume_has_pending_work_true_when_block_adaptation_is_pending_in_derived_state_dir():
    with tempfile.TemporaryDirectory() as td:
        brief_path = Path(td) / "brief.json"
        project_id = f"pending-block-{Path(td).name}"
        _write_minimal_brief(brief_path, project_id=project_id)
        progress_path = resolve_linkedin_state_dir(brief_path=brief_path) / "progress.json"
        progress_path.write_text(json.dumps({
            "strings": [
                {"id": 1, "status": "done"},
                {"id": 2, "status": "skipped"},
            ],
            "pending_block_name": "Compound Batch 1",
            "pending_block_string_ids": [1],
        }))

        assert _resume_has_pending_work(str(brief_path)) is True


def test_resume_prefers_canonical_queued_work_over_stale_exhausted_projection(tmp_path: Path):
    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id="canonical-queued")
    state_dir = resolve_linkedin_state_dir(brief_path=brief_path, state_dir=tmp_path / "state")
    (state_dir / "progress.json").write_text(json.dumps({"strings": [{"status": "done"}]}))
    _seed_runtime_state(state_dir, statuses=("done", "queued"))

    assert _resume_has_pending_work(str(brief_path), str(state_dir)) is True


def test_resume_prefers_canonical_exhaustion_over_stale_queued_projection(tmp_path: Path):
    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id="canonical-exhausted")
    state_dir = resolve_linkedin_state_dir(brief_path=brief_path, state_dir=tmp_path / "state")
    (state_dir / "progress.json").write_text(json.dumps({"strings": [{"status": "queued"}]}))
    _seed_runtime_state(state_dir, statuses=("done", "skipped"))

    assert _resume_has_pending_work(str(brief_path), str(state_dir)) is False


@pytest.mark.parametrize("status", ["error", "legacy-unknown"])
def test_resume_treats_any_nonterminal_canonical_status_as_pending(
    tmp_path: Path,
    status: str,
):
    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id=f"canonical-{status}")
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "state",
    )
    _seed_runtime_state(state_dir, statuses=("done", status))

    assert _resume_has_pending_work(str(brief_path), str(state_dir)) is True


def test_resume_reads_pending_block_context_from_canonical_state(tmp_path: Path):
    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id="canonical-pending-block")
    state_dir = resolve_linkedin_state_dir(brief_path=brief_path, state_dir=tmp_path / "state")
    (state_dir / "progress.json").write_text(json.dumps({"strings": [{"status": "done"}]}))
    _seed_runtime_state(
        state_dir,
        statuses=("done",),
        pending_block_string_ids=(1,),
    )

    assert _resume_has_pending_work(str(brief_path), str(state_dir)) is True


def test_resume_with_only_canonical_full_obligation_enters_run_full(
    tmp_path: Path,
):
    from shared.runtime_state.store import RuntimeStateStore

    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id="canonical-obligation")
    state_dir = resolve_linkedin_state_dir(
        brief_path=brief_path,
        state_dir=tmp_path / "state",
    )
    store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id="canonical-obligation",
        output_dir=str(state_dir),
        mode="fresh",
    )
    store.upsert_work_unit(
        run_id=run_id,
        source="linkedin",
        brief_id="canonical-obligation",
        kind="linkedin_string",
        source_unit_id="1",
        display_name="done string",
        ordering_index=0,
        status="done",
    )
    candidate_id = store.ensure_candidate(
        source="linkedin",
        brief_id="canonical-obligation",
        identity_key="/talent/profile/pending",
        profile_url="/talent/profile/pending",
    )
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO candidate_attempts("
            "run_id, candidate_id, stage, attempt_number, status, payload_json, "
            "source_cursor_json, started_at, ended_at"
            ") VALUES (?, ?, 'facial', 1, 'succeeded', ?, '{}', '', '')",
            (
                run_id,
                candidate_id,
                json.dumps(
                    {"facial_decision": {"decision": "FACIAL_YES"}}
                ),
            ),
        )

    governor = MagicMock()
    governor.can_start_session.return_value = (True, "")
    governor.end_session.return_value = {"profile_opens_session": 0}
    pipeline = MagicMock()
    pipeline.run_full = AsyncMock()
    pipeline.stats = {}

    with patch(
        "linkedin.session_orchestrator.SessionGovernor",
        return_value=governor,
    ), patch(
        "linkedin.session_orchestrator.cooldown.record_session_start",
        return_value=1,
    ), patch(
        "linkedin.session_orchestrator.cooldown.get_sessions_today",
        return_value=1,
    ), patch(
        "linkedin.session_orchestrator.cooldown.get_profile_opens_24h",
        return_value=0,
    ), patch(
        "linkedin.session_orchestrator.cooldown.record_session_end",
    ), patch(
        "linkedin.session_orchestrator.signal.signal",
    ), patch(
        "linkedin.orchestrator.Pipeline",
        return_value=pipeline,
    ) as pipeline_class:
        asyncio.run(
            _run_day_cycle_with_browser_lock(
                brief_path=str(brief_path),
                output_dir=str(state_dir),
                resume=True,
            )
        )

    pipeline_class.assert_called_once()
    pipeline.run_full.assert_awaited_once_with(
        resume=True,
        restart_string_id=None,
        restart_string_ids=None,
    )


def test_resume_falls_back_to_projection_when_canonical_db_is_unreadable(tmp_path: Path):
    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id="unreadable-canonical")
    state_dir = resolve_linkedin_state_dir(brief_path=brief_path, state_dir=tmp_path / "state")
    (state_dir / "runtime_state.sqlite3").write_text("not sqlite")
    (state_dir / "progress.json").write_text(json.dumps({"strings": [{"status": "queued"}]}))

    assert _resume_has_pending_work(str(brief_path), str(state_dir)) is True


def test_resume_opens_canonical_db_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import shared.runtime_state.read_models as read_models

    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id="readonly-canonical")
    state_dir = resolve_linkedin_state_dir(brief_path=brief_path, state_dir=tmp_path / "state")
    _seed_runtime_state(state_dir, statuses=("done",))
    calls: list[tuple[tuple, dict]] = []
    real_connect = sqlite3.connect

    def recording_connect(*args, **kwargs):
        calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(read_models.sqlite3, "connect", recording_connect)

    assert _resume_has_pending_work(str(brief_path), str(state_dir)) is False
    assert calls
    assert "mode=ro" in calls[0][0][0]
    assert calls[0][1]["uri"] is True


def test_global_browser_lock_blocks_second_process_before_browser_control(
    tmp_path: Path,
):
    from shared.runtime_state import RuntimeStateLock

    lock_dir = tmp_path
    held = RuntimeStateLock(
        lock_dir,
        filename="recruiter-browser.lock",
        resource_name="LinkedIn Recruiter browser",
    )
    marker = tmp_path / "browser-control-reached"
    code = f"""
import asyncio
from pathlib import Path
import linkedin.session_orchestrator as session_orchestrator
from shared.runtime_state import RuntimeStateLock

async def forbidden_browser_control(**kwargs):
    Path({str(marker)!r}).write_text("reached")

session_orchestrator._linkedin_browser_lock = lambda: RuntimeStateLock(
    Path({str(lock_dir)!r}),
    filename="recruiter-browser.lock",
    resource_name="LinkedIn Recruiter browser",
)
session_orchestrator._run_day_cycle_with_browser_lock = forbidden_browser_control
try:
    asyncio.run(session_orchestrator.run_day_cycle(
        brief_path="brief.json",
        output_dir=None,
    ))
except RuntimeError as exc:
    print(exc)
else:
    raise SystemExit(3)
"""

    held.acquire()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        held.release()

    assert completed.returncode == 0, completed.stderr
    assert "already locked" in completed.stdout
    assert not marker.exists()


def test_global_browser_lock_blocks_decoy_process_before_browser_control(
    tmp_path: Path,
):
    from shared.runtime_state import RuntimeStateLock

    lock_dir = tmp_path
    held = RuntimeStateLock(
        lock_dir,
        filename="recruiter-browser.lock",
        resource_name="LinkedIn Recruiter browser",
    )
    marker = tmp_path / "decoy-browser-control-reached"
    code = f"""
import asyncio
from pathlib import Path
import linkedin.session_orchestrator as session_orchestrator
from shared.runtime_state import RuntimeStateLock

async def forbidden_browser_control():
    Path({str(marker)!r}).write_text("reached")

session_orchestrator._linkedin_browser_lock = lambda: RuntimeStateLock(
    Path({str(lock_dir)!r}),
    filename="recruiter-browser.lock",
    resource_name="LinkedIn Recruiter browser",
)
session_orchestrator._run_decoy_only_with_browser_lock = forbidden_browser_control
try:
    asyncio.run(session_orchestrator.run_decoy_only())
except RuntimeError as exc:
    print(exc)
else:
    raise SystemExit(3)
"""

    held.acquire()
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        held.release()

    assert completed.returncode == 0, completed.stderr
    assert "already locked" in completed.stdout
    assert not marker.exists()


def test_governor_can_start_session_when_under_caps():
    # CLO-153: the session-count read is gone from can_start_session — only
    # the opens window and backoff state are consulted.
    governor = SessionGovernor()
    with patch("shared.governor.cooldown.get_profile_opens_24h", return_value=0), \
         patch("shared.governor.cooldown.get_active_backoff", return_value=None):
        ok, reason = governor.can_start_session(session_type="linkedin_sourcing")
        assert ok is True
        assert reason == "ok"


def test_classify_session_exception_distinguishes_interrupts_from_errors():
    assert _classify_session_exception(KeyboardInterrupt()) == "interrupted: KeyboardInterrupt"
    assert _classify_session_exception(NameError("boom")) == "error: NameError"


def test_parse_restart_strings_arg_parses_csv():
    assert _parse_restart_strings_arg("4, 11,12,16,27") == [4, 11, 12, 16, 27]


def test_parse_restart_strings_arg_ignores_empty_chunks():
    assert _parse_restart_strings_arg("4,, 11, ,27") == [4, 11, 27]


def test_session_error_shutdown_gives_geography_regime_a_stable_reason():
    """P3a (Codex review, Wave 1): a GeographyRegimeError session must
    classify under the stable reason run_day_cycle breaks on — never the
    free-text 'error: ...' the cycle retries into (each retry hits the same
    unappliable facet)."""
    from linkedin.orchestrator import GeographyRegimeError
    from linkedin.session_orchestrator import _session_error_shutdown
    from shared.cooldown import ShutdownKind

    reason, kind = _session_error_shutdown(GeographyRegimeError("facet miss"))
    assert reason == "geography_regime_error"
    assert kind == ShutdownKind.ERROR

    reason, kind = _session_error_shutdown(
        GeographyRegimeError("typeahead ranking varied", retryable=True)
    )
    assert reason == "geography_apply_transient"
    assert kind == ShutdownKind.ERROR

    reason, kind = _session_error_shutdown(RuntimeError("transient"))
    assert reason == "error: transient"
    assert kind == ShutdownKind.ERROR


def test_session_error_shutdown_survives_classifier_import_failure(
    monkeypatch, capsys
):
    """CLO-151: the classifier runs on the error path and must never replace
    the session error it is classifying. On 2026-08-10 an ImportError raised
    by its lazy imports was recorded as the session's cause (`error:
    ImportError`), erasing the real one. A poisoned `linkedin.orchestrator`
    entry (None in sys.modules makes the lazy import raise ImportError) must
    yield the ORIGINAL error's generic classification, not an exception."""
    import sys

    from linkedin.session_orchestrator import _session_error_shutdown
    from shared.cooldown import ShutdownKind

    monkeypatch.setitem(sys.modules, "linkedin.orchestrator", None)

    reason, kind = _session_error_shutdown(RuntimeError("real cause"))
    assert reason == "error: real cause"
    assert kind == ShutdownKind.ERROR
    assert "Session-error classifier failed" in capsys.readouterr().out


def test_session_error_shutdown_regime_reasons_survive_import_failure(
    monkeypatch,
):
    """Review finding (wave 1): the module-scope regime classes must keep
    their STABLE reasons even when the lazy orchestrator import is broken —
    demoting them to retryable 'error: ...' free text would reopen the
    absorb-retry loop those reasons exist to close (P3b/P4)."""
    import sys

    from shared.constraint_manifest import ConstraintManifestError
    from shared.preflight_v2 import PreflightRegimeError
    from linkedin.session_orchestrator import _session_error_shutdown
    from shared.cooldown import ShutdownKind

    monkeypatch.setitem(sys.modules, "linkedin.orchestrator", None)

    reason, kind = _session_error_shutdown(ConstraintManifestError("no owners"))
    assert reason == "constraint_manifest_error"
    assert kind == ShutdownKind.ERROR

    reason, kind = _session_error_shutdown(PreflightRegimeError("lint wall"))
    assert reason == "preflight_regime_error"
    assert kind == ShutdownKind.ERROR


def test_session_error_shutdown_names_stale_browser_environment_error(
    monkeypatch,
):
    """CLO-150: the stale-CDP-instance signature classifies as a stable
    environment stop the day cycle raises on first occurrence — never the
    'error: ...' free text the absorb loop would burn three attempts on.
    Locked with persistence and browser-crash resume both armed, so the
    refusal is the classification's doing, not a disabled absorb path."""
    from shared import config
    from linkedin.session_orchestrator import (
        _session_error_shutdown,
        _should_retry_session_error,
    )
    from shared.cooldown import ShutdownKind

    monkeypatch.setattr(config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)
    monkeypatch.setattr(config, "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", True)

    exc = RuntimeError(
        "BrowserType.connect_over_cdp: Protocol error "
        "(Browser.setDownloadBehavior): Browser context management is not "
        "supported."
    )
    reason, kind = _session_error_shutdown(exc)
    assert reason == "browser_environment_error"
    assert kind == ShutdownKind.ERROR
    assert (
        _should_retry_session_error(
            reason,
            consecutive_error_resumes=0,
            profile_opens=0,
            error=exc,
        )
        is False
    )


def _run_day_cycle_with_session_results(
    monkeypatch,
    session_results: list[dict],
    *,
    with_decoy: bool = False,
    resume: bool = False,
    decoy_close_error: Exception | None = None,
    playwright_stop_error: Exception | None = None,
    restart_string_ids: list[int] | None = None,
):
    import linkedin.session_orchestrator as so

    recorded: dict = {
        "run_calls": [],
        "session_end_reasons": [],
        "dormant_calls": 0,
        "dormant_sample_calls": 0,
        "error_backoff_calls": 0,
        "wait_calls": 0,
        "can_start_calls": 0,
    }
    current_profile_opens = 0

    class FakeGovernor:
        def can_start_session(self, session_type="linkedin_sourcing"):
            recorded["can_start_calls"] += 1
            return True, "ok"

        def start_session(self, session_duration_seconds=None):
            recorded.setdefault("session_durations", []).append(session_duration_seconds)

        def end_session(self):
            return {"profile_opens_session": current_profile_opens}

    class FakeDecoy:
        def __init__(self, context):
            recorded["decoy"] = self
            recorded["decoy_context"] = context

        async def run_dormant_loop(self, *_args, **_kwargs):
            recorded["dormant_calls"] += 1

        async def close(self):
            recorded["decoy_close_attempted"] = True
            if decoy_close_error is not None:
                raise decoy_close_error
            recorded["decoy_closed"] = True

    class FakeBrowser:
        contexts = [object()]

    class FakeChromium:
        async def connect_over_cdp(self, _url):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        async def stop(self):
            recorded["playwright_stop_attempted"] = True
            if playwright_stop_error is not None:
                raise playwright_stop_error
            recorded["playwright_stopped"] = True

    class FakePlaywrightManager:
        async def start(self):
            recorded["playwright_started"] = True
            return FakePlaywright()

    remaining_results = list(session_results)

    async def fake_run_sourcing_session(**kwargs):
        nonlocal current_profile_opens
        recorded["run_calls"].append(kwargs)
        if not remaining_results:
            raise AssertionError("run_day_cycle started an unexpected session")
        result = dict(remaining_results.pop(0))
        current_profile_opens = int(result.pop("profile_opens_session", 0) or 0)
        if result.pop("set_operator_stop", False):
            kwargs["operator_stop_event"].set()
        return result

    def fake_sample_dormant_duration():
        recorded["dormant_sample_calls"] += 1
        return 1.0

    def fake_sample_error_backoff(_attempt):
        recorded["error_backoff_calls"] += 1
        return 0.5

    async def fake_wait_for(awaitable, *, timeout):
        recorded["wait_calls"] += 1
        recorded["wait_timeout"] = timeout
        awaitable.close()
        raise asyncio.TimeoutError

    def fake_record_session_end(**kwargs):
        recorded["session_end_reasons"].append(kwargs["reason"])

    monkeypatch.setattr(so, "SessionGovernor", FakeGovernor)
    monkeypatch.setattr(so, "DecoyAgent", FakeDecoy)
    monkeypatch.setattr(so, "_sample_session_duration", lambda: 16666.0)
    monkeypatch.setattr(so, "_sample_dormant_duration", fake_sample_dormant_duration)
    monkeypatch.setattr(so, "_sample_error_backoff", fake_sample_error_backoff)
    monkeypatch.setattr(so, "_run_sourcing_session", fake_run_sourcing_session)
    monkeypatch.setattr(so, "_resume_has_pending_work", lambda *_args, **_kwargs: True)
    # The runless-resume refusal is exercised by its own tests
    # (test_assert_resume_target_*); these day-cycle mechanics tests use a
    # fabricated state dir, so neutralize it the same way the passive
    # pending-work check above is neutralized.
    monkeypatch.setattr(so, "_assert_resume_target_exists", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(so.asyncio, "wait_for", fake_wait_for)
    monkeypatch.setattr(so.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "rebrowser_playwright.async_api.async_playwright",
        lambda: FakePlaywrightManager(),
    )
    monkeypatch.setattr(so.cooldown, "record_session_start", lambda session_type: 42)
    monkeypatch.setattr(so.cooldown, "get_sessions_today", lambda session_type: 1)
    monkeypatch.setattr(so.cooldown, "get_profile_opens_24h", lambda: 0)
    monkeypatch.setattr(so.cooldown, "record_session_end", fake_record_session_end)

    try:
        asyncio.run(
            so.run_day_cycle(
                brief_path="brief.json",
                output_dir=None,
                multi_session=True,
                with_decoy=with_decoy,
                resume=resume,
                restart_string_ids=restart_string_ids,
            )
        )
    except BaseException as exc:
        recorded["raised"] = exc

    return recorded


@pytest.mark.parametrize(
    "message",
    [
        "Locator.wait_for: Target crashed",
        "Page crashed",
    ],
)
def test_local_browser_crash_can_resume_after_profile_activity(
    monkeypatch,
    message,
):
    import linkedin.session_orchestrator as so

    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)
    monkeypatch.setattr(so.config, "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", True)

    assert so._should_retry_session_error(
        "error: RuntimeError",
        consecutive_error_resumes=0,
        profile_opens=42,
        error=RuntimeError(message),
    ) is True


@pytest.mark.parametrize(
    "message",
    [
        "Connection closed",
        "[Errno 32] Broken pipe",
        "panel recovery exhausted for Target Closed-Won AE",
        "Profile scrape failed: This LinkedIn account has been closed",
    ],
)
def test_non_browser_fault_text_cannot_resume_after_profile_activity(
    monkeypatch,
    message,
):
    import linkedin.session_orchestrator as so

    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)
    monkeypatch.setattr(so.config, "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", True)

    assert so._should_retry_session_error(
        "error: RuntimeError",
        consecutive_error_resumes=0,
        profile_opens=42,
        error=RuntimeError(message),
    ) is False


def test_rebrowser_target_closed_error_type_can_resume_after_profile_activity(
    monkeypatch,
):
    from rebrowser_playwright._impl._errors import TargetClosedError

    import linkedin.session_orchestrator as so

    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)
    monkeypatch.setattr(so.config, "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", True)

    assert so._should_retry_session_error(
        "error: TargetClosedError",
        consecutive_error_resumes=0,
        profile_opens=42,
        error=TargetClosedError("target closed without a diagnostic message"),
    ) is True


def test_locator_timeout_after_profile_activity_remains_terminal(monkeypatch):
    import linkedin.session_orchestrator as so

    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)
    monkeypatch.setattr(so.config, "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", True)

    assert so._should_retry_session_error(
        "error: TimeoutError",
        consecutive_error_resumes=0,
        profile_opens=42,
        error=TimeoutError("Locator.wait_for: Timeout 30000ms exceeded"),
    ) is False


def test_api_timeout_after_profile_activity_remains_terminal(monkeypatch):
    import linkedin.session_orchestrator as so

    class APITimeoutError(TimeoutError):
        pass

    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)
    monkeypatch.setattr(so.config, "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", True)

    assert so._should_retry_session_error(
        "error: APITimeoutError",
        consecutive_error_resumes=0,
        profile_opens=1,
        error=APITimeoutError("Request timed out"),
    ) is False


def test_browser_crash_after_activity_resumes_after_dormant_gap(
    monkeypatch,
    capsys,
):
    import linkedin.session_orchestrator as so

    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)
    monkeypatch.setattr(so.config, "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", True)
    crash = RuntimeError("Locator.wait_for: Target crashed")
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "error: RuntimeError",
                "shutdown_kind": None,
                "stats": {},
                "error": crash,
                "profile_opens_session": 42,
            },
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
    )

    assert "raised" not in recorded
    assert [call["resume"] for call in recorded["run_calls"]] == [False, True]
    assert recorded["dormant_sample_calls"] == 1
    assert recorded["error_backoff_calls"] == 0
    assert recorded["wait_timeout"] == 1.0
    assert "dormant gap, then resuming" in capsys.readouterr().out


def test_consecutive_browser_crash_resumes_stop_at_configured_limit(
    monkeypatch,
):
    import linkedin.session_orchestrator as so

    monkeypatch.setattr(so.config, "LINKEDIN_CAMPAIGN_PERSIST", True)
    monkeypatch.setattr(so.config, "LINKEDIN_SESSION_ERROR_RETRIES", 2)
    monkeypatch.setattr(so.config, "LINKEDIN_BROWSER_CRASH_RESUME_ENABLED", True)
    crashes = [
        RuntimeError(f"Page crashed {index}")
        for index in range(3)
    ]
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "error: RuntimeError",
                "shutdown_kind": None,
                "stats": {},
                "error": crash,
                "profile_opens_session": 42,
            }
            for crash in crashes
        ],
    )

    assert len(recorded["run_calls"]) == 3
    assert recorded["raised"] is crashes[-1]
    assert recorded["dormant_sample_calls"] == 2
    assert recorded["error_backoff_calls"] == 0


@pytest.mark.parametrize("initial_resume", [False, True])
def test_run_day_cycle_retries_once_after_geography_apply_transient(
    monkeypatch,
    capsys,
    initial_resume,
):
    from linkedin.orchestrator import GeographyRegimeError

    transient = GeographyRegimeError("typeahead variance", retryable=True)
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "geography_apply_transient",
                "shutdown_kind": None,
                "stats": {},
                "error": transient,
            },
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
        resume=initial_resume,
        restart_string_ids=[7, 9],
    )

    out = capsys.readouterr().out
    assert len(recorded["run_calls"]) == 2
    assert [call["resume"] for call in recorded["run_calls"]] == [
        initial_resume,
        initial_resume,
    ]
    assert [
        call["restart_string_ids"] for call in recorded["run_calls"]
    ] == [[7, 9], [7, 9]]
    assert recorded["dormant_calls"] == 0
    assert recorded["wait_calls"] == 0
    assert recorded["session_end_reasons"] == [
        "geography_apply_transient",
        "pipeline_complete",
    ]
    assert (
        "[governor] Geography apply flaked on LinkedIn typeahead variance; "
        "retrying the session once."
    ) in out
    assert "Stopping day cycle: the brief's geography could not be applied" not in out


def test_run_day_cycle_breaks_after_two_consecutive_geography_apply_transients(
    monkeypatch,
    capsys,
):
    from linkedin.orchestrator import GeographyRegimeError

    first = GeographyRegimeError("first", retryable=True)
    second = GeographyRegimeError("second", retryable=True)
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "geography_apply_transient",
                "shutdown_kind": None,
                "stats": {},
                "error": first,
            },
            {
                "shutdown_reason": "geography_apply_transient",
                "shutdown_kind": None,
                "stats": {},
                "error": second,
            },
        ],
    )

    out = capsys.readouterr().out
    assert len(recorded["run_calls"]) == 2
    assert recorded["session_end_reasons"] == [
        "geography_apply_transient",
        "geography_apply_transient",
    ]
    assert out.count("retrying the session once") == 1
    assert "Stopping day cycle: the brief's geography could not be applied" in out
    assert recorded["raised"] is second


def test_run_day_cycle_breaks_immediately_on_non_retryable_geography_regime_error(
    monkeypatch,
    capsys,
):
    from linkedin.orchestrator import GeographyRegimeError

    error = GeographyRegimeError("bad facet")
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "geography_regime_error",
                "shutdown_kind": None,
                "stats": {},
                "error": error,
            },
        ],
    )

    out = capsys.readouterr().out
    assert len(recorded["run_calls"]) == 1
    assert recorded["session_end_reasons"] == ["geography_regime_error"]
    assert "retrying the session once" not in out
    assert recorded["raised"] is error


def test_multi_session_without_decoy_waits_then_resumes(monkeypatch):
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "session_duration_cap",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
    )

    assert [call["resume"] for call in recorded["run_calls"]] == [False, True]
    assert all(call["decoy"] is None for call in recorded["run_calls"])
    assert all(call["shared_browser"] is None for call in recorded["run_calls"])
    assert recorded["wait_calls"] == 1
    assert recorded["dormant_calls"] == 0
    assert "playwright_started" not in recorded


def test_resumed_multi_session_resumes_every_session(monkeypatch):
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "session_duration_cap",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
        resume=True,
    )

    assert [call["resume"] for call in recorded["run_calls"]] == [True, True]


def test_multi_session_with_decoy_owns_shared_connection_and_cleanup(monkeypatch):
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "session_duration_cap",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
        with_decoy=True,
    )

    assert [call["resume"] for call in recorded["run_calls"]] == [False, True]
    assert all(
        call["decoy"] is recorded["decoy"] for call in recorded["run_calls"]
    )
    assert all(call["shared_browser"] is not None for call in recorded["run_calls"])
    assert all(
        call["shared_context"] is recorded["decoy_context"]
        for call in recorded["run_calls"]
    )
    assert recorded["dormant_calls"] == 1
    assert recorded["wait_calls"] == 0
    assert recorded["decoy_closed"] is True
    assert recorded["playwright_stopped"] is True


def test_multi_session_cleanup_failures_are_fail_soft_and_both_attempted(
    monkeypatch,
):
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
        with_decoy=True,
        decoy_close_error=RuntimeError("decoy close failed"),
        playwright_stop_error=RuntimeError("playwright stop failed"),
    )

    assert "raised" not in recorded
    assert recorded["decoy_close_attempted"] is True
    assert recorded["playwright_stop_attempted"] is True


def test_multi_session_operational_error_beats_shared_stop_signal(
    monkeypatch,
):
    error = RuntimeError("interleave failed")
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": "error: interleave failed",
                "shutdown_kind": None,
                "stats": {},
                "error": error,
                "set_operator_stop": True,
            },
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
        with_decoy=True,
    )

    assert len(recorded["run_calls"]) == 1
    assert recorded["dormant_calls"] == 0
    assert recorded["raised"] is error


@pytest.mark.parametrize("reason", ["operator_stop", "profile_open_cap"])
def test_multi_session_controlled_stop_does_not_schedule_another_session(
    monkeypatch,
    reason,
):
    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": reason,
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
    )

    assert len(recorded["run_calls"]) == 1
    assert recorded["wait_calls"] == 0
    assert "raised" not in recorded


@pytest.mark.parametrize(
    "kind",
    ["generic", "api_budget", "constraint", "preflight"],
)
def test_multi_session_propagates_abnormal_error_without_retry(
    monkeypatch,
    kind,
):
    if kind == "api_budget":
        from shared.failures import ApiBudgetExhaustedError

        error = ApiBudgetExhaustedError("credits")
        reason = "api_budget_exhausted"
    elif kind == "constraint":
        from shared.constraint_manifest import ConstraintManifestError

        error = ConstraintManifestError("no owner")
        reason = "constraint_manifest_error"
    elif kind == "preflight":
        from shared.preflight_v2 import PreflightRegimeError

        error = PreflightRegimeError("bad criteria")
        reason = "preflight_regime_error"
    else:
        error = RuntimeError("boom")
        reason = "error: boom"

    recorded = _run_day_cycle_with_session_results(
        monkeypatch,
        [
            {
                "shutdown_reason": reason,
                "shutdown_kind": None,
                "stats": {},
                "error": error,
            },
            {
                "shutdown_reason": "pipeline_complete",
                "shutdown_kind": None,
                "stats": {},
                "error": None,
            },
        ],
        with_decoy=True,
    )

    assert len(recorded["run_calls"]) == 1
    assert recorded["dormant_calls"] == 0
    assert recorded["raised"] is error
    assert recorded["decoy_closed"] is True
    assert recorded["playwright_stopped"] is True


def test_multi_session_missing_shared_context_fails_without_retry_and_cleans_up(
    monkeypatch,
):
    import linkedin.session_orchestrator as so

    recorded: dict[str, bool] = {}

    class FakeBrowser:
        contexts: list[object] = []

    class FakeChromium:
        async def connect_over_cdp(self, _url):
            recorded["connected"] = True
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

        async def stop(self):
            recorded["stopped"] = True

    class FakePlaywrightManager:
        async def start(self):
            return FakePlaywright()

    monkeypatch.setattr(
        "rebrowser_playwright.async_api.async_playwright",
        lambda: FakePlaywrightManager(),
    )
    monkeypatch.setattr(so.signal, "signal", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="No browser contexts found"):
        asyncio.run(
            so._run_day_cycle_with_browser_lock(
                brief_path="brief.json",
                output_dir=None,
                multi_session=True,
                with_decoy=True,
            )
        )

    assert recorded == {"connected": True, "stopped": True}


def test_model_configuration_failure_precedes_shared_browser_connection(
    monkeypatch,
):
    import linkedin.session_orchestrator as so
    from linkedin.orchestrator import Pipeline

    browser_started = MagicMock()
    monkeypatch.setattr(
        Pipeline,
        "_validate_judgment_runtime_configuration",
        MagicMock(side_effect=RuntimeError("FIREWORKS_API_KEY missing")),
    )
    monkeypatch.setattr(
        "rebrowser_playwright.async_api.async_playwright",
        browser_started,
    )

    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        asyncio.run(
            so._run_day_cycle_with_browser_lock(
                brief_path="brief.json",
                output_dir=None,
                multi_session=True,
                with_decoy=True,
            )
        )

    browser_started.assert_not_called()


def test_session_error_shutdown_gives_constraint_manifest_a_stable_reason():
    """P3b (Wave 2): a ConstraintManifestError session is the same terminal
    configuration-failure class as a geography regime failure — every retried
    session hits the same zero-owner constraint. Stable reason, cycle breaks."""
    from shared.constraint_manifest import ConstraintManifestError
    from linkedin.session_orchestrator import _session_error_shutdown
    from shared.cooldown import ShutdownKind

    reason, kind = _session_error_shutdown(
        ConstraintManifestError("compensation stated, zero owners")
    )
    assert reason == "constraint_manifest_error"
    assert kind == ShutdownKind.ERROR


def test_session_error_shutdown_gives_preflight_regime_a_stable_reason():
    """Codex review, Wave 3 (F1): a PreflightRegimeError session — the
    generated-brief lint wall included, since the retry logic wraps
    GeneratedBriefLintError into it — is the same terminal configuration-
    failure class as geography/constraint regime errors. Every retried
    session re-runs preflight into the same lint wall, burning LLM calls
    while presenting as a generic transient error."""
    from shared.preflight_v2 import PreflightRegimeError
    from linkedin.session_orchestrator import _session_error_shutdown
    from shared.cooldown import ShutdownKind

    reason, kind = _session_error_shutdown(
        PreflightRegimeError("generated brief failed lint twice")
    )
    assert reason == "preflight_regime_error"
    assert kind == ShutdownKind.ERROR


def _run_sourcing_session_with_fake_pipeline(exc: BaseException):
    import linkedin.session_orchestrator as so

    async def drive():
        stop_event = asyncio.Event()
        seen: dict = {}

        class FakeBrowser:
            def __init__(self):
                self.connect = AsyncMock()

            def attach_existing_connection(self, *_args, **_kwargs):
                seen["attached"] = True

        class FakePipeline:
            def __init__(self, **_kwargs):
                self.browser = FakeBrowser()
                self.stats = {}
                self._progress = None

            async def run(self):
                raise AssertionError("legacy Pipeline.run must not be called")

            async def run_full(self, **_kwargs):
                seen["operator_stop_event_threaded"] = self._operator_stop_event is stop_event
                raise exc

        decoy = MagicMock()
        decoy.execute_burst = AsyncMock(return_value=[])
        with patch("linkedin.orchestrator.Pipeline", FakePipeline):
            result = await so._run_sourcing_session(
                brief_path="brief.json",
                output_dir=None,
                resume=True,
                input_mode="concurrent",
                governor=SessionGovernor(),
                decoy=decoy,
                session_duration=3600,
                operator_stop_event=stop_event,
            )
        return result, seen

    return asyncio.run(drive())


def _run_sourcing_session_with_controlled_pipeline(
    run_full,
    *,
    governor=None,
    decoy=None,
):
    import linkedin.session_orchestrator as so

    async def drive():
        stop_event = asyncio.Event()
        seen: dict = {}

        class FakePipeline:
            def __init__(self, **_kwargs):
                self.browser = MagicMock()
                self.stats = {}
                self._progress = None
                seen["pipeline"] = self

            async def run_full(self, **_kwargs):
                return await run_full(self)

        with patch("linkedin.orchestrator.Pipeline", FakePipeline):
            result = await so._run_sourcing_session(
                brief_path="brief.json",
                output_dir=None,
                resume=True,
                input_mode="concurrent",
                governor=governor or MagicMock(),
                decoy=decoy,
                session_duration=3600,
                operator_stop_event=stop_event,
            )
        return result, seen, stop_event

    return asyncio.run(drive())


def test_run_sourcing_session_classifies_operator_stop_as_interrupted():
    from shared.cooldown import ShutdownKind
    from shared.governor import OperatorStopRequested

    result, seen = _run_sourcing_session_with_fake_pipeline(OperatorStopRequested())

    assert seen["operator_stop_event_threaded"] is True
    assert result["shutdown_reason"] == "operator_stop"
    assert result["shutdown_kind"] == ShutdownKind.INTERRUPTED


def test_run_sourcing_session_classifies_keyboard_interrupt_without_pipeline_complete():
    from shared.cooldown import ShutdownKind

    result, seen = _run_sourcing_session_with_fake_pipeline(KeyboardInterrupt())

    assert seen["operator_stop_event_threaded"] is True
    assert result["shutdown_reason"] == "operator_stop"
    assert result["shutdown_kind"] == ShutdownKind.INTERRUPTED


def test_run_sourcing_session_returns_unexpected_error_to_caller():
    error = RuntimeError("boom")
    result, _ = _run_sourcing_session_with_fake_pipeline(error)

    assert result["shutdown_reason"] == "error: boom"
    assert result["error"] is error


def test_run_sourcing_session_without_decoy_has_no_prelock_connection_or_interleave():
    import linkedin.session_orchestrator as so

    observed: dict = {}

    class FakeBrowser:
        def __init__(self):
            self.connect = AsyncMock()

        def attach_existing_connection(self, *_args, **_kwargs):
            raise AssertionError("sourcing-only session received shared browser")

    class FakePipeline:
        def __init__(self, **_kwargs):
            self.browser = FakeBrowser()
            self.stats = {}
            self._progress = MagicMock()
            self.progress_path = "projection-must-not-be-written.json"
            observed["pipeline"] = self

        async def run_full(self, **_kwargs):
            observed["run_full"] = True

        async def run(self):
            raise AssertionError("legacy Pipeline.run must not be called")

    with patch("linkedin.orchestrator.Pipeline", FakePipeline), patch.object(
        so,
        "BurstScheduler",
        side_effect=AssertionError("sourcing-only session built interleave scheduler"),
    ):
        result = asyncio.run(
            so._run_sourcing_session(
                brief_path="brief.json",
                output_dir=None,
                resume=True,
                input_mode="concurrent",
                governor=SessionGovernor(),
                decoy=None,
                session_duration=3600,
            )
        )

    assert observed["run_full"] is True
    observed["pipeline"].browser.connect.assert_not_awaited()
    observed["pipeline"]._progress.save.assert_not_called()
    assert result["shutdown_reason"] == "pipeline_complete"


def test_status_task_failure_is_fail_soft(monkeypatch, capsys):
    import linkedin.session_orchestrator as so

    real_sleep = asyncio.sleep

    async def immediate_sleep(_delay):
        await real_sleep(0)

    async def complete_after_status_runs(_pipeline):
        for _ in range(4):
            await real_sleep(0)

    governor = MagicMock()
    governor.status_line.side_effect = RuntimeError("status render failed")
    monkeypatch.setattr(so.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(so.time, "time", MagicMock(side_effect=[0.0, 301.0]))

    result, _seen, _stop_event = _run_sourcing_session_with_controlled_pipeline(
        complete_after_status_runs,
        governor=governor,
    )

    assert result["shutdown_reason"] == "pipeline_complete"
    assert result["error"] is None
    assert "status render failed" in capsys.readouterr().out


def test_timer_task_failure_stops_session_and_returns_error(monkeypatch):
    import linkedin.session_orchestrator as so
    from shared.governor import OperatorStopRequested

    real_sleep = asyncio.sleep
    timer_error = RuntimeError("session timer failed")

    async def fail_only_timer(_delay):
        task_name = asyncio.current_task().get_coro().__qualname__
        if "_session_timer" in task_name:
            raise timer_error
        await real_sleep(0)

    async def wait_for_stop(pipeline):
        while not pipeline._operator_stop_event.is_set():
            await real_sleep(0)
        raise OperatorStopRequested()

    monkeypatch.setattr(so.asyncio, "sleep", fail_only_timer)
    result, _seen, stop_event = _run_sourcing_session_with_controlled_pipeline(
        wait_for_stop
    )

    assert stop_event.is_set()
    assert result["shutdown_reason"] == "error: session timer failed"
    assert result["error"] is timer_error


def test_interleave_failure_stops_session_and_returns_error(monkeypatch):
    import linkedin.session_orchestrator as so
    from shared.governor import OperatorStopRequested

    real_sleep = asyncio.sleep
    interleave_error = RuntimeError("decoy burst failed")

    async def immediate_sleep(_delay):
        await real_sleep(0)

    async def wait_for_stop(pipeline):
        while not pipeline._operator_stop_event.is_set():
            if pipeline._pause_requested.is_set():
                pipeline._pause_requested.clear()
                await pipeline._resume_event.wait()
            await real_sleep(0)
        raise OperatorStopRequested()

    scheduler = MagicMock()
    scheduler.next_interval.return_value = 0.0
    decoy = MagicMock()
    decoy.execute_burst = AsyncMock(side_effect=interleave_error)
    monkeypatch.setattr(so.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(so, "BurstScheduler", MagicMock(return_value=scheduler))

    result, _seen, stop_event = _run_sourcing_session_with_controlled_pipeline(
        wait_for_stop,
        decoy=decoy,
    )

    assert stop_event.is_set()
    assert result["shutdown_reason"] == "error: decoy burst failed"
    assert result["error"] is interleave_error
    decoy.execute_burst.assert_awaited_once()


def _run_main_without_launch(monkeypatch, argv: list[str]) -> dict:
    import linkedin.session_orchestrator as so

    recorded: dict = {}

    def fake_run_day_cycle(**kwargs):
        recorded["run_day_cycle_kwargs"] = kwargs
        return "fake-day-cycle"

    def fake_asyncio_run(value):
        recorded["asyncio_run_value"] = value

    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(so, "enable_console_tee", lambda _state_dir: None)
    monkeypatch.setattr(so, "run_day_cycle", fake_run_day_cycle)
    monkeypatch.setattr(so.asyncio, "run", fake_asyncio_run)
    so.main()
    return recorded


def test_main_rejects_flagless_generated_brief_when_resume_has_pending_work(
    tmp_path,
    monkeypatch,
    capsys,
):
    import linkedin.session_orchestrator as so

    brief_path = tmp_path / "brief.json"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_minimal_brief(brief_path, project_id="fresh-gate")
    (state_dir / "preflight_v2_brief.json").write_text("{}")

    monkeypatch.setattr(
        "sys.argv",
        [
            "session_orchestrator.py",
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
        ],
    )
    monkeypatch.setattr(so, "enable_console_tee", lambda _state_dir: None)
    monkeypatch.setattr(so, "_resume_has_pending_work", lambda *_args, **_kwargs: True)
    # The runless-resume refusal is exercised by its own tests
    # (test_assert_resume_target_*); these day-cycle mechanics tests use a
    # fabricated state dir, so neutralize it the same way the passive
    # pending-work check above is neutralized.
    monkeypatch.setattr(so, "_assert_resume_target_exists", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(so, "run_day_cycle", lambda **_kwargs: "unexpected-day-cycle")
    monkeypatch.setattr(
        so.asyncio,
        "run",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("fresh-regeneration gate did not stop launch")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        so.main()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--fresh" in err
    assert "--resume" in err


def test_main_fresh_bypasses_generated_brief_pending_resume_gate(tmp_path, monkeypatch):
    import linkedin.session_orchestrator as so

    brief_path = tmp_path / "brief.json"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_minimal_brief(brief_path, project_id="fresh-bypass")
    (state_dir / "preflight_v2_brief.json").write_text("{}")
    monkeypatch.setattr(so, "_resume_has_pending_work", lambda *_args, **_kwargs: True)
    # The runless-resume refusal is exercised by its own tests
    # (test_assert_resume_target_*); these day-cycle mechanics tests use a
    # fabricated state dir, so neutralize it the same way the passive
    # pending-work check above is neutralized.
    monkeypatch.setattr(so, "_assert_resume_target_exists", lambda *_args, **_kwargs: None)

    recorded = _run_main_without_launch(
        monkeypatch,
        [
            "session_orchestrator.py",
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
            "--fresh",
        ],
    )

    assert recorded["asyncio_run_value"] == "fake-day-cycle"
    assert recorded["run_day_cycle_kwargs"]["resume"] is False
    assert recorded["run_day_cycle_kwargs"]["output_dir"] == str(state_dir)
    assert recorded["run_day_cycle_kwargs"]["multi_session"] is False
    assert recorded["run_day_cycle_kwargs"]["with_decoy"] is False


@pytest.mark.parametrize(
    ("flags", "multi_session", "with_decoy"),
    [
        (["--single-session"], False, False),
        (["--multi-session"], True, False),
        (["--multi-session", "--with-decoy"], True, True),
    ],
)
def test_main_session_mode_contract(
    tmp_path,
    monkeypatch,
    flags,
    multi_session,
    with_decoy,
):
    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id="session-mode")

    recorded = _run_main_without_launch(
        monkeypatch,
        [
            "session_orchestrator.py",
            "--brief",
            str(brief_path),
            *flags,
        ],
    )

    assert recorded["run_day_cycle_kwargs"]["multi_session"] is multi_session
    assert recorded["run_day_cycle_kwargs"]["with_decoy"] is with_decoy


def test_main_rejects_decoy_without_multi_session(
    tmp_path,
    monkeypatch,
    capsys,
):
    import linkedin.session_orchestrator as so

    brief_path = tmp_path / "brief.json"
    _write_minimal_brief(brief_path, project_id="invalid-decoy")
    monkeypatch.setattr(
        "sys.argv",
        [
            "session_orchestrator.py",
            "--brief",
            str(brief_path),
            "--with-decoy",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        so.main()

    assert exc_info.value.code == 2
    assert "--with-decoy requires --multi-session" in capsys.readouterr().err


def test_main_resume_bypasses_generated_brief_pending_resume_gate(tmp_path, monkeypatch):
    import linkedin.session_orchestrator as so

    brief_path = tmp_path / "brief.json"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_minimal_brief(brief_path, project_id="resume-bypass")
    (state_dir / "preflight_v2_brief.json").write_text("{}")
    monkeypatch.setattr(so, "_resume_has_pending_work", lambda *_args, **_kwargs: True)
    # The runless-resume refusal is exercised by its own tests
    # (test_assert_resume_target_*); these day-cycle mechanics tests use a
    # fabricated state dir, so neutralize it the same way the passive
    # pending-work check above is neutralized.
    monkeypatch.setattr(so, "_assert_resume_target_exists", lambda *_args, **_kwargs: None)

    recorded = _run_main_without_launch(
        monkeypatch,
        [
            "session_orchestrator.py",
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
            "--resume",
        ],
    )

    assert recorded["asyncio_run_value"] == "fake-day-cycle"
    assert recorded["run_day_cycle_kwargs"]["resume"] is True
    assert recorded["run_day_cycle_kwargs"]["output_dir"] == str(state_dir)


def test_main_flagless_execution_plan_with_pending_work_refuses_regeneration(
    tmp_path,
    monkeypatch,
    capsys,
):
    import linkedin.session_orchestrator as so

    brief_path = tmp_path / "brief.json"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_minimal_brief(brief_path, project_id="execution-plan-gate")
    (state_dir / "execution_plan.json").write_text("{}")
    monkeypatch.setattr(so, "_resume_has_pending_work", lambda *_args, **_kwargs: True)
    # The runless-resume refusal is exercised by its own tests
    # (test_assert_resume_target_*); these day-cycle mechanics tests use a
    # fabricated state dir, so neutralize it the same way the passive
    # pending-work check above is neutralized.
    monkeypatch.setattr(so, "_assert_resume_target_exists", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "session_orchestrator.py",
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
        ],
    )
    monkeypatch.setattr(so, "enable_console_tee", lambda _state_dir: None)
    monkeypatch.setattr(so, "run_day_cycle", lambda **_kwargs: "unexpected-day-cycle")
    monkeypatch.setattr(
        so.asyncio,
        "run",
        lambda _value: (_ for _ in ()).throw(
            AssertionError("execution-plan fresh-regeneration gate did not stop launch")
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        so.main()

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--fresh" in err
    assert "--resume" in err


@pytest.mark.parametrize(
    ("flag", "resume"),
    [("--resume", True), ("--fresh", False)],
)
def test_main_explicit_mode_bypasses_execution_plan_pending_gate(
    tmp_path,
    monkeypatch,
    flag,
    resume,
):
    import linkedin.session_orchestrator as so

    brief_path = tmp_path / "brief.json"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_minimal_brief(brief_path, project_id="execution-plan-explicit")
    (state_dir / "execution_plan.json").write_text("{}")
    monkeypatch.setattr(so, "_resume_has_pending_work", lambda *_args, **_kwargs: True)
    # The runless-resume refusal is exercised by its own tests
    # (test_assert_resume_target_*); these day-cycle mechanics tests use a
    # fabricated state dir, so neutralize it the same way the passive
    # pending-work check above is neutralized.
    monkeypatch.setattr(so, "_assert_resume_target_exists", lambda *_args, **_kwargs: None)

    recorded = _run_main_without_launch(
        monkeypatch,
        [
            "session_orchestrator.py",
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
            flag,
        ],
    )

    assert recorded["asyncio_run_value"] == "fake-day-cycle"
    assert recorded["run_day_cycle_kwargs"]["resume"] is resume
    assert recorded["run_day_cycle_kwargs"]["output_dir"] == str(state_dir)


def test_main_flagless_new_project_without_generated_artifacts_bypasses_gate(
    tmp_path,
    monkeypatch,
):
    import linkedin.session_orchestrator as so

    brief_path = tmp_path / "brief.json"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_minimal_brief(brief_path, project_id="new-project")
    monkeypatch.setattr(so, "_resume_has_pending_work", lambda *_args, **_kwargs: True)
    # The runless-resume refusal is exercised by its own tests
    # (test_assert_resume_target_*); these day-cycle mechanics tests use a
    # fabricated state dir, so neutralize it the same way the passive
    # pending-work check above is neutralized.
    monkeypatch.setattr(so, "_assert_resume_target_exists", lambda *_args, **_kwargs: None)

    recorded = _run_main_without_launch(
        monkeypatch,
        [
            "session_orchestrator.py",
            "--brief",
            str(brief_path),
            "--state-dir",
            str(state_dir),
        ],
    )

    assert recorded["asyncio_run_value"] == "fake-day-cycle"
    assert recorded["run_day_cycle_kwargs"]["resume"] is False
    assert recorded["run_day_cycle_kwargs"]["output_dir"] == str(state_dir)


def test_sample_session_duration_draws_uniformly_within_the_4_to_5h_band():
    """Sam's 2026-07-07 budget ruling (widened same day): 4-5h, randomly
    drawn per session. Bounds over many draws + spread (a fixed cap must
    fail)."""
    from linkedin.session_orchestrator import _sample_session_duration

    draws = [_sample_session_duration() for _ in range(500)]
    assert all(14400.0 <= d <= 18000.0 for d in draws)
    assert max(draws) - min(draws) > 1200  # genuinely random, not a constant


def test_default_day_cycle_is_one_sourcing_only_session(monkeypatch):
    import linkedin.session_orchestrator as so

    sampled_duration = 16666.0
    recorded: dict = {}

    class FakeGovernor:
        def can_start_session(self, session_type="linkedin_sourcing"):
            return True, "ok"

        def start_session(self, session_duration_seconds=None):
            recorded["governor_start_duration"] = session_duration_seconds

        def end_session(self):
            return {"profile_opens_session": 0}

    class FakeDecoy:
        def __init__(self, _context):
            raise AssertionError("default session should not construct a decoy")

    def fail_playwright_start():
        raise AssertionError(
            "default session should not pre-connect a shared browser"
        )

    async def fake_run_sourcing_session(**kwargs):
        recorded.setdefault("run_calls", []).append(kwargs)
        recorded["run_session_duration"] = kwargs["session_duration"]
        return {
            "shutdown_reason": "pipeline_complete",
            "shutdown_kind": None,
            "stats": {},
            "error": None,
        }

    monkeypatch.setattr(so, "SessionGovernor", FakeGovernor)
    monkeypatch.setattr(so, "DecoyAgent", FakeDecoy)
    monkeypatch.setattr(so, "_sample_session_duration", lambda: sampled_duration)
    monkeypatch.setattr(so, "_run_sourcing_session", fake_run_sourcing_session)
    monkeypatch.setattr(so.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "rebrowser_playwright.async_api.async_playwright",
        fail_playwright_start,
    )
    monkeypatch.setattr(so.cooldown, "record_session_start", lambda session_type: 42)
    monkeypatch.setattr(so.cooldown, "get_sessions_today", lambda session_type: 1)
    monkeypatch.setattr(so.cooldown, "get_profile_opens_24h", lambda: 0)
    monkeypatch.setattr(
        so.cooldown,
        "record_session_end",
        lambda **kwargs: recorded.setdefault("record_session_end", kwargs),
    )

    asyncio.run(
        so.run_day_cycle(
            brief_path="brief.json",
            output_dir=None,
        )
    )

    assert len(recorded["run_calls"]) == 1
    assert recorded["run_calls"][0]["decoy"] is None
    assert recorded["run_calls"][0]["shared_browser"] is None
    assert recorded["run_calls"][0]["shared_context"] is None
    assert recorded["governor_start_duration"] == sampled_duration
    assert recorded["run_session_duration"] == sampled_duration
    assert "decoy_closed" not in recorded
    assert "playwright_stopped" not in recorded
