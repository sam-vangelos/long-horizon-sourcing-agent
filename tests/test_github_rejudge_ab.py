"""Acceptance tests for tools/github_rejudge_ab.py."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from linkedin.judgment_templates import FullEvaluationResult
from shared.failures import ApiBudgetExhaustedError
from tools import github_rejudge_ab

BRIEF_PATH = Path("config/prre-code/brief-prre-code-github-v2.json")


def _payload(
    *,
    username: str,
    candidate_text: str,
    decision: str,
    confidence: float,
    source: str = "github",
    render_route: str = "github.full.v1",
    system_prompt_sha256: str | None = None,
) -> dict[str, Any]:
    prompt_capture: dict[str, Any] = {
        "candidate_text": candidate_text,
        "source": source,
        "render_route": render_route,
    }
    if system_prompt_sha256 is not None:
        prompt_capture["system_prompt_sha256"] = system_prompt_sha256
    return {
        "candidate_record": {"username": username},
        "prompt_capture": prompt_capture,
        "full_decision": {
            "decision": decision,
            "confidence": confidence,
        },
    }


def _insert_attempt(
    conn: sqlite3.Connection,
    *,
    attempt_id: int,
    stage: str,
    payload: dict[str, Any] | None,
    attempt_number: int = 1,
) -> None:
    conn.execute(
        """
        INSERT INTO candidate_attempts (
            id, run_id, candidate_id, stage, attempt_number, status,
            payload_json, started_at
        ) VALUES (?, 1, ?, ?, ?, 'succeeded', ?, '2026-08-01T00:00:00Z')
        """,
        (
            attempt_id,
            attempt_id,
            stage,
            attempt_number,
            json.dumps(payload) if payload is not None else "{}",
        ),
    )


def _make_fixture_db(state_dir: Path, payloads: list[tuple[str, dict[str, Any] | None]]) -> None:
    db_path = state_dir / "runtime_state.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE candidate_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            work_unit_id INTEGER,
            stage TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            batch_key TEXT,
            status TEXT NOT NULL,
            failure_kind TEXT,
            failure_reason TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            source_cursor_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL,
            ended_at TEXT
        )
        """
    )
    for attempt_id, (stage, payload) in enumerate(payloads, start=1):
        _insert_attempt(conn, attempt_id=attempt_id, stage=stage, payload=payload)
    conn.commit()
    conn.close()


def _snapshot_dir(path: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    if not path.exists():
        return snapshot
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            snapshot[str(item.relative_to(path))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _run_tool(
    tmp_path: Path,
    state_dir: Path,
    *,
    dry_run: bool = False,
    out_name: str = "report.md",
) -> tuple[int, str]:
    out_path = tmp_path / out_name
    exit_code = github_rejudge_ab.run_rejudge(
        state_dir=state_dir,
        brief_path=BRIEF_PATH,
        model_name="claude-cli:claude-opus-4-8",
        out_path=out_path,
        dry_run=dry_run,
    )
    return exit_code, out_path.read_text(encoding="utf-8")


def test_reads_only_full_stage_rows_with_capture(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            ("facial", _payload(username="face-only", candidate_text="Username: face-only\n", decision="FACIAL_YES", confidence=0.5)),
            ("full", {"full_decision": {"decision": "REJECT", "confidence": 0.1}}),
            (
                "full",
                _payload(
                    username="alpha",
                    candidate_text="Username: alpha\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
            (
                "full",
                _payload(
                    username="beta",
                    candidate_text="Username: beta\nEvidence",
                    decision="REJECT",
                    confidence=0.9,
                ),
            ),
            (
                "full",
                _payload(
                    username="gamma",
                    candidate_text="Username: gamma\nEvidence",
                    decision="TRANSFERABLE_SAVE",
                    confidence=0.7,
                ),
            ),
        ],
    )

    load_result = github_rejudge_ab._load_candidates(state_dir, limit=None)
    assert [row.username for row in load_result.candidates] == ["alpha", "beta", "gamma"]


def test_opens_state_readonly_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="alpha",
                    candidate_text="Username: alpha\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
        ],
    )

    with tempfile_copy_db(state_dir) as db_path:
        conn = github_rejudge_ab._open_state_db(db_path)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("CREATE TABLE readonly_probe (id INTEGER)")
        finally:
            conn.close()

    before = _snapshot_dir(state_dir)
    monkeypatch.setattr(github_rejudge_ab, "opus_llm_cached", lambda *args, **kwargs: "DECISION: SAVE\nCONFIDENCE: 0.8")
    exit_code, _ = _run_tool(tmp_path, state_dir, dry_run=False)
    after = _snapshot_dir(state_dir)

    assert exit_code == 0
    assert before == after


class tempfile_copy_db:
    """Copy fixture DB into a temp dir for readonly probe (mirrors production copy path)."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self._tmpdir: str | None = None
        self.db_path: Path | None = None

    def __enter__(self) -> Path:
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        tmp_path = Path(self._tmpdir)
        self.db_path = github_rejudge_ab._copy_state_db_snapshot(self.state_dir, tmp_path)
        return self.db_path

    def __exit__(self, *args: object) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)


def test_out_path_inside_state_dir_refused(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="alpha",
                    candidate_text="Username: alpha\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
        ],
    )

    with pytest.raises(SystemExit, match="--out must not be inside --state-dir"):
        github_rejudge_ab.run_rejudge(
            state_dir=state_dir,
            brief_path=BRIEF_PATH,
            model_name="claude-cli:claude-opus-4-8",
            out_path=state_dir / "report.md",
            dry_run=True,
        )


def test_out_path_case_insensitive_refused(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="alpha",
                    candidate_text="Username: alpha\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
        ],
    )

    with pytest.raises(SystemExit, match="--out must not be inside --state-dir"):
        github_rejudge_ab.run_rejudge(
            state_dir=state_dir,
            brief_path=BRIEF_PATH,
            model_name="claude-cli:claude-opus-4-8",
            out_path=tmp_path / "STATE" / "report.md",
            dry_run=True,
        )


def test_foreign_source_rows_skipped(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="github-user",
                    candidate_text="Username: github-user\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
            (
                "full",
                _payload(
                    username="linkedin-user",
                    candidate_text="Username: linkedin-user\nEvidence",
                    decision="REJECT",
                    confidence=0.5,
                    source="linkedin",
                    render_route="linkedin.full.v2",
                ),
            ),
        ],
    )

    load_result = github_rejudge_ab._load_candidates(state_dir, limit=None)
    assert [row.username for row in load_result.candidates] == ["github-user"]
    assert load_result.skipped_foreign == 1
    assert load_result.skipped_duplicate == 0


def test_duplicate_attempts_keep_latest(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    db_path = state_dir / "runtime_state.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE candidate_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            work_unit_id INTEGER,
            stage TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            batch_key TEXT,
            status TEXT NOT NULL,
            failure_kind TEXT,
            failure_reason TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            source_cursor_json TEXT NOT NULL DEFAULT '{}',
            started_at TEXT NOT NULL,
            ended_at TEXT
        )
        """
    )
    _insert_attempt(
        conn,
        attempt_id=1,
        stage="full",
        attempt_number=1,
        payload=_payload(
            username="dup-user",
            candidate_text="Username: dup-user\nOld",
            decision="REJECT",
            confidence=0.4,
        ),
    )
    _insert_attempt(
        conn,
        attempt_id=2,
        stage="full",
        attempt_number=3,
        payload=_payload(
            username="dup-user",
            candidate_text="Username: dup-user\nNew",
            decision="SAVE",
            confidence=0.9,
        ),
    )
    conn.commit()
    conn.close()

    load_result = github_rejudge_ab._load_candidates(state_dir, limit=None)
    assert len(load_result.candidates) == 1
    assert load_result.candidates[0].old_decision == "SAVE"
    assert "New" in load_result.candidates[0].candidate_text
    assert load_result.skipped_duplicate == 1


def test_malformed_payload_recorded_as_error_row(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="good-user",
                    candidate_text="Username: good-user\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
            (
                "full",
                {
                    "candidate_record": {"username": "bad-user"},
                    "prompt_capture": {
                        "candidate_text": "Username: bad-user\nEvidence",
                        "source": "github",
                        "render_route": "github.full.v1",
                    },
                    "full_decision": {"confidence": 0.5},
                },
            ),
        ],
    )

    monkeypatch.setattr(github_rejudge_ab, "opus_llm_cached", lambda *args, **kwargs: "DECISION: SAVE\nCONFIDENCE: 0.8")
    exit_code, report = _run_tool(tmp_path, state_dir, dry_run=False)
    assert exit_code == 2
    assert "| bad-user |  |  | | | | ERROR |" in report
    assert "- Errors: 1" in report


def test_report_header_uses_brief_json_id(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="alpha",
                    candidate_text="Username: alpha\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                    system_prompt_sha256="a" * 64,
                ),
            ),
        ],
    )

    _, report = _run_tool(tmp_path, state_dir, dry_run=True)
    assert "(`prre-code-github`)" in report
    assert "- **Selected:** 1" in report
    assert f"- **Old-prompt sha256 (stored):** `{('a' * 64)}`" in report
    assert "Old captures may predate evidence-format changes" in report


def test_confidence_only_movement_counted(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="stable-user",
                    candidate_text="Username: stable-user\nEvidence",
                    decision="SAVE",
                    confidence=0.80,
                ),
            ),
            (
                "full",
                _payload(
                    username="shift-user",
                    candidate_text="Username: shift-user\nEvidence",
                    decision="SAVE",
                    confidence=0.50,
                ),
            ),
        ],
    )

    responses = {
        "stable-user": "DECISION: SAVE\nCONFIDENCE: 0.85",
        "shift-user": "DECISION: SAVE\nCONFIDENCE: 0.65",
    }

    def _mock_llm(_system: str, candidate_text: str, **kwargs: Any) -> str:
        for username, raw in responses.items():
            if f"Username: {username}" in candidate_text:
                return raw
        raise AssertionError(f"unexpected candidate_text: {candidate_text!r}")

    monkeypatch.setattr(github_rejudge_ab, "opus_llm_cached", _mock_llm)

    exit_code, report = _run_tool(tmp_path, state_dir, dry_run=False)
    assert exit_code == 0
    assert "| stable-user | SAVE | 0.8 | SAVE | 0.85 | +0.05 | no |" in report
    assert "| shift-user | SAVE | 0.5 | SAVE | 0.65 | +0.15 | no |" in report
    assert "- Confidence moved ≥ 0.10 (same decision): 1" in report


def test_dry_run_emits_table_without_llm_calls(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="alpha",
                    candidate_text="Username: alpha\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
        ],
    )

    calls: list[tuple[Any, ...]] = []

    def _fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("opus_llm_cached must not be called in dry-run")

    monkeypatch.setattr(github_rejudge_ab, "opus_llm_cached", _fail_if_called)

    exit_code, report = _run_tool(tmp_path, state_dir, dry_run=True)
    assert exit_code == 0
    assert calls == []
    assert "| alpha | SAVE | 0.8 | (dry) |  |  | (dry) |" in report


def test_new_decisions_and_flips_reported(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="hold-user",
                    candidate_text="Username: hold-user\nEvidence",
                    decision="SAVE",
                    confidence=0.82,
                ),
            ),
            (
                "full",
                _payload(
                    username="flip-user",
                    candidate_text="Username: flip-user\nEvidence",
                    decision="SAVE",
                    confidence=0.75,
                ),
            ),
        ],
    )

    responses = {
        "hold-user": FullEvaluationResult(
            decision="SAVE",
            match_type="DIRECT",
            capability_area="area",
            capability_evidence="",
            depth="BUILDER",
            depth_evidence="",
            transferability="N/A",
            transferability_evidence="",
            case_for="",
            case_against="",
            confidence=0.85,
            post_save_modifier="NONE",
            summary="hold",
            raw_response="",
        ),
        "flip-user": FullEvaluationResult(
            decision="REJECT",
            match_type="NONE",
            capability_area=None,
            capability_evidence="",
            depth="USER",
            depth_evidence="",
            transferability="NOT_TRANSFERABLE",
            transferability_evidence="",
            case_for="",
            case_against="",
            confidence=0.91,
            post_save_modifier="NONE",
            summary="reject",
            raw_response="",
        ),
    }

    def _mock_llm(_system: str, candidate_text: str, **kwargs: Any) -> str:
        for username, result in responses.items():
            if f"Username: {username}" in candidate_text:
                return result.raw_response or f"DECISION: {result.decision}\nCONFIDENCE: {result.confidence}"
        raise AssertionError(f"unexpected candidate_text: {candidate_text!r}")

    def _mock_parse(raw: str) -> FullEvaluationResult:
        for result in responses.values():
            if result.decision in raw:
                return result
        raise AssertionError(f"unexpected raw response: {raw!r}")

    monkeypatch.setattr(github_rejudge_ab, "opus_llm_cached", _mock_llm)
    monkeypatch.setattr(github_rejudge_ab, "parse_full_evaluation_response", _mock_parse)

    exit_code, report = _run_tool(tmp_path, state_dir, dry_run=False)
    assert exit_code == 0
    assert "| hold-user | SAVE | 0.82 | SAVE | 0.85 | +0.03 | no |" in report
    assert "| flip-user | SAVE | 0.75 | REJECT | 0.91 | +0.16 | YES |" in report
    assert "- Unchanged: 1" in report
    assert "- SAVE→REJECT: 1" in report
    assert "- REJECT→SAVE: 0" in report
    assert "- Other decision changes: 0" in report
    assert "- Errors: 0" in report


def test_per_candidate_error_recorded_and_run_continues(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="user-one",
                    candidate_text="Username: user-one\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
            (
                "full",
                _payload(
                    username="user-two",
                    candidate_text="Username: user-two\nEvidence",
                    decision="REJECT",
                    confidence=0.9,
                ),
            ),
            (
                "full",
                _payload(
                    username="user-three",
                    candidate_text="Username: user-three\nEvidence",
                    decision="SAVE",
                    confidence=0.7,
                ),
            ),
        ],
    )

    call_count = {"n": 0}

    def _mock_llm(_system: str, candidate_text: str, **kwargs: Any) -> str:
        call_count["n"] += 1
        if "user-two" in candidate_text:
            raise RuntimeError("mock failure for user-two")
        return "DECISION: SAVE\nCONFIDENCE: 0.8"

    monkeypatch.setattr(github_rejudge_ab, "opus_llm_cached", _mock_llm)

    exit_code, report = _run_tool(tmp_path, state_dir, dry_run=False)
    assert exit_code == 2
    assert call_count["n"] == 3
    assert report.count("| user-one |") == 1
    assert "| user-two | REJECT | 0.9 | | | | ERROR |" in report
    assert report.count("| user-three |") == 1
    assert "- Errors: 1" in report


def test_budget_exhaustion_aborts_with_partial_report(tmp_path: Path, monkeypatch) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="user-one",
                    candidate_text="Username: user-one\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
            (
                "full",
                _payload(
                    username="user-two",
                    candidate_text="Username: user-two\nEvidence",
                    decision="REJECT",
                    confidence=0.9,
                ),
            ),
            (
                "full",
                _payload(
                    username="user-three",
                    candidate_text="Username: user-three\nEvidence",
                    decision="SAVE",
                    confidence=0.7,
                ),
            ),
        ],
    )

    call_count = {"n": 0}

    def _mock_llm(_system: str, candidate_text: str, **kwargs: Any) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "DECISION: SAVE\nCONFIDENCE: 0.81"
        raise ApiBudgetExhaustedError("provider credits exhausted")

    monkeypatch.setattr(github_rejudge_ab, "opus_llm_cached", _mock_llm)

    exit_code, report = _run_tool(tmp_path, state_dir, dry_run=False)
    assert exit_code == 1
    assert call_count["n"] == 2
    assert "| user-one | SAVE | 0.8 | SAVE | 0.81 | +0.01 | no |" in report
    assert "ABORTED (API budget exhausted)" in report
    assert "user-three" not in report


def test_immutable_flag_absent() -> None:
    uri = github_rejudge_ab.build_state_db_uri(Path("/tmp/runtime_state.sqlite3"))
    assert "mode=ro" in uri
    assert "immutable" not in uri


def test_direct_open_mutates_state_dir(tmp_path: Path, monkeypatch) -> None:
    """Red-proof: opening the live state DB directly can touch sidecar mtimes."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _make_fixture_db(
        state_dir,
        [
            (
                "full",
                _payload(
                    username="alpha",
                    candidate_text="Username: alpha\nEvidence",
                    decision="SAVE",
                    confidence=0.8,
                ),
            ),
        ],
    )

    conn = sqlite3.connect(state_dir / "runtime_state.sqlite3")
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    conn.close()

    before = _snapshot_dir(state_dir)

    def _direct_open_snapshot(state_dir_arg: Path, tmp_path_arg: Path) -> Path:
        return state_dir_arg / "runtime_state.sqlite3"

    monkeypatch.setattr(github_rejudge_ab, "_copy_state_db_snapshot", _direct_open_snapshot)
    monkeypatch.setattr(github_rejudge_ab, "opus_llm_cached", lambda *args, **kwargs: "DECISION: SAVE\nCONFIDENCE: 0.8")

    _run_tool(tmp_path, state_dir, dry_run=False)
    after = _snapshot_dir(state_dir)

    assert before != after, "direct open should mutate state-dir sidecar metadata"
