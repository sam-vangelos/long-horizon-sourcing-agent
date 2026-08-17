"""Contract tests for companion context + agent behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import shared.output_paths as output_paths
from cloris.conversation.agent import ConversationAgent
from cloris.conversation.context import build_live_context_and_citations
from shared.runtime_state.orchestration_store import OrchestrationStateStore


def _minimal_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            source TEXT,
            brief_id TEXT,
            output_dir TEXT,
            mode TEXT,
            status TEXT,
            stop_reason TEXT,
            started_at TEXT,
            ended_at TEXT,
            resumed_from_run_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY,
            identity_key TEXT,
            display_name TEXT,
            profile_url TEXT,
            terminal_decision TEXT,
            terminal_payload_json TEXT,
            last_seen_at TEXT,
            current_lifecycle_state TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS candidate_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            candidate_id INTEGER
        );
        """
    )


def test_build_live_context_empty_brief(monkeypatch, tmp_path: Path) -> None:
    orch = tmp_path / "orchestration" / "runtime_state.sqlite3"

    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        output_paths, "resolve_orchestration_db_path", lambda: orch
    )

    orch.parent.mkdir(parents=True, exist_ok=True)
    OrchestrationStateStore(orch)

    ctx, citations = build_live_context_and_citations(
        "no_such_workspace_brief_xyz",
    )
    assert ctx["brief_id"]
    assert ctx["specialists"] == []
    assert citations == []


def test_conversation_answer_persists_turns(monkeypatch, tmp_path: Path) -> None:
    state_root = tmp_path / "state"
    li = state_root / "linkedin" / "fixture_key"
    li.mkdir(parents=True)
    db_path = li / "runtime_state.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    _minimal_schema(conn)
    conn.execute(
        "INSERT INTO runs(id, source, brief_id, output_dir, mode, status, "
        "stop_reason, started_at, ended_at) "
        "VALUES (1, 'linkedin', 'fixture_brief', ?, 'fresh', "
        "'finished', '', '2026-01-01T00:00:00+00:00', "
        "'2026-01-02T00:00:00+00:00')",
        ("out/dir",),
    )
    conn.execute(
        "INSERT INTO candidates(id, identity_key, display_name, "
        "profile_url, terminal_decision, terminal_payload_json, last_seen_at) "
        "VALUES (1,'k','Ada','','SAVE','{}','2026-01-01')"
    )
    conn.execute(
        "INSERT INTO candidate_attempts(run_id, candidate_id) VALUES (1,1)"
    )
    conn.commit()
    conn.close()

    orch = tmp_path / "orch" / "runtime_state.sqlite3"
    orch.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        output_paths, "resolve_orchestration_db_path", lambda: orch
    )

    agent = ConversationAgent(state_root=state_root)

    def _fake_llm(
        system: str,
        user: str,
        expect_json: bool = True,
        usage_context: dict | None = None,
    ) -> str:
        return (
            "Across LinkedIn I returned one candidate marked SAVE on that run "
            "and stayed inside the scripted fixture."
        )

    monkeypatch.setattr("cloris.conversation.agent.cheap_llm", _fake_llm)
    monkeypatch.setattr(
        "cloris.conversation.agent._has_llm_access",
        lambda: True,
    )

    res = agent.answer(
        brief_id="fixture_brief",
        message="What's running?",
    )
    assert res.kind == "ok"
    assert "LinkedIn" in res.assistant_text

    store = OrchestrationStateStore(orch)
    tid = store.get_or_create_conversation_thread(brief_id="fixture_brief")
    hist = store.list_conversation_turns(thread_id=tid, limit=10)
    roles = [h["role"] for h in hist]
    assert "user" in roles and "assistant" in roles


def test_conversation_llm_failure_is_degraded(monkeypatch, tmp_path: Path) -> None:
    orch = tmp_path / "solo" / "runtime_state.sqlite3"
    orch.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        output_paths, "resolve_orchestration_db_path", lambda: orch
    )

    agent = ConversationAgent(state_root=None)

    def _boom(
        _s: str,
        _u: str,
        expect_json: bool = True,
        usage_context: dict | None = None,
    ) -> str:
        raise RuntimeError("simulated outage")

    monkeypatch.setattr("cloris.conversation.agent.cheap_llm", _boom)
    monkeypatch.setattr(
        "cloris.conversation.agent._has_llm_access",
        lambda: True,
    )
    monkeypatch.setattr(
        "cloris.conversation.context.state_dirs_for_brief_id",
        lambda _bid, **kwargs: [],
    )

    result = agent.answer(brief_id="solo_brief", message="Ping")
    assert result.kind == "degraded"
    assert "processing" in result.assistant_text.lower()
