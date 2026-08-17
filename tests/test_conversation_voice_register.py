"""Golden snapshot discipline for recruiter-facing companion prose."""

from __future__ import annotations

from pathlib import Path

import pytest

import shared.output_paths as output_paths
from cloris.conversation import voice_discipline
from cloris.conversation.agent import ConversationAgent


FIXTURE_QUERIES: tuple[str, ...] = (
    "Anything surprise you on this LinkedIn sweep?",
    "Where should I spend attention first?",
    "How noisy was the funnel?",
    "Did GitHub specialists contribute?",
    "What did the nightly cadence look like?",
    "Any circuit-breaker pauses I should know about?",
    "How many surfaced saves cleared the facial gate?",
    "Are we stalled on profiling?",
    "Where is orchestration sequencing headed?",
    "Summarize today's cross-specialist stance.",
)


GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "conversation_voice_register_golden.txt"


def test_voice_fixture_queries_banned_tokens_and_tropes(monkeypatch, tmp_path) -> None:
    orch = tmp_path / "voice_orch.sqlite3"

    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        output_paths, "resolve_orchestration_db_path", lambda: orch
    )
    monkeypatch.setattr(
        "cloris.conversation.context.state_dirs_for_brief_id",
        lambda _bid, **kwargs: [],
    )

    counter = {"i": 0}

    def _fake_llm(
        _s: str,
        _u: str,
        expect_json: bool = True,
        usage_context: dict | None = None,
    ) -> str:
        idx = counter["i"]
        counter["i"] = idx + 1
        return (
            "Across LinkedIn I'm watching run telemetry segment "
            f"{idx} anchored on grounded counts."
        )

    monkeypatch.setattr("cloris.conversation.agent.cheap_llm", _fake_llm)
    monkeypatch.setattr(
        "cloris.conversation.agent._has_llm_access",
        lambda: True,
    )

    agent = ConversationAgent(state_root=None)
    lines: list[str] = []
    for q in FIXTURE_QUERIES:
        res = agent.answer(brief_id="voice_brief", message=q)
        text = res.assistant_text
        lines.append(text)

        violations = voice_discipline.voice_violations(text)
        assert not violations, violations

        lowered = text.lower()
        for trope in voice_discipline.CHATBOT_TROPE_SUBSTRINGS:
            assert trope not in lowered

    assembled = "\n---\n".join(lines).strip() + "\n"
    expected = GOLDEN_PATH.read_text(encoding="utf-8")
    assert assembled == expected
