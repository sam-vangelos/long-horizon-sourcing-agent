"""Tests for the conversational intake orchestrator (Phase C2).

Covers:

- ``stream_next_turn`` happy path: stub Opus stream → concatenated
  deltas equal expected response, usage tuple lands at end-of-stream.
- Failure-mode contract: stream raises mid-token → trailing
  ``LLM_PARTIAL_INTERRUPT`` delta + ``("degraded", ...)`` marker +
  synthetic empty-usage tuple, generator closes cleanly.
- Failure-mode contract: stream raises BEFORE any delta → no fallback
  delta is emitted; only the ``("degraded", ...)`` marker + synthetic
  usage tuple, so the C5 endpoint translates the marker into a
  structurally distinct ``degraded`` SSE event (audit finding F-2).
- ``build_orchestrator_system_prompt`` — substring assertions on the
  scaffold sections so C9 has a stable reference for what to extend
  vs. what to rewrite.
- ``build_orchestrator_user_prompt`` — JSON shape + source_packet
  presence/absence handling.

The Opus stream is monkeypatched at the
``shared.intake_conversation.orchestrator.opus_llm_cached_stream``
binding (NOT at the ``shared.llm_clients`` source) so tests don't
require an Anthropic API key.
"""

from __future__ import annotations

import json

import pytest

from shared.intake_conversation import ConversationMessage
from shared.intake_conversation.orchestrator import (
    DEGRADED_REASON_PROVIDER_FAILED,
    stream_next_turn,
)
from shared.intake_conversation.prompts import (
    LLM_PARTIAL_INTERRUPT,
    build_orchestrator_system_prompt,
    build_orchestrator_user_prompt,
)


def _msg(role: str, content: str, ts: str = "2026-05-13T12:00:00+00:00") -> ConversationMessage:
    return {"role": role, "content": content, "ts": ts}  # type: ignore[typeddict-item]


def _fake_stream(deltas: list[str], usage: dict | None = None):
    """Build a fake opus_llm_cached_stream-shaped iterator."""

    def _factory(system_prompt, user_prompt, *, usage_context=None):
        for d in deltas:
            yield ("delta", d)
        yield ("usage", usage or {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        })

    return _factory


def _raising_stream(deltas_before_raise: list[str]):
    """Build a fake stream that yields N deltas then raises."""

    def _factory(system_prompt, user_prompt, *, usage_context=None):
        for d in deltas_before_raise:
            yield ("delta", d)
        raise RuntimeError("simulated mid-stream failure")

    return _factory


# -------------------------------------------------------------------------
# stream_next_turn — happy path
# -------------------------------------------------------------------------


def test_stream_next_turn_yields_concatenated_deltas_then_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _fake_stream(["Hi — ", "I'm Cloris. ", "Tell me about the role."]),
    )

    events = list(
        stream_next_turn(
            messages=[],
            v2_draft={},
            source_packet=None,
            sufficiency_state=(False, ["role_title"]),
        )
    )

    deltas = [p for k, p in events if k == "delta"]
    usages = [p for k, p in events if k == "usage"]

    assert "".join(deltas) == "Hi — I'm Cloris. Tell me about the role."
    assert len(usages) == 1
    assert usages[0]["input_tokens"] == 10
    assert usages[0]["output_tokens"] == 5


def test_stream_next_turn_passes_usage_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def _capturing(system_prompt, user_prompt, *, usage_context=None):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        captured["context"] = usage_context
        yield ("delta", "ok")
        yield ("usage", {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        })

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _capturing,
    )

    list(
        stream_next_turn(
            messages=[],
            v2_draft={},
            source_packet=None,
            sufficiency_state=(False, []),
            session_id=42,
            cap_state="soft",
            dropped_turn=True,
        )
    )

    assert captured["context"]["stage"] == "intake_conversation_orchestrator"
    assert captured["context"]["session_id"] == 42
    assert captured["context"]["cap_state"] == "soft"
    assert captured["context"]["dropped_turn"] is True


# -------------------------------------------------------------------------
# stream_next_turn — failure modes
# -------------------------------------------------------------------------


def test_stream_raises_before_any_delta_emits_degraded_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit finding F-2: provider failure with no streamed text emits a
    ``degraded`` marker only — no fallback delta. The C5 endpoint
    translates the marker into a structurally distinct SSE event so the
    recruiter sees a banner rather than a normal-shaped Cloris turn."""

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _raising_stream([]),
    )

    events = list(
        stream_next_turn(
            messages=[],
            v2_draft={},
            source_packet=None,
            sufficiency_state=(False, []),
        )
    )

    deltas = [p for k, p in events if k == "delta"]
    degradeds = [p for k, p in events if k == "degraded"]
    usages = [p for k, p in events if k == "usage"]

    assert deltas == []
    assert len(degradeds) == 1
    assert degradeds[0]["reason"] == DEGRADED_REASON_PROVIDER_FAILED
    assert degradeds[0]["any_delta"] is False
    assert len(usages) == 1
    # Synthetic usage means failed turn doesn't get billed to recruiter.
    assert usages[0]["input_tokens"] == 0
    assert usages[0]["output_tokens"] == 0


def test_stream_raises_mid_token_yields_partial_interrupt_then_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit finding F-2: provider failure mid-stream preserves the
    partial Cloris text, terminates with a ``LLM_PARTIAL_INTERRUPT``
    delta so the persisted transcript reads cleanly, then emits the
    ``degraded`` marker before the synthetic usage tuple."""

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _raising_stream(["Got it — let me ", "think about whether"]),
    )

    events = list(
        stream_next_turn(
            messages=[],
            v2_draft={},
            source_packet=None,
            sufficiency_state=(False, []),
        )
    )

    deltas = [p for k, p in events if k == "delta"]
    degradeds = [p for k, p in events if k == "degraded"]
    usages = [p for k, p in events if k == "usage"]

    # Partial deltas are preserved; an interrupt marker appended; THEN
    # the degraded marker; THEN the synthetic usage closes the stream.
    assert deltas[:2] == ["Got it — let me ", "think about whether"]
    assert deltas[-1] == LLM_PARTIAL_INTERRUPT
    assert len(degradeds) == 1
    assert degradeds[0]["any_delta"] is True
    assert degradeds[0]["reason"] == DEGRADED_REASON_PROVIDER_FAILED
    assert len(usages) == 1
    assert usages[0]["input_tokens"] == 0


def test_stream_failure_does_not_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The orchestrator MUST NOT raise — the C5 SSE endpoint depends on
    always being able to drain the generator and close the stream cleanly.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _raising_stream([]),
    )

    # Should not raise:
    events = list(
        stream_next_turn(
            messages=[],
            v2_draft={},
            source_packet=None,
            sufficiency_state=(False, []),
        )
    )
    assert events  # at least the fallback + usage tuples


# -------------------------------------------------------------------------
# build_orchestrator_system_prompt — section presence
# -------------------------------------------------------------------------


def test_system_prompt_cites_voice_guide_by_name() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "~/Downloads/Sam_Vangelos_Voice_Guide_v2.md" in prompt


def test_system_prompt_includes_role_section() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "# ROLE" in prompt
    assert "You are Cloris" in prompt


def test_system_prompt_includes_schema_description() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    lower = prompt.lower()
    assert "# BRIEF UNDERSTANDING" in prompt
    assert "capability areas" in lower
    assert "what separates someone who has really done" in lower
    assert "LinkedIn / GitHub / both" not in prompt


def test_system_prompt_includes_dropped_turn_branch_when_active() -> None:
    prompt_with = build_orchestrator_system_prompt(
        sufficiency_state=(False, []), dropped_turn=True
    )
    prompt_without = build_orchestrator_system_prompt(
        sufficiency_state=(False, []), dropped_turn=False
    )

    assert "# RESUME-FROM-DROPPED-TURN" in prompt_with
    assert "# RESUME-FROM-DROPPED-TURN" not in prompt_without
    # Substring is wrapped across a newline in the source; just check both parts.
    assert "Picking\nback up" in prompt_with or "Picking back up" in prompt_with
    assert "interrupted" in prompt_with


def test_system_prompt_includes_soft_cap_when_active() -> None:
    prompt = build_orchestrator_system_prompt(
        sufficiency_state=(False, []), cap_state="soft"
    )
    assert "# CAP STATE — SOFT" in prompt
    assert "20 turns" in prompt or "$1" in prompt


def test_system_prompt_includes_hard_cap_when_active() -> None:
    prompt = build_orchestrator_system_prompt(
        sufficiency_state=(False, []), cap_state="hard"
    )
    assert "# CAP STATE — HARD" in prompt
    assert "FORCE COMPOSITION" in prompt


def test_system_prompt_omits_cap_state_when_normal() -> None:
    prompt = build_orchestrator_system_prompt(
        sufficiency_state=(False, []), cap_state="normal"
    )
    assert "# CAP STATE" not in prompt


def test_system_prompt_includes_sufficiency_volunteer_when_ready() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(True, []))
    assert "# SUFFICIENCY STATE" in prompt
    # Case-insensitive match; the C9-rewritten sufficiency-volunteer
    # template uses "Want to see the brief I'd run with?".
    assert "want to see the brief" in prompt.lower()


def test_system_prompt_lists_missing_fields_when_not_ready() -> None:
    prompt = build_orchestrator_system_prompt(
        sufficiency_state=(False, ["role_title", "capability_areas[0].description"])
    )
    assert "job title" in prompt
    assert "capability with a concrete bar" in prompt


def test_system_prompt_includes_all_source_capabilities() -> None:
    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    for source in ("linkedin", "github", "researcher", "designer", "exec_search"):
        assert f"`{source}`" in prompt
    assert "Evidence boundaries are not permission to skip" in prompt


def test_system_prompt_voice_content_has_landed() -> None:
    """C9 replaced the C2 scaffold's TODO sections with substantive
    voice content. Pin the absence of TODO markers + the presence of
    the load-bearing section headings so a future regression that
    accidentally restored the scaffold would surface here.
    """

    prompt = build_orchestrator_system_prompt(sufficiency_state=(False, []))
    assert "TODO C9" not in prompt
    assert "# HOW TO CONSTRUCT SENTENCES" in prompt
    assert "# FORBIDDEN PATTERNS" in prompt
    assert "# SPECIFICITY RULE" in prompt


# -------------------------------------------------------------------------
# build_orchestrator_user_prompt — payload shape
# -------------------------------------------------------------------------


def test_user_prompt_serializes_messages_and_draft() -> None:
    user = build_orchestrator_user_prompt(
        messages=[
            _msg("cloris", "opener"),
            _msg("recruiter", "Senior tax associate at Northwind."),
        ],
        v2_draft={"role_title": "Senior Tax Associate"},
        source_packet=None,
    )

    assert user.startswith("INPUT:\n")
    payload = json.loads(user[len("INPUT:\n"):])
    assert len(payload["messages"]) == 2
    assert payload["messages"][1]["content"] == "Senior tax associate at Northwind."
    assert payload["v2_draft"] == {"role_title": "Senior Tax Associate"}
    assert "source_packet" not in payload


def test_user_prompt_includes_source_packet_when_present() -> None:
    user = build_orchestrator_user_prompt(
        messages=[],
        v2_draft={},
        source_packet={"raw_text": "JD body...", "geography": "NYC"},
    )

    payload = json.loads(user[len("INPUT:\n"):])
    assert payload["source_packet"]["raw_text"] == "JD body..."
    assert payload["source_packet"]["geography"] == "NYC"


def test_user_prompt_handles_empty_inputs_cleanly() -> None:
    user = build_orchestrator_user_prompt(
        messages=[],
        v2_draft={},
        source_packet=None,
    )

    payload = json.loads(user[len("INPUT:\n"):])
    assert payload["messages"] == []
    assert payload["v2_draft"] == {}
