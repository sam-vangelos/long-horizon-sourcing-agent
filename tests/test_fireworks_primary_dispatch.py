"""Provider dispatch for non-Anthropic primaries (the GLM promotion, item 19).

Pins:
- A Fireworks model id ("accounts/...") on the opus_llm family, facial_llm,
  or cheap_llm routes to the primary Fireworks client instead of raising —
  and never constructs an Anthropic client.
- Anthropic ids keep the exact pre-dispatch path.
- Primary posture: truncation (finish_reason != "stop") raises loudly,
  capture is filled BEFORE the raise, usage is recorded provider="fireworks".
- reasoning_content lands in capture["thinking_summary"] (the
  provider-agnostic key Anthropic primaries fill).
- The streaming intake path has no Fireworks route and says so.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import shared.judger as judger
import shared.config as config
from shared.failures import ApiBudgetExhaustedError
from shared.llm_clients import (
    cheap_llm,
    facial_llm,
    get_llm_client,
    opus_llm,
    opus_llm_cached,
)
from shared.llm_policy import FireworksStagePolicy, FireworksToolContract
from shared.llm_spend_budget import (
    FIREWORKS_SPEND_COHORT_CONTEXT_KEY,
    reset_fireworks_spend_budget_for_testing,
)
from shared.schemas import CandidateSnippet

FIREWORKS_ID = "accounts/fireworks/models/glm-5p2"


def _fireworks_response(
    content: str = '{"ok": true}',
    finish_reason: str = "stop",
    reasoning: str | None = "chain of thought",
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, reasoning_content=reasoning
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )


def _patched_openai(response):
    """Patch the OpenAI constructor; returns (context manager, client mock)."""
    client = MagicMock()
    raw = MagicMock()
    raw.parse.return_value = response
    raw.headers = {}
    raw.request_id = None
    client.chat.completions.with_raw_response.create.return_value = raw
    return patch("openai.OpenAI", return_value=client), client


def _create_mock(client):
    return client.chat.completions.with_raw_response.create


def test_opus_llm_routes_fireworks_id_and_never_touches_anthropic():
    openai_patch, client = _patched_openai(_fireworks_response())
    with openai_patch, patch("anthropic.Anthropic") as anthropic_ctor:
        result = opus_llm(
            "system", "user", expect_json=True, model_name=FIREWORKS_ID
        )
    assert result == {"ok": True}
    anthropic_ctor.assert_not_called()
    kwargs = _create_mock(client).call_args.kwargs
    assert kwargs["model"] == FIREWORKS_ID
    # Vendor calibration rides along; no thinking kwargs on the OpenAI wire.
    assert kwargs["temperature"] == 1.0
    assert "thinking" not in kwargs


def test_opus_llm_cached_routes_fireworks_id():
    openai_patch, client = _patched_openai(_fireworks_response())
    with openai_patch, patch("anthropic.Anthropic") as anthropic_ctor:
        result = opus_llm_cached(
            "system", "user", expect_json=False, model_name=FIREWORKS_ID
        )
    assert result == '{"ok": true}'
    anthropic_ctor.assert_not_called()
    assert _create_mock(client).called


def test_opus_llm_anthropic_id_keeps_anthropic_path():
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text='{"ok": true}')],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    client = MagicMock()
    client.messages.create.return_value = message
    with patch("anthropic.Anthropic", return_value=client), patch(
        "openai.OpenAI"
    ) as openai_ctor:
        result = opus_llm(
            "system", "user", expect_json=True, model_name="claude-opus-4-8"
        )
    assert result == {"ok": True}
    openai_ctor.assert_not_called()


def test_fireworks_truncation_raises_after_filling_capture():
    openai_patch, _ = _patched_openai(
        _fireworks_response(content="partial", finish_reason="length")
    )
    capture: dict = {}
    with openai_patch:
        with pytest.raises(RuntimeError, match="finish_reason=length"):
            opus_llm(
                "system",
                "user",
                expect_json=False,
                model_name=FIREWORKS_ID,
                capture=capture,
            )
    # Capture filled before the raise — same contract as the Anthropic
    # primaries' non-end_turn raise.
    assert capture["stop_reason"] == "length"
    assert capture["thinking_summary"] == "chain of thought"


def test_fireworks_floors_max_tokens_for_reasoning_headroom(monkeypatch):
    """Fireworks counts reasoning against max_tokens — Anthropic-calibrated
    caller caps (4096-8192) starved generation into finish_reason=length
    judgment failures on the first GLM-primary session (2026-07-07). Small
    caps are floored to FIREWORKS_PRIMARY_MIN_MAX_TOKENS; larger caller caps
    pass through untouched."""
    monkeypatch.setattr(config, "FIREWORKS_PRIMARY_MIN_MAX_TOKENS", 16384)
    openai_patch, client = _patched_openai(_fireworks_response())
    with openai_patch:
        opus_llm(
            "system", "user", expect_json=True,
            max_tokens=2048, model_name=FIREWORKS_ID,
        )
    assert _create_mock(client).call_args.kwargs["max_tokens"] == 16384

    get_llm_client.cache_clear()
    openai_patch, client = _patched_openai(_fireworks_response())
    with openai_patch:
        opus_llm(
            "system", "user", expect_json=True,
            max_tokens=32768, model_name=FIREWORKS_ID,
        )
    assert _create_mock(client).call_args.kwargs["max_tokens"] == 32768


def test_opus_llm_fireworks_timeout_override_sets_client_and_request(monkeypatch):
    monkeypatch.setattr(config, "FIREWORKS_PRIMARY_MIN_MAX_TOKENS", 16384)
    openai_patch, client = _patched_openai(_fireworks_response())
    with openai_patch as openai_ctor:
        opus_llm(
            "system",
            "user",
            expect_json=True,
            max_tokens=24576,
            model_name=FIREWORKS_ID,
            timeout_seconds=420,
        )

    assert openai_ctor.call_args.kwargs["timeout"] == 420
    assert openai_ctor.call_args.kwargs["max_retries"] == 0
    kwargs = _create_mock(client).call_args.kwargs
    assert kwargs["max_tokens"] == 24576
    assert kwargs["timeout"] == 420


def test_opus_llm_fireworks_default_timeout_uses_config(monkeypatch):
    monkeypatch.setattr(config, "SHADOW_LLM_TIMEOUT_SECONDS", 333)
    openai_patch, client = _patched_openai(_fireworks_response())
    with openai_patch as openai_ctor:
        opus_llm("system", "user", expect_json=True, model_name=FIREWORKS_ID)

    assert openai_ctor.call_args.kwargs["timeout"] == 333
    assert _create_mock(client).call_args.kwargs["timeout"] == 333


def test_fireworks_usage_recorded_with_provider(monkeypatch):
    recorded = {}

    def fake_record(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr("shared.llm_clients.record_llm_usage", fake_record)
    openai_patch, _ = _patched_openai(_fireworks_response())
    with openai_patch:
        opus_llm("system", "user", expect_json=True, model_name=FIREWORKS_ID)
    assert recorded["provider"] == "fireworks"
    assert recorded["model"] == FIREWORKS_ID
    assert recorded["usage"]["input_tokens"] == 100
    assert recorded["request"]["finish_reason"] == "stop"
    assert recorded["request"]["stop_reason"] == recorded["request"]["finish_reason"]


def test_fireworks_keyboard_interrupt_records_conservative_attempt_and_usage(
    monkeypatch: pytest.MonkeyPatch,
):
    client = MagicMock()
    client.chat.completions.with_raw_response.create.side_effect = KeyboardInterrupt
    attempts: list[dict] = []
    aggregates: list[dict] = []
    settlements: list[dict] = []

    monkeypatch.setattr(
        "shared.llm_clients.record_llm_attempt",
        lambda **kwargs: attempts.append(kwargs),
    )
    monkeypatch.setattr(
        "shared.llm_clients.record_llm_usage",
        lambda **kwargs: aggregates.append(kwargs),
    )
    monkeypatch.setattr(
        "shared.llm_clients.settle_fireworks_spend",
        lambda reservation, **kwargs: settlements.append(
            {"reservation": reservation, **kwargs}
        ),
    )

    with patch("openai.OpenAI", return_value=client), pytest.raises(KeyboardInterrupt):
        opus_llm(
            "system",
            "user",
            expect_json=True,
            model_name=FIREWORKS_ID,
            policy=FireworksStagePolicy(
                stage="full",
                max_attempts=1,
                attempt_timeout_seconds=5,
                total_deadline_seconds=5,
            ),
            usage_context={"logical_call_id": "interrupted-call"},
        )

    assert len(attempts) == 1
    assert attempts[0]["logical_call_id"] == "interrupted-call"
    assert attempts[0]["status"] == "terminal_error"
    assert attempts[0]["usage_status"] == "unavailable"
    assert attempts[0]["metadata"]["failure_reason"] == "interrupted"
    assert attempts[0]["metadata"]["error_type"] == "KeyboardInterrupt"
    assert len(aggregates) == 1
    assert aggregates[0]["actual_status"] == "error"
    assert aggregates[0]["usage_status"] == "unavailable"
    assert aggregates[0]["request"]["attempt_count"] == 1
    assert aggregates[0]["request"]["measured_attempt_count"] == 0
    assert aggregates[0]["request"]["error_type"] == "KeyboardInterrupt"
    assert settlements == [
        {
            "reservation": None,
            "measured_cost_usd": None,
            "usage_complete": False,
        }
    ]


def test_fireworks_facial_judge_usage_context_stage(monkeypatch):
    recorded = {}

    def fake_record(**kwargs):
        recorded.update(kwargs)

    brief = MagicMock()
    brief.has_v2_schema = True
    brief._new_brief = MagicMock()
    snippet = CandidateSnippet(
        name="Jane Doe",
        headline="ML Researcher",
        current_title="Research Scientist",
        current_company="OpenLab",
        location="SF",
        education_snippet="PhD, MIT",
        profile_url="https://www.linkedin.com/in/janedoe",
        source_string_id=1,
        source_string_name="string",
        page=1,
        result_rank=1,
    )

    monkeypatch.setattr(config, "FACIAL_MODEL_NAME", FIREWORKS_ID)
    monkeypatch.setattr(config, "SHADOW_FACIAL_MODEL_ENABLED", False)
    monkeypatch.setattr("shared.llm_clients.record_llm_usage", fake_record)
    openai_patch, _ = _patched_openai(
        _fireworks_response(
            content="DECISION: FACIAL_YES\nREASON: strong builder signal",
        )
    )
    with openai_patch, patch.object(judger, "assemble_facial_system", return_value="SYSTEM"):
        decision = judger.facial_judge(snippet, brief)

    assert decision.decision == "FACIAL_YES"
    assert recorded["usage_context"]["stage"] == "facial"


def test_facial_llm_routes_when_facial_model_is_fireworks(monkeypatch):
    monkeypatch.setattr(config, "FACIAL_MODEL_NAME", FIREWORKS_ID)
    openai_patch, client = _patched_openai(_fireworks_response())
    with openai_patch, patch("anthropic.Anthropic") as anthropic_ctor:
        result = facial_llm("system", "user", expect_json=True)
    assert result == {"ok": True}
    anthropic_ctor.assert_not_called()
    assert _create_mock(client).call_args.kwargs["model"] == FIREWORKS_ID


def test_cheap_llm_fireworks_provider_branch(monkeypatch):
    monkeypatch.setattr(config, "CHEAP_MODEL_PROVIDER", "fireworks")
    monkeypatch.setattr(config, "CHEAP_MODEL_NAME", FIREWORKS_ID)
    openai_patch, client = _patched_openai(_fireworks_response())
    with openai_patch:
        result = cheap_llm("system", "user", expect_json=True)
    assert result == {"ok": True}
    assert _create_mock(client).call_args.kwargs["model"] == FIREWORKS_ID


def test_stream_path_refuses_fireworks_opus_model(monkeypatch):
    monkeypatch.setattr(config, "OPUS_MODEL_NAME", FIREWORKS_ID)
    from shared.llm_clients import opus_llm_cached_stream

    with pytest.raises(RuntimeError, match="no Fireworks route"):
        list(opus_llm_cached_stream("system", "user"))


def test_forced_tool_contract_returns_decoded_arguments_and_exact_wire() -> None:
    contract = FireworksToolContract(
        name="submit_facial_batch",
        description="Submit judgments",
        parameters={
            "type": "object",
            "properties": {"results": {"type": "array"}},
            "required": ["results"],
            "additionalProperties": False,
        },
    )
    response = _fireworks_response(content="", finish_reason="tool_calls")
    response.choices[0].message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="submit_facial_batch",
                arguments='{"results":[{"candidate_id":"c1","decision":"FACIAL_YES"}]}',
            )
        )
    ]
    openai_patch, client = _patched_openai(response)
    with openai_patch, patch.object(config, "FACIAL_MODEL_NAME", FIREWORKS_ID):
        result = facial_llm(
            "system",
            "user",
            expect_json=False,
            tool_contract=contract,
        )
    assert result == {
        "results": [{"candidate_id": "c1", "decision": "FACIAL_YES"}]
    }
    kwargs = _create_mock(client).call_args.kwargs
    assert kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_facial_batch"},
    }
    assert kwargs["parallel_tool_calls"] is False
    assert len(kwargs["tools"]) == 1


def test_forced_tool_contract_rejects_multiple_calls() -> None:
    contract = FireworksToolContract(
        name="submit_full_evaluation",
        description="Submit one evaluation",
        parameters={"type": "object"},
    )
    response = _fireworks_response(content="", finish_reason="tool_calls")
    response.choices[0].message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(name=contract.name, arguments="{}")
        ),
        SimpleNamespace(
            function=SimpleNamespace(name=contract.name, arguments="{}")
        ),
    ]
    openai_patch, _client = _patched_openai(response)
    with openai_patch, pytest.raises(RuntimeError, match="exactly one"):
        opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            model_name=FIREWORKS_ID,
            tool_contract=contract,
        )


def test_fireworks_interrupt_after_response_is_not_recorded_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = FireworksToolContract(
        name="submit_full_evaluation",
        description="Submit one evaluation",
        parameters={"type": "object"},
    )
    response = _fireworks_response(content="", finish_reason="tool_calls")
    response.choices[0].message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(name=contract.name, arguments="{}")
        )
    ]
    aggregates: list[dict] = []
    monkeypatch.setattr(
        "shared.llm_clients.record_llm_usage",
        lambda **kwargs: aggregates.append(kwargs),
    )
    openai_patch, _client = _patched_openai(response)

    with openai_patch, patch(
        "shared.llm_clients._decode_forced_tool_call",
        side_effect=KeyboardInterrupt,
    ), pytest.raises(KeyboardInterrupt):
        opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            model_name=FIREWORKS_ID,
            policy=FireworksStagePolicy(
                stage="full",
                max_attempts=1,
                attempt_timeout_seconds=5,
                total_deadline_seconds=5,
            ),
            tool_contract=contract,
            usage_context={"logical_call_id": "post-response-interrupt"},
        )

    assert len(aggregates) == 1
    assert aggregates[0]["actual_status"] == "error"
    assert aggregates[0]["usage_status"] == "measured"


def test_fireworks_policy_is_not_silently_ignored_on_anthropic_model() -> None:
    policy = FireworksStagePolicy(stage="facial")
    with pytest.raises(ValueError, match="requires a Fireworks model"):
        opus_llm(
            "system",
            "user",
            model_name="claude-opus-4-8",
            policy=policy,
        )


def test_live_facial_spend_cohort_dispatches_zero_calls_when_only_two_fit(
    monkeypatch,
) -> None:
    fast_model = "accounts/fireworks/routers/glm-5p2-fast"
    cohort = threading.Barrier(3)
    errors = []
    lock = threading.Lock()
    openai_patch, client = _patched_openai(_fireworks_response())
    monkeypatch.setattr(config, "FIREWORKS_PRIMARY_MIN_MAX_TOKENS", 10_000)
    monkeypatch.setattr(config, "FIREWORKS_PRIMARY_MAX_COST_USD", 0.6)
    monkeypatch.setattr(
        "shared.llm_clients._fireworks_input_token_upper_bound",
        lambda _request: 100_000,
    )
    reset_fireworks_spend_budget_for_testing()

    def worker(index: int) -> None:
        try:
            opus_llm(
                "system",
                f"user-{index}",
                expect_json=True,
                max_tokens=10_000,
                model_name=fast_model,
                policy=FireworksStagePolicy(
                    stage="facial",
                    max_attempts=1,
                    attempt_timeout_seconds=5,
                    total_deadline_seconds=5,
                ),
                usage_context={
                    "logical_call_id": f"cohort-{index}",
                    FIREWORKS_SPEND_COHORT_CONTEXT_KEY: cohort,
                },
            )
        except ApiBudgetExhaustedError as exc:
            with lock:
                errors.append(exc)

    try:
        with openai_patch:
            threads = [threading.Thread(target=worker, args=(index,)) for index in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=5)
    finally:
        reset_fireworks_spend_budget_for_testing()

    assert len(errors) == 3
    assert _create_mock(client).call_count == 0


def test_explicit_policy_redacts_reasoning_capture_but_legacy_retains_it() -> None:
    policy_capture: dict = {}
    openai_patch, _client = _patched_openai(_fireworks_response())
    with openai_patch:
        opus_llm(
            "system",
            "user",
            model_name=FIREWORKS_ID,
            capture=policy_capture,
            policy=FireworksStagePolicy(stage="full", max_attempts=1),
        )
    assert policy_capture["thinking_summary"] is None

    legacy_capture: dict = {}
    openai_patch, _client = _patched_openai(_fireworks_response())
    with openai_patch:
        opus_llm(
            "system",
            "user",
            model_name=FIREWORKS_ID,
            capture=legacy_capture,
        )
    assert legacy_capture["thinking_summary"] == "chain of thought"


def test_tool_contract_requires_explicit_tool_calls_finish_reason() -> None:
    contract = FireworksToolContract(
        name="submit_full_evaluation",
        description="Submit one evaluation",
        parameters={"type": "object"},
    )
    response = _fireworks_response(content="", finish_reason=None)
    response.choices[0].message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(name=contract.name, arguments="{}")
        )
    ]
    openai_patch, _client = _patched_openai(response)
    with openai_patch, pytest.raises(RuntimeError, match="finish_reason=None"):
        opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            model_name=FIREWORKS_ID,
            tool_contract=contract,
        )
