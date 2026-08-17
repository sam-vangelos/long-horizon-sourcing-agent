from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from shared.claude_cli_transport import ClaudeCliTransportError, claude_cli_chat
from shared.failures import ApiBudgetExhaustedError
from shared.llm_clients import cheap_llm, opus_llm_cached


def _success_envelope_dict(*, result: str = "plain text", is_error: bool = False) -> dict:
    return {
        "type": "result",
        "result": result,
        "is_error": is_error,
        "usage": {
            "input_tokens": 2,
            "output_tokens": 42,
            "cache_read_input_tokens": 8835,
            "cache_creation_input_tokens": 0,
        },
        "total_cost_usd": 0.053,
        "duration_api_ms": 28000,
    }


def _success_envelope_list(*, result: str = "plain text") -> list[dict]:
    return [
        {"type": "system", "subtype": "init"},
        _success_envelope_dict(result=result),
    ]


def _mock_completed(stdout: str, *, returncode: int = 0, stderr: str = "") -> MagicMock:
    completed = MagicMock()
    completed.returncode = returncode
    completed.stdout = stdout
    completed.stderr = stderr
    return completed


def test_cli_prefix_routes_opus_cached_to_transport():
    envelope = _success_envelope_dict(result="OK")
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run, patch(
        "shared.llm_clients.get_llm_client",
        side_effect=AssertionError("Anthropic SDK must not be called"),
    ):
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        result = opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            model_name="claude-cli:claude-opus-4-8",
        )
    assert result == "OK"
    assert mock_run.call_count == 1


def test_cheap_provider_claude_cli_routes_to_transport():
    envelope = _success_envelope_dict(result='{"ok": true}')
    with patch("shared.config.CHEAP_MODEL_PROVIDER", "claude_cli"), patch(
        "shared.config.CHEAP_MODEL_NAME", "claude-cli:claude-haiku-4-5"
    ), patch("shared.claude_cli_transport.subprocess.run") as mock_run, patch(
        "shared.llm_clients.get_llm_client",
        side_effect=AssertionError("Anthropic SDK must not be called"),
    ):
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        result = cheap_llm("system", "user", expect_json=True)
    assert result == {"ok": True}
    assert mock_run.call_count == 1


def test_child_env_scrubbed_of_anthropic_key(monkeypatch):
    scrubbed_keys = {
        "ANTHROPIC_API_KEY": "sk-test",
        "ANTHROPIC_AUTH_TOKEN": "auth-test",
        "ANTHROPIC_BASE_URL": "https://example.invalid",
        "CLAUDECODE": "1",
        "CLAUDE_CODE_ENTRYPOINT": "entry",
        "SAFE_VAR": "keep-me",
    }
    for key, value in scrubbed_keys.items():
        monkeypatch.setenv(key, value)

    envelope = _success_envelope_dict()
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run:
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        claude_cli_chat(
            model="claude-opus-4-8",
            system_prompt="system",
            user_prompt="user",
            expect_json=False,
            usage_context=None,
        )
        child_env = mock_run.call_args.kwargs["env"]

    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
    ):
        assert key not in child_env
    assert child_env.get("SAFE_VAR") == "keep-me"


def test_user_prompt_travels_on_stdin_not_argv():
    user_prompt = "x" * 16000
    envelope = _success_envelope_dict()
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run:
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        claude_cli_chat(
            model="claude-opus-4-8",
            system_prompt="system",
            user_prompt=user_prompt,
            expect_json=False,
            usage_context=None,
        )
        assert mock_run.call_args.kwargs["input"] == user_prompt
        argv = mock_run.call_args.args[0]
        assert user_prompt not in argv


def test_result_list_shape_parses_terminal_result_entry():
    envelope = _success_envelope_list(result="from-list")
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run:
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        result = claude_cli_chat(
            model="claude-opus-4-8",
            system_prompt="system",
            user_prompt="user",
            expect_json=False,
            usage_context=None,
        )
    assert result == "from-list"


def test_result_dict_shape_parses():
    envelope = _success_envelope_dict(result="from-dict")
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run:
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        result = claude_cli_chat(
            model="claude-opus-4-8",
            system_prompt="system",
            user_prompt="user",
            expect_json=False,
            usage_context=None,
        )
    assert result == "from-dict"


def test_expect_json_returns_parsed_dict():
    envelope = _success_envelope_dict(result='{"decision": "SAVE", "confidence": 0.87}')
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run:
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        result = claude_cli_chat(
            model="claude-opus-4-8",
            system_prompt="system",
            user_prompt="user",
            expect_json=True,
            usage_context=None,
        )
    assert result == {"decision": "SAVE", "confidence": 0.87}


def test_nonzero_exit_raises_transport_error():
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run:
        mock_run.return_value = _mock_completed("", returncode=1, stderr="boom")
        with pytest.raises(ClaudeCliTransportError, match="exited 1"):
            claude_cli_chat(
                model="claude-opus-4-8",
                system_prompt="system",
                user_prompt="user",
                expect_json=False,
                usage_context=None,
            )


def test_is_error_payload_raises_transport_error():
    envelope = _success_envelope_dict(result="bad", is_error=True)
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run:
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        with pytest.raises(ClaudeCliTransportError, match="is_error"):
            claude_cli_chat(
                model="claude-opus-4-8",
                system_prompt="system",
                user_prompt="user",
                expect_json=False,
                usage_context=None,
            )


def test_usage_limit_text_raises_api_budget_exhausted():
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run:
        mock_run.return_value = _mock_completed(
            "",
            returncode=1,
            stderr="You have hit your usage limit for this session",
        )
        with pytest.raises(ApiBudgetExhaustedError):
            claude_cli_chat(
                model="claude-opus-4-8",
                system_prompt="system",
                user_prompt="user",
                expect_json=False,
                usage_context=None,
            )


def test_usage_recorded_with_provider_claude_cli():
    envelope = _success_envelope_dict()
    usage_context: dict = {}
    with patch("shared.claude_cli_transport.subprocess.run") as mock_run, patch(
        "shared.claude_cli_transport.record_llm_usage"
    ) as mock_record:
        mock_run.return_value = _mock_completed(json.dumps(envelope))
        claude_cli_chat(
            model="claude-opus-4-8",
            system_prompt="system",
            user_prompt="user",
            expect_json=False,
            usage_context=usage_context,
        )

    mock_record.assert_called_once()
    kwargs = mock_record.call_args.kwargs
    assert kwargs["provider"] == "claude_cli"
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["usage"] == {
        "input_tokens": 2,
        "output_tokens": 42,
        "cache_read_input_tokens": 8835,
        "cache_creation_input_tokens": 0,
    }
    assert usage_context["cli_reported_cost_usd"] == 0.053
