"""Subprocess transport for Claude CLI (``claude -p``).

Routes judgment calls through the local CLI seat so subscription usage
replaces Anthropic API credits. One subprocess attempt per call; no
internal retry loop.

The CLI cannot enforce ``max_tokens``. When callers pass it, the value is
recorded in the telemetry ``request`` dict only.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from shared.failures import ApiBudgetExhaustedError
from shared.llm_usage import record_llm_usage

_ENV_VARS_TO_STRIP = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "LINKEDIN_ANTHROPIC_API_KEY",
        "CLAUDECODE",
    }
)
_ENV_PREFIXES_TO_STRIP = ("CLAUDE_CODE_",)


class ClaudeCliTransportError(RuntimeError):
    """CLI subprocess failed; carries returncode and stderr tail."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


def _build_child_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _ENV_VARS_TO_STRIP:
            continue
        if any(key.startswith(prefix) for prefix in _ENV_PREFIXES_TO_STRIP):
            continue
        env[key] = value
    return env


def _stderr_tail(stderr: str, *, limit: int = 500) -> str:
    text = (stderr or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def _is_usage_limit_failure(text: str) -> bool:
    lower = text.lower()
    if any(
        phrase in lower
        for phrase in (
            "usage limit",
            "rate limit",
            "out of extra usage",
        )
    ):
        return True
    return "exceeded" in lower and "limit" in lower


def _raise_for_failure_text(
    text: str,
    *,
    model: str,
    request: dict[str, Any],
    usage_context: dict | None,
) -> None:
    if _is_usage_limit_failure(text):
        _record_transport_error(
            provider="claude_cli",
            model=model,
            request=request,
            usage_context=usage_context,
            exc=ApiBudgetExhaustedError(text[:240]),
        )
        raise ApiBudgetExhaustedError(text[:240])


def _record_transport_error(
    *,
    provider: str,
    model: str,
    request: dict[str, Any],
    usage_context: dict | None,
    exc: Exception,
) -> None:
    from shared.llm_clients import _record_llm_error

    _record_llm_error(
        provider=provider,
        model=model,
        request=request,
        usage_context=usage_context,
        exc=exc,
    )


def _usage_from_envelope(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cache_creation_input_tokens": int(
            usage.get("cache_creation_input_tokens", 0) or 0
        ),
    }


def _extract_envelope_payload(parsed: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for entry in reversed(parsed):
            if isinstance(entry, dict) and entry.get("type") == "result":
                return entry
    raise ClaudeCliTransportError("CLI stdout missing terminal result entry")


def claude_cli_chat(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    expect_json: bool,
    usage_context: dict | None,
    request: dict | None = None,
    timeout_seconds: float = 600.0,
    binary: str = "claude",
) -> str | dict | list:
    """Invoke ``claude -p`` once and return the assistant result text or JSON."""

    telemetry_request: dict[str, Any] = dict(request or {})
    telemetry_request.setdefault("system_prompt_chars", len(system_prompt))
    telemetry_request.setdefault("user_prompt_chars", len(user_prompt))
    telemetry_request.setdefault("expect_json", bool(expect_json))

    cmd = [
        binary,
        "-p",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--tools",
        "",
        "--disable-slash-commands",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--output-format",
        "json",
    ]

    try:
        completed = subprocess.run(
            cmd,
            input=user_prompt,
            capture_output=True,
            text=True,
            env=_build_child_env(),
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_error = ClaudeCliTransportError(
            f"Claude CLI timed out after {timeout_seconds}s",
            stderr_tail=_stderr_tail(getattr(exc, "stderr", "") or ""),
        )
        _record_transport_error(
            provider="claude_cli",
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
            exc=timeout_error,
        )
        raise timeout_error from exc

    if completed.returncode != 0:
        failure_text = (completed.stderr or "") + (completed.stdout or "")
        _raise_for_failure_text(
            failure_text,
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
        )
        transport_error = ClaudeCliTransportError(
            f"Claude CLI exited {completed.returncode}",
            returncode=completed.returncode,
            stderr_tail=_stderr_tail(completed.stderr),
        )
        _record_transport_error(
            provider="claude_cli",
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
            exc=transport_error,
        )
        raise transport_error

    stdout = (completed.stdout or "").strip()
    if not stdout:
        transport_error = ClaudeCliTransportError(
            "Claude CLI returned empty stdout",
            returncode=completed.returncode,
            stderr_tail=_stderr_tail(completed.stderr),
        )
        _record_transport_error(
            provider="claude_cli",
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
            exc=transport_error,
        )
        raise transport_error

    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        transport_error = ClaudeCliTransportError(
            f"Claude CLI stdout is not valid JSON: {stdout[:240]}",
            returncode=completed.returncode,
            stderr_tail=_stderr_tail(completed.stderr),
        )
        _record_transport_error(
            provider="claude_cli",
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
            exc=transport_error,
        )
        raise transport_error from exc

    try:
        envelope = _extract_envelope_payload(parsed)
    except ClaudeCliTransportError as exc:
        _record_transport_error(
            provider="claude_cli",
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
            exc=exc,
        )
        raise

    if envelope.get("is_error"):
        failure_text = json.dumps(envelope)
        if not failure_text:
            failure_text = str(envelope.get("result") or "")
        _raise_for_failure_text(
            failure_text,
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
        )
        transport_error = ClaudeCliTransportError(
            f"Claude CLI returned is_error payload: {failure_text[:240]}",
            returncode=completed.returncode,
            stderr_tail=_stderr_tail(completed.stderr),
        )
        _record_transport_error(
            provider="claude_cli",
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
            exc=transport_error,
        )
        raise transport_error

    result_text = envelope.get("result")
    if not isinstance(result_text, str) or not result_text.strip():
        transport_error = ClaudeCliTransportError(
            "Claude CLI envelope missing non-empty result text",
            returncode=completed.returncode,
            stderr_tail=_stderr_tail(completed.stderr),
        )
        _record_transport_error(
            provider="claude_cli",
            model=model,
            request=telemetry_request,
            usage_context=usage_context,
            exc=transport_error,
        )
        raise transport_error

    result_text = result_text.strip()
    total_cost_usd = envelope.get("total_cost_usd")
    duration_api_ms = envelope.get("duration_api_ms")
    if usage_context is not None:
        if total_cost_usd is not None:
            usage_context["cli_reported_cost_usd"] = total_cost_usd
        if duration_api_ms is not None:
            usage_context["cli_duration_api_ms"] = duration_api_ms

    record_llm_usage(
        provider="claude_cli",
        model=model,
        usage=_usage_from_envelope(envelope.get("usage")),
        request=telemetry_request,
        usage_context=usage_context,
    )

    from shared.llm_clients import _parse_json_response, _stash_langfuse_context

    _stash_langfuse_context(usage_context)

    if expect_json:
        return _parse_json_response(result_text)
    return result_text
