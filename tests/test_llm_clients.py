from __future__ import annotations

import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpcore
import httpx
import pytest

from shared.llm_clients import (
    _create_fireworks_streaming_completion,
    _fireworks_policy_error,
    _fireworks_timeout_metadata,
    _run_fireworks_with_policy,
    cheap_llm,
    get_llm_client,
    opus_llm,
    opus_llm_cached,
)
from shared.llm_policy import (
    FIREWORKS_GLM_5P2_FAST_MODEL,
    FireworksDeadlineExceeded,
    FireworksStagePolicy,
)
from shared.llm_usage import llm_usage_session
from shared.storage import read_jsonl


class _SocketPairStream(httpcore.NetworkStream):
    def __init__(self, sock: socket.socket):
        self.sock = sock

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.sock.settimeout(timeout)
        try:
            return self.sock.recv(max_bytes)
        except TimeoutError as exc:
            raise httpcore.ReadTimeout from exc

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.sock.settimeout(timeout)
        try:
            self.sock.sendall(buffer)
        except TimeoutError as exc:
            raise httpcore.WriteTimeout from exc

    def close(self) -> None:
        self.sock.close()

    def start_tls(self, *_args, **_kwargs):
        raise AssertionError("the inactivity probe uses plain HTTP")

    def get_extra_info(self, info: str):
        return self.sock if info == "socket" else None


class _SocketPairBackend(httpcore.NetworkBackend):
    def __init__(self):
        self.threads: list[threading.Thread] = []

    @staticmethod
    def _read_request(sock: socket.socket) -> dict:
        request = b""
        while b"\r\n\r\n" not in request:
            request += sock.recv(65536)
        header_bytes, body = request.split(b"\r\n\r\n", 1)
        headers = {}
        for line in header_bytes.decode().split("\r\n")[1:]:
            name, value = line.split(":", 1)
            headers[name.lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        while len(body) < length:
            body += sock.recv(65536)
        return json.loads(body[:length])

    @staticmethod
    def _event(payload: dict | str) -> bytes:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        return f"data: {data}\n\n".encode()

    @classmethod
    def _serve(cls, sock: socket.socket) -> None:
        try:
            request = cls._read_request(sock)
            sock.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n"
            )

            def chunk(content: str, *, finish_reason=None):
                return {
                    "id": "chatcmpl-inactivity-probe",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": finish_reason,
                        }
                    ],
                }

            if request["model"] == "steady-stream":
                for index in range(12):
                    sock.sendall(cls._event(chunk(str(index))))
                    if index < 11:
                        time.sleep(0.06)
                sock.sendall(cls._event(chunk("", finish_reason="stop")))
                sock.sendall(cls._event("[DONE]"))
            else:
                sock.sendall(cls._event(chunk("first")))
                time.sleep(0.4)
                sock.sendall(cls._event(chunk("late", finish_reason="stop")))
                sock.sendall(cls._event("[DONE]"))
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            sock.close()

    def connect_tcp(self, *_args, **_kwargs) -> httpcore.NetworkStream:
        client_sock, server_sock = socket.socketpair()
        thread = threading.Thread(
            target=self._serve,
            args=(server_sock,),
            daemon=True,
        )
        self.threads.append(thread)
        thread.start()
        return _SocketPairStream(client_sock)

    def connect_unix_socket(self, *_args, **_kwargs) -> httpcore.NetworkStream:
        raise AssertionError("the inactivity probe connects through socketpair")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def join(self) -> None:
        for thread in self.threads:
            thread.join(timeout=1.0)


def _inactivity_probe_client(timeout: float):
    from openai import OpenAI

    backend = _SocketPairBackend()
    transport = httpx.HTTPTransport()
    transport._pool = httpcore.ConnectionPool(network_backend=backend)
    http_client = httpx.Client(transport=transport, timeout=timeout)
    return backend, OpenAI(
        api_key="local-test-key",
        base_url="http://localhost/v1",
        timeout=timeout,
        max_retries=0,
        http_client=http_client,
    )


def test_fireworks_stream_timeout_is_read_inactivity_not_total_response(
):
    timeout = 0.5
    backend, client = _inactivity_probe_client(timeout)
    started = time.monotonic()
    try:
        envelope = _create_fireworks_streaming_completion(
            client,
            {
                "model": "steady-stream",
                "messages": [{"role": "user", "content": "probe"}],
                "timeout": timeout,
            },
            deadline_at=time.monotonic() + 30.0,
        )
    finally:
        client.close()
        backend.join()

    assert time.monotonic() - started > timeout
    assert envelope.response.choices[0].message.content == "01234567891011"
    assert envelope.response.choices[0].finish_reason == "stop"


def test_fireworks_stream_silence_fails_as_read_inactivity():
    timeout = 0.2
    backend, client = _inactivity_probe_client(timeout)

    try:
        with pytest.raises(Exception) as raised:
            _create_fireworks_streaming_completion(
                client,
                {
                    "model": "silent-stream",
                    "messages": [{"role": "user", "content": "probe"}],
                    "timeout": timeout,
                },
                deadline_at=time.monotonic() + 30.0,
            )
    finally:
        client.close()
        backend.join()

    assert _fireworks_timeout_metadata(raised.value) == {
        "timeout_phase": "read_inactivity",
        "transport_timeout_type": "ReadTimeout",
    }
    assert raised.value._cloris_stream_metadata["stream_event_count"] == 1


class _FakeAnthropicMessage:
    def __init__(self, stop_reason: str) -> None:
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(
            input_tokens=111,
            output_tokens=22,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        self.content = [SimpleNamespace(text='{"status":"partial"}')]


class _FakeAnthropicClient:
    def __init__(self, *, stop_reason: str) -> None:
        self.messages = SimpleNamespace(
            create=lambda **kwargs: _FakeAnthropicMessage(stop_reason)
        )


class _FailingAnthropicClient:
    def __init__(self, exc: Exception) -> None:
        self.messages = SimpleNamespace(
            create=lambda **kwargs: (_ for _ in ()).throw(exc)
        )


def test_opus_llm_logs_usage_even_when_response_truncates():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        fake_module = SimpleNamespace(
            Anthropic=lambda **_kwargs: _FakeAnthropicClient(stop_reason="max_tokens")
        )

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            with llm_usage_session(log_path, pipeline="test_pipeline"):
                with pytest.raises(RuntimeError, match="stop_reason=max_tokens"):
                    opus_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=False,
                        max_tokens=123,
                        usage_context={"stage": "test_stage"},
                    )

        records = read_jsonl(log_path)
        assert records
        assert records[0]["stage"] == "test_stage"
        assert records[0]["input_tokens"] == 111
        assert records[0]["output_tokens"] == 22
        assert records[0]["stop_reason"] == "max_tokens"


def test_opus_llm_logs_error_receipt_when_provider_call_fails():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        usage_context = {"stage": "test_error_stage"}
        fake_module = SimpleNamespace(
            Anthropic=lambda **_kwargs: _FailingAnthropicClient(
                ValueError("bad provider response")
            )
        )

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            with llm_usage_session(log_path, pipeline="test_pipeline"):
                with pytest.raises(ValueError, match="bad provider response"):
                    opus_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=False,
                        max_tokens=123,
                        usage_context=usage_context,
                    )

        records = read_jsonl(log_path)
        assert len(records) == 1
        record = records[0]
        receipt = record["receipt"]
        assert record["stage"] == "test_error_stage"
        assert record["input_tokens"] == 0
        assert record["output_tokens"] == 0
        assert record["error_type"] == "ValueError"
        assert receipt["receipt_type"] == "llm_call"
        assert receipt["actual_status"] == "error"
        assert receipt["actual_detail"]["error"]["type"] == "ValueError"
        assert usage_context["_llm_receipt"]["receipt_id"] == receipt["receipt_id"]


def _fake_openai_response(
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None = None,
    cached_tokens: int | None = None,
) -> SimpleNamespace:
    usage_kwargs = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if cached_tokens is not None:
        # Chat Completions cache-field shape (Fireworks/OpenAI convention):
        # usage.prompt_tokens_details.cached_tokens.
        usage_kwargs["prompt_tokens_details"] = SimpleNamespace(cached_tokens=cached_tokens)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            ),
        ],
        usage=SimpleNamespace(**usage_kwargs),
    )


class _FakeOpenAICompletions:
    def __init__(self, response_factory) -> None:
        self._response_factory = response_factory

    def create(self, **kwargs):
        return self._response_factory()


class _FakeOpenAIChat:
    def __init__(self, response_factory) -> None:
        self.completions = _FakeOpenAICompletions(response_factory)


class _FakeOpenAIClientCtor:
    def __init__(self, response_factory) -> None:
        self._factory = response_factory

    def __call__(self, **kwargs):
        return SimpleNamespace(chat=_FakeOpenAIChat(self._factory))


def test_llm_client_factory_disables_sdk_retries_and_reuses_by_key():
    anthropic_calls: list[dict] = []
    openai_calls: list[dict] = []

    def _anthropic(**kwargs):
        anthropic_calls.append(kwargs)
        return object()

    def _openai(**kwargs):
        openai_calls.append(kwargs)
        return object()

    with patch.dict(
        sys.modules,
        {
            "anthropic": SimpleNamespace(Anthropic=_anthropic),
            "openai": SimpleNamespace(OpenAI=_openai),
        },
    ):
        anthropic_client = get_llm_client("anthropic", "key-a", 30.0)
        assert get_llm_client("anthropic", "key-a", 30.0) is anthropic_client

        openai_client = get_llm_client("openai", "key-a", 30.0)
        assert get_llm_client("openai", "key-a", 30.0) is openai_client
        different_provider = get_llm_client("perplexity", "key-a", 30.0)
        different_key = get_llm_client("openai", "key-b", 30.0)
        different_timeout = get_llm_client("openai", "key-a", 31.0)
        custom_client = get_llm_client(
            "perplexity",
            "key-a",
            30.0,
            max_retries=2,
            base_url="https://custom.example/v1",
        )
        assert (
            get_llm_client(
                "perplexity",
                "key-a",
                30.0,
                max_retries=2,
                base_url="https://custom.example/v1",
            )
            is custom_client
        )

    assert anthropic_calls == [
        {"api_key": "key-a", "timeout": 30.0, "max_retries": 0}
    ]
    assert all(call["max_retries"] == 0 for call in openai_calls[:-1])
    assert openai_calls[-1] == {
        "api_key": "key-a",
        "timeout": 30.0,
        "max_retries": 2,
        "base_url": "https://custom.example/v1",
    }
    assert openai_client is not different_provider
    assert openai_client is not different_key
    assert openai_client is not different_timeout
    assert custom_client is not different_provider


def test_retry_wrapper_default_still_caps_at_five_attempts():
    from shared.llm_clients import _retry_with_backoff

    attempts = 0

    class _RetryableError(RuntimeError):
        status_code = 503

    def _call():
        nonlocal attempts
        attempts += 1
        raise _RetryableError("temporarily unavailable")

    with patch("shared.llm_clients.time.sleep"):
        with pytest.raises(_RetryableError):
            _retry_with_backoff(_call)

    assert attempts == 5


def test_retry_wrapper_caps_retry_after_at_thirty_seconds():
    from shared.llm_clients import _retry_with_backoff

    attempts = 0

    class _RateLimitError(RuntimeError):
        status_code = 429
        response = SimpleNamespace(headers={"Retry-After": "120"})

    def _call():
        nonlocal attempts
        attempts += 1
        raise _RateLimitError("rate limited")

    with patch("shared.llm_clients.time.sleep") as sleep:
        with pytest.raises(_RateLimitError):
            _retry_with_backoff(_call, max_attempts=2)

    assert attempts == 2
    sleep.assert_called_once_with(30.0)


def test_cheap_llm_openai_logs_usage_when_session_active():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        payload = _fake_openai_response('{"k": true}', prompt_tokens=100, completion_tokens=25)
        fake_openai = SimpleNamespace(
            OpenAI=_FakeOpenAIClientCtor(lambda: payload),
        )
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "openai"), patch(
                "shared.config.CHEAP_MODEL_NAME", "gpt-4o-mini"
            ):
                with llm_usage_session(log_path, pipeline="unit_pipeline"):
                    out = cheap_llm("system prompt", "user prompt", expect_json=True)

        records = read_jsonl(log_path)
        assert len(records) == 1
        r = records[0]
        assert r["pipeline"] == "unit_pipeline"
        assert r["provider"] == "openai"
        assert r["model"] == "gpt-4o-mini"
        assert r["input_tokens"] == 100
        assert r["output_tokens"] == 25
        assert r["estimated_cost_usd"] is not None
        assert r["estimated_cost_usd"] > 0
        assert r["receipt"]["receipt_type"] == "llm_call"
        assert r["receipt"]["actual_status"] == "ok"
        assert out == {"k": True}


def test_cheap_llm_openai_returns_receipt_in_usage_context_without_session():
    payload = _fake_openai_response('{"k": true}', prompt_tokens=100, completion_tokens=25)
    fake_openai = SimpleNamespace(
        OpenAI=_FakeOpenAIClientCtor(lambda: payload),
    )
    usage_context = {"stage": "cheap_extraction"}

    with patch.dict(sys.modules, {"openai": fake_openai}):
        with patch("shared.config.CHEAP_MODEL_PROVIDER", "openai"), patch(
            "shared.config.CHEAP_MODEL_NAME", "gpt-4o-mini"
        ):
            out = cheap_llm(
                "system prompt",
                "user prompt",
                expect_json=True,
                usage_context=usage_context,
            )

    receipt = usage_context["_llm_receipt"]
    assert out == {"k": True}
    assert receipt["receipt_type"] == "llm_call"
    assert receipt["stage"] == "llm:cheap_extraction"
    assert receipt["actual_detail"]["provider"] == "openai"
    assert receipt["actual_detail"]["input_tokens"] == 100


def test_cheap_llm_minimax_disables_thinking_and_records_exact_usage():
    captured_client: dict = {}
    captured_request: dict = {}
    payload = _fake_openai_response(
        '{"k": true}',
        prompt_tokens=100,
        completion_tokens=25,
        finish_reason="stop",
        cached_tokens=0,
    )

    def _openai(**kwargs):
        captured_client.update(kwargs)

        def _create(**request):
            captured_request.update(request)
            return payload

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
        )

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": SimpleNamespace(OpenAI=_openai)}), patch(
            "shared.config.CHEAP_MODEL_PROVIDER", "minimax"
        ), patch("shared.config.CHEAP_MODEL_NAME", "MiniMax-M3"), patch(
            "shared.config.MINIMAX_API_KEY", "synthetic"
        ), patch("shared.config.MINIMAX_BASE_URL", "https://api.minimax.io/v1"), patch(
            "shared.config.MINIMAX_CHEAP_MAX_ATTEMPTS", 1
        ), patch("shared.config.CHEAP_MODEL_FALLBACK_PROVIDER", ""):
            with llm_usage_session(log_path, pipeline="minimax_unit"):
                out = cheap_llm("system", "user", expect_json=True)
        record = read_jsonl(log_path)[0]

    assert out == {"k": True}
    assert captured_client["base_url"] == "https://api.minimax.io/v1"
    assert captured_client["max_retries"] == 0
    assert captured_request["model"] == "MiniMax-M3"
    assert captured_request["extra_body"] == {
        "thinking": {"type": "disabled"},
        "service_tier": "standard",
    }
    assert record["provider"] == "minimax"
    assert record["input_tokens"] == 100
    assert record["output_tokens"] == 25
    assert record["estimated_cost_usd"] == pytest.approx(0.00006)


def test_cheap_llm_minimax_parse_failure_falls_back_to_named_haiku():
    payload = _fake_openai_response(
        "not-json",
        prompt_tokens=10,
        completion_tokens=2,
        finish_reason="stop",
        cached_tokens=0,
    )
    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(lambda: payload))
    fake_anthropic = SimpleNamespace(
        Anthropic=lambda **_kwargs: _FakeAnthropicClient(stop_reason="end_turn")
    )
    usage_context = {"stage": "linkedin_profile_extraction"}

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(
            sys.modules,
            {"openai": fake_openai, "anthropic": fake_anthropic},
        ), patch("shared.config.CHEAP_MODEL_PROVIDER", "minimax"), patch(
            "shared.config.CHEAP_MODEL_NAME", "MiniMax-M3"
        ), patch("shared.config.MINIMAX_API_KEY", "minimax-test-key"), patch(
            "shared.config.MINIMAX_CHEAP_MAX_ATTEMPTS", 1
        ), patch(
            "shared.config.CHEAP_MODEL_FALLBACK_PROVIDER", "anthropic"
        ), patch(
            "shared.config.CHEAP_MODEL_FALLBACK_NAME",
            "claude-haiku-4-5-20251001",
        ):
            with llm_usage_session(log_path, pipeline="minimax_fallback"):
                out = cheap_llm(
                    "system", "user", expect_json=True, usage_context=usage_context
                )

        records = read_jsonl(log_path)

    assert out == {"status": "partial"}
    assert usage_context["cheap_fallback_used"] is True
    assert usage_context["_llm_receipt"]["actual_detail"]["provider"] == "anthropic"
    assert [row["provider"] for row in records] == ["minimax", "anthropic"]
    assert records[1]["model"] == "claude-haiku-4-5-20251001"
    assert records[1]["provider_role"] == "fallback"


class _FakeGoogleModels:
    def __init__(self, text: str, usage_metadata) -> None:
        self._text = text
        self._metadata = usage_metadata

    def generate_content(self, **kwargs):
        return SimpleNamespace(text=self._text, usage_metadata=self._metadata)


class _FakeGoogleClient:
    def __init__(self, **kwargs) -> None:
        self.models = _FakeGoogleModels(
            '{"k": 1}',
            SimpleNamespace(prompt_token_count=300, candidates_token_count=40),
        )


def test_cheap_llm_google_logs_usage_when_session_active():
    pytest.importorskip("google.genai")

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"

        with patch("google.genai.Client", side_effect=lambda **kwargs: _FakeGoogleClient()):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "google"):
                with llm_usage_session(log_path, pipeline="google_unit"):
                    out = cheap_llm("system prompt", "user prompt", expect_json=True)

        records = read_jsonl(log_path)
        assert len(records) == 1
        r = records[0]
        assert r["provider"] == "google"
        assert r["model"] == "gemini-2.0-flash"
        assert r["input_tokens"] == 300
        assert r["output_tokens"] == 40
        assert r["estimated_cost_usd"] is not None
        assert r["estimated_cost_usd"] > 0
        assert r["receipt"]["receipt_type"] == "llm_call"
        assert out == {"k": 1}


def test_cheap_llm_anthropic_logs_usage_and_receipt_when_session_active():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        fake_module = SimpleNamespace(
            Anthropic=lambda **_kwargs: _FakeAnthropicClient(
                stop_reason="end_turn"
            )
        )
        usage_context = {"stage": "anthropic_cheap"}

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "anthropic"), patch(
                "shared.config.CHEAP_MODEL_NAME", "claude-opus"
            ):
                with llm_usage_session(log_path, pipeline="anthropic_cheap_unit"):
                    out = cheap_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=True,
                        usage_context=usage_context,
                    )

        records = read_jsonl(log_path)
        assert len(records) == 1
        r = records[0]
        assert out == {"status": "partial"}
        assert r["provider"] == "anthropic"
        assert r["stage"] == "anthropic_cheap"
        assert r["input_tokens"] == 111
        assert r["output_tokens"] == 22
        assert r["receipt"]["receipt_type"] == "llm_call"
        assert usage_context["_llm_receipt"]["receipt_id"] == r["receipt"]["receipt_id"]


def test_cheap_llm_openai_writes_single_row_when_first_attempt_retries():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        attempts = {"n": 0}

        def responses():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("transient failure")
            return _fake_openai_response("{}", prompt_tokens=10, completion_tokens=2)

        fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(responses))
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.llm_clients.time.sleep"):
                with patch("shared.config.CHEAP_MODEL_PROVIDER", "openai"), patch(
                    "shared.config.CHEAP_MODEL_NAME", "gpt-4o-mini"
                ):
                    with llm_usage_session(log_path, pipeline="retry_test"):
                        cheap_llm("s", "u", expect_json=True)

        records = read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["input_tokens"] == 10
        assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# P4.3.3 — cheap-model finish-reason truncation checking
# ---------------------------------------------------------------------------


def test_cheap_llm_anthropic_marks_truncation_on_max_tokens_stop_reason():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        fake_module = SimpleNamespace(
            Anthropic=lambda **_kwargs: _FakeAnthropicClient(stop_reason="max_tokens")
        )
        usage_context = {"stage": "anthropic_cheap_truncated"}

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "anthropic"), patch(
                "shared.config.CHEAP_MODEL_NAME", "claude-opus"
            ):
                with llm_usage_session(log_path, pipeline="anthropic_truncation"):
                    cheap_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=True,
                        usage_context=usage_context,
                    )

        assert usage_context.get("llm_truncated") is True
        records = read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["llm_truncated"] is True


def test_cheap_llm_anthropic_does_not_mark_truncation_on_end_turn():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        fake_module = SimpleNamespace(
            Anthropic=lambda **_kwargs: _FakeAnthropicClient(stop_reason="end_turn")
        )
        usage_context = {"stage": "anthropic_cheap_clean"}

        with patch.dict(sys.modules, {"anthropic": fake_module}):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "anthropic"), patch(
                "shared.config.CHEAP_MODEL_NAME", "claude-opus"
            ):
                with llm_usage_session(log_path, pipeline="anthropic_clean"):
                    cheap_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=True,
                        usage_context=usage_context,
                    )

        # No affirmative "not truncated" flag — absence IS the signal.
        assert "llm_truncated" not in usage_context
        records = read_jsonl(log_path)
        assert "llm_truncated" not in records[0]


def test_cheap_llm_openai_marks_truncation_on_length_finish_reason():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        payload = _fake_openai_response(
            '{"k": true}', prompt_tokens=100, completion_tokens=25, finish_reason="length"
        )
        fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(lambda: payload))
        usage_context = {"stage": "openai_truncated"}

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "openai"), patch(
                "shared.config.CHEAP_MODEL_NAME", "gpt-4o-mini"
            ):
                with llm_usage_session(log_path, pipeline="openai_truncation"):
                    cheap_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=True,
                        usage_context=usage_context,
                    )

        assert usage_context.get("llm_truncated") is True
        records = read_jsonl(log_path)
        assert records[0]["llm_truncated"] is True


def test_cheap_llm_openai_does_not_mark_truncation_on_stop_finish_reason():
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        payload = _fake_openai_response(
            '{"k": true}', prompt_tokens=100, completion_tokens=25, finish_reason="stop"
        )
        fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(lambda: payload))
        usage_context = {"stage": "openai_clean"}

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "openai"), patch(
                "shared.config.CHEAP_MODEL_NAME", "gpt-4o-mini"
            ):
                with llm_usage_session(log_path, pipeline="openai_clean"):
                    cheap_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=True,
                        usage_context=usage_context,
                    )

        assert "llm_truncated" not in usage_context
        records = read_jsonl(log_path)
        assert "llm_truncated" not in records[0]


class _FakeGoogleCandidate:
    def __init__(self, finish_reason: str) -> None:
        self.finish_reason = finish_reason


class _FakeGoogleModelsWithFinishReason:
    def __init__(self, text: str, usage_metadata, finish_reason: str) -> None:
        self._text = text
        self._metadata = usage_metadata
        self._finish_reason = finish_reason

    def generate_content(self, **kwargs):
        return SimpleNamespace(
            text=self._text,
            usage_metadata=self._metadata,
            candidates=[_FakeGoogleCandidate(self._finish_reason)],
        )


class _FakeGoogleClientWithFinishReason:
    def __init__(self, finish_reason: str, **kwargs) -> None:
        self.models = _FakeGoogleModelsWithFinishReason(
            '{"k": 1}',
            SimpleNamespace(prompt_token_count=300, candidates_token_count=40),
            finish_reason,
        )


def test_cheap_llm_google_marks_truncation_on_max_tokens_finish_reason():
    pytest.importorskip("google.genai")

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        usage_context = {"stage": "google_truncated"}

        with patch(
            "google.genai.Client",
            side_effect=lambda **kwargs: _FakeGoogleClientWithFinishReason("MAX_TOKENS"),
        ):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "google"):
                with llm_usage_session(log_path, pipeline="google_truncation"):
                    cheap_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=True,
                        usage_context=usage_context,
                    )

        assert usage_context.get("llm_truncated") is True
        records = read_jsonl(log_path)
        assert records[0]["llm_truncated"] is True


# ---------------------------------------------------------------------------
# Always-thinking models (claude-fable-5): summarized-thinking request is
# scoped to those models only, and opus_llm's `capture` dict carries the
# summary + stop_reason back — filled even when the call ends in the
# non-end_turn raise, so a refusal still yields its reasoning.
# ---------------------------------------------------------------------------


class _RecordingAnthropicClient:
    """Records messages.create kwargs; returns a Fable-shaped response
    (ThinkingBlock first, text block second)."""

    def __init__(self, captured: dict, *, stop_reason: str = "end_turn") -> None:
        def _create(**kwargs):
            captured.update(kwargs)
            message = _FakeAnthropicMessage(stop_reason)
            message.content = [
                SimpleNamespace(type="thinking", thinking="I weighed three angles."),
                SimpleNamespace(type="text", text='{"status":"ok"}'),
            ]
            return message

        self.messages = SimpleNamespace(create=_create)


def test_opus_llm_requests_summarized_thinking_only_for_always_thinking_models():
    for model, expects_thinking in [
        ("claude-fable-5", True),
        ("claude-mythos-5", True),
        (None, False),  # config default (Opus family) — param must be ABSENT
    ]:
        get_llm_client.cache_clear()
        captured: dict = {}
        fake_module = SimpleNamespace(
            Anthropic=lambda **_kwargs: _RecordingAnthropicClient(captured)
        )
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            opus_llm("system", "user", expect_json=False, model_name=model)
        if expects_thinking:
            assert captured["thinking"] == {
                "type": "adaptive",
                "display": "summarized",
            }, model
        else:
            # Sending a thinking param to Opus-family models would CHANGE
            # primary behavior (omitted means thinking off there).
            assert "thinking" not in captured, model


def test_opus_llm_capture_carries_thinking_summary_and_stop_reason():
    captured: dict = {}
    fake_module = SimpleNamespace(
        Anthropic=lambda **_kwargs: _RecordingAnthropicClient(captured)
    )
    capture: dict = {}
    with patch.dict(sys.modules, {"anthropic": fake_module}):
        out = opus_llm(
            "system", "user", expect_json=False,
            model_name="claude-fable-5", capture=capture,
        )
    assert out == '{"status":"ok"}'
    assert capture["thinking_summary"] == "I weighed three angles."
    assert capture["stop_reason"] == "end_turn"


def test_opus_llm_capture_filled_before_non_end_turn_raise():
    """A refusal/truncation raises AFTER capture is filled — the caller
    (shared/strategy_shadow.py's worker) persists the reasoning either way."""
    captured: dict = {}
    fake_module = SimpleNamespace(
        Anthropic=lambda **_kwargs: _RecordingAnthropicClient(
            captured, stop_reason="refusal"
        )
    )
    capture: dict = {}
    with patch.dict(sys.modules, {"anthropic": fake_module}):
        with pytest.raises(RuntimeError, match="stop_reason=refusal"):
            opus_llm(
                "system", "user", expect_json=False,
                model_name="claude-fable-5", capture=capture,
            )
    assert capture["thinking_summary"] == "I weighed three angles."
    assert capture["stop_reason"] == "refusal"


def test_opus_llm_cached_requests_summarized_thinking_only_for_always_thinking_models():
    for model, expects_thinking in [
        ("claude-fable-5", True),
        ("claude-opus-4-8", False),
    ]:
        get_llm_client.cache_clear()
        captured: dict = {}
        fake_module = SimpleNamespace(
            Anthropic=lambda **_kwargs: _RecordingAnthropicClient(captured)
        )
        with patch.dict(sys.modules, {"anthropic": fake_module}):
            opus_llm_cached("system", "user", expect_json=False, model_name=model)

        assert captured["system"] == [
            {
                "type": "text",
                "text": "system",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if expects_thinking:
            assert captured["thinking"] == {
                "type": "adaptive",
                "display": "summarized",
            }, model
        else:
            # Sending a thinking param to Opus-family models would CHANGE
            # primary behavior (omitted means thinking off there).
            assert "thinking" not in captured, model


def test_opus_llm_cached_capture_carries_thinking_summary_and_stop_reason():
    captured: dict = {}
    fake_module = SimpleNamespace(
        Anthropic=lambda **_kwargs: _RecordingAnthropicClient(captured)
    )
    capture: dict = {}
    with patch.dict(sys.modules, {"anthropic": fake_module}):
        out = opus_llm_cached(
            "system", "user", expect_json=False,
            model_name="claude-fable-5", capture=capture,
        )
    assert out == '{"status":"ok"}'
    assert capture["thinking_summary"] == "I weighed three angles."
    assert capture["stop_reason"] == "end_turn"


def test_opus_llm_cached_capture_filled_before_non_end_turn_raise():
    """A refusal/truncation raises AFTER capture is filled — the caller
    can persist the reasoning either way."""
    captured: dict = {}
    fake_module = SimpleNamespace(
        Anthropic=lambda **_kwargs: _RecordingAnthropicClient(
            captured, stop_reason="refusal"
        )
    )
    capture: dict = {}
    with patch.dict(sys.modules, {"anthropic": fake_module}):
        with pytest.raises(RuntimeError, match="stop_reason=refusal"):
            opus_llm_cached(
                "system", "user", expect_json=False,
                model_name="claude-fable-5", capture=capture,
            )
    assert capture["thinking_summary"] == "I weighed three angles."
    assert capture["stop_reason"] == "refusal"


def test_opus_llm_cached_default_disables_anthropic_sdk_retries():
    constructor_kwargs: dict = {}

    def _anthropic(**kwargs):
        constructor_kwargs.update(kwargs)
        return _RecordingAnthropicClient({})

    with patch.dict(sys.modules, {"anthropic": SimpleNamespace(Anthropic=_anthropic)}):
        opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            model_name="claude-opus-4-8",
        )

    assert constructor_kwargs["timeout"] == 300.0
    assert constructor_kwargs["max_retries"] == 0


def test_opus_llm_cached_stream_uses_sdk_retries_for_connection_setup():
    from shared.llm_clients import opus_llm_cached_stream

    final = _FakeAnthropicMessage("end_turn")

    class _Stream:
        text_stream = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def get_final_message(self):
            return final

    client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_kwargs: _Stream())
    )
    with patch("shared.config.ANTHROPIC_API_KEY", "stream-key"), patch(
        "shared.llm_clients.get_llm_client", return_value=client
    ) as factory:
        list(opus_llm_cached_stream("system", "user"))

    factory.assert_called_once_with(
        "anthropic",
        "stream-key",
        300.0,
        max_retries=2,
    )


@pytest.mark.parametrize(
    ("bound_kwargs", "expected_timeout"),
    [
        ({"timeout_seconds": 9.0}, 9.0),
        ({"max_attempts": 1}, 300.0),
    ],
)
def test_opus_llm_cached_any_explicit_bound_disables_anthropic_sdk_retries(
    bound_kwargs,
    expected_timeout,
):
    constructor_kwargs: dict = {}

    def _anthropic(**kwargs):
        constructor_kwargs.update(kwargs)
        return _RecordingAnthropicClient({})

    with patch.dict(sys.modules, {"anthropic": SimpleNamespace(Anthropic=_anthropic)}):
        opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            model_name="claude-opus-4-8",
            **bound_kwargs,
        )

    assert constructor_kwargs["timeout"] == expected_timeout
    assert constructor_kwargs["max_retries"] == 0


def test_opus_llm_cached_explicit_bounds_own_anthropic_attempts():
    constructor_kwargs: dict = {}
    provider_calls = 0

    class _RetryableProviderError(RuntimeError):
        status_code = 503

    def _create(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise _RetryableProviderError("temporarily unavailable")

    def _anthropic(**kwargs):
        constructor_kwargs.update(kwargs)
        return SimpleNamespace(messages=SimpleNamespace(create=_create))

    with patch.dict(
        sys.modules,
        {"anthropic": SimpleNamespace(Anthropic=_anthropic)},
    ), patch("shared.llm_clients.time.sleep"):
        with pytest.raises(_RetryableProviderError, match="temporarily unavailable"):
            opus_llm_cached(
                "system",
                "user",
                expect_json=False,
                model_name="claude-opus-4-8",
                timeout_seconds=12.5,
                max_attempts=2,
            )

    assert constructor_kwargs["timeout"] == 12.5
    assert constructor_kwargs["max_retries"] == 0
    assert provider_calls == 2


def test_opus_llm_cached_explicit_bounds_create_fireworks_stage_policy():
    with patch("shared.llm_clients._fireworks_primary_chat", return_value="ok") as call:
        result = opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            model_name=FIREWORKS_GLM_5P2_FAST_MODEL,
            usage_context={"stage": "market_intel_synthesis"},
            timeout_seconds=11.0,
            max_attempts=2,
        )

    assert result == "ok"
    kwargs = call.call_args.kwargs
    assert kwargs["timeout_seconds"] == 11.0
    assert kwargs["policy"] == FireworksStagePolicy(
        stage="market_intel_synthesis",
        attempt_timeout_seconds=11.0,
        total_deadline_seconds=24.0,
        max_attempts=2,
    )


def test_opus_llm_cached_explicit_bounds_preserve_other_fireworks_policy_fields():
    policy = FireworksStagePolicy(
        stage="custom_stage",
        reasoning_effort="high",
        attempt_timeout_seconds=99.0,
        total_deadline_seconds=500.0,
        max_attempts=4,
        prompt_cache_key="stable-prefix",
    )
    with patch("shared.llm_clients._fireworks_primary_chat", return_value="ok") as call:
        opus_llm_cached(
            "system",
            "user",
            expect_json=False,
            model_name=FIREWORKS_GLM_5P2_FAST_MODEL,
            policy=policy,
            timeout_seconds=7.0,
            max_attempts=1,
        )

    bounded = call.call_args.kwargs["policy"]
    assert bounded is not policy
    assert bounded.stage == "custom_stage"
    assert bounded.reasoning_effort == "high"
    assert bounded.prompt_cache_key == "stable-prefix"
    assert bounded.total_deadline_seconds == 500.0
    assert bounded.attempt_timeout_seconds == 7.0
    assert bounded.max_attempts == 1


def test_opus_llm_skips_empty_thinking_blocks_in_summary():
    """display="omitted" responses carry thinking blocks with EMPTY text —
    the summary must read as absent, not as an empty string."""
    from shared.llm_clients import _thinking_summary

    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="answer"),
        ]
    )
    assert _thinking_summary(message) is None


# ---------------------------------------------------------------------------
# GLM-5.2 shadow-judge client (shadow_facial_llm, Fireworks OpenAI-compatible)
# ---------------------------------------------------------------------------


def test_shadow_facial_llm_records_usage_under_fireworks_provider():
    from shared.llm_clients import shadow_facial_llm

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        payload = _fake_openai_response(
            "DECISION: FACIAL_YES\nREASON: shadow says yes",
            prompt_tokens=50,
            completion_tokens=12,
            cached_tokens=0,
        )
        fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(lambda: payload))

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"), patch(
                "shared.config.SHADOW_FACIAL_MODEL_NAME", "accounts/fireworks/models/glm-5p2"
            ), patch("shared.config.SHADOW_FACIAL_BASE_URL", "https://api.fireworks.ai/inference/v1"):
                with llm_usage_session(log_path, pipeline="shadow_test"):
                    out = shadow_facial_llm("system", "user", usage_context={"stage": "facial_shadow"})

        assert out == "DECISION: FACIAL_YES\nREASON: shadow says yes"
        records = read_jsonl(log_path)
        assert len(records) == 1
        record = records[0]
        assert record["provider"] == "fireworks"
        assert record["model"] == "accounts/fireworks/models/glm-5p2"
        assert record["input_tokens"] == 50
        assert record["output_tokens"] == 12
        # Verified rate row (shared/llm_usage.py): $1.40 in / $4.40 out per MTok.
        assert record["estimated_cost_usd"] is not None
        assert record["estimated_cost_usd"] > 0


def test_shadow_facial_llm_records_vendor_calibrated_temperature():
    from shared.llm_clients import shadow_facial_llm

    captured_kwargs = {}

    class _CapturingCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _fake_openai_response(
                "DECISION: FACIAL_YES\nREASON: temperature",
                prompt_tokens=50,
                completion_tokens=12,
            )

    class _CapturingChat:
        completions = _CapturingCompletions()

    fake_openai = SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(chat=_CapturingChat())
    )
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(log_path, pipeline="shadow_temperature_test"):
                    shadow_facial_llm("system", "user")
        records = read_jsonl(log_path)

    assert captured_kwargs["temperature"] == 1.0
    assert "top_p" not in captured_kwargs
    assert len(records) == 1
    assert records[0]["temperature"] == 1.0


def test_shadow_full_llm_temperature_override_updates_call_and_request(monkeypatch):
    import shared.llm_clients as llm_clients

    monkeypatch.setattr(llm_clients, "SHADOW_LLM_TEMPERATURE", 0.1)
    captured_kwargs = {}

    class _CapturingCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _fake_openai_response(
                "DECISION: SAVE\nSUMMARY: temperature",
                prompt_tokens=500,
                completion_tokens=120,
            )

    class _CapturingChat:
        completions = _CapturingCompletions()

    fake_openai = SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(chat=_CapturingChat())
    )
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(
                    log_path, pipeline="shadow_temperature_override_test"
                ):
                    llm_clients.shadow_full_llm("system", "user")
        records = read_jsonl(log_path)

    assert captured_kwargs["temperature"] == 0.1
    assert "top_p" not in captured_kwargs
    assert len(records) == 1
    assert records[0]["temperature"] == 0.1


def test_shadow_facial_llm_retries_exactly_once_on_retryable_error():
    from shared.llm_clients import shadow_facial_llm

    attempts = {"n": 0}

    def responses():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("transient network blip")
        return _fake_openai_response("DECISION: FACIAL_NO\nREASON: ok", prompt_tokens=5, completion_tokens=3)

    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(responses))
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(log_path, pipeline="shadow_retry_test"):
                    out = shadow_facial_llm("system", "user")

    assert out == "DECISION: FACIAL_NO\nREASON: ok"
    assert attempts["n"] == 2  # exactly one retry, not the 5-attempt primary backoff loop


def test_shadow_facial_llm_does_not_retry_beyond_one_attempt():
    from shared.llm_clients import shadow_facial_llm

    attempts = {"n": 0}

    def responses():
        attempts["n"] += 1
        raise ConnectionError("still failing")

    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(responses))
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(log_path, pipeline="shadow_no_retry_test"):
                    with pytest.raises(ConnectionError):
                        shadow_facial_llm("system", "user")

        # 2 total attempts (1 original + 1 retry) — never the primary path's 5.
        assert attempts["n"] == 2
        records = read_jsonl(log_path)
        assert len(records) == 1
        assert records[0]["provider"] == "fireworks"
        assert records[0]["error_type"] == "ConnectionError"


def test_shadow_facial_llm_does_not_retry_on_terminal_error():
    from shared.llm_clients import shadow_facial_llm

    attempts = {"n": 0}

    def responses():
        attempts["n"] += 1
        raise ValueError("not retryable")

    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(responses))
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(log_path, pipeline="shadow_terminal_test"):
                    with pytest.raises(ValueError):
                        shadow_facial_llm("system", "user")

    assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# GLM-5.2 shadow-judge client (shadow_full_llm) — sibling to
# shadow_facial_llm, same _shadow_llm_call shared implementation.
# ---------------------------------------------------------------------------


def test_shadow_full_llm_records_usage_under_fireworks_provider():
    from shared.llm_clients import shadow_full_llm

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        payload = _fake_openai_response(
            "DECISION: SAVE\nSUMMARY: shadow says save",
            prompt_tokens=500,
            completion_tokens=120,
            cached_tokens=0,
        )
        fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(lambda: payload))

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"), patch(
                "shared.config.SHADOW_FACIAL_MODEL_NAME", "accounts/fireworks/models/glm-5p2"
            ), patch("shared.config.SHADOW_FACIAL_BASE_URL", "https://api.fireworks.ai/inference/v1"):
                with llm_usage_session(log_path, pipeline="shadow_full_test"):
                    out = shadow_full_llm("system", "user", usage_context={"stage": "full_shadow"})

        assert out == "DECISION: SAVE\nSUMMARY: shadow says save"
        records = read_jsonl(log_path)
        assert len(records) == 1
        record = records[0]
        assert record["provider"] == "fireworks"
        assert record["model"] == "accounts/fireworks/models/glm-5p2"
        assert record["input_tokens"] == 500
        assert record["output_tokens"] == 120
        assert record["estimated_cost_usd"] is not None
        assert record["estimated_cost_usd"] > 0


def test_shadow_full_llm_defaults_max_tokens_to_8192():
    """Mirrors opus_llm_cached's default (full_judge's primary call doesn't
    override max_tokens), unlike shadow_facial_llm's 2048 default."""
    from shared.llm_clients import shadow_full_llm

    captured_kwargs = {}

    class _CapturingCompletions:
        def create(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _fake_openai_response("DECISION: REJECT\nSUMMARY: ok", prompt_tokens=5, completion_tokens=3)

    class _CapturingChat:
        completions = _CapturingCompletions()

    fake_openai = SimpleNamespace(OpenAI=lambda **kwargs: SimpleNamespace(chat=_CapturingChat()))
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(log_path, pipeline="shadow_full_max_tokens_test"):
                    shadow_full_llm("system", "user")

    assert captured_kwargs["max_tokens"] == 8192


def test_shadow_full_llm_retries_exactly_once_on_retryable_error():
    from shared.llm_clients import shadow_full_llm

    attempts = {"n": 0}

    def responses():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ConnectionError("transient network blip")
        return _fake_openai_response("DECISION: SAVE\nSUMMARY: ok", prompt_tokens=5, completion_tokens=3)

    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(responses))
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(log_path, pipeline="shadow_full_retry_test"):
                    out = shadow_full_llm("system", "user")

    assert out == "DECISION: SAVE\nSUMMARY: ok"
    assert attempts["n"] == 2  # exactly one retry, not the 5-attempt primary backoff loop


def test_shadow_full_llm_does_not_retry_on_terminal_error():
    from shared.llm_clients import shadow_full_llm

    attempts = {"n": 0}

    def responses():
        attempts["n"] += 1
        raise ValueError("not retryable")

    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(responses))
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(log_path, pipeline="shadow_full_terminal_test"):
                    with pytest.raises(ValueError):
                        shadow_full_llm("system", "user")

    assert attempts["n"] == 1


# ---------------------------------------------------------------------------
# Fireworks automatic-prefix-cache accounting (Task B): cached tokens must
# be (a) actually read from usage.prompt_tokens_details.cached_tokens and
# (b) priced at the cached rate, with the billable input count reduced by
# the cached amount (inclusive convention) so it isn't double-priced.
# ---------------------------------------------------------------------------


def test_shadow_facial_llm_prices_cached_tokens_at_cache_rate_not_double_counted():
    from shared.llm_clients import shadow_facial_llm

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        # 1000 total prompt tokens, 800 of them cache hits (inclusive —
        # cached_tokens is a SUBSET of prompt_tokens, not additional).
        payload = _fake_openai_response(
            "DECISION: FACIAL_YES\nREASON: cached",
            prompt_tokens=1000,
            completion_tokens=50,
            cached_tokens=800,
        )
        fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(lambda: payload))

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"), patch(
                "shared.config.SHADOW_FACIAL_MODEL_NAME", "accounts/fireworks/models/glm-5p2"
            ):
                with llm_usage_session(log_path, pipeline="shadow_cache_test"):
                    shadow_facial_llm("system", "user")

        records = read_jsonl(log_path)
        assert len(records) == 1
        record = records[0]
        # Billable input excludes the cached slice (1000 - 800 = 200), not
        # the raw 1000 — otherwise the cached tokens get priced twice: once
        # at the full input rate (via input_tokens) and again at the cache
        # rate (via cache_read_input_tokens).
        assert record["input_tokens"] == 200
        assert record["cache_read_input_tokens"] == 800
        # GLM-5.2 rates (shared/llm_usage.py): input $1.40/MTok, output
        # $4.40/MTok, cache_read $0.14/MTok.
        expected_cost = (
            (200 / 1_000_000) * 1.40
            + (50 / 1_000_000) * 4.40
            + (800 / 1_000_000) * 0.14
        )
        assert record["estimated_cost_usd"] == round(expected_cost, 6)


def test_shadow_facial_llm_without_cache_details_records_output_only_lower_bound():
    """No cache split means partial usage, never a fabricated cache miss."""
    from shared.llm_clients import shadow_facial_llm

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        payload = _fake_openai_response(
            "DECISION: FACIAL_NO\nREASON: no cache", prompt_tokens=1000, completion_tokens=50
        )
        fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClientCtor(lambda: payload))

        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"), patch(
                "shared.config.SHADOW_FACIAL_MODEL_NAME", "accounts/fireworks/models/glm-5p2"
            ):
                with llm_usage_session(log_path, pipeline="shadow_no_cache_test"):
                    shadow_facial_llm("system", "user")

        records = read_jsonl(log_path)
        record = records[0]
        assert record["input_tokens"] is None
        assert record["cache_read_input_tokens"] is None
        assert record["usage_status"] == "partial"
        assert record["cost_completeness"] == "lower_bound"
        expected_cost = (50 / 1_000_000) * 4.40
        assert record["estimated_cost_usd"] == round(expected_cost, 6)


def test_cheap_llm_google_does_not_mark_truncation_on_stop_finish_reason():
    pytest.importorskip("google.genai")

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "token-cost-log.jsonl"
        usage_context = {"stage": "google_clean"}

        with patch(
            "google.genai.Client",
            side_effect=lambda **kwargs: _FakeGoogleClientWithFinishReason("STOP"),
        ):
            with patch("shared.config.CHEAP_MODEL_PROVIDER", "google"):
                with llm_usage_session(log_path, pipeline="google_clean"):
                    cheap_llm(
                        "system prompt",
                        "user prompt",
                        expect_json=True,
                        usage_context=usage_context,
                    )

        assert "llm_truncated" not in usage_context
        records = read_jsonl(log_path)
        assert "llm_truncated" not in records[0]


def test_shadow_client_disables_sdk_retries_and_uses_call_base_url():
    """The shadow call is synchronous in the judge path and implements its
    own one-retry doctrine; the OpenAI SDK's default max_retries=2 (with
    backoff) must NOT stack under it (2026-07-05 SPL run: stacking turned a
    60s timeout into 238s+ of run-blocking on long profiles)."""
    from shared.llm_clients import _shadow_llm_call

    captured: dict = {}

    class _RecordingCtor:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                chat=_FakeOpenAIChat(
                    lambda: _fake_openai_response("DECISION: REJECT\nREASON: ok", 5, 3)
                )
            )

    fake_openai = SimpleNamespace(OpenAI=_RecordingCtor())
    with tempfile.TemporaryDirectory() as td:
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch("shared.config.FIREWORKS_API_KEY", "fw-test-key"):
                with llm_usage_session(Path(td) / "t.jsonl", pipeline="shadow_retry_ceiling"):
                    _shadow_llm_call(
                        model="accounts/fireworks/models/glm-5p2",
                        base_url="https://shadow.example/v1",
                        system_prompt="system",
                        user_prompt="profile text",
                        max_tokens=8192,
                        usage_context=None,
                    )

    assert captured.get("max_retries") == 0, (
        "shadow Fireworks client must pass max_retries=0 so the SDK's default "
        "2-retries-with-backoff cannot stack under the manual one-retry"
    )
    assert captured["base_url"] == "https://shadow.example/v1"


def _policy_for_classifier_tests() -> FireworksStagePolicy:
    return FireworksStagePolicy(
        stage="test_stage",
        attempt_timeout_seconds=5.0,
        total_deadline_seconds=60.0,
        max_attempts=2,
    )


def test_policy_retries_raw_httpx_read_timeout():
    retryable, status_code, reason = _fireworks_policy_error(
        httpx.ReadTimeout("The read operation timed out"),
        _policy_for_classifier_tests(),
    )
    assert retryable is True
    assert status_code is None
    assert reason == "timeout"


def test_policy_retries_chained_transport_error():
    outer = RuntimeError("stream consumption failed")
    outer.__cause__ = httpcore.ReadTimeout("The read operation timed out")
    retryable, _status, reason = _fireworks_policy_error(
        outer, _policy_for_classifier_tests()
    )
    assert retryable is True
    assert reason == "timeout"

    conn_outer = RuntimeError("stream consumption failed")
    conn_outer.__cause__ = httpx.RemoteProtocolError("peer closed connection")
    retryable, _status, reason = _fireworks_policy_error(
        conn_outer, _policy_for_classifier_tests()
    )
    assert retryable is True
    assert reason == "connection_error"


def test_policy_deadline_exceeded_stays_terminal():
    retryable, _status, reason = _fireworks_policy_error(
        FireworksDeadlineExceeded("Fireworks total deadline expired"),
        _policy_for_classifier_tests(),
    )
    assert retryable is False
    assert reason == "logical_deadline"


def test_policy_existing_classifications_unchanged():
    policy = _policy_for_classifier_tests()

    class APITimeoutError(Exception):
        """Name-matched stand-in for the OpenAI SDK class."""

    retryable, _status, reason = _fireworks_policy_error(APITimeoutError("t"), policy)
    assert retryable is True
    assert reason == "timeout"

    status_exc = RuntimeError("throttled")
    status_exc.status_code = 429
    retryable, status, reason = _fireworks_policy_error(status_exc, policy)
    assert retryable is True
    assert status == 429
    assert reason == "http_429"

    retryable, _status, reason = _fireworks_policy_error(ValueError("boom"), policy)
    assert retryable is False
    assert reason == "terminal"


def test_policy_runner_retries_stream_read_timeout_then_succeeds():
    policy = _policy_for_classifier_tests()
    sentinel = object()
    attempts = {"count": 0}

    def call(_attempt_timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ReadTimeout("The read operation timed out")
        return sentinel

    observed = []

    def on_attempt(**kwargs):
        observed.append(kwargs)

    with patch("shared.llm_clients.time.sleep"):
        result = _run_fireworks_with_policy(
            call,
            model="accounts/fireworks/models/test-classifier",
            label="test",
            policy=policy,
            on_attempt=on_attempt,
        )

    assert result is sentinel
    assert attempts["count"] == 2
    assert observed[0]["will_retry"] is True
    assert observed[0]["classification"]["reason"] == "timeout"
    assert observed[1]["exc"] is None


def test_policy_deadline_stays_terminal_even_when_chaining_a_timeout():
    """The live wrapping: the streaming helper rebinds the raised exception
    to FireworksDeadlineExceeded with the transport timeout in __context__ —
    the deadline must still classify terminal ahead of the transport walk."""
    exc = FireworksDeadlineExceeded(
        "Fireworks total deadline expired while consuming stream"
    )
    exc.__context__ = httpx.ReadTimeout("The read operation timed out")
    retryable, _status, reason = _fireworks_policy_error(
        exc, _policy_for_classifier_tests()
    )
    assert retryable is False
    assert reason == "logical_deadline"


def test_policy_transport_subclasses_classify_via_mro():
    """Exact-name matching defeated the base-class entries: an httpx
    CloseError IS a NetworkError and an SSLEOFError IS an SSLError, and both
    are the same mid-stream teardown class the walk exists for."""
    import ssl

    retryable, _status, reason = _fireworks_policy_error(
        httpx.CloseError("peer closed connection"),
        _policy_for_classifier_tests(),
    )
    assert retryable is True
    assert reason == "connection_error"

    retryable, _status, reason = _fireworks_policy_error(
        ssl.SSLEOFError("EOF occurred in violation of protocol"),
        _policy_for_classifier_tests(),
    )
    assert retryable is True
    assert reason == "connection_error"


def test_policy_terminal_status_ignores_transport_context():
    """A terminal 4xx that happens to carry a transport fault in its context
    chain stays terminal — the walk only runs for status-less failures."""
    status_exc = RuntimeError("bad request")
    status_exc.status_code = 400
    status_exc.__context__ = httpx.ReadTimeout("The read operation timed out")
    retryable, status, reason = _fireworks_policy_error(
        status_exc, _policy_for_classifier_tests()
    )
    assert retryable is False
    assert status == 400
    assert reason == "http_400"
