"""The streaming judgment deadline is a hard bound, whatever the server sends.

These tests run the real streaming helper through a real OpenAI client and a
real loopback TCP socket, because the defect lives below every fake: a read
blocked in ``recv`` is not woken when another thread closes its socket, and the
SDK swallows SSE comment lines, so a ping-only stream never hands the loop a
chunk at which to check the deadline. Each consuming call runs in a daemon
thread with a hard join bound, so a regression fails the test instead of
hanging the suite.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any, Callable

import pytest

from shared.llm_clients import (
    _create_fireworks_streaming_completion,
    _fireworks_policy_error,
    _fireworks_primary_chat,
    _fireworks_timeout_metadata,
)
from shared.llm_policy import FireworksDeadlineExceeded, FireworksStagePolicy

# The read timeout the pings keep resetting: far past the deadline, so only
# the deadline can end the call inside the test's bound.
_READ_TIMEOUT_SECONDS = 30.0
_DEADLINE_SECONDS = 2.0
# "Within a few seconds of the deadline": generous for a loaded CI box, and
# still far below the read timeout that bounded the call before the fix.
_DEADLINE_MARGIN_SECONDS = 1.5
_HARD_BOUND_SECONDS = 8.0


def _sse_chunk(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


def _http_chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n"


class _FakeSSEServer:
    """Loopback HTTP/1.1 server: headers, optionally one chunk, then pings or silence."""

    def __init__(self, mode: str) -> None:
        if mode not in {"ping", "silent"}:
            raise ValueError(mode)
        self.mode = mode
        self._stop = threading.Event()
        self._conns: list[socket.socket] = []
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen()
        self._listener.settimeout(0.2)
        self.port = self._listener.getsockname()[1]
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _addr = self._listener.accept()
            except (socket.timeout, OSError):
                continue
            self._conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _read_request(self, conn: socket.socket) -> None:
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            data = conn.recv(65536)
            if not data:
                return
            buffer += data
        head, _sep, body = buffer.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n"):
            name, _colon, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value.strip())
        while len(body) < length:
            data = conn.recv(65536)
            if not data:
                return
            body += data

    def _serve(self, conn: socket.socket) -> None:
        try:
            self._read_request(conn)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"x-request-id: fake-request-1\r\n"
                b"\r\n"
            )
            if self.mode == "ping":
                conn.sendall(
                    _http_chunk(
                        _sse_chunk(
                            {
                                "id": "fake-response-1",
                                "object": "chat.completion.chunk",
                                "created": 0,
                                "model": "fake-model",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": "partial"},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    )
                )
            while not self._stop.is_set():
                if self.mode == "ping":
                    # An SSE comment: resets the client's read-inactivity
                    # timer, never surfaces as a chunk.
                    conn.sendall(_http_chunk(b": ping\n\n"))
                self._stop.wait(0.2)
        except OSError:
            pass

    def close(self) -> None:
        self._stop.set()
        for conn in self._conns:
            try:
                conn.close()
            except OSError:
                pass
        self._listener.close()
        self._accept_thread.join(timeout=1.0)


@pytest.fixture(params=["ping", "silent"])
def fake_server(request):
    server = _FakeSSEServer(request.param)
    try:
        yield server
    finally:
        server.close()


def _real_client(base_url: str):
    from openai import OpenAI

    return OpenAI(
        api_key="local-test-key",
        base_url=base_url,
        timeout=_READ_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _run_bounded(fn: Callable[[], Any]) -> tuple[BaseException | None, float]:
    """Run ``fn`` in a daemon thread; fail (never hang) past the hard bound."""

    outcome: dict[str, Any] = {}

    def _target() -> None:
        started = time.monotonic()
        try:
            outcome["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - the test inspects it
            outcome["exc"] = exc
        outcome["elapsed"] = time.monotonic() - started

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=_HARD_BOUND_SECONDS)
    if worker.is_alive():
        pytest.fail(
            f"streaming call still blocked {_HARD_BOUND_SECONDS:.0f}s after start "
            f"with a {_DEADLINE_SECONDS:.0f}s deadline: the deadline is not a "
            "hard bound"
        )
    if "result" in outcome:
        pytest.fail(f"streaming call returned instead of raising: {outcome['result']!r}")
    return outcome.get("exc"), float(outcome["elapsed"])


def test_stream_deadline_bounds_ping_only_and_silent_streams(fake_server):
    client = _real_client(fake_server.base_url)
    try:
        exc, elapsed = _run_bounded(
            lambda: _create_fireworks_streaming_completion(
                client,
                {
                    "model": "fake-model",
                    "messages": [{"role": "user", "content": "probe"}],
                    "timeout": _READ_TIMEOUT_SECONDS,
                },
                deadline_at=time.monotonic() + _DEADLINE_SECONDS,
            )
        )
    finally:
        fake_server.close()
        client.close()

    assert isinstance(exc, FireworksDeadlineExceeded), repr(exc)
    assert _DEADLINE_SECONDS - 0.1 <= elapsed < _DEADLINE_SECONDS + _DEADLINE_MARGIN_SECONDS
    # The deadline keeps today's classification: terminal, never retried.
    assert _fireworks_policy_error(
        exc, FireworksStagePolicy(stage="full")
    ) == (False, None, "logical_deadline")
    assert _fireworks_timeout_metadata(exc) == {
        "timeout_phase": "logical_deadline",
        "transport_timeout_type": "FireworksDeadlineExceeded",
    }
    # Pings are SSE comments, never chunks: only the real chunk is counted.
    expected_events = 1 if fake_server.mode == "ping" else 0
    assert exc._cloris_stream_metadata["stream_event_count"] == expected_events
    assert exc._cloris_provider_request_id == "fake-request-1"


def test_primary_chat_stream_policy_stops_at_the_deadline_without_retry(
    monkeypatch,
):
    """The production path: the policy runner sees the deadline error once,
    records it as a terminal logical-deadline attempt, and does not retry."""

    server = _FakeSSEServer("ping")
    client = _real_client(server.base_url)
    attempts: list[dict[str, Any]] = []
    monkeypatch.setattr("shared.llm_clients.get_llm_client", lambda *_a, **_k: client)
    monkeypatch.setattr(
        "shared.llm_clients.record_llm_attempt",
        lambda **kwargs: attempts.append(kwargs),
    )
    monkeypatch.setattr("shared.llm_clients.record_llm_usage", lambda **_k: None)
    monkeypatch.setattr(
        "shared.llm_clients.settle_fireworks_spend", lambda *_a, **_k: None
    )
    # The read timeout is clipped to the remaining deadline, so a short
    # deadline would mask the overshoot; a longer one exposes it.
    deadline = 3.0
    policy = FireworksStagePolicy(
        stage="full",
        max_attempts=2,
        attempt_timeout_seconds=_READ_TIMEOUT_SECONDS,
        total_deadline_seconds=deadline,
        response_transport="stream",
    )
    try:
        exc, elapsed = _run_bounded(
            lambda: _fireworks_primary_chat(
                model="accounts/fireworks/models/deepseek-v4p1-flash",
                system_prompt="system",
                user_prompt="user",
                expect_json=False,
                max_tokens=16,
                usage_context={},
                capture={},
                label="Fireworks-cached",
                prompt_cache=True,
                policy=policy,
            )
        )
    finally:
        server.close()
        client.close()

    assert isinstance(exc, FireworksDeadlineExceeded), repr(exc)
    assert deadline - 0.1 <= elapsed < deadline + _DEADLINE_MARGIN_SECONDS
    assert len(attempts) == 1
    metadata = attempts[0]["metadata"]
    assert attempts[0]["status"] == "terminal_error"
    assert metadata["failure_reason"] == "logical_deadline"
    assert metadata["retry_decision"] == "stop"
    assert metadata["timeout_phase"] == "logical_deadline"
    assert metadata["stream_event_count"] == 1
