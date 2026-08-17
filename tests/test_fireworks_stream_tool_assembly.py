from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from shared.llm_clients import (
    _create_fireworks_streaming_completion,
    _decode_forced_tool_call,
    _fireworks_primary_chat,
)
from shared.llm_policy import (
    FireworksDeadlineExceeded,
    FireworksStagePolicy,
    FireworksToolContract,
)


class _RawStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.headers = {"x-request-id": "request-1"}
        self.request_id = None
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def parse(self):
        return iter(self._chunks) if isinstance(self._chunks, list) else self._chunks

    def close(self):
        self.closed.set()


class _StreamingSurface:
    def __init__(self, raw: _RawStream):
        self.raw = raw
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.raw


def _client(raw: _RawStream):
    surface = _StreamingSurface(raw)
    return (
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(with_streaming_response=surface)
            )
        ),
        surface,
    )


def _chunk(delta, *, finish_reason=None, chunk_id="response-1", usage=None):
    return SimpleNamespace(
        id=chunk_id,
        usage=usage,
        choices=[
            SimpleNamespace(delta=delta, finish_reason=finish_reason)
        ],
    )


def _tool_contract() -> FireworksToolContract:
    return FireworksToolContract(
        name="submit_judgment",
        description="Submit one judgment",
        parameters={"type": "object"},
    )


def _consume(chunks):
    raw = _RawStream(chunks)
    client, surface = _client(raw)
    envelope = _create_fireworks_streaming_completion(
        client,
        {"model": "accounts/fireworks/models/glm-5p2"},
        deadline_at=time.monotonic() + 1.0,
    )
    return envelope, raw, surface


def test_split_tool_arguments_assemble_and_decode_across_real_delta_shapes():
    expected = '{"candidate_id":"c1","decision":"FACIAL_YES"}'
    chunks = [
        _chunk(
            SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        model_extra={
                            "index": 0,
                            "id": "call-1",
                            "function": SimpleNamespace(
                                model_extra={
                                    "name": "submit_judgment",
                                    "arguments": '{"candidate_id":',
                                }
                            ),
                        }
                    )
                ],
            )
        ),
        _chunk(
            SimpleNamespace(
                content=None,
                model_extra={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "",
                            "function": {
                                "name": "",
                                "arguments": '"c1","decision":',
                            },
                        }
                    ]
                },
            )
        ),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": '"FACIAL_YES"}'},
                    }
                ]
            },
            finish_reason="tool_calls",
        ),
    ]

    envelope, _raw, surface = _consume(chunks)
    message = envelope.response.choices[0].message

    assert message.tool_calls[0].id == "call-1"
    assert message.tool_calls[0].function.name == "submit_judgment"
    assert message.tool_calls[0].function.arguments == expected
    assert _decode_forced_tool_call(message, _tool_contract()) == {
        "candidate_id": "c1",
        "decision": "FACIAL_YES",
    }
    assert surface.kwargs["stream"] is True
    assert surface.kwargs["stream_options"] == {"include_usage": True}


def test_multi_index_tool_calls_are_ordered_and_interleaved_fragments_stay_separate():
    chunks = [
        _chunk(
            SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        index=1,
                        id="call-1",
                        function=SimpleNamespace(name="second", arguments='{"b":'),
                    ),
                    SimpleNamespace(
                        index=0,
                        id="call-0",
                        function=SimpleNamespace(name="first", arguments='{"a":'),
                    ),
                ]
            )
        ),
        _chunk(
            SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(name=None, arguments="1}"),
                    ),
                    SimpleNamespace(
                        index=1,
                        id=None,
                        function=SimpleNamespace(name=None, arguments="2}"),
                    ),
                ]
            ),
            finish_reason="tool_calls",
        ),
    ]

    envelope, _raw, _surface = _consume(chunks)
    calls = envelope.response.choices[0].message.tool_calls

    assert [call.id for call in calls] == ["call-0", "call-1"]
    assert [call.function.name for call in calls] == ["first", "second"]
    assert [call.function.arguments for call in calls] == ['{"a":1}', '{"b":2}']


def test_non_tool_stream_preserves_content_and_has_no_tool_calls_attribute():
    chunks = [
        _chunk(SimpleNamespace(content="hello ")),
        _chunk(
            SimpleNamespace(content="world", reasoning_content="private"),
            finish_reason="stop",
        ),
    ]

    envelope, _raw, _surface = _consume(chunks)
    message = envelope.response.choices[0].message

    assert message.content == "hello world"
    assert message.reasoning_content is None
    assert not hasattr(message, "tool_calls")


def test_streamed_tool_length_finish_reaches_existing_primary_truncation(monkeypatch):
    chunks = [
        _chunk(
            SimpleNamespace(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call-1",
                        function=SimpleNamespace(
                            name="submit_judgment", arguments="{}"
                        ),
                    )
                ]
            ),
            finish_reason="length",
        )
    ]
    raw = _RawStream(chunks)
    client, _surface = _client(raw)
    monkeypatch.setattr("shared.llm_clients.get_llm_client", lambda *_args: client)
    monkeypatch.setattr("shared.llm_clients.record_llm_attempt", lambda **_kwargs: None)
    monkeypatch.setattr("shared.llm_clients.record_llm_usage", lambda **_kwargs: None)
    monkeypatch.setattr("shared.llm_clients.settle_fireworks_spend", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="response truncated: finish_reason=length"):
        _fireworks_primary_chat(
            model="accounts/fireworks/models/glm-5p2",
            system_prompt="system",
            user_prompt="user",
            expect_json=False,
            max_tokens=16,
            usage_context={},
            capture={},
            label="test",
            prompt_cache=False,
            policy=FireworksStagePolicy(
                stage="facial",
                max_attempts=1,
                attempt_timeout_seconds=1,
                total_deadline_seconds=1,
                response_transport="stream",
            ),
            tool_contract=_tool_contract(),
        )


def test_deadline_close_midstream_keeps_deadline_error_and_stream_metadata():
    first = _chunk(SimpleNamespace(content="partial"))
    raw = _RawStream([])

    def blocking_chunks():
        yield first
        assert raw.closed.wait(timeout=1.0)
        raise RuntimeError("transport closed")

    raw._chunks = blocking_chunks()
    client, _surface = _client(raw)

    with pytest.raises(FireworksDeadlineExceeded) as raised:
        _create_fireworks_streaming_completion(
            client,
            {"model": "accounts/fireworks/models/glm-5p2"},
            deadline_at=time.monotonic() + 0.03,
        )

    assert raw.closed.is_set()
    assert raised.value._cloris_stream_metadata == {
        "response_transport": "stream",
        "stream_event_count": 1,
        "stream_content_delta_count": 1,
        "stream_reasoning_delta_count": 0,
        "stream_first_event_ms": pytest.approx(0, abs=20),
        "stream_first_content_ms": pytest.approx(0, abs=20),
    }
    assert raised.value._cloris_fireworks_headers == {
        "x-request-id": "request-1"
    }
    assert raised.value._cloris_provider_request_id == "request-1"


@pytest.mark.parametrize(
    "fragment,match",
    [
        ({"index": 0, "id": "call-1"}, "expected 'submit_judgment'"),
        (
            {
                "index": 0,
                "id": "call-1",
                "function": {"name": "submit_judgment", "arguments": {"bad": True}},
            },
            "not valid JSON",
        ),
    ],
)
def test_malformed_tool_fragments_surface_only_at_decode_time(fragment, match):
    envelope, _raw, _surface = _consume(
        [
            _chunk(
                {"tool_calls": [fragment]},
                finish_reason="tool_calls",
            )
        ]
    )
    message = envelope.response.choices[0].message

    with pytest.raises(RuntimeError, match=match):
        _decode_forced_tool_call(message, _tool_contract())
