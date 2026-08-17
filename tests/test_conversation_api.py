"""API tests for recruiter conversation endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import cloris.api as api_mod
import shared.output_paths as output_paths
from cloris.app import create_app
from cloris.api.conversation import api_conversation_stream
from cloris.conversation.agent import ConversationQueryResult


@pytest.fixture(autouse=True)
def _reset_rate_buckets() -> None:
    api_mod._CONVERSATION_QUERY_BUCKETS.clear()
    yield
    api_mod._CONVERSATION_QUERY_BUCKETS.clear()


@pytest.fixture()
def chat_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)

    class StubAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def answer(
            self,
            *,
            brief_id: str,
            message: str,
            debug_citations: bool = False,
            **_: object,
        ) -> ConversationQueryResult:
            del brief_id, message, debug_citations
            return ConversationQueryResult(
                assistant_text="LinkedIn held the fixture line.",
                kind="ok",
            )

    monkeypatch.setattr(
        "cloris.conversation.agent.ConversationAgent", StubAgent
    )
    return TestClient(create_app())


def test_conversation_query_200(chat_client: TestClient) -> None:
    r = chat_client.post(
        "/api/conversation/fixture_brief/query",
        json={"message": "What is running?"},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["kind"] == "ok"
    assert "LinkedIn" in payload["assistant_text"]


def test_conversation_query_rate_limit_429(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)

    class StubAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def answer(self, **kwargs: object) -> ConversationQueryResult:
            return ConversationQueryResult(assistant_text="ok", kind="ok")

    monkeypatch.setattr(
        "cloris.conversation.agent.ConversationAgent", StubAgent
    )
    client = TestClient(create_app())
    brief = "rate_brief"
    for _ in range(10):
        r = client.post(
            f"/api/conversation/{brief}/query",
            json={"message": "pulse"},
        )
        assert r.status_code == 200
    r11 = client.post(
        f"/api/conversation/{brief}/query",
        json={"message": "pulse"},
    )
    assert r11.status_code == 429
    assert r11.json()["detail"] == "Cloris needs a moment."


def test_conversation_query_degraded_200(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)

    class StubAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def answer(self, **kwargs: object) -> ConversationQueryResult:
            return ConversationQueryResult(
                assistant_text="Cloris is processing.",
                kind="degraded",
                degraded_reason="llm_raise",
            )

    monkeypatch.setattr(
        "cloris.conversation.agent.ConversationAgent", StubAgent
    )
    client = TestClient(create_app())
    res = client.post(
        "/api/conversation/br/query",
        json={"message": "Ping"},
    )
    assert res.status_code == 200
    assert res.json()["kind"] == "degraded"


def test_conversation_mute_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)
    client = TestClient(create_app())

    bid = "mute_brief"
    r_patch = client.patch(
        f"/api/conversation/{bid}/mute",
        json={"ambient_muted": True},
    )
    assert r_patch.status_code == 200
    body = r_patch.json()
    assert body["ambient_muted"] is True

    r_get = client.get(f"/api/conversation/{bid}/mute")
    assert r_get.status_code == 200
    assert r_get.json()["ambient_muted"] is True


def test_conversation_stream_emits_initial_ping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(output_paths, "OUTPUT_ROOT", tmp_path)

    class DisconnectingRequest:
        async def is_disconnected(self) -> bool:
            return True

    async def read_first_chunk() -> str:
        response = await api_conversation_stream(
            "stream_brief", DisconnectingRequest()  # type: ignore[arg-type]
        )
        iterator = response.body_iterator
        try:
            return await anext(iterator)
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()

    first_chunk = asyncio.run(read_first_chunk())

    assert "event: ping" in first_chunk
