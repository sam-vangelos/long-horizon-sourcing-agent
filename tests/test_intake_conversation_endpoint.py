"""Tests for the conversational intake SSE endpoint (Phase C5).

Coverage matches the C5 phase block:

- Happy path: send recruiter message → SSE chunks land in order → state
  persisted (recruiter + cloris messages, conversation_meta bumped,
  v2_draft updated from extraction).
- LLM error mid-stream → ``error`` event present in stream, recruiter
  message persisted at tail so the NEXT turn detects dropped-turn.
- Dropped-turn detection on session load → orchestrator's prompt
  builder is invoked with ``dropped_turn=True`` (asserted via
  prompt-builder spy).
- Concurrent request → second returns 409 ``turn_in_flight``.
- Lock eviction: ``_CONVERSATION_LOCKS`` is empty after every test
  path, including 409 + error paths.
- Auth: SSE-exempt block in :mod:`cloris.api.auth` accepts ``?token=``
  matching ``SESSION_TOKEN``; rejects missing/wrong token with 401.

The orchestrator's ``opus_llm_cached_stream`` and the extractor's
``cheap_llm`` are monkeypatched at their import-bound locations so
tests don't require live API keys.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloris import intake_sessions as intake_module
from cloris.app import create_app
from shared.runtime_state.store import RuntimeStateStore


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")


@pytest.fixture
def api_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    """TestClient backed by a tmp_path intake DB. Mirrors test_intake_sessions."""

    tmp_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    tmp_store = RuntimeStateStore(tmp_db_path)

    monkeypatch.setattr(
        "cloris.api.intake._intake_store", lambda: tmp_store
    )
    monkeypatch.setattr(
        "cloris.api.intake._intake_db_path", lambda: tmp_db_path
    )
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clear_locks_between_tests():
    """Reset the module-level lock dict between tests for isolation."""

    from cloris.api.intake import _CONVERSATION_LOCKS

    _CONVERSATION_LOCKS.clear()
    yield
    _CONVERSATION_LOCKS.clear()


def _stub_orchestrator_stream(deltas: list[str], usage: dict | None = None):
    """Build a fake opus_llm_cached_stream-shaped iterator factory."""

    def _factory(system_prompt, user_prompt, *, usage_context=None, max_tokens=4096):
        for d in deltas:
            yield ("delta", d)
        yield ("usage", usage or {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        })

    return _factory


def _stub_orchestrator_raises_immediately():
    def _factory(*args, **kwargs):
        raise RuntimeError("simulated Opus outage")
        yield  # noqa — make it a generator

    return _factory


def _stub_cheap_llm(updates: dict):
    def _impl(system, user, expect_json=True, usage_context=None):
        return updates

    return _impl


def _create_conversation_session(api_client: TestClient) -> int:
    """Create a session + start the conversation (writes opener)."""

    create_resp = api_client.post(
        "/api/intake/sessions",
        json={"role_title": "Senior Tax Associate"},
    )
    assert create_resp.status_code == 201
    session_id = create_resp.json()["session"]["id"]

    start_resp = api_client.post(
        f"/api/intake/sessions/{session_id}/start_conversation"
    )
    assert start_resp.status_code == 200
    return session_id


# -------------------------------------------------------------------------
# start_conversation
# -------------------------------------------------------------------------


def test_start_conversation_writes_opener_no_packet(api_client: TestClient) -> None:
    create_resp = api_client.post(
        "/api/intake/sessions", json={"role_title": "Tax Associate"}
    )
    session_id = create_resp.json()["session"]["id"]

    start_resp = api_client.post(
        f"/api/intake/sessions/{session_id}/start_conversation"
    )
    assert start_resp.status_code == 200
    body = start_resp.json()
    session = body["session"]
    assert session["current_step"] == "conversation"
    messages = session["state_json"]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "cloris"
    assert "Tell me about the role" in messages[0]["content"]


def test_start_conversation_uses_with_packet_opener(
    api_client: TestClient,
) -> None:
    create_resp = api_client.post(
        "/api/intake/sessions", json={"role_title": None}
    )
    session_id = create_resp.json()["session"]["id"]

    api_client.patch(
        f"/api/intake/sessions/{session_id}",
        json={
            "state_json": {
                "source_packet": {
                    "job_description_text": "JD body about tax associate"
                }
            }
        },
    )

    start_resp = api_client.post(
        f"/api/intake/sessions/{session_id}/start_conversation"
    )
    assert start_resp.status_code == 200
    messages = start_resp.json()["session"]["state_json"]["messages"]
    assert "Got the JD" in messages[0]["content"]


def test_start_conversation_idempotent_on_already_initialized(
    api_client: TestClient,
) -> None:
    session_id = _create_conversation_session(api_client)
    first = api_client.get(f"/api/intake/sessions/{session_id}").json()

    second_start = api_client.post(
        f"/api/intake/sessions/{session_id}/start_conversation"
    )
    assert second_start.status_code == 200
    second = api_client.get(f"/api/intake/sessions/{session_id}").json()

    assert (
        len(second["session"]["state_json"]["messages"])
        == len(first["session"]["state_json"]["messages"])
        == 1
    )


def test_start_conversation_404_on_missing_session(api_client: TestClient) -> None:
    resp = api_client.post("/api/intake/sessions/9999/start_conversation")
    assert resp.status_code == 404


# -------------------------------------------------------------------------
# compose_from_conversation
# -------------------------------------------------------------------------


def test_compose_from_conversation_recovers_session_19_style_transcript(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access", lambda: False
    )
    create_resp = api_client.post("/api/intake/sessions", json={})
    session_id = create_resp.json()["session"]["id"]
    api_client.patch(
        f"/api/intake/sessions/{session_id}",
        json={
            "state_json": {
                "messages": [
                    {"role": "cloris", "content": "Hi — I'm Cloris.", "ts": "t0"},
                    {
                        "role": "recruiter",
                        "content": (
                            "We need a Head of Applied AI Lab for BFS, US remote. "
                            "They need to lead applied AI work, evaluate research, "
                            "ship prototypes, and screen out AI-adjacent managers "
                            "who have not built real systems. Minimum bar is direct "
                            "ownership of applied AI systems."
                        ),
                        "ts": "t1",
                    },
                    {
                        "role": "cloris",
                        "content": "I'd start on LinkedIn and corroborate depth.",
                        "ts": "t2",
                    },
                ],
                "v2_draft": {},
            }
        },
    )

    response = api_client.post(
        f"/api/intake/sessions/{session_id}/compose_from_conversation"
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["slice"] == "v0-intake-compose-job-1"
    assert body["job"]["status"] == "ready"
    assert body["job"]["result"]["compose_status"] == "composed"
    session = body["session"]
    assert session["current_step"] == "review"
    draft = session["state_json"]["v2_draft"]
    assert draft["role_title"] == "Head of Applied AI Lab"
    assert draft["capability_areas"]
    assert draft["minimum_bar_description"]
    assert draft["non_fit_patterns"]
    assert draft["geography"] == "Remote"
    assert "linkedin" in draft["target_modules"]
    assert "github" in draft["target_modules"]
    assert "researcher" in draft["target_modules"]
    assert draft["engagement_context"] == {
        "selectivity_posture": "selective"
    }
    assert session["state_json"]["conversation_compose_meta"]["status"] == "composed"


def test_compose_from_conversation_returns_deficits_without_overwrite(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access", lambda: False
    )
    create_resp = api_client.post(
        "/api/intake/sessions", json={"role_title": "Manual Title"}
    )
    session_id = create_resp.json()["session"]["id"]
    useful_partial = {"role_title": "Manual Title"}
    api_client.patch(
        f"/api/intake/sessions/{session_id}",
        json={"state_json": {"v2_draft": useful_partial, "messages": []}},
    )

    response = api_client.post(
        f"/api/intake/sessions/{session_id}/compose_from_conversation"
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job"]["status"] == "ready"
    assert body["job"]["result"]["compose_status"] == "deficits"
    session = body["session"]
    assert session["current_step"] == "welcome"
    assert session["state_json"]["v2_draft"] == useful_partial
    assert body["job"]["result"]["deficits"]


_REALISTIC_JD = (
    "Head of Applied AI Lab, Banking & Financial Services\n\n"
    "Owns applied AI strategy, lab buildout, executive stakeholder "
    "alignment, regulated financial-services AI delivery, and production "
    "GenAI evaluation.\n\n"
    "Needs someone who has actually built and led applied AI teams in "
    "banking or financial services, not just advised on AI strategy."
)


def test_compose_from_conversation_consumes_uploaded_jd_after_opener_only(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty transcript + uploaded JD must produce a materially populated draft.

    Guards against the silent-empty-draft regression: a recruiter who uploads
    a JD and immediately clicks "Show me the brief" without further chat must
    still get useful intake structure derived from the upload, not a blank
    review surface.
    """

    monkeypatch.setattr(
        "shared.source_packet_synthesis._has_llm_access", lambda: False
    )
    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access", lambda: False
    )

    session_id = _create_conversation_session(api_client)

    upload = api_client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
        files={
            "files": (
                "head-of-applied-ai.txt",
                _REALISTIC_JD.encode("utf-8"),
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200, upload.text

    # Upload now schedules synthesis in a background worker; wait for
    # it to finish before composing so ``v2_draft`` reflects the JD.
    from cloris.api.intake_synthesis import wait_for_synthesis

    assert wait_for_synthesis(session_id, timeout=5.0)

    response = api_client.post(
        f"/api/intake/sessions/{session_id}/compose_from_conversation"
    )
    assert response.status_code == 200, response.text
    body = response.json()
    draft = body["session"]["state_json"]["v2_draft"]
    # Even if compose returns deficits (the transcript is opener-only), the
    # uploaded JD must materially shape the draft.
    assert draft["role_title"], body
    assert "Applied AI" in draft["role_title"]
    assert isinstance(draft["role_summary"], str) and draft["role_summary"].strip()
    assert draft["capability_areas"], body
    assert draft["target_modules"], body


def test_compose_from_conversation_respects_manual_locks(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access", lambda: False
    )
    create_resp = api_client.post(
        "/api/intake/sessions", json={"role_title": "Locked Title"}
    )
    session_id = create_resp.json()["session"]["id"]
    api_client.patch(
        f"/api/intake/sessions/{session_id}",
        json={
            "state_json": {
                "messages": [
                    {
                        "role": "recruiter",
                        "content": (
                            "Actually the public title might be Principal AI Engineer. "
                            "They own applied AI systems and need direct builder proof."
                        ),
                        "ts": "t1",
                    }
                ],
                "v2_draft": {"role_title": "Locked Title"},
                "conversation_meta": {"manually_edited_keys": ["role_title"]},
            }
        },
    )

    response = api_client.post(
        f"/api/intake/sessions/{session_id}/compose_from_conversation"
    )

    assert response.status_code == 200, response.text
    draft = response.json()["session"]["state_json"]["v2_draft"]
    assert draft["role_title"] == "Locked Title"
    assert draft["capability_areas"]


# -------------------------------------------------------------------------
# converse_stream — happy path
# -------------------------------------------------------------------------


def test_converse_stream_happy_path(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_stream(
            ["Got it — ", "Senior tax associate. ", "What's the team size?"]
        ),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({"role_summary": "Senior tax associate at Northwind."}),
    )

    session_id = _create_conversation_session(api_client)

    chunks: list[str] = []
    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "It's a senior tax associate role."},
    ) as response:
        assert response.status_code == 200
        for chunk in response.iter_text():
            chunks.append(chunk)

    raw = "".join(chunks)
    assert "event: message_chunk" in raw
    assert "Got it" in raw
    assert "team size" in raw
    assert "event: slot_update" in raw
    assert "event: done" in raw

    # State persisted: opener + recruiter + cloris message present.
    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    messages = refreshed["session"]["state_json"]["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "cloris"  # opener
    assert messages[1]["role"] == "recruiter"
    assert messages[2]["role"] == "cloris"
    assert "Got it" in messages[2]["content"]
    assert messages[2]["meta"]["model"]

    # v2_draft populated from extraction.
    assert (
        refreshed["session"]["state_json"]["v2_draft"]["role_summary"]
        == "Senior tax associate at Northwind."
    )

    # conversation_meta bumped.
    meta = refreshed["session"]["state_json"]["conversation_meta"]
    assert meta["turn_count"] == 1
    assert meta["cost_usd_running_total"] >= 0.0


def test_converse_stream_emits_extraction_partial_when_keys_dropped(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P9.4 SSE hookup: a structurally-invalid extractor key must surface as
    a visible ``extraction_partial`` event naming the dropped key, and the
    persisted ``conversation_meta.last_extraction`` must record it — never a
    silent drop. Mirrors the exact fixture from
    ``test_extract_slots_drops_only_the_invalid_key_on_structural_validation_failure``
    (``tests/test_intake_conversation_extractor.py``): a malformed
    ``capability_areas`` item alongside a valid ``depth_distinction`` update.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_stream(["Got it."]),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm(
            {
                "capability_areas": [
                    {"foo": "bar"},  # missing required name + description
                ],
                "depth_distinction": {
                    "builder_definition": "x",
                    "user_definition": "y",
                    "edge_case_guidance": "z",
                },
            }
        ),
    )

    session_id = _create_conversation_session(api_client)

    chunks: list[str] = []
    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "It's a senior tax associate role."},
    ) as response:
        assert response.status_code == 200
        for chunk in response.iter_text():
            chunks.append(chunk)

    raw = "".join(chunks)
    assert "event: slot_update" in raw
    assert "event: extraction_partial" in raw
    # The extraction_partial event must be emitted after slot_update, per
    # the SSE ordering contract this hookup mirrors.
    assert raw.index("event: slot_update") < raw.index("event: extraction_partial")
    assert "capability_areas" in raw

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    last_extraction = refreshed["session"]["state_json"]["conversation_meta"][
        "last_extraction"
    ]
    assert last_extraction["status"] == "partial"
    assert "capability_areas" in last_extraction["dropped_keys"]

    # The valid depth_distinction update still landed — a dropped key must
    # not discard the rest of the round (P9.4 core contract, exercised here
    # end-to-end through the API).
    assert (
        refreshed["session"]["state_json"]["v2_draft"]["depth_distinction"][
            "builder_definition"
        ]
        == "x"
    )


def test_converse_stream_omits_extraction_partial_on_clean_round(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_stream(["Got it."]),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({"role_title": "Senior Tax Associate"}),
    )

    session_id = _create_conversation_session(api_client)

    chunks: list[str] = []
    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "It's a senior tax associate role."},
    ) as response:
        assert response.status_code == 200
        for chunk in response.iter_text():
            chunks.append(chunk)

    raw = "".join(chunks)
    assert "event: extraction_partial" not in raw

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    last_extraction = refreshed["session"]["state_json"]["conversation_meta"][
        "last_extraction"
    ]
    assert last_extraction["status"] == "updated"
    assert last_extraction["dropped_keys"] == []


def test_converse_stream_skips_v2_draft_merge_while_synthesis_running(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chat extraction must not clobber v2_draft while upload synthesis runs."""

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_stream(["Got it — tell me more."]),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({"role_title": "From Chat Extraction"}),
    )

    session_id = _create_conversation_session(api_client)
    synthesis_owned = {
        "role_title": "From Synthesis Worker",
        "role_summary": "Owned by synthesis.",
        "capability_areas": [
            {"name": "Applied AI", "description": "Ships production systems."}
        ],
        "depth_distinction": {
            "builder_definition": "Has owned the work.",
            "user_definition": "Has supported it.",
            "edge_case_guidance": "Review adjacents.",
        },
        "non_fit_patterns": [],
        "target_modules": ["linkedin"],
    }
    patch = api_client.patch(
        f"/api/intake/sessions/{session_id}",
        json={
            "state_json": {
                "v2_draft": synthesis_owned,
                "source_packet_synthesis": {
                    "status": "running",
                    "revision": 1,
                },
            }
        },
    )
    assert patch.status_code == 200, patch.text

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "The role is Head of Applied AI."},
    ) as response:
        assert response.status_code == 200
        for _chunk in response.iter_text():
            pass

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    assert (
        refreshed["session"]["state_json"]["v2_draft"]["role_title"]
        == "From Synthesis Worker"
    )


def test_converse_stream_emits_ready_to_compose_when_threshold_crossed(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_stream(["Got it."]),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm(
            {
                "role_title": "Senior Tax Associate",
                "role_summary": "Owns multi-state SUT filings.",
                "capability_areas": [
                    {
                        "name": "Sales tax",
                        "description": "Owns SUT filings end-to-end.",
                    }
                ],
            }
        ),
    )

    session_id = _create_conversation_session(api_client)

    chunks: list[str] = []
    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "..."},
    ) as response:
        for chunk in response.iter_text():
            chunks.append(chunk)

    raw = "".join(chunks)
    assert "event: ready_to_compose" in raw

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    assert refreshed["session"]["state_json"]["conversation_meta"]["ready_to_compose"] is True


# -------------------------------------------------------------------------
# converse_stream — failure modes
# -------------------------------------------------------------------------


def test_converse_stream_emits_degraded_event_on_orchestrator_failure_before_any_delta(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit finding F-2: provider failure before any streamed delta must
    surface as a structurally distinct ``degraded`` SSE event — NOT as a
    normal ``message_chunk`` containing "Lost my train of thought".

    State-side: only the opener and the recruiter message persist; no
    Cloris fallback turn is committed. The next turn detects this as a
    dropped turn (last message is recruiter) and the C5 prompt builder
    flips ``dropped_turn=True`` so Cloris re-engages with the same
    question rather than asking something new.
    """

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_raises_immediately(),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "tell me about it"},
    ) as response:
        body = "".join(response.iter_text())

    assert "event: degraded" in body
    assert '"reason":"provider_failed"' in body
    assert '"recoverable":true' in body
    assert "event: done" in body
    # The recruiter must NOT see the legacy in-band fallback chunk.
    assert "Lost my train of thought" not in body
    # No standalone message_chunk frames either — orchestrator emitted no
    # deltas so the wire is silent on that channel before degraded.
    assert "event: message_chunk" not in body

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    messages = refreshed["session"]["state_json"]["messages"]
    # opener + recruiter only — no Cloris fallback turn.
    assert len(messages) == 2
    assert messages[0]["role"] == "cloris"  # opener
    assert messages[1]["role"] == "recruiter"


def test_converse_stream_emits_degraded_after_partial_then_keeps_partial(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit finding F-2: provider failure mid-stream preserves the
    partial Cloris turn (so the conversation isn't silently truncated)
    and STILL emits a ``degraded`` SSE event so the recruiter UI shows
    a banner."""

    def _factory(system_prompt, user_prompt, *, usage_context=None, max_tokens=4096):
        yield ("delta", "Got it — let me ")
        yield ("delta", "think about whether ")
        raise RuntimeError("simulated mid-stream outage")

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _factory,
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "tell me about it"},
    ) as response:
        body = "".join(response.iter_text())

    assert "event: message_chunk" in body
    assert "event: degraded" in body
    assert '"reason":"provider_failed"' in body
    assert "event: done" in body

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    messages = refreshed["session"]["state_json"]["messages"]
    # opener + recruiter + partial-cloris (preserved with degraded meta).
    assert len(messages) == 3
    assert messages[2]["role"] == "cloris"
    assert "Got it" in messages[2]["content"]
    assert messages[2].get("meta", {}).get("degraded") is True


def test_converse_stream_dropped_turn_detected_on_session_load(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the previous turn died mid-stream (last message is recruiter),
    the orchestrator's prompt builder is invoked with ``dropped_turn=True``.
    """

    captured = {"system_prompts": []}

    def _capturing_factory(system_prompt, user_prompt, *, usage_context=None, max_tokens=4096):
        captured["system_prompts"].append(system_prompt)
        yield ("delta", "OK.")
        yield ("usage", {
            "input_tokens": 5,
            "output_tokens": 1,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        })

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _capturing_factory,
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)

    # Simulate a dropped previous turn by manually appending a recruiter
    # message to state without a matching cloris reply, then sending a
    # new recruiter message.
    api_client.patch(
        f"/api/intake/sessions/{session_id}",
        json={
            "state_json": {
                "messages": [
                    {
                        "role": "cloris",
                        "content": "opener",
                        "ts": "2026-05-13T12:00:00+00:00",
                    },
                    {
                        "role": "recruiter",
                        "content": "got cut off mid-flight",
                        "ts": "2026-05-13T12:00:01+00:00",
                    },
                ]
            }
        },
    )

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "(repeated send after the drop)"},
    ) as response:
        for _ in response.iter_text():
            pass

    assert captured["system_prompts"], "orchestrator was not invoked"
    assert "# RESUME-FROM-DROPPED-TURN" in captured["system_prompts"][0]


# -------------------------------------------------------------------------
# Concurrency + lock lifecycle
# -------------------------------------------------------------------------


def test_converse_stream_409_when_turn_in_flight(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second concurrent request returns 409. We simulate in-flight by
    pre-acquiring the lock from outside before sending the request.
    """

    from cloris.api.intake import _CONVERSATION_LOCKS

    # Stub the orchestrator so we don't actually hit anything in flight.
    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_stream(["x"]),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)

    # Pre-acquire the lock to simulate an in-flight turn. asyncio.Lock
    # has to be acquired inside an event loop; we use asyncio.run.
    locked = asyncio.Lock()

    async def _grab():
        await locked.acquire()
        return locked

    asyncio.run(_grab())
    _CONVERSATION_LOCKS[session_id] = locked

    response = api_client.post(
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "concurrent send"},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "turn_in_flight"

    # Cleanup so the autouse fixture's clear() finds the dict consistent.
    locked.release()
    del _CONVERSATION_LOCKS[session_id]


def test_converse_stream_evicts_lock_after_successful_turn(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock dict must be empty after every turn so it doesn't grow
    monotonically across the trial.
    """

    from cloris.api.intake import _CONVERSATION_LOCKS

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_stream(["fine"]),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "first"},
    ) as response:
        for _ in response.iter_text():
            pass

    assert _CONVERSATION_LOCKS == {}


def test_converse_stream_evicts_lock_after_orchestrator_failure(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock eviction holds even when the orchestrator throws — the
    ``finally`` clause runs."""

    from cloris.api.intake import _CONVERSATION_LOCKS

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_raises_immediately(),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "first"},
    ) as response:
        for _ in response.iter_text():
            pass

    assert _CONVERSATION_LOCKS == {}


# -------------------------------------------------------------------------
# Auth (SSE-exempt block)
# -------------------------------------------------------------------------


def test_converse_stream_401_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When CLORIS_SKIP_AUTH_FOR_TESTING is unset, the SSE endpoint
    requires ``?token=`` matching SESSION_TOKEN.
    """

    monkeypatch.delenv("CLORIS_SKIP_AUTH_FOR_TESTING", raising=False)

    tmp_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    tmp_store = RuntimeStateStore(tmp_db_path)
    monkeypatch.setattr(
        "cloris.api.intake._intake_store", lambda: tmp_store
    )
    monkeypatch.setattr(
        "cloris.api.intake._intake_db_path", lambda: tmp_db_path
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/intake/sessions/1/converse/stream",
        json={"recruiter_message": "hi"},
    )

    assert response.status_code == 401


def test_converse_stream_200_with_correct_query_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Token in ``?token=`` matching SESSION_TOKEN is accepted (mirrors
    the existing /api/conversation/*/stream pattern).
    """

    monkeypatch.delenv("CLORIS_SKIP_AUTH_FOR_TESTING", raising=False)

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_orchestrator_stream(["ok"]),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    tmp_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    tmp_store = RuntimeStateStore(tmp_db_path)
    monkeypatch.setattr(
        "cloris.api.intake._intake_store", lambda: tmp_store
    )
    monkeypatch.setattr(
        "cloris.api.intake._intake_db_path", lambda: tmp_db_path
    )

    from cloris.api.auth import SESSION_TOKEN

    client = TestClient(
        create_app(),
        headers={"Authorization": f"Bearer {SESSION_TOKEN}"},
    )
    create_resp = client.post(
        "/api/intake/sessions", json={"role_title": "Tax Associate"}
    )
    session_id = create_resp.json()["session"]["id"]
    client.post(f"/api/intake/sessions/{session_id}/start_conversation")

    # Now try the SSE endpoint with NO Authorization header — must use
    # ?token= query param.
    raw_client = TestClient(create_app())
    response = raw_client.post(
        f"/api/intake/sessions/{session_id}/converse/stream?token={SESSION_TOKEN}",
        json={"recruiter_message": "hi"},
    )
    assert response.status_code == 200


def test_converse_stream_401_with_wrong_query_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CLORIS_SKIP_AUTH_FOR_TESTING", raising=False)

    tmp_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    tmp_store = RuntimeStateStore(tmp_db_path)
    monkeypatch.setattr(
        "cloris.api.intake._intake_store", lambda: tmp_store
    )
    monkeypatch.setattr(
        "cloris.api.intake._intake_db_path", lambda: tmp_db_path
    )

    client = TestClient(create_app())
    response = client.post(
        "/api/intake/sessions/1/converse/stream?token=not-the-real-token",
        json={"recruiter_message": "hi"},
    )

    assert response.status_code == 401
