"""Tests for the conversational intake cost + cap telemetry (Phase C11).

Three contracts to pin:

1. ``cap_state_for`` returns ``"normal" | "soft" | "hard"`` correctly
   on all four boundary cases (under both, soft on turns, soft on
   cost, hard on either).
2. The C5 endpoint computes cap_state from the PRE-turn meta and
   passes it through to the orchestrator's system-prompt builder
   (substring assertion on the captured prompt).
3. Hard cap forces ``ready_to_compose=True`` regardless of v2_draft
   state — the recruiter sees the draft after this turn whether or
   not the deterministic sufficiency check would have flipped it.
4. Cost rollup: after N stubbed turns with known token counts,
   ``state_json.conversation_meta.cost_usd_running_total`` equals the
   sum of :func:`shared.llm_usage.estimate_usage_cost_usd` over those
   turns. Grounded in the helper, not the JSONL log (per Decision #6:
   in-stream cost rollup is the runtime source).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloris.app import create_app
from shared import config as shared_config
from shared.intake_conversation import (
    HARD_CAP_TURNS,
    HARD_CAP_USD,
    SOFT_CAP_TURNS,
    SOFT_CAP_USD,
    cap_state_for,
)
from shared.llm_usage import estimate_usage_cost_usd
from shared.runtime_state.store import RuntimeStateStore


# -------------------------------------------------------------------------
# cap_state_for — pure boundary tests
# -------------------------------------------------------------------------


def test_cap_state_normal_under_both_caps() -> None:
    assert cap_state_for(turn_count=0, cost_usd_running_total=0.0) == "normal"
    assert cap_state_for(turn_count=1, cost_usd_running_total=0.05) == "normal"


def test_cap_state_soft_when_turns_at_threshold() -> None:
    assert (
        cap_state_for(turn_count=SOFT_CAP_TURNS, cost_usd_running_total=0.0)
        == "soft"
    )


def test_cap_state_soft_when_cost_at_threshold() -> None:
    assert (
        cap_state_for(turn_count=0, cost_usd_running_total=SOFT_CAP_USD)
        == "soft"
    )


def test_cap_state_hard_when_turns_at_threshold() -> None:
    assert (
        cap_state_for(turn_count=HARD_CAP_TURNS, cost_usd_running_total=0.0)
        == "hard"
    )


def test_cap_state_hard_when_cost_at_threshold() -> None:
    assert (
        cap_state_for(turn_count=0, cost_usd_running_total=HARD_CAP_USD)
        == "hard"
    )


def test_cap_state_hard_dominates_soft() -> None:
    """Hard cap on either dimension wins, even if the other is below soft."""

    assert (
        cap_state_for(turn_count=HARD_CAP_TURNS, cost_usd_running_total=0.01)
        == "hard"
    )
    assert (
        cap_state_for(turn_count=1, cost_usd_running_total=HARD_CAP_USD)
        == "hard"
    )


# -------------------------------------------------------------------------
# C5 endpoint — cap state passes through to the orchestrator
# -------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")


@pytest.fixture
def api_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    tmp_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    tmp_store = RuntimeStateStore(tmp_db_path)

    monkeypatch.setattr("cloris.api.intake._intake_store", lambda: tmp_store)
    monkeypatch.setattr("cloris.api.intake._intake_db_path", lambda: tmp_db_path)
    return TestClient(create_app())


@pytest.fixture(autouse=True)
def _clear_locks_between_tests():
    from cloris.api.intake import _CONVERSATION_LOCKS

    _CONVERSATION_LOCKS.clear()
    yield
    _CONVERSATION_LOCKS.clear()


def _capturing_orchestrator(captured: dict):
    def _factory(system_prompt, user_prompt, *, usage_context=None, max_tokens=4096):
        captured.setdefault("system_prompts", []).append(system_prompt)
        yield ("delta", "ok")
        yield (
            "usage",
            {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )

    return _factory


def _stub_cheap_llm(updates: dict):
    def _impl(system, user, expect_json=True, usage_context=None):
        return updates

    return _impl


def _create_conversation_session(api_client: TestClient) -> int:
    create_resp = api_client.post(
        "/api/intake/sessions", json={"role_title": "Tax Associate"}
    )
    session_id = create_resp.json()["session"]["id"]
    api_client.post(
        f"/api/intake/sessions/{session_id}/start_conversation"
    )
    return session_id


def _seed_conversation_meta(
    api_client: TestClient,
    session_id: int,
    *,
    turn_count: int,
    cost_usd_running_total: float,
) -> None:
    """Patch state_json.conversation_meta to seed cap-state telemetry."""

    api_client.patch(
        f"/api/intake/sessions/{session_id}",
        json={
            "state_json": {
                "messages": [
                    {
                        "role": "cloris",
                        "content": "opener",
                        "ts": "2026-05-13T12:00:00+00:00",
                    }
                ],
                "conversation_meta": {
                    "turn_count": turn_count,
                    "cost_usd_running_total": cost_usd_running_total,
                    "manually_edited_keys": [],
                },
            }
        },
    )


def test_soft_cap_seen_in_orchestrator_prompt(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _capturing_orchestrator(captured),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)
    _seed_conversation_meta(
        api_client,
        session_id,
        turn_count=SOFT_CAP_TURNS,
        cost_usd_running_total=0.0,
    )

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "still talking"},
    ) as response:
        for _ in response.iter_text():
            pass

    assert captured["system_prompts"]
    assert "# CAP STATE — SOFT" in captured["system_prompts"][0]


def test_hard_cap_seen_in_orchestrator_prompt(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _capturing_orchestrator(captured),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)
    _seed_conversation_meta(
        api_client,
        session_id,
        turn_count=HARD_CAP_TURNS,
        cost_usd_running_total=0.0,
    )

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "still talking"},
    ) as response:
        for _ in response.iter_text():
            pass

    assert "# CAP STATE — HARD" in captured["system_prompts"][0]


def test_hard_cap_forces_ready_to_compose_event(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with an empty v2_draft (sufficiency check would normally
    return False), the hard cap surfaces ready_to_compose=True so the
    recruiter sees the brief after this turn.
    """

    captured: dict = {}
    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _capturing_orchestrator(captured),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),  # no extraction, draft stays empty
    )

    session_id = _create_conversation_session(api_client)
    _seed_conversation_meta(
        api_client,
        session_id,
        turn_count=HARD_CAP_TURNS,
        cost_usd_running_total=0.0,
    )

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "stop"},
    ) as response:
        body = "".join(response.iter_text())

    assert "event: ready_to_compose" in body
    assert "event: slot_update" in body
    # Slot update at hard cap reports missing == [] regardless of the
    # actual sufficiency state.
    assert '"missing":[]' in body

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    meta = refreshed["session"]["state_json"]["conversation_meta"]
    assert meta["ready_to_compose"] is True
    compose = refreshed["session"]["state_json"]["conversation_compose"]
    assert compose["status"] in {"composing", "ready"}


def test_hard_cap_auto_schedules_compose_job(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hard cap must schedule composition so cap copy is not an orphan promise."""

    scheduled: dict[str, int] = {}

    def _capture_schedule(*, session_id: int, expected_revision: int):
        scheduled["session_id"] = session_id
        scheduled["revision"] = expected_revision

        class _NoThread:
            def start(self) -> None:
                return None

        return _NoThread()

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _capturing_orchestrator({}),
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )
    monkeypatch.setattr(
        "cloris.api.intake_compose.schedule_compose_from_conversation",
        _capture_schedule,
    )
    monkeypatch.setattr(
        "cloris.api.intake_compose.should_run_compose_synchronously",
        lambda: False,
    )

    session_id = _create_conversation_session(api_client)
    _seed_conversation_meta(
        api_client,
        session_id,
        turn_count=HARD_CAP_TURNS,
        cost_usd_running_total=0.0,
    )

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "stop"},
    ) as response:
        for _ in response.iter_text():
            pass

    assert scheduled["session_id"] == session_id
    assert scheduled["revision"] == 1


# -------------------------------------------------------------------------
# Cost rollup (in-stream, sourced from MessageStreamEvent.usage)
# -------------------------------------------------------------------------


def test_cost_rollup_matches_estimate_usage_cost_usd(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After one turn with known token counts, the persisted
    cost_usd_running_total equals the value returned by the
    estimate_usage_cost_usd helper. Grounded in the helper, not the
    JSONL log — Decision #6 in the implementation plan.
    """

    INPUT_TOKENS = 1234
    OUTPUT_TOKENS = 567
    CACHE_READ = 100
    CACHE_CREATION = 50

    def _stub_stream(system_prompt, user_prompt, *, usage_context=None, max_tokens=4096):
        yield ("delta", "ok")
        yield (
            "usage",
            {
                "input_tokens": INPUT_TOKENS,
                "output_tokens": OUTPUT_TOKENS,
                "cache_read_input_tokens": CACHE_READ,
                "cache_creation_input_tokens": CACHE_CREATION,
            },
        )

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_stream,
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)

    with api_client.stream(
        "POST",
        f"/api/intake/sessions/{session_id}/converse/stream",
        json={"recruiter_message": "first turn"},
    ) as response:
        for _ in response.iter_text():
            pass

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    persisted_cost = refreshed["session"]["state_json"]["conversation_meta"][
        "cost_usd_running_total"
    ]

    expected_cost, _ = estimate_usage_cost_usd(
        model=shared_config.OPUS_MODEL_NAME,
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        cache_read_input_tokens=CACHE_READ,
        cache_creation_input_tokens=CACHE_CREATION,
    )

    assert persisted_cost == pytest.approx(expected_cost, rel=1e-6)


def test_cost_rollup_accumulates_across_turns(
    api_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two turns should accumulate costs additively in
    cost_usd_running_total. Validates the bump_conversation_meta path
    for cost_delta_usd.
    """

    def _stub_stream(system_prompt, user_prompt, *, usage_context=None, max_tokens=4096):
        yield ("delta", "ok")
        yield (
            "usage",
            {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )

    monkeypatch.setattr(
        "shared.intake_conversation.orchestrator.opus_llm_cached_stream",
        _stub_stream,
    )
    monkeypatch.setattr(
        "shared.intake_conversation.extractor.cheap_llm",
        _stub_cheap_llm({}),
    )

    session_id = _create_conversation_session(api_client)

    for _ in range(2):
        with api_client.stream(
            "POST",
            f"/api/intake/sessions/{session_id}/converse/stream",
            json={"recruiter_message": "go"},
        ) as response:
            for _ in response.iter_text():
                pass

    refreshed = api_client.get(f"/api/intake/sessions/{session_id}").json()
    meta = refreshed["session"]["state_json"]["conversation_meta"]
    persisted_cost = meta["cost_usd_running_total"]
    persisted_turn_count = meta["turn_count"]

    one_turn_cost, _ = estimate_usage_cost_usd(
        model=shared_config.OPUS_MODEL_NAME,
        input_tokens=500,
        output_tokens=100,
    )

    assert persisted_turn_count == 2
    assert persisted_cost == pytest.approx(one_turn_cost * 2, rel=1e-6)
