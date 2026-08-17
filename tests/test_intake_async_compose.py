"""Backend tests for the async intake compose job contract.

Covers ``state_json.conversation_compose``, the revision-guarded scheduler,
and synchronous completion under cert / deterministic paths.

See ``plans/intake-product-contracts.md`` Slice 1.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def isolated_intake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from cloris import api as cloris_api
    from shared import output_paths

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cloris_api._paths, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(cloris_api._paths, "_CONFIG_PARENT", tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CLORIS_OUTPUT_ROOT", str(output_dir))

    intake_root = output_dir / "intake"
    intake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(output_paths, "INTAKE_ROOT", intake_root)

    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access", lambda: False
    )
    return tmp_path


@pytest.fixture()
def client(isolated_intake: Path) -> TestClient:
    from cloris.app import create_app

    return TestClient(create_app())


def _create_session(client: TestClient) -> int:
    response = client.post("/api/intake/sessions", json={})
    assert response.status_code == 201, response.text
    return int(response.json()["session"]["id"])


def _seed_transcript(client: TestClient, session_id: int) -> None:
    client.patch(
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


def _wait_for_compose_status(
    client: TestClient,
    session_id: int,
    *,
    expected: set[str],
    timeout: float = 30.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(
            f"/api/intake/sessions/{session_id}/compose_jobs/current"
        )
        assert response.status_code == 200, response.text
        last = response.json()["job"]
        if last.get("status") in expected:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"timeout waiting for compose status in {expected}; last job={last}"
    )


def test_compose_job_returns_fast_with_composing_or_ready(client: TestClient) -> None:
    session_id = _create_session(client)
    _seed_transcript(client, session_id)

    start = time.monotonic()
    response = client.post(f"/api/intake/sessions/{session_id}/compose_jobs")
    elapsed_ms = (time.monotonic() - start) * 1000.0

    assert response.status_code == 200, response.text
    assert elapsed_ms < 2000.0, f"compose job POST took {elapsed_ms:.0f}ms"

    body = response.json()
    job = body["job"]
    # Deterministic path completes inline — ready with result.
    assert job["status"] in {"composing", "ready"}
    assert job["revision"] == 1
    assert job["error"] is None
    assert job["started_at"] is not None

    if job["status"] == "composing":
        from cloris.api.intake_compose import wait_for_compose

        wait_for_compose(session_id, timeout=5.0)
        job = _wait_for_compose_status(client, session_id, expected={"ready"})

    assert job["status"] == "ready"
    assert job["result"]["compose_status"] == "composed"


def test_compose_from_conversation_alias_matches_compose_jobs(
    client: TestClient,
) -> None:
    session_id = _create_session(client)
    _seed_transcript(client, session_id)

    response = client.post(
        f"/api/intake/sessions/{session_id}/compose_from_conversation"
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["slice"] == "v0-intake-compose-job-1"
    assert body["job"]["status"] == "ready"
    assert body["job"]["result"]["compose_status"] == "composed"


def test_compose_completes_and_writes_draft(client: TestClient) -> None:
    session_id = _create_session(client)
    _seed_transcript(client, session_id)

    client.post(f"/api/intake/sessions/{session_id}/compose_jobs")
    session = client.get(f"/api/intake/sessions/{session_id}").json()["session"]
    block = session["state_json"]["conversation_compose"]
    assert block["status"] == "ready"
    assert block["error"] is None
    assert block["completed_at"] is not None
    assert block["result"]["compose_status"] == "composed"

    draft = session["state_json"]["v2_draft"]
    assert draft["role_title"] == "Head of Applied AI Lab"
    assert session["current_step"] == "review"
    assert session["state_json"]["conversation_compose_meta"]["status"] == "composed"


def test_compose_failure_sets_failed_with_recruiter_safe_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris.api import intake_compose as mod

    def _boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("simulated compose explosion with sensitive token")

    monkeypatch.setattr(mod, "compose_from_conversation_pure", _boom)
    monkeypatch.setattr(mod, "should_run_compose_synchronously", lambda: False)

    session_id = _create_session(client)
    _seed_transcript(client, session_id)

    client.post(f"/api/intake/sessions/{session_id}/compose_jobs")
    job = _wait_for_compose_status(client, session_id, expected={"failed"})

    assert job["status"] == "failed"
    error = job["error"]
    assert isinstance(error, str) and error
    assert "simulated compose explosion" not in error
    assert "RuntimeError" not in error


def test_stale_revision_compose_commit_is_dropped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cloris.api import intake_compose as mod

    real_pure = mod.compose_from_conversation_pure
    barrier = threading.Event()
    release = threading.Event()
    call_count = {"n": 0}
    stale_marker = {"role_title": "STALE STALE STALE"}

    def _slow_pure(**kwargs: Any) -> mod.ComposeProduct:
        call_count["n"] += 1
        if call_count["n"] == 1:
            barrier.set()
            release.wait(timeout=5.0)
            product = real_pure(**kwargs)
            return mod.ComposeProduct(
                compose_status=product.compose_status,
                deficits=product.deficits,
                missing_keys=product.missing_keys,
                invalid_keys=product.invalid_keys,
                insight_deficits=product.insight_deficits,
                v2_draft={**product.v2_draft, **stale_marker},
                insight_updates=product.insight_updates,
                metadata=product.metadata,
                role_title=stale_marker["role_title"],
                current_step=product.current_step,
                conversation_meta=product.conversation_meta,
            )
        return real_pure(**kwargs)

    monkeypatch.setattr(mod, "compose_from_conversation_pure", _slow_pure)
    monkeypatch.setattr(mod, "should_run_compose_synchronously", lambda: False)

    session_id = _create_session(client)
    _seed_transcript(client, session_id)

    first = client.post(f"/api/intake/sessions/{session_id}/compose_jobs")
    assert first.json()["job"]["revision"] == 1
    assert barrier.wait(timeout=2.0)

    second = client.post(f"/api/intake/sessions/{session_id}/compose_jobs")
    assert second.json()["job"]["revision"] == 2

    release.set()
    _wait_for_compose_status(client, session_id, expected={"ready"}, timeout=10.0)

    session = client.get(f"/api/intake/sessions/{session_id}").json()["session"]
    assert session["state_json"]["conversation_compose"]["revision"] == 2
    assert session["state_json"]["v2_draft"]["role_title"] != stale_marker["role_title"]
    assert "Applied AI Lab" in session["state_json"]["v2_draft"]["role_title"]


def test_compose_deficits_without_overwrite(client: TestClient) -> None:
    session_id = _create_session(client)
    useful_partial = {"role_title": "Manual Title"}
    client.patch(
        f"/api/intake/sessions/{session_id}",
        json={"state_json": {"v2_draft": useful_partial, "messages": []}},
    )

    response = client.post(f"/api/intake/sessions/{session_id}/compose_jobs")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job"]["status"] == "ready"
    assert body["job"]["result"]["compose_status"] == "deficits"
    session = body["session"]
    assert session["current_step"] == "welcome"
    assert session["state_json"]["v2_draft"] == useful_partial
    assert body["job"]["result"]["deficits"]


def test_certify_env_runs_compose_inline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLORIS_CERTIFY", "1")
    monkeypatch.setattr(
        "shared.intake_conversation.composer._has_llm_access", lambda: True
    )

    session_id = _create_session(client)
    _seed_transcript(client, session_id)

    response = client.post(f"/api/intake/sessions/{session_id}/compose_jobs")
    assert response.status_code == 200, response.text
    assert response.json()["job"]["status"] == "ready"
