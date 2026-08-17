"""Backend tests for the async intake upload split.

Covers the new ``state_json.source_packet_synthesis`` shape, the
revision-guarded scheduler, and the event-loop liveness guarantee the
async upload route is supposed to give back.

See ``plans/browser-observed-certification.md`` for the contract.
"""

from __future__ import annotations

import asyncio
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------
# Shared fixtures.
# ---------------------------------------------------------------------


@pytest.fixture()
def isolated_intake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point intake storage at a clean tmp dir for this test."""

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

    # Force deterministic synthesis so tests don't hit live providers.
    monkeypatch.setattr(
        "shared.source_packet_synthesis._has_llm_access", lambda: False
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


_REALISTIC_JD = (
    "Head of Applied AI Lab, Banking & Financial Services\n\n"
    "Owns applied AI strategy, lab buildout, executive stakeholder "
    "alignment, regulated financial-services AI delivery, and production "
    "GenAI evaluation.\n\n"
    "Needs someone who has actually built and led applied AI teams in "
    "banking or financial services, not just advised on AI strategy."
).encode("utf-8")


def _upload_jd(client: TestClient, session_id: int) -> dict[str, Any]:
    response = client.post(
        f"/api/intake/sessions/{session_id}/source_packet/files",
        data={"kind": "job_description"},
        files={"files": ("head-of-applied-ai.txt", _REALISTIC_JD, "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for_status(
    client: TestClient,
    session_id: int,
    *,
    expected: set[str],
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Poll the session until ``source_packet_synthesis.status`` is in ``expected``."""

    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/intake/sessions/{session_id}")
        assert response.status_code == 200, response.text
        last = response.json()["session"]["state_json"]
        block = last.get("source_packet_synthesis") or {}
        if block.get("status") in expected:
            return last
        time.sleep(0.05)
    raise AssertionError(
        f"timeout waiting for status in {expected}; last state={last}"
    )


# ---------------------------------------------------------------------
# Slice 1 contract tests.
# ---------------------------------------------------------------------


def test_upload_returns_fast_with_running_status(client: TestClient) -> None:
    """Fast upload contract: response carries status=running + bumped revision."""

    session_id = _create_session(client)

    start = time.monotonic()
    payload = _upload_jd(client, session_id)
    elapsed_ms = (time.monotonic() - start) * 1000.0

    # SLA: with the deterministic stub returning instantly (no cert
    # delay set), the upload response itself should land well under
    # 2s. The exact threshold is a budget — generous enough to absorb
    # CI variance but tight enough to fail if synthesis sneaks back
    # onto the request thread.
    assert elapsed_ms < 2000.0, f"upload took {elapsed_ms:.0f}ms"

    state = payload["session"]["state_json"]
    block = state["source_packet_synthesis"]
    # Status is ``running`` at the moment we return — the worker may
    # have already flipped it to ``ready`` server-side, but the
    # response captured the snapshot taken just before the scheduler
    # ran.
    assert block["status"] == "running"
    assert block["revision"] == 1
    assert block["error"] is None
    assert block["started_at"] is not None
    assert block["completed_at"] is None

    # Wait for the worker to land so the next test/teardown isn't racing.
    from cloris.api.intake_synthesis import wait_for_synthesis

    wait_for_synthesis(session_id, timeout=5.0)


def test_uploaded_file_persists_before_synthesis_completes(
    client: TestClient,
) -> None:
    """The file row must be visible immediately, not after synthesis returns."""

    session_id = _create_session(client)
    payload = _upload_jd(client, session_id)
    files = payload["session"]["state_json"]["source_packet"]["files"]
    assert len(files) == 1
    assert files[0]["filename"] == "head-of-applied-ai.txt"
    assert files[0]["kind"] == "job_description"
    assert "Applied AI Lab" in files[0]["text"]

    from cloris.api.intake_synthesis import wait_for_synthesis

    wait_for_synthesis(session_id, timeout=5.0)


def test_synthesis_completes_and_flips_to_ready(client: TestClient) -> None:
    """Happy path: background worker writes synthesis-owned fields + sets ready."""

    session_id = _create_session(client)
    _upload_jd(client, session_id)

    state = _wait_for_status(client, session_id, expected={"ready"}, timeout=10.0)
    block = state["source_packet_synthesis"]
    assert block["status"] == "ready"
    assert block["error"] is None
    assert block["completed_at"] is not None

    draft = state["v2_draft"]
    assert "Applied AI Lab" in draft["role_title"]
    assert state["distillation"]["prose"]
    assert state["field_provenance"]
    assert "gap_questions" in state


def test_synthesis_failure_sets_failed_with_recruiter_safe_copy(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthesis exception: status=failed, recruiter-safe error, no stack leak."""

    from cloris.api import intake_synthesis as mod

    def _boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001 - signature parity
        raise RuntimeError("simulated synthesis explosion with sensitive token")

    monkeypatch.setattr(mod, "refresh_source_packet_artifacts_pure", _boom)

    session_id = _create_session(client)
    _upload_jd(client, session_id)

    state = _wait_for_status(client, session_id, expected={"failed"}, timeout=10.0)
    block = state["source_packet_synthesis"]
    assert block["status"] == "failed"
    error = block["error"]
    assert isinstance(error, str) and error
    # Truthful, recruiter-safe copy — must not mention being offline,
    # must not leak the exception text or stack trace.
    assert "offline" not in error.lower()
    assert "simulated synthesis explosion" not in error
    assert "RuntimeError" not in error


def test_stale_revision_commit_is_dropped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale worker (revision=N) must not overwrite a fresher revision=N+1."""

    from cloris.api import intake_synthesis as mod

    real_pure = mod.refresh_source_packet_artifacts_pure
    barrier = threading.Event()
    release = threading.Event()
    call_count = {"n": 0}
    stale_marker = {"role_title": "STALE STALE STALE"}

    def _slow_pure(*, state_snapshot: Any, session_id: int) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First worker: block until the test stages a second
            # upload (bumping revision), then return a sentinel that
            # MUST NOT land in state because we're now stale.
            barrier.set()
            release.wait(timeout=5.0)
            product = real_pure(
                state_snapshot=state_snapshot, session_id=session_id
            )
            return mod.SynthesisProduct(
                v2_draft={**product.v2_draft, **stale_marker},
                v2_draft_polish_meta=product.v2_draft_polish_meta,
                field_provenance=product.field_provenance,
                gap_questions=product.gap_questions,
                retrieval_meta=product.retrieval_meta,
                distillation=product.distillation,
                role_title=stale_marker["role_title"],
            )
        return real_pure(state_snapshot=state_snapshot, session_id=session_id)

    monkeypatch.setattr(mod, "refresh_source_packet_artifacts_pure", _slow_pure)

    session_id = _create_session(client)

    # Upload #1 — schedules worker #1, which we now have paused
    # inside ``_slow_pure``.
    first = _upload_jd(client, session_id)
    assert first["session"]["state_json"]["source_packet_synthesis"]["revision"] == 1
    assert barrier.wait(timeout=2.0)

    # Stage a second upload to bump revision to 2 before worker #1
    # commits. Worker #2 launches; its commit will be the only one
    # that lands (revision check drops worker #1).
    second = _upload_jd(client, session_id)
    assert second["session"]["state_json"]["source_packet_synthesis"]["revision"] == 2

    # Release worker #1 — its commit must be dropped.
    release.set()

    # Wait for the final ``ready`` state, then assert the stale marker
    # never reached durable state.
    state = _wait_for_status(client, session_id, expected={"ready"}, timeout=10.0)
    assert state["source_packet_synthesis"]["revision"] == 2
    assert state["v2_draft"]["role_title"] != stale_marker["role_title"]
    assert "Applied AI Lab" in state["v2_draft"]["role_title"]


def test_non_synthesis_state_preserved_across_commit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recruiter manual edit to a non-synthesis-owned field survives the commit.

    The worker reloads the latest session before committing and only
    writes synthesis-owned keys; everything else under ``state_json``
    must round-trip untouched. To prove this against a real race, we
    pause the worker mid-flight, do a read-modify-write that injects a
    non-synthesis-owned field, then release the worker.
    """

    from cloris.api import intake_synthesis as mod

    real_pure = mod.refresh_source_packet_artifacts_pure
    barrier = threading.Event()
    release = threading.Event()

    def _paused_pure(*, state_snapshot: Any, session_id: int) -> Any:
        barrier.set()
        release.wait(timeout=5.0)
        return real_pure(state_snapshot=state_snapshot, session_id=session_id)

    monkeypatch.setattr(mod, "refresh_source_packet_artifacts_pure", _paused_pure)

    session_id = _create_session(client)
    _upload_jd(client, session_id)
    assert barrier.wait(timeout=2.0)

    # Read-modify-write to inject the recruiter manual edit without
    # wiping the synthesis state block (the PATCH endpoint replaces
    # ``state_json`` wholesale, mirroring how the real frontend does
    # it: GET → mutate → PATCH).
    current = client.get(f"/api/intake/sessions/{session_id}").json()
    state = current["session"]["state_json"]
    state["affirmed_fields"] = ["role_title"]
    patch = client.patch(
        f"/api/intake/sessions/{session_id}",
        json={"state_json": state},
    )
    assert patch.status_code == 200, patch.text

    release.set()

    final = _wait_for_status(client, session_id, expected={"ready"}, timeout=10.0)
    assert final["affirmed_fields"] == ["role_title"]
    assert final["v2_draft"]["role_title"]


# ---------------------------------------------------------------------
# Event-loop liveness — the load-bearing guarantee this whole plan is
# about.
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_status_remains_responsive_during_synthesis(
    isolated_intake: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While synthesis is running, ``/api/status`` keeps answering fast."""

    # Force the deterministic stub path to take a real ~750ms so the
    # background worker is genuinely in flight while we hammer status.
    monkeypatch.setenv("CLORIS_DISABLE_INTAKE_LLM", "1")
    monkeypatch.setenv("CLORIS_CERTIFY_SYNTHESIS_DELAY_MS", "750")

    from cloris.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as ac:
        create = await ac.post("/api/intake/sessions", json={})
        assert create.status_code == 201, create.text
        session_id = int(create.json()["session"]["id"])

        files_part = {"files": ("jd.txt", _REALISTIC_JD, "text/plain")}
        upload = await ac.post(
            f"/api/intake/sessions/{session_id}/source_packet/files",
            data={"kind": "job_description"},
            files=files_part,
        )
        assert upload.status_code == 200, upload.text
        assert (
            upload.json()["session"]["state_json"]["source_packet_synthesis"][
                "status"
            ]
            == "running"
        )

        # Hammer /api/status while the background worker is sleeping.
        latencies_ms: list[float] = []
        deadline = time.monotonic() + 0.6  # within the 750ms stub delay
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            status = await ac.get("/api/status", timeout=2.0)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            assert status.status_code == 200, (
                f"status returned {status.status_code} during synthesis: {status.text}"
            )
            latencies_ms.append(elapsed_ms)
            await asyncio.sleep(0.05)

    assert len(latencies_ms) >= 5
    p95 = statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(
        latencies_ms
    )
    assert p95 < 500.0, (
        f"/api/status latencies suggest the event loop is blocked: "
        f"p95={p95:.0f}ms samples={latencies_ms}"
    )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------
# Stub-path-only delay.
# ---------------------------------------------------------------------


def test_cert_delay_only_applies_to_stub_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CLORIS_CERTIFY_SYNTHESIS_DELAY_MS`` is honored only when LLM is disabled."""

    from shared import source_packet_synthesis as mod

    monkeypatch.setenv("CLORIS_DISABLE_INTAKE_LLM", "1")
    monkeypatch.setenv("CLORIS_CERTIFY_SYNTHESIS_DELAY_MS", "250")

    long_text = " ".join(["Staff Platform Engineer with real context"] * 20)

    t0 = time.monotonic()
    result = mod.synthesize_v2_from_source_packet(
        source_text=long_text,
        job_description_text=long_text,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    assert result.source == "deterministic"
    assert elapsed_ms >= 200.0, (
        f"stub path did not honor delay: elapsed={elapsed_ms:.0f}ms"
    )

    # With the delay env unset, the stub completes near-instantly even
    # though LLM is still disabled.
    monkeypatch.delenv("CLORIS_CERTIFY_SYNTHESIS_DELAY_MS", raising=False)
    t0 = time.monotonic()
    mod.synthesize_v2_from_source_packet(
        source_text=long_text,
        job_description_text=long_text,
    )
    fast_ms = (time.monotonic() - t0) * 1000.0
    assert fast_ms < 100.0, (
        f"stub path took {fast_ms:.0f}ms with delay unset; "
        f"unexpected baseline latency"
    )
