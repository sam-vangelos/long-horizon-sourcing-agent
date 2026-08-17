"""Tests for the Phase 0 ``apikey-ui`` slice — first-launch
onboarding state and HTTP surface.

Pins:

- ``upsert_credential`` writes ``KEY=value`` to the user-data ``.env``
  with chmod 600, replaces an existing entry rather than duplicating,
  and updates ``os.environ`` in-process.
- ``record_acknowledgment`` writes a structured timestamp + version
  payload that the welcome gate can verify without re-launching.
- ``onboarding_status`` returns the right ``welcome_complete``
  combinations across (anthropic_present, acknowledged) corners.
- ``GET /api/onboarding/status`` mirrors the module read.
- ``POST /api/onboarding/credential`` handles the success path,
  rejects unknown keys (HTTP 422), and rejects invalid values (HTTP 422).
- ``POST /api/onboarding/acknowledge`` rejects ``acknowledged=False``
  (HTTP 422) and records the acknowledgment on the success path.

The tests use a tmp user-data dir (set via ``CLORIS_USER_DATA_DIR``)
so the recipient's real ``~/Library/Application Support/Cloris`` is
never touched.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cloris import onboarding
from cloris.app import create_app


@pytest.fixture
def user_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point Cloris's user-data resolution at a fresh tmp dir for the test.

    Also wipes any pre-loaded API-key env vars so ``credential_present``
    sees an empty starting state. We use ``setenv("", "")`` to force an
    empty string rather than ``delenv`` because the project's dev
    ``.env`` (loaded at ``shared.config`` import time, before any
    fixture runs) leaves these set in ``os.environ``, and
    :func:`onboarding.credential_present` reads ``os.environ.get(...,
    "").strip()`` — empty string and unset are treated identically by
    that read.
    """

    monkeypatch.setenv("CLORIS_USER_DATA_DIR", str(tmp_path))
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "PERPLEXITY_API_KEY",
    ):
        monkeypatch.setenv(var, "")
    return tmp_path


@pytest.fixture
def client(user_data_dir: Path) -> TestClient:
    return TestClient(create_app())


def test_upsert_credential_writes_env_with_mode_0o600(user_data_dir: Path) -> None:
    onboarding.upsert_credential("anthropic_api_key", "sk-ant-fake-1")

    env_path = user_data_dir / ".env"
    assert env_path.exists()
    content = env_path.read_text()
    assert "ANTHROPIC_API_KEY=sk-ant-fake-1" in content

    mode = stat.S_IMODE(env_path.stat().st_mode)
    assert mode == 0o600, (
        f"env file must be owner-only; observed mode {oct(mode)}"
    )


def test_upsert_credential_replaces_existing_value(user_data_dir: Path) -> None:
    onboarding.upsert_credential("anthropic_api_key", "sk-ant-stale")
    onboarding.upsert_credential("anthropic_api_key", "sk-ant-fresh")

    content = (user_data_dir / ".env").read_text()
    assert content.count("ANTHROPIC_API_KEY=") == 1
    assert "ANTHROPIC_API_KEY=sk-ant-fresh" in content
    assert "sk-ant-stale" not in content


def test_upsert_credential_preserves_unrelated_lines(user_data_dir: Path) -> None:
    """A handwritten ``.env`` carrying other env vars must survive an
    upsert intact — the welcome surface sets one key at a time and
    must not eat the recipient's other env state."""

    env_path = user_data_dir / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-existing\nSOME_OTHER_VAR=value\n"
    )
    os.chmod(env_path, 0o600)

    onboarding.upsert_credential("anthropic_api_key", "sk-ant-1")

    content = env_path.read_text()
    assert "OPENAI_API_KEY=sk-existing" in content
    assert "SOME_OTHER_VAR=value" in content
    assert "ANTHROPIC_API_KEY=sk-ant-1" in content


def test_upsert_credential_updates_os_environ(user_data_dir: Path) -> None:
    onboarding.upsert_credential("anthropic_api_key", "sk-ant-live")

    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-live"


def test_upsert_credential_updates_loaded_shared_config(user_data_dir: Path) -> None:
    """LLM clients read ``shared.config`` module attributes at call time.

    Updating only ``os.environ`` makes the welcome/settings UI look
    successful while already-imported clients keep using the stale key.
    """

    from shared import config

    previous = config.ANTHROPIC_API_KEY
    config.ANTHROPIC_API_KEY = "sk-ant-stale"
    try:
        onboarding.upsert_credential("anthropic_api_key", "sk-ant-live")

        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-live"
        assert config.ANTHROPIC_API_KEY == "sk-ant-live"
    finally:
        config.ANTHROPIC_API_KEY = previous


def test_upsert_anthropic_credential_clears_health_cache(user_data_dir: Path) -> None:
    from cloris import anthropic_health

    anthropic_health.clear_cache()
    anthropic_health._CACHE = (  # type: ignore[attr-defined]
        0.0,
        anthropic_health.AnthropicHealth(
            state="unhealthy",
            message="stale",
            checked_at="2026-05-13T00:00:00+00:00",
            cache_age_s=0.0,
        ),
    )
    try:
        onboarding.upsert_credential("anthropic_api_key", "sk-ant-live")

        assert anthropic_health._CACHE is None  # type: ignore[attr-defined]
    finally:
        anthropic_health.clear_cache()


def test_upsert_credential_rejects_unknown_key(user_data_dir: Path) -> None:
    with pytest.raises(onboarding.UnknownCredentialKeyError):
        onboarding.upsert_credential("nonexistent_api_key", "x")


def test_upsert_credential_rejects_empty_value(user_data_dir: Path) -> None:
    with pytest.raises(ValueError):
        onboarding.upsert_credential("anthropic_api_key", "   ")


def test_credential_present_reads_env_file_when_environ_absent(
    user_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A previous launch's credential survives a fresh process via the
    ``.env`` on disk, not via the in-process ``os.environ``."""

    onboarding.upsert_credential("anthropic_api_key", "sk-ant-disk")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert onboarding.credential_present("anthropic_api_key") is True


def test_record_and_read_acknowledgment(user_data_dir: Path) -> None:
    onboarding.record_acknowledgment(cloris_version="0.0.1-test")

    record = onboarding.acknowledgment_record()
    assert record is not None
    assert isinstance(record["acknowledged_at"], str)
    assert record["cloris_version"] == "0.0.1-test"
    assert "linkedin" in record["message"].lower()


def test_onboarding_status_complete_only_with_both_artifacts(
    user_data_dir: Path,
) -> None:
    s = onboarding.onboarding_status()
    assert s.welcome_complete is False
    assert s.anthropic_present is False
    assert s.acknowledged is False

    onboarding.upsert_credential("anthropic_api_key", "sk-ant-1")
    s = onboarding.onboarding_status()
    assert s.anthropic_present is True
    assert s.acknowledged is False
    assert s.welcome_complete is False, (
        "API key alone must not unlock the welcome gate — the "
        "acknowledgment is the relational-layer half of the bargain."
    )

    onboarding.record_acknowledgment(cloris_version="0.0.1-test")
    s = onboarding.onboarding_status()
    assert s.welcome_complete is True
    assert s.anthropic_present is True
    assert s.acknowledged is True
    assert isinstance(s.acknowledged_at, str)


def test_get_onboarding_status_endpoint(client: TestClient) -> None:
    resp = client.get("/api/onboarding/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slice"] == "v0-onboarding-status-1"
    assert body["welcome_complete"] is False
    assert body["anthropic_present"] is False
    assert body["acknowledged"] is False


def test_post_onboarding_credential_round_trip_rejects_injected_env_line(
    client: TestClient, user_data_dir: Path
) -> None:
    resp = client.post(
        "/api/onboarding/credential",
        json={"key": "anthropic_api_key", "value": "sk-ant-via-api"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["anthropic_present"] is True
    assert body["welcome_complete"] is False  # acknowledgment still missing

    env_path = user_data_dir / ".env"
    before = env_path.read_bytes()
    assert before == b"ANTHROPIC_API_KEY=sk-ant-via-api\n"

    rejected = client.post(
        "/api/onboarding/credential",
        json={
            "key": "anthropic_api_key",
            "value": "sk-ant-injected\nCLORIS_SKIP_AUTH_FOR_TESTING=1",
        },
    )
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["error"] == "invalid_credential_value"
    assert env_path.read_bytes() == before


def test_post_onboarding_credential_rejects_unknown_key(
    client: TestClient,
) -> None:
    resp = client.post(
        "/api/onboarding/credential",
        json={"key": "stripe_api_key", "value": "x"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"]["error"] == "unknown_credential_key"
    assert body["detail"]["key"] == "stripe_api_key"
    assert "anthropic_api_key" in body["detail"]["allowed"]


def test_post_onboarding_credential_rejects_empty_value(
    client: TestClient,
) -> None:
    resp = client.post(
        "/api/onboarding/credential",
        json={"key": "anthropic_api_key", "value": "   "},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "empty_credential_value"


def test_post_onboarding_acknowledge_success(
    client: TestClient, user_data_dir: Path
) -> None:
    resp = client.post(
        "/api/onboarding/acknowledge",
        json={"acknowledged": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["acknowledged"] is True
    assert isinstance(body["acknowledged_at"], str)
    assert (user_data_dir / "acknowledged.json").exists()


def test_post_onboarding_acknowledge_rejects_false(client: TestClient) -> None:
    resp = client.post(
        "/api/onboarding/acknowledge",
        json={"acknowledged": False},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "acknowledgment_required"


def test_full_welcome_flow_unlocks_welcome_complete(
    client: TestClient, user_data_dir: Path
) -> None:
    """End-to-end: empty state → upsert credential → acknowledge → gate
    unlocks. Mirrors the welcome-screen flow byte-for-byte."""

    initial = client.get("/api/onboarding/status").json()
    assert initial["welcome_complete"] is False

    cred_resp = client.post(
        "/api/onboarding/credential",
        json={"key": "anthropic_api_key", "value": "sk-ant-flow"},
    )
    assert cred_resp.json()["welcome_complete"] is False

    ack_resp = client.post(
        "/api/onboarding/acknowledge",
        json={"acknowledged": True},
    )
    assert ack_resp.status_code == 200
    assert ack_resp.json()["welcome_complete"] is True
